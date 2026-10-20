"""PIT-safe RMB risk-free-rate product and excess-return attribution.

The module retains the legacy zero-rate result, then adds CCDC ChinaBond
government-bond curve scenarios without changing signals, portfolios, costs,
execution, or the historical evaluation window.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from quant_research.capm_attribution import capm_record
from quant_research.ingest import ROOT, Lake
from quant_research.style_factor_attribution import multi_factor_record


CONFIG_PATH = ROOT / "config" / "risk_free_rate_attribution.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def request_windows(start: str, end: str, maximum_days: int) -> list[tuple[str, str]]:
    """Inclusive request windows short enough for the public history endpoint."""
    cursor, final = pd.Timestamp(start), pd.Timestamp(end)
    windows: list[tuple[str, str]] = []
    while cursor <= final:
        window_end = min(cursor + pd.Timedelta(days=maximum_days - 1), final)
        windows.append((cursor.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def fetch_chinabond_history(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Download public ChinaBond yield-curve history through AKShare's adapter."""
    import akshare as ak

    download_cfg = cfg["download"]
    frames: list[pd.DataFrame] = []
    for start, end in request_windows(
        str(download_cfg["start_date"]), str(download_cfg["end_date"]),
        int(download_cfg["maximum_window_days"]),
    ):
        last_error: Exception | None = None
        for attempt in range(int(download_cfg["request_retries"])):
            try:
                frame = ak.bond_china_yield(start_date=start, end_date=end)
                if frame.empty:
                    raise RuntimeError("ChinaBond history response was empty")
                frames.append(frame)
                print(f"ChinaBond {start}-{end}: {len(frame)} rows", flush=True)
                break
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                last_error = exc
                if attempt + 1 < int(download_cfg["request_retries"]):
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"ChinaBond download failed for {start}-{end}: {last_error}")
    pulled_at = datetime.now(timezone.utc).isoformat()
    return pd.concat(frames, ignore_index=True), pulled_at


def normalize_observations(raw: pd.DataFrame, cfg: dict, pulled_at_utc: str) -> pd.DataFrame:
    """Normalize selected public curve tenors into a traceable long data product."""
    download_cfg = cfg["download"]
    required = {"曲线名称", "日期"}
    if not required.issubset(raw.columns):
        raise ValueError(f"ChinaBond response lacks columns: {sorted(required - set(raw.columns))}")
    curve = raw.loc[raw["曲线名称"].eq(download_cfg["curve_name"])].copy()
    if curve.empty:
        raise RuntimeError(f"curve absent from ChinaBond response: {download_cfg['curve_name']}")
    curve["observation_date"] = pd.to_datetime(curve["日期"], errors="coerce")
    records: list[pd.DataFrame] = []
    for scenario in cfg["rate_scenarios"]:
        source_column = scenario.get("source_column")
        if source_column is None:
            continue
        if source_column not in curve.columns:
            raise ValueError(f"ChinaBond tenor column missing: {source_column}")
        current = curve[["observation_date", source_column]].copy()
        current["annual_rate_percent"] = pd.to_numeric(
            current.pop(source_column), errors="coerce"
        )
        current["scenario_id"] = str(scenario["scenario_id"])
        current["series_id"] = (
            "ccdc_chinabond_government_bond_ytm_" + str(scenario["tenor"]).lower()
        )
        current["tenor"] = str(scenario["tenor"])
        records.append(current)
    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["observation_date", "annual_rate_percent"])
    result = result.loc[result["annual_rate_percent"].ge(0)].copy()
    result["observation_date"] = result["observation_date"].dt.normalize()
    result["publication_date"] = result["observation_date"]
    lag = int(download_cfg["availability_lag_calendar_days"])
    result["available_from_date"] = result["publication_date"] + pd.Timedelta(days=lag)
    result["annual_rate_decimal"] = result["annual_rate_percent"] / 100.0
    result["annual_quote_convention"] = str(download_cfg["annual_quote_convention"])
    result["raw_quote_unit"] = str(download_cfg["raw_quote_unit"])
    result["publication_timestamp_status"] = str(
        download_cfg["publication_timestamp_status"]
    )
    result["availability_lag_calendar_days"] = lag
    result["source_provider"] = str(download_cfg["provider"])
    result["source_name"] = str(download_cfg["source_name"])
    result["source_url"] = str(download_cfg["source_url"])
    result["api_url"] = str(download_cfg["api_url"])
    result["downloaded_at_utc"] = pulled_at_utc
    result["source_missing_flag"] = False
    result["interpolation_flag"] = False
    result = result.drop_duplicates(["scenario_id", "observation_date"], keep="last")
    result = result.sort_values(["scenario_id", "observation_date"]).reset_index(drop=True)
    if result.duplicated(["scenario_id", "observation_date"]).any():
        raise RuntimeError("duplicate normalized risk-free observations")
    date_columns = ["observation_date", "publication_date", "available_from_date"]
    result[date_columns] = result[date_columns].map(
        lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else None
    )
    return result


def annual_quote_to_holding_return(annual_rate_decimal: pd.Series, calendar_days: pd.Series) -> pd.Series:
    """ACT/365 effective holding-period return; no annual-rate / 252 shortcut."""
    rates = pd.to_numeric(annual_rate_decimal, errors="coerce")
    days = pd.to_numeric(calendar_days, errors="coerce")
    return np.power(1.0 + rates, days / 365.0) - 1.0


def strategy_clock(lake: Lake) -> pd.DataFrame:
    daily = lake.read("capm_aligned_daily", "v1")
    if daily.empty:
        raise RuntimeError("capm_aligned_daily/v1 is missing")
    dates = pd.DataFrame({
        "trade_date": pd.to_datetime(daily["trade_date"].drop_duplicates().sort_values())
    })
    benchmark = lake.read("daily_bars_validated", "sh_000905")
    if benchmark.empty:
        raise RuntimeError("CSI500 validated bars are missing")
    market_dates = pd.to_datetime(benchmark["trade_date"], errors="coerce").dropna().sort_values()
    prior = pd.DataFrame({"period_start_date": market_dates})
    dates = pd.merge_asof(
        dates.sort_values("trade_date"), prior,
        left_on="trade_date", right_on="period_start_date",
        direction="backward", allow_exact_matches=False,
    )
    if dates["period_start_date"].isna().any():
        raise RuntimeError("cannot resolve prior benchmark trading day")
    dates["calendar_days"] = (dates["trade_date"] - dates["period_start_date"]).dt.days
    if dates["calendar_days"].le(0).any():
        raise RuntimeError("non-positive holding-period day count")
    return dates


def align_rates_to_clock(
    observations: pd.DataFrame, clock: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Backward as-of alignment on known-available dates, with no future fill."""
    records: list[pd.DataFrame] = []
    base = clock.copy()
    for scenario in cfg["rate_scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        if scenario.get("source_column") is None:
            current = base.copy()
            current["scenario_id"] = scenario_id
            current["tenor"] = str(scenario["tenor"])
            current["observation_date"] = pd.NaT
            current["publication_date"] = pd.NaT
            current["available_from_date"] = pd.NaT
            current["annual_rate_percent"] = 0.0
            current["annual_rate_decimal"] = 0.0
            current["risk_free_return"] = 0.0
            current["source_provider"] = "synthetic_zero_rate_control"
            current["source_url"] = "not_applicable"
            current["source_missing_flag"] = False
            current["interpolation_flag"] = False
            current["stale_flag"] = False
            current["staleness_calendar_days"] = np.nan
            records.append(current)
            continue
        available = observations.loc[observations["scenario_id"].eq(scenario_id)].copy()
        for column in ["observation_date", "publication_date", "available_from_date"]:
            available[column] = pd.to_datetime(available[column], errors="coerce")
        available = available.sort_values("available_from_date")
        current = pd.merge_asof(
            base.sort_values("period_start_date"), available,
            left_on="period_start_date", right_on="available_from_date",
            direction="backward", allow_exact_matches=True,
        )
        current["scenario_id"] = scenario_id
        current["source_missing_flag"] = current["annual_rate_decimal"].isna()
        current["staleness_calendar_days"] = (
            current["period_start_date"] - current["observation_date"]
        ).dt.days
        current["stale_flag"] = current["staleness_calendar_days"].gt(
            int(cfg["maximum_staleness_calendar_days"])
        )
        unusable = current["source_missing_flag"] | current["stale_flag"]
        current["risk_free_return"] = annual_quote_to_holding_return(
            current["annual_rate_decimal"], current["calendar_days"]
        ).mask(unusable)
        records.append(current)
    result = pd.concat(records, ignore_index=True, sort=False)
    sourced = result.loc[~result["scenario_id"].eq(cfg["zero_rate_scenario"])]
    if (sourced["available_from_date"] > sourced["period_start_date"]).fillna(False).any():
        raise RuntimeError("future risk-free observation used")
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.strftime("%Y-%m-%d")
    result["period_start_date"] = pd.to_datetime(
        result["period_start_date"]
    ).dt.strftime("%Y-%m-%d")
    for column in ["observation_date", "publication_date", "available_from_date"]:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result.sort_values(["scenario_id", "trade_date"]).reset_index(drop=True)


def coverage_summary(aligned: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records: list[dict] = []
    for scenario_id, group in aligned.groupby("scenario_id", sort=False):
        sourced = scenario_id != cfg["zero_rate_scenario"]
        missing = int(group["risk_free_return"].isna().sum())
        records.append({
            "scenario_id": scenario_id,
            "dates": len(group),
            "start_date": group["trade_date"].min(),
            "end_date": group["trade_date"].max(),
            "missing_dates": missing,
            "missing_rate": missing / len(group),
            "interpolated_dates": int(group["interpolation_flag"].fillna(False).sum()),
            "stale_dates": int(group["stale_flag"].fillna(False).sum()),
            "maximum_staleness_calendar_days": (
                float(group["staleness_calendar_days"].max()) if sourced else np.nan
            ),
            "minimum_annual_rate_percent": (
                float(group["annual_rate_percent"].min()) if sourced else 0.0
            ),
            "maximum_annual_rate_percent": (
                float(group["annual_rate_percent"].max()) if sourced else 0.0
            ),
        })
    return pd.DataFrame(records)


def attribution_panel(lake: Lake, rates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    base = lake.read("capm_aligned_daily", "v1").rename(
        columns={"scenario": "execution_scenario"}
    )
    rate_columns = [
        "trade_date", "scenario_id", "period_start_date", "calendar_days",
        "observation_date", "available_from_date", "annual_rate_percent",
        "annual_rate_decimal", "risk_free_return", "source_provider", "source_url",
        "source_missing_flag", "interpolation_flag", "stale_flag",
        "staleness_calendar_days",
    ]
    panel = base.merge(rates[rate_columns], on="trade_date", how="left")
    # pandas has no join-key cross expansion that preserves trade_date; merge on
    # trade_date expands one strategy row to every configured rate scenario.
    if panel["risk_free_return"].isna().any():
        missing = panel.loc[panel["risk_free_return"].isna(), "trade_date"].unique()[:5]
        raise RuntimeError(f"risk-free rate missing after alignment: {missing.tolist()}")
    panel = panel.rename(columns={"scenario_id": "rate_scenario"})
    panel["strategy_excess_return"] = panel["net_return"] - panel["risk_free_return"]
    for benchmark in ["CSI300", "CSI500", "CSI1000"]:
        panel[f"{benchmark}_excess_return"] = panel[benchmark] - panel["risk_free_return"]
        panel[f"arithmetic_excess_vs_{benchmark}"] = panel["net_return"] - panel[benchmark]
        panel[f"geometric_relative_vs_{benchmark}"] = (
            (1.0 + panel["net_return"]) / (1.0 + panel[benchmark]) - 1.0
        )
    factor_daily = lake.read("style_factor_daily_returns", "v1")
    factor_wide = factor_daily.pivot(
        index="trade_date", columns="factor", values="factor_return"
    ).reset_index()
    panel = panel.merge(factor_wide, on="trade_date", how="left", validate="many_to_one")
    styles = ["size", "beta", "volatility", "liquidity", "value", "momentum"]
    if panel[styles].isna().any().any():
        raise RuntimeError("style-factor dates do not align with attribution panel")
    return panel.sort_values([
        "rate_scenario", "model", "alpha_pipeline", "execution_scenario", "trade_date"
    ]).reset_index(drop=True)


def performance_summary(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = ["rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario"]
    records: list[dict] = []
    annualization = int(cfg["annualization_days"])
    for values, group in panel.groupby(keys, sort=False):
        group = group.sort_values("trade_date")
        returns = group["net_return"].to_numpy(dtype="float64")
        rf = group["risk_free_return"].to_numpy(dtype="float64")
        excess = returns - rf
        wealth = np.cumprod(1.0 + returns)
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        records.append({
            **dict(zip(keys, values)),
            "observations": len(group),
            "annualized_portfolio_return_geometric": float(
                wealth[-1] ** (annualization / len(group)) - 1.0
            ),
            "annualized_portfolio_return_arithmetic": float(returns.mean() * annualization),
            "annualized_risk_free_return_geometric": float(
                np.prod(1.0 + rf) ** (annualization / len(group)) - 1.0
            ),
            "annualized_risk_free_return_arithmetic": float(rf.mean() * annualization),
            "annualized_excess_return_arithmetic": float(excess.mean() * annualization),
            "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(annualization)),
            "sharpe_excess_return": float(
                excess.mean() / excess.std(ddof=1) * np.sqrt(annualization)
            ),
            "maximum_drawdown": float(drawdown.min()),
        })
    return pd.DataFrame(records)


def capm_summaries(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = ["rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario"]
    records: list[dict] = []
    for values, group in panel.groupby(keys, sort=False):
        for benchmark in ["CSI300", "CSI500", "CSI1000"]:
            records.append({
                **dict(zip(keys, values)), "benchmark": benchmark,
                **capm_record(
                    group["strategy_excess_return"], group[f"{benchmark}_excess_return"],
                    lags=int(cfg["newey_west_lags"]),
                    annualization_days=int(cfg["annualization_days"]),
                    confidence_level=float(cfg["confidence_level"]),
                ),
            })
    return pd.DataFrame(records)


def style_summaries(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = ["rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario"]
    style_cfg = {
        "styles": ["size", "beta", "volatility", "liquidity", "value", "momentum"],
        "newey_west_lags": int(cfg["newey_west_lags"]),
        "annualization_days": int(cfg["annualization_days"]),
        "confidence_level": float(cfg["confidence_level"]),
    }
    records: list[dict] = []
    for values, group in panel.groupby(keys, sort=False):
        factors = group[["CSI500_excess_return", *style_cfg["styles"]]].rename(
            columns={"CSI500_excess_return": "market"}
        )
        records.append({
            **dict(zip(keys, values)),
            **multi_factor_record(group["strategy_excess_return"], factors, style_cfg),
        })
    return pd.DataFrame(records)


def regression_checks(
    lake: Lake, capm: pd.DataFrame, style: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    tolerance = float(cfg["regression_tolerance"])
    zero = str(cfg["zero_rate_scenario"])
    checks: list[dict] = []
    comparisons = [
        (
            "capm_summary_v1",
            lake.read("capm_summary", "v1").rename(columns={"scenario": "execution_scenario"}),
            capm.loc[capm["rate_scenario"].eq(zero)],
            ["model", "alpha_pipeline", "strategy", "execution_scenario", "benchmark"],
        ),
        (
            "style_factor_regression_summary_v1",
            lake.read("style_factor_regression_summary", "v1").rename(
                columns={"scenario": "execution_scenario"}
            ),
            style.loc[style["rate_scenario"].eq(zero)],
            ["model", "alpha_pipeline", "strategy", "execution_scenario"],
        ),
    ]
    for name, legacy, current, keys in comparisons:
        common_numeric = [
            column for column in legacy.select_dtypes(include=[np.number]).columns
            if column in current.columns
        ]
        merged = legacy[keys + common_numeric].merge(
            current[keys + common_numeric], on=keys, how="outer",
            suffixes=("_legacy", "_recomputed"), indicator=True,
        )
        errors = []
        for column in common_numeric:
            errors.append(np.abs(
                merged[f"{column}_legacy"] - merged[f"{column}_recomputed"]
            ).to_numpy(dtype="float64"))
        maximum = float(np.nanmax(np.concatenate(errors))) if errors else np.nan
        passed = bool(merged["_merge"].eq("both").all() and maximum <= tolerance)
        checks.append({
            "check": name,
            "legacy_rows": len(legacy),
            "recomputed_rows": len(current),
            "numeric_columns_checked": len(common_numeric),
            "maximum_absolute_error": maximum,
            "tolerance": tolerance,
            "passed": passed,
        })
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        raise RuntimeError(f"zero-rate regression failed:\n{result.to_string(index=False)}")
    return result


def protocol_frame(cfg: dict, rates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario in cfg["rate_scenarios"]:
        records.append({
            "experiment_id": cfg["experiment_id"],
            "rate_scenario": scenario["scenario_id"],
            "role": scenario["role"],
            "tenor": scenario["tenor"],
            "day_count_convention": cfg["day_count_convention"],
            "annual_to_holding_period_conversion": cfg[
                "annual_to_holding_period_conversion"
            ],
            "availability_lag_calendar_days": (
                0 if scenario.get("source_column") is None
                else int(cfg["download"]["availability_lag_calendar_days"])
            ),
            "strategy_dependent_variable": "net_return_minus_risk_free_return",
            "market_factor": "price_index_return_minus_risk_free_return",
            "style_factor_treatment": cfg["style_factor_return_convention"],
            "market_benchmark_return_type": cfg["market_benchmark_return_type"],
            "frequency": "daily_nonoverlapping",
            "regression_intercept": True,
            "standard_errors": f"Newey-West({cfg['newey_west_lags']}) Bartlett HAC",
            "alpha_annualization": f"arithmetic_daily_alpha_times_{cfg['annualization_days']}",
            "historical_oos_used_for_selection": False,
            "dates": int(rates.loc[rates["scenario_id"].eq(scenario["scenario_id"]), "trade_date"].nunique()),
        })
    return pd.DataFrame(records)


def write_report(
    cfg: dict, coverage: pd.DataFrame, performance: pd.DataFrame,
    capm: pd.DataFrame, style: pd.DataFrame, checks: pd.DataFrame,
) -> None:
    primary_exec = str(cfg["primary_execution_scenario"])
    selected_perf = performance.loc[
        performance["execution_scenario"].eq(primary_exec)
    ].copy()
    perf_columns = [
        "rate_scenario", "model", "alpha_pipeline", "observations",
        "annualized_portfolio_return_geometric", "annualized_risk_free_return_geometric",
        "annualized_excess_return_arithmetic", "sharpe_excess_return", "maximum_drawdown",
    ]
    selected_capm = capm.loc[
        capm["execution_scenario"].eq(primary_exec)
        & capm["benchmark"].eq(cfg["primary_benchmark"])
    ][[
        "rate_scenario", "model", "alpha_pipeline", "observations",
        "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
        "annualized_alpha_ci_high", "alpha_hac_t_stat", "beta", "r_squared",
    ]]
    selected_style = style.loc[
        style["execution_scenario"].eq(primary_exec)
    ][[
        "rate_scenario", "model", "alpha_pipeline", "observations",
        "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
        "annualized_alpha_ci_high", "alpha_hac_t_stat", "r_squared",
    ]]
    report = "\n".join([
        "# Phase 2：人民币无风险利率与收益归因口径升级", "",
        "## 研究边界", "",
        "本实验只修正无风险利率与归因口径；信号、模型、组合、20bp基线成本、逐日成交规则和"
        "2021–2026 historical OOS 区间均未改变，也未据此重新选择模型或参数。R0旧结果保留，"
        "R1/R2作为描述性重算，不能被表述为全新独立样本证据。", "",
        "## 数据选择与非等价候选", "",
        f"- R1：{cfg['download']['provider']} 发布的中债国债收益率曲线3个月期限；这是主有来源利率基线。",
        "- R2：同一曲线1年期限，仅作期限敏感性，避免跨数据源差异混入比较。",
        "- 国债曲线是估值/到期收益率曲线，不是可直接按该收益率逐日无摩擦投资的现金账户。",
        "- Shibor 是报价银行的人民币无担保批发性同业拆出利率，带银行信用和融资市场含义；"
        "央行逆回购/MLF 是政策工具利率。二者都不能与短期国债收益率直接视为同一对象。", "",
        "## PIT 对齐与换算", "",
        f"- 原始单位：年度百分比报价；换算：`{cfg['annual_to_holding_period_conversion']}`，"
        f"日计数为 {cfg['day_count_convention']}，没有使用年利率/252近似。",
        "- 历史接口未提供可靠的逐条盘中发布时间。本实验保守规定观察日后1个日历日才可用；"
        "每个收益区间在上一交易日仅向后查找当时已可用的最近观察，绝不未来填充。",
        "- 风格因子是零净投资多空收益，不再减无风险利率；组合左侧和价格指数市场因子均减同一rf。",
        "- CSI300/500/1000仍是价格指数收益、不含股息；因此相对可投资总收益基准的Alpha可能被高估。",
        "", "## 覆盖审计", "", coverage.to_markdown(index=False, floatfmt=".8f"),
        "", "## Sharpe 与收益口径", "",
        "`annualized_portfolio_return_geometric`是组合复合年化；`annualized_excess_return_arithmetic`"
        "是日超额收益均值乘252；Sharpe用组合超额收益计算。最大回撤来自原组合净收益，rf变化不会改变它。",
        "", selected_perf[perf_columns].to_markdown(index=False, floatfmt=".6f"),
        "", f"## CAPM（相对 {cfg['primary_benchmark']}）", "",
        f"日频含常数OLS，Newey–West({cfg['newey_west_lags']}) Bartlett HAC，Alpha按"
        f"{cfg['annualization_days']}日算术年化；每行明确报告样本数和95%置信区间。",
        "", selected_capm.to_markdown(index=False, floatfmt=".6f"),
        "", "## 本地六风格归因", "",
        "日频含常数回归；右侧为CSI500超额收益以及六个PIT自融资风格因子，推断和年化口径与CAPM一致。",
        "", selected_style.to_markdown(index=False, floatfmt=".6f"),
        "", "## R0回归核对", "", checks.to_markdown(index=False, floatfmt=".3e"),
        "", "## 结论与限制", "",
        "更换rf只改变超额收益、Sharpe和回归截距定义，不改变组合财富或最大回撤，也不能据此宣称"
        "经济Alpha得到改善。主结论应比较R0/R1/R2的量级与统计区间稳定性。公开曲线缺少可交易"
        "现金工具的费用、滚动和再投资实现；指数又不含股息，故本结果是口径改进后的研究归因，"
        "不是实盘可复制的无风险套利账本。", "",
    ])
    destination = ROOT / "reports" / "risk_free_rate_attribution.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(tables / "risk_free_rate_coverage.csv", index=False)
    selected_perf.to_csv(tables / "risk_free_rate_performance.csv", index=False)
    selected_capm.to_csv(tables / "risk_free_rate_capm.csv", index=False)
    selected_style.to_csv(tables / "risk_free_rate_style_attribution.csv", index=False)


def run(
    lake: Lake | None = None,
    *,
    reuse_observations: bool = False,
    fetcher: Callable[[dict], tuple[pd.DataFrame, str]] = fetch_chinabond_history,
) -> None:
    lake = lake or Lake()
    cfg = load_config()
    existing = lake.read("risk_free_rate_observations", "v1")
    if reuse_observations:
        if existing.empty:
            raise RuntimeError("--reuse-observations requested but data product is missing")
        observations = existing
    else:
        raw, pulled_at = fetcher(cfg)
        observations = normalize_observations(raw, cfg, pulled_at)
        lake.write(
            "risk_free_rate_observations", observations, key="v1",
            source="ccdc.chinabond.government_bond_yield_curve",
            request=cfg["download"],
        )
    rates = align_rates_to_clock(observations, strategy_clock(lake), cfg)
    coverage = coverage_summary(rates, cfg)
    if int(coverage["missing_dates"].sum()) != 0:
        raise RuntimeError(f"risk-free coverage incomplete:\n{coverage.to_string(index=False)}")
    panel = attribution_panel(lake, rates, cfg)
    performance = performance_summary(panel, cfg)
    capm = capm_summaries(panel, cfg)
    style = style_summaries(panel, cfg)
    checks = regression_checks(lake, capm, style, cfg)
    protocol = protocol_frame(cfg, rates)
    request = cfg.copy()
    for table, frame in {
        "risk_free_rate_daily": rates,
        "risk_free_rate_coverage": coverage,
        "risk_free_rate_attribution_protocol": protocol,
        "risk_free_rate_attribution_daily": panel,
        "risk_free_rate_performance_summary": performance,
        "risk_free_rate_capm_summary": capm,
        "risk_free_rate_style_summary": style,
        "risk_free_rate_regression_checks": checks,
    }.items():
        lake.write(
            table, frame, key="v1", source="risk_free_rate_attribution.v1", request=request
        )
    write_report(cfg, coverage, performance, capm, style, checks)
    print(coverage.to_string(index=False), flush=True)
    print(performance.loc[
        performance["execution_scenario"].eq(cfg["primary_execution_scenario"])
    ].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PIT risk-free-rate data and recompute excess-return attribution"
    )
    parser.add_argument(
        "--reuse-observations", action="store_true",
        help="reuse the existing normalized official snapshot instead of downloading again",
    )
    args = parser.parse_args()
    run(reuse_observations=args.reuse_observations)


if __name__ == "__main__":
    main()
