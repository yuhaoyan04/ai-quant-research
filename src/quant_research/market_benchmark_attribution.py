"""Price-index and total-return-index attribution on one audited daily clock."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
import requests
import yaml

from quant_research.capm_attribution import capm_record
from quant_research.ingest import ROOT, Lake
from quant_research.style_factor_attribution import multi_factor_record


CONFIG_PATH = ROOT / "config" / "market_benchmark_attribution.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fetch_csindex_history(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Download price and total-return series from the official CSIndex API."""
    download = cfg["download"]
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        session.headers.update({"Referer": str(download["referer"])})
        for item in cfg["indices"]:
            for return_type, field in [
                ("price_return", "official_price_code"),
                ("total_return", "official_total_return_code"),
            ]:
                code = str(item[field])
                last_error: Exception | None = None
                for attempt in range(int(download["request_retries"])):
                    try:
                        response = session.get(
                            str(download["api_url"]),
                            params={
                                "indexCode": code,
                                "startDate": str(download["start_date"]).replace("-", ""),
                                "endDate": str(download["end_date"]).replace("-", ""),
                            },
                            timeout=float(download["timeout_seconds"]),
                        )
                        response.raise_for_status()
                        payload = response.json()
                        if str(payload.get("code")) != "200" or not payload.get("data"):
                            raise RuntimeError(f"CSIndex API error: {payload.get('msg')}")
                        current = pd.DataFrame(payload["data"])
                        current["benchmark"] = str(item["benchmark"])
                        current["return_type"] = return_type
                        current["requested_index_code"] = code
                        frames.append(current)
                        print(
                            f"CSIndex {item['benchmark']} {return_type} {code}: "
                            f"{len(current)} rows", flush=True,
                        )
                        break
                    except Exception as exc:  # pragma: no cover - network-specific
                        last_error = exc
                        if attempt + 1 < int(download["request_retries"]):
                            time.sleep(2**attempt)
                else:
                    raise RuntimeError(
                        f"CSIndex download failed for {item['benchmark']}/{code}: {last_error}"
                    )
    return pd.concat(frames, ignore_index=True), datetime.now(timezone.utc).isoformat()


def normalize_official_observations(
    raw: pd.DataFrame, cfg: dict, pulled_at_utc: str,
) -> pd.DataFrame:
    """Normalize official close levels and recompute close-to-close returns."""
    required = {
        "tradeDate", "indexCode", "indexNameCnAll", "indexNameEnAll", "close",
        "changePct", "benchmark", "return_type", "requested_index_code",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"CSIndex response lacks columns: {sorted(required - set(raw.columns))}")
    frame = raw.copy()
    frame["observation_date"] = pd.to_datetime(frame["tradeDate"], errors="coerce")
    frame["close_level"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["vendor_change_return"] = pd.to_numeric(
        frame["changePct"], errors="coerce"
    ) / 100.0
    frame = frame.dropna(subset=["observation_date", "close_level"])
    frame = frame.loc[frame["close_level"].gt(0)].copy()
    if not frame["indexCode"].astype(str).eq(frame["requested_index_code"].astype(str)).all():
        raise RuntimeError("CSIndex response index code does not match request")
    frame = frame.sort_values(["benchmark", "return_type", "observation_date"])
    frame["benchmark_return"] = frame.groupby(
        ["benchmark", "return_type"], sort=False
    )["close_level"].pct_change(fill_method=None)
    frame["vendor_return_crosscheck_error"] = (
        frame["benchmark_return"] - frame["vendor_change_return"]
    )
    frame["publication_date"] = frame["observation_date"]
    frame["available_from_date"] = frame["publication_date"] + pd.Timedelta(
        days=int(cfg["download"]["availability_lag_calendar_days"])
    )
    frame["index_code"] = frame["indexCode"].astype(str)
    frame["index_name_cn"] = frame["indexNameCnAll"].astype(str)
    frame["index_name_en"] = frame["indexNameEnAll"].astype(str)
    frame["variant_id"] = frame["return_type"].map({
        "price_return": "official_csindex_price",
        "total_return": "official_csindex_total_return",
    })
    frame["source_provider"] = str(cfg["download"]["provider"])
    frame["source_name"] = str(cfg["download"]["source_name"])
    frame["source_url"] = str(cfg["download"]["source_url"])
    frame["api_url"] = str(cfg["download"]["api_url"])
    frame["downloaded_at_utc"] = pulled_at_utc
    frame["publication_timestamp_status"] = str(
        cfg["download"]["publication_timestamp_status"]
    )
    frame["availability_lag_calendar_days"] = int(
        cfg["download"]["availability_lag_calendar_days"]
    )
    frame["source_missing_flag"] = False
    frame["interpolation_flag"] = False
    keep = [
        "benchmark", "variant_id", "return_type", "index_code", "index_name_cn",
        "index_name_en", "observation_date", "publication_date", "available_from_date",
        "close_level", "benchmark_return", "vendor_change_return",
        "vendor_return_crosscheck_error", "source_provider", "source_name", "source_url",
        "api_url", "downloaded_at_utc", "publication_timestamp_status",
        "availability_lag_calendar_days", "source_missing_flag", "interpolation_flag",
    ]
    result = frame[keep].drop_duplicates(
        ["benchmark", "variant_id", "observation_date"], keep="last"
    )
    if result.duplicated(["benchmark", "variant_id", "observation_date"]).any():
        raise RuntimeError("duplicate official benchmark observations")
    for column in ["observation_date", "publication_date", "available_from_date"]:
        result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result.sort_values(
        ["benchmark", "variant_id", "observation_date"]
    ).reset_index(drop=True)


def build_benchmark_daily(
    lake: Lake, observations: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Align official and retained legacy series to exactly the strategy dates."""
    capm = lake.read("capm_aligned_daily", "v1")
    if capm.empty:
        raise RuntimeError("capm_aligned_daily/v1 is missing")
    strategy_dates = sorted(capm["trade_date"].astype(str).unique())
    official = observations.loc[observations["observation_date"].isin(strategy_dates)].copy()
    official = official.rename(columns={"observation_date": "trade_date"})
    counts = official.groupby(["benchmark", "variant_id"])["trade_date"].nunique()
    expected = len(strategy_dates)
    if len(counts) != 6 or not counts.eq(expected).all():
        raise RuntimeError(
            f"official benchmark coverage incomplete; expected {expected}: {counts.to_dict()}"
        )
    unique_legacy = capm[["trade_date", "CSI300", "CSI500", "CSI1000"]].drop_duplicates()
    if unique_legacy["trade_date"].duplicated().any():
        raise RuntimeError("legacy benchmark return differs across strategy rows")
    index_lookup = {str(item["benchmark"]): item for item in cfg["indices"]}
    legacy_records = []
    for benchmark in ["CSI300", "CSI500", "CSI1000"]:
        current = unique_legacy[["trade_date", benchmark]].rename(
            columns={benchmark: "benchmark_return"}
        )
        current["benchmark"] = benchmark
        current["variant_id"] = "legacy_baostock_price"
        current["return_type"] = "price_return"
        current["index_code"] = str(index_lookup[benchmark]["legacy_price_instrument"])
        current["index_name_cn"] = benchmark
        current["index_name_en"] = benchmark
        current["publication_date"] = current["trade_date"]
        current["available_from_date"] = pd.to_datetime(
            current["trade_date"]
        ).add(pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        current["close_level"] = np.nan
        current["vendor_change_return"] = current["benchmark_return"]
        current["vendor_return_crosscheck_error"] = 0.0
        current["source_provider"] = "BaoStock"
        current["source_name"] = "validated BaoStock index pct_chg legacy baseline"
        current["source_url"] = "http://www.baostock.com/"
        current["api_url"] = "not_applicable_existing_lake_product"
        current["downloaded_at_utc"] = "retained_from_existing_lake_manifest"
        current["publication_timestamp_status"] = "post_trade_attribution_only"
        current["availability_lag_calendar_days"] = 1
        current["source_missing_flag"] = False
        current["interpolation_flag"] = False
        legacy_records.append(current)
    legacy = pd.concat(legacy_records, ignore_index=True)
    columns = list(legacy.columns)
    result = pd.concat([legacy, official[columns]], ignore_index=True)
    if result["benchmark_return"].isna().any():
        raise RuntimeError("benchmark return missing on strategy clock")
    if result.duplicated(["trade_date", "benchmark", "variant_id"]).any():
        raise RuntimeError("duplicate benchmark daily key")
    return result.sort_values(["benchmark", "variant_id", "trade_date"]).reset_index(drop=True)


def coverage_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby(
        ["benchmark", "variant_id", "return_type", "index_code", "source_provider"],
        as_index=False, sort=False,
    ).agg(
        dates=("trade_date", "nunique"),
        start_date=("trade_date", "min"),
        end_date=("trade_date", "max"),
        missing_returns=("benchmark_return", lambda values: int(values.isna().sum())),
        interpolated_dates=("interpolation_flag", "sum"),
    )


def benchmark_comparison(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    annualization = int(cfg["annualization_days"])
    material_threshold = float(cfg["material_daily_total_minus_price_threshold"])
    records = []
    for benchmark, group in daily.groupby("benchmark", sort=False):
        official = group.loc[group["variant_id"].isin([
            cfg["same_provider_price_variant"], cfg["primary_total_return_variant"]
        ])].pivot(index="trade_date", columns="variant_id", values="benchmark_return")
        price = official[str(cfg["same_provider_price_variant"])]
        total = official[str(cfg["primary_total_return_variant"])]
        observations = len(official)
        price_wealth = float(np.prod(1.0 + price))
        total_wealth = float(np.prod(1.0 + total))
        records.append({
            "benchmark": benchmark,
            "observations": observations,
            "official_price_annualized_return_geometric": (
                price_wealth ** (annualization / observations) - 1.0
            ),
            "official_total_return_annualized_return_geometric": (
                total_wealth ** (annualization / observations) - 1.0
            ),
            "annualized_arithmetic_total_minus_price": float(
                (total - price).mean() * annualization
            ),
            "annualized_geometric_total_relative_to_price": float(
                (total_wealth / price_wealth) ** (annualization / observations) - 1.0
            ),
            "maximum_abs_daily_total_minus_price": float((total - price).abs().max()),
            "material_daily_difference_threshold": material_threshold,
            "days_abs_total_minus_price_above_threshold": int(
                (total - price).abs().gt(material_threshold).sum()
            ),
        })
    return pd.DataFrame(records)


def source_reconciliation(
    daily: pd.DataFrame, observations: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    records = []
    tolerance = float(cfg["official_change_pct_tolerance"])
    for benchmark, group in daily.groupby("benchmark", sort=False):
        wide = group.pivot(index="trade_date", columns="variant_id", values="benchmark_return")
        official = wide[str(cfg["same_provider_price_variant"])]
        legacy = wide[str(cfg["legacy_price_variant"])]
        errors = official - legacy
        raw = observations.loc[
            observations["benchmark"].eq(benchmark)
            & observations["variant_id"].eq(cfg["same_provider_price_variant"])
            & observations["benchmark_return"].notna()
        ]
        crosscheck_max = float(raw["vendor_return_crosscheck_error"].abs().max())
        records.append({
            "benchmark": benchmark,
            "observations": len(wide),
            "official_price_vs_legacy_correlation": float(official.corr(legacy)),
            "official_price_vs_legacy_mean_error": float(errors.mean()),
            "official_price_vs_legacy_mean_abs_error": float(errors.abs().mean()),
            "official_price_vs_legacy_max_abs_error": float(errors.abs().max()),
            "official_close_return_vs_published_change_pct_max_abs_error": crosscheck_max,
            "published_change_pct_rounding_tolerance": tolerance,
            "published_change_pct_crosscheck_passed": crosscheck_max <= tolerance,
        })
    result = pd.DataFrame(records)
    if not result["published_change_pct_crosscheck_passed"].all():
        raise RuntimeError(f"official return crosscheck failed:\n{result.to_string(index=False)}")
    return result


def attribution_panel(lake: Lake, benchmark_daily: pd.DataFrame) -> pd.DataFrame:
    base = lake.read("risk_free_rate_attribution_daily", "v1")
    if base.empty:
        raise RuntimeError("risk_free_rate_attribution_daily/v1 is missing")
    market_columns = [
        "trade_date", "benchmark", "variant_id", "return_type", "index_code",
        "index_name_cn", "index_name_en", "benchmark_return", "source_provider",
    ]
    panel = base.merge(benchmark_daily[market_columns], on="trade_date", how="left")
    if panel["benchmark_return"].isna().any():
        raise RuntimeError("market benchmark missing in attribution panel")
    panel["benchmark_excess_return"] = (
        panel["benchmark_return"] - panel["risk_free_return"]
    )
    panel["arithmetic_excess_vs_benchmark"] = (
        panel["net_return"] - panel["benchmark_return"]
    )
    panel["geometric_relative_vs_benchmark"] = (
        (1.0 + panel["net_return"]) / (1.0 + panel["benchmark_return"]) - 1.0
    )
    return panel


def relative_performance(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = [
        "rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario",
        "benchmark", "variant_id", "return_type", "index_code",
    ]
    annualization = int(cfg["annualization_days"])
    records = []
    for values, group in panel.groupby(keys, sort=False):
        strategy = group["net_return"].to_numpy(dtype="float64")
        benchmark = group["benchmark_return"].to_numpy(dtype="float64")
        difference = strategy - benchmark
        observations = len(group)
        records.append({
            **dict(zip(keys, values)),
            "observations": observations,
            "benchmark_annualized_return_geometric": float(
                np.prod(1.0 + benchmark) ** (annualization / observations) - 1.0
            ),
            "strategy_annualized_return_geometric": float(
                np.prod(1.0 + strategy) ** (annualization / observations) - 1.0
            ),
            "annualized_arithmetic_excess": float(difference.mean() * annualization),
            "annualized_geometric_relative_return": float(
                (np.prod(1.0 + strategy) / np.prod(1.0 + benchmark))
                ** (annualization / observations) - 1.0
            ),
            "information_ratio": float(
                difference.mean() / difference.std(ddof=1) * np.sqrt(annualization)
            ),
        })
    return pd.DataFrame(records)


def capm_summaries(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = [
        "rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario",
        "benchmark", "variant_id", "return_type", "index_code",
    ]
    records = []
    for values, group in panel.groupby(keys, sort=False):
        records.append({
            **dict(zip(keys, values)),
            **capm_record(
                group["strategy_excess_return"], group["benchmark_excess_return"],
                lags=int(cfg["newey_west_lags"]),
                annualization_days=int(cfg["annualization_days"]),
                confidence_level=float(cfg["confidence_level"]),
            ),
        })
    return pd.DataFrame(records)


def style_summaries(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    primary = panel.loc[panel["benchmark"].eq(cfg["primary_benchmark"])].copy()
    keys = [
        "rate_scenario", "model", "alpha_pipeline", "strategy", "execution_scenario",
        "benchmark", "variant_id", "return_type", "index_code",
    ]
    styles = ["size", "beta", "volatility", "liquidity", "value", "momentum"]
    style_cfg = {
        "styles": styles,
        "newey_west_lags": int(cfg["newey_west_lags"]),
        "annualization_days": int(cfg["annualization_days"]),
        "confidence_level": float(cfg["confidence_level"]),
    }
    records = []
    for values, group in primary.groupby(keys, sort=False):
        factors = group[["benchmark_excess_return", *styles]].rename(
            columns={"benchmark_excess_return": "market"}
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
    legacy_variant = str(cfg["legacy_price_variant"])
    checks = []
    comparisons = [
        (
            "risk_free_rate_capm_summary_v1",
            lake.read("risk_free_rate_capm_summary", "v1"),
            capm.loc[capm["variant_id"].eq(legacy_variant)],
            [
                "rate_scenario", "model", "alpha_pipeline", "strategy",
                "execution_scenario", "benchmark",
            ],
        ),
        (
            "risk_free_rate_style_summary_v1",
            lake.read("risk_free_rate_style_summary", "v1"),
            style.loc[style["variant_id"].eq(legacy_variant)],
            [
                "rate_scenario", "model", "alpha_pipeline", "strategy",
                "execution_scenario",
            ],
        ),
    ]
    for name, legacy, current, keys in comparisons:
        numeric = [
            column for column in legacy.select_dtypes(include=[np.number]).columns
            if column in current.columns
        ]
        merged = legacy[keys + numeric].merge(
            current[keys + numeric], on=keys, how="outer",
            suffixes=("_legacy", "_recomputed"), indicator=True,
        )
        errors = [
            np.abs(
                merged[f"{column}_legacy"] - merged[f"{column}_recomputed"]
            ).to_numpy(dtype="float64")
            for column in numeric
        ]
        maximum = float(np.nanmax(np.concatenate(errors))) if errors else np.nan
        passed = bool(merged["_merge"].eq("both").all() and maximum <= tolerance)
        checks.append({
            "check": name,
            "legacy_rows": len(legacy),
            "recomputed_rows": len(current),
            "numeric_columns_checked": len(numeric),
            "maximum_absolute_error": maximum,
            "tolerance": tolerance,
            "passed": passed,
        })
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        raise RuntimeError(f"legacy price-index regression failed:\n{result.to_string(index=False)}")
    return result


def protocol_frame(cfg: dict, daily: pd.DataFrame) -> pd.DataFrame:
    roles = {str(item["variant_id"]): str(item["role"]) for item in cfg["benchmark_variants"]}
    rows = daily[[
        "benchmark", "variant_id", "return_type", "index_code", "source_provider"
    ]].drop_duplicates()
    rows["role"] = rows["variant_id"].map(roles)
    rows["experiment_id"] = str(cfg["experiment_id"])
    rows["frequency"] = "daily_nonoverlapping"
    rows["return_calculation"] = "close_level_t_divided_by_close_level_t_minus_1_minus_1"
    rows["dividend_treatment"] = np.where(
        rows["return_type"].eq("total_return"),
        "pre_tax_cash_dividend_reinvestment_per_CSIndex_methodology",
        "cash_dividends_excluded",
    )
    rows["post_trade_attribution_only"] = True
    rows["strategy_dependent_variable"] = "net_return_minus_same_risk_free_return"
    rows["market_factor"] = "benchmark_return_minus_same_risk_free_return"
    rows["standard_errors"] = f"Newey-West({cfg['newey_west_lags']}) Bartlett HAC"
    rows["alpha_annualization"] = f"arithmetic_daily_alpha_times_{cfg['annualization_days']}"
    rows["historical_oos_used_for_selection"] = False
    return rows.sort_values(["benchmark", "variant_id"]).reset_index(drop=True)


def write_report(
    cfg: dict, coverage: pd.DataFrame, comparison: pd.DataFrame,
    reconciliation: pd.DataFrame, relative: pd.DataFrame, capm: pd.DataFrame,
    style: pd.DataFrame, checks: pd.DataFrame,
) -> None:
    selection = (
        relative["rate_scenario"].eq(cfg["primary_rate_scenario"])
        & relative["execution_scenario"].eq(cfg["primary_execution_scenario"])
        & relative["benchmark"].eq(cfg["primary_benchmark"])
    )
    selected_relative = relative.loc[selection, [
        "variant_id", "return_type", "model", "alpha_pipeline", "observations",
        "benchmark_annualized_return_geometric", "strategy_annualized_return_geometric",
        "annualized_arithmetic_excess", "annualized_geometric_relative_return",
        "information_ratio",
    ]]
    selected_capm = capm.loc[
        capm["rate_scenario"].eq(cfg["primary_rate_scenario"])
        & capm["execution_scenario"].eq(cfg["primary_execution_scenario"])
        & capm["benchmark"].eq(cfg["primary_benchmark"]),
        [
            "variant_id", "return_type", "model", "alpha_pipeline", "observations",
            "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
            "annualized_alpha_ci_high", "alpha_hac_t_stat", "beta", "r_squared",
        ],
    ]
    selected_style = style.loc[
        style["rate_scenario"].eq(cfg["primary_rate_scenario"])
        & style["execution_scenario"].eq(cfg["primary_execution_scenario"]),
        [
            "variant_id", "return_type", "model", "alpha_pipeline", "observations",
            "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
            "annualized_alpha_ci_high", "alpha_hac_t_stat", "r_squared",
        ],
    ]
    report = "\n".join([
        "# 价格指数与全收益指数双口径归因", "", "## 研究边界", "",
        "本实验仅替换市场基准收益定义。策略、持仓、成交、成本、无风险利率、风格因子和日期均不变；"
        "2021–2026仍是已查看的historical OOS，没有据此重新选择模型。旧BaoStock价格指数结果完整保留。",
        "", "## 数据与可比性设计", "",
        "- `legacy_baostock_price`：原价格指数序列，只用于历史结果零漂移核对。",
        "- `official_csindex_price`：中证指数有限公司官网价格指数收盘点位计算的收益。",
        "- `official_csindex_total_return`：同一官网全收益指数收盘点位计算的收益；按中证规则计入税前现金红利再投资。",
        "- 股息效应只比较后两个同源序列；不能把BaoStock与CSIndex之间的供应商/精度差异解释成股息。",
        "- 官方接口的`changePct`仅保留两位百分数。本实验用收盘点位重算收益，并把该字段作为舍入容差内的交叉核对。",
        "", "## 覆盖", "", coverage.to_markdown(index=False),
        "", "## 官方价格与全收益差异", "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "", "## 供应商与收益字段核对", "",
        reconciliation.to_markdown(index=False, floatfmt=".8f"),
        "", f"## 相对 {cfg['primary_benchmark']} 的收益效率（R1 3M国债rf）", "",
        "算术超额收益为策略减基准的日收益均值乘252；几何相对收益按两条财富曲线之比年化，二者不混用。",
        "", selected_relative.to_markdown(index=False, floatfmt=".6f"),
        "", f"## CAPM（R1 3M国债rf，{cfg['primary_benchmark']}）", "",
        f"日频含常数OLS；左右两侧均减同一rf；Newey–West({cfg['newey_west_lags']}) Bartlett HAC；"
        f"Alpha按{cfg['annualization_days']}日算术年化。",
        "", selected_capm.to_markdown(index=False, floatfmt=".6f"),
        "", "## CSI500 + 六风格归因", "",
        "市场因子使用相应价格/全收益指数超额收益；六个零净投资风格因子不重复减rf。",
        "", selected_style.to_markdown(index=False, floatfmt=".6f"),
        "", "## 历史价格指数回归核对", "", checks.to_markdown(index=False, floatfmt=".3e"),
        "", "## 解释限制", "",
        "全收益指数比价格指数更适合衡量持有股票基准的经济回报，但仍是理论指数：没有ETF费率、"
        "跟踪误差、税费和实际再投资摩擦。价格与全收益结果均应保留；前者用于和旧研究连续比较，"
        "后者作为主要经济归因口径。任何Alpha变化首先是基准定义变化，不是模型产生了新的Alpha。", "",
    ])
    destination = ROOT / "reports" / "market_benchmark_attribution.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables / "market_benchmark_comparison.csv", index=False)
    reconciliation.to_csv(tables / "market_benchmark_reconciliation.csv", index=False)
    selected_relative.to_csv(tables / "market_benchmark_relative_performance.csv", index=False)
    selected_capm.to_csv(tables / "market_benchmark_capm.csv", index=False)
    selected_style.to_csv(tables / "market_benchmark_style_attribution.csv", index=False)


def run(
    lake: Lake | None = None,
    *,
    reuse_observations: bool = False,
    fetcher: Callable[[dict], tuple[pd.DataFrame, str]] = fetch_csindex_history,
) -> None:
    lake = lake or Lake()
    cfg = load_config()
    existing = lake.read("market_benchmark_observations", "v1")
    if reuse_observations:
        if existing.empty:
            raise RuntimeError("--reuse-observations requested but benchmark snapshot is missing")
        observations = existing
    else:
        raw, pulled_at = fetcher(cfg)
        observations = normalize_official_observations(raw, cfg, pulled_at)
        lake.write(
            "market_benchmark_observations", observations, key="v1",
            source="csindex.official_index_performance_history", request=cfg["download"],
        )
    daily = build_benchmark_daily(lake, observations, cfg)
    coverage = coverage_summary(daily)
    comparison = benchmark_comparison(daily, cfg)
    reconciliation = source_reconciliation(daily, observations, cfg)
    panel = attribution_panel(lake, daily)
    relative = relative_performance(panel, cfg)
    capm = capm_summaries(panel, cfg)
    style = style_summaries(panel, cfg)
    checks = regression_checks(lake, capm, style, cfg)
    protocol = protocol_frame(cfg, daily)
    request = cfg.copy()
    for table, frame in {
        "market_benchmark_daily": daily,
        "market_benchmark_coverage": coverage,
        "market_benchmark_comparison": comparison,
        "market_benchmark_reconciliation": reconciliation,
        "market_benchmark_protocol": protocol,
        "market_benchmark_relative_performance": relative,
        "market_benchmark_capm_summary": capm,
        "market_benchmark_style_summary": style,
        "market_benchmark_regression_checks": checks,
    }.items():
        lake.write(
            table, frame, key="v1", source="market_benchmark_attribution.v1", request=request
        )
    write_report(cfg, coverage, comparison, reconciliation, relative, capm, style, checks)
    print(coverage.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(checks.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official price and total-return benchmark attribution"
    )
    parser.add_argument("--reuse-observations", action="store_true")
    args = parser.parse_args()
    run(reuse_observations=args.reuse_observations)


if __name__ == "__main__":
    main()
