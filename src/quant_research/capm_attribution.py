"""Daily executable-portfolio CAPM alpha, HAC inference, and rolling beta."""

from __future__ import annotations

import argparse
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "capm_attribution.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def newey_west_ols(
    y: np.ndarray, x: np.ndarray, lags: int,
) -> dict[str, np.ndarray | float | int]:
    """OLS coefficients and Bartlett-kernel Newey-West covariance.

    ``x`` must include any desired intercept column. The covariance uses the
    finite-sample n/(n-k) correction and no prewhitening.
    """
    dependent = np.asarray(y, dtype="float64").reshape(-1)
    design = np.asarray(x, dtype="float64")
    if design.ndim != 2 or len(dependent) != len(design):
        raise ValueError("y and x dimensions do not align")
    usable = np.isfinite(dependent) & np.isfinite(design).all(axis=1)
    dependent, design = dependent[usable], design[usable]
    observations, parameters = design.shape
    if observations <= parameters or observations <= lags:
        raise ValueError("not enough observations for HAC regression")
    gram_inverse = np.linalg.pinv(design.T @ design)
    coefficients = gram_inverse @ design.T @ dependent
    residual = dependent - design @ coefficients
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(int(lags), observations - 1) + 1):
        weight = 1.0 - lag / (int(lags) + 1.0)
        covariance = scores[lag:].T @ scores[:-lag]
        meat += weight * (covariance + covariance.T)
    hac_covariance = gram_inverse @ meat @ gram_inverse
    hac_covariance *= observations / (observations - parameters)
    standard_errors = np.sqrt(np.maximum(np.diag(hac_covariance), 0.0))
    total = float(np.sum((dependent - dependent.mean()) ** 2))
    residual_sum = float(residual @ residual)
    r_squared = 1.0 - residual_sum / total if total > 0 else np.nan
    return {
        "observations": observations,
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "covariance": hac_covariance,
        "residuals": residual,
        "r_squared": float(r_squared),
    }


def capm_record(
    strategy_return: pd.Series,
    benchmark_return: pd.Series,
    *,
    lags: int,
    annualization_days: int,
    confidence_level: float,
) -> dict[str, float | int]:
    strategy = pd.to_numeric(strategy_return, errors="coerce").to_numpy(dtype="float64")
    benchmark = pd.to_numeric(benchmark_return, errors="coerce").to_numpy(dtype="float64")
    result = newey_west_ols(
        strategy, np.column_stack([np.ones(len(benchmark)), benchmark]), lags
    )
    coefficients = np.asarray(result["coefficients"])
    standard_errors = np.asarray(result["standard_errors"])
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    daily_alpha, beta = float(coefficients[0]), float(coefficients[1])
    daily_alpha_se, beta_se = float(standard_errors[0]), float(standard_errors[1])
    annual_alpha = daily_alpha * annualization_days
    annual_alpha_se = daily_alpha_se * annualization_days
    residuals = np.asarray(result["residuals"])
    alpha_t = daily_alpha / daily_alpha_se if daily_alpha_se > 0 else np.nan
    beta_t = (beta - 1.0) / beta_se if beta_se > 0 else np.nan
    return {
        "observations": int(result["observations"]),
        "daily_alpha": daily_alpha,
        "annualized_alpha_arithmetic": annual_alpha,
        "annualized_alpha_hac_se": annual_alpha_se,
        "annualized_alpha_ci_low": annual_alpha - critical * annual_alpha_se,
        "annualized_alpha_ci_high": annual_alpha + critical * annual_alpha_se,
        "alpha_hac_t_stat": alpha_t,
        "beta": beta,
        "beta_hac_se": beta_se,
        "beta_ci_low": beta - critical * beta_se,
        "beta_ci_high": beta + critical * beta_se,
        "beta_equals_one_hac_t_stat": beta_t,
        "r_squared": float(result["r_squared"]),
        "annualized_residual_volatility": float(
            residuals.std(ddof=2) * np.sqrt(annualization_days)
        ),
        "mean_daily_strategy_return": float(np.nanmean(strategy)),
        "mean_daily_benchmark_return": float(np.nanmean(benchmark)),
        "newey_west_lags": int(lags),
    }


def benchmark_panel(lake: Lake, cfg: dict) -> pd.DataFrame:
    panel: pd.DataFrame | None = None
    for item in cfg["benchmarks"]:
        name, instrument = str(item["name"]), str(item["instrument_id"])
        bars = lake.read("daily_bars_validated", instrument.replace(".", "_"))
        if bars.empty:
            raise RuntimeError(f"validated benchmark bars missing: {instrument}")
        current = bars[["trade_date", "pct_chg"]].copy()
        current[name] = pd.to_numeric(current.pop("pct_chg"), errors="coerce") / 100.0
        current = current.drop_duplicates("trade_date", keep="last")
        panel = current if panel is None else panel.merge(
            current, on="trade_date", how="outer", validate="one_to_one"
        )
    if panel is None:
        raise RuntimeError("no benchmarks configured")
    return panel.sort_values("trade_date").reset_index(drop=True)


def aligned_returns(lake: Lake, cfg: dict) -> pd.DataFrame:
    daily = lake.read("constrained_daily_execution_daily", "v1")
    if daily.empty:
        raise RuntimeError("constrained daily execution returns missing")
    benchmarks = benchmark_panel(lake, cfg)
    result = daily.merge(benchmarks, on="trade_date", how="left", validate="many_to_one")
    benchmark_names = [str(item["name"]) for item in cfg["benchmarks"]]
    missing = result[benchmark_names].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f"benchmark returns missing on strategy dates: {missing.to_dict()}")
    return result


def full_sample_summary(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records: list[dict] = []
    keys = ["model", "alpha_pipeline", "strategy", "scenario"]
    for values, group in panel.groupby(keys, sort=False):
        group = group.sort_values("trade_date")
        for benchmark in cfg["benchmarks"]:
            name = str(benchmark["name"])
            record = dict(zip(keys, values))
            record.update({"benchmark": name, **capm_record(
                group["net_return"], group[name],
                lags=int(cfg["newey_west_lags"]),
                annualization_days=int(cfg["annualization_days"]),
                confidence_level=float(cfg["confidence_level"]),
            )})
            records.append(record)
    return pd.DataFrame(records)


def annual_summary(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frame = panel.copy()
    frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
    records: list[dict] = []
    keys = ["year", "model", "alpha_pipeline", "strategy", "scenario"]
    for values, group in frame.groupby(keys, sort=False):
        group = group.sort_values("trade_date")
        for benchmark in cfg["benchmarks"]:
            name = str(benchmark["name"])
            record = dict(zip(keys, values))
            record.update({"benchmark": name, **capm_record(
                group["net_return"], group[name],
                lags=int(cfg["newey_west_lags"]),
                annualization_days=int(cfg["annualization_days"]),
                confidence_level=float(cfg["confidence_level"]),
            )})
            records.append(record)
    return pd.DataFrame(records)


def rolling_summary(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    primary = str(cfg["primary_benchmark"])
    window = int(cfg["rolling"]["window_trading_days"])
    step = int(cfg["rolling"]["step_trading_days"])
    records: list[dict] = []
    keys = ["model", "alpha_pipeline", "strategy", "scenario"]
    for values, group in panel.groupby(keys, sort=False):
        group = group.sort_values("trade_date").reset_index(drop=True)
        endpoints = list(range(window - 1, len(group), step))
        if endpoints[-1] != len(group) - 1:
            endpoints.append(len(group) - 1)
        for endpoint in endpoints:
            sample = group.iloc[endpoint - window + 1:endpoint + 1]
            record = dict(zip(keys, values))
            record.update({
                "benchmark": primary,
                "window_start": str(sample["trade_date"].iloc[0]),
                "window_end": str(sample["trade_date"].iloc[-1]),
                **capm_record(
                    sample["net_return"], sample[primary],
                    lags=int(cfg["newey_west_lags"]),
                    annualization_days=int(cfg["annualization_days"]),
                    confidence_level=float(cfg["confidence_level"]),
                ),
            })
            records.append(record)
    return pd.DataFrame(records)


def write_report(
    cfg: dict, summary: pd.DataFrame, annual: pd.DataFrame, rolling: pd.DataFrame,
) -> None:
    primary = summary.loc[
        summary["benchmark"].eq(cfg["primary_benchmark"])
        & summary["scenario"].eq(cfg["primary_scenario"])
    ].copy().sort_values(["model", "alpha_pipeline"])
    annual_primary = annual.loc[
        annual["benchmark"].eq(cfg["primary_benchmark"])
        & annual["scenario"].eq(cfg["primary_scenario"])
    ]
    positive_years = annual_primary.groupby(["model", "alpha_pipeline"])[
        "annualized_alpha_arithmetic"
    ].apply(lambda values: int(values.gt(0).sum()))
    rolling_ranges = rolling.loc[
        rolling["scenario"].eq(cfg["primary_scenario"])
    ].groupby(["model", "alpha_pipeline"]).agg(
        rolling_beta_min=("beta", "min"),
        rolling_beta_max=("beta", "max"),
        rolling_alpha_min=("annualized_alpha_arithmetic", "min"),
        rolling_alpha_max=("annualized_alpha_arithmetic", "max"),
    )
    primary = primary.merge(
        positive_years.rename("positive_alpha_years"),
        on=["model", "alpha_pipeline"], validate="one_to_one"
    ).merge(
        rolling_ranges, on=["model", "alpha_pipeline"], validate="one_to_one"
    )
    display = primary[[
        "model", "alpha_pipeline", "observations", "annualized_alpha_arithmetic",
        "annualized_alpha_ci_low", "annualized_alpha_ci_high", "alpha_hac_t_stat",
        "beta", "beta_ci_low", "beta_ci_high", "r_squared",
        "positive_alpha_years", "rolling_beta_min", "rolling_beta_max",
    ]].copy()
    numeric = [column for column in display if column not in {
        "model", "alpha_pipeline", "observations", "positive_alpha_years"
    }]
    for column in numeric:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    significant = primary.loc[
        primary["annualized_alpha_ci_low"].gt(0), ["model", "alpha_pipeline"]
    ].apply(lambda row: f"{row['model']}/{row['alpha_pipeline']}", axis=1).tolist()
    report = "\n".join([
        "# 日频可成交组合 CAPM Alpha 与滚动 Beta",
        "",
        "## 研究口径",
        "",
        "- 收益：约束组合的唯一非重叠日频净值时钟，包含20bp换手成本。",
        "- 主结果：停牌/涨跌停订单逐日重试情景，相对 CSI500 日收益。",
        "- 稳健性：同样回归 CSI300、CSI1000，并保留理想全成交上界。",
        f"- 推断：OLS + Newey–West({cfg['newey_west_lags']})；Alpha 按252日算术年化。",
        "- 无风险利率按0处理；指数为价格收益、不含股息，因此 Alpha 可能被上估。",
        "- 2021–2026 是已查看的 historical OOS，不用于本实验重新选择模型或参数。",
        "",
        "## 主结果",
        "",
        f"- 相对 CSI500、95% Alpha 区间下界为正：{', '.join(significant) or '无'}。",
        "- 年度正 Alpha 数量和滚动 Beta 区间用于识别时间稳定性，不用于事后择时。",
        "",
        display.to_markdown(index=False),
        "",
        "## 限制",
        "",
        "单因子 CAPM 不能吸收规模、价值、动量、流动性等本地风格溢价。这里的 Alpha 是"
        "相对价格指数的一因子截距，不等同于纯个股选股能力；应与本地多因子归因联合解释。",
        "",
    ])
    destination = ROOT / "reports" / "capm_attribution.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    table = ROOT / "reports" / "tables" / "capm_attribution.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(table, index=False)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    panel = aligned_returns(lake, cfg)
    summary = full_sample_summary(panel, cfg)
    annual = annual_summary(panel, cfg)
    rolling = rolling_summary(panel, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "source_experiment": cfg["source_experiment"],
        "period": cfg["period"],
        "historical_oos_used_for_selection": False,
        "strategy_return": cfg["strategy_return"],
        "risk_free_rate_convention": cfg["risk_free_rate_convention"],
        "annualization_days": int(cfg["annualization_days"]),
        "newey_west_lags": int(cfg["newey_west_lags"]),
        "primary_benchmark": cfg["primary_benchmark"],
        "primary_scenario": cfg["primary_scenario"],
        "benchmark_price_returns_exclude_dividends": True,
        "unique_nonoverlap_daily_clock": True,
        "daily_dates": int(panel["trade_date"].nunique()),
    }])
    request = cfg.copy()
    for table, frame in {
        "capm_protocol": protocol,
        "capm_aligned_daily": panel,
        "capm_summary": summary,
        "capm_annual": annual,
        "capm_rolling": rolling,
    }.items():
        lake.write(table, frame, key="v1", source="capm_attribution.v1", request=request)
    write_report(cfg, summary, annual, rolling)
    print(summary.loc[
        summary["benchmark"].eq(cfg["primary_benchmark"])
        & summary["scenario"].eq(cfg["primary_scenario"])
    ].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate daily executable-portfolio CAPM alpha and rolling beta"
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
