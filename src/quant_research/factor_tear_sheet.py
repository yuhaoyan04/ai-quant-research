"""Historical-OOS single-factor decile, turnover, stability, and regime tear sheet."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_neutralization import _read_year, transformed_factor_designs
from quant_research.ic_decay import newey_west_mean_t
from quant_research.ingest import ROOT, Lake
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "factor_tear_sheet.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def equal_weight_turnover(previous: set[str] | None, current: set[str]) -> float:
    """One-way target-weight turnover for two equal-weight portfolios."""
    if previous is None:
        return np.nan
    if not previous or not current:
        return 1.0
    overlap = len(previous & current)
    shared_weight = overlap * min(1.0 / len(previous), 1.0 / len(current))
    return float(1.0 - shared_weight)


def assign_buckets(values: np.ndarray, bucket_count: int) -> np.ndarray:
    """Assign deterministic, near-equal sized buckets from low to high signal."""
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least two")
    count = len(values)
    if count < bucket_count:
        raise ValueError("cross section is smaller than bucket count")
    order = np.argsort(values, kind="stable")
    buckets = np.empty(count, dtype="int64")
    buckets[order] = np.floor(np.arange(count) * bucket_count / count).astype(int) + 1
    return buckets


def benchmark_regimes(lake: Lake, cfg: dict, dates: list[str]) -> pd.DataFrame:
    """Build signal-date regimes exclusively from trailing benchmark observations."""
    instrument = str(cfg["benchmark_instrument"])
    bars = lake.read("daily_bars_validated", instrument.replace(".", "_"))
    if bars.empty:
        raise RuntimeError(f"validated benchmark bars missing: {instrument}")
    lookback = int(cfg["regime"]["lookback_trading_days"])
    minimum = int(cfg["regime"]["expanding_median_min_periods"])
    frame = bars[["trade_date", "close"]].copy().sort_values("trade_date")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    close = pd.to_numeric(frame["close"], errors="coerce")
    daily_return = close.pct_change()
    frame["trailing_return"] = close / close.shift(lookback) - 1.0
    frame["trailing_volatility"] = (
        daily_return.rolling(lookback, min_periods=lookback).std(ddof=1)
        * np.sqrt(int(cfg["regime"]["annualization_days"]))
    )
    # Shift makes the volatility threshold itself known before the current day.
    frame["volatility_threshold"] = frame["trailing_volatility"].expanding(
        min_periods=minimum
    ).median().shift(1)
    frame = frame.dropna(subset=[
        "trailing_return", "trailing_volatility", "volatility_threshold"
    ])
    signals = pd.DataFrame({"asof_date": pd.to_datetime(dates)}).sort_values("asof_date")
    merged = pd.merge_asof(
        signals, frame, left_on="asof_date", right_on="trade_date", direction="backward"
    )
    if merged[["trailing_return", "trailing_volatility"]].isna().any().any():
        raise RuntimeError("benchmark regime unavailable on one or more signal dates")
    merged["trend_regime"] = np.where(merged["trailing_return"].ge(0), "up", "down")
    merged["volatility_regime"] = np.where(
        merged["trailing_volatility"].gt(merged["volatility_threshold"]), "high", "low"
    )
    merged["market_regime"] = merged["trend_regime"] + "_" + merged["volatility_regime"]
    merged["asof_date"] = merged["asof_date"].dt.strftime("%Y-%m-%d")
    merged["benchmark_trade_date"] = merged["trade_date"].dt.strftime("%Y-%m-%d")
    return merged.drop(columns=["trade_date", "close"])


def build_tear_sheet_panel(
    lake: Lake, cfg: dict, factors: list[str], directions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_map = directions.set_index("factor")["direction"].astype(int).to_dict()
    missing = sorted(set(factors) - set(direction_map))
    if missing:
        raise RuntimeError(f"training-locked factor directions missing: {missing}")
    bucket_count = int(cfg["bucket_count"])
    target = str(cfg["target"])
    selected_pipelines = list(cfg["pipelines"])
    weekly_records: list[dict] = []
    bucket_records: list[dict] = []
    previous: dict[tuple[str, str, str], set[str]] = {}
    columns = ["instrument_id", "asof_date", target, *factors]
    for year in range(2021, 2027):
        frame = _read_year(lake, year, columns)
        for asof_date, group in frame.groupby("asof_date", sort=True):
            group = group.reset_index(drop=True)
            returns = pd.to_numeric(group[target], errors="coerce").to_numpy(dtype="float64")
            designs, risks = transformed_factor_designs(group, factors, cfg)
            exposure_complete = risks["valid_size"] & risks["valid_industry"]
            instruments = group["instrument_id"].astype(str).to_numpy()
            for pipeline in selected_pipelines:
                design = designs[pipeline]
                for factor_index, factor in enumerate(factors):
                    values = design[:, factor_index] * direction_map[factor]
                    usable = exposure_complete & np.isfinite(values) & np.isfinite(returns)
                    if usable.sum() < int(cfg["minimum_cross_section_size"]):
                        continue
                    used_values = values[usable]
                    used_returns = returns[usable]
                    used_instruments = instruments[usable]
                    buckets = assign_buckets(used_values, bucket_count)
                    bucket_means = np.array([
                        used_returns[buckets == bucket].mean()
                        for bucket in range(1, bucket_count + 1)
                    ])
                    top = set(used_instruments[buckets == bucket_count])
                    bottom = set(used_instruments[buckets == 1])
                    top_key = (pipeline, factor, "top")
                    bottom_key = (pipeline, factor, "bottom")
                    top_turnover = equal_weight_turnover(previous.get(top_key), top)
                    bottom_turnover = equal_weight_turnover(previous.get(bottom_key), bottom)
                    previous[top_key], previous[bottom_key] = top, bottom
                    gross_spread = float(bucket_means[-1] - bucket_means[0])
                    cost = float(cfg["assumed_cost_bps_per_leg_turnover"]) / 10_000
                    transaction_cost = (
                        np.nan if not np.isfinite(top_turnover + bottom_turnover)
                        else cost * (top_turnover + bottom_turnover)
                    )
                    rank_ic = pd.Series(used_values).rank(method="average").corr(
                        pd.Series(used_returns).rank(method="average")
                    )
                    weekly_records.append({
                        "asof_date": str(asof_date), "year": year,
                        "pipeline": pipeline, "factor": factor,
                        "direction": direction_map[factor],
                        "cross_section_size": int(usable.sum()),
                        "oriented_rank_ic": float(rank_ic),
                        "gross_top_minus_bottom": gross_spread,
                        "top_turnover": top_turnover,
                        "bottom_turnover": bottom_turnover,
                        "long_short_turnover": (
                            np.nan if not np.isfinite(top_turnover + bottom_turnover)
                            else top_turnover + bottom_turnover
                        ),
                        "transaction_cost": transaction_cost,
                        "net_top_minus_bottom": gross_spread - transaction_cost
                        if np.isfinite(transaction_cost) else np.nan,
                        "same_date_bucket_monotonicity": pd.Series(
                            np.arange(1, bucket_count + 1)
                        ).corr(pd.Series(bucket_means), method="spearman"),
                    })
                    for bucket, bucket_return in enumerate(bucket_means, start=1):
                        bucket_records.append({
                            "asof_date": str(asof_date), "year": year,
                            "pipeline": pipeline, "factor": factor,
                            "bucket": bucket,
                            "stock_count": int((buckets == bucket).sum()),
                            "equal_weight_return": float(bucket_return),
                        })
        print(f"factor tear sheet year={year} dates={frame['asof_date'].nunique()}", flush=True)
    weekly = pd.DataFrame(weekly_records)
    buckets = pd.DataFrame(bucket_records)
    regimes = benchmark_regimes(lake, cfg, sorted(weekly["asof_date"].unique()))
    weekly = weekly.merge(regimes, on="asof_date", how="left", validate="many_to_one")
    buckets = buckets.merge(
        regimes[["asof_date", "market_regime"]], on="asof_date", how="left",
        validate="many_to_one",
    )
    return weekly, buckets


def _monotonicity(group: pd.DataFrame, bucket_count: int) -> tuple[float, float]:
    means = group.groupby("bucket")["equal_weight_return"].mean().reindex(
        range(1, bucket_count + 1)
    )
    correlation = pd.Series(range(1, bucket_count + 1)).corr(
        means.reset_index(drop=True), method="spearman"
    )
    adjacent = float(means.diff().dropna().gt(0).mean())
    return float(correlation), adjacent


def summarize(
    weekly: pd.DataFrame, buckets: pd.DataFrame, cfg: dict,
    grouping: list[str],
) -> pd.DataFrame:
    records: list[dict] = []
    annualization = int(cfg["annualization_periods"])
    bucket_count = int(cfg["bucket_count"])
    group_key: str | list[str] = grouping[0] if len(grouping) == 1 else grouping
    for keys, group in weekly.groupby(group_key, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        selector = np.ones(len(buckets), dtype=bool)
        for column, value in zip(grouping, keys):
            selector &= buckets[column].eq(value).to_numpy()
        selected_buckets = buckets.loc[selector]
        monotonicity, adjacent = _monotonicity(selected_buckets, bucket_count)
        ic_se, ic_t = newey_west_mean_t(
            group["oriented_rank_ic"], int(cfg["newey_west_lags"])
        )
        spread_se, spread_t = newey_west_mean_t(
            group["gross_top_minus_bottom"], int(cfg["newey_west_lags"])
        )
        ic_std = float(group["oriented_rank_ic"].std(ddof=1))
        record = dict(zip(grouping, keys))
        record.update({
            "observations": int(len(group)),
            "mean_oriented_rank_ic": float(group["oriented_rank_ic"].mean()),
            "annualized_icir": (
                float(group["oriented_rank_ic"].mean() / ic_std * np.sqrt(annualization))
                if ic_std > 0 else np.nan
            ),
            "positive_rank_ic_fraction": float(group["oriented_rank_ic"].gt(0).mean()),
            "rank_ic_newey_west_se": ic_se,
            "rank_ic_newey_west_t": ic_t,
            "bucket_return_spearman": monotonicity,
            "positive_adjacent_bucket_fraction": adjacent,
            "annualized_gross_top_minus_bottom": float(
                group["gross_top_minus_bottom"].mean() * annualization
            ),
            "gross_spread_newey_west_se": spread_se,
            "gross_spread_newey_west_t": spread_t,
            "positive_gross_spread_fraction": float(
                group["gross_top_minus_bottom"].gt(0).mean()
            ),
            "annualized_net_top_minus_bottom": float(
                group["net_top_minus_bottom"].mean() * annualization
            ),
            "mean_top_turnover": float(group["top_turnover"].mean()),
            "mean_bottom_turnover": float(group["bottom_turnover"].mean()),
            "mean_long_short_turnover": float(group["long_short_turnover"].mean()),
        })
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_summary(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records: list[dict] = []
    bootstrap = cfg["bootstrap"]
    for (pipeline, factor), group in weekly.groupby(["pipeline", "factor"], sort=False):
        for metric, scale in (
            ("oriented_rank_ic", 1.0),
            ("gross_top_minus_bottom", float(cfg["annualization_periods"])),
            ("net_top_minus_bottom", float(cfg["annualization_periods"])),
        ):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            estimates = circular_block_bootstrap_mean(
                values, int(bootstrap["block_length_weeks"]),
                int(bootstrap["samples"]), int(bootstrap["random_seed"]),
            ) * scale
            records.append({
                "pipeline": pipeline, "factor": factor, "metric": metric,
                "observations": len(values), "point_estimate": float(values.mean() * scale),
                "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                "one_sided_probability_nonpositive": float(
                    (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                ),
                "block_length_weeks": int(bootstrap["block_length_weeks"]),
            })
    return pd.DataFrame(records)


def write_report(
    cfg: dict, summary: pd.DataFrame, annual: pd.DataFrame,
    regime: pd.DataFrame, bootstrap: pd.DataFrame,
) -> None:
    neutral = summary.loc[summary["pipeline"].eq("industry_size_neutralized")].copy()
    intervals = bootstrap.loc[
        bootstrap["pipeline"].eq("industry_size_neutralized")
        & bootstrap["metric"].eq("gross_top_minus_bottom"),
        ["factor", "ci_2_5pct", "ci_97_5pct"],
    ]
    neutral = neutral.merge(intervals, on="factor", validate="one_to_one")
    neutral["spread_significant"] = np.where(neutral["ci_2_5pct"].gt(0), "yes", "no")
    neutral = neutral.sort_values("mean_oriented_rank_ic", ascending=False)
    annual_positive = annual.loc[annual["pipeline"].eq("industry_size_neutralized")].groupby(
        "factor"
    )["mean_oriented_rank_ic"].apply(lambda values: int(values.gt(0).sum()))
    neutral["positive_ic_years"] = neutral["factor"].map(annual_positive)
    regime_positive = regime.loc[regime["pipeline"].eq("industry_size_neutralized")].groupby(
        "factor"
    )["mean_oriented_rank_ic"].apply(lambda values: int(values.gt(0).sum()))
    neutral["positive_ic_regimes"] = neutral["factor"].map(regime_positive)
    display_columns = [
        "factor", "mean_oriented_rank_ic", "annualized_icir",
        "bucket_return_spearman", "annualized_gross_top_minus_bottom",
        "ci_2_5pct", "ci_97_5pct", "annualized_net_top_minus_bottom",
        "mean_long_short_turnover", "positive_ic_years", "positive_ic_regimes",
        "spread_significant",
    ]
    display = neutral[display_columns].copy()
    numeric = [column for column in display.columns if column not in {
        "factor", "positive_ic_years", "positive_ic_regimes", "spread_significant"
    }]
    for column in numeric:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    significant = neutral.loc[neutral["ci_2_5pct"].gt(0), "factor"].tolist()
    stable = neutral.loc[neutral["positive_ic_years"].eq(6), "factor"].tolist()
    report = "\n".join([
        "# PIT 单因子 Tear Sheet",
        "",
        "## 口径",
        "",
        "- 样本：2021–2026 historical OOS，共用 PIT 可交易股票池与完整行业/规模暴露样本。",
        "- 方向：仅按 2006–2015 raw 训练期平均 RankIC 的符号锁定；OOS 不选因子、不翻向。",
        "- 组合：每周十分位等权，D10 为预测收益最高组；Top–Bottom 为 D10 减 D1。",
        f"- 成本：多空两腿各按目标权重换手乘 {cfg['assumed_cost_bps_per_leg_turnover']:.0f}bp，仅作敏感性估计。",
        "- 市场状态：信号日可见的中证500过去60交易日趋势，并以历史扩张中位数区分高/低波动。",
        "- 推断：Newey–West(4) 与 13周 circular block bootstrap 95% 区间。",
        "",
        "## 核心发现",
        "",
        f"- 行业+规模中性化后，Top–Bottom 年化算术价差 95% 下界为正的因子：{', '.join(significant) or '无'}。",
        f"- 六个 OOS 年度定向 RankIC 全为正的因子：{', '.join(stable) or '无'}。",
        "- `annualized_net_top_minus_bottom` 对换手成本非常敏感，不等同于可直接成交的多空策略回测。",
        "",
        "## 行业+规模中性化结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 限制",
        "",
        "分位数组合未施加涨跌停、停牌、容量和借券约束；因此它是因子诊断而不是独立策略业绩。"
        "市场状态仅用于事后分层描述，但其分类变量在每个信号日均可获得。",
        "",
    ])
    destination = ROOT / "reports" / "factor_tear_sheet.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    table = ROOT / "reports" / "tables" / "factor_tear_sheet.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    neutral.to_csv(table, index=False)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    directions = lake.read("factor_neutralization_directions", "v1")
    weekly, buckets = build_tear_sheet_panel(lake, cfg, factors, directions)
    summary = summarize(weekly, buckets, cfg, ["pipeline", "factor"])
    annual = summarize(weekly, buckets, cfg, ["year", "pipeline", "factor"])
    regime = summarize(weekly, buckets, cfg, ["market_regime", "pipeline", "factor"])
    bootstrap = bootstrap_summary(weekly, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "source_experiment": cfg["source_experiment"],
        "period": cfg["period"],
        "historical_oos_used_for_selection": False,
        "direction_rule": cfg["direction_rule"],
        "pipelines": ",".join(cfg["pipelines"]),
        "bucket_count": int(cfg["bucket_count"]),
        "benchmark_instrument": cfg["benchmark_instrument"],
        "regime_uses_trailing_information_only": True,
        "weekly_dates": int(weekly["asof_date"].nunique()),
        "factors": len(factors),
    }])
    request = {**cfg, "factor_columns": factors}
    for table, frame in {
        "factor_tear_sheet_protocol": protocol,
        "factor_tear_sheet_weekly": weekly,
        "factor_tear_sheet_buckets": buckets,
        "factor_tear_sheet_summary": summary,
        "factor_tear_sheet_annual": annual,
        "factor_tear_sheet_regime": regime,
        "factor_tear_sheet_bootstrap": bootstrap,
    }.items():
        lake.write(table, frame, key="v1", source="factor_tear_sheet.v1", request=request)
    write_report(cfg, summary, annual, regime, bootstrap)
    print(summary.loc[summary["pipeline"].eq("industry_size_neutralized")].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIT single-factor tear sheets")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
