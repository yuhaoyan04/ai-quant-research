"""Validation-first leave-one-factor-out marginal contribution analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
    _read_year,
    _update_portfolio,
    transform_cross_section,
)
from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import (
    SufficientStatistics,
    fit_ridge,
    purged_end_date,
    sum_statistics,
    summarize,
)
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "leave_one_factor_out.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_feature_indices(factors: list[str]) -> dict[str, list[int]]:
    """Remove both a factor exposure and its paired missingness indicator."""
    dimension = 2 * len(factors)
    result = {"full": list(range(dimension))}
    for factor_index, factor in enumerate(factors):
        removed = {factor_index, factor_index + len(factors)}
        result[f"without_{factor}"] = [index for index in range(dimension) if index not in removed]
    return result


def embedded_beta(statistics: SufficientStatistics, penalty: float,
                  indices: list[int], dimension: int) -> np.ndarray:
    beta = np.zeros(dimension, dtype="float64")
    beta[indices] = fit_ridge(statistics, penalty, indices)
    return beta


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Monotone Benjamini-Hochberg adjusted p-values."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype="float64")
    result = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return pd.Series(result, index=p_values.index)
    order = finite_indices[np.argsort(values[finite_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def _transform(group: pd.DataFrame, factors: list[str], cfg: dict) -> np.ndarray:
    design, _ = transform_cross_section(
        group, factors, winsor_method="quantile", normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    return design


def _fit_matrix(statistics: SufficientStatistics, model_indices: dict[str, list[int]],
                penalty: float, dimension: int) -> tuple[list[str], np.ndarray]:
    names = list(model_indices)
    matrix = np.column_stack([
        embedded_beta(statistics, penalty, model_indices[name], dimension)
        for name in names
    ])
    return names, matrix


def _evaluate_period(frames: list[pd.DataFrame], factors: list[str], cfg: dict,
                     model_names: list[str], beta_by_year: dict[int, np.ndarray],
                     split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[dict] = []
    portfolio_records: list[dict] = []
    states = {name: {} for name in model_names}
    target = str(cfg["target"])
    minimum_size = int(cfg["minimum_cross_section_size"])
    for frame in frames:
        year = int(frame["asof_date"].iloc[0][:4])
        beta_matrix = beta_by_year[year]
        for asof_date, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < minimum_size:
                continue
            design = _transform(group, factors, cfg)
            scores = design @ beta_matrix
            realized = group.loc[usable, target].reset_index(drop=True)
            for model_index, model_name in enumerate(model_names):
                score = scores[:, model_index]
                usable_score = score[usable]
                percentile = pd.Series(usable_score).rank(method="first", pct=True)
                bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
                bucket_returns = realized.groupby(bucket.to_numpy()).mean()
                daily_records.append({
                    "split": split, "asof_date": asof_date, "year": year,
                    "model": model_name, "rank_ic": _rank_ic(usable_score, realized),
                    "top_minus_bottom": bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan),
                    "cross_section_size": int(usable.sum()),
                })
                portfolio = _update_portfolio(
                    group.reset_index(drop=True), score, target,
                    int(cfg["portfolio"]["top_n"]),
                    float(cfg["portfolio"]["round_trip_cost_bps"]),
                    float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                    states[model_name],
                )
                portfolio.update({
                    "split": split, "model": model_name, "year": year,
                })
                portfolio_records.append(portfolio)
        print(f"lofo evaluation split={split} year={year}", flush=True)
    return pd.DataFrame(daily_records), pd.DataFrame(portfolio_records)


def _summaries(daily: pd.DataFrame, portfolio: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records = []
    for model, group in daily.groupby("model", sort=False):
        portfolio_group = portfolio.loc[portfolio["model"].eq(model)]
        records.append({
            "model": model,
            **summarize(group, int(cfg["annualization_periods"])),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, int(cfg["annualization_periods"])
            ).items()},
        })
    return pd.DataFrame(records)


def _bootstrap_delta(values: pd.Series, cfg: dict, scale: float = 1.0) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    estimates = circular_block_bootstrap_mean(
        clean.to_numpy(), int(cfg["bootstrap"]["block_length_weeks"]),
        int(cfg["bootstrap"]["samples"]), int(cfg["bootstrap"]["random_seed"]),
    ) * scale
    probability_nonpositive = float((np.sum(estimates <= 0) + 1) / (len(estimates) + 1))
    probability_nonnegative = float((np.sum(estimates >= 0) + 1) / (len(estimates) + 1))
    return {
        "point_estimate": float(clean.mean() * scale),
        "ci_2_5pct": float(np.quantile(estimates, 0.025)),
        "ci_97_5pct": float(np.quantile(estimates, 0.975)),
        "one_sided_probability_nonpositive": probability_nonpositive,
        "two_sided_p": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
        "observations": len(clean),
    }


def _paired_results(daily: pd.DataFrame, portfolio: pd.DataFrame,
                    cfg: dict, split: str, factors: list[str]) -> pd.DataFrame:
    sample_daily = daily.loc[daily["split"].eq(split)]
    sample_portfolio = portfolio.loc[
        portfolio["split"].eq(split) & portfolio["included_in_performance"]
    ]
    full_ic = sample_daily.loc[sample_daily["model"].eq("full")].set_index("asof_date")["rank_ic"]
    full_return = sample_portfolio.loc[
        sample_portfolio["model"].eq("full")
    ].set_index("asof_date")["net_return"]
    records = []
    for factor in factors:
        model = f"without_{factor}"
        leave_ic = sample_daily.loc[sample_daily["model"].eq(model)].set_index("asof_date")["rank_ic"]
        leave_return = sample_portfolio.loc[
            sample_portfolio["model"].eq(model)
        ].set_index("asof_date")["net_return"]
        common_ic = full_ic.index.intersection(leave_ic.index)
        common_return = full_return.index.intersection(leave_return.index)
        ic = _bootstrap_delta(full_ic.loc[common_ic] - leave_ic.loc[common_ic], cfg)
        returns = _bootstrap_delta(
            full_return.loc[common_return] - leave_return.loc[common_return],
            cfg, float(cfg["annualization_periods"]),
        )
        records.append({
            "split": split, "factor": factor,
            **{f"rank_ic_{key}": value for key, value in ic.items()},
            **{f"net_return_{key}": value for key, value in returns.items()},
        })
    result = pd.DataFrame(records)
    result["rank_ic_bh_adjusted_p"] = benjamini_hochberg(
        result["rank_ic_one_sided_probability_nonpositive"]
    )
    result["net_return_bh_adjusted_p"] = benjamini_hochberg(
        result["net_return_one_sided_probability_nonpositive"]
    )
    result["rank_ic_two_sided_bh_adjusted_p"] = benjamini_hochberg(
        result["rank_ic_two_sided_p"]
    )
    result["net_return_two_sided_bh_adjusted_p"] = benjamini_hochberg(
        result["net_return_two_sided_p"]
    )
    return result


def comparison_table(paired: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_ic_point_estimate", "rank_ic_ci_2_5pct", "rank_ic_ci_97_5pct",
        "rank_ic_bh_adjusted_p", "rank_ic_two_sided_bh_adjusted_p",
        "net_return_point_estimate", "net_return_ci_2_5pct",
        "net_return_ci_97_5pct", "net_return_bh_adjusted_p",
        "net_return_two_sided_bh_adjusted_p",
    ]
    comparison = paired.pivot(index="factor", columns="split", values=metrics)
    comparison.columns = [f"{metric}__{split}" for metric, split in comparison.columns]
    comparison = comparison.reset_index()
    comparison["validation_direction"] = np.select(
        [
            comparison["rank_ic_point_estimate__validation"].gt(0),
            comparison["rank_ic_point_estimate__validation"].lt(0),
        ], ["helps_full_model", "hurts_full_model"], default="neutral"
    )
    comparison["oos_direction_confirmed"] = (
        np.sign(comparison["rank_ic_point_estimate__validation"])
        == np.sign(comparison["rank_ic_point_estimate__historical_oos"])
    )
    return comparison


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    dimension = 2 * len(factors)
    indices = model_feature_indices(factors)
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])
    full_statistics: dict[int, SufficientStatistics] = {}
    purged_statistics: dict[int, SufficientStatistics] = {}

    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        full_accumulator = MutableStatistics.create(dimension)
        purged_accumulator = MutableStatistics.create(dimension)
        for asof_date, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < int(cfg["minimum_cross_section_size"]):
                continue
            design = _transform(group, factors, cfg)[usable]
            ranked_target = rank_zscore(group.loc[usable, target])
            full_accumulator.add(design, ranked_target)
            if asof_date <= cutoff:
                purged_accumulator.add(design, ranked_target)
        full_statistics[year] = full_accumulator.freeze()
        purged_statistics[year] = purged_accumulator.freeze()
        print(f"lofo statistics year={year}", flush=True)

    penalty = float(cfg["ridge_penalty_per_observation"])
    train_end = int(cfg["train"]["end_year"])
    train_statistics = sum_statistics(
        [full_statistics[year] for year in range(start_year, train_end)]
        + [purged_statistics[train_end]]
    )
    model_names, validation_matrix = _fit_matrix(train_statistics, indices, penalty, dimension)
    validation_frames = []
    validation_end = int(cfg["validation"]["end_year"])
    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    for year in range(int(cfg["validation"]["start_year"]), validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(validation_cutoff)]
        validation_frames.append(frame)
    validation_betas = {
        year: validation_matrix for year in range(
            int(cfg["validation"]["start_year"]), validation_end + 1
        )
    }
    validation_daily, validation_portfolio = _evaluate_period(
        validation_frames, factors, cfg, model_names, validation_betas, "validation"
    )

    oos_frames = []
    oos_betas = {}
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        fit_end = test_year - 1
        statistics = sum_statistics(
            [full_statistics[year] for year in range(start_year, fit_end)]
            + [purged_statistics[fit_end]]
        )
        names, oos_betas[test_year] = _fit_matrix(statistics, indices, penalty, dimension)
        if names != model_names:
            raise RuntimeError("model ordering changed")
        oos_frames.append(_read_year(lake, test_year, columns))
    oos_daily, oos_portfolio = _evaluate_period(
        oos_frames, factors, cfg, model_names, oos_betas, "historical_oos"
    )

    daily = pd.concat([validation_daily, oos_daily], ignore_index=True)
    portfolio = pd.concat([validation_portfolio, oos_portfolio], ignore_index=True)
    summary = pd.concat([
        _summaries(validation_daily, validation_portfolio, cfg).assign(split="validation"),
        _summaries(oos_daily, oos_portfolio, cfg).assign(split="historical_oos"),
    ], ignore_index=True)
    paired = pd.concat([
        _paired_results(daily, portfolio, cfg, "validation", factors),
        _paired_results(daily, portfolio, cfg, "historical_oos", factors),
    ], ignore_index=True)
    comparison = comparison_table(paired)

    request = {**cfg, "factor_columns": factors}
    source = "leave_one_factor_out.v1"
    lake.write("lofo_daily", daily, key="v1", source=source, request=request)
    lake.write("lofo_portfolio", portfolio, key="v1", source=source, request=request)
    lake.write("lofo_summary", summary, key="v1", source=source, request=request)
    lake.write("lofo_paired_bootstrap", paired, key="v1", source=source, request=request)
    lake.write("lofo_comparison", comparison, key="v1", source=source, request=request)
    print(comparison.sort_values(
        "rank_ic_point_estimate__validation", ascending=False
    ).to_string(index=False))


if __name__ == "__main__":
    run()
