"""Locked, expanding-window historical OOS evaluation of preprocessing choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
    _read_year,
    _update_portfolio,
    annual_summaries,
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


CONFIG_PATH = ROOT / "config" / "factor_preprocessing_oos.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expanding_fit_statistics(
    full: dict[int, SufficientStatistics],
    purged: dict[int, SufficientStatistics],
    start_year: int,
    test_year: int,
) -> SufficientStatistics:
    """Use full years before test_year-1 and the purged immediately prior year."""
    fit_end = test_year - 1
    return sum_statistics(
        [full[year] for year in range(start_year, fit_end)] + [purged[fit_end]]
    )


def _pipeline_transform(group: pd.DataFrame, factors: list[str], pipeline: dict,
                        cfg: dict) -> np.ndarray:
    design, _ = transform_cross_section(
        group,
        factors,
        winsor_method=str(pipeline["winsor_method"]),
        normalization_method=str(pipeline["normalization_method"]),
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    return design


def _bootstrap_record(metric: str, values: pd.Series, scale: float, cfg: dict) -> dict:
    estimates = circular_block_bootstrap_mean(
        values.to_numpy(dtype="float64"),
        int(cfg["bootstrap"]["block_length_weeks"]),
        int(cfg["bootstrap"]["samples"]),
        int(cfg["bootstrap"]["random_seed"]),
    ) * scale
    return {
        "metric": metric,
        "observations": int(values.notna().sum()),
        "point_estimate": float(values.mean() * scale),
        "ci_2_5pct": float(np.quantile(estimates, 0.025)),
        "ci_97_5pct": float(np.quantile(estimates, 0.975)),
        "one_sided_probability_nonpositive": float(
            (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
        ),
        "block_length_weeks": int(cfg["bootstrap"]["block_length_weeks"]),
    }


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    preprocessing_cfg = load_preprocessing_config()
    factors = list(preprocessing_cfg["factor_columns"])
    target = str(cfg["target"])
    pipelines = dict(cfg["pipelines"])
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    minimum_size = int(cfg["minimum_cross_section_size"])
    start_year = int(cfg["train_start_year"])
    test_start = int(cfg["test"]["start_year"])
    test_end = int(cfg["test"]["end_year"])
    calendar = lake.read("calendar", "SSE")

    full_statistics: dict[str, dict[int, SufficientStatistics]] = {
        name: {} for name in pipelines
    }
    purged_statistics: dict[str, dict[int, SufficientStatistics]] = {
        name: {} for name in pipelines
    }
    for year in range(start_year, test_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        full_accumulators = {
            name: MutableStatistics.create(dimension) for name in pipelines
        }
        purged_accumulators = {
            name: MutableStatistics.create(dimension) for name in pipelines
        }
        for asof_date, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < minimum_size:
                continue
            ranked_target = rank_zscore(group.loc[usable, target])
            for name, pipeline in pipelines.items():
                design = _pipeline_transform(group, factors, pipeline, cfg)[usable]
                full_accumulators[name].add(design, ranked_target)
                if asof_date <= cutoff:
                    purged_accumulators[name].add(design, ranked_target)
        for name in pipelines:
            full_statistics[name][year] = full_accumulators[name].freeze()
            purged_statistics[name][year] = purged_accumulators[name].freeze()
        print(f"oos statistics year={year} weekly_dates={frame['asof_date'].nunique()}", flush=True)

    daily_records: list[dict] = []
    portfolio_records: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_records: list[dict] = []
    portfolio_states = {name: {} for name in pipelines}
    selected_name = next(
        name for name, pipeline in pipelines.items()
        if bool(pipeline["selected_on_validation"])
    )
    for test_year in range(test_start, test_end + 1):
        betas: dict[str, np.ndarray] = {}
        for name, pipeline in pipelines.items():
            fit_stats = expanding_fit_statistics(
                full_statistics[name], purged_statistics[name], start_year, test_year
            )
            penalty = float(pipeline["ridge_penalty_per_observation"])
            betas[name] = fit_ridge(fit_stats, penalty)
            coefficient_records.extend({
                "test_year": test_year,
                "pipeline": name,
                "feature": (
                    factors[index] if index < len(factors)
                    else f"{factors[index - len(factors)]}__missing"
                ),
                "coefficient": float(value),
                "fit_observations": fit_stats.observations,
                "ridge_penalty": penalty,
            } for index, value in enumerate(betas[name]))

        frame = _read_year(lake, test_year, columns)
        for asof_date, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < minimum_size:
                continue
            realized = group.loc[usable, target].reset_index(drop=True)
            for name, pipeline in pipelines.items():
                design = _pipeline_transform(group, factors, pipeline, cfg)
                score = design @ betas[name]
                usable_score = score[usable]
                percentile = pd.Series(usable_score).rank(method="first", pct=True)
                bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
                bucket_returns = realized.groupby(bucket.to_numpy()).mean()
                daily_records.append({
                    "asof_date": asof_date,
                    "test_year": test_year,
                    "variant": name,
                    "ridge_penalty": float(pipeline["ridge_penalty_per_observation"]),
                    "cross_section_size": int(usable.sum()),
                    "rank_ic": _rank_ic(usable_score, realized),
                    "top_minus_bottom": (
                        bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan)
                    ),
                })
                portfolio = _update_portfolio(
                    group.reset_index(drop=True), score, target,
                    int(cfg["portfolio"]["top_n"]),
                    float(cfg["portfolio"]["round_trip_cost_bps"]),
                    float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                    portfolio_states[name],
                )
                portfolio.update({
                    "variant": name,
                    "ridge_penalty": float(pipeline["ridge_penalty_per_observation"]),
                    "test_year": test_year,
                })
                portfolio_records.append(portfolio)
                if name == selected_name:
                    predictions = group[[
                        "instrument_id", "asof_date", "available_date", target
                    ]].copy()
                    predictions["score"] = score.astype("float32")
                    predictions["test_year"] = test_year
                    prediction_frames.append(predictions)
        print(f"oos evaluation year={test_year} weekly_dates={frame['asof_date'].nunique()}", flush=True)

    daily = pd.DataFrame(daily_records)
    portfolio = pd.DataFrame(portfolio_records)
    annual = annual_summaries(daily, portfolio, int(cfg["annualization_periods"]))
    summary_records = []
    for name, pipeline in pipelines.items():
        model_daily = daily.loc[daily["variant"].eq(name)]
        portfolio_daily = portfolio.loc[portfolio["variant"].eq(name)]
        summary_records.append({
            "pipeline": name,
            "selected_on_validation": bool(pipeline["selected_on_validation"]),
            **summarize(model_daily, int(cfg["annualization_periods"])),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_daily, int(cfg["annualization_periods"])
            ).items()},
        })
    summary = pd.DataFrame(summary_records)

    selected_daily = daily.loc[daily["variant"].eq(selected_name)].set_index("asof_date")
    reference_name = next(name for name in pipelines if name != selected_name)
    reference_daily = daily.loc[daily["variant"].eq(reference_name)].set_index("asof_date")
    selected_portfolio = portfolio.loc[
        portfolio["variant"].eq(selected_name)
        & portfolio["included_in_performance"]
    ].set_index("asof_date")
    reference_portfolio = portfolio.loc[
        portfolio["variant"].eq(reference_name)
        & portfolio["included_in_performance"]
    ].set_index("asof_date")
    common_ic = selected_daily.index.intersection(reference_daily.index)
    common_portfolio = selected_portfolio.index.intersection(reference_portfolio.index)
    bootstrap = pd.DataFrame([
        _bootstrap_record(
            "selected_minus_reference_rank_ic",
            selected_daily.loc[common_ic, "rank_ic"] - reference_daily.loc[common_ic, "rank_ic"],
            1.0, cfg,
        ),
        _bootstrap_record(
            "selected_minus_reference_net_return_annualized_arithmetic",
            selected_portfolio.loc[common_portfolio, "net_return"]
            - reference_portfolio.loc[common_portfolio, "net_return"],
            float(cfg["annualization_periods"]), cfg,
        ),
    ])

    request = {**cfg, "factor_columns": factors}
    source = "factor_preprocessing_oos.v1"
    lake.write("factor_preprocessing_oos_daily", daily, key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_portfolio", portfolio, key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_annual", annual, key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_summary", summary, key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_bootstrap", bootstrap, key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_predictions", pd.concat(prediction_frames, ignore_index=True),
               key="v1", source=source, request=request)
    lake.write("factor_preprocessing_oos_coefficients", pd.DataFrame(coefficient_records),
               key="v1", source=source, request=request)
    print(summary.to_string(index=False))
    print(annual.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    run()

