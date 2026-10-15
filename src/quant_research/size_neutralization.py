"""Paired point-in-time size-neutralization stress test for the locked Ridge model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
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
    weekly_rows,
)
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "size_neutralization.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def neutralize_design_to_size(
    design: np.ndarray, size: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS-residualize every feature on same-date winsorized log float cap."""
    raw_size = pd.to_numeric(size, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(raw_size) & (raw_size > 0)
    if valid.sum() < 3:
        raise ValueError("not enough valid size observations for neutralization")
    log_size = np.full(len(raw_size), np.nan, dtype="float64")
    log_size[valid] = np.log(raw_size[valid])
    lower, upper = np.quantile(log_size[valid], [0.01, 0.99])
    clipped = np.clip(log_size[valid], lower, upper)
    mean, standard_deviation = clipped.mean(), clipped.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("same-date size exposure has no dispersion")
    size_z = np.zeros(len(raw_size), dtype="float64")
    size_z[valid] = (clipped - mean) / standard_deviation
    regressors = np.column_stack([np.ones(valid.sum()), size_z[valid]])
    coefficients = np.linalg.lstsq(regressors, design[valid], rcond=None)[0]
    fitted = coefficients[0] + np.outer(size_z, coefficients[1])
    residual = design - fitted
    residual_mean = residual.mean(axis=0)
    residual_std = residual.std(axis=0, ddof=1)
    neutral = np.divide(
        residual - residual_mean,
        residual_std,
        out=np.zeros_like(residual),
        where=residual_std > 0,
    )
    if not np.isfinite(neutral).all():
        raise ValueError("size neutralization produced non-finite features")
    return neutral, size_z, valid


def _transform(group: pd.DataFrame, factors: list[str], cfg: dict
               ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    raw, _ = transform_cross_section(
        group, factors,
        winsor_method="quantile", normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    neutral, size_z, valid_size = neutralize_design_to_size(
        raw, group["float_cap_proxy_cny"]
    )
    return {"raw": raw, "size_neutralized": neutral}, size_z, valid_size


def _read_year(lake: Lake, year: int, columns: list[str]) -> pd.DataFrame:
    panel = weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=columns
    ))
    size = pd.read_parquet(
        lake.table_path("pit_size_exposures", str(year)),
        columns=["instrument_id", "asof_date", "float_cap_proxy_cny"],
    )
    result = panel.merge(
        size, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )
    return result


def _year_statistics(frame: pd.DataFrame, factors: list[str], target: str,
                     cutoff: str, cfg: dict
                     ) -> tuple[dict[str, SufficientStatistics],
                                dict[str, SufficientStatistics]]:
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    full = {name: MutableStatistics.create(dimension) for name in cfg["pipelines"]}
    purged = {name: MutableStatistics.create(dimension) for name in cfg["pipelines"]}
    for asof_date, group in frame.groupby("asof_date", sort=True):
        usable = pd.to_numeric(group[target], errors="coerce").notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        designs, _, _ = _transform(group, factors, cfg)
        ranked_target = rank_zscore(group.loc[usable, target])
        for name, design in designs.items():
            full[name].add(design[usable], ranked_target)
            if asof_date <= cutoff:
                purged[name].add(design[usable], ranked_target)
    return (
        {name: value.freeze() for name, value in full.items()},
        {name: value.freeze() for name, value in purged.items()},
    )


def _fit(full: dict[str, dict[int, SufficientStatistics]],
         purged: dict[str, dict[int, SufficientStatistics]],
         pipeline: str, start_year: int, end_year: int, penalty: float
         ) -> tuple[np.ndarray, SufficientStatistics]:
    statistics = sum_statistics(
        [full[pipeline][year] for year in range(start_year, end_year)]
        + [purged[pipeline][end_year]]
    )
    return fit_ridge(statistics, penalty), statistics


def _evaluate(frame: pd.DataFrame, factors: list[str], target: str,
              betas: dict[str, np.ndarray], states: dict[str, dict], cfg: dict,
              split: str, test_year: int | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_records, portfolio_records, exposure_records = [], [], []
    for asof_date, group in frame.groupby("asof_date", sort=True):
        group = group.reset_index(drop=True)
        designs, size_z, valid_size = _transform(group, factors, cfg)
        realized = pd.to_numeric(group[target], errors="coerce")
        usable = realized.notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        for name, beta in betas.items():
            score = designs[name] @ beta
            usable_score = score[usable]
            usable_returns = realized.loc[usable].reset_index(drop=True)
            percentile = pd.Series(usable_score).rank(method="first", pct=True)
            bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
            bucket_returns = usable_returns.groupby(bucket.to_numpy()).mean()
            daily_records.append({
                "asof_date": asof_date, "split": split, "test_year": test_year,
                "pipeline": name, "cross_section_size": int(usable.sum()),
                "rank_ic": _rank_ic(usable_score, usable_returns),
                "top_minus_bottom": (
                    bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan)
                ),
            })
            portfolio = _update_portfolio(
                group, score, target, int(cfg["portfolio"]["top_n"]),
                float(cfg["portfolio"]["round_trip_cost_bps"]),
                float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                states[name],
            )
            portfolio.update({
                "split": split, "test_year": test_year, "pipeline": name,
            })
            portfolio_records.append(portfolio)
            ranked = np.argsort(-score, kind="stable")[:int(cfg["portfolio"]["top_n"])]
            size_percentile = pd.Series(size_z).rank(method="average", pct=True).to_numpy()
            correlation = float(np.corrcoef(score[valid_size], size_z[valid_size])[0, 1])
            exposure_records.append({
                "asof_date": asof_date, "split": split, "test_year": test_year,
                "pipeline": name, "score_log_size_correlation": correlation,
                "selected_mean_size_z": float(np.mean(size_z[ranked])),
                "selected_mean_size_percentile": float(np.mean(size_percentile[ranked])),
            })
    return (
        pd.DataFrame(daily_records), pd.DataFrame(portfolio_records),
        pd.DataFrame(exposure_records),
    )


def _summary(daily: pd.DataFrame, portfolio: pd.DataFrame,
             exposures: pd.DataFrame, periods: int, split: str) -> pd.DataFrame:
    records = []
    for pipeline, group in daily.groupby("pipeline", sort=False):
        portfolio_group = portfolio.loc[portfolio["pipeline"].eq(pipeline)]
        exposure_group = exposures.loc[exposures["pipeline"].eq(pipeline)]
        records.append({
            "split": split, "pipeline": pipeline,
            **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, periods
            ).items()},
            "mean_score_log_size_correlation": float(
                exposure_group["score_log_size_correlation"].mean()
            ),
            "mean_selected_size_z": float(exposure_group["selected_mean_size_z"].mean()),
            "mean_selected_size_percentile": float(
                exposure_group["selected_mean_size_percentile"].mean()
            ),
        })
    return pd.DataFrame(records)


def _bootstrap(oos_daily: pd.DataFrame, oos_portfolio: pd.DataFrame,
               cfg: dict) -> pd.DataFrame:
    records = []
    bootstrap_cfg = cfg["bootstrap"]
    for metric, frame, column, scale in (
        ("size_neutralized_minus_raw_rank_ic", oos_daily, "rank_ic", 1.0),
        ("size_neutralized_minus_raw_net_return_annualized_arithmetic",
         oos_portfolio.loc[oos_portfolio["included_in_performance"]],
         "net_return", float(cfg["annualization_periods"])),
    ):
        left = frame.loc[frame["pipeline"].eq("size_neutralized"), ["asof_date", column]]
        right = frame.loc[frame["pipeline"].eq("raw"), ["asof_date", column]]
        paired = left.merge(right, on="asof_date", suffixes=("_neutral", "_raw"),
                            validate="one_to_one")
        difference = paired[f"{column}_neutral"] - paired[f"{column}_raw"]
        estimates = circular_block_bootstrap_mean(
            difference.to_numpy(dtype="float64"),
            int(bootstrap_cfg["block_length_weeks"]),
            int(bootstrap_cfg["samples"]), int(bootstrap_cfg["random_seed"]),
        ) * scale
        records.append({
            "metric": metric, "observations": len(difference),
            "point_estimate": float(difference.mean() * scale),
            "ci_2_5pct": float(np.quantile(estimates, 0.025)),
            "ci_97_5pct": float(np.quantile(estimates, 0.975)),
            "one_sided_probability_nonpositive": float(
                (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
            ),
            "block_length_weeks": int(bootstrap_cfg["block_length_weeks"]),
        })
    return pd.DataFrame(records)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])
    full = {name: {} for name in cfg["pipelines"]}
    purged = {name: {} for name in cfg["pipelines"]}
    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        year_full, year_purged = _year_statistics(frame, factors, target, cutoff, cfg)
        for name in cfg["pipelines"]:
            full[name][year], purged[name][year] = year_full[name], year_purged[name]
        print(f"size-neutral statistics year={year} dates={frame['asof_date'].nunique()}",
              flush=True)

    penalty = float(cfg["ridge_penalty_per_observation"])
    train_end = int(cfg["train"]["end_year"])
    validation_betas = {
        name: _fit(full, purged, name, start_year, train_end, penalty)[0]
        for name in cfg["pipelines"]
    }
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
    validation = pd.concat(validation_frames, ignore_index=True)
    validation_daily, validation_portfolio, validation_exposures = _evaluate(
        validation, factors, target, validation_betas,
        {name: {} for name in cfg["pipelines"]}, cfg, "validation",
    )
    validation_summary = _summary(
        validation_daily, validation_portfolio, validation_exposures,
        int(cfg["annualization_periods"]), "validation",
    )

    oos_daily, oos_portfolio, oos_exposures, coefficient_records = [], [], [], []
    states = {name: {} for name in cfg["pipelines"]}
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        betas = {}
        for name in cfg["pipelines"]:
            beta, statistics = _fit(
                full, purged, name, start_year, test_year - 1, penalty
            )
            betas[name] = beta
            coefficient_records.extend({
                "test_year": test_year, "pipeline": name,
                "feature": (factors[index] if index < len(factors)
                            else f"{factors[index-len(factors)]}__missing"),
                "coefficient": float(value), "fit_observations": statistics.observations,
            } for index, value in enumerate(beta))
        frame = _read_year(lake, test_year, columns)
        daily, portfolio, exposures = _evaluate(
            frame, factors, target, betas, states, cfg, "historical_oos", test_year
        )
        oos_daily.append(daily); oos_portfolio.append(portfolio); oos_exposures.append(exposures)
        print(f"size-neutral oos year={test_year} dates={frame['asof_date'].nunique()}",
              flush=True)
    oos_daily_frame = pd.concat(oos_daily, ignore_index=True)
    oos_portfolio_frame = pd.concat(oos_portfolio, ignore_index=True)
    oos_exposure_frame = pd.concat(oos_exposures, ignore_index=True)
    oos_summary = _summary(
        oos_daily_frame, oos_portfolio_frame, oos_exposure_frame,
        int(cfg["annualization_periods"]), "historical_oos",
    )
    bootstrap = _bootstrap(oos_daily_frame, oos_portfolio_frame, cfg)
    annual_records = []
    for (pipeline, year), group in oos_daily_frame.groupby(["pipeline", "test_year"]):
        portfolio_group = oos_portfolio_frame.loc[
            oos_portfolio_frame["pipeline"].eq(pipeline)
            & oos_portfolio_frame["test_year"].eq(year)
        ]
        annual_records.append({
            "pipeline": pipeline, "year": int(year),
            **summarize(group, int(cfg["annualization_periods"])),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, int(cfg["annualization_periods"])
            ).items()},
        })

    request = {**cfg, "factor_columns": factors}
    outputs = {
        "size_neutralization_validation_summary": validation_summary,
        "size_neutralization_validation_daily": validation_daily,
        "size_neutralization_validation_portfolio": validation_portfolio,
        "size_neutralization_validation_exposures": validation_exposures,
        "size_neutralization_oos_summary": oos_summary,
        "size_neutralization_oos_daily": oos_daily_frame,
        "size_neutralization_oos_portfolio": oos_portfolio_frame,
        "size_neutralization_oos_exposures": oos_exposure_frame,
        "size_neutralization_oos_annual": pd.DataFrame(annual_records),
        "size_neutralization_oos_bootstrap": bootstrap,
        "size_neutralization_coefficients": pd.DataFrame(coefficient_records),
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="size_neutralization.v1", request=request)
    print(validation_summary.to_string(index=False))
    print(oos_summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    run()
