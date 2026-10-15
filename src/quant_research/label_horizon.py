"""Validation-selected 5/10/20-day target and target-transformation study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
    _update_portfolio,
    transform_cross_section,
)
from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.ic_decay import newey_west_mean_t
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


CONFIG_PATH = ROOT / "config" / "label_horizon.yaml"


@dataclass(frozen=True)
class TargetSpec:
    horizon: int
    mode: str

    @property
    def name(self) -> str:
        return f"h{self.horizon}_{self.mode}"

    @property
    def column(self) -> str:
        return f"fwd_return_{self.horizon}d_lag1"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configured_specs(cfg: dict) -> list[TargetSpec]:
    return [
        TargetSpec(int(horizon), str(mode))
        for horizon in cfg["horizons"] for mode in cfg["target_modes"]
    ]


def target_values(values: pd.Series, mode: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("target_values requires a non-missing target series")
    if mode == "raw_return":
        return numeric.to_numpy(dtype="float64")
    if mode == "cross_sectional_rank":
        return rank_zscore(numeric)
    raise ValueError(f"unknown target mode: {mode}")


def select_validation_winners(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select penalty within target, then target using only validation portfolio data."""
    selected_rows = []
    for _, group in grid.groupby("target_spec", sort=False):
        selected_rows.append(group.sort_values(
            ["mean_rank_ic", "annualized_icir", "portfolio_sharpe_zero_rate"],
            ascending=False,
        ).iloc[0])
    candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
    winner = candidates.sort_values(
        ["portfolio_sharpe_zero_rate", "portfolio_annualized_return", "mean_rank_ic"],
        ascending=False,
    ).iloc[0]
    return candidates, winner


def _transform(group: pd.DataFrame, factors: list[str], cfg: dict) -> np.ndarray:
    design, _ = transform_cross_section(
        group, factors,
        winsor_method="quantile", normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    return design


def _read_year(lake: Lake, year: int, columns: list[str]) -> pd.DataFrame:
    path = lake.table_path("multi_horizon_factor_panel", str(year))
    if not path.exists():
        raise FileNotFoundError(f"multi-horizon factor panel missing for {year}: {path}")
    return pd.read_parquet(path, columns=columns)


def _panel_audit(lake: Lake, year: int, frame: pd.DataFrame,
                 label_columns: list[str]) -> dict:
    eligible_path = lake.table_path("eligible_factor_panel_staging", str(year))
    expected_rows = pq.ParquetFile(eligible_path).metadata.num_rows
    duplicate_keys = int(frame.duplicated(["instrument_id", "asof_date"]).sum())
    record = {
        "year": year,
        "rows": len(frame),
        "eligible_rows": int(expected_rows),
        "row_count_matches_eligible": len(frame) == expected_rows,
        "duplicate_instrument_date_keys": duplicate_keys,
    }
    record.update({f"{column}_coverage": float(frame[column].notna().mean())
                   for column in label_columns})
    if not record["row_count_matches_eligible"] or duplicate_keys:
        raise RuntimeError(f"multi-horizon panel audit failed for {year}: {record}")
    return record


def _year_statistics(frame: pd.DataFrame, specs: list[TargetSpec], factors: list[str],
                     cutoffs: dict[int, str], cfg: dict
                     ) -> tuple[dict[str, SufficientStatistics],
                                dict[str, SufficientStatistics]]:
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    full = {spec.name: MutableStatistics.create(dimension) for spec in specs}
    purged = {spec.name: MutableStatistics.create(dimension) for spec in specs}
    minimum = int(cfg["minimum_cross_section_size"])
    for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=True):
        design = _transform(group, factors, cfg)
        for spec in specs:
            usable = pd.to_numeric(group[spec.column], errors="coerce").notna().to_numpy()
            if usable.sum() < minimum:
                continue
            target = target_values(group.loc[usable, spec.column], spec.mode)
            full[spec.name].add(design[usable], target)
            if asof_date <= cutoffs[spec.horizon]:
                purged[spec.name].add(design[usable], target)
    return (
        {name: item.freeze() for name, item in full.items()},
        {name: item.freeze() for name, item in purged.items()},
    )


def _fit_training(statistics_full: dict[str, dict[int, SufficientStatistics]],
                  statistics_purged: dict[str, dict[int, SufficientStatistics]],
                  spec_name: str, start_year: int, end_year: int,
                  penalty: float) -> tuple[np.ndarray, SufficientStatistics]:
    stats = sum_statistics(
        [statistics_full[spec_name][year] for year in range(start_year, end_year)]
        + [statistics_purged[spec_name][end_year]]
    )
    return fit_ridge(stats, penalty), stats


def _evaluate(
    frame: pd.DataFrame,
    specs_by_name: dict[str, TargetSpec],
    betas: dict[tuple[str, float], np.ndarray],
    factors: list[str],
    cfg: dict,
    states: dict[tuple[str, float], dict],
    *,
    split: str,
    test_year: int | None = None,
    prediction_spec: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_records, portfolio_records, prediction_frames = [], [], []
    minimum = int(cfg["minimum_cross_section_size"])
    portfolio_target = str(cfg["portfolio_evaluation_target"])
    for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=True):
        group = group.reset_index(drop=True)
        design = _transform(group, factors, cfg)
        for (spec_name, penalty), beta in betas.items():
            spec = specs_by_name[spec_name]
            score = design @ beta
            realized = pd.to_numeric(group[spec.column], errors="coerce")
            usable = realized.notna().to_numpy() & np.isfinite(score)
            if usable.sum() >= minimum:
                usable_returns = realized.loc[usable].reset_index(drop=True)
                usable_score = score[usable]
                percentile = pd.Series(usable_score).rank(method="first", pct=True)
                bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
                buckets = usable_returns.groupby(bucket.to_numpy()).mean()
                daily_records.append({
                    "asof_date": asof_date,
                    "split": split,
                    "test_year": test_year,
                    "target_spec": spec_name,
                    "horizon": spec.horizon,
                    "target_mode": spec.mode,
                    "ridge_penalty": penalty,
                    "cross_section_size": int(usable.sum()),
                    "rank_ic": _rank_ic(usable_score, usable_returns),
                    "top_minus_bottom": buckets.get(9, np.nan) - buckets.get(0, np.nan),
                })
            portfolio = _update_portfolio(
                group, score, portfolio_target,
                int(cfg["portfolio"]["top_n"]),
                float(cfg["portfolio"]["round_trip_cost_bps"]),
                float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                states[(spec_name, penalty)],
            )
            portfolio.update({
                "split": split,
                "test_year": test_year,
                "target_spec": spec_name,
                "horizon": spec.horizon,
                "target_mode": spec.mode,
                "ridge_penalty": penalty,
                "evaluation_target": portfolio_target,
            })
            portfolio_records.append(portfolio)
            if prediction_spec == spec_name:
                prediction_columns = list(dict.fromkeys([
                    "instrument_id", "asof_date", "available_date",
                    "fwd_return_5d_lag1", spec.column,
                ]))
                prediction = group[prediction_columns].copy()
                prediction["score"] = score.astype("float32")
                prediction["target_spec"] = spec_name
                prediction["test_year"] = test_year
                prediction_frames.append(prediction)
    predictions = (pd.concat(prediction_frames, ignore_index=True)
                   if prediction_frames else pd.DataFrame())
    return pd.DataFrame(daily_records), pd.DataFrame(portfolio_records), predictions


def _summaries(daily: pd.DataFrame, portfolio: pd.DataFrame,
               periods: int, split: str, cfg: dict) -> pd.DataFrame:
    records = []
    for (spec_name, penalty), group in daily.groupby(
        ["target_spec", "ridge_penalty"], sort=False
    ):
        portfolio_group = portfolio.loc[
            portfolio["target_spec"].eq(spec_name)
            & portfolio["ridge_penalty"].eq(penalty)
        ]
        first = group.iloc[0]
        horizon = int(first["horizon"])
        lags = int(cfg["newey_west_lags_by_horizon"][horizon])
        hac_se, hac_t = newey_west_mean_t(group["rank_ic"], lags)
        records.append({
            "split": split,
            "target_spec": spec_name,
            "horizon": horizon,
            "target_mode": first["target_mode"],
            "ridge_penalty": float(penalty),
            **summarize(group, periods),
            "newey_west_lags": lags,
            "newey_west_mean_ic_se": hac_se,
            "newey_west_mean_ic_t_stat": hac_t,
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, periods
            ).items()},
        })
    return pd.DataFrame(records)


def _annual_summaries(daily: pd.DataFrame, portfolio: pd.DataFrame,
                      periods: int) -> pd.DataFrame:
    dated_daily = daily.assign(year=daily["asof_date"].str[:4].astype(int))
    dated_portfolio = portfolio.assign(year=portfolio["asof_date"].str[:4].astype(int))
    records = []
    for (spec_name, year), group in dated_daily.groupby(["target_spec", "year"]):
        portfolio_group = dated_portfolio.loc[
            dated_portfolio["target_spec"].eq(spec_name)
            & dated_portfolio["year"].eq(year)
        ]
        records.append({
            "target_spec": spec_name,
            "year": int(year),
            **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, periods
            ).items()},
        })
    return pd.DataFrame(records)


def _bootstrap_comparisons(selected: str, reference: str, portfolio: pd.DataFrame,
                           cfg: dict) -> pd.DataFrame:
    usable = portfolio.loc[portfolio["included_in_performance"]]
    right = usable.loc[usable["target_spec"].eq(reference), ["asof_date", "net_return"]]
    bootstrap_cfg = cfg["bootstrap"]
    records = []
    for challenger in sorted(usable["target_spec"].unique()):
        if challenger == reference:
            continue
        left = usable.loc[
            usable["target_spec"].eq(challenger), ["asof_date", "net_return"]
        ]
        paired = left.merge(
            right, on="asof_date", suffixes=("_challenger", "_reference"),
            validate="one_to_one",
        )
        difference = paired["net_return_challenger"] - paired["net_return_reference"]
        estimates = circular_block_bootstrap_mean(
            difference.to_numpy(dtype="float64"),
            int(bootstrap_cfg["block_length_weeks"]),
            int(bootstrap_cfg["samples"]),
            int(bootstrap_cfg["random_seed"]),
        ) * int(cfg["annualization_periods"])
        records.append({
            "challenger": challenger,
            "reference": reference,
            "selected_on_validation": challenger == selected,
            "post_hoc_diagnostic": challenger != selected,
            "metric": "annualized_arithmetic_5d_net_return_difference",
            "observations": len(difference),
            "point_estimate": float(
                difference.mean() * int(cfg["annualization_periods"])
            ),
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
    specs = configured_specs(cfg)
    specs_by_name = {spec.name: spec for spec in specs}
    labels = sorted({spec.column for spec in specs})
    columns = ["instrument_id", "asof_date", "available_date", *labels, *factors]
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])

    statistics_full = {spec.name: {} for spec in specs}
    statistics_purged = {spec.name: {} for spec in specs}
    audit_records = []
    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        audit_records.append(_panel_audit(lake, year, frame, labels))
        cutoffs = {
            horizon: purged_end_date(calendar, f"{year}-12-31", horizon)
            for horizon in cfg["horizons"]
        }
        full, purged = _year_statistics(frame, specs, factors, cutoffs, cfg)
        for spec in specs:
            statistics_full[spec.name][year] = full[spec.name]
            statistics_purged[spec.name][year] = purged[spec.name]
        print(f"label statistics year={year} weekly_dates={weekly_rows(frame)['asof_date'].nunique()}",
              flush=True)

    train_end = int(cfg["train"]["end_year"])
    penalties = [float(value) for value in cfg["ridge_penalties_per_observation"]]
    validation_betas = {}
    for spec in specs:
        for penalty in penalties:
            validation_betas[(spec.name, penalty)], _ = _fit_training(
                statistics_full, statistics_purged, spec.name,
                start_year, train_end, penalty,
            )

    validation_states = {key: {} for key in validation_betas}
    validation_daily_frames, validation_portfolio_frames = [], []
    validation_end = int(cfg["validation"]["end_year"])
    common_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31",
        int(cfg["common_validation_boundary_purge_horizon"]),
    )
    for year in range(int(cfg["validation"]["start_year"]), validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(common_cutoff)]
        daily, portfolio, _ = _evaluate(
            frame, specs_by_name, validation_betas, factors, cfg,
            validation_states, split="validation",
        )
        validation_daily_frames.append(daily)
        validation_portfolio_frames.append(portfolio)
        print(f"label validation year={year} weekly_dates={weekly_rows(frame)['asof_date'].nunique()}",
              flush=True)
    validation_daily = pd.concat(validation_daily_frames, ignore_index=True)
    validation_portfolio = pd.concat(validation_portfolio_frames, ignore_index=True)
    validation_grid = _summaries(
        validation_daily, validation_portfolio, int(cfg["annualization_periods"]),
        "validation", cfg,
    )
    candidates, winner_row = select_validation_winners(validation_grid)
    winner_name = str(winner_row["target_spec"])
    selected_penalties = dict(zip(
        candidates["target_spec"], candidates["ridge_penalty"].astype(float)
    ))
    selection = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "selected_target_spec": winner_name,
        "selected_horizon": int(winner_row["horizon"]),
        "selected_target_mode": winner_row["target_mode"],
        "selected_ridge_penalty": float(winner_row["ridge_penalty"]),
        "validation_mean_rank_ic": float(winner_row["mean_rank_ic"]),
        "validation_portfolio_sharpe_zero_rate": float(
            winner_row["portfolio_sharpe_zero_rate"]
        ),
        "selection_period": cfg["selection"]["selection_period"],
        "historical_oos_used_for_selection": False,
        "portfolio_evaluation_target": cfg["portfolio_evaluation_target"],
    }])

    oos_daily_frames, oos_portfolio_frames, prediction_frames = [], [], []
    coefficient_records = []
    oos_states = {(name, penalty): {} for name, penalty in selected_penalties.items()}
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        betas = {}
        for spec in specs:
            penalty = selected_penalties[spec.name]
            beta, fit_stats = _fit_training(
                statistics_full, statistics_purged, spec.name,
                start_year, test_year - 1, penalty,
            )
            betas[(spec.name, penalty)] = beta
            coefficient_records.extend({
                "test_year": test_year,
                "target_spec": spec.name,
                "feature": (factors[index] if index < len(factors)
                            else f"{factors[index - len(factors)]}__missing"),
                "coefficient": float(value),
                "fit_observations": fit_stats.observations,
                "ridge_penalty": penalty,
            } for index, value in enumerate(beta))
        frame = _read_year(lake, test_year, columns)
        if test_year == oos_end:
            audit_records.append(_panel_audit(lake, test_year, frame, labels))
        daily, portfolio, predictions = _evaluate(
            frame, specs_by_name, betas, factors, cfg, oos_states,
            split="historical_oos", test_year=test_year,
            prediction_spec=winner_name,
        )
        oos_daily_frames.append(daily)
        oos_portfolio_frames.append(portfolio)
        prediction_frames.append(predictions)
        print(f"label oos year={test_year} weekly_dates={weekly_rows(frame)['asof_date'].nunique()}",
              flush=True)
    oos_daily = pd.concat(oos_daily_frames, ignore_index=True)
    oos_portfolio = pd.concat(oos_portfolio_frames, ignore_index=True)
    oos_predictions = pd.concat(prediction_frames, ignore_index=True)
    oos_summary = _summaries(
        oos_daily, oos_portfolio, int(cfg["annualization_periods"]),
        "historical_oos", cfg,
    )
    annual = _annual_summaries(oos_daily, oos_portfolio, int(cfg["annualization_periods"]))
    reference = "h5_cross_sectional_rank"
    bootstrap = _bootstrap_comparisons(winner_name, reference, oos_portfolio, cfg)

    request = {**cfg, "factor_columns": factors}
    source = "label_horizon.v1"
    outputs = {
        "label_horizon_panel_audit": pd.DataFrame(audit_records),
        "label_horizon_validation_grid": validation_grid,
        "label_horizon_validation_candidates": candidates,
        "label_horizon_selection": selection,
        "label_horizon_validation_daily": validation_daily,
        "label_horizon_validation_portfolio": validation_portfolio,
        "label_horizon_oos_daily": oos_daily,
        "label_horizon_oos_portfolio": oos_portfolio,
        "label_horizon_oos_summary": oos_summary,
        "label_horizon_oos_annual": annual,
        "label_horizon_oos_predictions": oos_predictions,
        "label_horizon_coefficients": pd.DataFrame(coefficient_records),
        "label_horizon_oos_bootstrap": bootstrap,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source=source, request=request)
    print(selection.to_string(index=False))
    print(candidates.to_string(index=False))
    print(oos_summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    run()
