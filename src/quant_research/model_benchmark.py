"""Unified validation-first model-family benchmark on the locked factor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

from quant_research.factor_preprocessing_ablation import (
    MutableStatistics, _rank_ic, _update_portfolio, transform_cross_section,
)
from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import (
    SufficientStatistics, fit_ridge, purged_end_date, sum_statistics,
    summarize, weekly_rows,
)
from quant_research.model_challenger import (
    TrainingSample, concatenate_samples, deterministic_indices,
)
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "model_benchmark.yaml"


@dataclass
class FittedCandidate:
    candidate: str
    family: str
    kind: str
    estimator: Any
    fit_seconds: float
    fit_observations: int

    def predict(self, design: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, np.ndarray):
            return design @ self.estimator
        return np.asarray(self.estimator.predict(design), dtype="float64")


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_family_candidates(grid: pd.DataFrame, metric: str,
                             tiebreakers: list[str]) -> pd.DataFrame:
    rows = []
    columns = [metric, *tiebreakers]
    for _, group in grid.groupby("family", sort=False):
        rows.append(group.sort_values(columns, ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _transform(group: pd.DataFrame, factors: list[str], cfg: dict) -> np.ndarray:
    design, _ = transform_cross_section(
        group, factors, winsor_method="quantile", normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    return design


def _read_year(lake: Lake, year: int, columns: list[str]) -> pd.DataFrame:
    return weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=columns
    ))


def build_year_training(
    frame: pd.DataFrame, factors: list[str], target: str, cutoff: str,
    cfg: dict,
) -> tuple[SufficientStatistics, SufficientStatistics, TrainingSample, TrainingSample]:
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    full_stats, purged_stats = (MutableStatistics.create(dimension),
                                MutableStatistics.create(dimension))
    full_x, full_y, purged_x, purged_y = [], [], [], []
    full_dates = purged_dates = 0
    for asof_date, group in frame.groupby("asof_date", sort=True):
        usable = pd.to_numeric(group[target], errors="coerce").notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        design = _transform(group, factors, cfg)[usable]
        target_values = rank_zscore(group.loc[usable, target])
        full_stats.add(design, target_values)
        usable_group = group.loc[usable]
        chosen = deterministic_indices(usable_group, int(cfg["sample_per_training_date"]))
        full_x.append(design[chosen].astype("float32"))
        full_y.append(target_values[chosen].astype("float32"))
        full_dates += 1
        if asof_date <= cutoff:
            purged_stats.add(design, target_values)
            purged_x.append(design[chosen].astype("float32"))
            purged_y.append(target_values[chosen].astype("float32"))
            purged_dates += 1
    def sample(xs: list[np.ndarray], ys: list[np.ndarray], dates: int) -> TrainingSample:
        return TrainingSample(np.concatenate(xs), np.concatenate(ys), dates)
    return (full_stats.freeze(), purged_stats.freeze(),
            sample(full_x, full_y, full_dates), sample(purged_x, purged_y, purged_dates))


def _ols(statistics: SufficientStatistics) -> np.ndarray:
    return np.linalg.lstsq(statistics.gram, statistics.cross, rcond=1e-10)[0]


def _naive(statistics: SufficientStatistics, factor_count: int) -> np.ndarray:
    beta = np.zeros(len(statistics.cross), dtype="float64")
    directions = np.sign(statistics.cross[:factor_count])
    beta[:factor_count] = directions / max(np.sqrt((directions != 0).sum()), 1.0)
    return beta


def fit_candidate(spec: dict, statistics: SufficientStatistics,
                  sample: TrainingSample, factor_count: int,
                  cfg: dict) -> FittedCandidate:
    started = perf_counter()
    kind = str(spec["kind"])
    if kind == "naive":
        estimator: Any = _naive(statistics, factor_count)
        observations = statistics.observations
    elif kind == "ols":
        estimator = _ols(statistics)
        observations = statistics.observations
    elif kind == "ridge":
        estimator = fit_ridge(statistics, float(spec["penalty"]))
        observations = statistics.observations
    elif kind == "elastic_net":
        estimator = ElasticNet(
            alpha=float(spec["alpha"]), l1_ratio=float(spec["l1_ratio"]),
            fit_intercept=False, max_iter=5000, tol=1e-4,
            selection="cyclic", random_state=int(cfg["random_state"]),
        ).fit(sample.x, sample.y)
        observations = len(sample.y)
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            n_jobs=-1, random_state=int(spec.get("random_state", cfg["random_state"])),
        ).fit(sample.x, sample.y)
        observations = len(sample.y)
    elif kind == "hgb":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=False, random_state=int(cfg["random_state"]),
        ).fit(sample.x, sample.y)
        observations = len(sample.y)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return FittedCandidate(
        str(spec["candidate"]), str(spec["family"]), kind, estimator,
        perf_counter() - started, observations,
    )


def _evaluate(frame: pd.DataFrame, models: dict[str, FittedCandidate],
              factors: list[str], target: str, states: dict[str, dict],
              cfg: dict, split: str, test_year: int | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records, portfolio_records = [], []
    for asof_date, group in frame.groupby("asof_date", sort=True):
        group = group.reset_index(drop=True)
        design = _transform(group, factors, cfg)
        realized = pd.to_numeric(group[target], errors="coerce")
        usable = realized.notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        for candidate, model in models.items():
            score = model.predict(design)
            usable_returns = realized.loc[usable].reset_index(drop=True)
            usable_score = score[usable]
            percentile = pd.Series(usable_score).rank(method="first", pct=True)
            bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
            bucket_returns = usable_returns.groupby(bucket.to_numpy()).mean()
            daily_records.append({
                "asof_date": asof_date, "split": split, "test_year": test_year,
                "candidate": candidate, "family": model.family, "kind": model.kind,
                "cross_section_size": int(usable.sum()),
                "rank_ic": _rank_ic(usable_score, usable_returns),
                "top_minus_bottom": (
                    bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan)
                ),
            })
            portfolio = _update_portfolio(
                group, score, target, int(cfg["portfolio"]["top_n"]),
                float(cfg["portfolio"]["round_trip_cost_bps"]),
                float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                states[candidate],
            )
            portfolio.update({
                "split": split, "test_year": test_year,
                "candidate": candidate, "family": model.family, "kind": model.kind,
            })
            portfolio_records.append(portfolio)
    return pd.DataFrame(daily_records), pd.DataFrame(portfolio_records)


def _summary(daily: pd.DataFrame, portfolio: pd.DataFrame,
             models: dict[str, FittedCandidate], periods: int,
             split: str) -> pd.DataFrame:
    rows = []
    for candidate, group in daily.groupby("candidate", sort=False):
        model = models[candidate]
        portfolio_group = portfolio.loc[portfolio["candidate"].eq(candidate)]
        rows.append({
            "split": split, "candidate": candidate, "family": model.family,
            "kind": model.kind, "fit_seconds": model.fit_seconds,
            "fit_observations": model.fit_observations,
            **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, periods
            ).items()},
        })
    return pd.DataFrame(rows)


def _bootstrap(daily: pd.DataFrame, portfolio: pd.DataFrame,
               reference: str, cfg: dict) -> pd.DataFrame:
    records = []
    bootstrap_cfg = cfg["bootstrap"]
    for candidate in sorted(daily["candidate"].unique()):
        if candidate == reference:
            continue
        for metric, frame, column, scale in (
            ("rank_ic", daily, "rank_ic", 1.0),
            ("net_return_annualized_arithmetic",
             portfolio.loc[portfolio["included_in_performance"]],
             "net_return", float(cfg["annualization_periods"])),
        ):
            left = frame.loc[frame["candidate"].eq(candidate), ["asof_date", column]]
            right = frame.loc[frame["candidate"].eq(reference), ["asof_date", column]]
            paired = left.merge(right, on="asof_date", suffixes=("_candidate", "_reference"),
                                validate="one_to_one")
            difference = paired[f"{column}_candidate"] - paired[f"{column}_reference"]
            estimates = circular_block_bootstrap_mean(
                difference.to_numpy(dtype="float64"),
                int(bootstrap_cfg["block_length_weeks"]),
                int(bootstrap_cfg["samples"]), int(bootstrap_cfg["random_seed"]),
            ) * scale
            records.append({
                "candidate": candidate, "reference": reference, "metric": metric,
                "observations": len(difference),
                "point_estimate": float(difference.mean() * scale),
                "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                "one_sided_probability_nonpositive": float(
                    (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                ),
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
    full_stats, purged_stats, full_samples, purged_samples = {}, {}, {}, {}
    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        (full_stats[year], purged_stats[year],
         full_samples[year], purged_samples[year]) = build_year_training(
            frame, factors, target, cutoff, cfg
        )
        print(f"benchmark statistics year={year} dates={full_stats[year].dates} "
              f"sample_rows={len(full_samples[year].y)}", flush=True)

    train_end = int(cfg["train"]["end_year"])
    train_stats = sum_statistics(
        [full_stats[year] for year in range(start_year, train_end)]
        + [purged_stats[train_end]]
    )
    train_sample = concatenate_samples(
        [full_samples[year] for year in range(start_year, train_end)]
        + [purged_samples[train_end]]
    )
    validation_models = {
        str(spec["candidate"]): fit_candidate(
            spec, train_stats, train_sample, len(factors), cfg
        ) for spec in cfg["candidates"]
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
    validation_daily, validation_portfolio = _evaluate(
        validation, validation_models, factors, target,
        {name: {} for name in validation_models}, cfg, "validation",
    )
    grid = _summary(
        validation_daily, validation_portfolio, validation_models,
        int(cfg["annualization_periods"]), "validation",
    )
    selected = select_family_candidates(
        grid, str(cfg["selection_metric"]), list(cfg["selection_tiebreakers"])
    )
    selected_specs = {
        row["family"]: next(
            spec for spec in cfg["candidates"] if spec["candidate"] == row["candidate"]
        ) for _, row in selected.iterrows()
    }
    overall = selected.sort_values(
        [str(cfg["selection_metric"]), *cfg["selection_tiebreakers"]], ascending=False
    ).iloc[0]
    selection = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "selected_candidate": overall["candidate"],
        "selected_family": overall["family"],
        "validation_mean_rank_ic": overall["mean_rank_ic"],
        "selection_period": "validation_2016_2020_only",
        "historical_oos_used_for_selection": False,
    }])

    oos_daily_frames, oos_portfolio_frames, fit_records = [], [], []
    states = {str(spec["candidate"]): {} for spec in selected_specs.values()}
    selected_models_last: dict[str, FittedCandidate] = {}
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        end_year = test_year - 1
        statistics = sum_statistics(
            [full_stats[year] for year in range(start_year, end_year)]
            + [purged_stats[end_year]]
        )
        sample = concatenate_samples(
            [full_samples[year] for year in range(start_year, end_year)]
            + [purged_samples[end_year]]
        )
        models = {}
        for family, spec in selected_specs.items():
            model = fit_candidate(spec, statistics, sample, len(factors), cfg)
            models[model.candidate] = model
            selected_models_last[model.candidate] = model
            fit_records.append({
                "test_year": test_year, "candidate": model.candidate,
                "family": family, "fit_seconds": model.fit_seconds,
                "fit_observations": model.fit_observations,
            })
        frame = _read_year(lake, test_year, columns)
        daily, portfolio = _evaluate(
            frame, models, factors, target, states, cfg, "historical_oos", test_year
        )
        oos_daily_frames.append(daily); oos_portfolio_frames.append(portfolio)
        print(f"benchmark oos year={test_year} models={len(models)}", flush=True)
    oos_daily = pd.concat(oos_daily_frames, ignore_index=True)
    oos_portfolio = pd.concat(oos_portfolio_frames, ignore_index=True)
    # Summary fit-time fields use the final expanding fit; the per-year audit keeps every fit.
    oos_summary = _summary(
        oos_daily, oos_portfolio, selected_models_last,
        int(cfg["annualization_periods"]), "historical_oos",
    )
    ridge_reference = str(selected.loc[selected["family"].eq("ridge"), "candidate"].iloc[0])
    bootstrap = _bootstrap(oos_daily, oos_portfolio, ridge_reference, cfg)
    hgb_reference = str(selected.loc[selected["family"].eq("hgb"), "candidate"].iloc[0])
    winner_name = str(overall["candidate"])
    winner_pairwise = pd.concat([
        _bootstrap(oos_daily, oos_portfolio, reference, cfg).loc[
            lambda frame: frame["candidate"].eq(winner_name)
        ]
        for reference in (ridge_reference, hgb_reference)
        if reference != winner_name
    ], ignore_index=True)
    annual_records = []
    for (candidate, year), group in oos_daily.groupby(["candidate", "test_year"]):
        portfolio_group = oos_portfolio.loc[
            oos_portfolio["candidate"].eq(candidate)
            & oos_portfolio["test_year"].eq(year)
        ]
        annual_records.append({
            "candidate": candidate, "year": int(year),
            **summarize(group, int(cfg["annualization_periods"])),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                portfolio_group, int(cfg["annualization_periods"])
            ).items()},
        })
    availability = pd.DataFrame([
        {"model": model, "status": status} for model, status in cfg["unavailable_models"].items()
    ] + [{"model": family, "status": "executed"} for family in selected["family"]])
    request = {**cfg, "factor_columns": factors}
    outputs = {
        "model_benchmark_validation_grid": grid,
        "model_benchmark_family_selection": selected,
        "model_benchmark_selection": selection,
        "model_benchmark_validation_daily": validation_daily,
        "model_benchmark_validation_portfolio": validation_portfolio,
        "model_benchmark_oos_summary": oos_summary,
        "model_benchmark_oos_daily": oos_daily,
        "model_benchmark_oos_portfolio": oos_portfolio,
        "model_benchmark_oos_annual": pd.DataFrame(annual_records),
        "model_benchmark_oos_bootstrap": bootstrap,
        "model_benchmark_winner_pairwise": winner_pairwise,
        "model_benchmark_fit_audit": pd.DataFrame(fit_records),
        "model_benchmark_availability": availability,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="model_benchmark.v1", request=request)
    print(selection.to_string(index=False))
    print(selected.to_string(index=False))
    print(oos_summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    run()
