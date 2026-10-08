"""Purged, walk-forward ridge baseline for cross-sectional stock ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake


@dataclass(frozen=True)
class SufficientStatistics:
    gram: np.ndarray
    cross: np.ndarray
    observations: int
    dates: int


def load_config() -> dict:
    with (ROOT / "config" / "model_baseline.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_feature_sets() -> dict:
    with (ROOT / "config" / "model_features.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def design_names(factors: list[str], include_missing: bool) -> list[str]:
    names = [f"{factor}__z" for factor in factors]
    if include_missing:
        names += [f"{factor}__missing_cs_z" for factor in factors]
    return names


def cross_sectional_design(group: pd.DataFrame, factors: list[str],
                           include_missing: bool) -> np.ndarray:
    """Use only statistics observable in the current date's cross section."""
    blocks = [group[[f"{factor}__z" for factor in factors]].to_numpy(dtype="float64")]
    if include_missing:
        missing = group[[f"{factor}__missing" for factor in factors]].to_numpy(dtype="float64")
        means = missing.mean(axis=0)
        standard_deviations = missing.std(axis=0, ddof=1)
        standardized = np.divide(
            missing - means, standard_deviations, out=np.zeros_like(missing),
            where=standard_deviations > 0,
        )
        blocks.append(standardized)
    design = np.concatenate(blocks, axis=1)
    if not np.isfinite(design).all():
        raise ValueError("model design contains non-finite values")
    return design


def weekly_rows(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(frame["asof_date"].drop_duplicates())
    parsed = pd.to_datetime(dates)
    selected = dates.groupby(parsed.dt.to_period("W-FRI")).max()
    return frame.loc[frame["asof_date"].isin(set(selected))].copy()


def purged_end_date(calendar: pd.DataFrame, end_date: str, horizon: int) -> str:
    open_days = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
    ).sort_values().drop_duplicates()
    eligible = open_days.loc[open_days.le(pd.Timestamp(end_date))]
    if len(eligible) <= horizon:
        raise ValueError("not enough trading dates to apply label purge")
    return eligible.iloc[-(horizon + 1)].strftime("%Y-%m-%d")


def year_statistics(frame: pd.DataFrame, factors: list[str], target: str,
                    include_missing: bool, purge_cutoff: str) -> tuple[SufficientStatistics, SufficientStatistics]:
    dimension = len(design_names(factors, include_missing))
    full_gram = np.zeros((dimension, dimension), dtype="float64")
    full_cross = np.zeros(dimension, dtype="float64")
    purged_gram = np.zeros_like(full_gram)
    purged_cross = np.zeros_like(full_cross)
    full_observations = purged_observations = full_dates = purged_dates = 0
    for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=False):
        target_available = group[target].notna().to_numpy()
        if not target_available.any():
            continue
        # Cross-sectional scaling uses the full then-observable universe. Only
        # after feature construction are rows without a realized label removed.
        x = cross_sectional_design(group, factors, include_missing)[target_available]
        usable_target = group.loc[target_available, target]
        y = rank_zscore(usable_target)
        gram, cross = x.T @ x, x.T @ y
        full_gram += gram
        full_cross += cross
        full_observations += len(usable_target)
        full_dates += 1
        if asof_date <= purge_cutoff:
            purged_gram += gram
            purged_cross += cross
            purged_observations += len(usable_target)
            purged_dates += 1
    return (
        SufficientStatistics(full_gram, full_cross, full_observations, full_dates),
        SufficientStatistics(purged_gram, purged_cross, purged_observations, purged_dates),
    )


def sum_statistics(items: list[SufficientStatistics]) -> SufficientStatistics:
    if not items:
        raise ValueError("no sufficient statistics supplied")
    return SufficientStatistics(
        sum((item.gram for item in items), np.zeros_like(items[0].gram)),
        sum((item.cross for item in items), np.zeros_like(items[0].cross)),
        sum(item.observations for item in items), sum(item.dates for item in items),
    )


def fit_ridge(statistics: SufficientStatistics, penalty_per_observation: float,
              indices: list[int] | None = None) -> np.ndarray:
    gram, cross = statistics.gram, statistics.cross
    if indices is not None:
        gram, cross = gram[np.ix_(indices, indices)], cross[indices]
    penalty = penalty_per_observation * statistics.observations
    return np.linalg.solve(gram + penalty * np.eye(len(cross)), cross)


def feature_indices(all_factors: list[str], selected_factors: list[str],
                    include_missing: bool) -> list[int]:
    all_names = design_names(all_factors, include_missing)
    return [all_names.index(name) for name in design_names(selected_factors, include_missing)]


def evaluate(frame: pd.DataFrame, factors: list[str], target: str, include_missing: bool,
             beta: np.ndarray, minimum_cross_section_size: int, bucket_count: int
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []
    for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=False):
        score = cross_sectional_design(group, factors, include_missing) @ beta
        predictions = group[["instrument_id", "asof_date", "available_date", target]].copy()
        predictions["score"] = score.astype("float32")
        prediction_frames.append(predictions)
        usable = predictions.loc[predictions[target].notna()]
        if len(usable) < minimum_cross_section_size:
            continue
        rank_ic = usable["score"].rank(method="average").corr(usable[target].rank(method="average"))
        percentile = usable["score"].rank(method="first", pct=True)
        bucket = (percentile * bucket_count).clip(upper=bucket_count - 1e-9).astype(int)
        bucket_returns = usable.assign(bucket=bucket).groupby("bucket")[target].mean()
        daily_records.append({
            "asof_date": asof_date, "cross_section_size": len(usable), "rank_ic": rank_ic,
            "top_minus_bottom": bucket_returns.get(bucket_count - 1, np.nan) - bucket_returns.get(0, np.nan),
        })
    return pd.DataFrame(daily_records), pd.concat(prediction_frames, ignore_index=True)


def summarize(daily: pd.DataFrame, annualization_periods: int) -> dict[str, float | int]:
    mean_ic, std_ic = daily["rank_ic"].mean(), daily["rank_ic"].std(ddof=1)
    return {
        "observations": len(daily), "mean_rank_ic": mean_ic, "rank_ic_std": std_ic,
        "annualized_icir": mean_ic / std_ic * np.sqrt(annualization_periods) if std_ic > 0 else np.nan,
        "mean_top_minus_bottom": daily["top_minus_bottom"].mean(),
    }


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg, feature_cfg = load_config(), load_feature_sets()
    all_factors = list(feature_cfg["full"])
    include_missing = bool(feature_cfg["include_missing_indicators"])
    target = str(cfg["target"])
    calendar = lake.read("calendar", "SSE")
    horizon = int(cfg["label_horizon_trading_days"])
    start_year, test_end = int(cfg["train"]["start_year"]), int(cfg["test"]["end_year"])
    stats_full: dict[int, SufficientStatistics] = {}
    stats_purged: dict[int, SufficientStatistics] = {}
    read_columns = ["instrument_id", "asof_date", "available_date", target]
    read_columns += [f"{factor}__z" for factor in all_factors]
    read_columns += [f"{factor}__missing" for factor in all_factors]

    for year in range(start_year, test_end + 1):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=read_columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31", horizon)
        stats_full[year], stats_purged[year] = year_statistics(
            frame, all_factors, target, include_missing, cutoff)
        print(f"statistics year={year} dates={stats_full[year].dates} purged_dates={stats_purged[year].dates}", flush=True)

    train_end = int(cfg["train"]["end_year"])
    train_stats = sum_statistics([stats_full[y] for y in range(start_year, train_end)] + [stats_purged[train_end]])
    validation_start, validation_end = int(cfg["validation"]["start_year"]), int(cfg["validation"]["end_year"])
    validation_frames = []
    for year in range(validation_start, validation_end + 1):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=read_columns)
        if year == validation_end:
            cutoff = purged_end_date(calendar, f"{year}-12-31", horizon)
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        validation_frames.append(frame)
    validation = pd.concat(validation_frames, ignore_index=True)

    grid_records: list[dict[str, float | str | int]] = []
    for set_name in ("full", "compact"):
        selected_factors = list(feature_cfg[set_name])
        indices = feature_indices(all_factors, selected_factors, include_missing)
        for penalty in cfg["ridge_penalties_per_observation"]:
            beta = fit_ridge(train_stats, float(penalty), indices)
            daily, _ = evaluate(validation, selected_factors, target, include_missing, beta,
                                int(cfg["minimum_cross_section_size"]), int(cfg["portfolio_buckets"]))
            record = {"feature_set": set_name, "ridge_penalty": float(penalty),
                      **summarize(daily, int(cfg["annualization_periods"]))}
            grid_records.append(record)
            print(f"validation {record}", flush=True)
    grid = pd.DataFrame(grid_records).sort_values(
        [str(cfg["selection_metric"]), "mean_top_minus_bottom"], ascending=False)
    winner = grid.iloc[0]
    selected_name, selected_penalty = str(winner["feature_set"]), float(winner["ridge_penalty"])
    selected_factors = list(feature_cfg[selected_name])
    selected_indices = feature_indices(all_factors, selected_factors, include_missing)
    validation_beta = fit_ridge(train_stats, selected_penalty, selected_indices)
    validation_daily, validation_predictions = evaluate(
        validation, selected_factors, target, include_missing, validation_beta,
        int(cfg["minimum_cross_section_size"]), int(cfg["portfolio_buckets"]),
    )

    prediction_frames, daily_frames, coefficient_records = [], [], []
    for test_year in range(int(cfg["test"]["start_year"]), test_end + 1):
        fit_end = test_year - 1
        fit_stats = sum_statistics([stats_full[y] for y in range(start_year, fit_end)] + [stats_purged[fit_end]])
        beta = fit_ridge(fit_stats, selected_penalty, selected_indices)
        test_frame = pd.read_parquet(lake.table_path("model_features", str(test_year)), columns=read_columns)
        daily, predictions = evaluate(test_frame, selected_factors, target, include_missing, beta,
                                      int(cfg["minimum_cross_section_size"]), int(cfg["portfolio_buckets"]))
        daily["test_year"], predictions["test_year"] = test_year, test_year
        daily_frames.append(daily)
        prediction_frames.append(predictions)
        coefficient_records.extend({
            "test_year": test_year, "feature": name, "coefficient": float(value),
            "fit_observations": fit_stats.observations, "ridge_penalty": selected_penalty,
        } for name, value in zip(design_names(selected_factors, include_missing), beta))
        print(f"test year={test_year} {summarize(daily, int(cfg['annualization_periods']))}", flush=True)

    daily_oos, predictions_oos = pd.concat(daily_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)
    annual = pd.DataFrame([{"test_year": year, **summarize(group, int(cfg["annualization_periods"]))}
                           for year, group in daily_oos.groupby("test_year")])
    selection = pd.DataFrame([{
        "selected_feature_set": selected_name, "selected_ridge_penalty": selected_penalty,
        "validation_mean_rank_ic": float(winner["mean_rank_ic"]),
        "validation_annualized_icir": float(winner["annualized_icir"]), "target": target,
        "temporal_scaler": "none; same-date cross-sectional transforms only",
        "selection_data_end": "2020-12-31",
    }])
    lake.write("model_validation_grid", grid, key="ridge_v1", source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_validation_daily", validation_daily, key="ridge_v1",
               source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_validation_predictions", validation_predictions, key="ridge_v1",
               source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_selection", selection, key="ridge_v1", source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_oos_daily", daily_oos, key="ridge_v1", source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_oos_annual", annual, key="ridge_v1", source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_oos_predictions", predictions_oos, key="ridge_v1", source="model_baseline.ridge.v1", request=cfg)
    lake.write("model_coefficients", pd.DataFrame(coefficient_records), key="ridge_v1",
               source="model_baseline.ridge.v1", request=cfg)
    print(selection.to_string(index=False))
    print(annual.to_string(index=False))


if __name__ == "__main__":
    run()
