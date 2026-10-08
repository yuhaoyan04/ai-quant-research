"""Purged walk-forward nonlinear challenger and validation-selected blend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import (
    cross_sectional_design,
    design_names,
    load_feature_sets,
    purged_end_date,
    summarize,
    weekly_rows,
)


@dataclass(frozen=True)
class TrainingSample:
    x: np.ndarray
    y: np.ndarray
    dates: int


def load_config() -> dict:
    with (ROOT / "config" / "model_challenger.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deterministic_indices(group: pd.DataFrame, maximum: int) -> np.ndarray:
    """Outcome-independent, reproducible equal-count sampling within a date."""
    if len(group) <= maximum:
        return np.arange(len(group))
    hashes = pd.util.hash_pandas_object(
        group[["instrument_id", "asof_date"]], index=False).to_numpy(dtype="uint64")
    return np.argpartition(hashes, maximum - 1)[:maximum]


def build_training_sample(frame: pd.DataFrame, factors: list[str], target: str,
                          include_missing: bool, sample_per_date: int,
                          end_date: str | None = None) -> TrainingSample:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dates = 0
    for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=False):
        if end_date is not None and str(asof_date) > end_date:
            continue
        usable_mask = group[target].notna().to_numpy()
        if not usable_mask.any():
            continue
        usable_group = group.loc[usable_mask]
        x = cross_sectional_design(group, factors, include_missing)[usable_mask]
        y = np.asarray(rank_zscore(usable_group[target]), dtype="float64")
        chosen = deterministic_indices(usable_group, sample_per_date)
        xs.append(x[chosen].astype("float32"))
        ys.append(y[chosen].astype("float32"))
        dates += 1
    if not xs:
        width = len(design_names(factors, include_missing))
        return TrainingSample(np.empty((0, width), dtype="float32"),
                              np.empty(0, dtype="float32"), 0)
    return TrainingSample(np.concatenate(xs), np.concatenate(ys), dates)


def concatenate_samples(samples: list[TrainingSample]) -> TrainingSample:
    usable = [sample for sample in samples if len(sample.y)]
    if not usable:
        raise ValueError("no nonlinear training observations")
    return TrainingSample(np.concatenate([sample.x for sample in usable]),
                          np.concatenate([sample.y for sample in usable]),
                          sum(sample.dates for sample in usable))


def make_model(parameters: dict, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(parameters["max_iter"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        early_stopping=False,
        random_state=random_state,
    )


def prediction_metrics(predictions: pd.DataFrame, target: str, minimum_size: int,
                       bucket_count: int) -> pd.DataFrame:
    records: list[dict] = []
    for asof_date, group in predictions.groupby("asof_date", sort=False):
        usable = group.loc[group[target].notna()]
        if len(usable) < minimum_size:
            continue
        rank_ic = usable["score"].rank(method="average").corr(
            usable[target].rank(method="average"))
        percentile = usable["score"].rank(method="first", pct=True)
        bucket = (percentile * bucket_count).clip(upper=bucket_count - 1e-9).astype(int)
        returns = usable.assign(bucket=bucket).groupby("bucket")[target].mean()
        records.append({
            "asof_date": asof_date,
            "cross_section_size": len(usable),
            "rank_ic": rank_ic,
            "top_minus_bottom": returns.get(bucket_count - 1, np.nan) - returns.get(0, np.nan),
        })
    return pd.DataFrame(records)


def predict_frame(frame: pd.DataFrame, model: HistGradientBoostingRegressor,
                  factors: list[str], target: str, include_missing: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in weekly_rows(frame).groupby("asof_date", sort=False):
        x = cross_sectional_design(group, factors, include_missing)
        result = group[["instrument_id", "asof_date", "available_date", target]].copy()
        result["score"] = model.predict(x).astype("float32")
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def rank_normalized_score(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("asof_date")[column].rank(method="average", pct=True) - 0.5


def blend_predictions(tree: pd.DataFrame, ridge: pd.DataFrame, tree_weight: float,
                      target: str) -> pd.DataFrame:
    merged = tree.rename(columns={"score": "tree_score"}).merge(
        ridge[["instrument_id", "asof_date", "score"]].rename(columns={"score": "ridge_score"}),
        on=["instrument_id", "asof_date"], how="inner", validate="one_to_one")
    tree_rank = rank_normalized_score(merged, "tree_score")
    ridge_rank = rank_normalized_score(merged, "ridge_score")
    merged["score"] = (tree_weight * tree_rank + (1 - tree_weight) * ridge_rank).astype("float32")
    return merged[["instrument_id", "asof_date", "available_date", target, "score"]]


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg, feature_cfg = load_config(), load_feature_sets()
    factors = list(feature_cfg["full"])
    include_missing = bool(feature_cfg["include_missing_indicators"])
    target = str(cfg["target"])
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    test_end = int(cfg["test"]["end_year"])
    horizon = int(cfg["label_horizon_trading_days"])
    columns = ["instrument_id", "asof_date", "available_date", target]
    columns += [f"{factor}__z" for factor in factors]
    columns += [f"{factor}__missing" for factor in factors]

    full_samples: dict[int, TrainingSample] = {}
    purged_samples: dict[int, TrainingSample] = {}
    for year in range(start_year, test_end):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31", horizon)
        full_samples[year] = build_training_sample(
            frame, factors, target, include_missing, int(cfg["sample_per_training_date"]))
        purged_samples[year] = build_training_sample(
            frame, factors, target, include_missing, int(cfg["sample_per_training_date"]), cutoff)
        print(f"sample year={year} rows={len(full_samples[year].y)} dates={full_samples[year].dates}",
              flush=True)

    train_end = int(cfg["train"]["end_year"])
    train = concatenate_samples(
        [full_samples[year] for year in range(start_year, train_end)]
        + [purged_samples[train_end]])
    models: dict[str, HistGradientBoostingRegressor] = {}
    for parameters in cfg["candidates"]:
        model = make_model(parameters, int(cfg["random_state"]))
        model.fit(train.x, train.y)
        models[str(parameters["name"])] = model
        print(f"fit candidate={parameters['name']} rows={len(train.y)}", flush=True)

    validation_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in models}
    validation_start = int(cfg["validation"]["start_year"])
    validation_end = int(cfg["validation"]["end_year"])
    for year in range(validation_start, validation_end + 1):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=columns)
        if year == validation_end:
            cutoff = purged_end_date(calendar, f"{year}-12-31", horizon)
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        for name, model in models.items():
            validation_predictions[name].append(
                predict_frame(frame, model, factors, target, include_missing))
        print(f"validation predictions year={year}", flush=True)

    grid_rows: list[dict] = []
    combined_validation: dict[str, pd.DataFrame] = {}
    for parameters in cfg["candidates"]:
        name = str(parameters["name"])
        predictions = pd.concat(validation_predictions[name], ignore_index=True)
        combined_validation[name] = predictions
        daily = prediction_metrics(predictions, target, int(cfg["minimum_cross_section_size"]),
                                   int(cfg["portfolio_buckets"]))
        grid_rows.append({"candidate": name, **parameters,
                          **summarize(daily, int(cfg["annualization_periods"]))})
    grid = pd.DataFrame(grid_rows).sort_values(
        [str(cfg["selection_metric"]), "mean_top_minus_bottom"], ascending=False)
    selected_name = str(grid.iloc[0]["candidate"])
    selected_tree_validation = combined_validation[selected_name]

    ridge_validation = lake.read("model_validation_predictions", "ridge_v1")
    blend_rows: list[dict] = []
    validation_blends: dict[float, pd.DataFrame] = {}
    for weight in cfg["blend_tree_weights"]:
        blended = blend_predictions(selected_tree_validation, ridge_validation, float(weight), target)
        validation_blends[float(weight)] = blended
        daily = prediction_metrics(blended, target, int(cfg["minimum_cross_section_size"]),
                                   int(cfg["portfolio_buckets"]))
        blend_rows.append({"tree_weight": float(weight),
                           **summarize(daily, int(cfg["annualization_periods"]))})
    blend_grid = pd.DataFrame(blend_rows).sort_values(
        [str(cfg["selection_metric"]), "mean_top_minus_bottom"], ascending=False)
    selected_tree_weight = float(blend_grid.iloc[0]["tree_weight"])
    selected_validation = validation_blends[selected_tree_weight]
    selected_validation_daily = prediction_metrics(
        selected_validation, target, int(cfg["minimum_cross_section_size"]),
        int(cfg["portfolio_buckets"]))

    test_prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict] = []
    selected_parameters = next(item for item in cfg["candidates"] if item["name"] == selected_name)
    ridge_oos = lake.read("model_oos_predictions", "ridge_v1")
    for test_year in range(int(cfg["test"]["start_year"]), test_end + 1):
        fit_end = test_year - 1
        fit_sample = concatenate_samples(
            [full_samples[year] for year in range(start_year, fit_end)]
            + [purged_samples[fit_end]])
        model = make_model(selected_parameters, int(cfg["random_state"]))
        model.fit(fit_sample.x, fit_sample.y)
        frame = pd.read_parquet(lake.table_path("model_features", str(test_year)), columns=columns)
        tree_predictions = predict_frame(frame, model, factors, target, include_missing)
        ridge_year = ridge_oos.loc[ridge_oos["test_year"].eq(test_year)]
        blended = blend_predictions(tree_predictions, ridge_year, selected_tree_weight, target)
        blended["test_year"] = test_year
        test_prediction_frames.append(blended)
        fit_rows.append({"test_year": test_year, "fit_end_year": fit_end,
                         "fit_rows": len(fit_sample.y), "fit_dates": fit_sample.dates})
        print(f"walk-forward test_year={test_year} fit_rows={len(fit_sample.y)}", flush=True)

    oos_predictions = pd.concat(test_prediction_frames, ignore_index=True)
    oos_daily = prediction_metrics(oos_predictions, target, int(cfg["minimum_cross_section_size"]),
                                   int(cfg["portfolio_buckets"]))
    oos_daily["test_year"] = oos_daily["asof_date"].str.slice(0, 4).astype(int)
    annual = pd.DataFrame([
        {"test_year": int(year), **summarize(group, int(cfg["annualization_periods"]))}
        for year, group in oos_daily.groupby("test_year")
    ])
    selection = pd.DataFrame([{
        "selected_tree_candidate": selected_name,
        "selected_tree_weight": selected_tree_weight,
        "ridge_weight": 1 - selected_tree_weight,
        "target": target,
        "selection_period": "validation_2016_2020_only",
        "early_stopping": False,
        "sample_per_training_date": int(cfg["sample_per_training_date"]),
    }])
    lake.write("model_challenger_grid", grid, key="hgb_v1", source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_blend_grid", blend_grid, key="hgb_v1", source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_selection", selection, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_validation_daily", selected_validation_daily, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_validation_predictions", selected_validation, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_oos_daily", oos_daily, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_oos_annual", annual, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_oos_predictions", oos_predictions, key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    lake.write("model_challenger_fit_audit", pd.DataFrame(fit_rows), key="hgb_v1",
               source="model_challenger.hgb.v1", request=cfg)
    print(selection.to_string(index=False))
    print(grid.to_string(index=False))
    print(blend_grid.to_string(index=False))
    print(annual.to_string(index=False))


if __name__ == "__main__":
    run()
