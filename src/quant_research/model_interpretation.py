"""Grouped within-date permutation importance for the selected tree model."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import yaml

from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import cross_sectional_design, load_feature_sets, purged_end_date, weekly_rows
from quant_research.model_challenger import (
    build_training_sample,
    concatenate_samples,
    deterministic_indices,
    load_config as load_model_config,
    make_model,
)
from quant_research.robustness import circular_block_bootstrap_mean


def load_config() -> dict:
    with (ROOT / "config" / "model_interpretation.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _spearman(score: np.ndarray, target: np.ndarray) -> float:
    return float(pd.Series(score).rank(method="average").corr(
        pd.Series(target).rank(method="average")))


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    model_cfg, feature_cfg = load_model_config(), load_feature_sets()
    selection = lake.read("model_challenger_selection", "hgb_v1")
    if selection.empty:
        raise RuntimeError("challenger selection missing")
    selected_name = str(selection.iloc[0]["selected_tree_candidate"])
    parameters = next(item for item in model_cfg["candidates"] if item["name"] == selected_name)
    factors = list(feature_cfg["full"])
    include_missing = bool(feature_cfg["include_missing_indicators"])
    target = str(model_cfg["target"])
    calendar = lake.read("calendar", "SSE")
    columns = ["instrument_id", "asof_date", "available_date", target]
    columns += [f"{factor}__z" for factor in factors]
    columns += [f"{factor}__missing" for factor in factors]

    samples = []
    train_start, train_end = int(model_cfg["train"]["start_year"]), int(model_cfg["train"]["end_year"])
    for year in range(train_start, train_end + 1):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=columns)
        cutoff = (purged_end_date(calendar, f"{year}-12-31",
                                  int(model_cfg["label_horizon_trading_days"]))
                  if year == train_end else None)
        samples.append(build_training_sample(
            frame, factors, target, include_missing,
            int(model_cfg["sample_per_training_date"]), cutoff))
    train = concatenate_samples(samples)
    model = make_model(parameters, int(model_cfg["random_state"]))
    model.fit(train.x, train.y)
    print(f"interpretation model fit rows={len(train.y)} candidate={selected_name}", flush=True)

    records: list[dict] = []
    for year in range(int(model_cfg["validation"]["start_year"]),
                      int(model_cfg["validation"]["end_year"]) + 1):
        frame = pd.read_parquet(lake.table_path("model_features", str(year)), columns=columns)
        if year == int(model_cfg["validation"]["end_year"]):
            cutoff = purged_end_date(calendar, f"{year}-12-31",
                                     int(model_cfg["label_horizon_trading_days"]))
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        for asof_date, group in weekly_rows(frame).groupby("asof_date", sort=False):
            usable_mask = group[target].notna().to_numpy()
            if usable_mask.sum() < int(model_cfg["minimum_cross_section_size"]):
                continue
            usable = group.loc[usable_mask]
            x = cross_sectional_design(group, factors, include_missing)[usable_mask]
            y = np.asarray(rank_zscore(usable[target]), dtype="float64")
            chosen = deterministic_indices(usable, int(cfg["evaluation_sample_per_date"]))
            x, y = x[chosen], y[chosen]
            baseline_ic = _spearman(model.predict(x), y)
            for factor_index, factor in enumerate(factors):
                seed_text = f"{asof_date}|{factor}|{cfg['random_seed']}".encode()
                rng = np.random.default_rng(zlib.crc32(seed_text))
                permutation = rng.permutation(len(x))
                permuted = x.copy()
                permuted[:, factor_index] = x[permutation, factor_index]
                if include_missing:
                    missing_index = factor_index + len(factors)
                    permuted[:, missing_index] = x[permutation, missing_index]
                permuted_ic = _spearman(model.predict(permuted), y)
                records.append({
                    "asof_date": asof_date, "factor": factor,
                    "baseline_rank_ic": baseline_ic, "permuted_rank_ic": permuted_ic,
                    "rank_ic_drop": baseline_ic - permuted_ic,
                })
        print(f"interpretation validation year={year}", flush=True)

    daily = pd.DataFrame(records)
    summary_rows: list[dict] = []
    for factor, group in daily.groupby("factor", sort=False):
        draws = circular_block_bootstrap_mean(
            group.sort_values("asof_date")["rank_ic_drop"].to_numpy(),
            int(cfg["block_length_weeks"]), int(cfg["bootstrap_samples"]), int(cfg["random_seed"]),
        )
        summary_rows.append({
            "factor": factor,
            "mean_rank_ic_drop": group["rank_ic_drop"].mean(),
            "median_rank_ic_drop": group["rank_ic_drop"].median(),
            "ci_2_5pct": np.quantile(draws, 0.025),
            "ci_97_5pct": np.quantile(draws, 0.975),
            "probability_importance_nonpositive": (np.sum(draws <= 0) + 1) / (len(draws) + 1),
            "dates": group["asof_date"].nunique(),
        })
    summary = pd.DataFrame(summary_rows).sort_values("mean_rank_ic_drop", ascending=False)
    lake.write("model_permutation_importance_daily", daily, key="hgb_v1",
               source="model_interpretation.permutation.v1", request=cfg)
    lake.write("model_permutation_importance", summary, key="hgb_v1",
               source="model_interpretation.permutation.v1", request=cfg)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
