"""Redundancy and marginal-signal diagnostics using selection periods only."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.preprocessing import load_config as load_preprocessing_config


def load_config() -> dict:
    with (ROOT / "config" / "feature_analysis.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rank_zscore(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average").to_numpy(dtype="float64")
    standard_deviation = ranks.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.zeros(len(ranks), dtype="float64")
    return (ranks - ranks.mean()) / standard_deviation


def daily_diagnostics(frame: pd.DataFrame, feature_columns: list[str], target: str,
                      minimum_observations: int, ridge_penalty_per_observation: float
                      ) -> tuple[np.ndarray, int, pd.DataFrame]:
    """Return equal-day-weighted correlation sums and daily marginal betas."""
    correlation_sum = np.zeros((len(feature_columns), len(feature_columns)), dtype="float64")
    usable_days = 0
    beta_records: list[dict[str, float | str | int]] = []
    for asof_date, group in frame.groupby("asof_date", sort=False):
        usable = group[[*feature_columns, target]].dropna(subset=[target])
        if len(usable) < minimum_observations:
            continue
        x = usable[feature_columns].to_numpy(dtype="float64")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite preprocessed feature on {asof_date}")
        centered = x - x.mean(axis=0)
        norms = np.sqrt((centered * centered).sum(axis=0))
        denominator = np.outer(norms, norms)
        daily_correlation = np.divide(
            centered.T @ centered,
            denominator,
            out=np.zeros((len(feature_columns), len(feature_columns)), dtype="float64"),
            where=denominator > 0,
        )
        # A factor can be structurally unavailable for an entire early date;
        # its preprocessed exposure is then constant zero. It contributes zero
        # association without raising divide-by-zero warnings.
        correlation_sum += daily_correlation
        usable_days += 1

        y = rank_zscore(usable[target])
        gram = x.T @ x
        penalty = ridge_penalty_per_observation * len(x)
        beta = np.linalg.solve(gram + penalty * np.eye(len(feature_columns)), x.T @ y)
        beta_records.extend({
            "asof_date": asof_date,
            "factor": feature.removesuffix("__z"),
            "marginal_rank_coefficient": float(value),
            "cross_section_size": len(x),
        } for feature, value in zip(feature_columns, beta))
    return correlation_sum, usable_days, pd.DataFrame(beta_records)


def connected_correlation_clusters(correlation: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Connected components of the absolute-correlation graph."""
    factors = list(correlation.index)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1:]:
            if abs(float(correlation.loc[left, right])) >= threshold:
                neighbours[left].add(right)
                neighbours[right].add(left)
    records: list[dict[str, str | int]] = []
    visited: set[str] = set()
    cluster_number = 0
    for factor in factors:
        if factor in visited:
            continue
        cluster_number += 1
        stack = [factor]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(neighbours[current] - visited)
        for member in sorted(component):
            records.append({"cluster": cluster_number, "factor": member, "cluster_size": len(component)})
    return pd.DataFrame(records)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    preprocessing = load_preprocessing_config()
    factors = list(preprocessing["factor_columns"])
    features = [f"{factor}__z" for factor in factors]
    target = str(preprocessing["target"])
    all_correlation_sum = np.zeros((len(features), len(features)), dtype="float64")
    all_days = 0
    correlation_records: list[pd.DataFrame] = []
    daily_beta_records: list[pd.DataFrame] = []

    for split in cfg["selection_splits"]:
        split_sum = np.zeros_like(all_correlation_sum)
        split_days = 0
        split_betas: list[pd.DataFrame] = []
        for year in range(int(split["start_year"]), int(split["end_year"]) + 1):
            path = lake.table_path("model_features", str(year))
            if not path.exists():
                raise RuntimeError(f"model feature year missing: {year}")
            frame = pd.read_parquet(path, columns=["asof_date", target, *features])
            correlation_sum, days, daily_betas = daily_diagnostics(
                frame, features, target, int(cfg["minimum_daily_observations"]),
                float(cfg["ridge_penalty_per_observation"]),
            )
            split_sum += correlation_sum
            split_days += days
            if not daily_betas.empty:
                daily_betas["sample_split"] = split["name"]
                split_betas.append(daily_betas)
            print(f"split={split['name']} year={year} days={days}", flush=True)
        if split_days == 0:
            raise RuntimeError(f"no usable dates in split {split['name']}")
        matrix = pd.DataFrame(split_sum / split_days, index=factors, columns=factors)
        long = matrix.rename_axis("factor_a").reset_index().melt(
            id_vars="factor_a", var_name="factor_b", value_name="mean_daily_correlation")
        long["sample_split"] = split["name"]
        long["trading_days"] = split_days
        correlation_records.append(long)
        daily_beta_records.extend(split_betas)
        all_correlation_sum += split_sum
        all_days += split_days

    selection_matrix = pd.DataFrame(all_correlation_sum / all_days, index=factors, columns=factors)
    selection_long = selection_matrix.rename_axis("factor_a").reset_index().melt(
        id_vars="factor_a", var_name="factor_b", value_name="mean_daily_correlation")
    selection_long["sample_split"] = "train_plus_validation"
    selection_long["trading_days"] = all_days
    correlation_records.append(selection_long)
    correlations = pd.concat(correlation_records, ignore_index=True)

    threshold = float(cfg["correlation_cluster_threshold"])
    pairs = selection_long.loc[
        selection_long["factor_a"].lt(selection_long["factor_b"])
        & selection_long["mean_daily_correlation"].abs().ge(threshold)
    ].copy().sort_values("mean_daily_correlation", key=lambda values: values.abs(), ascending=False)
    clusters = connected_correlation_clusters(selection_matrix, threshold)
    daily_betas = pd.concat(daily_beta_records, ignore_index=True)
    marginal = daily_betas.groupby(["sample_split", "factor"], as_index=False).agg(
        observations=("marginal_rank_coefficient", "size"),
        mean_marginal_coefficient=("marginal_rank_coefficient", "mean"),
        std_marginal_coefficient=("marginal_rank_coefficient", "std"),
    )
    marginal["t_stat"] = marginal["mean_marginal_coefficient"] / (
        marginal["std_marginal_coefficient"] / np.sqrt(marginal["observations"])
    )
    lake.write("factor_correlation", correlations, key="v1", source="feature_analysis.v1", request=cfg)
    lake.write("factor_redundancy_pairs", pairs, key="v1", source="feature_analysis.v1", request=cfg)
    lake.write("factor_clusters", clusters, key="v1", source="feature_analysis.v1", request=cfg)
    lake.write("factor_marginal_daily", daily_betas, key="v1", source="feature_analysis.v1", request=cfg)
    lake.write("factor_marginal_summary", marginal, key="v1", source="feature_analysis.v1", request=cfg)
    print("high-correlation pairs")
    print(pairs[["factor_a", "factor_b", "mean_daily_correlation"]].to_string(index=False))
    print("marginal coefficients")
    print(marginal.to_string(index=False))


if __name__ == "__main__":
    run()
