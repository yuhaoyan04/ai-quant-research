"""Validation-only local sensitivity study for the benchmark random forest."""

from __future__ import annotations

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import purged_end_date, sum_statistics
from quant_research.model_benchmark import (
    _evaluate, _read_year, _summary, build_year_training, fit_candidate,
)
from quant_research.model_challenger import concatenate_samples
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "rf_hyperparameter_stability.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_local_grid(candidates: list[dict], baseline_candidate: str) -> None:
    """Require a genuine one-factor-at-a-time grid around one frozen baseline."""
    names = [str(spec["candidate"]) for spec in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    matches = [spec for spec in candidates if spec["candidate"] == baseline_candidate]
    if len(matches) != 1:
        raise ValueError("baseline_candidate must identify exactly one candidate")
    baseline = matches[0]
    parameters = ("n_estimators", "max_depth", "min_samples_leaf",
                  "max_features", "random_state")
    for spec in candidates:
        if spec is baseline:
            if spec.get("axis") != "baseline":
                raise ValueError("baseline must use axis=baseline")
            continue
        axis = str(spec.get("axis"))
        if axis not in parameters:
            raise ValueError(f"unsupported sensitivity axis: {axis}")
        changed = [name for name in parameters if spec.get(name) != baseline.get(name)]
        if changed != [axis]:
            raise ValueError(
                f"{spec['candidate']} must change only {axis}; changed={changed}"
            )


def annotate_robustness(summary: pd.DataFrame, candidates: list[dict],
                        baseline_candidate: str, thresholds: dict) -> pd.DataFrame:
    """Attach predeclared baseline deltas and pass/fail flags to validation results."""
    specs = pd.DataFrame([{
        "candidate": spec["candidate"], "axis": spec["axis"], "value": spec["value"],
        "n_estimators": spec["n_estimators"], "max_depth": spec["max_depth"],
        "min_samples_leaf": spec["min_samples_leaf"],
        "max_features": spec["max_features"], "random_state": spec["random_state"],
    } for spec in candidates])
    result = specs.merge(summary, on="candidate", validate="one_to_one")
    baseline = result.loc[result["candidate"].eq(baseline_candidate)]
    if len(baseline) != 1:
        raise ValueError("validation summary must contain the baseline exactly once")
    base = baseline.iloc[0]
    result["rank_ic_delta_vs_baseline"] = result["mean_rank_ic"] - base["mean_rank_ic"]
    result["sharpe_delta_vs_baseline"] = (
        result["portfolio_sharpe_zero_rate"] - base["portfolio_sharpe_zero_rate"]
    )
    result["passes_rank_ic_floor"] = result["rank_ic_delta_vs_baseline"].ge(
        -float(thresholds["maximum_rank_ic_drop"])
    )
    result["passes_sharpe_floor"] = result["sharpe_delta_vs_baseline"].ge(
        -float(thresholds["maximum_sharpe_drop"])
    )
    result["passes_robustness_floor"] = (
        result["passes_rank_ic_floor"] & result["passes_sharpe_floor"]
    )
    result["is_baseline"] = result["candidate"].eq(baseline_candidate)
    return result


def build_protocol_result(annotated: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    neighbors = annotated.loc[~annotated["is_baseline"]]
    pass_fraction = float(neighbors["passes_robustness_floor"].mean())
    threshold = float(cfg["robustness_thresholds"]["minimum_neighbor_pass_fraction"])
    return pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "purpose": cfg["purpose"],
        "baseline_candidate": cfg["baseline_candidate"],
        "training_period": "2006_2015",
        "evaluation_period": "validation_2016_2020_only",
        "historical_oos_accessed": False,
        "candidate_count": len(annotated),
        "neighbor_count": len(neighbors),
        "neighbors_passing": int(neighbors["passes_robustness_floor"].sum()),
        "neighbor_pass_fraction": pass_fraction,
        "minimum_neighbor_pass_fraction": threshold,
        "local_robustness_pass": bool(pass_fraction >= threshold),
        "best_candidate_is_not_promoted": True,
    }])


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    candidates = list(cfg["candidates"])
    validate_local_grid(candidates, str(cfg["baseline_candidate"]))
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")

    full_stats, purged_stats, full_samples, purged_samples = {}, {}, {}, {}
    start_year, train_end = int(cfg["train"]["start_year"]), int(cfg["train"]["end_year"])
    for year in range(start_year, train_end + 1):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31",
                                 int(cfg["label_horizon_trading_days"]))
        (full_stats[year], purged_stats[year], full_samples[year],
         purged_samples[year]) = build_year_training(
            frame, factors, target, cutoff, cfg
        )
        print(f"rf stability training year={year} sample_rows={len(full_samples[year].y)}",
              flush=True)
    statistics = sum_statistics(
        [full_stats[year] for year in range(start_year, train_end)]
        + [purged_stats[train_end]]
    )
    sample = concatenate_samples(
        [full_samples[year] for year in range(start_year, train_end)]
        + [purged_samples[train_end]]
    )
    models = {}
    for spec in candidates:
        model = fit_candidate(spec, statistics, sample, len(factors), cfg)
        models[model.candidate] = model
        print(f"rf stability fit candidate={model.candidate} seconds={model.fit_seconds:.3f}",
              flush=True)

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
    daily, portfolio = _evaluate(
        validation, models, factors, target, {name: {} for name in models}, cfg,
        "validation",
    )
    summary = _summary(daily, portfolio, models, int(cfg["annualization_periods"]),
                       "validation")
    annotated = annotate_robustness(
        summary, candidates, str(cfg["baseline_candidate"]),
        cfg["robustness_thresholds"],
    )
    protocol = build_protocol_result(annotated, cfg)
    axis_summary = annotated.loc[~annotated["is_baseline"]].groupby("axis").agg(
        candidate_count=("candidate", "size"),
        minimum_mean_rank_ic=("mean_rank_ic", "min"),
        maximum_mean_rank_ic=("mean_rank_ic", "max"),
        minimum_portfolio_sharpe=("portfolio_sharpe_zero_rate", "min"),
        maximum_portfolio_sharpe=("portfolio_sharpe_zero_rate", "max"),
        pass_fraction=("passes_robustness_floor", "mean"),
    ).reset_index()
    request = {**cfg, "factor_columns": factors}
    outputs = {
        "rf_hyperparameter_validation_summary": annotated,
        "rf_hyperparameter_axis_summary": axis_summary,
        "rf_hyperparameter_protocol": protocol,
        "rf_hyperparameter_validation_daily": daily,
        "rf_hyperparameter_validation_portfolio": portfolio,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="rf_hyperparameter_stability.v1",
                   request=request)
    print(annotated.to_string(index=False), flush=True)
    print(protocol.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
