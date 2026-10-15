import pandas as pd
import pytest

from quant_research.rf_hyperparameter_stability import (
    annotate_robustness, build_protocol_result, validate_local_grid,
)


def candidates() -> list[dict]:
    baseline = {
        "candidate": "base", "axis": "baseline", "value": 0,
        "n_estimators": 100, "max_depth": 6, "min_samples_leaf": 100,
        "max_features": 0.7, "random_state": 1,
    }
    return [baseline, {**baseline, "candidate": "depth4", "axis": "max_depth",
                       "value": 4, "max_depth": 4}]


def test_grid_requires_one_factor_at_a_time() -> None:
    validate_local_grid(candidates(), "base")
    broken = candidates()
    broken[1]["max_features"] = 0.5
    with pytest.raises(ValueError, match="must change only"):
        validate_local_grid(broken, "base")


def test_robustness_thresholds_and_protocol_are_predeclared() -> None:
    summary = pd.DataFrame([
        {"candidate": "base", "mean_rank_ic": 0.10,
         "portfolio_sharpe_zero_rate": 1.0},
        {"candidate": "depth4", "mean_rank_ic": 0.096,
         "portfolio_sharpe_zero_rate": 0.76},
    ])
    thresholds = {"maximum_rank_ic_drop": 0.005, "maximum_sharpe_drop": 0.25,
                  "minimum_neighbor_pass_fraction": 0.75}
    annotated = annotate_robustness(summary, candidates(), "base", thresholds)
    neighbor = annotated.loc[annotated["candidate"].eq("depth4")].iloc[0]
    assert neighbor["passes_robustness_floor"]
    cfg = {
        "experiment_id": "test", "purpose": "validation_only",
        "baseline_candidate": "base", "robustness_thresholds": thresholds,
    }
    protocol = build_protocol_result(annotated, cfg).iloc[0]
    assert protocol["historical_oos_accessed"] == False
    assert protocol["local_robustness_pass"]
    assert protocol["best_candidate_is_not_promoted"]
