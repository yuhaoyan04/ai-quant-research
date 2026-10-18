import numpy as np
import pandas as pd
import pytest

from quant_research.incremental_alpha_test import (
    _drifted_target_state,
    _paired_metrics,
    _turnover_controlled_target,
    match_industry_peers,
)


def config() -> dict:
    return {
        "matching": {
            "previous_peer_cost_multiplier": 0.0,
            "held_peer_fallback_penalty": 1000.0,
            "self_match_penalty": 1_000_000.0,
        }
    }


def test_peer_matching_is_unique_same_industry_and_excludes_holdings() -> None:
    source = pd.DataFrame([
        {"instrument_id": "A", "industry_broad_code": "I", "target_weight": 0.6,
         "style_size": -1.0, "style_beta": 0.0},
        {"instrument_id": "B", "industry_broad_code": "I", "target_weight": 0.4,
         "style_size": 1.0, "style_beta": 0.0},
    ])
    universe = pd.concat([source.drop(columns="target_weight"), pd.DataFrame([
        {"instrument_id": "C", "industry_broad_code": "I", "style_size": -0.9, "style_beta": 0.1},
        {"instrument_id": "D", "industry_broad_code": "I", "style_size": 1.1, "style_beta": -0.1},
    ])], ignore_index=True)

    result = match_industry_peers(source, universe, ["size", "beta"], {}, config())

    assert set(result["instrument_id"]) == {"C", "D"}
    assert result["instrument_id"].is_unique
    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert not result["used_held_peer_fallback"].any()


def test_turnover_overlay_respects_limit_and_preserves_weight_cap() -> None:
    previous = pd.Series({"A": 0.5, "B": 0.5})
    fresh = pd.Series({"C": 0.5, "D": 0.5})

    controlled, unconstrained, realized = _turnover_controlled_target(
        fresh, previous, 0.3, maximum_weight=0.5
    )

    assert unconstrained == pytest.approx(1.0)
    assert realized == pytest.approx(0.3)
    assert controlled.sum() == pytest.approx(1.0)
    assert controlled.max() <= 0.5


def test_turnover_overlay_repairs_drift_above_stock_cap() -> None:
    previous = pd.Series({"A": 0.51, "B": 0.49})
    fresh = pd.Series({"A": 0.50, "B": 0.50})

    controlled, _, realized = _turnover_controlled_target(
        fresh, previous, 0.30, maximum_weight=0.5
    )

    assert controlled.max() <= 0.5
    assert controlled.sum() == pytest.approx(1.0)
    assert realized <= 0.30 + 1e-10


def test_paired_metrics_use_positive_improvement_direction() -> None:
    model = np.array([0.01, 0.02, -0.01, 0.01])
    style = np.array([0.005, 0.01, -0.02, 0.005])
    model_turnover = np.full(4, 0.01)
    style_turnover = np.full(4, 0.02)

    result = _paired_metrics(model, style, model_turnover, style_turnover, 252)

    assert result["incremental_net_return_annualized_arithmetic"] > 0
    assert result["turnover_improvement"] > 0
    assert result["sharpe_improvement"] > 0


def test_target_state_drifts_before_next_turnover_constraint() -> None:
    state = {
        "weights": pd.Series({"A": 0.5, "B": 0.5}),
        "returns": pd.Series({"A": 0.1, "B": 0.0}),
    }

    drifted = _drifted_target_state(state)

    assert drifted.sum() == pytest.approx(1.0)
    assert drifted["A"] == pytest.approx(0.55 / 1.05)
