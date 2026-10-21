from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from quant_research.idiosyncratic_portfolio_validation import (
    load_config,
    solve_style_replica_weights,
    validate_config,
)


def test_portfolio_config_is_validation_only_and_cost_matched() -> None:
    cfg = load_config(); validate_config(cfg)
    assert cfg["validation"]["end_year"] == 2020
    assert cfg["portfolio"]["round_trip_cost_bps"] == 20.0
    assert cfg["historical_oos_evaluation_enabled"] is False
    assert cfg["future_lockbox_access_allowed"] is False


def test_portfolio_config_rejects_oos_access_and_cost_drift() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["historical_oos_evaluation_enabled"] = True
    with pytest.raises(ValueError, match="historical OOS"):
        validate_config(cfg)
    cfg = copy.deepcopy(load_config())
    cfg["portfolio"]["round_trip_cost_bps"] = 10.0
    with pytest.raises(ValueError, match="20bp"):
        validate_config(cfg)


def test_primary_families_are_fixed_before_portfolio_run() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["inference"]["primary_families"] = ["residual_target_minus_raw_target"]
    with pytest.raises(ValueError, match="families changed"):
        validate_config(cfg)


def test_promotion_gate_is_not_share_level_claim() -> None:
    cfg = load_config()
    assert cfg["scope_note"].startswith("Validation-only")
    assert cfg["promotion_gate"]["allow_turnover_relaxation"] is False
    assert cfg["portfolio"]["minimum_realized_weight_fraction"] == 0.95


def test_style_replica_lp_matches_pit_exposures_without_model_holdings() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["style_benchmark"]["maximum_stock_weight"] = 0.02
    styles = list(cfg["controls"]["styles"])
    base = pd.DataFrame({
        "instrument_id": [f"S{i:03d}" for i in range(50)],
        "industry_broad_code": ["A"] * 25 + ["B"] * 25,
        "target_weight": np.full(50, 0.02),
    })
    for number, style in enumerate(styles):
        base[f"style_{style}"] = np.tile(np.linspace(-1, 1, 25), 2) * (number + 1) / 6
    peers = base.copy()
    peers["instrument_id"] = [f"P{i:03d}" for i in range(50)]
    universe = pd.concat([base, peers], ignore_index=True)
    universe["fwd_return_5d_lag1"] = np.linspace(-0.2, 0.2, len(universe))

    weights, diagnostics = solve_style_replica_weights(
        base, universe, None, styles, cfg
    )

    assert diagnostics["excluded_ml_holdings"] is True
    assert not set(weights.index) & set(base["instrument_id"])
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights.max() <= 0.02 + 1e-9
    selected = universe.set_index("instrument_id").loc[weights.index]
    for style in styles:
        expected = float(base["target_weight"] @ base[f"style_{style}"])
        actual = float(weights @ selected[f"style_{style}"])
        assert actual == pytest.approx(expected, abs=1e-8)


def test_style_replica_retries_with_overlap_when_exclusion_removes_industry() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["style_benchmark"]["maximum_stock_weight"] = 0.5
    styles = list(cfg["controls"]["styles"])
    source = pd.DataFrame({
        "instrument_id": ["A1", "B1"],
        "industry_broad_code": ["A", "B"],
        "target_weight": [0.5, 0.5],
        **{f"style_{style}": [0.0, 0.0] for style in styles},
    })
    universe = source.copy()
    universe["fwd_return_5d_lag1"] = [0.01, -0.01]

    weights, diagnostics = solve_style_replica_weights(
        source, universe, None, styles, cfg
    )

    assert diagnostics["excluded_ml_holdings"] is False
    assert weights.to_dict() == pytest.approx({"A1": 0.5, "B1": 0.5})
