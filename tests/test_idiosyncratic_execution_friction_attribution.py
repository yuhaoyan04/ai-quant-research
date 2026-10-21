from copy import deepcopy

import pandas as pd
import pytest

from quant_research.idiosyncratic_execution_friction_attribution import (
    _attribution,
    _waterfall,
    load_config,
    validate_config,
)


def test_friction_config_relaxes_exactly_one_constraint_per_scenario() -> None:
    cfg = load_config()
    validate_config(cfg)

    invalid = deepcopy(cfg)
    invalid["scenarios"][0]["enforce_volume_participation"] = False
    with pytest.raises(ValueError, match="must relax exactly"):
        validate_config(invalid)


def test_attribution_uses_same_path_baseline_and_discloses_non_additivity() -> None:
    cfg = load_config()
    common = {
        "fill_ratio_by_shares": 0.9,
        "mean_target_tracking_l1": 0.1,
        "p95_target_tracking_l1": 0.2,
        "mean_blocked_buy_weight": 0.01,
        "mean_blocked_sell_weight": 0.01,
        "mean_cash_weight": 0.001,
        "maximum_drawdown": -0.2,
        "sharpe_reliable_risk_free": 0.5,
    }
    summary = pd.DataFrame([
        {
            **common, "path_id": path, "scenario_id": scenario,
            "relaxed_constraint": relaxed, "annualized_return": value,
        }
        for path, offset in (("raw", 0.0), ("residual", 0.1))
        for scenario, relaxed, value in (
            ("all_constraints", "none", 0.10 + offset),
            ("no_trade_flags", "trade_flags", 0.15 + offset),
            ("fractional_upper_bound", "all_constraints", 0.20 + offset),
        )
    ])

    result = _attribution(summary, cfg)
    relaxed = result.loc[result["scenario_id"].eq("no_trade_flags")]

    assert relaxed["annualized_return_uplift_vs_all_constraints"].round(12).eq(0.05).all()
    assert relaxed["fraction_of_upper_bound_gap_closed"].round(12).eq(0.5).all()
    assert result["effects_are_non_additive"].all()


def test_ordered_waterfall_reconciles_to_upper_bound() -> None:
    cfg = load_config()
    rows = []
    for path in ("raw", "residual"):
        for scenario, value in (
            ("all_constraints", 0.10),
            ("no_trade_flags", 0.14),
            ("no_trade_flags_no_volume", 0.18),
            ("no_trade_flags_no_volume_no_lot", 0.19),
            ("fractional_upper_bound", 0.20),
        ):
            rows.append({
                "path_id": path, "scenario_id": scenario,
                "annualized_return": value,
            })
    result = _waterfall(pd.DataFrame(rows), cfg)

    increments = result.groupby("path_id")[
        "ordered_incremental_annualized_return"
    ].sum()
    assert increments.round(12).eq(0.10).all()
    assert result.groupby("path_id").tail(1)[
        "cumulative_fraction_of_upper_bound_gap"
    ].round(12).eq(1.0).all()
    assert result["increments_are_order_dependent"].all()
