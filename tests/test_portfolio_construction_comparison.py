import numpy as np
import pandas as pd

from quant_research.portfolio_construction_comparison import (
    _capped_normalize,
    carry_forward_fallback,
    common_buffer_candidates,
    construct_targets,
    load_config,
    optimizer_candidates,
    simple_weights,
    solve_constrained_weights,
)


def test_signal_input_columns_are_stably_unique() -> None:
    factors = ["momentum_20d", "volatility_20d"]
    columns = list(dict.fromkeys([
        "instrument_id", "asof_date", "available_date", "target",
        "volatility_20d", *factors,
    ]))
    assert columns.count("volatility_20d") == 1


def _frame(count: int = 240, date: str = "2024-01-05") -> pd.DataFrame:
    codes = [f"sh.{600000 + i:06d}" for i in range(count)]
    return pd.DataFrame({
        "instrument_id": codes,
        "asof_date": date,
        "available_date": "2024-01-08",
        "score": np.linspace(1.0, 0.0, count),
        "volatility_20d": np.linspace(0.01, 0.04, count),
        "industry_broad_code": np.where(np.arange(count) % 2, "A", "B"),
        "size_z": np.linspace(-1.0, 1.0, count),
        "fwd_return_5d_lag1": 0.0,
    })


def test_buffer_retains_old_top400_and_adds_new_top200() -> None:
    frame = _frame(450)
    previous = {frame.iloc[250]["instrument_id"], frame.iloc[449]["instrument_id"]}
    selected, audit = common_buffer_candidates(frame, previous, 200, 400)
    codes = set(selected["instrument_id"])
    assert frame.iloc[250]["instrument_id"] in codes
    assert frame.iloc[449]["instrument_id"] not in codes
    assert frame.iloc[0]["instrument_id"] in codes
    assert audit["buffer_retained_count"] == 1


def test_inverse_volatility_can_lower_retained_weight() -> None:
    cfg = load_config()
    candidate = _frame(201)
    candidate.loc[200, "volatility_20d"] = 0.20
    equal = simple_weights(candidate, "P0_equal", cfg)
    inverse = simple_weights(candidate, "P1_inverse_volatility", cfg)
    assert inverse[200] < equal[200]
    assert np.isclose(inverse.sum(), 1.0)
    assert inverse.max() <= cfg["weighting"]["maximum_stock_weight"] + 1e-12


def test_capped_normalize_rejects_infeasible_cap() -> None:
    try:
        _capped_normalize(np.ones(99), 0.01)
    except ValueError as exc:
        assert "infeasible" in str(exc)
    else:
        raise AssertionError("infeasible cap should fail")


def test_carry_fallback_preserves_old_weight_and_repairs_exit() -> None:
    cfg = load_config()
    candidate = _frame(200)
    previous = {
        candidate.iloc[0]["instrument_id"]: 0.01,
        "sh.999999": 0.99,
    }
    weights = carry_forward_fallback(candidate, previous, cfg)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] == cfg["weighting"]["maximum_stock_weight"]
    assert weights.max() <= cfg["weighting"]["maximum_stock_weight"] + 1e-12


def test_too_tight_turnover_is_explicitly_infeasible() -> None:
    cfg = load_config()
    cfg["optimizer"]["one_way_turnover_limit"] = 0.01
    universe = _frame(240)
    candidate = universe.iloc[:200].copy()
    previous = {"sh.999999": 1.0}
    weights, audit = solve_constrained_weights(universe, candidate, previous, cfg)
    assert weights is None
    assert audit["optimizer_success"] is False
    assert audit["solver_status"] == "turnover_limit_below_forced_exit"
    assert audit["constraint_relaxed"] is False


def test_industry_candidate_conflict_is_not_silently_relaxed() -> None:
    cfg = load_config()
    universe = _frame(240)
    universe.loc[:119, "industry_broad_code"] = "A"
    universe.loc[120:, "industry_broad_code"] = "B"
    candidate = universe.iloc[:100].copy()
    cfg["weighting"]["maximum_stock_weight"] = 0.02
    weights, audit = solve_constrained_weights(universe, candidate, {}, cfg)
    assert weights is None
    assert audit["optimizer_success"] is False
    assert audit["constraint_relaxed"] is False


def test_optimizer_support_is_explicit_and_preserves_base_candidates() -> None:
    cfg = load_config()
    cfg["buffer"]["entry_rank"] = 100
    cfg["buffer"]["exit_rank"] = 150
    universe = _frame(240)
    universe.loc[:119, "industry_broad_code"] = "A"
    universe.loc[120:, "industry_broad_code"] = "B"
    base, _ = common_buffer_candidates(universe, set(), 100, 150)
    support = optimizer_candidates(universe, base, {}, cfg)
    assert set(base["instrument_id"]).issubset(set(support["instrument_id"]))
    assert support["constraint_support"].any()
    assert support.loc[support["constraint_support"], "entry_eligible"].eq(False).all()


def test_construct_targets_records_constraint_support_without_relaxation() -> None:
    cfg = load_config()
    cfg["buffer"]["entry_rank"] = 100
    cfg["buffer"]["exit_rank"] = 150
    cfg["weighting"]["maximum_stock_weight"] = 0.02
    frame = _frame(240)
    frame.loc[:119, "industry_broad_code"] = "A"
    frame.loc[120:, "industry_broad_code"] = "B"
    targets, audit = construct_targets(frame, cfg)
    p4 = audit.loc[audit["method"].eq("P4_constrained")].iloc[0]
    assert not bool(p4["fallback_used"])
    assert bool(p4["optimizer_success"])
    assert p4["constraint_support_count"] > 0
    assert not bool(p4["constraint_relaxed"])
    sums = targets.groupby(["execution_date", "method"])["target_weight"].sum()
    assert np.allclose(sums, 1.0)
