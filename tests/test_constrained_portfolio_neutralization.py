import numpy as np
import pandas as pd

from quant_research.constrained_portfolio_neutralization import (
    load_config,
    portfolio_risk,
    solve_risk_constrained_weights,
    top_equal_weights,
)


def _universe() -> pd.DataFrame:
    count = 900
    return pd.DataFrame({
        "instrument_id": [f"S{index:04d}" for index in range(count)],
        "industry_broad_code": np.repeat(["A", "B", "C"], count // 3),
        "size_z": np.tile(np.linspace(-2.0, 2.0, count // 3), 3),
    })


def test_linear_program_satisfies_prespecified_risk_bounds() -> None:
    cfg = load_config()
    group = _universe()
    score = np.random.default_rng(7).normal(size=len(group))

    weights, diagnostics = solve_risk_constrained_weights(
        group, score, {}, cfg
    )
    risk, allocation = portfolio_risk(group, weights, cfg)

    assert diagnostics["optimizer_success"]
    assert np.isclose(weights.sum(), 1.0)
    assert risk["maximum_stock_weight"] <= 0.005 + 1e-10
    assert risk["abs_portfolio_size_z"] <= 0.10 + 1e-10
    assert allocation["active_weight"].abs().max() <= 0.03 + 1e-10


def test_linear_program_honors_turnover_cap_when_feasible() -> None:
    cfg = load_config()
    group = _universe()
    first_score = np.linspace(1.0, 0.0, len(group))
    first, _ = solve_risk_constrained_weights(group, first_score, {}, cfg)
    previous = dict(zip(group["instrument_id"], first))
    previous = {code: weight for code, weight in previous.items() if weight > 0}

    second, diagnostics = solve_risk_constrained_weights(
        group, -first_score, previous, cfg
    )
    turnover = 0.5 * np.abs(second - first).sum()

    assert diagnostics["effective_turnover_limit"] == 0.30
    assert not diagnostics["turnover_limit_relaxed"]
    assert turnover <= 0.30 + 1e-9


def test_top_equal_weights_select_exactly_two_hundred_names() -> None:
    group = _universe()
    score = np.arange(len(group), dtype="float64")
    weights = top_equal_weights(group, score, 200)
    assert np.count_nonzero(weights) == 200
    np.testing.assert_allclose(weights[weights > 0], 0.005)
    assert np.isclose(weights.sum(), 1.0)
