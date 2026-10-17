import numpy as np
import pandas as pd

from quant_research.risk_aware_portfolio_optimization import (
    _industry_limits,
    _linear_oracle,
    _score_utility,
    active_variance,
    load_config,
    solve_risk_aware_weights,
)


def _problem():
    count = 240
    industry = np.repeat(["A", "B", "C"], count // 3)
    size = np.tile(np.linspace(-1.5, 1.5, count // 3), 3)
    score = np.linspace(1.0, 0.0, count)
    candidate = pd.DataFrame({
        "instrument_id": [f"S{index:04d}" for index in range(count)],
        "industry_broad_code": industry,
        "size_z": size,
        "score": score,
    })
    factor_loadings = np.column_stack([
        np.ones(count), industry == "A", industry == "B", industry == "C", size,
    ]).astype("float64")
    benchmark = np.array([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    covariance = np.diag([0.002, 0.001, 0.001, 0.001, 0.0015])
    specific = np.linspace(0.001, 0.003, count)
    return candidate, factor_loadings, benchmark, covariance, specific


def test_risk_aware_solver_preserves_alpha_and_reduces_active_variance():
    cfg = load_config()
    candidate, loadings, benchmark, covariance, specific = _problem()
    benchmark_industry = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    weights, diagnostics = solve_risk_aware_weights(
        candidate, {}, benchmark_industry, loadings, benchmark,
        covariance, specific, 1 / 4000, cfg,
    )
    alpha = _score_utility(candidate["score"])
    limits = _industry_limits(
        benchmark_industry, cfg["optimizer"]["industry_active_weight_bound"]
    )
    alpha_max, _, _ = _linear_oracle(
        candidate, -alpha, {}, limits, 1.0, cfg
    )

    assert diagnostics["optimizer_success"]
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= cfg["optimizer"]["maximum_stock_weight"] + 1e-10
    assert abs(weights @ candidate["size_z"]) <= 0.10 + 1e-9
    allocation = pd.Series(weights).groupby(candidate["industry_broad_code"]).sum()
    assert max(abs(allocation[item] - 1 / 3) for item in allocation.index) <= 0.03 + 1e-9
    assert alpha @ weights >= 0.98 * (alpha @ alpha_max) - 1e-9
    assert active_variance(
        weights, loadings, benchmark, covariance, specific, 1 / 4000
    ) <= active_variance(
        alpha_max, loadings, benchmark, covariance, specific, 1 / 4000
    ) + 1e-12


def test_risk_aware_solver_honors_turnover_when_feasible():
    cfg = load_config()
    candidate, loadings, benchmark, covariance, specific = _problem()
    benchmark_industry = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    first, _ = solve_risk_aware_weights(
        candidate, {}, benchmark_industry, loadings, benchmark,
        covariance, specific, 1 / 4000, cfg,
    )
    previous = {
        code: float(weight) for code, weight in zip(candidate.instrument_id, first)
        if weight > 0
    }
    changed = candidate.copy()
    changed["score"] = changed["score"].to_numpy()[::-1]
    second, diagnostics = solve_risk_aware_weights(
        changed, previous, benchmark_industry, loadings, benchmark,
        covariance, specific, 1 / 4000, cfg,
    )

    assert diagnostics["effective_turnover_limit"] == 0.30
    assert 0.5 * np.abs(second - first).sum() <= 0.30 + 1e-8
