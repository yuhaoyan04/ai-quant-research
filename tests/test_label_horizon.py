import numpy as np
import pandas as pd
import pytest

from quant_research.label_horizon import select_validation_winners, target_values


def test_target_values_distinguishes_raw_and_cross_sectional_rank() -> None:
    values = pd.Series([0.03, -0.01, 0.01])
    assert np.allclose(target_values(values, "raw_return"), values.to_numpy())
    ranked = target_values(values, "cross_sectional_rank")
    assert ranked.mean() == pytest.approx(0.0)
    assert ranked.std(ddof=1) == pytest.approx(1.0)
    assert list(np.argsort(ranked)) == [1, 2, 0]


def test_validation_selection_is_two_stage() -> None:
    grid = pd.DataFrame([
        {"target_spec": "a", "ridge_penalty": 0.1, "mean_rank_ic": 0.10,
         "annualized_icir": 1.0, "portfolio_sharpe_zero_rate": 0.5,
         "portfolio_annualized_return": 0.1},
        {"target_spec": "a", "ridge_penalty": 1.0, "mean_rank_ic": 0.09,
         "annualized_icir": 2.0, "portfolio_sharpe_zero_rate": 9.0,
         "portfolio_annualized_return": 0.9},
        {"target_spec": "b", "ridge_penalty": 1.0, "mean_rank_ic": 0.08,
         "annualized_icir": 1.0, "portfolio_sharpe_zero_rate": 0.7,
         "portfolio_annualized_return": 0.1},
    ])
    candidates, winner = select_validation_winners(grid)
    assert dict(zip(candidates["target_spec"], candidates["ridge_penalty"])) == {
        "a": 0.1, "b": 1.0,
    }
    assert winner["target_spec"] == "b"
