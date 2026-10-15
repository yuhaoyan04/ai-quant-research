import numpy as np
import pandas as pd

from quant_research.model_benchmark import _naive, _ols, select_family_candidates
from quant_research.model_baseline import SufficientStatistics


def test_ols_solves_sufficient_statistics() -> None:
    statistics = SufficientStatistics(
        gram=np.array([[2.0, 0.0], [0.0, 4.0]]),
        cross=np.array([4.0, 8.0]), observations=10, dates=2,
    )
    assert np.allclose(_ols(statistics), [2.0, 2.0])


def test_naive_uses_equal_directional_factor_weights_only() -> None:
    statistics = SufficientStatistics(
        gram=np.eye(4), cross=np.array([2.0, -1.0, 5.0, -5.0]),
        observations=10, dates=2,
    )
    beta = _naive(statistics, factor_count=2)
    assert np.allclose(beta[:2], [1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert np.allclose(beta[2:], 0.0)


def test_family_selection_never_uses_other_family_score() -> None:
    grid = pd.DataFrame([
        {"candidate": "r1", "family": "ridge", "mean_rank_ic": 0.1,
         "portfolio_sharpe_zero_rate": 0.5, "portfolio_annualized_return": 0.1},
        {"candidate": "r2", "family": "ridge", "mean_rank_ic": 0.09,
         "portfolio_sharpe_zero_rate": 9.0, "portfolio_annualized_return": 0.9},
        {"candidate": "h1", "family": "hgb", "mean_rank_ic": 0.08,
         "portfolio_sharpe_zero_rate": 0.7, "portfolio_annualized_return": 0.1},
    ])
    selected = select_family_candidates(
        grid, "mean_rank_ic", ["portfolio_sharpe_zero_rate", "portfolio_annualized_return"]
    )
    assert set(selected["candidate"]) == {"r1", "h1"}
