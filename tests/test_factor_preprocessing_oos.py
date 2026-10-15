import numpy as np

from quant_research.factor_preprocessing_oos import expanding_fit_statistics
from quant_research.model_baseline import SufficientStatistics


def _stats(value: float, observations: int = 10) -> SufficientStatistics:
    return SufficientStatistics(
        np.array([[value]], dtype="float64"),
        np.array([value], dtype="float64"),
        observations,
        1,
    )


def test_expanding_fit_uses_only_prior_years_and_purges_boundary_year() -> None:
    full = {2018: _stats(1), 2019: _stats(2), 2020: _stats(100)}
    purged = {2018: _stats(1), 2019: _stats(2), 2020: _stats(3, observations=8)}
    result = expanding_fit_statistics(full, purged, 2018, 2021)
    assert result.gram[0, 0] == 1 + 2 + 3
    assert result.observations == 28

