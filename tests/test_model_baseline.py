import numpy as np
import pandas as pd
import pytest

from quant_research.model_baseline import (
    SufficientStatistics,
    cross_sectional_design,
    fit_ridge,
    purged_end_date,
)


def test_cross_sectional_scaling_does_not_use_other_dates() -> None:
    first = pd.DataFrame({"alpha__z": [-1.0, 0.0, 1.0], "alpha__missing": [0, 0, 1]})
    design_before = cross_sectional_design(first, ["alpha"], True)
    unrelated_future = pd.DataFrame({"alpha__z": [100.0] * 3, "alpha__missing": [1, 1, 1]})
    _ = cross_sectional_design(unrelated_future, ["alpha"], True)
    np.testing.assert_allclose(design_before, cross_sectional_design(first, ["alpha"], True))


def test_purge_removes_label_horizon_from_period_end() -> None:
    calendar = pd.DataFrame({
        "cal_date": pd.date_range("2020-12-21", periods=9, freq="B").strftime("%Y-%m-%d"),
        "is_open": ["1"] * 9,
    })
    assert purged_end_date(calendar, "2020-12-31", 5) == "2020-12-24"


def test_ridge_recovers_positive_signal() -> None:
    x = np.arange(-50, 50, dtype="float64").reshape(-1, 1)
    y = 2 * x[:, 0]
    stats = SufficientStatistics(x.T @ x, x.T @ y, len(x), 1)
    assert fit_ridge(stats, 0.0)[0] == pytest.approx(2.0)
