import numpy as np
import pandas as pd

from quant_research.leave_one_factor_out import (
    benjamini_hochberg,
    model_feature_indices,
)


def test_leave_one_factor_out_removes_exposure_and_missing_indicator() -> None:
    result = model_feature_indices(["a", "b", "c"])
    assert result["full"] == [0, 1, 2, 3, 4, 5]
    assert result["without_b"] == [0, 2, 3, 5]


def test_benjamini_hochberg_is_monotone_in_p_value_order() -> None:
    values = pd.Series([0.04, 0.001, 0.02, np.nan])
    adjusted = benjamini_hochberg(values)
    ordered = pd.DataFrame({"p": values, "q": adjusted}).dropna().sort_values("p")
    assert ordered["q"].is_monotonic_increasing
    assert adjusted.iloc[1] <= adjusted.iloc[2] <= adjusted.iloc[0]

