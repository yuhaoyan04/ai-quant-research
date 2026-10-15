import numpy as np
import pandas as pd

from quant_research.risk_exposure import float_cap_proxy


def test_float_cap_proxy_converts_percent_turnover_to_fraction() -> None:
    result = float_cap_proxy(pd.Series([1_000_000.0]), pd.Series([2.0]))
    assert result.iloc[0] == 50_000_000.0


def test_float_cap_proxy_does_not_fabricate_invalid_values() -> None:
    result = float_cap_proxy(
        pd.Series([1.0, 0.0, -1.0, 1.0]),
        pd.Series([0.0, 1.0, 1.0, np.nan]),
    )
    assert result.isna().all()
