import numpy as np
import pandas as pd

from quant_research.ic_decay import incremental_return, newey_west_mean_t


def test_incremental_return_recovers_tail_compound_return() -> None:
    first = pd.Series([0.10])
    tail = 0.20
    cumulative = pd.Series([(1 + first.iloc[0]) * (1 + tail) - 1])
    result = incremental_return(cumulative, first)
    assert np.isclose(result.iloc[0], tail)


def test_incremental_return_preserves_missingness() -> None:
    result = incremental_return(pd.Series([0.2, np.nan]), pd.Series([np.nan, 0.1]))
    assert result.isna().all()


def test_newey_west_constant_positive_series_has_undefined_zero_variance_t() -> None:
    standard_error, t_stat = newey_west_mean_t(pd.Series([0.1] * 20), 4)
    assert np.isclose(standard_error, 0.0)
    assert np.isnan(t_stat)

