import pandas as pd
import pytest

from quant_research.label_panel import merge_horizon_labels


def test_label_merge_preserves_eligible_universe_and_adds_horizons() -> None:
    eligible = pd.DataFrame({
        "instrument_id": ["a", "b"], "asof_date": ["2020-01-01"] * 2,
        "fwd_return_5d_lag1": [0.1, 0.2], "factor": [1.0, 2.0],
    })
    labels = pd.DataFrame({
        "instrument_id": ["a"], "asof_date": ["2020-01-01"],
        "fwd_return_5d_lag1": [0.1], "fwd_return_10d_lag1": [0.3],
        "fwd_return_20d_lag1": [0.4],
    })
    result = merge_horizon_labels(eligible, labels)
    assert len(result) == 2
    assert result.loc[0, "fwd_return_10d_lag1"] == 0.3
    assert pd.isna(result.loc[1, "fwd_return_20d_lag1"])


def test_label_merge_rejects_duplicate_keys() -> None:
    eligible = pd.DataFrame({"instrument_id": ["a"], "asof_date": ["2020-01-01"]})
    labels = pd.DataFrame({
        "instrument_id": ["a", "a"], "asof_date": ["2020-01-01"] * 2,
        "fwd_return_5d_lag1": [0.1, 0.1], "fwd_return_10d_lag1": [0.2, 0.2],
        "fwd_return_20d_lag1": [0.3, 0.3],
    })
    with pytest.raises(ValueError, match="duplicate"):
        merge_horizon_labels(eligible, labels)

