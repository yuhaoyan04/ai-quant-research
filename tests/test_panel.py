import pandas as pd

from quant_research.panel import build_labels, instrument_type


def test_instrument_classification_excludes_sse_index() -> None:
    assert instrument_type("sh.000001") == "index"
    assert instrument_type("sh.600000") == "a_share"
    assert instrument_type("sz.300750") == "a_share"


def test_label_starts_after_signal_date() -> None:
    data = pd.DataFrame({
        "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "instrument_id": ["sh.600000"] * 4,
        "pct_chg": ["99", "10", "20", "30"],
        "trade_status": [1] * 4,
        "is_st": [0] * 4,
        "quality_status": ["pass"] * 4,
    })
    labels = build_labels(data, [2])
    # t=Jan 2 uses Jan 3 and Jan 4: (1.10 * 1.20) - 1, not its 99% return.
    assert round(labels.loc[0, "fwd_return_2d"], 6) == 0.32
    assert pd.isna(labels.loc[2, "fwd_return_2d"])


def test_executable_label_waits_until_after_next_day_entry() -> None:
    data = pd.DataFrame({
        "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "instrument_id": ["sh.600000"] * 4,
        "pct_chg": ["99", "10", "20", "30"],
        "trade_status": [1] * 4,
        "is_st": [0] * 4,
        "quality_status": ["pass"] * 4,
    })
    labels = build_labels(data, [2], execution_lag=1)
    # t=Jan 2 excludes both its own return and Jan 3's pre-entry return.
    assert round(labels.loc[0, "fwd_return_2d_lag1"], 6) == 0.56
    assert pd.isna(labels.loc[1, "fwd_return_2d_lag1"])
