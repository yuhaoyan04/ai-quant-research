import numpy as np
import pandas as pd

from quant_research.factor_tear_sheet import (
    _monotonicity,
    assign_buckets,
    equal_weight_turnover,
)


def test_assign_buckets_is_balanced_and_ordered() -> None:
    values = np.array([9, 1, 7, 3, 5, 0, 8, 2, 6, 4, 10], dtype="float64")
    buckets = assign_buckets(values, 5)

    assert buckets[np.argmin(values)] == 1
    assert buckets[np.argmax(values)] == 5
    counts = pd.Series(buckets).value_counts()
    assert counts.max() - counts.min() <= 1


def test_equal_weight_turnover_handles_membership_and_size_change() -> None:
    assert np.isnan(equal_weight_turnover(None, {"a", "b"}))
    assert equal_weight_turnover({"a", "b"}, {"a", "b"}) == 0
    assert equal_weight_turnover({"a", "b"}, {"c", "d"}) == 1
    assert equal_weight_turnover({"a", "b"}, {"a", "b", "c", "d"}) == 0.5


def test_monotonicity_uses_aggregate_bucket_returns() -> None:
    frame = pd.DataFrame({
        "bucket": np.tile(np.arange(1, 6), 2),
        "equal_weight_return": np.tile(np.arange(1, 6), 2) / 100,
    })

    correlation, adjacent = _monotonicity(frame, 5)

    assert np.isclose(correlation, 1)
    assert adjacent == 1
