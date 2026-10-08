import pandas as pd

from quant_research.factor_research import factor_statistics, split_factor_statistics


def test_factor_statistics_detects_monotonic_signal() -> None:
    rows = []
    for day in range(5):
        for rank in range(1, 121):
            rows.append({"asof_date": f"2024-01-{day + 1:02d}", "signal": rank, "target": rank / 10000})
    result = factor_statistics(pd.DataFrame(rows), ["signal"], "target", 100, 10)
    assert result.loc[0, "observations"] == 5
    assert result.loc[0, "mean_rank_ic"] > 0.99
    assert result.loc[0, "mean_top_minus_bottom"] > 0


def test_time_splits_do_not_mix_dates() -> None:
    panel = pd.DataFrame([
        {"asof_date": day, "signal": rank, "target": rank / 10000}
        for day in ("2019-12-31", "2020-01-02") for rank in range(1, 101)
    ])
    result = split_factor_statistics(panel, ["signal"], "target", 100, 2, [
        {"name": "early", "start": "2019-01-01", "end": "2019-12-31"},
        {"name": "late", "start": "2020-01-01", "end": "2020-12-31"},
    ])
    assert result["sample_split"].tolist() == ["early", "late"]
    assert result["observations"].tolist() == [1, 1]
