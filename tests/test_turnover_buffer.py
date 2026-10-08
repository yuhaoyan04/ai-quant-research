import pandas as pd
import pytest

from quant_research.turnover_buffer import simulate_buffered_long_only


def _predictions() -> pd.DataFrame:
    rows = []
    scores = {
        "2020-01-03": {"A": 4, "B": 3, "C": 2, "D": 1},
        "2020-01-10": {"C": 4, "D": 3, "A": 2, "B": 1},
    }
    for date, cross_section in scores.items():
        for code, score in cross_section.items():
            rows.append({
                "asof_date": date, "available_date": date, "instrument_id": code,
                "score": score, "label": 0.01, "amihud_percentile": 0.5,
                "turnover_percentile": 0.5,
            })
    return pd.DataFrame(rows)


def test_buffer_retains_names_inside_exit_threshold():
    weekly, holdings = simulate_buffered_long_only(
        _predictions(), top_n=2, exit_n=4, target="label", round_trip_cost_bps=0,
        minimum_realized_weight_fraction=1.0)
    second = holdings.loc[holdings["asof_date"].eq("2020-01-10")]
    assert set(second["instrument_id"]) == {"A", "B"}
    assert weekly.iloc[1]["retention_rate"] == pytest.approx(1.0)
    assert weekly.iloc[1]["additions"] == 0


def test_no_buffer_replaces_names_outside_top_n():
    weekly, holdings = simulate_buffered_long_only(
        _predictions(), top_n=2, exit_n=2, target="label", round_trip_cost_bps=0,
        minimum_realized_weight_fraction=1.0)
    second = holdings.loc[holdings["asof_date"].eq("2020-01-10")]
    assert set(second["instrument_id"]) == {"C", "D"}
    assert weekly.iloc[1]["retention_rate"] == pytest.approx(0.0)
    assert weekly.iloc[1]["additions"] == 2


def test_exit_threshold_cannot_be_smaller_than_entry_threshold():
    with pytest.raises(ValueError):
        simulate_buffered_long_only(
            _predictions(), top_n=3, exit_n=2, target="label", round_trip_cost_bps=0,
            minimum_realized_weight_fraction=1.0)
