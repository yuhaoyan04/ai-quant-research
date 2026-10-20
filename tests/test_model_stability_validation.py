import copy

import pandas as pd
import pytest

from quant_research.model_stability_validation import (
    label_interval_is_disjoint,
    label_maturity_date,
    load_config,
    summarize_rf_seeds,
    validate_protocol,
    window_years,
)


def test_protocol_is_locked_against_oos_reselection() -> None:
    cfg = load_config()
    validate_protocol(cfg)
    changed = copy.deepcopy(cfg)
    changed["parameter_promotion_allowed"] = True
    with pytest.raises(ValueError, match="cannot promote"):
        validate_protocol(changed)


def test_window_years_are_expanding_or_fixed_ten_years() -> None:
    cfg = load_config()
    assert window_years(2016, "expanding", cfg) == list(range(2006, 2016))
    assert window_years(2026, "expanding", cfg) == list(range(2006, 2026))
    assert window_years(2026, "rolling_10y", cfg) == list(range(2016, 2026))


def test_label_maturity_uses_open_day_horizon() -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2020-12-29", "2020-12-30", "2020-12-31",
        "2021-01-04", "2021-01-05", "2021-01-06",
    ]))
    assert label_maturity_date("2020-12-29", 5, calendar) == "2021-01-06"


def test_purge_rejects_both_sides_of_overlapping_label_interval() -> None:
    assert label_interval_is_disjoint(
        "2020-12-20", "2020-12-28", "2021-01-01", "2021-12-31"
    )
    assert not label_interval_is_disjoint(
        "2020-12-30", "2021-01-05", "2021-01-01", "2021-12-31"
    )
    assert not label_interval_is_disjoint(
        "2021-12-31", "2022-01-07", "2021-01-01", "2022-01-07"
    )
    assert label_interval_is_disjoint(
        "2022-01-10", "2022-01-17", "2021-01-01", "2022-01-07"
    )


def test_rf_seed_summary_reports_distribution_without_selecting_seed() -> None:
    frame = pd.DataFrame({
        "axis": ["baseline", "random_state", "random_state", "max_depth"],
        "mean_rank_ic": [0.09, 0.10, 0.11, 0.50],
        "portfolio_annualized_return": [0.2, 0.3, 0.4, 9.0],
        "portfolio_sharpe_zero_rate": [1.0, 1.1, 1.2, 8.0],
        "portfolio_maximum_drawdown": [-0.3, -0.2, -0.1, -0.01],
    })
    result = summarize_rf_seeds(frame)
    assert set(result["seed_count"]) == {3}
    assert not result["best_seed_selected"].any()
    rank_ic = result.loc[result["metric"].eq("mean_rank_ic")].iloc[0]
    assert rank_ic["mean"] == pytest.approx(0.10)
    assert rank_ic["minimum"] == pytest.approx(0.09)
    assert rank_ic["maximum"] == pytest.approx(0.11)
