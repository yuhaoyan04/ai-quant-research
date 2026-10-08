import pandas as pd
import pytest

from quant_research.capacity_execution import execute_toward_target, simulate_capacity_constrained


def test_execution_respects_adv_and_cash() -> None:
    weights, cash, trades = execute_toward_target(
        {}, 1.0, {"A": 0.6, "B": 0.4}, {"A": 100.0, "B": 100.0},
        nav_cny=1_000.0, participation_limit=0.1)
    assert trades == {"A": pytest.approx(0.01), "B": pytest.approx(0.01)}
    assert weights == {"A": pytest.approx(0.01), "B": pytest.approx(0.01)}
    assert cash == pytest.approx(0.98)


def test_large_adv_reaches_target_and_computes_return() -> None:
    market = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "label": 0.10,
         "adv20_cny": 1e9, "volatility_20d": 0.02},
        {"asof_date": "2024-01-05", "instrument_id": "B", "label": 0.00,
         "adv20_cny": 1e9, "volatility_20d": 0.02},
    ])
    holdings = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "target_weight": 0.5},
        {"asof_date": "2024-01-05", "instrument_id": "B", "target_weight": 0.5},
    ])
    result = simulate_capacity_constrained(
        market, holdings, "label", 1_000_000, 0.05, 0, 0, 1.0)
    assert result.loc[0, "cash_weight"] == pytest.approx(0.0)
    assert result.loc[0, "gross_return"] == pytest.approx(0.05)
    assert result.loc[0, "order_fill_ratio"] == pytest.approx(1.0)


def test_low_adv_leaves_cash_and_reports_partial_fill() -> None:
    market = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "label": 0.10,
         "adv20_cny": 10_000, "volatility_20d": 0.02},
    ])
    holdings = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "target_weight": 1.0},
    ])
    result = simulate_capacity_constrained(
        market, holdings, "label", 1_000_000, 0.01, 0, 0, 0.0)
    assert result.loc[0, "cash_weight"] == pytest.approx(0.9999)
    assert result.loc[0, "gross_return"] == pytest.approx(0.00001)
    assert result.loc[0, "order_fill_ratio"] == pytest.approx(0.0001)


def test_missing_held_return_does_not_drop_chronological_period() -> None:
    market = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "label": float("nan"),
         "adv20_cny": 1e9, "volatility_20d": 0.02},
    ])
    holdings = pd.DataFrame([
        {"asof_date": "2024-01-05", "instrument_id": "A", "target_weight": 1.0},
    ])
    result = simulate_capacity_constrained(
        market, holdings, "label", 1_000_000, 0.05, 0, 0, 0.98)
    assert not result.loc[0, "coverage_pass"]
    assert result.loc[0, "included_in_performance"]
    assert result.loc[0, "gross_return"] == 0.0
