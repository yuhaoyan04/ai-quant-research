import pandas as pd
import pytest

from quant_research.daily_execution import simulate_daily_retry, simulate_daily_retry_with_fills


def market(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "trade_date", "instrument_id", "daily_return", "buy_allowed", "sell_allowed"
    ])


def test_blocked_buy_is_retried_and_earns_returns_only_after_fill() -> None:
    data = market([
        ("2024-01-02", "A", 0.10, False, True),
        ("2024-01-03", "A", 0.20, True, True),
        ("2024-01-04", "A", 0.30, True, True),
    ])
    targets = pd.DataFrame([
        {"execution_date": "2024-01-02", "instrument_id": "A", "target_weight": 1.0}
    ])
    result = simulate_daily_retry(data, targets, 0)
    assert result["net_return"].tolist() == pytest.approx([0.0, 0.0, 0.30])
    assert result["blocked_buy_weight"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert result.loc[1, "buy_weight"] == pytest.approx(1.0)


def test_blocked_sell_retains_old_position_until_later_close() -> None:
    data = market([
        ("2024-01-02", "A", 0.00, True, True),
        ("2024-01-02", "B", 0.00, True, True),
        ("2024-01-03", "A", 0.10, True, False),
        ("2024-01-03", "B", 0.50, True, True),
        ("2024-01-04", "A", 0.00, True, True),
        ("2024-01-04", "B", 0.20, True, True),
        ("2024-01-05", "A", 0.00, True, True),
        ("2024-01-05", "B", 0.30, True, True),
    ])
    targets = pd.DataFrame([
        {"execution_date": "2024-01-02", "instrument_id": "A", "target_weight": 1.0},
        {"execution_date": "2024-01-03", "instrument_id": "B", "target_weight": 1.0},
    ])
    result = simulate_daily_retry(data, targets, 0)
    assert result["net_return"].tolist() == pytest.approx([0.0, 0.10, 0.0, 0.30])
    assert result.loc[1, "blocked_sell_weight"] == pytest.approx(1.0)
    assert result.loc[2, "sell_weight"] == pytest.approx(1.0)
    assert result.loc[2, "buy_weight"] == pytest.approx(1.0)


def test_cost_reservation_preserves_accounting_identity() -> None:
    data = market([
        ("2024-01-02", "A", 0.00, True, True),
        ("2024-01-03", "A", 0.01, True, True),
    ])
    targets = pd.DataFrame([
        {"execution_date": "2024-01-02", "instrument_id": "A", "target_weight": 1.0}
    ])
    result = simulate_daily_retry(data, targets, 20)
    assert result["accounting_weight_sum"].tolist() == pytest.approx([1.0, 1.0])
    assert result.loc[0, "buy_weight"] == pytest.approx(1 / 1.002)
    assert result.loc[0, "transaction_cost"] > 0
    assert result.loc[1, "net_return"] == pytest.approx(0.01, rel=1e-5)


def test_natural_price_drift_does_not_trigger_daily_rebalancing() -> None:
    data = market([
        ("2024-01-02", "A", 0.00, True, True),
        ("2024-01-02", "B", 0.00, True, True),
        ("2024-01-03", "A", 0.10, True, True),
        ("2024-01-03", "B", 0.00, True, True),
        ("2024-01-04", "A", 0.00, True, True),
        ("2024-01-04", "B", 0.00, True, True),
    ])
    targets = pd.DataFrame([
        {"execution_date": "2024-01-02", "instrument_id": "A", "target_weight": 0.5},
        {"execution_date": "2024-01-02", "instrument_id": "B", "target_weight": 0.5},
    ])
    result = simulate_daily_retry(data, targets, 0)
    assert result["net_return"].tolist() == pytest.approx([0.0, 0.05, 0.0])
    assert result["turnover"].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_fill_ledger_and_standard_turnover_fields_reconcile() -> None:
    data = market([
        ("2024-01-02", "A", 0.00, True, True),
        ("2024-01-02", "B", 0.00, True, True),
        ("2024-01-03", "A", 0.00, True, True),
        ("2024-01-03", "B", 0.00, True, True),
    ])
    targets = pd.DataFrame([
        {"execution_date": "2024-01-02", "instrument_id": "A", "target_weight": 0.6},
        {"execution_date": "2024-01-02", "instrument_id": "B", "target_weight": 0.4},
    ])

    daily, fills = simulate_daily_retry_with_fills(data, targets, 20)

    first = daily.iloc[0]
    assert first["buy_notional"] == pytest.approx(fills["notional"].sum())
    assert first["sell_notional"] == 0.0
    assert first["gross_two_way_turnover"] == pytest.approx(first["buy_turnover"])
    assert first["one_way_turnover"] == pytest.approx(first["turnover"])
    assert first["transaction_cost"] == pytest.approx(
        fills["total_cost"].sum() / first["previous_nav"]
    )
    assert fills["total_cost"].sum() == pytest.approx(
        fills["fixed_round_trip_cost"].sum()
    )
    assert daily.loc[1, "gross_two_way_turnover"] == 0.0
