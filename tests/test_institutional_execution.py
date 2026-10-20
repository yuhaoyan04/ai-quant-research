from __future__ import annotations

import pandas as pd
import pytest

from quant_research.institutional_execution import _target_shares, simulate_execution


def config(**overrides) -> dict:
    base = {
        "initial_cash_cny": 100_000.0,
        "round_trip_cost_bps": 20.0,
        "max_volume_participation": 0.10,
        "max_pending_trading_days": 5,
        "order_quantity_rules": {
            "main_board_buy_lot": 100,
            "main_board_sell_lot": 100,
            "star_market_minimum_order": 200,
            "star_market_increment": 1,
            "odd_lot_liquidation_allowed": True,
        },
    }
    base.update(overrides)
    return base


def scenario(**overrides) -> dict:
    base = {
        "scenario": "test",
        "enforce_trade_flags": True,
        "enforce_volume_participation": True,
        "enforce_order_quantity_rules": True,
        "enforce_t_plus_one": True,
    }
    base.update(overrides)
    return base


def market(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "volume": 1_000_000, "trade_status": True,
        "buy_allowed": True, "sell_allowed": True,
        "at_upper_limit": False, "at_lower_limit": False,
        "share_factor": 1.0, "cash_dividend_per_share": 0.0,
        "action_source": "none", "daily_return": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def targets(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["execution_date", "instrument_id", "target_weight"])


def test_partial_fill_retries_and_expires_with_full_audit_trail() -> None:
    data = market([
        {"trade_date": date, "instrument_id": "sh.600000", "close": 10.0, "volume": 2_000}
        for date in ("2024-01-02", "2024-01-03", "2024-01-04")
    ])
    result = simulate_execution(
        data, targets([("2024-01-02", "sh.600000", 1.0)]),
        config(max_pending_trading_days=10), scenario=scenario(),
    )

    order = result.orders.iloc[0]
    assert order["filled_shares"] == pytest.approx(600.0)
    assert order["remaining_shares"] > 0
    assert order["status"] == "EXPIRED"
    assert order["terminal_reason"] == "end_of_simulation"
    assert (result.events["status"] == "PARTIALLY_FILLED").sum() == 3
    assert len(result.fills) == 3


def test_new_target_replaces_blocked_order_instead_of_silently_dropping_it() -> None:
    data = market([
        {"trade_date": "2024-01-02", "instrument_id": "sh.600000", "close": 10.0,
         "trade_status": False, "buy_allowed": False, "sell_allowed": False},
        {"trade_date": "2024-01-02", "instrument_id": "sz.000001", "close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "sh.600000", "close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "sz.000001", "close": 10.0},
    ])
    result = simulate_execution(
        data,
        targets([
            ("2024-01-02", "sh.600000", 1.0),
            ("2024-01-03", "sz.000001", 1.0),
        ]),
        config(), scenario=scenario(enforce_volume_participation=False),
    )

    old = result.orders.loc[result.orders["instrument_id"].eq("sh.600000")].iloc[0]
    assert old["status"] == "REPLACED"
    assert old["terminal_reason"] == "new_target"
    assert result.events.loc[result.events["order_id"].eq(old["order_id"]), "status"].tolist() == [
        "NEW", "BLOCKED", "REPLACED"
    ]


def test_suspension_upper_limit_and_lower_limit_have_distinct_reasons() -> None:
    data = market([
        {"trade_date": "2024-01-02", "instrument_id": "A", "close": 10.0,
         "trade_status": False, "buy_allowed": False, "sell_allowed": False},
        {"trade_date": "2024-01-03", "instrument_id": "A", "close": 10.0,
         "buy_allowed": False, "at_upper_limit": True},
    ])
    result = simulate_execution(
        data, targets([("2024-01-02", "A", 1.0)]), config(), scenario=scenario(),
    )
    reasons = result.events.loc[result.events["status"].eq("BLOCKED"), "reason"].tolist()
    assert reasons == ["suspended", "upper_limit"]

    sell_market = market([
        {"trade_date": "2024-01-02", "instrument_id": "A", "close": 10.0},
        {"trade_date": "2024-01-02", "instrument_id": "B", "close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "A", "close": 10.0,
         "sell_allowed": False, "at_lower_limit": True},
        {"trade_date": "2024-01-03", "instrument_id": "B", "close": 10.0},
    ])
    sold = simulate_execution(
        sell_market,
        targets([("2024-01-02", "A", 1.0), ("2024-01-03", "B", 1.0)]),
        config(), scenario=scenario(enforce_volume_participation=False),
    )
    assert "lower_limit" in sold.events.loc[sold.events["status"].eq("BLOCKED"), "reason"].tolist()


def test_t_plus_one_inventory_settles_next_trading_day_and_cash_never_negative() -> None:
    data = market([
        {"trade_date": "2024-01-02", "instrument_id": "A", "close": 10.0},
        {"trade_date": "2024-01-02", "instrument_id": "B", "close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "A", "close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "B", "close": 10.0},
    ])
    result = simulate_execution(
        data,
        targets([("2024-01-02", "A", 1.0), ("2024-01-03", "B", 1.0)]),
        config(), scenario=scenario(enforce_volume_participation=False),
    )

    first_a = result.positions.loc[
        result.positions["trade_date"].eq("2024-01-02")
        & result.positions["instrument_id"].eq("A")
    ].iloc[0]
    assert first_a["available_shares"] == 0.0
    assert first_a["unsettled_shares"] > 0.0
    assert (result.fills["side"] == "sell").any()
    sells = result.fills.loc[result.fills["side"].eq("sell")]
    assert (sells["executed_shares"] <= sells["available_before_fill"]).all()
    assert result.daily["cash"].min() >= 0.0
    assert result.daily["accounting_weight_sum"].tolist() == pytest.approx([1.0, 1.0])


def test_corporate_action_factor_preserves_value_without_phantom_pnl() -> None:
    data = market([
        {"trade_date": "2024-01-02", "instrument_id": "A", "close": 10.0,
         "previous_close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "A", "close": 5.0,
         "previous_close": 10.0, "share_factor": 2.0,
         "action_source": "test_split"},
    ])
    result = simulate_execution(
        data, targets([("2024-01-02", "A", 1.0)]),
        config(round_trip_cost_bps=0.0),
        scenario=scenario(enforce_volume_participation=False),
    )

    assert result.daily["net_return"].tolist() == pytest.approx([0.0, 0.0])
    assert result.corporate_actions.iloc[0]["value_reconciliation_error"] == pytest.approx(0.0)
    second = result.positions.loc[result.positions["trade_date"].eq("2024-01-03")].iloc[0]
    assert second["shares"] == pytest.approx(2 * result.positions.iloc[0]["shares"])


def test_star_market_quantity_rule_has_200_share_minimum_and_one_share_increment() -> None:
    rules = config()["order_quantity_rules"]
    assert _target_shares(1_990.0, 10.0, "sh.688001", rules, True) == 0.0
    assert _target_shares(2_010.0, 10.0, "sh.688001", rules, True) == 201.0
    assert _target_shares(2_990.0, 10.0, "sh.600000", rules, True) == 200.0
