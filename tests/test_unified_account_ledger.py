from __future__ import annotations

import pandas as pd
import pytest

from quant_research.institutional_execution import simulate_execution
from quant_research.unified_account_ledger import replay_transactions


def _market(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "volume": 1_000_000, "trade_status": True,
        "buy_allowed": True, "sell_allowed": True,
        "at_upper_limit": False, "at_lower_limit": False,
        "share_factor": 1.0, "cash_dividend_per_share": 0.0,
        "action_source": "none", "daily_return": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _config() -> dict:
    return {
        "initial_cash_cny": 1_000.0, "round_trip_cost_bps": 0.0,
        "max_volume_participation": 0.10, "max_pending_trading_days": 5,
        "order_quantity_rules": {
            "main_board_buy_lot": 1, "main_board_sell_lot": 1,
            "star_market_minimum_order": 1, "star_market_increment": 1,
            "odd_lot_liquidation_allowed": True,
        },
    }


def _scenario() -> dict:
    return {
        "scenario": "micro", "enforce_trade_flags": True,
        "enforce_volume_participation": False,
        "enforce_order_quantity_rules": True, "enforce_t_plus_one": True,
    }


def test_hand_calculated_cash_position_nav_and_fees() -> None:
    marks = pd.DataFrame([
        ("2024-01-02", "A", 100.0),
        ("2024-01-03", "A", 110.0),
        ("2024-01-04", "A", 110.0),
    ], columns=["trade_date", "instrument_id", "price"])
    fills = pd.DataFrame([
        ("2024-01-02", "A", "buy", 1.0, 100.0, 1.0),
        ("2024-01-04", "A", "sell", 0.5, 110.0, 0.5),
    ], columns=["trade_date", "instrument_id", "side", "shares", "price", "fee"])
    daily, positions = replay_transactions(1_000.0, marks, fills)

    assert daily[["cash", "market_value", "nav", "fees"]].to_numpy().ravel().tolist() == pytest.approx([
        899.0, 100.0, 999.0, 1.0,
        899.0, 110.0, 1009.0, 0.0,
        953.5, 55.0, 1008.5, 0.5,
    ])
    assert positions.iloc[-1]["actual_shares"] == pytest.approx(0.5)


def test_limit_blocks_replacement_and_corporate_action_are_auditable() -> None:
    market = _market([
        {"trade_date": "2024-01-02", "instrument_id": "A", "close": 10.0,
         "buy_allowed": False, "at_upper_limit": True, "previous_close": 10.0},
        {"trade_date": "2024-01-02", "instrument_id": "B", "close": 10.0,
         "previous_close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "A", "close": 10.0,
         "previous_close": 10.0},
        {"trade_date": "2024-01-03", "instrument_id": "B", "close": 10.0,
         "previous_close": 10.0},
        {"trade_date": "2024-01-04", "instrument_id": "A", "close": 10.0,
         "previous_close": 10.0},
        {"trade_date": "2024-01-04", "instrument_id": "B", "close": 5.0,
         "sell_allowed": False, "at_lower_limit": True, "previous_close": 10.0,
         "share_factor": 2.0, "action_source": "test_split"},
    ])
    targets = pd.DataFrame([
        ("2024-01-02", "A", 1.0),
        ("2024-01-03", "B", 1.0),
        ("2024-01-04", "A", 1.0),
    ], columns=["execution_date", "instrument_id", "target_weight"])
    result = simulate_execution(market, targets, _config(), scenario=_scenario())

    blocked = result.events.loc[result.events["status"].eq("BLOCKED"), "reason"].tolist()
    assert "upper_limit" in blocked
    assert "lower_limit" in blocked
    assert "REPLACED" in result.events["status"].tolist()
    assert len(result.corporate_actions) == 1
    assert result.corporate_actions.iloc[0]["shares_after"] == pytest.approx(
        2 * result.corporate_actions.iloc[0]["shares_before"]
    )
    assert result.corporate_actions.iloc[0]["value_reconciliation_error"] == pytest.approx(0.0)
