from __future__ import annotations

import pandas as pd
import pytest

from quant_research.transaction_costs import COST_COLUMNS, price_fills


def _fills() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "portfolio": "p", "fill_id": "f1", "trade_date": "2023-08-25",
            "instrument_id": "sh.600000", "side": "buy", "notional": 60.0,
        },
        {
            "portfolio": "p", "fill_id": "f2", "trade_date": "2023-08-25",
            "instrument_id": "sz.000001", "side": "sell", "notional": 100.0,
        },
    ])


def test_fixed_round_trip_cost_is_applied_once_to_one_way_notional() -> None:
    scenario = {
        "scenario_id": "fixed20", "mode": "fixed_round_trip",
        "round_trip_cost_bps": 20.0, "evidence_status": "test",
    }

    result = price_fills(_fills(), scenario, batch_columns=["portfolio"])

    assert result["total_cost"].sum() == pytest.approx(100.0 * 20.0 / 10_000.0)
    assert result["fixed_round_trip_cost"].sum() == pytest.approx(0.2)
    assert result[[column for column in COST_COLUMNS if column not in {
        "fixed_round_trip_cost", "total_cost",
    }]].to_numpy().sum() == 0.0


def test_zero_fixed_cost_is_strictly_zero() -> None:
    scenario = {
        "scenario_id": "fixed0", "mode": "fixed_round_trip",
        "round_trip_cost_bps": 0.0, "evidence_status": "test",
    }

    result = price_fills(_fills(), scenario, batch_columns=["portfolio"])

    assert result[list(COST_COLUMNS)].to_numpy().sum() == 0.0


def test_component_costs_are_side_and_effective_date_aware() -> None:
    fills = pd.concat([
        _fills(),
        pd.DataFrame([{
            "portfolio": "p", "fill_id": "f3", "trade_date": "2023-08-28",
            "instrument_id": "sh.600000", "side": "sell", "notional": 100.0,
        }]),
    ], ignore_index=True)
    scenario = {
        "scenario_id": "components", "mode": "components",
        "evidence_status": "test",
        "components": {
            "commission": {"rules": [{
                "start_date": "2021-01-01", "buy_bps": 3.0, "sell_bps": 3.0,
            }]},
            "stamp_duty": {"rules": [
                {"start_date": "2021-01-01", "end_date": "2023-08-27",
                 "buy_bps": 0.0, "sell_bps": 10.0},
                {"start_date": "2023-08-28", "buy_bps": 0.0, "sell_bps": 5.0},
            ]},
            "transfer_fee": {"rules": [{
                "start_date": "2021-01-01", "buy_bps": 0.1, "sell_bps": 0.1,
            }]},
            "spread": {"rules": [{
                "start_date": "2021-01-01", "buy_bps": 0.0, "sell_bps": 0.0,
            }]},
            "market_impact": {"rules": [{
                "start_date": "2021-01-01", "buy_bps": 0.0, "sell_bps": 0.0,
            }]},
        },
    }

    result = price_fills(fills, scenario)
    before = result.loc[result["fill_id"].eq("f2")].iloc[0]
    after = result.loc[result["fill_id"].eq("f3")].iloc[0]
    buy = result.loc[result["fill_id"].eq("f1")].iloc[0]

    assert before["stamp_duty_cost"] == pytest.approx(100 * 10 / 10_000)
    assert after["stamp_duty_cost"] == pytest.approx(100 * 5 / 10_000)
    assert buy["stamp_duty_cost"] == 0.0
    assert before["commission_cost"] == pytest.approx(100 * 3 / 10_000)
    assert before["transfer_fee_cost"] == pytest.approx(100 * 0.1 / 10_000)
    assert result["total_cost"].sum() == pytest.approx(
        result[[
            "commission_cost", "stamp_duty_cost", "transfer_fee_cost",
            "spread_cost", "market_impact_cost", "fixed_round_trip_cost",
        ]].to_numpy().sum()
    )


def test_minimum_commission_and_invalid_double_counting() -> None:
    scenario = {
        "scenario_id": "minimum", "mode": "components", "evidence_status": "test",
        "components": {
            component: {
                "minimum_fee_per_fill": 5.0 if component == "commission" else 0.0,
                "rules": [{"start_date": "2021-01-01", "buy_bps": 1.0, "sell_bps": 1.0}],
            }
            for component in ("commission", "stamp_duty", "transfer_fee", "spread", "market_impact")
        },
    }
    result = price_fills(_fills(), scenario)
    assert result["commission_cost"].tolist() == [5.0, 5.0]

    bad = {
        "scenario_id": "bad", "mode": "fixed_round_trip",
        "round_trip_cost_bps": 20.0, "evidence_status": "test",
        "components": {"commission": {}},
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        price_fills(_fills(), bad, batch_columns=["portfolio"])


def test_overlapping_component_rules_are_rejected() -> None:
    scenario = {
        "scenario_id": "overlap", "mode": "components", "evidence_status": "test",
        "components": {
            component: {"rules": [
                {"start_date": "2021-01-01", "buy_bps": 1.0, "sell_bps": 1.0},
                {"start_date": "2022-01-01", "buy_bps": 1.0, "sell_bps": 1.0},
            ]}
            for component in ("commission", "stamp_duty", "transfer_fee", "spread", "market_impact")
        },
    }
    with pytest.raises(ValueError, match="overlapping fee rules"):
        price_fills(_fills(), scenario)
