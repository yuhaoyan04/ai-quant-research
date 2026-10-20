from __future__ import annotations

import pandas as pd
import pytest

from quant_research.transaction_cost_analysis import reprice_daily_path
from quant_research.transaction_costs import price_fills


def test_repricing_uses_fill_sum_over_previous_nav_without_fee_feedback() -> None:
    daily = pd.DataFrame([{
        "model": "m", "alpha_pipeline": "raw", "strategy": "s",
        "source_execution_scenario": "constrained", "trade_date": "2024-01-02",
        "gross_return": 0.10, "previous_nav": 1.0, "pretrade_nav": 1.1,
        "commission_cost": 0.0, "stamp_duty_cost": 0.0,
        "transfer_fee_cost": 0.0, "spread_cost": 0.0,
        "market_impact_cost": 0.0, "fixed_round_trip_cost": 0.0022,
        "total_cost": 0.0022,
    }])
    fills = pd.DataFrame([{
        "model": "m", "alpha_pipeline": "raw", "strategy": "s",
        "source_execution_scenario": "constrained", "fill_id": "f",
        "trade_date": "2024-01-02", "instrument_id": "sh.600000",
        "side": "buy", "notional": 1.1,
    }])
    scenario = {
        "scenario_id": "c0", "scenario_group": "C0", "mode": "fixed_round_trip",
        "round_trip_cost_bps": 20.0, "evidence_status": "test",
    }
    priced = price_fills(
        fills, scenario,
        batch_columns=["model", "alpha_pipeline", "strategy", "source_execution_scenario"],
    )

    result = reprice_daily_path(daily, priced, scenario).iloc[0]

    assert result["transaction_cost"] == pytest.approx(0.0022)
    assert result["net_return"] == pytest.approx(0.0978)
    assert result["nav"] == pytest.approx(1.0978)
    assert result["c0_reference_total_cost"] == pytest.approx(0.0022)
    assert result["cash_path_is_c0_reference"]
