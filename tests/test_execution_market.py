from __future__ import annotations

import pandas as pd
import pytest

from quant_research.execution_market import (
    EXECUTION_MARKET_COLUMNS,
    validated_bars_to_execution_market,
)


def rules() -> dict:
    return {
        "st_limit_pct": 5.0,
        "main_board_limit_pct": 10.0,
        "star_limit_pct": 20.0,
        "chinext_limit_pct_before_2020_08_24": 10.0,
        "chinext_limit_pct_after_2020_08_24": 20.0,
    }


def test_canonical_execution_market_flags_limits_and_action_proxy() -> None:
    bars = pd.DataFrame([
        {"instrument_id": "sh.600000", "trade_date": "2024-01-02", "close": 10.0,
         "volume": 1000, "pct_chg": 0.0, "trade_status": 1, "is_st": 0},
        {"instrument_id": "sh.600000", "trade_date": "2024-01-03", "close": 11.0,
         "volume": 1000, "pct_chg": 10.0, "trade_status": 1, "is_st": 0},
        {"instrument_id": "sh.600000", "trade_date": "2024-01-04", "close": 5.5,
         "volume": 1000, "pct_chg": 0.0, "trade_status": 0, "is_st": 0},
    ])
    result = validated_bars_to_execution_market(
        bars, instrument_id="sh.600000", rules=rules(),
        corporate_action_threshold=0.005,
        start_date="2024-01-02", end_date="2024-01-04",
    )
    assert result.columns.tolist() == EXECUTION_MARKET_COLUMNS
    assert result["at_upper_limit"].tolist() == [False, True, False]
    assert result["buy_allowed"].tolist() == [True, False, False]
    assert result["sell_allowed"].tolist() == [True, True, False]
    assert result.loc[2, "share_factor"] == pytest.approx(2.0)
    assert result.loc[2, "action_source"] == "inferred_value_preserving_factor"


def test_canonical_execution_market_is_sorted_and_date_bounded() -> None:
    bars = pd.DataFrame([
        {"instrument_id": "sz.000001", "trade_date": date, "close": 10.0,
         "volume": 1000, "pct_chg": 0.0, "trade_status": 1, "is_st": 0}
        for date in ["2024-01-04", "2024-01-02", "2024-01-03"]
    ])
    result = validated_bars_to_execution_market(
        bars, instrument_id="sz.000001", rules=rules(),
        corporate_action_threshold=0.005,
        start_date="2024-01-03", end_date="2024-01-04",
    )
    assert result["trade_date"].tolist() == ["2024-01-03", "2024-01-04"]
    assert result["daily_return"].eq(0).all()


def test_canonical_execution_market_rejects_incomplete_schema() -> None:
    with pytest.raises(KeyError, match="missing columns"):
        validated_bars_to_execution_market(
            pd.DataFrame({"trade_date": ["2024-01-02"]}),
            instrument_id="sh.600000", rules=rules(),
            corporate_action_threshold=0.005,
            start_date="2024-01-02", end_date="2024-01-02",
        )
