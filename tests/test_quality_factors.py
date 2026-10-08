import pandas as pd

from quant_research.factors import build_raw_factors, next_trading_day_map
from quant_research.quality import validate_daily_bars


def raw_bars() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=25, freq="B").strftime("%Y-%m-%d"),
        "instrument_id": ["sh.600000"] * 25,
        "open": [10.0] * 25, "high": [10.2] * 25, "low": [9.8] * 25,
        "close": [10.0] * 25, "pre_close": [10.0] * 25,
        "volume": [1000] * 25, "amount": [10000] * 25,
        "trade_status": ["1"] * 25, "is_st": ["0"] * 25,
        "pct_chg": ["0.1"] * 25, "turn": ["0.2"] * 25,
        "pe_ttm": ["20"] * 25, "pb_mrq": ["2"] * 25,
    })


def test_invalid_ohlc_is_failed_but_retained() -> None:
    raw = raw_bars()
    raw.loc[0, "high"] = 9.0
    checked = validate_daily_bars(raw).frame
    assert len(checked) == 25
    assert checked.loc[0, "quality_status"] == "fail"
    assert "invalid_ohlc" in checked.loc[0, "quality_flags"]


def test_factors_are_available_only_next_trading_day() -> None:
    checked = validate_daily_bars(raw_bars()).frame
    calendar = pd.DataFrame({"cal_date": pd.date_range("2024-01-01", periods=30, freq="B").strftime("%Y-%m-%d"), "is_open": ["1"] * 30})
    factors = build_raw_factors(checked, next_trading_day_map(calendar))
    row = factors.loc[factors["momentum_20d"].notna()].iloc[0]
    assert row["available_date"] > row["asof_date"]
    assert row["is_tradable"]
