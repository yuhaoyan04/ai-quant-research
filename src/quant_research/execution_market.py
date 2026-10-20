"""Canonical conversion from validated daily bars to share-execution market rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.ingest import Lake
from quant_research.rf_daily_retry_execution import _limit_array


EXECUTION_MARKET_COLUMNS = [
    "trade_date", "instrument_id", "close", "previous_close", "volume",
    "daily_return", "trade_status", "buy_allowed", "sell_allowed",
    "at_upper_limit", "at_lower_limit", "share_factor",
    "cash_dividend_per_share", "action_source",
]


def validated_bars_to_execution_market(
    bars: pd.DataFrame,
    *,
    instrument_id: str,
    rules: dict,
    corporate_action_threshold: float,
    start_date: str,
    end_date: str,
    limit_tolerance_percentage_points: float = 0.20,
) -> pd.DataFrame:
    """Apply the one canonical daily-bar execution and action-proxy convention."""
    required = {
        "instrument_id", "trade_date", "close", "volume", "pct_chg",
        "trade_status", "is_st",
    }
    if missing := required - set(bars):
        raise KeyError(f"validated daily bars missing columns: {sorted(missing)}")
    frame = bars.sort_values("trade_date").copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous = close.shift(1)
    returns = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100.0
    raw_factor = previous * (1.0 + returns) / close
    factor = raw_factor.where(
        raw_factor.sub(1.0).abs().gt(float(corporate_action_threshold)), 1.0
    ).fillna(1.0)
    status = pd.to_numeric(frame["trade_status"], errors="coerce").eq(1)
    is_st = pd.to_numeric(frame["is_st"], errors="coerce").fillna(0).eq(1)
    limits = _limit_array(str(instrument_id), frame["trade_date"], is_st, rules)
    pct = pd.to_numeric(frame["pct_chg"], errors="coerce")
    tolerance = float(limit_tolerance_percentage_points)
    result = frame.assign(
        previous_close=previous, daily_return=returns,
        buy_allowed=status & pct.lt(limits - tolerance),
        sell_allowed=status & pct.gt(-limits + tolerance),
        at_upper_limit=status & pct.ge(limits - tolerance),
        at_lower_limit=status & pct.le(-limits + tolerance),
        share_factor=factor, cash_dividend_per_share=0.0,
        action_source=np.where(
            factor.ne(1.0), "inferred_value_preserving_factor", "none"
        ),
    )
    return result.loc[result["trade_date"].between(start_date, end_date), EXECUTION_MARKET_COLUMNS]


def read_execution_market_rows(
    lake: Lake,
    instrument_id: str,
    *,
    rules: dict,
    corporate_action_threshold: float,
    start_date: str,
    end_date: str,
    limit_tolerance_percentage_points: float = 0.20,
) -> pd.DataFrame:
    bars = pd.read_parquet(
        lake.table_path("daily_bars_validated", instrument_id.replace(".", "_")),
        columns=[
            "instrument_id", "trade_date", "close", "volume", "pct_chg",
            "trade_status", "is_st",
        ],
    )
    return validated_bars_to_execution_market(
        bars, instrument_id=instrument_id, rules=rules,
        corporate_action_threshold=corporate_action_threshold,
        start_date=start_date, end_date=end_date,
        limit_tolerance_percentage_points=limit_tolerance_percentage_points,
    )
