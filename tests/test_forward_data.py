from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_research.factors import build_raw_factors, next_trading_day_map
from quant_research.forward_data import (
    completed_weekly_dates,
    forward_table,
    load_config,
    materialize_incremental,
    read_forward,
)
from quant_research.ingest import Lake
from quant_research.panel import build_labels, load_config as load_panel_config


def _calendar(periods: int = 210) -> pd.DataFrame:
    days = pd.bdate_range("2025-01-02", periods=periods)
    return pd.DataFrame({
        "cal_date": days.strftime("%Y-%m-%d"), "is_open": "1", "exchange": "SSE",
    })


def _raw(calendar: pd.DataFrame, rows: int) -> pd.DataFrame:
    dates = calendar["cal_date"].iloc[:rows].reset_index(drop=True)
    returns = np.sin(np.arange(rows) / 17.0) * 1.5
    close = 10 * np.cumprod(1 + returns / 100)
    previous = np.r_[10.0, close[:-1]]
    return pd.DataFrame({
        "trade_date": dates, "instrument_id": "sh.600000",
        "open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
        "close": close, "pre_close": previous, "volume": 1_000_000 + np.arange(rows),
        "amount": close * (1_000_000 + np.arange(rows)), "adjustflag": "3",
        "turn": 1.0 + np.arange(rows) / 10_000, "trade_status": "1",
        "pct_chg": returns, "pe_ttm": 12.0, "pb_mrq": 1.5,
        "ps_ttm": 2.0, "pcf_ncf_ttm": 8.0, "is_st": "0",
    })


def test_completed_weekly_dates_excludes_open_partial_week() -> None:
    calendar = _calendar(12)
    end = str(calendar["cal_date"].iloc[3])  # a Tuesday in the second week
    result = completed_weekly_dates(calendar, end)
    assert result == [str(calendar["cal_date"].iloc[1])]


def test_zero_byte_forward_file_falls_back_to_historical(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    calendar = _calendar(5)
    lake.write("calendar", calendar, key="SSE", source="test", request={})
    path = lake.table_path(forward_table("calendar"), "SSE")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert_frame_equal(read_forward(lake, "calendar", "SSE"), calendar)


def test_incremental_materialization_matches_full_recomputation(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    calendar = _calendar()
    lake.write("calendar", calendar, key="SSE", source="test", request={})
    first = _raw(calendar, 170)
    lake.write("daily_bars_baostock", first, key="sh.600000", source="test", request={})
    cfg = copy.deepcopy(load_config())
    cfg["status_path"] = str(tmp_path / "status.json")
    materialize_incremental(lake=lake, cfg=cfg, codes=["sh.600000"])

    extended = _raw(calendar, 180)
    lake.write("daily_bars_baostock", extended, key="sh.600000", source="test", request={})
    materialize_incremental(lake=lake, cfg=cfg, codes=["sh.600000"])

    validated = lake.read(forward_table("daily_bars_validated"), "sh.600000")
    actual_factors = lake.read(forward_table("factors_raw"), "sh.600000").reset_index(drop=True)
    expected_factors = build_raw_factors(validated, next_trading_day_map(calendar))
    assert_frame_equal(actual_factors, expected_factors, check_dtype=False)

    panel_cfg = load_panel_config()
    actual_labels = lake.read(forward_table("labels_raw"), "sh.600000").reset_index(drop=True)
    expected_labels = build_labels(
        validated, list(panel_cfg["label_horizons_trading_days"]),
        int(panel_cfg["execution_lag_trading_days"]),
    )
    assert_frame_equal(actual_labels, expected_labels, check_dtype=False)
    assert len(validated) == len(extended)
