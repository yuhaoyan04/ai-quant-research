from pathlib import Path

import pandas as pd

from quant_research.ingest import Lake, date_windows, merge_records


def test_new_vendor_record_replaces_prior_natural_key() -> None:
    old = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"], "close": [10.0]})
    new = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"], "close": [10.1]})
    result = merge_records(old, new, ["ts_code", "trade_date"])
    assert len(result) == 1
    assert result.loc[0, "close"] == 10.1


def test_calendar_rejects_duplicate_natural_key(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    duplicated = pd.DataFrame({"exchange": ["SSE", "SSE"], "cal_date": ["20240102", "20240102"]})
    try:
        lake.write("calendar", duplicated, key="SSE", source="test", request={})
    except ValueError as exc:
        assert "duplicate natural key" in str(exc)
    else:
        raise AssertionError("duplicate calendar rows must be rejected")


def test_date_windows_are_contiguous_and_cover_boundaries() -> None:
    windows = list(date_windows("2020-01-15", "2020-04-10", months=1))
    assert windows == [
        ("20200115", "20200214"),
        ("20200215", "20200314"),
        ("20200315", "20200410"),
    ]
