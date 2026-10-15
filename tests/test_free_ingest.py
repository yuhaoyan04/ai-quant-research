import pandas as pd
import pytest

from quant_research.free_ingest import verify_download
from quant_research.ingest import Lake


def _write_required_inputs(lake: Lake) -> None:
    calendar = pd.DataFrame(
        {
            "cal_date": ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-12"],
            "is_open": ["1", "1", "1", "1"],
        }
    )
    lake.write(
        "calendar", calendar, key="SSE", source="test",
        request={"start": "2024-01-01", "end": "2024-01-12"},
    )
    codes = pd.DataFrame({"instrument_id": ["sh.600000", "sz.000001", "sh.000001"]})
    lake.write(
        "historical_security_union", codes, key="all", source="test", request={},
    )
    for day in ("2024-01-05", "2024-01-12"):
        lake.write(
            "security_snapshot",
            pd.DataFrame({"instrument_id": ["sh.600000", "sz.000001"]}),
            key=day, source="test", request={"day": day},
        )


def _write_bar(lake: Lake, code: str, start: str = "2024-01-01") -> None:
    lake.write(
        "daily_bars_baostock",
        pd.DataFrame(
            {
                "instrument_id": [code],
                "trade_date": ["2024-01-05"],
                "close": [10.0],
            }
        ),
        key=code,
        source="test",
        request={"code": code, "start": start, "end": "2024-01-12", "adjustflag": "3"},
    )


def test_verify_download_requires_every_a_share(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    _write_bar(lake, "sh.600000")

    with pytest.raises(RuntimeError, match="missing bar files"):
        verify_download(lake, "2024-01-01", "2024-01-12")

    _write_bar(lake, "sz.000001")
    assert verify_download(lake, "2024-01-01", "2024-01-12") == {
        "a_share_codes": 2,
        "weekly_snapshots": 2,
        "covered_bar_requests": 2,
    }


def test_verify_download_rejects_shorter_pilot_range(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    _write_bar(lake, "sh.600000", start="2024-01-04")
    _write_bar(lake, "sz.000001")

    with pytest.raises(RuntimeError, match="without matching"):
        verify_download(lake, "2024-01-01", "2024-01-12")
