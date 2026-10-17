import pandas as pd
import pytest

from quant_research.free_ingest import (
    build_bars,
    build_industry_snapshots,
    industry_snapshot_dates,
    verify_download,
    verify_industry_download,
)
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
        "benchmark_codes": 0,
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


def test_verify_download_rejects_readable_truncated_snapshot(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    _write_bar(lake, "sh.600000")
    _write_bar(lake, "sz.000001")
    lake.write(
        "security_snapshot",
        pd.DataFrame({"instrument_id": ["sh.600000"]}),
        key="2024-01-12", source="test", request={"day": "2024-01-12"},
    )

    with pytest.raises(RuntimeError, match="truncated weekly snapshots"):
        verify_download(lake, "2024-01-01", "2024-01-12")


def test_verify_download_requires_configured_benchmarks(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    _write_bar(lake, "sh.600000")
    _write_bar(lake, "sz.000001")

    with pytest.raises(RuntimeError, match="missing bar files"):
        verify_download(
            lake, "2024-01-01", "2024-01-12",
            required_benchmarks=["sh.000300"],
        )

    _write_bar(lake, "sh.000300")
    assert verify_download(
        lake, "2024-01-01", "2024-01-12",
        required_benchmarks=["sh.000300"],
    )["benchmark_codes"] == 1


def test_verify_industry_download_rejects_future_dated_snapshot(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    for day, update in (("2024-01-05", "2024-01-01"),
                        ("2024-01-12", "2024-01-15")):
        lake.write(
            "industry_snapshot_baostock",
            pd.DataFrame({
                "instrument_id": ["sh.600000"],
                "update_date": [update],
                "industry_raw": ["J66货币金融服务"],
                "industry_classification": ["证监会行业分类"],
                "asof_date": [day],
            }),
            key=day, source="test", request={"date": day},
        )

    with pytest.raises(RuntimeError, match="future=1"):
        verify_industry_download(lake, "2024-01-01", "2024-01-12")


def test_verify_industry_download_accepts_complete_pit_snapshots(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    for day in ("2024-01-05", "2024-01-12"):
        lake.write(
            "industry_snapshot_baostock",
            pd.DataFrame({
                "instrument_id": ["sh.600000", "sz.000001"],
                "update_date": ["2024-01-01"] * 2,
                "industry_raw": ["J66货币金融服务", "C39计算机、通信和其他电子设备制造业"],
                "industry_classification": ["证监会行业分类"] * 2,
                "asof_date": [day] * 2,
            }),
            key=day, source="test", request={"date": day},
        )

    assert verify_industry_download(
        lake, "2024-01-01", "2024-01-12"
    )["industry_snapshots"] == 2


def test_verify_industry_download_rejects_truncated_membership(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    for day in ("2024-01-05", "2024-01-12"):
        lake.write(
            "industry_snapshot_baostock",
            pd.DataFrame({
                "instrument_id": ["sh.600000"],
                "update_date": ["2024-01-01"],
                "industry_raw": ["J66货币金融服务"],
                "industry_classification": ["证监会行业分类"],
                "asof_date": [day],
            }),
            key=day, source="test", request={"date": day},
        )

    with pytest.raises(RuntimeError, match="security_coverage=50.0000%"):
        verify_industry_download(lake, "2024-01-01", "2024-01-12")


def test_verify_industry_download_reports_missing_columns(tmp_path):
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)
    for day in ("2024-01-05", "2024-01-12"):
        lake.write(
            "industry_snapshot_baostock",
            pd.DataFrame({"instrument_id": ["sh.600000"], "asof_date": [day]}),
            key=day, source="test", request={"date": day},
        )

    with pytest.raises(RuntimeError, match="missing_columns"):
        verify_industry_download(lake, "2024-01-01", "2024-01-12")


def test_industry_schedule_uses_initial_week_and_month_ends() -> None:
    calendar = pd.DataFrame({
        "cal_date": [
            "2024-01-05", "2024-01-12", "2024-01-31",
            "2024-02-02", "2024-02-29",
        ],
        "is_open": ["1"] * 5,
    })
    assert industry_snapshot_dates(calendar) == [
        "2024-01-05", "2024-01-12", "2024-02-29"
    ]


def test_build_bars_rejects_concurrent_baostock_sessions(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    with pytest.raises(ValueError, match="prohibits concurrent connections"):
        build_bars(
            object(), lake, ["sh.600000"], "2024-01-01", "2024-01-12",
            workers=2,
        )


def test_industry_download_stops_after_first_vendor_failure(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    _write_required_inputs(lake)

    class FailingProvider:
        calls = 0

        def industry_asof(self, asof):
            self.calls += 1
            raise RuntimeError(f"vendor failure on {asof}")

    provider = FailingProvider()
    with pytest.raises(RuntimeError, match="vendor failure"):
        build_industry_snapshots(
            provider, lake, "2024-01-01", "2024-01-12", workers=1,
        )
    assert provider.calls == 1
