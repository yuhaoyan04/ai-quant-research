import numpy as np
import pandas as pd

from quant_research.ingest import Lake
from quant_research.risk_exposure import (
    build_industry_exposures,
    float_cap_proxy,
    normalize_csrc_industry,
)


def test_float_cap_proxy_converts_percent_turnover_to_fraction() -> None:
    result = float_cap_proxy(pd.Series([1_000_000.0]), pd.Series([2.0]))
    assert result.iloc[0] == 50_000_000.0


def test_float_cap_proxy_does_not_fabricate_invalid_values() -> None:
    result = float_cap_proxy(
        pd.Series([1.0, 0.0, -1.0, 1.0]),
        pd.Series([0.0, 1.0, 1.0, np.nan]),
    )
    assert result.isna().all()


def test_normalize_modern_csrc_industry_uses_broad_section() -> None:
    assert normalize_csrc_industry("J66货币金融服务") == (
        "csrc_2012", "J66", "J"
    )


def test_normalize_legacy_csrc_industry_preserves_taxonomy_provenance() -> None:
    assert normalize_csrc_industry("金融保险业-银行业") == (
        "csrc_legacy", "LEGACY:金融保险业", "J"
    )
    assert normalize_csrc_industry("社会服务业-租赁服务业") == (
        "csrc_legacy", "LEGACY:社会服务业", "LEGACY_SOCIAL_SERVICES"
    )


def test_normalize_missing_industry_does_not_invent_a_group() -> None:
    assert normalize_csrc_industry("") == (None, None, None)


def test_build_industry_exposures_uses_latest_prior_snapshot(tmp_path, monkeypatch) -> None:
    lake = Lake(tmp_path / "lake")
    calendar = pd.DataFrame({
        "cal_date": ["2024-01-05", "2024-01-12", "2024-01-26"],
        "is_open": ["1", "1", "1"],
    })
    lake.write("calendar", calendar, key="SSE", source="test", request={})
    panel = pd.DataFrame({
        "instrument_id": ["sh.600000", "sz.000001"] * 3,
        "asof_date": np.repeat(["2024-01-05", "2024-01-12", "2024-01-26"], 2),
    })
    lake.write(
        "multi_horizon_factor_panel", panel, key="2024", source="test", request={}
    )
    for snapshot_date, update_date, industries in (
        ("2024-01-05", "2024-01-01", ["J66货币金融服务", "C39计算机制造"]),
        ("2024-01-26", "2024-01-22", ["J66货币金融服务", "I65软件服务"]),
    ):
        lake.write(
            "industry_snapshot_baostock",
            pd.DataFrame({
                "instrument_id": ["sh.600000", "sz.000001"],
                "update_date": [update_date] * 2,
                "industry_raw": industries,
                "industry_classification": ["证监会行业分类"] * 2,
            }),
            key=snapshot_date, source="test", request={"date": snapshot_date},
        )
    monkeypatch.setattr(
        "quant_research.risk_exposure.load_config",
        lambda: {
            "start_year": 2024,
            "end_year": 2024,
            "industry_minimum_coverage": 0.95,
        },
    )

    build_industry_exposures(lake)

    result = lake.read("pit_industry_exposures", "2024")
    middle = result.loc[result["asof_date"].eq("2024-01-12")]
    assert middle["source_snapshot_date"].eq("2024-01-05").all()
    assert set(middle["industry_broad_code"]) == {"C", "J"}
    audit = lake.read("pit_industry_exposure_index", "v1")
    assert audit.iloc[0]["future_dated_rows"] == 0
