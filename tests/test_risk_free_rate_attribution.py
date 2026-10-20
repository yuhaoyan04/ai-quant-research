import numpy as np
import pandas as pd

from quant_research.risk_free_rate_attribution import (
    align_rates_to_clock,
    annual_quote_to_holding_return,
    normalize_observations,
    request_windows,
)


def _config() -> dict:
    return {
        "download": {
            "curve_name": "中债国债收益率曲线",
            "provider": "CCDC",
            "source_name": "ChinaBond Government Bond Yield Curve",
            "source_url": "https://example.test/history",
            "api_url": "https://example.test/api",
            "annual_quote_convention": "ytm_annual_percent",
            "raw_quote_unit": "percent_per_annum",
            "publication_timestamp_status": "not_provided",
            "availability_lag_calendar_days": 1,
        },
        "rate_scenarios": [
            {"scenario_id": "R0_zero_rate", "tenor": "zero", "source_column": None},
            {"scenario_id": "R1_ccdc_cgb_3m", "tenor": "3M", "source_column": "3月"},
            {"scenario_id": "R2_ccdc_cgb_1y", "tenor": "1Y", "source_column": "1年"},
        ],
        "zero_rate_scenario": "R0_zero_rate",
        "maximum_staleness_calendar_days": 10,
    }


def test_request_windows_are_inclusive_without_overlap() -> None:
    windows = request_windows("2024-01-01", "2024-01-10", 4)
    assert windows == [
        ("20240101", "20240104"),
        ("20240105", "20240108"),
        ("20240109", "20240110"),
    ]


def test_annual_quote_uses_act365_compounding() -> None:
    result = annual_quote_to_holding_return(pd.Series([0.0365]), pd.Series([3]))
    expected = (1.0 + 0.0365) ** (3.0 / 365.0) - 1.0
    assert np.isclose(result.iloc[0], expected)
    assert not np.isclose(result.iloc[0], 0.0365 / 252.0)


def test_normalize_and_alignment_are_pit_safe_and_weekend_aware() -> None:
    raw = pd.DataFrame({
        "曲线名称": ["中债国债收益率曲线"] * 3,
        "日期": ["2024-01-03", "2024-01-04", "2024-01-05"],
        "3月": [1.8, 1.9, 2.0],
        "1年": [2.1, 2.2, 2.3],
    })
    cfg = _config()
    observations = normalize_observations(raw, cfg, "2024-01-06T00:00:00+00:00")
    clock = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
        "period_start_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
        "calendar_days": [1, 3],
    })

    aligned = align_rates_to_clock(observations, clock, cfg)
    r1 = aligned.loc[aligned["scenario_id"].eq("R1_ccdc_cgb_3m")]

    # Jan-04 quote is not usable until Jan-05, so the Jan-05 close interval
    # uses the Jan-03 observation known at the Jan-04 period start.
    assert r1.iloc[0]["observation_date"] == "2024-01-03"
    assert r1.iloc[0]["available_from_date"] == "2024-01-04"
    # Friday-to-Monday accrual uses three ACT/365 calendar days.
    assert r1.iloc[1]["observation_date"] == "2024-01-04"
    assert np.isclose(r1.iloc[1]["risk_free_return"], (1.019 ** (3 / 365)) - 1)
    assert not r1["source_missing_flag"].any()
    assert not r1["interpolation_flag"].any()


def test_missing_or_stale_observation_is_explicit_not_forward_filled() -> None:
    raw = pd.DataFrame({
        "曲线名称": ["中债国债收益率曲线"],
        "日期": ["2024-01-01"],
        "3月": [1.8],
        "1年": [2.1],
    })
    cfg = _config()
    cfg["maximum_staleness_calendar_days"] = 2
    observations = normalize_observations(raw, cfg, "2024-01-02T00:00:00+00:00")
    clock = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-11"]),
        "period_start_date": pd.to_datetime(["2024-01-10"]),
        "calendar_days": [1],
    })

    aligned = align_rates_to_clock(observations, clock, cfg)
    r1 = aligned.loc[aligned["scenario_id"].eq("R1_ccdc_cgb_3m")].iloc[0]
    assert r1["stale_flag"]
    assert pd.isna(r1["risk_free_return"])
