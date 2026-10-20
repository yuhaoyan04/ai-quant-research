import numpy as np
import pandas as pd

from quant_research.market_benchmark_attribution import (
    benchmark_comparison,
    normalize_official_observations,
)


def _config() -> dict:
    return {
        "download": {
            "provider": "CSI",
            "source_name": "official",
            "source_url": "https://example.test",
            "api_url": "https://example.test/api",
            "publication_timestamp_status": "not_provided",
            "availability_lag_calendar_days": 1,
        },
        "annualization_days": 252,
        "material_daily_total_minus_price_threshold": 0.0001,
        "same_provider_price_variant": "official_csindex_price",
        "primary_total_return_variant": "official_csindex_total_return",
    }


def _raw_series(code: str, return_type: str, levels: list[float]) -> pd.DataFrame:
    changes = [0.0, levels[1] / levels[0] - 1, levels[2] / levels[1] - 1]
    return pd.DataFrame({
        "tradeDate": ["20240102", "20240103", "20240104"],
        "indexCode": [code] * 3,
        "indexNameCnAll": [code] * 3,
        "indexNameEnAll": [code] * 3,
        "close": levels,
        "changePct": [value * 100 for value in changes],
        "benchmark": ["CSI500"] * 3,
        "return_type": [return_type] * 3,
        "requested_index_code": [code] * 3,
    })


def test_normalize_recomputes_returns_from_close_and_marks_availability() -> None:
    raw = _raw_series("000905", "price_return", [100.0, 101.0, 99.99])
    result = normalize_official_observations(raw, _config(), "2024-01-05T00:00:00+00:00")

    assert result.iloc[0]["variant_id"] == "official_csindex_price"
    assert pd.isna(result.iloc[0]["benchmark_return"])
    assert np.isclose(result.iloc[1]["benchmark_return"], 0.01)
    assert result.iloc[1]["available_from_date"] == "2024-01-04"
    assert result.iloc[1]["source_provider"] == "CSI"
    assert not result.iloc[1]["interpolation_flag"]


def test_benchmark_comparison_separates_geometric_and_arithmetic_wedge() -> None:
    dates = ["2024-01-02", "2024-01-03"]
    daily = pd.DataFrame({
        "trade_date": dates + dates,
        "benchmark": ["CSI500"] * 4,
        "variant_id": ["official_csindex_price"] * 2 + ["official_csindex_total_return"] * 2,
        "benchmark_return": [0.01, -0.01, 0.011, -0.009],
    })

    result = benchmark_comparison(daily, _config()).iloc[0]

    assert result["observations"] == 2
    assert np.isclose(result["annualized_arithmetic_total_minus_price"], 0.252)
    assert result["annualized_geometric_total_relative_to_price"] > 0
    assert result["days_abs_total_minus_price_above_threshold"] == 2


def test_total_return_variant_is_identified_from_official_code() -> None:
    raw = _raw_series("H00905", "total_return", [100.0, 102.0, 104.0])
    result = normalize_official_observations(raw, _config(), "2024-01-05T00:00:00+00:00")
    assert result["variant_id"].eq("official_csindex_total_return").all()
    assert result["return_type"].eq("total_return").all()
