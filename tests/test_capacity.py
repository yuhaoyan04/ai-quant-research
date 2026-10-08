import pandas as pd
import pytest

from quant_research.capacity import capacity_by_date


def test_capacity_declines_when_aum_exceeds_adv_limit() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "a", "asof_date": "2024-01-05", "score": 2.0, "target": 0.01,
         "amihud_percentile": 0.5, "turnover_percentile": 0.5, "adv20_cny": 1_000_000,
         "volatility_20d": 0.02},
        {"instrument_id": "b", "asof_date": "2024-01-05", "score": 1.0, "target": 0.02,
         "amihud_percentile": 0.5, "turnover_percentile": 0.5, "adv20_cny": 1_000_000,
         "volatility_20d": 0.02},
    ])
    small = capacity_by_date(predictions, 2, "target", None, None, 10_000, 0.05, 0.5)
    large = capacity_by_date(predictions, 2, "target", None, None, 10_000_000, 0.05, 0.5)
    assert small.loc[0, "fill_ratio"] == 1.0
    assert large.loc[0, "fill_ratio"] < 0.02


def test_screen_removes_illiquid_high_score_before_capacity() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "illiquid", "asof_date": "2024-01-05", "score": 10.0, "target": 0.1,
         "amihud_percentile": 1.0, "turnover_percentile": 0.0, "adv20_cny": 1.0,
         "volatility_20d": 0.02},
        {"instrument_id": "liquid", "asof_date": "2024-01-05", "score": 1.0, "target": 0.01,
         "amihud_percentile": 0.5, "turnover_percentile": 0.5, "adv20_cny": 1_000_000,
         "volatility_20d": 0.02},
    ])
    result = capacity_by_date(predictions, 1, "target", 0.9, 0.1, 10_000, 0.05, 0.5)
    assert result.loc[0, "fill_ratio"] == 1.0


def test_impact_is_charged_only_on_fillable_trade() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "a", "asof_date": "2024-01-05", "score": 2.0, "target": 0.01,
         "amihud_percentile": 0.5, "turnover_percentile": 0.5, "adv20_cny": 1_000_000,
         "volatility_20d": 0.02},
        {"instrument_id": "b", "asof_date": "2024-01-05", "score": 1.0, "target": 0.02,
         "amihud_percentile": 0.5, "turnover_percentile": 0.5, "adv20_cny": 1_000_000,
         "volatility_20d": 0.02},
    ])
    result = capacity_by_date(
        predictions, 2, "target", None, None,
        aum_cny=10_000_000, participation_limit=0.01, impact_eta=0.5)
    # Each name can trade only 10k. Impact is 0.5*2%*sqrt(1%) = 10bp
    # on the 20k actually filled, or 0.02bp of the 10m portfolio.
    assert result.loc[0, "estimated_impact_fraction_aum"] == pytest.approx(0.000002)
