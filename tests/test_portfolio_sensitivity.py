from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.portfolio_sensitivity import simulate_weighted_long_only


def _predictions() -> pd.DataFrame:
    rows = []
    for date_number, date in enumerate(("2020-01-03", "2020-01-10")):
        for number in range(4):
            rows.append({
                "instrument_id": f"s{number}", "asof_date": date,
                "available_date": date, "score": 4 - number,
                "fwd_return_5d_lag1": np.nan if number == 0 and date_number == 0 else number / 100,
                "volatility_20d": 0.01 * (number + 1),
            })
    return pd.DataFrame(rows)


def test_weighting_is_ex_ante_and_missing_outcome_is_not_removed() -> None:
    result = simulate_weighted_long_only(
        _predictions(), top_n=2, weighting="equal_weight",
        target="fwd_return_5d_lag1", volatility="volatility_20d",
        clip_quantiles=(0.05, 0.95), cost_bps=20,
        minimum_realized_weight_fraction=0.4,
    )
    assert result.iloc[0]["holdings"] == 2
    assert result.iloc[0]["realized_weight_fraction"] == 0.5
    assert np.isclose(result.iloc[0]["gross_return"], 0.005)


def test_inverse_volatility_is_less_concentrated_in_high_volatility_name() -> None:
    result = simulate_weighted_long_only(
        _predictions(), top_n=4, weighting="inverse_volatility",
        target="fwd_return_5d_lag1", volatility="volatility_20d",
        clip_quantiles=(0.0, 1.0), cost_bps=0,
        minimum_realized_weight_fraction=0.0,
    )
    assert result["maximum_target_weight"].between(0.47, 0.49).all()
    assert np.isclose(result.iloc[0]["turnover"], 1.0)
