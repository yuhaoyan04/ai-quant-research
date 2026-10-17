import numpy as np
import pandas as pd
import pytest

from quant_research.multi_factor_risk_attribution import (
    build_style_exposures,
    ewma_covariance,
    fit_factor_cross_section,
)


def config() -> dict:
    return {
        "target": "forward_return",
        "style_winsor_quantiles": [0.01, 0.99],
        "covariance": {
            "minimum_weeks": 3, "trailing_weeks": 4,
            "ewma_half_life_weeks": 2, "diagonal_shrinkage": 0.2,
        },
    }


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append({
            "instrument_id": f"S{index:02d}",
            "industry_broad_code": "A" if index < 6 else "B",
            "size_z": index - 5.5,
            "beta_252d": 0.7 + index * 0.05,
            "volatility_20d": 0.01 + index * 0.001,
            "amihud_20d": 1e-10 * (index + 1),
            "turnover_20d": 0.1 + index * 0.02,
            "earnings_yield": 0.03 + index * 0.002,
            "book_to_price": 0.4 + index * 0.03,
            "momentum_120d": -0.1 + index * 0.02,
            "momentum_20d_skip_5d": -0.05 + index * 0.01,
            "forward_return": 0.01 + (0.002 if index >= 6 else -0.001)
                              + index * 0.0003,
        })
    return pd.DataFrame(rows)


def test_style_exposures_are_finite_and_zero_mean() -> None:
    styles = build_style_exposures(sample_frame(), config())

    assert np.isfinite(styles.to_numpy()).all()
    assert styles.mean().abs().max() < 1e-12


def test_factor_cross_section_has_exact_stock_return_identity() -> None:
    result = fit_factor_cross_section(sample_frame(), config())
    frame = result["frame"]
    reconstructed = []
    for row in frame.itertuples():
        value = result["industry_coefficients"][row.industry_broad_code]
        value += sum(
            getattr(row, f"style_{style}") * factor_return
            for style, factor_return in result["style_returns"].items()
        )
        value += row.residual_return
        reconstructed.append(value)

    assert reconstructed == pytest.approx(frame["forward_return"].tolist(), abs=1e-12)
    assert result["market_return"] == pytest.approx(frame["forward_return"].mean())


def test_ewma_covariance_requires_only_prior_history_and_shrinks() -> None:
    history = [
        np.array([0.01, 0.02]), np.array([-0.01, 0.00]),
        np.array([0.02, -0.01]),
    ]

    covariance = ewma_covariance(history, config())

    assert covariance is not None
    assert covariance.shape == (2, 2)
    assert np.all(np.diag(covariance) >= 0)
    assert ewma_covariance(history[:2], config()) is None
