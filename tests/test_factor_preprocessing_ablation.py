import numpy as np
import pandas as pd

from quant_research.factor_preprocessing_ablation import (
    _update_portfolio,
    annual_summaries,
    configured_variants,
    transform_cross_section,
)


def test_configured_ablation_is_full_three_by_three_factorial() -> None:
    cfg = {
        "winsor_methods": ["none", "quantile", "mad"],
        "normalization_methods": ["raw", "zscore", "rank_zscore"],
    }
    variants = configured_variants(cfg)
    assert len(variants) == 9
    assert len({item[0] for item in variants}) == 9


def test_transform_is_date_local_finite_and_preserves_missing_indicator() -> None:
    frame = pd.DataFrame({
        "factor": [1.0, 2.0, 3.0, 4.0, np.nan, 1000.0],
    })
    design, audit = transform_cross_section(
        frame, ["factor"], winsor_method="mad", normalization_method="zscore",
        mad_multiplier=3.0,
    )
    assert design.shape == (6, 2)
    assert np.isfinite(design).all()
    assert audit.iloc[0]["clipped_fraction"] > 0
    assert design[-2, 1] > 0


def test_rank_zscore_is_invariant_to_positive_linear_rescaling() -> None:
    frame = pd.DataFrame({"factor": [1.0, 4.0, 2.0, 3.0]})
    scaled = frame.assign(factor=frame["factor"] * 100 + 7)
    left, _ = transform_cross_section(
        frame, ["factor"], winsor_method="none", normalization_method="rank_zscore",
        include_missing_indicators=False,
    )
    right, _ = transform_cross_section(
        scaled, ["factor"], winsor_method="none", normalization_method="rank_zscore",
        include_missing_indicators=False,
    )
    np.testing.assert_allclose(left, right)


def test_annual_summaries_keep_model_and_portfolio_years_aligned() -> None:
    daily = pd.DataFrame({
        "asof_date": ["2020-01-03", "2020-01-10"],
        "variant": ["v", "v"], "ridge_penalty": [1.0, 1.0],
        "rank_ic": [0.1, 0.2], "top_minus_bottom": [0.01, 0.02],
    })
    portfolio = pd.DataFrame({
        "asof_date": daily["asof_date"], "variant": ["v", "v"],
        "ridge_penalty": [1.0, 1.0], "net_return": [0.01, 0.02],
        "net_excess_return": [0.005, 0.01], "turnover": [1.0, 0.5],
        "realized_weight_fraction": [1.0, 1.0],
        "included_in_performance": [True, True],
    })
    result = annual_summaries(daily, portfolio, 52)
    assert result.loc[0, "year"] == 2020
    assert result.loc[0, "observations"] == 2
    assert result.loc[0, "portfolio_periods"] == 2


def test_portfolio_does_not_filter_on_future_return_availability() -> None:
    group = pd.DataFrame({
        "instrument_id": ["a", "b", "c"],
        "asof_date": ["2020-01-03"] * 3,
        "fwd": [np.nan, 0.02, 0.01],
    })
    result = _update_portfolio(
        group, np.array([3.0, 2.0, 1.0]), "fwd", 2, 20.0, 0.0, {},
    )
    assert result["realized_weight_fraction"] == 0.5
    assert result["gross_return"] == 0.01
