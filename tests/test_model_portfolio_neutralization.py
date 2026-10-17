import numpy as np
import pandas as pd

from quant_research.model_portfolio_neutralization import (
    model_designs,
    portfolio_risk_diagnostics,
)


def _config() -> dict:
    return {
        "pipelines": [
            "raw", "size_neutralized", "industry_neutralized",
            "industry_size_neutralized",
        ],
        "minimum_industry_group_size": 5,
        "quantile_limits": [0.01, 0.99],
        "mad_multiplier": 5.0,
        "include_missing_indicators": True,
    }


def _cross_section() -> pd.DataFrame:
    rows = 120
    index = np.arange(rows, dtype="float64")
    industries = np.repeat(["A", "B", "C"], rows // 3)
    size = np.exp(10.0 + index / rows)
    factor_one = 0.8 * np.log(size) + np.repeat([1.0, -0.5, 0.2], rows // 3)
    factor_two = np.sin(index / 8.0) + np.repeat([-0.2, 0.4, 0.7], rows // 3)
    factor_two[[3, 44, 91]] = np.nan
    return pd.DataFrame({
        "instrument_id": [f"S{item:03d}" for item in range(rows)],
        "asof_date": "2025-01-03",
        "industry_broad_code": industries,
        "float_cap_proxy_cny": size,
        "factor_one": factor_one,
        "factor_two": factor_two,
    })


def test_model_designs_share_sample_and_neutralize_every_model_feature() -> None:
    group = _cross_section()
    designs, risks, complete = model_designs(
        group, ["factor_one", "factor_two"], _config()
    )
    assert complete.all()
    assert set(designs) == set(_config()["pipelines"])
    assert all(design.shape == (120, 4) for design in designs.values())
    combined = designs["industry_size_neutralized"]
    for industry in ("A", "B", "C"):
        mask = group["industry_broad_code"].eq(industry).to_numpy()
        np.testing.assert_allclose(combined[mask].mean(axis=0), 0.0, atol=1e-10)
    for column in range(4):
        if combined[:, column].std(ddof=1) > 0:
            assert abs(np.corrcoef(combined[:, column], risks["size_z"])[0, 1]) < 1e-10


def test_model_designs_use_one_exposure_complete_mask() -> None:
    group = _cross_section()
    group.loc[5, "float_cap_proxy_cny"] = np.nan
    group.loc[70, "industry_broad_code"] = ""
    _, _, complete = model_designs(group, ["factor_one", "factor_two"], _config())
    assert complete.sum() == 118
    assert not complete[5]
    assert not complete[70]


def test_portfolio_risk_diagnostics_match_equal_weight_allocations() -> None:
    group = _cross_section().reset_index(drop=True)
    score = np.linspace(1.0, 0.0, len(group))
    selected = group.iloc[:20].copy()
    selected["portfolio_weight"] = 1.0 / len(selected)
    selected["portfolio_rank"] = np.arange(1, len(selected) + 1)
    diagnostics, allocation = portfolio_risk_diagnostics(
        group, selected, score, np.linspace(-1.0, 1.0, len(group))
    )
    assert np.isclose(allocation["portfolio_weight"].sum(), 1.0)
    assert np.isclose(allocation["universe_weight"].sum(), 1.0)
    assert np.isclose(diagnostics["portfolio_industry_hhi"], 1.0)
    assert diagnostics["industry_active_weight_l1"] > 0.6
    assert diagnostics["maximum_industry_weight"] == 1.0
    assert diagnostics["score_industry_max_abs_mean"] > 0
