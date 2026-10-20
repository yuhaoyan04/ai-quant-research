from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.alpha_evidence_matrix import incremental_regression_record


def _config() -> dict:
    return {
        "styles": ["size", "beta", "volatility", "liquidity", "value", "momentum"],
        "annualization_days": 252,
        "confidence_level": 0.95,
    }


def test_incremental_regression_recovers_known_intercept() -> None:
    rng = np.random.default_rng(7)
    observations = 800
    frame = pd.DataFrame({
        "benchmark_excess_return": rng.normal(0, 0.01, observations),
        **{style: rng.normal(0, 0.01, observations) for style in _config()["styles"]},
    })
    frame["paired_incremental_net_return"] = (
        0.0002 + 0.15 * frame["benchmark_excess_return"]
        + 0.05 * frame["size"] - 0.03 * frame["momentum"]
    )
    result = incremental_regression_record(frame, _config(), lags=5)
    assert result["annualized_alpha_arithmetic"] == pytest.approx(0.0002 * 252)
    assert result["loading_market"] == pytest.approx(0.15)
    assert result["loading_size"] == pytest.approx(0.05)
    assert result["loading_momentum"] == pytest.approx(-0.03)


def test_incremental_regression_uses_arithmetic_alpha_annualization() -> None:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "benchmark_excess_return": rng.normal(0, 0.01, 400),
        **{style: rng.normal(0, 0.01, 400) for style in _config()["styles"]},
    })
    frame["paired_incremental_net_return"] = 0.0001 + rng.normal(0, 0.001, 400)
    result = incremental_regression_record(frame, _config(), lags=10)
    assert result["newey_west_lags"] == 10
    assert result["observations"] == 400
    assert result["annualized_alpha_ci_low"] < result["annualized_alpha_arithmetic"]
    assert result["annualized_alpha_ci_high"] > result["annualized_alpha_arithmetic"]
    assert 0 <= result["one_sided_p_value"] <= 1
