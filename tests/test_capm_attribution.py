import numpy as np
import pandas as pd

from quant_research.capm_attribution import capm_record, newey_west_ols


def test_newey_west_ols_recovers_exact_linear_coefficients() -> None:
    market = np.linspace(-0.03, 0.03, 200)
    design = np.column_stack([np.ones(len(market)), market])
    strategy = 0.0004 + 0.8 * market

    result = newey_west_ols(strategy, design, lags=5)

    np.testing.assert_allclose(result["coefficients"], [0.0004, 0.8], atol=1e-12)
    assert result["r_squared"] == 1


def test_capm_record_annualizes_alpha_and_reports_beta() -> None:
    rng = np.random.default_rng(17)
    market = rng.normal(0.0002, 0.01, 1000)
    noise = rng.normal(0, 0.002, 1000)
    strategy = 0.0003 + 1.2 * market + noise

    result = capm_record(
        pd.Series(strategy), pd.Series(market), lags=5,
        annualization_days=252, confidence_level=0.95,
    )

    assert abs(result["annualized_alpha_arithmetic"] - 0.0003 * 252) < 0.01
    assert abs(result["beta"] - 1.2) < 0.03
    assert result["annualized_alpha_ci_low"] < result["annualized_alpha_arithmetic"]
    assert result["annualized_alpha_ci_high"] > result["annualized_alpha_arithmetic"]
