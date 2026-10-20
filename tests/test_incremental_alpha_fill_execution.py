from __future__ import annotations

import pandas as pd
import pytest

from quant_research.incremental_alpha_fill_execution import _comparison


def test_fill_comparison_reports_model_minus_style_differences() -> None:
    common = {
        "model": "m", "alpha_pipeline": "raw", "strategy": "s",
        "maximum_drawdown": -0.2, "mean_turnover": 0.05,
        "mean_target_tracking_l1": 0.02, "mean_cash_weight": 0.001,
        "total_orders": 10, "total_fills": 12, "total_cost_cny": 100.0,
    }
    summary = pd.DataFrame([
        {**common, "portfolio_type": "ml_model", "annualized_return": 0.20,
         "sharpe_zero_rate": 1.0, "fill_ratio_by_shares": 0.90},
        {**common, "portfolio_type": "style_matched_benchmark", "annualized_return": 0.15,
         "sharpe_zero_rate": 0.8, "fill_ratio_by_shares": 0.95},
    ])
    result = _comparison(summary).iloc[0]
    assert result["annualized_return_difference_model_minus_style"] == pytest.approx(0.05)
    assert result["fill_ratio_by_shares_difference_model_minus_style"] == pytest.approx(-0.05)
    assert result["model_total_orders"] == 10


def test_fill_comparison_keeps_candidate_identity() -> None:
    rows = []
    for portfolio_type in ("ml_model", "style_matched_benchmark"):
        rows.append({
            "model": "ridge", "alpha_pipeline": "neutral", "strategy": "locked",
            "portfolio_type": portfolio_type, "annualized_return": 0.1,
            "sharpe_zero_rate": 0.5, "maximum_drawdown": -0.2,
            "mean_turnover": 0.05, "mean_target_tracking_l1": 0.02,
            "mean_cash_weight": 0.001, "fill_ratio_by_shares": 0.9,
            "total_orders": 10, "total_fills": 10, "total_cost_cny": 1.0,
        })
    result = _comparison(pd.DataFrame(rows))
    assert result[["model", "alpha_pipeline", "strategy"]].iloc[0].tolist() == [
        "ridge", "neutral", "locked",
    ]
