import pandas as pd
import pytest

from quant_research.constrained_portfolio_daily_execution import (
    paired_execution_bootstrap,
    prepare_targets,
)


def test_prepare_targets_attaches_execution_date_and_preserves_weights() -> None:
    holdings = pd.DataFrame([
        {
            "instrument_id": code, "asof_date": "2024-01-05",
            "target_weight": weight, "model": "ridge_locked",
            "alpha_pipeline": "raw", "strategy": "raw__risk_constrained",
            "construction": "risk_constrained", "test_year": 2024,
        }
        for code, weight in (("A", 0.6), ("B", 0.4))
    ])
    dates = pd.DataFrame([{
        "asof_date": "2024-01-05", "available_date": "2024-01-08",
    }])

    result = prepare_targets(holdings, dates, "risk_constrained")

    assert result["execution_date"].unique().tolist() == ["2024-01-08"]
    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert result["signal_date"].unique().tolist() == ["2024-01-05"]


def test_prepare_targets_rejects_missing_execution_date() -> None:
    holdings = pd.DataFrame([{
        "instrument_id": "A", "asof_date": "2024-01-05", "target_weight": 1.0,
        "model": "ridge_locked", "alpha_pipeline": "raw",
        "strategy": "raw__risk_constrained", "construction": "risk_constrained",
        "test_year": 2024,
    }])

    with pytest.raises(ValueError, match="execution dates missing"):
        prepare_targets(holdings, pd.DataFrame(columns=["asof_date", "available_date"]),
                        "risk_constrained")


def test_paired_execution_bootstrap_uses_same_daily_clock() -> None:
    rows = []
    for date, ideal, constrained in (
        ("2024-01-02", 0.01, 0.00),
        ("2024-01-03", 0.02, 0.01),
        ("2024-01-04", -0.01, -0.02),
    ):
        for scenario, value in (
            ("nonoverlap_all_fills_upper_bound", ideal),
            ("daily_retry_limit_and_suspension_constrained", constrained),
        ):
            rows.append({
                "model": "ridge_locked", "alpha_pipeline": "raw",
                "strategy": "raw__risk_constrained", "trade_date": date,
                "scenario": scenario, "net_return": value,
            })
    config = {
        "annualization_periods": 252,
        "bootstrap": {
            "block_length_trading_days": 2, "samples": 100, "random_seed": 7,
        },
    }

    result = paired_execution_bootstrap(pd.DataFrame(rows), config).iloc[0]

    assert result["observations"] == 3
    assert result["point_estimate"] == pytest.approx(-0.01 * 252)
