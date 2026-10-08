import pandas as pd

from quant_research.portfolio import performance_summary, simulate_long_only


def test_selection_uses_scores_and_costs_turnover() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "a", "asof_date": "2024-01-05", "available_date": "2024-01-08", "score": 2, "target": 0.10},
        {"instrument_id": "b", "asof_date": "2024-01-05", "available_date": "2024-01-08", "score": 1, "target": -0.10},
        {"instrument_id": "a", "asof_date": "2024-01-12", "available_date": "2024-01-15", "score": 1, "target": 0.00},
        {"instrument_id": "b", "asof_date": "2024-01-12", "available_date": "2024-01-15", "score": 2, "target": 0.20},
    ])
    result = simulate_long_only(predictions, 1, "target", 20, 1.0)
    assert result.loc[0, "gross_return"] == 0.10
    assert result.loc[1, "gross_return"] == 0.20
    assert result.loc[1, "turnover"] == 1.0
    assert result.loc[1, "transaction_cost"] == 0.002


def test_missing_outcome_is_not_renormalized_away() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "a", "asof_date": "2024-01-05", "available_date": "2024-01-08", "score": 2, "target": None},
        {"instrument_id": "b", "asof_date": "2024-01-05", "available_date": "2024-01-08", "score": 1, "target": 0.10},
    ])
    result = simulate_long_only(predictions, 2, "target", 0, 0.4)
    assert result.loc[0, "gross_return"] == 0.05
    assert result.loc[0, "realized_weight_fraction"] == 0.5


def test_summary_reports_drawdown_and_turnover() -> None:
    weekly = pd.DataFrame({
        "net_return": [0.02, -0.01, 0.01], "net_excess_return": [0.01, -0.02, 0.0],
        "turnover": [1.0, 0.5, 0.5], "realized_weight_fraction": [1.0] * 3,
        "included_in_performance": [True] * 3,
    })
    summary = performance_summary(weekly, 52)
    assert summary["maximum_drawdown"] < 0
    assert summary["mean_turnover"] == 2 / 3


def test_liquidity_screen_is_applied_before_top_n_selection() -> None:
    predictions = pd.DataFrame([
        {"instrument_id": "illiquid", "asof_date": "2024-01-05", "available_date": "2024-01-08",
         "score": 10, "target": 1.0, "amihud_percentile": 1.0, "turnover_percentile": 0.0},
        {"instrument_id": "liquid", "asof_date": "2024-01-05", "available_date": "2024-01-08",
         "score": 1, "target": 0.1, "amihud_percentile": 0.5, "turnover_percentile": 0.5},
    ])
    result = simulate_long_only(predictions, 1, "target", 0, 1.0, 0.9, 0.1)
    assert result.loc[0, "gross_return"] == 0.1
    assert result.loc[0, "candidate_pool_size"] == 1
