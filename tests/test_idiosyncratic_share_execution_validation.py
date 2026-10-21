from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.idiosyncratic_share_execution_validation import (
    _paired_metrics,
    exposure_execution_audit,
    load_config,
    validate_config,
)


def test_frozen_execution_config_is_validation_only() -> None:
    cfg = load_config()
    validate_config(cfg)
    assert cfg["period"]["end_date"] == "2020-12-31"
    assert cfg["round_trip_cost_bps"] == 20.0
    assert cfg["historical_oos_evaluation_enabled"] is False
    assert cfg["future_lockbox_access_allowed"] is False
    assert len(cfg["paths"]) == 3


def test_paired_metrics_use_reliable_excess_returns_for_sharpe() -> None:
    candidate = pd.DataFrame({
        "net_return": [0.01, -0.005, 0.012, 0.004],
        "excess_return": [0.009, -0.006, 0.011, 0.003],
        "one_way_turnover": [0.1] * 4,
        "target_tracking_l1": [0.02] * 4,
        "cash_weight": [0.01] * 4,
    })
    reference = pd.DataFrame({
        "net_return": [0.004, -0.006, 0.005, 0.001],
        "excess_return": [0.003, -0.007, 0.004, 0.0],
        "one_way_turnover": [0.08] * 4,
        "target_tracking_l1": [0.03] * 4,
        "cash_weight": [0.02] * 4,
    })
    result = _paired_metrics(candidate, reference, 252)
    expected = (candidate["net_return"] - reference["net_return"]).mean() * 252
    assert result["incremental_net_return_annualized_arithmetic"] == pytest.approx(expected)
    assert result["sharpe_reliable_rf_improvement"] > 0
    assert result["target_tracking_l1_improvement"] == pytest.approx(0.01)
    assert result["cash_weight_improvement"] == pytest.approx(0.01)


def test_execution_exposure_audit_weights_actual_shares_and_cash() -> None:
    cfg = load_config()
    targets = pd.DataFrame({
        "path_id": ["p", "p"], "execution_date": ["2020-01-02"] * 2,
        "signal_date": ["2019-12-31"] * 2, "instrument_id": ["a", "b"],
        "target_weight": [0.6, 0.4],
    })
    positions = pd.DataFrame({
        "path_id": ["p", "p"], "trade_date": ["2020-01-02"] * 2,
        "instrument_id": ["a", "b"], "actual_weight": [0.55, 0.40],
    })
    daily = pd.DataFrame({
        "path_id": ["p"], "trade_date": ["2020-01-02"],
        "active_target_date": ["2020-01-02"], "new_target": [True],
        "target_tracking_l1": [0.10],
    })
    exposure_map = pd.DataFrame({
        "signal_date": ["2019-12-31"] * 2, "instrument_id": ["a", "b"],
        "industry_broad_code": ["I1", "I2"],
        **{f"style_{style}": [1.0, -1.0] for style in cfg["style_exposure_audit"]["styles"]},
    })
    result = exposure_execution_audit(targets, positions, daily, exposure_map, cfg).iloc[0]
    assert result["actual_exposure_coverage_weight"] == pytest.approx(0.95)
    assert result["target_exposure_coverage_weight"] == pytest.approx(1.0)
    assert result["size_execution_gap"] == pytest.approx(-0.05)
    assert result["maximum_abs_industry_execution_gap"] == pytest.approx(0.05)
    assert np.isfinite(result["target_tracking_l1"])
