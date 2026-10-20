from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research.paper_trading as paper
from quant_research.institutional_execution import simulate_execution
from quant_research.paper_trading import (
    append_operation,
    construct_primary_target,
    load_config,
    score_signal_universe,
    validate_config,
    verify_operations_log,
)
from quant_research.preprocessing import load_config as load_preprocessing_config


def test_phase11_config_freezes_forward_only_boundaries() -> None:
    cfg = load_config()
    validate_config(cfg)
    changed = copy.deepcopy(cfg)
    changed["allow_retroactive_backfill"] = True
    with pytest.raises(ValueError, match="backfill"):
        validate_config(changed)
    changed = copy.deepcopy(cfg)
    changed["allow_interim_lockbox_performance_access"] = True
    with pytest.raises(ValueError, match="sealed"):
        validate_config(changed)


def test_operations_log_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    first = append_operation(path, {
        "occurred_at_utc": "2026-10-21T08:00:00+00:00",
        "trade_date": "2026-10-21", "status": "FAILED",
        "run_directory": "runs/a", "run_manifest_sha256": "a" * 64,
        "error": "missing data", "result_accessed": False,
    })
    second = append_operation(path, {
        "occurred_at_utc": "2026-10-22T08:00:00+00:00",
        "trade_date": "2026-10-22", "status": "SUCCESS",
        "run_directory": "runs/b", "run_manifest_sha256": "b" * 64,
        "error": None, "result_accessed": False,
    })
    assert second["previous_hash"] == first["event_hash"]
    assert len(verify_operations_log(path)) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["status"] = "SUCCESS"
    lines[0] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_operations_log(path)


class _FirstColumnEstimator:
    def predict(self, design: np.ndarray) -> np.ndarray:
        return design[:, 0]


def test_signal_scoring_uses_only_pit_cross_section() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["model"]["minimum_cross_section_size"] = 10
    factors = list(load_preprocessing_config()["factor_columns"])
    rows = 20
    frame = pd.DataFrame({
        "instrument_id": [f"sh.{600000 + index:06d}" for index in range(rows)],
        "asof_date": ["2026-10-23"] * rows,
        "available_date": ["2026-10-26"] * rows,
        "float_cap_proxy_cny": np.linspace(1e9, 3e9, rows),
        "industry_broad_code": ["A"] * 10 + ["B"] * 10,
    })
    for number, factor in enumerate(factors):
        frame[factor] = np.linspace(-1, 1, rows) + number / 100.0
    scored = score_signal_universe(frame, cfg, _FirstColumnEstimator())
    assert len(scored) == rows
    assert scored["score"].notna().all()
    assert set(scored["model_name"]) == {"hgb_shallow15_locked"}
    assert "fwd_return_5d_lag1" not in scored


def test_weekly_signal_gate_rejects_midweek_and_accepts_next_session(
    tmp_path: Path,
) -> None:
    from quant_research.ingest import Lake

    lake = Lake(tmp_path / "lake")
    days = pd.bdate_range("2026-10-12", "2026-10-30")
    calendar = pd.DataFrame({
        "cal_date": days.strftime("%Y-%m-%d"), "is_open": "1", "exchange": "SSE",
    })
    lake.write("calendar", calendar, key="SSE", source="test", request={})
    assert paper._signal_asof_date(lake, "2026-10-21") is None
    assert paper._signal_asof_date(lake, "2026-10-26") == "2026-10-23"


def test_primary_target_is_long_only_fully_invested_and_audited() -> None:
    cfg = load_config()
    rows = 500
    signal = pd.DataFrame({
        "instrument_id": [f"sz.{index:06d}" for index in range(rows)],
        "score": np.linspace(1, 0, rows),
        "asof_date": ["2026-10-23"] * rows,
        "available_date": ["2026-10-26"] * rows,
        "industry_broad_code": [f"I{index % 5}" for index in range(rows)],
        "size_z": np.linspace(-1, 1, rows),
        "volatility_20d": np.linspace(0.01, 0.05, rows),
    })
    target, audit = construct_primary_target(
        signal, cfg, pd.DataFrame(), pd.DataFrame()
    )
    assert np.isclose(target["target_weight"].sum(), 1.0)
    assert target["target_weight"].ge(0).all()
    assert target["target_weight"].max() <= 0.01 + 1e-12
    assert audit["constraint_relaxed"] is False
    assert audit["new_target"] is True


def test_paper_replay_keeps_open_orders_alive_at_current_boundary() -> None:
    market = pd.DataFrame([{
        "trade_date": "2026-10-21", "instrument_id": "sh.600000",
        "close": 10.0, "volume": 100.0, "trade_status": "tradable",
        "buy_allowed": True, "sell_allowed": True, "share_factor": 1.0,
        "cash_dividend_per_share": 0.0,
    }])
    targets = pd.DataFrame([{
        "execution_date": "2026-10-21", "instrument_id": "sh.600000",
        "target_weight": 1.0,
    }])
    config = {
        "initial_cash_cny": 1_000_000.0, "round_trip_cost_bps": 20.0,
        "max_volume_participation": 0.10, "max_pending_trading_days": 5,
        "order_quantity_rules": {
            "main_board_buy_lot": 100, "main_board_sell_lot": 100,
            "star_market_minimum_order": 200, "star_market_increment": 1,
            "odd_lot_liquidation_allowed": True,
        },
    }
    scenario = {
        "scenario": "paper", "enforce_trade_flags": True,
        "enforce_volume_participation": True,
        "enforce_order_quantity_rules": True, "enforce_t_plus_one": True,
    }
    result = simulate_execution(
        market, targets, config, scenario=scenario, expire_open_orders_at_end=False
    )
    assert result.orders.iloc[0]["status"] == "BLOCKED"
    assert not result.events["status"].eq("EXPIRED").any()


def test_run_publication_is_immutable_and_hash_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    cfg = copy.deepcopy(load_config())
    cfg["storage"]["run_directory"] = "paper/runs"
    cfg["storage"]["operations_log"] = "paper/operations.jsonl"
    frame = pd.DataFrame([{"check": "cash_nonnegative", "passed": True}])
    path, manifest_hash = paper._publish_run(
        cfg, "2026-10-21", "SUCCESS", "2026-10-21T08:00:00+00:00",
        {"health_checks": frame}, {"performance_claim_allowed": False},
    )
    assert path.exists()
    assert len(manifest_hash) == 64
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS"
    assert manifest["performance_claim_allowed"] is False
    assert manifest["artifacts"]["health_checks.parquet"] == paper.sha256_file(
        path / "health_checks.parquet"
    )
