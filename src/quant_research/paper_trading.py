"""Forward-only paper trading and future-lockbox operational controls.

The module deliberately separates observable operational health from sealed
future-lockbox performance.  It never backfills a successful paper observation,
and every attempted date is published as an immutable, hash-addressed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml

from quant_research.execution_market import read_execution_market_rows
from quant_research.experiment_registry import sha256_file, source_tree_hash, stable_json_hash
from quant_research.factor_neutralization import _read_year
from quant_research.ingest import ROOT, Lake
from quant_research.institutional_execution import simulate_execution
from quant_research.model_baseline import purged_end_date
from quant_research.model_challenger import concatenate_samples
from quant_research.model_portfolio_neutralization import (
    build_year_training,
    fit_models,
    model_designs,
)
from quant_research.portfolio_construction_comparison import (
    carry_forward_fallback,
    common_buffer_candidates,
    optimizer_candidates,
    risk_diagnostics,
    solve_constrained_weights,
)
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.research_governance import append_access_event


CONFIG_PATH = ROOT / "config" / "paper_trading.yaml"
REPORT_PATH = ROOT / "reports" / "paper_trading.md"
GENESIS_HASH = "0" * 64
TERMINAL_ATTEMPT_STATUSES = {
    "SUCCESS", "FAILED", "NOT_ACTIVE", "NON_TRADING_DAY", "REJECTED_BACKFILL",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg["evidence_class"] != "governance_only":
        raise ValueError("an empty Phase 11 pipeline is governance evidence, not performance evidence")
    if cfg.get("claims_independent_evidence") is not False:
        raise ValueError("the pipeline cannot claim independent evidence before observations exist")
    if cfg.get("allow_retroactive_backfill") is not False:
        raise ValueError("paper trading cannot allow retroactive backfill")
    if cfg.get("allow_interim_lockbox_performance_access") is not False:
        raise ValueError("interim future-lockbox performance must remain sealed")
    if cfg["model"]["pipeline"] != "raw" or cfg["model"]["name"] != "hgb_shallow15_locked":
        raise ValueError("Phase 11 must retain the frozen primary candidate")
    if cfg["portfolio"]["method"] != "P4_constrained":
        raise ValueError("Phase 11 must retain the frozen risk-constrained portfolio")
    if cfg["portfolio"].get("allow_constraint_relaxation") is not False:
        raise ValueError("constraint relaxation cannot be enabled silently")
    if float(cfg["execution"]["round_trip_cost_bps"]) != 20.0:
        raise ValueError("Phase 11 primary cost must remain the frozen 20 bp scenario")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / str(cfg["storage"][key])


def _git_state() -> dict[str, Any]:
    def call(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=30,
        )
        return completed.stdout.strip()

    return {"commit": call("rev-parse", "HEAD"), "dirty": bool(call("status", "--porcelain"))}


def _model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg["model"]
    return {
        **model,
        "pipelines": ["raw", "size_neutralized", "industry_neutralized", "industry_size_neutralized"],
        "models": {str(model["name"]): dict(model["parameters"])},
    }


def _dependency_rows(lake: Lake, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    start = int(cfg["model"]["train_start_year"])
    end = pd.Timestamp(cfg["model"]["training_data_cutoff"]).year
    rows: list[dict[str, Any]] = []
    for year in range(start, end + 1):
        for table in (
            cfg["signal"]["factor_table"], cfg["signal"]["industry_table"],
            cfg["signal"]["size_table"],
        ):
            path = lake.table_path(str(table), str(year))
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append({
                "table": str(table), "key": str(year), "path": _relative(path),
                "sha256": sha256_file(path),
            })
    calendar = lake.table_path("calendar", "SSE")
    if not calendar.exists():
        raise FileNotFoundError(calendar)
    rows.append({
        "table": "calendar", "key": "SSE", "path": _relative(calendar),
        "sha256": sha256_file(calendar),
    })
    return rows


def freeze_model_bundle(lake: Lake | None = None) -> dict[str, Any]:
    """Fit and immutably serialize the already selected primary candidate."""
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("model freeze requires a clean Git working tree")
    bundle = _path(cfg, "model_bundle_directory")
    if bundle.exists():
        raise FileExistsError(f"model bundle already frozen: {bundle}")

    factors = list(load_preprocessing_config()["factor_columns"])
    model_cfg = _model_config(cfg)
    target = str(cfg["model"]["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start = int(cfg["model"]["train_start_year"])
    final_year = pd.Timestamp(cfg["model"]["training_data_cutoff"]).year
    horizon = int(cfg["model"]["label_horizon_trading_days"])
    full_samples = []
    final_purged_sample = None
    training_dates = 0
    for year in range(start, final_year + 1):
        frame = _read_year(lake, year, columns)
        boundary = min(
            pd.Timestamp(f"{year}-12-31"),
            pd.Timestamp(cfg["model"]["training_data_cutoff"]),
        ).strftime("%Y-%m-%d")
        cutoff = purged_end_date(calendar, boundary, horizon)
        _, _, samples, purged_samples = build_year_training(
            frame, factors, target, cutoff, model_cfg
        )
        if year < final_year:
            full_samples.append(samples["raw"])
            training_dates += samples["raw"].dates
        else:
            final_purged_sample = purged_samples["raw"]
            training_dates += purged_samples["raw"].dates
    if final_purged_sample is None:
        raise RuntimeError("final purged training sample was not built")
    sample = concatenate_samples([*full_samples, final_purged_sample])
    # HGB ignores the sufficient-statistics object. Supply a shape-correct
    # placeholder so the shared frozen fit API remains the sole implementation.
    dimension = sample.x.shape[1]
    from quant_research.model_baseline import SufficientStatistics
    placeholder = SufficientStatistics(
        np.zeros((dimension, dimension)), np.zeros(dimension), len(sample.y), training_dates
    )
    fitted = fit_models(placeholder, sample, model_cfg)[str(cfg["model"]["name"])]

    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate_bundle_", dir=bundle.parent))
    try:
        model_path = temporary / "model.joblib"
        joblib.dump(fitted.estimator, model_path, compress=3)
        dependencies = _dependency_rows(lake, cfg)
        manifest = {
            "schema_version": 1,
            "bundle_id": cfg["candidate_bundle_id"],
            "paper_id": cfg["paper_id"],
            "model_name": cfg["model"]["name"],
            "model_pipeline": cfg["model"]["pipeline"],
            "model_parameters": cfg["model"]["parameters"],
            "feature_columns": factors,
            "design_dimension": int(dimension),
            "fit_observations": int(len(sample.y)),
            "fit_dates": int(training_dates),
            "training_data_cutoff": cfg["model"]["training_data_cutoff"],
            "label_purge_days": horizon,
            "random_state": int(cfg["model"]["random_state"]),
            "freeze_commit": git["commit"],
            "source_tree_sha256": source_tree_hash(ROOT),
            "config_sha256": sha256_file(CONFIG_PATH),
            "dependency_snapshot_sha256": stable_json_hash(dependencies),
            "dependencies": dependencies,
            "model_file": "model.joblib",
            "model_sha256": sha256_file(model_path),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable": True,
            "performance_results_included": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(bundle)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    public_path = _path(cfg, "public_model_manifest")
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_model_bundle(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    bundle = _path(cfg, "model_bundle_directory")
    manifest_path = bundle / "manifest.json"
    model_path = bundle / "model.joblib"
    if not manifest_path.exists() or not model_path.exists():
        raise FileNotFoundError("frozen paper model bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("immutable"):
        raise ValueError("paper model manifest is not immutable")
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("paper model artifact hash mismatch")
    if manifest["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("paper model config changed after freeze")
    return manifest


def _event_payload(event: dict[str, Any]) -> bytes:
    content = {key: value for key, value in event.items() if key != "event_hash"}
    return json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def verify_operations_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event.get("sequence", -1)) != len(events) + 1:
                raise ValueError(f"non-sequential paper log event at line {number}")
            if event.get("previous_hash") != previous:
                raise ValueError(f"broken paper log hash chain at line {number}")
            actual = hashlib.sha256(_event_payload(event)).hexdigest()
            if event.get("event_hash") != actual:
                raise ValueError(f"paper log hash mismatch at line {number}")
            events.append(event)
            previous = actual
    return events


def append_operation(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    events = verify_operations_log(path)
    payload = {
        **event,
        "sequence": len(events) + 1,
        "previous_hash": events[-1]["event_hash"] if events else GENESIS_HASH,
    }
    payload["event_hash"] = hashlib.sha256(_event_payload(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _successful_run_paths(cfg: dict[str, Any]) -> list[Path]:
    events = verify_operations_log(_path(cfg, "operations_log"))
    return [ROOT / event["run_directory"] for event in events if event["status"] == "SUCCESS"]


def _latest_state(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = _successful_run_paths(cfg)
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    target_path = runs[-1] / "target_state.parquet"
    position_path = runs[-1] / "positions.parquet"
    target = pd.read_parquet(target_path) if target_path.exists() else pd.DataFrame()
    positions = pd.read_parquet(position_path) if position_path.exists() else pd.DataFrame()
    return target, positions


def _load_signal_universe(
    lake: Lake, cfg: dict[str, Any], trade_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    factors = list(load_preprocessing_config()["factor_columns"])
    year = str(pd.Timestamp(trade_date).year)
    source_paths = {
        "factor": lake.table_path(str(cfg["signal"]["factor_table"]), year),
        "industry": lake.table_path(str(cfg["signal"]["industry_table"]), year),
        "size": lake.table_path(str(cfg["signal"]["size_table"]), year),
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    panel = pd.read_parquet(
        source_paths["factor"],
        columns=["instrument_id", "asof_date", "available_date", *factors],
    )
    panel = panel.loc[panel["available_date"].astype(str).eq(trade_date)].copy()
    if panel.empty:
        return panel, [
            {"source": name, "path": _relative(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        ]
    asof_dates = panel["asof_date"].astype(str).unique()
    if len(asof_dates) != 1:
        raise RuntimeError(f"signal packet has non-unique asof dates: {asof_dates.tolist()}")
    asof = str(asof_dates[0])
    size = pd.read_parquet(
        source_paths["size"], columns=["instrument_id", "asof_date", "float_cap_proxy_cny"]
    )
    industry = pd.read_parquet(
        source_paths["industry"], columns=["instrument_id", "asof_date", "industry_broad_code"]
    )
    size = size.loc[size["asof_date"].astype(str).eq(asof)]
    industry = industry.loc[industry["asof_date"].astype(str).eq(asof)]
    result = panel.merge(
        size, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    ).merge(
        industry, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )
    dependencies = [
        {"source": name, "path": _relative(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    ]
    return result.sort_values("instrument_id").reset_index(drop=True), dependencies


def score_signal_universe(
    universe: pd.DataFrame, cfg: dict[str, Any], estimator: Any,
) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()
    factors = list(load_preprocessing_config()["factor_columns"])
    designs, risks, complete = model_designs(universe, factors, _model_config(cfg))
    scored = universe.loc[complete].reset_index(drop=True).copy()
    if len(scored) < int(cfg["model"]["minimum_cross_section_size"]):
        raise RuntimeError(f"signal universe too small after PIT exposure filter: {len(scored)}")
    scored["size_z"] = risks["size_z"][complete]
    scored["score"] = np.asarray(estimator.predict(designs["raw"][complete]), dtype=float)
    if not np.isfinite(scored["score"]).all():
        raise ValueError("paper signal contains non-finite model scores")
    scored["model_name"] = cfg["model"]["name"]
    scored["model_pipeline"] = cfg["model"]["pipeline"]
    return scored


def _portfolio_config(cfg: dict[str, Any]) -> dict[str, Any]:
    portfolio = cfg["portfolio"]
    return {
        "buffer": {"entry_rank": portfolio["entry_rank"], "exit_rank": portfolio["exit_rank"]},
        "weighting": {"maximum_stock_weight": portfolio["maximum_stock_weight"]},
        "optimizer": {
            "industry_active_weight_bound": portfolio["industry_active_weight_bound"],
            "absolute_size_z_bound": portfolio["absolute_size_z_bound"],
            "one_way_turnover_limit": portfolio["one_way_turnover_limit"],
            "industry_capacity_buffer_names": portfolio["industry_capacity_buffer_names"],
            "size_tail_candidates_each_side": portfolio["size_tail_candidates_each_side"],
        },
    }


def construct_primary_target(
    signal: pd.DataFrame, cfg: dict[str, Any], previous_target: pd.DataFrame,
    previous_positions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if signal.empty:
        return previous_target.copy(), {"new_target": False, "reason": "no_signal_due"}
    p_cfg = _portfolio_config(cfg)
    prior_candidates = set()
    if not previous_target.empty:
        ordinary = previous_target.get(
            "constraint_support", pd.Series(False, index=previous_target.index)
        ).astype(bool)
        prior_candidates = set(previous_target.loc[~ordinary, "instrument_id"].astype(str))
    base, buffer_audit = common_buffer_candidates(
        signal, prior_candidates, int(cfg["portfolio"]["entry_rank"]),
        int(cfg["portfolio"]["exit_rank"]),
    )
    previous_weights: dict[str, float] = {}
    if not previous_positions.empty:
        previous_weights = dict(zip(
            previous_positions["instrument_id"].astype(str),
            previous_positions["actual_weight"].astype(float),
        ))
    candidate = optimizer_candidates(signal, base, previous_weights, p_cfg)
    weights, optimizer = solve_constrained_weights(signal, candidate, previous_weights, p_cfg)
    fallback = weights is None
    if fallback:
        weights = carry_forward_fallback(candidate, previous_weights, p_cfg)
    risk = risk_diagnostics(signal, candidate, weights)
    result = candidate[[
        "instrument_id", "score", "score_rank", "entry_eligible", "buffer_retained",
        "industry_broad_code", "size_z", "volatility_20d",
    ]].copy()
    result["constraint_support"] = candidate.get(
        "constraint_support", pd.Series(False, index=candidate.index)
    ).to_numpy(bool)
    result["target_weight"] = weights
    result["signal_date"] = str(signal["asof_date"].iloc[0])
    result["execution_date"] = str(signal["available_date"].iloc[0])
    result["method"] = cfg["portfolio"]["method"]
    audit = {
        "new_target": True, "fallback_used": bool(fallback), **buffer_audit, **optimizer, **risk,
        "weight_sum": float(weights.sum()), "minimum_weight": float(weights.min()),
        "constraint_relaxed": False,
    }
    return result.sort_values("instrument_id").reset_index(drop=True), audit


def _execution_config(cfg: dict[str, Any]) -> dict[str, Any]:
    execution = cfg["execution"]
    return {
        "initial_cash_cny": execution["initial_cash_cny"],
        "round_trip_cost_bps": execution["round_trip_cost_bps"],
        "max_volume_participation": execution["max_volume_participation"],
        "max_pending_trading_days": execution["max_pending_trading_days"],
        "order_quantity_rules": execution["order_quantity_rules"],
    }


def _scenario(cfg: dict[str, Any]) -> dict[str, Any]:
    execution = cfg["execution"]
    return {
        "scenario": "paper_candidate_bundle_v1",
        "enforce_trade_flags": execution["enforce_trade_flags"],
        "enforce_volume_participation": execution["enforce_volume_participation"],
        "enforce_order_quantity_rules": execution["enforce_order_quantity_rules"],
        "enforce_t_plus_one": execution["enforce_t_plus_one"],
    }


def _market_snapshot(
    lake: Lake, cfg: dict[str, Any], trade_date: str, codes: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    with (ROOT / "config" / "constrained_portfolio_daily_execution.yaml").open(
        encoding="utf-8"
    ) as handle:
        rules = yaml.safe_load(handle)["rules"]
    frames = []
    dependencies = []
    for code in sorted(codes):
        path = lake.table_path("daily_bars", code)
        if path.exists():
            dependencies.append({
                "source": "daily_bars", "instrument_id": code,
                "path": _relative(path), "sha256": sha256_file(path),
            })
        rows = read_execution_market_rows(
            lake, code, rules=rules,
            corporate_action_threshold=float(
                cfg["execution"]["corporate_action_proxy"]["activation_absolute_deviation"]
            ),
            start_date=trade_date, end_date=trade_date,
        )
        if not rows.empty:
            frames.append(rows)
    if not frames:
        raise RuntimeError(f"no execution-market observations for {trade_date}")
    market = pd.concat(frames, ignore_index=True).sort_values("instrument_id").reset_index(drop=True)
    return market, dependencies


def _replay(
    cfg: dict[str, Any], current_market: pd.DataFrame, current_target: pd.DataFrame,
    new_target: bool,
) -> Any:
    market_frames = []
    target_frames = []
    for run in _successful_run_paths(cfg):
        market_frames.append(pd.read_parquet(run / "market.parquet"))
        submitted = run / "submitted_target.parquet"
        if submitted.exists():
            target_frames.append(pd.read_parquet(submitted))
    market_frames.append(current_market)
    if new_target:
        target_frames.append(current_target)
    if not target_frames:
        raise RuntimeError("paper account cannot start before its first frozen signal target")
    market = pd.concat(market_frames, ignore_index=True).drop_duplicates(
        ["trade_date", "instrument_id"], keep="last"
    )
    targets = pd.concat(target_frames, ignore_index=True)[
        ["execution_date", "instrument_id", "target_weight"]
    ].drop_duplicates(["execution_date", "instrument_id"], keep="last")
    return simulate_execution(
        market, targets, _execution_config(cfg), scenario=_scenario(cfg),
        expire_open_orders_at_end=False,
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _publish_run(
    cfg: dict[str, Any], trade_date: str, status: str, started: str,
    frames: dict[str, pd.DataFrame], metadata: dict[str, Any],
) -> tuple[Path, str]:
    runs = _path(cfg, "run_directory")
    runs.mkdir(parents=True, exist_ok=True)
    prior = verify_operations_log(_path(cfg, "operations_log"))
    attempt = 1 + sum(str(event.get("trade_date")) == trade_date for event in prior)
    stamp = started.replace("-", "").replace(":", "").replace("+00:00", "Z").replace(".", "")
    run_id = f"{stamp}__{trade_date.replace('-', '')}__a{attempt:02d}"
    target = runs / run_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}_", dir=runs))
    try:
        for name, frame in frames.items():
            _write_frame(temporary / f"{name}.parquet", frame)
        artifact_hashes = {
            path.name: sha256_file(path) for path in sorted(temporary.glob("*.parquet"))
        }
        run_manifest = {
            "schema_version": 1, "run_id": run_id, "paper_id": cfg["paper_id"],
            "lockbox_id": cfg["lockbox_id"], "trade_date": trade_date,
            "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "artifacts": artifact_hashes, **metadata,
        }
        manifest_path = temporary / "run.json"
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        manifest_hash = sha256_file(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
        return target, manifest_hash
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _attempted_dates(cfg: dict[str, Any]) -> set[str]:
    return {
        str(event["trade_date"]) for event in verify_operations_log(_path(cfg, "operations_log"))
        if event.get("status") in TERMINAL_ATTEMPT_STATUSES
    }


def _assert_no_silent_dates(lake: Lake, cfg: dict[str, Any], trade_date: str) -> None:
    calendar = lake.read("calendar", "SSE")
    if calendar.empty:
        raise RuntimeError("SSE calendar is unavailable")
    cal = calendar.copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"]).dt.strftime("%Y-%m-%d")
    required = set(cal.loc[
        cal["is_open"].astype(str).eq("1")
        & cal["cal_date"].ge(str(cfg["activation_not_before"]))
        & cal["cal_date"].lt(trade_date), "cal_date"
    ])
    missing = sorted(required - _attempted_dates(cfg))
    if missing:
        raise RuntimeError(f"unrecorded prior trading dates: {missing[:10]}")


def _is_open_date(lake: Lake, trade_date: str) -> bool:
    calendar = lake.read("calendar", "SSE")
    if calendar.empty:
        raise RuntimeError("SSE calendar is unavailable")
    dates = pd.to_datetime(calendar["cal_date"]).dt.strftime("%Y-%m-%d")
    selected = calendar.loc[dates.eq(trade_date), "is_open"]
    return bool(not selected.empty and selected.astype(str).eq("1").any())


def _record_terminal(
    cfg: dict[str, Any], trade_date: str, status: str, started: str,
    *, error: str | None = None, frames: dict[str, pd.DataFrame] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run, manifest_hash = _publish_run(
        cfg, trade_date, status, started, frames or {},
        {**(metadata or {}), "error": error},
    )
    event = append_operation(_path(cfg, "operations_log"), {
        "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        "trade_date": trade_date, "status": status,
        "run_directory": _relative(run), "run_manifest_sha256": manifest_hash,
        "error": error, "result_accessed": False,
    })
    return event


def run_daily(
    trade_date: str | None = None, *, lake: Lake | None = None,
    now: datetime | None = None, estimator_loader: Callable[[Path], Any] = joblib.load,
) -> dict[str, Any]:
    """Run one real-time date. Tests may inject ``now`` and an estimator loader."""
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    current = now or datetime.now(ZoneInfo(str(cfg["timezone"])))
    local_date = current.astimezone(ZoneInfo(str(cfg["timezone"]))).date().isoformat()
    requested = trade_date or local_date
    started = datetime.now(timezone.utc).isoformat()
    if requested != local_date:
        return _record_terminal(
            cfg, requested, "REJECTED_BACKFILL", started,
            error=f"requested_date={requested}; local_date={local_date}",
        )
    if requested < str(cfg["activation_not_before"]):
        return _record_terminal(cfg, requested, "NOT_ACTIVE", started)
    if not _is_open_date(lake, requested):
        return _record_terminal(cfg, requested, "NON_TRADING_DAY", started)

    try:
        _assert_no_silent_dates(lake, cfg, requested)
        manifest = verify_model_bundle(cfg)
        estimator = estimator_loader(_path(cfg, "model_bundle_directory") / "model.joblib")
        universe, signal_dependencies = _load_signal_universe(lake, cfg, requested)
        signal = score_signal_universe(universe, cfg, estimator)
        previous_target, previous_positions = _latest_state(cfg)
        target, target_audit = construct_primary_target(
            signal, cfg, previous_target, previous_positions
        )
        new_target = bool(target_audit["new_target"])
        active_target = target if new_target else previous_target
        codes = set(active_target.get("instrument_id", pd.Series(dtype=str)).astype(str))
        codes.update(previous_positions.get("instrument_id", pd.Series(dtype=str)).astype(str))
        if not codes:
            raise RuntimeError("no signal target or existing account positions are available")
        market, market_dependencies = _market_snapshot(lake, cfg, requested, codes)
        result = _replay(cfg, market, target, new_target)
        daily = result.daily.loc[result.daily["trade_date"].eq(requested)].copy()
        if len(daily) != 1:
            raise RuntimeError(f"paper account produced {len(daily)} daily rows for {requested}")
        positions = result.positions.loc[result.positions["trade_date"].eq(requested)].copy()
        positions = positions.rename(columns={"shares": "actual_shares"})
        events = result.events.loc[result.events["trade_date"].eq(requested)].copy()
        fills = result.fills.loc[result.fills.get("trade_date", pd.Series(dtype=str)).eq(requested)].copy()
        nav_error = float(daily["nav_reconciliation_error"].abs().max())
        minimum_cash = float(daily["cash"].min())
        checks = pd.DataFrame([
            {"check": "nav_reconciles", "passed": nav_error <= float(
                cfg["health_thresholds"]["maximum_nav_reconciliation_error_cny"]
            ), "observed": nav_error},
            {"check": "cash_nonnegative", "passed": minimum_cash >= float(
                cfg["health_thresholds"]["minimum_cash_cny"]
            ), "observed": minimum_cash},
            {"check": "target_weights_sum_to_one", "passed": bool(
                np.isclose(active_target["target_weight"].sum(), 1.0, atol=1e-8)
            ), "observed": float(active_target["target_weight"].sum())},
            {"check": "no_constraint_relaxation", "passed": not bool(
                target_audit.get("constraint_relaxed", False)
            ), "observed": bool(target_audit.get("constraint_relaxed", False))},
        ])
        if not checks["passed"].all():
            raise RuntimeError("paper account health checks failed")
        frames = {
            "signal_input": universe,
            "signals": signal,
            "market": market,
            "target_state": active_target,
            "account": daily,
            "positions": positions,
            "orders": result.orders,
            "order_events": events,
            "fills": fills,
            "health_checks": checks,
        }
        if new_target:
            frames["submitted_target"] = target
        metadata = {
            "model_bundle_sha256": manifest["model_sha256"],
            "model_freeze_commit": manifest["freeze_commit"],
            "config_sha256": sha256_file(CONFIG_PATH),
            "input_dependencies": [*signal_dependencies, *market_dependencies],
            "input_dependency_snapshot_sha256": stable_json_hash(
                [*signal_dependencies, *market_dependencies]
            ),
            "new_target": new_target,
            "target_audit": target_audit,
            "performance_claim_allowed": False,
        }
        event = _record_terminal(
            cfg, requested, "SUCCESS", started, frames=frames, metadata=metadata
        )
        governance_cfg = ROOT / "config" / "research_governance.yaml"
        with governance_cfg.open(encoding="utf-8") as handle:
            governance = yaml.safe_load(handle)
        access_log = ROOT / str(governance["access_log"]["path"])
        prior_access = verify_operations_log(_path(cfg, "operations_log"))
        if sum(item["status"] == "SUCCESS" for item in prior_access) == 1:
            append_access_event(
                access_log, actor="paper_trading_pipeline", evidence_class="future_lockbox",
                action="LOCKBOX_DATA_ARRIVED", reason="first forward-only daily input captured",
                result_accessed=False, used_for_change=False, lockbox_id=cfg["lockbox_id"],
            )
        append_access_event(
            access_log, actor="paper_trading_pipeline", evidence_class="paper_trading",
            action="PAPER_OBSERVATION_RECORDED", reason=f"immutable daily run {requested}",
            result_accessed=False, used_for_change=False,
        )
        return event
    except Exception as exc:
        return _record_terminal(cfg, requested, "FAILED", started, error=str(exc))


def build_protocol(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    events = verify_operations_log(_path(cfg, "operations_log"))
    successes = [event for event in events if event["status"] == "SUCCESS"]
    bundle_ok = False
    bundle_detail = "not_frozen"
    try:
        manifest = verify_model_bundle(cfg)
        bundle_ok = True
        bundle_detail = str(manifest["model_sha256"])
    except (FileNotFoundError, ValueError) as exc:
        bundle_detail = str(exc)
    source_contracts = pd.DataFrame([
        {"artifact": "signal_input", "required": True,
         "contract": "PIT factors plus same-date industry and size; available_date equals run date"},
        {"artifact": "model_bundle", "required": True,
         "contract": "immutable HGB artifact, config hash, source hash, dependency hashes"},
        {"artifact": "target", "required": True,
         "contract": "P4 constraints frozen; fallback explicit; no silent relaxation"},
        {"artifact": "account", "required": True,
         "contract": "orders, fills, cash, positions and NAV reconcile daily"},
        {"artifact": "operations_log", "required": True,
         "contract": "append-only SHA-256 chain retains failed and missing-data dates"},
    ])
    checks = pd.DataFrame([
        {"check": "config_valid", "status": "PASS", "detail": cfg["experiment_id"]},
        {"check": "model_bundle_frozen", "status": "PASS" if bundle_ok else "FAIL",
         "detail": bundle_detail},
        {"check": "operations_log_hash_chain_valid", "status": "PASS",
         "detail": f"events={len(events)}"},
        {"check": "retroactive_backfill_disabled", "status": "PASS",
         "detail": str(cfg["allow_retroactive_backfill"])},
        {"check": "interim_lockbox_performance_sealed", "status": "PASS",
         "detail": str(cfg["allow_interim_lockbox_performance_access"])},
        {"check": "failed_runs_retained", "status": "PASS",
         "detail": f"failed={sum(event['status'] == 'FAILED' for event in events)}"},
        {"check": "no_false_forward_claim", "status": "PASS",
         "detail": f"successful_observations={len(successes)}"},
    ])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "paper_id": cfg["paper_id"],
        "lockbox_id": cfg["lockbox_id"], "candidate_bundle_id": cfg["candidate_bundle_id"],
        "activation_not_before": cfg["activation_not_before"],
        "model": cfg["model"]["name"], "pipeline": cfg["model"]["pipeline"],
        "portfolio": cfg["portfolio"]["method"],
        "round_trip_cost_bps": cfg["execution"]["round_trip_cost_bps"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": False,
        "successful_forward_observations": len(successes),
        "ready_for_live_capital": False,
    }])
    status = pd.DataFrame([{
        "paper_id": cfg["paper_id"],
        "state": "COLLECTING" if successes else "FROZEN_NOT_STARTED",
        "attempts": len(events), "successful_observations": len(successes),
        "failed_attempts": sum(event["status"] == "FAILED" for event in events),
        "last_attempt_date": events[-1]["trade_date"] if events else None,
        "last_event_hash": events[-1]["event_hash"] if events else None,
        "performance_validated": False,
    }])
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "detail": "all Phase 11 implementation gates passed",
        }])
    outputs = {
        "paper_trading_protocol": protocol,
        "paper_trading_source_contracts": source_contracts,
        "paper_trading_status": status,
        "paper_trading_checks": checks,
        "paper_trading_failed_assertions": failures,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="paper_trading.v1", request=cfg)
    lines = [
        "# Phase 11：Paper Trading 与 Future Lockbox", "", "## 当前结论", "",
        f"管线已冻结；成功前瞻观测为 **{len(successes)}**。在满足预设样本门槛前，"
        "不宣称已经得到前瞻验证或实盘验证。Future lockbox 的中途绩效仍封存。", "",
        "## 冻结协议", "", protocol.to_markdown(index=False), "",
        "## 输入与输出契约", "", source_contracts.to_markdown(index=False), "",
        "## 运营状态", "", status.to_markdown(index=False), "",
        "## 机器验收", "", checks.to_markdown(index=False), "",
        "失败运行、缺失数据和非交易日尝试均进入不可变运行目录与 SHA-256 链；成功观测只允许在"
        "激活日之后按真实日期产生，不能历史回填。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Phase 11 implementation gates failed")
    return checks


def status() -> dict[str, Any]:
    cfg = load_config()
    events = verify_operations_log(_path(cfg, "operations_log"))
    successes = [event for event in events if event["status"] == "SUCCESS"]
    return {
        "paper_id": cfg["paper_id"], "activation_not_before": cfg["activation_not_before"],
        "attempts": len(events), "successful_observations": len(successes),
        "last_status": events[-1]["status"] if events else "FROZEN_NOT_STARTED",
        "operations_log_valid": True, "performance_validated": False,
        "future_lockbox_interim_performance_sealed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only paper trading pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-model")
    daily = sub.add_parser("run-daily")
    daily.add_argument("--date")
    sub.add_parser("build-protocol")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "freeze-model":
        print(json.dumps(freeze_model_bundle(), ensure_ascii=False, indent=2))
    elif args.command == "run-daily":
        print(json.dumps(run_daily(args.date), ensure_ascii=False, indent=2))
    elif args.command == "build-protocol":
        print(build_protocol().to_string(index=False))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
