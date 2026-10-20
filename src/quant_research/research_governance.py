"""Phase 4 evidence governance, candidate freeze, and hash-chained access log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_research.experiment_registry import sha256_file, source_tree_hash, stable_json_hash
from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "research_governance.yaml"
REPORT_PATH = ROOT / "reports" / "research_governance.md"
TABLE_ROOT = ROOT / "reports" / "tables"
GENESIS_HASH = "0" * 64
CONSUME_ACTIONS = {"VIEW_RESULTS", "EXPORT_RESULTS", "USE_FOR_CHANGE"}
EVIDENCE_CLASSES = {
    "training", "validation_selection", "historical_oos",
    "future_lockbox", "paper_trading", "governance_only",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    periods = cfg["evidence_periods"]
    training = periods["training"]
    validation = periods["validation_selection"]
    historical = periods["historical_oos"]
    lockbox = periods["future_lockbox"]
    if not (
        training["end_date"] < validation["start_date"]
        <= validation["end_date"] < historical["start_date"]
        <= historical["end_date"] < lockbox["start_date"]
    ):
        raise ValueError("evidence periods must be ordered and non-overlapping")
    if historical.get("consumed") is not True:
        raise ValueError("2021-2026 historical OOS must be marked consumed")
    if historical.get("selection_allowed_for_new_revisions") is not False:
        raise ValueError("historical OOS cannot select new revisions")
    if lockbox.get("already_observed") is not False:
        raise ValueError("future lockbox must be unobserved at freeze")
    if lockbox.get("interim_performance_access_allowed") is not False:
        raise ValueError("future lockbox cannot expose interim performance")
    if periods["paper_trading"].get("retroactive_backfill_allowed") is not False:
        raise ValueError("paper trading cannot be backfilled retroactively")
    if cfg.get("new_model_selected") is not False:
        raise ValueError("Phase 4 cannot select a new model")
    plan = cfg["statistical_plan"]
    if int(plan["familywise_strategy_count"]) != 1 + len(plan["secondary_hypotheses"]):
        raise ValueError("statistical hypothesis family count is inconsistent")
    if plan["multiplicity_control"] != "benjamini_hochberg":
        raise ValueError("strategy family must retain BH multiplicity control")


def _git_call(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=20,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_snapshot() -> dict[str, Any]:
    commit = _git_call("rev-parse", "HEAD")
    status = _git_call("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest() if status else None,
    }


def _event_payload(event: dict[str, Any]) -> bytes:
    content = {key: value for key, value in event.items() if key != "event_hash"}
    return json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def verify_access_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid access-log JSON at line {line_number}") from exc
            if int(event.get("sequence", -1)) != len(events) + 1:
                raise ValueError(f"non-sequential access-log event at line {line_number}")
            if event.get("previous_hash") != previous:
                raise ValueError(f"broken access-log hash chain at line {line_number}")
            actual = hashlib.sha256(_event_payload(event)).hexdigest()
            if event.get("event_hash") != actual:
                raise ValueError(f"access-log event hash mismatch at line {line_number}")
            events.append(event)
            previous = actual
    return events


def append_access_event(
    path: Path, *, actor: str, evidence_class: str, action: str, reason: str,
    result_accessed: bool, used_for_change: bool, lockbox_id: str | None = None,
    occurred_at_utc: str | None = None,
) -> dict[str, Any]:
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unsupported evidence class: {evidence_class}")
    if not actor.strip() or not reason.strip() or not action.strip():
        raise ValueError("actor, action, and reason are required")
    action = action.upper()
    if action in CONSUME_ACTIONS and not result_accessed:
        raise ValueError("result-access actions must set result_accessed=true")
    if used_for_change and not result_accessed:
        raise ValueError("a result cannot drive change unless it was accessed")
    events = verify_access_log(path)
    event = {
        "sequence": len(events) + 1,
        "occurred_at_utc": occurred_at_utc or datetime.now(timezone.utc).isoformat(),
        "actor": actor, "evidence_class": evidence_class, "action": action,
        "reason": reason, "result_accessed": bool(result_accessed),
        "used_for_change": bool(used_for_change), "lockbox_id": lockbox_id,
        "previous_hash": events[-1]["event_hash"] if events else GENESIS_HASH,
    }
    event["event_hash"] = hashlib.sha256(_event_payload(event)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def lockbox_state(events: list[dict[str, Any]], lockbox_id: str,
                  initial_state: str = "SEALED_EMPTY") -> tuple[str, bool, str | None]:
    relevant = [event for event in events if event.get("lockbox_id") == lockbox_id]
    accessed = [event for event in relevant if event.get("result_accessed")]
    if accessed:
        return "OPENED_CONSUMED", True, str(accessed[0]["occurred_at_utc"])
    actions = {str(event["action"]) for event in relevant}
    if "LOCKBOX_READY" in actions:
        return "READY_SEALED", False, None
    if "LOCKBOX_DATA_ARRIVED" in actions:
        return "COLLECTING", False, None
    return initial_state, False, None


def _source_artifacts(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in cfg["source_artifacts"]:
        table, key = str(item["table"]), str(item["key"])
        path = lake.table_path(table, key)
        frame = lake.read(table, key)
        if frame.empty or not path.exists():
            raise RuntimeError(f"governance source artifact is missing: {table}/{key}")
        rows.append({
            "table": table, "key": key, "rows": len(frame),
            "path": str(path.resolve()), "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


def _config_hashes() -> pd.DataFrame:
    names = [
        "research_governance.yaml", "model_benchmark.yaml",
        "model_stability_validation.yaml", "constrained_portfolio_neutralization.yaml",
        "constrained_portfolio_daily_execution.yaml", "transaction_cost_model.yaml",
        "incremental_alpha_test.yaml", "market_benchmark_attribution.yaml",
        "risk_free_rate_attribution.yaml", "research_conclusion_audit.yaml",
    ]
    rows = []
    for name in names:
        path = ROOT / "config" / name
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"config_file": name, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def _evidence_registry(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, period in cfg["evidence_periods"].items():
        rows.append({
            "period_name": name, "evidence_class": period["evidence_class"],
            "start_date": period.get("start_date"),
            "end_date": period.get("end_date", period.get("planned_end_date")),
            "already_observed_at_freeze": bool(period.get("already_observed", False)),
            "consumed_at_freeze": bool(period.get("consumed", False)),
            "selection_allowed": bool(
                period.get("selection_allowed", period.get("selection_allowed_for_new_revisions", False))
            ),
            "disclosure": period.get("disclosure", ""),
        })
    return pd.DataFrame(rows)


def _hypothesis_registry(cfg: dict[str, Any]) -> pd.DataFrame:
    plan = cfg["statistical_plan"]
    hypotheses = [plan["primary_hypothesis"], *plan["secondary_hypotheses"]]
    return pd.DataFrame([{
        "hypothesis_order": index + 1, "hypothesis": hypothesis,
        "is_primary": index == 0, "metric": "incremental_net_return_annualized_arithmetic",
        "paired_clock": plan["paired_clock"], "bootstrap": plan["bootstrap"],
        "block_length_trading_days": plan["block_length_trading_days"],
        "bootstrap_samples": plan["bootstrap_samples"],
        "multiplicity_control": plan["multiplicity_control"],
        "false_discovery_rate": plan["false_discovery_rate"],
        "frozen_before_future_data": True,
    } for index, hypothesis in enumerate(hypotheses)])


def _candidate_freeze(cfg: dict[str, Any], artifacts: pd.DataFrame,
                      configs: pd.DataFrame, git: dict[str, Any]) -> pd.DataFrame:
    bundle = cfg["candidate_bundle"]
    primary = bundle["primary"]
    dependency_hash = stable_json_hash(
        artifacts[["table", "key", "sha256"]].to_dict(orient="records")
    )
    config_hash = stable_json_hash(configs.to_dict(orient="records"))
    return pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "bundle_id": bundle["bundle_id"],
        "freeze_commit": git["commit"], "working_tree_dirty_at_freeze": git["dirty"],
        "source_tree_sha256": source_tree_hash(ROOT),
        "dependency_snapshot_sha256": dependency_hash,
        "config_bundle_sha256": config_hash,
        "primary_model": primary["model"],
        "primary_alpha_pipeline": primary["alpha_pipeline"],
        "primary_portfolio_construction": primary["portfolio_construction"],
        "selection_source": primary["selection_source"],
        "target": bundle["target"], "training_window": bundle["model_training_window"],
        "primary_cost_scenario": cfg["cost_and_execution"]["primary_cost_scenario"],
        "primary_round_trip_cost_bps": cfg["cost_and_execution"]["primary_round_trip_cost_bps"],
        "execution_convention": cfg["cost_and_execution"]["execution_convention"],
        "historical_oos_used_for_selection": False,
        "new_model_selected_in_phase4": False,
        "freeze_status": "FROZEN_FOR_FUTURE_EVIDENCE",
    }])


def _lockbox_registry(cfg: dict[str, Any], events: list[dict[str, Any]]) -> pd.DataFrame:
    lockbox = cfg["evidence_periods"]["future_lockbox"]
    state, consumed, first_access = lockbox_state(
        events, str(lockbox["lockbox_id"]), str(lockbox["initial_state"])
    )
    return pd.DataFrame([{
        "lockbox_id": lockbox["lockbox_id"], "state": state,
        "start_date": lockbox["start_date"], "planned_end_date": lockbox["planned_end_date"],
        "minimum_daily_observations": lockbox["minimum_daily_observations"],
        "minimum_weekly_signal_observations": lockbox["minimum_weekly_signal_observations"],
        "minimum_calendar_months": lockbox["minimum_calendar_months"],
        "minimum_coverage_fraction": lockbox["minimum_coverage_fraction"],
        "interim_performance_access_allowed": False,
        "operational_health_access_allowed": True,
        "consumed": consumed, "first_result_access_utc": first_access,
        "open_rule": lockbox["open_rule"],
        "cryptographic_access_control_implemented": bool(
            lockbox["local_enforcement_is_cryptographic"]
        ),
    }])


def _paper_registry(cfg: dict[str, Any], events: list[dict[str, Any]]) -> pd.DataFrame:
    paper = cfg["evidence_periods"]["paper_trading"]
    observations = sum(event["action"] == "PAPER_OBSERVATION_RECORDED" for event in events)
    return pd.DataFrame([{
        "paper_id": paper["paper_id"],
        "state": "ACTIVE" if observations else "FROZEN_NOT_STARTED",
        "activation_rule": paper["activation_rule"],
        "retroactive_backfill_allowed": False,
        "recorded_observations": observations,
        "performance_reporting_frequency": paper["performance_reporting_frequency"],
        "operational_health_reporting_frequency": paper["operational_health_reporting_frequency"],
        "outcome_driven_change_requires_new_version": True,
        "is_live_capital": False,
    }])


def _checks(cfg: dict[str, Any], freeze: pd.DataFrame, lockbox: pd.DataFrame,
            paper: pd.DataFrame, events: list[dict[str, Any]],
            artifacts: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("historical_oos_explicitly_consumed",
         bool(cfg["evidence_periods"]["historical_oos"]["consumed"])),
        ("historical_oos_reselection_prohibited",
         not bool(cfg["evidence_periods"]["historical_oos"]["selection_allowed_for_new_revisions"])),
        ("candidate_commit_recorded", bool(freeze.iloc[0]["freeze_commit"])),
        ("candidate_source_tree_clean_at_freeze",
         freeze.iloc[0]["working_tree_dirty_at_freeze"] is False
         or freeze.iloc[0]["working_tree_dirty_at_freeze"] == False),
        ("dependency_hashes_complete", artifacts["sha256"].str.len().eq(64).all()),
        ("future_lockbox_unconsumed", not bool(lockbox.iloc[0]["consumed"])),
        ("future_lockbox_interim_performance_sealed",
         not bool(lockbox.iloc[0]["interim_performance_access_allowed"])),
        ("paper_trading_not_backfilled", not bool(paper.iloc[0]["retroactive_backfill_allowed"])),
        ("access_log_hash_chain_valid", len(events) >= 4),
        ("phase4_did_not_select_new_model", cfg["new_model_selected"] is False),
    ]
    return pd.DataFrame([{
        "check": name, "status": "PASS" if passed else "FAIL",
        "passed": bool(passed),
    } for name, passed in checks])


def _protocol(cfg: dict[str, Any], git: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "policy_version": cfg["policy_version"],
        "purpose": cfg["purpose"], "current_data_cutoff": cfg["current_data_cutoff"],
        "historical_oos_classification": "consumed_descriptive_not_pristine",
        "future_lockbox_id": cfg["evidence_periods"]["future_lockbox"]["lockbox_id"],
        "paper_trading_id": cfg["evidence_periods"]["paper_trading"]["paper_id"],
        "candidate_freeze_commit": git["commit"],
        "historical_oos_used_for_selection": False,
        "future_results_available_at_freeze": False,
        "paper_observations_available_at_freeze": False,
        "local_access_control_limitation": "hash_chained_audit_not_cryptographic_secrecy",
    }])


def _write_report(outputs: dict[str, pd.DataFrame]) -> None:
    evidence = outputs["research_evidence_registry"]
    freeze = outputs["research_candidate_freeze"]
    lockbox = outputs["research_lockbox_registry"]
    paper = outputs["research_paper_trading_registry"]
    hypotheses = outputs["research_hypothesis_registry"]
    checks = outputs["research_governance_checks"]
    lines = [
        "# Phase 4：Historical OOS 污染治理与未来证据冻结", "",
        "## 决策", "",
        "2021–2026（当前数据截止 2026-09-30）已经被反复查看，永久归类为 consumed historical OOS。Phase 4 没有重新选模；当前候选包仅用于未来 lockbox 与真实时间 paper tracking。",
        "", "## 证据分层", "", evidence.to_markdown(index=False),
        "", "## 当前候选冻结", "", freeze.to_markdown(index=False),
        "", "## Future lockbox", "", lockbox.to_markdown(index=False),
        "", "本地 hash-chain 能发现访问日志篡改，但不是密码学保密或独立托管；机构部署应把结果密钥和访问权限交给独立研究治理角色。",
        "", "## Paper trading", "", paper.to_markdown(index=False),
        "", "Paper trading 不能回填。任何基于其结果的修改都必须产生新候选版本，旧版本继续保留。",
        "", "## 冻结假设族", "", hypotheses.to_markdown(index=False),
        "", "## 验收", "", checks.to_markdown(index=False),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    for table, frame in outputs.items():
        frame.to_csv(TABLE_ROOT / f"{table}.csv", index=False)


def initialize(lake: Lake | None = None, *, require_clean_git: bool = True) -> dict[str, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    log_path = ROOT / str(cfg["access_log"]["path"])
    if log_path.exists():
        raise FileExistsError(f"governance access log already initialized: {log_path}")
    git = git_snapshot()
    if not git["commit"]:
        raise RuntimeError("candidate freeze requires a Git commit")
    if require_clean_git and git["dirty"] is not False:
        raise RuntimeError("candidate freeze requires a clean Git working tree")
    artifacts = _source_artifacts(lake, cfg)
    configs = _config_hashes()
    lockbox_id = str(cfg["evidence_periods"]["future_lockbox"]["lockbox_id"])
    fixed_time = f"{cfg['policy_frozen_date_utc']}T00:00:00+00:00"
    initial_events = [
        dict(actor="research_governance", evidence_class="historical_oos",
             action="PRIOR_ACCESS_DISCLOSED",
             reason="2021-2026 results were repeatedly inspected during development",
             result_accessed=True, used_for_change=True, lockbox_id=None),
        dict(actor="research_governance", evidence_class="governance_only",
             action="CANDIDATE_BUNDLE_FROZEN",
             reason="freeze validation-selected candidate before future evidence",
             result_accessed=False, used_for_change=False, lockbox_id=None),
        dict(actor="research_governance", evidence_class="future_lockbox",
             action="LOCKBOX_SEALED",
             reason="create empty future lockbox before loading post-cutoff data",
             result_accessed=False, used_for_change=False, lockbox_id=lockbox_id),
        dict(actor="research_governance", evidence_class="paper_trading",
             action="PAPER_PROTOCOL_FROZEN",
             reason="paper observations may begin only after candidate freeze commit",
             result_accessed=False, used_for_change=False, lockbox_id=None),
    ]
    for index, event in enumerate(initial_events):
        append_access_event(log_path, occurred_at_utc=fixed_time, **event)
    events = verify_access_log(log_path)
    evidence = _evidence_registry(cfg)
    freeze = _candidate_freeze(cfg, artifacts, configs, git)
    lockbox = _lockbox_registry(cfg, events)
    paper = _paper_registry(cfg, events)
    outputs = {
        "research_governance_protocol": _protocol(cfg, git),
        "research_evidence_registry": evidence,
        "research_candidate_freeze": freeze,
        "research_lockbox_registry": lockbox,
        "research_paper_trading_registry": paper,
        "research_hypothesis_registry": _hypothesis_registry(cfg),
        "research_governance_dependency_hashes": pd.concat([
            artifacts.assign(dependency_type="data_artifact").rename(columns={"table": "name"}),
            configs.assign(dependency_type="config").rename(columns={"config_file": "name"}),
        ], ignore_index=True, sort=False),
        "research_access_log_snapshot": pd.DataFrame(events),
    }
    outputs["research_governance_checks"] = _checks(
        cfg, freeze, lockbox, paper, events, artifacts
    )
    if outputs["research_governance_checks"]["status"].ne("PASS").any():
        raise RuntimeError("research governance acceptance checks failed")
    request = {**cfg, "candidate_freeze_commit": git["commit"]}
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="research_governance.v1", request=request)
    _write_report(outputs)
    return outputs


def refresh_status(lake: Lake | None = None) -> dict[str, Any]:
    lake = lake or Lake()
    cfg = load_config()
    log_path = ROOT / str(cfg["access_log"]["path"])
    events = verify_access_log(log_path)
    lockbox = _lockbox_registry(cfg, events)
    paper = _paper_registry(cfg, events)
    return {
        "event_count": len(events),
        "last_event_hash": events[-1]["event_hash"] if events else None,
        "lockbox": lockbox.iloc[0].to_dict(),
        "paper_trading": paper.iloc[0].to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research evidence and lockbox governance")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("initialize", help="freeze candidate and create empty future evidence records")
    sub.add_parser("status", help="verify the access-log chain and show current state")
    record = sub.add_parser("record-access", help="append an auditable evidence-access event")
    record.add_argument("--actor", required=True)
    record.add_argument("--evidence-class", required=True, choices=sorted(EVIDENCE_CLASSES))
    record.add_argument("--action", required=True)
    record.add_argument("--reason", required=True)
    record.add_argument("--result-accessed", action="store_true")
    record.add_argument("--used-for-change", action="store_true")
    record.add_argument("--lockbox-id")
    args = parser.parse_args()
    if args.command == "initialize":
        outputs = initialize()
        print(outputs["research_governance_checks"].to_string(index=False))
    elif args.command == "status":
        print(json.dumps(refresh_status(), ensure_ascii=False, indent=2, default=str))
    else:
        cfg = load_config()
        event = append_access_event(
            ROOT / str(cfg["access_log"]["path"]), actor=args.actor,
            evidence_class=args.evidence_class, action=args.action, reason=args.reason,
            result_accessed=args.result_accessed, used_for_change=args.used_for_change,
            lockbox_id=args.lockbox_id,
        )
        print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
