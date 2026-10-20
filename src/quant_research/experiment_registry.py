"""Immutable experiment registration for reproducible research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


DEFAULT_SPEC = ROOT / "config" / "frozen_baselines.yaml"
DEFAULT_REGISTRY = ROOT / "reports" / "experiments"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
TEXT_ARTIFACT_SUFFIXES = {
    ".csv", ".html", ".json", ".md", ".py", ".svg", ".toml", ".txt", ".yaml", ".yml",
}
EVIDENCE_CLASSES = {
    "validation_selection", "historical_oos_descriptive",
    "post_historical_oos_adaptation", "future_lockbox", "paper_trading",
    "governance_only", "mixed_disclosed",
}
RESEARCH_SELECTION_ROLES = {
    "selection", "diagnostic", "final_evaluation", "operational_tracking",
    "post_hoc_adaptation", "governance",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hash_variants(path: Path) -> set[str]:
    """Return byte-exact and cross-platform text-line-ending hashes.

    Git may normalize a frozen text artifact from Windows CRLF to LF on
    checkout.  That is not a content mutation, while any other byte change
    must continue to fail the immutable-snapshot gate.
    """
    exact = sha256_file(path)
    if path.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
        return {exact}
    content = path.read_bytes()
    lf = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        exact,
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def source_tree_hash(project_root: Path) -> str:
    candidates = sorted((project_root / "src").rglob("*.py"))
    candidates += [project_root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.exists():
            continue
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_governance_declaration(global_spec: dict[str, Any],
                                    spec: dict[str, Any]) -> None:
    """Require an explicit evidence role for experiments registered after Phase 4."""
    if not global_spec.get("governance_required_for_new_registrations", False):
        return
    required = {
        "evidence_class", "research_selection_role", "claims_independent_evidence",
        "historical_oos_used_for_selection",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"governance declaration missing fields: {missing}")
    evidence_class = str(spec["evidence_class"])
    role = str(spec["research_selection_role"])
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unsupported evidence_class: {evidence_class}")
    if role not in RESEARCH_SELECTION_ROLES:
        raise ValueError(f"unsupported research_selection_role: {role}")
    if evidence_class == "future_lockbox" and not spec.get("lockbox_id"):
        raise ValueError("future_lockbox experiments require lockbox_id")
    if evidence_class in {"historical_oos_descriptive", "post_historical_oos_adaptation"}:
        if spec.get("claims_independent_evidence") is not False:
            raise ValueError("historical-OOS experiments cannot claim independent evidence")
    if spec.get("historical_oos_used_for_selection") is True:
        if evidence_class != "post_historical_oos_adaptation" or role != "post_hoc_adaptation":
            raise ValueError(
                "historical-OOS selection must be declared as post_historical_oos_adaptation"
            )


def git_state(project_root: Path) -> dict[str, Any]:
    def call(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={project_root}", *args],
                cwd=project_root, check=True, capture_output=True,
                text=True, timeout=15,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = call("rev-parse", "HEAD")
    status = call("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None,
    }


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("pandas", "pyarrow", "PyYAML", "scikit-learn", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def publish_immutable_directory(temp: Path, target: Path) -> None:
    """Publish a complete snapshot without overwriting an existing target.

    Some mounted workspaces permit file creation but reject directory-level
    ``rename(2)``. Prefer atomic rename, then fall back to an exclusive target
    and copy ``FROZEN.json`` last as the commit marker.
    """
    try:
        temp.replace(target)
        return
    except PermissionError:
        pass
    target.mkdir(exist_ok=False)
    try:
        for source in sorted(temp.iterdir(), key=lambda path: path.name):
            if source.name == "FROZEN.json":
                continue
            destination = target / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        shutil.copy2(temp / "FROZEN.json", target / "FROZEN.json")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    shutil.rmtree(temp)


def filter_frame(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column, expected in spec.get("where", {}).items():
        if column not in result:
            raise KeyError(f"filter column missing: {column}")
        result = result.loc[result[column].astype(str).eq(str(expected))]
    for column, prefix in spec.get("where_prefix", {}).items():
        if column not in result:
            raise KeyError(f"prefix filter column missing: {column}")
        result = result.loc[result[column].astype(str).str.startswith(str(prefix))]
    return result.reset_index(drop=True)


class ExperimentRegistry:
    def __init__(self, project_root: Path = ROOT, registry_root: Path = DEFAULT_REGISTRY,
                 lake: Lake | None = None) -> None:
        self.project_root = project_root.resolve()
        self.registry_root = registry_root.resolve()
        self.lake = lake or Lake()

    def _validate_id(self, experiment_id: str) -> None:
        if not SAFE_ID.fullmatch(experiment_id):
            raise ValueError("experiment_id must match [a-z0-9][a-z0-9_-]{2,79}")

    def _artifact_record(self, table: str, key: str) -> dict[str, Any]:
        path = self.lake.table_path(table, key).resolve()
        if not path.exists():
            raise FileNotFoundError(f"required input artifact missing: {table}/{key}: {path}")
        return {
            "table": table,
            "key": key,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def _config_snapshot(self, experiment_id: str, global_spec: dict, spec: dict) -> dict:
        configs = {}
        for filename in spec.get("config_files", []):
            path = self.project_root / "config" / filename
            if not path.exists():
                raise FileNotFoundError(f"config missing: {path}")
            with path.open(encoding="utf-8") as handle:
                configs[filename] = yaml.safe_load(handle)
        return {
            "experiment_id": experiment_id,
            "registry_version": global_spec.get("registry_version", 1),
            "selection_policy": global_spec.get("selection_policy"),
            "historical_oos_disclosure": global_spec.get("historical_oos_disclosure"),
            "governance_required_for_new_registrations": global_spec.get(
                "governance_required_for_new_registrations", False
            ),
            "governance_policy": global_spec.get("governance_policy"),
            "experiment": spec,
            "resolved_configs": configs,
        }

    @staticmethod
    def _directory_manifest(directory: Path) -> dict[str, str]:
        records = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "FROZEN.json":
                records[path.relative_to(directory).as_posix()] = sha256_file(path)
        return records

    def register(self, experiment_id: str, global_spec: dict, spec: dict) -> Path:
        self._validate_id(experiment_id)
        validate_governance_declaration(global_spec, spec)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        target = self.registry_root / experiment_id
        if target.exists():
            raise FileExistsError(f"experiment already exists and will not be overwritten: {target}")

        temp = Path(tempfile.mkdtemp(prefix=f".{experiment_id}.", dir=self.registry_root))
        try:
            (temp / "tables").mkdir()
            (temp / "figures").mkdir()
            config_snapshot = self._config_snapshot(experiment_id, global_spec, spec)
            (temp / "config.yaml").write_text(
                yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

            inputs = [self._artifact_record(str(item["table"]), str(item["key"]))
                      for item in spec.get("input_tables", [])]
            ingestion_manifest = self.lake.manifests / "ingestion.jsonl"
            manifest_record = ({
                "path": str(ingestion_manifest.resolve()),
                "size_bytes": ingestion_manifest.stat().st_size,
                "sha256": sha256_file(ingestion_manifest),
            } if ingestion_manifest.exists() else None)
            dataset_version = stable_json_hash({
                "inputs": [{"table": item["table"], "key": item["key"], "sha256": item["sha256"]}
                           for item in inputs],
                "ingestion_manifest_sha256": manifest_record["sha256"] if manifest_record else None,
            })

            metrics: dict[str, Any] = {}
            metric_artifacts = []
            for item in spec.get("metric_tables", []):
                table, key = str(item["table"]), str(item["key"])
                frame = self.lake.read(table, key)
                if frame.empty:
                    raise RuntimeError(f"metric table missing or empty: {table}/{key}")
                frame = filter_frame(frame, item)
                if frame.empty:
                    raise RuntimeError(f"metric filter produced no rows: {table}/{key}")
                name = f"{table}__{key}"
                csv_path = temp / "tables" / f"{name}.csv"
                frame.to_csv(csv_path, index=False, encoding="utf-8")
                metrics[name] = [jsonable(record) for record in frame.to_dict(orient="records")]
                metric_artifacts.append(self._artifact_record(table, key))
            (temp / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

            copied_figures = []
            for relative in spec.get("figures", []):
                source = (self.project_root / relative).resolve()
                if not source.exists():
                    raise FileNotFoundError(f"figure missing: {source}")
                destination = temp / "figures" / source.name
                shutil.copy2(source, destination)
                copied_figures.append({
                    "source": str(source), "snapshot": destination.name,
                    "sha256": sha256_file(destination),
                })

            metadata = {
                "experiment_id": experiment_id,
                "title": spec.get("title"),
                "stage": spec.get("stage"),
                "model_version": spec.get("model_version"),
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "immutable": True,
                "selection_policy": global_spec.get("selection_policy"),
                "historical_oos_used_for_selection": bool(
                    spec.get("historical_oos_used_for_selection", False)),
                "historical_oos_disclosure": global_spec.get("historical_oos_disclosure"),
                "evidence_class": spec.get("evidence_class"),
                "research_selection_role": spec.get("research_selection_role"),
                "claims_independent_evidence": spec.get("claims_independent_evidence"),
                "lockbox_id": spec.get("lockbox_id"),
                "dataset_version": dataset_version,
                "dataset_manifest": manifest_record,
                "input_artifacts": inputs,
                "metric_artifacts": metric_artifacts,
                "figures": copied_figures,
                "source_tree_sha256": source_tree_hash(self.project_root),
                "git": git_state(self.project_root),
                "environment": environment_snapshot(),
            }
            (temp / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            notes = (
                f"# {spec.get('title', experiment_id)}\n\n"
                "This directory is an immutable snapshot of an existing research baseline.\n\n"
                f"- Selection policy: `{global_spec.get('selection_policy')}`\n"
                f"- Dataset version: `{dataset_version}`\n"
                f"- Historical OOS used for selection: "
                f"`{bool(spec.get('historical_oos_used_for_selection', False))}`\n\n"
                f"> {global_spec.get('historical_oos_disclosure')}\n"
            )
            (temp / "notes.md").write_text(notes, encoding="utf-8")

            files = self._directory_manifest(temp)
            frozen = {
                "experiment_id": experiment_id,
                "frozen_at_utc": metadata["registered_at_utc"],
                "content_sha256": stable_json_hash(files),
                "files": files,
            }
            (temp / "FROZEN.json").write_text(
                json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
            publish_immutable_directory(temp, target)
            self._update_index()
            return target
        except Exception:
            if temp.exists() and temp.parent == self.registry_root:
                shutil.rmtree(temp)
            raise

    def verify(self, experiment_id: str, verify_live_inputs: bool = True) -> dict[str, Any]:
        self._validate_id(experiment_id)
        directory = self.registry_root / experiment_id
        frozen_path = directory / "FROZEN.json"
        if not frozen_path.exists():
            raise FileNotFoundError(f"frozen experiment missing: {directory}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected_files = frozen["files"]
        actual_paths = {
            path.relative_to(directory).as_posix(): path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "FROZEN.json"
        }
        snapshot_mismatches = sorted(
            relative for relative in set(expected_files) | set(actual_paths)
            if relative not in expected_files
            or relative not in actual_paths
            or expected_files[relative] not in artifact_hash_variants(actual_paths[relative])
        )
        snapshot_ok = (
            not snapshot_mismatches
            and frozen["content_sha256"] == stable_json_hash(expected_files)
        )
        input_checks = []
        if verify_live_inputs:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            for item in metadata["input_artifacts"] + metadata["metric_artifacts"]:
                # The path captured at registration documents provenance on the
                # originating machine.  Live verification must resolve the
                # logical table/key against this registry's lake so a rebuilt
                # dataset can be checked after moving to another machine.
                path = self.lake.table_path(str(item["table"]), str(item["key"])).resolve()
                actual = sha256_file(path) if path.exists() else None
                input_checks.append({
                    "table": item["table"], "key": item["key"],
                    "path": str(path),
                    "expected_sha256": item["sha256"], "actual_sha256": actual,
                    "matches": actual == item["sha256"],
                })
        return {
            "experiment_id": experiment_id,
            "snapshot_integrity": snapshot_ok,
            "snapshot_mismatches": snapshot_mismatches,
            "live_inputs_checked": verify_live_inputs,
            "live_inputs_match": all(item["matches"] for item in input_checks) if input_checks else None,
            "input_checks": input_checks,
        }

    def _update_index(self) -> None:
        records = []
        for path in sorted(self.registry_root.iterdir()):
            metadata_path = path / "metadata.json"
            frozen_path = path / "FROZEN.json"
            if not path.is_dir() or not metadata_path.exists() or not frozen_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records.append({
                "experiment_id": metadata["experiment_id"],
                "title": metadata["title"],
                "stage": metadata["stage"],
                "model_version": metadata["model_version"],
                "registered_at_utc": metadata["registered_at_utc"],
                "dataset_version": metadata["dataset_version"],
                "immutable": metadata["immutable"],
            })
        (self.registry_root / "index.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_specs(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result.get("experiments"), dict):
        raise ValueError("frozen baseline spec requires an experiments mapping")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Register and verify immutable research experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register", help="register one or all configured frozen baselines")
    register.add_argument("experiment_id", nargs="?", help="omit with --all")
    register.add_argument("--all", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a frozen experiment")
    verify.add_argument("experiment_id")
    verify.add_argument("--snapshot-only", action="store_true")
    subparsers.add_parser("list", help="list registered experiments")
    args = parser.parse_args()

    registry = ExperimentRegistry()
    if args.command == "register":
        specs = load_specs()
        ids = list(specs["experiments"]) if args.all else [args.experiment_id]
        if not ids or ids == [None]:
            parser.error("provide experiment_id or --all")
        for experiment_id in ids:
            if experiment_id not in specs["experiments"]:
                raise KeyError(f"experiment not configured: {experiment_id}")
            print(registry.register(experiment_id, specs, specs["experiments"][experiment_id]))
    elif args.command == "verify":
        result = registry.verify(args.experiment_id, not args.snapshot_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["snapshot_integrity"] or result["live_inputs_match"] is False:
            raise SystemExit(1)
    else:
        index = registry.registry_root / "index.json"
        print(index.read_text(encoding="utf-8") if index.exists() else "[]")


if __name__ == "__main__":
    main()
