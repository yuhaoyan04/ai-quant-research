import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from quant_research.experiment_registry import (
    ExperimentRegistry,
    publish_immutable_directory,
    validate_governance_declaration,
)
from quant_research.ingest import Lake


def _setup(tmp_path: Path) -> tuple[ExperimentRegistry, dict, dict]:
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "config").mkdir()
    (project / "config" / "model.yaml").write_text("alpha: 1\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    lake = Lake(project / "lake")
    lake.write("features", pd.DataFrame({"x": [1.0]}), key="v1", source="test", request={})
    lake.write("summary", pd.DataFrame({"model": ["ridge"], "metric": [0.1]}),
               key="v1", source="test", request={})
    figure = project / "figure.svg"
    figure.write_text("<svg></svg>", encoding="utf-8")
    global_spec = {
        "registry_version": 1, "selection_policy": "validation_only",
        "historical_oos_disclosure": "not pristine",
    }
    spec = {
        "title": "test baseline", "stage": "test", "model_version": "v1",
        "historical_oos_used_for_selection": False,
        "config_files": ["model.yaml"],
        "input_tables": [{"table": "features", "key": "v1"}],
        "metric_tables": [{"table": "summary", "key": "v1", "where": {"model": "ridge"}}],
        "figures": ["figure.svg"],
    }
    registry = ExperimentRegistry(project, project / "reports" / "experiments", lake)
    return registry, global_spec, spec


def test_register_creates_complete_immutable_snapshot(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    path = registry.register("ridge_v1", global_spec, spec)
    assert {"config.yaml", "metadata.json", "metrics.json", "notes.md", "FROZEN.json"}.issubset(
        {item.name for item in path.iterdir()})
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["immutable"] is True
    assert metadata["historical_oos_used_for_selection"] is False
    assert len(metadata["dataset_version"]) == 64
    verification = registry.verify("ridge_v1")
    assert verification["snapshot_integrity"]
    assert verification["live_inputs_match"]


def test_registration_does_not_mark_its_own_staging_directory_dirty(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    subprocess.run(["git", "init"], cwd=registry.project_root, check=True,
                   capture_output=True)
    subprocess.run(["git", "add", "."], cwd=registry.project_root, check=True,
                   capture_output=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-m", "baseline",
    ], cwd=registry.project_root, check=True, capture_output=True)

    path = registry.register("ridge_v1", global_spec, spec)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["git"]["commit"]
    assert metadata["git"]["dirty"] is False


def test_frozen_experiment_cannot_be_overwritten(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    registry.register("ridge_v1", global_spec, spec)
    with pytest.raises(FileExistsError):
        registry.register("ridge_v1", global_spec, spec)


def test_verification_detects_snapshot_tampering(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    path = registry.register("ridge_v1", global_spec, spec)
    (path / "metrics.json").write_text("{}", encoding="utf-8")
    assert not registry.verify("ridge_v1", verify_live_inputs=False)["snapshot_integrity"]


def test_snapshot_verification_accepts_git_line_ending_normalization(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    path = registry.register("ridge_v1", global_spec, spec)
    metrics = path / "metrics.json"
    metrics.write_bytes(metrics.read_bytes().replace(b"\n", b"\r\n"))

    verification = registry.verify("ridge_v1", verify_live_inputs=False)
    assert verification["snapshot_integrity"]
    assert verification["snapshot_mismatches"] == []


def test_live_verification_resolves_tables_in_current_lake(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    registry.register("ridge_v1", global_spec, spec)

    relocated_root = tmp_path / "relocated_lake"
    shutil.copytree(registry.lake.root, relocated_root)
    shutil.rmtree(registry.lake.root)
    relocated = ExperimentRegistry(
        registry.project_root, registry.registry_root, Lake(relocated_root),
    )

    verification = relocated.verify("ridge_v1")
    assert verification["live_inputs_match"]
    assert all(str(relocated_root.resolve()) in item["path"]
               for item in verification["input_checks"])


def test_unsafe_experiment_id_is_rejected(tmp_path: Path) -> None:
    registry, global_spec, spec = _setup(tmp_path)
    with pytest.raises(ValueError):
        registry.register("../escape", global_spec, spec)


def test_publish_falls_back_when_directory_rename_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "tables").mkdir(parents=True)
    (source / "tables" / "metric.csv").write_text("value\n1\n", encoding="utf-8")
    (source / "FROZEN.json").write_text("{}", encoding="utf-8")

    def denied_replace(self: Path, destination: Path) -> Path:
        raise PermissionError(f"rename denied: {self} -> {destination}")

    monkeypatch.setattr(Path, "replace", denied_replace)
    publish_immutable_directory(source, target)

    assert not source.exists()
    assert (target / "tables" / "metric.csv").is_file()
    assert (target / "FROZEN.json").is_file()


def test_new_registration_requires_explicit_evidence_governance() -> None:
    global_spec = {"governance_required_for_new_registrations": True}
    with pytest.raises(ValueError, match="missing fields"):
        validate_governance_declaration(global_spec, {})
    valid = {
        "evidence_class": "governance_only",
        "research_selection_role": "governance",
        "claims_independent_evidence": False,
        "historical_oos_used_for_selection": False,
    }
    validate_governance_declaration(global_spec, valid)


def test_historical_oos_selection_must_be_declared_post_hoc() -> None:
    global_spec = {"governance_required_for_new_registrations": True}
    invalid = {
        "evidence_class": "historical_oos_descriptive",
        "research_selection_role": "diagnostic",
        "claims_independent_evidence": False,
        "historical_oos_used_for_selection": True,
    }
    with pytest.raises(ValueError, match="post_historical_oos_adaptation"):
        validate_governance_declaration(global_spec, invalid)
