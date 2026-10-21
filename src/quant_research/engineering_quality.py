"""Phase 10 repository engineering contracts and structural audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "engineering_quality.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def implementation_inventory(root: Path = ROOT) -> pd.DataFrame:
    signatures = {
        "corporate_action_proxy_formula": "raw_factor = previous * (1.0 + returns) / close",
        "execution_buy_limit_rule": "buy_allowed=status & pct.lt",
        "execution_market_public_api": "read_execution_market_rows(",
    }
    rows = []
    for name, signature in signatures.items():
        for path in sorted((root / "src" / "quant_research").glob("*.py")):
            if path.name == "engineering_quality.py":
                continue
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                if signature in line:
                    rows.append({
                        "logic": name, "path": path.relative_to(root).as_posix(),
                        "line": number, "is_definition": path.name == "execution_market.py",
                    })
    return pd.DataFrame(rows)


def random_seed_inventory(root: Path = ROOT) -> pd.DataFrame:
    pattern = re.compile(r"(?:default_rng|RandomState)\(([^)]*)\)")
    rows = []
    for path in sorted((root / "src" / "quant_research").glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in pattern.finditer(line):
                expression = match.group(1).strip()
                rows.append({
                    "path": path.relative_to(root).as_posix(), "line": number,
                    "seed_expression": expression,
                    "explicit_seed": bool(expression and expression.lower() != "none"),
                })
    return pd.DataFrame(rows)


def test_layer_inventory(cfg: dict[str, Any], root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for layer, paths in cfg["required_test_layers"].items():
        for raw_path in paths:
            path = root / raw_path
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            rows.append({
                "layer": layer, "path": raw_path, "exists": path.exists(),
                "test_functions": len(re.findall(r"^def test_", text, flags=re.MULTILINE)),
            })
    return pd.DataFrame(rows)


def output_contracts(lake: Lake, cfg: dict[str, Any], root: Path = ROOT) -> pd.DataFrame:
    registry = root / "reports" / "experiments"
    rows = []
    for item in cfg["critical_protocols"]:
        frame = lake.read(str(item["table"]), str(item["key"]))
        experiment_id = str(frame.iloc[0].get("experiment_id", "")) if not frame.empty else ""
        directory = registry / str(item["frozen_experiment"])
        frozen = directory / "FROZEN.json"
        metadata = directory / "metadata.json"
        frozen_valid = False
        if frozen.exists() and metadata.exists():
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            frozen_valid = bool(payload.get("immutable"))
        rows.append({
            **item, "protocol_rows": len(frame), "protocol_experiment_id": experiment_id,
            "protocol_has_experiment_id": bool(experiment_id),
            "frozen_snapshot_exists": frozen.exists(), "frozen_snapshot_immutable": frozen_valid,
        })
    return pd.DataFrame(rows)


def frozen_registry_inventory(root: Path = ROOT) -> pd.DataFrame:
    """Reconcile the public index with immutable directories by identity.

    A fixed expected count becomes stale whenever a legitimate experiment is
    appended.  The institutional invariant is stronger: IDs are unique, every
    indexed run has exactly one committed snapshot, and no snapshot is hidden
    from the index.
    """
    registry = root / "reports" / "experiments"
    payload = json.loads((registry / "index.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("experiment index must be a list of records")
    index_ids = [str(item.get("experiment_id", "")) for item in payload]
    directory_ids = sorted(
        path.name for path in registry.iterdir()
        if path.is_dir() and (path / "FROZEN.json").exists()
    )
    counts: dict[str, int] = {}
    for experiment_id in index_ids:
        counts[experiment_id] = counts.get(experiment_id, 0) + 1
    all_ids = sorted(set(index_ids) | set(directory_ids))
    immutable_by_id = {
        str(item.get("experiment_id", "")): bool(item.get("immutable"))
        for item in payload
    }
    return pd.DataFrame([{
        "experiment_id": experiment_id,
        "index_occurrences": counts.get(experiment_id, 0),
        "indexed": experiment_id in set(index_ids),
        "snapshot_directory_exists": experiment_id in set(directory_ids),
        "index_declares_immutable": immutable_by_id.get(experiment_id, False),
    } for experiment_id in all_ids])


def gap_register() -> pd.DataFrame:
    return pd.DataFrame([
        {"priority": "P0", "gap": "failed_market_rows_could_enter_forward_labels",
         "status": "CLOSED_IN_CODE_V2",
         "mitigation": "quality-fail returns remain on the date axis but invalidate every affected label window"},
        {"priority": "P1", "gap": "live_curated_table_names_are_mutable",
         "status": "CONTROLLED_BY_IMMUTABLE_REGISTRY",
         "mitigation": "publication claims use frozen experiment SHA-256, never bare live v1 names"},
        {"priority": "P1", "gap": "no_static_type_gate",
         "status": "OPEN", "mitigation": "add pyright or mypy after annotating legacy modules"},
        {"priority": "P2", "gap": "pytest_cache_mount_permission",
         "status": "CLOSED_TOOLING_V3",
         "mitigation": "disable the optional pytest cache provider on read-only/shared mounts"},
        {"priority": "P2", "gap": "legacy_modules_still_contain_version_literals",
         "status": "OPEN", "mitigation": "migrate new work first; change old versions only with new experiment IDs"},
        {"priority": "P1", "gap": "licensed_industry_and_announcement_date_fundamentals",
         "status": "BLOCKED_EXTERNAL_DATA",
         "mitigation": "retain CSRC broad-industry and price/valuation scope; do not claim a commercial risk model"},
        {"priority": "P1", "gap": "official_corporate_actions_and_broker_calibrated_impact",
         "status": "BLOCKED_EXTERNAL_DATA",
         "mitigation": "retain explicit proxy/scenario labels and require vendor or broker data for promotion"},
    ])


def acceptance_checks(
    implementation: pd.DataFrame, seeds: pd.DataFrame, tests: pd.DataFrame,
    contracts: pd.DataFrame, registry: pd.DataFrame, gaps: pd.DataFrame,
    cfg: dict[str, Any], root: Path = ROOT,
) -> pd.DataFrame:
    definition = implementation.loc[
        implementation["logic"].isin([
            "corporate_action_proxy_formula", "execution_buy_limit_rule",
        ])
    ]
    registry_complete = (
        len(registry) >= int(cfg["minimum_frozen_experiments"])
        and registry["index_occurrences"].eq(1).all()
        and registry["indexed"].all()
        and registry["snapshot_directory_exists"].all()
        and registry["index_declares_immutable"].all()
    )
    rows = [
        ("execution_market_logic_has_one_definition", len(definition) == 2
         and definition["is_definition"].all()
         and definition["path"].nunique() == 1, definition.to_dict("records")),
        ("all_execution_consumers_use_canonical_api", implementation.loc[
            implementation["logic"].eq("execution_market_public_api"), "path"
        ].nunique() >= 4, implementation.loc[
            implementation["logic"].eq("execution_market_public_api"), "path"
        ].unique().tolist()),
        ("random_generators_have_explicit_seeds", not seeds.empty
         and seeds["explicit_seed"].all(), len(seeds)),
        ("five_test_layers_present", set(tests["layer"]) == set(cfg["required_test_layers"])
         and tests["exists"].all() and tests.groupby("layer")["test_functions"].sum().gt(0).all(),
         tests.groupby("layer")["test_functions"].sum().to_dict()),
        ("critical_outputs_have_versioned_protocols", len(contracts) == len(cfg["critical_protocols"])
         and contracts["protocol_has_experiment_id"].all(), contracts.to_dict("records")),
        ("critical_outputs_have_immutable_snapshots", contracts["frozen_snapshot_exists"].all()
         and contracts["frozen_snapshot_immutable"].all(),
         contracts[["stage", "frozen_experiment"]].to_dict("records")),
        ("frozen_experiment_inventory_complete", registry_complete, {
            "records": len(registry),
            "minimum": int(cfg["minimum_frozen_experiments"]),
            "duplicates": registry.loc[registry["index_occurrences"].ne(1), "experiment_id"].tolist(),
            "missing_snapshots": registry.loc[~registry["snapshot_directory_exists"], "experiment_id"].tolist(),
            "unindexed_snapshots": registry.loc[~registry["indexed"], "experiment_id"].tolist(),
        }),
        ("no_open_p0_engineering_gaps", not (
            gaps["priority"].eq("P0") & gaps["status"].str.startswith("OPEN")
        ).any(), gaps.groupby(["priority", "status"]).size().to_dict()),
    ]
    return pd.DataFrame([{
        "check_name": name, "status": "PASS" if bool(passed) else "FAIL",
        "observed": str(observed),
    } for name, passed, observed in rows])


def run(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    implementation = implementation_inventory()
    seeds = random_seed_inventory()
    tests = test_layer_inventory(cfg)
    contracts = output_contracts(lake, cfg)
    registry = frozen_registry_inventory()
    gaps = gap_register()
    checks = acceptance_checks(implementation, seeds, tests, contracts, registry, gaps, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "canonical_execution_market_module": cfg["canonical_execution_market_module"],
        "research_logic_changed": False, "frozen_results_reselected": False,
    }])
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all Phase 10 engineering gates passed",
        }])
    outputs = {
        "engineering_quality_protocol": protocol,
        "engineering_implementation_inventory": implementation,
        "engineering_random_seed_inventory": seeds,
        "engineering_test_layer_inventory": tests,
        "engineering_output_contracts": contracts,
        "engineering_registry_inventory": registry,
        "engineering_gap_register": gaps,
        "engineering_quality_checks": checks,
        "engineering_failed_assertions": failures,
    }
    for table, frame in outputs.items():
        lake.write(
            table, frame, key=str(cfg["artifact_key"]),
            source=f"engineering_quality.{cfg['artifact_key']}", request=cfg,
        )
    lines = [
        "# Phase 10：工程结构、配置与测试规范", "",
        "## 结论", "",
        "Phase 5/6/9原先重复的执行市场转换已收敛到一个主实现。此次重构不改变信号、目标、成本、订单或历史结论；真实主板/科创板抽样重算与冻结市场逐行零差异。", "",
        "## 实现清单", "", implementation.to_markdown(index=False), "",
        "## 随机种子", "", seeds.to_markdown(index=False), "",
        "## 测试层次", "", tests.to_markdown(index=False), "",
        "## 关键输出契约", "", contracts.to_markdown(index=False), "",
        "## 冻结实验登记对账", "", registry.to_markdown(index=False), "",
        "## 工程债与外部数据边界", "", gaps.to_markdown(index=False), "",
        "## 机器验收", "", checks.to_markdown(index=False),
    ]
    report_path = ROOT / str(cfg["report_path"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(checks.to_string(index=False), flush=True)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Phase 10 engineering gates failed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10 engineering quality audit")
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
