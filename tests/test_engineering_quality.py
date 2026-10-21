from __future__ import annotations

from pathlib import Path
import json

from quant_research.engineering_quality import (
    frozen_registry_inventory, implementation_inventory, random_seed_inventory,
)


def test_execution_market_formula_has_one_canonical_definition() -> None:
    inventory = implementation_inventory()
    definitions = inventory.loc[inventory["logic"].isin([
        "corporate_action_proxy_formula", "execution_buy_limit_rule",
    ])]
    assert len(definitions) == 2
    assert definitions["is_definition"].all()
    assert definitions["path"].unique().tolist() == [
        "src/quant_research/execution_market.py"
    ]


def test_all_discovered_random_generators_have_explicit_seed_expression() -> None:
    inventory = random_seed_inventory()
    assert not inventory.empty
    assert inventory["explicit_seed"].all()


def test_registry_inventory_reconciles_ids_not_a_brittle_exact_count(tmp_path: Path) -> None:
    registry = tmp_path / "reports" / "experiments"
    registry.mkdir(parents=True)
    payload = [
        {"experiment_id": "experiment_a", "immutable": True},
        {"experiment_id": "experiment_b", "immutable": True},
    ]
    (registry / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    for experiment_id in ("experiment_a", "experiment_b"):
        directory = registry / experiment_id
        directory.mkdir()
        (directory / "FROZEN.json").write_text("{}", encoding="utf-8")
    inventory = frozen_registry_inventory(tmp_path)
    assert len(inventory) == 2
    assert inventory["index_occurrences"].eq(1).all()
    assert inventory["snapshot_directory_exists"].all()
    assert inventory["index_declares_immutable"].all()


def test_registry_inventory_exposes_orphan_and_duplicate_entries(tmp_path: Path) -> None:
    registry = tmp_path / "reports" / "experiments"
    registry.mkdir(parents=True)
    payload = [
        {"experiment_id": "experiment_a", "immutable": True},
        {"experiment_id": "experiment_a", "immutable": True},
    ]
    (registry / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    for experiment_id in ("experiment_a", "orphan_snapshot"):
        directory = registry / experiment_id
        directory.mkdir()
        (directory / "FROZEN.json").write_text("{}", encoding="utf-8")
    inventory = frozen_registry_inventory(tmp_path).set_index("experiment_id")
    assert inventory.loc["experiment_a", "index_occurrences"] == 2
    assert not bool(inventory.loc["orphan_snapshot", "indexed"])
