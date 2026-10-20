from __future__ import annotations

from quant_research.engineering_quality import implementation_inventory, random_seed_inventory


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
