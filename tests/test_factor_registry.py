from __future__ import annotations

import copy

import pytest

from quant_research.factor_registry import load_registry, validate_registry


def test_registry_matches_model_features() -> None:
    registry = load_registry()
    expected = [entry["name"] for entry in registry["factors"]]
    validate_registry(registry, expected)


def test_registry_rejects_duplicate_names() -> None:
    registry = copy.deepcopy(load_registry())
    registry["factors"].append(copy.deepcopy(registry["factors"][0]))
    expected = [entry["name"] for entry in registry["factors"][:-1]]
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry(registry, expected)
