from __future__ import annotations

import copy

import pytest

from quant_research.hgb_hyperparameter_stability import (
    candidate_specs, load_config, validate_surface,
)


def test_surface_is_complete_and_contains_baseline() -> None:
    cfg = load_config()
    specs = candidate_specs(cfg)
    validate_surface(specs, cfg)
    assert len(specs) == 9


def test_surface_rejects_missing_coordinate() -> None:
    cfg = copy.deepcopy(load_config())
    specs = candidate_specs(cfg)[:-1]
    with pytest.raises(ValueError, match="Cartesian"):
        validate_surface(specs, cfg)
