from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.statistical_inference import (
    _stable_seed,
    bootstrap_mean_record,
    hac_mean_record,
)


def test_hac_mean_reports_scaled_mean_and_valid_interval() -> None:
    values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02, 0.01] * 10)
    result = hac_mean_record(values, scale=252.0, lags=3, confidence_level=0.95)
    assert result["point_estimate"] == pytest.approx(values.mean() * 252.0)
    assert result["observations"] == 60
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]
    assert 0.0 <= result["one_sided_p_value"] <= 1.0


def test_block_bootstrap_is_reproducible_for_hypothesis_seed() -> None:
    values = pd.Series(np.linspace(-0.02, 0.03, 100))
    seed = _stable_seed(20261020, "family", "hypothesis", "20")
    first = bootstrap_mean_record(values, scale=252, block_length=20, samples=200, seed=seed)
    second = bootstrap_mean_record(values, scale=252, block_length=20, samples=200, seed=seed)
    assert first == second


def test_different_family_names_receive_different_stable_seeds() -> None:
    assert _stable_seed(7, "family_a", "x") != _stable_seed(7, "family_b", "x")
