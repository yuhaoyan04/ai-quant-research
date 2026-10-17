from __future__ import annotations

import pandas as pd

from quant_research.factor_visual_diagnostics import (
    correlation_heatmap_svg, distribution_svg,
)


def test_heatmap_svg_contains_labels_and_values() -> None:
    matrix = pd.DataFrame([[1.0, -0.5], [-0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    svg = correlation_heatmap_svg(matrix, "test")
    assert svg.startswith("<svg")
    assert "a vs b: -0.500" in svg


def test_distribution_svg_handles_constant_factor() -> None:
    summary = pd.DataFrame([{
        "factor": "constant", "p01": 1.0, "p25": 1.0, "p50": 1.0,
        "p75": 1.0, "p99": 1.0, "missing_fraction": 0.0,
    }])
    assert "constant" in distribution_svg(summary)
