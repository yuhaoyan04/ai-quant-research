from __future__ import annotations

import pandas as pd

from quant_research.research_conclusion_audit import (
    factor_multiplicity,
    lifecycle_audit,
)


class FrameLake:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def read(self, table: str, key: str) -> pd.DataFrame:
        assert (table, key) == ("factor_tear_sheet_bootstrap", "v1")
        return self.frame.copy()


def test_factor_multiplicity_applies_bh_within_selected_family() -> None:
    frame = pd.DataFrame({
        "pipeline": ["industry_size_neutralized"] * 3 + ["raw"],
        "metric": ["gross_top_minus_bottom"] * 4,
        "factor": ["a", "b", "c", "ignored"],
        "point_estimate": [0.3, 0.2, 0.1, 1.0],
        "ci_2_5pct": [0.1, 0.01, -0.1, 0.9],
        "ci_97_5pct": [0.5, 0.4, 0.3, 1.1],
        "one_sided_probability_nonpositive": [0.001, 0.02, 0.2, 0.001],
    })

    result = factor_multiplicity(FrameLake(frame), 0.05)

    assert set(result["factor"]) == {"a", "b", "c"}
    assert result.set_index("factor").loc["a", "survives_bh_fdr"]
    assert result.set_index("factor").loc["b", "survives_bh_fdr"]
    assert not result.set_index("factor").loc["c", "survives_bh_fdr"]


def test_lifecycle_distinguishes_research_from_deployment() -> None:
    lifecycle = lifecycle_audit().set_index("stage")

    assert lifecycle.loc["reproducibility_and_governance", "status"] == "COMPLETE"
    assert lifecycle.loc["paper_live_monitoring", "status"] == "NEXT_STAGE"
