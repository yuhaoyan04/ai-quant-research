import numpy as np
import pandas as pd

from quant_research.preprocessing import preprocess_cross_section


def test_preprocessing_is_date_local_and_preserves_missingness() -> None:
    frame = pd.DataFrame({
        "asof_date": ["2024-01-02"] * 5 + ["2024-01-03"] * 5,
        "factor": [1.0, 2.0, 3.0, 4.0, np.nan, 100.0, 200.0, 300.0, 400.0, np.nan],
    })
    result = preprocess_cross_section(frame, ["factor"], mad_multiplier=5.0)
    assert result.features["factor__missing"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert abs(result.features.loc[:4, "factor__z"].mean()) < 1e-6
    assert abs(result.features.loc[5:, "factor__z"].mean()) < 1e-6
    assert result.features["factor__z"].notna().all()


def test_extreme_value_is_mad_clipped_and_audited() -> None:
    frame = pd.DataFrame({
        "asof_date": ["2024-01-02"] * 7,
        "factor": [1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 1000.0],
    })
    result = preprocess_cross_section(frame, ["factor"], mad_multiplier=3.0)
    audit = result.audit.iloc[0]
    assert audit["clipped_count"] == 1
    assert result.features["factor__z"].max() < 3.0


def test_all_missing_factor_becomes_neutral_with_indicator() -> None:
    frame = pd.DataFrame({"asof_date": ["2024-01-02"] * 3, "factor": [np.nan] * 3})
    result = preprocess_cross_section(frame, ["factor"])
    assert result.features["factor__z"].eq(0).all()
    assert result.features["factor__missing"].eq(1).all()
