import numpy as np
import pandas as pd

from quant_research.size_neutralization import neutralize_design_to_size


def test_size_neutralization_removes_linear_size_exposure() -> None:
    size = pd.Series(np.exp(np.linspace(1.0, 5.0, 200)))
    log_size = np.log(size.to_numpy())
    design = np.column_stack([
        3.0 * log_size + np.sin(log_size),
        -2.0 * log_size + np.cos(log_size),
    ])
    neutral, size_z, valid = neutralize_design_to_size(design, size)
    correlations = [np.corrcoef(neutral[valid, index], size_z[valid])[0, 1]
                    for index in range(neutral.shape[1])]
    assert max(abs(value) for value in correlations) < 1e-12
    assert np.allclose(neutral.mean(axis=0), 0.0, atol=1e-12)


def test_size_neutralization_keeps_rows_with_missing_size() -> None:
    design = np.arange(40, dtype="float64").reshape(20, 2)
    size = pd.Series(np.exp(np.linspace(1.0, 3.0, 20)))
    size.iloc[0] = np.nan
    neutral, _, valid = neutralize_design_to_size(design, size)
    assert neutral.shape == design.shape
    assert not valid[0]
    assert np.isfinite(neutral).all()
