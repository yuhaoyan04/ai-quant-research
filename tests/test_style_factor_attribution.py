import numpy as np
import pandas as pd

from quant_research.style_factor_attribution import (
    residual_factor_weights,
    simulate_factor_returns,
)


def test_residual_factor_weights_are_self_financing_and_control_neutral() -> None:
    rng = np.random.default_rng(11)
    rows = 300
    industry = pd.Series(np.repeat(["A", "B", "C"], rows // 3))
    controls = rng.normal(size=(rows, 3))
    target = (
        controls @ np.array([0.8, -0.4, 0.2])
        + industry.map({"A": -1.0, "B": 0.2, "C": 1.2}).to_numpy()
        + rng.normal(size=rows)
    )
    styles = pd.DataFrame({
        "target": target, "one": controls[:, 0],
        "two": controls[:, 1], "three": controls[:, 2],
    })

    weights, diagnostics = residual_factor_weights(styles, industry, "target")

    assert abs(weights.sum()) < 1e-12
    assert np.isclose(np.clip(weights, 0, None).sum(), 1)
    assert np.isclose(np.clip(-weights, 0, None).sum(), 1)
    assert diagnostics["maximum_abs_industry_weight"] < 1e-12
    assert diagnostics["maximum_abs_control_exposure"] < 1e-12
    assert diagnostics["target_style_exposure"] > 0


def test_factor_return_begins_after_execution_close() -> None:
    targets = pd.DataFrame({
        "instrument_id": ["a", "b"],
        "signal_date": ["2024-01-01"] * 2,
        "execution_date": ["2024-01-02"] * 2,
        "weight_size": [1.0, -1.0],
    })
    market = pd.DataFrame({
        "instrument_id": ["a", "b", "a", "b"],
        "trade_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "daily_return": [0.10, -0.10, 0.03, -0.02],
    })

    result = simulate_factor_returns(targets, market, ["size"])

    assert result.iloc[0]["factor_return"] == 0
    assert np.isclose(result.iloc[1]["factor_return"], 0.05)
