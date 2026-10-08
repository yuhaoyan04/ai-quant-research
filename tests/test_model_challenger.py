import numpy as np
import pandas as pd

from quant_research.model_challenger import blend_predictions, deterministic_indices, make_model


def test_deterministic_sampling_is_reproducible_and_bounded() -> None:
    frame = pd.DataFrame({
        "instrument_id": [f"s{i}" for i in range(100)],
        "asof_date": ["2020-01-03"] * 100,
    })
    first = deterministic_indices(frame, 20)
    second = deterministic_indices(frame, 20)
    assert len(first) == 20
    assert np.array_equal(np.sort(first), np.sort(second))


def test_challenger_disables_random_internal_early_stopping() -> None:
    model = make_model({
        "learning_rate": 0.1, "max_iter": 2, "max_leaf_nodes": 3,
        "min_samples_leaf": 2, "l2_regularization": 1.0,
    }, 1)
    assert model.early_stopping is False


def test_blend_endpoints_equal_component_ranks() -> None:
    base = pd.DataFrame({
        "instrument_id": ["a", "b"], "asof_date": ["2020-01-03"] * 2,
        "available_date": ["2020-01-06"] * 2, "target": [0.0, 1.0],
    })
    tree = base.assign(score=[2.0, 1.0])
    ridge = base.assign(score=[1.0, 2.0])
    tree_only = blend_predictions(tree, ridge, 1.0, "target")
    ridge_only = blend_predictions(tree, ridge, 0.0, "target")
    assert tree_only.loc[0, "score"] > tree_only.loc[1, "score"]
    assert ridge_only.loc[0, "score"] < ridge_only.loc[1, "score"]
