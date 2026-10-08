import numpy as np

from quant_research.robustness import circular_block_bootstrap_mean


def test_block_bootstrap_is_reproducible_and_preserves_constant_mean() -> None:
    values = np.full(30, 0.02)
    first = circular_block_bootstrap_mean(values, 5, 100, 7)
    second = circular_block_bootstrap_mean(values, 5, 100, 7)
    assert np.array_equal(first, second)
    assert np.allclose(first, 0.02)
