import pandas as pd
import pytest

from quant_research.feature_analysis import connected_correlation_clusters, daily_diagnostics


def test_daily_diagnostics_detects_redundancy_and_marginal_signal() -> None:
    rows = []
    for day in ("2020-01-02", "2020-01-03"):
        for rank in range(1, 121):
            rows.append({
                "asof_date": day,
                "alpha__z": (rank - 60.5) / 35,
                "copy__z": (rank - 60.5) / 35,
                "noise__z": (-1) ** rank,
                "target": rank / 10000,
            })
    frame = pd.DataFrame(rows)
    correlation_sum, days, betas = daily_diagnostics(
        frame, ["alpha__z", "copy__z", "noise__z"], "target", 100, 0.001)
    matrix = correlation_sum / days
    assert days == 2
    assert matrix[0, 1] > 0.99
    assert betas.groupby("factor")["marginal_rank_coefficient"].mean()["noise"] == pytest.approx(0, abs=0.02)


def test_connected_clusters_join_highly_correlated_factors() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )
    clusters = connected_correlation_clusters(matrix, 0.75)
    indexed = clusters.set_index("factor")
    assert indexed.loc["a", "cluster"] == indexed.loc["b", "cluster"]
    assert indexed.loc["c", "cluster_size"] == 1
