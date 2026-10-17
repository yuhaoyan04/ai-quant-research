import numpy as np
import pandas as pd

from quant_research.factor_neutralization import (
    _safe_correlation,
    _tail_masks,
    apply_training_directions,
    build_daily_diagnostics,
    conclusion_table,
    neutralize_factor_matrix,
    paired_bootstrap,
)
from quant_research.ingest import Lake
from quant_research.factor_preprocessing_ablation import _rank_ic


def test_industry_size_neutralization_removes_both_linear_exposures() -> None:
    rows = 240
    industry = pd.Series(np.repeat(["A", "B", "C", "D"], rows // 4))
    size = pd.Series(np.exp(np.linspace(2.0, 8.0, rows)))
    size_z = (np.log(size) - np.log(size).mean()) / np.log(size).std(ddof=1)
    industry_effect = industry.map({"A": -2.0, "B": -0.5, "C": 0.5, "D": 2.0})
    design = np.column_stack([
        3.0 * size_z + industry_effect + np.sin(np.arange(rows)),
        -2.0 * size_z + 0.5 * industry_effect + np.cos(np.arange(rows)),
    ])

    neutral, diagnostics = neutralize_factor_matrix(
        design, industry=industry, size=size, minimum_industry_group_size=5
    )

    valid = diagnostics["fit_valid"]
    for column in range(neutral.shape[1]):
        assert abs(np.corrcoef(
            neutral[valid, column], diagnostics["size_z"][valid]
        )[0, 1]) < 1e-12
        means = pd.DataFrame({
            "value": neutral[:, column], "industry": industry,
        }).groupby("industry")["value"].mean()
        assert means.abs().max() < 1e-12


def test_industry_neutralization_keeps_missing_industry_rows_finite() -> None:
    industry = pd.Series(["A"] * 10 + ["B"] * 10 + [None])
    design = np.arange(42, dtype="float64").reshape(21, 2)

    neutral, diagnostics = neutralize_factor_matrix(
        design, industry=industry, minimum_industry_group_size=2
    )

    assert not diagnostics["fit_valid"][-1]
    assert np.isfinite(neutral).all()


def test_safe_correlation_returns_nan_for_constant_factor() -> None:
    assert np.isnan(_safe_correlation(np.ones(10), np.arange(10, dtype="float64")))


def test_rank_ic_returns_nan_for_constant_factor_without_warning() -> None:
    assert np.isnan(_rank_ic(np.ones(10), pd.Series(np.arange(10, dtype="float64"))))


def test_tail_masks_follow_configured_fraction() -> None:
    top, bottom = _tail_masks(np.arange(100, dtype="float64"), 0.2)

    assert top.sum() == 20
    assert bottom.sum() == 20
    assert np.flatnonzero(top)[0] == 80
    assert np.flatnonzero(bottom)[-1] == 19


def test_factor_direction_is_fixed_from_raw_training_data() -> None:
    daily = pd.DataFrame({
        "split": ["train", "train", "historical_oos", "historical_oos"],
        "pipeline": ["raw", "raw", "raw", "industry_size_neutralized"],
        "factor": ["negative_factor"] * 4,
        "rank_ic": [-0.1, -0.2, -0.05, -0.02],
        "top_minus_bottom": [-0.02, -0.03, -0.01, -0.005],
    })

    oriented, directions = apply_training_directions(daily)

    assert directions.iloc[0]["direction"] == -1
    assert oriented.loc[2, "oriented_rank_ic"] == 0.05
    assert oriented.loc[3, "oriented_rank_ic"] == 0.02


def test_paired_bootstrap_drops_undefined_constant_factor_dates() -> None:
    dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI").strftime("%Y-%m-%d")
    records = []
    for pipeline in ("raw", "industry_neutralized"):
        for index, asof_date in enumerate(dates):
            value = np.nan if index == 0 else 0.01 * (index + 1)
            records.append({
                "asof_date": asof_date,
                "split": "historical_oos",
                "pipeline": pipeline,
                "factor": "factor_1",
                "rank_ic": value,
                "oriented_rank_ic": value,
                "top_minus_bottom": value,
                "oriented_top_minus_bottom": value,
            })
    cfg = {
        "pipelines": ["raw", "industry_neutralized"],
        "bootstrap": {
            "block_length_weeks": 2, "samples": 20, "random_seed": 7,
        },
    }

    result = paired_bootstrap(pd.DataFrame(records), cfg)

    assert result["observations"].eq(4).all()
    assert result["point_estimate"].eq(0).all()


def test_conclusion_table_uses_oriented_ic_confidence_interval() -> None:
    summary = pd.DataFrame([
        {"split": "historical_oos", "pipeline": pipeline, "factor": "factor_1",
         "mean_oriented_rank_ic": ic, "mean_oriented_top_minus_bottom": spread,
         "mean_size_correlation": size, "mean_industry_max_abs_mean": industry}
        for pipeline, ic, spread, size, industry in (
            ("raw", 0.02, 0.001, 0.3, 0.2),
            ("industry_size_neutralized", 0.04, 0.002, 0.0, 0.0),
        )
    ])
    bootstrap = pd.DataFrame([
        {"pipeline": "industry_size_neutralized", "factor": "factor_1",
         "metric": metric, "point_estimate": estimate,
         "ci_2_5pct": low, "ci_97_5pct": high}
        for metric, estimate, low, high in (
            ("oriented_rank_ic", 0.02, 0.01, 0.03),
            ("oriented_top_minus_bottom", 0.001, -0.001, 0.003),
        )
    ])
    directions = pd.DataFrame({
        "factor": ["factor_1"], "direction": [1],
        "training_raw_mean_rank_ic": [0.03],
    })

    result = conclusion_table(summary, bootstrap, directions)

    assert result.iloc[0]["rank_ic_effect"] == "strengthened"
    assert result.iloc[0]["rank_ic_delta"] == 0.02


def test_factor_diagnostics_keep_paired_samples_and_remove_industry_means(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    rng = np.random.default_rng(7)
    rows = []
    sizes = []
    industries = []
    for asof_date in ("2024-01-05", "2024-01-12"):
        for index in range(120):
            group = "A" if index < 60 else "B"
            factor_1 = (2.0 if group == "B" else -2.0) + rng.normal()
            factor_2 = index / 120 + rng.normal(scale=0.1)
            rows.append({
                "instrument_id": f"sh.{600000 + index:06d}",
                "asof_date": asof_date,
                "factor_1": factor_1,
                "factor_2": factor_2,
                "target": 0.01 * factor_1 + rng.normal(scale=0.02),
            })
            sizes.append({
                "instrument_id": f"sh.{600000 + index:06d}",
                "asof_date": asof_date,
                # A different missing risk exposure verifies that each
                # residualizer is fitted on the same complete-case sample.
                "float_cap_proxy_cny": (
                    np.nan if index == 118 else float(np.exp(10 + index / 120))
                ),
            })
            industries.append({
                "instrument_id": f"sh.{600000 + index:06d}",
                "asof_date": asof_date,
                # One missing exposure verifies that every pipeline uses the
                # same complete-case cross section.
                "industry_broad_code": None if index == 119 else group,
            })
    lake.write(
        "multi_horizon_factor_panel", pd.DataFrame(rows), key="2024",
        source="test", request={},
    )
    lake.write(
        "pit_size_exposures", pd.DataFrame(sizes), key="2024",
        source="test", request={},
    )
    lake.write(
        "pit_industry_exposures", pd.DataFrame(industries), key="2024",
        source="test", request={},
    )
    cfg = {
        "train": {"start_year": 2024, "end_year": 2024},
        "validation": {"end_year": 2024},
        "historical_oos": {"end_year": 2024},
        "minimum_cross_section_size": 100,
        "minimum_industry_group_size": 5,
        "quantile_limits": [0.01, 0.99],
        "mad_multiplier": 5.0,
        "factor_top_quantile": 0.2,
    }

    daily = build_daily_diagnostics(
        lake, cfg, ["factor_1", "factor_2"], "target"
    )

    assert len(daily) == 2 * 4 * 2
    assert daily["cross_section_size"].eq(118).all()
    neutral = daily.loc[daily["pipeline"].eq("industry_neutralized")]
    assert neutral["industry_max_abs_mean"].max() < 1e-10
