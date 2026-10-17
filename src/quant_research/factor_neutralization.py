"""Paired PIT industry/size neutralization diagnostics for individual factors."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import _rank_ic, transform_cross_section
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import weekly_rows
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "factor_neutralization.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def standardized_log_size(size: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(size, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(raw) & (raw > 0)
    if valid.sum() < 3:
        raise ValueError("not enough valid size observations")
    log_size = np.log(raw[valid])
    lower, upper = np.quantile(log_size, [0.01, 0.99])
    clipped = np.clip(log_size, lower, upper)
    deviation = clipped.std(ddof=1)
    if not np.isfinite(deviation) or deviation <= 0:
        raise ValueError("same-date size exposure has no dispersion")
    result = np.zeros(len(raw), dtype="float64")
    result[valid] = (clipped - clipped.mean()) / deviation
    return result, valid


def consolidated_industry_groups(
    industry: pd.Series, minimum_group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = industry.fillna("").astype(str).str.strip()
    valid = values.ne("").to_numpy()
    counts = values.loc[valid].value_counts()
    rare = set(counts.loc[counts.lt(minimum_group_size)].index)
    groups = values.to_numpy(dtype=object)
    if rare:
        groups[np.isin(groups, list(rare))] = "__OTHER__"
    groups[~valid] = ""
    return groups.astype(str), valid


def neutralize_factor_matrix(
    design: np.ndarray,
    *,
    industry: pd.Series | None = None,
    size: pd.Series | None = None,
    minimum_industry_group_size: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Residualize factor columns on same-date industry dummies and/or log size."""
    if industry is None and size is None:
        raise ValueError("at least one risk exposure is required")
    row_count = len(design)
    fit_valid = np.ones(row_count, dtype=bool)
    full_columns = [np.ones(row_count, dtype="float64")]
    diagnostics: dict[str, np.ndarray] = {}

    if size is not None:
        size_z, valid_size = standardized_log_size(size)
        fit_valid &= valid_size
        full_columns.append(size_z)
        diagnostics["size_z"] = size_z
        diagnostics["valid_size"] = valid_size

    if industry is not None:
        groups, valid_industry = consolidated_industry_groups(
            industry, minimum_industry_group_size
        )
        fit_valid &= valid_industry
        categories = sorted(set(groups[valid_industry]))
        if len(categories) < 2:
            raise ValueError("industry neutralization requires at least two groups")
        # Drop the first group; the intercept captures its mean and avoids a
        # singular full-dummy design while preserving the same projection.
        full_columns.extend(
            (groups == category).astype("float64") for category in categories[1:]
        )
        diagnostics["industry_groups"] = groups
        diagnostics["valid_industry"] = valid_industry

    if fit_valid.sum() <= len(full_columns):
        raise ValueError("not enough fully observed rows for risk neutralization")
    regressors = np.column_stack(full_columns)
    coefficients = np.linalg.lstsq(
        regressors[fit_valid], design[fit_valid], rcond=None
    )[0]
    residual = design - regressors @ coefficients
    center = residual[fit_valid].mean(axis=0)
    scale = residual[fit_valid].std(axis=0, ddof=1)
    neutral = np.divide(
        residual - center,
        scale,
        out=np.zeros_like(residual),
        where=scale > 0,
    )
    if not np.isfinite(neutral).all():
        raise ValueError("risk neutralization produced non-finite factor values")
    diagnostics["fit_valid"] = fit_valid
    return neutral, diagnostics


def transformed_factor_designs(
    group: pd.DataFrame, factors: list[str], cfg: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw, _ = transform_cross_section(
        group, factors,
        winsor_method="quantile",
        normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=False,
    )
    _, valid_size = standardized_log_size(group["float_cap_proxy_cny"])
    _, valid_industry = consolidated_industry_groups(
        group["industry_broad_code"], int(cfg["minimum_industry_group_size"])
    )
    # Fit every residualizer on the same exposure-complete cross section used
    # for evaluation.  Otherwise even a single missing size row can make the
    # industry-only residual lose exact within-industry orthogonality after the
    # final common-sample filter.
    exposure_complete = valid_size & valid_industry
    common_size = group["float_cap_proxy_cny"].where(exposure_complete)
    common_industry = group["industry_broad_code"].where(exposure_complete)
    size_z, common_valid_size = standardized_log_size(common_size)
    industry_groups, common_valid_industry = consolidated_industry_groups(
        common_industry, int(cfg["minimum_industry_group_size"])
    )
    if not np.array_equal(common_valid_size, exposure_complete):
        raise RuntimeError("common size sample does not match exposure-complete sample")
    if not np.array_equal(common_valid_industry, exposure_complete):
        raise RuntimeError("common industry sample does not match exposure-complete sample")
    size_neutral, _ = neutralize_factor_matrix(
        raw, size=common_size
    )
    industry_neutral, _ = neutralize_factor_matrix(
        raw, industry=common_industry,
        minimum_industry_group_size=int(cfg["minimum_industry_group_size"]),
    )
    combined_neutral, _ = neutralize_factor_matrix(
        raw,
        industry=common_industry,
        size=common_size,
        minimum_industry_group_size=int(cfg["minimum_industry_group_size"]),
    )
    return {
        "raw": raw,
        "size_neutralized": size_neutral,
        "industry_neutralized": industry_neutral,
        "industry_size_neutralized": combined_neutral,
    }, {
        "size_z": size_z,
        "valid_size": valid_size,
        "industry_groups": industry_groups,
        "valid_industry": valid_industry,
    }


def _read_year(lake: Lake, year: int, columns: list[str]) -> pd.DataFrame:
    panel = weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=columns
    ))
    size = pd.read_parquet(
        lake.table_path("pit_size_exposures", str(year)),
        columns=["instrument_id", "asof_date", "float_cap_proxy_cny"],
    )
    industry = pd.read_parquet(
        lake.table_path("pit_industry_exposures", str(year)),
        columns=["instrument_id", "asof_date", "industry_broad_code"],
    )
    return panel.merge(
        size, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    ).merge(
        industry, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )


def _split_for_year(year: int, cfg: dict) -> str:
    if year <= int(cfg["train"]["end_year"]):
        return "train"
    if year <= int(cfg["validation"]["end_year"]):
        return "validation"
    return "historical_oos"


def _industry_max_abs_mean(
    values: np.ndarray, groups: np.ndarray, valid: np.ndarray,
) -> float:
    frame = pd.DataFrame({"value": values[valid], "industry": groups[valid]})
    return float(frame.groupby("industry")["value"].mean().abs().max())


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return NaN for an undefined constant-vector correlation, without warnings."""
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3:
        return np.nan
    left_usable, right_usable = left[usable], right[usable]
    if left_usable.std(ddof=1) <= 0 or right_usable.std(ddof=1) <= 0:
        return np.nan
    return float(np.corrcoef(left_usable, right_usable)[0, 1])


def _tail_masks(values: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return percentile masks for matching upper and lower factor tails."""
    if not 0 < fraction <= 0.5:
        raise ValueError("factor_top_quantile must be in (0, 0.5]")
    percentile = pd.Series(values).rank(method="average", pct=True)
    return percentile.gt(1.0 - fraction).to_numpy(), percentile.le(fraction).to_numpy()


def build_daily_diagnostics(
    lake: Lake, cfg: dict, factors: list[str], target: str,
) -> pd.DataFrame:
    records: list[dict] = []
    prior_top: dict[tuple[str, str], set[str]] = {}
    start_year = int(cfg["train"]["start_year"])
    end_year = int(cfg["historical_oos"]["end_year"])
    columns = ["instrument_id", "asof_date", target, *factors]
    top_fraction = float(cfg["factor_top_quantile"])
    for year in range(start_year, end_year + 1):
        frame = _read_year(lake, year, columns)
        split = _split_for_year(year, cfg)
        for asof_date, group in frame.groupby("asof_date", sort=True):
            group = group.reset_index(drop=True)
            returns = pd.to_numeric(group[target], errors="coerce")
            designs, risks = transformed_factor_designs(group, factors, cfg)
            # All four pipelines use one exposure-complete sample.  This keeps
            # paired IC/return differences from being contaminated by changing
            # the stock universe when industry or size is unavailable.
            exposure_complete = risks["valid_size"] & risks["valid_industry"]
            usable = returns.notna().to_numpy() & exposure_complete
            if usable.sum() < int(cfg["minimum_cross_section_size"]):
                continue
            usable_instruments = group.loc[
                usable, "instrument_id"
            ].astype(str).reset_index(drop=True)
            for pipeline, design in designs.items():
                for factor_index, factor in enumerate(factors):
                    values = design[:, factor_index]
                    usable_values = values[usable]
                    usable_returns = returns.loc[usable].reset_index(drop=True)
                    top, bottom = _tail_masks(usable_values, top_fraction)
                    current_top_count = max(
                        1, int(np.ceil(len(usable_values) * top_fraction))
                    )
                    selected_index = np.argsort(
                        -usable_values, kind="stable"
                    )[:current_top_count]
                    selected = set(usable_instruments.iloc[selected_index])
                    state_key = (pipeline, factor)
                    previous = prior_top.get(state_key)
                    turnover = (
                        np.nan if previous is None
                        else 1.0 - len(previous & selected) / max(len(previous), 1)
                    )
                    prior_top[state_key] = selected
                    valid_size = exposure_complete
                    size_correlation = _safe_correlation(
                        values[valid_size], risks["size_z"][valid_size]
                    )
                    records.append({
                        "asof_date": str(asof_date),
                        "year": year,
                        "split": split,
                        "pipeline": pipeline,
                        "factor": factor,
                        "cross_section_size": int(usable.sum()),
                        "raw_factor_coverage": float(group[factor].notna().mean()),
                        "industry_coverage": float(risks["valid_industry"].mean()),
                        "size_coverage": float(valid_size.mean()),
                        "rank_ic": _rank_ic(usable_values, usable_returns),
                        "top_minus_bottom": float(
                            usable_returns.loc[top].mean()
                            - usable_returns.loc[bottom].mean()
                        ),
                        "top_quantile_turnover": turnover,
                        "size_correlation": size_correlation,
                        "industry_max_abs_mean": _industry_max_abs_mean(
                            values, risks["industry_groups"], exposure_complete
                        ),
                    })
        print(
            f"factor neutralization year={year} dates={frame['asof_date'].nunique()}",
            flush=True,
        )
    return pd.DataFrame(records)


def apply_training_directions(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient every factor using only its raw training-period RankIC sign."""
    training_raw = daily.loc[
        daily["split"].eq("train") & daily["pipeline"].eq("raw")
    ]
    means = training_raw.groupby("factor")["rank_ic"].mean()
    missing = sorted(set(daily["factor"]) - set(means.dropna().index))
    if missing:
        raise RuntimeError(f"factor directions unavailable from training data: {missing}")
    directions = pd.DataFrame({
        "factor": means.index,
        "training_raw_mean_rank_ic": means.to_numpy(dtype="float64"),
        "direction": np.where(means.to_numpy(dtype="float64") >= 0, 1, -1),
        "direction_rule": "sign_of_raw_training_mean_rank_ic",
    }).reset_index(drop=True)
    result = daily.merge(
        directions[["factor", "direction"]], on="factor", how="left",
        validate="many_to_one",
    )
    result["oriented_rank_ic"] = result["rank_ic"] * result["direction"]
    result["oriented_top_minus_bottom"] = (
        result["top_minus_bottom"] * result["direction"]
    )
    return result, directions


def summarize_daily(daily: pd.DataFrame, annualization_periods: int) -> pd.DataFrame:
    records = []
    for (split, pipeline, factor), group in daily.groupby(
        ["split", "pipeline", "factor"], sort=False
    ):
        ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        standard_deviation = float(ic.std(ddof=1))
        records.append({
            "split": split,
            "pipeline": pipeline,
            "factor": factor,
            "observations": len(ic),
            "mean_rank_ic": float(ic.mean()),
            "rank_ic_std": standard_deviation,
            "annualized_icir": (
                float(ic.mean() / standard_deviation * np.sqrt(annualization_periods))
                if standard_deviation > 0 else np.nan
            ),
            "positive_rank_ic_fraction": float(ic.gt(0).mean()),
            "mean_oriented_rank_ic": float(group["oriented_rank_ic"].mean()),
            "mean_top_minus_bottom": float(group["top_minus_bottom"].mean()),
            "mean_oriented_top_minus_bottom": float(
                group["oriented_top_minus_bottom"].mean()
            ),
            "mean_top_quantile_turnover": float(group["top_quantile_turnover"].mean()),
            "mean_size_correlation": float(group["size_correlation"].mean()),
            "mean_industry_max_abs_mean": float(group["industry_max_abs_mean"].mean()),
            "mean_raw_factor_coverage": float(group["raw_factor_coverage"].mean()),
            "mean_industry_coverage": float(group["industry_coverage"].mean()),
            "mean_size_coverage": float(group["size_coverage"].mean()),
        })
    return pd.DataFrame(records)


def annual_summary(daily: pd.DataFrame, annualization_periods: int) -> pd.DataFrame:
    records = []
    for (year, split, pipeline, factor), group in daily.groupby(
        ["year", "split", "pipeline", "factor"], sort=False
    ):
        ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        standard_deviation = float(ic.std(ddof=1))
        records.append({
            "year": int(year), "split": split, "pipeline": pipeline, "factor": factor,
            "observations": len(ic), "mean_rank_ic": float(ic.mean()),
            "rank_ic_std": standard_deviation,
            "annualized_icir": (
                float(ic.mean() / standard_deviation * np.sqrt(annualization_periods))
                if standard_deviation > 0 else np.nan
            ),
            "mean_oriented_rank_ic": float(group["oriented_rank_ic"].mean()),
            "mean_top_minus_bottom": float(group["top_minus_bottom"].mean()),
            "mean_oriented_top_minus_bottom": float(
                group["oriented_top_minus_bottom"].mean()
            ),
            "mean_top_quantile_turnover": float(group["top_quantile_turnover"].mean()),
        })
    return pd.DataFrame(records)


def paired_bootstrap(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    oos = daily.loc[daily["split"].eq("historical_oos")]
    bootstrap_cfg = cfg["bootstrap"]
    records = []
    for factor in sorted(oos["factor"].unique()):
        raw = oos.loc[
            oos["factor"].eq(factor) & oos["pipeline"].eq("raw"),
            [
                "asof_date", "rank_ic", "oriented_rank_ic",
                "top_minus_bottom", "oriented_top_minus_bottom",
            ],
        ]
        for pipeline in cfg["pipelines"]:
            if pipeline == "raw":
                continue
            challenger = oos.loc[
                oos["factor"].eq(factor) & oos["pipeline"].eq(pipeline),
                [
                    "asof_date", "rank_ic", "oriented_rank_ic",
                    "top_minus_bottom", "oriented_top_minus_bottom",
                ],
            ]
            paired = challenger.merge(
                raw, on="asof_date", suffixes=("_neutral", "_raw"),
                validate="one_to_one",
            )
            for metric in (
                "rank_ic", "oriented_rank_ic",
                "top_minus_bottom", "oriented_top_minus_bottom",
            ):
                difference = pd.to_numeric(
                    paired[f"{metric}_neutral"] - paired[f"{metric}_raw"],
                    errors="coerce",
                ).dropna().to_numpy(dtype="float64")
                if len(difference) < int(bootstrap_cfg["block_length_weeks"]):
                    raise RuntimeError(
                        f"not enough paired observations for {factor}/{pipeline}/{metric}: "
                        f"{len(difference)}"
                    )
                estimates = circular_block_bootstrap_mean(
                    difference,
                    int(bootstrap_cfg["block_length_weeks"]),
                    int(bootstrap_cfg["samples"]),
                    int(bootstrap_cfg["random_seed"]),
                )
                records.append({
                    "factor": factor,
                    "pipeline": pipeline,
                    "reference": "raw",
                    "metric": metric,
                    "observations": len(difference),
                    "point_estimate": float(difference.mean()),
                    "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                    "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                    "one_sided_probability_nonpositive": float(
                        (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                    ),
                    "block_length_weeks": int(bootstrap_cfg["block_length_weeks"]),
                })
    return pd.DataFrame(records)


def conclusion_table(
    summary: pd.DataFrame, bootstrap: pd.DataFrame, directions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the locked OOS raw-vs-combined comparison used in the report."""
    oos = summary.loc[summary["split"].eq("historical_oos")]
    metric_columns = [
        "factor", "mean_oriented_rank_ic", "mean_oriented_top_minus_bottom",
        "mean_size_correlation", "mean_industry_max_abs_mean",
    ]
    raw = oos.loc[oos["pipeline"].eq("raw"), metric_columns].rename(columns={
        column: f"raw_{column}" for column in metric_columns if column != "factor"
    })
    combined = oos.loc[
        oos["pipeline"].eq("industry_size_neutralized"), metric_columns
    ].rename(columns={
        column: f"neutral_{column}" for column in metric_columns if column != "factor"
    })
    result = raw.merge(combined, on="factor", validate="one_to_one").merge(
        directions[["factor", "direction", "training_raw_mean_rank_ic"]],
        on="factor", validate="one_to_one",
    )
    for metric, prefix in (
        ("oriented_rank_ic", "rank_ic"),
        ("oriented_top_minus_bottom", "spread"),
    ):
        interval = bootstrap.loc[
            bootstrap["pipeline"].eq("industry_size_neutralized")
            & bootstrap["metric"].eq(metric),
            ["factor", "point_estimate", "ci_2_5pct", "ci_97_5pct"],
        ].rename(columns={
            "point_estimate": f"{prefix}_delta",
            "ci_2_5pct": f"{prefix}_delta_ci_low",
            "ci_97_5pct": f"{prefix}_delta_ci_high",
        })
        result = result.merge(interval, on="factor", validate="one_to_one")
    result["rank_ic_effect"] = np.select(
        [result["rank_ic_delta_ci_low"].gt(0),
         result["rank_ic_delta_ci_high"].lt(0)],
        ["strengthened", "weakened"],
        default="inconclusive",
    )
    return result.sort_values(
        "neutral_mean_oriented_rank_ic", ascending=False
    ).reset_index(drop=True)


def write_markdown_report(
    lake: Lake, cfg: dict, conclusions: pd.DataFrame,
) -> None:
    exposure = lake.read("pit_industry_exposure_index", "v1")
    annual = lake.read("factor_neutralization_annual", "v1")
    coverage = float(exposure["eligible_industry_coverage"].min())
    strengthened = conclusions.loc[
        conclusions["rank_ic_effect"].eq("strengthened"), "factor"
    ].tolist()
    weakened = conclusions.loc[
        conclusions["rank_ic_effect"].eq("weakened"), "factor"
    ].tolist()
    spread_strengthened = conclusions.loc[
        conclusions["spread_delta_ci_low"].gt(0), "factor"
    ].tolist()
    spread_weakened = conclusions.loc[
        conclusions["spread_delta_ci_high"].lt(0), "factor"
    ].tolist()
    combined_annual = annual.loc[
        annual["split"].eq("historical_oos")
        & annual["pipeline"].eq("industry_size_neutralized")
    ]
    stable_factors = combined_annual.groupby("factor")["mean_oriented_rank_ic"].apply(
        lambda values: bool(values.gt(0).all())
    )
    positive_every_year = int(stable_factors.sum())
    oos_years = int(combined_annual["year"].nunique())
    display = conclusions[[
        "factor", "direction", "raw_mean_oriented_rank_ic",
        "neutral_mean_oriented_rank_ic", "rank_ic_delta",
        "rank_ic_delta_ci_low", "rank_ic_delta_ci_high",
        "neutral_mean_oriented_top_minus_bottom", "spread_delta",
        "spread_delta_ci_low", "spread_delta_ci_high",
        "rank_ic_effect",
    ]].copy()
    for column in display.columns:
        if column not in {"factor", "direction", "rank_ic_effect"}:
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    report = "\n".join([
        "# PIT 行业与规模中性化因子评价",
        "",
        "## 研究口径",
        "",
        f"- 历史 OOS：{cfg['historical_oos']['start_year']}–"
        f"{cfg['historical_oos']['end_year']}；未用于因子选择或翻向。",
        "- 比较：raw 与 industry + log-size neutralized，使用完全相同的暴露完整样本。",
        "- 因子方向：仅由训练期 raw 平均 RankIC 符号确定。",
        f"- PIT 行业最低年度覆盖率：{coverage:.4%}。",
        "- 推断：13 周 circular block bootstrap，95% 区间。",
        "",
        "## 结论摘要",
        "",
        f"- 中性化后定向 RankIC 显著增强：{', '.join(strengthened) or '无'}。",
        f"- 中性化后定向 RankIC 显著减弱：{', '.join(weakened) or '无'}。",
        f"- 其余 {len(conclusions) - len(strengthened) - len(weakened)} 个因子差异不显著。",
        f"- 行业+规模中性化后，{positive_every_year}/{len(stable_factors)} 个因子在全部 "
        f"{oos_years} 个 OOS 年度切片保持正向定向 RankIC。",
        f"- 定向 Top/Bottom 价差显著增强：{', '.join(spread_strengthened) or '无'}；"
        f"显著减弱：{', '.join(spread_weakened) or '无'}。",
        "",
        "## 逐因子结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 限制",
        "",
        "行业暴露来自按月抽样的历史证监会宽行业，不等同于申万细分行业；规模暴露为成交额除以"
        "换手率得到的流通市值代理。结果用于识别风险暴露依赖，不构成可交易组合收益承诺。",
        "",
    ])
    destination = ROOT / "reports" / "factor_neutralization.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    daily = build_daily_diagnostics(lake, cfg, factors, target)
    daily, directions = apply_training_directions(daily)
    summary = summarize_daily(daily, int(cfg["annualization_periods"]))
    annual = annual_summary(daily, int(cfg["annualization_periods"]))
    bootstrap = paired_bootstrap(daily, cfg)
    conclusions = conclusion_table(summary, bootstrap, directions)
    request = {**cfg, "factor_columns": factors}
    for table, frame in {
        "factor_neutralization_daily": daily,
        "factor_neutralization_directions": directions,
        "factor_neutralization_summary": summary,
        "factor_neutralization_annual": annual,
        "factor_neutralization_bootstrap": bootstrap,
        "factor_neutralization_conclusions": conclusions,
    }.items():
        lake.write(
            table, frame, key="v1",
            source="factor_neutralization.v1", request=request,
        )
    write_markdown_report(lake, cfg, conclusions)
    print(summary.loc[summary["split"].eq("historical_oos")].to_string(index=False))
    print(bootstrap.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate factors before and after PIT industry/size neutralization"
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
