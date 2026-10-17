"""Locked-model, end-to-end PIT factor-neutralization experiment.

This experiment isolates the effect of residualizing factor exposures on
same-date CSRC broad-industry dummies and/or log size.  Model hyperparameters,
portfolio construction, costs, and the exposure-complete stock sample are held
fixed across pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from quant_research.factor_neutralization import (
    _read_year,
    _safe_correlation,
    neutralize_factor_matrix,
    transformed_factor_designs,
)
from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
    _update_portfolio,
    transform_cross_section,
)
from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import (
    SufficientStatistics,
    fit_ridge,
    purged_end_date,
    sum_statistics,
    summarize,
)
from quant_research.model_challenger import (
    TrainingSample,
    concatenate_samples,
    deterministic_indices,
)
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "model_portfolio_neutralization.yaml"
REPORT_PATH = ROOT / "reports" / "model_portfolio_neutralization.md"


@dataclass
class FittedModel:
    model: str
    kind: str
    estimator: Any
    fit_seconds: float
    fit_observations: int

    def predict(self, design: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, np.ndarray):
            return design @ self.estimator
        return np.asarray(self.estimator.predict(design), dtype="float64")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_designs(
    group: pd.DataFrame, factors: list[str], cfg: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Create paired model designs and one shared exposure-complete mask.

    Both the factor block and the missingness-indicator block are residualized.
    Otherwise the model could reconstruct industry or size exposure through
    systematic differences in vendor/fundamental coverage.
    """
    factor_designs, risks = transformed_factor_designs(group, factors, cfg)
    exposure_complete = risks["valid_size"] & risks["valid_industry"]
    if bool(cfg["include_missing_indicators"]):
        full_raw, _ = transform_cross_section(
            group,
            factors,
            winsor_method="quantile",
            normalization_method="zscore",
            quantile_limits=tuple(cfg["quantile_limits"]),
            mad_multiplier=float(cfg["mad_multiplier"]),
            include_missing_indicators=True,
        )
        missing_block = full_raw[:, len(factors):]
        common_size = group["float_cap_proxy_cny"].where(exposure_complete)
        common_industry = group["industry_broad_code"].where(exposure_complete)
        size_missing, _ = neutralize_factor_matrix(
            missing_block, size=common_size
        )
        industry_missing, _ = neutralize_factor_matrix(
            missing_block,
            industry=common_industry,
            minimum_industry_group_size=int(cfg["minimum_industry_group_size"]),
        )
        combined_missing, _ = neutralize_factor_matrix(
            missing_block,
            industry=common_industry,
            size=common_size,
            minimum_industry_group_size=int(cfg["minimum_industry_group_size"]),
        )
        # An all-zero missingness column can acquire machine-epsilon residuals
        # from lstsq; standardizing those residuals would manufacture a fake
        # risk signal. Preserve structural constants as exact zeros.
        variable_missing = missing_block[exposure_complete].std(axis=0, ddof=1) > 1e-12
        for neutral_missing in (size_missing, industry_missing, combined_missing):
            neutral_missing[:, ~variable_missing] = 0.0
        missing_designs = {
            "raw": missing_block,
            "size_neutralized": size_missing,
            "industry_neutralized": industry_missing,
            "industry_size_neutralized": combined_missing,
        }
        designs = {
            pipeline: np.concatenate([design, missing_designs[pipeline]], axis=1)
            for pipeline, design in factor_designs.items()
        }
    else:
        designs = factor_designs
    expected = set(cfg["pipelines"])
    if set(designs) != expected:
        raise RuntimeError(
            f"configured pipelines do not match factor designs: {expected ^ set(designs)}"
        )
    return designs, risks, exposure_complete


def build_year_training(
    frame: pd.DataFrame,
    factors: list[str],
    target: str,
    cutoff: str,
    cfg: dict,
) -> tuple[
    dict[str, SufficientStatistics],
    dict[str, SufficientStatistics],
    dict[str, TrainingSample],
    dict[str, TrainingSample],
]:
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    full_stats = {
        pipeline: MutableStatistics.create(dimension) for pipeline in cfg["pipelines"]
    }
    purged_stats = {
        pipeline: MutableStatistics.create(dimension) for pipeline in cfg["pipelines"]
    }
    full_x = {pipeline: [] for pipeline in cfg["pipelines"]}
    full_y = {pipeline: [] for pipeline in cfg["pipelines"]}
    purged_x = {pipeline: [] for pipeline in cfg["pipelines"]}
    purged_y = {pipeline: [] for pipeline in cfg["pipelines"]}
    full_dates = purged_dates = 0

    for asof_date, group in frame.groupby("asof_date", sort=True):
        group = group.reset_index(drop=True)
        designs, _, complete = model_designs(group, factors, cfg)
        target_available = pd.to_numeric(group[target], errors="coerce").notna().to_numpy()
        usable = complete & target_available
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        y = np.asarray(rank_zscore(group.loc[usable, target]), dtype="float64")
        usable_group = group.loc[usable]
        chosen = deterministic_indices(
            usable_group, int(cfg["sample_per_training_date"])
        )
        for pipeline, design in designs.items():
            x = design[usable]
            full_stats[pipeline].add(x, y)
            full_x[pipeline].append(x[chosen].astype("float32"))
            full_y[pipeline].append(y[chosen].astype("float32"))
            if str(asof_date) <= cutoff:
                purged_stats[pipeline].add(x, y)
                purged_x[pipeline].append(x[chosen].astype("float32"))
                purged_y[pipeline].append(y[chosen].astype("float32"))
        full_dates += 1
        if str(asof_date) <= cutoff:
            purged_dates += 1

    def sample(xs: list[np.ndarray], ys: list[np.ndarray], dates: int) -> TrainingSample:
        if not xs:
            return TrainingSample(
                np.empty((0, dimension), dtype="float32"),
                np.empty(0, dtype="float32"),
                0,
            )
        return TrainingSample(np.concatenate(xs), np.concatenate(ys), dates)

    return (
        {name: value.freeze() for name, value in full_stats.items()},
        {name: value.freeze() for name, value in purged_stats.items()},
        {
            name: sample(full_x[name], full_y[name], full_dates)
            for name in cfg["pipelines"]
        },
        {
            name: sample(purged_x[name], purged_y[name], purged_dates)
            for name in cfg["pipelines"]
        },
    )


def _aggregate_training(
    yearly_stats: dict[int, dict[str, SufficientStatistics]],
    yearly_purged_stats: dict[int, dict[str, SufficientStatistics]],
    yearly_samples: dict[int, dict[str, TrainingSample]],
    yearly_purged_samples: dict[int, dict[str, TrainingSample]],
    pipeline: str,
    start_year: int,
    end_year: int,
) -> tuple[SufficientStatistics, TrainingSample]:
    statistics = sum_statistics(
        [yearly_stats[year][pipeline] for year in range(start_year, end_year)]
        + [yearly_purged_stats[end_year][pipeline]]
    )
    sample = concatenate_samples(
        [yearly_samples[year][pipeline] for year in range(start_year, end_year)]
        + [yearly_purged_samples[end_year][pipeline]]
    )
    return statistics, sample


def fit_models(
    statistics: SufficientStatistics,
    sample: TrainingSample,
    cfg: dict,
) -> dict[str, FittedModel]:
    fitted: dict[str, FittedModel] = {}
    for model_name, spec in cfg["models"].items():
        started = perf_counter()
        kind = str(spec["kind"])
        if kind == "ridge":
            estimator: Any = fit_ridge(
                statistics, float(spec["penalty_per_observation"])
            )
            observations = statistics.observations
        elif kind == "hgb":
            estimator = HistGradientBoostingRegressor(
                learning_rate=float(spec["learning_rate"]),
                max_iter=int(spec["max_iter"]),
                max_leaf_nodes=int(spec["max_leaf_nodes"]),
                min_samples_leaf=int(spec["min_samples_leaf"]),
                l2_regularization=float(spec["l2_regularization"]),
                early_stopping=False,
                random_state=int(cfg["random_state"]),
            ).fit(sample.x, sample.y)
            observations = len(sample.y)
        else:
            raise ValueError(f"unknown locked model kind: {kind}")
        fitted[model_name] = FittedModel(
            model=model_name,
            kind=kind,
            estimator=estimator,
            fit_seconds=perf_counter() - started,
            fit_observations=observations,
        )
    return fitted


def _selection_frame(
    group: pd.DataFrame, score: np.ndarray, top_n: int,
) -> pd.DataFrame:
    selected = group.assign(_score=score).sort_values(
        ["_score", "instrument_id"], ascending=[False, True]
    ).head(top_n).copy()
    selected["portfolio_weight"] = 1.0 / len(selected)
    selected["portfolio_rank"] = np.arange(1, len(selected) + 1)
    return selected


def portfolio_risk_diagnostics(
    group: pd.DataFrame,
    selected: pd.DataFrame,
    score: np.ndarray,
    size_z: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    industries = group["industry_broad_code"].astype(str)
    selected_industries = selected["industry_broad_code"].astype(str)
    universe_weights = industries.value_counts(normalize=True)
    portfolio_weights = selected_industries.value_counts(normalize=True)
    labels = sorted(set(universe_weights.index) | set(portfolio_weights.index))
    rows = []
    for industry in labels:
        universe_weight = float(universe_weights.get(industry, 0.0))
        portfolio_weight = float(portfolio_weights.get(industry, 0.0))
        rows.append({
            "industry_broad_code": industry,
            "portfolio_weight": portfolio_weight,
            "universe_weight": universe_weight,
            "active_weight": portfolio_weight - universe_weight,
            "holdings": int(selected_industries.eq(industry).sum()),
        })
    allocation = pd.DataFrame(rows)
    active = allocation["active_weight"].to_numpy(dtype="float64")
    selected_index = selected.index.to_numpy(dtype=int)
    score_size_correlation = _safe_correlation(score, size_z)
    score_industry_max_abs_mean = float(
        pd.DataFrame({"score": score, "industry": industries.to_numpy()})
        .groupby("industry")["score"].mean().abs().max()
    )
    diagnostics = {
        "score_log_size_correlation": score_size_correlation,
        "abs_score_log_size_correlation": abs(score_size_correlation),
        "score_industry_max_abs_mean": score_industry_max_abs_mean,
        "selected_mean_size_z": float(np.mean(size_z[selected_index])),
        "universe_mean_size_z": float(np.mean(size_z)),
        "active_mean_size_z": float(
            np.mean(size_z[selected_index]) - np.mean(size_z)
        ),
        "portfolio_industry_hhi": float(
            np.square(allocation["portfolio_weight"]).sum()
        ),
        "universe_industry_hhi": float(
            np.square(allocation["universe_weight"]).sum()
        ),
        "industry_active_weight_l1": float(np.abs(active).sum() / 2.0),
        "maximum_industry_weight": float(allocation["portfolio_weight"].max()),
        "maximum_abs_industry_active_weight": float(np.abs(active).max()),
    }
    return diagnostics, allocation


def evaluate(
    frame: pd.DataFrame,
    models: dict[str, dict[str, FittedModel]],
    factors: list[str],
    target: str,
    states: dict[tuple[str, str], dict],
    cfg: dict,
    split: str,
    test_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_records: list[dict] = []
    portfolio_records: list[dict] = []
    risk_records: list[dict] = []
    allocation_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    top_n = int(cfg["portfolio"]["top_n"])

    for asof_date, original_group in frame.groupby("asof_date", sort=True):
        original_group = original_group.reset_index(drop=True)
        designs, risks, complete = model_designs(original_group, factors, cfg)
        if complete.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        group = original_group.loc[complete].reset_index(drop=True)
        size_z = risks["size_z"][complete]
        realized = pd.to_numeric(group[target], errors="coerce")
        usable = realized.notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        usable_returns = realized.loc[usable].reset_index(drop=True)

        for pipeline in cfg["pipelines"]:
            design = designs[pipeline][complete]
            for model_name, fitted in models[pipeline].items():
                score = fitted.predict(design)
                usable_score = score[usable]
                percentile = pd.Series(usable_score).rank(method="first", pct=True)
                bucket = (
                    percentile * int(cfg["portfolio_buckets"])
                ).clip(upper=int(cfg["portfolio_buckets"]) - 1e-9).astype(int)
                bucket_returns = usable_returns.groupby(bucket.to_numpy()).mean()
                base = {
                    "asof_date": str(asof_date),
                    "split": split,
                    "test_year": test_year,
                    "pipeline": pipeline,
                    "model": model_name,
                    "kind": fitted.kind,
                }
                daily_records.append({
                    **base,
                    "cross_section_size": int(usable.sum()),
                    "rank_ic": _rank_ic(usable_score, usable_returns),
                    "top_minus_bottom": float(
                        bucket_returns.get(int(cfg["portfolio_buckets"]) - 1, np.nan)
                        - bucket_returns.get(0, np.nan)
                    ),
                })
                portfolio = _update_portfolio(
                    group,
                    score,
                    target,
                    top_n,
                    float(cfg["portfolio"]["round_trip_cost_bps"]),
                    float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                    states[(pipeline, model_name)],
                )
                portfolio_records.append({**portfolio, **base})

                selected = _selection_frame(group, score, top_n)
                diagnostics, allocation = portfolio_risk_diagnostics(
                    group, selected, score, size_z
                )
                risk_records.append({**base, **diagnostics})
                allocation_frames.append(allocation.assign(**base))
                holding_frames.append(selected[[
                    "instrument_id", "industry_broad_code", "float_cap_proxy_cny",
                    "portfolio_weight", "portfolio_rank", "_score",
                ]].rename(columns={"_score": "score"}).assign(**base))

    return (
        pd.DataFrame(daily_records),
        pd.DataFrame(portfolio_records),
        pd.DataFrame(risk_records),
        pd.concat(allocation_frames, ignore_index=True),
        pd.concat(holding_frames, ignore_index=True),
    )


def summarize_results(
    daily: pd.DataFrame,
    portfolio: pd.DataFrame,
    risk: pd.DataFrame,
    cfg: dict,
    split: str,
) -> pd.DataFrame:
    records = []
    periods = int(cfg["annualization_periods"])
    for (pipeline, model), group in daily.groupby(["pipeline", "model"], sort=False):
        portfolio_group = portfolio.loc[
            portfolio["pipeline"].eq(pipeline) & portfolio["model"].eq(model)
        ]
        risk_group = risk.loc[
            risk["pipeline"].eq(pipeline) & risk["model"].eq(model)
        ]
        records.append({
            "split": split,
            "pipeline": pipeline,
            "model": model,
            **summarize(group, periods),
            **{
                f"portfolio_{key}": value
                for key, value in performance_summary(portfolio_group, periods).items()
            },
            "mean_score_log_size_correlation": float(
                risk_group["score_log_size_correlation"].mean()
            ),
            "mean_abs_score_log_size_correlation": float(
                risk_group["abs_score_log_size_correlation"].mean()
            ),
            "mean_score_industry_max_abs_mean": float(
                risk_group["score_industry_max_abs_mean"].mean()
            ),
            "mean_selected_size_z": float(risk_group["selected_mean_size_z"].mean()),
            "mean_active_size_z": float(risk_group["active_mean_size_z"].mean()),
            "mean_portfolio_industry_hhi": float(
                risk_group["portfolio_industry_hhi"].mean()
            ),
            "mean_industry_active_weight_l1": float(
                risk_group["industry_active_weight_l1"].mean()
            ),
            "mean_maximum_industry_weight": float(
                risk_group["maximum_industry_weight"].mean()
            ),
            "mean_maximum_abs_industry_active_weight": float(
                risk_group["maximum_abs_industry_active_weight"].mean()
            ),
        })
    return pd.DataFrame(records)


def annual_results(
    daily: pd.DataFrame, portfolio: pd.DataFrame, risk: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    records = []
    periods = int(cfg["annualization_periods"])
    for (pipeline, model, year), group in daily.groupby(
        ["pipeline", "model", "test_year"], sort=False
    ):
        selector = (
            portfolio["pipeline"].eq(pipeline)
            & portfolio["model"].eq(model)
            & portfolio["test_year"].eq(year)
        )
        risk_selector = (
            risk["pipeline"].eq(pipeline)
            & risk["model"].eq(model)
            & risk["test_year"].eq(year)
        )
        risk_group = risk.loc[risk_selector]
        records.append({
            "pipeline": pipeline,
            "model": model,
            "year": int(year),
            **summarize(group, periods),
            **{
                f"portfolio_{key}": value
                for key, value in performance_summary(portfolio.loc[selector], periods).items()
            },
            "mean_industry_active_weight_l1": float(
                risk_group["industry_active_weight_l1"].mean()
            ),
            "mean_abs_score_log_size_correlation": float(
                risk_group["abs_score_log_size_correlation"].mean()
            ),
        })
    return pd.DataFrame(records)


def paired_bootstrap(
    daily: pd.DataFrame, portfolio: pd.DataFrame, risk: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    records = []
    bootstrap_cfg = cfg["bootstrap"]
    sources = (
        ("rank_ic", daily, "rank_ic", 1.0, "higher_is_better"),
        (
            "net_return_annualized_arithmetic",
            portfolio.loc[portfolio["included_in_performance"]],
            "net_return",
            float(cfg["annualization_periods"]),
            "higher_is_better",
        ),
        (
            "industry_active_weight_l1",
            risk,
            "industry_active_weight_l1",
            1.0,
            "lower_is_better",
        ),
        (
            "abs_score_log_size_correlation",
            risk,
            "abs_score_log_size_correlation",
            1.0,
            "lower_is_better",
        ),
        (
            "score_industry_max_abs_mean",
            risk,
            "score_industry_max_abs_mean",
            1.0,
            "lower_is_better",
        ),
        (
            "portfolio_industry_hhi",
            risk,
            "portfolio_industry_hhi",
            1.0,
            "lower_is_better",
        ),
    )
    for model in cfg["models"]:
        for pipeline in cfg["pipelines"]:
            if pipeline == "raw":
                continue
            for metric, frame, column, scale, direction in sources:
                left = frame.loc[
                    frame["model"].eq(model) & frame["pipeline"].eq(pipeline),
                    ["asof_date", column],
                ]
                right = frame.loc[
                    frame["model"].eq(model) & frame["pipeline"].eq("raw"),
                    ["asof_date", column],
                ]
                paired = left.merge(
                    right,
                    on="asof_date",
                    suffixes=("_neutral", "_raw"),
                    validate="one_to_one",
                )
                difference = pd.to_numeric(
                    paired[f"{column}_neutral"] - paired[f"{column}_raw"],
                    errors="coerce",
                ).dropna().to_numpy(dtype="float64")
                if len(difference) < int(bootstrap_cfg["block_length_weeks"]):
                    raise RuntimeError(
                        f"not enough paired rows for {model}/{pipeline}/{metric}"
                    )
                estimates = circular_block_bootstrap_mean(
                    difference,
                    int(bootstrap_cfg["block_length_weeks"]),
                    int(bootstrap_cfg["samples"]),
                    int(bootstrap_cfg["random_seed"]),
                ) * scale
                records.append({
                    "model": model,
                    "pipeline": pipeline,
                    "reference": "raw",
                    "metric": metric,
                    "preferred_direction": direction,
                    "observations": len(difference),
                    "point_estimate": float(difference.mean() * scale),
                    "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                    "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                    "probability_delta_nonpositive": float(
                        (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                    ),
                    "block_length_weeks": int(bootstrap_cfg["block_length_weeks"]),
                })
    return pd.DataFrame(records)


def conclusion_table(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in summary["model"].unique():
        raw = summary.loc[
            summary["model"].eq(model) & summary["pipeline"].eq("raw")
        ].iloc[0]
        for pipeline in summary["pipeline"].unique():
            if pipeline == "raw":
                continue
            neutral = summary.loc[
                summary["model"].eq(model) & summary["pipeline"].eq(pipeline)
            ].iloc[0]
            intervals = bootstrap.loc[
                bootstrap["model"].eq(model) & bootstrap["pipeline"].eq(pipeline)
            ].set_index("metric")
            ic_interval = intervals.loc["rank_ic"]
            return_interval = intervals.loc["net_return_annualized_arithmetic"]
            industry_interval = intervals.loc["industry_active_weight_l1"]
            rows.append({
                "model": model,
                "pipeline": pipeline,
                "raw_mean_rank_ic": float(raw["mean_rank_ic"]),
                "neutral_mean_rank_ic": float(neutral["mean_rank_ic"]),
                "rank_ic_delta": float(ic_interval["point_estimate"]),
                "rank_ic_ci_low": float(ic_interval["ci_2_5pct"]),
                "rank_ic_ci_high": float(ic_interval["ci_97_5pct"]),
                "raw_annualized_return": float(raw["portfolio_annualized_return"]),
                "neutral_annualized_return": float(
                    neutral["portfolio_annualized_return"]
                ),
                "annualized_arithmetic_return_delta": float(
                    return_interval["point_estimate"]
                ),
                "return_delta_ci_low": float(return_interval["ci_2_5pct"]),
                "return_delta_ci_high": float(return_interval["ci_97_5pct"]),
                "raw_industry_active_weight_l1": float(
                    raw["mean_industry_active_weight_l1"]
                ),
                "neutral_industry_active_weight_l1": float(
                    neutral["mean_industry_active_weight_l1"]
                ),
                "industry_active_delta": float(industry_interval["point_estimate"]),
                "industry_active_delta_ci_low": float(industry_interval["ci_2_5pct"]),
                "industry_active_delta_ci_high": float(industry_interval["ci_97_5pct"]),
            })
    return pd.DataFrame(rows)


def write_report(conclusions: pd.DataFrame, cfg: dict) -> None:
    combined = conclusions.loc[
        conclusions["pipeline"].eq("industry_size_neutralized")
    ].copy()
    display = combined[[
        "model", "raw_mean_rank_ic", "neutral_mean_rank_ic", "rank_ic_delta",
        "rank_ic_ci_low", "rank_ic_ci_high", "raw_annualized_return",
        "neutral_annualized_return", "annualized_arithmetic_return_delta",
        "return_delta_ci_low", "return_delta_ci_high",
        "raw_industry_active_weight_l1", "neutral_industry_active_weight_l1",
        "industry_active_delta_ci_low", "industry_active_delta_ci_high",
    ]].copy()
    lines = [
        "# 模型与组合层面的 PIT 行业/规模中性化",
        "",
        "## 研究设计",
        "",
        "- 模型：锁定 Ridge 与 `hgb_shallow15`；没有使用 2021+ 数据重新选模或调参。",
        "- 因子：同日截面 1%/99% 缩尾、z-score，再对 CSRC 大类行业哑变量与 log-size 做 OLS 残差化。",
        "- 样本：所有 pipeline 共用行业与规模暴露均完整的股票；训练标签按 5 个交易日 purge。",
        "- 组合：Top-200 等权、理想完整成交、20bp round-trip 成本；本实验没有施加组合层行业约束。",
        f"- 推断：{cfg['bootstrap']['block_length_weeks']} 周 circular block bootstrap，"
        f"{cfg['bootstrap']['samples']} 次抽样。",
        "",
        "## 历史样本外核心结果（2021–2026）",
        "",
        display.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 解释边界",
        "",
        "因子中性化不等于组合行业中性。若行业主动权重仍然偏高，下一步应在组合构建器中加入"
        "显式行业 active-weight、个股权重和换手约束，而不是再次用 OOS 数据选择因子处理方式。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])

    yearly_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_purged_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_samples: dict[int, dict[str, TrainingSample]] = {}
    yearly_purged_samples: dict[int, dict[str, TrainingSample]] = {}
    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        (
            yearly_stats[year],
            yearly_purged_stats[year],
            yearly_samples[year],
            yearly_purged_samples[year],
        ) = build_year_training(frame, factors, target, cutoff, cfg)
        print(
            f"model-neutral statistics year={year} "
            f"dates={yearly_stats[year]['raw'].dates} "
            f"sample_rows={len(yearly_samples[year]['raw'].y)}",
            flush=True,
        )

    train_end = int(cfg["train"]["end_year"])
    validation_models: dict[str, dict[str, FittedModel]] = {}
    fit_records: list[dict] = []
    coefficient_records: list[dict] = []
    feature_names = factors + (
        [f"{factor}__missing" for factor in factors]
        if cfg["include_missing_indicators"] else []
    )
    for pipeline in cfg["pipelines"]:
        statistics, sample = _aggregate_training(
            yearly_stats, yearly_purged_stats, yearly_samples,
            yearly_purged_samples, pipeline, start_year, train_end,
        )
        validation_models[pipeline] = fit_models(statistics, sample, cfg)
        for model in validation_models[pipeline].values():
            fit_records.append({
                "split": "validation", "test_year": None, "pipeline": pipeline,
                "model": model.model, "kind": model.kind,
                "fit_seconds": model.fit_seconds,
                "fit_observations": model.fit_observations,
            })

    validation_end = int(cfg["validation"]["end_year"])
    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    validation_frames = []
    for year in range(int(cfg["validation"]["start_year"]), validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(validation_cutoff)]
        validation_frames.append(frame)
    validation = pd.concat(validation_frames, ignore_index=True)
    validation_states = {
        (pipeline, model): {}
        for pipeline in cfg["pipelines"] for model in cfg["models"]
    }
    (
        validation_daily, validation_portfolio, validation_risk,
        validation_allocation, validation_holdings,
    ) = evaluate(
        validation, validation_models, factors, target, validation_states,
        cfg, "validation",
    )
    validation_summary = summarize_results(
        validation_daily, validation_portfolio, validation_risk, cfg, "validation"
    )

    oos_daily_frames = []
    oos_portfolio_frames = []
    oos_risk_frames = []
    oos_allocation_frames = []
    oos_holding_frames = []
    oos_states = {
        (pipeline, model): {}
        for pipeline in cfg["pipelines"] for model in cfg["models"]
    }
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        models: dict[str, dict[str, FittedModel]] = {}
        for pipeline in cfg["pipelines"]:
            statistics, sample = _aggregate_training(
                yearly_stats, yearly_purged_stats, yearly_samples,
                yearly_purged_samples, pipeline, start_year, test_year - 1,
            )
            models[pipeline] = fit_models(statistics, sample, cfg)
            for model in models[pipeline].values():
                fit_records.append({
                    "split": "historical_oos", "test_year": test_year,
                    "pipeline": pipeline, "model": model.model, "kind": model.kind,
                    "fit_seconds": model.fit_seconds,
                    "fit_observations": model.fit_observations,
                })
                if model.kind == "ridge":
                    coefficient_records.extend({
                        "test_year": test_year, "pipeline": pipeline,
                        "model": model.model, "feature": feature,
                        "coefficient": float(value),
                        "fit_observations": model.fit_observations,
                    } for feature, value in zip(feature_names, model.estimator))
        frame = _read_year(lake, test_year, columns)
        daily, portfolio, risk, allocation, holdings = evaluate(
            frame, models, factors, target, oos_states, cfg,
            "historical_oos", test_year,
        )
        oos_daily_frames.append(daily)
        oos_portfolio_frames.append(portfolio)
        oos_risk_frames.append(risk)
        oos_allocation_frames.append(allocation)
        oos_holding_frames.append(holdings)
        print(
            f"model-neutral oos year={test_year} dates={daily['asof_date'].nunique()}",
            flush=True,
        )

    oos_daily = pd.concat(oos_daily_frames, ignore_index=True)
    oos_portfolio = pd.concat(oos_portfolio_frames, ignore_index=True)
    oos_risk = pd.concat(oos_risk_frames, ignore_index=True)
    oos_allocation = pd.concat(oos_allocation_frames, ignore_index=True)
    oos_holdings = pd.concat(oos_holding_frames, ignore_index=True)
    oos_summary = summarize_results(
        oos_daily, oos_portfolio, oos_risk, cfg, "historical_oos"
    )
    oos_annual = annual_results(oos_daily, oos_portfolio, oos_risk, cfg)
    bootstrap = paired_bootstrap(oos_daily, oos_portfolio, oos_risk, cfg)
    conclusions = conclusion_table(oos_summary, bootstrap)
    write_report(conclusions, cfg)

    request = {**cfg, "factor_columns": factors}
    outputs = {
        "model_neutralization_protocol": pd.DataFrame([{
            "experiment_id": cfg["experiment_id"],
            "historical_oos_used_for_selection": bool(
                cfg["historical_oos_used_for_selection"]
            ),
            "models_locked_before_historical_oos": True,
            "common_exposure_complete_sample": True,
            "factor_and_missingness_features_residualized": True,
            "portfolio_industry_constraint_applied": bool(
                cfg["portfolio"]["explicit_industry_constraint"]
            ),
            "interpretation": cfg["interpretation"],
        }]),
        "model_neutralization_validation_summary": validation_summary,
        "model_neutralization_validation_daily": validation_daily,
        "model_neutralization_validation_portfolio": validation_portfolio,
        "model_neutralization_validation_risk": validation_risk,
        "model_neutralization_validation_industry_allocation": validation_allocation,
        "model_neutralization_validation_holdings": validation_holdings,
        "model_neutralization_oos_summary": oos_summary,
        "model_neutralization_oos_daily": oos_daily,
        "model_neutralization_oos_portfolio": oos_portfolio,
        "model_neutralization_oos_risk": oos_risk,
        "model_neutralization_oos_industry_allocation": oos_allocation,
        "model_neutralization_oos_holdings": oos_holdings,
        "model_neutralization_oos_annual": oos_annual,
        "model_neutralization_oos_bootstrap": bootstrap,
        "model_neutralization_conclusions": conclusions,
        "model_neutralization_fit_audit": pd.DataFrame(fit_records),
        "model_neutralization_ridge_coefficients": pd.DataFrame(coefficient_records),
    }
    for table, frame in outputs.items():
        lake.write(
            table, frame, key="v1", source="model_portfolio_neutralization.v1",
            request=request,
        )
    print(oos_summary.to_string(index=False))
    print(conclusions.to_string(index=False))


if __name__ == "__main__":
    run()
