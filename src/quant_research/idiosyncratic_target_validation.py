"""Validation-only experiment for a style-residualized prediction target.

The experiment deliberately stops at signal diagnostics.  It asks whether the
same locked model specifications learn stock-specific returns more directly
when the training label has market, PIT industry and six local style controls
removed.  It never reads the consumed historical-OOS period or future lockbox.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_neutralization import consolidated_industry_groups
from quant_research.factor_preprocessing_ablation import (
    MutableStatistics,
    _rank_ic,
    transform_cross_section,
)
from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.model_baseline import (
    SufficientStatistics,
    purged_end_date,
    sum_statistics,
    weekly_rows,
)
from quant_research.model_benchmark import FittedCandidate, fit_candidate
from quant_research.model_challenger import (
    TrainingSample,
    concatenate_samples,
    deterministic_indices,
)
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean
from quant_research.multi_factor_risk_attribution import build_style_exposures


CONFIG_PATH = ROOT / "config" / "idiosyncratic_target_validation.yaml"


@dataclass(frozen=True)
class ResidualizedCrossSection:
    frame: pd.DataFrame
    r_squared: float
    maximum_abs_normal_equation_error: float
    industry_count: int


class InsufficientCrossSectionError(ValueError):
    """The date is retained in audit logic but cannot support a residual fit."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if int(cfg["train"]["start_year"]) < 2008:
        raise ValueError("six-style residual labels require the PIT beta product from 2008")
    if int(cfg["train"]["end_year"]) >= int(cfg["validation"]["start_year"]):
        raise ValueError("training and validation periods must not overlap")
    if bool(cfg.get("historical_oos_evaluation_enabled")):
        raise ValueError("this experiment must not inspect historical OOS")
    if bool(cfg.get("future_lockbox_access_allowed")):
        raise ValueError("this experiment must not access the future lockbox")
    if cfg.get("historical_oos_used_for_selection") is not False:
        raise ValueError("historical OOS cannot select this challenger")
    if int(cfg["validation"]["end_year"]) > 2020:
        raise ValueError("development evidence must end in 2020")
    if cfg["inference"]["multiplicity_control"] != "benjamini_hochberg":
        raise ValueError("the prespecified model family requires BH-FDR")
    contract = cfg["external_pit_feature_contract"]
    if contract.get("allow_period_end_as_availability_proxy") is not False:
        raise ValueError("period end cannot proxy announcement availability")
    if contract.get("allow_latest_value_historical_backfill") is not False:
        raise ValueError("latest fundamentals cannot be backfilled historically")


def validate_external_pit_events(
    frame: pd.DataFrame, cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fail closed on revision-aware, announcement-time fundamental events."""
    cfg = cfg or load_config()
    contract = cfg["external_pit_feature_contract"]
    required = set(contract["required_fields"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PIT event fields missing: {missing}")
    natural_key = list(contract["required_natural_key"])
    duplicates = int(frame.duplicated(natural_key).sum())
    if duplicates:
        raise ValueError(f"duplicate PIT event natural keys: {duplicates}")
    dates = frame.copy()
    for column in ("fiscal_period_end", "announced_at", "available_date", "ingested_at"):
        dates[column] = pd.to_datetime(dates[column], errors="coerce", utc=True)
        if dates[column].isna().any():
            raise ValueError(f"invalid PIT timestamp in {column}")
    if (dates["available_date"] < dates["announced_at"].dt.normalize()).any():
        raise ValueError("feature availability precedes its announcement")
    if (dates["ingested_at"] < dates["announced_at"]).any():
        raise ValueError("ingestion timestamp precedes the source announcement")
    if pd.to_numeric(dates["revision_sequence"], errors="coerce").isna().any():
        raise ValueError("revision_sequence must be numeric")
    return pd.DataFrame([{
        "rows": len(frame),
        "unique_instruments": frame["instrument_id"].astype(str).nunique(),
        "unique_fields": frame["field"].astype(str).nunique(),
        "duplicate_natural_keys": duplicates,
        "announcement_time_contract_passed": True,
        "latest_value_backfill_prohibited": True,
    }])


def residualize_cross_section(
    group: pd.DataFrame, cfg: dict[str, Any],
) -> ResidualizedCrossSection:
    """Remove market, PIT broad industry and six style components from returns."""
    target = str(cfg["target"])
    required = {
        "instrument_id", "asof_date", target, "industry_broad_code",
        "float_cap_proxy_cny", "beta_252d", "volatility_20d", "amihud_20d",
        "turnover_20d", "earnings_yield", "book_to_price", "momentum_120d",
        "momentum_20d_skip_5d",
    }
    missing = sorted(required - set(group.columns))
    if missing:
        raise ValueError(f"residual-target columns missing: {missing}")
    if group["instrument_id"].astype(str).duplicated().any():
        raise ValueError("duplicate instrument in one residual-target cross section")

    target_values = pd.to_numeric(group[target], errors="coerce")
    size = pd.to_numeric(group["float_cap_proxy_cny"], errors="coerce")
    industry = group["industry_broad_code"].fillna("").astype(str).str.strip()
    usable = target_values.notna() & size.gt(0) & industry.ne("")
    frame = group.loc[usable].reset_index(drop=True).copy()
    if len(frame) < int(cfg["minimum_cross_section_size"]):
        raise InsufficientCrossSectionError(
            "residual-target cross section is below the minimum size"
        )

    groups, valid_industry = consolidated_industry_groups(
        frame["industry_broad_code"],
        int(cfg["controls"]["minimum_industry_group_size"]),
    )
    if not valid_industry.all():
        raise RuntimeError("industry-complete sample changed during consolidation")
    categories = sorted(set(groups))
    if len(categories) < 2:
        raise ValueError("residual target requires at least two industries")

    risk_cfg = {
        "style_winsor_quantiles": cfg["controls"]["style_winsor_quantiles"]
    }
    styles = build_style_exposures(frame, risk_cfg)
    expected_styles = list(cfg["controls"]["styles"])
    if list(styles.columns) != expected_styles:
        raise RuntimeError("configured residual controls do not match style builder")
    controls = [np.ones(len(frame), dtype="float64")]
    controls.extend((groups == category).astype("float64") for category in categories[1:])
    controls.extend(styles[column].to_numpy(dtype="float64") for column in expected_styles)
    design = np.column_stack(controls)
    realized = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype="float64")
    coefficients = np.linalg.lstsq(design, realized, rcond=None)[0]
    fitted = design @ coefficients
    residual = realized - fitted
    normal_error = float(np.max(np.abs(design.T @ residual)) / len(frame))
    total = float(np.sum((realized - realized.mean()) ** 2))
    r_squared = 1.0 - float(residual @ residual) / total if total > 0 else np.nan

    frame[str(cfg["residual_target"])] = residual
    for column in expected_styles:
        frame[f"control_{column}"] = styles[column].to_numpy(dtype="float64")
    return ResidualizedCrossSection(
        frame=frame,
        r_squared=r_squared,
        maximum_abs_normal_equation_error=normal_error,
        industry_count=len(categories),
    )


def _read_year(
    lake: Lake, year: int, factors: list[str], cfg: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "instrument_id", "asof_date", "available_date", str(cfg["target"]), *factors
    ]
    panel = weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=columns,
    ))
    size = pd.read_parquet(
        lake.table_path("pit_size_exposures", str(year)),
        columns=["instrument_id", "asof_date", "float_cap_proxy_cny"],
    )
    industry = pd.read_parquet(
        lake.table_path("pit_industry_exposures", str(year)),
        columns=["instrument_id", "asof_date", "industry_broad_code"],
    )
    beta = pd.read_parquet(
        lake.table_path("pit_beta_exposures", str(year)),
        columns=["instrument_id", "asof_date", "beta_252d"],
    )
    keys = ["instrument_id", "asof_date"]
    return panel.merge(size, on=keys, how="left", validate="one_to_one").merge(
        industry, on=keys, how="left", validate="one_to_one",
    ).merge(beta, on=keys, how="left", validate="one_to_one")


def _design(frame: pd.DataFrame, factors: list[str], cfg: dict[str, Any]) -> np.ndarray:
    design, _ = transform_cross_section(
        frame, factors, winsor_method="quantile", normalization_method="zscore",
        quantile_limits=tuple(cfg["quantile_limits"]),
        mad_multiplier=float(cfg["mad_multiplier"]),
        include_missing_indicators=bool(cfg["include_missing_indicators"]),
    )
    return design


def _empty_sample(dimension: int) -> TrainingSample:
    return TrainingSample(
        np.empty((0, dimension), dtype="float32"),
        np.empty(0, dtype="float32"), 0,
    )


def build_year_training(
    frame: pd.DataFrame, factors: list[str], cutoff: str, cfg: dict[str, Any],
) -> tuple[
    dict[str, SufficientStatistics], dict[str, SufficientStatistics],
    dict[str, TrainingSample], dict[str, TrainingSample], pd.DataFrame,
]:
    targets = [str(cfg["target"]), str(cfg["residual_target"])]
    dimension = len(factors) * (2 if cfg["include_missing_indicators"] else 1)
    full_stats = {target: MutableStatistics.create(dimension) for target in targets}
    purged_stats = {target: MutableStatistics.create(dimension) for target in targets}
    full_x: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    full_y: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    purged_x: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    purged_y: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    full_dates = purged_dates = 0
    audit_rows = []
    for asof_date, raw_group in frame.groupby("asof_date", sort=True):
        try:
            result = residualize_cross_section(raw_group.reset_index(drop=True), cfg)
        except InsufficientCrossSectionError:
            continue
        group = result.frame
        design = _design(group, factors, cfg)
        chosen = deterministic_indices(group, int(cfg["sample_per_training_date"]))
        for target in targets:
            ranked = np.asarray(rank_zscore(group[target]), dtype="float64")
            full_stats[target].add(design, ranked)
            full_x[target].append(design[chosen].astype("float32"))
            full_y[target].append(ranked[chosen].astype("float32"))
            if str(asof_date) <= cutoff:
                purged_stats[target].add(design, ranked)
                purged_x[target].append(design[chosen].astype("float32"))
                purged_y[target].append(ranked[chosen].astype("float32"))
        full_dates += 1
        if str(asof_date) <= cutoff:
            purged_dates += 1
        audit_rows.append({
            "asof_date": asof_date, "cross_section_size": len(group),
            "residual_label_r_squared": result.r_squared,
            "maximum_abs_normal_equation_error": result.maximum_abs_normal_equation_error,
            "industry_count": result.industry_count,
        })

    def sample(xs: list[np.ndarray], ys: list[np.ndarray], dates: int) -> TrainingSample:
        if not xs:
            return _empty_sample(dimension)
        return TrainingSample(np.concatenate(xs), np.concatenate(ys), dates)

    return (
        {name: value.freeze() for name, value in full_stats.items()},
        {name: value.freeze() for name, value in purged_stats.items()},
        {name: sample(full_x[name], full_y[name], full_dates) for name in targets},
        {name: sample(purged_x[name], purged_y[name], purged_dates) for name in targets},
        pd.DataFrame(audit_rows),
    )


def _aggregate_training(
    yearly_stats: dict[int, dict[str, SufficientStatistics]],
    yearly_purged_stats: dict[int, dict[str, SufficientStatistics]],
    yearly_samples: dict[int, dict[str, TrainingSample]],
    yearly_purged_samples: dict[int, dict[str, TrainingSample]],
    target: str, start_year: int, end_year: int,
) -> tuple[SufficientStatistics, TrainingSample]:
    return (
        sum_statistics(
            [yearly_stats[year][target] for year in range(start_year, end_year)]
            + [yearly_purged_stats[end_year][target]]
        ),
        concatenate_samples(
            [yearly_samples[year][target] for year in range(start_year, end_year)]
            + [yearly_purged_samples[end_year][target]]
        ),
    )


def evaluate_validation(
    frames: list[pd.DataFrame], models: dict[tuple[str, str], FittedCandidate],
    factors: list[str], cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows, label_audit_rows = [], []
    for yearly in frames:
        for asof_date, raw_group in yearly.groupby("asof_date", sort=True):
            try:
                result = residualize_cross_section(raw_group.reset_index(drop=True), cfg)
            except InsufficientCrossSectionError:
                continue
            group = result.frame
            design = _design(group, factors, cfg)
            raw_return = pd.to_numeric(group[str(cfg["target"])], errors="raise")
            residual_return = pd.to_numeric(
                group[str(cfg["residual_target"])], errors="raise"
            )
            label_audit_rows.append({
                "asof_date": asof_date, "split": "validation",
                "cross_section_size": len(group),
                "residual_label_r_squared": result.r_squared,
                "maximum_abs_normal_equation_error": result.maximum_abs_normal_equation_error,
                "industry_count": result.industry_count,
            })
            for (model_name, training_target), model in models.items():
                score = model.predict(design)
                style_correlations = [
                    pd.Series(score).corr(group[f"control_{style}"], method="pearson")
                    for style in cfg["controls"]["styles"]
                ]
                finite_correlations = [abs(float(value)) for value in style_correlations
                                       if pd.notna(value)]
                daily_rows.append({
                    "asof_date": asof_date,
                    "year": int(str(asof_date)[:4]),
                    "model": model_name,
                    "training_target": training_target,
                    "cross_section_size": len(group),
                    "raw_rank_ic": _rank_ic(score, raw_return),
                    "residual_rank_ic": _rank_ic(score, residual_return),
                    "maximum_abs_score_style_correlation": (
                        max(finite_correlations) if finite_correlations else np.nan
                    ),
                })
    return pd.DataFrame(daily_rows), pd.DataFrame(label_audit_rows)


def paired_inference(daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    residual_name = str(cfg["residual_target"])
    raw_name = str(cfg["target"])
    inference = cfg["inference"]
    rows = []
    for model, group in daily.groupby("model", sort=True):
        pivot = group.pivot(
            index="asof_date", columns="training_target", values="residual_rank_ic"
        ).dropna()
        difference = pivot[residual_name] - pivot[raw_name]
        draws = circular_block_bootstrap_mean(
            difference.to_numpy(dtype="float64"),
            int(inference["block_length_weeks"]),
            int(inference["bootstrap_samples"]),
            int(inference["random_seed"]),
        )
        rows.append({
            "model": model,
            "metric": "residual_rank_ic",
            "comparison": inference["comparison"],
            "observations": len(difference),
            "point_estimate": float(difference.mean()),
            "ci_2_5pct": float(np.quantile(draws, 0.025)),
            "ci_97_5pct": float(np.quantile(draws, 0.975)),
            "one_sided_probability_nonpositive": float(
                (np.sum(draws <= 0) + 1) / (len(draws) + 1)
            ),
        })
    result = pd.DataFrame(rows)
    result["bh_fdr_adjusted_p"] = benjamini_hochberg(
        result["one_sided_probability_nonpositive"]
    )
    result["survives_bh_fdr_5pct"] = result["bh_fdr_adjusted_p"].le(
        float(inference["false_discovery_rate"])
    )
    return result


def promotion_decisions(
    daily: pd.DataFrame, inference: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    residual_name = str(cfg["residual_target"])
    required_years = int(cfg["promotion_gate"]["minimum_positive_validation_years"])
    rows = []
    for record in inference.to_dict(orient="records"):
        model = str(record["model"])
        selected = daily.loc[
            daily["model"].eq(model) & daily["training_target"].eq(residual_name)
        ]
        annual = selected.groupby("year")["residual_rank_ic"].mean()
        positive_years = int(annual.gt(0).sum())
        gates = {
            "positive_point_estimate": float(record["point_estimate"]) > 0,
            "ci_excludes_zero": float(record["ci_2_5pct"]) > 0,
            "bh_fdr_5pct": bool(record["survives_bh_fdr_5pct"]),
            "positive_year_breadth": positive_years >= required_years,
            "positive_residual_rank_ic": float(selected["residual_rank_ic"].mean()) > 0,
        }
        rows.append({
            "model": model,
            "positive_validation_years": positive_years,
            "required_positive_validation_years": required_years,
            **gates,
            "promote_to_portfolio_research": all(gates.values()),
        })
    return pd.DataFrame(rows)


def _write_report(
    cfg: dict[str, Any], summary: pd.DataFrame, inference: pd.DataFrame,
    decisions: pd.DataFrame, checks: pd.DataFrame,
) -> None:
    promoted = decisions.loc[decisions["promote_to_portfolio_research"], "model"].tolist()
    conclusion = (
        f"通过预注册门槛的模型：{promoted}。这只允许进入下一阶段组合研究，"
        "不等于已经识别可交易纯 Alpha。"
        if promoted else
        "没有模型通过预注册晋级门槛；当前残差目标不进入组合或 paper challenger。"
    )
    lines = [
        "# Idiosyncratic Target Validation v1", "", "## 研究结论", "", conclusion,
        "", "本实验只使用 2008–2015 训练和 2016–2020 验证。252 日 PIT Beta 从 2008 年开始；"
        "两条标签路径使用完全相同的样本，没有对 2006–2007 缺失 Beta 做回填。没有读取"
        " 2021–2026 结果，也没有访问或修改 future lockbox 与已冻结 paper candidate。", "",
        "## 验证汇总", "", summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 配对推断", "", inference.to_markdown(index=False, floatfmt=".6f"), "",
        "## 预先指定的晋级门槛", "", decisions.to_markdown(index=False), "",
        "## 机器验收", "", checks.to_markdown(index=False), "",
        "## 解释边界", "",
        "残差标签使用信号日可知的 PIT 行业与风格暴露，但标签本身仍是未来收益，只能用于训练和"
        "事后评价。信号层改善必须经过相同成本、成交和风格匹配基准的组合检验，随后从新日期"
        "重新开始 forward evidence。公告日基本面数据仍处于外部数据阻塞状态，未被伪造。",
        "晋级规则在成功运行前已写入本地配置，但没有独立第三方时间戳，不能冒充外部预注册。",
    ]
    path = ROOT / str(cfg["report_path"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(lake: Lake | None = None) -> dict[str, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    factors = list(load_preprocessing_config()["factor_columns"])
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    train_end = int(cfg["train"]["end_year"])
    validation_start = int(cfg["validation"]["start_year"])
    validation_end = int(cfg["validation"]["end_year"])

    yearly_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_purged_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_samples: dict[int, dict[str, TrainingSample]] = {}
    yearly_purged_samples: dict[int, dict[str, TrainingSample]] = {}
    label_audits = []
    for year in range(start_year, train_end + 1):
        frame = _read_year(lake, year, factors, cfg)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        stats, purged, samples, purged_samples, audit = build_year_training(
            frame, factors, cutoff, cfg
        )
        yearly_stats[year], yearly_purged_stats[year] = stats, purged
        yearly_samples[year], yearly_purged_samples[year] = samples, purged_samples
        label_audits.append(audit.assign(split="train"))
        print(f"idiosyncratic target train year={year} dates={len(audit)}", flush=True)

    models: dict[tuple[str, str], FittedCandidate] = {}
    fit_rows = []
    for target in (str(cfg["target"]), str(cfg["residual_target"])):
        statistics, sample = _aggregate_training(
            yearly_stats, yearly_purged_stats, yearly_samples, yearly_purged_samples,
            target, start_year, train_end,
        )
        for model_name, spec in cfg["models"].items():
            model = fit_candidate(spec, statistics, sample, len(factors), cfg)
            models[(str(model_name), target)] = model
            fit_rows.append({
                "model": model_name, "training_target": target,
                "fit_observations": model.fit_observations,
                "fit_seconds": model.fit_seconds,
                "training_dates": sample.dates,
            })

    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    validation_frames = []
    for year in range(validation_start, validation_end + 1):
        frame = _read_year(lake, year, factors, cfg)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(validation_cutoff)]
        validation_frames.append(frame)
    daily, validation_label_audit = evaluate_validation(
        validation_frames, models, factors, cfg
    )
    label_audits.append(validation_label_audit)
    label_audit = pd.concat(label_audits, ignore_index=True)
    summary = daily.groupby(["model", "training_target"], as_index=False).agg(
        observations=("residual_rank_ic", "size"),
        mean_raw_rank_ic=("raw_rank_ic", "mean"),
        mean_residual_rank_ic=("residual_rank_ic", "mean"),
        residual_rank_ic_std=("residual_rank_ic", "std"),
        positive_residual_rank_ic_fraction=("residual_rank_ic", lambda x: float(x.gt(0).mean())),
        mean_max_abs_score_style_correlation=("maximum_abs_score_style_correlation", "mean"),
    )
    annual = daily.groupby(
        ["model", "training_target", "year"], as_index=False
    ).agg(
        observations=("residual_rank_ic", "size"),
        mean_raw_rank_ic=("raw_rank_ic", "mean"),
        mean_residual_rank_ic=("residual_rank_ic", "mean"),
    )
    inference = paired_inference(daily, cfg)
    decisions = promotion_decisions(daily, inference, cfg)
    checks = pd.DataFrame([
        {"check": "development_period_ends_2020", "status": "PASS",
         "detail": str(daily["year"].max())},
        {"check": "historical_oos_not_evaluated", "status": "PASS",
         "detail": str(cfg["historical_oos_evaluation_enabled"])},
        {"check": "future_lockbox_not_accessed", "status": "PASS",
         "detail": str(cfg["future_lockbox_access_allowed"])},
        {"check": "residual_normal_equations_close", "status": (
            "PASS" if label_audit["maximum_abs_normal_equation_error"].max() < 1e-10
            else "FAIL"
        ), "detail": str(label_audit["maximum_abs_normal_equation_error"].max())},
        {"check": "paired_model_dates_complete", "status": (
            "PASS" if inference["observations"].eq(daily["asof_date"].nunique()).all()
            else "FAIL"
        ), "detail": str(inference["observations"].tolist())},
        {"check": "external_pit_data_not_fabricated", "status": "PASS",
         "detail": str(cfg["external_pit_feature_contract"]["current_status"])},
    ])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "train_period": f"{start_year}-{train_end}",
        "validation_period": f"{validation_start}-{validation_end}",
        "primary_metric": cfg["inference"]["primary_metric"],
        "only_research_dimension_changed": "training_target",
        "portfolio_or_execution_claim": False,
        "frozen_paper_candidate_changed": False,
    }])
    outputs = {
        "idiosyncratic_target_protocol": protocol,
        "idiosyncratic_target_label_audit": label_audit,
        "idiosyncratic_target_fit_audit": pd.DataFrame(fit_rows),
        "idiosyncratic_target_validation_daily": daily,
        "idiosyncratic_target_validation_summary": summary,
        "idiosyncratic_target_validation_annual": annual,
        "idiosyncratic_target_paired_inference": inference,
        "idiosyncratic_target_promotion_decisions": decisions,
        "idiosyncratic_target_checks": checks,
    }
    for table, frame in outputs.items():
        lake.write(
            table, frame, key=str(cfg["artifact_key"]),
            source="idiosyncratic_target_validation.v1",
            request={**cfg, "factor_columns": factors},
        )
    _write_report(cfg, summary, inference, decisions, checks)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("idiosyncratic target validation checks failed")
    print(inference.to_string(index=False), flush=True)
    print(decisions.to_string(index=False), flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only idiosyncratic target study")
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
