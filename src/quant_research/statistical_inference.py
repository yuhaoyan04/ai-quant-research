"""Phase 8 unified paired inference, block sensitivity and trial registry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.capm_attribution import newey_west_ols
from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "statistical_inference.yaml"
REPORT_PATH = ROOT / "reports" / "statistical_inference.md"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int((int(base) + int.from_bytes(digest[:4], "big")) % (2**32 - 1))


def hac_mean_record(values: pd.Series, *, scale: float, lags: int,
                    confidence_level: float) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    result = newey_west_ols(clean, np.ones((len(clean), 1)), int(lags))
    point = float(np.asarray(result["coefficients"])[0]) * scale
    standard_error = float(np.asarray(result["standard_errors"])[0]) * scale
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    t_stat = point / standard_error if standard_error > 0 else (
        np.inf if point > 0 else (-np.inf if point < 0 else np.nan)
    )
    p_value = 1.0 - NormalDist().cdf(t_stat) if np.isfinite(t_stat) else (
        0.0 if t_stat == np.inf else (1.0 if t_stat == -np.inf else np.nan)
    )
    return {
        "observations": len(clean), "point_estimate": point,
        "hac_standard_error": standard_error, "hac_t_stat": float(t_stat),
        "ci_low": point - critical * standard_error,
        "ci_high": point + critical * standard_error,
        "one_sided_p_value": float(p_value), "hac_lags": int(lags),
    }


def bootstrap_mean_record(values: pd.Series, *, scale: float, block_length: int,
                          samples: int, seed: int) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    estimates = circular_block_bootstrap_mean(clean, block_length, samples, seed) * scale
    return {
        "observations": len(clean), "point_estimate": float(clean.mean() * scale),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "one_sided_p_value": float((np.sum(estimates <= 0) + 1) / (len(estimates) + 1)),
        "block_length": int(block_length), "bootstrap_samples": int(samples),
        "random_seed": int(seed),
    }


def _factor_series(frame: pd.DataFrame, family_id: str, spec: dict[str, Any],
                   evidence_class: str) -> pd.DataFrame:
    selected = frame.loc[frame["pipeline"].eq(str(spec["pipeline"]))].copy()
    records = selected[["asof_date", "factor", str(spec["metric"])]].rename(columns={
        "asof_date": "date", "factor": "hypothesis_id",
        str(spec["metric"]): "candidate_value",
    })
    records["reference_value"] = 0.0
    records["paired_difference"] = records["candidate_value"]
    records["family_id"] = family_id
    records["hypothesis_role"] = str(spec["hypothesis_role"])
    records["evidence_class"] = evidence_class
    records["frequency"] = str(spec["frequency"])
    records["annualization_scale"] = float(spec["annualization_scale"])
    records["positive_direction"] = "positive_factor_efficacy"
    return records


def _model_incremental_series(frame: pd.DataFrame, family_id: str,
                              spec: dict[str, Any], evidence_class: str) -> pd.DataFrame:
    keys = ["model", "alpha_pipeline", "strategy", "trade_date"]
    if frame.duplicated([*keys, "portfolio_type"]).any():
        raise ValueError("duplicate incremental-alpha portfolio/date keys")
    pivot = frame.pivot(index=keys, columns="portfolio_type", values="net_return").reset_index()
    pivot = pivot.rename(columns={
        "trade_date": "date", "ml_model": "candidate_value",
        "style_matched_benchmark": "reference_value",
    })
    pivot["hypothesis_id"] = (
        pivot["model"].astype(str) + "__" + pivot["alpha_pipeline"].astype(str)
    )
    pivot["paired_difference"] = pivot["candidate_value"] - pivot["reference_value"]
    pivot["family_id"] = family_id
    pivot["hypothesis_role"] = str(spec["hypothesis_role"])
    pivot["evidence_class"] = evidence_class
    pivot["frequency"] = str(spec["frequency"])
    pivot["annualization_scale"] = float(spec["annualization_scale"])
    pivot["positive_direction"] = "ml_model_minus_style_benchmark_positive"
    return pivot


def _portfolio_series(frame: pd.DataFrame, family_id: str,
                      spec: dict[str, Any], evidence_class: str) -> pd.DataFrame:
    if frame.duplicated(["method", "trade_date"]).any():
        raise ValueError("duplicate portfolio-construction method/date keys")
    pivot = frame.pivot(index="trade_date", columns="method", values="net_return")
    reference = str(spec["reference"])
    records = []
    for method in sorted(set(pivot.columns) - {reference}):
        current = pd.DataFrame({
            "date": pivot.index.astype(str), "hypothesis_id": method,
            "candidate_value": pivot[method].to_numpy(),
            "reference_value": pivot[reference].to_numpy(),
        })
        current["paired_difference"] = current["candidate_value"] - current["reference_value"]
        records.append(current)
    result = pd.concat(records, ignore_index=True)
    result["family_id"] = family_id
    result["hypothesis_role"] = str(spec["hypothesis_role"])
    result["evidence_class"] = evidence_class
    result["frequency"] = str(spec["frequency"])
    result["annualization_scale"] = float(spec["annualization_scale"])
    result["positive_direction"] = "candidate_method_minus_equal_weight_positive"
    return result


def build_series(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for family_id, spec in cfg["families"].items():
        source = spec["source"]
        frame = lake.read(str(source["table"]), str(source["key"]))
        if frame.empty:
            raise RuntimeError(f"inference source missing: {source}")
        if family_id.startswith("factor_"):
            result = _factor_series(frame, family_id, spec, str(cfg["evidence_class"]))
        elif family_id == "model_incremental_alpha":
            result = _model_incremental_series(
                frame, family_id, spec, str(cfg["evidence_class"]),
            )
        else:
            result = _portfolio_series(frame, family_id, spec, str(cfg["evidence_class"]))
        frames.append(result)
    result = pd.concat(frames, ignore_index=True, sort=False)
    ordered = [
        "family_id", "hypothesis_id", "hypothesis_role", "evidence_class", "date",
        "candidate_value", "reference_value", "paired_difference", "frequency",
        "annualization_scale", "positive_direction",
    ]
    result = result[ordered]
    result["is_complete_pair"] = result[["candidate_value", "reference_value"]].notna().all(axis=1)
    return result.sort_values(["family_id", "hypothesis_id", "date"]).reset_index(drop=True)


def run_inference(series: pd.DataFrame, cfg: dict[str, Any]
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hac_rows, bootstrap_rows, alignment_rows = [], [], []
    for (family_id, hypothesis_id), group in series.groupby(
        ["family_id", "hypothesis_id"], sort=True
    ):
        spec = cfg["families"][family_id]
        scale = float(spec["annualization_scale"])
        complete = group["is_complete_pair"].astype(bool)
        alignment_rows.append({
            "family_id": family_id, "hypothesis_id": hypothesis_id,
            "scheduled_observations": len(group), "complete_observations": int(complete.sum()),
            "missing_observations": int((~complete).sum()),
            "duplicate_dates": int(group.duplicated("date").sum()),
            "start_date": str(group["date"].min()), "end_date": str(group["date"].max()),
        })
        values = group.loc[complete, "paired_difference"]
        for lags in spec["hac_lags"]:
            hac_rows.append({
                "family_id": family_id, "hypothesis_id": hypothesis_id,
                **hac_mean_record(
                    values, scale=scale, lags=int(lags),
                    confidence_level=float(cfg["confidence_level"]),
                ),
            })
        for block in spec["block_lengths"]:
            seed = _stable_seed(int(cfg["random_seed"]), family_id, str(hypothesis_id), str(block))
            bootstrap_rows.append({
                "family_id": family_id, "hypothesis_id": hypothesis_id,
                **bootstrap_mean_record(
                    values, scale=scale, block_length=int(block),
                    samples=int(cfg["bootstrap_samples"]), seed=seed,
                ),
            })
    hac = pd.DataFrame(hac_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    alignment = pd.DataFrame(alignment_rows)
    primary = []
    for family_id, group in bootstrap.groupby("family_id", sort=True):
        spec = cfg["families"][family_id]
        selected = group.loc[group["block_length"].eq(int(spec["primary_block_length"]))].copy()
        selected["bh_fdr_adjusted_p"] = benjamini_hochberg(selected["one_sided_p_value"])
        selected["false_discovery_rate"] = float(cfg["false_discovery_rate"])
        selected["survives_bh_fdr"] = (
            selected["bh_fdr_adjusted_p"] <= float(cfg["false_discovery_rate"])
        ) & selected["point_estimate"].gt(0)
        threshold = float(spec["economic_materiality_threshold"])
        selected["economic_materiality_threshold"] = threshold
        selected["economically_material_positive"] = selected["point_estimate"].ge(threshold)
        selected["claim_status"] = np.select(
            [
                selected["survives_bh_fdr"] & selected["economically_material_positive"],
                selected["survives_bh_fdr"],
            ],
            [
                "historical_diagnostic_positive_not_future_evidence",
                "statistical_signal_below_materiality_threshold",
            ],
            default="positive_value_not_established_after_family_fdr",
        )
        selected["family_test_count"] = len(selected)
        primary.append(selected)
    return hac, bootstrap, pd.concat(primary, ignore_index=True), alignment


def build_trial_registry(lake: Lake) -> pd.DataFrame:
    """Inventory major tried specifications without merging unlike FDR questions."""
    sources = [
        ("model_family_grid", "model_benchmark_validation_grid", "v1", ["candidate"]),
        ("hgb_hyperparameter_grid", "hgb_hyperparameter_summary", "v1", ["candidate"]),
        ("rf_hyperparameter_grid", "rf_hyperparameter_validation_summary", "v1", ["candidate"]),
        ("preprocessing_grid", "factor_preprocessing_ablation_grid", "v2", ["variant", "ridge_penalty"]),
        ("label_grid", "label_horizon_validation_grid", "v1", ["target_spec", "ridge_penalty"]),
        ("factor_family", "factor_registry", "v1", ["factor"]),
        ("portfolio_sensitivity", "portfolio_sensitivity_summary", "v1", ["split", "top_n", "weighting"]),
        ("portfolio_construction", "portfolio_construction_execution_summary", "v1", ["method"]),
        ("transaction_cost", "transaction_cost_protocol", "v1", ["scenario_id"]),
        ("incremental_alpha", "incremental_alpha_comparison", "v1", ["model", "alpha_pipeline"]),
        ("risk_aware_portfolio", "risk_aware_portfolio_summary", "v2", ["model", "alpha_pipeline", "scenario"]),
    ]
    rows = []
    for category, table, key, columns in sources:
        frame = lake.read(table, key)
        if frame.empty:
            raise RuntimeError(f"trial-registry source missing: {table}/{key}")
        unique = frame[columns].drop_duplicates().astype(str)
        for values in unique.itertuples(index=False, name=None):
            rows.append({
                "trial_category": category,
                "trial_id": "__".join(values),
                "source_table": f"{table}/{key}",
                "trial_count_unit": "unique_" + "_".join(columns),
                "included_in_cross_category_fdr": False,
                "reason_not_pooled": "distinct_research_question_or_selection_stage",
            })
    return pd.DataFrame(rows).sort_values(["trial_category", "trial_id"]).reset_index(drop=True)


def acceptance_checks(series: pd.DataFrame, hac: pd.DataFrame, bootstrap: pd.DataFrame,
                      fdr: pd.DataFrame, alignment: pd.DataFrame,
                      trials: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    expected_families = set(cfg["families"])
    expected_counts = {"factor_rank_ic": 15, "factor_gross_spread": 15,
                       "model_incremental_alpha": 4,
                       "portfolio_construction_increment": 3}
    observed_counts = fdr.groupby("family_id")["hypothesis_id"].nunique().to_dict()
    block_ok = all(
        set(bootstrap.loc[bootstrap["family_id"].eq(family), "block_length"].astype(int))
        == set(map(int, spec["block_lengths"]))
        for family, spec in cfg["families"].items()
    )
    hac_ok = all(
        set(hac.loc[hac["family_id"].eq(family), "hac_lags"].astype(int))
        == set(map(int, spec["hac_lags"]))
        for family, spec in cfg["families"].items()
    )
    paired = alignment["family_id"].isin([
        "model_incremental_alpha", "portfolio_construction_increment",
    ])
    primary_raw = fdr.loc[
        fdr["family_id"].eq("model_incremental_alpha")
        & fdr["hypothesis_id"].eq("hgb_shallow15_locked__raw")
    ]
    rows = [
        ("raw_series_keys_unique", not series.duplicated(["family_id", "hypothesis_id", "date"]).any(), len(series)),
        ("paired_daily_clocks_complete", alignment.loc[paired, "missing_observations"].eq(0).all() and alignment.loc[paired, "duplicate_dates"].eq(0).all(), alignment.loc[paired, "complete_observations"].min()),
        ("families_separate_and_complete", set(fdr["family_id"]) == expected_families and observed_counts == expected_counts, observed_counts),
        ("block_length_sensitivity_complete", block_ok, sorted(bootstrap["block_length"].unique())),
        ("hac_lag_sensitivity_complete", hac_ok, sorted(hac["hac_lags"].unique())),
        ("p_values_and_q_values_valid", bootstrap["one_sided_p_value"].between(0, 1).all() and hac["one_sided_p_value"].between(0, 1).all() and fdr["bh_fdr_adjusted_p"].between(0, 1).all(), len(fdr)),
        ("point_estimates_recompute_from_raw", all(np.isclose(row.point_estimate, series.loc[(series["family_id"].eq(row.family_id)) & (series["hypothesis_id"].eq(row.hypothesis_id)), "paired_difference"].mean() * float(cfg["families"][row.family_id]["annualization_scale"]), atol=1e-12) for row in fdr.itertuples()), len(fdr)),
        ("trial_registry_covers_major_search_space", set(trials["trial_category"]) == {"model_family_grid", "hgb_hyperparameter_grid", "rf_hyperparameter_grid", "preprocessing_grid", "label_grid", "factor_family", "portfolio_sensitivity", "portfolio_construction", "transaction_cost", "incremental_alpha", "risk_aware_portfolio"}, trials.groupby("trial_category").size().to_dict()),
        ("primary_incremental_alpha_not_overclaimed", len(primary_raw) == 1 and (not bool(primary_raw.iloc[0]["survives_bh_fdr"])) and str(primary_raw.iloc[0]["claim_status"]) == "positive_value_not_established_after_family_fdr", primary_raw[["point_estimate", "ci_low", "ci_high", "bh_fdr_adjusted_p", "claim_status"]].to_dict(orient="records")),
        ("historical_oos_not_claimed_independent", not bool(cfg["claims_independent_evidence"]) and str(cfg["evidence_class"]) == "post_historical_oos_adaptation", cfg["scope_note"]),
    ]
    return pd.DataFrame([{
        "check_name": name, "status": "PASS" if bool(condition) else "FAIL",
        "observed": str(observed),
    } for name, condition, observed in rows])


def run(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    series = build_series(lake, cfg)
    hac, bootstrap, fdr, alignment = run_inference(series, cfg)
    trials = build_trial_registry(lake)
    checks = acceptance_checks(series, hac, bootstrap, fdr, alignment, trials, cfg)
    family_summary = fdr.groupby("family_id", as_index=False).agg(
        hypotheses=("hypothesis_id", "size"),
        individually_positive=("ci_low", lambda x: int(x.gt(0).sum())),
        survives_family_bh_fdr=("survives_bh_fdr", "sum"),
        economically_material_positive=("economically_material_positive", "sum"),
        minimum_q_value=("bh_fdr_adjusted_p", "min"),
    )
    sensitivity = bootstrap.groupby(["family_id", "hypothesis_id"], as_index=False).agg(
        point_estimate=("point_estimate", "first"),
        minimum_ci_low=("ci_low", "min"), maximum_ci_low=("ci_low", "max"),
        minimum_ci_high=("ci_high", "min"), maximum_ci_high=("ci_high", "max"),
        minimum_p_value=("one_sided_p_value", "min"), maximum_p_value=("one_sided_p_value", "max"),
        all_block_intervals_positive=("ci_low", lambda x: bool(x.gt(0).all())),
        all_block_intervals_negative=("ci_high", lambda x: bool(x.lt(0).all())),
    )
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "paired_comparisons_required": True, "hac_kernel": "Bartlett",
        "bootstrap_method": "circular_block_bootstrap",
        "block_length_selected_from_results": False,
        "fdr_method": "Benjamini-Hochberg_within_prespecified_family",
        "future_success_probability_claimed": False,
        "family_count": len(cfg["families"]), "registered_trial_count": len(trials),
    }])
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all Phase 8 inference protocol assertions passed",
        }])
    tables = {
        "statistical_inference_protocol": protocol,
        "statistical_inference_series": series,
        "statistical_inference_alignment": alignment,
        "statistical_inference_hac": hac,
        "statistical_inference_block_bootstrap": bootstrap,
        "statistical_inference_fdr": fdr,
        "statistical_inference_family_summary": family_summary,
        "statistical_inference_block_sensitivity": sensitivity,
        "statistical_trial_registry": trials,
        "statistical_inference_checks": checks,
        "statistical_inference_failed_assertions": failures,
    }
    for table, frame in tables.items():
        lake.write(table, frame, key="v1", source="statistical_inference.v1", request=cfg)
    raw = fdr.loc[
        fdr["family_id"].eq("model_incremental_alpha")
        & fdr["hypothesis_id"].eq("hgb_shallow15_locked__raw")
    ].iloc[0]
    lines = [
        "# Phase 8：统一统计推断与多重检验", "",
        "## 证据边界", "",
        "本阶段在已查看的 2021–2026 historical OOS 上统一推断口径，属于 post-historical-OOS adaptation；不选择新模型、因子、块长或组合方法，也不提供未来盈利概率。", "",
        "## 检验族", "", family_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "BH-FDR 只在同一研究问题内部执行，因子 RankIC、因子多空毛收益、模型增量和组合方法增量绝不混池。", "",
        "## 主模型增量 Alpha", "",
        f"raw HGB 相对风格复制基准的年化算术收益差为 {raw['point_estimate']:.2%}，主 20 日块长 95% 区间 [{raw['ci_low']:.2%}, {raw['ci_high']:.2%}]，族内 BH-FDR q={raw['bh_fdr_adjusted_p']:.4f}。结论：{raw['claim_status']}。", "",
        "## Block length 敏感性", "", sensitivity.to_markdown(index=False, floatfmt=".6f"), "",
        "## 完整试验台账", "",
        trials.groupby("trial_category").size().rename("trials").reset_index().to_markdown(index=False), "",
        "## 机器验收", "", checks.to_markdown(index=False), "",
        "> 统计显著不自动等于可交易 Alpha；经济阈值、成本、换手、容量与未来 lockbox/paper evidence 仍是独立门禁。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(family_summary.to_string(index=False), flush=True)
    print(checks.to_string(index=False), flush=True)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Phase 8 inference protocol checks failed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 8 unified statistical inference")
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
