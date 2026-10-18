"""Create a decision-oriented audit of the final research conclusions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg


CONFIG_PATH = ROOT / "config" / "research_conclusion_audit.yaml"
REPORT_PATH = ROOT / "reports" / "research_conclusion_audit.md"
JSON_PATH = ROOT / "reports" / "research_conclusion_audit.json"
TABLE_ROOT = ROOT / "reports" / "tables"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def factor_multiplicity(lake: Lake, false_discovery_rate: float) -> pd.DataFrame:
    """Apply BH-FDR across the 15 neutralized single-factor spread tests."""
    bootstrap = lake.read("factor_tear_sheet_bootstrap", "v1")
    selected = bootstrap.loc[
        bootstrap["pipeline"].eq("industry_size_neutralized")
        & bootstrap["metric"].eq("gross_top_minus_bottom")
    ].copy()
    if selected.empty:
        raise RuntimeError("neutralized factor spread bootstrap is missing")
    selected["bh_fdr_adjusted_p"] = benjamini_hochberg(
        selected["one_sided_probability_nonpositive"]
    )
    selected["individual_95pct_interval_above_zero"] = selected["ci_2_5pct"].gt(0)
    selected["survives_bh_fdr"] = selected["bh_fdr_adjusted_p"].le(
        false_discovery_rate
    )
    return selected.sort_values(
        ["bh_fdr_adjusted_p", "factor"], ascending=[True, True]
    ).reset_index(drop=True)


def _pairwise_row(lake: Lake, cfg: dict, metric: str) -> pd.Series:
    pairwise = lake.read("model_benchmark_winner_pairwise", "v1")
    selected = pairwise.loc[
        pairwise["candidate"].eq(cfg["model_increment_candidate"])
        & pairwise["reference"].eq(cfg["model_increment_reference"])
        & pairwise["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"expected one model pairwise row for {metric}, got {len(selected)}")
    return selected.iloc[0]


def conclusion_decisions(
    lake: Lake, cfg: dict, multiplicity: pd.DataFrame,
) -> pd.DataFrame:
    execution = lake.read("constrained_daily_execution_summary", "v1")
    primary = execution.loc[
        execution["model"].eq(cfg["primary_model"])
        & execution["alpha_pipeline"].eq(cfg["primary_alpha_pipeline"])
        & execution["scenario"].eq(cfg["primary_execution_scenario"])
    ]
    if len(primary) != 1:
        raise RuntimeError(f"expected one primary execution row, got {len(primary)}")
    primary = primary.iloc[0]

    cost_stress = lake.read("execution_cost_stress", "v1")
    stress = cost_stress.loc[
        cost_stress["model"].eq(cfg["primary_model"])
        & cost_stress["alpha_pipeline"].eq(cfg["primary_alpha_pipeline"])
        & cost_stress["cost_bps"].eq(100)
    ]
    if len(stress) != 1:
        raise RuntimeError(f"expected one 100bp cost-stress row, got {len(stress)}")
    stress = stress.iloc[0]

    style = lake.read("style_factor_regression_summary", "v1")
    style = style.loc[
        style["scenario"].eq(cfg["primary_execution_scenario"])
    ].copy()
    if len(style) != 4:
        raise RuntimeError(f"expected four primary style regressions, got {len(style)}")
    style_all_cross_zero = bool(
        style["annualized_alpha_ci_low"].le(0).all()
        and style["annualized_alpha_ci_high"].ge(0).all()
    )

    rank_ic = _pairwise_row(lake, cfg, "rank_ic")
    model_return = _pairwise_row(lake, cfg, "net_return_annualized_arithmetic")
    increment_inconclusive = bool(
        float(rank_ic["ci_2_5pct"]) <= 0 <= float(rank_ic["ci_97_5pct"])
        and float(model_return["ci_2_5pct"]) <= 0 <= float(model_return["ci_97_5pct"])
    )
    fdr_count = int(multiplicity["survives_bh_fdr"].sum())
    factor_count = len(multiplicity)
    incremental = lake.read("incremental_alpha_bootstrap", "v1")
    incremental_alpha = incremental.loc[
        incremental["metric"].eq("incremental_net_return_annualized_arithmetic")
    ]
    if len(incremental_alpha) != 4:
        raise RuntimeError(
            f"expected four style-matched incremental alpha rows, got {len(incremental_alpha)}"
        )
    individual_incremental = int(incremental_alpha["ci_2_5pct"].gt(0).sum())
    fdr_incremental = int(
        incremental_alpha["survives_bh_fdr_5pct"].astype(bool).sum()
    )
    implementation = lake.read("incremental_alpha_comparison", "v1")
    implementation_survivors = int(
        implementation[
            "multiplicity_controlled_risk_or_efficiency_value_detected"
        ].astype(bool).sum()
    )
    hgb_neutral = incremental_alpha.loc[
        incremental_alpha["model"].eq("hgb_shallow15_locked")
        & incremental_alpha["alpha_pipeline"].eq("industry_size_neutralized")
    ].iloc[0]

    records = [
        {
            "question": "neutralized_factor_signal_exists",
            "verdict": "SUPPORTED_AS_DIAGNOSTIC_EVIDENCE",
            "evidence": (
                f"{fdr_count}/{factor_count} neutralized gross factor spreads survive "
                f"BH-FDR {float(cfg['false_discovery_rate']):.0%}"
            ),
            "statistical_interpretation": "cross-factor multiplicity controlled",
            "scope_limit": "diagnostic long-short portfolios omit borrow and full execution",
        },
        {
            "question": "hgb_adds_stable_increment_over_ridge",
            "verdict": "NOT_ESTABLISHED" if increment_inconclusive else "REVIEW_REQUIRED",
            "evidence": (
                f"RankIC delta 95% CI [{float(rank_ic['ci_2_5pct']):.4f}, "
                f"{float(rank_ic['ci_97_5pct']):.4f}]; annual return delta 95% CI "
                f"[{float(model_return['ci_2_5pct']):.2%}, "
                f"{float(model_return['ci_97_5pct']):.2%}]"
            ),
            "statistical_interpretation": "both paired intervals include zero",
            "scope_limit": "historical OOS has been inspected",
        },
        {
            "question": "ml_adds_value_over_style_matched_baseline",
            "verdict": "MIXED_LOCAL_EVIDENCE",
            "evidence": (
                f"incremental return intervals above zero {individual_incremental}/4; "
                f"BH-FDR survivors {fdr_incremental}/4; HGB neutral delta "
                f"{float(hgb_neutral['point_estimate']):.2%} with 95% CI "
                f"[{float(hgb_neutral['ci_2_5pct']):.2%}, "
                f"{float(hgb_neutral['ci_97_5pct']):.2%}]; multiplicity-controlled "
                f"risk/efficiency survivors {implementation_survivors}/4"
            ),
            "statistical_interpretation": (
                "one local alpha interval excludes zero but not after cross-strategy BH-FDR; "
                "Ridge raw turnover efficiency survives its metric family"
            ),
            "scope_limit": "matched peers use local six-style proxies and monthly broad industry",
        },
        {
            "question": "executable_backtest_survives_costs",
            "verdict": "SUPPORTED_HISTORICALLY",
            "evidence": (
                f"20bp annualized return {float(primary['annualized_return']):.2%}, "
                f"Sharpe {float(primary['sharpe_zero_rate']):.2f}, max drawdown "
                f"{float(primary['maximum_drawdown']):.2%}; fixed-path 100bp annualized "
                f"return {float(stress['annualized_return']):.2%}"
            ),
            "statistical_interpretation": "economic stress result, not a pure-alpha test",
            "scope_limit": "close execution, approximate limits, no broker-calibrated impact",
        },
        {
            "question": "pure_stock_selection_alpha_is_significant",
            "verdict": "NOT_SUPPORTED" if style_all_cross_zero else "REVIEW_REQUIRED",
            "evidence": (
                f"style-adjusted alpha range {float(style['annualized_alpha_arithmetic'].min()):.2%} "
                f"to {float(style['annualized_alpha_arithmetic'].max()):.2%}; "
                f"all_95pct_intervals_cross_zero={style_all_cross_zero}; "
                f"R2 range {float(style['r_squared'].min()):.3f}-"
                f"{float(style['r_squared'].max()):.3f}"
            ),
            "statistical_interpretation": "CAPM alpha is absorbed by local style factors",
            "scope_limit": "local factors are research proxies, not a commercial risk model",
        },
        {
            "question": "ready_for_live_capital",
            "verdict": "NO",
            "evidence": "no unseen forward sample, paper track record, or broker fill calibration",
            "statistical_interpretation": "backtest completion is not deployment validation",
            "scope_limit": "next gate is frozen-parameter paper trading on future data",
        },
    ]
    return pd.DataFrame(records)


def lifecycle_audit() -> pd.DataFrame:
    rows = [
        ("research_mandate", "COMPLETE", "question hierarchy and falsifiable hypotheses", "final report sections 1 and 10"),
        ("data_acquisition", "COMPLETE_WITH_LIMIT", "free-source PIT lake and manifests", "BaoStock revisions prevent byte-identical redownloads"),
        ("data_quality", "COMPLETE", "schema, coverage, truncation, and key gates", "publication audit"),
        ("point_in_time_universe", "COMPLETE_WITH_LIMIT", "historical A-share snapshots delayed one day", "not official historical CSI membership"),
        ("factor_definition", "COMPLETE", "versioned registry, formulas, missingness, direction", "15-factor registry"),
        ("label_and_timing", "COMPLETE", "t signal, t+1 close execution, t+2..t+6 label", "purged split boundaries"),
        ("preprocessing_and_neutralization", "COMPLETE_WITH_LIMIT", "same-date transforms and PIT industry/size residualization", "monthly CSRC broad industry"),
        ("signal_validation", "COMPLETE", "IC, deciles, regimes, decay, LOFO, BH-FDR", "historical OOS is not pristine"),
        ("model_selection", "COMPLETE", "validation-only benchmark and parameter stability", "HGB increment over Ridge not established"),
        ("portfolio_construction", "COMPLETE", "single-name, industry, size, and turnover constraints", "long-only research policy"),
        ("execution_simulation", "COMPLETE_WITH_LIMIT", "share orders, cash, suspensions, limits, retry clock", "no intraday queue or IPO no-limit window"),
        ("cost_capacity_stress", "COMPLETE_WITH_LIMIT", "0-100bp and ADV/impact scenarios", "impact not calibrated to broker orders"),
        ("risk_and_attribution", "COMPLETE_WITH_LIMIT", "CAPM, six-style and cross-sectional risk attribution", "price index and local proxy factors"),
        ("robustness_and_inference", "COMPLETE", "paired block bootstrap, HAC, common-date tests, failed experiments", "no untouched production lockbox"),
        ("reproducibility_and_governance", "COMPLETE", "tests, immutable experiments, hashes, invalidations", "full data lake intentionally excluded from Git"),
        ("paper_live_monitoring", "NEXT_STAGE", "frozen-parameter forward evidence required", "not part of a completed historical research backtest"),
    ]
    return pd.DataFrame(rows, columns=["stage", "status", "evidence", "boundary"])


def write_outputs(
    decisions: pd.DataFrame, multiplicity: pd.DataFrame, lifecycle: pd.DataFrame,
    cfg: dict,
) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(TABLE_ROOT / "research_conclusion_decisions.csv", index=False)
    multiplicity.to_csv(TABLE_ROOT / "factor_multiplicity_control.csv", index=False)
    lifecycle.to_csv(TABLE_ROOT / "research_lifecycle_audit.csv", index=False)
    payload = {
        "experiment_id": cfg["experiment_id"],
        "decisions": decisions.to_dict(orient="records"),
        "factor_multiplicity": multiplicity.to_dict(orient="records"),
        "lifecycle": lifecycle.to_dict(orient="records"),
    }
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    decision_display = decisions[["question", "verdict", "evidence", "scope_limit"]]
    lifecycle_display = lifecycle[["stage", "status", "evidence", "boundary"]]
    fdr = multiplicity[[
        "factor", "point_estimate", "ci_2_5pct", "ci_97_5pct",
        "one_sided_probability_nonpositive", "bh_fdr_adjusted_p", "survives_bh_fdr",
    ]]
    lines = [
        "# 研究结论与全链路审计",
        "",
        "本页把研究结果拆成六个独立决策，防止把预测力、回测收益、模型增量、"
        "风险调整 Alpha 和上线资格混为一谈。",
        "",
        "## 决策矩阵",
        "",
        decision_display.to_markdown(index=False),
        "",
        "## Quant Research 生命周期",
        "",
        lifecycle_display.to_markdown(index=False),
        "",
        "结论：历史研究链路已经闭环；`paper_live_monitoring` 是下一阶段，不能用回测替代。",
        "",
        "## 单因子多重检验",
        "",
        f"在 {len(multiplicity)} 个行业+规模中性因子上，按 BH-FDR "
        f"{float(cfg['false_discovery_rate']):.0%} 控制后，"
        f"{int(multiplicity['survives_bh_fdr'].sum())} 个毛 Top–Bottom 价差仍通过。",
        "这加强了因子诊断证据，但不把未建模借券和完整成交的多空组合称为可交易策略。",
        "",
        fdr.to_markdown(index=False),
        "",
        "## 年化指标口径",
        "",
        "- 最终组合年化收益是日频净值的几何复合年化。",
        "- CAPM/风格 Alpha 是日回归截距的算术年化。",
        "- 风险贡献是周频贡献的算术年化；它与几何收益不能逐项直接相加比较。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    multiplicity = factor_multiplicity(lake, float(cfg["false_discovery_rate"]))
    decisions = conclusion_decisions(lake, cfg, multiplicity)
    lifecycle = lifecycle_audit()
    artifact_key = str(cfg["experiment_id"]).rsplit("_", 1)[-1]
    request = {**cfg, "source": "registered_research_outputs"}
    lake.write(
        "research_conclusion_decisions", decisions, key=artifact_key,
        source=f"research_conclusion_audit.{artifact_key}", request=request,
    )
    lake.write(
        "research_factor_multiplicity", multiplicity, key=artifact_key,
        source=f"research_conclusion_audit.{artifact_key}", request=request,
    )
    lake.write(
        "research_lifecycle_audit", lifecycle, key=artifact_key,
        source=f"research_conclusion_audit.{artifact_key}", request=request,
    )
    write_outputs(decisions, multiplicity, lifecycle, cfg)
    return decisions, multiplicity, lifecycle


def main() -> None:
    decisions, _, lifecycle = run()
    print(decisions.to_string(index=False))
    print(f"lifecycle stages={len(lifecycle)}")


if __name__ == "__main__":
    main()
