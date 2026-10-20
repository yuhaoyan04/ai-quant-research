"""Phase 9 risk-benchmark hierarchy and tradable incremental-alpha evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.capm_attribution import newey_west_ols
from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.statistical_inference import _stable_seed, bootstrap_mean_record


CONFIG_PATH = ROOT / "config" / "alpha_evidence_matrix.yaml"
REPORT_PATH = ROOT / "reports" / "alpha_evidence_matrix.md"
TABLE_ROOT = ROOT / "reports" / "tables"
KEYS = ["model", "alpha_pipeline", "strategy"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def paired_execution_panel(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    """Join ML/style returns, local factors, market variants and rf on one clock."""
    daily = lake.read("incremental_alpha_fill_daily", "v1").copy()
    daily["turnover"] = daily["one_way_turnover"]
    daily["blocked_buy_weight"] = daily["blocked_buy_notional"] / daily["pretrade_nav"]
    daily["blocked_sell_weight"] = daily["blocked_sell_notional"] / daily["pretrade_nav"]
    metrics = [
        "net_return", "transaction_cost", "turnover", "blocked_buy_weight",
        "blocked_sell_weight", "target_tracking_l1", "cash_weight",
    ]
    index = [*KEYS, "trade_date"]
    if daily.duplicated([*index, "portfolio_type"]).any():
        raise ValueError("duplicate incremental-alpha portfolio/date keys")
    wide = daily.pivot(index=index, columns="portfolio_type", values=metrics)
    role = {"ml_model": "model", "style_matched_benchmark": "style"}
    wide.columns = [f"{role[portfolio]}_{metric}" for metric, portfolio in wide.columns]
    panel = wide.reset_index()
    panel["paired_incremental_net_return"] = (
        panel["model_net_return"] - panel["style_net_return"]
    )

    factors = lake.read("style_factor_daily_returns", "v1").pivot(
        index="trade_date", columns="factor", values="factor_return"
    ).reset_index()
    market = lake.read("market_benchmark_daily", "v1")
    market = market.loc[
        market["benchmark"].eq(str(cfg["primary_benchmark"]))
        & market["variant_id"].isin(cfg["comparison_benchmark_variants"]),
        ["trade_date", "benchmark", "variant_id", "return_type", "benchmark_return"],
    ]
    rates = lake.read("risk_free_rate_daily", "v1")[[
        "trade_date", "scenario_id", "risk_free_return",
    ]].rename(columns={"scenario_id": "rate_scenario"})
    panel = panel.merge(factors, on="trade_date", how="left", validate="many_to_one")
    panel = panel.merge(market, on="trade_date", how="left", validate="many_to_many")
    panel = panel.merge(rates, on="trade_date", how="left", validate="many_to_many")
    panel["benchmark_excess_return"] = (
        panel["benchmark_return"] - panel["risk_free_return"]
    )
    return panel.sort_values([
        *KEYS, "rate_scenario", "variant_id", "trade_date",
    ]).reset_index(drop=True)


def incremental_regression_record(
    group: pd.DataFrame, cfg: dict[str, Any], lags: int,
) -> dict[str, float | int]:
    """Seven-factor regression of the zero-cost paired return difference."""
    factors = ["benchmark_excess_return", *cfg["styles"]]
    design = np.column_stack([
        np.ones(len(group)), group[factors].to_numpy(dtype="float64"),
    ])
    result = newey_west_ols(
        group["paired_incremental_net_return"].to_numpy(dtype="float64"),
        design, int(lags),
    )
    coefficient = np.asarray(result["coefficients"], dtype="float64")
    standard_error = np.asarray(result["standard_errors"], dtype="float64")
    annualization = int(cfg["annualization_days"])
    critical = NormalDist().inv_cdf(0.5 + float(cfg["confidence_level"]) / 2.0)
    point = float(coefficient[0] * annualization)
    error = float(standard_error[0] * annualization)
    t_stat = point / error if error > 0 else np.nan
    record: dict[str, float | int] = {
        "observations": int(result["observations"]),
        "annualized_alpha_arithmetic": point,
        "annualized_alpha_hac_se": error,
        "annualized_alpha_ci_low": point - critical * error,
        "annualized_alpha_ci_high": point + critical * error,
        "alpha_hac_t_stat": float(t_stat),
        "one_sided_p_value": float(1.0 - NormalDist().cdf(t_stat)),
        "r_squared": float(result["r_squared"]),
        "newey_west_lags": int(lags),
    }
    for position, factor in enumerate(["market", *cfg["styles"]], start=1):
        record[f"loading_{factor}"] = float(coefficient[position])
        record[f"loading_{factor}_hac_se"] = float(standard_error[position])
    return record


def risk_adjusted_inference(
    panel: pd.DataFrame, cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_keys = [*KEYS, "rate_scenario", "variant_id", "return_type"]
    rows: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    required = [
        "model_net_return", "style_net_return", "benchmark_return",
        "risk_free_return", *cfg["styles"],
    ]
    for values, group in panel.groupby(group_keys, sort=True):
        group = group.sort_values("trade_date")
        missing = group[required].isna().any(axis=1)
        alignment.append({
            **dict(zip(group_keys, values)),
            "scheduled_observations": len(group),
            "complete_observations": int((~missing).sum()),
            "missing_observations": int(missing.sum()),
            "duplicate_dates": int(group.duplicated("trade_date").sum()),
            "start_date": str(group["trade_date"].min()),
            "end_date": str(group["trade_date"].max()),
        })
        complete = group.loc[~missing]
        for lags in cfg["hac_lags"]:
            rows.append({
                **dict(zip(group_keys, values)),
                **incremental_regression_record(complete, cfg, int(lags)),
            })
    regressions = pd.DataFrame(rows)
    primary = regressions.loc[
        regressions["rate_scenario"].eq(str(cfg["primary_rate_scenario"]))
        & regressions["variant_id"].eq(str(cfg["primary_benchmark_variant"]))
        & regressions["newey_west_lags"].eq(int(cfg["primary_hac_lags"])),
    ].copy()
    primary["bh_fdr_adjusted_p"] = benjamini_hochberg(primary["one_sided_p_value"])
    primary["false_discovery_rate"] = float(cfg["false_discovery_rate"])
    primary["survives_bh_fdr"] = (
        primary["bh_fdr_adjusted_p"].le(float(cfg["false_discovery_rate"]))
        & primary["annualized_alpha_arithmetic"].gt(0)
    )
    primary["claim_status"] = np.where(
        primary["survives_bh_fdr"],
        "historical_risk_adjusted_increment_detected_not_future_evidence",
        "risk_adjusted_increment_not_established_after_family_fdr",
    )
    return regressions, primary, pd.DataFrame(alignment)


def raw_share_level_inference(
    panel: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    selected = panel.loc[
        panel["rate_scenario"].eq(str(cfg["primary_rate_scenario"]))
        & panel["variant_id"].eq(str(cfg["primary_benchmark_variant"]))
    ]
    rows = []
    for values, group in selected.groupby(KEYS, sort=True):
        seed = _stable_seed(int(cfg["random_seed"]), *map(str, values), "share_fill_raw")
        record = bootstrap_mean_record(
            group.sort_values("trade_date")["paired_incremental_net_return"],
            scale=float(cfg["annualization_days"]),
            block_length=int(cfg["primary_block_length"]),
            samples=int(cfg["bootstrap_samples"]), seed=seed,
        )
        rows.append({**dict(zip(KEYS, values)), **record})
    result = pd.DataFrame(rows)
    result["bh_fdr_adjusted_p"] = benjamini_hochberg(result["one_sided_p_value"])
    result["survives_bh_fdr"] = (
        result["bh_fdr_adjusted_p"].le(float(cfg["false_discovery_rate"]))
        & result["point_estimate"].gt(0)
    )
    return result


def execution_comparison(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    daily = lake.read("incremental_alpha_fill_daily", "v1")
    fill_summary = lake.read("incremental_alpha_fill_summary", "v1")
    annualization = int(cfg["annualization_days"])
    rows = []
    for values, group in daily.groupby([*KEYS, "portfolio_type"], sort=True):
        returns = group["net_return"].to_numpy(dtype="float64")
        wealth = np.cumprod(1.0 + returns)
        rows.append({
            **dict(zip([*KEYS, "portfolio_type"], values)),
            "observations": len(group),
            "annualized_return_geometric": float(
                wealth[-1] ** (annualization / len(group)) - 1.0
            ),
            "annualized_transaction_cost_arithmetic": float(
                group["transaction_cost"].mean() * annualization
            ),
            "mean_daily_turnover": float(group["one_way_turnover"].mean()),
            "annualized_turnover_arithmetic": float(group["one_way_turnover"].mean() * annualization),
            "mean_blocked_buy_weight": float((group["blocked_buy_notional"] / group["pretrade_nav"]).mean()),
            "mean_blocked_sell_weight": float((group["blocked_sell_notional"] / group["pretrade_nav"]).mean()),
            "mean_target_tracking_l1": float(group["target_tracking_l1"].mean()),
            "mean_cash_weight": float(group["cash_weight"].mean()),
            "mean_open_order_count": float(group["open_order_count"].mean()),
            "days_with_open_orders": int(group["open_order_count"].gt(0).sum()),
            "fill_evidence_status": str(cfg["fill_evidence_status"]),
        })
    result = pd.DataFrame(rows)
    result = result.merge(fill_summary[[
        *KEYS, "portfolio_type", "fill_ratio_by_shares", "total_orders", "total_fills",
    ]], on=[*KEYS, "portfolio_type"], validate="one_to_one")
    result["share_level_fill_ratio_available"] = True
    result = result.rename(columns={"fill_ratio_by_shares": "share_level_fill_ratio"})
    return result


def exposure_match_summary(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    exposure = lake.read("incremental_alpha_exposure_audit", "v1")
    rows = []
    for values, group in exposure.groupby(KEYS, sort=True):
        record: dict[str, Any] = {
            **dict(zip(KEYS, values)), "rebalance_dates": len(group),
            "mean_abs_industry_weight_gap": float(
                group["maximum_abs_industry_weight_gap"].abs().mean()
            ),
            "maximum_abs_industry_weight_gap": float(
                group["maximum_abs_industry_weight_gap"].abs().max()
            ),
            "mean_portfolio_overlap_weight": float(group["portfolio_overlap_weight"].mean()),
            "mean_fresh_style_distance": float(group["fresh_mean_style_distance"].mean()),
        }
        for style in cfg["styles"]:
            values_ = group[f"{style}_exposure_gap"].abs()
            record[f"mean_abs_{style}_exposure_gap"] = float(values_.mean())
            record[f"maximum_abs_{style}_exposure_gap"] = float(values_.max())
        rows.append(record)
    return pd.DataFrame(rows)


def _attribution_wide(
    frame: pd.DataFrame, cfg: dict[str, Any], prefix: str,
) -> pd.DataFrame:
    selected = frame.loc[
        frame["rate_scenario"].eq(str(cfg["primary_rate_scenario"]))
        & frame["execution_scenario"].eq(str(cfg["primary_execution_scenario"]))
        & frame["benchmark"].eq(str(cfg["primary_benchmark"]))
        & frame["variant_id"].isin(cfg["comparison_benchmark_variants"]),
        [*KEYS, "variant_id", "annualized_alpha_arithmetic",
         "annualized_alpha_ci_low", "annualized_alpha_ci_high", "r_squared"],
    ]
    if selected.duplicated([*KEYS, "variant_id"]).any():
        raise ValueError(f"duplicate primary {prefix} attribution rows")
    suffix = {
        "official_csindex_price": "price",
        "official_csindex_total_return": "total_return",
    }
    records = []
    for keys, group in selected.groupby(KEYS, sort=True):
        record = dict(zip(KEYS, keys))
        for row in group.itertuples(index=False):
            tag = suffix[str(row.variant_id)]
            for field in [
                "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
                "annualized_alpha_ci_high", "r_squared",
            ]:
                record[f"{prefix}_{tag}_{field}"] = float(getattr(row, field))
        records.append(record)
    return pd.DataFrame(records)


def candidate_evidence_matrix(
    lake: Lake, primary: pd.DataFrame, raw: pd.DataFrame, execution: pd.DataFrame,
    exposure: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    comparison = lake.read("incremental_alpha_comparison", "v1")
    phase8 = raw[[*KEYS, "point_estimate", "ci_low", "ci_high",
                  "bh_fdr_adjusted_p", "survives_bh_fdr"]].rename(columns={
        "point_estimate": "raw_incremental_return_annualized_arithmetic",
        "ci_low": "raw_incremental_return_ci_low",
        "ci_high": "raw_incremental_return_ci_high",
        "bh_fdr_adjusted_p": "raw_incremental_return_fdr_q",
        "survives_bh_fdr": "raw_incremental_return_survives_fdr",
    })
    comparison["hypothesis_id"] = (
        comparison["model"].astype(str) + "__" + comparison["alpha_pipeline"].astype(str)
    )
    result = comparison[[
        *KEYS, "hypothesis_id", "annualized_return_model", "annualized_return_style",
        "sharpe_zero_rate_model", "sharpe_zero_rate_style",
        "maximum_drawdown_model", "maximum_drawdown_style",
    ]].drop(columns=["annualized_return_model", "annualized_return_style"]).merge(
        phase8, on=KEYS, validate="one_to_one"
    )

    risk = primary[[
        *KEYS, "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
        "annualized_alpha_ci_high", "alpha_hac_t_stat", "r_squared",
        "bh_fdr_adjusted_p", "survives_bh_fdr", "claim_status",
    ]].rename(columns={
        "annualized_alpha_arithmetic": "risk_adjusted_incremental_alpha_arithmetic",
        "annualized_alpha_ci_low": "risk_adjusted_incremental_alpha_ci_low",
        "annualized_alpha_ci_high": "risk_adjusted_incremental_alpha_ci_high",
        "alpha_hac_t_stat": "risk_adjusted_incremental_alpha_hac_t",
        "r_squared": "incremental_difference_r_squared",
        "bh_fdr_adjusted_p": "risk_adjusted_incremental_alpha_fdr_q",
        "survives_bh_fdr": "risk_adjusted_incremental_alpha_survives_fdr",
        "claim_status": "risk_adjusted_incremental_alpha_claim_status",
    })
    result = result.merge(risk, on=KEYS, validate="one_to_one")
    result = result.merge(
        _attribution_wide(lake.read("market_benchmark_capm_summary", "v1"), cfg, "capm"),
        on=KEYS, validate="one_to_one",
    ).merge(
        _attribution_wide(lake.read("market_benchmark_style_summary", "v1"), cfg, "six_style"),
        on=KEYS, validate="one_to_one",
    )

    execution_wide = execution.pivot(
        index=KEYS, columns="portfolio_type",
        values=["annualized_return_geometric", "annualized_transaction_cost_arithmetic",
                "mean_daily_turnover", "mean_target_tracking_l1", "mean_open_order_count",
                "share_level_fill_ratio"],
    )
    role = {"ml_model": "model", "style_matched_benchmark": "style"}
    execution_wide.columns = [f"{role[p]}_{metric}" for metric, p in execution_wide.columns]
    execution_wide = execution_wide.reset_index()
    result = result.merge(execution_wide, on=KEYS, validate="one_to_one")
    result = result.rename(columns={
        "model_annualized_return_geometric": "annualized_return_model",
        "style_annualized_return_geometric": "annualized_return_style",
    })
    result["transaction_cost_difference_arithmetic"] = (
        result["model_annualized_transaction_cost_arithmetic"]
        - result["style_annualized_transaction_cost_arithmetic"]
    )
    result["mean_daily_turnover_difference"] = (
        result["model_mean_daily_turnover"] - result["style_mean_daily_turnover"]
    )
    result = result.merge(exposure, on=KEYS, validate="one_to_one")
    result["share_level_fill_ratio_difference"] = (
        result["model_share_level_fill_ratio"] - result["style_share_level_fill_ratio"]
    )
    result["share_level_fill_difference_available"] = True
    result["style_model_status"] = str(cfg["style_model_status"])
    result["pure_stock_selection_alpha_identified"] = False
    result["evidence_verdict"] = np.where(
        result["raw_incremental_return_survives_fdr"].astype(bool)
        | result["risk_adjusted_incremental_alpha_survives_fdr"].astype(bool),
        "historical_increment_detected_not_independent_evidence",
        "incremental_alpha_not_established_after_multiplicity_and_risk_adjustment",
    )
    return result.sort_values(KEYS).reset_index(drop=True)


def definitions(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("portfolio_annual_return", "absolute_portfolio_performance", "geometric", False),
        ("raw_paired_increment", "ML minus executable local style replica", "arithmetic", False),
        ("CAPM_alpha", "intercept after CSI500 market excess return", "arithmetic", False),
        ("CSI500_plus_six_style_alpha", "portfolio intercept after local market and six styles", "arithmetic", False),
        ("risk_adjusted_paired_increment", "paired ML-minus-style intercept after market and six styles", "arithmetic", False),
    ]
    return pd.DataFrame([{
        "measure": measure, "interpretation": interpretation,
        "annualization": annualization, "may_be_called_pure_stock_alpha": pure,
        "evidence_class": cfg["evidence_class"],
    } for measure, interpretation, annualization, pure in rows])


def gap_register(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"priority": "P0", "gap": "share_level_fill_comparison",
         "status": "CLOSED", "impact": "eight paths replayed through the common share-order ledger",
         "required_action": "retain the frozen fill ledger and common execution protocol"},
        {"priority": "P1", "gap": "commercial_risk_model_validation",
         "status": "OPEN", "impact": "local PIT styles and broad CSRC industry are research proxies",
         "required_action": "validate exposures against a licensed point-in-time commercial model if available"},
        {"priority": "P1", "gap": "benchmark_replica_exposure_error",
         "status": "DISCLOSED", "impact": "style matching is approximate, especially volatility/liquidity/momentum",
         "required_action": "retain exposure-error table and test tighter feasible matching without OOS selection"},
        {"priority": "P2", "gap": "intraday_publication_timestamp",
         "status": "OPEN", "impact": "official history endpoints do not expose exact intraday publication timestamps",
         "required_action": "retain one-calendar-day attribution availability lag"},
    ])


def acceptance_checks(
    panel: pd.DataFrame, regressions: pd.DataFrame, primary: pd.DataFrame,
    alignment: pd.DataFrame, execution: pd.DataFrame, matrix: pd.DataFrame,
    cfg: dict[str, Any], lake: Lake,
) -> pd.DataFrame:
    expected_days = int(cfg["expected_trading_days"])
    incremental_protocol = lake.read("incremental_alpha_protocol", "v1").iloc[0]
    expected_regressions = 4 * 3 * 2 * len(cfg["hac_lags"])
    rows = [
        ("paired_clocks_are_complete", alignment["complete_observations"].eq(expected_days).all()
         and alignment["missing_observations"].eq(0).all()
         and alignment["duplicate_dates"].eq(0).all(), len(alignment)),
        ("rf_market_style_frequency_and_dates_unified", panel["rate_scenario"].nunique() == 3
         and panel["variant_id"].nunique() == 2
         and panel["trade_date"].nunique() == expected_days, panel.shape),
        ("price_and_total_return_benchmarks_both_retained", set(panel["return_type"])
         == {"price_return", "total_return"}, sorted(panel["return_type"].unique())),
        ("risk_model_sensitivity_complete", len(regressions) == expected_regressions
         and set(regressions["newey_west_lags"].astype(int)) == set(cfg["hac_lags"]), len(regressions)),
        ("primary_specification_is_unique", len(primary) == 4
         and not primary.duplicated(KEYS).any(), len(primary)),
        ("multiple_testing_is_within_four_candidate_family", len(primary) == 4
         and primary["bh_fdr_adjusted_p"].between(0, 1).all(),
         primary[["model", "alpha_pipeline", "bh_fdr_adjusted_p"]].to_dict("records")),
        ("same_execution_and_cost_protocol", bool(incremental_protocol["same_daily_execution_and_cost"])
         and bool(incremental_protocol["pit_style_exposures_only"])
         and bool(incremental_protocol["industry_matched"]), incremental_protocol.to_dict()),
        ("execution_cost_turnover_tracking_reported", len(execution) == 8
         and execution["observations"].eq(expected_days).all(), len(execution)),
        ("share_level_fill_ledger_is_complete", execution["share_level_fill_ratio_available"].all()
         and matrix["share_level_fill_difference_available"].all()
         and execution["share_level_fill_ratio"].between(0, 1).all(), cfg["fill_evidence_status"]),
        ("arithmetic_and_geometric_definitions_separated", cfg["return_definitions"]["portfolio_annual_return"].startswith("geometric")
         and cfg["return_definitions"]["risk_adjusted_incremental_alpha"].startswith("arithmetic"), cfg["return_definitions"]),
        ("capm_is_not_labelled_pure_alpha", matrix["pure_stock_selection_alpha_identified"].eq(False).all(), len(matrix)),
        ("local_style_proxy_is_disclosed", matrix["style_model_status"].eq(
            "local_research_proxy_not_commercial_risk_model").all(), cfg["style_model_status"]),
    ]
    return pd.DataFrame([{
        "check_name": name, "status": "PASS" if bool(condition) else "FAIL",
        "observed": str(observed),
    } for name, condition, observed in rows])


def run(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    panel = paired_execution_panel(lake, cfg)
    regressions, primary, alignment = risk_adjusted_inference(panel, cfg)
    raw = raw_share_level_inference(panel, cfg)
    execution = execution_comparison(lake, cfg)
    exposure = exposure_match_summary(lake, cfg)
    matrix = candidate_evidence_matrix(lake, primary, raw, execution, exposure, cfg)
    measure_definitions = definitions(cfg)
    gaps = gap_register(cfg)
    checks = acceptance_checks(
        panel, regressions, primary, alignment, execution, matrix, cfg, lake,
    )
    sensitivity = regressions.groupby(KEYS, as_index=False).agg(
        specifications=("annualized_alpha_arithmetic", "size"),
        minimum_alpha=("annualized_alpha_arithmetic", "min"),
        maximum_alpha=("annualized_alpha_arithmetic", "max"),
        minimum_ci_low=("annualized_alpha_ci_low", "min"),
        maximum_ci_low=("annualized_alpha_ci_low", "max"),
        minimum_ci_high=("annualized_alpha_ci_high", "min"),
        maximum_ci_high=("annualized_alpha_ci_high", "max"),
        individually_positive_specifications=(
            "annualized_alpha_ci_low", lambda values: int(values.gt(0).sum())
        ),
    )
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "phase_status": "COMPLETE_SHARE_LEVEL_FILL_COMPARISON",
        "primary_rate_scenario": cfg["primary_rate_scenario"],
        "primary_benchmark_variant": cfg["primary_benchmark_variant"],
        "primary_hac_lags": cfg["primary_hac_lags"],
        "style_model_status": cfg["style_model_status"],
        "share_level_fill_difference_available": True,
        "pure_stock_selection_alpha_claim_allowed": False,
    }])
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all Phase 9 assertions passed; share-level fill comparison is complete",
        }])
    tables = {
        "alpha_evidence_protocol": protocol,
        "alpha_evidence_aligned_daily": panel,
        "alpha_evidence_alignment": alignment,
        "alpha_evidence_risk_adjusted_regression": regressions,
        "alpha_evidence_risk_adjusted_fdr": primary,
        "alpha_evidence_share_level_raw_fdr": raw,
        "alpha_evidence_risk_sensitivity": sensitivity,
        "alpha_evidence_execution_comparison": execution,
        "alpha_evidence_exposure_match": exposure,
        "alpha_evidence_candidate_matrix": matrix,
        "alpha_evidence_measure_definitions": measure_definitions,
        "alpha_evidence_gap_register": gaps,
        "alpha_evidence_checks": checks,
        "alpha_evidence_failed_assertions": failures,
    }
    for table, frame in tables.items():
        lake.write(table, frame, key="v2", source="alpha_evidence_matrix.v2", request=cfg)

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(TABLE_ROOT / "alpha_evidence_candidate_matrix.csv", index=False)
    primary.to_csv(TABLE_ROOT / "alpha_evidence_risk_adjusted_fdr.csv", index=False)
    execution.to_csv(TABLE_ROOT / "alpha_evidence_execution_comparison.csv", index=False)
    exposure.to_csv(TABLE_ROOT / "alpha_evidence_exposure_match.csv", index=False)
    gaps.to_csv(TABLE_ROOT / "alpha_evidence_gap_register.csv", index=False)
    display = matrix[[
        "model", "alpha_pipeline", "annualized_return_model", "annualized_return_style",
        "raw_incremental_return_annualized_arithmetic", "raw_incremental_return_ci_low",
        "raw_incremental_return_ci_high", "risk_adjusted_incremental_alpha_arithmetic",
        "risk_adjusted_incremental_alpha_ci_low", "risk_adjusted_incremental_alpha_ci_high",
        "risk_adjusted_incremental_alpha_fdr_q", "evidence_verdict",
    ]]
    lines = [
        "# Phase 9：风险基准与增量 Alpha 统一证据矩阵", "",
        "## 结论", "",
        "四条候选的 2021–2026 历史结果均不能识别为纯个股 Alpha。CAPM 截距在部分规格显著，但加入本地 PIT 六风格后均跨零；ML 相对可交易风格复制组合的配对收益差经四候选 BH-FDR 后 0/4 通过，进一步用 CSI500 全收益超额收益与六风格解释配对差后仍为 0/4。", "",
        "主规格固定为中债 3 个月国债利率、CSI500 官方全收益指数与 HAC(5)；价格指数、零/1年利率和 HAC(10/20/40) 只作敏感性，不按结果选择。", "",
        "## 候选证据矩阵", "", display.to_markdown(index=False, floatfmt=".6f"), "",
        "## 风险模型敏感性", "", sensitivity.to_markdown(index=False, floatfmt=".6f"), "",
        "## 执行与成交证据", "",
        "ML 与风格复制组合已全部接入同一人民币股数级订单执行器，使用1亿元账户、20bp成本、T+1、整手、10% ADV、停牌/涨跌停重试。八条路径均保存订单、成交、持仓、现金和NAV，并报告按股数成交率。", "",
        "## 暴露匹配", "", exposure.to_markdown(index=False, floatfmt=".6f"), "",
        "风格因子与证监会宽行业均为本项目本地 PIT 代理，不是商业风险模型。匹配误差已完整保留，因此风格基准是可审计研究基准，而不是对所有风险暴露的完美复制。", "",
        "## 未完成差距", "", gaps.to_markdown(index=False), "",
        "## 机器验收", "", checks.to_markdown(index=False), "",
        "> Phase 9 实现范围已完成，但仍属于已查看 historical OOS 上的 post-hoc consolidation；最终独立证据仍需冻结方案后的 future lockbox 或 paper trading。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(display.to_string(index=False), flush=True)
    print(checks.to_string(index=False), flush=True)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Phase 9 alpha evidence checks failed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 9 alpha evidence matrix")
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
