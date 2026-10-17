from pathlib import Path

import pandas as pd

from quant_research.audit import AuditCheck, ResearchAuditor, write_reports


class FrameLake:
    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self.frames = frames

    def read(self, table: str, key: str = "all") -> pd.DataFrame:
        return self.frames.get((table, key), pd.DataFrame()).copy()


def test_model_benchmark_audit_handles_hgb_winner_without_hgb_reference_row() -> None:
    frames = {
        ("model_benchmark_selection", "v1"): pd.DataFrame([{
            "selected_candidate": "hgb_shallow15",
            "selected_family": "hgb",
            "historical_oos_used_for_selection": False,
        }]),
        ("model_benchmark_oos_summary", "v1"): pd.DataFrame([
            {
                "candidate": "ridge_1", "family": "ridge",
                "mean_rank_ic": 0.09, "portfolio_annualized_return": 0.20,
            },
            {
                "candidate": "hgb_shallow15", "family": "hgb",
                "mean_rank_ic": 0.10, "portfolio_annualized_return": 0.24,
            },
        ]),
        ("model_benchmark_oos_annual", "v1"): pd.DataFrame([{
            "candidate": "hgb_shallow15", "mean_rank_ic": 0.10,
        }]),
        ("model_benchmark_winner_pairwise", "v1"): pd.DataFrame([
            {
                "candidate": "hgb_shallow15", "reference": "ridge_1",
                "metric": "rank_ic", "ci_2_5pct": -0.002,
                "ci_97_5pct": 0.007,
            },
            {
                "candidate": "hgb_shallow15", "reference": "ridge_1",
                "metric": "net_return_annualized_arithmetic",
                "ci_2_5pct": -0.009, "ci_97_5pct": 0.073,
            },
        ]),
        ("model_benchmark_availability", "v1"): pd.DataFrame([{
            "model": "hgb", "status": "executed",
        }]),
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_model_benchmark()

    checks = {check.name: check for check in auditor.checks}
    assert checks["model_winner_return_pairwise_vs_ridge"].status == "PASS"
    assert checks["model_winner_return_interval_valid"].status == "PASS"
    assert "inference=inconclusive" in checks[
        "model_winner_return_interval_valid"
    ].detail
    assert checks["random_forest_not_claimed_superior_to_hgb"].status == "PASS"
    assert auditor.findings["model_benchmark_validation_winner"] == "hgb_shallow15"


def test_audit_markdown_has_no_trailing_space_for_empty_finding(tmp_path: Path) -> None:
    markdown = tmp_path / "audit.md"
    write_reports(
        [AuditCheck(name="check", status="PASS", detail="detail")],
        {"empty_result": ""},
        md_path=markdown,
        json_path=tmp_path / "audit.json",
    )

    assert "- `empty_result`:\n" in markdown.read_text(encoding="utf-8")


def test_model_neutralization_audit_checks_locked_paired_protocol() -> None:
    metrics = [
        "rank_ic", "net_return_annualized_arithmetic",
        "industry_active_weight_l1", "abs_score_log_size_correlation",
        "score_industry_max_abs_mean", "portfolio_industry_hhi",
    ]
    frames = {
        ("model_neutralization_protocol", "v1"): pd.DataFrame([{
            "historical_oos_used_for_selection": False,
            "models_locked_before_historical_oos": True,
            "common_exposure_complete_sample": True,
        }]),
        ("model_neutralization_oos_summary", "v1"): pd.DataFrame([{
            "pipeline": "industry_size_neutralized", "model": "ridge_locked",
            "mean_score_log_size_correlation": 1e-16,
            "mean_score_industry_max_abs_mean": 1e-4,
            "mean_rank_ic": 0.09, "portfolio_annualized_return": 0.16,
        }]),
        ("model_neutralization_oos_daily", "v1"): pd.DataFrame([
            {"pipeline": pipeline, "model": model, "asof_date": date}
            for pipeline in ("raw", "industry_size_neutralized")
            for model in ("ridge_locked", "hgb_shallow15_locked")
            for date in ("2021-01-08", "2021-01-15")
        ]),
        ("model_neutralization_oos_bootstrap", "v1"): pd.DataFrame([
            {
                "pipeline": "industry_size_neutralized", "model": "ridge_locked",
                "metric": metric, "ci_2_5pct": -0.01, "ci_97_5pct": 0.01,
            }
            for metric in metrics
        ]),
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_model_portfolio_neutralization()

    checks = {check.name: check.status for check in auditor.checks}
    assert checks["model_neutralization_protocol_locked"] == "PASS"
    assert checks["model_neutralization_paired_dates"] == "PASS"
    assert checks["linear_model_neutral_score_residual_small"] == "PASS"
    assert checks["model_neutralization_inference_complete"] == "PASS"


def test_constrained_portfolio_audit_checks_protocol_and_bounds() -> None:
    strategies = [
        "raw__top200_equal", "raw__risk_constrained",
        "industry_size_neutralized__top200_equal",
        "industry_size_neutralized__risk_constrained",
    ]
    weekly_rows = []
    for strategy in strategies:
        construction = "risk_constrained" if strategy.endswith("risk_constrained") else "top200_equal"
        for date in ("2021-01-08", "2021-01-15"):
            weekly_rows.append({
                "strategy": strategy, "asof_date": date, "construction": construction,
                "optimizer_success": True, "stock_weight_violation": 0.0,
                "size_bound_violation": 1e-16, "industry_bound_violation": 1e-16,
                "turnover": 0.3, "effective_turnover_limit": (
                    0.3 if construction == "risk_constrained" else float("nan")
                ),
                "turnover_limit_relaxed": False,
            })
    frames = {
        ("constrained_portfolio_protocol", "v1"): pd.DataFrame([{
            "historical_oos_used_for_selection": False,
            "constraint_policy_prespecified": True,
            "source_experiment": "model_portfolio_neutralization_v2",
        }]),
        ("constrained_portfolio_oos_weekly", "v1"): pd.DataFrame(weekly_rows),
        ("constrained_portfolio_oos_summary", "v1"): pd.DataFrame([{
            "model": "hgb_shallow15_locked", "strategy": "raw__risk_constrained",
            "annualized_return": 0.24, "sharpe_zero_rate": 0.92,
            "maximum_drawdown": -0.33, "mean_industry_active_weight_l1": 0.095,
        }]),
        ("constrained_portfolio_oos_bootstrap", "v1"): pd.DataFrame([
            {"comparison": "full_neutral_stack", "metric": metric}
            for metric in (
                "rank_ic", "net_return_annualized_arithmetic",
                "industry_active_weight_l1", "abs_portfolio_size_z", "turnover",
            )
        ]),
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_constrained_portfolio_neutralization()

    checks = {check.name: check.status for check in auditor.checks}
    assert checks["constrained_portfolio_protocol_locked"] == "PASS"
    assert checks["constrained_portfolio_paired_dates"] == "PASS"
    assert checks["constrained_portfolio_risk_bounds"] == "PASS"
    assert checks["constrained_portfolio_turnover_bound"] == "PASS"
    assert checks["constrained_portfolio_inference_complete"] == "PASS"


def test_constrained_daily_execution_audit_checks_clock_and_accounting() -> None:
    strategies = [
        ("ridge_locked", "raw", "raw__risk_constrained"),
        ("hgb_shallow15_locked", "raw", "raw__risk_constrained"),
    ]
    daily_rows, target_rows, summary_rows, bootstrap_rows = [], [], [], []
    for model, pipeline, strategy in strategies:
        for target_number in range(293):
            target_rows.append({
                "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
                "execution_date": f"d{target_number:03d}",
            })
        for scenario, annualized_return in (
            ("daily_retry_limit_and_suspension_constrained", 0.18),
            ("nonoverlap_all_fills_upper_bound", 0.20),
        ):
            summary_rows.append({
                "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
                "scenario": scenario, "annualized_return": annualized_return,
                "sharpe_zero_rate": 0.8, "maximum_drawdown": -0.3,
                "mean_blocked_buy_weight": 0.01,
            })
            for date, new_target in (("2024-01-02", True), ("2024-01-03", False)):
                daily_rows.append({
                    "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
                    "scenario": scenario, "trade_date": date,
                    "accounting_weight_sum": 1.0, "net_return": 0.01,
                    "gross_return": 0.01, "transaction_cost": 0.0,
                    "new_target": new_target, "turnover": 0.2 if new_target else 0.0,
                })
        bootstrap_rows.append({
            "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
            "ci_97_5pct": -0.001,
        })
    frames = {
        ("constrained_daily_execution_protocol", "v1"): pd.DataFrame([{
            "historical_oos_used_for_selection": False,
            "unique_nonoverlap_daily_clock": True,
            "persistent_blocked_share_orders": True,
            "source_experiment": "constrained_portfolio_neutralization_v1",
        }]),
        ("constrained_daily_execution_daily", "v1"): pd.DataFrame(daily_rows),
        ("constrained_daily_execution_summary", "v1"): pd.DataFrame(summary_rows),
        ("constrained_daily_execution_bootstrap", "v1"): pd.DataFrame(bootstrap_rows),
        ("constrained_execution_targets", "v1"): pd.DataFrame(target_rows),
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_constrained_portfolio_daily_execution()

    checks = {check.name: check.status for check in auditor.checks}
    assert checks["constrained_daily_execution_protocol_locked"] == "PASS"
    assert checks["constrained_daily_execution_paired_targets"] == "PASS"
    assert checks["constrained_daily_execution_accounting_identity"] == "PASS"
    assert checks["constrained_daily_execution_no_hidden_rebalance"] == "PASS"
    assert checks["constrained_daily_execution_haircut_detected"] == "PASS"


def test_multi_factor_risk_attribution_audit_checks_pit_identity_and_lag() -> None:
    attribution_rows = []
    for model in ("ridge_locked", "hgb_shallow15_locked"):
        for date_number in range(293):
            attribution_rows.append({
                "model": model, "alpha_pipeline": "raw",
                "strategy": "raw__risk_constrained",
                "asof_date": f"d{date_number:03d}",
                "attribution_error": 1e-16, "reported_net_error": 1e-16,
            })
    metrics = [
        "industry_active_contribution", "style_contribution",
        "specific_selection_contribution", "active_gross_return", "active_net_return",
    ]
    frames = {
        ("risk_model_protocol", "v1"): pd.DataFrame([{
            "historical_oos_used_for_selection": False,
            "factor_covariance_uses_current_or_future_return": False,
            "attribution_identity_required": True,
            "source_experiment": "constrained_portfolio_neutralization_v1",
        }]),
        ("pit_beta_exposure_index", "v1"): pd.DataFrame([{
            "beta_coverage": 0.91, "duplicate_instrument_date_keys": 0,
        }]),
        ("risk_model_factor_returns", "v1"): pd.DataFrame([{
            "asof_date": "d000", "factor": "market", "factor_return": 0.01,
        }]),
        ("risk_model_portfolio_attribution", "v1"): pd.DataFrame(attribution_rows),
        ("risk_model_portfolio_forecasts", "v1"): pd.DataFrame([{
            "factor_history_weeks": 600, "forecast_annualized_volatility": 0.25,
            "forecast_annualized_tracking_error": 0.08,
        }]),
        ("risk_model_attribution_summary", "v1"): pd.DataFrame([{
            "model": "hgb_shallow15_locked", "alpha_pipeline": "raw",
            "annualized_style_contribution_arithmetic": 0.14,
            "annualized_specific_selection_contribution_arithmetic": -0.01,
            "mean_forecast_annualized_tracking_error": 0.08,
        }]),
        ("risk_model_attribution_bootstrap", "v1"): pd.DataFrame([
            {
                "model": model, "alpha_pipeline": "raw", "metric": metric,
                "ci_2_5pct": -0.02, "ci_97_5pct": 0.02,
            }
            for model in ("ridge_locked", "hgb_shallow15_locked")
            for metric in metrics
        ]),
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_multi_factor_risk_attribution()

    checks = {check.name: check.status for check in auditor.checks}
    assert checks["risk_attribution_protocol_locked"] == "PASS"
    assert checks["pit_beta_coverage_and_keys"] == "PASS"
    assert checks["risk_attribution_exact_and_paired"] == "PASS"
    assert checks["risk_forecast_lagged_finite_positive"] == "PASS"
    assert checks["risk_attribution_inference_complete"] == "PASS"


def test_risk_aware_portfolio_audit_rejects_return_destructive_policy() -> None:
    weekly = pd.DataFrame([
        {
            "model": model, "alpha_pipeline": "raw", "scenario": scenario,
            "asof_date": f"d{date:03d}", "turnover_limit_relaxed": False,
        }
        for model in ("ridge_locked", "hgb_shallow15_locked")
        for scenario in (
            "source_risk_constrained", "risk_aware_full", "risk_aware_target_vol",
        )
        for date in range(293)
    ])
    summary = pd.DataFrame([
        {
            "model": "hgb_shallow15_locked", "alpha_pipeline": "raw",
            "scenario": scenario, "mean_forecast_annualized_volatility": volatility,
            "mean_forecast_annualized_tracking_error": tracking_error,
            "mean_risky_weight": risky_weight, "annualized_return": annualized_return,
        }
        for scenario, volatility, tracking_error, risky_weight, annualized_return in (
            ("source_risk_constrained", 0.26, 0.08, 1.0, 0.24),
            ("risk_aware_full", 0.27, 0.05, 1.0, 0.19),
            ("risk_aware_target_vol", 0.20, 0.09, 0.79, 0.10),
        )
    ])
    bootstrap = pd.DataFrame([
        {
            "model": "hgb_shallow15_locked", "alpha_pipeline": "raw",
            "scenario": "risk_aware_full", "metric": metric,
            "point_estimate": point, "ci_2_5pct": low, "ci_97_5pct": high,
        }
        for metric, point, low, high in (
            ("forecast_tracking_error", -0.03, -0.04, -0.02),
            ("net_return_annualized_arithmetic", -0.05, -0.08, -0.01),
        )
    ])
    frames = {
        ("risk_aware_portfolio_protocol", "v2"): pd.DataFrame([{
            "historical_oos_used_for_selection": False,
            "factor_covariance_uses_current_or_future_return": False,
            "specific_variance_uses_current_or_future_residual": False,
            "candidate_set_uses_future_return": False,
            "source_experiment": "multi_factor_risk_attribution_v1",
        }]),
        ("risk_aware_portfolio_weekly", "v2"): weekly,
        ("risk_aware_portfolio_constraint_audit", "v2"): pd.DataFrame([{
            "stock_weight_violation": 0.0, "size_bound_violation": 0.0,
            "industry_bound_violation": 0.0, "turnover_bound_violation": 0.0,
            "alpha_utility_shortfall": 0.0,
        }]),
        ("risk_aware_portfolio_forecasts", "v2"): pd.DataFrame([{
            "forecast_annualized_volatility": 0.20,
        }]),
        ("risk_aware_portfolio_summary", "v2"): summary,
        ("risk_aware_portfolio_bootstrap", "v2"): bootstrap,
    }
    auditor = ResearchAuditor(lake=FrameLake(frames))

    auditor.audit_risk_aware_portfolio_optimization()

    checks = {check.name: check.status for check in auditor.checks}
    assert checks["risk_aware_protocol_lagged_and_locked"] == "PASS"
    assert checks["risk_aware_paired_dates"] == "PASS"
    assert checks["risk_aware_constraints_hold"] == "PASS"
    assert checks["target_volatility_forecast_calibrated"] == "PASS"
    assert checks["risk_aware_te_reduction_confirmed"] == "PASS"
    assert checks["risk_aware_return_cost_detected"] == "PASS"
    assert checks["risk_aware_policy_rejected"] == "WARN"
    assert auditor.findings["risk_aware_v1_policy_adopted"] is False
