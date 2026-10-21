"""Audit derived research artifacts before reporting or publication."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.experiment_registry import ExperimentRegistry, load_specs, sha256_file
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.portfolio import performance_summary


REPORT_MD = ROOT / "reports" / "research_audit.md"
REPORT_JSON = ROOT / "reports" / "research_audit.json"


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    detail: str


def _load_yaml(name: str) -> dict:
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True))


class ResearchAuditor:
    def __init__(self, lake: Lake | None = None) -> None:
        self.lake = lake or Lake()
        self.checks: list[AuditCheck] = []
        self.findings: dict[str, float | int | str | bool] = {}

    def add(self, name: str, condition: bool, detail: str,
            *, failure_status: str = "FAIL") -> None:
        self.checks.append(AuditCheck(name, "PASS" if condition else failure_status, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(AuditCheck(name, "WARN", detail))

    def require_table(self, table: str, key: str) -> pd.DataFrame:
        frame = self.lake.read(table, key)
        self.add(
            f"artifact:{table}/{key}", not frame.empty,
            f"rows={len(frame):,}" if not frame.empty else "missing or empty",
        )
        return frame

    def audit_configs(self) -> None:
        cfg = _load_yaml("model_baseline.yaml")
        train = cfg["train"]
        validation = cfg["validation"]
        test = cfg["test"]
        ordered = (
            int(train["start_year"]) <= int(train["end_year"])
            < int(validation["start_year"]) <= int(validation["end_year"])
            < int(test["start_year"]) <= int(test["end_year"])
        )
        self.add(
            "chronological_split_order", ordered,
            f"train={train}, validation={validation}, historical_oos={test}",
        )
        self.warn(
            "historical_oos_is_not_pristine",
            "The 2021-2026 block has already been inspected during project development. "
            "It is historical out-of-sample evidence, not a never-seen production lockbox.",
        )

    def audit_frozen_experiments(self) -> None:
        """Treat experiment immutability and input lineage as publication gates."""
        specs = load_specs()
        registry = ExperimentRegistry(lake=self.lake)
        invalidation_path = ROOT / "config" / "experiment_invalidations.yaml"
        invalidations: dict[str, dict] = {}
        if invalidation_path.exists():
            with invalidation_path.open(encoding="utf-8") as handle:
                invalidations = yaml.safe_load(handle).get("invalidations", {})
        experiment_ids = list(specs["experiments"])
        self.findings["registered_frozen_experiments"] = len(experiment_ids)
        for experiment_id in experiment_ids:
            try:
                result = registry.verify(experiment_id, verify_live_inputs=True)
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.add(
                    f"frozen_experiment:{experiment_id}", False,
                    f"verification_error={type(exc).__name__}: {exc}",
                )
                continue
            self.add(
                f"frozen_snapshot_integrity:{experiment_id}",
                bool(result["snapshot_integrity"]),
                "all frozen files match FROZEN.json",
            )
            mismatches = [
                f"{item['table']}/{item['key']}"
                for item in result["input_checks"] if not item["matches"]
            ]
            self.add(
                f"frozen_live_inputs:{experiment_id}",
                bool(result["live_inputs_match"]),
                "all registered inputs match their SHA-256 hashes"
                if not mismatches else f"mismatches={mismatches}",
                # An explicitly invalidated/superseded experiment remains an
                # immutable historical record. Its snapshot must still pass,
                # while expected drift from the repaired live lake is a warning.
                failure_status=("WARN" if experiment_id in invalidations else "FAIL"),
            )
        if invalidations:
            for experiment_id, record in invalidations.items():
                self.warn(
                    f"invalidated_experiment:{experiment_id}",
                    f"severity={record['severity']}; superseded_by={record['superseded_by']}; "
                    f"reason={record['reason']}",
                )
            self.findings["invalidated_experiments"] = len(invalidations)

    def audit_coverage(self) -> None:
        coverage = self.require_table("research_coverage", "a_share")
        if coverage.empty:
            return
        values = coverage.set_index("metric")["value"]
        rate = float(values.get("download_coverage", np.nan))
        missing = int(values.get("missing_a_share_files", -1))
        self.add(
            "historical_universe_download_coverage", _close(rate, 1.0) and missing == 0,
            f"coverage={rate:.2%}, missing_files={missing}",
        )

    def audit_predictions(self, table: str, key: str, label: str,
                          start_year: int, end_year: int) -> pd.DataFrame:
        frame = self.require_table(table, key)
        if frame.empty:
            return frame
        keys = ["instrument_id", "asof_date"]
        duplicate_count = int(frame.duplicated(keys).sum())
        years = frame["asof_date"].str.slice(0, 4).astype(int)
        availability_ok = (frame["available_date"] > frame["asof_date"]).all()
        finite_scores = np.isfinite(pd.to_numeric(frame["score"], errors="coerce")).all()
        self.add(f"{label}_prediction_keys_unique", duplicate_count == 0,
                 f"duplicate_rows={duplicate_count}")
        self.add(f"{label}_prediction_years", years.between(start_year, end_year).all(),
                 f"observed={years.min()}-{years.max()}, expected={start_year}-{end_year}")
        self.add(f"{label}_availability_after_signal", bool(availability_ok),
                 "available_date must be strictly after asof_date")
        self.add(f"{label}_scores_finite", bool(finite_scores),
                 f"rows={len(frame):,}, dates={frame['asof_date'].nunique():,}")
        return frame

    def audit_selection_provenance(self) -> None:
        items = [
            ("model_selection", "ridge_v1", "selection_data_end", "2020-12-31"),
            ("portfolio_selection", "long_only_v1", "selection_period", "validation_2016_2020_only"),
            ("turnover_buffer_selection", "v1", "selection_period", "validation_2016_2020_only"),
            ("model_challenger_selection", "hgb_v1", "selection_period", "validation_2016_2020_only"),
            ("label_horizon_selection", "v1", "selection_period", "validation_2016_2020_only"),
            ("model_benchmark_selection", "v1", "selection_period", "validation_2016_2020_only"),
        ]
        for table, key, column, expected in items:
            frame = self.require_table(table, key)
            if frame.empty:
                continue
            actual = str(frame.iloc[0][column])
            self.add(f"selection_provenance:{table}", actual == expected,
                     f"{column}={actual}; expected={expected}")

    def audit_weekly_identity(self, table: str, key: str, label: str,
                              selector: Callable[[pd.DataFrame], pd.Series] | None = None) -> pd.DataFrame:
        frame = self.require_table(table, key)
        if frame.empty:
            return frame
        if selector is not None:
            frame = frame.loc[selector(frame)].copy()
        gross = pd.to_numeric(frame["gross_return"], errors="coerce")
        cost = pd.to_numeric(frame["transaction_cost"], errors="coerce")
        net = pd.to_numeric(frame["net_return"], errors="coerce")
        identity_error = float((net - (gross - cost)).abs().max())
        valid_returns = np.isfinite(net).all() and net.gt(-1).all()
        self.add(f"{label}_net_return_identity", identity_error < 1e-10,
                 f"max_abs_error={identity_error:.3e}")
        self.add(f"{label}_returns_finite_and_above_minus_one", bool(valid_returns),
                 f"dates={frame['asof_date'].nunique():,}")
        return frame

    def audit_holdings(self, table: str, key: str, label: str) -> None:
        holdings = self.require_table(table, key)
        if holdings.empty:
            return
        totals = holdings.groupby("asof_date")["target_weight"].sum()
        maximum_error = float((totals - 1.0).abs().max())
        duplicate_count = int(holdings.duplicated(["asof_date", "instrument_id"]).sum())
        self.add(f"{label}_weights_sum_to_one", maximum_error < 1e-9,
                 f"max_abs_error={maximum_error:.3e}")
        self.add(f"{label}_holding_keys_unique", duplicate_count == 0,
                 f"duplicate_rows={duplicate_count}")

    def audit_paired_results(self, ridge: pd.DataFrame, challenger: pd.DataFrame) -> None:
        if ridge.empty or challenger.empty:
            return
        common = sorted(set(ridge["asof_date"]) & set(challenger["asof_date"]))
        self.add("paired_model_dates_available", len(common) >= 250,
                 f"common_dates={len(common)}, first={common[0]}, last={common[-1]}")

        costs = self.require_table("robustness_cost_grid", "v1")
        bootstrap = self.require_table("robustness_block_bootstrap", "v1")
        annual_ic = self.require_table("model_oos_annual", "ridge_v1")
        if costs.empty or bootstrap.empty or annual_ic.empty:
            return

        selected = costs.loc[costs["cost_bps"].eq(20)].set_index("model")
        ridge_return = float(selected.loc["ridge", "annualized_return"])
        challenger_return = float(selected.loc["hgb_ridge_blend", "annualized_return"])
        ridge_excess = bootstrap.loc[
            bootstrap["metric"].eq("ridge_net_excess_vs_CSI500")].iloc[0]
        model_increment = bootstrap.loc[
            bootstrap["metric"].eq("challenger_minus_ridge_net_return")].iloc[0]
        all_year_ic_positive = bool(annual_ic["mean_rank_ic"].gt(0).all())

        self.findings.update({
            "common_oos_weeks": len(common),
            "ridge_annualized_return_20bp": ridge_return,
            "challenger_annualized_return_20bp": challenger_return,
            "ridge_excess_vs_csi500_ci_low": float(ridge_excess["ci_2_5pct"]),
            "ridge_excess_vs_csi500_ci_high": float(ridge_excess["ci_97_5pct"]),
            "challenger_minus_ridge_ci_low": float(model_increment["ci_2_5pct"]),
            "challenger_minus_ridge_ci_high": float(model_increment["ci_97_5pct"]),
            "ridge_rank_ic_positive_every_oos_year": all_year_ic_positive,
        })
        self.add("ridge_rank_ic_positive_each_oos_year", all_year_ic_positive,
                 f"minimum_annual_rank_ic={annual_ic['mean_rank_ic'].min():.4f}")
        self.add("ridge_excess_bootstrap_interval_above_zero",
                 float(ridge_excess["ci_2_5pct"]) > 0,
                 f"95% CI=[{ridge_excess['ci_2_5pct']:.2%}, {ridge_excess['ci_97_5pct']:.2%}]",
                 failure_status="WARN")
        increment_low = float(model_increment["ci_2_5pct"])
        increment_high = float(model_increment["ci_97_5pct"])
        increment_inference = (
            "positive" if increment_low > 0
            else "negative" if increment_high < 0
            else "inconclusive"
        )
        self.add(
            "challenger_increment_interval_valid",
            np.isfinite([increment_low, increment_high]).all()
            and increment_low <= increment_high,
            f"95% CI=[{increment_low:.2%}, {increment_high:.2%}]; "
            f"inference={increment_inference}; historical OOS is not pristine",
        )

    def audit_lofo_consistency(self) -> None:
        lofo = self.require_table("lofo_summary", "v1")
        preprocessing = self.require_table("factor_preprocessing_oos_summary", "v1")
        if lofo.empty or preprocessing.empty:
            return
        lofo_full = lofo.loc[
            lofo["model"].eq("full") & lofo["split"].eq("historical_oos")
        ].iloc[0]
        preprocessing_full = preprocessing.loc[
            preprocessing["pipeline"].eq("selected_quantile_zscore")
        ].iloc[0]
        for metric in ("mean_rank_ic", "annualized_icir", "portfolio_annualized_return"):
            left, right = float(lofo_full[metric]), float(preprocessing_full[metric])
            self.add(
                f"lofo_full_reproduces_preprocessing:{metric}", _close(left, right, 1e-9),
                f"lofo={left:.10f}, preprocessing_oos={right:.10f}",
            )

    def audit_label_horizon(self) -> None:
        panel = self.require_table("label_horizon_panel_audit", "v1")
        selection = self.require_table("label_horizon_selection", "v1")
        summary = self.require_table("label_horizon_oos_summary", "v1")
        bootstrap = self.require_table("label_horizon_oos_bootstrap", "v1")
        if panel.empty or selection.empty or summary.empty or bootstrap.empty:
            return
        row_match = bool(panel["row_count_matches_eligible"].astype(bool).all())
        duplicates = int(panel["duplicate_instrument_date_keys"].sum())
        ordered_coverage = bool((
            panel["fwd_return_5d_lag1_coverage"]
            >= panel["fwd_return_10d_lag1_coverage"]
        ).all() and (
            panel["fwd_return_10d_lag1_coverage"]
            >= panel["fwd_return_20d_lag1_coverage"]
        ).all())
        self.add("label_panel_preserves_eligible_rows", row_match,
                 f"years={len(panel)}, all_years_match={row_match}")
        self.add("label_panel_keys_unique", duplicates == 0,
                 f"duplicate_rows={duplicates}")
        self.add("label_coverage_monotone_by_horizon", ordered_coverage,
                 "coverage(5d) >= coverage(10d) >= coverage(20d) for every year")
        no_oos_selection = not bool(selection.iloc[0]["historical_oos_used_for_selection"])
        self.add("label_selection_excludes_historical_oos", no_oos_selection,
                 f"selected={selection.iloc[0]['selected_target_spec']}")

        preprocessing = self.require_table("factor_preprocessing_oos_summary", "v1")
        if not preprocessing.empty:
            reproduced = summary.loc[
                summary["target_spec"].eq("h5_cross_sectional_rank")
            ].iloc[0]
            baseline = preprocessing.loc[
                preprocessing["pipeline"].eq("selected_quantile_zscore")
            ].iloc[0]
            for metric in ("mean_rank_ic", "portfolio_annualized_return"):
                left, right = float(reproduced[metric]), float(baseline[metric])
                self.add(
                    f"label_h5_rank_reproduces_preprocessing:{metric}",
                    _close(left, right, 1e-9),
                    f"label_study={left:.10f}, preprocessing_oos={right:.10f}",
                )
        selected_name = str(selection.iloc[0]["selected_target_spec"])
        selected_test = bootstrap.loc[bootstrap["challenger"].eq(selected_name)].iloc[0]
        interval_includes_zero = (
            float(selected_test["ci_2_5pct"]) <= 0 <= float(selected_test["ci_97_5pct"])
        )
        self.add(
            "label_selected_increment_not_overstated", interval_includes_zero,
            f"selected={selected_name}; 95% CI=[{selected_test['ci_2_5pct']:.2%}, "
            f"{selected_test['ci_97_5pct']:.2%}]",
        )
        self.findings.update({
            "label_validation_selected_target": selected_name,
            "label_selected_oos_increment_ci_low": float(selected_test["ci_2_5pct"]),
            "label_selected_oos_increment_ci_high": float(selected_test["ci_97_5pct"]),
        })

    def audit_size_neutralization(self) -> None:
        exposure_index = self.require_table("pit_size_exposure_index", "v1")
        summary = self.require_table("size_neutralization_oos_summary", "v1")
        bootstrap = self.require_table("size_neutralization_oos_bootstrap", "v1")
        if exposure_index.empty or summary.empty or bootstrap.empty:
            return
        minimum_coverage = float(exposure_index["eligible_float_cap_coverage"].min())
        duplicate_count = int(exposure_index["duplicate_instrument_date_keys"].sum())
        self.add("pit_size_coverage", minimum_coverage > 0.9999,
                 f"minimum_eligible_weekly_coverage={minimum_coverage:.6%}")
        self.add("pit_size_keys_unique", duplicate_count == 0,
                 f"duplicate_rows={duplicate_count}")
        neutral = summary.loc[summary["pipeline"].eq("size_neutralized")].iloc[0]
        residual_correlation = abs(float(neutral["mean_score_log_size_correlation"]))
        self.add("size_neutral_score_orthogonality", residual_correlation < 1e-12,
                 f"abs_mean_score_log_size_correlation={residual_correlation:.3e}")
        preprocessing = self.require_table("factor_preprocessing_oos_summary", "v1")
        if not preprocessing.empty:
            raw = summary.loc[summary["pipeline"].eq("raw")].iloc[0]
            baseline = preprocessing.loc[
                preprocessing["pipeline"].eq("selected_quantile_zscore")
            ].iloc[0]
            for metric in ("mean_rank_ic", "portfolio_annualized_return"):
                left, right = float(raw[metric]), float(baseline[metric])
                self.add(
                    f"size_raw_reproduces_preprocessing:{metric}",
                    _close(left, right, 1e-9),
                    f"size_study={left:.10f}, preprocessing_oos={right:.10f}",
                )
        return_test = bootstrap.loc[
            bootstrap["metric"].eq(
                "size_neutralized_minus_raw_net_return_annualized_arithmetic"
            )
        ].iloc[0]
        dependence_detected = float(return_test["ci_97_5pct"]) < 0
        self.add(
            "small_size_return_dependence_detected", dependence_detected,
            f"neutral-minus-raw 95% CI=[{return_test['ci_2_5pct']:.2%}, "
            f"{return_test['ci_97_5pct']:.2%}]",
        )
        self.findings.update({
            "size_neutralized_oos_annualized_return": float(
                neutral["portfolio_annualized_return"]
            ),
            "size_neutralized_oos_mean_rank_ic": float(neutral["mean_rank_ic"]),
            "size_neutralized_minus_raw_return_ci_low": float(return_test["ci_2_5pct"]),
            "size_neutralized_minus_raw_return_ci_high": float(return_test["ci_97_5pct"]),
        })

    def audit_factor_neutralization(self) -> None:
        exposure_index = self.require_table("pit_industry_exposure_index", "v1")
        summary = self.require_table("factor_neutralization_summary", "v1")
        bootstrap = self.require_table("factor_neutralization_bootstrap", "v1")
        if any(frame.empty for frame in (exposure_index, summary, bootstrap)):
            return
        minimum_coverage = float(exposure_index["eligible_industry_coverage"].min())
        duplicate_rows = int(exposure_index["duplicate_instrument_date_keys"].sum())
        future_rows = int(exposure_index[[
            "future_dated_rows", "future_snapshot_rows", "update_after_snapshot_rows"
        ]].to_numpy().sum())
        self.add(
            "pit_industry_coverage", minimum_coverage >= 0.95,
            f"minimum_eligible_weekly_coverage={minimum_coverage:.4%}",
        )
        self.add("pit_industry_keys_unique", duplicate_rows == 0,
                 f"duplicate_rows={duplicate_rows}")
        self.add("pit_industry_no_future_information", future_rows == 0,
                 f"future_or_invalid_timing_rows={future_rows}")
        oos = summary.loc[summary["split"].eq("historical_oos")]
        industry = oos.loc[oos["pipeline"].eq("industry_neutralized")]
        combined = oos.loc[oos["pipeline"].eq("industry_size_neutralized")]
        industry_residual = float(industry["mean_industry_max_abs_mean"].max())
        combined_industry_residual = float(combined["mean_industry_max_abs_mean"].max())
        combined_size_residual = float(combined["mean_size_correlation"].abs().max())
        self.add(
            "industry_neutral_factor_orthogonality", industry_residual < 1e-10,
            f"maximum_mean_abs_industry_group_mean={industry_residual:.3e}",
        )
        self.add(
            "industry_size_neutral_factor_orthogonality",
            combined_industry_residual < 1e-10 and combined_size_residual < 1e-10,
            f"industry={combined_industry_residual:.3e}, size={combined_size_residual:.3e}",
        )
        paired = bootstrap.loc[
            bootstrap["pipeline"].eq("industry_size_neutralized")
            & bootstrap["metric"].eq("oriented_rank_ic")
        ]
        strengthened = paired.loc[paired["ci_2_5pct"].gt(0), "factor"].tolist()
        weakened = paired.loc[paired["ci_97_5pct"].lt(0), "factor"].tolist()
        self.findings.update({
            "pit_industry_minimum_coverage": minimum_coverage,
            "factor_industry_size_neutral_significantly_strengthened": ",".join(strengthened),
            "factor_industry_size_neutral_significantly_weakened": ",".join(weakened),
        })

    def audit_factor_tear_sheet(self) -> None:
        protocol = self.require_table("factor_tear_sheet_protocol", "v1")
        weekly = self.require_table("factor_tear_sheet_weekly", "v1")
        buckets = self.require_table("factor_tear_sheet_buckets", "v1")
        summary = self.require_table("factor_tear_sheet_summary", "v1")
        annual = self.require_table("factor_tear_sheet_annual", "v1")
        regime = self.require_table("factor_tear_sheet_regime", "v1")
        bootstrap = self.require_table("factor_tear_sheet_bootstrap", "v1")
        if any(frame.empty for frame in (
            protocol, weekly, buckets, summary, annual, regime, bootstrap
        )):
            return
        pipelines = {"raw", "industry_size_neutralized"}
        factors = set(self.lake.read("factor_neutralization_directions", "v1")["factor"])
        paired = (
            weekly.groupby(["pipeline", "factor"])["asof_date"].nunique()
            .eq(293).all()
        )
        self.add(
            "factor_tear_sheet_paired_oos_panel",
            paired and set(weekly["pipeline"]) == pipelines
            and set(weekly["factor"]) == factors,
            f"dates={weekly['asof_date'].nunique()}, pipelines={weekly['pipeline'].nunique()}, "
            f"factors={weekly['factor'].nunique()}",
        )
        expected_buckets = set(range(1, 11))
        bucket_complete = buckets.groupby(["asof_date", "pipeline", "factor"])[
            "bucket"
        ].apply(lambda values: set(values) == expected_buckets).all()
        self.add(
            "factor_tear_sheet_deciles_complete", bool(bucket_complete),
            f"rows={len(buckets):,}, expected={293 * 2 * len(factors) * 10:,}",
        )
        benchmark_timing = pd.to_datetime(weekly["benchmark_trade_date"]).le(
            pd.to_datetime(weekly["asof_date"])
        ).all()
        self.add(
            "factor_tear_sheet_regime_is_pit", bool(benchmark_timing)
            and bool(protocol.iloc[0]["regime_uses_trailing_information_only"]),
            "benchmark_trade_date<=signal_date and protocol declares trailing-only state",
        )
        cfg = _load_yaml("factor_tear_sheet.yaml")
        cost_rate = float(cfg["assumed_cost_bps_per_leg_turnover"]) / 10_000
        cost_rows = weekly["transaction_cost"].notna()
        identity = np.allclose(
            weekly.loc[cost_rows, "net_top_minus_bottom"],
            weekly.loc[cost_rows, "gross_top_minus_bottom"]
            - weekly.loc[cost_rows, "long_short_turnover"] * cost_rate,
            rtol=1e-12, atol=1e-12,
        )
        self.add(
            "factor_tear_sheet_cost_identity", bool(identity),
            f"cost_bps_per_leg_turnover={cost_rate * 10_000:.1f}",
        )
        self.add(
            "factor_tear_sheet_no_oos_reselection",
            not bool(protocol.iloc[0]["historical_oos_used_for_selection"])
            and protocol.iloc[0]["direction_rule"] == "sign_of_raw_training_mean_rank_ic",
            f"direction_rule={protocol.iloc[0]['direction_rule']}",
        )
        neutral_spread = bootstrap.loc[
            bootstrap["pipeline"].eq("industry_size_neutralized")
            & bootstrap["metric"].eq("gross_top_minus_bottom")
        ]
        significant = neutral_spread.loc[
            neutral_spread["ci_2_5pct"].gt(0), "factor"
        ].tolist()
        down_high = regime.loc[
            regime["pipeline"].eq("industry_size_neutralized")
            & regime["market_regime"].eq("down_high")
        ]
        fragile = down_high.loc[
            down_high["mean_oriented_rank_ic"].le(0), "factor"
        ].tolist()
        self.findings.update({
            "factor_tear_sheet_oos_weeks": int(weekly["asof_date"].nunique()),
            "factor_tear_sheet_significant_neutral_spreads": ",".join(significant),
            "factor_tear_sheet_down_high_nonpositive_ic": ",".join(fragile),
        })

    def audit_capm_attribution(self) -> None:
        protocol = self.require_table("capm_protocol", "v1")
        daily = self.require_table("capm_aligned_daily", "v1")
        summary = self.require_table("capm_summary", "v1")
        annual = self.require_table("capm_annual", "v1")
        rolling = self.require_table("capm_rolling", "v1")
        if any(frame.empty for frame in (protocol, daily, summary, annual, rolling)):
            return
        self.add(
            "capm_protocol_uses_executable_nonoverlap_returns",
            bool(protocol.iloc[0]["unique_nonoverlap_daily_clock"])
            and not bool(protocol.iloc[0]["historical_oos_used_for_selection"])
            and protocol.iloc[0]["strategy_return"] == "daily_net_return",
            f"dates={protocol.iloc[0]['daily_dates']}, return={protocol.iloc[0]['strategy_return']}",
        )
        counts = summary.groupby([
            "model", "alpha_pipeline", "scenario", "benchmark"
        ])["observations"].first()
        self.add(
            "capm_complete_paired_panel",
            len(summary) == 24 and counts.eq(1387).all()
            and len(annual) == 144,
            f"summary_rows={len(summary)}, annual_rows={len(annual)}, "
            f"observations={counts.min()}-{counts.max()}",
        )
        ci_valid = (
            summary["annualized_alpha_ci_low"].le(summary["annualized_alpha_arithmetic"])
            & summary["annualized_alpha_arithmetic"].le(
                summary["annualized_alpha_ci_high"]
            )
            & summary["beta_ci_low"].le(summary["beta"])
            & summary["beta"].le(summary["beta_ci_high"])
        ).all()
        self.add(
            "capm_hac_inference_complete", bool(ci_valid)
            and summary[["alpha_hac_t_stat", "beta", "r_squared"]].notna().all().all(),
            f"newey_west_lags={summary['newey_west_lags'].unique().tolist()}",
        )
        identity_errors = []
        keys = ["model", "alpha_pipeline", "strategy", "scenario"]
        for values, group in daily.groupby(keys, sort=False):
            for benchmark in ("CSI300", "CSI500", "CSI1000"):
                row = summary.loc[
                    summary["benchmark"].eq(benchmark)
                    & np.logical_and.reduce([
                        summary[column].eq(value)
                        for column, value in zip(keys, values)
                    ])
                ].iloc[0]
                implied = (
                    group["net_return"].mean()
                    - float(row["beta"]) * group[benchmark].mean()
                )
                identity_errors.append(abs(implied - float(row["daily_alpha"])))
        maximum_identity_error = max(identity_errors)
        self.add(
            "capm_ols_intercept_identity", maximum_identity_error < 1e-12,
            f"maximum_abs_error={maximum_identity_error:.3e}",
        )
        primary = summary.loc[
            summary["benchmark"].eq("CSI500")
            & summary["scenario"].eq("daily_retry_limit_and_suspension_constrained")
        ]
        significant = primary.loc[
            primary["annualized_alpha_ci_low"].gt(0),
            ["model", "alpha_pipeline"],
        ].apply(lambda row: f"{row['model']}/{row['alpha_pipeline']}", axis=1).tolist()
        unstable_windows = int((
            rolling["scenario"].eq("daily_retry_limit_and_suspension_constrained")
            & rolling["annualized_alpha_arithmetic"].lt(0)
        ).sum())
        self.findings.update({
            "capm_csi500_significant_executable_strategies": ",".join(significant),
            "capm_csi500_executable_alpha_min": float(
                primary["annualized_alpha_arithmetic"].min()
            ),
            "capm_csi500_executable_alpha_max": float(
                primary["annualized_alpha_arithmetic"].max()
            ),
            "capm_negative_rolling_alpha_windows": unstable_windows,
        })

    def audit_style_factor_attribution(self) -> None:
        protocol = self.require_table("style_factor_protocol", "v1")
        targets = self.require_table("style_factor_targets", "v1")
        exposures = self.require_table("style_factor_exposure_audit", "v1")
        factors = self.require_table("style_factor_daily_returns", "v1")
        diagnostics = self.require_table("style_factor_diagnostics", "v1")
        multicollinearity = self.require_table("style_factor_multicollinearity", "v1")
        summary = self.require_table("style_factor_regression_summary", "v1")
        annual = self.require_table("style_factor_regression_annual", "v1")
        if any(frame.empty for frame in (
            protocol, targets, exposures, factors, diagnostics,
            multicollinearity, summary, annual,
        )):
            return
        style_names = {"size", "beta", "volatility", "liquidity", "value", "momentum"}
        self.add(
            "style_factor_protocol_is_pit_and_locked",
            not bool(protocol.iloc[0]["historical_oos_used_for_selection"])
            and bool(protocol.iloc[0]["unique_nonoverlap_daily_clock"])
            and float(protocol.iloc[0]["factor_transaction_cost_bps"]) == 0,
            f"construction={protocol.iloc[0]['factor_construction']}; "
            "explanatory factors explicitly gross of costs",
        )
        maximum_neutrality_error = float(exposures[[
            "net_weight", "maximum_abs_industry_weight",
            "maximum_abs_control_exposure",
        ]].abs().to_numpy().max())
        gross_valid = (
            np.allclose(exposures["long_weight"], 1.0, atol=1e-10)
            and np.allclose(exposures["short_weight"], 1.0, atol=1e-10)
            and exposures["target_style_exposure"].gt(0).all()
        )
        self.add(
            "style_factor_self_financing_and_orthogonal",
            maximum_neutrality_error < 1e-10 and gross_valid,
            f"maximum_neutrality_error={maximum_neutrality_error:.3e}",
        )
        factor_counts = factors.groupby("factor")["trade_date"].nunique()
        self.add(
            "style_factor_complete_daily_panel",
            set(factor_counts.index) == style_names and factor_counts.eq(1387).all()
            and factors["factor_return"].notna().all(),
            f"dates={factor_counts.min()}-{factor_counts.max()}, factors={len(factor_counts)}",
        )
        maximum_vif = float(multicollinearity["variance_inflation_factor"].max())
        condition_number = float(
            multicollinearity["standardized_design_condition_number"].max()
        )
        self.add(
            "style_factor_regression_not_ill_conditioned",
            maximum_vif < 5.0 and condition_number < 10.0,
            f"maximum_vif={maximum_vif:.3f}, condition_number={condition_number:.3f}",
        )
        inference_columns = [
            "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
            "annualized_alpha_ci_high", "alpha_hac_t_stat", "r_squared",
        ]
        complete = (
            len(summary) == 8 and len(annual) == 48
            and summary[inference_columns].notna().all().all()
            and summary["observations"].eq(1387).all()
            and summary["annualized_alpha_ci_low"].le(
                summary["annualized_alpha_arithmetic"]
            ).all()
            and summary["annualized_alpha_arithmetic"].le(
                summary["annualized_alpha_ci_high"]
            ).all()
        )
        self.add(
            "style_factor_multifactor_inference_complete", bool(complete),
            f"summary_rows={len(summary)}, annual_rows={len(annual)}",
        )
        primary = summary.loc[
            summary["scenario"].eq("daily_retry_limit_and_suspension_constrained")
        ]
        significant = primary.loc[
            primary["annualized_alpha_ci_low"].gt(0)
            | primary["annualized_alpha_ci_high"].lt(0),
            ["model", "alpha_pipeline"],
        ].apply(lambda row: f"{row['model']}/{row['alpha_pipeline']}", axis=1).tolist()
        self.findings.update({
            "style_factor_significant_residual_alpha_strategies": ",".join(significant),
            "style_factor_executable_alpha_min": float(
                primary["annualized_alpha_arithmetic"].min()
            ),
            "style_factor_executable_alpha_max": float(
                primary["annualized_alpha_arithmetic"].max()
            ),
            "style_factor_executable_r_squared_min": float(primary["r_squared"].min()),
            "style_factor_executable_r_squared_max": float(primary["r_squared"].max()),
        })

    def audit_model_portfolio_neutralization(self) -> None:
        protocol = self.require_table("model_neutralization_protocol", "v1")
        summary = self.require_table("model_neutralization_oos_summary", "v1")
        daily = self.require_table("model_neutralization_oos_daily", "v1")
        bootstrap = self.require_table("model_neutralization_oos_bootstrap", "v1")
        if any(frame.empty for frame in (protocol, summary, daily, bootstrap)):
            return
        row = protocol.iloc[0]
        locked = (
            not bool(row["historical_oos_used_for_selection"])
            and bool(row["models_locked_before_historical_oos"])
            and bool(row["common_exposure_complete_sample"])
        )
        self.add(
            "model_neutralization_protocol_locked", locked,
            "no historical-OOS selection; locked models; paired exposure-complete sample",
        )
        counts = daily.groupby(["pipeline", "model"])["asof_date"].nunique()
        self.add(
            "model_neutralization_paired_dates", counts.nunique() == 1,
            f"minimum_dates={counts.min()}, maximum_dates={counts.max()}",
        )
        combined_ridge = summary.loc[
            summary["pipeline"].eq("industry_size_neutralized")
            & summary["model"].eq("ridge_locked")
        ].iloc[0]
        size_residual = abs(float(
            combined_ridge["mean_score_log_size_correlation"]
        ))
        industry_residual = abs(float(
            combined_ridge["mean_score_industry_max_abs_mean"]
        ))
        # The residualizer pools date-specific groups with fewer than five
        # stocks into OTHER, while this reporting metric expands the original
        # broad codes. Exact fit-group orthogonality is audited above; the
        # expanded-code score residual should still remain economically tiny.
        self.add(
            "linear_model_neutral_score_residual_small",
            size_residual < 1e-10 and industry_residual < 5e-4,
            f"industry={industry_residual:.3e}, size={size_residual:.3e}",
        )
        expected_metrics = {
            "rank_ic", "net_return_annualized_arithmetic",
            "industry_active_weight_l1", "abs_score_log_size_correlation",
            "score_industry_max_abs_mean", "portfolio_industry_hhi",
        }
        observed_metrics = set(bootstrap["metric"].astype(str))
        self.add(
            "model_neutralization_inference_complete",
            expected_metrics.issubset(observed_metrics),
            f"metrics={','.join(sorted(observed_metrics))}",
        )
        combined = bootstrap.loc[
            bootstrap["pipeline"].eq("industry_size_neutralized")
        ]
        self.findings.update({
            "ridge_combined_neutral_oos_rank_ic": float(
                combined_ridge["mean_rank_ic"]
            ),
            "ridge_combined_neutral_oos_annualized_return": float(
                combined_ridge["portfolio_annualized_return"]
            ),
            "model_neutralization_combined_intervals": "; ".join(
                f"{item.model}/{item.metric}=[{item.ci_2_5pct:.6f},{item.ci_97_5pct:.6f}]"
                for item in combined.itertuples()
            ),
        })

    def audit_constrained_portfolio_neutralization(self) -> None:
        """Gate the explicit portfolio-level industry, size, and turnover controls."""
        protocol = self.require_table("constrained_portfolio_protocol", "v1")
        weekly = self.require_table("constrained_portfolio_oos_weekly", "v1")
        summary = self.require_table("constrained_portfolio_oos_summary", "v1")
        bootstrap = self.require_table("constrained_portfolio_oos_bootstrap", "v1")
        if any(frame.empty for frame in (protocol, weekly, summary, bootstrap)):
            return
        row = protocol.iloc[0]
        self.add(
            "constrained_portfolio_protocol_locked",
            not bool(row["historical_oos_used_for_selection"])
            and bool(row["constraint_policy_prespecified"])
            and str(row["source_experiment"]) == "model_portfolio_neutralization_v2",
            "no historical-OOS selection; prespecified constraints; repaired v2 source",
        )
        counts = weekly.groupby("strategy")["asof_date"].nunique()
        self.add(
            "constrained_portfolio_paired_dates", counts.nunique() == 1,
            f"minimum_dates={counts.min()}, maximum_dates={counts.max()}",
        )
        constrained = weekly.loc[weekly["construction"].eq("risk_constrained")]
        max_stock = float(constrained["stock_weight_violation"].max())
        max_size = float(constrained["size_bound_violation"].max())
        max_industry = float(constrained["industry_bound_violation"].max())
        self.add(
            "constrained_portfolio_risk_bounds",
            bool(constrained["optimizer_success"].all())
            and max(max_stock, max_size, max_industry) < 1e-10,
            f"stock={max_stock:.3e}, size={max_size:.3e}, industry={max_industry:.3e}",
        )
        turnover_rows = constrained.loc[constrained["effective_turnover_limit"].notna()]
        turnover_error = float(
            (turnover_rows["turnover"] - turnover_rows["effective_turnover_limit"])
            .clip(lower=0).max()
        )
        relaxation_fraction = float(constrained["turnover_limit_relaxed"].mean())
        self.add(
            "constrained_portfolio_turnover_bound",
            turnover_error < 1e-7 and relaxation_fraction == 0.0,
            f"maximum_excess={turnover_error:.3e}, relaxation_fraction={relaxation_fraction:.3%}",
        )
        expected_metrics = {
            "rank_ic", "net_return_annualized_arithmetic",
            "industry_active_weight_l1", "abs_portfolio_size_z", "turnover",
        }
        full_stack = bootstrap.loc[bootstrap["comparison"].eq("full_neutral_stack")]
        self.add(
            "constrained_portfolio_inference_complete",
            expected_metrics.issubset(set(full_stack["metric"].astype(str))),
            f"metrics={','.join(sorted(set(full_stack['metric'].astype(str))))}",
        )
        hgb_raw = summary.loc[
            summary["model"].eq("hgb_shallow15_locked")
            & summary["strategy"].eq("raw__risk_constrained")
        ].iloc[0]
        self.findings.update({
            "constrained_hgb_raw_oos_annualized_return": float(hgb_raw["annualized_return"]),
            "constrained_hgb_raw_oos_sharpe": float(hgb_raw["sharpe_zero_rate"]),
            "constrained_hgb_raw_oos_maximum_drawdown": float(hgb_raw["maximum_drawdown"]),
            "constrained_hgb_raw_mean_industry_active_weight_l1": float(
                hgb_raw["mean_industry_active_weight_l1"]
            ),
            "constrained_portfolio_turnover_relaxation_fraction": relaxation_fraction,
        })

    def audit_constrained_portfolio_daily_execution(self) -> None:
        """Gate the daily retry execution paths for constrained target holdings."""
        protocol = self.require_table("constrained_daily_execution_protocol", "v1")
        daily = self.require_table("constrained_daily_execution_daily", "v1")
        summary = self.require_table("constrained_daily_execution_summary", "v1")
        bootstrap = self.require_table("constrained_daily_execution_bootstrap", "v1")
        targets = self.require_table("constrained_execution_targets", "v1")
        if any(frame.empty for frame in (protocol, daily, summary, bootstrap, targets)):
            return
        row = protocol.iloc[0]
        self.add(
            "constrained_daily_execution_protocol_locked",
            not bool(row["historical_oos_used_for_selection"])
            and bool(row["unique_nonoverlap_daily_clock"])
            and bool(row["persistent_blocked_share_orders"])
            and str(row["source_experiment"]) == "constrained_portfolio_neutralization_v1",
            "frozen targets; unique daily clock; persistent blocked share orders",
        )
        target_counts = targets.groupby(
            ["model", "alpha_pipeline", "strategy"]
        )["execution_date"].nunique()
        self.add(
            "constrained_daily_execution_paired_targets",
            target_counts.nunique() == 1 and int(target_counts.min()) == 293,
            f"minimum_dates={target_counts.min()}, maximum_dates={target_counts.max()}",
        )
        daily_counts = daily.groupby(
            ["model", "alpha_pipeline", "strategy", "scenario"]
        )["trade_date"].nunique()
        accounting_error = float((daily["accounting_weight_sum"] - 1.0).abs().max())
        return_error = float((
            daily["net_return"] - daily["gross_return"] + daily["transaction_cost"]
        ).abs().max())
        self.add(
            "constrained_daily_execution_accounting_identity",
            daily_counts.nunique() == 1 and accounting_error < 1e-10 and return_error < 1e-10,
            f"dates={daily_counts.min()}-{daily_counts.max()}, "
            f"accounting={accounting_error:.3e}, return={return_error:.3e}",
        )
        ideal = daily.loc[daily["scenario"].eq("nonoverlap_all_fills_upper_bound")]
        hidden_turnover = float(ideal.loc[~ideal["new_target"].astype(bool), "turnover"].max())
        self.add(
            "constrained_daily_execution_no_hidden_rebalance",
            hidden_turnover < 1e-12,
            f"ideal_non_target_max_turnover={hidden_turnover:.3e}",
        )
        summary_pivot = summary.pivot(
            index=["model", "alpha_pipeline", "strategy"],
            columns="scenario", values="annualized_return",
        )
        haircut = (
            summary_pivot["daily_retry_limit_and_suspension_constrained"]
            < summary_pivot["nonoverlap_all_fills_upper_bound"]
        )
        interval_mask = pd.to_numeric(
            bootstrap["ci_97_5pct"], errors="coerce"
        ).lt(0)
        intervals_negative = bool(interval_mask.all())
        self.add(
            "constrained_daily_execution_haircut_detected",
            bool(haircut.all()) and intervals_negative,
            f"strategies_with_haircut={int(haircut.sum())}/{len(haircut)}, "
            f"negative_upper_intervals={int(interval_mask.sum())}/{len(interval_mask)}",
        )
        hgb_raw = summary.loc[
            summary["model"].eq("hgb_shallow15_locked")
            & summary["alpha_pipeline"].eq("raw")
            & summary["scenario"].eq("daily_retry_limit_and_suspension_constrained")
        ].iloc[0]
        self.findings.update({
            "constrained_daily_hgb_raw_annualized_return": float(
                hgb_raw["annualized_return"]
            ),
            "constrained_daily_hgb_raw_sharpe": float(hgb_raw["sharpe_zero_rate"]),
            "constrained_daily_hgb_raw_maximum_drawdown": float(
                hgb_raw["maximum_drawdown"]
            ),
            "constrained_daily_hgb_raw_mean_blocked_buy_weight": float(
                hgb_raw["mean_blocked_buy_weight"]
            ),
        })

    def audit_institutional_execution(self) -> None:
        """Gate the Phase-5 order state machine and account ledgers."""
        protocol = self.require_table("institutional_execution_protocol", "v1")
        daily = self.require_table("institutional_execution_daily", "v1")
        orders = self.require_table("institutional_execution_orders", "v1")
        events = self.require_table("institutional_execution_events", "v1")
        fills = self.require_table("institutional_execution_fills", "v1")
        positions = self.require_table("institutional_execution_positions", "v1")
        actions = self.require_table("institutional_execution_corporate_actions", "v1")
        summary = self.require_table("institutional_execution_summary", "v1")
        invariants = self.require_table("institutional_execution_invariants", "v1")
        if any(frame.empty for frame in (
            protocol, daily, orders, events, fills, positions, actions, summary, invariants,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "institutional_execution_evidence_disclosed",
            str(row["evidence_class"]) == "historical_oos_descriptive"
            and str(row["research_selection_role"]) == "diagnostic"
            and not bool(row["claims_independent_evidence"])
            and not bool(row["historical_oos_used_for_selection"]),
            "historical OOS diagnostic; no independent-evidence claim or model selection",
        )
        expected_scenarios = {
            "daily_close_institutional_constraints",
            "market_constraints_removed_upper_bound",
        }
        observed_scenarios = set(daily["scenario"].astype(str))
        dates = daily.groupby("scenario")["trade_date"].nunique()
        self.add(
            "institutional_execution_common_clock",
            observed_scenarios == expected_scenarios and dates.nunique() == 1
            and int(dates.min()) == 1387,
            f"scenarios={sorted(observed_scenarios)}, dates={dates.to_dict()}",
        )
        self.add(
            "institutional_execution_invariants",
            invariants["all_invariants_pass"].astype(bool).all()
            and invariants["t_plus_one_violation_count"].astype(int).eq(0).all()
            and invariants["negative_cash_day_count"].astype(int).eq(0).all()
            and invariants["negative_position_count"].astype(int).eq(0).all(),
            str(invariants.to_dict(orient="records")),
        )
        accounting_error = float((daily["accounting_weight_sum"] - 1.0).abs().max())
        nav_error = float(daily["nav_reconciliation_error"].abs().max())
        self.add(
            "institutional_execution_nav_cash_positions_reconcile",
            accounting_error <= 1e-12 and nav_error <= 1e-4
            and float(daily["cash"].min()) >= -1e-6,
            f"accounting_error={accounting_error:.3e}, nav_error_cny={nav_error:.3e}, "
            f"minimum_cash={daily['cash'].min():.6f}",
        )
        fee_error = abs(float(fills["total_cost"].sum() - daily["total_cost"].sum()))
        self.add(
            "institutional_execution_fill_fees_reconcile",
            fee_error <= 1e-6 and fills["total_cost"].ge(0).all(),
            f"fill_minus_daily_fee_cny={fee_error:.3e}",
        )
        sell_fills = fills.loc[fills["side"].eq("sell")]
        t1_violations = int((
            sell_fills["executed_shares"] > sell_fills["available_before_fill"] + 1e-8
        ).sum())
        self.add(
            "institutional_execution_t_plus_one_fill_gate",
            t1_violations == 0,
            f"sell_fills={len(sell_fills):,}, violations={t1_violations}",
        )
        allowed_events = {
            "NEW", "BLOCKED", "PARTIALLY_FILLED", "FILLED", "CANCELLED",
            "EXPIRED", "REPLACED", "ACTION_ADJUSTED",
        }
        event_statuses = set(events["status"].astype(str))
        new_orders = events.loc[
            events["status"].eq("NEW"), ["scenario", "order_id"]
        ].drop_duplicates()
        order_keys = orders[["scenario", "order_id"]].drop_duplicates()
        self.add(
            "institutional_execution_order_state_machine",
            event_statuses <= allowed_events
            and len(new_orders) == len(order_keys) == len(orders),
            f"event_statuses={sorted(event_statuses)}, orders={len(orders):,}, "
            f"new_events={len(new_orders):,}",
        )
        terminal = {"FILLED", "CANCELLED", "EXPIRED", "REPLACED"}
        self.add(
            "institutional_execution_orders_terminal",
            set(orders["status"].astype(str)) <= terminal
            and orders["remaining_shares"].ge(-1e-8).all(),
            f"terminal_statuses={sorted(set(orders['status'].astype(str)))}",
        )
        action_error = float(actions["value_reconciliation_error"].abs().max())
        self.add(
            "institutional_execution_corporate_action_proxy_reconciles",
            action_error <= 1e-6,
            f"actions={len(actions):,}, max_error_cny={action_error:.3e}; proxy_not_official",
        )
        by_scenario = summary.set_index("scenario")
        constrained = by_scenario.loc["daily_close_institutional_constraints"]
        upper = by_scenario.loc["market_constraints_removed_upper_bound"]
        self.add(
            "institutional_execution_constraints_have_measurable_haircut",
            float(constrained["annualized_return"]) < float(upper["annualized_return"])
            and float(constrained["mean_target_tracking_l1"])
            > float(upper["mean_target_tracking_l1"]),
            f"annualized={constrained['annualized_return']:.6f} vs {upper['annualized_return']:.6f}; "
            f"tracking_l1={constrained['mean_target_tracking_l1']:.6f} vs "
            f"{upper['mean_target_tracking_l1']:.6f}",
        )
        self.findings.update({
            "institutional_execution_hgb_raw_annualized_return": float(
                constrained["annualized_return"]
            ),
            "institutional_execution_hgb_raw_sharpe": float(constrained["sharpe_zero_rate"]),
            "institutional_execution_hgb_raw_maximum_drawdown": float(
                constrained["maximum_drawdown"]
            ),
            "institutional_execution_hgb_raw_fill_ratio": float(
                constrained["fill_ratio_by_shares"]
            ),
        })

    def audit_unified_account_ledger(self) -> None:
        """Gate the Phase-7 cross-table account ledger and replay checks."""
        protocol = self.require_table("unified_account_protocol", "v1")
        daily = self.require_table("unified_account_daily_ledger", "v1")
        security = self.require_table("unified_account_security_ledger", "v1")
        cash = self.require_table("unified_account_cash_journal", "v1")
        orders = self.require_table("unified_account_orders", "v1")
        events = self.require_table("unified_account_events", "v1")
        fills = self.require_table("unified_account_fills", "v1")
        actions = self.require_table("unified_account_corporate_actions", "v1")
        daily_recon = self.require_table("unified_account_daily_reconciliation", "v1")
        position_recon = self.require_table("unified_account_position_reconciliation", "v1")
        exceptions = self.require_table("unified_account_order_exceptions", "v1")
        failures = self.require_table("unified_account_failed_assertions", "v1")
        checks = self.require_table("unified_account_checks", "v1")
        if any(frame.empty for frame in (
            protocol, daily, security, cash, orders, events, fills, actions,
            daily_recon, position_recon, exceptions, failures, checks,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "unified_account_protocol_is_read_only_diagnostic",
            str(row["evidence_class"]) == "historical_oos_descriptive"
            and str(row["research_selection_role"]) == "diagnostic"
            and not bool(row["claims_independent_evidence"])
            and not bool(row["source_outputs_mutated"])
            and not bool(row["returns_recomputed_or_changed"]),
            "frozen Phase 5 outputs normalized without changing returns or claiming new evidence",
        )
        hard = checks.loc[checks["status"].eq("FAIL")]
        self.add(
            "unified_account_all_hard_reconciliations_pass",
            hard.empty and len(checks) == 16,
            f"checks={len(checks)}, pass={int(checks['status'].eq('PASS').sum())}, "
            f"warn={int(checks['status'].eq('WARN').sum())}, fail={len(hard)}",
        )
        self.add(
            "unified_account_common_daily_key",
            len(daily) == len(cash) == len(daily_recon) == 1387
            and daily["trade_date"].nunique() == 1387
            and not daily.duplicated(["account_id", "trade_date"]).any(),
            f"daily={len(daily):,}, cash={len(cash):,}, reconciliation={len(daily_recon):,}",
        )
        required_security = {
            "target_weight", "actual_shares", "actual_weight", "cash_weight", "weight_gap",
        }
        self.add(
            "unified_account_target_actual_cash_semantics",
            required_security <= set(security.columns)
            and not security.duplicated(["account_id", "trade_date", "instrument_id"]).any(),
            f"security_rows={len(security):,}; required_fields={sorted(required_security)}",
        )
        max_cash = float(cash["cash_reconciliation_error"].abs().max())
        max_nav = float(daily_recon["nav_identity_error"].abs().max())
        max_shares = float(position_recon["share_reconciliation_error"].abs().max())
        self.add(
            "unified_account_cash_nav_position_replay",
            max_cash <= 1e-4 and max_nav <= 1e-4 and max_shares <= 1e-6,
            f"cash_error_cny={max_cash:.3e}, nav_error_cny={max_nav:.3e}, "
            f"share_error={max_shares:.3e}",
        )
        self.add(
            "unified_account_assertion_log_clean",
            len(failures) == 1
            and str(failures.iloc[0]["check_name"]) == "NO_FAILED_ASSERTIONS"
            and str(failures.iloc[0]["status"]) == "PASS",
            failures.to_dict(orient="records").__str__(),
        )
        self.warn(
            "legacy_corporate_action_scenario_key_missing",
            "The frozen Phase 5 corporate-action table concatenated two scenarios without a "
            "scenario key. Phase 7 preserves it and reconstructs account-specific actions from "
            "prior actual holdings plus frozen market action fields.",
        )
        self.findings.update({
            "unified_account_daily_rows": len(daily),
            "unified_account_security_rows": len(security),
            "unified_account_order_exceptions": len(exceptions),
            "unified_account_max_cash_error_cny": max_cash,
            "unified_account_max_nav_error_cny": max_nav,
            "unified_account_max_share_error": max_shares,
        })

    def audit_statistical_inference(self) -> None:
        """Gate Phase-8 paired inference, dependence sensitivity and FDR families."""
        protocol = self.require_table("statistical_inference_protocol", "v1")
        series = self.require_table("statistical_inference_series", "v1")
        alignment = self.require_table("statistical_inference_alignment", "v1")
        hac = self.require_table("statistical_inference_hac", "v1")
        bootstrap = self.require_table("statistical_inference_block_bootstrap", "v1")
        fdr = self.require_table("statistical_inference_fdr", "v1")
        family = self.require_table("statistical_inference_family_summary", "v1")
        sensitivity = self.require_table("statistical_inference_block_sensitivity", "v1")
        trials = self.require_table("statistical_trial_registry", "v1")
        checks = self.require_table("statistical_inference_checks", "v1")
        failures = self.require_table("statistical_inference_failed_assertions", "v1")
        if any(frame.empty for frame in (
            protocol, series, alignment, hac, bootstrap, fdr, family,
            sensitivity, trials, checks, failures,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "statistical_inference_protocol_discloses_post_oos_status",
            str(row["evidence_class"]) == "post_historical_oos_adaptation"
            and str(row["research_selection_role"]) == "post_hoc_adaptation"
            and not bool(row["claims_independent_evidence"])
            and not bool(row["historical_oos_used_for_selection"])
            and not bool(row["block_length_selected_from_results"])
            and not bool(row["future_success_probability_claimed"]),
            "harmonized inference is post-historical-OOS and does not claim future probability",
        )
        self.add(
            "statistical_inference_all_protocol_checks_pass",
            checks["status"].eq("PASS").all() and len(checks) == 10,
            f"checks={len(checks)}, failures={int(checks['status'].eq('FAIL').sum())}",
        )
        expected = {
            "factor_rank_ic": 15, "factor_gross_spread": 15,
            "model_incremental_alpha": 4, "portfolio_construction_increment": 3,
        }
        counts = fdr.groupby("family_id")["hypothesis_id"].nunique().to_dict()
        self.add(
            "statistical_inference_fdr_families_not_pooled",
            counts == expected and fdr["family_test_count"].eq(
                fdr["family_id"].map(expected)
            ).all(),
            f"family_hypotheses={counts}",
        )
        paired = alignment["family_id"].isin([
            "model_incremental_alpha", "portfolio_construction_increment",
        ])
        self.add(
            "statistical_inference_paired_daily_clocks",
            alignment.loc[paired, "missing_observations"].eq(0).all()
            and alignment.loc[paired, "duplicate_dates"].eq(0).all()
            and alignment.loc[paired, "complete_observations"].eq(1387).all(),
            f"paired_hypotheses={int(paired.sum())}, observations="
            f"{sorted(alignment.loc[paired, 'complete_observations'].unique())}",
        )
        self.add(
            "statistical_inference_dependence_sensitivity_complete",
            len(hac) == 148 and len(bootstrap) == 155 and len(sensitivity) == 37,
            f"hac={len(hac)}, block_bootstrap={len(bootstrap)}, sensitivity={len(sensitivity)}",
        )
        self.add(
            "statistical_trial_registry_complete",
            len(trials) == 172 and trials["trial_category"].nunique() == 11
            and not trials["included_in_cross_category_fdr"].astype(bool).any(),
            f"trials={len(trials)}, categories={trials['trial_category'].nunique()}",
        )
        primary = fdr.loc[
            fdr["family_id"].eq("model_incremental_alpha")
            & fdr["hypothesis_id"].eq("hgb_shallow15_locked__raw")
        ]
        self.add(
            "statistical_primary_incremental_alpha_not_overclaimed",
            len(primary) == 1 and not bool(primary.iloc[0]["survives_bh_fdr"])
            and float(primary.iloc[0]["ci_low"]) <= 0
            and str(primary.iloc[0]["claim_status"])
            == "positive_value_not_established_after_family_fdr",
            primary[["point_estimate", "ci_low", "ci_high", "bh_fdr_adjusted_p",
                     "claim_status"]].to_dict(orient="records").__str__(),
        )
        self.add(
            "statistical_inference_assertion_log_clean",
            len(failures) == 1
            and str(failures.iloc[0]["check_name"]) == "NO_FAILED_ASSERTIONS",
            failures.to_dict(orient="records").__str__(),
        )
        self.findings.update({
            "statistical_inference_hypotheses": len(fdr),
            "statistical_trial_registry_rows": len(trials),
            "hgb_raw_incremental_return_point_estimate": float(primary.iloc[0]["point_estimate"]),
            "hgb_raw_incremental_return_ci_low": float(primary.iloc[0]["ci_low"]),
            "hgb_raw_incremental_return_ci_high": float(primary.iloc[0]["ci_high"]),
            "hgb_raw_incremental_return_fdr_q": float(primary.iloc[0]["bh_fdr_adjusted_p"]),
        })

    def audit_alpha_evidence_matrix(self) -> None:
        """Gate Phase-9 benchmark hierarchy and risk-adjusted incremental evidence."""
        protocol = self.require_table("alpha_evidence_protocol", "v2")
        aligned = self.require_table("alpha_evidence_aligned_daily", "v2")
        alignment = self.require_table("alpha_evidence_alignment", "v2")
        regressions = self.require_table("alpha_evidence_risk_adjusted_regression", "v2")
        fdr = self.require_table("alpha_evidence_risk_adjusted_fdr", "v2")
        sensitivity = self.require_table("alpha_evidence_risk_sensitivity", "v2")
        execution = self.require_table("alpha_evidence_execution_comparison", "v2")
        exposure = self.require_table("alpha_evidence_exposure_match", "v2")
        matrix = self.require_table("alpha_evidence_candidate_matrix", "v2")
        definitions = self.require_table("alpha_evidence_measure_definitions", "v2")
        gaps = self.require_table("alpha_evidence_gap_register", "v2")
        checks = self.require_table("alpha_evidence_checks", "v2")
        failures = self.require_table("alpha_evidence_failed_assertions", "v2")
        fill_protocol = self.require_table("incremental_alpha_fill_protocol", "v1")
        fill_summary = self.require_table("incremental_alpha_fill_summary", "v1")
        fill_invariants = self.require_table("incremental_alpha_fill_invariants", "v1")
        fill_comparison = self.require_table("incremental_alpha_fill_comparison", "v1")
        fill_checks = self.require_table("incremental_alpha_fill_checks", "v1")
        fill_failures = self.require_table("incremental_alpha_fill_failed_assertions", "v1")
        if any(frame.empty for frame in (
            protocol, aligned, alignment, regressions, fdr, sensitivity,
            execution, exposure, matrix, definitions, gaps, checks, failures,
            fill_protocol, fill_summary, fill_invariants, fill_comparison,
            fill_checks, fill_failures,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "alpha_evidence_protocol_discloses_nonindependent_history",
            str(row["evidence_class"]) == "post_historical_oos_adaptation"
            and str(row["research_selection_role"]) == "post_hoc_adaptation"
            and not bool(row["claims_independent_evidence"])
            and not bool(row["historical_oos_used_for_selection"]),
            f"phase_status={row['phase_status']}",
        )
        self.add(
            "alpha_evidence_all_implemented_scope_checks_pass",
            len(checks) == 12 and checks["status"].eq("PASS").all(),
            f"checks={len(checks)}, failures={int(checks['status'].eq('FAIL').sum())}",
        )
        self.add(
            "alpha_evidence_common_clock_complete",
            len(alignment) == 24
            and alignment["complete_observations"].eq(1387).all()
            and alignment["missing_observations"].eq(0).all()
            and alignment["duplicate_dates"].eq(0).all()
            and aligned["trade_date"].nunique() == 1387,
            f"alignment_rows={len(alignment)}, aligned_rows={len(aligned)}",
        )
        self.add(
            "alpha_evidence_risk_model_sensitivity_complete",
            len(regressions) == 96 and len(sensitivity) == 4
            and set(regressions["newey_west_lags"].astype(int)) == {5, 10, 20, 40}
            and set(regressions["return_type"]) == {"price_return", "total_return"}
            and regressions["rate_scenario"].nunique() == 3,
            f"regressions={len(regressions)}, rf={regressions['rate_scenario'].nunique()}, "
            f"return_types={sorted(regressions['return_type'].unique())}",
        )
        self.add(
            "alpha_evidence_four_candidate_fdr_family",
            len(fdr) == 4 and fdr["bh_fdr_adjusted_p"].between(0, 1).all()
            and not fdr["survives_bh_fdr"].astype(bool).any(),
            f"survivors={int(fdr['survives_bh_fdr'].astype(bool).sum())}, "
            f"q={fdr['bh_fdr_adjusted_p'].round(6).tolist()}",
        )
        self.add(
            "alpha_evidence_does_not_call_capm_pure_alpha",
            len(matrix) == 4
            and not matrix["pure_stock_selection_alpha_identified"].astype(bool).any()
            and not definitions["may_be_called_pure_stock_alpha"].astype(bool).any(),
            matrix[["model", "alpha_pipeline", "evidence_verdict"]]
            .to_dict(orient="records").__str__(),
        )
        self.add(
            "alpha_evidence_local_proxy_and_exposure_error_disclosed",
            len(exposure) == 4 and matrix["style_model_status"].eq(
                "local_research_proxy_not_commercial_risk_model"
            ).all(),
            f"exposure_rows={len(exposure)}",
        )
        self.add(
            "alpha_evidence_assertion_log_clean",
            len(failures) == 1
            and str(failures.iloc[0]["check_name"]) == "NO_FAILED_ASSERTIONS",
            failures.to_dict(orient="records").__str__(),
        )
        fill_gap = gaps.loc[gaps["gap"].eq("share_level_fill_comparison")]
        self.add(
            "alpha_evidence_share_level_fill_comparison_complete",
            len(fill_gap) == 1 and str(fill_gap.iloc[0]["priority"]) == "P0"
            and str(fill_gap.iloc[0]["status"]) == "CLOSED"
            and execution["share_level_fill_ratio_available"].astype(bool).all()
            and len(fill_summary) == 8 and len(fill_comparison) == 4
            and fill_summary["fill_ratio_by_shares"].between(0, 1).all(),
            f"paths={len(fill_summary)}, ratios="
            f"{fill_summary['fill_ratio_by_shares'].round(6).tolist()}",
        )
        self.add(
            "alpha_evidence_fill_accounting_invariants_pass",
            len(fill_invariants) == 8
            and fill_invariants["all_invariants_pass"].astype(bool).all()
            and fill_invariants["t_plus_one_violation_count"].eq(0).all()
            and fill_checks["status"].eq("PASS").all()
            and len(fill_failures) == 1
            and str(fill_failures.iloc[0]["check_name"]) == "NO_FAILED_ASSERTIONS",
            f"paths={len(fill_invariants)}, checks={len(fill_checks)}",
        )
        hgb_raw = matrix.loc[
            matrix["model"].eq("hgb_shallow15_locked")
            & matrix["alpha_pipeline"].eq("raw")
        ].iloc[0]
        hgb_neutral = matrix.loc[
            matrix["model"].eq("hgb_shallow15_locked")
            & matrix["alpha_pipeline"].eq("industry_size_neutralized")
        ].iloc[0]
        self.findings.update({
            "alpha_evidence_phase_status": str(row["phase_status"]),
            "alpha_evidence_regression_specifications": len(regressions),
            "alpha_evidence_risk_adjusted_fdr_survivors": int(
                fdr["survives_bh_fdr"].astype(bool).sum()
            ),
            "hgb_raw_risk_adjusted_incremental_alpha": float(
                hgb_raw["risk_adjusted_incremental_alpha_arithmetic"]
            ),
            "hgb_neutral_risk_adjusted_incremental_alpha": float(
                hgb_neutral["risk_adjusted_incremental_alpha_arithmetic"]
            ),
        })

    def audit_incremental_alpha_test(self) -> None:
        """Gate the characteristic-matched executable incremental-alpha test."""
        protocol = self.require_table("incremental_alpha_protocol", "v1")
        targets = self.require_table("incremental_alpha_style_targets", "v1")
        exposure = self.require_table("incremental_alpha_exposure_audit", "v1")
        daily = self.require_table("incremental_alpha_daily", "v1")
        summary = self.require_table("incremental_alpha_summary", "v1")
        bootstrap = self.require_table("incremental_alpha_bootstrap", "v1")
        comparison = self.require_table("incremental_alpha_comparison", "v1")
        if any(frame.empty for frame in (
            protocol, targets, exposure, daily, summary, bootstrap, comparison,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "incremental_alpha_protocol_locked",
            not bool(row["historical_oos_used_for_selection"])
            and bool(row["pit_style_exposures_only"])
            and bool(row["same_daily_execution_and_cost"])
            and bool(row["industry_matched"])
            and bool(row["volatility_matching_is_ex_post_diagnostic"])
            and str(row["source_experiment"]) == "constrained_portfolio_daily_execution_v1",
            "frozen ML targets; PIT style peers; same 20bp executable clock",
        )
        keys = ["model", "alpha_pipeline", "strategy"]
        date_counts = targets.groupby(keys)["execution_date"].nunique()
        weight_sums = targets.groupby([*keys, "execution_date"])["target_weight"].sum()
        max_weight = float(targets["target_weight"].max())
        self.add(
            "incremental_alpha_target_constraints",
            date_counts.nunique() == 1 and int(date_counts.min()) == 293
            and np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10)
            and max_weight <= 0.005000001,
            f"dates={date_counts.min()}-{date_counts.max()}, max_weight={max_weight:.8f}",
        )
        noninitial = exposure.groupby(keys).cumcount().gt(0)
        maximum_turnover = float(exposure.loc[noninitial, "target_turnover"].max())
        maximum_industry_gap = float(exposure["maximum_abs_industry_weight_gap"].max())
        mean_overlap = float(exposure["portfolio_overlap_weight"].mean())
        maximum_mean_style_gap = max(
            float(exposure.groupby(keys)[f"{style}_exposure_gap"].apply(
                lambda values: values.abs().mean()
            ).max())
            for style in ("size", "beta", "volatility", "liquidity", "value", "momentum")
        )
        self.add(
            "incremental_alpha_exposure_and_turnover_match",
            maximum_turnover <= 0.300001
            and maximum_industry_gap <= 0.02
            and mean_overlap <= 0.02
            and maximum_mean_style_gap <= 0.35,
            f"turnover={maximum_turnover:.6f}, industry_gap={maximum_industry_gap:.4f}, "
            f"overlap={mean_overlap:.4f}, max_mean_style_gap={maximum_mean_style_gap:.4f}",
        )
        daily_counts = daily.groupby([*keys, "portfolio_type"])["trade_date"].nunique()
        accounting_error = float((daily["accounting_weight_sum"] - 1.0).abs().max())
        return_error = float((
            daily["net_return"] - daily["gross_return"] + daily["transaction_cost"]
        ).abs().max())
        self.add(
            "incremental_alpha_common_executable_clock",
            daily_counts.nunique() == 1 and int(daily_counts.min()) == 1387
            and accounting_error < 1e-10 and return_error < 1e-10,
            f"dates={daily_counts.min()}-{daily_counts.max()}, "
            f"accounting={accounting_error:.3e}, return={return_error:.3e}",
        )
        original = self.require_table("constrained_daily_execution_summary", "v1")
        original = original.loc[
            original["scenario"].eq("daily_retry_limit_and_suspension_constrained")
        ][["model", "alpha_pipeline", "strategy", "annualized_return"]]
        reproduced = summary.loc[summary["portfolio_type"].eq("ml_model")][
            ["model", "alpha_pipeline", "strategy", "annualized_return"]
        ]
        merged = reproduced.merge(
            original, on=keys, suffixes=("_incremental", "_original"),
            validate="one_to_one",
        )
        reproduction_error = float((
            merged["annualized_return_incremental"] - merged["annualized_return_original"]
        ).abs().max())
        self.add(
            "incremental_alpha_reproduces_frozen_ml_path",
            len(merged) == 4 and reproduction_error < 1e-12,
            f"strategies={len(merged)}, max_annual_return_error={reproduction_error:.3e}",
        )
        expected_metrics = {
            "incremental_net_return_annualized_arithmetic",
            "volatility_matched_incremental_return_annualized_arithmetic",
            "sharpe_improvement", "maximum_drawdown_improvement",
            "turnover_improvement", "return_per_turnover_improvement",
        }
        complete = bootstrap.groupby(keys)["metric"].apply(set)
        self.add(
            "incremental_alpha_inference_and_multiplicity_complete",
            len(complete) == 4 and complete.apply(lambda values: expected_metrics <= values).all()
            and bootstrap["bh_fdr_adjusted_probability"].between(0, 1).all(),
            f"comparisons={len(complete)}, metrics={bootstrap['metric'].nunique()}",
        )
        alpha = bootstrap.loc[
            bootstrap["metric"].eq("incremental_net_return_annualized_arithmetic")
        ]
        hgb_neutral = alpha.loc[
            alpha["model"].eq("hgb_shallow15_locked")
            & alpha["alpha_pipeline"].eq("industry_size_neutralized")
        ].iloc[0]
        ridge_raw = comparison.loc[
            comparison["model"].eq("ridge_locked")
            & comparison["alpha_pipeline"].eq("raw")
        ].iloc[0]
        self.findings.update({
            "incremental_alpha_individual_intervals_above_zero": int(
                alpha["ci_2_5pct"].gt(0).sum()
            ),
            "incremental_alpha_bh_fdr_survivors": int(
                alpha["survives_bh_fdr_5pct"].astype(bool).sum()
            ),
            "hgb_neutral_incremental_return_annualized_arithmetic": float(
                hgb_neutral["point_estimate"]
            ),
            "hgb_neutral_incremental_return_ci_low": float(hgb_neutral["ci_2_5pct"]),
            "hgb_neutral_incremental_return_ci_high": float(hgb_neutral["ci_97_5pct"]),
            "ridge_raw_turnover_value_survives_multiplicity": bool(
                ridge_raw[
                    "multiplicity_controlled_risk_or_efficiency_value_detected"
                ]
            ),
        })

    def audit_engineering_quality(self) -> None:
        key = "v2"
        protocol = self.require_table("engineering_quality_protocol", key)
        implementation = self.require_table("engineering_implementation_inventory", key)
        seeds = self.require_table("engineering_random_seed_inventory", key)
        tests = self.require_table("engineering_test_layer_inventory", key)
        contracts = self.require_table("engineering_output_contracts", key)
        gaps = self.require_table("engineering_gap_register", key)
        checks = self.require_table("engineering_quality_checks", key)
        failures = self.require_table("engineering_failed_assertions", key)
        if any(frame.empty for frame in (
            protocol, implementation, seeds, tests, contracts, gaps, checks, failures,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "engineering_quality_governance_only",
            str(row["evidence_class"]) == "governance_only"
            and str(row["research_selection_role"]) == "governance"
            and not bool(row["research_logic_changed"])
            and not bool(row["frozen_results_reselected"]),
            f"canonical_module={row['canonical_execution_market_module']}",
        )
        self.add(
            "engineering_quality_all_checks_pass",
            len(checks) == 8 and checks["status"].eq("PASS").all(),
            f"checks={len(checks)}",
        )
        self.add(
            "engineering_quality_single_execution_market_implementation",
            implementation.loc[
                implementation["logic"].isin([
                    "corporate_action_proxy_formula", "execution_buy_limit_rule",
                ]), "path"
            ].nunique() == 1,
            implementation.to_dict(orient="records").__str__(),
        )
        self.add(
            "engineering_quality_tests_and_seeds_complete",
            tests["exists"].astype(bool).all()
            and tests.groupby("layer")["test_functions"].sum().gt(0).all()
            and seeds["explicit_seed"].astype(bool).all(),
            f"layers={tests['layer'].nunique()}, random_generators={len(seeds)}",
        )
        self.add(
            "engineering_quality_output_contracts_frozen",
            len(contracts) == 14
            and contracts["protocol_has_experiment_id"].astype(bool).all()
            and contracts["frozen_snapshot_immutable"].astype(bool).all(),
            contracts[["stage", "frozen_experiment"]].to_dict(orient="records").__str__(),
        )
        self.add(
            "engineering_quality_no_open_p0",
            not (gaps["priority"].eq("P0") & gaps["status"].str.startswith("OPEN")).any(),
            gaps.groupby(["priority", "status"]).size().to_dict().__str__(),
        )
        self.add(
            "engineering_quality_assertion_log_clean",
            len(failures) == 1
            and str(failures.iloc[0]["check_name"]) == "NO_FAILED_ASSERTIONS",
            failures.to_dict(orient="records").__str__(),
        )
        self.findings.update({
            "engineering_canonical_execution_market_module": str(
                row["canonical_execution_market_module"]
            ),
            "engineering_test_layers": int(tests["layer"].nunique()),
            "engineering_open_p0_gaps": int((
                gaps["priority"].eq("P0") & gaps["status"].str.startswith("OPEN")
            ).sum()),
        })

    def audit_paper_trading(self) -> None:
        """Gate the forward-only runner without opening lockbox performance."""
        protocol = self.require_table("paper_trading_protocol", "v1")
        contracts = self.require_table("paper_trading_source_contracts", "v1")
        status = self.require_table("paper_trading_status", "v1")
        checks = self.require_table("paper_trading_checks", "v1")
        failures = self.require_table("paper_trading_failed_assertions", "v1")
        if any(frame.empty for frame in (protocol, contracts, status, checks, failures)):
            return
        row = protocol.iloc[0]
        self.add(
            "paper_pipeline_does_not_claim_forward_evidence",
            str(row["evidence_class"]) == "governance_only"
            and not bool(row["claims_independent_evidence"])
            and not bool(row["ready_for_live_capital"]),
            f"observations={row['successful_forward_observations']}",
        )
        self.add(
            "paper_pipeline_contract_is_complete",
            {"signal_input", "model_bundle", "target", "account", "operations_log"}
            <= set(contracts["artifact"].astype(str))
            and contracts["required"].astype(bool).all(),
            f"contracts={len(contracts)}",
        )
        self.add(
            "paper_pipeline_checks_all_pass",
            len(checks) == 7 and checks["status"].eq("PASS").all(),
            f"checks={len(checks)}",
        )
        from quant_research.paper_trading import (
            load_config as load_paper_config,
            verify_model_bundle,
            verify_operations_log,
        )
        paper_cfg = load_paper_config()
        model_manifest = verify_model_bundle(paper_cfg)
        events = verify_operations_log(ROOT / paper_cfg["storage"]["operations_log"])
        success = [event for event in events if event["status"] == "SUCCESS"]
        success_dates = [str(event["trade_date"]) for event in success]
        self.add(
            "paper_model_bundle_is_commit_and_hash_anchored",
            len(str(model_manifest["freeze_commit"])) == 40
            and len(str(model_manifest["model_sha256"])) == 64
            and int(model_manifest["fit_observations"]) > 0
            and model_manifest["performance_results_included"] is False,
            f"commit={model_manifest['freeze_commit']}, observations={model_manifest['fit_observations']}",
        )
        self.add(
            "paper_operations_log_and_run_manifests_are_immutable",
            all(
                (ROOT / event["run_directory"] / "run.json").exists()
                and sha256_file(ROOT / event["run_directory"] / "run.json")
                == event["run_manifest_sha256"]
                for event in events
            ),
            f"events={len(events)}",
        )
        self.add(
            "paper_observations_are_forward_only_and_unique",
            len(success_dates) == len(set(success_dates))
            and all(date >= str(paper_cfg["activation_not_before"]) for date in success_dates),
            f"success_dates={success_dates}",
        )
        status_row = status.iloc[0]
        self.add(
            "paper_status_matches_append_only_log",
            int(status_row["attempts"]) == len(events)
            and int(status_row["successful_observations"]) == len(success)
            and not bool(status_row["performance_validated"]),
            f"attempts={len(events)}, successes={len(success)}",
        )
        self.findings.update({
            "paper_trading_attempts": len(events),
            "paper_trading_successful_observations": len(success),
            "future_lockbox_interim_performance_sealed": True,
            "paper_trading_performance_validated": False,
        })

    def audit_multi_factor_risk_attribution(self) -> None:
        """Gate PIT style coverage, exact attribution, and lagged risk forecasts."""
        protocol = self.require_table("risk_model_protocol", "v1")
        beta_index = self.require_table("pit_beta_exposure_index", "v1")
        factor_returns = self.require_table("risk_model_factor_returns", "v1")
        attribution = self.require_table("risk_model_portfolio_attribution", "v1")
        forecasts = self.require_table("risk_model_portfolio_forecasts", "v1")
        summary = self.require_table("risk_model_attribution_summary", "v1")
        bootstrap = self.require_table("risk_model_attribution_bootstrap", "v1")
        if any(frame.empty for frame in (
            protocol, beta_index, factor_returns, attribution,
            forecasts, summary, bootstrap,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "risk_attribution_protocol_locked",
            not bool(row["historical_oos_used_for_selection"])
            and not bool(row["factor_covariance_uses_current_or_future_return"])
            and bool(row["attribution_identity_required"])
            and str(row["source_experiment"]) == "constrained_portfolio_neutralization_v1",
            "prespecified PIT factors; lagged covariance; exact identity required",
        )
        minimum_beta_coverage = float(beta_index["beta_coverage"].min())
        beta_duplicates = int(beta_index["duplicate_instrument_date_keys"].sum())
        self.add(
            "pit_beta_coverage_and_keys",
            minimum_beta_coverage >= 0.90 and beta_duplicates == 0,
            f"minimum_coverage={minimum_beta_coverage:.2%}, duplicates={beta_duplicates}",
        )
        counts = attribution.groupby(
            ["model", "alpha_pipeline", "strategy"]
        )["asof_date"].nunique()
        maximum_identity_error = max(
            float(attribution["attribution_error"].abs().max()),
            float(attribution["reported_net_error"].abs().max()),
        )
        self.add(
            "risk_attribution_exact_and_paired",
            counts.nunique() == 1 and int(counts.min()) == 293
            and maximum_identity_error < 1e-10,
            f"dates={counts.min()}-{counts.max()}, max_error={maximum_identity_error:.3e}",
        )
        forecast_values = forecasts[[
            "forecast_annualized_volatility", "forecast_annualized_tracking_error",
        ]].to_numpy(dtype="float64")
        minimum_history = int(forecasts["factor_history_weeks"].min())
        self.add(
            "risk_forecast_lagged_finite_positive",
            minimum_history >= 52 and np.isfinite(forecast_values).all()
            and (forecast_values > 0).all(),
            f"minimum_prior_weeks={minimum_history}, forecasts={len(forecasts)}",
        )
        expected = {
            "industry_active_contribution", "style_contribution",
            "specific_selection_contribution", "active_gross_return", "active_net_return",
        }
        self.add(
            "risk_attribution_inference_complete",
            expected.issubset(set(bootstrap["metric"].astype(str))),
            f"metrics={','.join(sorted(set(bootstrap['metric'].astype(str))))}",
        )
        hgb_raw = summary.loc[
            summary["model"].eq("hgb_shallow15_locked")
            & summary["alpha_pipeline"].eq("raw")
        ].iloc[0]
        hgb_specific = bootstrap.loc[
            bootstrap["model"].eq("hgb_shallow15_locked")
            & bootstrap["alpha_pipeline"].eq("raw")
            & bootstrap["metric"].eq("specific_selection_contribution")
        ].iloc[0]
        self.findings.update({
            "pit_beta_minimum_coverage": minimum_beta_coverage,
            "hgb_raw_annualized_style_contribution_arithmetic": float(
                hgb_raw["annualized_style_contribution_arithmetic"]
            ),
            "hgb_raw_annualized_specific_selection_contribution_arithmetic": float(
                hgb_raw["annualized_specific_selection_contribution_arithmetic"]
            ),
            "hgb_raw_specific_selection_ci_low": float(hgb_specific["ci_2_5pct"]),
            "hgb_raw_specific_selection_ci_high": float(hgb_specific["ci_97_5pct"]),
            "hgb_raw_mean_forecast_tracking_error": float(
                hgb_raw["mean_forecast_annualized_tracking_error"]
            ),
        })

    def audit_risk_aware_portfolio_optimization(self) -> None:
        """Gate lagged inputs, hard constraints, and the risk/return decision."""
        protocol = self.require_table("risk_aware_portfolio_protocol", "v2")
        weekly = self.require_table("risk_aware_portfolio_weekly", "v2")
        constraints = self.require_table("risk_aware_portfolio_constraint_audit", "v2")
        forecasts = self.require_table("risk_aware_portfolio_forecasts", "v2")
        summary = self.require_table("risk_aware_portfolio_summary", "v2")
        bootstrap = self.require_table("risk_aware_portfolio_bootstrap", "v2")
        if any(frame.empty for frame in (
            protocol, weekly, constraints, forecasts, summary, bootstrap,
        )):
            return
        row = protocol.iloc[0]
        self.add(
            "risk_aware_protocol_lagged_and_locked",
            not bool(row["historical_oos_used_for_selection"])
            and not bool(row["factor_covariance_uses_current_or_future_return"])
            and not bool(row["specific_variance_uses_current_or_future_residual"])
            and not bool(row["candidate_set_uses_future_return"])
            and str(row["source_experiment"]) == "multi_factor_risk_attribution_v1",
            "lagged covariance/specific risk; outcome-independent candidate set",
        )
        counts = weekly.groupby(
            ["model", "alpha_pipeline", "scenario"]
        )["asof_date"].nunique()
        self.add(
            "risk_aware_paired_dates",
            counts.nunique() == 1 and int(counts.min()) == 293,
            f"dates={counts.min()}-{counts.max()}, strategies={len(counts)}",
        )
        violation_columns = [
            "stock_weight_violation", "size_bound_violation",
            "industry_bound_violation", "turnover_bound_violation",
            "alpha_utility_shortfall",
        ]
        maximum_violation = float(
            constraints[violation_columns].to_numpy(dtype="float64").max()
        )
        relaxation = float(
            weekly.loc[weekly["scenario"].eq("risk_aware_full"),
                       "turnover_limit_relaxed"].astype(bool).mean()
        )
        self.add(
            "risk_aware_constraints_hold",
            maximum_violation < 1e-6 and relaxation == 0.0,
            f"maximum_violation={maximum_violation:.3e}, "
            f"turnover_relaxation_fraction={relaxation:.2%}",
        )
        target = summary.loc[summary["scenario"].eq("risk_aware_target_vol")]
        target_error = float(
            (target["mean_forecast_annualized_volatility"] - 0.20).abs().max()
        )
        self.add(
            "target_volatility_forecast_calibrated",
            target_error < 0.005 and float(target["mean_risky_weight"].max()) <= 1.0,
            f"maximum_mean_target_error={target_error:.3%}, no_leverage=True",
        )
        hgb = bootstrap.loc[
            bootstrap["model"].eq("hgb_shallow15_locked")
            & bootstrap["alpha_pipeline"].eq("raw")
            & bootstrap["scenario"].eq("risk_aware_full")
        ]
        te = hgb.loc[hgb["metric"].eq("forecast_tracking_error")].iloc[0]
        returns = hgb.loc[
            hgb["metric"].eq("net_return_annualized_arithmetic")
        ].iloc[0]
        self.add(
            "risk_aware_te_reduction_confirmed",
            float(te["ci_97_5pct"]) < 0.0,
            f"delta={float(te['point_estimate']):.3%}, "
            f"95%CI=[{float(te['ci_2_5pct']):.3%},{float(te['ci_97_5pct']):.3%}]",
        )
        self.add(
            "risk_aware_return_cost_detected",
            float(returns["ci_97_5pct"]) < 0.0,
            f"delta={float(returns['point_estimate']):.3%}, "
            f"95%CI=[{float(returns['ci_2_5pct']):.3%},"
            f"{float(returns['ci_97_5pct']):.3%}]",
        )
        self.warn(
            "risk_aware_policy_rejected",
            "The v2-registered minimum-active-variance policy reduces ex-ante tracking error but "
            "causes a statistically negative return delta; retain as a failed experiment.",
        )
        hgb_summary = summary.loc[
            summary["model"].eq("hgb_shallow15_locked")
            & summary["alpha_pipeline"].eq("raw")
        ].set_index("scenario")
        self.findings.update({
            "risk_aware_v1_policy_adopted": False,
            "risk_aware_hgb_raw_te_before": float(hgb_summary.loc[
                "source_risk_constrained", "mean_forecast_annualized_tracking_error"
            ]),
            "risk_aware_hgb_raw_te_after": float(hgb_summary.loc[
                "risk_aware_full", "mean_forecast_annualized_tracking_error"
            ]),
            "risk_aware_hgb_raw_annualized_return": float(hgb_summary.loc[
                "risk_aware_full", "annualized_return"
            ]),
            "risk_aware_hgb_raw_return_delta_ci_high": float(returns["ci_97_5pct"]),
        })

    def audit_model_benchmark(self) -> None:
        selection = self.require_table("model_benchmark_selection", "v1")
        summary = self.require_table("model_benchmark_oos_summary", "v1")
        annual = self.require_table("model_benchmark_oos_annual", "v1")
        pairwise = self.require_table("model_benchmark_winner_pairwise", "v1")
        availability = self.require_table("model_benchmark_availability", "v1")
        if any(frame.empty for frame in (selection, summary, annual, pairwise, availability)):
            return
        winner = str(selection.iloc[0]["selected_candidate"])
        winner_family = str(selection.iloc[0]["selected_family"])
        no_oos_selection = not bool(selection.iloc[0]["historical_oos_used_for_selection"])
        self.add("model_benchmark_excludes_historical_oos_selection", no_oos_selection,
                 f"winner={winner}")
        winner_annual = annual.loc[annual["candidate"].eq(winner)]
        self.add(
            "model_winner_has_annual_oos_results", not winner_annual.empty,
            f"winner={winner}, annual_rows={len(winner_annual)}",
        )
        if winner_annual.empty:
            return
        self.add("model_winner_rank_ic_positive_each_oos_year",
                 bool(winner_annual["mean_rank_ic"].gt(0).all()),
                 f"minimum_annual_rank_ic={winner_annual['mean_rank_ic'].min():.4f}")
        preprocessing = self.require_table("factor_preprocessing_oos_summary", "v1")
        if not preprocessing.empty:
            ridge = summary.loc[summary["family"].eq("ridge")].iloc[0]
            baseline = preprocessing.loc[
                preprocessing["pipeline"].eq("selected_quantile_zscore")
            ].iloc[0]
            for metric in ("mean_rank_ic", "portfolio_annualized_return"):
                left, right = float(ridge[metric]), float(baseline[metric])
                self.add(
                    f"benchmark_ridge_reproduces_preprocessing:{metric}",
                    _close(left, right, 1e-9),
                    f"benchmark={left:.10f}, preprocessing_oos={right:.10f}",
                )
        ridge_rows = summary.loc[summary["family"].eq("ridge")]
        self.add(
            "model_benchmark_has_ridge_reference", len(ridge_rows) == 1,
            f"ridge_candidates={ridge_rows['candidate'].tolist()}",
        )
        ridge_return = ridge_ic = None
        if len(ridge_rows) == 1:
            ridge_reference = str(ridge_rows.iloc[0]["candidate"])
            for metric, check_name in (
                ("net_return_annualized_arithmetic", "model_winner_return_pairwise_vs_ridge"),
                ("rank_ic", "model_winner_rank_ic_pairwise_vs_ridge"),
            ):
                matches = pairwise.loc[
                    pairwise["candidate"].eq(winner)
                    & pairwise["reference"].eq(ridge_reference)
                    & pairwise["metric"].eq(metric)
                ]
                self.add(
                    check_name, len(matches) == 1,
                    f"candidate={winner}, reference={ridge_reference}, metric={metric}, "
                    f"rows={len(matches)}",
                )
                if len(matches) == 1:
                    if metric == "net_return_annualized_arithmetic":
                        ridge_return = matches.iloc[0]
                    else:
                        ridge_ic = matches.iloc[0]
        if ridge_return is not None:
            return_low = float(ridge_return["ci_2_5pct"])
            return_high = float(ridge_return["ci_97_5pct"])
            return_inference = (
                "positive" if return_low > 0
                else "negative" if return_high < 0
                else "inconclusive"
            )
            self.add(
                "model_winner_return_interval_valid",
                np.isfinite([return_low, return_high]).all()
                and return_low <= return_high,
                f"95% CI=[{return_low:.2%}, {return_high:.2%}]; "
                f"inference={return_inference}",
            )
        if ridge_ic is not None:
            self.add("model_winner_rank_ic_increment_not_overstated",
                     float(ridge_ic["ci_2_5pct"]) <= 0 <= float(ridge_ic["ci_97_5pct"]),
                     f"95% CI=[{ridge_ic['ci_2_5pct']:.4f}, "
                     f"{ridge_ic['ci_97_5pct']:.4f}]")
        if winner_family == "random_forest":
            hgb_rows = summary.loc[summary["family"].eq("hgb")]
            hgb_reference = (
                str(hgb_rows.iloc[0]["candidate"]) if len(hgb_rows) == 1 else None
            )
            matches = pairwise.loc[
                pairwise["candidate"].eq(winner)
                & pairwise["reference"].eq(hgb_reference)
                & pairwise["metric"].eq("net_return_annualized_arithmetic")
            ]
            self.add(
                "model_winner_return_pairwise_vs_hgb", len(matches) == 1,
                f"candidate={winner}, reference={hgb_reference}, rows={len(matches)}",
            )
            if len(matches) == 1:
                hgb_return = matches.iloc[0]
                self.add("random_forest_not_claimed_superior_to_hgb",
                         float(hgb_return["ci_2_5pct"]) <= 0
                         <= float(hgb_return["ci_97_5pct"]),
                         f"95% CI=[{hgb_return['ci_2_5pct']:.2%}, "
                         f"{hgb_return['ci_97_5pct']:.2%}]")
        else:
            self.add(
                "random_forest_not_claimed_superior_to_hgb", True,
                f"validation-selected winner={winner} ({winner_family}); "
                "random forest is not the selected winner",
            )
        unavailable = availability.loc[availability["status"].ne("executed"), "model"].tolist()
        if unavailable:
            self.warn("model_dependencies_unavailable", f"not executed={unavailable}")
        winner_rows = summary.loc[summary["candidate"].eq(winner)]
        self.add(
            "model_winner_has_oos_summary", len(winner_rows) == 1,
            f"winner={winner}, summary_rows={len(winner_rows)}",
        )
        if len(winner_rows) != 1:
            return
        winner_summary = winner_rows.iloc[0]
        self.findings.update({
            "model_benchmark_validation_winner": winner,
            "model_winner_oos_mean_rank_ic": float(winner_summary["mean_rank_ic"]),
            "model_winner_oos_annualized_return": float(
                winner_summary["portfolio_annualized_return"]
            ),
            "model_winner_minus_ridge_return_ci_low": (
                float(ridge_return["ci_2_5pct"]) if ridge_return is not None else float("nan")
            ),
            "model_winner_minus_ridge_return_ci_high": (
                float(ridge_return["ci_97_5pct"]) if ridge_return is not None else float("nan")
            ),
        })

    def audit_rf_hyperparameter_stability(self) -> None:
        summary = self.require_table("rf_hyperparameter_validation_summary", "v1")
        protocol = self.require_table("rf_hyperparameter_protocol", "v1")
        benchmark = self.require_table("model_benchmark_validation_grid", "v1")
        if any(frame.empty for frame in (summary, protocol, benchmark)):
            return
        record = protocol.iloc[0]
        neighbors = summary.loc[~summary["is_baseline"].astype(bool)]
        actual_fraction = float(neighbors["passes_robustness_floor"].astype(bool).mean())
        expected_outcome = actual_fraction >= float(record["minimum_neighbor_pass_fraction"])
        self.add(
            "rf_stability_excludes_historical_oos",
            not bool(record["historical_oos_accessed"]),
            f"evaluation_period={record['evaluation_period']}",
        )
        self.add(
            "rf_stability_protocol_arithmetic",
            int(record["candidate_count"]) == len(summary)
            and int(record["neighbor_count"]) == len(neighbors)
            and _close(float(record["neighbor_pass_fraction"]), actual_fraction, 1e-12)
            and bool(record["local_robustness_pass"]) == expected_outcome,
            f"neighbors={len(neighbors)}, pass_fraction={actual_fraction:.1%}",
        )
        baseline = summary.loc[summary["is_baseline"].astype(bool)].iloc[0]
        original = benchmark.loc[benchmark["candidate"].eq("random_forest_depth6")].iloc[0]
        self.add(
            "rf_stability_reproduces_frozen_validation_baseline",
            _close(float(baseline["mean_rank_ic"]), float(original["mean_rank_ic"]), 1e-12)
            and _close(
                float(baseline["portfolio_annualized_return"]),
                float(original["portfolio_annualized_return"]), 1e-12,
            ),
            f"RankIC={baseline['mean_rank_ic']:.6f}, "
            f"annual_return={baseline['portfolio_annualized_return']:.2%}",
        )
        self.add(
            "rf_stability_does_not_promote_posthoc_winner",
            bool(record["best_candidate_is_not_promoted"]),
            f"local_robustness_pass={bool(record['local_robustness_pass'])}",
        )
        self.findings.update({
            "rf_neighbor_pass_fraction": actual_fraction,
            "rf_local_robustness_pass": bool(record["local_robustness_pass"]),
            "rf_stability_minimum_neighbor_rank_ic": float(neighbors["mean_rank_ic"].min()),
            "rf_stability_minimum_neighbor_sharpe": float(
                neighbors["portfolio_sharpe_zero_rate"].min()
            ),
        })

    def audit_rf_execution_forensics(self) -> None:
        validation = self.require_table("rf_forensic_summary", "v1")
        holdings = self.require_table("rf_forensic_holdings_summary", "v1")
        labels = self.require_table("rf_forensic_label_audit", "v1")
        oos = self.require_table("rf_oos_execution_summary", "v1")
        diagnostics = self.require_table("rf_oos_execution_diagnostics", "v1")
        if any(frame.empty for frame in (validation, holdings, labels, oos, diagnostics)):
            return
        validation_ideal = validation.loc[
            validation["scenario"].eq("ideal_close_to_close")
        ].iloc[0]
        validation_constrained = validation.loc[
            validation["scenario"].eq("t_plus_1_limit_and_suspension_constrained")
        ].iloc[0]
        oos_ideal = oos.loc[oos["scenario"].eq("ideal_close_to_close")].iloc[0]
        oos_constrained = oos.loc[
            oos["scenario"].eq("t_plus_1_limit_and_suspension_constrained")
        ].iloc[0]
        self.add(
            "rf_forensic_sampled_labels_rebuild_exactly",
            len(labels) >= 250 and float(labels["absolute_error"].max()) <= 1e-12,
            f"rows={len(labels)}, max_abs_error={labels['absolute_error'].max():.3e}",
        )
        self.add(
            "rf_validation_execution_bias_detected",
            float(validation_constrained["annualized_return"])
            < float(validation_ideal["annualized_return"]),
            f"ideal={validation_ideal['annualized_return']:.2%}, "
            f"constrained={validation_constrained['annualized_return']:.2%}",
        )
        self.add(
            "rf_oos_execution_comparison_uses_common_periods",
            int(oos_ideal["periods"]) == int(oos_constrained["periods"]),
            f"ideal={int(oos_ideal['periods'])}, constrained={int(oos_constrained['periods'])}",
        )
        self.add(
            "rf_oos_execution_bias_detected",
            float(oos_constrained["annualized_return"]) < float(oos_ideal["annualized_return"])
            and not bool(diagnostics.iloc[0]["historical_oos_used_for_selection"]),
            f"ideal={oos_ideal['annualized_return']:.2%}, "
            f"constrained={oos_constrained['annualized_return']:.2%}",
        )
        self.findings.update({
            "rf_validation_idealized_annualized_return": float(
                validation_ideal["annualized_return"]
            ),
            "rf_validation_execution_constrained_annualized_return": float(
                validation_constrained["annualized_return"]
            ),
            "rf_oos_idealized_annualized_return": float(oos_ideal["annualized_return"]),
            "rf_oos_execution_constrained_annualized_return": float(
                oos_constrained["annualized_return"]
            ),
            "rf_oos_execution_constrained_sharpe": float(
                oos_constrained["sharpe_zero_rate"]
            ),
            "rf_validation_mean_size_percentile": float(
                holdings.iloc[0]["mean_size_percentile"]
            ),
            "rf_validation_mean_blocked_buy_weight": float(
                holdings.iloc[0]["mean_blocked_buy_weight"]
            ),
        })

    def audit_rf_daily_retry_execution(self) -> None:
        """Gate the share-order retry engine and its non-overlapping clock."""
        summaries: dict[str, tuple[pd.Series, pd.Series]] = {}
        for split in ("validation", "historical_oos"):
            constrained_daily = self.require_table(
                "rf_daily_retry_daily", f"{split}_v1"
            )
            ideal_daily = self.require_table("rf_daily_ideal_daily", f"{split}_v1")
            constrained_summary = self.require_table(
                "rf_daily_retry_summary", f"{split}_v1"
            )
            ideal_summary = self.require_table(
                "rf_daily_ideal_summary", f"{split}_v1"
            )
            timing = self.require_table("rf_daily_timing_audit", f"{split}_v1")
            if any(frame.empty for frame in (
                constrained_daily, ideal_daily, constrained_summary, ideal_summary, timing
            )):
                continue
            accounting_error = max(
                float((constrained_daily["accounting_weight_sum"] - 1.0).abs().max()),
                float((ideal_daily["accounting_weight_sum"] - 1.0).abs().max()),
            )
            return_error = max(
                float((
                    constrained_daily["net_return"]
                    - constrained_daily["gross_return"]
                    + constrained_daily["transaction_cost"]
                ).abs().max()),
                float((
                    ideal_daily["net_return"] - ideal_daily["gross_return"]
                    + ideal_daily["transaction_cost"]
                ).abs().max()),
            )
            non_rebalance_ideal_turnover = float(
                ideal_daily.loc[~ideal_daily["new_target"].astype(bool), "turnover"].max()
            )
            constrained = constrained_summary.iloc[0]
            ideal = ideal_summary.iloc[0]
            self.add(
                f"rf_daily_{split}_accounting_identity", accounting_error < 1e-10,
                f"max_abs_error={accounting_error:.3e}",
            )
            self.add(
                f"rf_daily_{split}_return_identity", return_error < 1e-10,
                f"max_abs_error={return_error:.3e}",
            )
            self.add(
                f"rf_daily_{split}_no_hidden_daily_rebalance",
                non_rebalance_ideal_turnover < 1e-12,
                f"ideal_non_rebalance_max_turnover={non_rebalance_ideal_turnover:.3e}",
            )
            self.add(
                f"rf_daily_{split}_execution_haircut_detected",
                float(constrained["annualized_return"]) < float(ideal["annualized_return"]),
                f"ideal={ideal['annualized_return']:.2%}, "
                f"constrained={constrained['annualized_return']:.2%}",
            )
            self.add(
                f"rf_daily_{split}_overlap_removed",
                int(timing.iloc[0]["duplicated_horizon_days"]) > 0,
                f"duplicated_horizon_days={int(timing.iloc[0]['duplicated_horizon_days'])}",
            )
            summaries[split] = (constrained, ideal)
        if "validation" in summaries and "historical_oos" in summaries:
            validation, validation_ideal = summaries["validation"]
            oos, oos_ideal = summaries["historical_oos"]
            self.findings.update({
                "rf_daily_validation_ideal_annualized_return": float(
                    validation_ideal["annualized_return"]
                ),
                "rf_daily_validation_constrained_annualized_return": float(
                    validation["annualized_return"]
                ),
                "rf_daily_oos_ideal_annualized_return": float(
                    oos_ideal["annualized_return"]
                ),
                "rf_daily_oos_constrained_annualized_return": float(
                    oos["annualized_return"]
                ),
                "rf_daily_oos_constrained_sharpe": float(oos["sharpe_zero_rate"]),
                "rf_daily_oos_constrained_maximum_drawdown": float(
                    oos["maximum_drawdown"]
                ),
            })

    def audit_final_research_extensions(self) -> None:
        """Gate the final registry, descriptive, parameter, and cost diagnostics."""
        registry = self.require_table("factor_registry", "v1")
        distribution = self.require_table("factor_distribution_summary", "v1")
        rolling = self.require_table("factor_rolling_correlation_stability", "v1")
        hgb_protocol = self.require_table("hgb_hyperparameter_protocol", "v1")
        hgb_surface = self.require_table("hgb_hyperparameter_summary", "v1")
        sensitivity_protocol = self.require_table("portfolio_sensitivity_protocol", "v1")
        sensitivity = self.require_table("portfolio_sensitivity_summary", "v1")
        costs = self.require_table("execution_cost_stress", "v1")
        if any(frame.empty for frame in (
            registry, distribution, rolling, hgb_protocol, hgb_surface,
            sensitivity_protocol, sensitivity, costs,
        )):
            return
        expected_factors = set(_load_yaml("preprocessing.yaml")["factor_columns"])
        registered = set(registry["factor"].astype(str))
        directions_complete = registry["locked_training_direction"].isin([-1, 1]).all()
        self.add(
            "factor_registry_complete_and_versioned",
            registered == expected_factors and len(registry) == 15
            and registry["registry_version"].nunique() == 1 and directions_complete,
            f"factors={len(registry)}, version={registry['registry_version'].iloc[0]}",
        )
        expected_pairs = len(expected_factors) * (len(expected_factors) - 1) // 2
        self.add(
            "factor_visual_diagnostics_complete",
            set(distribution["factor"].astype(str)) == expected_factors
            and len(rolling) == expected_pairs
            and rolling["rolling_windows"].gt(0).all(),
            f"distributions={len(distribution)}, rolling_pairs={len(rolling)}, "
            f"expected_pairs={expected_pairs}",
        )
        hgb_row = hgb_protocol.iloc[0]
        coordinates = hgb_surface[[
            "max_leaf_nodes", "l2_regularization"
        ]].drop_duplicates()
        self.add(
            "hgb_surface_validation_only_and_complete",
            not bool(hgb_row["historical_oos_accessed"])
            and bool(hgb_row["best_candidate_is_not_promoted"])
            and int(hgb_row["surface_points"]) == 9
            and len(coordinates) == 9,
            f"surface_points={len(coordinates)}, evaluation={hgb_row['evaluation_period']}",
        )
        protocol_row = sensitivity_protocol.iloc[0]
        counts = sensitivity.groupby("split")["periods"].nunique()
        variants = sensitivity.groupby("split").size()
        self.add(
            "portfolio_sensitivity_common_date_and_no_reselection",
            not bool(protocol_row["historical_oos_used_for_selection"])
            and bool(protocol_row["common_date_comparison"])
            and counts.eq(1).all() and variants.eq(8).all(),
            f"variants={variants.to_dict()}, performance_period_counts={counts.to_dict()}",
        )
        expected_costs = {0.0, 20.0, 50.0, 100.0}
        monotone = True
        for _, group in costs.sort_values("cost_bps").groupby(
            ["model", "alpha_pipeline"], sort=False
        ):
            monotone &= bool(group["annualized_return"].is_monotonic_decreasing)
        self.add(
            "execution_cost_stress_complete_and_monotone",
            set(costs["cost_bps"].astype(float)) == expected_costs
            and monotone
            and costs["stress_method"].eq("fixed_realized_fill_path_reprice").all(),
            f"costs={sorted(costs['cost_bps'].unique().tolist())}, monotone={monotone}",
        )
        self.findings.update({
            "hgb_validation_surface_rank_ic_min": float(hgb_surface["mean_rank_ic"].min()),
            "hgb_validation_surface_rank_ic_max": float(hgb_surface["mean_rank_ic"].max()),
            "factor_registry_count": len(registry),
        })

    def audit_research_conclusions(self) -> None:
        """Gate the decision hierarchy, multiplicity control, and lifecycle boundary."""
        decisions = self.require_table("research_conclusion_decisions", "v2")
        multiplicity = self.require_table("research_factor_multiplicity", "v2")
        lifecycle = self.require_table("research_lifecycle_audit", "v2")
        registry = self.require_table("factor_registry", "v1")
        if any(frame.empty for frame in (decisions, multiplicity, lifecycle, registry)):
            return
        expected = {
            "neutralized_factor_signal_exists": "SUPPORTED_AS_DIAGNOSTIC_EVIDENCE",
            "hgb_adds_stable_increment_over_ridge": "NOT_ESTABLISHED",
            "ml_adds_value_over_style_matched_baseline": "MIXED_LOCAL_EVIDENCE",
            "executable_backtest_survives_costs": "SUPPORTED_HISTORICALLY",
            "pure_stock_selection_alpha_is_significant": "NOT_SUPPORTED",
            "ready_for_live_capital": "NO",
        }
        observed = decisions.set_index("question")["verdict"].to_dict()
        self.add(
            "conclusion_decision_hierarchy_complete",
            all(observed.get(question) == verdict for question, verdict in expected.items()),
            f"verdicts={observed}",
        )
        recomputed = benjamini_hochberg(
            multiplicity["one_sided_probability_nonpositive"]
        )
        maximum_q_error = float(
            (recomputed - multiplicity["bh_fdr_adjusted_p"]).abs().max()
        )
        fdr_count = int(multiplicity["survives_bh_fdr"].astype(bool).sum())
        self.add(
            "single_factor_multiple_testing_controlled",
            len(multiplicity) == 15 and maximum_q_error < 1e-12,
            f"factors={len(multiplicity)}, bh_fdr_5pct_pass={fdr_count}, "
            f"maximum_q_error={maximum_q_error:.3e}",
        )
        statuses = lifecycle.set_index("stage")["status"].to_dict()
        self.add(
            "research_and_deployment_scope_separated",
            statuses.get("reproducibility_and_governance") == "COMPLETE"
            and statuses.get("paper_live_monitoring") == "NEXT_STAGE",
            "historical research chain is complete; paper/live evidence remains a separate gate",
        )
        price_convention = _load_yaml("factor_registry.yaml")["price_convention"]
        table_conventions = set(registry["price_convention"].astype(str))
        expected_convention = "baostock_unadjusted_ohlc_and_ex_right_reference_pct_chg"
        self.add(
            "price_and_return_convention_explicit",
            price_convention == expected_convention
            and table_conventions == {expected_convention},
            f"price_convention={price_convention}",
        )
        report = (ROOT / "reports" / "final_research_report.md").read_text(encoding="utf-8")
        required_phrases = (
            "不支持显著纯选股 Alpha",
            "historical OOS",
            "不是实盘",
            "BH-FDR",
        )
        missing_phrases = [phrase for phrase in required_phrases if phrase not in report]
        self.add(
            "final_report_states_decision_boundaries",
            not missing_phrases,
            f"missing_phrases={missing_phrases}",
        )
        self.findings.update({
            "neutral_factor_spreads_surviving_bh_fdr_5pct": fdr_count,
            "research_backtest_chain_complete": True,
            "ready_for_live_capital": False,
        })

    def audit_research_governance(self) -> None:
        """Gate evidence classification, candidate freeze, and lockbox consumption state."""
        protocol = self.require_table("research_governance_protocol", "v1")
        evidence = self.require_table("research_evidence_registry", "v1")
        freeze = self.require_table("research_candidate_freeze", "v1")
        lockbox = self.require_table("research_lockbox_registry", "v1")
        paper = self.require_table("research_paper_trading_registry", "v1")
        hypotheses = self.require_table("research_hypothesis_registry", "v1")
        checks = self.require_table("research_governance_checks", "v1")
        access = self.require_table("research_access_log_snapshot", "v1")
        if any(frame.empty for frame in (
            protocol, evidence, freeze, lockbox, paper, hypotheses, checks, access
        )):
            return
        roles = set(evidence["evidence_class"].astype(str))
        self.add(
            "evidence_classes_complete",
            {"training", "validation_selection", "historical_oos",
             "future_lockbox", "paper_trading"}.issubset(roles),
            f"classes={sorted(roles)}",
        )
        historical = evidence.loc[evidence["evidence_class"].eq("historical_oos")].iloc[0]
        self.add(
            "historical_oos_consumed_and_not_selectable",
            bool(historical["consumed_at_freeze"]) and not bool(historical["selection_allowed"]),
            f"consumed={historical['consumed_at_freeze']}, "
            f"selection_allowed={historical['selection_allowed']}",
        )
        current_lockbox = lockbox.iloc[0]
        self.add(
            "future_lockbox_is_sealed_and_unconsumed",
            current_lockbox["state"] in {"SEALED_EMPTY", "COLLECTING", "READY_SEALED"}
            and not bool(current_lockbox["consumed"]),
            f"state={current_lockbox['state']}, consumed={current_lockbox['consumed']}",
        )
        current_paper = paper.iloc[0]
        self.add(
            "paper_tracking_is_forward_only",
            not bool(current_paper["retroactive_backfill_allowed"]),
            f"state={current_paper['state']}, observations={current_paper['recorded_observations']}",
        )
        self.add(
            "candidate_freeze_is_commit_anchored",
            freeze["freeze_commit"].astype(str).str.len().eq(40).all()
            and freeze["working_tree_dirty_at_freeze"].eq(False).all(),
            f"commit={freeze.iloc[0]['freeze_commit']}, "
            f"dirty={freeze.iloc[0]['working_tree_dirty_at_freeze']}",
        )
        self.add(
            "governance_checks_all_pass",
            checks["status"].eq("PASS").all(),
            f"checks={len(checks)}, failures={int(checks['status'].ne('PASS').sum())}",
        )
        self.add(
            "future_hypothesis_family_frozen",
            len(hypotheses) == 4
            and hypotheses["frozen_before_future_data"].astype(bool).all()
            and hypotheses["multiplicity_control"].eq("benjamini_hochberg").all(),
            f"hypotheses={len(hypotheses)}",
        )
        from quant_research.research_governance import load_config, verify_access_log
        governance_cfg = load_config()
        live_events = verify_access_log(ROOT / governance_cfg["access_log"]["path"])
        snapshot_events = access.sort_values("sequence").to_dict(orient="records")
        snapshot_is_prefix = len(live_events) >= len(snapshot_events) and all(
            live_events[index]["event_hash"] == event["event_hash"]
            for index, event in enumerate(snapshot_events)
        )
        self.add(
            "access_log_hash_chain_extends_frozen_snapshot",
            snapshot_is_prefix,
            f"live_events={len(live_events)}, frozen_prefix_events={len(access)}",
        )
        self.findings.update({
            "future_lockbox_state": str(current_lockbox["state"]),
            "future_lockbox_consumed": bool(current_lockbox["consumed"]),
            "paper_trading_state": str(current_paper["state"]),
        })

    def run(self) -> tuple[list[AuditCheck], dict]:
        self.audit_configs()
        self.audit_frozen_experiments()
        self.audit_coverage()
        self.audit_predictions("model_validation_predictions", "ridge_v1", "ridge_validation", 2016, 2020)
        self.audit_predictions("model_oos_predictions", "ridge_v1", "ridge_oos", 2021, 2026)
        self.audit_predictions("model_challenger_oos_predictions", "hgb_v1", "challenger_oos", 2021, 2026)
        self.audit_predictions(
            "factor_preprocessing_oos_predictions", "v1",
            "factor_preprocessing_oos", 2021, 2026,
        )
        self.audit_selection_provenance()
        ridge = self.audit_weekly_identity(
            "turnover_buffer_oos_weekly", "v1", "ridge_buffered",
            lambda frame: (
                frame["selected_on_validation"].astype(bool)
                & frame["included_in_performance"].astype(bool)
            ),
        )
        challenger = self.audit_weekly_identity(
            "challenger_portfolio_oos_weekly", "v1", "challenger_buffered",
            lambda frame: frame["included_in_performance"].astype(bool),
        )
        self.audit_holdings("turnover_buffer_oos_holdings", "v1", "ridge")
        self.audit_holdings("challenger_portfolio_oos_holdings", "v1", "challenger")
        self.audit_paired_results(ridge, challenger)
        self.audit_lofo_consistency()
        self.audit_predictions(
            "label_horizon_oos_predictions", "v1", "label_horizon_oos", 2021, 2026,
        )
        self.audit_weekly_identity(
            "label_horizon_oos_portfolio", "v1", "label_horizon_selected",
            lambda frame: (
                frame["target_spec"].eq(
                    self.lake.read("label_horizon_selection", "v1").iloc[0][
                        "selected_target_spec"
                    ]
                ) & frame["included_in_performance"].astype(bool)
            ),
        )
        self.audit_label_horizon()
        self.audit_weekly_identity(
            "size_neutralization_oos_portfolio", "v1", "size_neutralization_raw",
            lambda frame: frame["pipeline"].eq("raw")
            & frame["included_in_performance"].astype(bool),
        )
        self.audit_weekly_identity(
            "size_neutralization_oos_portfolio", "v1", "size_neutralization_neutral",
            lambda frame: frame["pipeline"].eq("size_neutralized")
            & frame["included_in_performance"].astype(bool),
        )
        self.audit_size_neutralization()
        self.audit_factor_neutralization()
        self.audit_factor_tear_sheet()
        self.audit_capm_attribution()
        self.audit_style_factor_attribution()
        self.audit_model_portfolio_neutralization()
        self.audit_constrained_portfolio_neutralization()
        self.audit_constrained_portfolio_daily_execution()
        self.audit_institutional_execution()
        self.audit_unified_account_ledger()
        self.audit_statistical_inference()
        self.audit_incremental_alpha_test()
        self.audit_alpha_evidence_matrix()
        self.audit_engineering_quality()
        self.audit_paper_trading()
        self.audit_multi_factor_risk_attribution()
        self.audit_risk_aware_portfolio_optimization()
        self.audit_weekly_identity(
            "model_benchmark_oos_portfolio", "v1", "model_benchmark_winner",
            lambda frame: (
                frame["candidate"].eq(
                    self.lake.read("model_benchmark_selection", "v1").iloc[0][
                        "selected_candidate"
                    ]
                ) & frame["included_in_performance"].astype(bool)
            ),
        )
        self.audit_model_benchmark()
        self.audit_rf_hyperparameter_stability()
        self.audit_rf_execution_forensics()
        self.audit_rf_daily_retry_execution()
        self.audit_final_research_extensions()
        self.audit_research_conclusions()
        self.audit_research_governance()
        self.warn(
            "known_scope_limitations",
            "Industry neutralization uses monthly-sampled historical CSRC broad sections rather "
            "than Shenwan membership; there is no official historical CSI membership and no "
            "broker-calibrated market-impact model.",
        )
        return self.checks, self.findings


def write_reports(checks: list[AuditCheck], findings: dict,
                  md_path: Path = REPORT_MD, json_path: Path = REPORT_JSON) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated,
        "checks": [asdict(check) for check in checks],
        "findings": findings,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {status: sum(check.status == status for check in checks)
              for status in ("PASS", "WARN", "FAIL")}
    lines = [
        "# Research artifact audit",
        "",
        f"Generated at `{generated}`.",
        "",
        f"Summary: **{counts['PASS']} PASS / {counts['WARN']} WARN / {counts['FAIL']} FAIL**",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {check.status} | `{check.name}` | {check.detail.replace('|', '/')} |"
        for check in checks
    )
    lines.extend(["", "## Headline findings", ""])
    lines.extend(f"- `{name}`: {value}".rstrip() for name, value in findings.items())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit research outputs before publication")
    parser.add_argument("--no-write", action="store_true", help="run checks without updating reports")
    args = parser.parse_args()
    checks, findings = ResearchAuditor().run()
    if not args.no_write:
        write_reports(checks, findings)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    failures = [check for check in checks if check.status == "FAIL"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
