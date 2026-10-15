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
from quant_research.experiment_registry import ExperimentRegistry, load_specs
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
            )
        invalidation_path = ROOT / "config" / "experiment_invalidations.yaml"
        if invalidation_path.exists():
            with invalidation_path.open(encoding="utf-8") as handle:
                invalidations = yaml.safe_load(handle).get("invalidations", {})
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
        self.add("challenger_increment_not_overstated",
                 float(model_increment["ci_2_5pct"]) <= 0 <= float(model_increment["ci_97_5pct"]),
                 f"95% CI=[{model_increment['ci_2_5pct']:.2%}, {model_increment['ci_97_5pct']:.2%}]; "
                 "report as an economic signal, not proven superiority")

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

    def audit_model_benchmark(self) -> None:
        selection = self.require_table("model_benchmark_selection", "v1")
        summary = self.require_table("model_benchmark_oos_summary", "v1")
        annual = self.require_table("model_benchmark_oos_annual", "v1")
        pairwise = self.require_table("model_benchmark_winner_pairwise", "v1")
        availability = self.require_table("model_benchmark_availability", "v1")
        if any(frame.empty for frame in (selection, summary, annual, pairwise, availability)):
            return
        winner = str(selection.iloc[0]["selected_candidate"])
        no_oos_selection = not bool(selection.iloc[0]["historical_oos_used_for_selection"])
        self.add("model_benchmark_excludes_historical_oos_selection", no_oos_selection,
                 f"winner={winner}")
        winner_annual = annual.loc[annual["candidate"].eq(winner)]
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
        ridge_return = pairwise.loc[
            pairwise["reference"].eq("ridge_1")
            & pairwise["metric"].eq("net_return_annualized_arithmetic")
        ].iloc[0]
        ridge_ic = pairwise.loc[
            pairwise["reference"].eq("ridge_1")
            & pairwise["metric"].eq("rank_ic")
        ].iloc[0]
        hgb_return = pairwise.loc[
            pairwise["reference"].str.startswith("hgb_")
            & pairwise["metric"].eq("net_return_annualized_arithmetic")
        ].iloc[0]
        self.add("model_winner_return_increment_vs_ridge",
                 float(ridge_return["ci_2_5pct"]) > 0,
                 f"95% CI=[{ridge_return['ci_2_5pct']:.2%}, {ridge_return['ci_97_5pct']:.2%}]")
        self.add("model_winner_rank_ic_increment_not_overstated",
                 float(ridge_ic["ci_2_5pct"]) <= 0 <= float(ridge_ic["ci_97_5pct"]),
                 f"95% CI=[{ridge_ic['ci_2_5pct']:.4f}, {ridge_ic['ci_97_5pct']:.4f}]")
        self.add("random_forest_not_claimed_superior_to_hgb",
                 float(hgb_return["ci_2_5pct"]) <= 0 <= float(hgb_return["ci_97_5pct"]),
                 f"95% CI=[{hgb_return['ci_2_5pct']:.2%}, {hgb_return['ci_97_5pct']:.2%}]")
        unavailable = availability.loc[availability["status"].ne("executed"), "model"].tolist()
        if unavailable:
            self.warn("model_dependencies_unavailable", f"not executed={unavailable}")
        winner_summary = summary.loc[summary["candidate"].eq(winner)].iloc[0]
        self.findings.update({
            "model_benchmark_validation_winner": winner,
            "model_winner_oos_mean_rank_ic": float(winner_summary["mean_rank_ic"]),
            "model_winner_oos_annualized_return": float(
                winner_summary["portfolio_annualized_return"]
            ),
            "model_winner_minus_ridge_return_ci_low": float(ridge_return["ci_2_5pct"]),
            "model_winner_minus_ridge_return_ci_high": float(ridge_return["ci_97_5pct"]),
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
        self.warn(
            "known_scope_limitations",
            "No point-in-time industry neutralization, no official historical CSI membership, "
            "and no broker-calibrated market-impact model.",
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
    lines.extend(f"- `{name}`: {value}" for name, value in findings.items())
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
