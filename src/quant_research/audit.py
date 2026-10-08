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

    def run(self) -> tuple[list[AuditCheck], dict]:
        self.audit_configs()
        self.audit_coverage()
        self.audit_predictions("model_validation_predictions", "ridge_v1", "ridge_validation", 2016, 2020)
        self.audit_predictions("model_oos_predictions", "ridge_v1", "ridge_oos", 2021, 2026)
        self.audit_predictions("model_challenger_oos_predictions", "hgb_v1", "challenger_oos", 2021, 2026)
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
