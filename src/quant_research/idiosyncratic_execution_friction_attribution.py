"""Post-hoc, validation-only attribution of share-execution friction.

The experiment replays frozen targets while relaxing exactly one execution
constraint at a time.  It diagnoses implementation shortfall; it cannot select
a model, alter the frozen portfolio, or provide independent alpha evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.idiosyncratic_share_execution_validation import (
    _reliable_rf_summary,
    selected_targets,
)
from quant_research.ingest import ROOT, Lake
from quant_research.institutional_execution import _summaries, simulate_execution
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "idiosyncratic_execution_friction_attribution.yaml"
TABLE_ROOT = ROOT / "reports" / "tables"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if bool(cfg.get("historical_oos_evaluation_enabled")):
        raise ValueError("friction attribution cannot inspect historical OOS")
    if bool(cfg.get("future_lockbox_access_allowed")):
        raise ValueError("friction attribution cannot inspect the future lockbox")
    if bool(cfg.get("claims_independent_evidence")):
        raise ValueError("post-hoc friction attribution is not independent evidence")
    if str(cfg.get("research_selection_role")) != "diagnostic":
        raise ValueError("friction attribution must remain a post-hoc diagnostic")
    expected = {
        "trade_flags", "volume_participation", "order_quantity_rules", "t_plus_one",
    }
    scenarios = cfg["scenarios"]
    if {str(item["relaxed_constraint"]) for item in scenarios} != expected:
        raise ValueError("exactly four one-factor-at-a-time relaxations are required")
    toggle_by_constraint = {
        "trade_flags": "enforce_trade_flags",
        "volume_participation": "enforce_volume_participation",
        "order_quantity_rules": "enforce_order_quantity_rules",
        "t_plus_one": "enforce_t_plus_one",
    }
    for item in scenarios:
        relaxed = toggle_by_constraint[str(item["relaxed_constraint"])]
        toggles = {name: bool(item[name]) for name in toggle_by_constraint.values()}
        if toggles[relaxed] or sum(not value for value in toggles.values()) != 1:
            raise ValueError(f"{item['scenario_id']} must relax exactly {relaxed}")
    waterfall = cfg["waterfall_scenarios"]
    expected_waterfall = [
        (False, False, True, True),
        (False, False, False, True),
    ]
    observed_waterfall = [tuple(bool(item[name]) for name in (
        "enforce_trade_flags", "enforce_volume_participation",
        "enforce_order_quantity_rules", "enforce_t_plus_one",
    )) for item in waterfall]
    if observed_waterfall != expected_waterfall:
        raise ValueError("waterfall scenarios must follow the locked conditional sequence")
    contract = cfg["interpretation_contract"]
    if not all(bool(contract[name]) for name in (
        "one_factor_at_a_time_effects_are_not_additive",
        "waterfall_effects_are_order_dependent",
        "upper_bound_is_not_implementable",
        "may_not_promote_or_select_model",
        "may_not_create_paper_observations",
    )):
        raise ValueError("all diagnostic interpretation safeguards must be enabled")


def _decorate_summary(
    result: Any, rates: pd.DataFrame, execution_cfg: dict[str, Any],
    *, path_id: str, scenario_id: str, relaxed_constraint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    periods = int(execution_cfg["inference"]["annualization_periods"])
    summary, invariant = _summaries(result, periods)
    sharpe, annual_excess, missing = _reliable_rf_summary(result.daily, rates, execution_cfg)
    summary.update({
        "path_id": path_id,
        "scenario_id": scenario_id,
        "relaxed_constraint": relaxed_constraint,
        "sharpe_reliable_risk_free": sharpe,
        "annualized_excess_return_arithmetic": annual_excess,
        "risk_free_missing_dates": missing,
        "p95_target_tracking_l1": float(result.daily["target_tracking_l1"].quantile(0.95)),
        "mean_cash_weight": float(result.daily["cash_weight"].mean()),
    })
    invariant.update({
        "path_id": path_id,
        "scenario_id": scenario_id,
        "relaxed_constraint": relaxed_constraint,
    })
    return summary, invariant


def _existing_inputs(lake: Lake, cfg: dict[str, Any]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    key = str(cfg["artifact_key"])
    targets = selected_targets(lake, yaml.safe_load(
        (ROOT / "config" / "idiosyncratic_share_execution_validation.yaml").read_text(
            encoding="utf-8"
        )
    ))
    market = lake.read("idiosyncratic_execution_market", key)
    rates = lake.read("idiosyncratic_execution_rf_daily", key)
    base_daily = lake.read("idiosyncratic_execution_daily", key)
    upper_daily = lake.read("idiosyncratic_execution_upper_bound_daily", key)
    if any(frame.empty for frame in (targets, market, rates, base_daily, upper_daily)):
        raise RuntimeError("frozen Phase 12.2 inputs are incomplete")
    return targets, market, rates, base_daily, upper_daily


def _existing_summary_rows(lake: Lake, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = str(cfg["artifact_key"])
    base = lake.read("idiosyncratic_execution_summary", key).copy()
    upper = lake.read("idiosyncratic_execution_upper_bound_summary", key).copy()
    base["scenario_id"] = str(cfg["baseline_scenario_id"])
    base["relaxed_constraint"] = "none"
    upper["scenario_id"] = str(cfg["upper_bound_scenario_id"])
    upper["relaxed_constraint"] = "all_constraints"
    return base, upper


def _block_inference(daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    baseline_id = str(cfg["baseline_scenario_id"])
    block = int(cfg["inference"]["block_length_trading_days"])
    samples = int(cfg["inference"]["bootstrap_samples"])
    annualization = int(cfg["inference"]["annualization_periods"])
    seed = int(cfg["inference"]["random_seed"])
    records = []
    for path_number, (path_id, path) in enumerate(daily.groupby("path_id", sort=True)):
        base = path.loc[path["scenario_id"].eq(baseline_id), [
            "trade_date", "net_return"
        ]].rename(columns={"net_return": "baseline_return"})
        for scenario_number, (scenario_id, current) in enumerate(
            path.loc[~path["scenario_id"].eq(baseline_id)].groupby("scenario_id", sort=True)
        ):
            paired = current[["trade_date", "net_return"]].merge(
                base, on="trade_date", how="inner", validate="one_to_one"
            )
            if len(paired) != len(base):
                raise RuntimeError("friction scenarios do not share the frozen daily clock")
            difference = paired["net_return"] - paired["baseline_return"]
            estimates = circular_block_bootstrap_mean(
                difference.to_numpy(dtype="float64"), block, samples,
                seed + path_number * 100 + scenario_number,
            ) * annualization
            records.append({
                "path_id": str(path_id), "scenario_id": str(scenario_id),
                "metric": "daily_return_uplift_annualized_arithmetic",
                "observations": len(paired), "block_length_trading_days": block,
                "point_estimate": float(difference.mean() * annualization),
                "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                "probability_uplift_nonpositive": float(
                    (np.sum(estimates <= 0) + 1) / (samples + 1)
                ),
            })
    result = pd.DataFrame(records)
    result["bh_fdr_adjusted_probability"] = benjamini_hochberg(
        result["probability_uplift_nonpositive"]
    )
    result["survives_bh_fdr_5pct"] = (
        result["point_estimate"].gt(0)
        & result["bh_fdr_adjusted_probability"].le(
            float(cfg["inference"]["false_discovery_rate"])
        )
    )
    return result


def _attribution(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    baseline_id = str(cfg["baseline_scenario_id"])
    upper_id = str(cfg["upper_bound_scenario_id"])
    records = []
    for path_id, group in summary.groupby("path_id", sort=True):
        indexed = group.set_index("scenario_id")
        base = indexed.loc[baseline_id]
        upper = indexed.loc[upper_id]
        total_gap = float(upper["annualized_return"] - base["annualized_return"])
        for scenario_id, row in indexed.iterrows():
            uplift = float(row["annualized_return"] - base["annualized_return"])
            records.append({
                "path_id": path_id, "scenario_id": scenario_id,
                "relaxed_constraint": row["relaxed_constraint"],
                "annualized_return": float(row["annualized_return"]),
                "annualized_return_uplift_vs_all_constraints": uplift,
                "fraction_of_upper_bound_gap_closed": (
                    uplift / total_gap if abs(total_gap) > 1e-12 else np.nan
                ),
                "fill_ratio_by_shares": float(row["fill_ratio_by_shares"]),
                "mean_target_tracking_l1": float(row["mean_target_tracking_l1"]),
                "p95_target_tracking_l1": float(row["p95_target_tracking_l1"]),
                "mean_blocked_buy_weight": float(row["mean_blocked_buy_weight"]),
                "mean_blocked_sell_weight": float(row["mean_blocked_sell_weight"]),
                "mean_cash_weight": float(row["mean_cash_weight"]),
                "maximum_drawdown": float(row["maximum_drawdown"]),
                "sharpe_reliable_risk_free": float(row["sharpe_reliable_risk_free"]),
                "effects_are_non_additive": True,
            })
    return pd.DataFrame(records)


def _waterfall(attribution: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    steps = [
        (str(cfg["baseline_scenario_id"]), "baseline"),
        ("no_trade_flags", "relax_trade_flags"),
        ("no_trade_flags_no_volume", "then_relax_volume_participation"),
        ("no_trade_flags_no_volume_no_lot", "then_relax_order_quantity_rules"),
        (str(cfg["upper_bound_scenario_id"]), "then_relax_t_plus_one"),
    ]
    records = []
    for path_id, group in attribution.groupby("path_id", sort=True):
        indexed = group.set_index("scenario_id")
        base_return = float(indexed.loc[steps[0][0], "annualized_return"])
        upper_return = float(indexed.loc[steps[-1][0], "annualized_return"])
        total_gap = upper_return - base_return
        previous = base_return
        for step_number, (scenario_id, step_name) in enumerate(steps):
            current = float(indexed.loc[scenario_id, "annualized_return"])
            records.append({
                "path_id": path_id, "step_number": step_number,
                "scenario_id": scenario_id, "step_name": step_name,
                "annualized_return": current,
                "ordered_incremental_annualized_return": (
                    0.0 if step_number == 0 else current - previous
                ),
                "cumulative_fraction_of_upper_bound_gap": (
                    (current - base_return) / total_gap
                    if abs(total_gap) > 1e-12 else np.nan
                ),
                "increments_are_order_dependent": True,
            })
            previous = current
    return pd.DataFrame(records)


def _write_report(
    attribution: pd.DataFrame, inference: pd.DataFrame,
    waterfall: pd.DataFrame, blocker_summary: pd.DataFrame, cfg: dict[str, Any],
) -> None:
    focus = attribution.loc[
        ~attribution["scenario_id"].isin([
            cfg["baseline_scenario_id"], cfg["upper_bound_scenario_id"]
        ])
    ].sort_values(["path_id", "annualized_return_uplift_vs_all_constraints"], ascending=[True, False])
    lines = [
        "# Phase 12.3：股数执行摩擦归因（验证期、事后诊断）", "",
        "> 本实验固定 Phase 12.2 的信号、目标权重、成本和日期，只逐项解除一个执行约束。"
        "结果用于解释实施损耗，不用于选模、晋级、回看历史 OOS 或产生 paper 业绩。", "",
        "## 单约束反事实", "", focus.to_markdown(index=False, floatfmt=".6f"), "",
        "## 固定顺序约束 waterfall", "", waterfall.to_markdown(index=False, floatfmt=".6f"), "",
        "## 配对区块 bootstrap", "", inference.to_markdown(index=False, floatfmt=".6f"), "",
        "## 阻塞事件", "", blocker_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 解释边界", "",
        "- OFAT（one-factor-at-a-time）效应存在交互，不能相加，也不是结构因果估计。",
        "- waterfall 增量可以对账到上界，但依赖预先固定的解除顺序，不是唯一归因。",
        "- `fractional_upper_bound` 同时取消多项真实约束，仅是不可实施上界。",
        "- 所有结果来自已反复研究的 2016–2020 验证区间，不构成独立证据。",
        "- 本诊断不得改变当前冻结候选；任何执行友好新版本必须预注册后仅从新日期前瞻观察。",
    ]
    path = ROOT / str(cfg["report_path"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(lake: Lake | None = None) -> dict[str, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config(); validate_config(cfg)
    execution_cfg = yaml.safe_load(
        (ROOT / "config" / "idiosyncratic_share_execution_validation.yaml").read_text(
            encoding="utf-8"
        )
    )
    targets, market, rates, base_daily, upper_daily = _existing_inputs(lake, cfg)
    base_summary, upper_summary = _existing_summary_rows(lake, cfg)
    daily_frames = [
        base_daily.assign(scenario_id=str(cfg["baseline_scenario_id"])),
        upper_daily.assign(scenario_id=str(cfg["upper_bound_scenario_id"])),
    ]
    summary_frames = [base_summary, upper_summary]
    invariant_rows = []
    blocker_frames = []

    base_events = lake.read("idiosyncratic_execution_events", str(cfg["artifact_key"]))
    base_blocked = base_events.loc[base_events["status"].eq("BLOCKED")].copy()
    base_blocked["scenario_id"] = str(cfg["baseline_scenario_id"])
    blocker_frames.append(base_blocked)

    for scenario_spec in [*cfg["scenarios"], *cfg["waterfall_scenarios"]]:
        scenario_id = str(scenario_spec["scenario_id"])
        relaxed = str(scenario_spec["relaxed_constraint"])
        for path_id, group in targets.groupby("path_id", sort=False):
            codes = set(group["instrument_id"].astype(str))
            path_market = market.loc[market["instrument_id"].astype(str).isin(codes)]
            scenario = {
                "scenario": f"{path_id}::{scenario_id}",
                **{name: bool(scenario_spec[name]) for name in (
                    "enforce_trade_flags", "enforce_volume_participation",
                    "enforce_order_quantity_rules", "enforce_t_plus_one",
                )},
            }
            result = simulate_execution(
                path_market,
                group[["execution_date", "instrument_id", "target_weight"]],
                execution_cfg, scenario=scenario,
            )
            summary, invariant = _decorate_summary(
                result, rates, execution_cfg, path_id=str(path_id),
                scenario_id=scenario_id, relaxed_constraint=relaxed,
            )
            summary_frames.append(pd.DataFrame([summary]))
            invariant_rows.append(invariant)
            daily_frames.append(result.daily.assign(
                path_id=str(path_id), scenario_id=scenario_id,
            ))
            blocked = result.events.loc[result.events["status"].eq("BLOCKED")].copy()
            blocked["path_id"] = str(path_id); blocked["scenario_id"] = scenario_id
            blocker_frames.append(blocked)
            print(
                f"Phase 12.3 {scenario_id}/{path_id}: annual_return="
                f"{summary['annualized_return']:.2%}, fill={summary['fill_ratio_by_shares']:.2%}",
                flush=True,
            )

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    invariants = pd.DataFrame(invariant_rows)
    attribution = _attribution(summary, cfg)
    waterfall = _waterfall(attribution, cfg)
    inference = _block_inference(daily, cfg)
    blockers = pd.concat(blocker_frames, ignore_index=True)
    blocker_summary = blockers.groupby(
        ["path_id", "scenario_id", "reason"], as_index=False, dropna=False
    ).agg(
        blocked_event_count=("event_id", "size"),
        remaining_shares_event_sum=("remaining_shares", "sum"),
    )
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "source_experiment": cfg["source_experiment"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "signals_or_targets_changed": False,
        "promotion_or_selection_allowed": False,
        "paper_observations_created": 0,
        "ofat_effects_are_additive": False,
    }])
    outputs = {
        "idiosyncratic_friction_protocol": protocol,
        "idiosyncratic_friction_daily": daily,
        "idiosyncratic_friction_summary": summary,
        "idiosyncratic_friction_invariants": invariants,
        "idiosyncratic_friction_attribution": attribution,
        "idiosyncratic_friction_waterfall": waterfall,
        "idiosyncratic_friction_inference": inference,
        "idiosyncratic_friction_blockers": blocker_summary,
    }
    key = str(cfg["artifact_key"])
    for table, frame in outputs.items():
        lake.write(
            table, frame, key=key,
            source="idiosyncratic_execution_friction_attribution.v1", request=cfg,
        )
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    attribution.to_csv(TABLE_ROOT / "idiosyncratic_execution_friction_attribution.csv", index=False)
    _write_report(attribution, inference, waterfall, blocker_summary, cfg)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
