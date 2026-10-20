"""Share-order replay for all ML and style-replica incremental-alpha paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.institutional_execution import _summaries, simulate_execution
from quant_research.rf_daily_retry_execution import _limit_array


CONFIG_PATH = ROOT / "config" / "incremental_alpha_fill_execution.yaml"
REPORT_PATH = ROOT / "reports" / "incremental_alpha_fill_execution.md"
TABLE_ROOT = ROOT / "reports" / "tables"
KEYS = ["model", "alpha_pipeline", "strategy"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def combined_targets(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    model_spec = cfg["model_targets"]
    style_spec = cfg["style_targets"]
    model = lake.read(str(model_spec["table"]), str(model_spec["key"])).copy()
    model["portfolio_type"] = "ml_model"
    style = lake.read(str(style_spec["table"]), str(style_spec["key"])).copy()
    columns = [*KEYS, "portfolio_type", "execution_date", "instrument_id", "target_weight"]
    result = pd.concat([model[columns], style[columns]], ignore_index=True)
    result = result.loc[result["target_weight"].gt(0)].copy()
    duplicates = result.duplicated([*KEYS, "portfolio_type", "execution_date", "instrument_id"])
    if duplicates.any():
        raise ValueError("duplicate fill-replay target keys")
    sums = result.groupby([*KEYS, "portfolio_type", "execution_date"])["target_weight"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError("fill-replay target weights do not sum to one")
    return result.sort_values([*KEYS, "portfolio_type", "execution_date", "instrument_id"])


def build_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    """Extend the Phase-5 raw-close market only for missing replay target names."""
    lake = lake or Lake()
    cfg = load_config()
    destination = lake.table_path("incremental_alpha_fill_market", "v1")
    if destination.exists() and not overwrite:
        return lake.read("incremental_alpha_fill_market", "v1")
    targets = combined_targets(lake, cfg)
    base_spec = cfg["base_market"]
    existing = lake.read(str(base_spec["table"]), str(base_spec["key"]))
    start = str(targets["execution_date"].min())
    end = str(existing["trade_date"].max())
    codes = set(targets.loc[targets["execution_date"].le(end), "instrument_id"].astype(str))
    available = set(existing["instrument_id"].astype(str))
    frames = [existing.loc[existing["instrument_id"].astype(str).isin(codes)].copy()]
    missing = sorted(codes - available)
    with (ROOT / "config" / "constrained_portfolio_daily_execution.yaml").open(
        encoding="utf-8"
    ) as handle:
        rules = yaml.safe_load(handle)["rules"]
    threshold = float(cfg["corporate_action_proxy"]["activation_absolute_deviation"])
    tolerance = 0.20
    for number, code in enumerate(missing, start=1):
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", code.replace(".", "_")),
            columns=[
                "instrument_id", "trade_date", "close", "volume", "pct_chg",
                "trade_status", "is_st",
            ],
        ).sort_values("trade_date")
        close = pd.to_numeric(bars["close"], errors="coerce")
        previous = close.shift(1)
        returns = pd.to_numeric(bars["pct_chg"], errors="coerce") / 100.0
        raw_factor = previous * (1.0 + returns) / close
        factor = raw_factor.where(raw_factor.sub(1.0).abs().gt(threshold), 1.0).fillna(1.0)
        status = pd.to_numeric(bars["trade_status"], errors="coerce").eq(1)
        is_st = pd.to_numeric(bars["is_st"], errors="coerce").fillna(0).eq(1)
        limits = _limit_array(code, bars["trade_date"], is_st, rules)
        pct = pd.to_numeric(bars["pct_chg"], errors="coerce")
        current = bars.assign(
            previous_close=previous, daily_return=returns,
            buy_allowed=status & pct.lt(limits - tolerance),
            sell_allowed=status & pct.gt(-limits + tolerance),
            at_upper_limit=status & pct.ge(limits - tolerance),
            at_lower_limit=status & pct.le(-limits + tolerance),
            share_factor=factor, cash_dividend_per_share=0.0,
            action_source=np.where(
                factor.ne(1.0), "inferred_value_preserving_factor", "none"
            ),
        )
        frames.append(current.loc[current["trade_date"].between(start, end), [
            "trade_date", "instrument_id", "close", "previous_close", "volume",
            "daily_return", "trade_status", "buy_allowed", "sell_allowed",
            "at_upper_limit", "at_lower_limit", "share_factor",
            "cash_dividend_per_share", "action_source",
        ]])
        if number % 200 == 0:
            print(f"Phase 9 fill market files={number}/{len(missing)}", flush=True)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "instrument_id"], keep="last"
    ).sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
    missing_final = codes - set(result["instrument_id"].astype(str))
    if missing_final:
        raise RuntimeError(f"fill market missing {len(missing_final)} target codes")
    lake.write(
        "incremental_alpha_fill_market", result, key="v1",
        source="incremental_alpha_fill_execution.market.v1",
        request={**cfg, "start": start, "end": end, "codes": len(codes),
                 "reused_codes": len(codes & available), "new_code_files": len(missing)},
    )
    print(
        f"Phase 9 fill market rows={len(result):,} codes={len(codes):,} "
        f"reused={len(codes & available):,} new={len(missing):,}", flush=True,
    )
    return result


def _comparison(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "annualized_return", "sharpe_zero_rate", "maximum_drawdown", "mean_turnover",
        "mean_target_tracking_l1", "mean_cash_weight", "fill_ratio_by_shares",
        "total_orders", "total_fills", "total_cost_cny",
    ]
    wide = summary.pivot(index=KEYS, columns="portfolio_type", values=metrics)
    role = {"ml_model": "model", "style_matched_benchmark": "style"}
    wide.columns = [f"{role[p]}_{metric}" for metric, p in wide.columns]
    result = wide.reset_index()
    for metric in [
        "annualized_return", "sharpe_zero_rate", "maximum_drawdown", "mean_turnover",
        "mean_target_tracking_l1", "mean_cash_weight", "fill_ratio_by_shares",
        "total_cost_cny",
    ]:
        result[f"{metric}_difference_model_minus_style"] = (
            result[f"model_{metric}"] - result[f"style_{metric}"]
        )
    return result


def acceptance_checks(
    targets: pd.DataFrame, market: pd.DataFrame, daily: pd.DataFrame,
    summary: pd.DataFrame, invariants: pd.DataFrame, comparison: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    paths = targets[[*KEYS, "portfolio_type"]].drop_duplicates()
    counts = daily.groupby([*KEYS, "portfolio_type"])["trade_date"].nunique()
    rows = [
        ("eight_frozen_paths_present", len(paths) == int(cfg["expected_portfolio_paths"]), len(paths)),
        ("four_candidate_pairs_present", len(comparison) == int(cfg["expected_candidates"]), len(comparison)),
        ("target_weights_normalized", np.allclose(
            targets.groupby([*KEYS, "portfolio_type", "execution_date"])["target_weight"].sum(),
            1.0, atol=1e-8,
        ), len(targets)),
        ("market_covers_every_target_code", set(targets["instrument_id"].astype(str)).issubset(
            set(market["instrument_id"].astype(str))
        ), market["instrument_id"].nunique()),
        ("common_1387_day_clock", counts.eq(int(cfg["expected_trading_days"])).all(), counts.to_dict()),
        ("all_accounting_invariants_pass", len(invariants) == int(cfg["expected_portfolio_paths"])
         and invariants["all_invariants_pass"].astype(bool).all(),
         invariants["all_invariants_pass"].value_counts().to_dict()),
        ("share_fill_ratios_available", len(summary) == int(cfg["expected_portfolio_paths"])
         and summary["fill_ratio_by_shares"].between(0, 1).all(),
         summary["fill_ratio_by_shares"].round(6).tolist()),
        ("no_t_plus_one_violations", invariants["t_plus_one_violation_count"].eq(0).all(),
         int(invariants["t_plus_one_violation_count"].sum())),
        ("cash_and_nav_reconcile", invariants["negative_cash_day_count"].eq(0).all()
         and invariants["max_abs_nav_reconciliation_error"].le(1e-4).all(),
         float(invariants["max_abs_nav_reconciliation_error"].max())),
        ("same_cost_and_execution_configuration", summary["round_trip_cost_bps"].eq(
            float(cfg["round_trip_cost_bps"])
        ).all(), float(cfg["round_trip_cost_bps"])),
    ]
    return pd.DataFrame([{
        "check_name": name, "status": "PASS" if bool(passed) else "FAIL",
        "observed": str(observed),
    } for name, passed, observed in rows])


def run(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    targets = combined_targets(lake, cfg)
    market = build_market(lake)
    end = str(market["trade_date"].max())
    targets = targets.loc[targets["execution_date"].le(end)].copy()
    scenario_template = dict(cfg["scenario"])
    summary_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for values, group in targets.groupby([*KEYS, "portfolio_type"], sort=True):
        identity = dict(zip([*KEYS, "portfolio_type"], values))
        label = "__".join(map(str, values))
        scenario = {**scenario_template, "scenario": label}
        codes = set(group["instrument_id"].astype(str))
        path_market = market.loc[market["instrument_id"].astype(str).isin(codes)]
        path_targets = group[["execution_date", "instrument_id", "target_weight"]].reset_index(drop=True)
        result = simulate_execution(path_market, path_targets, cfg, scenario=scenario)
        summary, invariant = _summaries(result, int(cfg["annualization_periods"]))
        summary.update(identity)
        summary.update({
            "round_trip_cost_bps": float(cfg["round_trip_cost_bps"]),
            "mean_target_tracking_l1": float(result.daily["target_tracking_l1"].mean()),
            "mean_cash_weight": float(result.daily["cash_weight"].mean()),
            "mean_turnover": float(result.daily["one_way_turnover"].mean()),
        })
        summary_rows.append(summary)
        invariant_rows.append({**identity, **invariant})
        daily_frames.append(result.daily.assign(**identity))
        order_frames.append(result.orders.assign(**identity))
        event_frames.append(result.events.assign(**identity))
        fill_frames.append(result.fills.assign(**identity))
        position_frames.append(result.positions.assign(**identity))
        if not result.corporate_actions.empty:
            action_frames.append(result.corporate_actions.assign(**identity))
        print(
            f"Phase 9 fill replay {label} orders={len(result.orders):,} "
            f"fills={len(result.fills):,} ratio={summary['fill_ratio_by_shares']:.4%}",
            flush=True,
        )
    summary = pd.DataFrame(summary_rows)
    invariants = pd.DataFrame(invariant_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    comparison = _comparison(summary)
    checks = acceptance_checks(targets, market, daily, summary, invariants, comparison, cfg)
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all eight share-level execution paths passed",
        }])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "execution_engine": "institutional_execution.simulate_execution",
        "portfolio_paths": len(summary), "share_level_fill_ratio_available": True,
        "intraday_matching_claimed": False,
    }])
    outputs = {
        "incremental_alpha_fill_protocol": protocol,
        "incremental_alpha_fill_daily": daily,
        "incremental_alpha_fill_orders": pd.concat(order_frames, ignore_index=True),
        "incremental_alpha_fill_events": pd.concat(event_frames, ignore_index=True),
        "incremental_alpha_fill_fills": pd.concat(fill_frames, ignore_index=True),
        "incremental_alpha_fill_positions": pd.concat(position_frames, ignore_index=True),
        "incremental_alpha_fill_summary": summary,
        "incremental_alpha_fill_invariants": invariants,
        "incremental_alpha_fill_comparison": comparison,
        "incremental_alpha_fill_checks": checks,
        "incremental_alpha_fill_failed_assertions": failures,
    }
    if action_frames:
        outputs["incremental_alpha_fill_corporate_actions"] = pd.concat(
            action_frames, ignore_index=True
        )
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="incremental_alpha_fill_execution.v1", request=cfg)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_ROOT / "incremental_alpha_fill_summary.csv", index=False)
    comparison.to_csv(TABLE_ROOT / "incremental_alpha_fill_comparison.csv", index=False)
    lines = [
        "# Phase 9 P0：八路径股数级成交复跑", "",
        "## 范围", "",
        "四条冻结 ML 组合与各自的 PIT 行业/六风格复制组合，统一使用 1 亿元账户、未复权收盘价、T+1、整手、10% 日成交量参与率、停牌/涨跌停重试和 20bp 成本。", "",
        "## 结果", "", comparison.to_markdown(index=False, floatfmt=".6f"), "",
        "## 会计与订单不变量", "", invariants.to_markdown(index=False, floatfmt=".10f"), "",
        "## 机器验收", "", checks.to_markdown(index=False), "",
        "> 成交来自日线收盘近似执行器，不是真实券商或交易所回报；但八条路径采用完全相同的股数、订单和费用协议。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(checks.to_string(index=False), flush=True)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("Phase 9 share-level fill replay checks failed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Phase 9 incremental alpha at share/order level")
    parser.add_argument("command", choices=["build-market", "run", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in {"build-market", "all"}:
        build_market(overwrite=args.overwrite)
    if args.command in {"run", "all"}:
        run()


if __name__ == "__main__":
    main()
