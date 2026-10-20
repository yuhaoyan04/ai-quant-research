"""Phase 7 unified target, position, cash and NAV accounting ledger.

The builder is deliberately read-only with respect to the frozen Phase 5
execution outputs.  It normalizes those outputs and independently reconciles
cash, positions, fees, market value, NAV and target tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "unified_account_ledger.yaml"
REPORT_PATH = ROOT / "reports" / "unified_account_ledger.md"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def replay_transactions(initial_cash: float, marks: pd.DataFrame,
                        fills: pd.DataFrame, actions: pd.DataFrame | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independently replay a small account path for hand-checkable tests.

    Actions are applied before fills.  Required mark columns are
    ``trade_date``, ``instrument_id`` and ``price``.  Fill columns are side,
    shares, price and fee.  An action may specify share_factor and cash_delta.
    """
    required_marks = {"trade_date", "instrument_id", "price"}
    required_fills = {"trade_date", "instrument_id", "side", "shares", "price", "fee"}
    if missing := required_marks - set(marks):
        raise KeyError(f"mark columns missing: {sorted(missing)}")
    if missing := required_fills - set(fills):
        raise KeyError(f"fill columns missing: {sorted(missing)}")
    action_frame = actions.copy() if actions is not None else pd.DataFrame()
    cash = float(initial_cash)
    positions: dict[str, float] = {}
    daily_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for date, day_marks in marks.sort_values(["trade_date", "instrument_id"]).groupby(
        "trade_date", sort=True
    ):
        prices = dict(zip(day_marks["instrument_id"].astype(str), day_marks["price"].astype(float)))
        if not action_frame.empty:
            today_actions = action_frame.loc[action_frame["trade_date"].astype(str).eq(str(date))]
            for row in today_actions.itertuples(index=False):
                code = str(row.instrument_id)
                factor = float(getattr(row, "share_factor", 1.0))
                positions[code] = positions.get(code, 0.0) * factor
                cash += float(getattr(row, "cash_delta", 0.0))
        pretrade_nav = cash + sum(quantity * prices[code] for code, quantity in positions.items())
        today_fills = fills.loc[fills["trade_date"].astype(str).eq(str(date))]
        fees = 0.0
        for row in today_fills.itertuples(index=False):
            code, quantity = str(row.instrument_id), float(row.shares)
            notional, fee = quantity * float(row.price), float(row.fee)
            if str(row.side) == "buy":
                if cash + 1e-12 < notional + fee:
                    raise ValueError("insufficient cash")
                positions[code] = positions.get(code, 0.0) + quantity
                cash -= notional + fee
            elif str(row.side) == "sell":
                if positions.get(code, 0.0) + 1e-12 < quantity:
                    raise ValueError("oversell")
                positions[code] -= quantity
                cash += notional - fee
            else:
                raise ValueError(f"unsupported side: {row.side}")
            fees += fee
        positions = {code: quantity for code, quantity in positions.items() if quantity > 1e-12}
        market_value = sum(quantity * prices[code] for code, quantity in positions.items())
        nav = cash + market_value
        daily_rows.append({
            "trade_date": str(date), "pretrade_nav": pretrade_nav, "cash": cash,
            "market_value": market_value, "nav": nav, "fees": fees,
        })
        for code, quantity in sorted(positions.items()):
            position_rows.append({
                "trade_date": str(date), "instrument_id": code,
                "actual_shares": quantity, "mark_price": prices[code],
                "market_value": quantity * prices[code],
                "actual_weight": quantity * prices[code] / nav,
            })
    return pd.DataFrame(daily_rows), pd.DataFrame(position_rows)


def _selected_inputs(lake: Lake, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames = {
        name: lake.read(str(source["table"]), str(source["key"]))
        for name, source in cfg["sources"].items()
    }
    missing = [name for name, frame in frames.items() if frame.empty]
    if missing:
        raise RuntimeError(f"required Phase 5 inputs missing or empty: {missing}")
    scenario = str(cfg["scenario"])
    frames["daily"] = frames["daily"].loc[frames["daily"]["scenario"].eq(scenario)].copy()
    frames["orders"] = frames["orders"].loc[frames["orders"]["scenario"].eq(scenario)].copy()
    frames["events"] = frames["events"].loc[frames["events"]["scenario"].eq(scenario)].copy()
    frames["fills"] = frames["fills"].loc[frames["fills"]["portfolio"].eq(scenario)].copy()
    frames["positions"] = frames["positions"].loc[
        frames["positions"]["scenario"].eq(scenario)
    ].copy()
    candidate = cfg["candidate"]
    targets = frames["targets"]
    frames["targets"] = targets.loc[
        targets["model"].eq(str(candidate["model"]))
        & targets["alpha_pipeline"].eq(str(candidate["alpha_pipeline"]))
        & targets["strategy"].eq(str(candidate["strategy"])),
        ["execution_date", "instrument_id", "target_weight"],
    ].copy()
    for name in ("daily", "orders", "events", "fills", "positions", "targets"):
        if frames[name].empty:
            raise RuntimeError(f"selected account input is empty: {name}")
    return frames


def reconstruct_corporate_actions(daily: pd.DataFrame, positions: pd.DataFrame,
                                  market: pd.DataFrame) -> pd.DataFrame:
    """Rebuild scenario-specific actions from prior holdings and market fields."""
    dates = sorted(daily["trade_date"].astype(str).unique())
    next_date = dict(zip(dates[:-1], dates[1:]))
    prior = positions[["trade_date", "instrument_id", "shares"]].copy()
    prior["trade_date"] = prior["trade_date"].astype(str).map(next_date)
    prior = prior.dropna(subset=["trade_date"]).rename(columns={"shares": "shares_before"})
    action_market = market.loc[
        pd.to_numeric(market["share_factor"], errors="coerce").sub(1.0).abs().gt(1e-12)
        | pd.to_numeric(market["cash_dividend_per_share"], errors="coerce").fillna(0).ne(0),
        ["trade_date", "instrument_id", "close", "previous_close", "daily_return",
         "share_factor", "cash_dividend_per_share", "action_source"],
    ].copy()
    result = prior.merge(action_market, on=["trade_date", "instrument_id"], how="inner")
    if result.empty:
        return pd.DataFrame(columns=[
            "trade_date", "instrument_id", "share_factor", "cash_dividend_per_share",
            "shares_before", "shares_after", "cash_delta", "value_before_proxy",
            "value_after", "value_reconciliation_error", "action_source",
        ])
    result["shares_after"] = result["shares_before"] * result["share_factor"]
    result["cash_delta"] = result["shares_before"] * result["cash_dividend_per_share"]
    result["value_before_proxy"] = (
        result["shares_before"] * result["previous_close"] * (1.0 + result["daily_return"])
    )
    result["value_after"] = result["shares_after"] * result["close"] + result["cash_delta"]
    result["value_reconciliation_error"] = result["value_after"] - result["value_before_proxy"]
    return result[[
        "trade_date", "instrument_id", "share_factor", "cash_dividend_per_share",
        "shares_before", "shares_after", "cash_delta", "value_before_proxy",
        "value_after", "value_reconciliation_error", "action_source",
    ]].sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)


def build_security_ledger(daily: pd.DataFrame, positions: pd.DataFrame,
                          targets: pd.DataFrame, account_id: str) -> pd.DataFrame:
    target_daily = daily[["trade_date", "active_target_date", "nav", "cash_weight"]].merge(
        targets.rename(columns={"execution_date": "active_target_date"}),
        on="active_target_date", how="left", validate="many_to_many",
    )
    target_daily = target_daily[[
        "trade_date", "active_target_date", "instrument_id", "target_weight",
        "nav", "cash_weight",
    ]]
    actual = positions.rename(columns={"shares": "actual_shares"})[[
        "trade_date", "instrument_id", "actual_shares", "available_shares",
        "unsettled_shares", "mark_price", "market_value", "actual_weight",
    ]]
    result = target_daily.merge(
        actual, on=["trade_date", "instrument_id"], how="outer", validate="one_to_one",
    )
    daily_context = daily.set_index("trade_date")[["active_target_date", "nav", "cash_weight"]]
    for column in ("active_target_date", "nav", "cash_weight"):
        result[column] = result[column].fillna(result["trade_date"].map(daily_context[column]))
    result["target_weight"] = result["target_weight"].fillna(0.0)
    for column in ("actual_shares", "available_shares", "unsettled_shares",
                   "market_value", "actual_weight"):
        result[column] = result[column].fillna(0.0)
    result["weight_gap"] = result["actual_weight"] - result["target_weight"]
    result.insert(0, "account_id", account_id)
    return result.sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)


def build_position_reconciliation(daily: pd.DataFrame, positions: pd.DataFrame,
                                  fills: pd.DataFrame, actions: pd.DataFrame,
                                  account_id: str) -> pd.DataFrame:
    dates = sorted(daily["trade_date"].astype(str).unique())
    next_date = dict(zip(dates[:-1], dates[1:]))
    current = positions[["trade_date", "instrument_id", "shares"]].rename(
        columns={"shares": "actual_ending_shares"}
    )
    prior = positions[["trade_date", "instrument_id", "shares"]].copy()
    prior["trade_date"] = prior["trade_date"].astype(str).map(next_date)
    prior = prior.dropna(subset=["trade_date"]).rename(columns={"shares": "prior_ending_shares"})
    signed = fills.assign(
        signed_fill_shares=np.where(fills["side"].eq("buy"), fills["executed_shares"],
                                    -fills["executed_shares"])
    ).groupby(["trade_date", "instrument_id"], as_index=False)["signed_fill_shares"].sum()
    action = actions[["trade_date", "instrument_id", "share_factor"]].copy()
    keys = pd.concat([
        frame[["trade_date", "instrument_id"]] for frame in (current, prior, signed, action)
    ], ignore_index=True).drop_duplicates()
    result = keys.merge(prior, on=["trade_date", "instrument_id"], how="left")
    result = result.merge(action, on=["trade_date", "instrument_id"], how="left")
    result = result.merge(signed, on=["trade_date", "instrument_id"], how="left")
    result = result.merge(current, on=["trade_date", "instrument_id"], how="left")
    for column, default in (
        ("prior_ending_shares", 0.0), ("share_factor", 1.0),
        ("signed_fill_shares", 0.0), ("actual_ending_shares", 0.0),
    ):
        result[column] = result[column].fillna(default)
    result["expected_ending_shares"] = (
        result["prior_ending_shares"] * result["share_factor"]
        + result["signed_fill_shares"]
    )
    result["share_reconciliation_error"] = (
        result["actual_ending_shares"] - result["expected_ending_shares"]
    )
    result.insert(0, "account_id", account_id)
    return result.sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)


def build_daily_reconciliation(daily: pd.DataFrame, fills: pd.DataFrame,
                               positions: pd.DataFrame, actions: pd.DataFrame,
                               security: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    tol = cfg["tolerances"]
    fill_day = fills.groupby("trade_date").agg(
        fill_buy_notional=("notional", lambda x: float(x[fills.loc[x.index, "side"].eq("buy")].sum())),
        fill_sell_notional=("notional", lambda x: float(x[fills.loc[x.index, "side"].eq("sell")].sum())),
        fill_total_cost=("total_cost", "sum"),
    )
    position_day = positions.groupby("trade_date").agg(
        position_market_value=("market_value", "sum"),
        position_actual_weight=("actual_weight", "sum"),
    )
    action_cash = actions.groupby("trade_date")["cash_delta"].sum().rename("action_cash_delta")
    tracking = security.groupby("trade_date")["weight_gap"].apply(
        lambda x: float(x.abs().sum())
    ).rename("security_weight_gap_l1")
    result = daily.sort_values("trade_date").copy()
    result = result.join(fill_day, on="trade_date").join(position_day, on="trade_date")
    result = result.join(action_cash, on="trade_date").join(tracking, on="trade_date")
    numeric_zero = [
        "fill_buy_notional", "fill_sell_notional", "fill_total_cost",
        "position_market_value", "position_actual_weight", "action_cash_delta",
        "security_weight_gap_l1",
    ]
    result[numeric_zero] = result[numeric_zero].fillna(0.0)
    result["prior_cash"] = result["cash"].shift(1).fillna(float(cfg["initial_cash_cny"]))
    result["expected_cash"] = (
        result["prior_cash"] + result["fill_sell_notional"]
        - result["fill_buy_notional"] - result["fill_total_cost"]
        + result["action_cash_delta"]
    )
    result["cash_reconciliation_error"] = result["cash"] - result["expected_cash"]
    result["nav_identity_error"] = result["nav"] - result["cash"] - result["position_market_value"]
    result["market_value_reconciliation_error"] = (
        result["market_value"] - result["position_market_value"]
    )
    result["buy_notional_reconciliation_error"] = result["buy_notional"] - result["fill_buy_notional"]
    result["sell_notional_reconciliation_error"] = result["sell_notional"] - result["fill_sell_notional"]
    result["fee_reconciliation_error"] = result["total_cost"] - result["fill_total_cost"]
    result["weight_identity_error"] = (
        result["cash_weight"] + result["position_actual_weight"] - 1.0
    )
    result["tracking_reconciliation_error"] = (
        result["target_tracking_l1"]
        - result["security_weight_gap_l1"] - result["cash_weight"]
    )
    result["cash_pass"] = result["cash_reconciliation_error"].abs().le(float(tol["cash_cny"]))
    result["nav_pass"] = result["nav_identity_error"].abs().le(float(tol["nav_cny"]))
    result["market_value_pass"] = result["market_value_reconciliation_error"].abs().le(
        float(tol["market_value_cny"])
    )
    result["notional_pass"] = (
        result["buy_notional_reconciliation_error"].abs().le(float(tol["market_value_cny"]))
        & result["sell_notional_reconciliation_error"].abs().le(float(tol["market_value_cny"]))
    )
    result["fee_pass"] = result["fee_reconciliation_error"].abs().le(float(tol["fees_cny"]))
    result["weight_pass"] = result["weight_identity_error"].abs().le(float(tol["weight"]))
    result["tracking_pass"] = result["tracking_reconciliation_error"].abs().le(
        float(tol["weight"])
    )
    result.insert(0, "account_id", str(cfg["account_id"]))
    return result


def _checks(daily: pd.DataFrame, positions: pd.DataFrame, orders: pd.DataFrame,
            events: pd.DataFrame, fills: pd.DataFrame, actions: pd.DataFrame,
            position_recon: pd.DataFrame, legacy_actions: pd.DataFrame,
            cfg: dict[str, Any]) -> pd.DataFrame:
    tol = cfg["tolerances"]
    fill_by_order = fills.groupby("order_id")["executed_shares"].sum()
    order_error = (
        orders.set_index("order_id")["filled_shares"]
        - fill_by_order.reindex(orders["order_id"]).fillna(0).to_numpy()
    ).abs().max()
    terminal = {"FILLED", "CANCELLED", "EXPIRED", "REPLACED"}
    rows = [
        ("daily_cash_reconciles", daily["cash_pass"].all(), daily["cash_reconciliation_error"].abs().max()),
        ("daily_nav_reconciles", daily["nav_pass"].all(), daily["nav_identity_error"].abs().max()),
        ("daily_market_value_reconciles", daily["market_value_pass"].all(), daily["market_value_reconciliation_error"].abs().max()),
        ("daily_fill_notional_reconciles", daily["notional_pass"].all(), max(daily["buy_notional_reconciliation_error"].abs().max(), daily["sell_notional_reconciliation_error"].abs().max())),
        ("daily_fees_reconcile", daily["fee_pass"].all(), daily["fee_reconciliation_error"].abs().max()),
        ("cash_plus_actual_weights_equal_one", daily["weight_pass"].all(), daily["weight_identity_error"].abs().max()),
        ("target_actual_tracking_reconciles", daily["tracking_pass"].all(), daily["tracking_reconciliation_error"].abs().max()),
        ("position_share_transitions_reconcile", position_recon["share_reconciliation_error"].abs().le(float(tol["shares"])).all(), position_recon["share_reconciliation_error"].abs().max()),
        ("cash_never_negative", daily["cash"].ge(-float(tol["cash_cny"])).all(), daily["cash"].min()),
        ("positions_never_negative", positions["shares"].ge(-float(tol["shares"])).all(), positions["shares"].min()),
        ("t_plus_one_sell_inventory", (fills.loc[fills["side"].eq("sell"), "executed_shares"] <= fills.loc[fills["side"].eq("sell"), "available_before_fill"] + float(tol["shares"])).all(), int((fills.loc[fills["side"].eq("sell"), "executed_shares"] > fills.loc[fills["side"].eq("sell"), "available_before_fill"] + float(tol["shares"])).sum())),
        ("order_fill_shares_reconcile", float(order_error) <= float(tol["shares"]), float(order_error)),
        ("orders_are_terminal", set(orders["status"].astype(str)) <= terminal, sorted(set(orders["status"].astype(str)))),
        ("one_new_event_per_order", events.loc[events["status"].eq("NEW"), "order_id"].nunique() == len(orders), events.loc[events["status"].eq("NEW"), "order_id"].nunique()),
        ("corporate_actions_value_reconcile", actions["value_reconciliation_error"].abs().le(float(tol["market_value_cny"])).all(), actions["value_reconciliation_error"].abs().max()),
        ("legacy_action_scenario_identity_present", "scenario" in legacy_actions.columns, "legacy table omitted scenario during two-scenario concatenation"),
    ]
    return pd.DataFrame([
        {"check_name": name, "status": "PASS" if bool(ok) else ("WARN" if name == "legacy_action_scenario_identity_present" else "FAIL"), "observed": str(observed)}
        for name, ok, observed in rows
    ])


def run(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    frames = _selected_inputs(lake, cfg)
    account_id, scenario = str(cfg["account_id"]), str(cfg["scenario"])
    actions = reconstruct_corporate_actions(frames["daily"], frames["positions"], frames["market"])
    security = build_security_ledger(
        frames["daily"], frames["positions"], frames["targets"], account_id,
    )
    position_recon = build_position_reconciliation(
        frames["daily"], frames["positions"], frames["fills"], actions, account_id,
    )
    daily_recon = build_daily_reconciliation(
        frames["daily"], frames["fills"], frames["positions"], actions, security, cfg,
    )
    checks = _checks(
        daily_recon, frames["positions"], frames["orders"], frames["events"],
        frames["fills"], actions, position_recon, frames["corporate_actions_legacy"], cfg,
    )
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "account_id": account_id,
        "scenario": scenario, "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "timing_convention": cfg["timing_convention"],
        "source_outputs_mutated": False, "returns_recomputed_or_changed": False,
        "legacy_action_scenario_defect_disclosed": True,
    }])
    normalized_orders = frames["orders"].drop(columns=["scenario"]).copy()
    normalized_orders.insert(0, "account_id", account_id)
    normalized_orders.insert(1, "scenario", scenario)
    normalized_events = frames["events"].drop(columns=["scenario"]).copy()
    normalized_events.insert(0, "account_id", account_id)
    normalized_events.insert(1, "scenario", scenario)
    normalized_fills = frames["fills"].copy()
    normalized_fills.insert(0, "account_id", account_id)
    actions.insert(0, "account_id", account_id)
    exceptions = normalized_orders.loc[~normalized_orders["status"].eq("FILLED")].merge(
        normalized_events.loc[normalized_events["status"].eq("BLOCKED")]
        .groupby("order_id").agg(blocked_event_count=("event_id", "size"),
                                  blocked_reasons=("reason", lambda x: "|".join(sorted(set(x.dropna().astype(str))))))
        .reset_index(), on="order_id", how="left",
    )
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all hard Phase 7 accounting assertions passed",
        }])
    cash_journal = daily_recon[[
        "account_id", "trade_date", "prior_cash", "fill_sell_notional",
        "fill_buy_notional", "fill_total_cost", "action_cash_delta",
        "expected_cash", "cash", "cash_reconciliation_error",
    ]].copy()
    daily_ledger_columns = [
        "account_id", "scenario", "trade_date", "active_target_date", "new_target",
        "previous_nav", "pretrade_nav", "posttrade_nav", "gross_return", "net_return",
        "cash", "market_value", "nav", "cash_weight", "position_actual_weight",
        "buy_notional", "sell_notional", "total_cost", "open_order_count",
        "blocked_buy_notional", "blocked_sell_notional", "target_tracking_l1",
    ]
    daily_ledger = daily_recon[daily_ledger_columns].copy()
    tables = {
        "unified_account_protocol": protocol,
        "unified_account_daily_ledger": daily_ledger,
        "unified_account_security_ledger": security,
        "unified_account_cash_journal": cash_journal,
        "unified_account_orders": normalized_orders,
        "unified_account_events": normalized_events,
        "unified_account_fills": normalized_fills,
        "unified_account_corporate_actions": actions,
        "unified_account_daily_reconciliation": daily_recon,
        "unified_account_position_reconciliation": position_recon,
        "unified_account_order_exceptions": exceptions,
        "unified_account_failed_assertions": failures,
        "unified_account_checks": checks,
    }
    for table, frame in tables.items():
        lake.write(table, frame, key="v1", source="unified_account_ledger.v1", request=cfg)
    hard_failures = int(checks["status"].eq("FAIL").sum())
    warnings = int(checks["status"].eq("WARN").sum())
    lines = [
        "# Phase 7：现金、持仓、NAV 与目标权重统一时序账本", "",
        "## 定位", "",
        "本阶段只读规范化已冻结的 Phase 5 主账户，不改变目标、成交、费用、持仓或收益。",
        "2021–2026 仍是已查看的 historical OOS；本阶段是会计与谱系诊断，不构成新 Alpha 证据。", "",
        "## 时序约定", "",
        f"`{cfg['timing_convention']}`：先公司行动与盯市，得到交易前 NAV；新目标替换旧订单；随后在收盘执行、扣费并更新现金/持仓；最后计算交易后 NAV、实际权重和现金权重。", "",
        "## 统一对象", "",
        "- `target_weight` 只来自冻结目标；`actual_shares` 只来自账户持仓；`actual_weight` 由实际股数、收盘价与 NAV 计算；`cash_weight` 单列。",
        "- 未成交订单仅在订单/事件表出现，不进入实际持仓。",
        "- 公司行动从所选账户前日持仓与冻结市场字段重建，因为 Phase 5 旧公司行动合并表遗漏 `scenario`。旧表不被覆盖。", "",
        "## 机器对账", "", checks.to_markdown(index=False), "",
        "## 规模", "",
        f"- 账户日：{len(daily_ledger):,}；证券日：{len(security):,}；订单：{len(normalized_orders):,}；成交：{len(normalized_fills):,}；公司行动：{len(actions):,}。",
        f"- 异常/未完全成交订单：{len(exceptions):,}；硬失败：{hard_failures}；警告：{warnings}。", "",
        "## 结论", "",
        "若所有硬门禁通过，则 Phase 7 证明已有收益序列可由现金、成交、费用、公司行动和实际持仓逐日重建；该结论只证明会计一致性，不证明策略未来有效。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(checks.to_string(index=False), flush=True)
    if hard_failures:
        raise RuntimeError(f"Phase 7 accounting reconciliation has {hard_failures} hard failures")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 7 unified account ledger")
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
