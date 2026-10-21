"""Auditable daily-bar execution state machine for frozen target portfolios.

This is deliberately a conservative approximation, not an exchange matching
engine.  It turns target weights into share orders, events, fills and a cash /
position / NAV ledger while retaining every blocked or replaced order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.daily_execution import daily_performance_summary
from quant_research.execution_market import read_execution_market_rows
from quant_research.ingest import ROOT, Lake
from quant_research.transaction_costs import price_fills


CONFIG_PATH = ROOT / "config" / "institutional_execution.yaml"
REPORT_PATH = ROOT / "reports" / "institutional_execution.md"


@dataclass(frozen=True)
class ExecutionResult:
    daily: pd.DataFrame
    orders: pd.DataFrame
    events: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    corporate_actions: pd.DataFrame


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _order_rule(instrument_id: str, rules: dict[str, Any]) -> tuple[int, int]:
    if str(instrument_id).lower().startswith("sh.688"):
        return int(rules["star_market_minimum_order"]), int(rules["star_market_increment"])
    return int(rules["main_board_buy_lot"]), int(rules["main_board_buy_lot"])


def _target_shares(notional: float, price: float, instrument_id: str,
                   rules: dict[str, Any], enforce: bool) -> float:
    raw = max(float(notional) / float(price), 0.0)
    if not enforce:
        return raw
    minimum, increment = _order_rule(instrument_id, rules)
    if raw < minimum:
        return 0.0
    return float(minimum + np.floor((raw - minimum) / increment) * increment)


def _valid_sell_quantity(requested: float, holding: float, instrument_id: str,
                         rules: dict[str, Any], enforce: bool) -> float:
    quantity = min(max(float(requested), 0.0), max(float(holding), 0.0))
    if not enforce or quantity <= 0:
        return quantity
    minimum, increment = _order_rule(instrument_id, rules)
    if bool(rules.get("odd_lot_liquidation_allowed", True)) and np.isclose(
        quantity, holding, rtol=0, atol=1e-8
    ):
        return quantity
    if quantity < minimum:
        return 0.0
    return float(minimum + np.floor((quantity - minimum) / increment) * increment)


def _reason(row: pd.Series | None, side: str, enforce_flags: bool) -> str | None:
    if row is None:
        return "missing_market"
    if not enforce_flags:
        return None
    if not bool(row.get("trade_status", True)):
        return "suspended"
    allowed = bool(row.get(f"{side}_allowed", False))
    if allowed:
        return None
    if side == "buy" and bool(row.get("at_upper_limit", False)):
        return "upper_limit"
    if side == "sell" and bool(row.get("at_lower_limit", False)):
        return "lower_limit"
    return "trade_flag_blocked"


def _fixed_cost_scenario(cost_bps: float) -> dict[str, Any]:
    return {
        "scenario_id": "C0_fixed_round_trip",
        "scenario_group": "C0",
        "mode": "fixed_round_trip",
        "round_trip_cost_bps": float(cost_bps),
        "evidence_status": "existing_research_baseline",
    }


def simulate_execution(
    market: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    *,
    scenario: dict[str, Any],
    cancellation_requests: dict[str, set[str]] | None = None,
    expire_open_orders_at_end: bool = True,
) -> ExecutionResult:
    """Run one portfolio through the daily close execution state machine."""
    required_market = {
        "trade_date", "instrument_id", "close", "volume", "trade_status",
        "buy_allowed", "sell_allowed",
    }
    required_targets = {"execution_date", "instrument_id", "target_weight"}
    if missing := required_market - set(market):
        raise KeyError(f"institutional market columns missing: {sorted(missing)}")
    if missing := required_targets - set(targets):
        raise KeyError(f"institutional target columns missing: {sorted(missing)}")
    if market.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError("duplicate market date/instrument keys")
    if targets.duplicated(["execution_date", "instrument_id"]).any():
        raise ValueError("duplicate target date/instrument keys")
    target_sums = targets.groupby("execution_date")["target_weight"].sum()
    if not np.allclose(target_sums.to_numpy(float), 1.0, atol=1e-8):
        raise ValueError("target weights must sum to one on every execution date")

    market = market.copy()
    market["trade_date"] = market["trade_date"].astype(str)
    market["instrument_id"] = market["instrument_id"].astype(str)
    market_groups = {
        str(date): group.set_index("instrument_id")
        for date, group in market.groupby("trade_date", sort=True)
    }
    target_groups = {
        str(date): dict(zip(group["instrument_id"].astype(str), group["target_weight"].astype(float)))
        for date, group in targets.groupby("execution_date", sort=True)
    }
    first_date = min(target_groups)
    dates = [date for date in sorted(market_groups) if date >= first_date]
    if not dates:
        raise ValueError("market has no dates at or after the first target")

    initial_cash = float(config["initial_cash_cny"])
    cash = initial_cash
    positions: dict[str, float] = {}
    available: dict[str, float] = {}
    unsettled: dict[str, float] = {}
    prices: dict[str, float] = {}
    open_orders: dict[str, dict[str, Any]] = {}
    all_orders: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    previous_nav = initial_cash
    active_target_date: str | None = None
    active_target: dict[str, float] | None = None
    order_sequence = 0
    fill_sequence = 0
    enforce_flags = bool(scenario["enforce_trade_flags"])
    enforce_volume = bool(scenario["enforce_volume_participation"])
    enforce_quantity = bool(scenario["enforce_order_quantity_rules"])
    enforce_t1 = bool(scenario["enforce_t_plus_one"])
    quantity_rules = config["order_quantity_rules"]
    max_age = int(config["max_pending_trading_days"])
    participation = float(config["max_volume_participation"])
    cost_rate = float(config["round_trip_cost_bps"]) / 10_000.0
    cancellation_requests = cancellation_requests or {}

    def add_event(order: dict[str, Any], date: str, status: str, reason: str | None,
                  shares: float = 0.0) -> None:
        events.append({
            "event_id": f"E{len(events) + 1:09d}", "order_id": order["order_id"],
            "trade_date": date, "event_time": "close", "status": status,
            "reason": reason, "event_shares": float(shares),
            "remaining_shares": float(order["remaining_shares"]),
        })

    for date in dates:
        group = market_groups[date]
        if unsettled:
            for code, quantity in unsettled.items():
                available[code] = available.get(code, 0.0) + quantity
            unsettled = {}

        # Mark prices vectorially. Iterating all quoted names was the dominant
        # cost when the same market is replayed across portfolio methods.
        close_values = pd.to_numeric(group["close"], errors="coerce")
        valid_close = close_values.notna() & close_values.gt(0)
        prices.update(zip(
            group.index[valid_close].astype(str),
            close_values.loc[valid_close].astype(float),
        ))
        factors = pd.to_numeric(
            group.get("share_factor", pd.Series(1.0, index=group.index)), errors="coerce"
        )
        if factors.isna().any() or factors.le(0).any():
            invalid = str(factors.loc[factors.isna() | factors.le(0)].index[0])
            raise ValueError(f"invalid share factor for {invalid}/{date}")
        dividends = pd.to_numeric(
            group.get("cash_dividend_per_share", pd.Series(0.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0)
        action_mask = factors.sub(1.0).abs().gt(1e-12) | dividends.ne(0.0)

        # Explicit or inferred actions are applied before trading. Corporate
        # action rows are sparse, so iterate only rows that can change state.
        for code, row in group.loc[action_mask].iterrows():
            close = prices.get(code)
            if close is None:
                continue
            factor = float(row.get("share_factor", 1.0))
            dividend = float(row.get("cash_dividend_per_share", 0.0))
            old_position = positions.get(code, 0.0)
            if old_position > 0 and (not np.isclose(factor, 1.0) or dividend != 0.0):
                previous_close = float(row.get("previous_close", close))
                daily_return = float(row.get("daily_return", 0.0))
                expected_market_value = old_position * previous_close * (1.0 + daily_return)
                cash_delta = old_position * dividend
                cash += cash_delta
                positions[code] = old_position * factor
                available[code] = available.get(code, 0.0) * factor
                if code in unsettled:
                    unsettled[code] *= factor
                for order in open_orders.values():
                    if order["instrument_id"] == code:
                        order["remaining_shares"] *= factor
                        order["original_shares"] *= factor
                        add_event(order, date, "ACTION_ADJUSTED", "corporate_action", 0.0)
                value_after = positions[code] * close + cash_delta
                action_rows.append({
                    "trade_date": date, "instrument_id": code,
                    "share_factor": factor, "cash_dividend_per_share": dividend,
                    "shares_before": old_position, "shares_after": positions[code],
                    "cash_delta": cash_delta, "value_before_proxy": expected_market_value,
                    "value_after": value_after,
                    "value_reconciliation_error": value_after - expected_market_value,
                    "action_source": str(row.get("action_source", "provided")),
                })

        pretrade_nav = cash + sum(
            quantity * prices[code] for code, quantity in positions.items()
        )
        if pretrade_nav <= 0:
            raise RuntimeError("pre-trade NAV is non-positive")
        gross_return = pretrade_nav / previous_nav - 1.0

        if date in target_groups:
            for order in list(open_orders.values()):
                order["status"] = "REPLACED"
                order["terminal_reason"] = "new_target"
                order["terminal_date"] = date
                add_event(order, date, "REPLACED", "new_target")
            open_orders = {}
            active_target_date = date
            weights = target_groups[date]
            active_target = weights
            codes = set(positions) | set(weights)
            for code in sorted(codes):
                price = prices.get(code)
                if price is None or not np.isfinite(price) or price <= 0:
                    continue
                desired = _target_shares(
                    pretrade_nav * weights.get(code, 0.0), price, code,
                    quantity_rules, enforce_quantity,
                )
                delta = desired - positions.get(code, 0.0)
                if abs(delta) <= 1e-8:
                    continue
                side = "buy" if delta > 0 else "sell"
                quantity = abs(delta)
                if side == "sell":
                    quantity = _valid_sell_quantity(
                        quantity, positions.get(code, 0.0), code,
                        quantity_rules, enforce_quantity,
                    )
                if quantity <= 1e-8:
                    continue
                order_sequence += 1
                order = {
                    "order_id": f"O{order_sequence:09d}", "target_date": date,
                    "submitted_date": date, "instrument_id": code, "side": side,
                    "original_shares": float(quantity), "filled_shares": 0.0,
                    "remaining_shares": float(quantity), "status": "NEW",
                    "terminal_reason": None, "terminal_date": None, "age": 0,
                }
                open_orders[order["order_id"]] = order
                all_orders[order["order_id"]] = order
                add_event(order, date, "NEW", None)

        for order_id in sorted(cancellation_requests.get(date, set())):
            order = open_orders.get(str(order_id))
            if order is None:
                raise KeyError(f"cannot cancel unknown or terminal order: {order_id}")
            order["status"] = "CANCELLED"
            order["terminal_reason"] = "explicit_cancel_request"
            order["terminal_date"] = date
            add_event(order, date, "CANCELLED", order["terminal_reason"])
            open_orders.pop(order["order_id"], None)

        candidates: list[tuple[dict[str, Any], float, float, str | None]] = []
        for order in list(open_orders.values()):
            code = order["instrument_id"]
            row = group.loc[code] if code in group.index else None
            block_reason = _reason(row, order["side"], enforce_flags)
            if block_reason is not None:
                order["status"] = "BLOCKED"
                add_event(order, date, "BLOCKED", block_reason)
                continue
            price = prices.get(code)
            if price is None or price <= 0:
                order["status"] = "BLOCKED"
                add_event(order, date, "BLOCKED", "invalid_price")
                continue
            capacity = float(order["remaining_shares"])
            constraint_reason = None
            if order["side"] == "sell":
                sellable = available.get(code, 0.0) if enforce_t1 else positions.get(code, 0.0)
                if sellable <= 1e-8:
                    order["status"] = "BLOCKED"
                    add_event(order, date, "BLOCKED", "t_plus_one")
                    continue
                if sellable < capacity:
                    capacity = sellable
                    constraint_reason = "position_availability"
                capacity = _valid_sell_quantity(
                    capacity, positions.get(code, 0.0), code,
                    quantity_rules, enforce_quantity,
                )
            if enforce_volume:
                volume = max(float(row.get("volume", 0.0)), 0.0)
                volume_capacity = np.floor(volume * participation)
                if enforce_quantity and not str(code).lower().startswith("sh.688"):
                    lot = int(quantity_rules["main_board_buy_lot"])
                    volume_capacity = np.floor(volume_capacity / lot) * lot
                if volume_capacity < capacity:
                    capacity = float(volume_capacity)
                    constraint_reason = "volume_participation"
            if capacity <= 1e-8:
                order["status"] = "BLOCKED"
                add_event(order, date, "BLOCKED", constraint_reason or "zero_capacity")
                continue
            candidates.append((order, float(capacity), float(price), constraint_reason))

        sell_candidates = [item for item in candidates if item[0]["side"] == "sell"]
        buy_candidates = [item for item in candidates if item[0]["side"] == "buy"]
        sell_notional = sum(quantity * price for _, quantity, price, _ in sell_candidates)
        requested_buy = sum(quantity * price for _, quantity, price, _ in buy_candidates)
        cash_after_sells = cash + sell_notional
        if requested_buy <= 0:
            buy_capacity = 0.0
        else:
            # Reserve sell-side fees before budgeting buys, then reserve the
            # buy-side fee on the executable buy amount.  The old
            # requested_buy > sell_notional branch omitted sell fees; lot-size
            # rounding usually hid the error, while fractional-share upper-
            # bound runs could finish a rebalance with slightly negative cash.
            cash_after_sell_cost = max(
                cash_after_sells - cost_rate * sell_notional, 0.0
            )
            buy_capacity = min(
                requested_buy, cash_after_sell_cost / (1.0 + cost_rate)
            )
        buy_scale = buy_capacity / requested_buy if requested_buy > 0 else 1.0

        todays: list[dict[str, Any]] = []
        for order, capacity, price, constraint_reason in [*sell_candidates, *buy_candidates]:
            quantity = capacity
            reason = constraint_reason
            if order["side"] == "buy" and buy_scale < 1.0:
                quantity *= buy_scale
                reason = "cash_constraint"
                if enforce_quantity:
                    minimum, increment = _order_rule(order["instrument_id"], quantity_rules)
                    if quantity < minimum:
                        quantity = 0.0
                    else:
                        quantity = minimum + np.floor((quantity - minimum) / increment) * increment
            if quantity <= 1e-8:
                order["status"] = "BLOCKED"
                add_event(order, date, "BLOCKED", reason or "cash_constraint")
                continue
            fill_sequence += 1
            todays.append({
                "portfolio": str(scenario["scenario"]),
                "fill_id": f"F{fill_sequence:09d}", "order_id": order["order_id"],
                "trade_date": date, "fill_time": "close",
                "target_date": order["target_date"], "instrument_id": order["instrument_id"],
                "side": order["side"], "ordered_shares": order["original_shares"],
                "executed_shares": float(quantity), "execution_price": float(price),
                "notional": float(quantity * price), "liquidity_constraint": reason,
                "position_before_fill": float(positions.get(order["instrument_id"], 0.0)),
                "available_before_fill": (
                    float(available.get(order["instrument_id"], 0.0))
                    if order["side"] == "sell" else np.nan
                ),
            })

        if todays:
            priced = price_fills(
                pd.DataFrame(todays), _fixed_cost_scenario(config["round_trip_cost_bps"]),
                batch_columns=["portfolio"],
            )
            fills.extend(priced.to_dict(orient="records"))
            for fill in priced.to_dict(orient="records"):
                order = all_orders[fill["order_id"]]
                code, quantity = fill["instrument_id"], float(fill["executed_shares"])
                if fill["side"] == "buy":
                    positions[code] = positions.get(code, 0.0) + quantity
                    if enforce_t1:
                        unsettled[code] = unsettled.get(code, 0.0) + quantity
                    else:
                        available[code] = available.get(code, 0.0) + quantity
                    cash -= float(fill["notional"])
                else:
                    if quantity > positions.get(code, 0.0) + 1e-8:
                        raise RuntimeError("attempted to sell more than the held position")
                    positions[code] = positions.get(code, 0.0) - quantity
                    available[code] = max(available.get(code, 0.0) - quantity, 0.0)
                    cash += float(fill["notional"])
                cash -= float(fill["total_cost"])
                order["filled_shares"] += quantity
                order["remaining_shares"] = max(order["remaining_shares"] - quantity, 0.0)
                status = "FILLED" if order["remaining_shares"] <= 1e-8 else "PARTIALLY_FILLED"
                order["status"] = status
                add_event(order, date, status, fill.get("liquidity_constraint"), quantity)
                if status == "FILLED":
                    order["terminal_date"] = date
                    open_orders.pop(order["order_id"], None)

        if cash < -1e-6:
            raise RuntimeError(f"cash became negative: {cash}")
        cash = max(cash, 0.0)
        for code in list(positions):
            if positions[code] <= 1e-8:
                positions.pop(code, None)
                available.pop(code, None)
                unsettled.pop(code, None)

        for order in list(open_orders.values()):
            order["age"] += 1
            if order["age"] >= max_age:
                order["status"] = "EXPIRED"
                order["terminal_reason"] = "max_pending_trading_days"
                order["terminal_date"] = date
                add_event(order, date, "EXPIRED", order["terminal_reason"])
                open_orders.pop(order["order_id"], None)

        posttrade_nav = cash + sum(
            quantity * prices[code] for code, quantity in positions.items()
        )
        if todays:
            priced_today = priced
            buy_notional = float(priced_today.loc[priced_today["side"].eq("buy"), "notional"].sum())
            sell_notional_day = float(priced_today.loc[priced_today["side"].eq("sell"), "notional"].sum())
            fees = float(priced_today["total_cost"].sum())
        else:
            buy_notional = sell_notional_day = fees = 0.0
        nav_error = posttrade_nav - (pretrade_nav - fees)
        if abs(nav_error) > max(pretrade_nav, 1.0) * 1e-10:
            raise RuntimeError(f"NAV reconciliation failed on {date}: {nav_error}")
        if any(value < -1e-8 for value in positions.values()):
            raise RuntimeError("negative position detected")
        market_value = posttrade_nav - cash
        actual_weights = {
            code: quantity * prices[code] / posttrade_nav
            for code, quantity in positions.items()
        }
        target_tracking_l1 = (
            sum(
                abs(actual_weights.get(code, 0.0) - active_target.get(code, 0.0))
                for code in set(actual_weights) | set(active_target)
            ) + cash / posttrade_nav
            if active_target is not None else 0.0
        )
        daily_rows.append({
            "scenario": str(scenario["scenario"]), "trade_date": date,
            "active_target_date": active_target_date,
            "new_target": date in target_groups, "previous_nav": previous_nav,
            "pretrade_nav": pretrade_nav, "posttrade_nav": posttrade_nav,
            "gross_return": gross_return, "net_return": posttrade_nav / previous_nav - 1.0,
            "cash": cash, "market_value": market_value, "nav": posttrade_nav,
            "cash_weight": cash / posttrade_nav, "accounting_weight_sum":
                (cash + market_value) / posttrade_nav,
            "buy_notional": buy_notional, "sell_notional": sell_notional_day,
            "gross_two_way_turnover": (buy_notional + sell_notional_day) / pretrade_nav,
            "one_way_turnover": max(buy_notional, sell_notional_day) / pretrade_nav,
            "total_cost": fees, "transaction_cost": fees / previous_nav,
            "open_order_count": len(open_orders),
            "blocked_buy_notional": sum(
                o["remaining_shares"] * prices.get(o["instrument_id"], 0.0)
                for o in open_orders.values() if o["side"] == "buy"
            ),
            "blocked_sell_notional": sum(
                o["remaining_shares"] * prices.get(o["instrument_id"], 0.0)
                for o in open_orders.values() if o["side"] == "sell"
            ),
            "target_tracking_l1": target_tracking_l1,
            "nav_reconciliation_error": nav_error,
            "minimum_cash_check": cash,
        })
        for code, quantity in sorted(positions.items()):
            position_rows.append({
                "scenario": str(scenario["scenario"]), "trade_date": date,
                "instrument_id": code, "shares": quantity,
                "available_shares": available.get(code, 0.0),
                "unsettled_shares": unsettled.get(code, 0.0),
                "mark_price": prices[code], "market_value": quantity * prices[code],
                "actual_weight": quantity * prices[code] / posttrade_nav,
            })
        previous_nav = posttrade_nav

    if expire_open_orders_at_end:
        for order in list(open_orders.values()):
            order["status"] = "EXPIRED"
            order["terminal_reason"] = "end_of_simulation"
            order["terminal_date"] = dates[-1]
            add_event(order, dates[-1], "EXPIRED", "end_of_simulation")

    order_columns = [
        "order_id", "target_date", "submitted_date", "instrument_id", "side",
        "original_shares", "filled_shares", "remaining_shares", "status",
        "terminal_reason", "terminal_date", "age",
    ]
    return ExecutionResult(
        daily=pd.DataFrame(daily_rows),
        orders=pd.DataFrame(all_orders.values(), columns=order_columns),
        events=pd.DataFrame(events), fills=pd.DataFrame(fills),
        positions=pd.DataFrame(position_rows), corporate_actions=pd.DataFrame(action_rows),
    )


def build_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    """Build raw-close execution observations for the frozen primary candidate."""
    lake = lake or Lake()
    cfg = load_config()
    destination = lake.table_path("institutional_execution_market", "v1")
    if destination.exists() and not overwrite:
        return lake.read("institutional_execution_market", "v1")
    targets = lake.read(str(cfg["source_targets"]["table"]), str(cfg["source_targets"]["key"]))
    candidate = cfg["candidate"]
    mask = np.logical_and.reduce([
        targets["model"].eq(candidate["model"]),
        targets["alpha_pipeline"].eq(candidate["alpha_pipeline"]),
        targets["strategy"].eq(candidate["strategy"]),
    ])
    selected = targets.loc[mask]
    codes = sorted(selected["instrument_id"].astype(str).unique())
    start = str(selected["execution_date"].min())
    legacy_market = lake.read(str(cfg["source_market"]["table"]), str(cfg["source_market"]["key"]))
    end = str(legacy_market["trade_date"].max())
    rules = load_config(ROOT / "config" / "constrained_portfolio_daily_execution.yaml")["rules"]
    threshold = float(cfg["corporate_action_proxy"]["activation_absolute_deviation"])
    frames = []
    for number, code in enumerate(codes, start=1):
        frames.append(read_execution_market_rows(
            lake, code, rules=rules, corporate_action_threshold=threshold,
            start_date=start, end_date=end,
        ))
        if number % 500 == 0:
            print(f"institutional market files={number}/{len(codes)}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    lake.write(
        "institutional_execution_market", result, key="v1",
        source="institutional_execution.market.v1",
        request={**cfg, "start": start, "end": end, "codes": len(codes)},
    )
    return result


def _summaries(result: ExecutionResult, periods: int) -> tuple[dict[str, Any], dict[str, Any]]:
    daily = result.daily
    performance = daily_performance_summary(
        daily.rename(columns={"one_way_turnover": "turnover"}).assign(
            blocked_buy_weight=daily["blocked_buy_notional"] / daily["pretrade_nav"],
            blocked_sell_weight=daily["blocked_sell_notional"] / daily["pretrade_nav"],
        ), periods,
    )
    performance.update({
        "scenario": daily["scenario"].iloc[0],
        "total_orders": len(result.orders), "total_fills": len(result.fills),
        "fill_ratio_by_shares": float(result.orders["filled_shares"].sum() / result.orders["original_shares"].sum()),
        "total_cost_cny": float(daily["total_cost"].sum()),
    })
    sell_fills = result.fills.loc[result.fills.get("side", pd.Series(dtype=str)).eq("sell")]
    t1_violations = int((
        sell_fills.get("executed_shares", pd.Series(dtype=float))
        > sell_fills.get("available_before_fill", pd.Series(dtype=float)) + 1e-8
    ).sum())
    order_fill_totals = (
        result.fills.groupby(["portfolio", "order_id"])["executed_shares"].sum()
        if not result.fills.empty else pd.Series(dtype=float)
    )
    orders_indexed = result.orders.set_index("order_id")["filled_shares"]
    order_fill_error = 0.0
    if not order_fill_totals.empty:
        # A single simulation is summarized at a time, so portfolio is constant.
        order_fill_error = float((
            order_fill_totals.droplevel("portfolio")
            - orders_indexed.reindex(order_fill_totals.index.get_level_values("order_id"))
        ).abs().max())
    invariant = {
        "scenario": daily["scenario"].iloc[0],
        "max_abs_nav_reconciliation_error": float(daily["nav_reconciliation_error"].abs().max()),
        "minimum_cash": float(daily["cash"].min()),
        "max_abs_accounting_weight_error": float((daily["accounting_weight_sum"] - 1.0).abs().max()),
        "negative_position_count": int((result.positions.get("shares", pd.Series(dtype=float)) < -1e-8).sum()),
        "negative_cash_day_count": int((daily["cash"] < -1e-6).sum()),
        "t_plus_one_violation_count": t1_violations,
        "max_abs_order_fill_reconciliation_error": order_fill_error,
        "fee_reconciliation_error": float(
            result.fills.get("total_cost", pd.Series(dtype=float)).sum() - daily["total_cost"].sum()
        ),
        "unreconciled_corporate_action_count": int((
            result.corporate_actions.get("value_reconciliation_error", pd.Series(dtype=float)).abs() > 1e-6
        ).sum()),
        "all_invariants_pass": True,
    }
    invariant["all_invariants_pass"] = bool(
        invariant["max_abs_nav_reconciliation_error"] <= 1e-4
        and invariant["minimum_cash"] >= -1e-6
        and invariant["max_abs_accounting_weight_error"] <= 1e-12
        and invariant["negative_position_count"] == 0
        and invariant["negative_cash_day_count"] == 0
        and invariant["t_plus_one_violation_count"] == 0
        and invariant["max_abs_order_fill_reconciliation_error"] <= 1e-8
        and abs(invariant["fee_reconciliation_error"]) <= 1e-6
        and invariant["unreconciled_corporate_action_count"] == 0
    )
    return performance, invariant


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    targets = lake.read(str(cfg["source_targets"]["table"]), str(cfg["source_targets"]["key"]))
    candidate = cfg["candidate"]
    targets = targets.loc[
        targets["model"].eq(candidate["model"])
        & targets["alpha_pipeline"].eq(candidate["alpha_pipeline"])
        & targets["strategy"].eq(candidate["strategy"]),
        ["execution_date", "instrument_id", "target_weight"],
    ].reset_index(drop=True)
    market = build_market(lake)
    results: list[ExecutionResult] = []
    summary_rows, invariant_rows = [], []
    for scenario in cfg["scenarios"]:
        result = simulate_execution(market, targets, cfg, scenario=scenario)
        results.append(result)
        summary, invariant = _summaries(result, int(cfg["annualization_periods"]))
        summary_rows.append(summary)
        invariant_rows.append(invariant)
        print(f"institutional execution scenario={scenario['scenario']} complete", flush=True)

    summary = pd.DataFrame(summary_rows)
    invariants = pd.DataFrame(invariant_rows)
    daily = pd.concat([item.daily for item in results], ignore_index=True)
    orders = pd.concat([
        item.orders.assign(scenario=cfg["scenarios"][i]["scenario"])
        for i, item in enumerate(results)
    ], ignore_index=True)
    events = pd.concat([
        item.events.assign(scenario=cfg["scenarios"][i]["scenario"])
        for i, item in enumerate(results)
    ], ignore_index=True)
    fills = pd.concat([item.fills for item in results], ignore_index=True)
    positions = pd.concat([item.positions for item in results], ignore_index=True)
    actions = pd.concat([item.corporate_actions for item in results], ignore_index=True)
    status = orders.groupby(["scenario", "status", "terminal_reason"], dropna=False).agg(
        orders=("order_id", "size"), original_shares=("original_shares", "sum"),
        filled_shares=("filled_shares", "sum"), remaining_shares=("remaining_shares", "sum"),
    ).reset_index()
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "candidate": "hgb_shallow15_locked/raw/raw__risk_constrained",
        "execution_timing": cfg["return_timing"],
        "daily_bar_approximation": True,
        "intraday_matching_claimed": False,
    }])
    for table, frame in {
        "institutional_execution_protocol": protocol,
        "institutional_execution_daily": daily,
        "institutional_execution_orders": orders,
        "institutional_execution_events": events,
        "institutional_execution_fills": fills,
        "institutional_execution_positions": positions,
        "institutional_execution_corporate_actions": actions,
        "institutional_execution_summary": summary,
        "institutional_execution_invariants": invariants,
        "institutional_execution_order_status": status,
    }.items():
        if frame.empty and table == "institutional_execution_corporate_actions":
            frame = pd.DataFrame([{
                "trade_date": None, "instrument_id": None, "action_source": "no_held_action_observed",
                "value_reconciliation_error": 0.0,
            }])
        lake.write(table, frame, key="v1", source="institutional_execution.v1", request=cfg)

    constrained = summary.set_index("scenario").loc["daily_close_institutional_constraints"]
    ideal = summary.set_index("scenario").loc["market_constraints_removed_upper_bound"]
    lines = [
        "# Phase 5：机构级日频执行引擎", "",
        "## 证据定位", "",
        "本实验仅改变执行层，候选信号、目标权重和 20bp 成本保持冻结。",
        "2021–2026 已被研究者查看，因此结果属于 historical OOS 诊断，不是新的独立证据。", "",
        "## 执行约定", "",
        "- 每日先按未复权收盘价计价，再在收盘执行；收益只计入一次。",
        "- 新目标将旧未成交订单标为 REPLACED；订单可部分成交、受阻重试，并在 5 个交易日后过期。",
        "- 受限场景执行停牌/涨跌停、10% 日成交量参与率、现金、T+1 与申报数量规则。",
        "- 公司行动使用供应商涨跌幅与原始收盘价推导的价值保持因子；这是代理，不是官方公司行动记录。",
        "- 日线无法识别盘中排队、bid/ask、盘口深度或精确成交时刻。", "",
        "## 绩效与成交结果", "", summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 会计不变量", "", invariants.to_markdown(index=False, floatfmt=".10f"), "",
        "## 订单终态", "", status.to_markdown(index=False, floatfmt=".4f"), "",
        "## 诊断结论", "",
        f"受限场景年化收益 {constrained['annualized_return']:.2%}，理想上界 {ideal['annualized_return']:.2%}；"
        f"成交股数比例 {constrained['fill_ratio_by_shares']:.2%}。",
        "该差异衡量日频近似执行摩擦，不应解释为真实券商成交损失。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(invariants.to_string(index=False), flush=True)
    return summary, invariants


def main() -> None:
    parser = argparse.ArgumentParser(description="Institutional daily-bar execution state machine")
    parser.add_argument("command", choices=["build-market", "run", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in {"build-market", "all"}:
        build_market(overwrite=args.overwrite)
    if args.command in {"run", "all"}:
        run()


if __name__ == "__main__":
    main()
