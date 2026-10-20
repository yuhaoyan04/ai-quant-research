"""Daily close execution with persistent share orders and portfolio state."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _execute_share_orders(
    positions: dict[str, float], cash: float, pending: dict[str, float],
    prices: dict[str, float], buy_allowed: dict[str, bool],
    sell_allowed: dict[str, bool], cost_rate: float,
) -> tuple[dict[str, float], float, dict[str, float], dict[str, float], float]:
    """Execute eligible share orders, selling first and reserving fees."""
    sell_orders = {
        code: shares for code, shares in pending.items()
        if shares < 0 and bool(sell_allowed.get(code, False))
    }
    sell_notional = -sum(shares * prices[code] for code, shares in sell_orders.items())
    cash_after_sells = cash + sell_notional
    buy_orders = {
        code: shares for code, shares in pending.items()
        if shares > 0 and bool(buy_allowed.get(code, False))
    }
    requested_buy_notional = sum(
        shares * prices[code] for code, shares in buy_orders.items()
    )
    if requested_buy_notional <= 0:
        buy_notional = 0.0
    else:
        capacity_if_buy_dominant = cash_after_sells / (1.0 + cost_rate)
        capacity_if_sell_dominant = max(
            cash_after_sells - cost_rate * sell_notional, 0.0
        )
        if requested_buy_notional >= sell_notional:
            buy_notional = min(requested_buy_notional, capacity_if_buy_dominant)
        else:
            buy_notional = min(requested_buy_notional, capacity_if_sell_dominant)
        if buy_notional < sell_notional:
            buy_notional = min(requested_buy_notional, capacity_if_sell_dominant)
    buy_scale = (
        buy_notional / requested_buy_notional if requested_buy_notional > 0 else 1.0
    )
    executed = dict(sell_orders)
    executed.update({code: shares * buy_scale for code, shares in buy_orders.items()})
    cost = max(sell_notional, buy_notional) * cost_rate
    post_cash = cash_after_sells - buy_notional - cost
    if post_cash < -1e-10:
        raise RuntimeError("daily execution spent more cash than available")
    post_cash = max(post_cash, 0.0)
    post_positions = positions.copy()
    remaining = pending.copy()
    for code, shares in executed.items():
        post_positions[code] = post_positions.get(code, 0.0) + shares
        remaining[code] = remaining.get(code, 0.0) - shares
    # If no sale is still pending, an allowed buy clipped only by available
    # cash has deployed all fundable capital. Cancel its unaffordable residue;
    # otherwise a price rise would create a fictitious perpetual buy order.
    if not any(shares < -1e-12 for shares in remaining.values()):
        for code in buy_orders:
            remaining.pop(code, None)
    post_positions = {
        code: shares for code, shares in post_positions.items() if abs(shares) > 1e-12
    }
    remaining = {
        code: shares for code, shares in remaining.items() if abs(shares) > 1e-12
    }
    return post_positions, post_cash, remaining, executed, cost


FILL_COLUMNS = [
    "fill_id", "order_id", "trade_date", "active_target_date", "instrument_id",
    "venue", "side", "executed_shares", "synthetic_price", "notional",
    "commission_cost", "stamp_duty_cost", "transfer_fee_cost", "spread_cost",
    "market_impact_cost", "fixed_round_trip_cost", "total_cost",
]


def _venue(instrument_id: str) -> str:
    value = str(instrument_id).lower()
    if value.startswith("sh."):
        return "SSE"
    if value.startswith("sz."):
        return "SZSE"
    return "UNKNOWN"


def _simulate_daily_retry(
    market: pd.DataFrame, targets: pd.DataFrame, round_trip_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Submit weekly target-share orders and retry only their unfilled remainder.

    Returns accrue before close execution. A new weekly target cancels stale
    orders and creates fresh share orders from that close's NAV. Natural price
    drift does not trigger daily rebalancing.
    """
    required_market = {
        "trade_date", "instrument_id", "daily_return", "buy_allowed", "sell_allowed"
    }
    required_targets = {"execution_date", "instrument_id", "target_weight"}
    if missing := required_market - set(market):
        raise KeyError(f"daily market columns missing: {sorted(missing)}")
    if missing := required_targets - set(targets):
        raise KeyError(f"target columns missing: {sorted(missing)}")
    if targets.duplicated(["execution_date", "instrument_id"]).any():
        raise ValueError("duplicate target execution-date/instrument keys")
    target_sums = targets.groupby("execution_date")["target_weight"].sum()
    if not np.allclose(target_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise ValueError("target weights must sum to one on every execution date")

    target_groups = {
        str(date): dict(zip(group["instrument_id"].astype(str),
                            pd.to_numeric(group["target_weight"], errors="raise")))
        for date, group in targets.groupby("execution_date", sort=True)
    }
    prices: dict[str, float] = {}
    positions: dict[str, float] = {}
    pending: dict[str, float] = {}
    cash = 1.0
    active_target: dict[str, float] | None = None
    target_date: str | None = None
    target_age = 0
    nav = 1.0
    cost_rate = float(round_trip_cost_bps) / 10_000.0
    records = []
    fill_records = []
    first_target_date = min(target_groups)

    for raw_date, raw_group in market.groupby("trade_date", sort=True):
        date = str(raw_date)
        if date < first_target_date:
            continue
        group = raw_group.assign(
            instrument_id=raw_group["instrument_id"].astype(str)
        ).set_index("instrument_id")
        returns = pd.to_numeric(group["daily_return"], errors="coerce").to_dict()
        for code, value in returns.items():
            if code not in prices:
                prices[code] = 1.0
            elif pd.notna(value):
                prices[code] *= 1.0 + float(value)

        previous_nav = nav
        pretrade_nav = cash + sum(
            shares * prices.get(code, 1.0) for code, shares in positions.items()
        )
        if pretrade_nav <= 0:
            raise RuntimeError("portfolio value became non-positive before execution")
        gross_return = pretrade_nav / previous_nav - 1.0
        earning_target_date = target_date

        new_target = date in target_groups
        weight_based_one_way_turnover = np.nan
        if new_target:
            active_target = target_groups[date]
            target_date = date
            target_age = 0
            for code in active_target:
                prices.setdefault(code, 1.0)
            pretrade_weights = {
                code: shares * prices.get(code, 1.0) / pretrade_nav
                for code, shares in positions.items()
            }
            weight_based_one_way_turnover = 0.5 * sum(
                abs(active_target.get(code, 0.0) - pretrade_weights.get(code, 0.0))
                for code in set(active_target) | set(pretrade_weights)
            )
            pending = {
                code: (
                    pretrade_nav * active_target.get(code, 0.0)
                    / prices.get(code, 1.0) - positions.get(code, 0.0)
                )
                for code in set(positions) | set(active_target)
            }
            pending = {
                code: shares for code, shares in pending.items() if abs(shares) > 1e-12
            }
        elif active_target is not None:
            target_age += 1

        buy_allowed = group["buy_allowed"].astype(bool).to_dict()
        sell_allowed = group["sell_allowed"].astype(bool).to_dict()
        pretrade_cash = cash
        submitted_order_names = len(pending)
        if active_target is None:
            executed, cost = {}, 0.0
        else:
            positions, cash, pending, executed, cost = _execute_share_orders(
                positions, cash, pending, prices, buy_allowed, sell_allowed, cost_rate
            )
        nav = cash + sum(
            shares * prices.get(code, 1.0) for code, shares in positions.items()
        )
        net_return = nav / previous_nav - 1.0
        if abs((pretrade_nav - cost) - nav) > 1e-8 * max(pretrade_nav, 1.0):
            raise RuntimeError("post-trade NAV does not reconcile with transaction cost")

        buy_notional = sum(
            shares * prices[code] for code, shares in executed.items() if shares > 0
        )
        sell_notional = -sum(
            shares * prices[code] for code, shares in executed.items() if shares < 0
        )
        gross_fill_notional = buy_notional + sell_notional
        ordered_fills = []
        for code, shares in sorted(executed.items()):
            notional = abs(shares * prices[code])
            if notional <= 0:
                continue
            ordered_fills.append((code, shares, notional))
        allocated_fees = [
            cost * notional / gross_fill_notional if gross_fill_notional > 0 else 0.0
            for _, _, notional in ordered_fills
        ]
        if allocated_fees:
            allocated_fees[-1] += cost - sum(allocated_fees)
        for (code, shares, notional), allocated_fee in zip(
            ordered_fills, allocated_fees, strict=True
        ):
            side = "buy" if shares > 0 else "sell"
            fill_records.append({
                "fill_id": f"{date}:{target_date}:{code}",
                "order_id": f"{target_date}:{code}",
                "trade_date": date,
                "active_target_date": target_date,
                "instrument_id": code,
                "venue": _venue(code),
                "side": side,
                "executed_shares": abs(float(shares)),
                "synthetic_price": float(prices[code]),
                "notional": float(notional),
                "commission_cost": 0.0,
                "stamp_duty_cost": 0.0,
                "transfer_fee_cost": 0.0,
                "spread_cost": 0.0,
                "market_impact_cost": 0.0,
                "fixed_round_trip_cost": float(allocated_fee),
                "total_cost": float(allocated_fee),
            })
        blocked_buy = sum(
            shares * prices[code] for code, shares in pending.items() if shares > 0
        ) / pretrade_nav
        blocked_sell = -sum(
            shares * prices[code] for code, shares in pending.items() if shares < 0
        ) / pretrade_nav
        post_weights = {
            code: shares * prices[code] / nav for code, shares in positions.items()
        }
        target_tracking = (sum(
            abs(post_weights.get(code, 0.0) - active_target.get(code, 0.0))
            for code in set(post_weights) | set(active_target)
        ) + cash / nav if active_target is not None else 0.0)
        records.append({
            "trade_date": date, "earning_target_date": earning_target_date,
            "active_target_date": target_date, "new_target": new_target,
            "target_age_trading_days": target_age,
            "gross_return": gross_return, "transaction_cost": cost / previous_nav,
            "net_return": net_return, "nav": nav,
            "previous_nav": previous_nav, "pretrade_nav": pretrade_nav,
            "posttrade_nav": nav,
            "buy_notional": buy_notional, "sell_notional": sell_notional,
            "buy_turnover": buy_notional / pretrade_nav,
            "sell_turnover": sell_notional / pretrade_nav,
            "gross_two_way_turnover": (buy_notional + sell_notional) / pretrade_nav,
            "one_way_turnover": max(buy_notional, sell_notional) / pretrade_nav,
            "weight_based_one_way_turnover": weight_based_one_way_turnover,
            "commission_cost": 0.0, "stamp_duty_cost": 0.0,
            "transfer_fee_cost": 0.0, "spread_cost": 0.0,
            "market_impact_cost": 0.0, "fixed_round_trip_cost": cost,
            "total_cost": cost,
            "buy_weight": buy_notional / pretrade_nav,
            "sell_weight": sell_notional / pretrade_nav,
            "turnover": max(buy_notional, sell_notional) / pretrade_nav,
            "blocked_buy_weight": blocked_buy, "blocked_sell_weight": blocked_sell,
            "target_tracking_l1": target_tracking, "cash_weight": cash / nav,
            "actual_names": len(positions),
            "pending_order_names": len(pending),
            "accounting_weight_sum": cash / nav + sum(post_weights.values()),
            "pretrade_cash_weight": pretrade_cash / pretrade_nav,
            "submitted_order_names": submitted_order_names,
        })
    return pd.DataFrame(records), pd.DataFrame(fill_records, columns=FILL_COLUMNS)


def simulate_daily_retry(
    market: pd.DataFrame, targets: pd.DataFrame, round_trip_cost_bps: float,
) -> pd.DataFrame:
    """Run the legacy-compatible daily retry simulation and return daily rows."""
    daily, _ = _simulate_daily_retry(market, targets, round_trip_cost_bps)
    return daily


def simulate_daily_retry_with_fills(
    market: pd.DataFrame, targets: pd.DataFrame, round_trip_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run daily retry execution and retain its synthetic fill-level audit rows."""
    return _simulate_daily_retry(market, targets, round_trip_cost_bps)


def daily_performance_summary(daily: pd.DataFrame, annualization_periods: int = 252) -> dict:
    returns = pd.to_numeric(daily["net_return"], errors="raise")
    wealth = float((1.0 + returns).prod())
    periods = len(returns)
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization_periods))
    return {
        "periods": periods,
        "terminal_wealth": wealth,
        "annualized_return": wealth ** (annualization_periods / periods) - 1.0,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": float(
            returns.mean() / returns.std(ddof=1) * np.sqrt(annualization_periods)
        ),
        "maximum_drawdown": float(
            ((1.0 + returns).cumprod() / (1.0 + returns).cumprod().cummax() - 1.0).min()
        ),
        "mean_turnover": float(daily["turnover"].mean()),
        "mean_blocked_buy_weight": float(daily["blocked_buy_weight"].mean()),
        "mean_blocked_sell_weight": float(daily["blocked_sell_weight"].mean()),
        "mean_target_tracking_l1": float(daily["target_tracking_l1"].mean()),
        "mean_cash_weight": float(daily["cash_weight"].mean()),
    }
