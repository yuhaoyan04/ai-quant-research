"""Dynamic, participation-constrained execution of the selected portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.capacity import join_adv20
from quant_research.ingest import ROOT, Lake
from quant_research.liquidity_stress import enrich_liquidity
from quant_research.portfolio import performance_summary


def load_config() -> dict:
    with (ROOT / "config" / "capacity_execution.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_execution_market(market: pd.DataFrame, target_holdings: pd.DataFrame,
                           target: str, lake: Lake) -> pd.DataFrame:
    """Add prior target names that dropped out of the current model universe.

    Capacity exposures provide the same-date ADV needed to liquidate stale
    positions. The eligible panel supplies returns and volatility where they
    remain observable. Missing returns (primarily suspensions/end-of-sample)
    stay explicit and are marked to zero by the simulator.
    """
    relevant_codes = set(target_holdings["instrument_id"].astype(str))
    needed_dates = set(target_holdings["asof_date"].astype(str))
    base = market.copy()
    base["benchmark_eligible"] = True
    frames = [base]
    base_keys = set(zip(base["instrument_id"].astype(str), base["asof_date"].astype(str)))
    for year in sorted({int(date[:4]) for date in needed_dates}):
        year_dates = {date for date in needed_dates if int(date[:4]) == year}
        exposure = lake.read("capacity_exposures", str(year))
        exposure = exposure.loc[
            exposure["instrument_id"].astype(str).isin(relevant_codes)
            & exposure["asof_date"].astype(str).isin(year_dates)
        ].copy()
        if exposure.empty:
            continue
        stage = pd.read_parquet(
            lake.table_path("eligible_factor_panel_staging", str(year)),
            columns=["instrument_id", "asof_date", "volatility_20d", target],
        )
        stage = stage.loc[
            stage["instrument_id"].astype(str).isin(relevant_codes)
            & stage["asof_date"].astype(str).isin(year_dates)
        ]
        extra = exposure.merge(stage, on=["instrument_id", "asof_date"], how="left",
                               validate="one_to_one")
        keys = list(zip(extra["instrument_id"].astype(str), extra["asof_date"].astype(str)))
        extra = extra.loc[[key not in base_keys for key in keys]].copy()
        extra["benchmark_eligible"] = False
        frames.append(extra)
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.drop_duplicates(["instrument_id", "asof_date"], keep="first")


def execute_toward_target(
    current_weights: dict[str, float],
    cash_weight: float,
    target_weights: dict[str, float],
    adv_by_code: dict[str, float],
    nav_cny: float,
    participation_limit: float,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Clip trades by ADV, sell first, and never spend unavailable cash."""
    codes = set(current_weights) | set(target_weights)
    desired = {code: target_weights.get(code, 0.0) - current_weights.get(code, 0.0)
               for code in codes}
    cap_weight = {
        code: participation_limit * max(float(adv_by_code.get(code, 0.0)), 0.0) / nav_cny
        if nav_cny > 0 else 0.0
        for code in codes
    }
    trades: dict[str, float] = {}
    for code, delta in desired.items():
        if delta < 0:
            trades[code] = max(delta, -cap_weight[code])
    cash_after_sells = cash_weight - sum(value for value in trades.values() if value < 0)

    requested_buys = {
        code: min(delta, cap_weight[code])
        for code, delta in desired.items() if delta > 0
    }
    total_buys = sum(requested_buys.values())
    buy_scale = min(1.0, cash_after_sells / total_buys) if total_buys > 0 else 1.0
    trades.update({code: amount * buy_scale for code, amount in requested_buys.items()})

    updated = {
        code: current_weights.get(code, 0.0) + trades.get(code, 0.0)
        for code in codes
    }
    updated = {code: weight for code, weight in updated.items() if weight > 1e-14}
    new_cash = cash_after_sells - sum(value for value in trades.values() if value > 0)
    return updated, new_cash, trades


def simulate_capacity_constrained(
    market: pd.DataFrame,
    target_holdings: pd.DataFrame,
    target: str,
    aum_cny: float,
    participation_limit: float,
    round_trip_cost_bps: float,
    impact_eta: float,
    minimum_realized_weight_fraction: float,
    prepared_market: dict[str, pd.DataFrame] | None = None,
    prepared_targets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Carry unfilled orders implicitly by moving toward each new weekly target."""
    market_groups = prepared_market or {
        str(date): group.assign(instrument_id=group["instrument_id"].astype(str)).set_index("instrument_id")
        for date, group in market.groupby("asof_date", sort=True)
    }
    target_groups = prepared_targets or {
        str(date): group for date, group in target_holdings.groupby("asof_date", sort=True)
    }
    current_weights: dict[str, float] = {}
    cash_weight = 1.0
    nav_cny = float(aum_cny)
    records: list[dict] = []
    fixed_cost_rate = round_trip_cost_bps / 10_000

    for asof_date in sorted(target_groups):
        group = market_groups.get(asof_date)
        if group is None or group.empty:
            continue
        targets = target_groups[asof_date]
        target_weights = dict(zip(
            targets["instrument_id"].astype(str),
            pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0),
        ))
        indexed = group
        adv_by_code = pd.to_numeric(indexed["adv20_cny"], errors="coerce").fillna(0.0).to_dict()
        volatility_by_code = pd.to_numeric(
            indexed["volatility_20d"], errors="coerce").fillna(0.0).to_dict()
        post_trade, post_cash, trades = execute_toward_target(
            current_weights, cash_weight, target_weights, adv_by_code,
            nav_cny, participation_limit)

        buy_weight = sum(value for value in trades.values() if value > 0)
        sell_weight = -sum(value for value in trades.values() if value < 0)
        turnover = max(buy_weight, sell_weight)
        fixed_cost = turnover * fixed_cost_rate
        impact_cost = 0.0
        missing_adv_trade_weight = 0.0
        for code, trade_weight in trades.items():
            absolute_trade_cny = abs(trade_weight) * nav_cny
            adv = max(float(adv_by_code.get(code, 0.0)), 0.0)
            if absolute_trade_cny > 0 and adv <= 0:
                missing_adv_trade_weight += abs(trade_weight)
                continue
            if absolute_trade_cny > 0:
                impact_fraction = impact_eta * float(volatility_by_code.get(code, 0.0)) * np.sqrt(
                    absolute_trade_cny / adv)
                impact_cost += abs(trade_weight) * impact_fraction

        return_by_code = pd.to_numeric(indexed[target], errors="coerce").to_dict()
        risky_weight = sum(post_trade.values())
        realized_risky_weight = sum(
            weight for code, weight in post_trade.items() if pd.notna(return_by_code.get(code, np.nan)))
        realized_coverage = realized_risky_weight / risky_weight if risky_weight > 1e-14 else 1.0
        gross_return = sum(
            weight * (float(return_by_code[code]) if pd.notna(return_by_code.get(code, np.nan)) else 0.0)
            for code, weight in post_trade.items()
        )
        net_return = gross_return - fixed_cost - impact_cost
        benchmark_mask = group.get("benchmark_eligible", pd.Series(True, index=group.index)).fillna(False)
        benchmark_returns = pd.to_numeric(group.loc[benchmark_mask, target], errors="coerce")
        benchmark_return = float(benchmark_returns.fillna(0.0).mean())
        tracking_l1 = sum(
            abs(post_trade.get(code, 0.0) - target_weights.get(code, 0.0))
            for code in set(post_trade) | set(target_weights)
        ) + abs(post_cash)
        desired_l1 = sum(
            abs(target_weights.get(code, 0.0) - current_weights.get(code, 0.0))
            for code in set(current_weights) | set(target_weights)
        ) + abs(cash_weight)
        executed_l1 = sum(abs(value) for value in trades.values()) + abs(post_cash - cash_weight)
        fill_ratio = executed_l1 / desired_l1 if desired_l1 > 1e-14 else 1.0
        records.append({
            "asof_date": asof_date,
            "aum_initial_cny": float(aum_cny),
            "nav_cny_before_return": nav_cny,
            "participation_limit": participation_limit,
            "target_names": len(target_weights),
            "actual_names": len(post_trade),
            "cash_weight": post_cash,
            "risky_weight": risky_weight,
            "turnover": turnover,
            "order_fill_ratio": fill_ratio,
            "target_tracking_l1": tracking_l1,
            "fixed_transaction_cost": fixed_cost,
            "impact_cost": impact_cost,
            "gross_return": gross_return,
            "transaction_cost": fixed_cost + impact_cost,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "gross_excess_return": gross_return - benchmark_return,
            "net_excess_return": net_return - benchmark_return,
            "realized_risky_weight": realized_risky_weight,
            "realized_weight_fraction": realized_coverage,
            "missing_adv_trade_weight": missing_adv_trade_weight,
            "coverage_pass": realized_coverage >= minimum_realized_weight_fraction,
            # Missing held-name returns are explicitly marked to zero above.
            # Keep every chronological period to avoid scenario-dependent
            # selection of the backtest sample.
            "included_in_performance": True,
        })

        grown = {
            code: weight * (1 + (float(return_by_code[code])
                                      if pd.notna(return_by_code.get(code, np.nan)) else 0.0))
            for code, weight in post_trade.items()
        }
        gross_total = post_cash + sum(grown.values())
        if gross_total <= 0:
            raise RuntimeError("portfolio gross value became non-positive")
        current_weights = {code: value / gross_total for code, value in grown.items()}
        cash_weight = post_cash / gross_total
        nav_cny *= 1 + net_return
    return pd.DataFrame(records)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    model_selection = lake.read("model_selection", "ridge_v1")
    holdings = lake.read("turnover_buffer_oos_holdings", "v1")
    if model_selection.empty or holdings.empty:
        raise RuntimeError("model selection or buffered holdings missing")
    target = str(model_selection.iloc[0]["target"])
    market = enrich_liquidity(lake.read("model_oos_predictions", "ridge_v1"), lake)
    market = join_adv20(market, lake)
    market = build_execution_market(market, holdings, target, lake)
    prepared_market = {
        str(date): group.assign(instrument_id=group["instrument_id"].astype(str)).set_index("instrument_id")
        for date, group in market.groupby("asof_date", sort=True)
    }
    prepared_targets = {
        str(date): group for date, group in holdings.groupby("asof_date", sort=True)
    }

    weekly_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for aum in cfg["aum_cny"]:
        for participation in cfg["adv_participation_limits"]:
            weekly = simulate_capacity_constrained(
                market, holdings, target, float(aum), float(participation),
                float(cfg["round_trip_cost_bps"]), float(cfg["square_root_impact_eta"]),
                float(cfg["minimum_realized_weight_fraction"]),
                prepared_market, prepared_targets,
            )
            weekly_frames.append(weekly)
            summary = performance_summary(weekly, int(cfg["annualization_periods"]))
            summaries.append({
                "aum_cny": float(aum),
                "participation_limit": float(participation),
                **summary,
                "mean_order_fill_ratio": weekly["order_fill_ratio"].mean(),
                "p10_order_fill_ratio": weekly["order_fill_ratio"].quantile(0.10),
                "mean_cash_weight": weekly["cash_weight"].mean(),
                "mean_target_tracking_l1": weekly["target_tracking_l1"].mean(),
                "mean_fixed_cost_bps": weekly["fixed_transaction_cost"].mean() * 10_000,
                "mean_impact_cost_bps": weekly["impact_cost"].mean() * 10_000,
            })
    summary_frame = pd.DataFrame(summaries)
    lake.write("capacity_execution_weekly", pd.concat(weekly_frames, ignore_index=True), key="v1",
               source="capacity_execution.v1", request=cfg)
    lake.write("capacity_execution_summary", summary_frame, key="v1",
               source="capacity_execution.v1", request=cfg)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    run()
