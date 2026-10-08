"""Next-day suspension/price-limit execution stress test."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.capacity import join_adv20
from quant_research.capacity_execution import build_execution_market
from quant_research.ingest import ROOT, Lake
from quant_research.panel import instrument_type
from quant_research.portfolio import performance_summary


def load_config() -> dict:
    with (ROOT / "config" / "execution_constraints.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def price_limit_pct(code: str, trade_date: str, is_st: bool, rules: dict) -> float:
    if is_st:
        return float(rules["st_limit_pct"])
    if code.startswith("sh.688"):
        return float(rules["star_limit_pct"])
    if code.startswith(("sz.300", "sz.301")):
        key = ("chinext_limit_pct_after_2020_08_24"
               if trade_date >= "2020-08-24" else "chinext_limit_pct_before_2020_08_24")
        return float(rules[key])
    return float(rules["main_board_limit_pct"])


def build_execution_flags(frame: pd.DataFrame, needed_dates: set[str], cfg: dict) -> pd.DataFrame:
    data = frame.sort_values("trade_date").reset_index(drop=True)
    next_date = data["trade_date"].shift(-1)
    next_status = pd.to_numeric(data["trade_status"].shift(-1), errors="coerce")
    next_pct = pd.to_numeric(data["pct_chg"].shift(-1), errors="coerce")
    next_is_st = pd.to_numeric(data["is_st"].shift(-1), errors="coerce").fillna(0).eq(1)
    needed_mask = data["trade_date"].astype(str).isin(needed_dates)
    data = data.loc[needed_mask].copy()
    next_date = next_date.loc[needed_mask]
    next_status = next_status.loc[needed_mask]
    next_pct = next_pct.loc[needed_mask]
    next_is_st = next_is_st.loc[needed_mask]
    code = str(data["instrument_id"].iloc[0])
    limits = np.array([
        price_limit_pct(code, str(date), bool(st), cfg["rules"])
        for date, st in zip(next_date, next_is_st)
    ])
    tolerance = float(cfg["limit_detection_tolerance_percentage_points"])
    status_ok = next_status.eq(1) & next_date.notna()
    result = pd.DataFrame({
        "instrument_id": data["instrument_id"].astype(str),
        "asof_date": data["trade_date"].astype(str),
        "execution_date": next_date,
        "next_trade_status": next_status,
        "next_pct_chg": next_pct,
        "applicable_limit_pct": limits,
        "buy_allowed": status_ok & next_pct.lt(limits - tolerance),
        "sell_allowed": status_ok & next_pct.gt(-limits + tolerance),
    })
    return result


def materialize_flags(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    validation = lake.read("model_validation_predictions", "ridge_v1")
    test = lake.read("model_oos_predictions", "ridge_v1")
    dates = set(pd.concat([validation["asof_date"], test["asof_date"]]).astype(str))
    codes = set(pd.concat([validation["instrument_id"], test["instrument_id"]]).astype(str))
    years = sorted({int(date[:4]) for date in dates})
    buffers: dict[int, list[pd.DataFrame]] = {year: [] for year in years}
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    processed = 0
    for path in files:
        code = path.parent.name.removeprefix("key=").replace("_", ".")
        if code not in codes or instrument_type(code) != "a_share":
            continue
        frame = pd.read_parquet(
            path, columns=["instrument_id", "trade_date", "trade_status", "pct_chg", "is_st"])
        flags = build_execution_flags(frame, dates, cfg)
        if not flags.empty:
            flags["year"] = flags["asof_date"].str.slice(0, 4).astype(int)
            for year, group in flags.groupby("year", sort=False):
                buffers[int(year)].append(group.drop(columns="year"))
        processed += 1
        if processed % 500 == 0:
            print(f"execution flags files={processed}/{len(codes)}", flush=True)
    index_rows = []
    for year, chunks in sorted(buffers.items()):
        result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        lake.write("execution_flags", result, key=str(year), source="execution_constraints.flags.v1", request=cfg)
        index_rows.append({
            "year": year, "rows": len(result),
            "buy_block_rate": 1 - result["buy_allowed"].mean(),
            "sell_block_rate": 1 - result["sell_allowed"].mean(),
        })
        print(f"execution flags year={year} rows={len(result)}", flush=True)
    lake.write("execution_flag_index", pd.DataFrame(index_rows), key="v1",
               source="execution_constraints.flags.v1", request=cfg)


def join_flags(market: pd.DataFrame, lake: Lake) -> pd.DataFrame:
    dated = market.copy()
    dated["year"] = dated["asof_date"].str.slice(0, 4).astype(int)
    frames = []
    for year, group in dated.groupby("year", sort=True):
        flags = lake.read("execution_flags", str(year))
        frames.append(group.drop(columns="year").merge(
            flags[["instrument_id", "asof_date", "execution_date", "buy_allowed", "sell_allowed"]],
            on=["instrument_id", "asof_date"], how="left", validate="one_to_one"))
    result = pd.concat(frames, ignore_index=True)
    result[["buy_allowed", "sell_allowed"]] = result[["buy_allowed", "sell_allowed"]].fillna(False)
    return result


def execute_with_daily_constraints(current: dict[str, float], cash: float,
                                   target_weights: dict[str, float],
                                   buy_allowed: dict[str, bool],
                                   sell_allowed: dict[str, bool]
                                   ) -> tuple[dict[str, float], float, dict[str, float]]:
    codes = set(current) | set(target_weights)
    desired = {code: target_weights.get(code, 0.0) - current.get(code, 0.0) for code in codes}
    trades = {
        code: delta for code, delta in desired.items()
        if delta < 0 and bool(sell_allowed.get(code, False))
    }
    cash_after_sells = cash - sum(value for value in trades.values() if value < 0)
    buys = {
        code: delta for code, delta in desired.items()
        if delta > 0 and bool(buy_allowed.get(code, False))
    }
    total_buys = sum(buys.values())
    scale = min(1.0, cash_after_sells / total_buys) if total_buys > 0 else 1.0
    trades.update({code: value * scale for code, value in buys.items()})
    updated = {code: current.get(code, 0.0) + trades.get(code, 0.0) for code in codes}
    updated = {code: weight for code, weight in updated.items() if weight > 1e-14}
    new_cash = cash_after_sells - sum(value for value in trades.values() if value > 0)
    return updated, new_cash, trades


def simulate(market: pd.DataFrame, holdings: pd.DataFrame, target: str,
             cost_bps: float) -> pd.DataFrame:
    markets = {
        str(date): group.assign(instrument_id=group["instrument_id"].astype(str)).set_index("instrument_id")
        for date, group in market.groupby("asof_date", sort=True)
    }
    targets = {str(date): group for date, group in holdings.groupby("asof_date", sort=True)}
    current: dict[str, float] = {}
    cash = 1.0
    records = []
    for date in sorted(targets):
        group = markets.get(date)
        if group is None:
            continue
        target_weights = dict(zip(targets[date]["instrument_id"].astype(str),
                                  targets[date]["target_weight"].astype(float)))
        buy_allowed = group["buy_allowed"].astype(bool).to_dict()
        sell_allowed = group["sell_allowed"].astype(bool).to_dict()
        desired = {code: target_weights.get(code, 0.0) - current.get(code, 0.0)
                   for code in set(current) | set(target_weights)}
        post, post_cash, trades = execute_with_daily_constraints(
            current, cash, target_weights, buy_allowed, sell_allowed)
        blocked_buy = sum(delta for code, delta in desired.items()
                          if delta > 0 and not buy_allowed.get(code, False))
        blocked_sell = -sum(delta for code, delta in desired.items()
                            if delta < 0 and not sell_allowed.get(code, False))
        buys = sum(value for value in trades.values() if value > 0)
        sells = -sum(value for value in trades.values() if value < 0)
        turnover = max(buys, sells)
        transaction_cost = turnover * cost_bps / 10_000
        returns = pd.to_numeric(group[target], errors="coerce").to_dict()
        gross_return = sum(weight * (float(returns[code]) if pd.notna(returns.get(code, np.nan)) else 0.0)
                           for code, weight in post.items())
        benchmark_mask = group.get("benchmark_eligible", pd.Series(True, index=group.index)).fillna(False)
        benchmark_return = float(pd.to_numeric(group.loc[benchmark_mask, target], errors="coerce").fillna(0).mean())
        net_return = gross_return - transaction_cost
        tracking_l1 = sum(abs(post.get(code, 0.0) - target_weights.get(code, 0.0))
                          for code in set(post) | set(target_weights)) + abs(post_cash)
        records.append({
            "asof_date": date, "cash_weight": post_cash, "turnover": turnover,
            "blocked_buy_weight": blocked_buy, "blocked_sell_weight": blocked_sell,
            "target_tracking_l1": tracking_l1, "gross_return": gross_return,
            "transaction_cost": transaction_cost, "net_return": net_return,
            "benchmark_return": benchmark_return,
            "gross_excess_return": gross_return - benchmark_return,
            "net_excess_return": net_return - benchmark_return,
            "realized_weight_fraction": 1.0, "included_in_performance": True,
        })
        grown = {code: weight * (1 + (float(returns[code]) if pd.notna(returns.get(code, np.nan)) else 0.0))
                 for code, weight in post.items()}
        total = post_cash + sum(grown.values())
        current = {code: value / total for code, value in grown.items()}
        cash = post_cash / total
    return pd.DataFrame(records)


def analyze(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    sources = {
        "ridge": ("model_oos_predictions", "ridge_v1", "turnover_buffer_oos_holdings", "v1"),
        "hgb_ridge_blend": ("model_challenger_oos_predictions", "hgb_v1",
                            "challenger_portfolio_oos_holdings", "v1"),
    }
    weekly_frames, rows = [], []
    for model_name, (prediction_table, prediction_key, holding_table, holding_key) in sources.items():
        predictions = lake.read(prediction_table, prediction_key)
        holdings = lake.read(holding_table, holding_key)
        market = join_adv20(predictions, lake)
        market = build_execution_market(market, holdings, "fwd_return_5d_lag1", lake)
        market = join_flags(market, lake)
        weekly = simulate(market, holdings, "fwd_return_5d_lag1", float(cfg["round_trip_cost_bps"]))
        weekly["model"] = model_name
        weekly_frames.append(weekly)
        rows.append({
            "model": model_name,
            **performance_summary(weekly, int(cfg["annualization_periods"])),
            "mean_blocked_buy_weight": weekly["blocked_buy_weight"].mean(),
            "mean_blocked_sell_weight": weekly["blocked_sell_weight"].mean(),
            "mean_cash_weight": weekly["cash_weight"].mean(),
            "mean_target_tracking_l1": weekly["target_tracking_l1"].mean(),
        })
    summary = pd.DataFrame(rows)
    lake.write("execution_constraint_weekly", pd.concat(weekly_frames, ignore_index=True), key="v1",
               source="execution_constraints.v1", request=cfg)
    lake.write("execution_constraint_summary", summary, key="v1",
               source="execution_constraints.v1", request=cfg)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build execution flags or analyze constraints")
    parser.add_argument("stage", choices=["build-flags", "analyze"])
    args = parser.parse_args()
    if args.stage == "build-flags":
        materialize_flags()
    else:
        analyze()


if __name__ == "__main__":
    main()
