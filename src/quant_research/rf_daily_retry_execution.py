"""Build RF daily execution markets and run persistent-order close simulations."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.daily_execution import daily_performance_summary, simulate_daily_retry
from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "rf_daily_retry_execution.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _date_bounds(holdings: pd.DataFrame, calendar: pd.DataFrame,
                 horizon: int) -> tuple[str, str]:
    open_days = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
    ).sort_values().drop_duplicates().reset_index(drop=True)
    last_signal = pd.Timestamp(holdings["signal_date"].max())
    later = open_days.loc[open_days.gt(last_signal)]
    if len(later) < horizon + 1:
        end = open_days.iloc[-1]
    else:
        # Label convention is t+2...t+6 for h=5.
        end = later.iloc[horizon]
    return str(holdings["execution_date"].min()), end.strftime("%Y-%m-%d")


def _limit_array(code: str, dates: pd.Series, is_st: pd.Series, rules: dict) -> np.ndarray:
    if code.startswith("sh.688"):
        base = np.full(len(dates), float(rules["star_limit_pct"]))
    elif code.startswith(("sz.300", "sz.301")):
        base = np.where(
            dates.astype(str).ge("2020-08-24"),
            float(rules["chinext_limit_pct_after_2020_08_24"]),
            float(rules["chinext_limit_pct_before_2020_08_24"]),
        )
    else:
        base = np.full(len(dates), float(rules["main_board_limit_pct"]))
    return np.where(is_st, float(rules["st_limit_pct"]), base)


def build_market(split: str, lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    key = str(cfg["target_holdings"][split])
    destination_key = f"{split}_v1"
    if lake.table_path("rf_daily_execution_market", destination_key).exists() and not overwrite:
        return lake.read("rf_daily_execution_market", destination_key)
    holdings = lake.read("rf_target_holdings", key)
    calendar = lake.read("calendar", "SSE")
    start, end = _date_bounds(holdings, calendar, int(cfg["label_horizon_trading_days"]))
    codes = set(holdings["instrument_id"].astype(str))
    tolerance = float(cfg["limit_detection_tolerance_percentage_points"])
    rules = cfg["rules"]
    frames = []
    for number, code in enumerate(sorted(codes), start=1):
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", code.replace(".", "_")),
            columns=["instrument_id", "trade_date", "pct_chg", "trade_status", "is_st"],
        )
        bars = bars.loc[bars["trade_date"].between(start, end)].copy()
        if bars.empty:
            continue
        pct = pd.to_numeric(bars["pct_chg"], errors="coerce")
        status = pd.to_numeric(bars["trade_status"], errors="coerce").eq(1)
        is_st = pd.to_numeric(bars["is_st"], errors="coerce").fillna(0).eq(1)
        limits = _limit_array(code, bars["trade_date"], is_st, rules)
        bars["daily_return"] = pct / 100.0
        bars["buy_allowed"] = status & pct.lt(limits - tolerance)
        bars["sell_allowed"] = status & pct.gt(-limits + tolerance)
        frames.append(bars[[
            "instrument_id", "trade_date", "daily_return", "buy_allowed", "sell_allowed"
        ]])
        if number % 500 == 0:
            print(f"rf daily market split={split} files={number}/{len(codes)}", flush=True)
    market = pd.concat(frames, ignore_index=True)
    request = {**cfg, "split": split, "start": start, "end": end,
               "target_code_count": len(codes)}
    lake.write("rf_daily_execution_market", market, key=destination_key,
               source="rf_daily_retry_execution.market.v1", request=request)
    print(f"rf daily market split={split} rows={len(market):,} "
          f"dates={market['trade_date'].nunique()} codes={market['instrument_id'].nunique()}",
          flush=True)
    return market


def _ideal_fill_market(market: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Dense upper-bound market: missing prints return zero but fills are allowed."""
    dates = pd.DataFrame({"trade_date": sorted(market["trade_date"].unique())})
    codes = pd.DataFrame({"instrument_id": sorted(holdings["instrument_id"].unique())})
    dates["_join"] = 1
    codes["_join"] = 1
    dense = dates.merge(codes, on="_join").drop(columns="_join").merge(
        market[["trade_date", "instrument_id", "daily_return"]],
        on=["trade_date", "instrument_id"], how="left", validate="one_to_one",
    )
    dense["daily_return"] = dense["daily_return"].fillna(0.0)
    dense["buy_allowed"] = True
    dense["sell_allowed"] = True
    return dense


def _summaries(daily: pd.DataFrame, split: str, scenario: str,
               periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([{
        "split": split, "scenario": scenario,
        **daily_performance_summary(daily, periods),
    }])
    annual_source = daily.assign(year=daily["trade_date"].str[:4].astype(int))
    annual = pd.DataFrame([
        {"split": split, "scenario": scenario, "year": int(year),
         **daily_performance_summary(group, periods)}
        for year, group in annual_source.groupby("year", sort=True)
    ])
    return summary, annual


def _timing_audit(holdings: pd.DataFrame, daily: pd.DataFrame,
                  split: str, horizon: int) -> pd.DataFrame:
    dates = holdings[["signal_date", "execution_date"]].drop_duplicates().sort_values(
        "execution_date"
    )
    earning_days = int(daily["earning_target_date"].notna().sum())
    nominal_days = int(len(dates) * horizon)
    return pd.DataFrame([{
        "split": split, "target_dates": len(dates),
        "nominal_overlapping_label_days": nominal_days,
        "realized_nonoverlap_holding_days": earning_days,
        "duplicated_horizon_days": nominal_days - earning_days,
        "first_execution_date": dates["execution_date"].min(),
        "last_execution_date": dates["execution_date"].max(),
    }])


def run_split(split: str, lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    holdings = lake.read("rf_target_holdings", str(cfg["target_holdings"][split])).rename(
        columns={"execution_date": "execution_date"}
    )
    market = build_market(split, lake)
    daily = simulate_daily_retry(
        market, holdings[["execution_date", "instrument_id", "target_weight"]],
        float(cfg["round_trip_cost_bps"]),
    )
    periods = int(cfg["annualization_periods"])
    summary, annual = _summaries(
        daily, split, "daily_retry_limit_and_suspension_constrained", periods
    )
    ideal_daily = simulate_daily_retry(
        _ideal_fill_market(market, holdings),
        holdings[["execution_date", "instrument_id", "target_weight"]],
        float(cfg["round_trip_cost_bps"]),
    )
    ideal_summary, ideal_annual = _summaries(
        ideal_daily, split, "nonoverlap_all_fills_upper_bound", periods
    )
    timing = _timing_audit(
        holdings, daily, split, int(cfg["label_horizon_trading_days"])
    )
    request = {**cfg, "split": split}
    lake.write("rf_daily_retry_daily", daily, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_retry_summary", summary, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_retry_annual", annual, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_ideal_daily", ideal_daily, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_ideal_summary", ideal_summary, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_ideal_annual", ideal_annual, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    lake.write("rf_daily_timing_audit", timing, key=f"{split}_v1",
               source="rf_daily_retry_execution.v1", request=request)
    print(pd.concat([ideal_summary, summary], ignore_index=True).to_string(index=False),
          flush=True)
    print(annual.to_string(index=False), flush=True)
    return summary, annual


def main() -> None:
    parser = argparse.ArgumentParser(description="RF daily retry execution research")
    parser.add_argument("command", choices=["build-market", "run", "all"])
    parser.add_argument("--split", choices=["validation", "historical_oos", "both"],
                        default="both")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    splits = (["validation", "historical_oos"] if args.split == "both" else [args.split])
    for split in splits:
        if args.command in ("build-market", "all"):
            build_market(split, overwrite=args.overwrite)
        if args.command in ("run", "all"):
            run_split(split)


if __name__ == "__main__":
    main()
