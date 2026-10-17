"""Daily executable audit of the PIT risk-constrained target portfolios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_research.daily_execution import daily_performance_summary, simulate_daily_retry
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import weekly_rows
from quant_research.rf_daily_retry_execution import _ideal_fill_market, _limit_array
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "constrained_portfolio_daily_execution.yaml"
REPORT_PATH = ROOT / "reports" / "constrained_portfolio_daily_execution.md"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_targets(holdings: pd.DataFrame, date_map: pd.DataFrame,
                    construction: str) -> pd.DataFrame:
    """Attach the label's t+1 execution date to frozen target weights."""
    selected = holdings.loc[holdings["construction"].eq(construction)].copy()
    dates = date_map[["asof_date", "available_date"]].drop_duplicates()
    if dates.duplicated("asof_date").any():
        raise ValueError("one signal date maps to multiple execution dates")
    selected = selected.merge(dates, on="asof_date", how="left", validate="many_to_one")
    if selected["available_date"].isna().any():
        missing = sorted(selected.loc[selected["available_date"].isna(), "asof_date"].unique())
        raise ValueError(f"execution dates missing for signals: {missing[:10]}")
    selected = selected.rename(columns={
        "asof_date": "signal_date", "available_date": "execution_date",
    })
    keys = ["model", "alpha_pipeline", "strategy", "execution_date"]
    if selected.duplicated([*keys, "instrument_id"]).any():
        raise ValueError("duplicate target execution-date/instrument keys")
    sums = selected.groupby(keys)["target_weight"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise ValueError("target weights do not sum to one")
    return selected[[
        "instrument_id", "signal_date", "execution_date", "target_weight",
        "model", "alpha_pipeline", "strategy", "test_year",
    ]].sort_values(keys + ["instrument_id"]).reset_index(drop=True)


def build_targets(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    destination = lake.table_path("constrained_execution_targets", "v1")
    if destination.exists() and not overwrite:
        return lake.read("constrained_execution_targets", "v1")
    source = cfg["source_holdings"]
    holdings = lake.read(str(source["table"]), str(source["key"]))
    years = sorted(holdings["asof_date"].astype(str).str[:4].astype(int).unique())
    date_frames = []
    for year in years:
        panel = weekly_rows(pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["asof_date", "available_date"],
        ))
        date_frames.append(panel[["asof_date", "available_date"]].drop_duplicates())
    targets = prepare_targets(
        holdings, pd.concat(date_frames, ignore_index=True),
        str(cfg["target_construction"]),
    )
    lake.write(
        "constrained_execution_targets", targets, key="v1",
        source="constrained_portfolio_daily_execution.targets.v1", request=cfg,
    )
    print(
        f"constrained execution targets rows={len(targets):,} "
        f"dates={targets['execution_date'].nunique()} codes={targets['instrument_id'].nunique()}",
        flush=True,
    )
    return targets


def _market_end(targets: pd.DataFrame, calendar: pd.DataFrame, horizon: int) -> str:
    open_days = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
    ).sort_values().drop_duplicates().reset_index(drop=True)
    last_signal = pd.Timestamp(targets["signal_date"].max())
    later = open_days.loc[open_days.gt(last_signal)]
    return later.iloc[min(horizon, len(later) - 1)].strftime("%Y-%m-%d")


def build_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    destination = lake.table_path("constrained_daily_execution_market", "v1")
    if destination.exists() and not overwrite:
        return lake.read("constrained_daily_execution_market", "v1")
    targets = build_targets(lake)
    calendar = lake.read("calendar", "SSE")
    start = str(targets["execution_date"].min())
    end = _market_end(targets, calendar, int(cfg["label_horizon_trading_days"]))
    codes = sorted(targets["instrument_id"].astype(str).unique())
    tolerance = float(cfg["limit_detection_tolerance_percentage_points"])
    rules = cfg["rules"]
    frames = []
    for number, code in enumerate(codes, start=1):
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
            "instrument_id", "trade_date", "daily_return", "buy_allowed", "sell_allowed",
        ]])
        if number % 500 == 0:
            print(f"constrained daily market files={number}/{len(codes)}", flush=True)
    market = pd.concat(frames, ignore_index=True)
    request = {**cfg, "start": start, "end": end, "target_code_count": len(codes)}
    lake.write(
        "constrained_daily_execution_market", market, key="v1",
        source="constrained_portfolio_daily_execution.market.v1", request=request,
    )
    print(
        f"constrained daily market rows={len(market):,} "
        f"dates={market['trade_date'].nunique()} codes={market['instrument_id'].nunique()}",
        flush=True,
    )
    return market


def _summary(daily: pd.DataFrame, model: str, pipeline: str, strategy: str,
             scenario: str, annualization: int) -> dict:
    return {
        "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
        "scenario": scenario, **daily_performance_summary(daily, annualization),
    }


def paired_execution_bootstrap(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records = []
    boot = cfg["bootstrap"]
    for keys, group in daily.groupby(["model", "alpha_pipeline", "strategy"], sort=False):
        pivot = group.pivot(index="trade_date", columns="scenario", values="net_return")
        difference = (
            pivot["daily_retry_limit_and_suspension_constrained"]
            - pivot["nonoverlap_all_fills_upper_bound"]
        ).to_numpy(dtype="float64")
        estimates = circular_block_bootstrap_mean(
            difference, int(boot["block_length_trading_days"]),
            int(boot["samples"]), int(boot["random_seed"]),
        ) * int(cfg["annualization_periods"])
        records.append({
            "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
            "metric": "constrained_minus_ideal_net_return_annualized_arithmetic",
            "observations": len(difference),
            "point_estimate": float(difference.mean() * int(cfg["annualization_periods"])),
            "ci_2_5pct": float(np.quantile(estimates, 0.025)),
            "ci_97_5pct": float(np.quantile(estimates, 0.975)),
            "probability_delta_nonpositive": float(
                (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
            ),
        })
    return pd.DataFrame(records)


def _write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    lines = [
        "# 风险约束组合的日频可成交审计", "", "## 设计", "",
        "- 使用冻结的风险约束目标权重，不重新选择模型、alpha 或组合参数。",
        "- 信号后首个交易日收盘提交股数订单；停牌、涨停买入和跌停卖出订单逐日重试。",
        "- 新周目标取消旧未成交订单；自然价格漂移不会触发隐藏的日频再平衡。",
        "- 理想成交上界与受限场景共享唯一非重叠日频净值时钟和 20bp 成本。", "",
        "## 2021–2026 historical OOS", "",
        summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 受限成交减理想成交的配对区间", "",
        bootstrap.to_markdown(index=False, floatfmt=".6f"), "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    targets = build_targets(lake)
    market = build_market(lake)
    ideal_market = _ideal_fill_market(market, targets)
    daily_frames, summaries, annual_records = [], [], []
    periods = int(cfg["annualization_periods"])
    group_columns = ["model", "alpha_pipeline", "strategy"]
    for keys, group in targets.groupby(group_columns, sort=False):
        target = group[["execution_date", "instrument_id", "target_weight"]]
        scenarios = {
            "daily_retry_limit_and_suspension_constrained": market,
            "nonoverlap_all_fills_upper_bound": ideal_market,
        }
        for scenario, scenario_market in scenarios.items():
            daily = simulate_daily_retry(
                scenario_market, target, float(cfg["round_trip_cost_bps"])
            )
            daily.insert(0, "scenario", scenario)
            daily.insert(0, "strategy", keys[2])
            daily.insert(0, "alpha_pipeline", keys[1])
            daily.insert(0, "model", keys[0])
            daily_frames.append(daily)
            summaries.append(_summary(daily, *keys, scenario, periods))
            annual_source = daily.assign(year=daily["trade_date"].str[:4].astype(int))
            for year, annual_group in annual_source.groupby("year", sort=True):
                annual_records.append({
                    "model": keys[0], "alpha_pipeline": keys[1],
                    "strategy": keys[2], "scenario": scenario, "year": int(year),
                    **daily_performance_summary(annual_group, periods),
                })
        print(f"constrained daily execution model={keys[0]} pipeline={keys[1]}", flush=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    annual = pd.DataFrame(annual_records)
    bootstrap = paired_execution_bootstrap(daily, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "historical_oos_used_for_selection": False,
        "source_experiment": cfg["source_experiment"],
        "unique_nonoverlap_daily_clock": True,
        "persistent_blocked_share_orders": True,
    }])
    request = {**cfg, "strategies": summary["strategy"].unique().tolist()}
    for table, frame in {
        "constrained_daily_execution_protocol": protocol,
        "constrained_daily_execution_daily": daily,
        "constrained_daily_execution_summary": summary,
        "constrained_daily_execution_annual": annual,
        "constrained_daily_execution_bootstrap": bootstrap,
    }.items():
        lake.write(table, frame, key="v1",
                   source="constrained_portfolio_daily_execution.v1", request=request)
    _write_report(summary, bootstrap)
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    return summary, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily execution audit for constrained portfolios")
    parser.add_argument("command", choices=["build-targets", "build-market", "run", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in ("build-targets", "all"):
        build_targets(overwrite=args.overwrite)
    if args.command in ("build-market", "all"):
        build_market(overwrite=args.overwrite)
    if args.command in ("run", "all"):
        run()


if __name__ == "__main__":
    main()
