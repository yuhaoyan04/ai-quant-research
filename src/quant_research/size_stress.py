"""Point-in-time float-cap proxy and small-size dependence stress tests."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.liquidity_stress import enrich_liquidity
from quant_research.panel import instrument_type
from quant_research.portfolio import performance_summary
from quant_research.turnover_buffer import _screen_parameters, simulate_buffered_long_only


def load_config() -> dict:
    with (ROOT / "config" / "size_stress.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def materialize_size_proxy(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    validation = lake.read("model_validation_predictions", "ridge_v1")
    test = lake.read("model_oos_predictions", "ridge_v1")
    dates = set(pd.concat([validation["asof_date"], test["asof_date"]]).astype(str))
    relevant_codes = set(pd.concat([validation["instrument_id"], test["instrument_id"]]).astype(str))
    years = sorted({int(date[:4]) for date in dates})
    buffers: dict[int, list[pd.DataFrame]] = {year: [] for year in years}
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    processed = 0
    for path in files:
        code = path.parent.name.removeprefix("key=").replace("_", ".")
        if code not in relevant_codes or instrument_type(code) != "a_share":
            continue
        frame = pd.read_parquet(path, columns=["instrument_id", "trade_date", "amount", "turn"])
        frame = frame.loc[frame["trade_date"].isin(dates)].copy()
        if frame.empty:
            continue
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        turnover_fraction = pd.to_numeric(frame["turn"], errors="coerce") / 100
        frame["float_cap_proxy_cny"] = np.divide(
            amount, turnover_fraction,
            out=np.full(len(frame), np.nan, dtype="float64"),
            where=(turnover_fraction > 0) & (amount > 0),
        )
        frame = frame.rename(columns={"trade_date": "asof_date"})[
            ["instrument_id", "asof_date", "float_cap_proxy_cny"]]
        frame["year"] = frame["asof_date"].str.slice(0, 4).astype(int)
        for year, group in frame.groupby("year", sort=False):
            buffers[int(year)].append(group.drop(columns="year"))
        processed += 1
        if processed % 500 == 0:
            print(f"size proxy files={processed}/{len(relevant_codes)}", flush=True)
    index_rows = []
    for year, chunks in sorted(buffers.items()):
        result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        lake.write("size_exposures", result, key=str(year), source="size_stress.proxy.v1", request=cfg)
        index_rows.append({
            "year": year, "rows": len(result),
            "coverage": result["float_cap_proxy_cny"].notna().mean() if len(result) else 0.0,
            "median_float_cap_proxy_cny": result["float_cap_proxy_cny"].median() if len(result) else np.nan,
        })
        print(f"size proxy year={year} rows={len(result)}", flush=True)
    lake.write("size_exposure_index", pd.DataFrame(index_rows), key="v1",
               source="size_stress.proxy.v1", request=cfg)


def enrich_size(predictions: pd.DataFrame, lake: Lake) -> pd.DataFrame:
    dated = predictions.copy()
    dated["year"] = dated["asof_date"].str.slice(0, 4).astype(int)
    frames = []
    for year, group in dated.groupby("year", sort=True):
        exposure = lake.read("size_exposures", str(year))
        merged = group.drop(columns="year").merge(
            exposure, on=["instrument_id", "asof_date"], how="left", validate="one_to_one")
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)
    result["size_percentile"] = result.groupby("asof_date")["float_cap_proxy_cny"].rank(
        method="average", pct=True, na_option="bottom")
    return result


def analyze(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    policy = lake.read("turnover_buffer_selection", "v1").iloc[0]
    entry_n, exit_n = int(policy["entry_n"]), int(policy["selected_exit_n"])
    screen_name, max_amihud, min_turnover = _screen_parameters(lake)
    model_sources = {
        "ridge": ("model_validation_predictions", "ridge_v1", "model_oos_predictions", "ridge_v1"),
        "hgb_ridge_blend": ("model_challenger_validation_predictions", "hgb_v1",
                            "model_challenger_oos_predictions", "hgb_v1"),
    }
    rows: list[dict] = []
    weekly_frames: list[pd.DataFrame] = []
    for model_name, (val_table, val_key, test_table, test_key) in model_sources.items():
        target = "fwd_return_5d_lag1"
        for period, predictions in (
            ("validation", lake.read(val_table, val_key)),
            ("test_lockbox", lake.read(test_table, test_key)),
        ):
            sized = enrich_size(enrich_liquidity(predictions, lake), lake)
            for minimum_size in cfg["minimum_size_percentiles"]:
                weekly, _ = simulate_buffered_long_only(
                    sized, entry_n, exit_n, target, float(cfg["round_trip_cost_bps"]),
                    float(cfg["minimum_realized_weight_fraction"]), max_amihud, min_turnover,
                    float(minimum_size),
                )
                weekly["model"] = model_name
                weekly["period"] = period
                weekly["minimum_size_percentile"] = float(minimum_size)
                weekly_frames.append(weekly)
                rows.append({
                    "model": model_name, "period": period,
                    "minimum_size_percentile": float(minimum_size),
                    "mean_candidate_pool_size": weekly["candidate_pool_size"].mean(),
                    **performance_summary(weekly, int(cfg["annualization_periods"])),
                })
    summary = pd.DataFrame(rows)
    lake.write("size_stress_weekly", pd.concat(weekly_frames, ignore_index=True), key="v1",
               source="size_stress.v1", request=cfg)
    lake.write("size_stress_summary", summary, key="v1", source="size_stress.v1", request=cfg)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build size proxy or run size stress")
    parser.add_argument("stage", choices=["build-exposures", "analyze"])
    args = parser.parse_args()
    if args.stage == "build-exposures":
        materialize_size_proxy()
    else:
        analyze()


if __name__ == "__main__":
    main()
