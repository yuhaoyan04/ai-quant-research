"""ADV20 materialization and portfolio capacity diagnostics."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.liquidity_stress import enrich_liquidity, load_config as load_liquidity_config
from quant_research.panel import instrument_type


def load_config() -> dict:
    with (ROOT / "config" / "capacity.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def materialize_adv20(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    validation = lake.read("model_validation_predictions", "ridge_v1")
    test = lake.read("model_oos_predictions", "ridge_v1")
    dates = pd.concat([validation["asof_date"], test["asof_date"]]).drop_duplicates()
    dates_by_year = {
        int(year): set(group.astype(str))
        for year, group in dates.groupby(dates.str.slice(0, 4).astype(int))
    }
    all_needed_dates = set().union(*dates_by_year.values())
    buffers: dict[int, list[pd.DataFrame]] = {year: [] for year in dates_by_year}
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    window = int(cfg["adv_window_trading_days"])
    for number, path in enumerate(files, start=1):
        code = path.parent.name.removeprefix("key=").replace("_", ".")
        if instrument_type(code) != "a_share":
            continue
        frame = pd.read_parquet(path, columns=["instrument_id", "trade_date", "amount"])
        frame = frame.sort_values("trade_date")
        amount = pd.to_numeric(frame["amount"], errors="coerce").where(lambda values: values >= 0)
        frame["adv20_cny"] = amount.rolling(window, min_periods=window).mean()
        needed = frame.loc[frame["trade_date"].isin(all_needed_dates),
                           ["instrument_id", "trade_date", "adv20_cny"]].copy()
        if not needed.empty:
            needed["year"] = needed["trade_date"].str.slice(0, 4).astype(int)
            for year, chunk in needed.groupby("year", sort=False):
                if year in buffers:
                    buffers[year].append(
                        chunk.drop(columns="year").rename(columns={"trade_date": "asof_date"}))
        if number % 500 == 0:
            print(f"ADV20 files={number}/{len(files)}", flush=True)
    index_rows = []
    for year, chunks in sorted(buffers.items()):
        if not chunks:
            continue
        result = pd.concat(chunks, ignore_index=True)
        lake.write("capacity_exposures", result, key=str(year), source="capacity.adv20.v1", request=cfg)
        index_rows.append({"year": year, "rows": len(result), "median_adv20_cny": result["adv20_cny"].median()})
        print(f"ADV20 year={year} rows={len(result)}", flush=True)
    lake.write("capacity_exposure_index", pd.DataFrame(index_rows), key="v1",
               source="capacity.adv20.v1", request=cfg)


def join_adv20(predictions: pd.DataFrame, lake: Lake) -> pd.DataFrame:
    dated = predictions.copy()
    dated["year"] = dated["asof_date"].str.slice(0, 4).astype(int)
    frames = []
    for year, group in dated.groupby("year", sort=True):
        exposure = lake.read("capacity_exposures", str(year))
        frames.append(group.drop(columns="year").merge(
            exposure, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"))
    return pd.concat(frames, ignore_index=True)


def capacity_by_date(predictions: pd.DataFrame, top_n: int, target: str,
                     max_amihud_percentile: float | None,
                     min_turnover_percentile: float | None,
                     aum_cny: float, participation_limit: float,
                     impact_eta: float) -> pd.DataFrame:
    records = []
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    for asof_date, group in predictions.groupby("asof_date", sort=True):
        eligible = group
        if max_amihud_percentile is not None:
            eligible = eligible.loc[eligible["amihud_percentile"].le(max_amihud_percentile)]
        if min_turnover_percentile is not None:
            eligible = eligible.loc[eligible["turnover_percentile"].ge(min_turnover_percentile)]
        selected = eligible.sort_values(["score", "instrument_id"], ascending=[False, True]).head(top_n)
        if selected.empty:
            continue
        current_weights = dict(zip(selected["instrument_id"].astype(str), [1 / len(selected)] * len(selected)))
        if previous_weights:
            drifted_value = {code: weight * (1 + previous_returns.get(code, 0.0))
                             for code, weight in previous_weights.items()}
            total = sum(drifted_value.values())
            drifted_weights = {code: value / total for code, value in drifted_value.items()}
        else:
            drifted_weights = {}
        codes = set(current_weights) | set(drifted_weights)
        delta = {code: current_weights.get(code, 0.0) - drifted_weights.get(code, 0.0) for code in codes}
        exposure = group.assign(instrument_id=group["instrument_id"].astype(str)).set_index("instrument_id")
        code_index = pd.Index(list(codes))
        required = pd.Series(delta).abs().reindex(code_index).to_numpy(dtype="float64") * aum_cny
        aligned = exposure[["adv20_cny", "volatility_20d"]].reindex(code_index)
        adv = pd.to_numeric(aligned["adv20_cny"], errors="coerce").fillna(0.0).to_numpy()
        volatility = pd.to_numeric(aligned["volatility_20d"], errors="coerce").fillna(0.0).to_numpy()
        caps = participation_limit * adv
        fillable = np.minimum(required, caps)
        gross_required = required.sum()
        fill_ratio = fillable.sum() / gross_required if gross_required > 0 else 1.0
        participation = np.divide(required, adv, out=np.full_like(required, np.inf), where=adv > 0)
        # Impact applies only to the portion that can actually be executed.  The
        # earlier implementation charged impact on the full desired order even
        # when the ADV cap left most of it unfilled, overstating large-AUM cost.
        executed_participation = np.divide(
            fillable, adv, out=np.zeros_like(fillable), where=adv > 0)
        impact_fraction = impact_eta * volatility * np.sqrt(
            np.clip(executed_participation, 0, None))
        impact_cost_fraction_aum = float((fillable * impact_fraction).sum() / aum_cny)
        records.append({
            "asof_date": asof_date,
            "aum_cny": aum_cny,
            "participation_limit": participation_limit,
            "gross_trade_cny": gross_required,
            "fill_ratio": fill_ratio,
            "maximum_required_participation": float(np.nanmax(participation)) if len(participation) else 0.0,
            "estimated_impact_fraction_aum": impact_cost_fraction_aum,
            "missing_adv_trade_fraction": float(required[adv <= 0].sum() / gross_required) if gross_required > 0 else 0.0,
        })
        previous_weights = current_weights
        realized = pd.to_numeric(selected[target], errors="coerce").fillna(0.0)
        previous_returns = dict(zip(selected["instrument_id"].astype(str), realized))
    return pd.DataFrame(records)


def analyze_capacity(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    liquidity_cfg = load_liquidity_config()
    liquidity_selection = lake.read("liquidity_selection", "v1")
    portfolio_selection = lake.read("portfolio_selection", "long_only_v1")
    model_selection = lake.read("model_selection", "ridge_v1")
    selected_name = str(liquidity_selection.iloc[0]["selected_screen"])
    screen = next(item for item in liquidity_cfg["screens"] if item["name"] == selected_name)
    top_n = int(portfolio_selection.iloc[0]["selected_top_n"])
    target = str(model_selection.iloc[0]["target"])
    predictions = enrich_liquidity(lake.read("model_oos_predictions", "ridge_v1"), lake)
    predictions = join_adv20(predictions, lake)

    daily_frames = []
    summary_rows = []
    for aum in cfg["aum_cny"]:
        for participation_limit in cfg["adv_participation_limits"]:
            daily = capacity_by_date(
                predictions, top_n, target, screen["max_amihud_percentile"],
                screen["min_turnover_percentile"], float(aum), float(participation_limit),
                float(cfg["square_root_impact_eta"]),
            )
            daily_frames.append(daily)
            threshold = float(cfg["minimum_fill_ratio"])
            summary_rows.append({
                "aum_cny": float(aum),
                "participation_limit": float(participation_limit),
                "dates": len(daily),
                "fraction_dates_fillable": daily["fill_ratio"].ge(threshold).mean(),
                "median_fill_ratio": daily["fill_ratio"].median(),
                "p10_fill_ratio": daily["fill_ratio"].quantile(0.10),
                "median_estimated_impact_bps_aum": daily["estimated_impact_fraction_aum"].median() * 10_000,
                "median_missing_adv_trade_fraction": daily["missing_adv_trade_fraction"].median(),
                "impact_eta": float(cfg["square_root_impact_eta"]),
            })
    summary = pd.DataFrame(summary_rows)
    lake.write("capacity_daily", pd.concat(daily_frames, ignore_index=True), key="v1",
               source="capacity.analysis.v1", request=cfg)
    lake.write("capacity_summary", summary, key="v1", source="capacity.analysis.v1", request=cfg)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADV20 or run capacity stress tests")
    parser.add_argument("stage", choices=["build-exposures", "analyze"])
    args = parser.parse_args()
    if args.stage == "build-exposures":
        materialize_adv20()
    else:
        analyze_capacity()


if __name__ == "__main__":
    main()
