"""Liquidity-screen stress tests without touching the model lockbox for selection."""

from __future__ import annotations

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.portfolio import performance_summary, simulate_long_only


def load_config() -> dict:
    with (ROOT / "config" / "liquidity_stress.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def enrich_liquidity(predictions: pd.DataFrame, lake: Lake) -> pd.DataFrame:
    """Join only same-date factor exposures, then compute daily percentiles."""
    frames: list[pd.DataFrame] = []
    dated = predictions.copy()
    dated["year"] = dated["asof_date"].str.slice(0, 4).astype(int)
    for year, year_predictions in dated.groupby("year", sort=True):
        path = lake.table_path("eligible_factor_panel_staging", str(year))
        exposures = pd.read_parquet(
            path, columns=["instrument_id", "asof_date", "amihud_20d", "turnover_20d", "volatility_20d"])
        needed_dates = set(year_predictions["asof_date"])
        exposures = exposures.loc[exposures["asof_date"].isin(needed_dates)]
        merged = year_predictions.drop(columns="year").merge(
            exposures, on=["instrument_id", "asof_date"], how="left", validate="one_to_one")
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)
    result["amihud_percentile"] = result.groupby("asof_date")["amihud_20d"].rank(
        method="average", pct=True, na_option="bottom")
    result["turnover_percentile"] = result.groupby("asof_date")["turnover_20d"].rank(
        method="average", pct=True, na_option="bottom")
    return result


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    portfolio_selection = lake.read("portfolio_selection", "long_only_v1")
    model_selection = lake.read("model_selection", "ridge_v1")
    if portfolio_selection.empty or model_selection.empty:
        raise RuntimeError("portfolio/model selection missing")
    top_n = int(portfolio_selection.iloc[0]["selected_top_n"])
    target = str(model_selection.iloc[0]["target"])
    validation = enrich_liquidity(lake.read("model_validation_predictions", "ridge_v1"), lake)
    test = enrich_liquidity(lake.read("model_oos_predictions", "ridge_v1"), lake)

    validation_rows: list[dict] = []
    validation_weekly: dict[str, pd.DataFrame] = {}
    for screen in cfg["screens"]:
        weekly = simulate_long_only(
            validation, top_n, target, float(cfg["round_trip_cost_bps"]),
            float(cfg["minimum_realized_weight_fraction"]),
            screen["max_amihud_percentile"], screen["min_turnover_percentile"],
        )
        validation_weekly[screen["name"]] = weekly
        validation_rows.append({
            "screen": screen["name"],
            "max_amihud_percentile": screen["max_amihud_percentile"],
            "min_turnover_percentile": screen["min_turnover_percentile"],
            "mean_candidate_pool_size": weekly["candidate_pool_size"].mean(),
            **performance_summary(weekly, int(cfg["annualization_periods"])),
        })
    validation_grid = pd.DataFrame(validation_rows).sort_values(
        str(cfg["selection_metric"]), ascending=False)
    selected_screen = str(validation_grid.iloc[0]["screen"])

    test_rows: list[dict] = []
    test_weekly_frames: list[pd.DataFrame] = []
    for screen in cfg["screens"]:
        weekly = simulate_long_only(
            test, top_n, target, float(cfg["round_trip_cost_bps"]),
            float(cfg["minimum_realized_weight_fraction"]),
            screen["max_amihud_percentile"], screen["min_turnover_percentile"],
        )
        weekly["screen"] = screen["name"]
        weekly["selected_on_validation"] = screen["name"] == selected_screen
        test_weekly_frames.append(weekly)
        test_rows.append({
            "screen": screen["name"],
            "selected_on_validation": screen["name"] == selected_screen,
            "mean_candidate_pool_size": weekly["candidate_pool_size"].mean(),
            **performance_summary(weekly, int(cfg["annualization_periods"])),
        })

    selection = pd.DataFrame([{
        "selected_screen": selected_screen,
        "selection_period": cfg["selection_period"],
        "round_trip_cost_bps": float(cfg["round_trip_cost_bps"]),
        "capacity_status": cfg["capacity_status"],
    }])
    lake.write("liquidity_validation_grid", validation_grid, key="v1",
               source="liquidity_stress.v1", request=cfg)
    lake.write("liquidity_selection", selection, key="v1", source="liquidity_stress.v1", request=cfg)
    lake.write("liquidity_oos_weekly", pd.concat(test_weekly_frames, ignore_index=True), key="v1",
               source="liquidity_stress.v1", request=cfg)
    lake.write("liquidity_oos_summary", pd.DataFrame(test_rows), key="v1",
               source="liquidity_stress.v1", request=cfg)
    print(selection.to_string(index=False))
    print(validation_grid.to_string(index=False))
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    run()
