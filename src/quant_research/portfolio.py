"""Validation-selected, cost-aware long-only portfolio simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


def load_config() -> dict:
    with (ROOT / "config" / "portfolio.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def simulate_long_only(predictions: pd.DataFrame, top_n: int, target: str,
                       round_trip_cost_bps: float,
                       minimum_realized_weight_fraction: float,
                       max_amihud_percentile: float | None = None,
                       min_turnover_percentile: float | None = None) -> pd.DataFrame:
    """Equal-weight Top-N with drift-aware turnover and explicit missing coverage."""
    records: list[dict[str, float | str | int | bool]] = []
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    cost_rate = round_trip_cost_bps / 10_000
    for asof_date, group in predictions.groupby("asof_date", sort=True):
        benchmark_group = group
        if max_amihud_percentile is not None:
            group = group.loc[group["amihud_percentile"].le(max_amihud_percentile)]
        if min_turnover_percentile is not None:
            group = group.loc[group["turnover_percentile"].ge(min_turnover_percentile)]
        ranked = group.sort_values(["score", "instrument_id"], ascending=[False, True])
        selected = ranked.head(top_n).copy()
        if selected.empty:
            continue
        weight = 1.0 / len(selected)
        current_weights = dict(zip(selected["instrument_id"].astype(str), [weight] * len(selected)))

        if previous_weights:
            drifted_values = {
                code: old_weight * (1 + previous_returns.get(code, 0.0))
                for code, old_weight in previous_weights.items()
            }
            total_value = sum(drifted_values.values())
            drifted_weights = ({code: value / total_value for code, value in drifted_values.items()}
                               if total_value > 0 else previous_weights)
            names = set(current_weights) | set(drifted_weights)
            turnover = 0.5 * sum(abs(current_weights.get(code, 0.0) - drifted_weights.get(code, 0.0))
                                 for code in names)
        else:
            turnover = 1.0

        realized = pd.to_numeric(selected[target], errors="coerce")
        realized_weight_fraction = float(realized.notna().sum() / len(selected))
        # Weights are fixed before outcomes are known. Missing realized returns
        # are not removed and renormalized; they contribute zero and lower the
        # explicit coverage metric.
        gross_return = float((realized.fillna(0.0) * weight).sum())
        universe_returns = pd.to_numeric(benchmark_group[target], errors="coerce")
        benchmark_return = float(universe_returns.fillna(0.0).mean())
        transaction_cost = turnover * cost_rate
        net_return = gross_return - transaction_cost
        records.append({
            "asof_date": asof_date,
            "rebalance_date": selected["available_date"].dropna().min(),
            "holdings": len(selected),
            "candidate_pool_size": len(group),
            "turnover": turnover,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "gross_excess_return": gross_return - benchmark_return,
            "net_excess_return": net_return - benchmark_return,
            "realized_weight_fraction": realized_weight_fraction,
            "included_in_performance": realized_weight_fraction >= minimum_realized_weight_fraction,
        })
        previous_weights = current_weights
        previous_returns = dict(zip(selected["instrument_id"].astype(str), realized.fillna(0.0)))
    return pd.DataFrame(records)


def performance_summary(weekly: pd.DataFrame, annualization_periods: int) -> dict[str, float | int]:
    usable = weekly.loc[weekly["included_in_performance"]].copy()
    if usable.empty:
        raise ValueError("no portfolio periods pass realized-return coverage")
    returns = usable["net_return"]
    excess = usable["net_excess_return"]
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    periods = len(usable)
    annualized_return = wealth.iloc[-1] ** (annualization_periods / periods) - 1
    volatility = returns.std(ddof=1) * np.sqrt(annualization_periods)
    return {
        "periods": periods,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": returns.mean() / returns.std(ddof=1) * np.sqrt(annualization_periods),
        "annualized_excess_return_arithmetic": excess.mean() * annualization_periods,
        "information_ratio": excess.mean() / excess.std(ddof=1) * np.sqrt(annualization_periods),
        "maximum_drawdown": drawdown.min(),
        "mean_turnover": usable["turnover"].mean(),
        "mean_realized_weight_fraction": usable["realized_weight_fraction"].mean(),
    }


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    selection = lake.read("model_selection", "ridge_v1")
    if selection.empty:
        raise RuntimeError("model selection missing; run the ridge baseline first")
    target = str(selection.iloc[0]["target"])
    validation_predictions = lake.read("model_validation_predictions", "ridge_v1")
    test_predictions = lake.read("model_oos_predictions", "ridge_v1")
    if validation_predictions.empty or test_predictions.empty:
        raise RuntimeError("model predictions missing; rerun the ridge baseline")

    validation_records: list[dict[str, float | int]] = []
    selection_cost = float(cfg["selection_round_trip_cost_bps"])
    for top_n in cfg["candidate_top_n"]:
        weekly = simulate_long_only(
            validation_predictions, int(top_n), target, selection_cost,
            float(cfg["minimum_realized_weight_fraction"]),
        )
        validation_records.append({
            "top_n": int(top_n), "round_trip_cost_bps": selection_cost,
            **performance_summary(weekly, int(cfg["annualization_periods"])),
        })
    validation_grid = pd.DataFrame(validation_records).sort_values(
        ["sharpe_zero_rate", "annualized_return"], ascending=False)
    selected_top_n = int(validation_grid.iloc[0]["top_n"])

    weekly_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, float | int]] = []
    annual_records: list[dict[str, float | int]] = []
    for cost_bps in cfg["report_round_trip_cost_bps"]:
        weekly = simulate_long_only(
            test_predictions, selected_top_n, target, float(cost_bps),
            float(cfg["minimum_realized_weight_fraction"]),
        )
        weekly["round_trip_cost_bps"] = float(cost_bps)
        weekly["top_n"] = selected_top_n
        weekly_frames.append(weekly)
        summary_records.append({
            "top_n": selected_top_n, "round_trip_cost_bps": float(cost_bps),
            **performance_summary(weekly, int(cfg["annualization_periods"])),
        })
        included = weekly.loc[weekly["included_in_performance"]].copy()
        included["year"] = included["asof_date"].str.slice(0, 4).astype(int)
        annual_records.extend({
            "year": int(year), "top_n": selected_top_n, "round_trip_cost_bps": float(cost_bps),
            **performance_summary(group, int(cfg["annualization_periods"])),
        } for year, group in included.groupby("year"))

    portfolio_selection = pd.DataFrame([{
        "selected_top_n": selected_top_n,
        "selection_round_trip_cost_bps": selection_cost,
        "selection_period": "validation_2016_2020_only",
        "weighting": cfg["weighting"],
        "benchmark": cfg["benchmark"],
        "constraint_status": cfg["constraint_status"],
    }])
    lake.write("portfolio_validation_grid", validation_grid, key="long_only_v1",
               source="portfolio.long_only.v1", request=cfg)
    lake.write("portfolio_selection", portfolio_selection, key="long_only_v1",
               source="portfolio.long_only.v1", request=cfg)
    lake.write("portfolio_oos_weekly", pd.concat(weekly_frames, ignore_index=True), key="long_only_v1",
               source="portfolio.long_only.v1", request=cfg)
    lake.write("portfolio_oos_summary", pd.DataFrame(summary_records), key="long_only_v1",
               source="portfolio.long_only.v1", request=cfg)
    lake.write("portfolio_oos_annual", pd.DataFrame(annual_records), key="long_only_v1",
               source="portfolio.long_only.v1", request=cfg)
    print(portfolio_selection.to_string(index=False))
    print(validation_grid.to_string(index=False))
    print(pd.DataFrame(summary_records).to_string(index=False))


if __name__ == "__main__":
    run()
