"""Validation-selected rank buffer for reducing boundary churn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.liquidity_stress import enrich_liquidity
from quant_research.portfolio import performance_summary


def load_config() -> dict:
    with (ROOT / "config" / "turnover_buffer.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _screen_parameters(lake: Lake) -> tuple[str, float | None, float | None]:
    selection = lake.read("liquidity_selection", "v1")
    if selection.empty:
        raise RuntimeError("liquidity selection missing; run liquidity stress first")
    selected_name = str(selection.iloc[0]["selected_screen"])
    with (ROOT / "config" / "liquidity_stress.yaml").open(encoding="utf-8") as handle:
        screens = yaml.safe_load(handle)["screens"]
    matches = [screen for screen in screens if str(screen["name"]) == selected_name]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve selected liquidity screen: {selected_name}")
    screen = matches[0]
    return selected_name, screen["max_amihud_percentile"], screen["min_turnover_percentile"]


def apply_liquidity_screen(group: pd.DataFrame, max_amihud_percentile: float | None,
                           min_turnover_percentile: float | None) -> pd.DataFrame:
    result = group
    if max_amihud_percentile is not None:
        result = result.loc[result["amihud_percentile"].le(max_amihud_percentile)]
    if min_turnover_percentile is not None:
        result = result.loc[result["turnover_percentile"].ge(min_turnover_percentile)]
    return result


def simulate_buffered_long_only(
    predictions: pd.DataFrame,
    top_n: int,
    exit_n: int,
    target: str,
    round_trip_cost_bps: float,
    minimum_realized_weight_fraction: float,
    max_amihud_percentile: float | None = None,
    min_turnover_percentile: float | None = None,
    min_size_percentile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight portfolio with Top-N entry and wider rank-based exit.

    Existing names survive while their current score rank is at most ``exit_n``.
    Vacancies are filled by the best-ranked non-held names. All selection uses
    information available on the same decision date.
    """
    if exit_n < top_n:
        raise ValueError("exit_n must be greater than or equal to top_n")
    records: list[dict] = []
    holding_records: list[dict] = []
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    cost_rate = round_trip_cost_bps / 10_000

    for asof_date, raw_group in predictions.groupby("asof_date", sort=True):
        group = apply_liquidity_screen(
            raw_group, max_amihud_percentile, min_turnover_percentile)
        if min_size_percentile is not None:
            group = group.loc[group["size_percentile"].ge(min_size_percentile)]
        ranked = group.sort_values(["score", "instrument_id"], ascending=[False, True]).copy()
        if ranked.empty:
            continue
        ranked["current_rank"] = np.arange(1, len(ranked) + 1)
        rank_by_code = dict(zip(ranked["instrument_id"].astype(str), ranked["current_rank"]))

        previous_names = set(previous_weights)
        retained = {
            code for code in previous_names
            if code in rank_by_code and rank_by_code[code] <= exit_n
        }
        selected_codes = list(retained)
        for code in ranked["instrument_id"].astype(str):
            if len(selected_codes) >= top_n:
                break
            if code not in retained:
                selected_codes.append(code)
        selected_set = set(selected_codes)
        selected = ranked.loc[ranked["instrument_id"].astype(str).isin(selected_set)].copy()
        selected = selected.sort_values(["current_rank", "instrument_id"]).head(top_n)
        selected_codes = selected["instrument_id"].astype(str).tolist()
        selected_set = set(selected_codes)
        if selected.empty:
            continue

        weight = 1.0 / len(selected)
        current_weights = {code: weight for code in selected_codes}
        if previous_weights:
            drifted_values = {
                code: old_weight * (1 + previous_returns.get(code, 0.0))
                for code, old_weight in previous_weights.items()
            }
            total_value = sum(drifted_values.values())
            drifted_weights = ({code: value / total_value for code, value in drifted_values.items()}
                               if total_value > 0 else previous_weights)
            all_names = set(current_weights) | set(drifted_weights)
            turnover = 0.5 * sum(
                abs(current_weights.get(code, 0.0) - drifted_weights.get(code, 0.0))
                for code in all_names)
            additions = len(selected_set - previous_names)
            removals = len(previous_names - selected_set)
            retention_rate = len(selected_set & previous_names) / len(previous_names)
        else:
            turnover = 1.0
            additions = len(selected)
            removals = 0
            retention_rate = np.nan

        realized = pd.to_numeric(selected[target], errors="coerce")
        realized_weight_fraction = float(realized.notna().mean())
        gross_return = float((realized.fillna(0.0) * weight).sum())
        benchmark_returns = pd.to_numeric(raw_group[target], errors="coerce")
        benchmark_return = float(benchmark_returns.fillna(0.0).mean())
        transaction_cost = turnover * cost_rate
        net_return = gross_return - transaction_cost
        records.append({
            "asof_date": asof_date,
            "rebalance_date": selected["available_date"].dropna().min(),
            "holdings": len(selected),
            "candidate_pool_size": len(group),
            "retained_holdings": len(selected_set & previous_names),
            "additions": additions,
            "removals": removals,
            "retention_rate": retention_rate,
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
        holding_records.extend({
            "asof_date": asof_date,
            "rebalance_date": row.available_date,
            "instrument_id": str(row.instrument_id),
            "score": float(row.score),
            "current_rank": int(row.current_rank),
            "target_weight": weight,
            "was_retained": str(row.instrument_id) in previous_names,
            "realized_return": (float(getattr(row, target))
                                if pd.notna(getattr(row, target)) else np.nan),
        } for row in selected.itertuples(index=False))
        previous_weights = current_weights
        previous_returns = dict(zip(selected_codes, realized.fillna(0.0)))

    return pd.DataFrame(records), pd.DataFrame(holding_records)


def _augmented_summary(weekly: pd.DataFrame, annualization_periods: int) -> dict:
    summary = performance_summary(weekly, annualization_periods)
    transitions = weekly.loc[weekly["retention_rate"].notna()]
    summary.update({
        "mean_retention_rate": transitions["retention_rate"].mean(),
        "mean_additions": transitions["additions"].mean(),
        "mean_removals": transitions["removals"].mean(),
    })
    return summary


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    portfolio_selection = lake.read("portfolio_selection", "long_only_v1")
    model_selection = lake.read("model_selection", "ridge_v1")
    if portfolio_selection.empty or model_selection.empty:
        raise RuntimeError("portfolio/model selection missing")
    top_n = int(portfolio_selection.iloc[0]["selected_top_n"])
    target = str(model_selection.iloc[0]["target"])
    screen_name, max_amihud, min_turnover = _screen_parameters(lake)
    validation = enrich_liquidity(lake.read("model_validation_predictions", "ridge_v1"), lake)
    test = enrich_liquidity(lake.read("model_oos_predictions", "ridge_v1"), lake)

    validation_rows: list[dict] = []
    for exit_n in cfg["candidate_exit_n"]:
        weekly, _ = simulate_buffered_long_only(
            validation, top_n, int(exit_n), target, float(cfg["round_trip_cost_bps"]),
            float(cfg["minimum_realized_weight_fraction"]), max_amihud, min_turnover)
        validation_rows.append({
            "entry_n": top_n,
            "exit_n": int(exit_n),
            "liquidity_screen": screen_name,
            **_augmented_summary(weekly, int(cfg["annualization_periods"])),
        })
    validation_grid = pd.DataFrame(validation_rows).sort_values(
        [str(cfg["selection_metric"]), "annualized_return"], ascending=False)
    selected_exit_n = int(validation_grid.iloc[0]["exit_n"])

    test_rows: list[dict] = []
    test_weekly_frames: list[pd.DataFrame] = []
    selected_holdings = pd.DataFrame()
    for exit_n in cfg["candidate_exit_n"]:
        weekly, holdings = simulate_buffered_long_only(
            test, top_n, int(exit_n), target, float(cfg["round_trip_cost_bps"]),
            float(cfg["minimum_realized_weight_fraction"]), max_amihud, min_turnover)
        is_selected = int(exit_n) == selected_exit_n
        weekly["entry_n"] = top_n
        weekly["exit_n"] = int(exit_n)
        weekly["selected_on_validation"] = is_selected
        test_weekly_frames.append(weekly)
        test_rows.append({
            "entry_n": top_n,
            "exit_n": int(exit_n),
            "selected_on_validation": is_selected,
            "liquidity_screen": screen_name,
            **_augmented_summary(weekly, int(cfg["annualization_periods"])),
        })
        if is_selected:
            selected_holdings = holdings.assign(entry_n=top_n, exit_n=int(exit_n))

    selection = pd.DataFrame([{
        "entry_n": top_n,
        "selected_exit_n": selected_exit_n,
        "liquidity_screen": screen_name,
        "round_trip_cost_bps": float(cfg["round_trip_cost_bps"]),
        "selection_period": cfg["selection_period"],
    }])
    lake.write("turnover_buffer_validation_grid", validation_grid, key="v1",
               source="turnover_buffer.v1", request=cfg)
    lake.write("turnover_buffer_selection", selection, key="v1",
               source="turnover_buffer.v1", request=cfg)
    lake.write("turnover_buffer_oos_weekly", pd.concat(test_weekly_frames, ignore_index=True), key="v1",
               source="turnover_buffer.v1", request=cfg)
    lake.write("turnover_buffer_oos_summary", pd.DataFrame(test_rows), key="v1",
               source="turnover_buffer.v1", request=cfg)
    lake.write("turnover_buffer_oos_holdings", selected_holdings, key="v1",
               source="turnover_buffer.v1", request=cfg)
    print(selection.to_string(index=False))
    print(validation_grid.to_string(index=False))
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    run()
