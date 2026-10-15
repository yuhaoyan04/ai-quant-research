"""Execution-aware historical-OOS audit of the frozen random-forest winner."""

from __future__ import annotations

import pandas as pd
import yaml

from quant_research.execution_constraints import simulate
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import purged_end_date, sum_statistics
from quant_research.model_benchmark import _read_year, _transform, build_year_training, fit_candidate
from quant_research.model_challenger import concatenate_samples
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.rf_return_forensics import join_complete_execution_flags


CONFIG_PATH = ROOT / "config" / "rf_oos_execution_audit.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start = int(cfg["train_start_year"])
    oos_start = int(cfg["historical_oos"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])
    full_stats, purged_stats, full_samples, purged_samples = {}, {}, {}, {}
    for year in range(start, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31",
                                 int(cfg["label_horizon_trading_days"]))
        (full_stats[year], purged_stats[year], full_samples[year],
         purged_samples[year]) = build_year_training(frame, factors, target, cutoff, cfg)
        print(f"rf execution audit prepared year={year}", flush=True)

    market_frames, holding_frames, fit_records = [], [], []
    for test_year in range(oos_start, oos_end + 1):
        fit_end = test_year - 1
        statistics = sum_statistics(
            [full_stats[y] for y in range(start, fit_end)] + [purged_stats[fit_end]]
        )
        sample = concatenate_samples(
            [full_samples[y] for y in range(start, fit_end)] + [purged_samples[fit_end]]
        )
        model = fit_candidate(cfg["model"], statistics, sample, len(factors), cfg)
        fit_records.append({
            "test_year": test_year, "fit_seconds": model.fit_seconds,
            "fit_observations": model.fit_observations,
        })
        frame = _read_year(lake, test_year, columns)
        for _, group in frame.groupby("asof_date", sort=True):
            group = group.reset_index(drop=True)
            score = model.predict(_transform(group, factors, cfg))
            scored = group.assign(score=score)
            selected = scored.sort_values(
                ["score", "instrument_id"], ascending=[False, True]
            ).head(int(cfg["portfolio"]["top_n"])).copy()
            selected["target_weight"] = 1.0 / len(selected)
            holding_frames.append(selected[["instrument_id", "asof_date", "target_weight"]])
            market_frames.append(scored[[
                "instrument_id", "asof_date", "available_date", target, "score"
            ]])
        print(f"rf execution audit scored test_year={test_year} "
              f"fit_seconds={model.fit_seconds:.1f}", flush=True)

    market = pd.concat(market_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    flagged, supplemented = join_complete_execution_flags(market, lake)
    constrained = simulate(
        flagged, holdings, target, float(cfg["portfolio"]["round_trip_cost_bps"])
    )
    ideal = lake.read("model_benchmark_oos_portfolio", "v1").loc[
        lambda x: x["candidate"].eq("random_forest_depth6")
    ].copy()
    # Compare on exactly the dates emitted by the frozen benchmark. Two 2026
    # weeks fail its minimum usable cross-section and must not appear only in
    # the execution-aware leg.
    constrained["included_in_performance"] = constrained["asof_date"].isin(
        set(ideal["asof_date"])
    )
    periods = int(cfg["annualization_periods"])
    summary = pd.DataFrame([
        {"scenario": "ideal_close_to_close", **performance_summary(ideal, periods)},
        {"scenario": "t_plus_1_limit_and_suspension_constrained",
         **performance_summary(constrained, periods)},
    ])
    yearly = []
    for scenario, weekly in (("ideal", ideal), ("constrained", constrained)):
        dated = weekly.assign(year=weekly["asof_date"].str[:4].astype(int))
        for year, group in dated.groupby("year"):
            yearly.append({"scenario": scenario, "year": year,
                           **performance_summary(group, periods)})
    paired = ideal[["asof_date", "net_return"]].rename(
        columns={"net_return": "ideal_net_return"}
    ).merge(
        constrained[["asof_date", "net_return", "blocked_buy_weight",
                     "blocked_sell_weight", "target_tracking_l1"]],
        on="asof_date", validate="one_to_one",
    )
    paired["ideal_minus_constrained"] = (
        paired["ideal_net_return"] - paired["net_return"]
    )
    diagnostics = pd.DataFrame([{
        "periods": len(paired),
        "mean_blocked_buy_weight": float(constrained["blocked_buy_weight"].mean()),
        "maximum_blocked_buy_weight": float(constrained["blocked_buy_weight"].max()),
        "mean_blocked_sell_weight": float(constrained["blocked_sell_weight"].mean()),
        "mean_target_tracking_l1": float(constrained["target_tracking_l1"].mean()),
        "return_gap_correlation_with_blocked_buy": float(
            paired["ideal_minus_constrained"].corr(paired["blocked_buy_weight"])
        ),
        "annualized_arithmetic_ideal_minus_constrained": float(
            paired["ideal_minus_constrained"].mean() * periods
        ),
        "supplemented_execution_flag_rows": supplemented,
        "historical_oos_used_for_selection": False,
    }])
    request = {**cfg, "factor_columns": factors}
    outputs = {
        "rf_oos_execution_summary": summary,
        "rf_oos_execution_yearly": pd.DataFrame(yearly),
        "rf_oos_execution_diagnostics": diagnostics,
        "rf_oos_execution_weekly": constrained,
        "rf_oos_execution_fit_audit": pd.DataFrame(fit_records),
    }
    for table, result in outputs.items():
        lake.write(table, result, key="v1", source="rf_oos_execution_audit.v1",
                   request=request)
    print(summary.to_string(index=False), flush=True)
    print(diagnostics.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
