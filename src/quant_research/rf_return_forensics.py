"""Forensic checks for unusually high random-forest validation returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.execution_constraints import (
    build_execution_flags, load_config as load_execution_config, simulate,
)
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import purged_end_date, sum_statistics
from quant_research.model_benchmark import _read_year, _transform, build_year_training, fit_candidate
from quant_research.model_challenger import concatenate_samples
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "rf_return_forensics.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def return_concentration(weekly: pd.DataFrame, annualization_periods: int) -> pd.DataFrame:
    """Report how much CAGR depends on the strongest weeks without reordering time."""
    usable = weekly.loc[weekly["included_in_performance"].astype(bool)].copy()
    records = []
    for requested_drop_count in (0, 1, 5, 10, 20):
        drop_count = min(requested_drop_count, max(len(usable) - 1, 0))
        dropped = set(usable.nlargest(drop_count, "net_return").index) if drop_count else set()
        kept = usable.loc[~usable.index.isin(dropped)]
        wealth = float((1.0 + kept["net_return"]).prod())
        annualized = wealth ** (annualization_periods / len(kept)) - 1
        records.append({
            "requested_dropped_best_weeks": requested_drop_count,
            "dropped_best_weeks": drop_count, "remaining_periods": len(kept),
            "terminal_wealth": wealth, "annualized_return": annualized,
        })
    return pd.DataFrame(records)


def label_alignment_audit(holdings: pd.DataFrame, lake: Lake, target: str,
                          sample_rows: int) -> pd.DataFrame:
    """Independently rebuild sampled t+2...t+6 labels from per-security daily bars."""
    sample = holdings[["instrument_id", "asof_date", target]].copy()
    hashes = pd.util.hash_pandas_object(sample[["instrument_id", "asof_date"]], index=False)
    sample = sample.assign(_hash=hashes).nsmallest(sample_rows, "_hash")
    records = []
    for code, requested in sample.groupby("instrument_id"):
        key = str(code).replace(".", "_")
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", key),
            columns=["trade_date", "pct_chg"],
        ).sort_values("trade_date").reset_index(drop=True)
        returns = pd.to_numeric(bars["pct_chg"], errors="coerce") / 100.0
        rebuilt = (1.0 + returns.shift(-2).iloc[::-1]).rolling(
            5, min_periods=5
        ).apply(np.prod, raw=True).iloc[::-1] - 1.0
        lookup = dict(zip(bars["trade_date"].astype(str), rebuilt))
        for row in requested.itertuples(index=False):
            expected = lookup.get(str(row.asof_date), np.nan)
            observed = float(getattr(row, target))
            records.append({
                "instrument_id": code, "asof_date": row.asof_date,
                "observed_label": observed, "rebuilt_label": expected,
                "absolute_error": abs(observed - expected) if pd.notna(expected) else np.nan,
            })
    return pd.DataFrame(records)


def join_complete_execution_flags(market: pd.DataFrame, lake: Lake) -> tuple[pd.DataFrame, int]:
    """Join cached flags and rebuild any date-boundary gaps from raw daily bars."""
    dated = market.copy()
    dated["year"] = dated["asof_date"].str[:4].astype(int)
    cached = pd.concat([
        lake.read("execution_flags", str(year))
        for year in sorted(dated["year"].unique())
    ], ignore_index=True)
    flag_columns = [
        "instrument_id", "asof_date", "execution_date", "buy_allowed", "sell_allowed"
    ]
    result = dated.drop(columns="year").merge(
        cached[flag_columns], on=["instrument_id", "asof_date"], how="left",
        validate="one_to_one", indicator=True,
    )
    missing = result.loc[result["_merge"].eq("left_only"),
                         ["instrument_id", "asof_date"]]
    missing_count = len(missing)
    if missing_count:
        cfg = load_execution_config()
        supplements = []
        for code, keys in missing.groupby("instrument_id"):
            bars = pd.read_parquet(
                lake.table_path("daily_bars_validated", str(code).replace(".", "_")),
                columns=["instrument_id", "trade_date", "trade_status", "pct_chg", "is_st"],
            )
            supplements.append(build_execution_flags(
                bars, set(keys["asof_date"].astype(str)), cfg
            ))
        supplement = pd.concat(supplements, ignore_index=True)
        lookup = supplement.set_index(["instrument_id", "asof_date"])
        missing_mask = result["_merge"].eq("left_only")
        missing_index = pd.MultiIndex.from_frame(
            result.loc[missing_mask, ["instrument_id", "asof_date"]]
        )
        for column in ("execution_date", "buy_allowed", "sell_allowed"):
            result.loc[missing_mask, column] = lookup.reindex(missing_index)[column].to_numpy()
    if result[["buy_allowed", "sell_allowed"]].isna().any().any():
        raise RuntimeError("execution flags remain missing after raw-bar supplementation")
    result[["buy_allowed", "sell_allowed"]] = result[
        ["buy_allowed", "sell_allowed"]
    ].astype(bool)
    return result.drop(columns="_merge"), missing_count


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    full_stats, purged_stats, full_samples, purged_samples = {}, {}, {}, {}
    start, end = int(cfg["train"]["start_year"]), int(cfg["train"]["end_year"])
    for year in range(start, end + 1):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31",
                                 int(cfg["label_horizon_trading_days"]))
        (full_stats[year], purged_stats[year], full_samples[year],
         purged_samples[year]) = build_year_training(frame, factors, target, cutoff, cfg)
    statistics = sum_statistics(
        [full_stats[y] for y in range(start, end)] + [purged_stats[end]]
    )
    sample = concatenate_samples(
        [full_samples[y] for y in range(start, end)] + [purged_samples[end]]
    )
    model = fit_candidate(cfg["model"], statistics, sample, len(factors), cfg)

    market_frames, holding_frames, missing_records = [], [], []
    validation_end = int(cfg["validation"]["end_year"])
    cutoff = purged_end_date(calendar, f"{validation_end}-12-31",
                             int(cfg["label_horizon_trading_days"]))
    for year in range(int(cfg["validation"]["start_year"]), validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        size = lake.read("pit_size_exposures", str(year))
        frame = frame.merge(size, on=["instrument_id", "asof_date"], how="left",
                            validate="one_to_one")
        for asof_date, group in frame.groupby("asof_date", sort=True):
            group = group.reset_index(drop=True)
            score = model.predict(_transform(group, factors, cfg))
            scored = group.assign(score=score)
            scored["size_percentile"] = scored["float_cap_proxy_cny"].rank(pct=True)
            selected = scored.sort_values(
                ["score", "instrument_id"], ascending=[False, True]
            ).head(int(cfg["portfolio"]["top_n"])).copy()
            selected["target_weight"] = 1.0 / len(selected)
            holding_frames.append(selected[[
                "instrument_id", "asof_date", target, "target_weight",
                "size_percentile", "float_cap_proxy_cny", *factors,
            ]])
            market_frames.append(scored[[
                "instrument_id", "asof_date", "available_date", target, "score",
            ]])
            for factor in factors:
                missing_records.append({
                    "asof_date": asof_date, "factor": factor,
                    "universe_missing_fraction": float(scored[factor].isna().mean()),
                    "top200_missing_fraction": float(selected[factor].isna().mean()),
                })
        print(f"rf forensics scored year={year}", flush=True)
    market = pd.concat(market_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    missing = pd.DataFrame(missing_records)

    ideal = lake.read("model_benchmark_validation_portfolio", "v1").loc[
        lambda x: x["candidate"].eq("random_forest_depth6")
    ].copy()
    flagged_market, supplemented_flag_rows = join_complete_execution_flags(market, lake)
    constrained = simulate(
        flagged_market, holdings[["instrument_id", "asof_date", "target_weight"]],
        target, float(cfg["portfolio"]["round_trip_cost_bps"]),
    )
    ideal_summary = performance_summary(ideal, int(cfg["annualization_periods"]))
    constrained_summary = performance_summary(constrained, int(cfg["annualization_periods"]))
    summary = pd.DataFrame([
        {"scenario": "ideal_close_to_close", **ideal_summary},
        {"scenario": "t_plus_1_limit_and_suspension_constrained", **constrained_summary},
    ])
    holdings_summary = pd.DataFrame([{
        "dates": holdings["asof_date"].nunique(),
        "mean_size_percentile": float(holdings["size_percentile"].mean()),
        "bottom_size_decile_weight": float(holdings["size_percentile"].le(0.1).mean()),
        "bottom_size_30pct_weight": float(holdings["size_percentile"].le(0.3).mean()),
        "median_float_cap_proxy_cny": float(holdings["float_cap_proxy_cny"].median()),
        "mean_blocked_buy_weight": float(constrained["blocked_buy_weight"].mean()),
        "maximum_blocked_buy_weight": float(constrained["blocked_buy_weight"].max()),
        "mean_cash_weight": float(constrained["cash_weight"].mean()),
        "supplemented_execution_flag_rows": supplemented_flag_rows,
    }])
    yearly = []
    for scenario, weekly in (("ideal", ideal), ("constrained", constrained)):
        dated = weekly.assign(year=weekly["asof_date"].str[:4].astype(int))
        for year, group in dated.groupby("year"):
            yearly.append({"scenario": scenario, "year": year,
                           **performance_summary(group, int(cfg["annualization_periods"]))})
    label_audit = label_alignment_audit(
        holdings, lake, target, int(cfg["label_audit_sample_rows"])
    )
    missing_summary = missing.groupby("factor").agg(
        universe_missing_fraction=("universe_missing_fraction", "mean"),
        top200_missing_fraction=("top200_missing_fraction", "mean"),
    ).reset_index()
    missing_summary["top_minus_universe"] = (
        missing_summary["top200_missing_fraction"]
        - missing_summary["universe_missing_fraction"]
    )
    request = {**cfg, "factor_columns": factors}
    outputs = {
        "rf_forensic_summary": summary,
        "rf_forensic_yearly": pd.DataFrame(yearly),
        "rf_forensic_concentration": return_concentration(
            ideal, int(cfg["annualization_periods"])
        ),
        "rf_forensic_holdings_summary": holdings_summary,
        "rf_forensic_missingness": missing_summary,
        "rf_forensic_label_audit": label_audit,
        "rf_forensic_constrained_weekly": constrained,
    }
    for table, result in outputs.items():
        lake.write(table, result, key="v1", source="rf_return_forensics.v1", request=request)
    print(summary.to_string(index=False), flush=True)
    print(holdings_summary.to_string(index=False), flush=True)
    print(outputs["rf_forensic_concentration"].to_string(index=False), flush=True)
    print(f"label audit max error={label_audit['absolute_error'].max():.3e}", flush=True)


if __name__ == "__main__":
    run()
