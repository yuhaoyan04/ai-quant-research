"""Paired lockbox robustness, investable benchmarks, and block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.panel import build_labels
from quant_research.portfolio import performance_summary


def load_config() -> dict:
    with (ROOT / "config" / "robustness.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def circular_block_bootstrap_mean(values: np.ndarray, block_length: int,
                                  samples: int, seed: int) -> np.ndarray:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(len(clean) / block_length))
    starts = rng.integers(0, len(clean), size=(samples, blocks_needed))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % len(clean)
    draws = clean[indices.reshape(samples, -1)[:, :len(clean)]]
    return draws.mean(axis=1)


def bootstrap_record(name: str, values: pd.Series, scale: float, cfg: dict) -> dict:
    estimates = circular_block_bootstrap_mean(
        values.to_numpy(), int(cfg["block_length_weeks"]),
        int(cfg["bootstrap_samples"]), int(cfg["random_seed"]),
    ) * scale
    point = float(values.mean() * scale)
    return {
        "metric": name,
        "observations": int(values.notna().sum()),
        "point_estimate": point,
        "ci_2_5pct": float(np.quantile(estimates, 0.025)),
        "ci_97_5pct": float(np.quantile(estimates, 0.975)),
        "one_sided_probability_nonpositive": float((np.sum(estimates <= 0) + 1) / (len(estimates) + 1)),
        "block_length_weeks": int(cfg["block_length_weeks"]),
    }


def index_forward_returns(instrument_id: str, lake: Lake) -> pd.DataFrame:
    key = instrument_id.replace(".", "_")
    bars = lake.read("daily_bars_validated", key)
    labels = build_labels(bars, [5], execution_lag=1)
    return labels[["asof_date", "fwd_return_5d_lag1"]].rename(
        columns={"fwd_return_5d_lag1": instrument_id})


def _with_cost(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    result = frame.copy()
    result["transaction_cost"] = result["turnover"] * cost_bps / 10_000
    result["net_return"] = result["gross_return"] - result["transaction_cost"]
    result["net_excess_return"] = result["net_return"] - result["benchmark_return"]
    result["included_in_performance"] = True
    return result


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    ridge = lake.read("turnover_buffer_oos_weekly", "v1")
    ridge = ridge.loc[ridge["selected_on_validation"] & ridge["included_in_performance"]].copy()
    challenger = lake.read("challenger_portfolio_oos_weekly", "v1")
    challenger = challenger.loc[challenger["included_in_performance"]].copy()
    common_dates = sorted(set(ridge["asof_date"]) & set(challenger["asof_date"]))
    ridge = ridge.loc[ridge["asof_date"].isin(common_dates)].sort_values("asof_date").reset_index(drop=True)
    challenger = challenger.loc[challenger["asof_date"].isin(common_dates)].sort_values("asof_date").reset_index(drop=True)
    if not ridge["asof_date"].equals(challenger["asof_date"]):
        raise RuntimeError("paired portfolio dates are not aligned")

    benchmark_columns: list[str] = []
    for item in cfg["benchmarks"]:
        column = str(item["name"])
        returns = index_forward_returns(str(item["instrument_id"]), lake).rename(
            columns={str(item["instrument_id"]): column})
        ridge = ridge.merge(returns, on="asof_date", how="left", validate="one_to_one")
        challenger = challenger.merge(returns, on="asof_date", how="left", validate="one_to_one")
        benchmark_columns.append(column)

    cost_rows: list[dict] = []
    for model_name, frame in (("ridge", ridge), ("hgb_ridge_blend", challenger)):
        for cost in cfg["cost_grid_bps"]:
            adjusted = _with_cost(frame, float(cost))
            summary = performance_summary(adjusted, int(cfg["annualization_periods"]))
            cost_rows.append({"model": model_name, "cost_bps": float(cost), **summary})

    investable_rows: list[dict] = []
    for model_name, frame in (("ridge", ridge), ("hgb_ridge_blend", challenger)):
        for benchmark in benchmark_columns:
            usable = frame.loc[frame[benchmark].notna()].copy()
            usable["benchmark_return"] = usable[benchmark]
            usable["gross_excess_return"] = usable["gross_return"] - usable[benchmark]
            usable["net_excess_return"] = usable["net_return"] - usable[benchmark]
            usable["included_in_performance"] = True
            investable_rows.append({
                "model": model_name, "benchmark": benchmark,
                **performance_summary(usable, int(cfg["annualization_periods"])),
            })

    subperiod_rows: list[dict] = []
    for model_name, frame in (("ridge", ridge), ("hgb_ridge_blend", challenger)):
        for period in cfg["subperiods"]:
            sample = frame.loc[frame["asof_date"].between(period["start"], period["end"])].copy()
            sample["included_in_performance"] = True
            subperiod_rows.append({
                "model": model_name, "subperiod": period["name"],
                **performance_summary(sample, int(cfg["annualization_periods"])),
            })

    csi500 = "CSI500"
    paired = pd.DataFrame({
        "asof_date": ridge["asof_date"],
        "ridge_net": ridge["net_return"],
        "challenger_net": challenger["net_return"],
        "csi500": ridge[csi500],
    }).dropna()
    bootstrap_rows = [
        bootstrap_record("ridge_net_excess_vs_CSI500",
                         paired["ridge_net"] - paired["csi500"], 52, cfg),
        bootstrap_record("challenger_net_excess_vs_CSI500",
                         paired["challenger_net"] - paired["csi500"], 52, cfg),
        bootstrap_record("challenger_minus_ridge_net_return",
                         paired["challenger_net"] - paired["ridge_net"], 52, cfg),
    ]
    ridge_ic = lake.read("model_oos_daily", "ridge_v1")[["asof_date", "rank_ic"]].rename(
        columns={"rank_ic": "ridge_ic"})
    challenger_ic = lake.read("model_challenger_oos_daily", "hgb_v1")[["asof_date", "rank_ic"]].rename(
        columns={"rank_ic": "challenger_ic"})
    paired_ic = ridge_ic.merge(challenger_ic, on="asof_date", how="inner").dropna()
    bootstrap_rows.extend([
        bootstrap_record("ridge_mean_rank_ic", paired_ic["ridge_ic"], 1, cfg),
        bootstrap_record("challenger_mean_rank_ic", paired_ic["challenger_ic"], 1, cfg),
        bootstrap_record("challenger_minus_ridge_rank_ic",
                         paired_ic["challenger_ic"] - paired_ic["ridge_ic"], 1, cfg),
    ])

    paired_audit = pd.DataFrame([{
        "common_portfolio_dates": len(common_dates),
        "first_date": common_dates[0], "last_date": common_dates[-1],
        "ridge_original_dates": int(len(lake.read("turnover_buffer_oos_weekly", "v1").query(
            "selected_on_validation and included_in_performance"))),
        "challenger_original_dates": int(len(lake.read("challenger_portfolio_oos_weekly", "v1").query(
            "included_in_performance"))),
        "comparison_rule": "intersection_only",
    }])
    lake.write("robustness_paired_audit", paired_audit, key="v1", source="robustness.v1", request=cfg)
    lake.write("robustness_cost_grid", pd.DataFrame(cost_rows), key="v1",
               source="robustness.v1", request=cfg)
    lake.write("robustness_investable_benchmarks", pd.DataFrame(investable_rows), key="v1",
               source="robustness.v1", request=cfg)
    lake.write("robustness_subperiods", pd.DataFrame(subperiod_rows), key="v1",
               source="robustness.v1", request=cfg)
    lake.write("robustness_block_bootstrap", pd.DataFrame(bootstrap_rows), key="v1",
               source="robustness.v1", request=cfg)
    print(paired_audit.to_string(index=False))
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print(pd.DataFrame(investable_rows).to_string(index=False))
    print(pd.DataFrame(subperiod_rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    run()
