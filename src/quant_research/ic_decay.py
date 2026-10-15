"""Composite and single-factor IC decay on the locked historical OOS panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import transform_cross_section
from quant_research.ingest import ROOT, Lake
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "ic_decay.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def incremental_return(long_return: pd.Series, short_return: pd.Series) -> pd.Series:
    """Extract the non-overlapping tail window from two cumulative returns."""
    valid = long_return.notna() & short_return.notna() & short_return.gt(-1)
    result = pd.Series(np.nan, index=long_return.index, dtype="float64")
    result.loc[valid] = (
        (1.0 + long_return.loc[valid]) / (1.0 + short_return.loc[valid]) - 1.0
    )
    return result


def newey_west_mean_t(values: pd.Series, lags: int) -> tuple[float, float]:
    """Return HAC standard error and t-statistic for a time-series mean."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    count = len(clean)
    if count < 2:
        return np.nan, np.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned / count)
    for lag in range(1, min(lags, count - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(demeaned[lag:] @ demeaned[:-lag] / count)
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    tolerance = np.finfo("float64").eps * max(1.0, abs(float(clean.mean())))
    t_stat = float(clean.mean() / standard_error) if standard_error > tolerance else np.nan
    return float(standard_error), t_stat


def build_decay_panel(lake: Lake, factors: list[str]) -> pd.DataFrame:
    predictions = lake.read("factor_preprocessing_oos_predictions", "v1")
    if predictions.empty:
        raise RuntimeError("locked OOS predictions missing")
    prediction_dates = set(predictions["asof_date"].unique())
    factor_frames = []
    for year in range(2021, 2027):
        staged = pd.read_parquet(
            lake.table_path("eligible_factor_panel_staging", str(year)),
            columns=["instrument_id", "asof_date", *factors],
        )
        staged = staged.loc[staged["asof_date"].isin(prediction_dates)]
        factor_frames.append(staged)
    factor_panel = pd.concat(factor_frames, ignore_index=True)
    panel = predictions.merge(
        factor_panel, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )

    label_frames = []
    for instrument_id, prediction_group in panel.groupby("instrument_id", sort=False):
        key = str(instrument_id).replace(".", "_")
        path = lake.table_path("labels_raw", key)
        if not path.exists():
            continue
        labels = pd.read_parquet(
            path,
            columns=[
                "instrument_id", "asof_date", "fwd_return_10d_lag1",
                "fwd_return_20d_lag1",
            ],
        )
        labels = labels.loc[labels["asof_date"].isin(set(prediction_group["asof_date"]))]
        if not labels.empty:
            label_frames.append(labels)
    if not label_frames:
        raise RuntimeError("no multi-horizon labels available")
    labels = pd.concat(label_frames, ignore_index=True)
    panel = panel.merge(
        labels, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )
    panel["incremental_return_6_10d_lag1"] = incremental_return(
        panel["fwd_return_10d_lag1"], panel["fwd_return_5d_lag1"]
    )
    panel["incremental_return_11_20d_lag1"] = incremental_return(
        panel["fwd_return_20d_lag1"], panel["fwd_return_10d_lag1"]
    )
    return panel


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    panel = build_decay_panel(lake, factors)
    daily_records: list[dict] = []
    for asof_date, group in panel.groupby("asof_date", sort=True):
        exposures, _ = transform_cross_section(
            group, factors,
            winsor_method="quantile", normalization_method="zscore",
            quantile_limits=tuple(cfg["quantile_limits"]),
            mad_multiplier=float(cfg["mad_multiplier"]),
            include_missing_indicators=False,
        )
        signals = {"model_score": group["score"].to_numpy(dtype="float64")}
        signals.update({factor: exposures[:, index] for index, factor in enumerate(factors)})
        for target_spec in cfg["targets"]:
            target_name = str(target_spec["name"])
            target_column = str(target_spec["column"])
            target = pd.to_numeric(group[target_column], errors="coerce").reset_index(drop=True)
            for signal_name, signal_values in signals.items():
                usable = target.notna().to_numpy() & np.isfinite(signal_values)
                if usable.sum() < int(cfg["minimum_cross_section_size"]):
                    continue
                score = pd.Series(signal_values[usable])
                returns = target.loc[usable].reset_index(drop=True)
                percentile = score.rank(method="first", pct=True)
                bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
                bucket_returns = returns.groupby(bucket.to_numpy()).mean()
                daily_records.append({
                    "asof_date": asof_date,
                    "year": int(str(asof_date)[:4]),
                    "signal": signal_name,
                    "target": target_name,
                    "cross_section_size": int(usable.sum()),
                    "rank_ic": score.rank(method="average").corr(
                        returns.rank(method="average")
                    ),
                    "top_minus_bottom": (
                        bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan)
                    ),
                })
        print(f"ic decay date={asof_date}", flush=True)

    daily = pd.DataFrame(daily_records)
    summary_records = []
    for (signal, target), group in daily.groupby(["signal", "target"], sort=False):
        standard_error, t_stat = newey_west_mean_t(
            group["rank_ic"], int(cfg["newey_west_lags"])
        )
        mean_ic = float(group["rank_ic"].mean())
        std_ic = float(group["rank_ic"].std(ddof=1))
        summary_records.append({
            "signal": signal,
            "target": target,
            "observations": len(group),
            "mean_rank_ic": mean_ic,
            "rank_ic_std": std_ic,
            "annualized_icir": mean_ic / std_ic * np.sqrt(cfg["annualization_periods"]),
            "positive_rank_ic_fraction": float(group["rank_ic"].gt(0).mean()),
            "newey_west_lags": int(cfg["newey_west_lags"]),
            "newey_west_mean_se": standard_error,
            "newey_west_t_stat": t_stat,
            "mean_top_minus_bottom": float(group["top_minus_bottom"].mean()),
        })
    summary = pd.DataFrame(summary_records)
    annual = daily.groupby(["year", "signal", "target"], as_index=False).agg(
        observations=("rank_ic", "size"),
        mean_rank_ic=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        positive_rank_ic_fraction=("rank_ic", lambda values: float(values.gt(0).mean())),
        mean_top_minus_bottom=("top_minus_bottom", "mean"),
    )
    annual["annualized_icir"] = (
        annual["mean_rank_ic"] / annual["rank_ic_std"]
        * np.sqrt(int(cfg["annualization_periods"]))
    )
    request = {**cfg, "factor_columns": factors}
    lake.write("ic_decay_panel", panel, key="v1", source="ic_decay.v1", request=request)
    lake.write("ic_decay_daily", daily, key="v1", source="ic_decay.v1", request=request)
    lake.write("ic_decay_summary", summary, key="v1", source="ic_decay.v1", request=request)
    lake.write("ic_decay_annual", annual, key="v1", source="ic_decay.v1", request=request)
    print(summary.loc[summary["signal"].eq("model_score")].to_string(index=False))


if __name__ == "__main__":
    run()
