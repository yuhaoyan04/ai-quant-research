"""Share-level execution gate for the validation-only residual-target challenger."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.execution_market import read_execution_market_rows
from quant_research.idiosyncratic_target_validation import _read_year
from quant_research.ingest import ROOT, Lake
from quant_research.institutional_execution import _summaries, simulate_execution
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.multi_factor_risk_attribution import build_style_exposures
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.risk_free_rate_attribution import (
    align_rates_to_clock,
    coverage_summary,
    fetch_chinabond_history,
    normalize_observations,
)


CONFIG_PATH = ROOT / "config" / "idiosyncratic_share_execution_validation.yaml"
TABLE_ROOT = ROOT / "reports" / "tables"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if bool(cfg.get("historical_oos_evaluation_enabled")):
        raise ValueError("execution validation cannot inspect historical OOS")
    if bool(cfg.get("future_lockbox_access_allowed")):
        raise ValueError("execution validation cannot inspect the future lockbox")
    if str(cfg["period"]["end_date"]) > "2020-12-31":
        raise ValueError("validation-selection execution must end by 2020-12-31")
    if float(cfg["round_trip_cost_bps"]) != 20.0:
        raise ValueError("the frozen paired execution gate retains the 20bp cost")
    if len(cfg["paths"]) != 3 or len({item["path_id"] for item in cfg["paths"]}) != 3:
        raise ValueError("exactly three unique prespecified execution paths are required")
    path_ids = {str(item["path_id"]) for item in cfg["paths"]}
    for comparison in cfg["comparisons"]:
        if {str(comparison["candidate_path"]), str(comparison["reference_path"])} - path_ids:
            raise ValueError("comparison refers to an unknown execution path")
    inference = cfg["inference"]
    if int(inference["primary_block_length_trading_days"]) not in {
        int(value) for value in inference["block_length_sensitivity_trading_days"]
    }:
        raise ValueError("primary bootstrap block is absent from sensitivity grid")
    if inference["multiplicity_control"] != "benjamini_hochberg":
        raise ValueError("paired execution hypotheses require BH-FDR")


def selected_targets(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    source = cfg["source_targets"]
    targets = lake.read(str(source["table"]), str(source["key"])).copy()
    if targets.empty:
        raise RuntimeError(f"source targets are missing: {source}")
    frames = []
    for path in cfg["paths"]:
        current = targets.loc[
            targets["model"].eq(str(cfg["model"]))
            & targets["training_target"].eq(str(path["training_target"]))
            & targets["portfolio_type"].eq(str(path["portfolio_type"])),
            [
                "model", "training_target", "portfolio_type", "signal_date",
                "execution_date", "instrument_id", "target_weight",
            ],
        ].copy()
        if current.empty:
            raise RuntimeError(f"target path is empty: {path['path_id']}")
        current["path_id"] = str(path["path_id"])
        frames.append(current)
    result = pd.concat(frames, ignore_index=True)
    keys = ["path_id", "execution_date", "instrument_id"]
    if result.duplicated(keys).any():
        raise ValueError("duplicate execution target keys")
    sums = result.groupby(["path_id", "execution_date"])["target_weight"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError("execution target weights do not sum to one")
    if str(result["execution_date"].max()) > str(cfg["period"]["end_date"]):
        raise ValueError("execution target reads beyond validation period")
    return result.sort_values(keys).reset_index(drop=True)


def simulation_end_date(lake: Lake, targets: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """End after exactly one label horizon beyond the final execution close."""
    benchmark = lake.read("daily_bars_validated", "sh_000905")
    if benchmark.empty:
        raise RuntimeError("CSI500 validated bars are missing")
    dates = pd.Series(sorted(benchmark["trade_date"].astype(str).unique()))
    final_execution = str(targets["execution_date"].max())
    future = dates.loc[dates.gt(final_execution)]
    horizon = int(cfg["label_horizon_trading_days"])
    if len(future) < horizon:
        raise RuntimeError("insufficient market dates after final target")
    result = str(future.iloc[horizon - 1])
    if result > str(cfg["period"]["end_date"]):
        raise ValueError("final holding horizon exceeds validation period")
    return result


def build_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config(); validate_config(cfg)
    destination = lake.table_path("idiosyncratic_execution_market", str(cfg["artifact_key"]))
    if destination.exists() and not overwrite:
        return lake.read("idiosyncratic_execution_market", str(cfg["artifact_key"]))
    targets = selected_targets(lake, cfg)
    codes = sorted(targets["instrument_id"].astype(str).unique())
    start = str(targets["execution_date"].min())
    end = simulation_end_date(lake, targets, cfg)
    with (ROOT / "config" / "constrained_portfolio_daily_execution.yaml").open(
        encoding="utf-8"
    ) as handle:
        rules = yaml.safe_load(handle)["rules"]
    threshold = float(cfg["corporate_action_proxy"]["activation_absolute_deviation"])
    frames = []
    for number, code in enumerate(codes, start=1):
        current = read_execution_market_rows(
            lake, code, rules=rules, corporate_action_threshold=threshold,
            start_date=start, end_date=end,
        )
        if current.empty:
            raise RuntimeError(f"target instrument has no execution market rows: {code}")
        frames.append(current)
        if number % 500 == 0:
            print(f"Phase 12.2 market files={number}/{len(codes)}", flush=True)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "instrument_id"], keep="last"
    ).sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
    missing = set(codes) - set(result["instrument_id"].astype(str))
    if missing:
        raise RuntimeError(f"execution market misses {len(missing)} target instruments")
    lake.write(
        "idiosyncratic_execution_market", result, key=str(cfg["artifact_key"]),
        source="idiosyncratic_share_execution_validation.market.v1",
        request={**cfg, "market_start": start, "market_end": end, "codes": len(codes)},
    )
    return result


def build_risk_free_observations(
    lake: Lake | None = None, *, overwrite: bool = False,
) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config(); validate_config(cfg)
    key = str(cfg["artifact_key"])
    destination = lake.table_path("idiosyncratic_execution_rf_observations", key)
    if destination.exists() and not overwrite:
        return lake.read("idiosyncratic_execution_rf_observations", key)
    rf_cfg = cfg["risk_free_rate"]
    raw, pulled_at = fetch_chinabond_history(rf_cfg)
    observations = normalize_observations(raw, rf_cfg, pulled_at)
    lake.write(
        "idiosyncratic_execution_rf_observations", observations, key=key,
        source="ccdc.chinabond.government_bond_yield_curve",
        request=rf_cfg["download"],
    )
    return observations


def execution_clock(lake: Lake, targets: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    benchmark = lake.read("daily_bars_validated", "sh_000905")
    if benchmark.empty:
        raise RuntimeError("CSI500 validated bars are missing")
    all_dates = pd.to_datetime(benchmark["trade_date"], errors="coerce").dropna().sort_values()
    first = pd.Timestamp(str(targets["execution_date"].min()))
    final = pd.Timestamp(simulation_end_date(lake, targets, cfg))
    selected = all_dates.loc[all_dates.between(first, final)]
    clock = pd.DataFrame({"trade_date": selected})
    prior = pd.DataFrame({"period_start_date": all_dates})
    clock = pd.merge_asof(
        clock, prior, left_on="trade_date", right_on="period_start_date",
        direction="backward", allow_exact_matches=False,
    )
    if clock["period_start_date"].isna().any():
        raise RuntimeError("cannot resolve prior trading day for execution clock")
    clock["calendar_days"] = (clock["trade_date"] - clock["period_start_date"]).dt.days
    return clock


def aligned_risk_free(
    lake: Lake, targets: pd.DataFrame, cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = str(cfg["artifact_key"])
    observations = lake.read("idiosyncratic_execution_rf_observations", key)
    if observations.empty:
        raise RuntimeError("risk-free observations are missing; run build-risk-free first")
    rf_cfg = cfg["risk_free_rate"]
    aligned = align_rates_to_clock(observations, execution_clock(lake, targets, cfg), rf_cfg)
    coverage = coverage_summary(aligned, rf_cfg)
    lake.write(
        "idiosyncratic_execution_rf_daily", aligned, key=key,
        source="idiosyncratic_share_execution_validation.risk_free.v1", request=rf_cfg,
    )
    lake.write(
        "idiosyncratic_execution_rf_coverage", coverage, key=key,
        source="idiosyncratic_share_execution_validation.risk_free.v1", request=rf_cfg,
    )
    return aligned, coverage


def _reliable_rf_summary(
    daily: pd.DataFrame, rates: pd.DataFrame, cfg: dict[str, Any],
) -> tuple[float, float, int]:
    primary = str(cfg["risk_free_rate"]["primary_rate_scenario"])
    selected = rates.loc[
        rates["scenario_id"].eq(primary), ["trade_date", "risk_free_return"]
    ]
    merged = daily[["trade_date", "net_return"]].merge(
        selected, on="trade_date", how="left", validate="one_to_one"
    )
    missing = int(merged["risk_free_return"].isna().sum())
    excess = merged["net_return"] - merged["risk_free_return"]
    std = float(excess.std(ddof=1))
    annualization = int(cfg["inference"]["annualization_periods"])
    sharpe = float(excess.mean() / std * np.sqrt(annualization)) if std > 0 else np.nan
    annual_excess = float(excess.mean() * annualization)
    return sharpe, annual_excess, missing


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def _paired_metrics(
    candidate: pd.DataFrame, reference: pd.DataFrame, annualization: int,
) -> dict[str, float]:
    c = candidate["net_return"].to_numpy(dtype="float64")
    r = reference["net_return"].to_numpy(dtype="float64")
    ce = candidate["excess_return"].to_numpy(dtype="float64")
    re = reference["excess_return"].to_numpy(dtype="float64")
    ct = candidate["one_way_turnover"].to_numpy(dtype="float64")
    rt = reference["one_way_turnover"].to_numpy(dtype="float64")
    cstd, rstd = float(np.std(c, ddof=1)), float(np.std(r, ddof=1))
    scale = cstd / rstd if rstd > 0 else np.nan
    cesharpe = float(np.mean(ce) / np.std(ce, ddof=1) * np.sqrt(annualization))
    resharpe = float(np.mean(re) / np.std(re, ddof=1) * np.sqrt(annualization))
    return {
        "incremental_net_return_annualized_arithmetic": float(np.mean(c - r) * annualization),
        "volatility_matched_incremental_return_annualized_arithmetic": float(
            (np.mean(c) - np.mean(r) * scale) * annualization
        ),
        "sharpe_reliable_rf_improvement": cesharpe - resharpe,
        "maximum_drawdown_improvement": _maximum_drawdown(c) - _maximum_drawdown(r),
        "turnover_improvement": float(np.mean(rt) - np.mean(ct)),
        "return_per_turnover_improvement": float(
            np.mean(c) / max(np.mean(ct), 1e-12)
            - np.mean(r) / max(np.mean(rt), 1e-12)
        ),
        "target_tracking_l1_improvement": float(
            reference["target_tracking_l1"].mean() - candidate["target_tracking_l1"].mean()
        ),
        "cash_weight_improvement": float(
            reference["cash_weight"].mean() - candidate["cash_weight"].mean()
        ),
    }


def paired_inference(
    daily: pd.DataFrame, rates: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    primary = rates.loc[
        rates["scenario_id"].eq(str(cfg["risk_free_rate"]["primary_rate_scenario"])),
        ["trade_date", "risk_free_return"],
    ]
    panel = daily.merge(primary, on="trade_date", how="left", validate="many_to_one")
    panel["excess_return"] = panel["net_return"] - panel["risk_free_return"]
    records = []
    blocks = [int(value) for value in cfg["inference"]["block_length_sensitivity_trading_days"]]
    samples = int(cfg["inference"]["bootstrap_samples"])
    annualization = int(cfg["inference"]["annualization_periods"])
    for comparison_number, comparison in enumerate(cfg["comparisons"]):
        candidate = panel.loc[panel["path_id"].eq(str(comparison["candidate_path"]))].sort_values("trade_date")
        reference = panel.loc[panel["path_id"].eq(str(comparison["reference_path"]))].sort_values("trade_date")
        if not candidate["trade_date"].reset_index(drop=True).equals(
            reference["trade_date"].reset_index(drop=True)
        ):
            raise RuntimeError("paired execution paths do not share the same daily clock")
        if candidate["risk_free_return"].isna().any() or reference["risk_free_return"].isna().any():
            raise RuntimeError("reliable risk-free return is incomplete")
        candidate = candidate.reset_index(drop=True); reference = reference.reset_index(drop=True)
        point = _paired_metrics(candidate, reference, annualization)
        for block in blocks:
            rng = np.random.default_rng(
                int(cfg["inference"]["random_seed"]) + 1000 * comparison_number + block
            )
            estimates = {metric: np.empty(samples, dtype="float64") for metric in point}
            block_count = int(np.ceil(len(candidate) / block))
            offsets = np.arange(block)
            for sample in range(samples):
                starts = rng.integers(0, len(candidate), size=block_count)
                index = ((starts[:, None] + offsets) % len(candidate)).ravel()[:len(candidate)]
                result = _paired_metrics(candidate.iloc[index], reference.iloc[index], annualization)
                for metric, value in result.items():
                    estimates[metric][sample] = value
            for metric, value in point.items():
                distribution = estimates[metric]
                records.append({
                    "comparison_id": str(comparison["comparison_id"]),
                    "candidate_path": str(comparison["candidate_path"]),
                    "reference_path": str(comparison["reference_path"]),
                    "metric": metric, "observations": len(candidate),
                    "block_length_trading_days": block, "point_estimate": value,
                    "ci_2_5pct": float(np.quantile(distribution, 0.025)),
                    "ci_97_5pct": float(np.quantile(distribution, 0.975)),
                    "probability_improvement_nonpositive": float(
                        (np.sum(distribution <= 0) + 1) / (samples + 1)
                    ),
                })
    result = pd.DataFrame(records)
    result["bh_fdr_adjusted_probability"] = np.nan
    for _, indices in result.groupby(["metric", "block_length_trading_days"]).groups.items():
        result.loc[indices, "bh_fdr_adjusted_probability"] = benjamini_hochberg(
            result.loc[indices, "probability_improvement_nonpositive"]
        )
    result["survives_bh_fdr_5pct"] = result["bh_fdr_adjusted_probability"].le(
        float(cfg["inference"]["false_discovery_rate"])
    ) & result["point_estimate"].gt(0)
    result["is_primary_block"] = result["block_length_trading_days"].eq(
        int(cfg["inference"]["primary_block_length_trading_days"])
    )
    return result


def build_exposure_map(lake: Lake, targets: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    factors = list(load_preprocessing_config()["factor_columns"])
    target_cfg = yaml.safe_load(
        (ROOT / "config" / "idiosyncratic_target_validation.yaml").read_text(encoding="utf-8")
    )
    required_dates = set(targets["signal_date"].astype(str))
    frames = []
    for year in range(2016, 2021):
        yearly = _read_year(lake, year, factors, target_cfg)
        yearly = yearly.loc[yearly["asof_date"].astype(str).isin(required_dates)]
        for signal_date, raw in yearly.groupby("asof_date", sort=True):
            size = pd.to_numeric(raw["float_cap_proxy_cny"], errors="coerce")
            industry = raw["industry_broad_code"].fillna("").astype(str).str.strip()
            current = raw.loc[size.gt(0) & industry.ne("")].reset_index(drop=True).copy()
            styles = build_style_exposures(current, cfg["style_exposure_audit"])
            current = current[["instrument_id", "industry_broad_code"]].copy()
            current["signal_date"] = str(signal_date)
            for style in cfg["style_exposure_audit"]["styles"]:
                current[f"style_{style}"] = styles[style].to_numpy(dtype="float64")
            frames.append(current)
        print(f"Phase 12.2 exposure map year={year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate PIT exposure-map keys")
    missing_dates = required_dates - set(result["signal_date"].astype(str))
    if missing_dates:
        raise RuntimeError(f"PIT exposure map misses {len(missing_dates)} signal dates")
    return result


def exposure_execution_audit(
    targets: pd.DataFrame, positions: pd.DataFrame, daily: pd.DataFrame,
    exposure_map: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    styles = list(cfg["style_exposure_audit"]["styles"])
    rebalance = daily.loc[daily["new_target"].astype(bool), [
        "path_id", "trade_date", "active_target_date", "target_tracking_l1",
    ]].copy()
    actual = positions.merge(
        rebalance[["path_id", "trade_date", "active_target_date"]],
        on=["path_id", "trade_date"], how="inner", validate="many_to_one",
    )
    signal_map = targets[["path_id", "execution_date", "signal_date"]].drop_duplicates()
    if signal_map.duplicated(["path_id", "execution_date"]).any():
        raise RuntimeError("execution date maps to multiple signal dates")
    actual = actual.merge(
        signal_map, left_on=["path_id", "active_target_date"],
        right_on=["path_id", "execution_date"], how="left", validate="many_to_one",
    ).merge(exposure_map, on=["signal_date", "instrument_id"], how="left", validate="many_to_one")
    target = targets.merge(
        exposure_map, on=["signal_date", "instrument_id"], how="left", validate="many_to_one"
    )
    records = []
    tracking = rebalance.set_index(["path_id", "trade_date"])["target_tracking_l1"]
    for (path_id, trade_date), actual_group in actual.groupby(["path_id", "trade_date"], sort=True):
        target_group = target.loc[
            target["path_id"].eq(path_id) & target["execution_date"].eq(trade_date)
        ]
        if target_group.empty:
            raise RuntimeError(f"target exposure group is missing: {path_id}/{trade_date}")
        actual_valid = actual_group["industry_broad_code"].notna()
        target_valid = target_group["industry_broad_code"].notna()
        record: dict[str, Any] = {
            "path_id": path_id, "trade_date": trade_date,
            "signal_date": str(target_group["signal_date"].iloc[0]),
            "actual_exposure_coverage_weight": float(
                actual_group.loc[actual_valid, "actual_weight"].sum()
            ),
            "target_exposure_coverage_weight": float(
                target_group.loc[target_valid, "target_weight"].sum()
            ),
            "target_tracking_l1": float(tracking.loc[(path_id, trade_date)]),
        }
        for style in styles:
            column = f"style_{style}"
            actual_value = float(
                (actual_group.loc[actual_group[column].notna(), "actual_weight"]
                 * actual_group.loc[actual_group[column].notna(), column]).sum()
            )
            target_value = float(
                (target_group.loc[target_group[column].notna(), "target_weight"]
                 * target_group.loc[target_group[column].notna(), column]).sum()
            )
            record[f"actual_{style}_exposure"] = actual_value
            record[f"target_{style}_exposure"] = target_value
            record[f"{style}_execution_gap"] = actual_value - target_value
        actual_industry = actual_group.loc[actual_valid].groupby(
            "industry_broad_code"
        )["actual_weight"].sum()
        target_industry = target_group.loc[target_valid].groupby(
            "industry_broad_code"
        )["target_weight"].sum()
        industries = actual_industry.index.union(target_industry.index)
        record["maximum_abs_industry_execution_gap"] = float(
            (actual_industry.reindex(industries, fill_value=0.0)
             - target_industry.reindex(industries, fill_value=0.0)).abs().max()
        )
        records.append(record)
    return pd.DataFrame(records)


def annual_summary(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy(); frame["year"] = frame["trade_date"].str[:4].astype(int)
    return frame.groupby(["path_id", "year"], sort=True).agg(
        observations=("net_return", "size"),
        annualized_return_arithmetic=("net_return", lambda value: float(value.mean() * 252)),
        realized_return=("net_return", lambda value: float((1.0 + value).prod() - 1.0)),
        mean_turnover=("one_way_turnover", "mean"),
        mean_cash_weight=("cash_weight", "mean"),
        mean_target_tracking_l1=("target_tracking_l1", "mean"),
    ).reset_index()


def execution_bridge(
    lake: Lake, constrained: pd.DataFrame, upper_bound: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Bridge overlapping target-label returns to unique daily-NAV evidence."""
    weekly = lake.read("idiosyncratic_portfolio_weekly", str(cfg["artifact_key"]))
    path_specs = {str(item["path_id"]): item for item in cfg["paths"]}
    constrained_index = constrained.set_index("path_id")
    upper_index = upper_bound.set_index("path_id")
    records = []
    for path_id, path in path_specs.items():
        target = weekly.loc[
            weekly["model"].eq(str(cfg["model"]))
            & weekly["training_target"].eq(str(path["training_target"]))
            & weekly["portfolio_type"].eq(str(path["portfolio_type"]))
            & weekly["included_in_performance"].astype(bool)
        ].sort_values("signal_date")
        if target.empty:
            raise RuntimeError(f"target-weight weekly evidence missing: {path_id}")
        target_annual = float(
            (1.0 + target["net_return"]).prod()
            ** (52.0 / len(target)) - 1.0
        )
        execution_dates = target["execution_date"].astype(str).drop_duplicates().tolist()
        benchmark = lake.read("daily_bars_validated", "sh_000905")
        calendar = {date: index for index, date in enumerate(
            sorted(benchmark["trade_date"].astype(str).unique())
        )}
        gaps = np.diff([calendar[date] for date in execution_dates])
        upper_annual = float(upper_index.loc[path_id, "annualized_return"])
        constrained_annual = float(constrained_index.loc[path_id, "annualized_return"])
        records.append({
            "path_id": path_id, "target_weight_periods": len(target),
            "overlapping_target_intervals": int((gaps < int(cfg["label_horizon_trading_days"])).sum()),
            "target_weight_geometric_annual_return_not_executable": target_annual,
            "unique_nav_fractional_upper_bound_annual_return": upper_annual,
            "unique_nav_constrained_annual_return": constrained_annual,
            "overlap_and_clock_correction_upper_minus_target": upper_annual - target_annual,
            "implementation_shortfall_constrained_minus_upper": constrained_annual - upper_annual,
            "constrained_fill_ratio_by_shares": float(
                constrained_index.loc[path_id, "fill_ratio_by_shares"]
            ),
            "constrained_mean_target_tracking_l1": float(
                constrained_index.loc[path_id, "mean_target_tracking_l1"]
            ),
        })
    return pd.DataFrame(records)


def _checks_and_decision(
    targets: pd.DataFrame, market: pd.DataFrame, rates: pd.DataFrame,
    coverage: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame,
    invariants: pd.DataFrame, inference: pd.DataFrame, exposure: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_block = int(cfg["inference"]["primary_block_length_trading_days"])
    alpha = inference.loc[
        inference["is_primary_block"]
        & inference["metric"].eq("incremental_net_return_annualized_arithmetic")
        & inference["comparison_id"].isin(cfg["promotion_gate"]["primary_comparisons"])
    ]
    clocks = daily.groupby("path_id")["trade_date"].agg(["nunique", "min", "max"])
    sourced = rates.loc[~rates["scenario_id"].eq(
        str(cfg["risk_free_rate"]["zero_rate_scenario"])
    )]
    gates = {
        "incremental_return_ci_above_zero": bool(alpha["ci_2_5pct"].gt(0).all() and len(alpha) == 2),
        "incremental_return_survives_bh_fdr": bool(
            alpha["survives_bh_fdr_5pct"].astype(bool).all() and len(alpha) == 2
        ),
        "minimum_fill_ratio": bool(summary["fill_ratio_by_shares"].min() >= float(
            cfg["promotion_gate"]["minimum_fill_ratio_by_shares"]
        )),
        "maximum_mean_cash_weight": bool(summary["mean_cash_weight"].max() <= float(
            cfg["promotion_gate"]["maximum_mean_cash_weight"]
        )),
        "maximum_p95_target_tracking_l1": bool(summary["p95_target_tracking_l1"].max() <= float(
            cfg["promotion_gate"]["maximum_p95_target_tracking_l1"]
        )),
        "complete_reliable_risk_free_rate": bool(
            not sourced["risk_free_return"].isna().any()
            and int(coverage["missing_dates"].sum()) == 0
        ),
        "all_accounting_invariants": bool(invariants["all_invariants_pass"].astype(bool).all()),
        "minimum_exposure_coverage": bool(min(
            exposure["actual_exposure_coverage_weight"].min(),
            exposure["target_exposure_coverage_weight"].min(),
        ) >= float(cfg["style_exposure_audit"]["minimum_exposure_coverage_weight"])),
    }
    rows = [
        ("three_prespecified_paths", targets["path_id"].nunique() == 3, targets["path_id"].unique().tolist()),
        ("target_weights_normalized", np.allclose(
            targets.groupby(["path_id", "execution_date"])["target_weight"].sum(), 1.0, atol=1e-8
        ), len(targets)),
        ("market_covers_every_target_code", set(targets["instrument_id"].astype(str)).issubset(
            set(market["instrument_id"].astype(str))
        ), market["instrument_id"].nunique()),
        ("common_daily_clock", clocks["nunique"].nunique() == 1 and clocks["min"].nunique() == 1
         and clocks["max"].nunique() == 1, clocks.to_dict()),
        ("exact_nonoverlapping_daily_nav_horizon", daily["trade_date"].max() ==
         rates["trade_date"].max(),
         [daily["trade_date"].min(), daily["trade_date"].max()]),
        ("same_cost_configuration", summary["round_trip_cost_bps"].eq(
            float(cfg["round_trip_cost_bps"])
        ).all(), float(cfg["round_trip_cost_bps"])),
        ("no_t_plus_one_violations", invariants["t_plus_one_violation_count"].eq(0).all(),
         int(invariants["t_plus_one_violation_count"].sum())),
        ("cash_nav_and_orders_reconcile", gates["all_accounting_invariants"],
         invariants["all_invariants_pass"].tolist()),
        ("reliable_risk_free_complete", gates["complete_reliable_risk_free_rate"],
         coverage.to_dict(orient="records")),
        ("pit_exposure_coverage", gates["minimum_exposure_coverage"], float(
            exposure["actual_exposure_coverage_weight"].min()
        )),
        ("primary_incremental_return_ci_positive", gates["incremental_return_ci_above_zero"],
         alpha[["comparison_id", "ci_2_5pct", "ci_97_5pct"]].to_dict(orient="records")),
        ("primary_incremental_return_bh_fdr", gates["incremental_return_survives_bh_fdr"],
         alpha[["comparison_id", "bh_fdr_adjusted_probability"]].to_dict(orient="records")),
        ("share_fill_gate", gates["minimum_fill_ratio"], float(summary["fill_ratio_by_shares"].min())),
        ("cash_weight_gate", gates["maximum_mean_cash_weight"], float(summary["mean_cash_weight"].max())),
        ("tracking_error_gate", gates["maximum_p95_target_tracking_l1"], float(
            summary["p95_target_tracking_l1"].max()
        )),
    ]
    checks = pd.DataFrame([{
        "check_name": name, "status": "PASS" if bool(passed) else "FAIL",
        "observed": str(observed),
    } for name, passed, observed in rows])
    decision = pd.DataFrame([{
        **gates,
        "primary_block_length_trading_days": primary_block,
        "promote_to_forward_paper_challenger": bool(all(gates.values())),
        "evidence_class": cfg["evidence_class"],
        "claims_independent_evidence": False,
        "historical_oos_accessed": False,
        "paper_observations_created": 0,
    }])
    return checks, decision


def _write_report(
    summary: pd.DataFrame, upper_summary: pd.DataFrame, bridge: pd.DataFrame,
    annual: pd.DataFrame, inference: pd.DataFrame, upper_inference: pd.DataFrame,
    exposure: pd.DataFrame, checks: pd.DataFrame, decision: pd.DataFrame,
    trials: pd.DataFrame, cfg: dict[str, Any],
) -> None:
    primary = inference.loc[inference["is_primary_block"]].copy()
    exposure_summary = exposure.groupby("path_id", sort=True).agg(
        dates=("trade_date", "nunique"),
        minimum_actual_coverage=("actual_exposure_coverage_weight", "min"),
        mean_target_tracking_l1=("target_tracking_l1", "mean"),
        p95_target_tracking_l1=("target_tracking_l1", lambda value: float(value.quantile(0.95))),
        maximum_industry_execution_gap=("maximum_abs_industry_execution_gap", "max"),
    ).reset_index()
    for style in cfg["style_exposure_audit"]["styles"]:
        exposure_summary[f"p95_abs_{style}_execution_gap"] = exposure.groupby("path_id")[
            f"{style}_execution_gap"
        ].apply(lambda value: float(value.abs().quantile(0.95))).to_numpy()
    promoted = bool(decision.iloc[0]["promote_to_forward_paper_challenger"])
    conclusion = (
        "通过全部预先锁定的成交、统计、rf与会计门槛，可冻结为新的前瞻paper challenger。"
        if promoted else
        "未通过全部预先锁定门槛；保留结果但不晋级为新的前瞻paper challenger。"
    )
    lines = [
        "# Idiosyncratic Share Execution Validation v1", "", "## 结论", "", conclusion,
        "", "该结果仍属于2016–2020 validation-selection证据，不是pristine lockbox、paper或实盘业绩。",
        "", "## 同账户、同成交约束绩效", "", summary.to_markdown(index=False, floatfmt=".6f"),
        "", "## 主要配对检验与block敏感性", "", primary.to_markdown(index=False, floatfmt=".6f"),
        "", "## 收益时钟与实施损耗桥接", "", bridge.to_markdown(index=False, floatfmt=".6f"),
        "", "无交易限制的股数级上界", "", upper_summary.to_markdown(index=False, floatfmt=".6f"),
        "", "上界路径的主要配对检验", "",
        upper_inference.loc[upper_inference["is_primary_block"]].to_markdown(index=False, floatfmt=".6f"),
        "", "## 分年度结果", "", annual.to_markdown(index=False, floatfmt=".6f"),
        "", "## 实际持仓风险暴露漂移", "", exposure_summary.to_markdown(index=False, floatfmt=".6f"),
        "", "## 晋级决策", "", decision.to_markdown(index=False),
        "", "## 研究尝试登记", "", trials.to_markdown(index=False),
        "", "## 机器验收", "", checks.to_markdown(index=False),
        "", "## 边界", "",
        "成交是未复权日线收盘近似；公司行动为价值保持代理，不是官方公司行动记录，也不含盘口、排队或券商冲击校准。",
        "旧目标权重层把33个不足5个交易日的相邻周仍按完整5日标签复利，存在重叠收益窗口；目标与暴露本身保留，组合收益结论由唯一日频NAV取代。",
        "无风险利率来自CCDC中债国债收益率曲线，按观察日次日可用与ACT/365复利对齐；绝对Sharpe不再使用零利率。",
        "通过本门槛只允许从新的真实日期开始paper观测，不能把既有2016–2020结果重新命名为独立样本外证据。",
    ]
    (ROOT / str(cfg["report_path"])).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(lake: Lake | None = None) -> dict[str, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config(); validate_config(cfg)
    key = str(cfg["artifact_key"])
    targets = selected_targets(lake, cfg)
    market = build_market(lake)
    market = market.loc[
        market["trade_date"].le(simulation_end_date(lake, targets, cfg))
    ].copy()
    rates, coverage = aligned_risk_free(lake, targets, cfg)
    summary_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for path_id, group in targets.groupby("path_id", sort=False):
        codes = set(group["instrument_id"].astype(str))
        path_market = market.loc[market["instrument_id"].astype(str).isin(codes)]
        path_targets = group[["execution_date", "instrument_id", "target_weight"]]
        scenario = {**cfg["scenario"], "scenario": path_id}
        result = simulate_execution(path_market, path_targets, cfg, scenario=scenario)
        summary, invariant = _summaries(result, int(cfg["inference"]["annualization_periods"]))
        sharpe, annual_excess, rf_missing = _reliable_rf_summary(result.daily, rates, cfg)
        summary.update({
            "path_id": path_id, "round_trip_cost_bps": float(cfg["round_trip_cost_bps"]),
            "sharpe_reliable_risk_free": sharpe,
            "annualized_excess_return_arithmetic": annual_excess,
            "risk_free_missing_dates": rf_missing,
            "mean_target_tracking_l1": float(result.daily["target_tracking_l1"].mean()),
            "p95_target_tracking_l1": float(result.daily["target_tracking_l1"].quantile(0.95)),
            "mean_cash_weight": float(result.daily["cash_weight"].mean()),
            "mean_turnover": float(result.daily["one_way_turnover"].mean()),
        })
        summary_rows.append(summary); invariant_rows.append({"path_id": path_id, **invariant})
        daily_frames.append(result.daily.assign(path_id=path_id))
        order_frames.append(result.orders.assign(path_id=path_id))
        event_frames.append(result.events.assign(path_id=path_id))
        fill_frames.append(result.fills.assign(path_id=path_id))
        position_frames.append(result.positions.assign(path_id=path_id))
        if not result.corporate_actions.empty:
            action_frames.append(result.corporate_actions.assign(path_id=path_id))
        print(
            f"Phase 12.2 execution {path_id}: orders={len(result.orders):,} "
            f"fills={len(result.fills):,} ratio={summary['fill_ratio_by_shares']:.2%}", flush=True,
        )
    summary = pd.DataFrame(summary_rows)
    invariants = pd.DataFrame(invariant_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    orders = pd.concat(order_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    fills = pd.concat(fill_frames, ignore_index=True)
    positions = pd.concat(position_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame([{
        "path_id": None, "trade_date": None, "instrument_id": None,
        "action_source": "no_held_action_observed", "value_reconciliation_error": 0.0,
    }])
    inference = paired_inference(daily, rates, cfg)
    upper_summary_rows: list[dict[str, Any]] = []
    upper_invariant_rows: list[dict[str, Any]] = []
    upper_daily_frames: list[pd.DataFrame] = []
    for path_id, group in targets.groupby("path_id", sort=False):
        codes = set(group["instrument_id"].astype(str))
        path_market = market.loc[market["instrument_id"].astype(str).isin(codes)]
        scenario = {**cfg["diagnostic_upper_bound_scenario"], "scenario": path_id}
        result = simulate_execution(
            path_market, group[["execution_date", "instrument_id", "target_weight"]],
            cfg, scenario=scenario,
        )
        upper_summary, upper_invariant = _summaries(
            result, int(cfg["inference"]["annualization_periods"])
        )
        sharpe, annual_excess, rf_missing = _reliable_rf_summary(result.daily, rates, cfg)
        upper_summary.update({
            "path_id": path_id, "round_trip_cost_bps": float(cfg["round_trip_cost_bps"]),
            "sharpe_reliable_risk_free": sharpe,
            "annualized_excess_return_arithmetic": annual_excess,
            "risk_free_missing_dates": rf_missing,
            "mean_target_tracking_l1": float(result.daily["target_tracking_l1"].mean()),
            "p95_target_tracking_l1": float(result.daily["target_tracking_l1"].quantile(0.95)),
            "mean_cash_weight": float(result.daily["cash_weight"].mean()),
            "mean_turnover": float(result.daily["one_way_turnover"].mean()),
        })
        upper_summary_rows.append(upper_summary)
        upper_invariant_rows.append({"path_id": path_id, **upper_invariant})
        upper_daily_frames.append(result.daily.assign(path_id=path_id))
        print(
            f"Phase 12.2 upper bound {path_id}: annual_return="
            f"{upper_summary['annualized_return']:.2%}", flush=True,
        )
    upper_summary = pd.DataFrame(upper_summary_rows)
    upper_invariants = pd.DataFrame(upper_invariant_rows)
    upper_daily = pd.concat(upper_daily_frames, ignore_index=True)
    upper_inference = paired_inference(upper_daily, rates, cfg)
    bridge = execution_bridge(lake, summary, upper_summary, cfg)
    annual = annual_summary(daily)
    exposure_map = build_exposure_map(lake, targets, cfg)
    exposure = exposure_execution_audit(targets, positions, daily, exposure_map, cfg)
    checks, decision = _checks_and_decision(
        targets, market, rates, coverage, daily, summary, invariants,
        inference, exposure, cfg,
    )
    trials = pd.DataFrame(cfg["research_trial_registry"])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "source_experiment": cfg["source_experiment"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "execution_engine": "institutional_execution.simulate_execution",
        "portfolio_paths": targets["path_id"].nunique(), "initial_cash_cny": cfg["initial_cash_cny"],
        "reliable_risk_free_scenario": cfg["risk_free_rate"]["primary_rate_scenario"],
        "daily_bar_approximation": True, "intraday_matching_claimed": False,
    }])
    failures = checks.loc[checks["status"].eq("FAIL")].copy()
    if failures.empty:
        failures = pd.DataFrame([{
            "check_name": "NO_FAILED_ASSERTIONS", "status": "PASS",
            "observed": "all share-execution validation checks passed",
        }])
    outputs = {
        "idiosyncratic_execution_protocol": protocol,
        "idiosyncratic_execution_daily": daily,
        "idiosyncratic_execution_orders": orders,
        "idiosyncratic_execution_events": events,
        "idiosyncratic_execution_fills": fills,
        "idiosyncratic_execution_positions": positions,
        "idiosyncratic_execution_corporate_actions": actions,
        "idiosyncratic_execution_summary": summary,
        "idiosyncratic_execution_invariants": invariants,
        "idiosyncratic_execution_upper_bound_daily": upper_daily,
        "idiosyncratic_execution_upper_bound_summary": upper_summary,
        "idiosyncratic_execution_upper_bound_invariants": upper_invariants,
        "idiosyncratic_execution_upper_bound_inference": upper_inference,
        "idiosyncratic_execution_bridge": bridge,
        "idiosyncratic_execution_annual": annual,
        "idiosyncratic_execution_paired_inference": inference,
        "idiosyncratic_execution_exposure_map": exposure_map,
        "idiosyncratic_execution_exposure_audit": exposure,
        "idiosyncratic_execution_trial_registry": trials,
        "idiosyncratic_execution_checks": checks,
        "idiosyncratic_execution_promotion_decision": decision,
        "idiosyncratic_execution_failed_assertions": failures,
    }
    for table, frame in outputs.items():
        lake.write(
            table, frame, key=key,
            source="idiosyncratic_share_execution_validation.v1", request=cfg,
        )
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_ROOT / "idiosyncratic_execution_summary.csv", index=False)
    inference.loc[inference["is_primary_block"]].to_csv(
        TABLE_ROOT / "idiosyncratic_execution_inference.csv", index=False
    )
    _write_report(
        summary, upper_summary, bridge, annual, inference, upper_inference,
        exposure, checks, decision, trials, cfg,
    )
    print(summary.to_string(index=False), flush=True)
    print(decision.to_string(index=False), flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only residual HGB share execution")
    parser.add_argument(
        "command", choices=["build-risk-free", "build-market", "run", "all"]
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in {"build-risk-free", "all"}:
        build_risk_free_observations(overwrite=args.overwrite)
    if args.command in {"build-market", "all"}:
        build_market(overwrite=args.overwrite)
    if args.command in {"run", "all"}:
        run()


if __name__ == "__main__":
    main()
