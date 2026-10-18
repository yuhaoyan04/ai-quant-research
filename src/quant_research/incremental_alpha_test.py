"""Characteristic-matched incremental-alpha test for executable ML portfolios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from quant_research.constrained_portfolio_daily_execution import _market_end
from quant_research.daily_execution import daily_performance_summary, simulate_daily_retry
from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.model_baseline import weekly_rows
from quant_research.multi_factor_risk_attribution import (
    _read_year,
    build_style_exposures,
)
from quant_research.rf_daily_retry_execution import _limit_array


CONFIG_PATH = ROOT / "config" / "incremental_alpha_test.yaml"
REPORT_PATH = ROOT / "reports" / "incremental_alpha_test.md"
TABLE_ROOT = ROOT / "reports" / "tables"
FIGURE_PATH = ROOT / "reports" / "figures" / "incremental_alpha_equity_curve.svg"
KEY_COLUMNS = ["model", "alpha_pipeline", "strategy"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _exposure_universe(lake: Lake, year: int, cfg: dict) -> pd.DataFrame:
    risk_cfg = {
        "target": "fwd_return_5d_lag1",
        "style_winsor_quantiles": cfg["style_winsor_quantiles"],
    }
    frame = _read_year(lake, year, risk_cfg)
    frame = frame.loc[
        frame["industry_broad_code"].fillna("").astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    styles = build_style_exposures(frame, risk_cfg)
    for style in cfg["styles"]:
        frame[f"style_{style}"] = styles[str(style)].to_numpy(dtype="float64")
    return frame


def match_industry_peers(
    source: pd.DataFrame,
    universe: pd.DataFrame,
    styles: list[str],
    previous_mapping: dict[str, str],
    cfg: dict,
) -> pd.DataFrame:
    """Assign a unique same-industry characteristic peer to every ML holding."""
    required = {
        "instrument_id", "industry_broad_code", "target_weight",
        *[f"style_{style}" for style in styles],
    }
    if missing := required - set(source):
        raise KeyError(f"source matching columns missing: {sorted(missing)}")
    if missing := (required - {"target_weight"}) - set(universe):
        raise KeyError(f"universe matching columns missing: {sorted(missing)}")
    matching_cfg = cfg["matching"]
    held = set(source["instrument_id"].astype(str))
    records: list[dict] = []
    for industry, source_industry in source.groupby("industry_broad_code", sort=True):
        source_industry = source_industry.sort_values("instrument_id").reset_index(drop=True)
        all_candidates = universe.loc[
            universe["industry_broad_code"].astype(str).eq(str(industry))
        ].drop_duplicates("instrument_id").sort_values("instrument_id").reset_index(drop=True)
        clean_candidates = all_candidates.loc[
            ~all_candidates["instrument_id"].astype(str).isin(held)
        ].reset_index(drop=True)
        used_fallback = len(clean_candidates) < len(source_industry)
        candidates = all_candidates if used_fallback else clean_candidates
        if len(candidates) < len(source_industry):
            raise RuntimeError(
                f"industry {industry} has {len(candidates)} candidates for "
                f"{len(source_industry)} source holdings"
            )
        columns = [f"style_{style}" for style in styles]
        left = source_industry[columns].to_numpy(dtype="float64")
        right = candidates[columns].to_numpy(dtype="float64")
        distance = np.square(left[:, None, :] - right[None, :, :]).sum(axis=2)
        weights = source_industry["target_weight"].to_numpy(dtype="float64")
        weight_scale = weights / max(float(weights.mean()), 1e-12)
        cost = distance * weight_scale[:, None]
        candidate_codes = candidates["instrument_id"].astype(str).to_numpy()
        source_codes = source_industry["instrument_id"].astype(str).to_numpy()
        candidate_column = {code: index for index, code in enumerate(candidate_codes)}
        held_columns = np.fromiter(
            (candidate_column[code] for code in held if code in candidate_column),
            dtype="int64",
        )
        if len(held_columns):
            cost[:, held_columns] += float(matching_cfg["held_peer_fallback_penalty"])
        for row, source_code in enumerate(source_codes):
            prior = previous_mapping.get(source_code)
            if prior is not None and prior in candidate_column:
                cost[row, candidate_column[prior]] *= float(
                    matching_cfg["previous_peer_cost_multiplier"]
                )
            if source_code in candidate_column:
                cost[row, candidate_column[source_code]] += float(
                    matching_cfg["self_match_penalty"]
                )
        row_index, column_index = linear_sum_assignment(cost)
        if len(row_index) != len(source_industry):
            raise RuntimeError(f"incomplete peer assignment for industry {industry}")
        for row, column in zip(row_index, column_index):
            source_row = source_industry.iloc[int(row)]
            peer_row = candidates.iloc[int(column)]
            record = {
                "source_instrument_id": str(source_row["instrument_id"]),
                "instrument_id": str(peer_row["instrument_id"]),
                "industry_broad_code": str(industry),
                "target_weight": float(source_row["target_weight"]),
                "style_distance": float(np.sqrt(distance[int(row), int(column)])),
                "used_held_peer_fallback": bool(
                    str(peer_row["instrument_id"]) in held
                ),
                "industry_pool_fallback": bool(used_fallback),
            }
            for style in styles:
                record[f"source_style_{style}"] = float(source_row[f"style_{style}"])
                record[f"peer_style_{style}"] = float(peer_row[f"style_{style}"])
            records.append(record)
    result = pd.DataFrame(records)
    if result["instrument_id"].duplicated().any():
        raise RuntimeError("peer assignment is not unique")
    if not np.isclose(result["target_weight"].sum(), 1.0, atol=1e-10):
        raise RuntimeError("matched peer weights do not sum to one")
    return result


def _turnover_controlled_target(
    fresh: pd.Series, previous: pd.Series | None, limit: float,
    minimum_weight: float = 0.0, maximum_weight: float = 1.0,
) -> tuple[pd.Series, float, float]:
    fresh = fresh.groupby(level=0).sum().astype("float64")
    if previous is None:
        return fresh, 1.0, 1.0
    labels = fresh.index.union(previous.index)
    fresh_aligned = fresh.reindex(labels, fill_value=0.0)
    previous_aligned = previous.reindex(labels, fill_value=0.0)
    unconstrained = float(0.5 * np.abs(fresh_aligned - previous_aligned).sum())
    def capped(blend: float) -> pd.Series:
        values = previous_aligned + blend * (fresh_aligned - previous_aligned)
        values = values.clip(lower=0.0, upper=maximum_weight)
        deficit = 1.0 - float(values.sum())
        for _ in range(10):
            if deficit <= 1e-14:
                break
            headroom = (maximum_weight - values).clip(lower=0.0)
            # Reallocate forced sales only to names in the fresh matched
            # portfolio.  Feeding tiny legacy positions would create an
            # artificial ever-growing benchmark universe.
            eligible = fresh_aligned.gt(0) & headroom.gt(0)
            if not eligible.any():
                eligible = values.gt(0) & headroom.gt(0)
            capacity = headroom.where(eligible, 0.0)
            allocation = capacity / float(capacity.sum()) * deficit
            allocation = np.minimum(allocation, capacity)
            values += allocation
            deficit = 1.0 - float(values.sum())
        if abs(float(values.sum()) - 1.0) > 1e-10:
            raise RuntimeError("capped target projection failed to sum to one")
        return values

    upper = min(1.0, limit / unconstrained) if unconstrained > 0 else 1.0
    controlled = capped(upper)
    realized = float(0.5 * np.abs(controlled - previous_aligned).sum())
    if realized > limit + 1e-12:
        lower = 0.0
        forced = capped(lower)
        forced_turnover = float(0.5 * np.abs(forced - previous_aligned).sum())
        if forced_turnover > limit + 1e-10:
            raise RuntimeError("stock-weight cap is infeasible inside turnover limit")
        for _ in range(50):
            middle = (lower + upper) / 2.0
            candidate = capped(middle)
            candidate_turnover = float(
                0.5 * np.abs(candidate - previous_aligned).sum()
            )
            if candidate_turnover <= limit:
                lower, controlled = middle, candidate
            else:
                upper = middle
    controlled = controlled.loc[controlled.gt(minimum_weight)]
    controlled /= controlled.sum()
    if float(controlled.max()) > maximum_weight + 1e-14:
        excess = float((controlled - maximum_weight).clip(lower=0.0).sum())
        controlled = controlled.clip(upper=maximum_weight)
        recipients = fresh_aligned.reindex(controlled.index, fill_value=0.0).gt(0)
        headroom = (maximum_weight - controlled).clip(lower=0.0).where(
            recipients, 0.0
        )
        if float(headroom.sum()) < excess - 1e-12:
            raise RuntimeError("not enough fresh-name capacity after weight pruning")
        controlled += headroom / float(headroom.sum()) * excess
    realized = float(0.5 * np.abs(
        controlled.reindex(labels, fill_value=0.0) - previous_aligned
    ).sum())
    return controlled, unconstrained, realized


def _drifted_target_state(state: dict | None) -> pd.Series | None:
    """Drift the prior target through its subsequently realized five-day return."""
    if state is None:
        return None
    weights: pd.Series = state["weights"]
    returns: pd.Series = state["returns"]
    values = weights * (1.0 + returns.reindex(weights.index).fillna(0.0))
    total = float(values.sum())
    return values / total if total > 0 else weights.copy()


def build_style_targets(
    lake: Lake | None = None, *, overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    destinations = [
        lake.table_path("incremental_alpha_style_targets", "v1"),
        lake.table_path("incremental_alpha_peer_mapping", "v1"),
        lake.table_path("incremental_alpha_exposure_audit", "v1"),
    ]
    if all(path.exists() for path in destinations) and not overwrite:
        return (
            lake.read("incremental_alpha_style_targets", "v1"),
            lake.read("incremental_alpha_peer_mapping", "v1"),
            lake.read("incremental_alpha_exposure_audit", "v1"),
        )
    source_spec = cfg["source_targets"]
    targets = lake.read(str(source_spec["table"]), str(source_spec["key"]))
    styles = [str(style) for style in cfg["styles"]]
    states: dict[tuple[str, str, str], dict] = {}
    previous_mappings: dict[tuple[str, str, str], dict[str, str]] = {}
    target_records: list[pd.DataFrame] = []
    mapping_records: list[pd.DataFrame] = []
    audit_records: list[dict] = []
    for year in sorted(targets["signal_date"].str[:4].astype(int).unique()):
        universe_year = _exposure_universe(lake, int(year), cfg)
        date_lookup = {
            str(date): group.reset_index(drop=True)
            for date, group in universe_year.groupby("asof_date", sort=False)
        }
        year_targets = targets.loc[
            targets["signal_date"].str[:4].astype(int).eq(int(year))
        ]
        for keys, group in year_targets.groupby(
            [*KEY_COLUMNS, "signal_date", "execution_date"], sort=True
        ):
            model, pipeline, strategy, signal_date, execution_date = map(str, keys)
            state_key = (model, pipeline, strategy)
            universe = date_lookup[signal_date]
            source = group.merge(
                universe[[
                    "instrument_id", "industry_broad_code",
                    *[f"style_{style}" for style in styles],
                ]],
                on="instrument_id", how="left", validate="one_to_one",
            )
            if source[["industry_broad_code", *[f"style_{s}" for s in styles]]].isna().any().any():
                raise RuntimeError(f"source exposures missing on {signal_date} for {state_key}")
            matched = match_industry_peers(
                source, universe, styles, previous_mappings.get(state_key, {}), cfg
            )
            previous_mappings[state_key] = dict(zip(
                matched["source_instrument_id"], matched["instrument_id"]
            ))
            fresh = matched.set_index("instrument_id")["target_weight"]
            controlled, unconstrained_turnover, target_turnover = _turnover_controlled_target(
                fresh, _drifted_target_state(states.get(state_key)),
                float(cfg["matching"]["weekly_one_way_turnover_limit"]),
                float(cfg["matching"]["minimum_reported_weight"]),
                float(cfg["matching"]["maximum_stock_weight"]),
            )
            realized_returns = universe.set_index("instrument_id")[
                "fwd_return_5d_lag1"
            ].reindex(controlled.index)
            states[state_key] = {
                "weights": controlled,
                "returns": pd.to_numeric(realized_returns, errors="coerce").fillna(0.0),
            }
            final = controlled.rename("target_weight").reset_index().rename(
                columns={"index": "instrument_id"}
            )
            final = final.assign(
                signal_date=signal_date, execution_date=execution_date,
                model=model, alpha_pipeline=pipeline, strategy=strategy,
                test_year=int(year), portfolio_type="style_matched_benchmark",
            )
            target_records.append(final)
            matched = matched.assign(
                signal_date=signal_date, execution_date=execution_date,
                model=model, alpha_pipeline=pipeline, strategy=strategy,
            )
            mapping_records.append(matched)

            final_exposure = final.merge(
                universe[["instrument_id", "industry_broad_code",
                          *[f"style_{style}" for style in styles]]],
                on="instrument_id", how="left", validate="one_to_one",
            )
            source_industry = source.groupby("industry_broad_code")["target_weight"].sum()
            final_industry = final_exposure.groupby("industry_broad_code")["target_weight"].sum()
            industries = source_industry.index.union(final_industry.index)
            record = {
                "model": model, "alpha_pipeline": pipeline, "strategy": strategy,
                "signal_date": signal_date, "execution_date": execution_date,
                "source_holdings": int(len(source)),
                "benchmark_holdings": int(len(final)),
                "fresh_mean_style_distance": float(
                    np.average(matched["style_distance"], weights=matched["target_weight"])
                ),
                "held_peer_fallback_weight": float(matched.loc[
                    matched["used_held_peer_fallback"], "target_weight"
                ].sum()),
                "unconstrained_target_turnover": unconstrained_turnover,
                "target_turnover": target_turnover,
                "maximum_stock_weight": float(final["target_weight"].max()),
                "maximum_abs_industry_weight_gap": float(np.abs(
                    source_industry.reindex(industries, fill_value=0.0)
                    - final_industry.reindex(industries, fill_value=0.0)
                ).max()),
                "portfolio_overlap_weight": float(final.loc[
                    final["instrument_id"].isin(set(source["instrument_id"])),
                    "target_weight",
                ].sum()),
            }
            for style in styles:
                model_exposure = float(
                    source["target_weight"] @ source[f"style_{style}"]
                )
                benchmark_exposure = float(
                    final_exposure["target_weight"] @ final_exposure[f"style_{style}"]
                )
                record[f"model_{style}_exposure"] = model_exposure
                record[f"benchmark_{style}_exposure"] = benchmark_exposure
                record[f"{style}_exposure_gap"] = benchmark_exposure - model_exposure
            audit_records.append(record)
        print(f"incremental alpha matching year={year}", flush=True)
    style_targets = pd.concat(target_records, ignore_index=True)
    mappings = pd.concat(mapping_records, ignore_index=True)
    exposure_audit = pd.DataFrame(audit_records)
    request = {**cfg, "style_columns": styles}
    for table, frame in {
        "incremental_alpha_style_targets": style_targets,
        "incremental_alpha_peer_mapping": mappings,
        "incremental_alpha_exposure_audit": exposure_audit,
    }.items():
        lake.write(table, frame, key="v1", source="incremental_alpha_test.v1", request=request)
    return style_targets, mappings, exposure_audit


def build_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    lake = lake or Lake()
    cfg = load_config()
    destination = lake.table_path("incremental_alpha_market", "v1")
    if destination.exists() and not overwrite:
        return lake.read("incremental_alpha_market", "v1")
    style_targets, _, _ = build_style_targets(lake)
    base_spec = cfg["source_market"]
    base = lake.read(str(base_spec["table"]), str(base_spec["key"]))
    existing = set(base["instrument_id"].astype(str).unique())
    required = set(style_targets["instrument_id"].astype(str).unique())
    missing = sorted(required - existing)
    start, end = str(base["trade_date"].min()), str(base["trade_date"].max())
    tolerance = float(cfg["limit_detection_tolerance_percentage_points"])
    frames = [base]
    for number, code in enumerate(missing, start=1):
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", code.replace(".", "_")),
            columns=["instrument_id", "trade_date", "pct_chg", "trade_status", "is_st"],
        )
        bars = bars.loc[bars["trade_date"].between(start, end)].copy()
        if bars.empty:
            continue
        pct = pd.to_numeric(bars["pct_chg"], errors="coerce")
        status = pd.to_numeric(bars["trade_status"], errors="coerce").eq(1)
        is_st = pd.to_numeric(bars["is_st"], errors="coerce").fillna(0).eq(1)
        limits = _limit_array(code, bars["trade_date"], is_st, cfg["rules"])
        bars["daily_return"] = pct / 100.0
        bars["buy_allowed"] = status & pct.lt(limits - tolerance)
        bars["sell_allowed"] = status & pct.gt(-limits + tolerance)
        frames.append(bars[[
            "instrument_id", "trade_date", "daily_return", "buy_allowed", "sell_allowed"
        ]])
        if number % 100 == 0:
            print(f"incremental alpha market files={number}/{len(missing)}", flush=True)
    market = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instrument_id", "trade_date"], keep="last"
    )
    request = {**cfg, "start": start, "end": end, "supplemental_codes": len(missing)}
    lake.write(
        "incremental_alpha_market", market, key="v1",
        source="incremental_alpha_test.market.v1", request=request,
    )
    return market


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def _paired_metrics(model: np.ndarray, style: np.ndarray,
                    model_turnover: np.ndarray, style_turnover: np.ndarray,
                    annualization: int) -> dict[str, float]:
    model_std = float(np.std(model, ddof=1))
    style_std = float(np.std(style, ddof=1))
    scale = model_std / style_std if style_std > 0 else np.nan
    model_sharpe = float(np.mean(model) / model_std * np.sqrt(annualization))
    style_sharpe = float(np.mean(style) / style_std * np.sqrt(annualization))
    model_efficiency = float(np.mean(model) / max(np.mean(model_turnover), 1e-12))
    style_efficiency = float(np.mean(style) / max(np.mean(style_turnover), 1e-12))
    return {
        "incremental_net_return_annualized_arithmetic": float(
            np.mean(model - style) * annualization
        ),
        "volatility_matched_incremental_return_annualized_arithmetic": float(
            (np.mean(model) - np.mean(style) * scale) * annualization
        ),
        "sharpe_improvement": model_sharpe - style_sharpe,
        "maximum_drawdown_improvement": _maximum_drawdown(model) - _maximum_drawdown(style),
        "turnover_improvement": float(np.mean(style_turnover) - np.mean(model_turnover)),
        "return_per_turnover_improvement": model_efficiency - style_efficiency,
    }


def paired_bootstrap(paired: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records: list[dict] = []
    block = int(cfg["bootstrap"]["block_length_trading_days"])
    samples = int(cfg["bootstrap"]["samples"])
    annualization = int(cfg["annualization_periods"])
    for group_number, (keys, group) in enumerate(
        paired.groupby(KEY_COLUMNS, sort=False)
    ):
        group = group.sort_values("trade_date")
        model = group["model_net_return"].to_numpy(dtype="float64")
        style = group["style_net_return"].to_numpy(dtype="float64")
        model_turnover = group["model_turnover"].to_numpy(dtype="float64")
        style_turnover = group["style_turnover"].to_numpy(dtype="float64")
        point = _paired_metrics(model, style, model_turnover, style_turnover, annualization)
        rng = np.random.default_rng(int(cfg["bootstrap"]["random_seed"]) + group_number)
        estimates = {metric: np.empty(samples) for metric in point}
        blocks = int(np.ceil(len(group) / block))
        offsets = np.arange(block)
        for sample in range(samples):
            starts = rng.integers(0, len(group), size=blocks)
            indices = ((starts[:, None] + offsets[None, :]) % len(group)).ravel()[:len(group)]
            metrics = _paired_metrics(
                model[indices], style[indices], model_turnover[indices],
                style_turnover[indices], annualization,
            )
            for metric, value in metrics.items():
                estimates[metric][sample] = value
        for metric, value in point.items():
            values = estimates[metric]
            records.append({
                "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
                "metric": metric, "observations": len(group),
                "point_estimate": value,
                "ci_2_5pct": float(np.quantile(values, 0.025)),
                "ci_97_5pct": float(np.quantile(values, 0.975)),
                "probability_improvement_nonpositive": float(
                    (np.sum(values <= 0) + 1) / (samples + 1)
                ),
            })
    result = pd.DataFrame(records)
    result["bh_fdr_adjusted_probability"] = np.nan
    for _, indices in result.groupby("metric", sort=False).groups.items():
        result.loc[indices, "bh_fdr_adjusted_probability"] = benjamini_hochberg(
            result.loc[indices, "probability_improvement_nonpositive"]
        )
    result["survives_bh_fdr_5pct"] = result[
        "bh_fdr_adjusted_probability"
    ].le(float(cfg["false_discovery_rate"]))
    return result


def _comparison_table(daily: pd.DataFrame, bootstrap: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    summary = daily.groupby([*KEY_COLUMNS, "portfolio_type"], sort=False).apply(
        lambda group: pd.Series(daily_performance_summary(
            group, int(cfg["annualization_periods"])
        )), include_groups=False,
    ).reset_index()
    model = summary.loc[summary["portfolio_type"].eq("ml_model")].drop(
        columns="portfolio_type"
    )
    style = summary.loc[summary["portfolio_type"].eq("style_matched_benchmark")].drop(
        columns="portfolio_type"
    )
    comparison = model.merge(style, on=KEY_COLUMNS, suffixes=("_model", "_style"))
    wide = bootstrap.pivot(
        index=KEY_COLUMNS, columns="metric",
        values=[
            "point_estimate", "ci_2_5pct", "ci_97_5pct",
            "bh_fdr_adjusted_probability", "survives_bh_fdr_5pct",
        ],
    )
    wide.columns = [f"{metric}__{stat}" for stat, metric in wide.columns]
    comparison = comparison.merge(wide.reset_index(), on=KEY_COLUMNS, validate="one_to_one")
    alpha_metric = "incremental_net_return_annualized_arithmetic"
    comparison["incremental_alpha_significant_positive"] = comparison[
        f"{alpha_metric}__ci_2_5pct"
    ].gt(0)
    comparison["incremental_alpha_survives_bh_fdr_5pct"] = comparison[
        f"{alpha_metric}__survives_bh_fdr_5pct"
    ].astype(bool)
    implementation_metrics = [
        "sharpe_improvement", "maximum_drawdown_improvement",
        "turnover_improvement", "return_per_turnover_improvement",
    ]
    comparison["individual_risk_or_efficiency_value_detected"] = np.logical_or.reduce([
        comparison[f"{metric}__ci_2_5pct"].gt(0) for metric in implementation_metrics
    ])
    comparison["multiplicity_controlled_risk_or_efficiency_value_detected"] = np.logical_or.reduce([
        comparison[f"{metric}__survives_bh_fdr_5pct"].astype(bool)
        for metric in implementation_metrics
    ])
    comparison["overall_incremental_value_survives_multiplicity"] = (
        comparison["incremental_alpha_survives_bh_fdr_5pct"]
        | comparison["multiplicity_controlled_risk_or_efficiency_value_detected"]
    )
    return comparison


def _svg_equity_curve(daily: pd.DataFrame) -> None:
    width, height, margin = 980, 480, 55
    series = []
    colors = {
        ("hgb_shallow15_locked", "ml_model"): "#d9485f",
        ("hgb_shallow15_locked", "style_matched_benchmark"): "#e8a0aa",
        ("ridge_locked", "ml_model"): "#2866b1",
        ("ridge_locked", "style_matched_benchmark"): "#8bb2df",
    }
    selected = daily.loc[daily["alpha_pipeline"].eq("raw")]
    for keys, group in selected.groupby(["model", "portfolio_type"], sort=False):
        group = group.sort_values("trade_date")
        wealth = (1.0 + group["net_return"]).cumprod().to_numpy(dtype="float64")
        series.append((keys, group["trade_date"].to_numpy(), wealth))
    all_values = np.concatenate([item[2] for item in series])
    low, high = float(all_values.min()), float(all_values.max())
    def points(values: np.ndarray) -> str:
        x = np.linspace(margin, width - margin, len(values))
        y = height - margin - (values - low) / max(high - low, 1e-12) * (height - 2 * margin)
        return " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    labels = {
        ("hgb_shallow15_locked", "ml_model"): "HGB ML",
        ("hgb_shallow15_locked", "style_matched_benchmark"): "HGB style replica",
        ("ridge_locked", "ml_model"): "Ridge ML",
        ("ridge_locked", "style_matched_benchmark"): "Ridge style replica",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="28" font-family="sans-serif" font-size="20">Incremental Alpha Test — raw portfolios</text>',
    ]
    for tick in np.linspace(low, high, 5):
        y = height - margin - (tick - low) / max(high - low, 1e-12) * (height - 2 * margin)
        lines += [
            f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#e5e7eb"/>',
            f'<text x="8" y="{y+4:.2f}" font-family="sans-serif" font-size="12">{tick:.2f}</text>',
        ]
    for number, (keys, _, wealth) in enumerate(series):
        lines.append(
            f'<polyline fill="none" stroke="{colors[keys]}" stroke-width="2" points="{points(wealth)}"/>'
        )
        lines.append(
            f'<text x="{width-230}" y="{55+number*20}" font-family="sans-serif" '
            f'font-size="13" fill="{colors[keys]}">{labels[keys]}</text>'
        )
    lines.append("</svg>")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_report(comparison: pd.DataFrame, bootstrap: pd.DataFrame,
                  exposure: pd.DataFrame, cfg: dict) -> None:
    metrics = bootstrap.loc[bootstrap["metric"].isin([
        "incremental_net_return_annualized_arithmetic",
        "volatility_matched_incremental_return_annualized_arithmetic",
        "sharpe_improvement", "maximum_drawdown_improvement",
        "turnover_improvement", "return_per_turnover_improvement",
    ])]
    display = comparison[[
        "model", "alpha_pipeline", "annualized_return_model", "annualized_return_style",
        "sharpe_zero_rate_model", "sharpe_zero_rate_style",
        "maximum_drawdown_model", "maximum_drawdown_style",
        "mean_turnover_model", "mean_turnover_style",
        "incremental_alpha_significant_positive",
        "incremental_alpha_survives_bh_fdr_5pct",
        "multiplicity_controlled_risk_or_efficiency_value_detected",
    ]]
    styles = [str(style) for style in cfg["styles"]]
    exposure_summary = exposure.groupby(KEY_COLUMNS, sort=False).agg(
        dates=("signal_date", "nunique"),
        mean_industry_gap=("maximum_abs_industry_weight_gap", "mean"),
        max_industry_gap=("maximum_abs_industry_weight_gap", "max"),
        mean_overlap_weight=("portfolio_overlap_weight", "mean"),
        mean_target_turnover=("target_turnover", "mean"),
        max_target_turnover=("target_turnover", "max"),
        held_peer_fallback_weight=("held_peer_fallback_weight", "mean"),
        **{
            f"mean_abs_{style}_gap": (f"{style}_exposure_gap", lambda x: x.abs().mean())
            for style in styles
        },
    ).reset_index()
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLE_ROOT / "incremental_alpha_comparison.csv", index=False)
    bootstrap.to_csv(TABLE_ROOT / "incremental_alpha_bootstrap.csv", index=False)
    exposure_summary.to_csv(TABLE_ROOT / "incremental_alpha_exposure_summary.csv", index=False)
    alpha = bootstrap.loc[
        bootstrap["metric"].eq("incremental_net_return_annualized_arithmetic")
    ]
    individual_alpha_count = int(alpha["ci_2_5pct"].gt(0).sum())
    fdr_alpha_count = int(alpha["survives_bh_fdr_5pct"].sum())
    lines = [
        "# Incremental Alpha Test：机器学习相对风格复制基准", "",
        "## 识别设计", "",
        "- 每个 ML 目标组合在同一信号日构造一个行业内特征匹配的同业复制组合；匹配只使用 PIT 的规模、Beta、波动、流动性、价值和动量暴露。",
        "- 新鲜复制组合保持 ML 组合的行业权重和单票权重分布，并优先排除 ML 原持仓；随后施加与模型一致的 30% 周单边目标换手上限。",
        "- 两边使用同一历史区间、下一交易日收盘股数订单、停牌/涨跌停逐日重试和 20bp 线性成本。",
        "- 区间来自共同日历上的配对 20 日 circular block bootstrap；等波动收益差是事后诊断，不是可投资杠杆政策。", "",
        "## 结论先行", "",
        f"- 四条 ML 管线中，{individual_alpha_count}/4 条增量收益的单项 95% 区间排除零；对四条策略分别按指标执行 BH-FDR 5% 后为 {fdr_alpha_count}/4。",
        "- 因此，单项显著只能视为局部证据；只有 `survives_bh_fdr_5pct=true` 才作为跨策略检验族内的稳健增量证据。",
        "- 风险与交易效率同样同时披露单项区间和按指标跨四策略的 BH-FDR 结果。", "",
        "## 可成交绩效（2021–2026 historical OOS）", "",
        display.to_markdown(index=False, floatfmt=".6f"), "",
        "## 配对增量检验（正值均代表 ML 更优）", "",
        metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## 暴露匹配与交易约束审计", "",
        exposure_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "![raw 模型与风格复制组合净值](figures/incremental_alpha_equity_curve.svg)", "",
        "> 该检验隔离的是在相近行业/风格暴露与相同执行口径下的股票选择增量。若收益差区间跨零，但回撤、Sharpe 或换手效率显著改善，仍可认定模型存在非纯 Alpha 的实施价值。", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    style_targets, _, exposure = build_style_targets(lake)
    market = build_market(lake)
    source_spec = cfg["source_targets"]
    model_targets = lake.read(str(source_spec["table"]), str(source_spec["key"]))
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    annual_records: list[dict] = []
    paired_frames: list[pd.DataFrame] = []
    for keys, model_group in model_targets.groupby(KEY_COLUMNS, sort=False):
        style_group = style_targets.loc[
            style_targets["model"].eq(keys[0])
            & style_targets["alpha_pipeline"].eq(keys[1])
            & style_targets["strategy"].eq(keys[2])
        ]
        daily_pair = {}
        for portfolio_type, target_group in {
            "ml_model": model_group,
            "style_matched_benchmark": style_group,
        }.items():
            target = target_group[["execution_date", "instrument_id", "target_weight"]]
            codes = set(target["instrument_id"].astype(str))
            selected_market = market.loc[market["instrument_id"].astype(str).isin(codes)]
            daily = simulate_daily_retry(
                selected_market, target, float(cfg["round_trip_cost_bps"])
            )
            daily.insert(0, "portfolio_type", portfolio_type)
            daily.insert(0, "strategy", keys[2])
            daily.insert(0, "alpha_pipeline", keys[1])
            daily.insert(0, "model", keys[0])
            frames.append(daily)
            daily_pair[portfolio_type] = daily
            summaries.append({
                "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
                "portfolio_type": portfolio_type,
                **daily_performance_summary(daily, int(cfg["annualization_periods"])),
            })
            for year, annual in daily.assign(
                year=daily["trade_date"].str[:4].astype(int)
            ).groupby("year", sort=True):
                annual_records.append({
                    "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
                    "portfolio_type": portfolio_type, "year": int(year),
                    **daily_performance_summary(annual, int(cfg["annualization_periods"])),
                })
        paired = daily_pair["ml_model"][["trade_date", "net_return", "turnover"]].merge(
            daily_pair["style_matched_benchmark"][["trade_date", "net_return", "turnover"]],
            on="trade_date", suffixes=("_model", "_style"), validate="one_to_one",
        ).rename(columns={
            "net_return_model": "model_net_return", "net_return_style": "style_net_return",
            "turnover_model": "model_turnover", "turnover_style": "style_turnover",
        })
        if len(paired) != len(daily_pair["ml_model"]) or len(paired) != len(
            daily_pair["style_matched_benchmark"]
        ):
            raise RuntimeError(f"daily clocks are not identical for {keys}")
        paired_frames.append(paired.assign(
            model=keys[0], alpha_pipeline=keys[1], strategy=keys[2],
            incremental_net_return=paired["model_net_return"] - paired["style_net_return"],
        ))
        print(f"incremental alpha execution model={keys[0]} pipeline={keys[1]}", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    annual = pd.DataFrame(annual_records)
    bootstrap = paired_bootstrap(paired, cfg)
    comparison = _comparison_table(daily, bootstrap, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "historical_oos_used_for_selection": False,
        "source_experiment": cfg["source_experiment"],
        "pit_style_exposures_only": True,
        "same_daily_execution_and_cost": True,
        "industry_matched": True,
        "weekly_turnover_limit": float(cfg["matching"]["weekly_one_way_turnover_limit"]),
        "volatility_matching_is_ex_post_diagnostic": True,
    }])
    request = {**cfg, "common_daily_observations": int(paired.groupby(KEY_COLUMNS).size().min())}
    for table, frame in {
        "incremental_alpha_protocol": protocol,
        "incremental_alpha_daily": daily,
        "incremental_alpha_paired_daily": paired,
        "incremental_alpha_summary": summary,
        "incremental_alpha_annual": annual,
        "incremental_alpha_bootstrap": bootstrap,
        "incremental_alpha_comparison": comparison,
    }.items():
        lake.write(table, frame, key="v1", source="incremental_alpha_test.v1", request=request)
    _svg_equity_curve(daily)
    _write_report(comparison, bootstrap, exposure, cfg)
    print(comparison.to_string(index=False), flush=True)
    return comparison, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Executable characteristic-matched alpha test")
    parser.add_argument("command", choices=["build-targets", "build-market", "run", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in ("build-targets", "all"):
        build_style_targets(overwrite=args.overwrite)
    if args.command in ("build-market", "all"):
        build_market(overwrite=args.overwrite)
    if args.command in ("run", "all"):
        run()


if __name__ == "__main__":
    main()
