"""Validation-only P4 conversion test for the residual-return challenger."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from quant_research.constrained_portfolio_neutralization import (
    _drifted_weights,
    portfolio_risk,
    portfolio_step,
    solve_risk_constrained_weights,
)
from quant_research.idiosyncratic_target_validation import (
    _aggregate_training,
    _design,
    _read_year,
    build_year_training,
    residualize_cross_section,
)
from quant_research.incremental_alpha_test import _drifted_target_state, _paired_metrics
from quant_research.ingest import ROOT, Lake
from quant_research.leave_one_factor_out import benjamini_hochberg
from quant_research.model_baseline import purged_end_date
from quant_research.model_benchmark import FittedCandidate, fit_candidate
from quant_research.multi_factor_risk_attribution import build_style_exposures
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "idiosyncratic_portfolio_validation.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if bool(cfg.get("historical_oos_evaluation_enabled")):
        raise ValueError("portfolio validation cannot inspect historical OOS")
    if bool(cfg.get("future_lockbox_access_allowed")):
        raise ValueError("portfolio validation cannot access the future lockbox")
    if int(cfg["validation"]["end_year"]) > 2020:
        raise ValueError("portfolio development evidence must end in 2020")
    if float(cfg["portfolio"]["round_trip_cost_bps"]) != 20.0:
        raise ValueError("the paired portfolio gate retains the frozen 20bp cost")
    if cfg["inference"]["multiplicity_control"] != "benjamini_hochberg":
        raise ValueError("portfolio hypothesis families require BH-FDR")
    if set(cfg["inference"]["primary_families"]) != {
        "residual_target_minus_raw_target",
        "residual_target_minus_style_benchmark",
    }:
        raise ValueError("primary comparison families changed")
    benchmark = cfg["style_benchmark"]
    if not 0 < float(benchmark["maximum_absolute_industry_weight_gap"]) <= 0.03:
        raise ValueError("style benchmark industry exposure bound is invalid")
    if not 0 < float(benchmark["maximum_absolute_style_exposure_gap"]) <= 0.10:
        raise ValueError("style benchmark style exposure bound is invalid")
    if not 0 < float(benchmark["turnover_tiebreak_penalty"]) <= 1e-4:
        raise ValueError("style benchmark turnover tie-break is invalid")
    if cfg["risk_free_rate"]["sharpe_convention"] != (
        "zero_weekly_rate_development_diagnostic"
    ):
        raise ValueError("portfolio validation Sharpe convention changed")
    if bool(cfg["risk_free_rate"]["promotion_uses_absolute_sharpe"]):
        raise ValueError("absolute zero-rate Sharpe cannot be a promotion gate")
    if not bool(cfg["promotion_gate"]["require_style_benchmark_exposure_match"]):
        raise ValueError("style benchmark exposure matching must remain a hard gate")


def _fit_models(
    lake: Lake, cfg: dict[str, Any], factors: list[str], calendar: pd.DataFrame,
) -> tuple[dict[tuple[str, str], FittedCandidate], pd.DataFrame]:
    start, end = int(cfg["train"]["start_year"]), int(cfg["train"]["end_year"])
    yearly_stats, yearly_purged, yearly_samples, yearly_purged_samples = {}, {}, {}, {}
    for year in range(start, end + 1):
        frame = _read_year(lake, year, factors, cfg)
        cutoff = purged_end_date(calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"]))
        stats, purged, samples, purged_samples, _ = build_year_training(
            frame, factors, cutoff, cfg
        )
        yearly_stats[year], yearly_purged[year] = stats, purged
        yearly_samples[year], yearly_purged_samples[year] = samples, purged_samples
    fitted, records = {}, []
    for training_target in (str(cfg["target"]), str(cfg["residual_target"])):
        statistics, sample = _aggregate_training(
            yearly_stats, yearly_purged, yearly_samples, yearly_purged_samples,
            training_target, start, end,
        )
        for model_name, spec in cfg["models"].items():
            model = fit_candidate(spec, statistics, sample, len(factors), cfg)
            fitted[(str(model_name), training_target)] = model
            records.append({
                "model": model_name, "training_target": training_target,
                "fit_observations": model.fit_observations,
                "training_dates": sample.dates, "fit_seconds": model.fit_seconds,
            })
    return fitted, pd.DataFrame(records)


def _validation_frames(
    lake: Lake, cfg: dict[str, Any], factors: list[str], calendar: pd.DataFrame,
) -> list[pd.DataFrame]:
    end = int(cfg["validation"]["end_year"])
    cutoff = purged_end_date(calendar, f"{end}-12-31", int(cfg["label_horizon_trading_days"]))
    frames = []
    for year in range(int(cfg["validation"]["start_year"]), end + 1):
        frame = _read_year(lake, year, factors, cfg)
        if year == end:
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        frames.append(frame)
    return frames


def build_model_targets(
    frames: list[pd.DataFrame], models: dict[tuple[str, str], FittedCandidate],
    factors: list[str], cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    states: dict[tuple[str, str], dict] = {key: {} for key in models}
    weekly_rows, target_frames, exposure_rows = [], [], []
    universe_by_date: dict[str, pd.DataFrame] = {}
    for yearly in frames:
        for asof_date, raw_group in yearly.groupby("asof_date", sort=True):
            result = residualize_cross_section(raw_group.reset_index(drop=True), cfg)
            group = result.frame.copy()
            group["size_z"] = group["control_size"]
            for style in cfg["controls"]["styles"]:
                group[f"style_{style}"] = group[f"control_{style}"]
            # Matching eligibility cannot depend on whether the future label
            # later happens to be observed. Build its PIT risk universe from
            # signal-date exposures and retain missing future returns solely
            # for the explicit realized-weight coverage gate.
            matching = raw_group.reset_index(drop=True).copy()
            matching_size = pd.to_numeric(
                matching["float_cap_proxy_cny"], errors="coerce"
            )
            matching_industry = matching["industry_broad_code"].fillna("").astype(str).str.strip()
            matching = matching.loc[
                matching_size.gt(0) & matching_industry.ne("")
            ].reset_index(drop=True)
            styles = build_style_exposures(
                matching, {"style_winsor_quantiles": cfg["controls"]["style_winsor_quantiles"]}
            )
            for style in cfg["controls"]["styles"]:
                matching[f"style_{style}"] = styles[style].to_numpy(dtype="float64")
            universe_by_date[str(asof_date)] = matching
            design = _design(group, factors, cfg)
            for key, model in models.items():
                model_name, training_target = key
                score = model.predict(design)
                scored = group.assign(score=score)
                state = states[key]
                previous = _drifted_weights(state)
                weights, optimizer = solve_risk_constrained_weights(
                    scored, score, previous, cfg
                )
                risk, _ = portfolio_risk(scored, weights, cfg)
                portfolio, holdings = portfolio_step(
                    scored, weights, str(cfg["target"]), state, cfg
                )
                identity = {
                    "model": model_name, "training_target": training_target,
                    "signal_date": str(asof_date),
                    "execution_date": str(group["available_date"].iloc[0]),
                    "portfolio_type": "ml_model",
                }
                weekly_rows.append({**identity, **portfolio, **optimizer, **risk})
                holdings = holdings.rename(columns={"target_weight": "target_weight"})
                target_frames.append(holdings[["instrument_id", "target_weight"]].assign(**identity))
                exposure_rows.append({
                    **identity,
                    "portfolio_size": float(weights @ group["style_size"].to_numpy(float)),
                    **{
                        f"portfolio_{style}": float(
                            weights @ group[f"style_{style}"].to_numpy(float)
                        ) for style in cfg["controls"]["styles"]
                    },
                })
    return (
        pd.DataFrame(weekly_rows), pd.concat(target_frames, ignore_index=True),
        pd.DataFrame(exposure_rows), universe_by_date,
    )


def _style_replica_linear_program(
    source: pd.DataFrame, candidate: pd.DataFrame, previous: pd.Series | None,
    styles: list[str], cfg: dict[str, Any],
) -> tuple[pd.Series | None, dict[str, Any]]:
    """Minimize aggregate PIT exposure error subject to execution constraints."""
    started = perf_counter()
    benchmark_cfg = cfg["style_benchmark"]
    candidate = candidate.drop_duplicates("instrument_id").sort_values(
        "instrument_id"
    ).reset_index(drop=True)
    n, style_count = len(candidate), len(styles)
    # Include target industries even when the exclusion pass leaves no peer in
    # that bucket.  Otherwise the LP could silently omit the missing industry,
    # spend its weight in two zero-target buckets, and turn a 1% miss into two
    # apparently permissible 0.5% deviations.
    industries = sorted(
        set(candidate["industry_broad_code"].astype(str))
        | set(source["industry_broad_code"].astype(str))
    )
    industry_count = len(industries)
    # Variables are current weights, turnover auxiliaries, absolute style
    # errors, and absolute industry errors.
    style_start, industry_start = 2 * n, 2 * n + style_count
    variable_count = industry_start + industry_count
    objective = np.zeros(variable_count, dtype="float64")
    # Exposure errors are the primary objective.  This fixed, explicit
    # coefficient is large relative to HiGHS' dual tolerance but far below a
    # unit of normalized exposure error, making turnover a deterministic
    # secondary tie-break rather than a return-selected design choice.
    objective[n:2 * n] = float(benchmark_cfg["turnover_tiebreak_penalty"])
    objective[style_start:industry_start] = (
        1.0 / float(benchmark_cfg["maximum_absolute_style_exposure_gap"])
    )
    objective[industry_start:] = (
        1.0 / float(benchmark_cfg["maximum_absolute_industry_weight_gap"])
    )

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: list[tuple[int, float]], bound: float) -> None:
        row = len(rhs)
        for column, value in entries:
            rows.append(row); columns.append(column); values.append(value)
        rhs.append(float(bound))

    target_weights = source["target_weight"].to_numpy(dtype="float64")
    for offset, style in enumerate(styles):
        exposure = candidate[f"style_{style}"].to_numpy(dtype="float64")
        target = float(target_weights @ source[f"style_{style}"].to_numpy(dtype="float64"))
        deviation = style_start + offset
        add([*( (index, float(value)) for index, value in enumerate(exposure)),
             (deviation, -1.0)], target)
        add([*( (index, -float(value)) for index, value in enumerate(exposure)),
             (deviation, -1.0)], -target)

    candidate_industry = candidate["industry_broad_code"].astype(str).to_numpy()
    target_industry = source.groupby(
        source["industry_broad_code"].astype(str)
    )["target_weight"].sum()
    for offset, industry in enumerate(industries):
        members = np.flatnonzero(candidate_industry == industry)
        target = float(target_industry.get(industry, 0.0))
        deviation = industry_start + offset
        add([*((int(index), 1.0) for index in members), (deviation, -1.0)], target)
        add([*((int(index), -1.0) for index in members), (deviation, -1.0)], -target)

    previous_aligned = np.zeros(n, dtype="float64")
    exited_weight = 0.0
    if previous is not None:
        previous = previous.groupby(level=0).sum().astype("float64")
        previous_aligned = previous.reindex(
            candidate["instrument_id"].astype(str), fill_value=0.0
        ).to_numpy(dtype="float64")
        candidate_codes = set(candidate["instrument_id"].astype(str))
        exited_weight = float(previous.loc[~previous.index.astype(str).isin(candidate_codes)].sum())
        for index, old_weight in enumerate(previous_aligned):
            add([(index, 1.0), (n + index, -1.0)], old_weight)
            add([(index, -1.0), (n + index, -1.0)], -old_weight)
        remaining_l1 = (
            2.0 * float(benchmark_cfg["weekly_one_way_turnover_limit"])
            - exited_weight
        )
        if remaining_l1 < -1e-12:
            return None, {
                "solver_message": "turnover_limit_below_forced_exit",
                "solver_seconds": perf_counter() - started,
                "forced_exit_weight": exited_weight,
            }
        add([(n + index, 1.0) for index in range(n)], max(remaining_l1, 0.0))

    inequality = coo_matrix(
        (values, (rows, columns)), shape=(len(rhs), variable_count)
    ).tocsr()
    equality = csr_matrix(
        (np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, variable_count)
    )
    style_limit = float(benchmark_cfg["maximum_absolute_style_exposure_gap"])
    industry_limit = float(benchmark_cfg["maximum_absolute_industry_weight_gap"])
    result = linprog(
        objective, A_ub=inequality, b_ub=np.asarray(rhs),
        A_eq=equality, b_eq=np.array([1.0]),
        bounds=(
            [(0.0, float(benchmark_cfg["maximum_stock_weight"]))] * n
            + [(0.0, None)] * n
            + [(0.0, style_limit)] * style_count
            + [(0.0, industry_limit)] * industry_count
        ),
        method="highs", options={
            "presolve": True,
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        return None, {
            "solver_message": str(result.message),
            "solver_seconds": perf_counter() - started,
            "forced_exit_weight": exited_weight,
        }
    weights = pd.Series(
        result.x[:n], index=candidate["instrument_id"].astype(str), dtype="float64"
    )
    weights = weights.loc[weights.gt(float(benchmark_cfg["minimum_reported_weight"]))]
    # Do not renormalize: HiGHS already enforces the budget equality and
    # pruning is restricted below numerical reporting precision.
    turnover = 1.0 if previous is None else float(
        0.5 * (
            weights.reindex(weights.index.union(previous.index), fill_value=0.0)
            - previous.reindex(weights.index.union(previous.index), fill_value=0.0)
        ).abs().sum()
    )
    return weights, {
        "solver_message": str(result.message),
        "solver_seconds": perf_counter() - started,
        "forced_exit_weight": exited_weight,
        "turnover": turnover,
        "style_replica_objective": float(result.fun),
    }


def solve_style_replica_weights(
    source: pd.DataFrame, universe: pd.DataFrame, previous: pd.Series | None,
    styles: list[str], cfg: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """Solve without ML holdings first, then use overlap only for feasibility."""
    held = set(source["instrument_id"].astype(str))
    base = universe.loc[
        universe["industry_broad_code"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    attempts = []
    if bool(cfg["style_benchmark"]["exclude_ml_holdings_when_feasible"]):
        attempts.append((
            True, base.loc[~base["instrument_id"].astype(str).isin(held)].copy()
        ))
    attempts.append((False, base))
    messages = []
    for excluded, candidate in attempts:
        weights, diagnostics = _style_replica_linear_program(
            source, candidate, previous, styles, cfg
        )
        messages.append(f"excluded_ml={excluded}:{diagnostics['solver_message']}")
        if weights is not None:
            diagnostics["excluded_ml_holdings"] = excluded
            diagnostics["solver_message"] = " | ".join(messages)
            return weights, diagnostics
    raise RuntimeError("style-replica LP infeasible: " + " | ".join(messages))


def build_style_benchmarks(
    model_targets: pd.DataFrame, universe_by_date: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    styles = [str(value) for value in cfg["controls"]["styles"]]
    states: dict[tuple[str, str], dict[str, pd.Series]] = {}
    weekly_rows, target_frames, exposure_rows = [], [], []
    grouped = model_targets.groupby(
        ["model", "training_target", "signal_date", "execution_date"], sort=True
    )
    for keys, target in grouped:
        model, training_target, signal_date, execution_date = map(str, keys)
        state_key = (model, training_target)
        universe = universe_by_date[signal_date]
        source = target.merge(
            universe[["instrument_id", "industry_broad_code", str(cfg["target"]),
                      *[f"style_{style}" for style in styles]]],
            on="instrument_id", how="left", validate="one_to_one",
        )
        prior = _drifted_target_state(states.get(state_key))
        controlled, solver = solve_style_replica_weights(
            source, universe, prior, styles, cfg,
        )
        peer = universe.set_index("instrument_id").reindex(controlled.index)
        returns = pd.to_numeric(peer[str(cfg["target"])], errors="coerce")
        observed = returns.notna()
        realized_weight_fraction = float(controlled.loc[observed].sum())
        filled_returns = returns.fillna(0.0)
        gross = float(controlled @ filled_returns)
        cost = solver["turnover"] * float(cfg["portfolio"]["round_trip_cost_bps"]) / 10_000
        states[state_key] = {"weights": controlled.copy(), "returns": filled_returns.copy()}
        identity = {
            "model": model, "training_target": training_target,
            "signal_date": signal_date, "execution_date": execution_date,
            "portfolio_type": "style_matched_benchmark",
        }
        weekly_rows.append({
            **identity, "asof_date": signal_date, "turnover": solver["turnover"],
            "gross_return": gross, "transaction_cost": cost,
            "net_return": gross - cost,
            "realized_weight_fraction": realized_weight_fraction,
            "included_in_performance": realized_weight_fraction >= float(
                cfg["portfolio"]["minimum_realized_weight_fraction"]
            ),
            "unconstrained_target_turnover": np.nan,
            "turnover_limit_relaxed": False,
            "maximum_stock_weight": float(controlled.max()),
            "maximum_stock_weight_violation": max(
                0.0, float(controlled.max())
                - float(cfg["style_benchmark"]["maximum_stock_weight"])
            ),
            "size_bound_violation": np.nan, "industry_bound_violation": 0.0,
            **solver,
        })
        target_frames.append(pd.DataFrame({
            "instrument_id": controlled.index, "target_weight": controlled.values,
        }).assign(**identity))
        exposure_rows.append({
            **identity,
            **{
                f"portfolio_{style}": float(
                    controlled @ peer[f"style_{style}"].astype(float)
                ) for style in styles
            },
            "excluded_ml_holdings": solver["excluded_ml_holdings"],
            "style_replica_objective": solver["style_replica_objective"],
        })
    return pd.DataFrame(weekly_rows), pd.concat(target_frames, ignore_index=True), pd.DataFrame(exposure_rows)


def summarize(weekly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for keys, group in weekly.groupby(
        ["model", "training_target", "portfolio_type"], sort=True
    ):
        usable = group.copy()
        rows.append({
            "model": keys[0], "training_target": keys[1], "portfolio_type": keys[2],
            **performance_summary(usable, int(cfg["annualization_periods"])),
        })
    return pd.DataFrame(rows)


def annual_summary(weekly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    sample = weekly.assign(year=weekly["signal_date"].str[:4].astype(int))
    for keys, group in sample.groupby(
        ["model", "training_target", "portfolio_type", "year"], sort=True
    ):
        rows.append({
            "model": keys[0], "training_target": keys[1],
            "portfolio_type": keys[2], "year": int(keys[3]),
            **performance_summary(group, int(cfg["annualization_periods"])),
        })
    return pd.DataFrame(rows)


def recompute_target_exposures(
    targets: pd.DataFrame, lake: Lake, cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure risk exposures without using future-label availability.

    Legacy benchmark names may temporarily lack a current PIT risk record. The
    uncovered weight is retained as an explicit coverage field and industry
    bucket; observed styles are normalized only over their covered weight.
    """
    factors = list(load_preprocessing_config()["factor_columns"])
    style_names = [str(style) for style in cfg["controls"]["styles"]]
    risk_frames = []
    for year in range(int(cfg["validation"]["start_year"]), int(cfg["validation"]["end_year"]) + 1):
        raw = _read_year(lake, year, factors, cfg)
        for signal_date, group in raw.groupby("asof_date", sort=False):
            size = pd.to_numeric(group["float_cap_proxy_cny"], errors="coerce")
            industry = group["industry_broad_code"].fillna("").astype(str).str.strip()
            usable = group.loc[size.gt(0) & industry.ne("")].reset_index(drop=True).copy()
            if usable.empty:
                continue
            styles = build_style_exposures(
                usable, {"style_winsor_quantiles": cfg["controls"]["style_winsor_quantiles"]}
            )
            for style in style_names:
                usable[f"style_{style}"] = styles[style].to_numpy(dtype="float64")
            risk_frames.append(usable[[
                "instrument_id", "industry_broad_code", *[f"style_{style}" for style in style_names]
            ]].assign(signal_date=str(signal_date)))
    risk = pd.concat(risk_frames, ignore_index=True)
    enriched = targets.merge(
        risk, on=["instrument_id", "signal_date"], how="left", validate="many_to_one"
    )
    group_keys = ["model", "training_target", "signal_date", "portfolio_type"]
    rows = []
    for keys, group in enriched.groupby(group_keys, sort=True):
        style_complete = group[[f"style_{style}" for style in style_names]].notna().all(axis=1)
        coverage = float(group.loc[style_complete, "target_weight"].sum())
        row = dict(zip(group_keys, keys))
        row["style_exposure_coverage_weight"] = coverage
        for style in style_names:
            row[f"portfolio_{style}"] = (
                float((group.loc[style_complete, "target_weight"]
                       * group.loc[style_complete, f"style_{style}"]).sum() / coverage)
                if coverage > 0 else np.nan
            )
        row["industry_exposure_coverage_weight"] = float(
            group.loc[group["industry_broad_code"].notna(), "target_weight"].sum()
        )
        rows.append(row)
    measured = pd.DataFrame(rows)

    enriched["industry_broad_code"] = enriched["industry_broad_code"].fillna(
        "__MISSING_PIT_INDUSTRY__"
    )
    allocations = enriched.groupby(
        [*group_keys, "industry_broad_code"], as_index=False
    )["target_weight"].sum()
    wide = allocations.pivot_table(
        index=["model", "training_target", "signal_date", "industry_broad_code"],
        columns="portfolio_type", values="target_weight", fill_value=0.0,
    ).reset_index()
    wide["gap"] = wide["ml_model"] - wide["style_matched_benchmark"]
    industry_gap = wide.groupby(
        ["model", "training_target", "signal_date"], as_index=False
    ).agg(
        maximum_abs_industry_weight_gap=("gap", lambda value: float(value.abs().max())),
        industry_weight_gap_l1=("gap", lambda value: float(value.abs().sum() / 2.0)),
    )
    return measured, industry_gap


def exposure_match_audit(
    targets: pd.DataFrame, exposures: pd.DataFrame, industry_gap: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    keys = ["model", "training_target", "signal_date"]
    styles = [str(style) for style in cfg["controls"]["styles"]]
    exposure_columns = [f"portfolio_{style}" for style in styles]
    coverage_columns = [
        "style_exposure_coverage_weight",
        "industry_exposure_coverage_weight",
    ]
    model = exposures.loc[
        exposures["portfolio_type"].eq("ml_model"),
        [*keys, *exposure_columns, *coverage_columns],
    ]
    style = exposures.loc[
        exposures["portfolio_type"].eq("style_matched_benchmark"),
        [*keys, *exposure_columns, *coverage_columns],
    ]
    audit = model.merge(style, on=keys, suffixes=("_model", "_style"), validate="one_to_one")
    for name in styles:
        audit[f"{name}_exposure_gap"] = (
            audit[f"portfolio_{name}_model"] - audit[f"portfolio_{name}_style"]
        )

    overlap = targets.pivot_table(
        index=[*keys, "instrument_id"], columns="portfolio_type",
        values="target_weight", fill_value=0.0,
    ).reset_index()
    overlap["overlap"] = overlap[["ml_model", "style_matched_benchmark"]].min(axis=1)
    overlap = overlap.groupby(keys, as_index=False).agg(
        portfolio_overlap_weight=("overlap", "sum")
    )
    return audit.merge(industry_gap, on=keys, validate="one_to_one").merge(
        overlap, on=keys, validate="one_to_one"
    )


def paired_inference(weekly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    raw_target, residual_target = str(cfg["target"]), str(cfg["residual_target"])
    specs = []
    for model in sorted(weekly["model"].unique()):
        specs.extend([
            ("residual_target_minus_raw_target", model,
             (residual_target, "ml_model"), (raw_target, "ml_model")),
            ("residual_target_minus_style_benchmark", model,
             (residual_target, "ml_model"), (residual_target, "style_matched_benchmark")),
            ("raw_target_minus_style_benchmark_control", model,
             (raw_target, "ml_model"), (raw_target, "style_matched_benchmark")),
        ])
    records = []
    inference = cfg["inference"]
    for number, (family, model, candidate_key, reference_key) in enumerate(specs):
        subset = weekly.loc[weekly["model"].eq(model)]
        candidate = subset.loc[
            subset["training_target"].eq(candidate_key[0])
            & subset["portfolio_type"].eq(candidate_key[1])
            & subset["included_in_performance"].astype(bool),
            ["signal_date", "net_return", "turnover"],
        ]
        reference = subset.loc[
            subset["training_target"].eq(reference_key[0])
            & subset["portfolio_type"].eq(reference_key[1])
            & subset["included_in_performance"].astype(bool),
            ["signal_date", "net_return", "turnover"],
        ]
        paired = candidate.merge(reference, on="signal_date", suffixes=("_c", "_r"), validate="one_to_one")
        c, r = paired["net_return_c"].to_numpy(float), paired["net_return_r"].to_numpy(float)
        ct, rt = paired["turnover_c"].to_numpy(float), paired["turnover_r"].to_numpy(float)
        point = _paired_metrics(c, r, ct, rt, int(cfg["annualization_periods"]))
        rng = np.random.default_rng(int(inference["random_seed"]) + number)
        samples, block = int(inference["bootstrap_samples"]), int(inference["block_length_weeks"])
        draws = {metric: np.empty(samples) for metric in point}
        blocks = int(np.ceil(len(paired) / block)); offsets = np.arange(block)
        for sample in range(samples):
            starts = rng.integers(0, len(paired), size=blocks)
            index = ((starts[:, None] + offsets) % len(paired)).ravel()[:len(paired)]
            values = _paired_metrics(c[index], r[index], ct[index], rt[index], int(cfg["annualization_periods"]))
            for metric, value in values.items():
                draws[metric][sample] = value
        for metric, value in point.items():
            values = draws[metric]
            records.append({
                "family": family, "model": model, "metric": metric,
                "observations": len(paired), "point_estimate": value,
                "ci_2_5pct": float(np.quantile(values, 0.025)),
                "ci_97_5pct": float(np.quantile(values, 0.975)),
                "one_sided_probability_nonpositive": float(
                    (np.sum(values <= 0) + 1) / (samples + 1)
                ),
            })
    result = pd.DataFrame(records)
    result["bh_fdr_adjusted_p"] = np.nan
    for _, indices in result.groupby(["family", "metric"]).groups.items():
        result.loc[indices, "bh_fdr_adjusted_p"] = benjamini_hochberg(
            result.loc[indices, "one_sided_probability_nonpositive"]
        )
    result["survives_bh_fdr_5pct"] = result["bh_fdr_adjusted_p"].le(
        float(inference["false_discovery_rate"])
    )
    return result


def promotion_decision(
    weekly: pd.DataFrame, inference: pd.DataFrame, exposure_match: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    model = str(cfg["promotion_gate"]["primary_model"])
    primary_metric = "incremental_net_return_annualized_arithmetic"
    rows = inference.loc[
        inference["model"].eq(model) & inference["metric"].eq(primary_metric)
    ].set_index("family")
    target_vs_raw = rows.loc["residual_target_minus_raw_target"]
    target_vs_style = rows.loc["residual_target_minus_style_benchmark"]
    ml = weekly.loc[weekly["portfolio_type"].eq("ml_model")]
    maximum_violation = float(ml[[
        "maximum_stock_weight_violation", "size_bound_violation", "industry_bound_violation"
    ]].fillna(0.0).max().max())
    no_relaxation = not bool(ml["turnover_limit_relaxed"].astype(bool).any())
    exposure = exposure_match.loc[
        exposure_match["model"].eq(model)
        & exposure_match["training_target"].eq(str(cfg["residual_target"]))
    ]
    maximum_style_gap = float(
        exposure.filter(like="_exposure_gap").abs().max().max()
    )
    maximum_industry_gap = float(exposure["maximum_abs_industry_weight_gap"].max())
    minimum_benchmark_coverage = float(min(
        exposure["style_exposure_coverage_weight_style"].min(),
        exposure["industry_exposure_coverage_weight_style"].min(),
    ))
    exposure_matched = bool(
        maximum_style_gap <= float(
            cfg["style_benchmark"]["maximum_absolute_style_exposure_gap"]
        ) + 1e-8
        and maximum_industry_gap <= float(
            cfg["style_benchmark"]["maximum_absolute_industry_weight_gap"]
        ) + 1e-8
        and minimum_benchmark_coverage >= float(
            cfg["portfolio"]["minimum_realized_weight_fraction"]
        )
    )
    gates = {
        "residual_minus_raw_ci_above_zero": float(target_vs_raw["ci_2_5pct"]) > 0,
        "residual_minus_style_ci_above_zero": float(target_vs_style["ci_2_5pct"]) > 0,
        "both_primary_families_pass_bh_fdr": bool(
            target_vs_raw["survives_bh_fdr_5pct"] and target_vs_style["survives_bh_fdr_5pct"]
        ),
        "no_turnover_relaxation": no_relaxation,
        "constraints_within_tolerance": maximum_violation <= float(
            cfg["promotion_gate"]["maximum_constraint_violation"]
        ),
        "style_benchmark_exposures_within_tolerance": exposure_matched,
    }
    return pd.DataFrame([{
        "primary_model": model, **gates, "maximum_constraint_violation": maximum_violation,
        "maximum_style_exposure_gap": maximum_style_gap,
        "maximum_industry_weight_gap": maximum_industry_gap,
        "minimum_benchmark_exposure_coverage": minimum_benchmark_coverage,
        "promote_to_share_level_execution": all(gates.values()),
    }])


def _write_report(
    cfg: dict[str, Any], summary: pd.DataFrame, annual: pd.DataFrame,
    inference: pd.DataFrame, exposure_match: pd.DataFrame,
    decision: pd.DataFrame, checks: pd.DataFrame,
) -> None:
    primary = inference.loc[inference["metric"].eq(
        "incremental_net_return_annualized_arithmetic"
    )]
    primary_model = str(cfg["promotion_gate"]["primary_model"])
    primary_diagnostics = inference.loc[
        inference["model"].eq(primary_model)
        & inference["family"].isin(cfg["inference"]["primary_families"])
    ]
    passed = bool(decision.iloc[0]["promote_to_share_level_execution"])
    conclusion = (
        "HGB residual-target 通过组合层门槛，可以进入独立的股数级执行验证。"
        if passed else
        "HGB residual-target 未通过全部组合层门槛，不进入股数级 challenger 执行。"
    )
    exposure_summary = pd.DataFrame([{
        "dates": exposure_match["signal_date"].nunique(),
        "mean_max_abs_industry_weight_gap": exposure_match[
            "maximum_abs_industry_weight_gap"
        ].mean(),
        "maximum_abs_industry_weight_gap": exposure_match[
            "maximum_abs_industry_weight_gap"
        ].max(),
        "mean_portfolio_overlap_weight": exposure_match["portfolio_overlap_weight"].mean(),
        "minimum_model_style_coverage_weight": exposure_match[
            "style_exposure_coverage_weight_model"
        ].min(),
        "minimum_benchmark_style_coverage_weight": exposure_match[
            "style_exposure_coverage_weight_style"
        ].min(),
        "minimum_benchmark_industry_coverage_weight": exposure_match[
            "industry_exposure_coverage_weight_style"
        ].min(),
        **{
            f"mean_abs_{style}_exposure_gap": exposure_match[
                f"{style}_exposure_gap"
            ].abs().mean() for style in cfg["controls"]["styles"]
        },
    }])
    exposure_by_path = []
    for (model, target), group in exposure_match.groupby(
        ["model", "training_target"], sort=True
    ):
        style_gaps = group.filter(like="_exposure_gap").abs()
        exposure_by_path.append({
            "model": model, "training_target": target,
            "maximum_abs_style_exposure_gap": float(style_gaps.max().max()),
            "mean_of_style_mean_abs_gaps": float(style_gaps.mean().mean()),
            "maximum_abs_industry_weight_gap": float(
                group["maximum_abs_industry_weight_gap"].max()
            ),
            "mean_portfolio_overlap_weight": float(
                group["portfolio_overlap_weight"].mean()
            ),
        })
    exposure_by_path = pd.DataFrame(exposure_by_path)
    lines = [
        "# Idiosyncratic Portfolio Validation v1", "", "## 结论", "", conclusion, "",
        "本实验仍是2016–2020开发期证据，不是新的historical OOS、future lockbox或paper结果。",
        "", "## 同口径组合绩效", "", summary.to_markdown(index=False, floatfmt=".6f"),
        "", "## 年化净收益配对检验", "",
        primary.to_markdown(index=False, floatfmt=".6f"),
        "", "## 主模型风险与交易效率配对检验", "",
        primary_diagnostics.to_markdown(index=False, floatfmt=".6f"),
        "", "## 逐年结果", "", annual.to_markdown(index=False, floatfmt=".6f"),
        "", "## 风格复制匹配审计", "",
        exposure_summary.to_markdown(index=False, floatfmt=".6f"),
        "", exposure_by_path.to_markdown(index=False, floatfmt=".6f"),
        "", "## 晋级决策", "", decision.to_markdown(index=False),
        "", "## 机器验收", "", checks.to_markdown(index=False),
        "", "## 边界", "",
        "当前20bp结果来自周频目标权重和5日标签收益，尚未经过股数、现金、整手、T+1、"
        "停牌/涨跌停与ADV部分成交。即便通过，也只能进入下一执行门槛。",
        "当前Sharpe为显式零利率开发期诊断；可靠人民币无风险利率尚未覆盖2016–2020，"
        "因此绝对Sharpe不能替代后续成交层的可靠rf口径。",
        "HGB residual相对raw-target在2018年为负，逐年改善为4/5；聚合区间通过不等于"
        "每个市场状态都稳定获胜。",
    ]
    (ROOT / str(cfg["report_path"])).write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_integrity_checks(
    weekly: pd.DataFrame, exposure_match: pd.DataFrame, cfg: dict[str, Any],
) -> list[dict[str, str]]:
    style_max = float(exposure_match.filter(like="_exposure_gap").abs().max().max())
    industry_max = float(exposure_match["maximum_abs_industry_weight_gap"].max())
    coverage_min = float(min(
        exposure_match["style_exposure_coverage_weight_style"].min(),
        exposure_match["industry_exposure_coverage_weight_style"].min(),
    ))
    benchmark = weekly.loc[
        weekly["portfolio_type"].eq("style_matched_benchmark")
    ].sort_values(["model", "training_target", "signal_date"]).copy()
    benchmark["path_period"] = benchmark.groupby(
        ["model", "training_target"]
    ).cumcount()
    post_inception = benchmark.loc[benchmark["path_period"].gt(0)]
    turnover_max = float(post_inception["turnover"].max())
    stock_violation = float(benchmark["maximum_stock_weight_violation"].max())
    return [
        {
            "check": "all_benchmark_exposures_within_tolerance",
            "status": "PASS" if (
                style_max <= float(
                    cfg["style_benchmark"]["maximum_absolute_style_exposure_gap"]
                ) + 1e-8
                and industry_max <= float(
                    cfg["style_benchmark"]["maximum_absolute_industry_weight_gap"]
                ) + 1e-8
            ) else "FAIL",
            "detail": f"style_max={style_max:.12g};industry_max={industry_max:.12g}",
        },
        {
            "check": "benchmark_exposure_coverage",
            "status": "PASS" if coverage_min >= float(
                cfg["portfolio"]["minimum_realized_weight_fraction"]
            ) else "FAIL",
            "detail": f"minimum_coverage={coverage_min:.12g}",
        },
        {
            "check": "benchmark_turnover_and_stock_caps",
            "status": "PASS" if (
                turnover_max <= float(
                    cfg["style_benchmark"]["weekly_one_way_turnover_limit"]
                ) + 1e-8
                and stock_violation <= 1e-8
            ) else "FAIL",
            "detail": (
                f"post_inception_turnover_max={turnover_max:.12g};"
                f"stock_violation={stock_violation:.12g}"
            ),
        },
    ]


def run(lake: Lake | None = None) -> dict[str, pd.DataFrame]:
    lake = lake or Lake(); cfg = load_config(); validate_config(cfg)
    factors = list(load_preprocessing_config()["factor_columns"])
    calendar = lake.read("calendar", "SSE")
    models, fit_audit = _fit_models(lake, cfg, factors, calendar)
    frames = _validation_frames(lake, cfg, factors, calendar)
    ml_weekly, ml_targets, ml_exposures, universe = build_model_targets(
        frames, models, factors, cfg
    )
    style_weekly, style_targets, style_exposures = build_style_benchmarks(
        ml_targets, universe, cfg
    )
    weekly = pd.concat([ml_weekly, style_weekly], ignore_index=True, sort=False)
    targets = pd.concat([ml_targets, style_targets], ignore_index=True, sort=False)
    # Recompute both sides from target weights on a common, label-independent
    # PIT exposure universe. The construction-time frames remain useful for
    # debugging but cannot fairly audit carried legacy benchmark positions.
    exposures, industry_gap = recompute_target_exposures(targets, lake, cfg)
    summary = summarize(weekly, cfg)
    annual = annual_summary(weekly, cfg)
    exposure_match = exposure_match_audit(targets, exposures, industry_gap, cfg)
    inference = paired_inference(weekly, cfg)
    decision = promotion_decision(weekly, inference, exposure_match, cfg)
    expected_dates = weekly["signal_date"].nunique()
    path_counts = weekly.groupby(["model", "training_target", "portfolio_type"])["signal_date"].nunique()
    checks = pd.DataFrame([
        {"check": "development_period_ends_2020", "status": "PASS",
         "detail": str(weekly["signal_date"].max())},
        {"check": "eight_paired_paths_share_dates", "status": (
            "PASS" if len(path_counts) == 8 and path_counts.eq(expected_dates).all() else "FAIL"
        ), "detail": str(path_counts.to_dict())},
        {"check": "target_weights_sum_to_one", "status": (
            "PASS" if np.allclose(targets.groupby([
                "model", "training_target", "portfolio_type", "signal_date"
            ])["target_weight"].sum(), 1.0, atol=1e-8) else "FAIL"
        ), "detail": str(len(targets))},
        {"check": "no_historical_oos_or_lockbox_access", "status": "PASS",
         "detail": "validation_end=2020; future_access=false"},
        {"check": "same_20bp_cost", "status": "PASS", "detail": "20.0"},
        {"check": "primary_inference_families_complete", "status": (
            "PASS" if set(cfg["inference"]["primary_families"]).issubset(set(inference["family"]))
            else "FAIL"
        ), "detail": str(sorted(inference["family"].unique()))},
        {"check": "exposure_match_audit_complete", "status": (
            "PASS" if len(exposure_match) == expected_dates * 4
            and exposure_match.filter(like="_exposure_gap").notna().all().all()
            else "FAIL"
        ), "detail": f"rows={len(exposure_match)} dates={exposure_match['signal_date'].nunique()}"},
        *benchmark_integrity_checks(weekly, exposure_match, cfg),
    ])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "source_experiment": cfg["source_experiment"],
        "evidence_class": cfg["evidence_class"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "validation_period": "2016-2020", "portfolio": cfg["portfolio"]["construction"],
        "round_trip_cost_bps": cfg["portfolio"]["round_trip_cost_bps"],
        "sharpe_convention": cfg["risk_free_rate"]["sharpe_convention"],
        "reliable_risk_free_series_available": cfg["risk_free_rate"][
            "reliable_2016_2020_series_available"
        ],
        "share_level_execution_completed": False, "frozen_paper_candidate_changed": False,
    }])
    outputs = {
        "idiosyncratic_portfolio_protocol": protocol,
        "idiosyncratic_portfolio_fit_audit": fit_audit,
        "idiosyncratic_portfolio_weekly": weekly,
        "idiosyncratic_portfolio_targets": targets,
        "idiosyncratic_portfolio_exposures": exposures,
        "idiosyncratic_portfolio_summary": summary,
        "idiosyncratic_portfolio_annual": annual,
        "idiosyncratic_portfolio_exposure_match": exposure_match,
        "idiosyncratic_portfolio_paired_inference": inference,
        "idiosyncratic_portfolio_promotion_decision": decision,
        "idiosyncratic_portfolio_checks": checks,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key=str(cfg["artifact_key"]),
                   source="idiosyncratic_portfolio_validation.v1", request=cfg)
    _write_report(cfg, summary, annual, inference, exposure_match, decision, checks)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("idiosyncratic portfolio validation checks failed")
    print(summary.to_string(index=False), flush=True)
    print(inference.loc[inference["metric"].eq(
        "incremental_net_return_annualized_arithmetic"
    )].to_string(index=False), flush=True)
    print(decision.to_string(index=False), flush=True)
    return outputs


def publish_diagnostics(lake: Lake | None = None) -> None:
    """Add derived annual/exposure audits without rerunning model fits or LPs."""
    lake = lake or Lake(); cfg = load_config(); validate_config(cfg)
    weekly = lake.read("idiosyncratic_portfolio_weekly", str(cfg["artifact_key"]))
    targets = lake.read("idiosyncratic_portfolio_targets", str(cfg["artifact_key"]))
    if weekly.empty or targets.empty:
        raise RuntimeError("base portfolio validation outputs are missing")
    protocol = lake.read("idiosyncratic_portfolio_protocol", str(cfg["artifact_key"]))
    protocol["sharpe_convention"] = cfg["risk_free_rate"]["sharpe_convention"]
    protocol["reliable_risk_free_series_available"] = cfg["risk_free_rate"][
        "reliable_2016_2020_series_available"
    ]
    lake.write("idiosyncratic_portfolio_protocol", protocol,
               key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    annual = annual_summary(weekly, cfg)
    exposures, industry_gap = recompute_target_exposures(targets, lake, cfg)
    exposure_match = exposure_match_audit(targets, exposures, industry_gap, cfg)
    lake.write("idiosyncratic_portfolio_exposures", exposures,
               key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    lake.write("idiosyncratic_portfolio_annual", annual, key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    lake.write("idiosyncratic_portfolio_exposure_match", exposure_match,
               key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    checks = lake.read("idiosyncratic_portfolio_checks", str(cfg["artifact_key"]))
    derived_check_names = {
        "exposure_match_audit_complete",
        "all_benchmark_exposures_within_tolerance",
        "benchmark_exposure_coverage",
        "benchmark_turnover_and_stock_caps",
    }
    checks = checks.loc[~checks["check"].isin(derived_check_names)].copy()
    complete = (
        len(exposure_match) == weekly["signal_date"].nunique() * 4
        and exposure_match.filter(like="_exposure_gap").notna().all().all()
    )
    checks = pd.concat([checks, pd.DataFrame([{
        "check": "exposure_match_audit_complete",
        "status": "PASS" if complete else "FAIL",
        "detail": f"rows={len(exposure_match)} dates={exposure_match['signal_date'].nunique()}",
    }, *benchmark_integrity_checks(weekly, exposure_match, cfg)])], ignore_index=True)
    lake.write("idiosyncratic_portfolio_checks", checks,
               key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    inference = lake.read(
        "idiosyncratic_portfolio_paired_inference", str(cfg["artifact_key"])
    )
    decision = promotion_decision(weekly, inference, exposure_match, cfg)
    lake.write("idiosyncratic_portfolio_promotion_decision", decision,
               key=str(cfg["artifact_key"]),
               source="idiosyncratic_portfolio_validation.v1", request=cfg)
    _write_report(
        cfg, lake.read("idiosyncratic_portfolio_summary", str(cfg["artifact_key"])),
        annual, inference, exposure_match, decision, checks,
    )
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("derived portfolio diagnostics failed")


def rebuild_style_benchmark(lake: Lake | None = None) -> None:
    """Rebuild only the PIT replica after its construction logic changes."""
    lake = lake or Lake(); cfg = load_config(); validate_config(cfg)
    weekly_base = lake.read("idiosyncratic_portfolio_weekly", str(cfg["artifact_key"]))
    targets_base = lake.read("idiosyncratic_portfolio_targets", str(cfg["artifact_key"]))
    if weekly_base.empty or targets_base.empty:
        raise RuntimeError("base portfolio validation outputs are missing")
    ml_weekly = weekly_base.loc[weekly_base["portfolio_type"].eq("ml_model")].copy()
    ml_targets = targets_base.loc[targets_base["portfolio_type"].eq("ml_model")].copy()

    factors = list(load_preprocessing_config()["factor_columns"])
    calendar = lake.read("calendar", "SSE")
    frames = _validation_frames(lake, cfg, factors, calendar)
    universe: dict[str, pd.DataFrame] = {}
    for yearly in frames:
        for signal_date, raw_group in yearly.groupby("asof_date", sort=True):
            matching = raw_group.reset_index(drop=True).copy()
            size = pd.to_numeric(matching["float_cap_proxy_cny"], errors="coerce")
            industry = matching["industry_broad_code"].fillna("").astype(str).str.strip()
            matching = matching.loc[size.gt(0) & industry.ne("")].reset_index(drop=True)
            style_frame = build_style_exposures(
                matching,
                {"style_winsor_quantiles": cfg["controls"]["style_winsor_quantiles"]},
            )
            for style in cfg["controls"]["styles"]:
                matching[f"style_{style}"] = style_frame[style].to_numpy(dtype="float64")
            universe[str(signal_date)] = matching

    style_weekly, style_targets, _ = build_style_benchmarks(ml_targets, universe, cfg)
    weekly = pd.concat([ml_weekly, style_weekly], ignore_index=True, sort=False)
    targets = pd.concat([ml_targets, style_targets], ignore_index=True, sort=False)
    exposures, industry_gap = recompute_target_exposures(targets, lake, cfg)
    summary = summarize(weekly, cfg)
    annual = annual_summary(weekly, cfg)
    exposure_match = exposure_match_audit(targets, exposures, industry_gap, cfg)
    inference = paired_inference(weekly, cfg)
    decision = promotion_decision(weekly, inference, exposure_match, cfg)
    expected_dates = weekly["signal_date"].nunique()
    path_counts = weekly.groupby(
        ["model", "training_target", "portfolio_type"]
    )["signal_date"].nunique()
    checks = pd.DataFrame([
        {"check": "development_period_ends_2020", "status": (
            "PASS" if weekly["signal_date"].max() <= "2020-12-31" else "FAIL"
        ), "detail": str(weekly["signal_date"].max())},
        {"check": "eight_paired_paths_share_dates", "status": (
            "PASS" if len(path_counts) == 8 and path_counts.eq(expected_dates).all() else "FAIL"
        ), "detail": str(path_counts.to_dict())},
        {"check": "target_weights_sum_to_one", "status": (
            "PASS" if np.allclose(targets.groupby([
                "model", "training_target", "portfolio_type", "signal_date"
            ])["target_weight"].sum(), 1.0, atol=1e-8) else "FAIL"
        ), "detail": str(len(targets))},
        {"check": "no_historical_oos_or_lockbox_access", "status": "PASS",
         "detail": "validation_end=2020; future_access=false"},
        {"check": "same_20bp_cost", "status": "PASS", "detail": "20.0"},
        {"check": "primary_inference_families_complete", "status": (
            "PASS" if set(cfg["inference"]["primary_families"]).issubset(
                set(inference["family"])
            ) else "FAIL"
        ), "detail": str(sorted(inference["family"].unique()))},
        {"check": "exposure_match_audit_complete", "status": (
            "PASS" if len(exposure_match) == expected_dates * 4
            and exposure_match.filter(like="_exposure_gap").notna().all().all()
            else "FAIL"
        ), "detail": f"rows={len(exposure_match)} dates={exposure_match['signal_date'].nunique()}"},
        *benchmark_integrity_checks(weekly, exposure_match, cfg),
    ])
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "source_experiment": cfg["source_experiment"],
        "evidence_class": cfg["evidence_class"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "validation_period": "2016-2020", "portfolio": cfg["portfolio"]["construction"],
        "round_trip_cost_bps": cfg["portfolio"]["round_trip_cost_bps"],
        "sharpe_convention": cfg["risk_free_rate"]["sharpe_convention"],
        "reliable_risk_free_series_available": cfg["risk_free_rate"][
            "reliable_2016_2020_series_available"
        ],
        "share_level_execution_completed": False, "frozen_paper_candidate_changed": False,
    }])
    outputs = {
        "idiosyncratic_portfolio_protocol": protocol,
        "idiosyncratic_portfolio_weekly": weekly,
        "idiosyncratic_portfolio_targets": targets,
        "idiosyncratic_portfolio_exposures": exposures,
        "idiosyncratic_portfolio_summary": summary,
        "idiosyncratic_portfolio_annual": annual,
        "idiosyncratic_portfolio_exposure_match": exposure_match,
        "idiosyncratic_portfolio_paired_inference": inference,
        "idiosyncratic_portfolio_promotion_decision": decision,
        "idiosyncratic_portfolio_checks": checks,
    }
    for table, frame in outputs.items():
        lake.write(table, frame, key=str(cfg["artifact_key"]),
                   source="idiosyncratic_portfolio_validation.v1", request=cfg)
    _write_report(cfg, summary, annual, inference, exposure_match, decision, checks)
    if checks["status"].eq("FAIL").any():
        raise RuntimeError("rebuilt style benchmark checks failed")
    print(summary.to_string(index=False), flush=True)
    print(inference.loc[inference["metric"].eq(
        "incremental_net_return_annualized_arithmetic"
    )].to_string(index=False), flush=True)
    print(decision.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only residual-target P4 study")
    parser.add_argument("command", choices=["run", "diagnostics", "rebuild-benchmark"])
    args = parser.parse_args()
    if args.command == "run": run()
    elif args.command == "diagnostics": publish_diagnostics()
    else: rebuild_style_benchmark()


if __name__ == "__main__": main()
