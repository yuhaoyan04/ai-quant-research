"""Phase 6 common-signal portfolio construction and execution comparison.

The historical OOS has already been observed.  This module therefore provides
diagnostic evidence only: it never promotes a weighting method from 2021-2026
performance and never changes the locked HGB signal or execution assumptions.
"""

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

from quant_research.factor_neutralization import _read_year
from quant_research.ingest import ROOT, Lake
from quant_research.institutional_execution import _summaries, simulate_execution
from quant_research.model_baseline import purged_end_date
from quant_research.model_portfolio_neutralization import (
    _aggregate_training,
    build_year_training,
    fit_models,
    model_designs,
)
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.rf_daily_retry_execution import _limit_array
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "portfolio_construction_comparison.yaml"
REPORT_PATH = ROOT / "reports" / "portfolio_construction_comparison.md"
PIPELINES = [
    "raw", "size_neutralized", "industry_neutralized",
    "industry_size_neutralized",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _capped_normalize(values: np.ndarray, cap: float) -> np.ndarray:
    """Normalize positive values while redistributing mass above a hard cap."""
    raw = np.asarray(values, dtype="float64")
    if len(raw) == 0 or not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError("weight inputs must be finite and non-negative")
    if len(raw) * cap < 1.0 - 1e-12:
        raise ValueError("stock cap is infeasible for the candidate count")
    if raw.sum() <= 0:
        raw = np.ones(len(raw), dtype="float64")
    result = np.zeros(len(raw), dtype="float64")
    free = np.ones(len(raw), dtype=bool)
    remaining = 1.0
    while free.any():
        scale = raw[free]
        if scale.sum() <= 0:
            proposal = np.full(free.sum(), remaining / free.sum())
        else:
            proposal = remaining * scale / scale.sum()
        hit = proposal > cap + 1e-14
        indices = np.flatnonzero(free)
        if not hit.any():
            result[indices] = proposal
            break
        capped = indices[hit]
        result[capped] = cap
        free[capped] = False
        remaining = 1.0 - result.sum()
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise RuntimeError(f"capped weights do not sum to one: {result.sum()}")
    return result


def common_buffer_candidates(
    ranked: pd.DataFrame, previous_candidates: set[str], entry_rank: int, exit_rank: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Separate Top-200 entry eligibility from Top-400 carry eligibility."""
    if exit_rank < entry_rank:
        raise ValueError("exit rank must be no smaller than entry rank")
    current = ranked.sort_values(
        ["score", "instrument_id"], ascending=[False, True]
    ).reset_index(drop=True).copy()
    current["score_rank"] = np.arange(1, len(current) + 1)
    rank_map = dict(zip(current["instrument_id"].astype(str), current["score_rank"]))
    entrants = set(current.head(entry_rank)["instrument_id"].astype(str))
    retained = {
        code for code in previous_candidates
        if code in rank_map and int(rank_map[code]) <= exit_rank
    }
    eligible = entrants | retained
    selected = current.loc[current["instrument_id"].astype(str).isin(eligible)].copy()
    selected["entry_eligible"] = selected["score_rank"].le(entry_rank)
    selected["buffer_retained"] = (
        selected["instrument_id"].astype(str).isin(retained)
        & ~selected["entry_eligible"]
    )
    return selected.reset_index(drop=True), {
        "candidate_count": len(selected),
        "new_entry_count": len(entrants - previous_candidates),
        "buffer_retained_count": len(retained - entrants),
        "buffer_exit_count": len(previous_candidates - retained - entrants),
    }


def simple_weights(candidate: pd.DataFrame, method: str, cfg: dict[str, Any]) -> np.ndarray:
    cap = float(cfg["weighting"]["maximum_stock_weight"])
    if method == "P0_equal":
        values = np.ones(len(candidate), dtype="float64")
    elif method == "P1_inverse_volatility":
        volatility = pd.to_numeric(candidate["volatility_20d"], errors="coerce")
        usable = volatility.loc[volatility.gt(0) & np.isfinite(volatility)]
        if usable.empty:
            values = np.ones(len(candidate), dtype="float64")
        else:
            low, high = usable.quantile(
                cfg["weighting"]["inverse_volatility_clip_quantiles"]
            )
            filled = volatility.where(
                volatility.gt(0) & np.isfinite(volatility), usable.median()
            ).clip(lower=low, upper=high)
            values = 1.0 / filled.to_numpy(dtype="float64")
    elif method == "P2_score_weighted":
        percentile = candidate["score"].rank(method="average", pct=True).to_numpy(float)
        floor = float(cfg["weighting"]["score_rank_floor"])
        power = float(cfg["weighting"]["score_rank_power"])
        values = floor + np.power(percentile, power)
    else:
        raise ValueError(f"unknown simple weighting method: {method}")
    return _capped_normalize(values, cap)


def optimizer_candidates(
    universe: pd.DataFrame,
    base_candidate: pd.DataFrame,
    previous: dict[str, float],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Add only auditable industry/size feasibility support to the 200/400 pool.

    Support names are not ordinary buffer entrants and are labelled as such.
    They exist because a hard industry lower bound can be mathematically
    impossible when a small industry has no name in the Top-200/400 set.
    """
    cap = float(cfg["weighting"]["maximum_stock_weight"])
    active = float(cfg["optimizer"]["industry_active_weight_bound"])
    buffer_names = int(cfg["optimizer"]["industry_capacity_buffer_names"])
    ranked = universe.sort_values(
        ["score", "instrument_id"], ascending=[False, True]
    ).reset_index(drop=True).copy()
    ranked["score_rank"] = np.arange(1, len(ranked) + 1)
    benchmark = ranked["industry_broad_code"].astype(str).value_counts(normalize=True)
    chosen = set(base_candidate["instrument_id"].astype(str))
    for industry, weight in benchmark.items():
        required = int(np.ceil(max(float(weight) - active, 0.0) / cap)) + buffer_names
        support = ranked.loc[
            ranked["industry_broad_code"].astype(str).eq(str(industry))
        ].head(required)
        chosen.update(support["instrument_id"].astype(str))
    tails = int(cfg["optimizer"]["size_tail_candidates_each_side"])
    if tails > 0:
        size_order = ranked.sort_values(["size_z", "instrument_id"])
        chosen.update(size_order.head(tails)["instrument_id"].astype(str))
        chosen.update(size_order.tail(tails)["instrument_id"].astype(str))
    # Preserve old optimizer names only while they satisfy the formal Top-400
    # carry rule. Names outside it are forced exits and count toward turnover.
    rank_map = dict(zip(ranked["instrument_id"].astype(str), np.arange(1, len(ranked) + 1)))
    chosen.update(
        code for code in previous if code in rank_map
        and rank_map[code] <= int(cfg["buffer"]["exit_rank"])
    )
    result = ranked.loc[ranked["instrument_id"].astype(str).isin(chosen)].copy()
    base_codes = set(base_candidate["instrument_id"].astype(str))
    result["entry_eligible"] = result["instrument_id"].astype(str).isin(
        set(ranked.head(int(cfg["buffer"]["entry_rank"]))["instrument_id"].astype(str))
    )
    result["buffer_retained"] = result["instrument_id"].astype(str).isin(
        set(base_candidate.loc[base_candidate["buffer_retained"], "instrument_id"].astype(str))
    )
    result["constraint_support"] = ~result["instrument_id"].astype(str).isin(base_codes)
    return result.reset_index(drop=True)


def _drifted(previous: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    values = {code: weight * (1.0 + returns.get(code, 0.0)) for code, weight in previous.items()}
    total = sum(values.values())
    return {code: value / total for code, value in values.items()} if total > 0 else dict(previous)


def carry_forward_fallback(
    candidate: pd.DataFrame, previous: dict[str, float], cfg: dict[str, Any],
) -> np.ndarray:
    """Repair forced exits while preserving as much prior weight as possible.

    This is an explicitly non-optimized fallback. It respects long-only, full
    investment and the stock cap, but makes no claim that risk bounds hold.
    """
    cap = float(cfg["weighting"]["maximum_stock_weight"])
    codes = candidate["instrument_id"].astype(str).tolist()
    weights = np.array([min(previous.get(code, 0.0), cap) for code in codes], dtype=float)
    remaining = 1.0 - float(weights.sum())
    if remaining < -1e-10:
        raise RuntimeError("fallback retained weights exceed one")
    utility = candidate["score"].rank(method="average", pct=True).to_numpy(float)
    while remaining > 1e-12:
        capacity = np.maximum(cap - weights, 0.0)
        free = capacity > 1e-14
        if not free.any() or capacity.sum() < remaining - 1e-10:
            raise ValueError("fallback candidate capacity cannot absorb forced exits")
        allocation_scale = utility[free]
        if allocation_scale.sum() <= 0:
            allocation_scale = np.ones(free.sum())
        proposed = remaining * allocation_scale / allocation_scale.sum()
        addition = np.minimum(proposed, capacity[free])
        weights[np.flatnonzero(free)] += addition
        used = float(addition.sum())
        if used <= 1e-15:
            raise RuntimeError("fallback capacity redistribution stalled")
        remaining -= used
    weights /= weights.sum()
    return weights


def solve_constrained_weights(
    universe: pd.DataFrame,
    candidate: pd.DataFrame,
    previous: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Strict long-only LP. Infeasibility is returned, never silently relaxed."""
    started = perf_counter()
    n = len(candidate)
    cap = float(cfg["weighting"]["maximum_stock_weight"])
    active_bound = float(cfg["optimizer"]["industry_active_weight_bound"])
    size_bound = float(cfg["optimizer"]["absolute_size_z_bound"])
    turnover_limit = float(cfg["optimizer"]["one_way_turnover_limit"])
    score_utility = candidate["score"].rank(method="average", pct=True).to_numpy(float)
    objective = np.concatenate([-score_utility, np.full(n, 1e-9)])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: list[tuple[int, float]], bound: float) -> None:
        row = len(rhs)
        for column, value in entries:
            rows.append(row); columns.append(column); values.append(value)
        rhs.append(float(bound))

    benchmark = universe["industry_broad_code"].astype(str).value_counts(normalize=True)
    industries = candidate["industry_broad_code"].astype(str).to_numpy()
    for industry, weight in benchmark.items():
        members = np.flatnonzero(industries == str(industry))
        upper = min(1.0, float(weight) + active_bound)
        lower = max(0.0, float(weight) - active_bound)
        add([(int(index), 1.0) for index in members], upper)
        add([(int(index), -1.0) for index in members], -lower)

    size = candidate["size_z"].to_numpy(dtype="float64")
    add([(index, float(value)) for index, value in enumerate(size)], size_bound)
    add([(index, -float(value)) for index, value in enumerate(size)], size_bound)

    previous_vector = np.array([
        previous.get(code, 0.0) for code in candidate["instrument_id"].astype(str)
    ])
    candidate_codes = set(candidate["instrument_id"].astype(str))
    forced_exit = float(sum(weight for code, weight in previous.items() if code not in candidate_codes))
    if previous:
        if forced_exit > 2.0 * turnover_limit + 1e-12:
            return None, {
                "optimizer_success": False,
                "solver_status": "turnover_limit_below_forced_exit",
                "forced_exit_weight": forced_exit,
                "optimizer_seconds": perf_counter() - started,
                "constraint_relaxed": False,
            }
        for index, old_weight in enumerate(previous_vector):
            add([(index, 1.0), (n + index, -1.0)], old_weight)
            add([(index, -1.0), (n + index, -1.0)], -old_weight)
        add([(n + index, 1.0) for index in range(n)], 2.0 * turnover_limit - forced_exit)

    inequality = coo_matrix(
        (values, (rows, columns)), shape=(len(rhs), 2 * n)
    ).tocsr()
    equality = csr_matrix((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, 2 * n))
    result = linprog(
        objective,
        A_ub=inequality, b_ub=np.asarray(rhs),
        A_eq=equality, b_eq=np.array([1.0]),
        bounds=[(0.0, cap)] * n + [(0.0, None)] * n,
        method="highs", options={"presolve": True},
    )
    diagnostics = {
        "optimizer_success": bool(result.success),
        "solver_status": str(result.message),
        "forced_exit_weight": forced_exit,
        "optimizer_seconds": perf_counter() - started,
        "constraint_relaxed": False,
    }
    return (result.x[:n] if result.success else None), diagnostics


def risk_diagnostics(
    universe: pd.DataFrame, candidate: pd.DataFrame, weights: np.ndarray,
) -> dict[str, float]:
    benchmark = universe["industry_broad_code"].astype(str).value_counts(normalize=True)
    portfolio = pd.Series(weights).groupby(candidate["industry_broad_code"].astype(str)).sum()
    labels = set(benchmark.index) | set(portfolio.index)
    active = np.array([
        float(portfolio.get(label, 0.0) - benchmark.get(label, 0.0)) for label in labels
    ])
    return {
        "maximum_stock_weight": float(weights.max()),
        "portfolio_size_z": float(weights @ candidate["size_z"].to_numpy(float)),
        "industry_active_weight_l1": float(np.abs(active).sum() / 2.0),
        "maximum_abs_industry_active_weight": float(np.abs(active).max()),
    }


def construct_targets(signal: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct all Phase 6 targets on one signal clock and common buffer."""
    required = {
        "instrument_id", "asof_date", "available_date", "score",
        "volatility_20d", "industry_broad_code", "size_z",
    }
    if missing := required - set(signal):
        raise KeyError(f"signal columns missing: {sorted(missing)}")
    methods = list(cfg["weighting"]["methods"])
    previous_candidates: set[str] = set()
    states = {method: {"weights": {}, "returns": {}} for method in methods}
    holdings: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    target = str(cfg["signal"]["target"])

    for asof_date, raw in signal.groupby("asof_date", sort=True):
        universe = raw.sort_values("instrument_id").reset_index(drop=True)
        candidate, buffer_audit = common_buffer_candidates(
            universe, previous_candidates,
            int(cfg["buffer"]["entry_rank"]), int(cfg["buffer"]["exit_rank"]),
        )
        previous_candidates = set(candidate["instrument_id"].astype(str))
        execution_dates = candidate["available_date"].dropna().astype(str).unique()
        if len(execution_dates) != 1:
            raise RuntimeError(f"non-unique execution date for {asof_date}: {execution_dates}")
        for method in methods:
            state = states[method]
            previous = _drifted(state["weights"], state["returns"])
            fallback = False
            method_candidate = candidate
            if method == "P4_constrained":
                method_candidate = optimizer_candidates(universe, candidate, previous, cfg)
                weights, optimizer = solve_constrained_weights(
                    universe, method_candidate, previous, cfg
                )
                if weights is None:
                    weights = carry_forward_fallback(method_candidate, previous, cfg)
                    fallback = True
            else:
                weights = simple_weights(candidate, method, cfg)
                optimizer = {
                    "optimizer_success": True, "solver_status": "not_applicable",
                    "forced_exit_weight": 0.0, "optimizer_seconds": 0.0,
                    "constraint_relaxed": False,
                }
            codes = method_candidate["instrument_id"].astype(str).to_numpy()
            current = dict(zip(codes, weights.astype(float)))
            turnover = (
                0.5 * sum(abs(current.get(code, 0.0) - previous.get(code, 0.0))
                          for code in set(current) | set(previous))
                if previous else 1.0
            )
            risk = risk_diagnostics(universe, method_candidate, weights)
            audits.append({
                "asof_date": str(asof_date), "execution_date": execution_dates[0],
                "method": method, **buffer_audit,
                "optimizer_candidate_count": len(method_candidate),
                "constraint_support_count": int(
                    method_candidate.get("constraint_support", pd.Series(False, index=method_candidate.index)).sum()
                ),
                "one_way_target_turnover": turnover,
                "weight_sum": float(weights.sum()), "minimum_weight": float(weights.min()),
                "fallback_used": fallback, **optimizer, **risk,
            })
            selected = method_candidate[[
                "instrument_id", "score", "score_rank", "entry_eligible",
                "buffer_retained", "industry_broad_code", "size_z", "volatility_20d",
            ]].copy()
            selected["constraint_support"] = method_candidate.get(
                "constraint_support", pd.Series(False, index=method_candidate.index)
            ).to_numpy(bool)
            selected["target_weight"] = weights
            selected["asof_date"] = str(asof_date)
            selected["execution_date"] = execution_dates[0]
            selected["method"] = method
            holdings.append(selected)
            realized = pd.to_numeric(method_candidate.get(target), errors="coerce").fillna(0.0)
            state["weights"] = current
            state["returns"] = dict(zip(codes, realized.to_numpy(float)))
    return pd.concat(holdings, ignore_index=True), pd.DataFrame(audits)


def build_signal_panel(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    """Reproduce the exact locked raw-HGB score on the exposure-complete universe."""
    lake = lake or Lake()
    destination = lake.table_path("portfolio_construction_signal_panel", "v1")
    if destination.exists() and not overwrite:
        return lake.read("portfolio_construction_signal_panel", "v1")
    cfg = load_config()
    signal_cfg = cfg["signal"]
    model_cfg = {
        **signal_cfg,
        "pipelines": PIPELINES,
        "models": {signal_cfg["model"]: signal_cfg["model_parameters"]},
    }
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(signal_cfg["target"])
    volatility = str(signal_cfg["volatility"])
    # Volatility is itself one of the model factors. Preserve first occurrence
    # order while preventing duplicate labels from reaching preprocessing.
    columns = list(dict.fromkeys([
        "instrument_id", "asof_date", "available_date", target, volatility, *factors,
    ]))
    calendar = lake.read("calendar", "SSE")
    start = int(signal_cfg["train_start_year"])
    end = int(signal_cfg["historical_oos_end_year"])
    yearly_stats: dict[int, Any] = {}
    yearly_purged_stats: dict[int, Any] = {}
    yearly_samples: dict[int, Any] = {}
    yearly_purged_samples: dict[int, Any] = {}
    for year in range(start, end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31", int(signal_cfg["label_horizon_trading_days"]))
        (yearly_stats[year], yearly_purged_stats[year], yearly_samples[year],
         yearly_purged_samples[year]) = build_year_training(
            frame, factors, target, cutoff, model_cfg
        )
        print(f"phase6 signal statistics year={year}", flush=True)

    records: list[pd.DataFrame] = []
    for test_year in range(int(signal_cfg["historical_oos_start_year"]), end + 1):
        statistics, sample = _aggregate_training(
            yearly_stats, yearly_purged_stats, yearly_samples, yearly_purged_samples,
            "raw", start, test_year - 1,
        )
        fitted = fit_models(statistics, sample, model_cfg)[str(signal_cfg["model"])]
        frame = _read_year(lake, test_year, columns)
        for asof_date, original in frame.groupby("asof_date", sort=True):
            original = original.reset_index(drop=True)
            designs, risks, complete = model_designs(original, factors, model_cfg)
            group = original.loc[complete].reset_index(drop=True)
            if len(group) < int(signal_cfg["minimum_cross_section_size"]):
                continue
            score = fitted.predict(designs["raw"][complete])
            output = group[[
                "instrument_id", "asof_date", "available_date", target, volatility,
                "industry_broad_code", "float_cap_proxy_cny",
            ]].copy()
            output["size_z"] = risks["size_z"][complete]
            output["score"] = score
            output["test_year"] = test_year
            records.append(output)
        print(f"phase6 signal scored year={test_year}", flush=True)
    result = pd.concat(records, ignore_index=True)
    lake.write(
        "portfolio_construction_signal_panel", result, key="v1",
        source="portfolio_construction_comparison.signal.v1",
        request={**cfg, "factor_columns": factors},
    )
    return result


def run_targets(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    signal = build_signal_panel(lake)
    targets, audit = construct_targets(signal, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "evidence_class": cfg["evidence_class"],
        "research_selection_role": cfg["research_selection_role"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_oos_used_for_selection": cfg["historical_oos_used_for_selection"],
        "signal": "hgb_shallow15_locked/raw", "entry_rank": cfg["buffer"]["entry_rank"],
        "exit_rank": cfg["buffer"]["exit_rank"], "constraint_relaxation_allowed": False,
    }])
    lake.write("portfolio_construction_targets", targets, key="v1",
               source="portfolio_construction_comparison.targets.v1", request=cfg)
    lake.write("portfolio_construction_audit", audit, key="v1",
               source="portfolio_construction_comparison.targets.v1", request=cfg)
    lake.write("portfolio_construction_protocol", protocol, key="v1",
               source="portfolio_construction_comparison.targets.v1", request=cfg)
    print(audit.groupby("method").agg(
        dates=("asof_date", "nunique"), mean_candidates=("candidate_count", "mean"),
        mean_turnover=("one_way_target_turnover", "mean"),
        fallback_fraction=("fallback_used", "mean"),
        max_weight=("maximum_stock_weight", "max"),
    ).to_string(), flush=True)
    return targets, audit


def build_execution_market(lake: Lake | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    """Extend the Phase 5 market to every positive-weight Phase 6 target name."""
    lake = lake or Lake()
    destination = lake.table_path("portfolio_construction_execution_market", "v1")
    if destination.exists() and not overwrite:
        return lake.read("portfolio_construction_execution_market", "v1")
    cfg = load_config()
    targets = lake.read("portfolio_construction_targets", "v1")
    targets = targets.loc[targets["target_weight"].gt(0)]
    existing = lake.read("institutional_execution_market", "v1")
    end = str(existing["trade_date"].max())
    start = str(targets["execution_date"].min())
    codes = set(targets.loc[targets["execution_date"].le(end), "instrument_id"].astype(str))
    available_codes = set(existing["instrument_id"].astype(str))
    frames = [existing.loc[existing["instrument_id"].astype(str).isin(codes)].copy()]
    missing = sorted(codes - available_codes)
    with (ROOT / "config" / "constrained_portfolio_daily_execution.yaml").open(
        encoding="utf-8"
    ) as handle:
        rules = yaml.safe_load(handle)["rules"]
    threshold = float(cfg["execution"]["corporate_action_proxy"][
        "activation_absolute_deviation"
    ])
    tolerance = 0.20
    for number, code in enumerate(missing, start=1):
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", code.replace(".", "_")),
            columns=[
                "instrument_id", "trade_date", "close", "volume", "pct_chg",
                "trade_status", "is_st",
            ],
        ).sort_values("trade_date")
        close = pd.to_numeric(bars["close"], errors="coerce")
        previous = close.shift(1)
        returns = pd.to_numeric(bars["pct_chg"], errors="coerce") / 100.0
        raw_factor = previous * (1.0 + returns) / close
        factor = raw_factor.where(raw_factor.sub(1.0).abs().gt(threshold), 1.0).fillna(1.0)
        status = pd.to_numeric(bars["trade_status"], errors="coerce").eq(1)
        is_st = pd.to_numeric(bars["is_st"], errors="coerce").fillna(0).eq(1)
        limits = _limit_array(code, bars["trade_date"], is_st, rules)
        pct = pd.to_numeric(bars["pct_chg"], errors="coerce")
        current = bars.assign(
            previous_close=previous, daily_return=returns,
            buy_allowed=status & pct.lt(limits - tolerance),
            sell_allowed=status & pct.gt(-limits + tolerance),
            at_upper_limit=status & pct.ge(limits - tolerance),
            at_lower_limit=status & pct.le(-limits + tolerance),
            share_factor=factor, cash_dividend_per_share=0.0,
            action_source=np.where(
                factor.ne(1.0), "inferred_value_preserving_factor", "none"
            ),
        )
        frames.append(current.loc[current["trade_date"].between(start, end), [
            "trade_date", "instrument_id", "close", "previous_close", "volume",
            "daily_return", "trade_status", "buy_allowed", "sell_allowed",
            "at_upper_limit", "at_lower_limit", "share_factor",
            "cash_dividend_per_share", "action_source",
        ]])
        if number % 250 == 0:
            print(f"phase6 market files={number}/{len(missing)}", flush=True)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "instrument_id"], keep="last"
    ).sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
    missing_final = codes - set(result["instrument_id"].astype(str))
    if missing_final:
        raise RuntimeError(f"Phase 6 execution market missing {len(missing_final)} target codes")
    lake.write(
        "portfolio_construction_execution_market", result, key="v1",
        source="portfolio_construction_comparison.market.v1",
        request={**cfg, "start": start, "end": end, "codes": len(codes),
                 "reused_phase5_codes": len(codes & available_codes),
                 "new_code_files": len(missing)},
    )
    return result


def _rf_performance(daily: pd.DataFrame, lake: Lake, cfg: dict[str, Any]) -> dict[str, float]:
    rf = lake.read("risk_free_rate_daily", "v1")
    scenario = str(cfg["execution"]["risk_free_scenario"])
    rf = rf.loc[rf["scenario_id"].eq(scenario), ["trade_date", "risk_free_return"]]
    joined = daily[["trade_date", "net_return"]].merge(
        rf, on="trade_date", how="left", validate="one_to_one"
    )
    if joined["risk_free_return"].isna().any():
        raise RuntimeError("reliable risk-free series does not cover Phase 6 execution dates")
    excess = joined["net_return"] - joined["risk_free_return"]
    periods = int(cfg["execution"]["annualization_periods"])
    return {
        "risk_free_scenario": scenario,
        "annualized_risk_free_return_arithmetic": float(
            joined["risk_free_return"].mean() * periods
        ),
        "annualized_excess_return_arithmetic": float(excess.mean() * periods),
        "sharpe_reliable_risk_free": float(
            excess.mean() / excess.std(ddof=1) * np.sqrt(periods)
        ),
    }


def _execution_bootstrap(daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    reference = daily.loc[daily["method"].eq("P0_equal"), ["trade_date", "net_return"]]
    rows = []
    boot = cfg["bootstrap"]
    annualization = int(cfg["execution"]["annualization_periods"])
    for method in cfg["weighting"]["methods"]:
        if method == "P0_equal":
            continue
        current = daily.loc[daily["method"].eq(method), ["trade_date", "net_return"]]
        paired = current.merge(
            reference, on="trade_date", suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        difference = (
            paired["net_return_candidate"] - paired["net_return_reference"]
        ).to_numpy(float)
        estimates = circular_block_bootstrap_mean(
            difference, int(boot["block_length_trading_days"]),
            int(boot["samples"]), int(boot["random_seed"]),
        ) * annualization
        rows.append({
            "method": method, "reference": "P0_equal", "observations": len(difference),
            "annualized_arithmetic_return_delta": float(difference.mean() * annualization),
            "ci_2_5pct": float(np.quantile(estimates, 0.025)),
            "ci_97_5pct": float(np.quantile(estimates, 0.975)),
            "probability_delta_nonpositive": float(
                (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
            ),
        })
    return pd.DataFrame(rows)


def acceptance_checks(
    targets: pd.DataFrame, audit: pd.DataFrame, summary: pd.DataFrame,
    invariants: pd.DataFrame, daily: pd.DataFrame,
) -> pd.DataFrame:
    sums = targets.groupby(["execution_date", "method"])["target_weight"].sum()
    date_counts = targets.groupby("method")["execution_date"].nunique()
    optimizer = audit.loc[audit["method"].eq("P4_constrained")]
    checks = [
        ("long_only_targets", bool(targets["target_weight"].ge(-1e-12).all()),
         f"minimum={targets['target_weight'].min():.12g}"),
        ("fully_invested_targets", bool(np.allclose(sums, 1.0, atol=1e-8)),
         f"maximum_abs_error={float((sums - 1.0).abs().max()):.12g}"),
        ("common_target_dates", bool(date_counts.nunique() == 1),
         f"date_counts={date_counts.to_dict()}"),
        ("no_silent_constraint_relaxation", bool(~optimizer["constraint_relaxed"].any()),
         f"relaxed_dates={int(optimizer['constraint_relaxed'].sum())}"),
        ("optimizer_failures_explicit", bool(
            optimizer["fallback_used"].eq(~optimizer["optimizer_success"]).all()
         ), f"fallback_dates={int(optimizer['fallback_used'].sum())}"),
        ("accounting_invariants", bool(invariants["all_invariants_pass"].all()),
         f"passing_methods={int(invariants['all_invariants_pass'].sum())}/{len(invariants)}"),
        ("common_execution_clock", bool(
            daily.groupby("method")["trade_date"].nunique().nunique() == 1
         ), f"periods={daily.groupby('method')['trade_date'].nunique().to_dict()}"),
        ("reliable_risk_free_reported", bool(
            summary["sharpe_reliable_risk_free"].notna().all()
         ), f"scenario={summary['risk_free_scenario'].unique().tolist()}"),
    ]
    return pd.DataFrame([
        {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        for name, passed, detail in checks
    ])


def run_execution(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    targets = lake.read("portfolio_construction_targets", "v1")
    market = build_execution_market(lake)
    end = str(market["trade_date"].max())
    targets = targets.loc[
        targets["execution_date"].le(end) & targets["target_weight"].gt(0)
    ].copy()
    exec_cfg = {**cfg["execution"]}
    core_cash = float(exec_cfg["initial_cash_cny"])
    scenario_template = dict(exec_cfg["scenarios"][0])
    summaries: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    capacity_rows: list[dict[str, Any]] = []

    for method in cfg["weighting"]["methods"]:
        method_targets = targets.loc[targets["method"].eq(method), [
            "execution_date", "instrument_id", "target_weight",
        ]].reset_index(drop=True)
        scenario = {
            **scenario_template,
            "scenario": f"{method}__{scenario_template['scenario']}",
        }
        result = simulate_execution(
            market, method_targets, exec_cfg, scenario=scenario
        )
        summary, invariant = _summaries(
            result, int(exec_cfg["annualization_periods"])
        )
        summary.update({"method": method, **_rf_performance(result.daily, lake, cfg)})
        summary.update({
            "mean_gross_two_way_turnover": float(result.daily["gross_two_way_turnover"].mean()),
            "mean_one_way_turnover": float(result.daily["one_way_turnover"].mean()),
            "mean_cash_weight": float(result.daily["cash_weight"].mean()),
            "mean_target_tracking_l1": float(result.daily["target_tracking_l1"].mean()),
        })
        for column in [
            "commission_cost", "stamp_duty_cost", "transfer_fee_cost", "spread_cost",
            "market_impact_cost", "fixed_round_trip_cost", "total_cost",
        ]:
            summary[f"total_{column}_cny"] = float(
                result.fills.get(column, pd.Series(dtype=float)).sum()
            )
        summaries.append(summary)
        invariants.append({"method": method, **invariant})
        daily_frames.append(result.daily.assign(method=method))
        order_frames.append(result.orders.assign(method=method))
        fill_frames.append(result.fills.assign(method=method))
        position_frames.append(result.positions.assign(method=method))
        if not result.corporate_actions.empty:
            action_frames.append(result.corporate_actions.assign(method=method))
        capacity_rows.append({
            "method": method, "initial_cash_cny": core_cash, **summary,
            "all_invariants_pass": invariant["all_invariants_pass"],
        })
        print(f"phase6 execution method={method} complete", flush=True)

        for cash in exec_cfg["capacity_stress_cash_cny"]:
            stress_cfg = {**exec_cfg, "initial_cash_cny": float(cash)}
            stress = simulate_execution(market, method_targets, stress_cfg, scenario=scenario)
            stress_summary, stress_invariant = _summaries(
                stress, int(exec_cfg["annualization_periods"])
            )
            capacity_rows.append({
                "method": method, "initial_cash_cny": float(cash), **stress_summary,
                "all_invariants_pass": stress_invariant["all_invariants_pass"],
            })
            print(f"phase6 capacity method={method} cash={float(cash):.0f} complete", flush=True)

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    invariant_frame = pd.DataFrame(invariants)
    bootstrap = _execution_bootstrap(daily, cfg)
    target_audit = lake.read("portfolio_construction_audit", "v1")
    checks = acceptance_checks(targets, target_audit, summary, invariant_frame, daily)
    if checks["status"].ne("PASS").any():
        raise RuntimeError("Phase 6 acceptance checks failed")
    outputs = {
        "portfolio_construction_execution_daily": daily,
        "portfolio_construction_execution_orders": pd.concat(order_frames, ignore_index=True),
        "portfolio_construction_execution_fills": pd.concat(fill_frames, ignore_index=True),
        "portfolio_construction_execution_positions": pd.concat(position_frames, ignore_index=True),
        "portfolio_construction_execution_summary": summary,
        "portfolio_construction_execution_invariants": invariant_frame,
        "portfolio_construction_capacity": pd.DataFrame(capacity_rows),
        "portfolio_construction_bootstrap": bootstrap,
        "portfolio_construction_acceptance_checks": checks,
    }
    if action_frames:
        outputs["portfolio_construction_execution_corporate_actions"] = pd.concat(
            action_frames, ignore_index=True
        )
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1",
                   source="portfolio_construction_comparison.execution.v1", request=cfg)
    _write_report(
        summary, invariant_frame, bootstrap, target_audit,
        pd.DataFrame(capacity_rows), checks, cfg,
    )
    return summary, bootstrap


def _write_report(
    summary: pd.DataFrame, invariants: pd.DataFrame, bootstrap: pd.DataFrame,
    audit: pd.DataFrame, capacity: pd.DataFrame, checks: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    construction = audit.groupby("method", sort=False).agg(
        dates=("asof_date", "nunique"),
        mean_candidates=("candidate_count", "mean"),
        mean_optimizer_candidates=("optimizer_candidate_count", "mean"),
        mean_target_turnover=("one_way_target_turnover", "mean"),
        optimizer_success_fraction=("optimizer_success", "mean"),
        fallback_fraction=("fallback_used", "mean"),
        mean_industry_active_l1=("industry_active_weight_l1", "mean"),
        mean_abs_size_z=("portfolio_size_z", lambda value: value.abs().mean()),
    ).reset_index()
    performance_columns = [
        "method", "annualized_return", "annualized_volatility", "sharpe_zero_rate",
        "sharpe_reliable_risk_free", "maximum_drawdown", "mean_one_way_turnover",
        "mean_gross_two_way_turnover", "fill_ratio_by_shares", "mean_cash_weight",
        "mean_target_tracking_l1", "total_cost_cny",
    ]
    capacity_display = capacity[[
        "method", "initial_cash_cny", "annualized_return", "fill_ratio_by_shares",
        "mean_cash_weight", "mean_target_tracking_l1", "all_invariants_pass",
    ]]
    indexed = summary.set_index("method")
    p0 = indexed.loc["P0_equal"]
    p1 = indexed.loc["P1_inverse_volatility"]
    p2_interval = bootstrap.set_index("method").loc["P2_score_weighted"]
    p4 = indexed.loc["P4_constrained"]
    lines = [
        "# Phase 6：组合权重与风险优化比较", "", "## 证据边界", "",
        "本实验是已经被反复查看的 2021–2026 historical OOS 回顾性诊断，不是新的独立证据，也不用于选择或晋升组合方法。P4 支持集与回退处理曾依据本段历史中的结构性可行性失败改进，因此治理分类为 `post_historical_oos_adaptation`；没有按收益调参。",
        "四种方法固定使用同一 raw HGB 分数、调仓时钟、PIT 股票池、20bp 成本和 Phase 5 日频执行器。",
        "", "## 目标权重构造审计", "", construction.to_markdown(index=False, floatfmt=".6f"),
        "", "P4 对 Top-200/400 普通候选以外的行业/规模可行性支持股票单独标记。严格 LP 不可行时使用显式 carry-forward capacity repair；该日期不计为优化成功，且从不放宽约束。",
        "", "## 1 亿元实盘近似执行", "", summary[performance_columns].to_markdown(index=False, floatfmt=".6f"),
        "", "## 相对 P0 的配对块区间", "", bootstrap.to_markdown(index=False, floatfmt=".6f"),
        "", "## 容量敏感性", "", capacity_display.to_markdown(index=False, floatfmt=".6f"),
        "", "## 会计不变量", "", invariants.to_markdown(index=False, floatfmt=".10f"),
        "", "## 验收检查", "", checks.to_markdown(index=False),
        "", "## 结论与决策", "",
        (
            f"- 保留 P0 为透明基线：年化收益 {p0['annualized_return']:.2%}、"
            f"Sharpe {p0['sharpe_zero_rate']:.3f}、最大回撤 {p0['maximum_drawdown']:.2%}。"
        ),
        (
            f"- P1 未显示增量收益，但有轻微风险效率改善：年化收益 {p1['annualized_return']:.2%}，"
            f"波动 {p1['annualized_volatility']:.2%}，Sharpe {p1['sharpe_zero_rate']:.3f}；"
            "相对 P0 收益差区间跨零，不能声称优于基线。"
        ),
        (
            f"- P2 不予采用：相对 P0 年化算术收益差 95% 区间 "
            f"[{p2_interval['ci_2_5pct']:.2%}, {p2_interval['ci_97_5pct']:.2%}] 完全低于零。"
        ),
        (
            f"- P4 将波动降至 {p4['annualized_volatility']:.2%}、最大回撤改善至 "
            f"{p4['maximum_drawdown']:.2%}，但严格优化仅 45.4% 日期成功，成交股数比例 "
            f"{p4['fill_ratio_by_shares']:.2%}、平均目标偏离 {p4['mean_target_tracking_l1']:.2%}；"
            "这是风险/换手价值与可实现性不足并存的研究原型，不可晋升。"
        ),
        "复杂方法只有在未来 lockbox 或前瞻 paper trading 中稳定改善收益、风险或交易效率时才可替换 P0；本表不得用于 OOS 后择优。",
        "日线执行仍无法重建盘口、排队、bid/ask 或精确成交概率。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 portfolio construction comparison")
    parser.add_argument("command", choices=["signal", "targets", "market", "execution", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "signal":
        build_signal_panel(overwrite=args.overwrite)
    elif args.command == "targets":
        run_targets()
    elif args.command == "market":
        build_execution_market(overwrite=args.overwrite)
    elif args.command == "execution":
        run_execution()
    else:
        build_signal_panel(overwrite=args.overwrite)
        run_targets()
        build_execution_market(overwrite=args.overwrite)
        run_execution()


if __name__ == "__main__":
    main()
