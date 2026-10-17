"""Prespecified industry/size/turnover-constrained portfolio experiment."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from quant_research.factor_neutralization import _read_year
from quant_research.factor_preprocessing_ablation import _rank_ic
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import SufficientStatistics, purged_end_date
from quant_research.model_challenger import TrainingSample
from quant_research.model_portfolio_neutralization import (
    _aggregate_training,
    build_year_training,
    fit_models,
    model_designs,
)
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "constrained_portfolio_neutralization.yaml"
REPORT_PATH = ROOT / "reports" / "constrained_portfolio_neutralization.md"
FULL_PIPELINES = [
    "raw", "size_neutralized", "industry_neutralized",
    "industry_size_neutralized",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _drifted_weights(state: dict) -> dict[str, float]:
    previous = state.get("weights", {})
    if not previous:
        return {}
    returns = state.get("returns", {})
    values = {
        code: weight * (1.0 + returns.get(code, 0.0))
        for code, weight in previous.items()
    }
    total = sum(values.values())
    return ({code: value / total for code, value in values.items()}
            if total > 0 else dict(previous))


def _score_rank(score: np.ndarray) -> np.ndarray:
    return (
        pd.Series(score).rank(method="average", pct=True).to_numpy(dtype="float64")
        - 0.5
    )


def _industry_limits(group: pd.DataFrame, active_bound: float) -> dict[str, tuple[float, float, float]]:
    benchmark = group["industry_broad_code"].astype(str).value_counts(normalize=True)
    return {
        industry: (
            max(0.0, float(weight) - active_bound),
            min(1.0, float(weight) + active_bound),
            float(weight),
        )
        for industry, weight in benchmark.items()
    }


def candidate_indices(
    group: pd.DataFrame,
    score: np.ndarray,
    previous: dict[str, float],
    industry_limits: dict[str, tuple[float, float, float]],
    cfg: dict,
) -> np.ndarray:
    """Outcome-independent LP column screen with explicit capacity coverage."""
    optimizer = cfg["optimizer"]
    maximum_weight = float(optimizer["maximum_stock_weight"])
    ranked = group.assign(_score=score, _row=np.arange(len(group))).sort_values(
        ["_score", "instrument_id"], ascending=[False, True]
    )
    chosen = set(ranked.head(int(optimizer["candidate_top_n"])) ["_row"].astype(int))
    buffer_names = int(optimizer["industry_capacity_buffer_names"])
    for industry, (_, upper, _) in industry_limits.items():
        required = int(np.ceil(upper / maximum_weight)) + buffer_names
        industry_rows = ranked.loc[
            ranked["industry_broad_code"].astype(str).eq(industry)
        ].head(required)
        chosen.update(industry_rows["_row"].astype(int))
    tail = int(optimizer["size_tail_candidates_each_side"])
    if tail > 0:
        size_order = np.argsort(group["size_z"].to_numpy(dtype="float64"))
        chosen.update(size_order[:tail].tolist())
        chosen.update(size_order[-tail:].tolist())
    if previous:
        code_to_row = dict(zip(group["instrument_id"].astype(str), range(len(group))))
        chosen.update(code_to_row[code] for code in previous if code in code_to_row)
    return np.array(sorted(chosen), dtype=int)


def _linear_program(
    candidate: pd.DataFrame,
    score: np.ndarray,
    previous: dict[str, float],
    industry_limits: dict[str, tuple[float, float, float]],
    turnover_limit: float,
    cfg: dict,
) -> tuple[np.ndarray | None, str, float]:
    optimizer = cfg["optimizer"]
    n = len(candidate)
    variable_count = 2 * n
    objective = np.concatenate([-_score_rank(score), np.full(n, 1e-9)])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    bounds_rhs: list[float] = []

    def add_row(entries: list[tuple[int, float]], rhs: float) -> None:
        row = len(bounds_rhs)
        for column, value in entries:
            rows.append(row); columns.append(column); values.append(value)
        bounds_rhs.append(float(rhs))

    industries = candidate["industry_broad_code"].astype(str).to_numpy()
    for industry, (lower, upper, _) in industry_limits.items():
        members = np.flatnonzero(industries == industry)
        add_row([(int(item), 1.0) for item in members], upper)
        add_row([(int(item), -1.0) for item in members], -lower)

    size = candidate["size_z"].to_numpy(dtype="float64")
    size_bound = float(optimizer["absolute_size_z_bound"])
    add_row([(index, float(value)) for index, value in enumerate(size)], size_bound)
    add_row([(index, -float(value)) for index, value in enumerate(size)], size_bound)

    previous_candidate = np.array([
        previous.get(code, 0.0) for code in candidate["instrument_id"].astype(str)
    ], dtype="float64")
    candidate_codes = set(candidate["instrument_id"].astype(str))
    exited_weight = sum(weight for code, weight in previous.items() if code not in candidate_codes)
    if previous:
        for index, old_weight in enumerate(previous_candidate):
            add_row([(index, 1.0), (n + index, -1.0)], old_weight)
            add_row([(index, -1.0), (n + index, -1.0)], -old_weight)
        remaining_l1 = 2.0 * turnover_limit - exited_weight
        if remaining_l1 < -1e-12:
            return None, "turnover_limit_below_forced_exit", exited_weight
        add_row([(n + index, 1.0) for index in range(n)], max(remaining_l1, 0.0))

    inequality = coo_matrix(
        (values, (rows, columns)), shape=(len(bounds_rhs), variable_count)
    ).tocsr()
    equality = csr_matrix((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, variable_count))
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(bounds_rhs),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=(
            [(0.0, float(optimizer["maximum_stock_weight"]))] * n
            + [(0.0, None)] * n
        ),
        method="highs",
        options={"presolve": True},
    )
    return (
        (result.x[:n] if result.success else None),
        str(result.message),
        exited_weight,
    )


def solve_risk_constrained_weights(
    group: pd.DataFrame,
    score: np.ndarray,
    previous: dict[str, float],
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Solve the prespecified long-only risk and turnover constrained LP."""
    started = perf_counter()
    optimizer = cfg["optimizer"]
    limits = _industry_limits(
        group, float(optimizer["industry_active_weight_bound"])
    )
    chosen = candidate_indices(group, score, previous, limits, cfg)
    limit_sequence = ([1.0] if not previous else [
        float(value) for value in optimizer["turnover_feasibility_relaxation"]
    ])
    messages = []
    solution = None
    effective_limit = limit_sequence[-1]
    forced_exit = 0.0
    used_full_universe = False
    for use_full in (False, True):
        indices = np.arange(len(group), dtype=int) if use_full else chosen
        candidate = group.iloc[indices].reset_index(drop=True)
        candidate_score = score[indices]
        for limit in limit_sequence:
            weights, message, forced_exit = _linear_program(
                candidate, candidate_score, previous, limits, limit, cfg
            )
            messages.append(f"full={use_full},turnover={limit:.2f}:{message}")
            if weights is not None:
                solution = (indices, weights)
                effective_limit = limit
                used_full_universe = use_full
                break
        if solution is not None:
            break
    if solution is None:
        raise RuntimeError("risk-constrained LP infeasible: " + " | ".join(messages))
    indices, candidate_weights = solution
    result = np.zeros(len(group), dtype="float64")
    result[indices] = candidate_weights
    result[np.abs(result) < float(optimizer["minimum_reported_weight"])] = 0.0
    if not np.isclose(result.sum(), 1.0, atol=1e-8):
        raise RuntimeError(f"optimized weights do not sum to one: {result.sum()}")
    return result, {
        "optimizer_success": True,
        "optimizer_seconds": perf_counter() - started,
        "candidate_pool_size": int(len(indices)),
        "used_full_universe": used_full_universe,
        "requested_turnover_limit": float(optimizer["weekly_turnover_limit"]),
        "effective_turnover_limit": effective_limit,
        "turnover_limit_relaxed": bool(
            previous and effective_limit > float(optimizer["weekly_turnover_limit"])
        ),
        "forced_exit_weight": forced_exit,
        "solver_message": messages[-1],
    }


def top_equal_weights(group: pd.DataFrame, score: np.ndarray, top_n: int) -> np.ndarray:
    ranked = group.assign(_score=score, _row=np.arange(len(group))).sort_values(
        ["_score", "instrument_id"], ascending=[False, True]
    )
    selected = ranked.head(top_n)["_row"].to_numpy(dtype=int)
    weights = np.zeros(len(group), dtype="float64")
    weights[selected] = 1.0 / len(selected)
    return weights


def portfolio_risk(
    group: pd.DataFrame, weights: np.ndarray, cfg: dict,
) -> tuple[dict, pd.DataFrame]:
    industry = group["industry_broad_code"].astype(str)
    benchmark = industry.value_counts(normalize=True)
    portfolio = pd.Series(weights).groupby(industry.reset_index(drop=True)).sum()
    labels = sorted(set(benchmark.index) | set(portfolio.index))
    allocation = pd.DataFrame({
        "industry_broad_code": labels,
        "portfolio_weight": [float(portfolio.get(label, 0.0)) for label in labels],
        "universe_weight": [float(benchmark.get(label, 0.0)) for label in labels],
    })
    allocation["active_weight"] = (
        allocation["portfolio_weight"] - allocation["universe_weight"]
    )
    active = allocation["active_weight"].to_numpy(dtype="float64")
    size_exposure = float(
        weights @ group["size_z"].to_numpy(dtype="float64")
    )
    optimizer = cfg["optimizer"]
    maximum_active = float(np.abs(active).max())
    maximum_weight = float(weights.max())
    return {
        "holdings": int((weights > float(optimizer["minimum_reported_weight"])).sum()),
        "maximum_stock_weight": maximum_weight,
        "portfolio_size_z": size_exposure,
        "abs_portfolio_size_z": abs(size_exposure),
        "portfolio_industry_hhi": float(
            np.square(allocation["portfolio_weight"]).sum()
        ),
        "industry_active_weight_l1": float(np.abs(active).sum() / 2.0),
        "maximum_abs_industry_active_weight": maximum_active,
        "stock_weight_violation": max(
            0.0, maximum_weight - float(optimizer["maximum_stock_weight"])
        ),
        "size_bound_violation": max(
            0.0, abs(size_exposure) - float(optimizer["absolute_size_z_bound"])
        ),
        "industry_bound_violation": max(
            0.0, maximum_active - float(optimizer["industry_active_weight_bound"])
        ),
    }, allocation


def portfolio_step(
    group: pd.DataFrame,
    weights: np.ndarray,
    target: str,
    state: dict,
    cfg: dict,
) -> tuple[dict, pd.DataFrame]:
    codes = group["instrument_id"].astype(str).to_numpy()
    current = {
        code: float(weight) for code, weight in zip(codes, weights) if weight > 0
    }
    drifted = _drifted_weights(state)
    turnover = (
        0.5 * sum(abs(current.get(code, 0.0) - drifted.get(code, 0.0))
                  for code in set(current) | set(drifted))
        if drifted else 1.0
    )
    realized = pd.to_numeric(group[target], errors="coerce").to_numpy(dtype="float64")
    filled = np.nan_to_num(realized, nan=0.0)
    realized_fraction = float(weights[np.isfinite(realized)].sum())
    gross_return = float(weights @ filled)
    benchmark_return = float(np.mean(filled))
    transaction_cost = turnover * float(cfg["portfolio"]["round_trip_cost_bps"]) / 10_000
    state["weights"] = current
    state["returns"] = {
        code: float(value) for code, value, weight in zip(codes, filled, weights)
        if weight > 0
    }
    holdings = group.loc[weights > 0, [
        "instrument_id", "industry_broad_code", "float_cap_proxy_cny", "size_z",
    ]].copy()
    holdings["target_weight"] = weights[weights > 0]
    holdings["score"] = group.loc[weights > 0, "score"].to_numpy()
    holdings["realized_return"] = realized[weights > 0]
    return {
        "asof_date": str(group["asof_date"].iloc[0]),
        "turnover": turnover,
        "gross_return": gross_return,
        "transaction_cost": transaction_cost,
        "net_return": gross_return - transaction_cost,
        "benchmark_return": benchmark_return,
        "gross_excess_return": gross_return - benchmark_return,
        "net_excess_return": gross_return - transaction_cost - benchmark_return,
        "realized_weight_fraction": realized_fraction,
        "included_in_performance": (
            realized_fraction
            >= float(cfg["portfolio"]["minimum_realized_weight_fraction"])
        ),
    }, holdings


def evaluate_year(
    frame: pd.DataFrame,
    models: dict,
    factors: list[str],
    target: str,
    states: dict,
    cfg: dict,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly_records = []
    holding_frames = []
    allocation_frames = []
    for asof_date, original in frame.groupby("asof_date", sort=True):
        original = original.reset_index(drop=True)
        designs, risks, complete = model_designs(original, factors, cfg)
        group = original.loc[complete].reset_index(drop=True)
        group["size_z"] = risks["size_z"][complete]
        if len(group) < int(cfg["minimum_cross_section_size"]):
            continue
        target_values = pd.to_numeric(group[target], errors="coerce")
        usable = target_values.notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        for pipeline in cfg["alpha_pipelines"]:
            design = designs[pipeline][complete]
            for model_name, model in models[pipeline].items():
                score = model.predict(design)
                rank_ic = _rank_ic(score[usable], target_values.loc[usable])
                scored = group.assign(score=score)
                for construction in cfg["portfolio"]["constructions"]:
                    strategy = f"{pipeline}__{construction}"
                    state = states[(pipeline, model_name, construction)]
                    drifted = _drifted_weights(state)
                    if construction == "top200_equal":
                        weights = top_equal_weights(
                            scored, score, int(cfg["portfolio"]["baseline_top_n"])
                        )
                        optimizer_diagnostics = {
                            "optimizer_success": True,
                            "optimizer_seconds": 0.0,
                            "candidate_pool_size": len(scored),
                            "used_full_universe": False,
                            "requested_turnover_limit": np.nan,
                            "effective_turnover_limit": np.nan,
                            "turnover_limit_relaxed": False,
                            "forced_exit_weight": 0.0,
                            "solver_message": "not_applicable",
                        }
                    elif construction == "risk_constrained":
                        weights, optimizer_diagnostics = solve_risk_constrained_weights(
                            scored, score, drifted, cfg
                        )
                    else:
                        raise ValueError(f"unknown construction: {construction}")
                    risk_record, allocation = portfolio_risk(scored, weights, cfg)
                    portfolio, holdings = portfolio_step(
                        scored, weights, target, state, cfg
                    )
                    base = {
                        "test_year": test_year,
                        "alpha_pipeline": pipeline,
                        "model": model_name,
                        "construction": construction,
                        "strategy": strategy,
                    }
                    weekly_records.append({
                        **portfolio, **base, "rank_ic": rank_ic,
                        **risk_record, **optimizer_diagnostics,
                    })
                    holding_frames.append(holdings.assign(**base, asof_date=str(asof_date)))
                    allocation_frames.append(allocation.assign(
                        **base, asof_date=str(asof_date)
                    ))
    return (
        pd.DataFrame(weekly_records),
        pd.concat(holding_frames, ignore_index=True),
        pd.concat(allocation_frames, ignore_index=True),
    )


def summarize(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for (model, strategy), group in weekly.groupby(["model", "strategy"], sort=False):
        rows.append({
            "model": model,
            "strategy": strategy,
            "alpha_pipeline": group["alpha_pipeline"].iloc[0],
            "construction": group["construction"].iloc[0],
            "mean_rank_ic": float(group["rank_ic"].mean()),
            **performance_summary(group, int(cfg["annualization_periods"])),
            "mean_abs_portfolio_size_z": float(group["abs_portfolio_size_z"].mean()),
            "mean_industry_active_weight_l1": float(
                group["industry_active_weight_l1"].mean()
            ),
            "mean_maximum_abs_industry_active_weight": float(
                group["maximum_abs_industry_active_weight"].mean()
            ),
            "maximum_stock_weight_violation": float(
                group["stock_weight_violation"].max()
            ),
            "maximum_size_bound_violation": float(group["size_bound_violation"].max()),
            "maximum_industry_bound_violation": float(
                group["industry_bound_violation"].max()
            ),
            "turnover_relaxation_fraction": float(
                group["turnover_limit_relaxed"].astype(bool).mean()
            ),
            "mean_optimizer_seconds": float(group["optimizer_seconds"].mean()),
        })
    return pd.DataFrame(rows)


def annual_summary(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for (model, strategy, year), group in weekly.groupby(
        ["model", "strategy", "test_year"], sort=False
    ):
        rows.append({
            "model": model, "strategy": strategy, "year": int(year),
            "mean_rank_ic": float(group["rank_ic"].mean()),
            **performance_summary(group, int(cfg["annualization_periods"])),
            "mean_abs_portfolio_size_z": float(group["abs_portfolio_size_z"].mean()),
            "mean_industry_active_weight_l1": float(
                group["industry_active_weight_l1"].mean()
            ),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    comparisons = {
        "factor_neutralization_at_top200": (
            "industry_size_neutralized__top200_equal", "raw__top200_equal"
        ),
        "constraints_on_raw_alpha": (
            "raw__risk_constrained", "raw__top200_equal"
        ),
        "constraints_on_neutral_alpha": (
            "industry_size_neutralized__risk_constrained",
            "industry_size_neutralized__top200_equal",
        ),
        "full_neutral_stack": (
            "industry_size_neutralized__risk_constrained", "raw__top200_equal"
        ),
    }
    metrics = (
        ("rank_ic", "rank_ic", 1.0, "higher_is_better"),
        ("net_return_annualized_arithmetic", "net_return",
         float(cfg["annualization_periods"]), "higher_is_better"),
        ("industry_active_weight_l1", "industry_active_weight_l1", 1.0,
         "lower_is_better"),
        ("abs_portfolio_size_z", "abs_portfolio_size_z", 1.0, "lower_is_better"),
        ("turnover", "turnover", 1.0, "lower_is_better"),
    )
    bootstrap_cfg = cfg["bootstrap"]
    records = []
    for model in cfg["models"]:
        for comparison, (candidate, reference) in comparisons.items():
            for metric, column, scale, direction in metrics:
                left = weekly.loc[
                    weekly["model"].eq(model) & weekly["strategy"].eq(candidate),
                    ["asof_date", column],
                ]
                right = weekly.loc[
                    weekly["model"].eq(model) & weekly["strategy"].eq(reference),
                    ["asof_date", column],
                ]
                paired = left.merge(
                    right, on="asof_date", suffixes=("_candidate", "_reference"),
                    validate="one_to_one",
                )
                difference = pd.to_numeric(
                    paired[f"{column}_candidate"] - paired[f"{column}_reference"],
                    errors="coerce",
                ).dropna().to_numpy(dtype="float64")
                estimates = circular_block_bootstrap_mean(
                    difference,
                    int(bootstrap_cfg["block_length_weeks"]),
                    int(bootstrap_cfg["samples"]),
                    int(bootstrap_cfg["random_seed"]),
                ) * scale
                records.append({
                    "model": model, "comparison": comparison,
                    "candidate": candidate, "reference": reference,
                    "metric": metric, "preferred_direction": direction,
                    "observations": len(difference),
                    "point_estimate": float(difference.mean() * scale),
                    "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                    "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                    "probability_delta_nonpositive": float(
                        (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                    ),
                })
    return pd.DataFrame(records)


def write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame, cfg: dict) -> None:
    display = summary[[
        "model", "strategy", "mean_rank_ic", "annualized_return",
        "sharpe_zero_rate", "maximum_drawdown", "mean_turnover",
        "mean_abs_portfolio_size_z", "mean_industry_active_weight_l1",
        "mean_maximum_abs_industry_active_weight", "turnover_relaxation_fraction",
    ]]
    full_stack = bootstrap.loc[
        bootstrap["comparison"].eq("full_neutral_stack")
        & bootstrap["metric"].isin([
            "net_return_annualized_arithmetic", "industry_active_weight_l1",
            "abs_portfolio_size_z", "turnover",
        ])
    ][["model", "metric", "point_estimate", "ci_2_5pct", "ci_97_5pct"]]
    lines = [
        "# 组合层行业/规模/换手约束实验",
        "",
        "## 设计",
        "",
        "- 固定 raw 与 industry+size neutralized 两套 alpha、Ridge/HGB 参数和年度扩展窗口。",
        "- 线性规划目标为最大化截面 score rank；单股权重不超过 0.5%。",
        "- 相对 PIT 等权股票池的行业主动权重逐行业不超过 ±3%，组合 |size-z| 不超过 0.10。",
        "- 周度换手目标上限 30%；仅为保持可行性按预设阶梯放宽，并完整披露。",
        "- 收益仍是理想完整成交、20bp round-trip 的研究口径，不替代日频涨跌停执行审计。",
        "",
        "## 2021–2026 历史样本外结果",
        "",
        display.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 完整中性化栈相对 raw Top-200 的配对区间",
        "",
        full_stack.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    model_cfg = {**cfg, "pipelines": FULL_PIPELINES}
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start_year = int(cfg["train"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])
    yearly_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_purged_stats: dict[int, dict[str, SufficientStatistics]] = {}
    yearly_samples: dict[int, dict[str, TrainingSample]] = {}
    yearly_purged_samples: dict[int, dict[str, TrainingSample]] = {}
    for year in range(start_year, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        (
            yearly_stats[year], yearly_purged_stats[year],
            yearly_samples[year], yearly_purged_samples[year],
        ) = build_year_training(frame, factors, target, cutoff, model_cfg)
        print(f"constrained statistics year={year}", flush=True)

    states = {
        (pipeline, model, construction): {}
        for pipeline in cfg["alpha_pipelines"]
        for model in cfg["models"]
        for construction in cfg["portfolio"]["constructions"]
    }
    weekly_frames = []
    holding_frames = []
    allocation_frames = []
    fit_records = []
    for test_year in range(int(cfg["historical_oos"]["start_year"]), oos_end + 1):
        models = {}
        for pipeline in cfg["alpha_pipelines"]:
            statistics, sample = _aggregate_training(
                yearly_stats, yearly_purged_stats, yearly_samples,
                yearly_purged_samples, pipeline, start_year, test_year - 1,
            )
            models[pipeline] = fit_models(statistics, sample, cfg)
            fit_records.extend({
                "test_year": test_year, "alpha_pipeline": pipeline,
                "model": model.model, "kind": model.kind,
                "fit_seconds": model.fit_seconds,
                "fit_observations": model.fit_observations,
            } for model in models[pipeline].values())
        frame = _read_year(lake, test_year, columns)
        weekly, holdings, allocation = evaluate_year(
            frame, models, factors, target, states, model_cfg, test_year
        )
        weekly_frames.append(weekly)
        holding_frames.append(holdings)
        allocation_frames.append(allocation)
        constrained = weekly.loc[weekly["construction"].eq("risk_constrained")]
        print(
            f"constrained oos year={test_year} dates={weekly['asof_date'].nunique()} "
            f"mean_lp_seconds={constrained['optimizer_seconds'].mean():.3f}",
            flush=True,
        )

    weekly = pd.concat(weekly_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    summary = summarize(weekly, cfg)
    annual = annual_summary(weekly, cfg)
    bootstrap = paired_bootstrap(weekly, cfg)
    write_report(summary, bootstrap, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "historical_oos_used_for_selection": False,
        "constraint_policy_prespecified": True,
        "source_experiment": cfg["source_experiment"],
        "execution_assumption": "ideal_complete_fill_with_20bp_cost",
    }])
    request = {**cfg, "factor_columns": factors}
    outputs = {
        "constrained_portfolio_protocol": protocol,
        "constrained_portfolio_oos_weekly": weekly,
        "constrained_portfolio_oos_holdings": holdings,
        "constrained_portfolio_oos_industry_allocation": allocations,
        "constrained_portfolio_oos_summary": summary,
        "constrained_portfolio_oos_annual": annual,
        "constrained_portfolio_oos_bootstrap": bootstrap,
        "constrained_portfolio_fit_audit": pd.DataFrame(fit_records),
    }
    for table, frame in outputs.items():
        lake.write(
            table, frame, key="v1",
            source="constrained_portfolio_neutralization.v1", request=request,
        )
    print(summary.to_string(index=False))
    print(bootstrap.loc[
        bootstrap["comparison"].eq("full_neutral_stack")
    ].to_string(index=False))


if __name__ == "__main__":
    run()
