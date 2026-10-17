"""Lagged multi-factor-risk-aware portfolio optimization and target-vol overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from quant_research.ingest import ROOT, Lake
from quant_research.multi_factor_risk_attribution import (
    _read_year,
    fit_factor_cross_section,
)
from quant_research.portfolio import performance_summary
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "risk_aware_portfolio_optimization.yaml"
RISK_CONFIG_PATH = ROOT / "config" / "multi_factor_risk_attribution.yaml"
REPORT_PATH = ROOT / "reports" / "risk_aware_portfolio_optimization.md"
INDUSTRIES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "LEGACY_SOCIAL_SERVICES", "M", "N", "O", "P", "Q", "R", "S",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_risk_config(path: Path = RISK_CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _score_utility(score: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype="float64")


def _industry_limits(
    benchmark: dict[str, float], bound: float,
) -> dict[str, tuple[float, float]]:
    return {
        industry: (max(0.0, weight - bound), min(1.0, weight + bound))
        for industry, weight in benchmark.items()
    }


def _linear_oracle(
    candidate: pd.DataFrame,
    linear_objective: np.ndarray,
    previous: dict[str, float],
    industry_limits: dict[str, tuple[float, float]],
    turnover_limit: float,
    cfg: dict,
    *,
    alpha_utility: np.ndarray | None = None,
    alpha_floor: float | None = None,
) -> tuple[np.ndarray | None, str, float]:
    """Solve a linear objective over the common portfolio constraint set."""
    optimizer = cfg["optimizer"]
    n = len(candidate)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: list[tuple[int, float]], bound: float) -> None:
        row = len(rhs)
        for column, value in entries:
            rows.append(row)
            columns.append(column)
            values.append(value)
        rhs.append(float(bound))

    industries = candidate["industry_broad_code"].astype(str).to_numpy()
    for industry, (lower, upper) in industry_limits.items():
        members = np.flatnonzero(industries == industry)
        add([(int(index), 1.0) for index in members], upper)
        add([(int(index), -1.0) for index in members], -lower)

    size = candidate["size_z"].to_numpy(dtype="float64")
    size_bound = float(optimizer["absolute_size_z_bound"])
    add([(index, float(value)) for index, value in enumerate(size)], size_bound)
    add([(index, -float(value)) for index, value in enumerate(size)], size_bound)

    previous_candidate = np.array([
        previous.get(code, 0.0) for code in candidate["instrument_id"].astype(str)
    ])
    candidate_codes = set(candidate["instrument_id"].astype(str))
    forced_exit = float(sum(
        weight for code, weight in previous.items() if code not in candidate_codes
    ))
    if previous:
        for index, old_weight in enumerate(previous_candidate):
            add([(index, 1.0), (n + index, -1.0)], old_weight)
            add([(index, -1.0), (n + index, -1.0)], -old_weight)
        remaining_l1 = 2.0 * turnover_limit - forced_exit
        if remaining_l1 < -1e-12:
            return None, "turnover_limit_below_forced_exit", forced_exit
        add([(n + index, 1.0) for index in range(n)], max(remaining_l1, 0.0))

    if alpha_utility is not None and alpha_floor is not None:
        add(
            [(index, -float(value)) for index, value in enumerate(alpha_utility)],
            -float(alpha_floor) + 1e-11,
        )

    inequality = coo_matrix(
        (values, (rows, columns)), shape=(len(rhs), 2 * n)
    ).tocsr()
    equality = csr_matrix(
        (np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, 2 * n)
    )
    weight_objective = np.asarray(linear_objective, dtype="float64")
    objective_scale = max(float(np.abs(weight_objective).max()), 1e-12)
    objective = np.concatenate([
        weight_objective / objective_scale, np.full(n, 1e-10)
    ])
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(rhs),
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
        result.x[:n] if result.success else None,
        str(result.message),
        forced_exit,
    )


def active_variance(
    weights: np.ndarray,
    factor_loadings: np.ndarray,
    benchmark_exposure: np.ndarray,
    covariance: np.ndarray,
    specific_variance: np.ndarray,
    universe_weight: float,
) -> float:
    active_factor = factor_loadings.T @ weights - benchmark_exposure
    factor = float(active_factor @ covariance @ active_factor)
    specific = float(np.sum(
        np.square(weights - universe_weight) * specific_variance
    ))
    return factor + specific


def solve_risk_aware_weights(
    candidate: pd.DataFrame,
    previous: dict[str, float],
    benchmark_industry: dict[str, float],
    factor_loadings: np.ndarray,
    benchmark_exposure: np.ndarray,
    covariance: np.ndarray,
    specific_variance: np.ndarray,
    universe_weight: float,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Minimize convex active variance with a score-utility retention floor."""
    started = perf_counter()
    optimizer = cfg["optimizer"]
    alpha = (
        candidate["alpha_utility"].to_numpy(dtype="float64")
        if "alpha_utility" in candidate
        else _score_utility(candidate["score"])
    )
    limits = _industry_limits(
        benchmark_industry, float(optimizer["industry_active_weight_bound"])
    )
    sequence = ([1.0] if not previous else [
        float(value) for value in optimizer["turnover_feasibility_relaxation"]
    ])
    messages: list[str] = []
    initial = None
    effective_limit = sequence[-1]
    forced_exit = 0.0
    for limit in sequence:
        initial, message, forced_exit = _linear_oracle(
            candidate, -alpha, previous, limits, limit, cfg
        )
        messages.append(f"turnover={limit:.2f}:{message}")
        if initial is not None:
            effective_limit = limit
            break
    if initial is None:
        raise RuntimeError("alpha-feasible risk-aware LP failed: " + " | ".join(messages))

    maximum_utility = float(alpha @ initial)
    alpha_floor = float(optimizer["alpha_utility_retention_fraction"]) * maximum_utility
    weights = initial.copy()
    gap = np.inf
    iterations = 0
    for iterations in range(1, int(optimizer["maximum_iterations"]) + 1):
        active_factor = factor_loadings.T @ weights - benchmark_exposure
        gradient = (
            2.0 * factor_loadings @ covariance @ active_factor
            + 2.0 * specific_variance * (weights - universe_weight)
        )
        vertex, message, _ = _linear_oracle(
            candidate, gradient, previous, limits, effective_limit, cfg,
            alpha_utility=alpha, alpha_floor=alpha_floor,
        )
        if vertex is None:
            raise RuntimeError(f"conditional-gradient oracle failed: {message}")
        direction = vertex - weights
        gap = float(-gradient @ direction)
        if gap <= float(optimizer["duality_gap_tolerance"]):
            break
        factor_direction = factor_loadings.T @ direction
        curvature = float(
            factor_direction @ covariance @ factor_direction
            + np.sum(np.square(direction) * specific_variance)
        )
        step = 1.0 if curvature <= 0 else float(np.clip(gap / (2.0 * curvature), 0.0, 1.0))
        weights += step * direction

    weights[np.abs(weights) < float(optimizer["minimum_reported_weight"])] = 0.0
    weights /= weights.sum()
    optimized_utility = float(alpha @ weights)
    return weights, {
        "optimizer_success": True,
        "optimizer_seconds": perf_counter() - started,
        "optimizer_iterations": iterations,
        "duality_gap": gap,
        "candidate_pool_size": len(candidate),
        "maximum_alpha_utility": maximum_utility,
        "alpha_utility_floor": alpha_floor,
        "optimized_alpha_utility": optimized_utility,
        "alpha_utility_retention": optimized_utility / maximum_utility,
        "requested_turnover_limit": float(optimizer["weekly_turnover_limit"]),
        "effective_turnover_limit": effective_limit,
        "turnover_limit_relaxed": bool(
            previous and effective_limit > float(optimizer["weekly_turnover_limit"])
        ),
        "forced_exit_weight": forced_exit,
        "solver_message": messages[-1],
    }


def _factor_covariances(
    factor_returns: pd.DataFrame, factor_names: list[str], risk_cfg: dict,
) -> dict[str, np.ndarray]:
    pivot = factor_returns.pivot(
        index="asof_date", columns="factor", values="factor_return"
    ).sort_index().reindex(columns=factor_names)
    trailing = int(risk_cfg["covariance"]["trailing_weeks"])
    minimum = int(risk_cfg["covariance"]["minimum_weeks"])
    half_life = float(risk_cfg["covariance"]["ewma_half_life_weeks"])
    shrinkage = float(risk_cfg["covariance"]["diagonal_shrinkage"])
    history: list[np.ndarray] = []
    result: dict[str, np.ndarray] = {}
    for date, row in pivot.iterrows():
        if len(history) >= minimum:
            values = np.vstack(history[-trailing:])
            ages = np.arange(len(values) - 1, -1, -1, dtype="float64")
            weights = np.exp(-np.log(2.0) * ages / half_life)
            weights /= weights.sum()
            mean = np.sum(values * weights[:, None], axis=0)
            demeaned = values - mean
            covariance = (demeaned * weights[:, None]).T @ demeaned
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage * np.diag(np.diag(covariance))
            )
            result[str(date)] = covariance
        history.append(row.to_numpy(dtype="float64"))
    return result


def _drifted(state: dict) -> tuple[dict[str, float], float]:
    weights = state.get("weights", {})
    if not weights:
        return {}, 1.0
    returns = state.get("returns", {})
    cash = float(state.get("cash", 0.0))
    values = {
        code: weight * (1.0 + returns.get(code, 0.0))
        for code, weight in weights.items()
    }
    total = cash + sum(values.values())
    if total <= 0:
        return dict(weights), cash
    return ({code: value / total for code, value in values.items()}, cash / total)


def _turnover(new: dict[str, float], cash: float, state: dict) -> float:
    if not state.get("weights"):
        return float(sum(new.values()))
    old, old_cash = _drifted(state)
    return 0.5 * (
        sum(abs(new.get(code, 0.0) - old.get(code, 0.0)) for code in set(new) | set(old))
        + abs(cash - old_cash)
    )


def _risk_metrics(
    weights: np.ndarray,
    factor_loadings: np.ndarray,
    benchmark_exposure: np.ndarray,
    covariance: np.ndarray,
    selected_specific: np.ndarray,
    cross: pd.DataFrame,
    universe_specific: np.ndarray,
    annualization: int,
) -> dict[str, float]:
    portfolio_exposure = factor_loadings.T @ weights
    active_exposure = portfolio_exposure - benchmark_exposure
    factor_variance = float(portfolio_exposure @ covariance @ portfolio_exposure)
    specific_variance = float(np.sum(np.square(weights) * selected_specific))
    weight_map = dict(zip(
        cross.attrs["candidate_codes"], weights
    ))
    universe_weight = 1.0 / len(cross)
    deltas = np.array([
        weight_map.get(str(code), 0.0) - universe_weight
        for code in cross["instrument_id"]
    ])
    active_factor_variance = float(active_exposure @ covariance @ active_exposure)
    active_specific_variance = float(np.sum(np.square(deltas) * universe_specific))
    return {
        "portfolio_factor_variance": factor_variance,
        "portfolio_specific_variance": specific_variance,
        "active_factor_variance": active_factor_variance,
        "active_specific_variance": active_specific_variance,
        "forecast_annualized_volatility": np.sqrt(max(
            factor_variance + specific_variance, 0.0
        ) * annualization),
        "forecast_annualized_tracking_error": np.sqrt(max(
            active_factor_variance + active_specific_variance, 0.0
        ) * annualization),
    }


def _summary(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for keys, group in weekly.groupby(
        ["model", "alpha_pipeline", "scenario"], sort=False
    ):
        rows.append({
            "model": keys[0], "alpha_pipeline": keys[1], "scenario": keys[2],
            **performance_summary(group, int(cfg["annualization_periods"])),
            "mean_forecast_annualized_volatility": float(
                group["forecast_annualized_volatility"].mean()
            ),
            "mean_forecast_annualized_tracking_error": float(
                group["forecast_annualized_tracking_error"].mean()
            ),
            "mean_risky_weight": float(group["risky_weight"].mean()),
            "mean_alpha_utility_retention": float(
                group["alpha_utility_retention"].mean()
            ),
            "turnover_relaxation_fraction": float(
                group["turnover_limit_relaxed"].astype(bool).mean()
            ),
        })
    return pd.DataFrame(rows)


def _bootstrap(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records = []
    boot = cfg["bootstrap"]
    for (model, pipeline), group in weekly.groupby(["model", "alpha_pipeline"]):
        reference = group.loc[group["scenario"].eq("source_risk_constrained")]
        for scenario in ("risk_aware_full", "risk_aware_target_vol"):
            candidate = group.loc[group["scenario"].eq(scenario)]
            paired = candidate.merge(
                reference, on="asof_date", suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            metrics = {
                "net_return_annualized_arithmetic": (
                    paired["net_return_candidate"] - paired["net_return_reference"], 52.0
                ),
                "forecast_tracking_error": (
                    paired["forecast_annualized_tracking_error_candidate"]
                    - paired["forecast_annualized_tracking_error_reference"], 1.0
                ),
                "forecast_volatility": (
                    paired["forecast_annualized_volatility_candidate"]
                    - paired["forecast_annualized_volatility_reference"], 1.0
                ),
                "turnover": (
                    paired["turnover_candidate"] - paired["turnover_reference"], 1.0
                ),
            }
            for metric, (values, scale) in metrics.items():
                difference = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
                estimates = circular_block_bootstrap_mean(
                    difference, int(boot["block_length_weeks"]),
                    int(boot["samples"]), int(boot["random_seed"]),
                ) * scale
                records.append({
                    "model": model, "alpha_pipeline": pipeline,
                    "scenario": scenario, "reference": "source_risk_constrained",
                    "metric": metric, "observations": len(difference),
                    "point_estimate": float(difference.mean() * scale),
                    "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                    "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                })
    return pd.DataFrame(records)


def build_constraint_audit(
    lake: Lake, holdings: pd.DataFrame, weekly: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Recompute realized sleeve constraints from published target holdings."""
    source = lake.read("risk_model_portfolio_exposures", "v1")
    source = source.loc[source["factor"].astype(str).str.startswith("industry_"), [
        "asof_date", "model", "alpha_pipeline", "factor", "benchmark_exposure",
    ]].copy()
    source["industry_broad_code"] = source["factor"].astype(str).str.removeprefix(
        "industry_"
    )
    source = source.rename(columns={"benchmark_exposure": "universe_weight"}).drop_duplicates([
        "asof_date", "model", "alpha_pipeline", "industry_broad_code",
    ])
    full = holdings.loc[holdings["scenario"].eq("risk_aware_full")].copy()
    weekly_full = weekly.loc[weekly["scenario"].eq("risk_aware_full")].set_index(
        ["asof_date", "model", "alpha_pipeline"]
    )
    records = []
    keys = ["asof_date", "model", "alpha_pipeline"]
    for group_keys, group in full.groupby(keys, sort=False):
        date, model, pipeline = map(str, group_keys)
        allocation = group.groupby("industry_broad_code", as_index=False)[
            "target_weight"
        ].sum().rename(columns={"target_weight": "portfolio_weight"})
        benchmark = source.loc[
            source["asof_date"].astype(str).eq(date)
            & source["model"].astype(str).eq(model)
            & source["alpha_pipeline"].astype(str).eq(pipeline)
        ]
        allocation = benchmark.merge(
            allocation, on="industry_broad_code", how="outer"
        ).fillna(0.0)
        active = allocation["portfolio_weight"] - allocation["universe_weight"]
        row = weekly_full.loc[(date, model, pipeline)]
        turnover_limit = float(row["effective_turnover_limit"])
        is_initial = date == str(
            weekly_full.xs((model, pipeline), level=("model", "alpha_pipeline"))
            .index.min()
        )
        records.append({
            "asof_date": date, "model": model, "alpha_pipeline": pipeline,
            "maximum_stock_weight": float(group["target_weight"].max()),
            "absolute_size_z": abs(float(
                group["target_weight"] @ group["size_z"]
            )),
            "maximum_abs_industry_active_weight": float(active.abs().max()),
            "turnover": float(row["turnover"]),
            "effective_turnover_limit": turnover_limit,
            "is_initial_period": is_initial,
            "stock_weight_violation": max(
                0.0, float(group["target_weight"].max())
                - float(cfg["optimizer"]["maximum_stock_weight"])
            ),
            "size_bound_violation": max(
                0.0, abs(float(group["target_weight"] @ group["size_z"]))
                - float(cfg["optimizer"]["absolute_size_z_bound"])
            ),
            "industry_bound_violation": max(
                0.0, float(active.abs().max())
                - float(cfg["optimizer"]["industry_active_weight_bound"])
            ),
            "turnover_bound_violation": (
                0.0 if is_initial else max(
                    0.0, float(row["turnover"]) - turnover_limit
                )
            ),
            "alpha_utility_shortfall": max(
                0.0, float(cfg["optimizer"]["alpha_utility_retention_fraction"])
                - float(row["alpha_utility_retention"])
            ),
        })
    return pd.DataFrame(records)


def _write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame, cfg: dict) -> None:
    key = bootstrap.loc[bootstrap["metric"].isin([
        "net_return_annualized_arithmetic", "forecast_tracking_error",
        "forecast_volatility",
    ])]
    hgb = summary.loc[
        summary["model"].eq("hgb_shallow15_locked")
        & summary["alpha_pipeline"].eq("raw")
    ].set_index("scenario")
    hgb_return = bootstrap.loc[
        bootstrap["model"].eq("hgb_shallow15_locked")
        & bootstrap["alpha_pipeline"].eq("raw")
        & bootstrap["scenario"].eq("risk_aware_full")
        & bootstrap["metric"].eq("net_return_annualized_arithmetic")
    ].iloc[0]
    lines = [
        "# 事前多因子风险约束与目标波动实验", "", "## 方法", "",
        "- 候选股票为冻结的 Top-200 与约束组合持仓并集，并携带上期持仓；旧持仓若无当期分数则 alpha 效用为零。",
        "- 使用信号日前的 104 周 EWMA 收缩因子协方差和滞后个股特异方差。",
        "- 在保留最大可行 alpha score utility 98% 的前提下最小化主动风险。",
        "- 另设 20% 年化目标波动、只降仓不加杠杆的现金覆盖层。",
        "", "## 2021–2026 historical OOS 汇总", "",
        summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 相对原约束组合的配对块区间", "",
        key.to_markdown(index=False, floatfmt=".6f"), "",
        "## 决策", "",
        (
            "- **拒绝第一版风险政策（有效冻结登记 v2），不替换原约束组合。** raw HGB 的预测跟踪误差从 "
            f"{hgb.loc['source_risk_constrained', 'mean_forecast_annualized_tracking_error']:.2%} "
            f"降至 {hgb.loc['risk_aware_full', 'mean_forecast_annualized_tracking_error']:.2%}，"
            f"但年化收益从 {hgb.loc['source_risk_constrained', 'annualized_return']:.2%} "
            f"降至 {hgb.loc['risk_aware_full', 'annualized_return']:.2%}；年化算术收益差 95% 区间 "
            f"[{hgb_return['ci_2_5pct']:.2%}, {hgb_return['ci_97_5pct']:.2%}] 完全低于零。"
        ),
        "- 20% target-vol 覆盖层实现了预测波动目标，但没有改善 Sharpe；后续风险预算必须在独立验证期预设。",
        "",
        "> 周频 5 日标签口径；目标权重仍需通过日频涨跌停/停牌执行器复验。", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    artifact_key = str(cfg.get("artifact_key", "v2"))
    risk_cfg = load_risk_config()
    source_holdings = lake.read("constrained_portfolio_oos_holdings", "v1")
    source_weekly = lake.read("constrained_portfolio_oos_weekly", "v1")
    source_forecasts = lake.read("risk_model_portfolio_forecasts", "v1")
    factor_returns = lake.read("risk_model_factor_returns", "v1")
    allowed = set(cfg["candidate_source_constructions"])
    source_holdings = source_holdings.loc[
        source_holdings["construction"].isin(allowed)
    ].copy()
    candidates = {}
    for keys, group in source_holdings.groupby(
        ["asof_date", "model", "alpha_pipeline"], sort=False
    ):
        score_spread = group.groupby("instrument_id")["score"].agg(
            lambda value: float(value.max() - value.min())
        )
        if float(score_spread.max()) > 1e-10:
            raise RuntimeError(f"candidate scores disagree across source constructions: {keys}")
        candidates[tuple(map(str, keys))] = group.sort_values(
            ["instrument_id", "construction"]
        ).drop_duplicates("instrument_id")[["instrument_id", "score", "size_z"]]

    styles = list(cfg["styles"])
    factor_names = [
        "market", *[f"industry_{item}" for item in INDUSTRIES],
        *[f"style_{item}" for item in styles],
    ]
    covariances = _factor_covariances(factor_returns, factor_names, risk_cfg)
    specific_state: dict[str, float] = {}
    specific_counts: dict[str, int] = {}
    decay = np.exp(-np.log(2.0) / float(
        risk_cfg["specific_risk"]["ewma_half_life_weeks"]
    ))
    sleeve_states: dict[tuple[str, str], dict] = {}
    scenario_states: dict[tuple[str, str, str], dict] = {}
    weekly_records: list[dict] = []
    holding_frames: list[pd.DataFrame] = []
    forecast_records: list[dict] = []
    start_oos = int(cfg["historical_oos"]["start_year"])

    for year in range(int(cfg["risk_model"]["start_year"]),
                      int(cfg["risk_model"]["end_year"]) + 1):
        frame = _read_year(lake, year, risk_cfg)
        for asof_date, original in frame.groupby("asof_date", sort=True):
            date = str(asof_date)
            fitted = fit_factor_cross_section(original, risk_cfg)
            cross = fitted["frame"].copy()
            cross_codes = set(cross["instrument_id"].astype(str))
            matured = [
                value for code, value in specific_state.items()
                if specific_counts.get(code, 0) >= int(
                    risk_cfg["specific_risk"]["minimum_observations"]
                ) and np.isfinite(value)
            ]
            default_specific = float(np.median(matured)) if matured else float(
                np.var(cross["residual_return"], ddof=1)
            )
            covariance = covariances.get(date)
            keys_for_date = [key for key in candidates if key[0] == date]
            if year >= start_oos and covariance is None:
                raise RuntimeError(f"missing lagged covariance for {date}")
            for _, model, pipeline in keys_for_date:
                state_key = (model, pipeline)
                sleeve_state = sleeve_states.setdefault(state_key, {})
                previous, _ = _drifted(sleeve_state)
                current = candidates[(date, model, pipeline)].copy()
                current["alpha_utility"] = _score_utility(current["score"])
                current_codes = set(current["instrument_id"].astype(str))
                carry_codes = [
                    code for code, weight in previous.items()
                    if code not in current_codes
                    and code in cross_codes
                    and weight >= float(cfg["carry_minimum_weight"])
                ]
                carry = pd.DataFrame({
                    "instrument_id": carry_codes,
                    "score": np.nan,
                    "size_z": np.nan,
                    "alpha_utility": 0.0,
                })
                candidate = pd.concat([current, carry], ignore_index=True).merge(
                    cross, on="instrument_id", how="left", validate="one_to_one"
                )
                if candidate["industry_broad_code"].isna().any():
                    raise RuntimeError(f"candidate exposure missing: {(date, model, pipeline)}")
                # Use the same universe-standardized size exposure as the risk model.
                candidate["size_z"] = candidate["style_size"]
                style_columns = [f"style_{item}" for item in styles]
                industry = candidate["industry_broad_code"].astype(str)
                factor_loadings = np.column_stack([
                    np.ones(len(candidate)),
                    *[industry.eq(item).to_numpy(dtype="float64") for item in INDUSTRIES],
                    *[candidate[f"style_{item}"].to_numpy(dtype="float64") for item in styles],
                ])
                benchmark_industry = fitted["benchmark_industry"]
                benchmark_exposure = np.array([
                    1.0, *[benchmark_industry.get(item, 0.0) for item in INDUSTRIES],
                    *[0.0 for _ in styles],
                ])
                selected_specific = np.array([
                    specific_state.get(str(code), default_specific)
                    for code in candidate["instrument_id"]
                ])
                universe_specific = np.array([
                    specific_state.get(str(code), default_specific)
                    for code in cross["instrument_id"]
                ])
                weights, diagnostics = solve_risk_aware_weights(
                    candidate, previous, benchmark_industry, factor_loadings,
                    benchmark_exposure, covariance, selected_specific,
                    1.0 / len(cross), cfg,
                )
                candidate_codes = candidate["instrument_id"].astype(str).tolist()
                cross.attrs["candidate_codes"] = candidate_codes
                base_risk = _risk_metrics(
                    weights, factor_loadings, benchmark_exposure, covariance,
                    selected_specific, cross, universe_specific,
                    int(cfg["annualization_periods"]),
                )
                forecast_vol = base_risk["forecast_annualized_volatility"]
                target_cfg = cfg["target_volatility"]
                risky_weight = float(np.clip(
                    float(target_cfg["annualized_target"]) / max(forecast_vol, 1e-12),
                    float(target_cfg["minimum_risky_weight"]),
                    float(target_cfg["maximum_risky_weight"]),
                ))
                scenarios = {
                    "risk_aware_full": (1.0, weights, base_risk),
                    "risk_aware_target_vol": (risky_weight, risky_weight * weights, None),
                }
                realized = pd.to_numeric(
                    candidate[str(cfg["target"])], errors="coerce"
                ).to_numpy(dtype="float64")
                filled = np.nan_to_num(realized, nan=0.0)
                benchmark_return = float(fitted["market_return"])
                for scenario, (scale, actual_weights, metrics) in scenarios.items():
                    if metrics is None:
                        metrics = _risk_metrics(
                            actual_weights, factor_loadings, benchmark_exposure,
                            covariance, selected_specific, cross, universe_specific,
                            int(cfg["annualization_periods"]),
                        )
                    actual = {
                        code: float(weight) for code, weight in zip(
                            candidate_codes, actual_weights
                        ) if weight > float(cfg["optimizer"]["minimum_reported_weight"])
                    }
                    cash = 1.0 - float(actual_weights.sum())
                    scenario_state = scenario_states.setdefault(
                        (model, pipeline, scenario), {}
                    )
                    turnover = _turnover(actual, cash, scenario_state)
                    gross_return = float(actual_weights @ filled)
                    cost = turnover * float(cfg["portfolio"]["round_trip_cost_bps"]) / 10_000
                    realized_fraction = float(actual_weights[np.isfinite(realized)].sum() + cash)
                    weekly_records.append({
                        "asof_date": date, "test_year": year, "model": model,
                        "alpha_pipeline": pipeline, "scenario": scenario,
                        "turnover": turnover, "gross_return": gross_return,
                        "transaction_cost": cost, "net_return": gross_return - cost,
                        "benchmark_return": benchmark_return,
                        "gross_excess_return": gross_return - benchmark_return,
                        "net_excess_return": gross_return - cost - benchmark_return,
                        "realized_weight_fraction": realized_fraction,
                        "included_in_performance": realized_fraction >= float(
                            cfg["portfolio"]["minimum_realized_weight_fraction"]
                        ),
                        "risky_weight": scale, "cash_weight": cash,
                        **diagnostics, **metrics,
                    })
                    scenario_state["weights"] = actual
                    scenario_state["cash"] = cash
                    scenario_state["returns"] = dict(zip(candidate_codes, filled))
                    selected = candidate.loc[actual_weights > 0, [
                        "instrument_id", "industry_broad_code", "size_z", "score",
                    ]].copy()
                    selected["target_weight"] = actual_weights[actual_weights > 0]
                    holding_frames.append(selected.assign(
                        asof_date=date, test_year=year, model=model,
                        alpha_pipeline=pipeline, scenario=scenario,
                        risky_weight=scale, cash_weight=cash,
                    ))
                    forecast_records.append({
                        "asof_date": date, "model": model,
                        "alpha_pipeline": pipeline, "scenario": scenario,
                        "risky_weight": scale, **metrics,
                    })
                sleeve_state["weights"] = dict(zip(candidate_codes, weights))
                sleeve_state["cash"] = 0.0
                sleeve_state["returns"] = dict(zip(candidate_codes, filled))

            for row in cross[["instrument_id", "residual_return"]].itertuples(index=False):
                code, residual = str(row.instrument_id), float(row.residual_return)
                prior = specific_state.get(code, residual * residual)
                specific_state[code] = decay * prior + (1.0 - decay) * residual * residual
                specific_counts[code] = specific_counts.get(code, 0) + 1
        print(f"risk-aware year={year} dates={frame['asof_date'].nunique()}", flush=True)

    weekly = pd.DataFrame(weekly_records)
    holdings = pd.concat(holding_frames, ignore_index=True)
    forecasts = pd.DataFrame(forecast_records)
    # Add the frozen source portfolio to the paired comparison table.
    reference = source_weekly.loc[source_weekly["construction"].eq("risk_constrained")].copy()
    reference = reference.merge(
        source_forecasts[[
            "asof_date", "model", "alpha_pipeline",
            "forecast_annualized_volatility", "forecast_annualized_tracking_error",
        ]], on=["asof_date", "model", "alpha_pipeline"], how="left",
        validate="one_to_one",
    )
    reference["scenario"] = "source_risk_constrained"
    reference["risky_weight"] = 1.0
    reference["cash_weight"] = 0.0
    reference["alpha_utility_retention"] = 1.0
    reference["turnover_limit_relaxed"] = reference[
        "turnover_limit_relaxed"
    ].astype(bool)
    comparison_columns = sorted(set(weekly.columns) & set(reference.columns))
    comparison = pd.concat([
        reference.reindex(columns=weekly.columns), weekly
    ], ignore_index=True)
    summary = _summary(comparison, cfg)
    bootstrap = _bootstrap(comparison, cfg)
    constraint_audit = build_constraint_audit(lake, holdings, weekly, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "historical_oos_used_for_selection": False,
        "source_experiment": cfg["source_experiment"],
        "factor_covariance_uses_current_or_future_return": False,
        "specific_variance_uses_current_or_future_residual": False,
        "candidate_set_uses_future_return": False,
        "target_volatility_annualized": float(
            cfg["target_volatility"]["annualized_target"]
        ),
    }])
    outputs = {
        "risk_aware_portfolio_protocol": protocol,
        "risk_aware_portfolio_weekly": comparison,
        "risk_aware_portfolio_holdings": holdings,
        "risk_aware_portfolio_forecasts": forecasts,
        "risk_aware_portfolio_constraint_audit": constraint_audit,
        "risk_aware_portfolio_summary": summary,
        "risk_aware_portfolio_bootstrap": bootstrap,
    }
    for table, output in outputs.items():
        lake.write(
            table, output, key=artifact_key,
            source="risk_aware_portfolio_optimization.v1", request=cfg,
        )
    _write_report(summary, bootstrap, cfg)
    print(summary.to_string(index=False), flush=True)
    return summary, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lagged multi-factor risk-aware portfolio optimization"
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
