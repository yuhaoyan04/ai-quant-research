"""Validation-only HGB two-dimensional stability surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import purged_end_date, sum_statistics, summarize
from quant_research.model_benchmark import (
    _evaluate, _read_year, _summary, build_year_training, fit_candidate,
)
from quant_research.model_challenger import concatenate_samples
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "hgb_hyperparameter_stability.yaml"
REPORT_PATH = ROOT / "reports" / "hgb_hyperparameter_stability.md"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _candidate_name(leaves: int, l2: float) -> str:
    return f"hgb_leaf{int(leaves)}_l2_{float(l2):g}"


def candidate_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = cfg["fixed_parameters"]
    specs = []
    for leaves in cfg["surface"]["max_leaf_nodes"]:
        for l2 in cfg["surface"]["l2_regularization"]:
            specs.append({
                "candidate": _candidate_name(int(leaves), float(l2)),
                "family": "hgb", "kind": "hgb",
                "max_leaf_nodes": int(leaves), "l2_regularization": float(l2),
                "learning_rate": float(fixed["learning_rate"]),
                "max_iter": int(fixed["max_iter"]),
                "min_samples_leaf": int(fixed["min_samples_leaf"]),
            })
    return specs


def validate_surface(specs: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    expected = len(cfg["surface"]["max_leaf_nodes"]) * len(
        cfg["surface"]["l2_regularization"]
    )
    coordinates = {(item["max_leaf_nodes"], item["l2_regularization"]) for item in specs}
    if len(specs) != expected or len(coordinates) != expected:
        raise ValueError("HGB surface must be a complete unique Cartesian grid")
    baseline = cfg["baseline"]
    if (int(baseline["max_leaf_nodes"]), float(baseline["l2_regularization"])) not in coordinates:
        raise ValueError("HGB baseline must be present in the stability surface")


def annual_summary(daily: pd.DataFrame, portfolio: pd.DataFrame,
                   periods: int) -> pd.DataFrame:
    rows = []
    for (year, candidate), group in daily.groupby(["test_year", "candidate"], sort=True):
        holdings = portfolio.loc[
            portfolio["test_year"].eq(year) & portfolio["candidate"].eq(candidate)
        ]
        rows.append({
            "year": int(year), "candidate": candidate,
            **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                holdings, periods
            ).items()},
        })
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, annual: pd.DataFrame,
                  winners: pd.DataFrame, cfg: dict[str, Any]) -> None:
    display = summary[[
        "candidate", "mean_rank_ic", "annualized_icir",
        "portfolio_annualized_return", "portfolio_sharpe_zero_rate",
        "portfolio_maximum_drawdown", "fit_seconds",
    ]].sort_values("mean_rank_ic", ascending=False)
    lines = [
        "# HGB 二维参数稳定面", "", "## 研究约束", "",
        "仅使用 2006–2015 训练、2016–2020 验证；不访问 historical OOS，也不因网格最佳点改写已冻结模型。二维网格同时覆盖叶节点数与 L2 正则。",
        "", "## 全验证期", "", display.to_markdown(index=False, floatfmt=".6f"),
        "", "## 各年 RankIC 最优坐标", "", winners.to_markdown(index=False, floatfmt=".6f"),
        "", "年度完整结果保存于 `hgb_hyperparameter_annual/v1`。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    specs = candidate_specs(cfg)
    validate_surface(specs, cfg)
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
        print(f"hgb surface training year={year}", flush=True)
    statistics = sum_statistics(
        [full_stats[year] for year in range(start, end)] + [purged_stats[end]]
    )
    sample = concatenate_samples(
        [full_samples[year] for year in range(start, end)] + [purged_samples[end]]
    )
    models = {}
    for spec in specs:
        model = fit_candidate(spec, statistics, sample, len(factors), cfg)
        models[model.candidate] = model
        print(f"hgb surface fit={model.candidate} seconds={model.fit_seconds:.2f}", flush=True)

    daily_frames, portfolio_frames = [], []
    states = {name: {} for name in models}
    validation_end = int(cfg["validation"]["end_year"])
    cutoff = purged_end_date(calendar, f"{validation_end}-12-31",
                             int(cfg["label_horizon_trading_days"]))
    for year in range(int(cfg["validation"]["start_year"]), validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        daily, portfolio = _evaluate(
            frame, models, factors, target, states, cfg, "validation", year
        )
        daily_frames.append(daily)
        portfolio_frames.append(portfolio)
        print(f"hgb surface evaluated year={year}", flush=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    summary = _summary(
        daily, portfolio, models, int(cfg["annualization_periods"]), "validation"
    )
    spec_frame = pd.DataFrame(specs)[[
        "candidate", "max_leaf_nodes", "l2_regularization"
    ]]
    summary = spec_frame.merge(summary, on="candidate", validate="one_to_one")
    annual = spec_frame.merge(
        annual_summary(daily, portfolio, int(cfg["annualization_periods"])),
        on="candidate", validate="one_to_many",
    )
    winners = annual.sort_values(
        ["year", "mean_rank_ic", "portfolio_sharpe_zero_rate"],
        ascending=[True, False, False],
    ).groupby("year", as_index=False).first()
    baseline = cfg["baseline"]
    baseline_name = _candidate_name(
        int(baseline["max_leaf_nodes"]), float(baseline["l2_regularization"])
    )
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "purpose": cfg["purpose"],
        "training_period": "2006_2015", "evaluation_period": "2016_2020_validation_only",
        "historical_oos_accessed": False, "surface_points": len(specs),
        "baseline_candidate": baseline_name,
        "annual_winner_unique_coordinates": winners[[
            "max_leaf_nodes", "l2_regularization"
        ]].drop_duplicates().shape[0],
        "best_candidate_is_not_promoted": True,
    }])
    for table, frame in {
        "hgb_hyperparameter_protocol": protocol,
        "hgb_hyperparameter_summary": summary,
        "hgb_hyperparameter_annual": annual,
        "hgb_hyperparameter_annual_winners": winners,
    }.items():
        lake.write(table, frame, key="v1", source="hgb_hyperparameter_stability.v1",
                   request=cfg)
    _write_report(summary, annual, winners, cfg)
    print(summary.to_string(index=False), flush=True)
    print(winners.to_string(index=False), flush=True)
    return summary, winners


if __name__ == "__main__":
    run()
