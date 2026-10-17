"""Common-date Top-N, weighting, and executable cost sensitivity diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.daily_execution import daily_performance_summary
from quant_research.ingest import ROOT, Lake
from quant_research.portfolio import performance_summary


CONFIG_PATH = ROOT / "config" / "portfolio_sensitivity.yaml"
REPORT_PATH = ROOT / "reports" / "portfolio_sensitivity.md"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _target_weights(selected: pd.DataFrame, method: str, volatility: str,
                    clip_quantiles: tuple[float, float]) -> pd.Series:
    if method == "equal_weight":
        return pd.Series(1.0 / len(selected), index=selected.index)
    if method != "inverse_volatility":
        raise ValueError(f"unknown weighting method: {method}")
    values = pd.to_numeric(selected[volatility], errors="coerce")
    usable = values.loc[values.gt(0) & np.isfinite(values)]
    if usable.empty:
        return pd.Series(1.0 / len(selected), index=selected.index)
    filled = values.where(values.gt(0) & np.isfinite(values), usable.median())
    lower, upper = usable.quantile(list(clip_quantiles))
    inverse = 1.0 / filled.clip(lower=lower, upper=upper)
    return inverse / inverse.sum()


def simulate_weighted_long_only(
    predictions: pd.DataFrame, *, top_n: int, weighting: str, target: str,
    volatility: str, clip_quantiles: tuple[float, float], cost_bps: float,
    minimum_realized_weight_fraction: float,
) -> pd.DataFrame:
    """Form weights before outcomes and preserve missing realized returns as zero."""
    required = {"instrument_id", "asof_date", "available_date", "score", target, volatility}
    if missing := required - set(predictions):
        raise KeyError(f"portfolio sensitivity columns missing: {sorted(missing)}")
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    records = []
    for asof_date, group in predictions.groupby("asof_date", sort=True):
        selected = group.sort_values(
            ["score", "instrument_id"], ascending=[False, True]
        ).head(int(top_n)).copy()
        if selected.empty:
            continue
        weights = _target_weights(selected, weighting, volatility, clip_quantiles)
        codes = selected["instrument_id"].astype(str)
        current = dict(zip(codes, weights.astype(float)))
        if previous_weights:
            drifted_values = {
                code: weight * (1.0 + previous_returns.get(code, 0.0))
                for code, weight in previous_weights.items()
            }
            total = sum(drifted_values.values())
            drifted = ({code: value / total for code, value in drifted_values.items()}
                       if total > 0 else previous_weights)
            turnover = 0.5 * sum(
                abs(current.get(code, 0.0) - drifted.get(code, 0.0))
                for code in set(current) | set(drifted)
            )
        else:
            turnover = 1.0
        realized = pd.to_numeric(selected[target], errors="coerce")
        realized_values = dict(zip(codes, realized.fillna(0.0)))
        realized_weight = float(weights.loc[realized.notna()].sum())
        gross = float((weights * realized.fillna(0.0)).sum())
        benchmark = float(pd.to_numeric(group[target], errors="coerce").fillna(0.0).mean())
        cost = float(turnover * cost_bps / 10_000.0)
        records.append({
            "asof_date": str(asof_date),
            "rebalance_date": selected["available_date"].dropna().min(),
            "top_n": int(top_n), "weighting": weighting,
            "holdings": len(selected), "candidate_pool_size": len(group),
            "maximum_target_weight": float(weights.max()),
            "turnover": float(turnover), "gross_return": gross,
            "transaction_cost": cost, "net_return": gross - cost,
            "benchmark_return": benchmark,
            "gross_excess_return": gross - benchmark,
            "net_excess_return": gross - cost - benchmark,
            "realized_weight_fraction": realized_weight,
            "included_in_performance": (
                realized_weight >= minimum_realized_weight_fraction
            ),
        })
        previous_weights = current
        previous_returns = realized_values
    return pd.DataFrame(records)


def _attach_volatility(lake: Lake, predictions: pd.DataFrame, field: str) -> pd.DataFrame:
    years = sorted(predictions["asof_date"].astype(str).str[:4].astype(int).unique())
    prediction_dates = set(predictions["asof_date"].astype(str))
    frames = []
    for year in years:
        path = lake.table_path("multi_horizon_factor_panel", str(year))
        frame = pd.read_parquet(path, columns=["instrument_id", "asof_date", field])
        frame = frame.loc[frame["asof_date"].isin(prediction_dates)]
        frames.append(frame)
    exposure = pd.concat(frames, ignore_index=True)
    result = predictions.merge(
        exposure, on=["instrument_id", "asof_date"], how="left", validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError(f"{field} keys do not cover every prediction row")
    return result.drop(columns="_merge")


def _weekly_summary(weekly: pd.DataFrame, split: str, periods: int) -> pd.DataFrame:
    rows = []
    for (top_n, weighting), group in weekly.groupby(["top_n", "weighting"], sort=True):
        rows.append({
            "split": split, "top_n": int(top_n), "weighting": weighting,
            "common_dates": group["asof_date"].nunique(),
            "maximum_target_weight": float(group["maximum_target_weight"].max()),
            **performance_summary(group, periods),
        })
    return pd.DataFrame(rows)


def _daily_cost_stress(lake: Lake, cfg: dict[str, Any]) -> pd.DataFrame:
    daily = lake.read("constrained_daily_execution_daily", "v1")
    daily = daily.loc[
        daily["scenario"].eq("daily_retry_limit_and_suspension_constrained")
    ].copy()
    rows = []
    keys = ["model", "alpha_pipeline", "strategy", "scenario"]
    for values, group in daily.groupby(keys, sort=True):
        for cost_bps in cfg["execution_cost_stress_bps"]:
            repriced = group.copy()
            repriced["net_return"] = (
                repriced["gross_return"]
                - repriced["turnover"] * float(cost_bps) / 10_000.0
            )
            rows.append({
                **dict(zip(keys, values)), "cost_bps": float(cost_bps),
                "stress_method": "fixed_realized_fill_path_reprice",
                **daily_performance_summary(
                    repriced, int(cfg["daily_annualization_periods"])
                ),
            })
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, stress: pd.DataFrame, cfg: dict[str, Any]) -> None:
    validation = summary.loc[summary["split"].eq("validation")]
    oos = summary.loc[summary["split"].eq("historical_oos")]
    stress_display = stress[[
        "model", "alpha_pipeline", "cost_bps", "annualized_return",
        "annualized_volatility", "sharpe_zero_rate", "maximum_drawdown",
    ]]
    lines = [
        "# Top-N、权重与成本压力测试", "", "## 研究边界", "",
        "这是在历史 OOS 已被查看后的回顾性稳健性诊断，不用于重新选择已锁定的 Top-200 等权政策。所有 Top-N 与权重方案使用共同日期；inverse-vol 仅使用信号日已知的 20 日波动率。",
        "", "## 2016–2020 验证期", "", validation.to_markdown(index=False, floatfmt=".6f"),
        "", "## 2021–2026 historical OOS", "", oos.to_markdown(index=False, floatfmt=".6f"),
        "", "## 最终日频可成交组合的成本重定价", "",
        "固定实际成交路径，仅将线性费率重定价为 0/20/50/100bp；这不是含市场冲击、费率反馈与股数取整的完整重跑。",
        "", stress_display.to_markdown(index=False, floatfmt=".6f"), "",
        f"研究状态：`{cfg['research_status']}`。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    volatility = str(cfg["inverse_volatility_field"])
    clip = tuple(float(value) for value in cfg["inverse_volatility_clip_quantiles"])
    frames, summaries = [], []
    sources = {
        "validation": lake.read("model_validation_predictions", str(cfg["model"])),
        "historical_oos": lake.read("model_oos_predictions", str(cfg["model"])),
    }
    for split, source in sources.items():
        panel = _attach_volatility(lake, source, volatility)
        split_frames = []
        for top_n in cfg["top_n"]:
            for weighting in cfg["weighting"]:
                result = simulate_weighted_long_only(
                    panel, top_n=int(top_n), weighting=str(weighting),
                    target=str(cfg["target"]), volatility=volatility,
                    clip_quantiles=clip, cost_bps=float(cfg["base_round_trip_cost_bps"]),
                    minimum_realized_weight_fraction=float(
                        cfg["minimum_realized_weight_fraction"]
                    ),
                )
                result.insert(0, "split", split)
                split_frames.append(result)
        combined = pd.concat(split_frames, ignore_index=True)
        date_counts = combined.groupby(["top_n", "weighting"])["asof_date"].nunique()
        if date_counts.nunique() != 1:
            raise RuntimeError(f"{split}: sensitivity variants do not share common dates")
        variant_count = len(cfg["top_n"]) * len(cfg["weighting"])
        common_eligible_dates = set(
            combined.loc[combined["included_in_performance"]]
            .groupby("asof_date").size().loc[lambda values: values.eq(variant_count)].index
        )
        combined["included_in_performance"] = combined["asof_date"].isin(
            common_eligible_dates
        )
        frames.append(combined)
        summaries.append(_weekly_summary(
            combined, split, int(cfg["weekly_annualization_periods"])
        ))
    weekly = pd.concat(frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    stress = _daily_cost_stress(lake, cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "research_status": cfg["research_status"],
        "historical_oos_used_for_selection": False,
        "base_cost_bps": float(cfg["base_round_trip_cost_bps"]),
        "common_date_comparison": True,
        "daily_cost_stress_method": "fixed_realized_fill_path_reprice",
    }])
    for table, frame in {
        "portfolio_sensitivity_protocol": protocol,
        "portfolio_sensitivity_weekly": weekly,
        "portfolio_sensitivity_summary": summary,
        "execution_cost_stress": stress,
    }.items():
        lake.write(table, frame, key="v1", source="portfolio_sensitivity.v1", request=cfg)
    _write_report(summary, stress, cfg)
    print(summary.to_string(index=False))
    print(stress.to_string(index=False))
    return summary, stress


if __name__ == "__main__":
    run()
