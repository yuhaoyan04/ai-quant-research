"""Descriptive factor distributions, heatmap, and rolling-correlation stability."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import weekly_rows
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "factor_visual_diagnostics.yaml"
REPORT_PATH = ROOT / "reports" / "factor_visual_diagnostics.md"
HEATMAP_PATH = ROOT / "reports" / "figures" / "factor_correlation_heatmap.svg"
DISTRIBUTION_PATH = ROOT / "reports" / "figures" / "factor_distributions.svg"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def correlation_heatmap_svg(matrix: pd.DataFrame, title: str) -> str:
    names = list(matrix.index)
    cell, left, top = 34, 215, 210
    width, height = left + cell * len(names) + 40, top + cell * len(names) + 70
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-family="sans-serif" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]
    for index, name in enumerate(names):
        y = top + index * cell + 23
        x = left + index * cell + 17
        lines.append(f'<text x="{left - 8}" y="{y}" text-anchor="end" font-family="sans-serif" font-size="10">{escape(name)}</text>')
        lines.append(f'<text x="{x}" y="{top - 8}" transform="rotate(-55 {x} {top - 8})" text-anchor="start" font-family="sans-serif" font-size="10">{escape(name)}</text>')
    for row, row_name in enumerate(names):
        for column, column_name in enumerate(names):
            value = float(matrix.loc[row_name, column_name])
            strength = min(abs(value), 1.0)
            if value >= 0:
                red, other = 205, int(245 - 150 * strength)
                color = f'rgb({red},{other},{other})'
            else:
                blue, other = 205, int(245 - 150 * strength)
                color = f'rgb({other},{other},{blue})'
            x, y = left + column * cell, top + row * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="1"><title>{escape(row_name)} vs {escape(column_name)}: {value:.3f}</title></rect>')
    lines.append('<text x="20" y="65" font-family="sans-serif" font-size="11" fill="#555">Red = positive; blue = negative; hover a cell for the value.</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def distribution_svg(summary: pd.DataFrame) -> str:
    width, row_height, left = 980, 34, 220
    height = 80 + row_height * len(summary)
    plot_width = width - left - 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">OOS raw-factor distributions</text>',
        '<text x="20" y="52" font-family="sans-serif" font-size="11" fill="#555">Each row is scaled independently from p1 to p99; box = p25–p75, line = median.</text>',
    ]
    for index, row in summary.reset_index(drop=True).iterrows():
        y = 72 + index * row_height
        p01, p99 = float(row.p01), float(row.p99)
        scale = p99 - p01
        def x(value: float) -> float:
            return left + plot_width * ((value - p01) / scale if scale > 0 else 0.5)
        lines.extend([
            f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" font-family="sans-serif" font-size="11">{escape(str(row.factor))}</text>',
            f'<line x1="{left}" y1="{y + 14}" x2="{left + plot_width}" y2="{y + 14}" stroke="#b8c2cc" stroke-width="2"/>',
            f'<rect x="{x(float(row.p25)):.2f}" y="{y + 7}" width="{max(x(float(row.p75)) - x(float(row.p25)), 1):.2f}" height="14" fill="#9ecae1" stroke="#3182bd"/>',
            f'<line x1="{x(float(row.p50)):.2f}" y1="{y + 5}" x2="{x(float(row.p50)):.2f}" y2="{y + 23}" stroke="#08306b" stroke-width="2"/>',
            f'<text x="{left + plot_width + 8}" y="{y + 18}" font-family="sans-serif" font-size="10" fill="#555">missing {float(row.missing_fraction):.1%}</text>',
        ])
    lines.append('</svg>')
    return "\n".join(lines)


def _load_weekly_factors(lake: Lake, factors: list[str], start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        frame = pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["asof_date", *factors],
        )
        frames.append(weekly_rows(frame))
    return pd.concat(frames, ignore_index=True)


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    period = cfg["distribution_period"]
    panel = _load_weekly_factors(
        lake, factors, int(period["start_year"]), int(period["end_year"])
    )
    distribution_rows = []
    for factor in factors:
        values = pd.to_numeric(panel[factor], errors="coerce")
        quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        distribution_rows.append({
            "factor": factor, "observations": int(values.notna().sum()),
            "missing_fraction": float(values.isna().mean()),
            "p01": float(quantiles.loc[0.01]), "p25": float(quantiles.loc[0.25]),
            "p50": float(quantiles.loc[0.5]), "p75": float(quantiles.loc[0.75]),
            "p99": float(quantiles.loc[0.99]),
        })
    distributions = pd.DataFrame(distribution_rows)

    weekly_rows_list = []
    minimum = int(cfg["minimum_pair_observations"])
    for date, group in panel.groupby("asof_date", sort=True):
        if len(group) < minimum:
            continue
        correlations = group[factors].corr(method="spearman", min_periods=minimum)
        for left_index, left in enumerate(factors):
            for right in factors[left_index + 1:]:
                weekly_rows_list.append({
                    "asof_date": date, "factor_a": left, "factor_b": right,
                    "cross_sectional_spearman": correlations.loc[left, right],
                })
    weekly = pd.DataFrame(weekly_rows_list).sort_values(
        ["factor_a", "factor_b", "asof_date"]
    )
    window = int(cfg["rolling_window_weeks"])
    weekly["rolling_mean_correlation"] = weekly.groupby(
        ["factor_a", "factor_b"]
    )["cross_sectional_spearman"].transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )
    rolling = weekly.dropna(subset=["rolling_mean_correlation"])
    stability = rolling.groupby(["factor_a", "factor_b"]).agg(
        rolling_windows=("rolling_mean_correlation", "size"),
        mean_rolling_correlation=("rolling_mean_correlation", "mean"),
        minimum_rolling_correlation=("rolling_mean_correlation", "min"),
        maximum_rolling_correlation=("rolling_mean_correlation", "max"),
        rolling_correlation_std=("rolling_mean_correlation", "std"),
    ).reset_index()

    correlations = lake.read("factor_correlation", "v1")
    selected = correlations.loc[
        correlations["sample_split"].eq(str(cfg["display_split"]))
    ]
    matrix = selected.pivot(
        index="factor_a", columns="factor_b", values="mean_daily_correlation"
    ).reindex(index=factors, columns=factors)
    HEATMAP_PATH.write_text(
        correlation_heatmap_svg(matrix, "Validation mean cross-sectional factor correlation"),
        encoding="utf-8",
    )
    DISTRIBUTION_PATH.write_text(distribution_svg(distributions), encoding="utf-8")
    for table, frame in {
        "factor_distribution_summary": distributions,
        "factor_weekly_correlations": weekly,
        "factor_rolling_correlation_stability": stability,
    }.items():
        lake.write(table, frame, key="v1", source="factor_visual_diagnostics.v1", request=cfg)
    unstable = stability.assign(
        range=lambda frame: frame["maximum_rolling_correlation"]
        - frame["minimum_rolling_correlation"]
    ).nlargest(10, "range")
    lines = [
        "# 因子分布与相关性稳定性", "",
        "本页只做描述性展示，不用于重新筛选因子或模型。分布使用 2021–2026 周频信号截面；热图使用验证期，滚动相关使用 52 周历史 OOS 窗口。",
        "", "![因子相关性热图](figures/factor_correlation_heatmap.svg)", "",
        "![因子分布](figures/factor_distributions.svg)", "",
        "## 滚动相关变化最大的因子对", "",
        unstable.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(distributions.to_string(index=False), flush=True)
    print(unstable.to_string(index=False), flush=True)
    return distributions, stability


if __name__ == "__main__":
    run()
