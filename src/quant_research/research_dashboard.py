"""Generate a self-contained offline dashboard for final research results."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.ingest import ROOT, Lake
from quant_research.robustness import index_forward_returns


REPORT = ROOT / "reports" / "research_dashboard.html"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
PUBLIC_RESULTS = ROOT / "reports" / "public_results.json"
COLORS = {"Ridge": "#4f8cff", "AI blend": "#f97316", "CSI 500": "#64748b"}


def _multi_line(frame: pd.DataFrame, columns: list[str], width: int = 920,
                height: int = 320, *, percent: bool = False) -> str:
    values = frame[columns].astype(float)
    low, high = float(values.min().min()), float(values.max().max())
    padding = max((high - low) * 0.08, 0.02)
    low, high = low - padding, high + padding
    span = high - low or 1.0
    left, right, top, bottom = 66, 18, 44, 42
    plot_width, plot_height = width - left - right, height - top - bottom

    dates = (pd.to_datetime(frame["asof_date"]) if "asof_date" in frame
             else pd.Series(pd.RangeIndex(len(frame))))
    if len(frame) > 1 and "asof_date" in frame:
        date_numbers = dates.astype("int64").to_numpy(dtype="float64")
        x_values = left + (date_numbers - date_numbers.min()) / (
            date_numbers.max() - date_numbers.min()) * plot_width
    else:
        x_values = np.linspace(left, left + plot_width, max(len(frame), 1))

    grid = []
    for index, value in enumerate(np.linspace(low, high, 5)):
        y = top + plot_height - index / 4 * plot_height
        label = f"{value:.0%}" if percent else f"{value:.2f}"
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            f'stroke="#e5eaf1"/><text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#64748b" font-size="12">{label}</text>')

    tick_indices = np.linspace(0, max(len(frame) - 1, 0), min(6, max(len(frame), 1)), dtype=int)
    x_ticks = []
    for index in np.unique(tick_indices):
        label = (dates.iloc[index].strftime("%Y-%m") if "asof_date" in frame else str(index))
        x = x_values[index]
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" '
            f'y2="{top + plot_height + 5}" stroke="#94a3b8"/>'
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" '
            f'fill="#64748b" font-size="12">{label}</text>')

    paths = []
    for column in columns:
        points = " ".join(
            f"{x_values[i]:.1f},{top + plot_height - (value-low)/span*plot_height:.1f}"
            for i, value in enumerate(values[column]) if np.isfinite(value))
        paths.append(
            f'<polyline fill="none" stroke="{COLORS[column]}" stroke-width="2.6" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>')
    legend = []
    legend_x = left
    for column in columns:
        legend.append(
            f'<line x1="{legend_x}" y1="20" x2="{legend_x + 20}" y2="20" '
            f'stroke="{COLORS[column]}" stroke-width="3"/>'
            f'<text x="{legend_x + 27}" y="24" fill="#475569" font-size="12">'
            f'{html.escape(column)}</text>')
        legend_x += 50 + len(column) * 8
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" style="background:#fff;font-family:Inter,system-ui,Microsoft YaHei,sans-serif">'
        + "".join(legend) + "".join(grid) + "".join(x_ticks) + "".join(paths)
        + f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
          f'y2="{top + plot_height}" stroke="#94a3b8"/></svg>'
    )
    return svg


def _grouped_bars(frame: pd.DataFrame, category: str, columns: list[str],
                  width: int = 920, height: int = 330, *, percent: bool = True) -> str:
    left, right, top, bottom = 66, 18, 44, 50
    plot_width, plot_height = width - left - right, height - top - bottom
    values = frame[columns].astype(float)
    low = min(0.0, float(values.min().min()))
    high = max(0.0, float(values.max().max()))
    padding = max((high - low) * 0.10, 0.01)
    low, high = low - padding, high + padding
    span = high - low or 1.0
    zero_y = top + plot_height - (0 - low) / span * plot_height
    group_width = plot_width / max(len(frame), 1)
    bar_width = group_width * 0.70 / max(len(columns), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" style="background:#fff;font-family:Inter,system-ui,Microsoft YaHei,sans-serif">'
    ]
    legend_x = left
    for column in columns:
        parts.append(
            f'<rect x="{legend_x}" y="14" width="18" height="9" fill="{COLORS[column]}" rx="2"/>'
            f'<text x="{legend_x + 25}" y="24" fill="#475569" font-size="12">'
            f'{html.escape(column)}</text>')
        legend_x += 48 + len(column) * 8
    for index, tick in enumerate(np.linspace(low, high, 5)):
        y = top + plot_height - index / 4 * plot_height
        shown = f"{tick:.0%}" if percent else f"{tick:.3f}"
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5eaf1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#64748b" font-size="12">{shown}</text>')
    for row_index, row in frame.reset_index(drop=True).iterrows():
        base_x = left + row_index * group_width + group_width * 0.15
        for column_index, column in enumerate(columns):
            value = float(row[column])
            value_y = top + plot_height - (value - low) / span * plot_height
            y = min(value_y, zero_y)
            bar_height = max(abs(value_y - zero_y), 1.0)
            parts.append(
                f'<rect x="{base_x + column_index * bar_width:.1f}" y="{y:.1f}" '
                f'width="{bar_width - 2:.1f}" height="{bar_height:.1f}" '
                f'fill="{COLORS[column]}" rx="2"/>')
        parts.append(
            f'<text x="{left + (row_index + 0.5) * group_width:.1f}" y="{height - 18}" '
            f'text-anchor="middle" fill="#64748b" font-size="12">{html.escape(str(row[category]))}</text>')
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="#94a3b8"/>')
    parts.append('</svg>')
    return "".join(parts)


def _write_svg(fragment: str, path: Path) -> None:
    start = fragment.find("<svg")
    end = fragment.rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError(f"SVG fragment missing for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment[start:end + len("</svg>")], encoding="utf-8")


def _table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    rows = ["<tr>" + "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns) + "</tr>"]
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                shown = "—"
            elif column in formats:
                shown = formats[column].format(value)
            else:
                shown = str(value)
            cells.append(f"<td>{html.escape(shown)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='scroll'><table>" + "".join(rows) + "</table></div>"


def generate(output: Path = REPORT) -> Path:
    lake = Lake()
    ridge = lake.read("turnover_buffer_oos_weekly", "v1")
    ridge = ridge.loc[ridge["selected_on_validation"] & ridge["included_in_performance"]].copy()
    challenger = lake.read("challenger_portfolio_oos_weekly", "v1")
    challenger = challenger.loc[challenger["included_in_performance"]].copy()
    common = sorted(set(ridge["asof_date"]) & set(challenger["asof_date"]))
    ridge = ridge.loc[ridge["asof_date"].isin(common)].sort_values("asof_date")
    challenger = challenger.loc[challenger["asof_date"].isin(common)].sort_values("asof_date")
    csi = index_forward_returns("sh.000905", lake).rename(columns={"sh.000905": "CSI 500"})
    weekly_returns = pd.DataFrame({
        "asof_date": common,
        "Ridge": ridge["net_return"].to_numpy(),
        "AI blend": challenger["net_return"].to_numpy(),
    }).merge(csi, on="asof_date", how="left")
    weekly_returns["CSI 500"] = weekly_returns["CSI 500"].fillna(0)
    curve = weekly_returns.copy()
    for column in ("Ridge", "AI blend", "CSI 500"):
        curve[column] = (1 + curve[column]).cumprod()
    drawdown = curve.copy()
    for column in ("Ridge", "AI blend", "CSI 500"):
        drawdown[column] = drawdown[column] / drawdown[column].cummax() - 1

    annual_returns = weekly_returns.copy()
    annual_returns["year"] = pd.to_datetime(annual_returns["asof_date"]).dt.year
    annual_returns = pd.DataFrame([
        {"year": int(year), **{
            column: float((1 + group[column]).prod() - 1)
            for column in ("Ridge", "AI blend", "CSI 500")
        }}
        for year, group in annual_returns.groupby("year", sort=True)
    ])

    costs = lake.read("robustness_cost_grid", "v1")
    cost_display = costs[["model", "cost_bps", "annualized_return", "information_ratio",
                          "maximum_drawdown", "mean_turnover"]].copy()
    cost_display["model"] = cost_display["model"].replace(
        {"ridge": "Ridge", "hgb_ridge_blend": "AI blend"})
    bootstrap = lake.read("robustness_block_bootstrap", "v1")
    bootstrap_display = bootstrap[["metric", "point_estimate", "ci_2_5pct", "ci_97_5pct",
                                   "one_sided_probability_nonpositive"]].copy()
    size = lake.read("size_stress_summary", "v1")
    size = size.loc[size["period"].eq("test_lockbox"),
                    ["model", "minimum_size_percentile", "annualized_return",
                     "information_ratio", "maximum_drawdown", "mean_turnover"]].copy()
    size["model"] = size["model"].replace({"ridge": "Ridge", "hgb_ridge_blend": "AI blend"})
    capacity = lake.read("capacity_execution_summary", "v1")
    capacity = capacity[["aum_cny", "participation_limit", "annualized_return",
                         "mean_order_fill_ratio", "mean_cash_weight", "mean_target_tracking_l1",
                         "mean_impact_cost_bps"]].copy()
    importance = lake.read("model_permutation_importance", "hgb_v1").head(10)
    annual_ic = lake.read("model_challenger_oos_annual", "hgb_v1")
    ridge_ic = lake.read("model_oos_annual", "ridge_v1")
    annual_ic = ridge_ic[["test_year", "mean_rank_ic"]].rename(
        columns={"mean_rank_ic": "Ridge RankIC"}).merge(
        annual_ic[["test_year", "mean_rank_ic"]].rename(columns={"mean_rank_ic": "AI blend RankIC"}),
        on="test_year", how="inner")
    annual_ic = annual_ic.rename(columns={"test_year": "year"})
    execution = lake.read("execution_constraint_summary", "v1")
    selected_cost = cost_display.loc[cost_display["cost_bps"].eq(20)].set_index("model")
    increment = bootstrap.loc[bootstrap["metric"].eq("challenger_minus_ridge_net_return")].iloc[0]
    ridge_excess = bootstrap.loc[bootstrap["metric"].eq("ridge_net_excess_vs_CSI500")].iloc[0]
    cards = [
        ("共同历史样本外周", f"{len(common)}", "2021-01 至 2026-09"),
        ("Ridge 年化净收益", f"{selected_cost.loc['Ridge','annualized_return']:.1%}", "20bp，缓冲后"),
        ("Ridge 相对 CSI500 区间", f"[{ridge_excess.ci_2_5pct:.1%}, {ridge_excess.ci_97_5pct:.1%}]", "13周块 bootstrap，95%"),
        ("AI 相对增量区间", f"[{increment.ci_2_5pct:.1%}, {increment.ci_97_5pct:.1%}]", "13周块 bootstrap，95%"),
    ]
    cards_html = "".join(
        f'<section class="card"><small>{label}</small><strong>{value}</strong><span>{note}</span></section>'
        for label, value, note in cards)
    equity_svg = _multi_line(curve, ["Ridge", "AI blend", "CSI 500"])
    drawdown_svg = _multi_line(drawdown, ["Ridge", "AI blend", "CSI 500"], percent=True)
    annual_return_svg = _grouped_bars(
        annual_returns, "year", ["Ridge", "AI blend", "CSI 500"], percent=True)
    # Reuse the model palette for the human-readable RankIC labels.
    COLORS.update({"Ridge RankIC": COLORS["Ridge"], "AI blend RankIC": COLORS["AI blend"]})
    rank_ic_svg = _grouped_bars(
        annual_ic, "year", ["Ridge RankIC", "AI blend RankIC"], percent=False)
    _write_svg(equity_svg, FIGURES / "oos_equity_curve.svg")
    _write_svg(drawdown_svg, FIGURES / "oos_drawdown.svg")
    _write_svg(annual_return_svg, FIGURES / "annual_returns.svg")
    _write_svg(rank_ic_svg, FIGURES / "annual_rank_ic.svg")

    TABLES.mkdir(parents=True, exist_ok=True)
    public_tables = {
        "cost_sensitivity": cost_display,
        "bootstrap_intervals": bootstrap_display,
        "size_stress": size,
        "capacity_execution": capacity,
        "annual_rank_ic": annual_ic,
        "annual_returns": annual_returns,
    }
    for name, frame in public_tables.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8")
    public_results = {
        "sample": {"common_weeks": len(common), "start": common[0], "end": common[-1]},
        "ridge": {
            "annualized_return_after_20bp": float(selected_cost.loc["Ridge", "annualized_return"]),
            "excess_vs_csi500_annualized_arithmetic_ci_95": [
                float(ridge_excess.ci_2_5pct), float(ridge_excess.ci_97_5pct)],
        },
        "hgb_ridge_blend": {
            "annualized_return_after_20bp": float(selected_cost.loc["AI blend", "annualized_return"]),
            "increment_vs_ridge_annualized_ci_95": [
                float(increment.ci_2_5pct), float(increment.ci_97_5pct)],
        },
        "interpretation": (
            "Ridge shows positive historical OOS evidence after costs. The nonlinear blend has an "
            "economic uplift signal, but its incremental confidence interval crosses zero."
        ),
        "limitations": [
            "Historical OOS results have been inspected during development and are not a pristine live lockbox.",
            "No point-in-time industry neutralization.",
            "The universe is a historical A-share proxy, not official historical CSI membership.",
            "Market impact is scenario-based and not calibrated to broker fills.",
        ],
    }
    PUBLIC_RESULTS.write_text(
        json.dumps(public_results, ensure_ascii=False, indent=2), encoding="utf-8")

    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>A股点时多因子与机器学习研究面板</title><style>
:root{{--ink:#142033;--muted:#64748b;--paper:#f4f7fb;--line:#dde5ef}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:28px}}h1{{margin:0}}h2{{margin:0 0 14px;font-size:20px}}p{{line-height:1.65}}
.muted,small{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}
.card,.panel{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 2px 10px #94a3b815}}
.card strong{{display:block;font-size:25px;margin:9px 0}}.card span{{font-size:12px;color:var(--muted)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}.panel{{margin:16px 0}}svg{{width:100%;height:auto}}
.legend{{display:flex;gap:18px;font-size:13px;color:var(--muted)}}.legend i{{display:inline-block;width:18px;height:3px;margin:0 6px 3px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 9px;border-bottom:1px solid #edf1f6;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}.finding{{border-left:4px solid #f97316;padding-left:14px}}
code{{background:#edf2f7;padding:2px 5px;border-radius:4px}}@media(max-width:850px){{.grid,.two{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>A 股点时多因子与机器学习研究面板</h1><p class="muted">点时数据｜t+1 可执行标签｜Purged walk-forward｜成本、容量、规模与涨跌停压力测试｜生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
<div class="grid">{cards_html}</div>
<section class="panel"><h2>历史样本外累计净值（共同日期，20bp）</h2>{equity_svg}<p class="muted">CSI 500 使用与策略相同的 t+1 收盘入场、随后 5 日持有标签；图中不含指数交易费用。</p></section>
<section class="panel"><h2>回撤路径</h2>{drawdown_svg}</section>
<section class="panel"><h2>年度收益</h2>{annual_return_svg}</section>
<div class="two"><section class="panel"><h2>年度 RankIC</h2>{rank_ic_svg}</section>
<section class="panel"><h2>AI 模型验证期置换重要性</h2>{_table(importance[['factor','mean_rank_ic_drop','ci_2_5pct','ci_97_5pct']], {'mean_rank_ic_drop':'{:.4f}','ci_2_5pct':'{:.4f}','ci_97_5pct':'{:.4f}'})}</section></div>
<section class="panel"><h2>成本敏感性</h2>{_table(cost_display, {'cost_bps':'{:.0f}','annualized_return':'{:.1%}','information_ratio':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}'})}</section>
<section class="panel"><h2>配对块 Bootstrap</h2>{_table(bootstrap_display, {'point_estimate':'{:.3%}','ci_2_5pct':'{:.3%}','ci_97_5pct':'{:.3%}','one_sided_probability_nonpositive':'{:.3f}'})}</section>
<section class="panel"><h2>小市值剔除压力测试</h2>{_table(size, {'minimum_size_percentile':'{:.0%}','annualized_return':'{:.1%}','information_ratio':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}'})}</section>
<section class="panel"><h2>动态容量执行（Ridge 缓冲组合）</h2>{_table(capacity, {'aum_cny':'{:,.0f}','participation_limit':'{:.0%}','annualized_return':'{:.1%}','mean_order_fill_ratio':'{:.1%}','mean_cash_weight':'{:.1%}','mean_target_tracking_l1':'{:.3f}','mean_impact_cost_bps':'{:.2f}'})}</section>
<section class="panel"><h2>t+1 停牌/涨跌停执行约束</h2>{_table(execution, {'annualized_return':'{:.1%}','maximum_drawdown':'{:.1%}','mean_blocked_buy_weight':'{:.3%}','mean_blocked_sell_weight':'{:.3%}','mean_cash_weight':'{:.3%}','mean_target_tracking_l1':'{:.3f}'})}</section>
<section class="panel finding"><h2>最终 finding</h2><p>Ridge 多因子基线在各历史样本外年度均保持正 RankIC；20bp 成本后，相对 CSI 500 的年化算术超额块 bootstrap 95% 区间仍高于 0，这是当前最可靠的正向证据。非线性模型在验证集把 RankIC 从 0.0862 提高到约 0.0994，但共同历史样本外日期上的平均 RankIC 并未超过 Ridge。AI blend 的年化收益高约 3.2 个百分点，但增量区间跨 0，因此只能表述为经济增量迹象。结果仍受行业暴露、股票池代理和冲击模型未校准等限制；2021–2026 已在开发中被检查，应称为历史样本外，不再称作从未查看的 lockbox。</p></section>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    print(generate())


if __name__ == "__main__":
    main()
