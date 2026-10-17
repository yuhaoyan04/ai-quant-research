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
COLORS = {"Ridge": "#4f8cff", "AI challenger": "#f97316", "CSI 500": "#64748b"}


def _increment_interpretation(ci_low: float, ci_high: float) -> str:
    prefix = (
        "Ridge shows positive historical OOS evidence after costs. "
        "The validation-selected nonlinear challenger "
    )
    if ci_low > 0:
        return (
            prefix
            + "has a positive return increment whose block-bootstrap confidence interval "
            "is above zero in this rerun; treat it as historical evidence, not a pristine "
            "production lockbox result."
        )
    if ci_high < 0:
        return (
            prefix
            + "has a negative return increment whose block-bootstrap confidence interval "
            "is below zero in this rerun."
        )
    return (
        prefix
        + "has an economic uplift signal, but its incremental confidence interval crosses zero."
    )


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
    challenger_selection = lake.read("model_challenger_selection", "hgb_v1").iloc[0]
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
        "AI challenger": challenger["net_return"].to_numpy(),
    }).merge(csi, on="asof_date", how="left")
    weekly_returns["CSI 500"] = weekly_returns["CSI 500"].fillna(0)
    curve = weekly_returns.copy()
    for column in ("Ridge", "AI challenger", "CSI 500"):
        curve[column] = (1 + curve[column]).cumprod()
    drawdown = curve.copy()
    for column in ("Ridge", "AI challenger", "CSI 500"):
        drawdown[column] = drawdown[column] / drawdown[column].cummax() - 1

    annual_returns = weekly_returns.copy()
    annual_returns["year"] = pd.to_datetime(annual_returns["asof_date"]).dt.year
    annual_returns = pd.DataFrame([
        {"year": int(year), **{
            column: float((1 + group[column]).prod() - 1)
            for column in ("Ridge", "AI challenger", "CSI 500")
        }}
        for year, group in annual_returns.groupby("year", sort=True)
    ])

    costs = lake.read("robustness_cost_grid", "v1")
    cost_display = costs[["model", "cost_bps", "annualized_return", "information_ratio",
                          "maximum_drawdown", "mean_turnover"]].copy()
    cost_display["model"] = cost_display["model"].replace(
        {"ridge": "Ridge", "hgb_ridge_blend": "AI challenger"})
    bootstrap = lake.read("robustness_block_bootstrap", "v1")
    bootstrap_display = bootstrap[["metric", "point_estimate", "ci_2_5pct", "ci_97_5pct",
                                   "one_sided_probability_nonpositive"]].copy()
    size = lake.read("size_stress_summary", "v1")
    size = size.loc[size["period"].eq("test_lockbox"),
                    ["model", "minimum_size_percentile", "annualized_return",
                     "information_ratio", "maximum_drawdown", "mean_turnover"]].copy()
    size["model"] = size["model"].replace(
        {"ridge": "Ridge", "hgb_ridge_blend": "AI challenger"}
    )
    capacity = lake.read("capacity_execution_summary", "v1")
    capacity = capacity[["aum_cny", "participation_limit", "annualized_return",
                         "mean_order_fill_ratio", "mean_cash_weight", "mean_target_tracking_l1",
                         "mean_impact_cost_bps"]].copy()
    importance = lake.read("model_permutation_importance", "hgb_v1").head(10)
    annual_ic = lake.read("model_challenger_oos_annual", "hgb_v1")
    ridge_ic = lake.read("model_oos_annual", "ridge_v1")
    annual_ic = ridge_ic[["test_year", "mean_rank_ic"]].rename(
        columns={"mean_rank_ic": "Ridge RankIC"}).merge(
        annual_ic[["test_year", "mean_rank_ic"]].rename(
            columns={"mean_rank_ic": "AI challenger RankIC"}
        ),
        on="test_year", how="inner")
    annual_ic = annual_ic.rename(columns={"test_year": "year"})
    execution = lake.read("execution_constraint_summary", "v1")
    rf_validation_execution = pd.concat([
        lake.read("rf_daily_ideal_summary", "validation_v1"),
        lake.read("rf_daily_retry_summary", "validation_v1"),
    ], ignore_index=True)
    rf_validation_execution.insert(0, "period", "validation_2016_2020")
    rf_oos_execution = pd.concat([
        lake.read("rf_daily_ideal_summary", "historical_oos_v1"),
        lake.read("rf_daily_retry_summary", "historical_oos_v1"),
    ], ignore_index=True)
    rf_oos_execution.insert(0, "period", "historical_oos_2021_2026")
    rf_execution = pd.concat(
        [rf_validation_execution, rf_oos_execution], ignore_index=True
    )[["period", "scenario", "periods", "annualized_return", "sharpe_zero_rate",
       "maximum_drawdown", "mean_turnover"]]
    rf_oos_constrained = rf_oos_execution.loc[
        rf_oos_execution["scenario"].eq("daily_retry_limit_and_suspension_constrained")
    ].iloc[0]
    rf_oos_ideal = rf_oos_execution.loc[
        rf_oos_execution["scenario"].eq("nonoverlap_all_fills_upper_bound")
    ].iloc[0]
    rf_validation_constrained = rf_validation_execution.loc[
        rf_validation_execution["scenario"].eq(
            "daily_retry_limit_and_suspension_constrained"
        )
    ].iloc[0]
    rf_validation_ideal = rf_validation_execution.loc[
        rf_validation_execution["scenario"].eq("nonoverlap_all_fills_upper_bound")
    ].iloc[0]
    factor_neutralization = lake.read("factor_neutralization_conclusions", "v1")
    industry_exposure_index = lake.read("pit_industry_exposure_index", "v1")
    factor_neutralization_display = factor_neutralization[[
        "factor", "raw_mean_oriented_rank_ic", "neutral_mean_oriented_rank_ic",
        "rank_ic_delta", "rank_ic_delta_ci_low", "rank_ic_delta_ci_high",
        "rank_ic_effect",
    ]].copy()
    weakened_factors = factor_neutralization.loc[
        factor_neutralization["rank_ic_effect"].eq("weakened"), "factor"
    ].tolist()
    strengthened_factors = factor_neutralization.loc[
        factor_neutralization["rank_ic_effect"].eq("strengthened"), "factor"
    ].tolist()
    factor_tear_sheet = lake.read("factor_tear_sheet_summary", "v1")
    factor_tear_bootstrap = lake.read("factor_tear_sheet_bootstrap", "v1")
    factor_tear_intervals = factor_tear_bootstrap.loc[
        factor_tear_bootstrap["pipeline"].eq("industry_size_neutralized")
        & factor_tear_bootstrap["metric"].eq("gross_top_minus_bottom"),
        ["factor", "ci_2_5pct", "ci_97_5pct"],
    ]
    factor_tear_sheet_display = factor_tear_sheet.loc[
        factor_tear_sheet["pipeline"].eq("industry_size_neutralized"), [
            "factor", "mean_oriented_rank_ic", "annualized_icir",
            "bucket_return_spearman", "annualized_gross_top_minus_bottom",
            "annualized_net_top_minus_bottom", "mean_long_short_turnover",
        ],
    ].merge(factor_tear_intervals, on="factor", validate="one_to_one").sort_values(
        "mean_oriented_rank_ic", ascending=False
    )
    significant_factor_spreads = factor_tear_sheet_display.loc[
        factor_tear_sheet_display["ci_2_5pct"].gt(0), "factor"
    ].tolist()
    model_neutralization = lake.read("model_neutralization_conclusions", "v1")
    model_neutralization_display = model_neutralization.loc[
        model_neutralization["pipeline"].eq("industry_size_neutralized"), [
            "model", "raw_mean_rank_ic", "neutral_mean_rank_ic",
            "rank_ic_delta", "rank_ic_ci_low", "rank_ic_ci_high",
            "raw_annualized_return", "neutral_annualized_return",
            "annualized_arithmetic_return_delta", "return_delta_ci_low",
            "return_delta_ci_high", "raw_industry_active_weight_l1",
            "neutral_industry_active_weight_l1",
        ],
    ].copy()
    constrained_portfolio = lake.read("constrained_portfolio_oos_summary", "v1")
    constrained_portfolio_display = constrained_portfolio[[
        "model", "strategy", "annualized_return", "annualized_volatility",
        "sharpe_zero_rate", "maximum_drawdown", "mean_turnover",
        "mean_abs_portfolio_size_z", "mean_industry_active_weight_l1",
        "mean_maximum_abs_industry_active_weight", "turnover_relaxation_fraction",
    ]].copy()
    constrained_daily_execution = lake.read(
        "constrained_daily_execution_summary", "v1"
    )
    constrained_daily_execution_display = constrained_daily_execution[[
        "model", "alpha_pipeline", "scenario", "periods", "annualized_return",
        "annualized_volatility", "sharpe_zero_rate", "maximum_drawdown",
        "mean_turnover", "mean_blocked_buy_weight", "mean_blocked_sell_weight",
        "mean_target_tracking_l1", "mean_cash_weight",
    ]].copy()
    risk_attribution = lake.read("risk_model_attribution_summary", "v1")
    risk_attribution_display = risk_attribution[[
        "model", "alpha_pipeline", "annualized_return",
        "annualized_market_contribution_arithmetic",
        "annualized_industry_active_contribution_arithmetic",
        "annualized_style_contribution_arithmetic",
        "annualized_specific_selection_contribution_arithmetic",
        "annualized_transaction_cost_arithmetic",
        "mean_forecast_annualized_volatility",
        "mean_forecast_annualized_tracking_error",
    ]].copy()
    capm = lake.read("capm_summary", "v1")
    capm_display = capm.loc[
        capm["benchmark"].eq("CSI500")
        & capm["scenario"].eq("daily_retry_limit_and_suspension_constrained"), [
            "model", "alpha_pipeline", "observations",
            "annualized_alpha_arithmetic", "annualized_alpha_ci_low",
            "annualized_alpha_ci_high", "alpha_hac_t_stat", "beta",
            "beta_ci_low", "beta_ci_high", "r_squared",
        ],
    ].copy().sort_values(["model", "alpha_pipeline"])
    style_factor_diagnostics = lake.read("style_factor_diagnostics", "v1")
    style_regression = lake.read("style_factor_regression_summary", "v1")
    style_regression_display = style_regression.loc[
        style_regression["scenario"].eq(
            "daily_retry_limit_and_suspension_constrained"
        ), [
            "model", "alpha_pipeline", "annualized_alpha_arithmetic",
            "annualized_alpha_ci_low", "annualized_alpha_ci_high",
            "alpha_hac_t_stat", "r_squared", "loading_market", "loading_size",
            "loading_beta", "loading_volatility", "loading_liquidity",
            "loading_value", "loading_momentum",
        ],
    ].copy().sort_values(["model", "alpha_pipeline"])
    style_regression_display = style_regression_display.merge(
        capm_display[["model", "alpha_pipeline", "annualized_alpha_arithmetic"]].rename(
            columns={"annualized_alpha_arithmetic": "capm_annualized_alpha"}
        ), on=["model", "alpha_pipeline"], validate="one_to_one",
    )
    risk_aware_columns = [
        "model", "alpha_pipeline", "scenario", "annualized_return",
        "annualized_volatility", "sharpe_zero_rate", "maximum_drawdown",
        "mean_turnover", "mean_forecast_annualized_volatility",
        "mean_forecast_annualized_tracking_error", "mean_risky_weight",
        "mean_alpha_utility_retention", "turnover_relaxation_fraction",
    ]
    risk_aware = lake.read("risk_aware_portfolio_summary", "v2")
    risk_aware_display = (
        risk_aware[risk_aware_columns].copy()
        if not risk_aware.empty else pd.DataFrame(columns=risk_aware_columns)
    )
    portfolio_sensitivity_display = lake.read("portfolio_sensitivity_summary", "v1")
    execution_cost_stress_display = lake.read("execution_cost_stress", "v1")[[
        "model", "alpha_pipeline", "cost_bps", "annualized_return",
        "sharpe_zero_rate", "maximum_drawdown",
    ]]
    hgb_surface_display = lake.read("hgb_hyperparameter_summary", "v1")[[
        "candidate", "max_leaf_nodes", "l2_regularization", "mean_rank_ic",
        "annualized_icir", "portfolio_annualized_return",
        "portfolio_sharpe_zero_rate",
    ]].sort_values("mean_rank_ic", ascending=False)
    factor_distribution_display = lake.read("factor_distribution_summary", "v1")
    conclusion_decisions_display = lake.read("research_conclusion_decisions", "v1")[[
        "question", "verdict", "evidence", "scope_limit",
    ]]
    factor_multiplicity_display = lake.read("research_factor_multiplicity", "v1")[[
        "factor", "point_estimate", "ci_2_5pct", "ci_97_5pct",
        "bh_fdr_adjusted_p", "survives_bh_fdr",
    ]]
    lifecycle_display = lake.read("research_lifecycle_audit", "v1")
    minimum_industry_coverage = float(
        industry_exposure_index["eligible_industry_coverage"].min()
    )
    selected_cost = cost_display.loc[cost_display["cost_bps"].eq(20)].set_index("model")
    increment = bootstrap.loc[bootstrap["metric"].eq("challenger_minus_ridge_net_return")].iloc[0]
    ridge_excess = bootstrap.loc[bootstrap["metric"].eq("ridge_net_excess_vs_CSI500")].iloc[0]
    final_daily = constrained_daily_execution_display.loc[
        constrained_daily_execution_display["scenario"].eq(
            "daily_retry_limit_and_suspension_constrained"
        )
    ]
    hgb_raw = final_daily.loc[
        final_daily["model"].eq("hgb_shallow15_locked")
        & final_daily["alpha_pipeline"].eq("raw")
    ].iloc[0]
    hgb_neutral = final_daily.loc[
        final_daily["model"].eq("hgb_shallow15_locked")
        & final_daily["alpha_pipeline"].eq("industry_size_neutralized")
    ].iloc[0]
    hgb_style_raw = style_regression_display.loc[
        style_regression_display["model"].eq("hgb_shallow15_locked")
        & style_regression_display["alpha_pipeline"].eq("raw")
    ].iloc[0]
    cards = [
        ("最终日频净值日", f"{int(hgb_raw.periods):,}", "2021–2026 historical OOS"),
        ("HGB raw 可成交年化", f"{hgb_raw.annualized_return:.1%}", "20bp；停牌/涨跌停重试"),
        ("HGB neutral 可成交年化", f"{hgb_neutral.annualized_return:.1%}", "行业+规模中性化"),
        ("HGB 六风格残余 Alpha", f"{hgb_style_raw.annualized_alpha_arithmetic:.1%}",
         f"95% [{hgb_style_raw.annualized_alpha_ci_low:.1%}, {hgb_style_raw.annualized_alpha_ci_high:.1%}]"),
        ("冻结实验", str(sum(
            (path / "FROZEN.json").exists()
            for path in (ROOT / "reports" / "experiments").iterdir()
            if path.is_dir()
        )), "失败与被替代版本同样保留"),
    ]
    cards_html = "".join(
        f'<section class="card"><small>{label}</small><strong>{value}</strong><span>{note}</span></section>'
        for label, value, note in cards)
    equity_svg = _multi_line(curve, ["Ridge", "AI challenger", "CSI 500"])
    drawdown_svg = _multi_line(
        drawdown, ["Ridge", "AI challenger", "CSI 500"], percent=True
    )
    annual_return_svg = _grouped_bars(
        annual_returns, "year", ["Ridge", "AI challenger", "CSI 500"], percent=True)
    # Reuse the model palette for the human-readable RankIC labels.
    COLORS.update({
        "Ridge RankIC": COLORS["Ridge"],
        "AI challenger RankIC": COLORS["AI challenger"],
    })
    rank_ic_svg = _grouped_bars(
        annual_ic, "year", ["Ridge RankIC", "AI challenger RankIC"], percent=False)
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
        "rf_execution_forensics": rf_execution,
        "factor_neutralization": factor_neutralization_display,
        "factor_tear_sheet": factor_tear_sheet_display,
        "model_portfolio_neutralization": model_neutralization_display,
        "constrained_portfolio_neutralization": constrained_portfolio_display,
        "constrained_portfolio_daily_execution": constrained_daily_execution_display,
        "multi_factor_risk_attribution": risk_attribution_display,
        "capm_attribution": capm_display,
        "self_financing_style_factors": style_factor_diagnostics,
        "style_factor_attribution": style_regression_display,
        "risk_aware_portfolio_optimization": risk_aware_display,
        "portfolio_sensitivity": portfolio_sensitivity_display,
        "execution_cost_stress": execution_cost_stress_display,
        "hgb_hyperparameter_stability": hgb_surface_display,
        "factor_distribution_summary": factor_distribution_display,
        "research_conclusion_decisions": conclusion_decisions_display,
        "factor_multiplicity_control": factor_multiplicity_display,
        "research_lifecycle_audit": lifecycle_display,
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
            "schema_note": "legacy key retained for backward compatibility",
            "selected_tree_candidate": str(challenger_selection["selected_tree_candidate"]),
            "selected_tree_weight": float(challenger_selection["selected_tree_weight"]),
            "ridge_weight": float(challenger_selection["ridge_weight"]),
            "annualized_return_after_20bp": float(
                selected_cost.loc["AI challenger", "annualized_return"]
            ),
            "increment_vs_ridge_annualized_ci_95": [
                float(increment.ci_2_5pct), float(increment.ci_97_5pct)],
        },
        "random_forest": {
            "nonoverlap_ideal_oos_annualized_return": float(
                rf_oos_execution.loc[
                    rf_oos_execution["scenario"].eq("nonoverlap_all_fills_upper_bound"),
                    "annualized_return",
                ].iloc[0]
            ),
            "daily_retry_constrained_oos_annualized_return": float(
                rf_oos_constrained.annualized_return
            ),
            "daily_retry_constrained_oos_sharpe": float(rf_oos_constrained.sharpe_zero_rate),
            "daily_retry_constrained_oos_maximum_drawdown": float(
                rf_oos_constrained.maximum_drawdown
            ),
            "warning": (
                "The old 69.98%, 19.09%, and 24.19% figures are superseded: overlapping "
                "holding clocks and incomplete blocked-order handling were corrected."
            ),
        },
        "pit_factor_neutralization": {
            "minimum_annual_industry_coverage": minimum_industry_coverage,
            "historical_oos_weeks": 293,
            "significantly_strengthened_factors": strengthened_factors,
            "significantly_weakened_factors": weakened_factors,
            "method": "paired 13-week circular block bootstrap on a common exposure-complete sample",
        },
        "factor_tear_sheet": {
            "historical_oos_weeks": 293,
            "pipelines": ["raw", "industry_size_neutralized"],
            "significant_neutralized_top_bottom_factors": significant_factor_spreads,
            "cost_assumption_bps_per_leg_turnover": 20.0,
            "regime_definition": (
                "signal-date CSI500 trailing-60-day trend and expanding-median volatility"
            ),
            "factors": {
                str(row.factor): {
                    "mean_oriented_rank_ic": float(row.mean_oriented_rank_ic),
                    "annualized_icir": float(row.annualized_icir),
                    "bucket_return_spearman": float(row.bucket_return_spearman),
                    "annualized_gross_top_minus_bottom": float(
                        row.annualized_gross_top_minus_bottom
                    ),
                    "gross_spread_ci_95": [float(row.ci_2_5pct), float(row.ci_97_5pct)],
                    "annualized_net_top_minus_bottom": float(
                        row.annualized_net_top_minus_bottom
                    ),
                    "mean_long_short_turnover": float(row.mean_long_short_turnover),
                }
                for row in factor_tear_sheet_display.itertuples()
            },
        },
        "model_portfolio_neutralization": {
            "historical_oos_weeks": 293,
            "comparison": "industry+size neutralized minus raw on identical PIT samples",
            "models": {
                str(row.model): {
                    "raw_mean_rank_ic": float(row.raw_mean_rank_ic),
                    "neutral_mean_rank_ic": float(row.neutral_mean_rank_ic),
                    "rank_ic_delta_ci_95": [
                        float(row.rank_ic_ci_low), float(row.rank_ic_ci_high)
                    ],
                    "raw_annualized_return": float(row.raw_annualized_return),
                    "neutral_annualized_return": float(row.neutral_annualized_return),
                    "annualized_arithmetic_return_delta_ci_95": [
                        float(row.return_delta_ci_low), float(row.return_delta_ci_high)
                    ],
                    "raw_industry_active_weight_l1": float(
                        row.raw_industry_active_weight_l1
                    ),
                    "neutral_industry_active_weight_l1": float(
                        row.neutral_industry_active_weight_l1
                    ),
                }
                for row in model_neutralization_display.itertuples()
            },
            "portfolio_industry_constraint_applied": False,
        },
        "constrained_portfolio_neutralization": {
            "historical_oos_weeks": 293,
            "constraints": {
                "maximum_stock_weight": 0.005,
                "maximum_abs_industry_active_weight": 0.03,
                "maximum_abs_size_z": 0.10,
                "weekly_turnover_target": 0.30,
            },
            "strategies": {
                f"{row.model}__{row.strategy}": {
                    "annualized_return": float(row.annualized_return),
                    "annualized_volatility": float(row.annualized_volatility),
                    "sharpe_zero_rate": float(row.sharpe_zero_rate),
                    "maximum_drawdown": float(row.maximum_drawdown),
                    "mean_turnover": float(row.mean_turnover),
                    "mean_abs_portfolio_size_z": float(row.mean_abs_portfolio_size_z),
                    "mean_industry_active_weight_l1": float(
                        row.mean_industry_active_weight_l1
                    ),
                    "turnover_relaxation_fraction": float(
                        row.turnover_relaxation_fraction
                    ),
                }
                for row in constrained_portfolio_display.itertuples()
            },
            "execution_assumption": "ideal complete fill with 20bp round-trip cost",
        },
        "constrained_portfolio_daily_execution": {
            "daily_periods": 1387,
            "scenarios": {
                f"{row.model}__{row.alpha_pipeline}__{row.scenario}": {
                    "annualized_return": float(row.annualized_return),
                    "annualized_volatility": float(row.annualized_volatility),
                    "sharpe_zero_rate": float(row.sharpe_zero_rate),
                    "maximum_drawdown": float(row.maximum_drawdown),
                    "mean_blocked_buy_weight": float(row.mean_blocked_buy_weight),
                    "mean_blocked_sell_weight": float(row.mean_blocked_sell_weight),
                    "mean_target_tracking_l1": float(row.mean_target_tracking_l1),
                    "mean_cash_weight": float(row.mean_cash_weight),
                }
                for row in constrained_daily_execution_display.itertuples()
            },
            "execution_convention": (
                "returns accrue before close; blocked share orders retry daily; "
                "new weekly targets cancel stale orders"
            ),
        },
        "multi_factor_risk_attribution": {
            "benchmark": "equal-weight PIT exposure-complete universe",
            "styles": ["size", "beta", "volatility", "liquidity", "value", "momentum"],
            "models": {
                f"{row.model}__{row.alpha_pipeline}": {
                    "annualized_return": float(row.annualized_return),
                    "annualized_market_contribution_arithmetic": float(
                        row.annualized_market_contribution_arithmetic
                    ),
                    "annualized_industry_active_contribution_arithmetic": float(
                        row.annualized_industry_active_contribution_arithmetic
                    ),
                    "annualized_style_contribution_arithmetic": float(
                        row.annualized_style_contribution_arithmetic
                    ),
                    "annualized_specific_selection_contribution_arithmetic": float(
                        row.annualized_specific_selection_contribution_arithmetic
                    ),
                    "annualized_transaction_cost_arithmetic": float(
                        row.annualized_transaction_cost_arithmetic
                    ),
                    "mean_forecast_annualized_volatility": float(
                        row.mean_forecast_annualized_volatility
                    ),
                    "mean_forecast_annualized_tracking_error": float(
                        row.mean_forecast_annualized_tracking_error
                    ),
                }
                for row in risk_attribution_display.itertuples()
            },
            "scope": "weekly five-day label attribution; daily execution reported separately",
        },
        "capm_attribution": {
            "primary_benchmark": "CSI500",
            "risk_free_rate_convention": "zero_daily_rate",
            "benchmark_price_returns_exclude_dividends": True,
            "newey_west_lags": 5,
            "strategies": {
                f"{row.model}__{row.alpha_pipeline}": {
                    "annualized_alpha_arithmetic": float(
                        row.annualized_alpha_arithmetic
                    ),
                    "annualized_alpha_ci_95": [
                        float(row.annualized_alpha_ci_low),
                        float(row.annualized_alpha_ci_high),
                    ],
                    "alpha_hac_t_stat": float(row.alpha_hac_t_stat),
                    "beta": float(row.beta),
                    "beta_ci_95": [float(row.beta_ci_low), float(row.beta_ci_high)],
                    "r_squared": float(row.r_squared),
                }
                for row in capm_display.itertuples()
            },
        },
        "style_factor_attribution": {
            "factors": [
                {
                    "factor": str(row.factor),
                    "orientation": str(row.orientation),
                    "annualized_mean_return_arithmetic": float(
                        row.annualized_mean_return_arithmetic
                    ),
                    "annualized_volatility": float(row.annualized_volatility),
                    "mean_return_hac_t_stat": float(row.mean_return_hac_t_stat),
                    "mean_weekly_two_leg_turnover": float(
                        row.mean_weekly_two_leg_turnover
                    ),
                }
                for row in style_factor_diagnostics.itertuples()
            ],
            "factor_transaction_cost_bps": 0.0,
            "strategies": {
                f"{row.model}__{row.alpha_pipeline}": {
                    "capm_annualized_alpha": float(row.capm_annualized_alpha),
                    "multifactor_annualized_alpha": float(
                        row.annualized_alpha_arithmetic
                    ),
                    "multifactor_alpha_ci_95": [
                        float(row.annualized_alpha_ci_low),
                        float(row.annualized_alpha_ci_high),
                    ],
                    "alpha_hac_t_stat": float(row.alpha_hac_t_stat),
                    "r_squared": float(row.r_squared),
                    "loadings": {
                        factor: float(getattr(row, f"loading_{factor}"))
                        for factor in (
                            "market", "size", "beta", "volatility",
                            "liquidity", "value", "momentum",
                        )
                    },
                }
                for row in style_regression_display.itertuples()
            },
            "interpretation": (
                "All executable-strategy residual alpha confidence intervals cross zero; "
                "CAPM alpha is largely absorbed by local style factors."
            ),
        },
        "risk_aware_portfolio_optimization": {
            "policy_adopted": False,
            "reason": (
                "v1 materially reduced forecast tracking error but its paired "
                "historical-OOS return delta was significantly negative"
            ),
            "scenarios": {
                f"{row.model}__{row.alpha_pipeline}__{row.scenario}": {
                    "annualized_return": float(row.annualized_return),
                    "annualized_volatility": float(row.annualized_volatility),
                    "sharpe_zero_rate": float(row.sharpe_zero_rate),
                    "maximum_drawdown": float(row.maximum_drawdown),
                    "mean_forecast_annualized_volatility": float(
                        row.mean_forecast_annualized_volatility
                    ),
                    "mean_forecast_annualized_tracking_error": float(
                        row.mean_forecast_annualized_tracking_error
                    ),
                    "mean_risky_weight": float(row.mean_risky_weight),
                }
                for row in risk_aware_display.itertuples()
            },
        },
        "portfolio_sensitivity": {
            "research_status": "retrospective_diagnostic_not_a_new_oos_selection",
            "top_n": [50, 100, 200, 500],
            "weighting": ["equal_weight", "inverse_volatility"],
            "execution_cost_stress_bps": [0, 20, 50, 100],
        },
        "hgb_hyperparameter_stability": {
            "evaluation_period": "validation_2016_2020_only",
            "surface_points": int(len(hgb_surface_display)),
            "minimum_mean_rank_ic": float(hgb_surface_display["mean_rank_ic"].min()),
            "maximum_mean_rank_ic": float(hgb_surface_display["mean_rank_ic"].max()),
            "best_candidate_is_not_promoted": True,
        },
        "research_conclusion_audit": {
            "decisions": conclusion_decisions_display.to_dict(orient="records"),
            "neutralized_factor_spreads_surviving_bh_fdr_5pct": int(
                factor_multiplicity_display["survives_bh_fdr"].sum()
            ),
            "historical_research_chain_complete": True,
            "ready_for_live_capital": False,
        },
        "interpretation": _increment_interpretation(
            float(increment.ci_2_5pct), float(increment.ci_97_5pct)
        ),
        "limitations": [
            "Historical OOS results have been inspected during development and are not a pristine live lockbox.",
            "PIT factor neutralization uses monthly historical CSRC broad sections, not Shenwan subindustries.",
            "Constrained portfolios have a daily price-limit and suspension audit, but intraday queues, IPO no-limit windows, and broker-calibrated impact remain unmodeled.",
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
<section class="panel finding"><h2>五项最终判定</h2>{_table(conclusion_decisions_display)}<p class="muted">研究回测链路已经闭环；paper/live 是下一阶段，不能用历史回测替代。</p></section>
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
<section class="panel"><h2>RF 非重叠日频执行取证</h2>{_table(rf_execution, {'annualized_return':'{:.1%}','sharpe_zero_rate':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}'})}<p class="muted">所有场景使用唯一日频净值时钟与 20bp 成本。验证期从理想全成交上界 {rf_validation_ideal.annualized_return:.2%} 降至日频重试约束 {rf_validation_constrained.annualized_return:.2%}；historical OOS 从 {rf_oos_ideal.annualized_return:.2%} 降至 {rf_oos_constrained.annualized_return:.2%}。</p></section>
<section class="panel"><h2>PIT 行业与规模中性化</h2>{_table(factor_neutralization_display, {'raw_mean_oriented_rank_ic':'{:.4f}','neutral_mean_oriented_rank_ic':'{:.4f}','rank_ic_delta':'{:.4f}','rank_ic_delta_ci_low':'{:.4f}','rank_ic_delta_ci_high':'{:.4f}'})}<p class="muted">历史证监会宽行业最低年度覆盖率 {minimum_industry_coverage:.2%}；因子方向仅由训练期确定。行业+规模中性化后无因子 RankIC 显著增强，显著减弱的因子为 {html.escape(', '.join(weakened_factors))}。</p></section>
<section class="panel"><h2>中性化单因子 Tear Sheet</h2>{_table(factor_tear_sheet_display, {'mean_oriented_rank_ic':'{:.4f}','annualized_icir':'{:.2f}','bucket_return_spearman':'{:.2f}','annualized_gross_top_minus_bottom':'{:.1%}','annualized_net_top_minus_bottom':'{:.1%}','mean_long_short_turnover':'{:.3f}','ci_2_5pct':'{:.1%}','ci_97_5pct':'{:.1%}'})}<p class="muted">293 周十分位等权诊断；方向仅由训练期锁定。15 个因子中 14 个毛价差通过 BH-FDR 5%，盈利收益率除外；该表未建模借券与完整成交，不是可交易多空业绩。</p></section>
<section class="panel"><h2>单因子多重检验</h2>{_table(factor_multiplicity_display, {'point_estimate':'{:.1%}','ci_2_5pct':'{:.1%}','ci_97_5pct':'{:.1%}','bh_fdr_adjusted_p':'{:.4f}'})}</section>
<section class="panel"><h2>锁定模型与 Top-200 的中性化传导</h2>{_table(model_neutralization_display, {'raw_mean_rank_ic':'{:.4f}','neutral_mean_rank_ic':'{:.4f}','rank_ic_delta':'{:.4f}','rank_ic_ci_low':'{:.4f}','rank_ic_ci_high':'{:.4f}','raw_annualized_return':'{:.1%}','neutral_annualized_return':'{:.1%}','annualized_arithmetic_return_delta':'{:.1%}','return_delta_ci_low':'{:.1%}','return_delta_ci_high':'{:.1%}','raw_industry_active_weight_l1':'{:.3f}','neutral_industry_active_weight_l1':'{:.3f}'})}<p class="muted">固定 Ridge/HGB 参数、相同 PIT 暴露完整样本与 Top-200 规则。因子正交不保证排序尾部或非线性模型分数行业中性；本表未施加组合行业约束。</p></section>
<section class="panel"><h2>组合层行业、规模与换手约束</h2>{_table(constrained_portfolio_display, {'annualized_return':'{:.1%}','annualized_volatility':'{:.1%}','sharpe_zero_rate':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}','mean_abs_portfolio_size_z':'{:.3f}','mean_industry_active_weight_l1':'{:.3f}','mean_maximum_abs_industry_active_weight':'{:.3f}','turnover_relaxation_fraction':'{:.1%}'})}<p class="muted">单股 0.5%、行业主动权重 ±3%、|size-z|≤0.10、周换手目标 30%。所有约束组合均未触发换手放宽；收益是理想全成交、20bp 研究口径。</p></section>
<section class="panel"><h2>约束组合的日频可成交审计</h2>{_table(constrained_daily_execution_display, {'annualized_return':'{:.1%}','annualized_volatility':'{:.1%}','sharpe_zero_rate':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}','mean_blocked_buy_weight':'{:.2%}','mean_blocked_sell_weight':'{:.2%}','mean_target_tracking_l1':'{:.3f}','mean_cash_weight':'{:.2%}'})}<p class="muted">同一非重叠日频净值时钟、20bp 成本；停牌和涨跌停未成交股数订单逐日重试。raw HGB 从理想成交年化 22.18% 降至可成交 20.81%，折损区间显著低于零。</p></section>
<section class="panel"><h2>多因子风险与收益归因</h2>{_table(risk_attribution_display, {'annualized_return':'{:.1%}','annualized_market_contribution_arithmetic':'{:.1%}','annualized_industry_active_contribution_arithmetic':'{:.1%}','annualized_style_contribution_arithmetic':'{:.1%}','annualized_specific_selection_contribution_arithmetic':'{:.1%}','annualized_transaction_cost_arithmetic':'{:.1%}','mean_forecast_annualized_volatility':'{:.1%}','mean_forecast_annualized_tracking_error':'{:.1%}'})}<p class="muted">等权 PIT 股票池为市场基准。行业主动贡献接近零；raw HGB 的风格贡献约 13.61%，特异选股贡献约 -0.43%，其 95% 区间跨零。当前收益主要来自流动性、波动与动量风格，而非已证实的纯个股 alpha。</p></section>
<section class="panel"><h2>日频可成交组合 CAPM Alpha</h2>{_table(capm_display, {'annualized_alpha_arithmetic':'{:.1%}','annualized_alpha_ci_low':'{:.1%}','annualized_alpha_ci_high':'{:.1%}','alpha_hac_t_stat':'{:.2f}','beta':'{:.3f}','beta_ci_low':'{:.3f}','beta_ci_high':'{:.3f}','r_squared':'{:.3f}'})}<p class="muted">相对 CSI500，唯一非重叠日频净收益，Newey–West(5)。四条组合的全样本 Alpha 区间下界均为正，但滚动窗口曾转负；指数不含股息且无风险利率按0，因此不能将截距解释为纯个股 Alpha。</p></section>
<section class="panel"><h2>自融资风格组合</h2>{_table(style_factor_diagnostics, {'annualized_mean_return_arithmetic':'{:.1%}','annualized_volatility':'{:.1%}','mean_return_hac_t_stat':'{:.2f}','mean_weekly_two_leg_turnover':'{:.3f}'})}<p class="muted">目标风格先对宽行业与其余五风格残差化；正负残差形成单位多空腿。收益未扣成本，仅作时间序列解释变量。</p></section>
<section class="panel"><h2>六风格时间序列归因</h2>{_table(style_regression_display, {'annualized_alpha_arithmetic':'{:.1%}','annualized_alpha_ci_low':'{:.1%}','annualized_alpha_ci_high':'{:.1%}','alpha_hac_t_stat':'{:.2f}','r_squared':'{:.3f}','loading_market':'{:.3f}','loading_size':'{:.3f}','loading_beta':'{:.3f}','loading_volatility':'{:.3f}','loading_liquidity':'{:.3f}','loading_value':'{:.3f}','loading_momentum':'{:.3f}','capm_annualized_alpha':'{:.1%}'})}<p class="muted">加入 CSI500 与六个本地风格后，年化残余 Alpha 为 -2.55% 至 +1.07%，四条95%区间均跨零，R²约0.895。CAPM Alpha 主要被小盘、非流动性和反转暴露吸收；当前没有显著纯选股 Alpha 证据。</p></section>
<section class="panel"><h2>事前风险约束与目标波动（失败实验）</h2>{_table(risk_aware_display, {'annualized_return':'{:.1%}','annualized_volatility':'{:.1%}','sharpe_zero_rate':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}','mean_forecast_annualized_volatility':'{:.1%}','mean_forecast_annualized_tracking_error':'{:.1%}','mean_risky_weight':'{:.1%}','mean_alpha_utility_retention':'{:.1%}','turnover_relaxation_fraction':'{:.1%}'})}<p class="muted">v1 保留 98% alpha score utility 并使用严格滞后风险输入。raw HGB 预测 TE 从 8.07% 降至 4.43%，但年化收益从 24.68% 降至 18.69%，收益差区间完全低于零；该政策不采用。</p></section>
<section class="panel"><h2>Top-N 与 inverse-vol 共同日期诊断</h2>{_table(portfolio_sensitivity_display, {'maximum_target_weight':'{:.2%}','annualized_return':'{:.1%}','annualized_volatility':'{:.1%}','sharpe_zero_rate':'{:.2f}','annualized_excess_return_arithmetic':'{:.1%}','information_ratio':'{:.2f}','maximum_drawdown':'{:.1%}','mean_turnover':'{:.3f}','mean_realized_weight_fraction':'{:.2%}'})}<p class="muted">Top50/100/200/500 全部报告；historical OOS 已被查看，不据此重新选择已锁定的 Top-200 等权政策。</p></section>
<section class="panel"><h2>最终日频组合 0/20/50/100bp 成本压力</h2>{_table(execution_cost_stress_display, {'cost_bps':'{:.0f}','annualized_return':'{:.1%}','sharpe_zero_rate':'{:.2f}','maximum_drawdown':'{:.1%}'})}<p class="muted">固定实际成交路径的线性费用重定价，不包含费用反馈、股数取整或券商校准市场冲击。</p></section>
<section class="panel"><h2>HGB 二维参数稳定面</h2>{_table(hgb_surface_display, {'l2_regularization':'{:.1f}','mean_rank_ic':'{:.4f}','annualized_icir':'{:.2f}','portfolio_annualized_return':'{:.1%}','portfolio_sharpe_zero_rate':'{:.2f}'})}<p class="muted">仅使用 2006–2015 训练和 2016–2020 验证。RankIC 面较平坦，但理想成交收益与年度最优坐标不稳定；不据此提升新参数。</p></section>
<section class="panel finding"><h2>最终 finding</h2><p>Ridge 多因子基线在各历史样本外年度均保持正 RankIC；20bp 成本后，相对 CSI 500 的年化算术超额块 bootstrap 95% 区间仍高于 0，这是当前最可靠的正向证据。RF 的 RankIC 约 0.100，但旧组合回测同时含持有期重叠和不可成交偏差。改用股数订单、非重叠日频时钟及停牌/涨跌停逐日重试后，historical OOS 年化 {rf_oos_constrained.annualized_return:.2%}、Sharpe {rf_oos_constrained.sharpe_zero_rate:.2f}、最大回撤 {rf_oos_constrained.maximum_drawdown:.2%}；验证期为 {rf_validation_constrained.annualized_return:.2%}。显式组合约束把每个行业主动权重压到 ±3%、规模暴露压到 ±0.10，并把换手稳定在 30%。在非重叠日频成交口径下，原始 HGB 理想成交年化 22.18%，加入停牌/涨跌停逐日重试后为 20.81%、Sharpe 0.94、最大回撤 -33.32%。正式归因进一步表明行业主动贡献接近零，但 raw HGB 约 13.61% 的年化算术贡献来自流动性、波动、动量等风格，特异选股贡献 -0.43% 且区间 [-3.52%, 3.08%] 跨零；因此当前证据支持“可控风格溢价”，不支持“稳定纯个股 alpha”。第一版风险感知政策虽把 raw HGB 预测 TE 从 8.07% 降到 4.43%，但收益差区间显著为负，故明确拒绝该政策。结果仍受申万细分行业缺失、股票池代理、日内排队和冲击模型未校准等限制；2021–2026 已在开发中被检查，应称为历史样本外，不再称作从未查看的 lockbox。</p></section>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    print(generate())


if __name__ == "__main__":
    main()
