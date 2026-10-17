"""PIT self-financing style factors and daily multi-factor attribution."""

from __future__ import annotations

import argparse
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

from quant_research.capm_attribution import newey_west_ols
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import weekly_rows
from quant_research.multi_factor_risk_attribution import build_style_exposures


CONFIG_PATH = ROOT / "config" / "style_factor_attribution.yaml"
RISK_CONFIG_PATH = ROOT / "config" / "multi_factor_risk_attribution.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_risk_config() -> dict:
    with RISK_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def residual_factor_weights(
    styles: pd.DataFrame, industry: pd.Series, factor: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a zero-net factor portfolio orthogonal to industries and other styles."""
    if factor not in styles:
        raise KeyError(f"style missing: {factor}")
    industries = industry.fillna("").astype(str).str.strip()
    if industries.eq("").any():
        raise ValueError("industry must be complete before factor construction")
    dummies = pd.get_dummies(industries, dtype="float64")
    other_styles = styles.drop(columns=factor).astype("float64")
    controls = np.column_stack([dummies.to_numpy(), other_styles.to_numpy()])
    target = styles[factor].to_numpy(dtype="float64")
    coefficients = np.linalg.lstsq(controls, target, rcond=None)[0]
    residual = target - controls @ coefficients
    positive = np.clip(residual, 0.0, None)
    negative = np.clip(-residual, 0.0, None)
    if positive.sum() <= 0 or negative.sum() <= 0:
        raise RuntimeError(f"factor residual has no two-sided support: {factor}")
    weights = positive / positive.sum() - negative / negative.sum()
    industry_exposure = pd.Series(weights).groupby(industries.reset_index(drop=True)).sum()
    control_exposures = controls.T @ weights
    diagnostics = {
        "net_weight": float(weights.sum()),
        "gross_weight": float(np.abs(weights).sum()),
        "long_weight": float(np.clip(weights, 0, None).sum()),
        "short_weight": float(np.clip(-weights, 0, None).sum()),
        "maximum_abs_industry_weight": float(industry_exposure.abs().max()),
        "maximum_abs_control_exposure": float(np.abs(control_exposures).max()),
        "target_style_exposure": float(target @ weights),
    }
    return weights, diagnostics


def _read_signal_year(lake: Lake, year: int, risk_cfg: dict) -> pd.DataFrame:
    panel_columns = [
        "instrument_id", "asof_date", "available_date",
        "momentum_120d", "momentum_20d_skip_5d", "volatility_20d",
        "amihud_20d", "turnover_20d", "earnings_yield", "book_to_price",
    ]
    panel = weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=panel_columns
    ))
    size = pd.read_parquet(
        lake.table_path("pit_size_exposures", str(year)),
        columns=["instrument_id", "asof_date", "float_cap_proxy_cny"],
    )
    industry = pd.read_parquet(
        lake.table_path("pit_industry_exposures", str(year)),
        columns=["instrument_id", "asof_date", "industry_broad_code"],
    )
    beta = pd.read_parquet(
        lake.table_path("pit_beta_exposures", str(year)),
        columns=["instrument_id", "asof_date", "beta_252d"],
    )
    return panel.merge(
        size, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    ).merge(
        industry, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    ).merge(
        beta, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
    )


def build_factor_targets(
    lake: Lake, cfg: dict, risk_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    styles = list(cfg["styles"])
    target_frames: list[pd.DataFrame] = []
    audit_records: list[dict] = []
    for year in range(2021, 2027):
        frame = _read_signal_year(lake, year, risk_cfg)
        for asof_date, raw_group in frame.groupby("asof_date", sort=True):
            valid = (
                pd.to_numeric(raw_group["float_cap_proxy_cny"], errors="coerce").gt(0)
                & raw_group["industry_broad_code"].fillna("").astype(str).str.strip().ne("")
            )
            group = raw_group.loc[valid].reset_index(drop=True)
            if len(group) < 100:
                raise RuntimeError(f"style factor cross section too small: {asof_date}")
            style_values = build_style_exposures(group, risk_cfg)
            output = group[["instrument_id", "asof_date", "available_date"]].copy()
            output = output.rename(columns={
                "asof_date": "signal_date", "available_date": "execution_date"
            })
            for factor in styles:
                weights, diagnostics = residual_factor_weights(
                    style_values, group["industry_broad_code"], factor
                )
                output[f"weight_{factor}"] = weights
                audit_records.append({
                    "signal_date": str(asof_date),
                    "execution_date": str(output["execution_date"].iloc[0]),
                    "year": year, "factor": factor,
                    "cross_section_size": len(group), **diagnostics,
                })
            target_frames.append(output)
        print(f"style targets year={year} dates={frame['asof_date'].nunique()}", flush=True)
    targets = pd.concat(target_frames, ignore_index=True)
    if targets.duplicated(["execution_date", "instrument_id"]).any():
        raise RuntimeError("duplicate style target execution-date/instrument keys")
    return targets, pd.DataFrame(audit_records)


def build_factor_market(lake: Lake, targets: pd.DataFrame) -> pd.DataFrame:
    """Reuse audited daily market rows and supplement the remaining PIT universe."""
    base = lake.read("constrained_daily_execution_market", "v1")[[
        "instrument_id", "trade_date", "daily_return"
    ]]
    required_codes = set(targets["instrument_id"].astype(str))
    base = base.loc[base["instrument_id"].astype(str).isin(required_codes)].copy()
    present = set(base["instrument_id"].astype(str))
    missing_codes = sorted(required_codes - present)
    start = str(targets["execution_date"].min())
    end = str(base["trade_date"].max())
    frames = [base]
    for number, code in enumerate(missing_codes, start=1):
        bars = pd.read_parquet(
            lake.table_path("daily_bars_validated", code.replace(".", "_")),
            columns=["instrument_id", "trade_date", "pct_chg"],
        )
        bars = bars.loc[bars["trade_date"].between(start, end)].copy()
        bars["daily_return"] = pd.to_numeric(bars["pct_chg"], errors="coerce") / 100.0
        frames.append(bars[["instrument_id", "trade_date", "daily_return"]])
        if number % 250 == 0:
            print(f"style market supplements={number}/{len(missing_codes)}", flush=True)
    market = pd.concat(frames, ignore_index=True)
    market = market.drop_duplicates(["instrument_id", "trade_date"], keep="last")
    return market.sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)


def _leg_turnover(current: pd.Series | None, target: pd.Series, positive: bool) -> float:
    target_leg = target.clip(lower=0) if positive else (-target.clip(upper=0))
    if current is None:
        return 1.0
    current_leg = current.clip(lower=0) if positive else (-current.clip(upper=0))
    names = current_leg.index.union(target_leg.index)
    return float(0.5 * np.abs(
        current_leg.reindex(names, fill_value=0).to_numpy()
        - target_leg.reindex(names, fill_value=0).to_numpy()
    ).sum())


def simulate_factor_returns(
    targets: pd.DataFrame, market: pd.DataFrame, styles: list[str],
) -> pd.DataFrame:
    target_groups = {
        str(date): group.set_index("instrument_id")
        for date, group in targets.groupby("execution_date", sort=True)
    }
    current: dict[str, pd.Series | None] = {factor: None for factor in styles}
    records: list[dict] = []
    first_date = min(target_groups)
    for raw_date, raw_market in market.groupby("trade_date", sort=True):
        date = str(raw_date)
        if date < first_date:
            continue
        returns = pd.to_numeric(
            raw_market.set_index("instrument_id")["daily_return"], errors="coerce"
        )
        new_targets = target_groups.get(date)
        for factor in styles:
            weights = current[factor]
            if weights is None:
                long_return = short_return = factor_return = 0.0
                drifted = None
            else:
                aligned_return = returns.reindex(weights.index).fillna(0.0)
                long_weights = weights.clip(lower=0)
                short_weights = -weights.clip(upper=0)
                long_return = float(long_weights @ aligned_return)
                short_return = float(short_weights @ aligned_return)
                factor_return = long_return - short_return
                if 1.0 + long_return <= 0 or 1.0 + short_return <= 0:
                    raise RuntimeError(f"style factor leg became non-positive: {date}/{factor}")
                drifted = (
                    long_weights * (1.0 + aligned_return) / (1.0 + long_return)
                    - short_weights * (1.0 + aligned_return) / (1.0 + short_return)
                )
                drifted = drifted.loc[drifted.abs().gt(1e-16)]
            if new_targets is not None:
                target = new_targets[f"weight_{factor}"].astype("float64")
                long_turnover = _leg_turnover(drifted, target, True)
                short_turnover = _leg_turnover(drifted, target, False)
                current[factor] = target
            else:
                long_turnover = short_turnover = 0.0
                current[factor] = drifted
            records.append({
                "trade_date": date, "factor": factor,
                "factor_return": factor_return,
                "long_leg_return": long_return,
                "short_reference_return": short_return,
                "new_target": new_targets is not None,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "two_leg_turnover": long_turnover + short_turnover,
            })
    return pd.DataFrame(records)


def multi_factor_record(
    strategy_return: pd.Series, factors: pd.DataFrame, cfg: dict,
) -> dict[str, float | int]:
    factor_names = ["market", *cfg["styles"]]
    design = np.column_stack([
        np.ones(len(factors)),
        *[pd.to_numeric(factors[name], errors="coerce").to_numpy() for name in factor_names],
    ])
    result = newey_west_ols(
        pd.to_numeric(strategy_return, errors="coerce").to_numpy(),
        design, int(cfg["newey_west_lags"]),
    )
    coefficients = np.asarray(result["coefficients"])
    standard_errors = np.asarray(result["standard_errors"])
    critical = NormalDist().inv_cdf(0.5 + float(cfg["confidence_level"]) / 2.0)
    annualization = int(cfg["annualization_days"])
    record: dict[str, float | int] = {
        "observations": int(result["observations"]),
        "annualized_alpha_arithmetic": float(coefficients[0] * annualization),
        "annualized_alpha_hac_se": float(standard_errors[0] * annualization),
        "annualized_alpha_ci_low": float(
            (coefficients[0] - critical * standard_errors[0]) * annualization
        ),
        "annualized_alpha_ci_high": float(
            (coefficients[0] + critical * standard_errors[0]) * annualization
        ),
        "alpha_hac_t_stat": float(
            coefficients[0] / standard_errors[0] if standard_errors[0] > 0 else np.nan
        ),
        "r_squared": float(result["r_squared"]),
        "annualized_residual_volatility": float(
            np.asarray(result["residuals"]).std(ddof=len(factor_names) + 1)
            * np.sqrt(annualization)
        ),
        "newey_west_lags": int(cfg["newey_west_lags"]),
    }
    for index, factor in enumerate(factor_names, start=1):
        coefficient = float(coefficients[index])
        error = float(standard_errors[index])
        record.update({
            f"loading_{factor}": coefficient,
            f"loading_{factor}_hac_se": error,
            f"loading_{factor}_ci_low": coefficient - critical * error,
            f"loading_{factor}_ci_high": coefficient + critical * error,
            f"loading_{factor}_hac_t_stat": coefficient / error if error > 0 else np.nan,
        })
    return record


def regression_panel(
    lake: Lake, factor_daily: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    factor_wide = factor_daily.pivot(
        index="trade_date", columns="factor", values="factor_return"
    ).reset_index()
    capm_daily = lake.read("capm_aligned_daily", "v1")
    capm_daily = capm_daily.rename(columns={str(cfg["market_factor"]): "market"})
    return capm_daily.merge(
        factor_wide, on="trade_date", how="left", validate="many_to_one"
    )


def regressions(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "alpha_pipeline", "strategy", "scenario"]
    factor_columns = ["market", *cfg["styles"]]
    summary_records: list[dict] = []
    annual_records: list[dict] = []
    frame = panel.copy()
    frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
    for values, group in frame.groupby(keys, sort=False):
        group = group.sort_values("trade_date")
        summary_records.append({
            **dict(zip(keys, values)),
            **multi_factor_record(group["net_return"], group[factor_columns], cfg),
        })
        for year, annual_group in group.groupby("year", sort=True):
            annual_records.append({
                "year": int(year), **dict(zip(keys, values)),
                **multi_factor_record(
                    annual_group["net_return"], annual_group[factor_columns], cfg
                ),
            })
    return pd.DataFrame(summary_records), pd.DataFrame(annual_records)


def factor_diagnostics(factor_daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    records = []
    for factor, group in factor_daily.groupby("factor", sort=False):
        result = newey_west_ols(
            group["factor_return"].to_numpy(), np.ones((len(group), 1)),
            int(cfg["newey_west_lags"]),
        )
        mean, error = float(result["coefficients"][0]), float(result["standard_errors"][0])
        records.append({
            "factor": factor, "orientation": cfg["factor_orientation"][factor],
            "observations": len(group),
            "annualized_mean_return_arithmetic": mean * int(cfg["annualization_days"]),
            "annualized_volatility": float(
                group["factor_return"].std(ddof=1) * np.sqrt(int(cfg["annualization_days"]))
            ),
            "mean_return_hac_t_stat": mean / error if error > 0 else np.nan,
            "mean_weekly_two_leg_turnover": float(
                group.loc[group["new_target"], "two_leg_turnover"].mean()
            ),
        })
    return pd.DataFrame(records)


def multicollinearity_diagnostics(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    factors = ["market", *cfg["styles"]]
    frame = panel[["trade_date", *factors]].drop_duplicates("trade_date").dropna()
    standardized = (frame[factors] - frame[factors].mean()) / frame[factors].std(ddof=1)
    condition_number = float(np.linalg.cond(standardized.to_numpy(dtype="float64")))
    correlation = frame[factors].corr()
    records = []
    for factor in factors:
        other = [column for column in factors if column != factor]
        design = np.column_stack([np.ones(len(frame)), frame[other].to_numpy()])
        target = frame[factor].to_numpy()
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        total = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - float(residual @ residual) / total
        records.append({
            "factor": factor,
            "variance_inflation_factor": 1.0 / (1.0 - r_squared),
            "maximum_abs_pairwise_correlation": float(
                correlation.loc[factor].drop(factor).abs().max()
            ),
            "standardized_design_condition_number": condition_number,
            "observations": len(frame),
        })
    return pd.DataFrame(records)


def write_report(
    cfg: dict, summary: pd.DataFrame, annual: pd.DataFrame,
    diagnostics: pd.DataFrame, multicollinearity: pd.DataFrame, capm: pd.DataFrame,
) -> None:
    primary = summary.loc[summary["scenario"].eq(cfg["primary_scenario"])].copy()
    capm_primary = capm.loc[
        capm["benchmark"].eq(cfg["market_factor"])
        & capm["scenario"].eq(cfg["primary_scenario"]),
        ["model", "alpha_pipeline", "annualized_alpha_arithmetic"],
    ].rename(columns={"annualized_alpha_arithmetic": "capm_annualized_alpha"})
    primary = primary.merge(
        capm_primary, on=["model", "alpha_pipeline"], validate="one_to_one"
    )
    primary["alpha_reduction_vs_capm"] = (
        primary["annualized_alpha_arithmetic"] - primary["capm_annualized_alpha"]
    )
    annual_primary = annual.loc[annual["scenario"].eq(cfg["primary_scenario"])]
    positive_years = annual_primary.groupby(["model", "alpha_pipeline"])[
        "annualized_alpha_arithmetic"
    ].apply(lambda values: int(values.gt(0).sum()))
    primary = primary.merge(
        positive_years.rename("positive_multifactor_alpha_years"),
        on=["model", "alpha_pipeline"], validate="one_to_one",
    )
    display_columns = [
        "model", "alpha_pipeline", "annualized_alpha_arithmetic",
        "annualized_alpha_ci_low", "annualized_alpha_ci_high", "alpha_hac_t_stat",
        "capm_annualized_alpha", "alpha_reduction_vs_capm", "r_squared",
        "loading_market", "loading_size", "loading_beta", "loading_volatility",
        "loading_liquidity", "loading_value", "loading_momentum",
        "positive_multifactor_alpha_years",
    ]
    display = primary[display_columns].copy()
    numeric = [column for column in display if column not in {
        "model", "alpha_pipeline", "positive_multifactor_alpha_years"
    }]
    for column in numeric:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    significant = primary.loc[
        primary["annualized_alpha_ci_low"].gt(0), ["model", "alpha_pipeline"]
    ].apply(lambda row: f"{row['model']}/{row['alpha_pipeline']}", axis=1).tolist()
    report = "\n".join([
        "# PIT 自融资风格组合与多因子时间序列归因", "", "## 方法", "",
        "- 六个风格均使用信号日 PIT 暴露；目标风格先对宽行业和其余五风格残差化。",
        "- 残差正负分别形成单位多头与单位空头，净投资0、总杠杆2；下一交易日收盘换仓。",
        "- 风格因子为解释变量，未扣交易成本；双腿换手单独披露，不能视为直接可交易收益。",
        "- 组合收益为包含20bp成本及停牌/涨跌停重试的日频净收益。",
        f"- 时间序列回归包含 {cfg['market_factor']} 和六风格，使用 Newey–West({cfg['newey_west_lags']})。",
        "", "## 风格组合诊断", "", diagnostics.to_markdown(index=False, floatfmt=".6f"),
        "", "## 回归设计诊断", "", multicollinearity.to_markdown(index=False, floatfmt=".6f"),
        "", "## 可成交组合多因子 Alpha", "",
        f"- 95% Alpha 区间下界为正：{', '.join(significant) or '无'}。",
        "- `alpha_reduction_vs_capm` 为多因子 Alpha 减单因子 CAPM Alpha；负值表示风格因子解释了部分截距。",
        "", display.to_markdown(index=False), "", "## 限制", "",
        "风格组合使用免费数据、证监会宽行业和零成本理想多空腿；未建模借券、涨跌停、容量和"
        "指数股息。回归用于收益来源诊断，不构成可交易因子组合业绩。", "",
    ])
    destination = ROOT / "reports" / "style_factor_attribution.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    table = ROOT / "reports" / "tables" / "style_factor_attribution.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(table, index=False)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg, risk_cfg = load_config(), load_risk_config()
    targets, exposure_audit = build_factor_targets(lake, cfg, risk_cfg)
    market = build_factor_market(lake, targets)
    factor_daily = simulate_factor_returns(targets, market, list(cfg["styles"]))
    panel = regression_panel(lake, factor_daily, cfg)
    summary, annual = regressions(panel, cfg)
    diagnostics = factor_diagnostics(factor_daily, cfg)
    multicollinearity = multicollinearity_diagnostics(panel, cfg)
    correlation = factor_daily.pivot(
        index="trade_date", columns="factor", values="factor_return"
    ).corr().reset_index().rename(columns={"factor": "row_factor"})
    capm = lake.read("capm_summary", "v1")
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "source_experiment": cfg["source_experiment"],
        "historical_oos_used_for_selection": False,
        "factor_construction": cfg["factor_portfolio"]["construction"],
        "factor_controls": cfg["factor_portfolio"]["controls"],
        "factor_transaction_cost_bps": float(
            cfg["factor_portfolio"]["transaction_cost_bps"]
        ),
        "factor_execution": cfg["factor_portfolio"]["execution"],
        "unique_nonoverlap_daily_clock": True,
        "daily_dates": int(factor_daily["trade_date"].nunique()),
        "style_factors": len(cfg["styles"]),
    }])
    request = cfg.copy()
    for table, frame in {
        "style_factor_protocol": protocol,
        "style_factor_targets": targets,
        "style_factor_exposure_audit": exposure_audit,
        "style_factor_daily_market": market,
        "style_factor_daily_returns": factor_daily,
        "style_factor_diagnostics": diagnostics,
        "style_factor_multicollinearity": multicollinearity,
        "style_factor_correlation": correlation,
        "style_factor_regression_summary": summary,
        "style_factor_regression_annual": annual,
    }.items():
        lake.write(table, frame, key="v1", source="style_factor_attribution.v1", request=request)
    write_report(cfg, summary, annual, diagnostics, multicollinearity, capm)
    print(diagnostics.to_string(index=False), flush=True)
    print(summary.loc[summary["scenario"].eq(cfg["primary_scenario"])].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PIT self-financing style factors and multi-factor attribution"
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
