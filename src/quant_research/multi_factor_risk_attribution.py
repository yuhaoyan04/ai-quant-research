"""PIT multi-factor return attribution and ex-ante risk forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.daily_execution import daily_performance_summary
from quant_research.model_baseline import weekly_rows
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "multi_factor_risk_attribution.yaml"
REPORT_PATH = ROOT / "reports" / "multi_factor_risk_attribution.md"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _weekly_dates(lake: Lake, cfg: dict) -> set[str]:
    dates: set[str] = set()
    for year in range(int(cfg["risk_model"]["start_year"]),
                      int(cfg["risk_model"]["end_year"]) + 1):
        frame = pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["asof_date"],
        )
        dates.update(weekly_rows(frame)["asof_date"].astype(str).unique())
    return dates


def build_beta_exposures(lake: Lake | None = None, *, overwrite: bool = False) -> None:
    """Build trailing PIT market-beta exposures on weekly signal dates."""
    lake = lake or Lake()
    cfg = load_config()
    years = range(int(cfg["risk_model"]["start_year"]),
                  int(cfg["risk_model"]["end_year"]) + 1)
    if not overwrite and lake.table_path("pit_beta_exposure_index", "v1").exists() and all(
        lake.table_path("pit_beta_exposures", str(year)).exists() for year in years
    ):
        return
    benchmark = str(cfg["beta"]["benchmark_instrument"])
    bars = pd.read_parquet(
        lake.table_path("daily_bars_validated", benchmark.replace(".", "_")),
        columns=["trade_date", "pct_chg"],
    )
    benchmark_return = pd.Series(
        pd.to_numeric(bars["pct_chg"], errors="coerce").to_numpy(dtype="float64") / 100.0,
        index=bars["trade_date"].astype(str),
    )
    dates = _weekly_dates(lake, cfg)
    start_year, end_year = min(years), max(years)
    window = int(cfg["beta"]["lookback_trading_days"])
    minimum = int(cfg["beta"]["minimum_observations"])
    frames = []
    paths = sorted((lake.curated / "factors_raw").glob("key=*/data.parquet"))
    for number, path in enumerate(paths, start=1):
        frame = pd.read_parquet(
            path, columns=["instrument_id", "asof_date", "return_1d"]
        )
        stock = pd.to_numeric(frame["return_1d"], errors="coerce")
        market = pd.to_numeric(
            frame["asof_date"].astype(str).map(benchmark_return), errors="coerce"
        )
        beta = stock.rolling(window, min_periods=minimum).cov(market).div(
            market.rolling(window, min_periods=minimum).var()
        )
        keep = (
            frame["asof_date"].astype(str).isin(dates)
            & frame["asof_date"].astype(str).str[:4].astype(int).between(start_year, end_year)
        )
        if keep.any():
            selected = frame.loc[keep, ["instrument_id", "asof_date"]].copy()
            selected["beta_252d"] = beta.loc[keep].to_numpy(dtype="float64")
            frames.append(selected)
        if number % 500 == 0:
            print(f"PIT beta files={number}/{len(paths)}", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["asof_date"].str[:4].astype(int)
    request = {**cfg["beta"], "weekly_signal_dates": len(dates)}
    index_records = []
    for year, group in combined.groupby("year", sort=True):
        lake.write(
            "pit_beta_exposures", group.drop(columns="year"), key=str(year),
            source="multi_factor_risk_attribution.beta.v1", request=request,
        )
        print(
            f"PIT beta year={year} rows={len(group):,} "
            f"coverage={group['beta_252d'].notna().mean():.2%}", flush=True,
        )
        index_records.append({
            "year": int(year), "rows": len(group),
            "dates": group["asof_date"].nunique(),
            "instruments": group["instrument_id"].nunique(),
            "beta_coverage": float(group["beta_252d"].notna().mean()),
            "duplicate_instrument_date_keys": int(
                group.duplicated(["instrument_id", "asof_date"]).sum()
            ),
        })
    lake.write(
        "pit_beta_exposure_index", pd.DataFrame(index_records), key="v1",
        source="multi_factor_risk_attribution.beta.v1", request=request,
    )


def _zscore(values: pd.Series | np.ndarray, quantiles: tuple[float, float]) -> np.ndarray:
    raw = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(raw)
    if valid.sum() < 3:
        return np.zeros(len(raw), dtype="float64")
    lower, upper = np.quantile(raw[valid], quantiles)
    clipped = np.clip(raw, lower, upper)
    fill = float(np.median(clipped[valid]))
    clipped[~valid] = fill
    deviation = clipped.std(ddof=1)
    return ((clipped - clipped.mean()) / deviation
            if deviation > 0 else np.zeros(len(raw), dtype="float64"))


def build_style_exposures(group: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create prespecified same-date style exposures with zero universe means."""
    q = tuple(float(value) for value in cfg["style_winsor_quantiles"])
    positive_vol = pd.to_numeric(group["volatility_20d"], errors="coerce").where(
        lambda value: value > 0
    )
    amihud = pd.to_numeric(group["amihud_20d"], errors="coerce").where(
        lambda value: value > 0
    )
    turnover = pd.to_numeric(group["turnover_20d"], errors="coerce").where(
        lambda value: value > 0
    )
    result = pd.DataFrame(index=group.index)
    if "size_z" in group:
        size_source = group["size_z"]
    else:
        positive_size = pd.to_numeric(
            group["float_cap_proxy_cny"], errors="coerce"
        ).where(lambda value: value > 0)
        size_source = np.log(positive_size)
    result["size"] = _zscore(size_source, q)
    result["beta"] = _zscore(group["beta_252d"], q)
    result["volatility"] = _zscore(np.log(positive_vol), q)
    result["liquidity"] = _zscore(
        (_zscore(-np.log(amihud), q) + _zscore(np.log(turnover), q)) / 2.0, q
    )
    result["value"] = _zscore(
        (_zscore(group["earnings_yield"], q)
         + _zscore(group["book_to_price"], q)) / 2.0, q
    )
    result["momentum"] = _zscore(
        (_zscore(group["momentum_120d"], q)
         + _zscore(group["momentum_20d_skip_5d"], q)) / 2.0, q
    )
    return result


def fit_factor_cross_section(group: pd.DataFrame, cfg: dict) -> dict:
    """Fit full industry dummies plus styles and return an exact decomposition."""
    target = str(cfg["target"])
    valid = group["industry_broad_code"].fillna("").astype(str).str.strip().ne("")
    frame = group.loc[valid].reset_index(drop=True).copy()
    industries = frame["industry_broad_code"].astype(str)
    categories = sorted(industries.unique())
    industry_design = np.column_stack([
        industries.eq(category).to_numpy(dtype="float64") for category in categories
    ])
    styles = build_style_exposures(frame, cfg)
    design = np.column_stack([industry_design, styles.to_numpy(dtype="float64")])
    realized = pd.to_numeric(frame[target], errors="coerce").fillna(0.0).to_numpy(
        dtype="float64"
    )
    frame[target] = realized
    coefficients = np.linalg.lstsq(design, realized, rcond=None)[0]
    fitted = design @ coefficients
    residual = realized - fitted
    market = float(realized.mean())
    industry_coefficients = dict(zip(categories, coefficients[:len(categories)]))
    style_coefficients = dict(zip(styles.columns, coefficients[len(categories):]))
    benchmark_industry = industries.value_counts(normalize=True).to_dict()
    total_variation = float(np.sum((realized - market) ** 2))
    explained = (
        float(1.0 - np.sum(residual ** 2) / total_variation)
        if total_variation > 0 else np.nan
    )
    frame = frame.assign(residual_return=residual)
    for style in styles:
        frame[f"style_{style}"] = styles[style].to_numpy(dtype="float64")
    return {
        "frame": frame,
        "market_return": market,
        "industry_coefficients": industry_coefficients,
        "industry_active_returns": {
            industry: value - market for industry, value in industry_coefficients.items()
        },
        "style_returns": style_coefficients,
        "benchmark_industry": benchmark_industry,
        "r_squared": explained,
        "residual_std": float(residual.std(ddof=1)),
    }


def ewma_covariance(history: list[np.ndarray], cfg: dict) -> np.ndarray | None:
    covariance_cfg = cfg["covariance"]
    minimum = int(covariance_cfg["minimum_weeks"])
    if len(history) < minimum:
        return None
    values = np.vstack(history[-int(covariance_cfg["trailing_weeks"]):])
    half_life = float(covariance_cfg["ewma_half_life_weeks"])
    ages = np.arange(len(values) - 1, -1, -1, dtype="float64")
    weights = np.exp(-np.log(2.0) * ages / half_life)
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    demeaned = values - mean
    covariance = (demeaned * weights[:, None]).T @ demeaned
    shrinkage = float(covariance_cfg["diagonal_shrinkage"])
    return (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))


def _read_year(lake: Lake, year: int, cfg: dict) -> pd.DataFrame:
    columns = [
        "instrument_id", "asof_date", str(cfg["target"]),
        "momentum_120d", "momentum_20d_skip_5d", "volatility_20d",
        "amihud_20d", "turnover_20d", "earnings_yield", "book_to_price",
    ]
    panel = weekly_rows(pd.read_parquet(
        lake.table_path("multi_horizon_factor_panel", str(year)), columns=columns
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
    return panel.merge(size, on=["instrument_id", "asof_date"], how="left").merge(
        industry, on=["instrument_id", "asof_date"], how="left"
    ).merge(beta, on=["instrument_id", "asof_date"], how="left")


def _bootstrap(attribution: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    metrics = [
        "industry_active_contribution", "style_contribution",
        "specific_selection_contribution", "active_gross_return", "active_net_return",
    ]
    records = []
    boot = cfg["bootstrap"]
    scale = int(cfg["annualization_periods"])
    for keys, group in attribution.groupby(
        ["model", "alpha_pipeline", "strategy"], sort=False
    ):
        group = group.sort_values("asof_date")
        for metric in metrics:
            values = group[metric].to_numpy(dtype="float64")
            estimates = circular_block_bootstrap_mean(
                values, int(boot["block_length_weeks"]), int(boot["samples"]),
                int(boot["random_seed"]),
            ) * scale
            records.append({
                "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
                "metric": metric, "observations": len(values),
                "point_estimate": float(values.mean() * scale),
                "ci_2_5pct": float(np.quantile(estimates, 0.025)),
                "ci_97_5pct": float(np.quantile(estimates, 0.975)),
                "probability_nonpositive": float(
                    (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
                ),
            })
    return pd.DataFrame(records)


def _write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    specific = bootstrap.loc[
        bootstrap["metric"].eq("specific_selection_contribution")
    ]
    lines = [
        "# PIT 多因子风险模型与收益归因", "", "## 方法", "",
        "- 等权 PIT 暴露完整股票池为市场基准；截面 OLS 同时估计宽行业与六个预设风格。",
        "- 风格包括规模、252 日 Beta、波动、流动性、价值与动量；暴露只使用信号日可知信息。",
        "- 组合收益严格拆为市场、行业主动、风格、特异选股与交易成本。",
        "- 协方差预测只使用当期之前 104 周，26 周半衰期，并向对角矩阵收缩 20%。", "",
        "## 2021–2026 historical OOS 汇总", "",
        summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## 特异选股贡献的配对块区间", "",
        specific.to_markdown(index=False, floatfmt=".6f"), "",
        "> 风险和归因使用周频 5 日标签口径；日频可成交净值见独立执行审计。", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    cfg = load_config()
    build_beta_exposures(lake)
    holdings = lake.read("constrained_portfolio_oos_holdings", "v1")
    holdings = holdings.loc[
        holdings["construction"].eq(str(cfg["portfolio_construction"]))
    ].copy()
    weekly = lake.read("constrained_portfolio_oos_weekly", "v1")
    weekly = weekly.loc[
        weekly["construction"].eq(str(cfg["portfolio_construction"]))
    ].copy()
    holding_groups = {
        (str(date), str(model), str(pipeline), str(strategy)): group
        for (date, model, pipeline, strategy), group in holdings.groupby(
            ["asof_date", "model", "alpha_pipeline", "strategy"], sort=False
        )
    }
    weekly_lookup = weekly.set_index(
        ["asof_date", "model", "alpha_pipeline", "strategy"]
    )
    industry_categories = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "LEGACY_SOCIAL_SERVICES", "M", "N", "O", "P", "Q", "R", "S",
    ]
    styles = list(cfg["styles"])
    factor_names = ["market", *[f"industry_{item}" for item in industry_categories],
                    *[f"style_{item}" for item in styles]]
    factor_history: list[np.ndarray] = []
    specific_variance: dict[str, float] = {}
    specific_counts: dict[str, int] = {}
    specific_decay = np.exp(
        -np.log(2.0) / float(cfg["specific_risk"]["ewma_half_life_weeks"])
    )
    factor_records, attribution_records, risk_records, exposure_records = [], [], [], []
    start_attr = int(cfg["attribution"]["start_year"])
    for year in range(int(cfg["risk_model"]["start_year"]),
                      int(cfg["risk_model"]["end_year"]) + 1):
        frame = _read_year(lake, year, cfg)
        for asof_date, group in frame.groupby("asof_date", sort=True):
            date = str(asof_date)
            fitted = fit_factor_cross_section(group, cfg)
            cross = fitted["frame"]
            active_industry = fitted["industry_active_returns"]
            factor_vector = np.array([
                fitted["market_return"],
                *[active_industry.get(item, 0.0) for item in industry_categories],
                *[fitted["style_returns"].get(item, 0.0) for item in styles],
            ], dtype="float64")
            for name, value in zip(factor_names, factor_vector):
                factor_records.append({
                    "asof_date": date, "factor": name,
                    "factor_type": name.split("_", 1)[0], "factor_return": value,
                    "cross_section_size": len(cross), "r_squared": fitted["r_squared"],
                    "residual_std": fitted["residual_std"],
                })
            covariance = ewma_covariance(factor_history, cfg)
            matured = [
                value for code, value in specific_variance.items()
                if specific_counts.get(code, 0) >= int(cfg["specific_risk"]["minimum_observations"])
                and np.isfinite(value)
            ]
            default_specific = float(np.median(matured)) if matured else float(
                np.var(cross["residual_return"], ddof=1)
            )
            if year >= start_attr:
                benchmark_industry = fitted["benchmark_industry"]
                universe_codes = cross["instrument_id"].astype(str).to_numpy()
                universe_specific = np.array([
                    specific_variance.get(code, default_specific) for code in universe_codes
                ])
                universe_weight = 1.0 / len(cross)
                cross_index = cross.set_index("instrument_id")
                keys_for_date = [key for key in holding_groups if key[0] == date]
                for key in keys_for_date:
                    _, model, pipeline, strategy = key
                    target_holdings = holding_groups[key]
                    selected = target_holdings[["instrument_id", "target_weight"]].merge(
                        cross, on="instrument_id", how="left", validate="one_to_one"
                    )
                    if selected[str(cfg["target"])].isna().any():
                        raise RuntimeError(f"attribution universe misses holdings on {key}")
                    weights = selected["target_weight"].to_numpy(dtype="float64")
                    portfolio_industry = selected.groupby("industry_broad_code")[
                        "target_weight"
                    ].sum().to_dict()
                    industry_contribution = sum(
                        portfolio_industry.get(item, 0.0) * active_industry.get(item, 0.0)
                        for item in industry_categories
                    )
                    portfolio_styles = {
                        item: float(np.dot(weights, selected[f"style_{item}"]))
                        for item in styles
                    }
                    style_contributions = {
                        item: portfolio_styles[item] * fitted["style_returns"].get(item, 0.0)
                        for item in styles
                    }
                    style_total = float(sum(style_contributions.values()))
                    specific = float(np.dot(weights, selected["residual_return"]))
                    gross = float(np.dot(weights, selected[str(cfg["target"])]))
                    benchmark = float(fitted["market_return"])
                    active_gross = gross - benchmark
                    weekly_row = weekly_lookup.loc[(date, model, pipeline, strategy)]
                    transaction_cost = float(weekly_row["transaction_cost"])
                    reported_net = float(weekly_row["net_return"])
                    reconstructed = benchmark + industry_contribution + style_total + specific
                    attribution_error = gross - reconstructed
                    record = {
                        "asof_date": date, "test_year": year, "model": model,
                        "alpha_pipeline": pipeline, "strategy": strategy,
                        "market_contribution": benchmark,
                        "industry_active_contribution": industry_contribution,
                        "style_contribution": style_total,
                        "specific_selection_contribution": specific,
                        "transaction_cost_contribution": -transaction_cost,
                        "portfolio_gross_return": gross,
                        "portfolio_net_return": gross - transaction_cost,
                        "benchmark_return": benchmark,
                        "active_gross_return": active_gross,
                        "active_net_return": active_gross - transaction_cost,
                        "reported_net_return": reported_net,
                        "attribution_error": attribution_error,
                        "reported_net_error": reported_net - (gross - transaction_cost),
                    }
                    record.update({
                        f"style_{item}_contribution": style_contributions[item]
                        for item in styles
                    })
                    attribution_records.append(record)
                    for item in industry_categories:
                        exposure_records.append({
                            "asof_date": date, "model": model,
                            "alpha_pipeline": pipeline, "strategy": strategy,
                            "factor": f"industry_{item}",
                            "portfolio_exposure": portfolio_industry.get(item, 0.0),
                            "benchmark_exposure": benchmark_industry.get(item, 0.0),
                            "active_exposure": (
                                portfolio_industry.get(item, 0.0)
                                - benchmark_industry.get(item, 0.0)
                            ),
                        })
                    for item in styles:
                        exposure_records.append({
                            "asof_date": date, "model": model,
                            "alpha_pipeline": pipeline, "strategy": strategy,
                            "factor": f"style_{item}",
                            "portfolio_exposure": portfolio_styles[item],
                            "benchmark_exposure": 0.0,
                            "active_exposure": portfolio_styles[item],
                        })
                    if covariance is not None:
                        portfolio_exposure = np.array([
                            1.0, *[portfolio_industry.get(item, 0.0)
                                  for item in industry_categories],
                            *[portfolio_styles[item] for item in styles],
                        ])
                        active_exposure = np.array([
                            0.0, *[(portfolio_industry.get(item, 0.0)
                                   - benchmark_industry.get(item, 0.0))
                                  for item in industry_categories],
                            *[portfolio_styles[item] for item in styles],
                        ])
                        portfolio_factor_var = float(portfolio_exposure @ covariance @ portfolio_exposure)
                        active_factor_var = float(active_exposure @ covariance @ active_exposure)
                        selected_specific = np.array([
                            specific_variance.get(str(code), default_specific)
                            for code in selected["instrument_id"]
                        ])
                        portfolio_specific_var = float(np.sum(weights ** 2 * selected_specific))
                        portfolio_weight_map = dict(zip(
                            selected["instrument_id"].astype(str), weights
                        ))
                        deltas = np.array([
                            portfolio_weight_map.get(code, 0.0) - universe_weight
                            for code in universe_codes
                        ])
                        active_specific_var = float(np.sum(deltas ** 2 * universe_specific))
                        annualization = int(cfg["annualization_periods"])
                        risk_records.append({
                            "asof_date": date, "model": model,
                            "alpha_pipeline": pipeline, "strategy": strategy,
                            "factor_history_weeks": len(factor_history),
                            "portfolio_factor_variance": portfolio_factor_var,
                            "portfolio_specific_variance": portfolio_specific_var,
                            "active_factor_variance": active_factor_var,
                            "active_specific_variance": active_specific_var,
                            "forecast_annualized_volatility": np.sqrt(max(
                                portfolio_factor_var + portfolio_specific_var, 0.0
                            ) * annualization),
                            "forecast_annualized_tracking_error": np.sqrt(max(
                                active_factor_var + active_specific_var, 0.0
                            ) * annualization),
                            "portfolio_factor_risk_fraction": portfolio_factor_var / max(
                                portfolio_factor_var + portfolio_specific_var, 1e-18
                            ),
                            "active_factor_risk_fraction": active_factor_var / max(
                                active_factor_var + active_specific_var, 1e-18
                            ),
                        })
            factor_history.append(factor_vector)
            for row in cross[["instrument_id", "residual_return"]].itertuples(index=False):
                code, residual = str(row.instrument_id), float(row.residual_return)
                previous = specific_variance.get(code, residual * residual)
                specific_variance[code] = (
                    specific_decay * previous + (1.0 - specific_decay) * residual * residual
                )
                specific_counts[code] = specific_counts.get(code, 0) + 1
        print(f"risk attribution year={year} dates={frame['asof_date'].nunique()}", flush=True)
    factor_returns = pd.DataFrame(factor_records)
    attribution = pd.DataFrame(attribution_records)
    risk = pd.DataFrame(risk_records)
    exposures = pd.DataFrame(exposure_records)
    bootstrap = _bootstrap(attribution, cfg)
    summary_records = []
    for keys, group in attribution.groupby(
        ["model", "alpha_pipeline", "strategy"], sort=False
    ):
        performance = daily_performance_summary(
            group[["portfolio_net_return"]].rename(
                columns={"portfolio_net_return": "net_return"}
            ).assign(
                turnover=0.0, blocked_buy_weight=0.0, blocked_sell_weight=0.0,
                target_tracking_l1=0.0, cash_weight=0.0,
            ),
            int(cfg["annualization_periods"]),
        )
        group_risk = risk.loc[
            risk["model"].eq(keys[0]) & risk["alpha_pipeline"].eq(keys[1])
            & risk["strategy"].eq(keys[2])
        ]
        summary_records.append({
            "model": keys[0], "alpha_pipeline": keys[1], "strategy": keys[2],
            "periods": len(group), "annualized_return": performance["annualized_return"],
            "annualized_volatility": performance["annualized_volatility"],
            "sharpe_zero_rate": performance["sharpe_zero_rate"],
            "maximum_drawdown": performance["maximum_drawdown"],
            "annualized_market_contribution_arithmetic": group["market_contribution"].mean() * 52,
            "annualized_industry_active_contribution_arithmetic": group[
                "industry_active_contribution"
            ].mean() * 52,
            "annualized_style_contribution_arithmetic": group["style_contribution"].mean() * 52,
            **{
                f"annualized_style_{item}_contribution_arithmetic": group[
                    f"style_{item}_contribution"
                ].mean() * 52
                for item in styles
            },
            "annualized_specific_selection_contribution_arithmetic": group[
                "specific_selection_contribution"
            ].mean() * 52,
            "annualized_transaction_cost_arithmetic": group[
                "transaction_cost_contribution"
            ].mean() * 52,
            "mean_forecast_annualized_volatility": group_risk[
                "forecast_annualized_volatility"
            ].mean(),
            "mean_forecast_annualized_tracking_error": group_risk[
                "forecast_annualized_tracking_error"
            ].mean(),
            "mean_portfolio_factor_risk_fraction": group_risk[
                "portfolio_factor_risk_fraction"
            ].mean(),
            "maximum_abs_attribution_error": group["attribution_error"].abs().max(),
            "maximum_abs_reported_net_error": group["reported_net_error"].abs().max(),
        })
    summary = pd.DataFrame(summary_records)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "historical_oos_used_for_selection": False,
        "source_experiment": cfg["source_experiment"],
        "factor_covariance_uses_current_or_future_return": False,
        "attribution_identity_required": True,
    }])
    request = cfg
    outputs = {
        "risk_model_protocol": protocol,
        "risk_model_factor_returns": factor_returns,
        "risk_model_portfolio_attribution": attribution,
        "risk_model_portfolio_exposures": exposures,
        "risk_model_portfolio_forecasts": risk,
        "risk_model_attribution_summary": summary,
        "risk_model_attribution_bootstrap": bootstrap,
    }
    for table, output in outputs.items():
        lake.write(table, output, key="v1",
                   source="multi_factor_risk_attribution.v1", request=request)
    _write_report(summary, bootstrap)
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.loc[
        bootstrap["metric"].eq("specific_selection_contribution")
    ].to_string(index=False), flush=True)
    return summary, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT multi-factor risk and attribution")
    parser.add_argument("command", choices=["build-beta", "run", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in ("build-beta", "all"):
        build_beta_exposures(overwrite=args.overwrite)
    if args.command in ("run", "all"):
        run()


if __name__ == "__main__":
    main()
