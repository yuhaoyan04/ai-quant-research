"""Validation-only ablation of cross-sectional factor preprocessing choices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import yaml

from quant_research.feature_analysis import rank_zscore
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import SufficientStatistics, fit_ridge, purged_end_date, summarize, weekly_rows
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config


CONFIG_PATH = ROOT / "config" / "factor_preprocessing_ablation.yaml"


@dataclass
class MutableStatistics:
    gram: np.ndarray
    cross: np.ndarray
    observations: int = 0
    dates: int = 0

    @classmethod
    def create(cls, dimension: int) -> "MutableStatistics":
        return cls(np.zeros((dimension, dimension)), np.zeros(dimension))

    def add(self, design: np.ndarray, target: np.ndarray) -> None:
        self.gram += design.T @ design
        self.cross += design.T @ target
        self.observations += len(target)
        self.dates += 1

    def freeze(self) -> SufficientStatistics:
        return SufficientStatistics(self.gram, self.cross, self.observations, self.dates)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def variant_id(winsor_method: str, normalization_method: str) -> str:
    return f"winsor_{winsor_method}__norm_{normalization_method}"


def configured_variants(cfg: dict) -> list[tuple[str, str, str]]:
    return [
        (variant_id(winsor, normalization), winsor, normalization)
        for winsor, normalization in product(
            cfg["winsor_methods"], cfg["normalization_methods"]
        )
    ]


def _standardize(values: pd.DataFrame) -> pd.DataFrame:
    standard_deviation = values.std(axis=0, ddof=1).where(lambda item: item > 0)
    return values.subtract(values.mean(axis=0)).divide(standard_deviation).fillna(0.0)


def transform_cross_section(
    group: pd.DataFrame,
    factors: list[str],
    *,
    winsor_method: str,
    normalization_method: str,
    quantile_limits: tuple[float, float] = (0.01, 0.99),
    mad_multiplier: float = 5.0,
    include_missing_indicators: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Transform one date only; no statistic can cross a date or split boundary."""
    raw = group[factors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = raw.isna()
    median = raw.median(axis=0)
    if winsor_method == "none":
        transformed = raw.copy()
    elif winsor_method == "quantile":
        lower = raw.quantile(float(quantile_limits[0]))
        upper = raw.quantile(float(quantile_limits[1]))
        transformed = raw.clip(lower=lower, upper=upper, axis=1)
    elif winsor_method == "mad":
        mad = raw.subtract(median).abs().median(axis=0)
        scale = (1.4826 * mad).where(lambda item: item > 0)
        transformed = raw.clip(
            lower=median - mad_multiplier * scale,
            upper=median + mad_multiplier * scale,
            axis=1,
        )
    else:
        raise ValueError(f"unknown winsor method: {winsor_method}")

    clipped = raw.notna() & transformed.ne(raw)
    filled = transformed.fillna(median).fillna(0.0)
    if normalization_method == "raw":
        exposures = filled
    elif normalization_method == "zscore":
        exposures = _standardize(filled)
    elif normalization_method == "rank_zscore":
        exposures = _standardize(filled.rank(method="average", pct=True))
    else:
        raise ValueError(f"unknown normalization method: {normalization_method}")

    blocks = [exposures.to_numpy(dtype="float64")]
    if include_missing_indicators:
        blocks.append(_standardize(missing.astype("float64")).to_numpy(dtype="float64"))
    design = np.concatenate(blocks, axis=1)
    if not np.isfinite(design).all():
        raise ValueError("preprocessing ablation produced non-finite design values")

    audit = pd.DataFrame({
        "factor": factors,
        "coverage": raw.notna().mean(axis=0).to_numpy(),
        "clipped_fraction": clipped.mean(axis=0).to_numpy(),
        "transformed_mean": exposures.mean(axis=0).to_numpy(),
        "transformed_std": exposures.std(axis=0, ddof=1).to_numpy(),
        "transformed_p01": exposures.quantile(0.01).to_numpy(),
        "transformed_p99": exposures.quantile(0.99).to_numpy(),
    })
    return design, audit


def _rank_ic(score: np.ndarray, target: pd.Series) -> float:
    return float(pd.Series(score).rank(method="average").corr(target.reset_index(drop=True).rank(method="average")))


def _update_portfolio(
    group: pd.DataFrame,
    score: np.ndarray,
    target: str,
    top_n: int,
    cost_bps: float,
    minimum_realized_weight_fraction: float,
    state: dict[str, dict[str, float]],
) -> dict:
    ranked = group.assign(_score=score).sort_values(
        ["_score", "instrument_id"], ascending=[False, True]
    )
    selected = ranked.head(top_n)
    weight = 1.0 / len(selected)
    current = dict(zip(selected["instrument_id"].astype(str), [weight] * len(selected)))
    previous_weights = state.get("weights", {})
    previous_returns = state.get("returns", {})
    if previous_weights:
        drifted = {
            code: old_weight * (1.0 + previous_returns.get(code, 0.0))
            for code, old_weight in previous_weights.items()
        }
        total = sum(drifted.values())
        drifted = ({code: value / total for code, value in drifted.items()}
                   if total > 0 else previous_weights)
        turnover = 0.5 * sum(
            abs(current.get(code, 0.0) - drifted.get(code, 0.0))
            for code in set(current) | set(drifted)
        )
    else:
        turnover = 1.0
    realized = pd.to_numeric(selected[target], errors="coerce")
    gross_return = float(realized.fillna(0.0).mean())
    benchmark_return = float(pd.to_numeric(group[target], errors="coerce").fillna(0.0).mean())
    realized_fraction = float(realized.notna().mean())
    transaction_cost = turnover * cost_bps / 10_000
    state["weights"] = current
    state["returns"] = dict(zip(selected["instrument_id"].astype(str), realized.fillna(0.0)))
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
        "included_in_performance": realized_fraction >= minimum_realized_weight_fraction,
    }


def annual_summaries(daily: pd.DataFrame, portfolio_daily: pd.DataFrame,
                     annualization_periods: int) -> pd.DataFrame:
    """Create year-by-year stability evidence without refitting or reselection."""
    records: list[dict] = []
    dated_daily = daily.assign(year=daily["asof_date"].str.slice(0, 4).astype(int))
    dated_portfolio = portfolio_daily.assign(
        year=portfolio_daily["asof_date"].str.slice(0, 4).astype(int)
    )
    for (variant, penalty, year), group in dated_daily.groupby(
        ["variant", "ridge_penalty", "year"], sort=False
    ):
        portfolio_group = dated_portfolio.loc[
            dated_portfolio["variant"].eq(variant)
            & dated_portfolio["ridge_penalty"].eq(penalty)
            & dated_portfolio["year"].eq(year)
        ]
        model_summary = summarize(group, annualization_periods)
        portfolio_summary = performance_summary(portfolio_group, annualization_periods)
        records.append({
            "variant": variant,
            "ridge_penalty": penalty,
            "year": year,
            **model_summary,
            **{f"portfolio_{key}": value for key, value in portfolio_summary.items()},
        })
    return pd.DataFrame(records)


def _read_year(lake: Lake, year: int, columns: list[str]) -> pd.DataFrame:
    path = lake.table_path("eligible_factor_panel_staging", str(year))
    if not path.exists():
        raise FileNotFoundError(f"staged eligible panel missing for {year}: {path}")
    return weekly_rows(pd.read_parquet(path, columns=columns))


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    artifact_key = str(cfg["experiment_id"]).rsplit("_", 1)[-1]
    source = f"factor_preprocessing_ablation.{artifact_key}"
    preprocessing_cfg = load_preprocessing_config()
    factors = list(preprocessing_cfg["factor_columns"])
    target = str(cfg["target"])
    include_missing = bool(cfg["include_missing_indicators"])
    dimension = len(factors) * (2 if include_missing else 1)
    variants = configured_variants(cfg)
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    minimum_size = int(cfg["minimum_cross_section_size"])
    transform_args = {
        "quantile_limits": tuple(cfg["quantile_limits"]),
        "mad_multiplier": float(cfg["mad_multiplier"]),
        "include_missing_indicators": include_missing,
    }

    accumulators = {name: MutableStatistics.create(dimension) for name, _, _ in variants}
    train_start, train_end = int(cfg["train"]["start_year"]), int(cfg["train"]["end_year"])
    train_cutoff = purged_end_date(calendar, f"{train_end}-12-31", int(cfg["label_horizon_trading_days"]))
    for year in range(train_start, train_end + 1):
        frame = _read_year(lake, year, columns)
        if year == train_end:
            frame = frame.loc[frame["asof_date"].le(train_cutoff)]
        for _, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < minimum_size:
                continue
            y = rank_zscore(group.loc[usable, target])
            for name, winsor, normalization in variants:
                design, _ = transform_cross_section(
                    group, factors, winsor_method=winsor,
                    normalization_method=normalization, **transform_args,
                )
                accumulators[name].add(design[usable], y)
        print(f"ablation training year={year} weekly_dates={frame['asof_date'].nunique()}", flush=True)

    penalties = [float(item) for item in cfg["ridge_penalties_per_observation"]]
    betas = {
        (name, penalty): fit_ridge(accumulators[name].freeze(), penalty)
        for name, _, _ in variants for penalty in penalties
    }
    daily_records: list[dict] = []
    factor_records: list[dict] = []
    portfolio_records: list[dict] = []
    portfolio_states: dict[tuple[str, float], dict] = {
        (name, penalty): {} for name, _, _ in variants for penalty in penalties
    }
    validation_start = int(cfg["validation"]["start_year"])
    validation_end = int(cfg["validation"]["end_year"])
    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    for year in range(validation_start, validation_end + 1):
        frame = _read_year(lake, year, columns)
        if year == validation_end:
            frame = frame.loc[frame["asof_date"].le(validation_cutoff)]
        for asof_date, group in frame.groupby("asof_date", sort=False):
            usable = group[target].notna().to_numpy()
            if usable.sum() < minimum_size:
                continue
            evaluation_group = group.loc[usable].reset_index(drop=True)
            realized = evaluation_group[target]
            for name, winsor, normalization in variants:
                full_design, audit = transform_cross_section(
                    group, factors, winsor_method=winsor,
                    normalization_method=normalization, **transform_args,
                )
                design = full_design[usable]
                audit["asof_date"] = asof_date
                audit["variant"] = name
                for factor_index, factor in enumerate(factors):
                    audit.loc[audit["factor"].eq(factor), "rank_ic"] = _rank_ic(
                        design[:, factor_index], realized
                    )
                factor_records.extend(audit.to_dict(orient="records"))
                for penalty in penalties:
                    score = design @ betas[(name, penalty)]
                    rank_ic = _rank_ic(score, realized)
                    percentile = pd.Series(score).rank(method="first", pct=True)
                    bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
                    bucket_returns = realized.groupby(bucket.to_numpy()).mean()
                    daily_records.append({
                        "asof_date": asof_date,
                        "variant": name,
                        "ridge_penalty": penalty,
                        "cross_section_size": len(realized),
                        "rank_ic": rank_ic,
                        "top_minus_bottom": bucket_returns.get(9, np.nan) - bucket_returns.get(0, np.nan),
                    })
                    portfolio = _update_portfolio(
                        group.reset_index(drop=True),
                        full_design @ betas[(name, penalty)],
                        target, int(cfg["top_n"]),
                        float(cfg["round_trip_cost_bps"]),
                        float(cfg["minimum_realized_weight_fraction"]),
                        portfolio_states[(name, penalty)],
                    )
                    portfolio.update({"variant": name, "ridge_penalty": penalty})
                    portfolio_records.append(portfolio)
        print(f"ablation validation year={year} weekly_dates={frame['asof_date'].nunique()}", flush=True)

    daily = pd.DataFrame(daily_records)
    portfolio_daily = pd.DataFrame(portfolio_records)
    grid_records = []
    for (name, penalty), group in daily.groupby(["variant", "ridge_penalty"], sort=False):
        model_summary = summarize(group, int(cfg["annualization_periods"]))
        portfolio_group = portfolio_daily.loc[
            portfolio_daily["variant"].eq(name)
            & portfolio_daily["ridge_penalty"].eq(penalty)
        ]
        portfolio_summary = performance_summary(portfolio_group, int(cfg["annualization_periods"]))
        grid_records.append({
            "variant": name,
            "winsor_method": name.split("__")[0].removeprefix("winsor_"),
            "normalization_method": name.split("__")[1].removeprefix("norm_"),
            "ridge_penalty": penalty,
            **model_summary,
            **{f"portfolio_{key}": value for key, value in portfolio_summary.items()},
        })
    grid = pd.DataFrame(grid_records).sort_values(
        [str(cfg["selection_metric"]), *cfg["selection_tiebreakers"]], ascending=False
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    selection = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"],
        "selected_variant": winner["variant"],
        "selected_ridge_penalty": winner["ridge_penalty"],
        "validation_mean_rank_ic": winner["mean_rank_ic"],
        "validation_annualized_icir": winner["annualized_icir"],
        "selection_period": cfg["selection_period"],
        "historical_oos_used_for_selection": False,
    }])
    factor_daily = pd.DataFrame(factor_records)
    factor_summary = factor_daily.groupby(["variant", "factor"], as_index=False).agg(
        observations=("rank_ic", "size"),
        mean_rank_ic=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        positive_rank_ic_fraction=("rank_ic", lambda values: float(values.gt(0).mean())),
        mean_coverage=("coverage", "mean"),
        mean_clipped_fraction=("clipped_fraction", "mean"),
        mean_transformed_std=("transformed_std", "mean"),
        mean_transformed_p01=("transformed_p01", "mean"),
        mean_transformed_p99=("transformed_p99", "mean"),
    )
    factor_summary["annualized_icir"] = (
        factor_summary["mean_rank_ic"] / factor_summary["rank_ic_std"]
        * np.sqrt(int(cfg["annualization_periods"]))
    )
    annual = annual_summaries(daily, portfolio_daily, int(cfg["annualization_periods"]))

    request = {**cfg, "factor_columns": factors}
    lake.write("factor_preprocessing_ablation_grid", grid, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_selection", selection, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_daily", daily, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_portfolio", portfolio_daily, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_factor_daily", factor_daily, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_factor_summary", factor_summary, key=artifact_key,
               source=source, request=request)
    lake.write("factor_preprocessing_ablation_annual", annual, key=artifact_key,
               source=source, request=request)
    print(selection.to_string(index=False))
    print(grid.head(12).to_string(index=False))


if __name__ == "__main__":
    run()
