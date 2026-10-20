"""Phase 3 model stability, time-aware windows, and supplemental purged CV."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.factor_preprocessing_ablation import _rank_ic, _update_portfolio
from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import purged_end_date, sum_statistics, summarize
from quant_research.model_benchmark import (
    FittedCandidate, _read_year, _transform, build_year_training, fit_candidate,
)
from quant_research.model_challenger import TrainingSample, concatenate_samples
from quant_research.portfolio import performance_summary
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.robustness import circular_block_bootstrap_mean


CONFIG_PATH = ROOT / "config" / "model_stability_validation.yaml"
REPORT_PATH = ROOT / "reports" / "model_stability_validation.md"
TABLE_ROOT = ROOT / "reports" / "tables"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protocol(cfg: dict[str, Any]) -> None:
    """Reject any configuration that could silently turn Phase 3 into reselection."""
    if cfg.get("historical_oos_used_for_selection") is not False:
        raise ValueError("historical OOS must not be used for model selection")
    if cfg.get("parameter_promotion_allowed") is not False:
        raise ValueError("Phase 3 cannot promote parameters")
    cv = cfg["purged_cv"]
    if cv.get("promotion_allowed") is not False:
        raise ValueError("purged CV cannot promote a model")
    if cv.get("label_interval_purge_required") is not True:
        raise ValueError("purged CV must purge overlapping label intervals")
    if cv.get("future_training_samples_allowed") is not True:
        raise ValueError("this supplemental CV protocol must disclose future training samples")
    policies = list(cfg["window_comparison"]["policies"])
    if policies != ["expanding", "rolling_10y"]:
        raise ValueError("window policies must remain the preregistered pair")
    candidates = list(cfg["candidates"])
    families = [str(item["family"]) for item in candidates]
    if families != ["naive", "ridge", "elastic_net", "random_forest", "hgb"]:
        raise ValueError("candidate families or order differ from the frozen protocol")
    if len({str(item["candidate"]) for item in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")


def window_years(test_year: int, policy: str, cfg: dict[str, Any]) -> list[int]:
    start = int(cfg["window_comparison"]["expanding_start_year"])
    if policy == "expanding":
        first = start
    elif policy == "rolling_10y":
        first = max(start, test_year - int(cfg["window_comparison"]["rolling_years"]))
    else:
        raise ValueError(f"unknown window policy: {policy}")
    return list(range(first, test_year))


def open_trading_days(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    values = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
    ).sort_values().drop_duplicates()
    return pd.DatetimeIndex(values)


def label_maturity_date(asof_date: str, horizon: int,
                        trading_days: pd.DatetimeIndex) -> str:
    asof = pd.Timestamp(asof_date)
    position = int(trading_days.searchsorted(asof, side="left"))
    if position >= len(trading_days) or trading_days[position] != asof:
        raise ValueError(f"as-of date is not an open trading day: {asof_date}")
    maturity_position = position + int(horizon)
    if maturity_position >= len(trading_days):
        raise ValueError(f"calendar does not cover label maturity: {asof_date}")
    return trading_days[maturity_position].strftime("%Y-%m-%d")


def label_interval_is_disjoint(asof_date: str, maturity_date: str,
                               test_start: str, test_end_maturity: str) -> bool:
    """Return true only when a training label interval cannot touch the test interval."""
    return str(maturity_date) < str(test_start) or str(asof_date) > str(test_end_maturity)


def _model_version(kind: str) -> str:
    if kind in {"elastic_net", "random_forest", "hgb"}:
        return f"scikit-learn-{importlib.metadata.version('scikit-learn')}"
    return f"numpy-{importlib.metadata.version('numpy')}"


def _parameter_json(spec: dict[str, Any]) -> str:
    omitted = {"candidate", "family", "kind"}
    return json.dumps(
        {key: value for key, value in spec.items() if key not in omitted},
        sort_keys=True, separators=(",", ":"),
    )


def _score_hash(group: pd.DataFrame, scores: np.ndarray) -> bytes:
    identity = pd.util.hash_pandas_object(
        group[["instrument_id", "asof_date"]], index=False
    ).to_numpy(dtype="uint64")
    values = np.asarray(scores, dtype="<f8")
    return identity.tobytes() + values.tobytes()


def _evaluate_models(
    frame: pd.DataFrame,
    models: dict[str, FittedCandidate],
    metadata: dict[str, dict[str, str]],
    factors: list[str],
    target: str,
    states: dict[str, dict],
    cfg: dict[str, Any],
    split: str,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    daily_records: list[dict[str, Any]] = []
    portfolio_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    hashers = {name: hashlib.sha256() for name in models}
    quantiles = np.asarray(cfg["score_distribution_quantiles"], dtype="float64")
    for asof_date, group in frame.groupby("asof_date", sort=True):
        group = group.reset_index(drop=True)
        design = _transform(group, factors, cfg)
        realized = pd.to_numeric(group[target], errors="coerce")
        usable = realized.notna().to_numpy()
        if usable.sum() < int(cfg["minimum_cross_section_size"]):
            continue
        for strategy, model in models.items():
            meta = metadata[strategy]
            score = model.predict(design)
            hashers[strategy].update(_score_hash(group, score))
            usable_return = realized.loc[usable].reset_index(drop=True)
            usable_score = score[usable]
            percentile = pd.Series(usable_score).rank(method="first", pct=True)
            bucket = (percentile * 10).clip(upper=10 - 1e-9).astype(int)
            bucket_return = usable_return.groupby(bucket.to_numpy()).mean()
            common = {
                "asof_date": str(asof_date), "split": split,
                "test_year": int(test_year), **meta,
            }
            daily_records.append({
                **common, "cross_section_size": int(usable.sum()),
                "rank_ic": _rank_ic(usable_score, usable_return),
                "top_minus_bottom": (
                    bucket_return.get(9, np.nan) - bucket_return.get(0, np.nan)
                ),
            })
            distribution = np.quantile(usable_score, quantiles)
            score_records.append({
                **common, "score_count": int(len(usable_score)),
                "score_mean": float(np.mean(usable_score)),
                "score_std": float(np.std(usable_score, ddof=1)),
                **{
                    f"score_p{int(round(q * 100)):02d}": float(value)
                    for q, value in zip(quantiles, distribution)
                },
            })
            portfolio = _update_portfolio(
                group, score, target, int(cfg["portfolio"]["top_n"]),
                float(cfg["portfolio"]["round_trip_cost_bps"]),
                float(cfg["portfolio"]["minimum_realized_weight_fraction"]),
                states[strategy],
            )
            portfolio.update(common)
            portfolio_records.append(portfolio)
    return (
        pd.DataFrame(daily_records), pd.DataFrame(portfolio_records),
        pd.DataFrame(score_records),
        {name: digest.hexdigest() for name, digest in hashers.items()},
    )


def _annual_metrics(daily: pd.DataFrame, portfolio: pd.DataFrame,
                    cfg: dict[str, Any]) -> pd.DataFrame:
    periods = int(cfg["annualization_periods"])
    rows: list[dict[str, Any]] = []
    keys = ["split", "window_policy", "candidate", "family", "kind", "test_year"]
    for values, group in daily.groupby(keys, dropna=False, sort=True):
        selector = pd.Series(True, index=portfolio.index)
        for key, value in zip(keys, values):
            selector &= portfolio[key].fillna("").astype(str).eq("" if pd.isna(value) else str(value))
        holdings = portfolio.loc[selector]
        performance = performance_summary(holdings, periods)
        included = holdings.loc[holdings["included_in_performance"]]
        rows.append({
            **dict(zip(keys, values)), **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance.items()},
            "portfolio_mean_transaction_cost": float(included["transaction_cost"].mean()),
            "portfolio_total_transaction_cost": float(included["transaction_cost"].sum()),
            "portfolio_annualized_transaction_cost_arithmetic": float(
                included["transaction_cost"].mean() * periods
            ),
            "rank_ic_failure": bool(
                group["rank_ic"].mean() <= float(cfg["failure_rules"]["mean_rank_ic_nonpositive"])
            ),
            "net_return_failure": bool(
                performance["annualized_return"]
                <= float(cfg["failure_rules"]["annualized_net_return_nonpositive"])
            ),
        })
    return pd.DataFrame(rows)


def _overall_metrics(daily: pd.DataFrame, portfolio: pd.DataFrame,
                     cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = int(cfg["annualization_periods"])
    for keys, group in daily.groupby(
        ["split", "window_policy", "candidate", "family", "kind"],
        dropna=False, sort=True,
    ):
        split, policy, candidate, family, kind = keys
        holdings = portfolio.loc[
            portfolio["split"].eq(split)
            & portfolio["window_policy"].fillna("").eq("" if pd.isna(policy) else policy)
            & portfolio["candidate"].eq(candidate)
        ]
        included = holdings.loc[holdings["included_in_performance"]]
        rows.append({
            "split": split, "window_policy": policy, "candidate": candidate,
            "family": family, "kind": kind, **summarize(group, periods),
            **{f"portfolio_{key}": value for key, value in performance_summary(
                holdings, periods
            ).items()},
            "portfolio_mean_transaction_cost": float(included["transaction_cost"].mean()),
            "portfolio_total_transaction_cost": float(included["transaction_cost"].sum()),
            "portfolio_annualized_transaction_cost_arithmetic": float(
                included["transaction_cost"].mean() * periods
            ),
        })
    return pd.DataFrame(rows)


def _alignment_audit(daily: pd.DataFrame, portfolio: pd.DataFrame,
                     expected_candidates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split", "window_policy", "test_year"]
    for values, group in daily.groupby(keys, dropna=False, sort=True):
        split, policy, year = values
        portfolio_group = portfolio.loc[
            portfolio["split"].eq(split)
            & portfolio["test_year"].eq(year)
            & portfolio["window_policy"].fillna("").eq("" if pd.isna(policy) else policy)
        ]
        daily_counts = group.groupby("asof_date")["candidate"].nunique()
        cross_sections = group.groupby("asof_date")["cross_section_size"].nunique()
        portfolio_counts = portfolio_group.groupby("asof_date")["candidate"].nunique()
        rows.append({
            "split": split, "window_policy": policy, "test_year": int(year),
            "candidate_count": int(group["candidate"].nunique()),
            "evaluation_dates": int(group["asof_date"].nunique()),
            "all_candidates_share_dates": bool(
                len(daily_counts) > 0 and daily_counts.eq(expected_candidates).all()
            ),
            "all_candidates_share_cross_section_size": bool(cross_sections.eq(1).all()),
            "all_candidates_share_portfolio_dates": bool(
                len(portfolio_counts) > 0 and portfolio_counts.eq(expected_candidates).all()
            ),
        })
    return pd.DataFrame(rows)


def _bootstrap_difference(values: pd.Series, scale: float,
                          cfg: dict[str, Any]) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    boot = cfg["bootstrap"]
    estimates = circular_block_bootstrap_mean(
        clean.to_numpy(dtype="float64"), int(boot["block_length_weeks"]),
        int(boot["samples"]), int(boot["random_seed"]),
    ) * scale
    return {
        "observations": int(len(clean)),
        "point_estimate": float(clean.mean() * scale),
        "ci_2_5pct": float(np.quantile(estimates, 0.025)),
        "ci_97_5pct": float(np.quantile(estimates, 0.975)),
        "one_sided_probability_nonpositive": float(
            (np.sum(estimates <= 0) + 1) / (len(estimates) + 1)
        ),
        "block_length_weeks": int(boot["block_length_weeks"]),
    }


def _window_pairwise(daily: pd.DataFrame, portfolio: pd.DataFrame,
                     cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "historical_oos"):
        for candidate in sorted(daily["candidate"].unique()):
            for metric, frame, column, scale in (
                ("rank_ic", daily, "rank_ic", 1.0),
                ("net_return_annualized_arithmetic",
                 portfolio.loc[portfolio["included_in_performance"]],
                 "net_return", float(cfg["annualization_periods"])),
            ):
                subset = frame.loc[
                    frame["split"].eq(split) & frame["candidate"].eq(candidate),
                    ["asof_date", "window_policy", column],
                ]
                left = subset.loc[subset["window_policy"].eq("rolling_10y")]
                right = subset.loc[subset["window_policy"].eq("expanding")]
                paired = left.merge(
                    right, on="asof_date", suffixes=("_rolling", "_expanding"),
                    validate="one_to_one",
                )
                difference = paired[f"{column}_rolling"] - paired[f"{column}_expanding"]
                rows.append({
                    "split": split, "candidate": candidate, "metric": metric,
                    "contrast": "rolling_10y_minus_expanding",
                    **_bootstrap_difference(difference, scale, cfg),
                })
    return pd.DataFrame(rows)


def summarize_rf_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    seeds = frame.loc[frame["axis"].isin(["baseline", "random_state"])].copy()
    metrics = ["mean_rank_ic", "portfolio_annualized_return",
               "portfolio_sharpe_zero_rate", "portfolio_maximum_drawdown"]
    return pd.DataFrame([{
        "model": "random_forest_depth6", "metric": metric,
        "seed_count": int(len(seeds)), "mean": float(seeds[metric].mean()),
        "standard_deviation": float(seeds[metric].std(ddof=1)),
        "minimum": float(seeds[metric].min()), "maximum": float(seeds[metric].max()),
        "best_seed_selected": False,
    } for metric in metrics])


def _fit_models(specs: list[dict[str, Any]], statistics: Any,
                sample: TrainingSample, factor_count: int, cfg: dict[str, Any],
                prefix: str | None = None
                ) -> tuple[dict[str, FittedCandidate], dict[str, dict[str, str]]]:
    models: dict[str, FittedCandidate] = {}
    metadata: dict[str, dict[str, str]] = {}
    for spec in specs:
        fitted = fit_candidate(spec, statistics, sample, factor_count, cfg)
        strategy = f"{prefix}__{fitted.candidate}" if prefix else fitted.candidate
        fitted.candidate = strategy
        models[strategy] = fitted
        metadata[strategy] = {
            "window_policy": prefix,
            "candidate": str(spec["candidate"]), "family": str(spec["family"]),
            "kind": str(spec["kind"]),
        }
    return models, metadata


def _run_window_comparison(
    lake: Lake, cfg: dict[str, Any], factors: list[str], columns: list[str],
    calendar: pd.DataFrame, trading_days: pd.DatetimeIndex,
    full_stats: dict[int, Any], purged_stats: dict[int, Any],
    full_samples: dict[int, TrainingSample], purged_samples: dict[int, TrainingSample],
) -> dict[str, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    states = {
        f"{policy}__{spec['candidate']}": {}
        for policy in cfg["window_comparison"]["policies"]
        for spec in cfg["candidates"]
    }
    validation_end = int(cfg["validation"]["end_year"])
    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    for test_year in range(int(cfg["evaluation"]["start_year"]),
                           int(cfg["evaluation"]["end_year"]) + 1):
        models: dict[str, FittedCandidate] = {}
        metadata: dict[str, dict[str, str]] = {}
        pending_rows: dict[str, dict[str, Any]] = {}
        for policy in cfg["window_comparison"]["policies"]:
            years = window_years(test_year, str(policy), cfg)
            end_year = years[-1]
            statistics = sum_statistics(
                [full_stats[year] for year in years[:-1]] + [purged_stats[end_year]]
            )
            sample = concatenate_samples(
                [full_samples[year] for year in years[:-1]] + [purged_samples[end_year]]
            )
            fitted, meta = _fit_models(
                list(cfg["candidates"]), statistics, sample, len(factors), cfg, str(policy)
            )
            models.update(fitted); metadata.update(meta)
            cutoff = purged_end_date(
                calendar, f"{end_year}-12-31", int(cfg["label_horizon_trading_days"])
            )
            maturity = label_maturity_date(
                cutoff, int(cfg["label_horizon_trading_days"]), trading_days
            )
            for strategy, model in fitted.items():
                spec = next(item for item in cfg["candidates"]
                            if item["candidate"] == metadata[strategy]["candidate"])
                pending_rows[strategy] = {
                    "experiment_id": cfg["experiment_id"], "fit_type": "walk_forward",
                    "split": "validation" if test_year <= validation_end else "historical_oos",
                    "test_year": test_year, **metadata[strategy],
                    "training_start_date": f"{years[0]}-01-01",
                    "training_end_asof_date": cutoff,
                    "last_training_label_maturity_date": maturity,
                    "training_year_count": len(years),
                    "fit_observations": model.fit_observations,
                    "fit_seconds": model.fit_seconds,
                    "parameters_json": _parameter_json(spec),
                    "seed": spec.get("random_state", cfg["random_state"])
                    if spec["kind"] in {"random_forest", "hgb", "elastic_net"} else None,
                    "model_version": _model_version(str(spec["kind"])),
                }
        frame = _read_year(lake, test_year, columns)
        if test_year == validation_end:
            frame = frame.loc[frame["asof_date"].le(validation_cutoff)]
        split = "validation" if test_year <= validation_end else "historical_oos"
        daily, portfolio, score, hashes = _evaluate_models(
            frame, models, metadata, factors, str(cfg["target"]), states, cfg,
            split, test_year,
        )
        for strategy, row in pending_rows.items():
            row["prediction_output_sha256"] = hashes[strategy]
            fit_rows.append(row)
        daily_frames.append(daily); portfolio_frames.append(portfolio); score_frames.append(score)
        print(f"phase3 M3 year={test_year} fits={len(models)}", flush=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    return {
        "model_stability_window_daily": daily,
        "model_stability_window_portfolio": portfolio,
        "model_stability_score_distribution": pd.concat(score_frames, ignore_index=True),
        "model_stability_window_fit_audit": pd.DataFrame(fit_rows),
        "model_stability_window_annual": _annual_metrics(daily, portfolio, cfg),
        "model_stability_window_summary": _overall_metrics(daily, portfolio, cfg),
        "model_stability_window_pairwise": _window_pairwise(daily, portfolio, cfg),
    }


def _filtered_boundary_training(
    frame: pd.DataFrame, factors: list[str], cfg: dict[str, Any],
    trading_days: pd.DatetimeIndex, test_start: str, test_end_maturity: str,
) -> tuple[Any, TrainingSample, int, list[str]]:
    horizon = int(cfg["label_horizon_trading_days"])
    eligible_counts = frame.groupby("asof_date")[str(cfg["target"])].count()
    unique_dates = sorted(eligible_counts.loc[
        eligible_counts.ge(int(cfg["minimum_cross_section_size"]))
    ].index.astype(str))
    allowed_dates = [
        date for date in unique_dates
        if label_interval_is_disjoint(
            date, label_maturity_date(date, horizon, trading_days),
            test_start, test_end_maturity,
        )
    ]
    filtered = frame.loc[frame["asof_date"].isin(allowed_dates)]
    stats, _, sample, _ = build_year_training(
        filtered, factors, str(cfg["target"]), "9999-12-31", cfg
    )
    return stats, sample, len(unique_dates) - len(allowed_dates), allowed_dates


def _cv_pairwise(daily: pd.DataFrame, portfolio: pd.DataFrame,
                 cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = "naive_directional"
    for candidate in sorted(set(daily["candidate"]) - {reference}):
        for metric, frame, column, scale in (
            ("rank_ic", daily, "rank_ic", 1.0),
            ("net_return_annualized_arithmetic",
             portfolio.loc[portfolio["included_in_performance"]],
             "net_return", float(cfg["annualization_periods"])),
        ):
            left = frame.loc[frame["candidate"].eq(candidate), ["asof_date", column]]
            right = frame.loc[frame["candidate"].eq(reference), ["asof_date", column]]
            paired = left.merge(
                right, on="asof_date", suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            difference = paired[f"{column}_candidate"] - paired[f"{column}_reference"]
            rows.append({
                "candidate": candidate, "reference": reference, "metric": metric,
                **_bootstrap_difference(difference, scale, cfg),
            })
    return pd.DataFrame(rows)


def _run_purged_cv(
    lake: Lake, cfg: dict[str, Any], factors: list[str], columns: list[str],
    calendar: pd.DataFrame, trading_days: pd.DatetimeIndex,
    full_stats: dict[int, Any], full_samples: dict[int, TrainingSample],
    full_training_dates: dict[int, list[str]],
) -> dict[str, pd.DataFrame]:
    cv = cfg["purged_cv"]
    pool_start, pool_end = int(cv["training_pool_start_year"]), int(cv["training_pool_end_year"])
    daily_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    purge_rows: list[dict[str, Any]] = []
    validation_end = int(cfg["validation"]["end_year"])
    validation_cutoff = purged_end_date(
        calendar, f"{validation_end}-12-31", int(cfg["label_horizon_trading_days"])
    )
    for test_year in cv["test_years"]:
        test_year = int(test_year)
        test_frame = _read_year(lake, test_year, columns)
        if test_year == validation_end:
            test_frame = test_frame.loc[test_frame["asof_date"].le(validation_cutoff)]
        test_dates = sorted(test_frame["asof_date"].astype(str).unique())
        test_start, test_end = test_dates[0], test_dates[-1]
        test_end_maturity = label_maturity_date(
            test_end, int(cfg["label_horizon_trading_days"]), trading_days
        )
        statistics_parts, sample_parts = [], []
        overlap_dates_removed = 0
        training_years = []
        included_training_dates: list[str] = []
        for year in range(pool_start, pool_end + 1):
            if year == test_year:
                continue
            training_years.append(year)
            if year in {test_year - 1, test_year + 1}:
                boundary = _read_year(lake, year, columns)
                stats, sample, removed, allowed_dates = _filtered_boundary_training(
                    boundary, factors, cfg, trading_days, test_start, test_end_maturity
                )
                statistics_parts.append(stats); sample_parts.append(sample)
                overlap_dates_removed += removed
                included_training_dates.extend(allowed_dates)
            else:
                statistics_parts.append(full_stats[year]); sample_parts.append(full_samples[year])
                included_training_dates.extend(full_training_dates[year])
        statistics = sum_statistics(statistics_parts)
        sample = concatenate_samples(sample_parts)
        training_start_asof = min(included_training_dates)
        training_end_asof = max(included_training_dates)
        training_end_maturity = label_maturity_date(
            training_end_asof, int(cfg["label_horizon_trading_days"]), trading_days
        )
        models, metadata = _fit_models(
            list(cfg["candidates"]), statistics, sample, len(factors), cfg
        )
        daily, portfolio, score, hashes = _evaluate_models(
            test_frame, models, metadata, factors, str(cfg["target"]),
            {name: {} for name in models}, cfg, "purged_cv_supplemental", test_year,
        )
        daily_frames.append(daily); portfolio_frames.append(portfolio); score_frames.append(score)
        purge_rows.append({
            "test_year": test_year, "test_start_asof_date": test_start,
            "test_end_asof_date": test_end, "test_end_label_maturity_date": test_end_maturity,
            "training_pool_start_year": pool_start, "training_pool_end_year": pool_end,
            "training_years": ",".join(map(str, training_years)),
            "future_training_samples_used": bool(any(year > test_year for year in training_years)),
            "overlapping_training_dates_removed": overlap_dates_removed,
            "test_dates": len(test_dates), "portfolio_state_reset_at_fold_start": True,
        })
        for strategy, model in models.items():
            spec = next(item for item in cfg["candidates"]
                        if item["candidate"] == metadata[strategy]["candidate"])
            fit_rows.append({
                "experiment_id": cfg["experiment_id"], "fit_type": "purged_cv_supplemental",
                "split": "validation", "test_year": test_year, **metadata[strategy],
                "training_start_date": training_start_asof,
                "training_end_asof_date": training_end_asof,
                "last_training_label_maturity_date": training_end_maturity,
                "training_year_count": len(training_years),
                "fit_observations": model.fit_observations, "fit_seconds": model.fit_seconds,
                "parameters_json": _parameter_json(spec),
                "seed": spec.get("random_state", cfg["random_state"])
                if spec["kind"] in {"random_forest", "hgb", "elastic_net"} else None,
                "model_version": _model_version(str(spec["kind"])),
                "prediction_output_sha256": hashes[strategy],
            })
        print(f"phase3 M5 fold={test_year} removed_overlap_dates={overlap_dates_removed}",
              flush=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    return {
        "model_stability_purged_cv_daily": daily,
        "model_stability_purged_cv_portfolio": portfolio,
        "model_stability_purged_cv_score_distribution": pd.concat(score_frames, ignore_index=True),
        "model_stability_purged_cv_fit_audit": pd.DataFrame(fit_rows),
        "model_stability_purged_cv_purge_audit": pd.DataFrame(purge_rows),
        "model_stability_purged_cv_annual": _annual_metrics(daily, portfolio, cfg),
        "model_stability_purged_cv_summary": _overall_metrics(daily, portfolio, cfg),
        "model_stability_purged_cv_pairwise": _cv_pairwise(daily, portfolio, cfg),
    }


def _evidence_tables(lake: Lake) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rf = lake.read("rf_hyperparameter_validation_summary", "v1")
    hgb = lake.read("hgb_hyperparameter_summary", "v1")
    rf_protocol = lake.read("rf_hyperparameter_protocol", "v1").iloc[0]
    incremental = lake.read("incremental_alpha_comparison", "v1")
    evidence = pd.DataFrame([
        {"experiment": "M0", "status": "COMPLETE", "evidence": "model_benchmark_v1",
         "interpretation": "validation-selected frozen family baselines"},
        {"experiment": "M1", "status": "COMPLETE_WITH_INSTABILITY",
         "evidence": "hgb_hyperparameter_stability_v1; rf_hyperparameter_stability_v1",
         "interpretation": (
             f"RF neighbor pass fraction {float(rf_protocol['neighbor_pass_fraction']):.3f} "
             f"vs floor {float(rf_protocol['minimum_neighbor_pass_fraction']):.3f}; "
             f"HGB RankIC range {hgb['mean_rank_ic'].min():.4f}..{hgb['mean_rank_ic'].max():.4f}"
         )},
        {"experiment": "M2", "status": "COMPLETE", "evidence": "three preregistered RF seeds",
         "interpretation": "mean/std/range reported; no best-seed selection"},
        {"experiment": "M3", "status": "COMPLETE", "evidence": "expanding vs rolling_10y",
         "interpretation": "paired validation and descriptive historical-OOS comparison"},
        {"experiment": "M4", "status": "COMPLETE", "evidence": "annual RankIC and portfolio metrics",
         "interpretation": "nonpositive RankIC/return years explicitly flagged"},
        {"experiment": "M5", "status": "COMPLETE_SUPPLEMENTAL",
         "evidence": "five label-interval-purged validation-year folds",
         "interpretation": "noncausal generalization diagnostic; not trading performance"},
        {"experiment": "M6", "status": "COMPLETE_MIXED", "evidence": "incremental_alpha_test_v1",
         "interpretation": (
             f"positive significant raw intervals="
             f"{int(incremental['incremental_alpha_significant_positive'].sum())}/"
             f"{len(incremental)}; no model promotion"
         )},
    ])
    hyperparameters = pd.DataFrame([
        {
            "family": "hgb", "candidate_count": len(hgb),
            "mean_rank_ic_minimum": hgb["mean_rank_ic"].min(),
            "mean_rank_ic_maximum": hgb["mean_rank_ic"].max(),
            "mean_rank_ic_range": hgb["mean_rank_ic"].max() - hgb["mean_rank_ic"].min(),
            "portfolio_sharpe_minimum": hgb["portfolio_sharpe_zero_rate"].min(),
            "portfolio_sharpe_maximum": hgb["portfolio_sharpe_zero_rate"].max(),
            "local_robustness_pass": None,
            "interpretation": "RankIC surface narrow; portfolio outcomes materially wider",
        },
        {
            "family": "random_forest", "candidate_count": len(rf),
            "mean_rank_ic_minimum": rf["mean_rank_ic"].min(),
            "mean_rank_ic_maximum": rf["mean_rank_ic"].max(),
            "mean_rank_ic_range": rf["mean_rank_ic"].max() - rf["mean_rank_ic"].min(),
            "portfolio_sharpe_minimum": rf["portfolio_sharpe_zero_rate"].min(),
            "portfolio_sharpe_maximum": rf["portfolio_sharpe_zero_rate"].max(),
            "local_robustness_pass": bool(rf_protocol["local_robustness_pass"]),
            "interpretation": "predeclared neighbor robustness floor failed; no point promoted",
        },
    ])
    return evidence, summarize_rf_seeds(rf), hyperparameters


def _protocol_table(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "purpose": cfg["purpose"],
        "selection_period": "validation_2016_2020_only",
        "historical_oos_classification": "descriptive_historical_oos_not_pristine_lockbox",
        "historical_oos_used_for_selection": False,
        "parameter_promotion_allowed": False,
        "candidate_count": len(cfg["candidates"]),
        "same_universe_label_cost_execution_dates": True,
        "target": cfg["target"], "round_trip_cost_bps": cfg["portfolio"]["round_trip_cost_bps"],
        "window_policies": ",".join(cfg["window_comparison"]["policies"]),
        "purged_cv_future_training_samples_allowed": True,
        "purged_cv_is_trading_simulation": False,
        "deterministic_models_not_repeated_by_seed": "naive,ridge,elastic_net,hgb_early_stopping_false",
        "random_model_seed_evidence": "random_forest:20261016,20261017,20261018",
        "new_model_selected": False,
    }])


def _write_report(outputs: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    summary = outputs["model_stability_window_summary"]
    pairwise = outputs["model_stability_window_pairwise"]
    annual = outputs["model_stability_window_annual"]
    cv = outputs["model_stability_purged_cv_summary"]
    seeds = outputs["model_stability_seed_summary"]
    evidence = outputs["model_stability_evidence_matrix"]
    hyperparameters = outputs["model_stability_hyperparameter_range"]
    alignment = outputs["model_stability_alignment_audit"]
    display_summary = summary[[
        "split", "window_policy", "candidate", "mean_rank_ic",
        "portfolio_annualized_return", "portfolio_sharpe_zero_rate",
        "portfolio_maximum_drawdown", "portfolio_mean_turnover",
        "portfolio_annualized_transaction_cost_arithmetic",
    ]]
    failure = annual.groupby(["split", "window_policy", "candidate"], as_index=False).agg(
        years=("test_year", "size"), rank_ic_failure_years=("rank_ic_failure", "sum"),
        net_return_failure_years=("net_return_failure", "sum"),
    )
    lines = [
        "# Phase 3：模型稳定性、随机种子与时间感知验证", "",
        "## 结论边界", "",
        "本实验不重新选模、不晋升参数。2016–2020 仍是选择/验证区间；2021–2026 仅作已被反复查看的 historical OOS 描述，不是 pristine lockbox。M5 允许测试年份之后的数据进入训练，只用于跨时期泛化诊断，不能解释为可交易历史业绩。",
        "", "## M0–M6 完成度", "", evidence.to_markdown(index=False),
        "", "## Expanding 与 10 年 Rolling", "",
        display_summary.to_markdown(index=False, floatfmt=".6f"),
        "", "### 配对差异（rolling - expanding）", "",
        pairwise.to_markdown(index=False, floatfmt=".6f"),
        "", "### 失败年份计数", "", failure.to_markdown(index=False),
        "", "## RF 多随机种子", "", seeds.to_markdown(index=False, floatfmt=".6f"),
        "", "## 参数邻域", "", hyperparameters.to_markdown(index=False, floatfmt=".6f"),
        "", "## Purged CV（补充、非交易模拟）", "",
        cv[["candidate", "mean_rank_ic", "portfolio_annualized_return",
            "portfolio_sharpe_zero_rate", "portfolio_maximum_drawdown",
            "portfolio_mean_turnover",
            "portfolio_annualized_transaction_cost_arithmetic"]].to_markdown(
                index=False, floatfmt=".6f"
            ),
        "", "## 日期与横截面对齐审计", "",
        alignment.to_markdown(index=False),
        "", "## 审计说明", "",
        "每次 fit 的训练区间、标签成熟日期、参数、seed、依赖版本及预测输出 SHA-256 均写入 fit audit。逐日预测分数分布、逐年 RankIC/净收益/Sharpe/回撤/换手和成本均保留在数据湖。任何不稳定结果均原样报告，未以新结果改写冻结候选。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_stability_protocol", "model_stability_evidence_matrix",
        "model_stability_seed_summary", "model_stability_hyperparameter_range",
        "model_stability_alignment_audit", "model_stability_window_summary",
        "model_stability_window_annual", "model_stability_window_pairwise",
        "model_stability_purged_cv_summary", "model_stability_purged_cv_annual",
        "model_stability_purged_cv_pairwise", "model_stability_purged_cv_purge_audit",
    ):
        outputs[name].to_csv(TABLE_ROOT / f"{name}.csv", index=False)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    validate_protocol(cfg)
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    trading_days = open_trading_days(calendar)
    start = int(cfg["window_comparison"]["expanding_start_year"])
    end = int(cfg["evaluation"]["end_year"])
    full_stats: dict[int, Any] = {}
    purged_stats: dict[int, Any] = {}
    full_samples: dict[int, TrainingSample] = {}
    purged_samples: dict[int, TrainingSample] = {}
    full_training_dates: dict[int, list[str]] = {}
    for year in range(start, end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(
            calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
        )
        (full_stats[year], purged_stats[year], full_samples[year],
         purged_samples[year]) = build_year_training(frame, factors, target, cutoff, cfg)
        counts = frame.groupby("asof_date")[target].count()
        full_training_dates[year] = sorted(counts.loc[
            counts.ge(int(cfg["minimum_cross_section_size"]))
        ].index.astype(str))
        print(f"phase3 training statistics year={year}", flush=True)

    outputs = _run_window_comparison(
        lake, cfg, factors, columns, calendar, trading_days,
        full_stats, purged_stats, full_samples, purged_samples,
    )
    outputs.update(_run_purged_cv(
        lake, cfg, factors, columns, calendar, trading_days, full_stats, full_samples,
        full_training_dates,
    ))
    evidence, seed_summary, hyperparameters = _evidence_tables(lake)
    outputs["model_stability_protocol"] = _protocol_table(cfg)
    outputs["model_stability_evidence_matrix"] = evidence
    outputs["model_stability_seed_summary"] = seed_summary
    outputs["model_stability_hyperparameter_range"] = hyperparameters
    outputs["model_stability_alignment_audit"] = pd.concat([
        _alignment_audit(
            outputs["model_stability_window_daily"],
            outputs["model_stability_window_portfolio"], len(cfg["candidates"]),
        ),
        _alignment_audit(
            outputs["model_stability_purged_cv_daily"],
            outputs["model_stability_purged_cv_portfolio"], len(cfg["candidates"]),
        ),
    ], ignore_index=True)
    request = {**cfg, "factor_columns": factors}
    for table, frame in outputs.items():
        lake.write(table, frame, key="v1", source="model_stability_validation.v1",
                   request=request)
    _write_report(outputs, cfg)
    print(outputs["model_stability_protocol"].to_string(index=False), flush=True)
    print(outputs["model_stability_window_summary"].to_string(index=False), flush=True)
    print(outputs["model_stability_purged_cv_summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
