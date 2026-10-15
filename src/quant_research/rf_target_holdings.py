"""Materialize frozen RF target holdings once for reusable execution research."""

from __future__ import annotations

import pandas as pd

from quant_research.ingest import Lake
from quant_research.model_baseline import purged_end_date, sum_statistics
from quant_research.model_benchmark import _read_year, _transform, build_year_training, fit_candidate
from quant_research.model_challenger import concatenate_samples
from quant_research.preprocessing import load_config as load_preprocessing_config
from quant_research.rf_oos_execution_audit import load_config


def _score_holdings(frame: pd.DataFrame, model, factors: list[str], target: str,
                    cfg: dict, split: str, test_year: int | None = None) -> pd.DataFrame:
    frames = []
    for asof_date, group in frame.groupby("asof_date", sort=True):
        if pd.to_numeric(group[target], errors="coerce").notna().sum() < int(
            cfg["minimum_cross_section_size"]
        ):
            continue
        group = group.reset_index(drop=True)
        score = model.predict(_transform(group, factors, cfg))
        selected = group.assign(score=score).sort_values(
            ["score", "instrument_id"], ascending=[False, True]
        ).head(int(cfg["portfolio"]["top_n"])).copy()
        selected["target_weight"] = 1.0 / len(selected)
        selected["signal_date"] = str(asof_date)
        selected["execution_date"] = selected["available_date"].astype(str)
        selected["split"] = split
        selected["test_year"] = test_year
        frames.append(selected[[
            "instrument_id", "signal_date", "execution_date", "target_weight",
            "score", "split", "test_year",
        ]])
    return pd.concat(frames, ignore_index=True)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(load_preprocessing_config()["factor_columns"])
    target = str(cfg["target"])
    columns = ["instrument_id", "asof_date", "available_date", target, *factors]
    calendar = lake.read("calendar", "SSE")
    start = int(cfg["train_start_year"])
    oos_start = int(cfg["historical_oos"]["start_year"])
    oos_end = int(cfg["historical_oos"]["end_year"])
    full_stats, purged_stats, full_samples, purged_samples = {}, {}, {}, {}
    for year in range(start, oos_end):
        frame = _read_year(lake, year, columns)
        cutoff = purged_end_date(calendar, f"{year}-12-31",
                                 int(cfg["label_horizon_trading_days"]))
        (full_stats[year], purged_stats[year], full_samples[year],
         purged_samples[year]) = build_year_training(frame, factors, target, cutoff, cfg)
        print(f"rf targets prepared year={year}", flush=True)

    validation_train_end = oos_start - 6
    statistics = sum_statistics(
        [full_stats[y] for y in range(start, validation_train_end)]
        + [purged_stats[validation_train_end]]
    )
    sample = concatenate_samples(
        [full_samples[y] for y in range(start, validation_train_end)]
        + [purged_samples[validation_train_end]]
    )
    validation_model = fit_candidate(cfg["model"], statistics, sample, len(factors), cfg)
    # Repeated small cross-section predictions are faster and more stable
    # without starting a thread pool 250+ times. Fitting remains parallel.
    validation_model.estimator.n_jobs = 1
    validation_frames = []
    for year in range(validation_train_end + 1, oos_start):
        frame = _read_year(lake, year, columns)
        if year == oos_start - 1:
            cutoff = purged_end_date(
                calendar, f"{year}-12-31", int(cfg["label_horizon_trading_days"])
            )
            frame = frame.loc[frame["asof_date"].le(cutoff)]
        validation_frames.append(frame)
    validation = pd.concat(validation_frames, ignore_index=True)
    validation_holdings = _score_holdings(
        validation, validation_model, factors, target, cfg, "validation"
    )
    print(f"rf targets validation dates={validation_holdings['signal_date'].nunique()}",
          flush=True)

    oos_holdings, fit_records = [], []
    for test_year in range(oos_start, oos_end + 1):
        fit_end = test_year - 1
        statistics = sum_statistics(
            [full_stats[y] for y in range(start, fit_end)] + [purged_stats[fit_end]]
        )
        sample = concatenate_samples(
            [full_samples[y] for y in range(start, fit_end)] + [purged_samples[fit_end]]
        )
        model = fit_candidate(cfg["model"], statistics, sample, len(factors), cfg)
        model.estimator.n_jobs = 1
        holdings = _score_holdings(
            _read_year(lake, test_year, columns), model, factors, target, cfg,
            "historical_oos", test_year,
        )
        oos_holdings.append(holdings)
        fit_records.append({
            "split": "historical_oos", "test_year": test_year,
            "fit_seconds": model.fit_seconds, "fit_observations": model.fit_observations,
        })
        print(f"rf targets oos year={test_year} dates={holdings['signal_date'].nunique()} "
              f"fit_seconds={model.fit_seconds:.1f}", flush=True)
    request = {**cfg, "factor_columns": factors,
               "purpose": "reusable_target_holdings_for_execution_research"}
    lake.write("rf_target_holdings", validation_holdings, key="validation_v1",
               source="rf_target_holdings.v1", request=request)
    lake.write("rf_target_holdings", pd.concat(oos_holdings, ignore_index=True),
               key="historical_oos_v1", source="rf_target_holdings.v1", request=request)
    fit_records.insert(0, {
        "split": "validation", "test_year": None,
        "fit_seconds": validation_model.fit_seconds,
        "fit_observations": validation_model.fit_observations,
    })
    lake.write("rf_target_holdings_fit_audit", pd.DataFrame(fit_records), key="v1",
               source="rf_target_holdings.v1", request=request)


if __name__ == "__main__":
    run()
