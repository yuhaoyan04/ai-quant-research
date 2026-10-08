"""Point-in-time-safe raw factor construction.

Each output row has `asof_date` (information known after that close) and
`available_date` (the next exchange trading day).  Cross-sectional transforms
are separate from raw-factor construction to prevent global-sample leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research.ingest import Lake


def next_trading_day_map(calendar: pd.DataFrame) -> dict[str, str]:
    open_days = pd.to_datetime(calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"]).sort_values()
    return {day.strftime("%Y-%m-%d"): next_day.strftime("%Y-%m-%d")
            for day, next_day in zip(open_days.iloc[:-1], open_days.iloc[1:])}


def build_raw_factors(validated: pd.DataFrame, next_days: dict[str, str]) -> pd.DataFrame:
    data = validated.loc[validated["quality_status"].ne("fail")].copy()
    data = data.sort_values("trade_date").reset_index(drop=True)
    one_day_return = data["pct_chg"] / 100
    prior_turnover_mean = data["turn"].rolling(20, min_periods=20).mean().shift(1)
    prior_volume_mean = data["volume"].rolling(20, min_periods=20).mean().shift(1)
    log_volume_change = data["volume"].where(data["volume"] > 0).map(__import__("math").log).diff()
    daily_amplitude = (data["high"] - data["low"]) / data["pre_close"].where(data["pre_close"] > 0)
    # Do not fill missing return observations: a suspension/missing print must
    # not become a fabricated zero-return day in momentum or volatility.
    factor = pd.DataFrame({
        "instrument_id": data["instrument_id"],
        "asof_date": data["trade_date"],
        "return_1d": one_day_return,
        "momentum_20d": (1 + one_day_return).rolling(20, min_periods=20).apply(lambda x: x.prod() - 1, raw=True),
        "momentum_60d": (1 + one_day_return).rolling(60, min_periods=60).apply(lambda x: x.prod() - 1, raw=True),
        "momentum_120d": (1 + one_day_return).rolling(120, min_periods=120).apply(lambda x: x.prod() - 1, raw=True),
        # Skip the latest week to reduce overlap with short-term reversal.
        "momentum_20d_skip_5d": (1 + one_day_return.shift(5)).rolling(20, min_periods=20).apply(lambda x: x.prod() - 1, raw=True),
        "reversal_5d": -((1 + one_day_return).rolling(5, min_periods=5).apply(lambda x: x.prod() - 1, raw=True)),
        "volatility_20d": one_day_return.rolling(20, min_periods=20).std(ddof=1),
        "amihud_20d": (one_day_return.abs() / data["amount"].where(data["amount"] > 0)).rolling(20, min_periods=20).mean(),
        "turnover_20d": data["turn"].rolling(20, min_periods=20).mean(),
        "turnover_surprise_20d": data["turn"] / prior_turnover_mean - 1,
        "volume_surprise_20d": data["volume"] / prior_volume_mean - 1,
        "amplitude_20d": daily_amplitude.rolling(20, min_periods=20).mean(),
        "return_volume_corr_20d": one_day_return.rolling(20, min_periods=20).corr(log_volume_change),
        "max_return_20d": one_day_return.rolling(20, min_periods=20).max(),
        "earnings_yield": 1 / data["pe_ttm"].where(data["pe_ttm"] > 0),
        "book_to_price": 1 / data["pb_mrq"].where(data["pb_mrq"] > 0),
        "is_tradable": data["trade_status"].eq(1) & data["is_st"].ne(1),
    })
    factor["available_date"] = factor["asof_date"].map(next_days)
    return factor


def cross_sectional_rank_zscore(frame: pd.DataFrame, factor_columns: list[str]) -> pd.DataFrame:
    """Per-date robust transforms; use only after the full eligible universe is assembled."""
    result = frame.copy()
    for column in factor_columns:
        values = result[column]
        lo = result.groupby("asof_date")[column].transform(lambda x: x.quantile(0.01))
        hi = result.groupby("asof_date")[column].transform(lambda x: x.quantile(0.99))
        clipped = values.clip(lo, hi)
        mean = clipped.groupby(result["asof_date"]).transform("mean")
        std = clipped.groupby(result["asof_date"]).transform("std")
        result[f"{column}_z"] = (clipped - mean) / std.where(std > 0)
    return result


def run_factors(lake: Lake | None = None, *, limit: int | None = None,
                skip_existing: bool = False) -> None:
    lake = lake or Lake()
    calendar = lake.read("calendar", "SSE")
    if calendar.empty:
        raise RuntimeError("calendar is missing; run free ingestion first")
    next_days = next_trading_day_map(calendar)
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    if limit is not None:
        files = files[:limit]
    for number, path in enumerate(files, start=1):
        key = path.parent.name.removeprefix("key=")
        if skip_existing and lake.table_path("factors_raw", key).exists():
            continue
        factors = build_raw_factors(pd.read_parquet(path), next_days)
        lake.write("factors_raw", factors, key=key, source="factors.v1",
                   request={"validated_path": str(path), "availability": "t_close_to_t_plus_1"})
        print(f"[{number}/{len(files)}] factors {key}")
