"""Non-destructive data-quality gates for market data.

Raw vendor files are immutable evidence. This module writes a normalized copy
with row-level flags and a per-security report; it never silently drops or
repairs observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


@dataclass(frozen=True)
class QualityResult:
    frame: pd.DataFrame
    report: pd.DataFrame


def load_config() -> dict:
    with (ROOT / "config" / "quality.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["daily_bars"]


def validate_daily_bars(raw: pd.DataFrame, config: dict | None = None) -> QualityResult:
    cfg = config or load_config()
    missing_columns = set(cfg["required_columns"]) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"daily_bars schema missing required columns: {sorted(missing_columns)}")
    frame = raw.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in cfg["numeric_columns"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_status"] = pd.to_numeric(frame["trade_status"], errors="coerce")
    frame["is_st"] = pd.to_numeric(frame.get("is_st"), errors="coerce")

    flags: dict[str, pd.Series] = {
        "missing_key": frame["trade_date"].isna() | frame["instrument_id"].isna(),
        "duplicate_key": frame.duplicated(["instrument_id", "trade_date"], keep=False),
        "invalid_price": (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | frame[["open", "high", "low", "close"]].isna().any(axis=1),
        "invalid_ohlc": (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1)) | (frame["low"] > frame["high"]),
        "negative_activity": (frame[["volume", "amount"]] < 0).any(axis=1),
    }
    return_from_vendor = frame["pct_chg"] / 100
    flags["extreme_return_warning"] = return_from_vendor.abs() > float(cfg["maximum_absolute_daily_return"])
    flags["not_tradable"] = frame["trade_status"].ne(1)
    flags["st_warning"] = frame["is_st"].eq(1)
    frame["quality_flags"] = pd.DataFrame(flags).apply(
        lambda row: "|".join(name for name, present in row.items() if present), axis=1
    )
    critical = ["missing_key", "duplicate_key", "invalid_price", "invalid_ohlc", "negative_activity"]
    frame["quality_status"] = "pass"
    frame.loc[pd.DataFrame(flags)[critical].any(axis=1), "quality_status"] = "fail"
    frame.loc[(frame["quality_status"] == "pass") & (frame["quality_flags"] != ""), "quality_status"] = "warn"
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")

    report = pd.DataFrame([
        {"metric": "rows", "value": len(frame)},
        {"metric": "pass_rows", "value": int(frame["quality_status"].eq("pass").sum())},
        {"metric": "warn_rows", "value": int(frame["quality_status"].eq("warn").sum())},
        {"metric": "fail_rows", "value": int(frame["quality_status"].eq("fail").sum())},
        *({"metric": f"flag_{name}", "value": int(present.sum())} for name, present in flags.items()),
    ])
    return QualityResult(frame=frame, report=report)


def run_quality(lake: Lake | None = None, *, limit: int | None = None,
                skip_existing: bool = False) -> None:
    lake = lake or Lake()
    root = lake.curated / "daily_bars_baostock"
    files = sorted(root.glob("key=*/data.parquet"))
    if limit is not None:
        files = files[:limit]
    for number, path in enumerate(files, start=1):
        key = path.parent.name.removeprefix("key=")
        if skip_existing and lake.table_path("daily_bars_validated", key).exists() \
                and lake.table_path("quality_report", key).exists():
            continue
        result = validate_daily_bars(pd.read_parquet(path))
        lake.write("daily_bars_validated", result.frame, key=key, source="quality.daily_bars.v1",
                   request={"raw_table": "daily_bars_baostock", "raw_path": str(path)})
        lake.write("quality_report", result.report, key=key, source="quality.daily_bars.v1",
                   request={"raw_table": "daily_bars_baostock"})
        print(f"[{number}/{len(files)}] validated {key}")
