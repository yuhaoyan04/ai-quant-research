"""Research-panel primitives: instrument classification and future labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


A_SHARE_PATTERNS = (
    re.compile(r"^sh\.(600|601|603|605|688)\d{3}$"),
    re.compile(r"^sz\.(000|001|002|003|300|301)\d{3}$"),
    re.compile(r"^bj\.(4|8|9)\d{5}$"),
)


def instrument_type(instrument_id: str) -> str:
    """Classify vendor codes before they enter an equity cross section."""
    if any(pattern.fullmatch(instrument_id) for pattern in A_SHARE_PATTERNS):
        return "a_share"
    if instrument_id.startswith(("sh.000", "sz.399")):
        return "index"
    return "other_security"


def build_labels(validated: pd.DataFrame, horizons: list[int], execution_lag: int = 1) -> pd.DataFrame:
    """Create future-only compounded close-to-close returns.

    At date t, the first future return is t+1. This deliberately excludes the
    t-day return already embedded in a t-close signal. Missing future prints
    make a label missing rather than a fabricated zero return.
    """
    data = validated.sort_values("trade_date").reset_index(drop=True).copy()
    daily_return = pd.to_numeric(data["pct_chg"], errors="coerce") / 100
    output = pd.DataFrame({
        "instrument_id": data["instrument_id"],
        "asof_date": data["trade_date"],
        "quality_status": data["quality_status"],
        "instrument_type": data["instrument_id"].map(instrument_type),
        "is_tradable_at_asof": data["trade_status"].eq(1) & data["is_st"].ne(1),
    })
    for horizon in horizons:
        # reverse rolling is equivalent to product of returns t+1...t+h.
        future = daily_return.shift(-1)
        output[f"fwd_return_{horizon}d"] = (
            (1 + future.iloc[::-1]).rolling(horizon, min_periods=horizon).apply(lambda x: x.prod() - 1, raw=True).iloc[::-1]
        )
        executable_future = daily_return.shift(-(1 + execution_lag))
        output[f"fwd_return_{horizon}d_lag{execution_lag}"] = (
            (1 + executable_future.iloc[::-1]).rolling(horizon, min_periods=horizon)
            .apply(lambda x: x.prod() - 1, raw=True).iloc[::-1]
        )
    return output


def load_config() -> dict:
    with (ROOT / "config" / "research_panel.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_labels(lake: Lake | None = None, *, limit: int | None = None,
               skip_existing: bool = False) -> None:
    lake = lake or Lake()
    cfg = load_config()
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    if limit is not None:
        files = files[:limit]
    for number, path in enumerate(files, start=1):
        key = path.parent.name.removeprefix("key=")
        if skip_existing and lake.table_path("labels_raw", key).exists():
            continue
        labels = build_labels(
            pd.read_parquet(path), list(cfg["label_horizons_trading_days"]),
            int(cfg["execution_lag_trading_days"]),
        )
        lake.write("labels_raw", labels, key=key, source="labels.v1",
                   request={"validated_path": str(path), "convention": cfg["label_convention"],
                            "executable_convention": cfg["executable_label_convention"]})
        print(f"[{number}/{len(files)}] labels {key}")
