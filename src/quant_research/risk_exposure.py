"""Point-in-time risk exposures that are safe for neutralization research."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.model_baseline import weekly_rows
from quant_research.panel import instrument_type


CONFIG_PATH = ROOT / "config" / "risk_exposure.yaml"
OUTPUT_COLUMNS = ["instrument_id", "asof_date", "float_cap_proxy_cny"]


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def float_cap_proxy(amount: pd.Series, turnover_percent: pd.Series) -> pd.Series:
    """Return amount / turnover fraction without fabricating invalid observations."""
    amount_numeric = pd.to_numeric(amount, errors="coerce").to_numpy(dtype="float64")
    turnover_fraction = (
        pd.to_numeric(turnover_percent, errors="coerce").to_numpy(dtype="float64") / 100.0
    )
    values = np.divide(
        amount_numeric, turnover_fraction,
        out=np.full(len(amount_numeric), np.nan, dtype="float64"),
        where=(amount_numeric > 0) & (turnover_fraction > 0),
    )
    return pd.Series(values, index=amount.index, dtype="float64")


def weekly_date_sets(lake: Lake, start_year: int, end_year: int) -> dict[int, set[str]]:
    result = {}
    for year in range(start_year, end_year + 1):
        dates = pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["asof_date"],
        ).drop_duplicates()
        result[year] = set(weekly_rows(dates)["asof_date"].astype(str))
    return result


def audit_size_exposures(lake: Lake, start_year: int, end_year: int,
                         cfg: dict) -> pd.DataFrame:
    """Measure coverage on the actual eligible weekly panel, not the source table."""
    records = []
    for year in range(start_year, end_year + 1):
        exposure = pd.read_parquet(
            lake.table_path("pit_size_exposures", str(year)), columns=OUTPUT_COLUMNS
        )
        duplicates = int(exposure.duplicated(["instrument_id", "asof_date"]).sum())
        if duplicates:
            raise RuntimeError(f"PIT size exposure duplicate keys in {year}: {duplicates}")
        eligible = weekly_rows(pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["instrument_id", "asof_date"],
        ))
        merged = eligible.merge(
            exposure, on=["instrument_id", "asof_date"], how="left", validate="one_to_one"
        )
        records.append({
            "year": year,
            "source_rows": len(exposure),
            "eligible_weekly_rows": len(eligible),
            "eligible_weekly_dates": eligible["asof_date"].nunique(),
            "eligible_float_cap_coverage": float(
                merged["float_cap_proxy_cny"].notna().mean()
            ),
            "median_float_cap_proxy_cny": float(
                merged["float_cap_proxy_cny"].median()
            ),
            "duplicate_instrument_date_keys": duplicates,
        })
    result = pd.DataFrame(records)
    lake.write(
        "pit_size_exposure_index", result, key="v1",
        source="risk_exposure.pit_size.v1", request=cfg,
    )
    return result


def build_size_exposures(lake: Lake | None = None, *, overwrite: bool = False) -> None:
    lake = lake or Lake()
    cfg = load_config()
    start_year, end_year = int(cfg["start_year"]), int(cfg["end_year"])
    dates_by_year = weekly_date_sets(lake, start_year, end_year)
    date_to_year = {
        date: year for year, dates in dates_by_year.items() for date in dates
    }
    all_dates = set(date_to_year)
    writers: dict[int, pq.ParquetWriter] = {}
    schemas: dict[int, pa.Schema] = {}
    temporary: dict[int, Path] = {}
    final: dict[int, Path] = {}
    counts = {year: 0 for year in dates_by_year}
    non_missing = {year: 0 for year in dates_by_year}
    skipped_years = {
        year for year in dates_by_year
        if lake.table_path("pit_size_exposures", str(year)).exists() and not overwrite
    }
    files = sorted((lake.curated / "daily_bars_validated").glob("key=*/data.parquet"))
    processed = 0
    try:
        for path in files:
            code = path.parent.name.removeprefix("key=").replace("_", ".")
            if instrument_type(code) != "a_share":
                continue
            frame = pd.read_parquet(
                path, columns=["instrument_id", "trade_date", "amount", "turn"]
            )
            frame = frame.loc[frame["trade_date"].isin(all_dates)].copy()
            if frame.empty:
                continue
            frame["float_cap_proxy_cny"] = float_cap_proxy(frame["amount"], frame["turn"])
            frame["asof_date"] = frame["trade_date"].astype(str)
            frame["year"] = frame["asof_date"].map(date_to_year)
            for year, group in frame.groupby("year", sort=False):
                year = int(year)
                if year in skipped_years:
                    continue
                output = group[OUTPUT_COLUMNS]
                if year not in writers:
                    destination = lake.table_path("pit_size_exposures", str(year))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp = destination.with_suffix(".tmp.parquet")
                    if temp.exists():
                        temp.unlink()
                    table = pa.Table.from_pandas(output, preserve_index=False)
                    schemas[year] = table.schema
                    writers[year] = pq.ParquetWriter(temp, table.schema, compression="zstd")
                    temporary[year], final[year] = temp, destination
                else:
                    table = pa.Table.from_pandas(
                        output, schema=schemas[year], preserve_index=False
                    )
                writers[year].write_table(table)
                counts[year] += len(output)
                non_missing[year] += int(output["float_cap_proxy_cny"].notna().sum())
            processed += 1
            if processed % 500 == 0:
                print(f"PIT size exposure files={processed}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for year, temp in temporary.items():
        temp.replace(final[year])

    for year in range(start_year, end_year + 1):
        path = lake.table_path("pit_size_exposures", str(year))
        if not path.exists():
            raise FileNotFoundError(f"PIT size exposure not created for {year}")
        frame = pd.read_parquet(path, columns=OUTPUT_COLUMNS)
        duplicates = int(frame.duplicated(["instrument_id", "asof_date"]).sum())
        if duplicates:
            raise RuntimeError(f"PIT size exposure duplicate keys in {year}: {duplicates}")
        print(
            f"PIT size exposure year={year} rows={len(frame):,} "
            f"coverage={frame['float_cap_proxy_cny'].notna().mean():.4%}", flush=True,
        )
    audit = audit_size_exposures(lake, start_year, end_year, cfg)
    print(audit.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time risk exposures")
    parser.add_argument("command", choices=["build-size"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "build-size":
        build_size_exposures(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
