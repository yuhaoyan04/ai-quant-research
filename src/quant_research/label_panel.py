"""Materialize one PIT factor panel with executable 5/10/20-day labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from quant_research.ingest import Lake


LABEL_COLUMNS = [
    "instrument_id", "asof_date", "fwd_return_5d_lag1",
    "fwd_return_10d_lag1", "fwd_return_20d_lag1",
]


def merge_horizon_labels(eligible: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Preserve the eligible PIT universe and enforce one label row per key."""
    keys = ["instrument_id", "asof_date"]
    if labels.duplicated(keys).any():
        raise ValueError("multi-horizon labels contain duplicate instrument-date keys")
    additional = [column for column in LABEL_COLUMNS[2:] if column not in eligible.columns]
    result = eligible.merge(
        labels[[*keys, *additional]], on=keys, how="left", validate="one_to_one"
    )
    if len(result) != len(eligible):
        raise RuntimeError("multi-horizon merge changed eligible universe row count")
    return result


def stage_annual_labels(lake: Lake | None = None, *, overwrite: bool = False) -> None:
    lake = lake or Lake()
    source_root = lake.curated / "labels_raw"
    dataset = ds.dataset(source_root, format="parquet", partitioning="hive")
    writers: dict[str, pq.ParquetWriter] = {}
    temporary: dict[str, Path] = {}
    final: dict[str, Path] = {}
    counts: dict[str, int] = {}
    try:
        scanner = dataset.scanner(columns=LABEL_COLUMNS, batch_size=250_000)
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            for year, group in frame.groupby(frame["asof_date"].str.slice(0, 4), sort=False):
                destination = lake.table_path("multi_horizon_labels_staging", str(year))
                if destination.exists() and not overwrite:
                    continue
                if year not in writers:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp = destination.with_suffix(".tmp.parquet")
                    if temp.exists():
                        temp.unlink()
                    temporary[year], final[year] = temp, destination
                    table = pa.Table.from_pandas(group[LABEL_COLUMNS], preserve_index=False)
                    writers[year] = pq.ParquetWriter(temp, table.schema, compression="zstd")
                else:
                    table = pa.Table.from_pandas(
                        group[LABEL_COLUMNS], schema=writers[year].schema,
                        preserve_index=False,
                    )
                writers[year].write_table(table)
                counts[year] = counts.get(year, 0) + len(group)
    finally:
        for writer in writers.values():
            writer.close()
    for year, temp in temporary.items():
        temp.replace(final[year])
        print(f"label staging year={year} rows={counts[year]:,}", flush=True)


def build_annual_panels(lake: Lake | None = None, *, overwrite: bool = False) -> None:
    lake = lake or Lake()
    for year in range(2006, 2027):
        destination = lake.table_path("multi_horizon_factor_panel", str(year))
        if destination.exists() and not overwrite:
            print(f"multi-horizon panel year={year} exists", flush=True)
            continue
        eligible_path = lake.table_path("eligible_factor_panel_staging", str(year))
        labels_path = lake.table_path("multi_horizon_labels_staging", str(year))
        if not eligible_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"missing eligible or label staging year={year}")
        eligible = pd.read_parquet(eligible_path)
        labels = pd.read_parquet(labels_path)
        panel = merge_horizon_labels(eligible, labels)
        lake.write(
            "multi_horizon_factor_panel", panel, key=str(year),
            source="label_panel.multi_horizon.v1",
            request={"year": year, "label_columns": LABEL_COLUMNS},
        )
        coverage = {
            column: float(panel[column].notna().mean()) for column in LABEL_COLUMNS[2:]
        }
        print(f"multi-horizon panel year={year} rows={len(panel):,} coverage={coverage}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build annual multi-horizon PIT panels")
    parser.add_argument("command", choices=["stage-labels", "build-panels", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in ("stage-labels", "all"):
        stage_annual_labels(overwrite=args.overwrite)
    if args.command in ("build-panels", "all"):
        build_annual_panels(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

