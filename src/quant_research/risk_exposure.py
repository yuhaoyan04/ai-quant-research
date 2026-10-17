"""Point-in-time risk exposures that are safe for neutralization research."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.free_ingest import industry_snapshot_dates
from quant_research.model_baseline import weekly_rows
from quant_research.panel import instrument_type


CONFIG_PATH = ROOT / "config" / "risk_exposure.yaml"
SIZE_OUTPUT_COLUMNS = ["instrument_id", "asof_date", "float_cap_proxy_cny"]
INDUSTRY_OUTPUT_COLUMNS = [
    "instrument_id", "asof_date", "source_snapshot_date", "source_update_date",
    "industry_raw", "industry_classification", "industry_taxonomy",
    "industry_detail_code", "industry_broad_code",
]

LEGACY_CSRC_BROAD = {
    "农林牧渔业": "A",
    "采掘业": "B",
    "制造业": "C",
    "电力煤气及水的生产和供应业": "D",
    "建筑业": "E",
    "批发和零售贸易": "F",
    "交通运输仓储业": "G",
    "信息技术业": "I",
    "金融保险业": "J",
    "房地产业": "K",
    # The legacy social-services bucket spans several modern sections, so it
    # remains explicit instead of being assigned a false modern equivalent.
    "社会服务业": "LEGACY_SOCIAL_SERVICES",
    "传播与文化产业": "R",
    "综合类": "S",
}
MODERN_CSRC_PATTERN = re.compile(r"^([A-Z])(\d{2})")


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


def normalize_csrc_industry(value: object) -> tuple[str | None, str | None, str | None]:
    """Map raw historical CSRC labels to an auditable broad contemporaneous group.

    Modern records contain a letter plus two-digit division code.  Legacy
    records use named top-level groups.  The bridge is semantic and explicit;
    ambiguous legacy social services are deliberately not forced into a modern
    section.
    """
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return None, None, None
    modern = MODERN_CSRC_PATTERN.match(raw)
    if modern:
        return "csrc_2012", modern.group(0), modern.group(1)
    root = raw.split("-", 1)[0].strip()
    broad = LEGACY_CSRC_BROAD.get(root)
    return "csrc_legacy", f"LEGACY:{root}", broad


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
            lake.table_path("pit_size_exposures", str(year)), columns=SIZE_OUTPUT_COLUMNS
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
                output = group[SIZE_OUTPUT_COLUMNS]
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
        frame = pd.read_parquet(path, columns=SIZE_OUTPUT_COLUMNS)
        duplicates = int(frame.duplicated(["instrument_id", "asof_date"]).sum())
        if duplicates:
            raise RuntimeError(f"PIT size exposure duplicate keys in {year}: {duplicates}")
        print(
            f"PIT size exposure year={year} rows={len(frame):,} "
            f"coverage={frame['float_cap_proxy_cny'].notna().mean():.4%}", flush=True,
        )
    audit = audit_size_exposures(lake, start_year, end_year, cfg)
    print(audit.to_string(index=False), flush=True)


def audit_industry_exposures(
    lake: Lake, start_year: int, end_year: int, cfg: dict,
) -> pd.DataFrame:
    records = []
    for year in range(start_year, end_year + 1):
        frame = pd.read_parquet(
            lake.table_path("pit_industry_exposures", str(year)),
            columns=INDUSTRY_OUTPUT_COLUMNS,
        )
        duplicates = int(frame.duplicated(["instrument_id", "asof_date"]).sum())
        if duplicates:
            raise RuntimeError(f"PIT industry exposure duplicate keys in {year}: {duplicates}")
        update = pd.to_datetime(frame["source_update_date"], errors="coerce")
        snapshot = pd.to_datetime(frame["source_snapshot_date"], errors="coerce")
        asof = pd.to_datetime(frame["asof_date"], errors="coerce")
        future_rows = int((update > asof).sum())
        future_snapshot_rows = int((snapshot > asof).sum())
        update_after_snapshot_rows = int((update > snapshot).sum())
        if future_rows or future_snapshot_rows or update_after_snapshot_rows:
            raise RuntimeError(f"PIT industry exposure contains future rows in {year}: {future_rows}")
        covered = frame["industry_broad_code"].notna()
        coverage = float(covered.mean())
        records.append({
            "year": year,
            "eligible_weekly_rows": len(frame),
            "eligible_weekly_dates": frame["asof_date"].nunique(),
            "eligible_industry_coverage": coverage,
            "industry_broad_groups": int(frame.loc[covered, "industry_broad_code"].nunique()),
            "legacy_rows": int(frame["industry_taxonomy"].eq("csrc_legacy").sum()),
            "modern_rows": int(frame["industry_taxonomy"].eq("csrc_2012").sum()),
            "median_source_lag_days": float((asof - update).dt.days.median()),
            "maximum_source_lag_days": int((asof - update).dt.days.max()),
            "median_snapshot_lag_days": float((asof - snapshot).dt.days.median()),
            "maximum_snapshot_lag_days": int((asof - snapshot).dt.days.max()),
            "future_dated_rows": future_rows,
            "future_snapshot_rows": future_snapshot_rows,
            "update_after_snapshot_rows": update_after_snapshot_rows,
            "duplicate_instrument_date_keys": duplicates,
        })
    result = pd.DataFrame(records)
    minimum = float(cfg["industry_minimum_coverage"])
    if float(result["eligible_industry_coverage"].min()) < minimum:
        failing = result.loc[
            result["eligible_industry_coverage"].lt(minimum),
            ["year", "eligible_industry_coverage"],
        ].to_dict("records")
        raise RuntimeError(
            f"PIT industry coverage below {minimum:.1%}: {failing}"
        )
    lake.write(
        "pit_industry_exposure_index", result, key="v1",
        source="risk_exposure.pit_industry.v1", request=cfg,
    )
    return result


def build_industry_exposures(
    lake: Lake | None = None, *, overwrite: bool = False,
) -> None:
    """Align historical CSRC snapshots exactly to the eligible weekly panel."""
    lake = lake or Lake()
    cfg = load_config()
    start_year, end_year = int(cfg["start_year"]), int(cfg["end_year"])
    calendar = lake.read("calendar", "SSE")
    scheduled = industry_snapshot_dates(calendar.loc[
        calendar["cal_date"].astype(str).between(
            f"{start_year}-01-01", f"{end_year}-12-31"
        )
    ])
    available_schedule = [
        day for day in scheduled
        if lake.table_path("industry_snapshot_baostock", day).exists()
    ]
    if len(available_schedule) != len(scheduled):
        missing = sorted(set(scheduled) - set(available_schedule))
        raise FileNotFoundError(
            f"scheduled industry snapshots missing={len(missing)} sample={missing[:5]}"
        )
    for year in range(start_year, end_year + 1):
        destination = lake.table_path("pit_industry_exposures", str(year))
        if destination.exists() and not overwrite:
            continue
        panel = weekly_rows(pd.read_parquet(
            lake.table_path("multi_horizon_factor_panel", str(year)),
            columns=["instrument_id", "asof_date"],
        ))
        pieces = []
        snapshot_cache: dict[str, pd.DataFrame] = {}
        for asof_date, eligible in panel.groupby("asof_date", sort=True):
            position = bisect_right(available_schedule, str(asof_date)) - 1
            if position < 0:
                raise FileNotFoundError(
                    f"no PIT industry snapshot on or before {asof_date}"
                )
            snapshot_date = available_schedule[position]
            if snapshot_date not in snapshot_cache:
                snapshot = lake.read("industry_snapshot_baostock", snapshot_date)
                if snapshot.empty:
                    raise FileNotFoundError(
                        f"industry snapshot missing for {snapshot_date}"
                    )
                required = {
                    "instrument_id", "update_date", "industry_raw",
                    "industry_classification",
                }
                missing_columns = sorted(required - set(snapshot.columns))
                if missing_columns:
                    raise RuntimeError(
                        f"industry snapshot {snapshot_date} missing {missing_columns}"
                    )
                snapshot = snapshot[list(required)].copy()
                if snapshot.duplicated("instrument_id").any():
                    raise RuntimeError(
                        f"industry snapshot duplicate instruments: {snapshot_date}"
                    )
                update_date = pd.to_datetime(snapshot["update_date"], errors="coerce")
                if update_date.gt(pd.Timestamp(snapshot_date)).any():
                    raise RuntimeError(
                        f"industry snapshot contains future updates: {snapshot_date}"
                    )
                snapshot = snapshot.rename(
                    columns={"update_date": "source_update_date"}
                )
                snapshot["source_snapshot_date"] = snapshot_date
                snapshot_cache[snapshot_date] = snapshot
            snapshot = snapshot_cache[snapshot_date]
            merged = eligible.merge(
                snapshot, on="instrument_id", how="left", validate="one_to_one"
            )
            normalized = merged["industry_raw"].map(normalize_csrc_industry)
            merged[[
                "industry_taxonomy", "industry_detail_code", "industry_broad_code"
            ]] = pd.DataFrame(normalized.tolist(), index=merged.index)
            pieces.append(merged[INDUSTRY_OUTPUT_COLUMNS])
        output = pd.concat(pieces, ignore_index=True)
        lake.write(
            "pit_industry_exposures", output, key=str(year),
            source="risk_exposure.pit_industry.v1", request=cfg,
        )
        print(
            f"PIT industry exposure year={year} rows={len(output):,} "
            f"coverage={output['industry_broad_code'].notna().mean():.4%}",
            flush=True,
        )
    audit = audit_industry_exposures(lake, start_year, end_year, cfg)
    print(audit.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time risk exposures")
    parser.add_argument("command", choices=["build-size", "build-industry", "build-all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command in {"build-size", "build-all"}:
        build_size_exposures(overwrite=args.overwrite)
    if args.command in {"build-industry", "build-all"}:
        build_industry_exposures(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
