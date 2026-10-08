"""Leakage-safe, auditable cross-sectional preprocessing for model features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from quant_research.factor_research import eligible_research_rows, pit_snapshot_membership
from quant_research.ingest import ROOT, Lake
from quant_research.panel import instrument_type


@dataclass(frozen=True)
class PreprocessingResult:
    features: pd.DataFrame
    audit: pd.DataFrame


def load_config() -> dict:
    with (ROOT / "config" / "preprocessing.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preprocess_cross_section(
    frame: pd.DataFrame,
    factor_columns: list[str],
    *,
    date_column: str = "asof_date",
    mad_multiplier: float = 5.0,
) -> PreprocessingResult:
    """Median/MAD winsorize, median-impute and z-score independently each day.

    Missing indicators preserve the information that a value was unavailable.
    An all-missing or zero-variance daily factor becomes a neutral zero z-score.
    No statistic is shared between dates, so future observations cannot affect
    an earlier feature value.
    """
    required = {date_column, *factor_columns}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"preprocessing columns missing: {sorted(missing_columns)}")

    output = frame.copy()
    audit_frames: list[pd.DataFrame] = []
    dates = output[date_column]
    for factor in factor_columns:
        values = pd.to_numeric(output[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = values.isna()
        median = values.groupby(dates).transform("median")
        absolute_deviation = (values - median).abs()
        mad = absolute_deviation.groupby(dates).transform("median")
        robust_scale = 1.4826 * mad
        usable_scale = robust_scale.where(robust_scale > 0)
        lower = median - mad_multiplier * usable_scale
        upper = median + mad_multiplier * usable_scale
        winsorized = values.where(lower.isna() | (values >= lower), lower)
        winsorized = winsorized.where(upper.isna() | (winsorized <= upper), upper)
        clipped = values.notna() & winsorized.ne(values)

        # A missing value receives a neutral cross-sectional exposure.  If the
        # entire factor is missing that day, zero is used and the indicator is 1.
        filled = winsorized.fillna(median).fillna(0.0)
        daily_mean = filled.groupby(dates).transform("mean")
        daily_std = filled.groupby(dates).transform("std")
        zscore = ((filled - daily_mean) / daily_std.where(daily_std > 0)).fillna(0.0)

        output[f"{factor}__z"] = zscore.astype("float32")
        output[f"{factor}__missing"] = missing.astype("int8")

        audit = pd.DataFrame({
            date_column: dates,
            "factor": factor,
            "missing": missing.astype("int64"),
            "clipped": clipped.astype("int64"),
            "median": median,
            "mad": mad,
            "lower": lower,
            "upper": upper,
        }).groupby([date_column, "factor"], as_index=False).agg(
            observations=("missing", "size"),
            missing_count=("missing", "sum"),
            clipped_count=("clipped", "sum"),
            median=("median", "first"),
            mad=("mad", "first"),
            lower=("lower", "first"),
            upper=("upper", "first"),
        )
        audit["missing_fraction"] = audit["missing_count"] / audit["observations"]
        audit["clipped_fraction"] = audit["clipped_count"] / audit["observations"]
        audit_frames.append(audit)

    return PreprocessingResult(output, pd.concat(audit_frames, ignore_index=True))


def load_eligible_period(lake: Lake, start: str, end: str, factor_columns: list[str],
                         target: str) -> pd.DataFrame:
    """Assemble one bounded period without constructing the full 20-year panel."""
    factor_files = {p.parent.name.removeprefix("key="): p
                    for p in (lake.curated / "factors_raw").glob("key=*/data.parquet")}
    label_files = {p.parent.name.removeprefix("key="): p
                   for p in (lake.curated / "labels_raw").glob("key=*/data.parquet")}
    memberships, snapshot_state = pit_snapshot_membership(lake)
    frames: list[pd.DataFrame] = []
    for key in sorted(set(factor_files) & set(label_files)):
        code = key.replace("_", ".")
        if instrument_type(code) != "a_share":
            continue
        factor_fields = ["instrument_id", "asof_date", "available_date", "is_tradable", *factor_columns]
        factors = pd.read_parquet(factor_files[key], columns=factor_fields)
        factors = factors.loc[factors["asof_date"].between(start, end)]
        if factors.empty:
            continue
        labels = pd.read_parquet(
            label_files[key],
            columns=["instrument_id", "asof_date", "quality_status", "is_tradable_at_asof", target],
        )
        labels = labels.loc[labels["asof_date"].between(start, end)]
        merged = factors.merge(labels, on=["instrument_id", "asof_date"], how="inner")
        eligible = eligible_research_rows(merged, memberships.get(code, set()), snapshot_state)
        if not eligible.empty:
            frames.append(eligible)
    if not frames:
        raise RuntimeError(f"no eligible model rows between {start} and {end}")
    return pd.concat(frames, ignore_index=True)


def _flush_year_buffer(year: str, buffers: dict[str, list[pd.DataFrame]],
                       writers: dict[str, pq.ParquetWriter], temporary_paths: dict[str, Path],
                       row_counts: dict[str, int]) -> None:
    chunks = buffers.get(year, [])
    if not chunks:
        return
    batch = pd.concat(chunks, ignore_index=True)
    table = pa.Table.from_pandas(batch, preserve_index=False)
    if year not in writers:
        temporary_paths[year].parent.mkdir(parents=True, exist_ok=True)
        writers[year] = pq.ParquetWriter(temporary_paths[year], table.schema, compression="zstd")
    writers[year].write_table(table)
    row_counts[year] = row_counts.get(year, 0) + len(batch)
    buffers[year] = []


def stage_eligible_panels(lake: Lake | None = None, *, overwrite: bool = False,
                          buffer_rows: int = 100_000) -> pd.DataFrame:
    """Transpose per-security inputs into atomic yearly PIT-eligible panels.

    This scans every security once.  Subsequent preprocessing/model experiments
    read one year directly rather than rescanning thousands of Parquet files.
    """
    lake = lake or Lake()
    cfg = load_config()
    factor_columns = list(cfg["factor_columns"])
    target = str(cfg["target"])
    factor_files = {p.parent.name.removeprefix("key="): p
                    for p in (lake.curated / "factors_raw").glob("key=*/data.parquet")}
    label_files = {p.parent.name.removeprefix("key="): p
                   for p in (lake.curated / "labels_raw").glob("key=*/data.parquet")}
    memberships, snapshot_state = pit_snapshot_membership(lake)
    buffers: dict[str, list[pd.DataFrame]] = {}
    buffered_rows: dict[str, int] = {}
    writers: dict[str, pq.ParquetWriter] = {}
    temporary_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    row_counts: dict[str, int] = {}

    try:
        for key in sorted(set(factor_files) & set(label_files)):
            code = key.replace("_", ".")
            if instrument_type(code) != "a_share":
                continue
            factors = pd.read_parquet(
                factor_files[key],
                columns=["instrument_id", "asof_date", "available_date", "is_tradable", *factor_columns],
            )
            labels = pd.read_parquet(
                label_files[key],
                columns=["instrument_id", "asof_date", "quality_status", "is_tradable_at_asof", target],
            )
            merged = factors.merge(labels, on=["instrument_id", "asof_date"], how="inner")
            eligible = eligible_research_rows(merged, memberships.get(code, set()), snapshot_state)
            if eligible.empty:
                continue
            for year, chunk in eligible.groupby(eligible["asof_date"].str.slice(0, 4), sort=False):
                final = lake.table_path("eligible_factor_panel_staging", year)
                if final.exists() and not overwrite:
                    continue
                if year not in final_paths:
                    final_paths[year] = final
                    temporary_paths[year] = final.with_suffix(".tmp.parquet")
                    if temporary_paths[year].exists():
                        temporary_paths[year].unlink()
                buffers.setdefault(year, []).append(chunk)
                buffered_rows[year] = buffered_rows.get(year, 0) + len(chunk)
                if buffered_rows[year] >= buffer_rows:
                    _flush_year_buffer(year, buffers, writers, temporary_paths, row_counts)
                    buffered_rows[year] = 0
        for year in list(buffers):
            _flush_year_buffer(year, buffers, writers, temporary_paths, row_counts)
    finally:
        for writer in writers.values():
            writer.close()

    for year, temporary in temporary_paths.items():
        final_paths[year].parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_paths[year])
    index = pd.DataFrame([
        {"year": int(year), "rows": rows, "path": str(final_paths[year])}
        for year, rows in sorted(row_counts.items())
    ])
    if not index.empty:
        lake.write("preprocessing_staging_index", index, key="all",
                   source="preprocessing.eligible_panel_staging.v1", request=cfg)
    return index


def materialize_year(year: int, lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = load_config()
    factors = list(cfg["factor_columns"])
    target = str(cfg["target"])
    staged_path = lake.table_path("eligible_factor_panel_staging", str(year))
    panel = (pd.read_parquet(staged_path) if staged_path.exists()
             else load_eligible_period(lake, f"{year}-01-01", f"{year}-12-31", factors, target))
    result = preprocess_cross_section(
        panel, factors, mad_multiplier=float(cfg["winsorization"]["mad_multiplier"]),
    )
    usable_dates = result.features.groupby("asof_date")["instrument_id"].transform("size") \
        >= int(cfg["minimum_cross_section_size"])
    features = result.features.loc[usable_dates]
    model_columns = ["instrument_id", "asof_date", "available_date", target]
    model_columns += [f"{factor}__z" for factor in factors]
    model_columns += [f"{factor}__missing" for factor in factors]
    lake.write("model_features", features[model_columns], key=str(year),
               source="preprocessing.cross_sectional.v1", request=cfg)
    lake.write("preprocessing_audit", result.audit, key=str(year),
               source="preprocessing.cross_sectional.v1", request=cfg)
    print(f"year={year} rows={len(features)} dates={features['asof_date'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize leakage-safe model features")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--year", type=int)
    mode.add_argument("--stage", action="store_true", help="build yearly eligible-panel staging files")
    mode.add_argument("--all", action="store_true", help="preprocess every staged year")
    parser.add_argument("--overwrite-stage", action="store_true")
    args = parser.parse_args()
    if args.stage:
        print(stage_eligible_panels(overwrite=args.overwrite_stage).to_string(index=False))
    elif args.all:
        lake = Lake()
        paths = sorted((lake.curated / "eligible_factor_panel_staging").glob("key=*/data.parquet"))
        if not paths:
            raise RuntimeError("no staged yearly panels; run --stage first")
        for path in paths:
            materialize_year(int(path.parent.name.removeprefix("key=")), lake)
    else:
        materialize_year(args.year)


if __name__ == "__main__":
    main()
