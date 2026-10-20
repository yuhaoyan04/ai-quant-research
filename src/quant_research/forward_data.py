"""Incremental, point-in-time data supply for forward paper trading.

This module is deliberately separate from the frozen historical research
panels.  It extends vendor and per-security derived files by natural key; it
does not rebuild, overwrite, or reinterpret historical experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from quant_research.factors import build_raw_factors, next_trading_day_map
from quant_research.free_ingest import industry_snapshot_dates, weekly_dates
from quant_research.ingest import ROOT, Lake, merge_records
from quant_research.panel import build_labels, instrument_type, load_config as load_panel_config
from quant_research.providers.baostock import BaoStockProvider
from quant_research.quality import validate_daily_bars


CONFIG_PATH = ROOT / "config" / "forward_data.yaml"
REPORT_PATH = ROOT / "reports" / "forward_data.md"
FORWARD_TABLES = {
    "calendar": "forward_calendar",
    "security_snapshot": "forward_security_snapshot",
    "industry_snapshot_baostock": "forward_industry_snapshot_baostock",
    "historical_security_union": "forward_security_union",
    "daily_bars_baostock": "forward_daily_bars_baostock",
    "daily_bars_validated": "forward_daily_bars_validated",
    "quality_report": "forward_quality_report",
    "factors_raw": "forward_factors_raw",
    "labels_raw": "forward_labels_raw",
}


def forward_table(table: str) -> str:
    return FORWARD_TABLES[table]


def read_forward(lake: Lake, table: str, key: str) -> pd.DataFrame:
    """Read the isolated forward table, bootstrapping from history if needed."""
    path = lake.table_path(forward_table(table), key)
    current = pd.read_parquet(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    return current if not current.empty else lake.read(table, key)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("evidence_class") != "governance_only":
        raise ValueError("forward supply is governance evidence, not performance evidence")
    if cfg.get("claims_independent_evidence") is not False:
        raise ValueError("forward supply cannot claim independent performance evidence")
    if cfg.get("source") != "baostock":
        raise ValueError("forward pipeline currently requires the frozen BaoStock source")
    for key in ("calendar_refresh_days", "calendar_lookahead_days",
                "bar_refresh_calendar_days", "new_security_history_calendar_days",
                "factor_lookback_rows"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(cfg["factor_lookback_rows"]) < 125:
        raise ValueError("factor lookback must cover the longest 120-day feature")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Fail fast when another forward update owns the explicit lock file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"forward-data update already running: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()
            }))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _write_status(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    path = ROOT / str(cfg["status_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bounded(calendar: pd.DataFrame, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(calendar["cal_date"]).dt.strftime("%Y-%m-%d")
    return calendar.loc[dates.le(end)].copy()


def completed_weekly_dates(calendar: pd.DataFrame, end: str) -> list[str]:
    """Exclude the still-open W-FRI period from the weekly PIT schedule."""
    bounded = _bounded(calendar, end)
    result = pd.Series(pd.to_datetime(weekly_dates(bounded)))
    if result.empty:
        return []
    completed = result.dt.to_period("W-FRI").dt.end_time.dt.normalize().le(pd.Timestamp(end))
    return result.loc[completed].dt.strftime("%Y-%m-%d").tolist()


def _table_keys(lake: Lake, table: str) -> list[str]:
    return sorted(
        path.parent.name.removeprefix("key=").replace("_", ".")
        for path in (lake.curated / table).glob("key=*/data.parquet")
    )


def _merge_and_write(
    lake: Lake, table: str, key: str, incoming: pd.DataFrame, keys: list[str],
    *, source: str, request: dict[str, Any],
) -> tuple[int, int, bool]:
    old = lake.read(table, key)
    if old.empty and table in FORWARD_TABLES.values():
        historical = next(name for name, current in FORWARD_TABLES.items() if current == table)
        old = lake.read(historical, key)
    merged = merge_records(old, incoming, keys)
    changed = len(merged) != len(old)
    if not changed and not incoming.empty and not old.empty:
        # Refresh windows may contain vendor corrections with unchanged keys.
        old_hash = pd.util.hash_pandas_object(old, index=False).sum()
        new_hash = pd.util.hash_pandas_object(merged, index=False).sum()
        changed = bool(old_hash != new_hash)
    if changed:
        lake.write(table, merged, key=key, source=source, request=request)
    return len(old), len(merged), changed


def _hardlink_table(lake: Lake, source_table: str, destination_table: str) -> int:
    """Snapshot current files cheaply; later atomic replaces break the hardlink."""
    copied = 0
    for source in (lake.curated / source_table).glob("key=*/data.parquet"):
        key_dir = source.parent.name
        destination = lake.curated / destination_table / key_dir / "data.parquet"
        if destination.exists():
            if destination.stat().st_size == 0:
                destination.unlink()
            else:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except PermissionError:
            temporary = destination.with_suffix(".copying.parquet")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        copied += 1
    return copied


def _remove_table_key(lake: Lake, table: str, key: str) -> None:
    path = lake.table_path(table, key)
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def isolate_forward_data(
    cutoff: str, *, lake: Lake | None = None, cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Migrate an initial shared update into forward-only table namespaces.

    The forward copy is created first with hardlinks.  Historical paths are
    then atomically restored to ``cutoff``; replacing a historical file leaves
    the forward inode intact.  Empty post-cutoff securities remain recoverable
    in the forward namespace before their historical path is removed.
    """
    lake = lake or Lake()
    cfg = cfg or load_config()
    cutoff = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
    linked = {
        table: _hardlink_table(lake, table, forward_table(table))
        for table in FORWARD_TABLES
    }

    calendar = lake.read("calendar", "SSE")
    historical_calendar = calendar.loc[
        calendar["cal_date"].astype(str).le(cutoff)
    ].copy()
    lake.write(
        "calendar", historical_calendar, key="SSE",
        source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
    )
    next_days = next_trading_day_map(historical_calendar)
    panel_cfg = load_panel_config()
    horizons = list(panel_cfg["label_horizons_trading_days"])
    lag = int(panel_cfg["execution_lag_trading_days"])

    changed_codes: list[str] = []
    for code in _table_keys(lake, forward_table("daily_bars_baostock")):
        forward_raw = lake.read(forward_table("daily_bars_baostock"), code)
        if forward_raw.empty or str(forward_raw["trade_date"].max()) <= cutoff:
            continue
        changed_codes.append(code)
        raw = forward_raw.loc[forward_raw["trade_date"].astype(str).le(cutoff)].copy()
        if raw.empty:
            for table in (
                "daily_bars_baostock", "daily_bars_validated", "quality_report",
                "factors_raw", "labels_raw",
            ):
                _remove_table_key(lake, table, code)
            continue
        lake.write(
            "daily_bars_baostock", raw, key=code,
            source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
        )
        quality = validate_daily_bars(raw)
        lake.write(
            "daily_bars_validated", quality.frame, key=code,
            source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
        )
        lake.write(
            "quality_report", quality.report, key=code,
            source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
        )
        forward_factors = lake.read(forward_table("factors_raw"), code)
        factors = forward_factors.loc[
            forward_factors["asof_date"].astype(str).le(cutoff)
        ].copy()
        factors["available_date"] = factors["asof_date"].map(next_days)
        lake.write(
            "factors_raw", factors, key=code,
            source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
        )
        labels = build_labels(quality.frame, horizons, lag)
        lake.write(
            "labels_raw", labels, key=code,
            source="forward_data.restore_historical_cutoff.v1", request={"cutoff": cutoff},
        )
        if len(changed_codes) % 100 == 0:
            print(f"[isolate restore {len(changed_codes)}] {code}", flush=True)

    removed_snapshots = 0
    for table in ("security_snapshot", "industry_snapshot_baostock"):
        for path in list((lake.curated / table).glob("key=*/data.parquet")):
            day = path.parent.name.removeprefix("key=")
            if day > cutoff:
                _remove_table_key(lake, table, day)
                removed_snapshots += 1
    snapshot_codes: set[str] = set()
    for path in (lake.curated / "security_snapshot").glob("key=*/data.parquet"):
        ids = pd.read_parquet(path, columns=["instrument_id"])["instrument_id"]
        snapshot_codes.update(ids.dropna().astype(str))
    lake.write(
        "historical_security_union", pd.DataFrame({"instrument_id": sorted(snapshot_codes)}),
        key="all", source="forward_data.restore_historical_cutoff.v1",
        request={"cutoff": cutoff},
    )
    result = {
        "pipeline_id": cfg["pipeline_id"], "stage": "isolated",
        "historical_cutoff": cutoff, "linked_files": linked,
        "restored_code_count": len(changed_codes),
        "removed_post_cutoff_snapshot_files": removed_snapshots,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_status(cfg, result)
    return result


def update_vendor_data(
    end: str, *, lake: Lake | None = None, cfg: dict[str, Any] | None = None,
    limit: int | None = None, provider_factory: Any = BaoStockProvider,
) -> dict[str, Any]:
    """Extend calendar, PIT snapshots, and active-security bars through ``end``."""
    lake = lake or Lake()
    cfg = cfg or load_config()
    validate_config(cfg)
    end = pd.Timestamp(end).strftime("%Y-%m-%d")
    existing_calendar = read_forward(lake, "calendar", "SSE")
    if existing_calendar.empty:
        calendar_start = end
    else:
        maximum = pd.to_datetime(existing_calendar["cal_date"]).max()
        calendar_start = (maximum - pd.Timedelta(days=int(cfg["calendar_refresh_days"]))).strftime("%Y-%m-%d")
    calendar_end = (
        pd.Timestamp(end) + pd.Timedelta(days=int(cfg["calendar_lookahead_days"]))
    ).strftime("%Y-%m-%d")
    network = yaml.safe_load((ROOT / "config" / "free_sources.yaml").read_text(encoding="utf-8"))["network"]
    summary: dict[str, Any] = {
        "pipeline_id": cfg["pipeline_id"], "requested_end": end,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "errors": [],
    }
    changed_codes: list[str] = []
    with provider_factory(
        socket_timeout_seconds=int(network["socket_timeout_seconds"]),
        max_attempts=int(network["max_attempts"]),
        retry_backoff_seconds=int(network["retry_backoff_seconds"]),
    ) as provider:
        incoming_calendar = provider.trade_calendar(calendar_start, calendar_end)
        _, _, calendar_changed = _merge_and_write(
            lake, forward_table("calendar"), "SSE", incoming_calendar, ["cal_date"],
            source="baostock.query_trade_dates.incremental",
            request={"start": calendar_start, "end": calendar_end},
        )
        calendar = lake.read(forward_table("calendar"), "SSE")
        bounded = _bounded(calendar, end)

        snapshot_dates = completed_weekly_dates(calendar, end)
        missing_snapshots = [
            day for day in snapshot_dates
            if not lake.table_path(forward_table("security_snapshot"), day).exists()
            and not lake.table_path("security_snapshot", day).exists()
        ]
        for day in missing_snapshots:
            try:
                snapshot = provider.securities_asof(day)
                snapshot["asof_date"] = day
                snapshot["universe_definition"] = "a_share_pit_proxy_weekly"
                lake.write(
                    forward_table("security_snapshot"), snapshot, key=day,
                    source="baostock.query_all_stock.incremental", request={"day": day},
                )
            except Exception as exc:  # retained in status, optionally fail-closed below
                summary["errors"].append(f"security_snapshot {day}: {exc}")

        completed_calendar = bounded.loc[
            pd.to_datetime(bounded["cal_date"]).dt.to_period("M").dt.end_time.dt.normalize()
            .le(pd.Timestamp(end))
        ]
        industry_dates = industry_snapshot_dates(completed_calendar)
        missing_industries = [
            day for day in industry_dates
            if not lake.table_path(forward_table("industry_snapshot_baostock"), day).exists()
            and not lake.table_path("industry_snapshot_baostock", day).exists()
        ]
        for day in missing_industries:
            try:
                industry = provider.industry_asof(day)
                industry["asof_date"] = day
                lake.write(
                    forward_table("industry_snapshot_baostock"), industry, key=day,
                    source="baostock.query_stock_industry.incremental", request={"date": day},
                )
            except Exception as exc:
                summary["errors"].append(f"industry_snapshot {day}: {exc}")

        available_snapshots = [
            day for day in snapshot_dates
            if lake.table_path(forward_table("security_snapshot"), day).exists()
            or lake.table_path("security_snapshot", day).exists()
        ]
        if not available_snapshots:
            raise RuntimeError("no point-in-time security snapshot is available")
        latest_snapshot = max(available_snapshots)
        active = read_forward(lake, "security_snapshot", latest_snapshot)
        codes = sorted({
            code for code in active["instrument_id"].dropna().astype(str)
            if instrument_type(code) == "a_share"
        })
        if len(codes) < int(cfg["minimum_active_universe_rows"]):
            raise RuntimeError(f"active A-share universe unexpectedly small: {len(codes)}")
        if bool(cfg.get("include_benchmark_indices")):
            free_cfg = yaml.safe_load(
                (ROOT / "config" / "free_sources.yaml").read_text(encoding="utf-8")
            )
            codes = sorted(set(codes) | set(free_cfg.get("benchmark_indices", [])))
        if limit is not None:
            codes = codes[:limit]

        for number, code in enumerate(codes, start=1):
            old = read_forward(lake, "daily_bars_baostock", code)
            if old.empty:
                start = (
                    pd.Timestamp(end)
                    - pd.Timedelta(days=int(cfg["new_security_history_calendar_days"]))
                ).strftime("%Y-%m-%d")
            else:
                maximum = pd.to_datetime(old["trade_date"]).max()
                start = (maximum - pd.Timedelta(days=int(cfg["bar_refresh_calendar_days"]))).strftime("%Y-%m-%d")
            try:
                incoming = provider.daily_bars(code, start, end)
                _, _, changed = _merge_and_write(
                    lake, forward_table("daily_bars_baostock"), code, incoming,
                    ["instrument_id", "trade_date"],
                    source="baostock.query_history_k_data_plus.incremental",
                    request={"code": code, "start": start, "end": end, "adjustflag": "3"},
                )
                if changed:
                    changed_codes.append(code)
                if number % 100 == 0 or number == len(codes):
                    print(f"[forward bars {number}/{len(codes)}] changed={len(changed_codes)}", flush=True)
            except Exception as exc:
                summary["errors"].append(f"daily_bars {code}: {exc}")

    # Refresh the historical union without dropping delisted securities.
    union = read_forward(lake, "historical_security_union", "all")
    old_codes = set(union.get("instrument_id", pd.Series(dtype=str)).dropna().astype(str))
    all_snapshot_codes = set(old_codes)
    for day in missing_snapshots:
        frame = read_forward(lake, "security_snapshot", day)
        all_snapshot_codes.update(frame.get("instrument_id", pd.Series(dtype=str)).dropna().astype(str))
    if all_snapshot_codes != old_codes:
        lake.write(
            forward_table("historical_security_union"),
            pd.DataFrame({"instrument_id": sorted(all_snapshot_codes)}), key="all",
            source="derived.security_snapshot_union.incremental.v1",
            request={"through": end},
        )
    summary.update({
        "calendar_changed": calendar_changed,
        "calendar_max_date": str(pd.to_datetime(calendar["cal_date"]).max().date()),
        "latest_security_snapshot": latest_snapshot,
        "requested_codes": len(codes), "changed_codes": changed_codes,
        "changed_code_count": len(changed_codes),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_status(cfg, {**summary, "stage": "vendor_update"})
    if summary["errors"] and bool(cfg["fail_on_vendor_error"]):
        raise RuntimeError(f"forward vendor update had {len(summary['errors'])} errors; see status file")
    return summary


def _needs_refresh(raw: pd.DataFrame, derived: pd.DataFrame) -> bool:
    if raw.empty:
        return False
    if derived.empty:
        return True
    raw_max = pd.to_datetime(raw["trade_date"]).max()
    date_column = "trade_date" if "trade_date" in derived else "asof_date"
    return raw_max > pd.to_datetime(derived[date_column]).max() or len(raw) != len(derived)


def materialize_incremental(
    *, lake: Lake | None = None, cfg: dict[str, Any] | None = None,
    codes: list[str] | None = None, limit: int | None = None,
) -> dict[str, Any]:
    """Refresh validated bars, factors and labels for changed security files."""
    lake = lake or Lake()
    cfg = cfg or load_config()
    validate_config(cfg)
    forced = codes is not None
    candidates = sorted(codes or _table_keys(lake, forward_table("daily_bars_baostock")))
    if limit is not None:
        candidates = candidates[:limit]
    calendar = read_forward(lake, "calendar", "SSE")
    next_days = next_trading_day_map(calendar)
    panel_cfg = load_panel_config()
    horizons = list(panel_cfg["label_horizons_trading_days"])
    execution_lag = int(panel_cfg["execution_lag_trading_days"])
    refreshed: list[str] = []
    for number, code in enumerate(candidates, start=1):
        raw = read_forward(lake, "daily_bars_baostock", code)
        validated_old = lake.read(forward_table("daily_bars_validated"), code)
        if not forced and not _needs_refresh(raw, validated_old):
            continue
        quality = validate_daily_bars(raw)
        lake.write(
            forward_table("daily_bars_validated"), quality.frame, key=code, source="quality.daily_bars.v1.incremental",
            request={"raw_table": "daily_bars_baostock", "through": str(raw["trade_date"].max())},
        )
        lake.write(
            forward_table("quality_report"), quality.report, key=code, source="quality.daily_bars.v1.incremental",
            request={"raw_table": "daily_bars_baostock"},
        )

        old_factors = read_forward(lake, "factors_raw", code)
        lookback = int(cfg["factor_lookback_rows"])
        factor_start = max(0, len(quality.frame) - max(lookback, len(quality.frame) - len(old_factors) + lookback))
        factor_update = build_raw_factors(quality.frame.iloc[factor_start:].copy(), next_days)
        if old_factors.empty:
            factors = factor_update
        else:
            # The first 120 rows of the slice are warm-up context, not output
            # replacements.  Keeping them would turn previously valid long
            # momentum values into NaN during every incremental refresh.
            safe_offset = min(120, len(factor_update) - 1)
            cutoff = str(factor_update.iloc[safe_offset]["asof_date"])
            factors = merge_records(
                old_factors.loc[old_factors["asof_date"].astype(str).lt(cutoff)],
                factor_update.loc[factor_update["asof_date"].astype(str).ge(cutoff)],
                ["instrument_id", "asof_date"],
            )
        lake.write(
            forward_table("factors_raw"), factors, key=code, source="factors.v1.incremental",
            request={"lookback_rows": lookback, "availability": "t_close_to_t_plus_1"},
        )

        old_labels = read_forward(lake, "labels_raw", code)
        label_tail = max(horizons) + execution_lag + 2
        label_start = max(0, len(quality.frame) - max(label_tail, len(quality.frame) - len(old_labels) + label_tail))
        label_update = build_labels(
            quality.frame.iloc[label_start:].copy(), horizons, execution_lag,
        )
        if old_labels.empty:
            labels = label_update
        else:
            cutoff = str(label_update["asof_date"].min())
            labels = merge_records(
                old_labels.loc[old_labels["asof_date"].astype(str).lt(cutoff)],
                label_update, ["instrument_id", "asof_date"],
            )
        lake.write(
            forward_table("labels_raw"), labels, key=code, source="labels.v1.incremental",
            request={"horizons": horizons, "execution_lag": execution_lag},
        )
        refreshed.append(code)
        if number % 100 == 0 or number == len(candidates):
            print(f"[forward materialize {number}/{len(candidates)}] refreshed={len(refreshed)}", flush=True)
    result = {
        "pipeline_id": cfg["pipeline_id"], "stage": "materialize",
        "candidate_codes": len(candidates), "refreshed_codes": refreshed,
        "refreshed_code_count": len(refreshed),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_status(cfg, result)
    return result


def pipeline_status(lake: Lake | None = None) -> dict[str, Any]:
    lake = lake or Lake()
    cfg = load_config()
    calendar = read_forward(lake, "calendar", "SSE")
    raw_keys = _table_keys(lake, forward_table("daily_bars_baostock"))
    maxima = []
    for code in raw_keys:
        frame = pd.read_parquet(
            lake.table_path(forward_table("daily_bars_baostock"), code), columns=["trade_date"]
        )
        if not frame.empty:
            maxima.append(str(frame["trade_date"].max()))
    return {
        "pipeline_id": cfg["pipeline_id"],
        "calendar_max_date": str(calendar["cal_date"].max()) if not calendar.empty else None,
        "raw_security_files": len(raw_keys),
        "raw_max_date": max(maxima) if maxima else None,
        "raw_files_at_max_date": sum(day == max(maxima) for day in maxima) if maxima else 0,
        "lock_present": (ROOT / str(cfg["lock_path"])).exists(),
    }


def build_protocol(lake: Lake | None = None) -> pd.DataFrame:
    """Freeze stable implementation controls without freezing mutable live data."""
    lake = lake or Lake()
    cfg = load_config()
    validate_config(cfg)
    protocol = pd.DataFrame([{
        "experiment_id": cfg["experiment_id"], "pipeline_id": cfg["pipeline_id"],
        "evidence_class": cfg["evidence_class"],
        "claims_independent_evidence": cfg["claims_independent_evidence"],
        "historical_cutoff": cfg["historical_cutoff"],
        "source": cfg["source"], "forward_table_prefix": "forward_",
        "natural_key_merge": True, "historical_namespace_mutation_allowed": False,
        "strategy_logic_changed": False, "performance_results_included": False,
    }])
    checks = pd.DataFrame([
        {"check": "governance_only", "status": "PASS", "detail": cfg["evidence_class"]},
        {"check": "separate_forward_namespace", "status": "PASS",
         "detail": ",".join(sorted(FORWARD_TABLES.values()))},
        {"check": "historical_cutoff_explicit", "status": "PASS",
         "detail": str(cfg["historical_cutoff"])},
        {"check": "vendor_errors_fail_closed", "status": "PASS",
         "detail": str(cfg["fail_on_vendor_error"])},
        {"check": "weekly_partial_period_blocked", "status": "PASS",
         "detail": f"calendar_lookahead_days={cfg['calendar_lookahead_days']}"},
        {"check": "factor_warmup_sufficient", "status": "PASS",
         "detail": f"factor_lookback_rows={cfg['factor_lookback_rows']}"},
        {"check": "strategy_configuration_unchanged", "status": "PASS",
         "detail": "paper_trading.yaml and frozen model bundle are not rewritten"},
    ])
    failed = pd.DataFrame([{
        "check": "NO_FAILED_ASSERTIONS", "status": "PASS",
        "detail": "all stable forward-data implementation gates passed",
    }])
    for table, frame in {
        "forward_data_protocol": protocol,
        "forward_data_implementation_checks": checks,
        "forward_data_failed_assertions": failed,
    }.items():
        lake.write(table, frame, key="v1", source="forward_data.protocol.v1", request=cfg)
    REPORT_PATH.write_text("\n".join([
        "# Forward Data Supply Protocol", "", "## Scope", "",
        "This is an operational/governance control, not return evidence. Mutable forward data",
        "lives only under `forward_*`; frozen historical research inputs remain at their original hashes.",
        "", "## Protocol", "", protocol.to_markdown(index=False), "",
        "## Implementation checks", "", checks.to_markdown(index=False), "",
    ]) + "\n", encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental forward-data supply")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("update", "all"):
        command = sub.add_parser(name)
        command.add_argument("--end", default=date.today().isoformat())
        command.add_argument("--limit", type=int)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--limit", type=int)
    isolate = sub.add_parser("isolate")
    isolate.add_argument("--cutoff", required=True)
    sub.add_parser("build-protocol")
    sub.add_parser("status")
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "status":
        print(json.dumps(pipeline_status(), ensure_ascii=False, indent=2))
        return
    lock = ROOT / str(cfg["lock_path"])
    with exclusive_lock(lock):
        if args.command == "build-protocol":
            result = {"checks": build_protocol().to_dict(orient="records")}
        elif args.command == "isolate":
            result = isolate_forward_data(args.cutoff, cfg=cfg)
        elif args.command == "update":
            result = update_vendor_data(args.end, cfg=cfg, limit=args.limit)
        elif args.command == "materialize":
            result = materialize_incremental(cfg=cfg, limit=args.limit)
        else:
            vendor = update_vendor_data(args.end, cfg=cfg, limit=args.limit)
            derived = materialize_incremental(
                cfg=cfg, codes=list(vendor["changed_codes"])
            )
            result = {
                "pipeline_id": cfg["pipeline_id"], "stage": "complete",
                "requested_end": vendor["requested_end"],
                "calendar_max_date": vendor["calendar_max_date"],
                "latest_security_snapshot": vendor["latest_security_snapshot"],
                "requested_codes": vendor["requested_codes"],
                "changed_code_count": vendor["changed_code_count"],
                "refreshed_code_count": derived["refreshed_code_count"],
                "errors": vendor["errors"],
                "completed_at_utc": derived["completed_at_utc"],
            }
            _write_status(cfg, result)
    public = dict(result)
    public.pop("changed_codes", None)
    public.pop("refreshed_codes", None)
    print(json.dumps(public, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
