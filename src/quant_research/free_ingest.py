"""Free-source, auditable A-share data-lake bootstrap.

This first phase creates a point-in-time *eligible A-share* universe from
historical BaoStock snapshots. It does not claim to reconstruct historical CSI
membership; a later licensed/official constituent source replaces the proxy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from quant_research.ingest import LAKE, ROOT, Lake, merge_records
from quant_research.providers.akshare import AKShareProvider
from quant_research.providers.baostock import BaoStockProvider
from quant_research.panel import instrument_type


CONFIG_PATH = ROOT / "config" / "free_sources.yaml"
def config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def weekly_dates(calendar: pd.DataFrame) -> list[str]:
    open_days = calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].copy()
    dates = pd.to_datetime(open_days)
    # Last trading day of each ISO week, so a weekly strategy only sees an
    # eligible-universe snapshot that existed before its next rebalance.
    return dates.groupby(dates.dt.to_period("W-FRI")).max().dt.strftime("%Y-%m-%d").tolist()


def industry_snapshot_dates(calendar: pd.DataFrame) -> list[str]:
    """Return a PIT-safe, rate-limit-friendly historical industry schedule."""
    strategy_dates = pd.Series(pd.to_datetime(weekly_dates(calendar)))
    if strategy_dates.empty:
        return []
    # Sample the last strategy signal date of each month, not a later calendar
    # month-end that the weekly strategy never uses.
    month_ends = strategy_dates.groupby(strategy_dates.dt.to_period("M")).max()
    first_week = strategy_dates.min()
    return sorted(
        set(month_ends.dt.strftime("%Y-%m-%d")) | {first_week.strftime("%Y-%m-%d")}
    )


def record_state(lake: Lake, stage: str, *, completed: str | None = None,
                 failure: str | None = None) -> None:
    """Persist a human-readable checkpoint after every successful unit of work."""
    root = lake.root / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "free_ingest.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prior[stage] = {"last_completed": completed, "last_failure": failure,
                    "updated_at": pd.Timestamp.now(tz="UTC").isoformat()}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_snapshots(provider: BaoStockProvider, lake: Lake, start: str, end: str,
                    *, limit: int | None = None) -> list[str]:
    calendar = provider.trade_calendar(start, end)
    lake.write("calendar", calendar, key="SSE", source="baostock.query_trade_dates",
               request={"start": start, "end": end})
    dates = weekly_dates(calendar)
    if limit:
        dates = dates[:limit]
    all_codes: set[str] = set()
    for number, asof in enumerate(dates, start=1):
        existing_path = lake.table_path("security_snapshot", asof)
        if existing_path.exists():
            # Read only the required column; snapshots are intentionally large.
            existing_ids = pd.read_parquet(existing_path, columns=["instrument_id"])["instrument_id"]
            all_codes.update(existing_ids.dropna().astype(str))
            continue
        try:
            snapshot = provider.securities_asof(asof)
            snapshot["asof_date"] = asof
            snapshot["universe_definition"] = "a_share_pit_proxy_weekly"
            lake.write("security_snapshot", snapshot, key=asof, source="baostock.query_all_stock",
                       request={"day": asof})
            all_codes.update(snapshot["instrument_id"].dropna().astype(str))
            record_state(lake, "snapshots", completed=asof)
            print(f"[snapshot {number}/{len(dates)}] {asof}", flush=True)
        except Exception as exc:
            record_state(lake, "snapshots", failure=f"{asof}: {exc}")
            print(f"[snapshot failed] {asof}: {exc}", flush=True)
    codes = sorted(all_codes)
    lake.write("historical_security_union", pd.DataFrame({"instrument_id": codes}), key="all",
               source="derived.security_snapshot_union.v1", request={"snapshot_frequency": "weekly"})
    return codes


def build_industry_snapshots(
    provider: BaoStockProvider, lake: Lake, start: str, end: str,
    *, limit: int | None = None, workers: int = 1,
) -> None:
    """Download historical CSRC-industry snapshots on the research week-ends."""
    if workers < 1:
        raise ValueError("industry workers must be at least 1")
    if workers != 1:
        raise ValueError(
            "BaoStock prohibits concurrent connections; industry workers must equal 1"
        )
    calendar = lake.read("calendar", "SSE")
    if calendar.empty:
        calendar = provider.trade_calendar(start, end)
    bounded = calendar.loc[
        calendar["cal_date"].astype(str).between(start, end)
    ].copy()
    dates = industry_snapshot_dates(bounded)
    if limit:
        dates = dates[:limit]
    pending = [
        (number, asof) for number, asof in enumerate(dates, start=1)
        if not lake.table_path("industry_snapshot_baostock", asof).exists()
    ]
    for number, asof in pending:
        try:
            frame = provider.industry_asof(asof)
            frame["asof_date"] = asof
            lake.write(
                "industry_snapshot_baostock", frame, key=asof,
                source="baostock.query_stock_industry", request={"date": asof},
            )
            record_state(lake, "industries", completed=asof)
            print(f"[industry {number}/{len(dates)}] {asof}", flush=True)
        except Exception as exc:
            record_state(lake, "industries", failure=f"{asof}: {exc}")
            print(f"[industry failed] {asof}: {exc}", flush=True)
            # Stop immediately instead of hammering the vendor after a broken
            # connection or renewed rate-limit event.  Successful atomic files
            # remain resumable on the next invocation.
            raise


def build_bars(provider: BaoStockProvider, lake: Lake, codes: list[str], start: str, end: str,
               *, limit: int | None = None, isolate_requests: bool = False,
               request_timeout_seconds: int = 180, workers: int = 1,
               extra_codes: list[str] | None = None) -> None:
    # Keep the research cross section restricted to A shares, while allowing
    # explicitly configured benchmark indices required by downstream audits.
    codes = sorted({
        code for code in codes if instrument_type(code) == "a_share"
    } | set(extra_codes or []))
    if limit:
        codes = codes[:limit]
    if workers < 1:
        raise ValueError("bar workers must be at least 1")
    if workers != 1:
        raise ValueError(
            "BaoStock prohibits concurrent connections; bar workers must equal 1"
        )
    if isolate_requests and workers != 1:
        raise ValueError("isolated bar requests cannot be combined with multiple workers")

    pending = [
        (number, code) for number, code in enumerate(codes, start=1)
        if not lake.table_path("daily_bars_baostock", code).exists()
    ]
    for number, code in pending:
        # Atomic parquet writes guarantee that file existence means completion;
        # loading every prior 20-year history here would make resume unusably slow.
        try:
            incoming = (isolated_daily_bars(code, start, end, provider.socket_timeout_seconds,
                                            provider.max_attempts, provider.retry_backoff_seconds,
                                            request_timeout_seconds)
                        if isolate_requests else provider.daily_bars(code, start, end))
            lake.write("daily_bars_baostock", incoming, key=code,
                       source="baostock.query_history_k_data_plus",
                       request={"code": code, "start": start, "end": end, "adjustflag": "3"})
            record_state(lake, "bars", completed=code)
            print(f"[bar {number}/{len(codes)}] {code}", flush=True)
        except Exception as exc:
            record_state(lake, "bars", failure=f"{code}: {exc}")
            print(f"[bar failed] {code}: {exc}", flush=True)


def _download_bar_worker(result_queue: mp.Queue, code: str, start: str, end: str,
                         socket_timeout_seconds: int, max_attempts: int,
                         retry_backoff_seconds: int) -> None:
    """Windows-spawn-safe worker: a vendor call cannot strand the parent batch."""
    try:
        with BaoStockProvider(socket_timeout_seconds=socket_timeout_seconds, max_attempts=max_attempts,
                              retry_backoff_seconds=retry_backoff_seconds) as worker:
            result_queue.put(("ok", worker.daily_bars(code, start, end)))
    except Exception as exc:
        result_queue.put(("error", repr(exc)))


def isolated_daily_bars(code: str, start: str, end: str, socket_timeout_seconds: int,
                        max_attempts: int, retry_backoff_seconds: int,
                        request_timeout_seconds: int) -> pd.DataFrame:
    """Fetch one code in a killable process, for recovery from vendor hangs."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_download_bar_worker,
        args=(result_queue, code, start, end, socket_timeout_seconds, max_attempts, retry_backoff_seconds),
    )
    worker.start()
    try:
        status, payload = result_queue.get(timeout=request_timeout_seconds)
    except queue.Empty:
        worker.terminate()
        worker.join(timeout=10)
        raise TimeoutError(f"BaoStock request exceeded {request_timeout_seconds}s for {code}")
    finally:
        if worker.is_alive():
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        worker.close()
        result_queue.close()
    if status != "ok":
        raise RuntimeError(f"BaoStock worker failed for {code}: {payload}")
    return payload


def crosscheck(lake: Lake, start: str, end: str, sample_size: int) -> None:
    files = sorted((LAKE / "curated" / "daily_bars_baostock").glob("key=*/data.parquet"))[:sample_size]
    ak = AKShareProvider()
    for path in files:
        code = path.parent.name.removeprefix("key=").replace("_", ".")
        primary = pd.read_parquet(path)
        alternate = ak.daily_bars(code, start, end)
        joined = primary.merge(alternate, on=["instrument_id", "trade_date"], suffixes=("_bs", "_ak"))
        if joined.empty:
            continue
        report = pd.DataFrame({
            "instrument_id": joined["instrument_id"], "trade_date": joined["trade_date"],
            "close_abs_diff": (pd.to_numeric(joined["close_bs"]) - pd.to_numeric(joined["close_ak"])).abs(),
            "volume_abs_diff": (pd.to_numeric(joined["volume_bs"]) - pd.to_numeric(joined["volume_ak"])).abs(),
        })
        lake.write("crosscheck_baostock_akshare", report, key=code,
                   source="baostock+akshare", request={"start": start, "end": end})


def verify_download(lake: Lake, start: str, end: str,
                    required_benchmarks: list[str] | None = None) -> dict[str, int]:
    """Fail closed unless the requested free-source download is complete.

    File existence alone is insufficient: a pilot file downloaded for a shorter
    range must not be mistaken for a complete history during a resumed full run.
    The append-only ingestion manifest supplies the request-range evidence.
    """
    union = lake.read("historical_security_union", "all")
    calendar = lake.read("calendar", "SSE")
    if union.empty or "instrument_id" not in union.columns:
        raise RuntimeError("historical_security_union is missing or empty")
    if calendar.empty:
        raise RuntimeError("SSE calendar is missing or empty")

    expected_a_shares = {
        code for code in union["instrument_id"].dropna().astype(str)
        if instrument_type(code) == "a_share"
    }
    benchmark_codes = set(required_benchmarks or [])
    expected_codes = expected_a_shares | benchmark_codes
    expected_snapshots = set(weekly_dates(calendar))
    missing_snapshots = sorted(
        day for day in expected_snapshots
        if not lake.table_path("security_snapshot", day).exists()
    )
    # A silent vendor pagination failure can leave a readable, full-page
    # snapshot (for example 2,000 or 4,000 rows). File existence alone would
    # accept it. Weekly A-share membership changes gradually, so compare each
    # snapshot with its immediate neighbors and fail closed on a sudden drop.
    snapshot_counts: dict[str, int] = {}
    for day in sorted(expected_snapshots - set(missing_snapshots)):
        identifiers = pd.read_parquet(
            lake.table_path("security_snapshot", day), columns=["instrument_id"]
        )["instrument_id"].dropna().astype(str)
        snapshot_counts[day] = int(sum(
            instrument_type(code) == "a_share" for code in identifiers
        ))
    minimum_adjacent_coverage = float(
        config()["universe"].get("minimum_adjacent_snapshot_coverage", 0.95)
    )
    ordered_snapshot_days = sorted(snapshot_counts)
    truncated_snapshots = []
    for index, day in enumerate(ordered_snapshot_days):
        neighbors = []
        if index:
            neighbors.append(snapshot_counts[ordered_snapshot_days[index - 1]])
        if index + 1 < len(ordered_snapshot_days):
            neighbors.append(snapshot_counts[ordered_snapshot_days[index + 1]])
        reference = max(neighbors, default=snapshot_counts[day])
        coverage = snapshot_counts[day] / reference if reference else 0.0
        if coverage < minimum_adjacent_coverage:
            truncated_snapshots.append((day, snapshot_counts[day], reference, coverage))
    missing_bar_files = sorted(
        code for code in expected_codes
        if not lake.table_path("daily_bars_baostock", code).exists()
    )

    latest_bar_request: dict[str, tuple[str | None, str | None, str | None]] = {}
    manifest_path = lake.manifests / "ingestion.jsonl"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid ingestion manifest JSON at line {line_number}: {exc}"
                    ) from exc
                if item.get("table") == "daily_bars_baostock":
                    request = item.get("request", {})
                    latest_bar_request[str(item.get("key"))] = (
                        item.get("status"), request.get("start"), request.get("end")
                    )
    covered_codes = {
        code for code, request_state in latest_bar_request.items()
        if request_state == ("ok", start, end)
    }
    uncovered_ranges = sorted(expected_codes - covered_codes)

    problems = []
    if missing_snapshots:
        problems.append(
            f"missing weekly snapshots={len(missing_snapshots)} sample={missing_snapshots[:5]}"
        )
    if truncated_snapshots:
        problems.append(
            "truncated weekly snapshots="
            f"{len(truncated_snapshots)} sample={truncated_snapshots[:5]}"
        )
    if missing_bar_files:
        problems.append(
            f"missing bar files={len(missing_bar_files)} sample={missing_bar_files[:5]}"
        )
    if uncovered_ranges:
        problems.append(
            f"bars without matching {start}..{end} manifest={len(uncovered_ranges)} "
            f"sample={uncovered_ranges[:5]}"
        )
    if problems:
        raise RuntimeError("download verification failed: " + "; ".join(problems))

    summary = {
        "a_share_codes": len(expected_a_shares),
        "benchmark_codes": len(benchmark_codes),
        "weekly_snapshots": len(expected_snapshots),
        "covered_bar_requests": len(covered_codes & expected_codes),
    }
    print(f"download verification passed: {summary}", flush=True)
    return summary


def verify_industry_download(lake: Lake, start: str, end: str) -> dict[str, int]:
    """Fail closed unless every scheduled industry snapshot is PIT-safe and readable."""
    calendar = lake.read("calendar", "SSE")
    if calendar.empty:
        raise RuntimeError("SSE calendar is missing or empty")
    bounded = calendar.loc[
        calendar["cal_date"].astype(str).between(start, end)
    ].copy()
    expected = industry_snapshot_dates(bounded)
    missing, invalid, rows, classified = [], [], 0, 0
    minimum_coverage = float(
        config()["industry"].get("minimum_security_snapshot_coverage", 0.95)
    )
    for asof in expected:
        frame = lake.read("industry_snapshot_baostock", asof)
        if frame.empty:
            missing.append(asof)
            continue
        required = {
            "instrument_id", "update_date", "industry_raw",
            "industry_classification", "asof_date",
        }
        absent = sorted(required - set(frame.columns))
        if absent:
            invalid.append(f"{asof}: missing_columns={absent}")
            continue
        update_dates = pd.to_datetime(frame["update_date"], errors="coerce")
        future_rows = int((update_dates > pd.Timestamp(asof)).sum())
        duplicate_rows = int(frame.duplicated("instrument_id").sum())
        wrong_asof = int(frame["asof_date"].astype(str).ne(asof).sum())
        securities = lake.read("security_snapshot", asof)
        expected_ids = {
            code for code in securities.get("instrument_id", pd.Series(dtype=str))
            .dropna().astype(str)
            if instrument_type(code) == "a_share"
        }
        received_ids = set(frame["instrument_id"].dropna().astype(str))
        security_coverage = (
            len(expected_ids & received_ids) / len(expected_ids)
            if expected_ids else 0.0
        )
        if (absent or future_rows or duplicate_rows or wrong_asof
                or security_coverage < minimum_coverage):
            invalid.append(
                f"{asof}: missing_columns={absent}, future={future_rows}, "
                f"duplicates={duplicate_rows}, wrong_asof={wrong_asof}, "
                f"security_coverage={security_coverage:.4%}"
            )
        rows += len(frame)
        classified += int(frame["industry_raw"].fillna("").astype(str).str.strip().ne("").sum())
    if missing or invalid:
        raise RuntimeError(
            "industry download verification failed: "
            f"missing={len(missing)} sample={missing[:5]}; "
            f"invalid={len(invalid)} sample={invalid[:3]}"
        )
    summary = {
        "industry_snapshots": len(expected),
        "industry_rows": rows,
        "classified_rows": classified,
    }
    print(f"industry download verification passed: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build free-source A-share research inputs")
    parser.add_argument(
        "command",
        choices=[
            "snapshots", "bars", "bootstrap", "industries",
            "crosscheck", "verify", "verify-industries",
        ],
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, help="pilot mode: cap snapshot dates or securities")
    parser.add_argument("--isolate-bars", action="store_true",
                        help="fetch each bar request in a killable worker (recovery mode)")
    parser.add_argument("--request-timeout", type=int, default=180,
                        help="hard timeout per isolated bar request, in seconds")
    parser.add_argument("--bar-workers", type=int, default=1,
                        help="must be 1: BaoStock prohibits concurrent connections")
    parser.add_argument("--industry-workers", type=int, default=None,
                        help="must be 1: BaoStock prohibits concurrent connections")
    args = parser.parse_args()
    if args.isolate_bars and args.bar_workers != 1:
        parser.error("--isolate-bars and --bar-workers > 1 are mutually exclusive")
    cfg = config()
    start = args.start or cfg["research_start"]
    lake = Lake()
    network = cfg["network"]
    if args.command in {"snapshots", "bootstrap", "bars", "industries"}:
        with BaoStockProvider(socket_timeout_seconds=int(network["socket_timeout_seconds"]),
                              max_attempts=int(network["max_attempts"]),
                              retry_backoff_seconds=int(network["retry_backoff_seconds"])) as provider:
            if args.command in {"snapshots", "bootstrap", "bars"}:
                codes = build_snapshots(provider, lake, start, args.end, limit=args.limit)
            if args.command in {"bars", "bootstrap"}:
                build_bars(provider, lake, codes, start, args.end, limit=args.limit,
                           isolate_requests=args.isolate_bars,
                           request_timeout_seconds=args.request_timeout,
                           workers=args.bar_workers,
                           extra_codes=(list(cfg.get("benchmark_indices", []))
                                        if args.limit is None else []))
            if args.command == "industries":
                build_industry_snapshots(
                    provider, lake, start, args.end, limit=args.limit,
                    workers=(args.industry_workers
                             or int(cfg["industry"]["default_workers"])),
                )
    if args.command == "crosscheck":
        crosscheck(lake, start, args.end, int(cfg["crosscheck"]["sample_size"]))
    if args.command == "verify":
        verify_download(lake, start, args.end,
                        required_benchmarks=list(cfg.get("benchmark_indices", [])))
    if args.command == "verify-industries":
        verify_industry_download(lake, start, args.end)


if __name__ == "__main__":
    main()
