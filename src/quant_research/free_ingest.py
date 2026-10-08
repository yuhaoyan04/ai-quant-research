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


def record_state(lake: Lake, stage: str, *, completed: str | None = None,
                 failure: str | None = None) -> None:
    """Persist a human-readable checkpoint after every successful unit of work."""
    root = lake.root / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "free_ingest.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prior[stage] = {"last_completed": completed, "last_failure": failure,
                    "updated_at": pd.Timestamp.utcnow().isoformat()}
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


def build_bars(provider: BaoStockProvider, lake: Lake, codes: list[str], start: str, end: str,
               *, limit: int | None = None, isolate_requests: bool = False,
               request_timeout_seconds: int = 180) -> None:
    # Index and other-security data are useful benchmarks, but are separate
    # products. Prioritize the A-share cross section needed by this project.
    codes = [code for code in codes if instrument_type(code) == "a_share"]
    if limit:
        codes = codes[:limit]
    for number, code in enumerate(codes, start=1):
        # Atomic parquet writes guarantee that file existence means completion;
        # loading every prior 20-year history here would make resume unusably slow.
        if lake.table_path("daily_bars_baostock", code).exists():
            continue
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build free-source A-share research inputs")
    parser.add_argument("command", choices=["snapshots", "bars", "bootstrap", "crosscheck"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, help="pilot mode: cap snapshot dates or securities")
    parser.add_argument("--isolate-bars", action="store_true",
                        help="fetch each bar request in a killable worker (recovery mode)")
    parser.add_argument("--request-timeout", type=int, default=180,
                        help="hard timeout per isolated bar request, in seconds")
    args = parser.parse_args()
    cfg = config()
    start = args.start or cfg["research_start"]
    lake = Lake()
    network = cfg["network"]
    if args.command in {"snapshots", "bootstrap", "bars"}:
        with BaoStockProvider(socket_timeout_seconds=int(network["socket_timeout_seconds"]),
                              max_attempts=int(network["max_attempts"]),
                              retry_backoff_seconds=int(network["retry_backoff_seconds"])) as provider:
            codes = build_snapshots(provider, lake, start, args.end, limit=args.limit)
            if args.command in {"bars", "bootstrap"}:
                build_bars(provider, lake, codes, start, args.end, limit=args.limit,
                           isolate_requests=args.isolate_bars,
                           request_timeout_seconds=args.request_timeout)
    if args.command == "crosscheck":
        crosscheck(lake, start, args.end, int(cfg["crosscheck"]["sample_size"]))


if __name__ == "__main__":
    main()
