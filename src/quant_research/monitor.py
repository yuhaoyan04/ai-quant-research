"""Low-overhead progress monitor for the long-running research pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_research.ingest import ROOT
from quant_research.panel import instrument_type


LAKE = ROOT / "data" / "lake"


def process_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, errors="replace")
    return str(pid) in result.stdout


def collect(collector_pid: int | None, materialization_pid: int | None, threshold: float) -> dict:
    union_path = LAKE / "curated" / "historical_security_union" / "key=all" / "data.parquet"
    if not union_path.exists():
        raise RuntimeError("Historical security union is not available yet.")
    union = pd.read_parquet(union_path, columns=["instrument_id"])["instrument_id"].dropna().astype(str)
    a_shares = {code for code in union if instrument_type(code) == "a_share"}
    bar_root = LAKE / "curated" / "daily_bars_baostock"
    bars = {path.parent.name.removeprefix("key=").replace("_", ".") for path in bar_root.glob("key=*/data.parquet")}
    downloaded = len(a_shares & bars)
    coverage = downloaded / len(a_shares) if a_shares else 0.0
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_a_share_union": len(a_shares),
        "downloaded_a_share_files": downloaded,
        "missing_a_share_files": len(a_shares) - downloaded,
        "download_coverage": round(coverage, 6),
        "coverage_threshold": threshold,
        "ready_for_factor_research": coverage >= threshold,
        "collector_running": process_alive(collector_pid),
        "materialization_running": process_alive(materialization_pid),
    }


def write_status(status: dict) -> None:
    root = LAKE / "monitoring"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "status.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "status.json")
    with (root / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(status, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor A-share data coverage")
    parser.add_argument("--collector-pid", type=int)
    parser.add_argument("--materialization-pid", type=int)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            status = collect(args.collector_pid, args.materialization_pid, args.threshold)
            write_status(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc), "checked_at_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
