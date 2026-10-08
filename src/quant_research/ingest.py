"""Tushare-to-Parquet ingestion with auditable, point-in-time-safe raw inputs.

The module never derives features or forward fills data.  It only preserves
vendor records and their retrieval metadata so later research can apply an
explicit availability convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "universes.yaml"
LAKE = ROOT / "data" / "lake"


@dataclass(frozen=True)
class Config:
    start_date: str
    indices: list[dict[str, str]]
    refresh_days: int
    fundamental_endpoints: list[str]

    @classmethod
    def load(cls) -> "Config":
        with CONFIG_PATH.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            start_date=str(raw["start_date"]),
            indices=list(raw["indices"]),
            refresh_days=int(raw["recent_refresh_calendar_days"]),
            fundamental_endpoints=list(raw["fundamental_endpoints"]),
        )


class Lake:
    def __init__(self, root: Path = LAKE) -> None:
        self.root = root
        self.curated = root / "curated"
        self.manifests = root / "manifests"

    def table_path(self, table: str, key: str = "all") -> Path:
        safe_key = key.replace(".", "_").replace("/", "_")
        return self.curated / table / f"key={safe_key}" / "data.parquet"

    def write(self, table: str, frame: pd.DataFrame, *, key: str, source: str,
              request: dict[str, Any]) -> None:
        if frame.empty:
            self._manifest(table, key, source, request, frame, "empty")
            return
        self._validate(frame, table)
        path = self.table_path(table, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temp, index=False, compression="zstd")
        temp.replace(path)
        self._manifest(table, key, source, request, frame, "ok")

    def read(self, table: str, key: str = "all") -> pd.DataFrame:
        path = self.table_path(table, key)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _validate(self, frame: pd.DataFrame, table: str) -> None:
        if frame.columns.duplicated().any():
            raise ValueError(f"{table}: duplicate column names")
        # These are vendor natural keys used by the downstream deduplication job.
        candidate_keys = {
            "calendar": ["cal_date", "exchange"],
            "securities": ["ts_code"],
            "index_membership": ["index_code", "trade_date", "con_code"],
        }.get(table)
        if candidate_keys and all(c in frame.columns for c in candidate_keys):
            if frame.duplicated(candidate_keys).any():
                raise ValueError(f"{table}: duplicate natural key")

    def _manifest(self, table: str, key: str, source: str, request: dict[str, Any],
                  frame: pd.DataFrame, status: str) -> None:
        self.manifests.mkdir(parents=True, exist_ok=True)
        payload = {
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
            "table": table,
            "key": key,
            "source": source,
            "request": request,
            "status": status,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "content_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest() if not frame.empty else None,
        }
        with (self.manifests / "ingestion.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class TushareClient:
    def __init__(self) -> None:
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set. Set it in your shell; do not commit it.")
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError("Tushare is not installed. Run: pip install -e .") from exc
        ts.set_token(token)
        self.pro = ts.pro_api()

    def call(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        method = getattr(self.pro, endpoint)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                result = method(**{k: v for k, v in kwargs.items() if v is not None})
                if not isinstance(result, pd.DataFrame):
                    raise TypeError(f"{endpoint} did not return a DataFrame")
                return result
            except Exception as exc:  # Vendor/network errors should be visible after retries.
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Tushare endpoint '{endpoint}' failed after retries: {last_error}")


def ymd(value: str | date) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def merge_records(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Prefer newly retrieved vendor rows while retaining prior non-overlapping history."""
    if old.empty:
        return new.copy()
    if new.empty:
        return old.copy()
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    usable = [k for k in keys if k in combined.columns]
    if not usable:
        return combined.drop_duplicates().reset_index(drop=True)
    return combined.drop_duplicates(usable, keep="last").sort_values(usable).reset_index(drop=True)


def date_windows(start: str, end: str, *, months: int) -> Iterable[tuple[str, str]]:
    """Return inclusive API windows, keeping each response below vendor row caps."""
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cursor <= final:
        next_cursor = cursor + pd.DateOffset(months=months)
        window_end = min(next_cursor - pd.Timedelta(days=1), final)
        yield ymd(cursor), ymd(window_end)
        cursor = next_cursor


def call_windowed(client: TushareClient, endpoint: str, *, start: str, end: str,
                  months: int, **kwargs: Any) -> pd.DataFrame:
    frames = [
        client.call(endpoint, **kwargs, start_date=window_start, end_date=window_end)
        for window_start, window_end in date_windows(start, end, months=months)
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def download_reference(client: TushareClient, lake: Lake) -> None:
    securities = client.call("stock_basic", exchange="", list_status="L,D,P",
                             fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status")
    lake.write("securities", securities, key="all", source="tushare.stock_basic",
               request={"list_status": "L,D,P"})
    calendar = client.call("trade_cal", exchange="SSE", start_date="19900101", end_date=ymd(date.today()))
    lake.write("calendar", calendar, key="SSE", source="tushare.trade_cal",
               request={"exchange": "SSE", "start_date": "19900101"})


def download_membership(client: TushareClient, lake: Lake, config: Config) -> list[str]:
    union: set[str] = set()
    for item in config.indices:
        code = item["code"]
        # The endpoint can return far more than its per-request row cap for a
        # 20-year index history.  A monthly query window retains every snapshot.
        frame = call_windowed(client, "index_weight", index_code=code,
                              start=ymd(config.start_date), end=ymd(date.today()), months=1)
        if not frame.empty:
            if "index_code" not in frame.columns:
                frame["index_code"] = code
            else:
                frame["index_code"] = frame["index_code"].fillna(code)
            union.update(frame["con_code"].dropna().astype(str))
        lake.write("index_membership", frame, key=code, source="tushare.index_weight",
                   request={"index_code": code, "start_date": ymd(config.start_date), "end_date": ymd(date.today())})
    if not union:
        raise RuntimeError("No constituents downloaded. Check Tushare permissions for index_weight.")
    return sorted(union)


def per_security_history(client: TushareClient, lake: Lake, codes: Iterable[str], config: Config,
                         *, update: bool) -> None:
    endpoint_specs = {
        "daily_bars": ("daily", ["ts_code", "trade_date"]),
        "daily_valuation": ("daily_basic", ["ts_code", "trade_date"]),
        "adjustment_factors": ("adj_factor", ["ts_code", "trade_date"]),
        "daily_limits": ("stk_limit", ["ts_code", "trade_date"]),
        "suspensions": ("suspend_d", ["ts_code", "suspend_date"]),
        "name_changes": ("namechange", ["ts_code", "start_date", "end_date"]),
    }
    end = ymd(date.today())
    for code in codes:
        for table, (endpoint, keys) in endpoint_specs.items():
            old = lake.read(table, code)
            start = ymd(config.start_date)
            if update and not old.empty:
                date_col = next((c for c in ("trade_date", "suspend_date", "start_date") if c in old.columns), None)
                if date_col:
                    start = ymd(pd.to_datetime(old[date_col].max()) - timedelta(days=config.refresh_days))
            kwargs: dict[str, Any] = {"ts_code": code, "start_date": start, "end_date": end}
            # Twenty years of daily records can exceed a vendor response cap.
            # Event tables remain small; daily tables are split into ten-year windows.
            new = call_windowed(client, endpoint, start=start, end=end, months=120, ts_code=code) \
                if endpoint in {"daily", "daily_basic", "adj_factor", "stk_limit"} \
                else client.call(endpoint, **kwargs)
            merged = merge_records(old, new, keys)
            lake.write(table, merged, key=code, source=f"tushare.{endpoint}", request=kwargs)

        for endpoint in config.fundamental_endpoints:
            table = f"fundamental_{endpoint}"
            old = lake.read(table, code)
            # Full re-pull is intentional: vendors can restate old reports.  This
            # lake retains announcement fields; a later PIT resolver selects the
            # version available at each decision date.
            kwargs = {"ts_code": code, "start_date": ymd(config.start_date), "end_date": end}
            new = client.call(endpoint, **kwargs)
            keys = [k for k in ["ts_code", "ann_date", "end_date", "report_type", "comp_type"] if k in new.columns]
            lake.write(table, merge_records(old, new, keys), key=code,
                       source=f"tushare.{endpoint}", request=kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the point-in-time A-share research lake")
    parser.add_argument("command", choices=["bootstrap", "update"])
    parser.add_argument("--start", help="override configured start date (YYYY-MM-DD)")
    args = parser.parse_args()
    config = Config.load()
    if args.start:
        config = Config(args.start, config.indices, config.refresh_days, config.fundamental_endpoints)
    client, lake = TushareClient(), Lake()
    download_reference(client, lake)
    codes = download_membership(client, lake, config)
    per_security_history(client, lake, codes, config, update=args.command == "update")
    print(f"Completed {args.command}: {len(codes)} historical constituent securities.")


if __name__ == "__main__":
    main()
