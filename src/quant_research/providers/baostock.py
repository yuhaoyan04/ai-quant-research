"""BaoStock adapter with no credentials embedded in code."""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, TypeVar

import pandas as pd


T = TypeVar("T")


class BaoStockProvider:
    source = "baostock"

    def __init__(self, *, socket_timeout_seconds: int = 45, max_attempts: int = 4,
                 retry_backoff_seconds: int = 5) -> None:
        self._bs: Any | None = None
        self.socket_timeout_seconds = socket_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def __enter__(self) -> "BaoStockProvider":
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("BaoStock is not installed. Create the Python 3.11 environment and run pip install -e .") from exc
        self._install_socket_timeout(bs)
        error: Exception | None = None
        result: Any | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = bs.login()
                if result.error_code == "0":
                    break
                error = RuntimeError(result.error_msg)
                if result.error_code == "10001011":
                    # Retrying a provider blacklist only extends noisy traffic;
                    # fail fast and resume after the published release window.
                    break
            except Exception as exc:
                error = exc
            if attempt < self.max_attempts:
                time.sleep(self.retry_backoff_seconds * attempt)
        if result is None or result.error_code != "0":
            raise RuntimeError(
                f"BaoStock login failed after {self.max_attempts} attempts: {error}"
            )
        self._bs = bs
        return self

    def __exit__(self, *_: object) -> None:
        if self._bs is not None:
            self._bs.logout()

    def _frame(self, result: Any, operation: str) -> pd.DataFrame:
        if result.error_code != "0":
            raise RuntimeError(
                f"BaoStock {operation} failed [{result.error_code}]: {result.error_msg}"
            )
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise RuntimeError(
                f"BaoStock {operation} pagination failed "
                f"[{result.error_code}]: {result.error_msg}"
            )
        # ResultData leaves the previous full page in ``data`` when fetching
        # the next page fails.  A successful end-of-data response replaces it
        # with an empty page, so this detects the otherwise silent 2,000-row
        # truncation observed in historical industry downloads.
        if len(result.data) == int(result.per_page_count):
            raise RuntimeError(
                f"BaoStock {operation} ended on a full page; response may be truncated"
            )
        return pd.DataFrame(rows, columns=result.fields)

    def _install_socket_timeout(self, bs: Any) -> None:
        """Patch BaoStock's connection factory before login.

        BaoStock's shipped client creates a socket without a timeout.  A stalled
        vendor response can otherwise block a multi-day batch indefinitely.
        """
        import baostock.util.socketutil as socketutil
        import baostock.common.contants as constants

        timeout = self.socket_timeout_seconds

        def connect_with_timeout(self: Any, api_key: str) -> None:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(timeout)
            host = constants.BAOSTOCK_VIP_SERVER_IP if api_key.startswith("bs-") else constants.BAOSTOCK_SERVER_IP
            connection.connect((host, constants.BAOSTOCK_SERVER_PORT))
            setattr(socketutil.context, "default_socket", connection)

        socketutil.SocketUtil.connect = connect_with_timeout

    def _retry(self, operation: str, call: Callable[[], T]) -> T:
        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return call()
            except Exception as exc:
                error = exc
                if "10001011" in str(exc):
                    # A blacklist is time-based; reconnecting cannot clear it
                    # and only creates more prohibited traffic.
                    break
                if attempt == self.max_attempts:
                    break
                # Reconnect after a timeout/broken connection before retrying.
                try:
                    assert self._bs is not None
                    self._bs.logout()
                    result = self._bs.login()
                    if result.error_code != "0":
                        raise RuntimeError(result.error_msg)
                except Exception:
                    pass
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"BaoStock {operation} failed after {self.max_attempts} attempts: {error}")

    def trade_calendar(self, start: str, end: str) -> pd.DataFrame:
        assert self._bs is not None
        frame = self._retry("query_trade_dates", lambda: self._frame(
            self._bs.query_trade_dates(start_date=start, end_date=end), "query_trade_dates"))
        return frame.rename(columns={"calendar_date": "cal_date", "is_trading_day": "is_open"})

    def securities_asof(self, asof_date: str) -> pd.DataFrame:
        assert self._bs is not None
        frame = self._retry("query_all_stock", lambda: self._frame(
            self._bs.query_all_stock(day=asof_date), "query_all_stock"))
        return frame.rename(columns={"code": "instrument_id", "code_name": "name", "tradeStatus": "trade_status"})

    def industry_asof(self, asof_date: str) -> pd.DataFrame:
        """Return the historical CSRC industry snapshot available on ``asof_date``."""
        assert self._bs is not None
        frame = self._retry("query_stock_industry", lambda: self._frame(
            self._bs.query_stock_industry(date=asof_date), "query_stock_industry"
        ))
        return frame.rename(columns={
            "updateDate": "update_date",
            "code": "instrument_id",
            "code_name": "name",
            "industry": "industry_raw",
            "industryClassification": "industry_classification",
        })

    def daily_bars(self, instrument_id: str, start: str, end: str) -> pd.DataFrame:
        assert self._bs is not None
        fields = (
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
            "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        )
        frame = self._retry("query_history_k_data_plus", lambda: self._frame(
            self._bs.query_history_k_data_plus(
                instrument_id, fields, start_date=start, end_date=end, frequency="d", adjustflag="3"
            ), "query_history_k_data_plus"))
        return frame.rename(columns={
            "date": "trade_date", "code": "instrument_id", "preclose": "pre_close",
            "tradestatus": "trade_status", "pctChg": "pct_chg", "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq", "psTTM": "ps_ttm", "pcfNcfTTM": "pcf_ncf_ttm", "isST": "is_st",
        })
