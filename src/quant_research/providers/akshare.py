"""AKShare adapter used only as an independent public-source cross-check."""

from __future__ import annotations

import pandas as pd


class AKShareProvider:
    source = "akshare.eastmoney"

    def daily_bars(self, instrument_id: str, start: str, end: str) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Create the Python 3.11 environment and run pip install -e .") from exc
        exchange, code = instrument_id.split(".", maxsplit=1)
        # AKShare's A-share endpoint expects the six-digit code; exchange is
        # retained in our canonical ID to avoid ambiguous tickers downstream.
        _ = exchange
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start.replace("-", ""),
            end_date=end.replace("-", ""), adjust=""
        )
        columns = {
            "日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount", "换手率": "turn", "涨跌幅": "pct_chg",
        }
        frame = raw.rename(columns=columns)
        keep = [c for c in columns.values() if c in frame.columns]
        frame = frame[keep].copy()
        frame["instrument_id"] = instrument_id
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
        return frame
