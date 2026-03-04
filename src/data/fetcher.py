"""Data acquisition via yfinance with bulk and incremental download modes."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import ALL_TICKERS, DATA_START_DATE


def fetch_bulk(
    tickers: List[str] | None = None,
    start: str = DATA_START_DATE,
    end: str | None = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Download full history for each ticker from *start* to *end*.

    Returns a dict mapping ticker -> DataFrame with columns
    [open, high, low, close, volume].
    """
    tickers = tickers or ALL_TICKERS
    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()

    result: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        print(f"  [FETCH] {tkr} ({interval}) {start} -> {end}")
        df = yf.download(
            tkr,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            print(f"  [FETCH] WARNING: No data returned for {tkr}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df.dropna(subset=["close"], inplace=True)
        result[tkr] = df

    return result


def fetch_incremental(
    tickers: List[str] | None = None,
    last_dates: Dict[str, str] | None = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Download only new data after *last_dates* for each ticker.

    *last_dates* maps ticker -> last stored date string (YYYY-MM-DD).
    Tickers missing from *last_dates* get a full download.
    """
    tickers = tickers or ALL_TICKERS
    last_dates = last_dates or {}
    end = (dt.date.today() + dt.timedelta(days=1)).isoformat()

    result: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        start = last_dates.get(tkr)
        if start is None:
            start = DATA_START_DATE
        else:
            start = (pd.Timestamp(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if pd.Timestamp(start) >= pd.Timestamp(end):
            continue

        print(f"  [FETCH] {tkr} incremental {start} -> {end}")
        df = yf.download(
            tkr, start=start, end=end, interval=interval,
            progress=False, auto_adjust=True,
        )
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df.dropna(subset=["close"], inplace=True)
        result[tkr] = df

    return result


def fetch_intraday(
    tickers: List[str] | None = None,
    interval: str = "5m",
    period: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch recent intraday data for live mode."""
    tickers = tickers or ALL_TICKERS
    result: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        df = yf.download(
            tkr, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        result[tkr] = df

    return result
