"""SQLite-backed storage with upsert semantics and merged DataFrame reads.

Handles the full 27-ticker universe including index symbols (^VIX, ^TNX)
by sanitizing ticker names into valid SQLite table names.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_PATH, ALL_TICKERS


def _table_name(ticker: str) -> str:
    return "ohlcv_" + re.sub(r"[^a-zA-Z0-9]", "_", ticker).lower()


def _engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db() -> None:
    """Create tables for all tickers if they don't exist."""
    engine = _engine()
    with engine.begin() as conn:
        for tkr in ALL_TICKERS:
            tbl = _table_name(tkr)
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low  REAL,
                    close REAL,
                    volume REAL
                )
            """))


def upsert(ticker: str, df: pd.DataFrame) -> int:
    """Insert or replace rows for *ticker*. Returns number of rows written."""
    if df.empty:
        return 0

    tbl = _table_name(ticker)
    engine = _engine()

    tmp = df.copy()
    if tmp.index.name == "date" or "date" not in tmp.columns:
        tmp = tmp.reset_index()
    tmp["date"] = tmp["date"].astype(str).str[:10]

    with engine.begin() as conn:
        for _, row in tmp.iterrows():
            conn.execute(text(f"""
                INSERT OR REPLACE INTO {tbl} (date, open, high, low, close, volume)
                VALUES (:date, :open, :high, :low, :close, :volume)
            """), {
                "date": row["date"],
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })

    return len(tmp)


def upsert_all(data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Upsert data for multiple tickers. Returns dict of row counts."""
    counts = {}
    for tkr, df in data.items():
        counts[tkr] = upsert(tkr, df)
    return counts


def last_date(ticker: str) -> Optional[str]:
    """Return the most recent date stored for *ticker*, or None."""
    tbl = _table_name(ticker)
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(text(f"SELECT MAX(date) FROM {tbl}")).fetchone()
    if row and row[0]:
        return row[0]
    return None


def last_dates_all(tickers: List[str] | None = None) -> Dict[str, str]:
    """Return {ticker: last_date} for all tickers that have data."""
    tickers = tickers or ALL_TICKERS
    result = {}
    for tkr in tickers:
        d = last_date(tkr)
        if d:
            result[tkr] = d
    return result


def read_ticker(ticker: str) -> pd.DataFrame:
    """Read full OHLCV history for a single ticker."""
    tbl = _table_name(ticker)
    engine = _engine()
    df = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY date", engine, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


def read_all(tickers: List[str] | None = None) -> pd.DataFrame:
    """Read and merge all tickers into a single DataFrame.

    Returns a DataFrame with a DatetimeIndex and MultiIndex columns:
    (ticker, field) where field is one of [open, high, low, close, volume].
    """
    tickers = tickers or ALL_TICKERS
    frames = {}
    for tkr in tickers:
        df = read_ticker(tkr)
        if not df.empty:
            frames[tkr] = df

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1)
    merged.sort_index(inplace=True)
    return merged
