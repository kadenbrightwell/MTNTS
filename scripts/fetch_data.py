"""CLI: Download and update historical OHLCV data for the full 27-ticker universe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

from config import ALL_TICKERS, TARGET_TICKERS
from src.data.storage import init_db, upsert_all, last_dates_all
from src.data.fetcher import fetch_bulk, fetch_incremental


@click.command()
@click.option("--full", is_flag=True, help="Force full re-download (ignore existing data).")
@click.option("--interval", default="1d", help="Data interval (1d, 1h, etc.).")
def main(full, interval):
    """Download or update market data for all feature and target tickers.

    Target tickers: UVXY, SPXU, SVIX, SPXL
    Plus 23 feature tickers spanning indices, volatility, bonds, commodities,
    currency, sectors, global markets, and crypto.
    """
    init_db()

    print(f"[FETCH] Targets: {', '.join(TARGET_TICKERS)}")
    print(f"[FETCH] Total universe: {len(ALL_TICKERS)} tickers")

    if full:
        print("[FETCH] Full download for all tickers...")
        data = fetch_bulk(tickers=ALL_TICKERS, interval=interval)
    else:
        existing = last_dates_all()
        if existing:
            print(f"[FETCH] Incremental update ({len(existing)} tickers have data)...")
            for tkr, d in existing.items():
                print(f"  {tkr}: last date = {d}")
        else:
            print("[FETCH] No existing data. Running full download...")

        data = fetch_incremental(
            tickers=ALL_TICKERS, last_dates=existing, interval=interval
        )

    counts = upsert_all(data)
    print("\n[FETCH] Upserted rows:")
    for tkr, n in counts.items():
        print(f"  {tkr}: {n} rows")

    total = sum(counts.values())
    print(f"\n  Total: {total} rows across {len(counts)} tickers.")


if __name__ == "__main__":
    main()
