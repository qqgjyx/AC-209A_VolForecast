"""Fetch daily 5-minute realized variance from Alpaca for every company in the index since 2016.

Usage:
    uv run python scripts/build_rv5.py   # needs ALPACA_* in .env; hours, resumable per ticker

Writes data/rv5/<TICKER>.parquet. The notebook's validation sample comes first.
"""

from pathlib import Path

from dotenv import load_dotenv

from volforecast.data import (
    ALPACA_START,
    fetch_alpaca_rv,
    fetch_membership,
    rv_sample,
    to_current,
    universe,
)

ROOT = Path(__file__).resolve().parents[1]
END = "2026-11-07"


def main() -> None:
    load_dotenv(ROOT / ".env")
    m = to_current(fetch_membership(ROOT / "data" / "sp500_membership.csv"))
    sample = rv_sample(m)  # the notebook's validation sample first
    rest = sorted(set(universe(m, ALPACA_START)) - set(sample))
    rv = fetch_alpaca_rv(sample + rest, END, ROOT / "data" / "rv5")
    print(f"{rv['ticker'].nunique()} tickers, {len(rv):,} stock-days")


if __name__ == "__main__":
    main()
