# Data

**No raw data is committed to this repository**; everything here except this file is git-ignored.
On a first run [`ms2.ipynb`](../ms2.ipynb) unpacks the team bucket of these caches from Google
Drive (`scripts/pack_data.py`); with `BUCKET_ID=""` it downloads every source through
`volforecast.data` instead. The Yahoo, Alpaca daily and `crsp/` caches hold a `request.json`, so a
changed ticker list or date range cannot silently reuse them; the other caches are per file and
never refreshed, so delete them to refetch.

## Sources

| Path | Source | Pinned to | Licence / terms |
|---|---|---|---|
| `sp500_membership.csv` | [fja05680/sp500](https://github.com/fja05680/sp500), `S&P 500 Historical Components & Changes (Updated).csv` | commit `a2430f2` (2026-09-07); last snapshot 2026-08-18 | see repo |
| `prices/`, `market/` | Yahoo Finance via [yfinance](https://pypi.org/project/yfinance/) | 2005-01-01 to 2026-11-06 | Yahoo terms restrict redistribution |
| `alpaca_daily/`, `rv5/` | [Alpaca market data](https://docs.alpaca.markets/docs/about-market-data-api), SIP feed | 2016-01-01 to 2026-11-06 | free account; no redistribution |
| `fred/` | [FRED](https://fred.stlouisfed.org/): VIXCLS, VXVCLS, BAA10Y, T10Y2Y, DTB3 | downloaded on first run | public |
| `sec/` | [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) submissions | downloaded on first run | public domain |
| `crsp/` | [CRSP](https://wrds-www.wharton.upenn.edu) `dsf_v2` and `dsp500list_v2` via WRDS | 2005-01-01 to 2025-12-31 (annual vintage) | team licence; in the private bucket only |

**Licence note.** Yahoo's and Alpaca's terms restrict redistribution and CRSP is licensed, so no
data is ever committed. The bucket is shared only with the team and the TF, CRSP under the team's
WRDS licence, and its link lives only in this private repository and the course submission.

## Identity tables

Both live in `src/volforecast/data.py` and were checked by hand.

- `RENAMES`: 30 old index tickers whose company Yahoo and Alpaca now file under a new symbol
  (BK -> BNY, MMC -> MRSH, ANTM -> ELV, ...). The new symbol's history covers the old ticker's spell.
- `WRONG_SERIES`: 9 tickers whose Yahoo series sits inside an index spell but belongs to another
  security (PARA near 100,000 USD, CCU a Chilean ADR, ...); found as an in-index median dollar
  volume under $5M a day.

## Known gaps

- **Survivorship before 2016.** Yahoo keeps no history for delisted companies, and Alpaca's free
  history starts in 2016, so firms that left before 2016 are missing. The notebook reports coverage
  by month, and section 3l measures the cost against CRSP: a third fewer stock-weeks before 2016,
  with a mean log variance 0.04 below CRSP's in the same year; from 2016 the gap is under 0.01.
- **Ticker reuse.** A symbol can now belong to another company; prices are kept only near each
  ticker's index spells (120 days before, 14 after).
- **Membership after 2026-08-18** is carried forward from the last snapshot.
- **Industry codes** from SEC are today's, not historical.
