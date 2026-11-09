# Data

Everything here except this file is gitignored and rebuilt by `notebooks/00_eda.ipynb`.

| Source | What | Access | Licence / terms |
|:--|:--|:--|:--|
| [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/) | daily OHLCV, 2005-2026 | unofficial scraper, may rate-limit | Yahoo terms restrict redistribution |
| [Stooq](https://stooq.com/db/h/) | daily OHLCV fallback | bulk download | personal use |
| [fja05680/sp500](https://github.com/fja05680/sp500) | S&P 500 membership by date | GitHub CSV | see repo |
| [SEC EDGAR XBRL company facts](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | quarterly fundamentals with filing dates, from mid-2009 | JSON API, needs a User-Agent with contact | public domain |
| [FRED](https://fred.stlouisfed.org/) | VIXCLS, DTB3, T10Y2Y | CSV download | public |

Record the download date and a checksum of each snapshot here once the data is pulled.
