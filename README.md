# [CS1090A / AC209A Fall 2026] Forecasting Stock Volatility on the S&P 500

**Can pooled machine learning beat per-stock HAR at forecasting next-week realized volatility?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qqgjyx/AC-209A_VolForecast/blob/main/ms2.ipynb)
[![CI](https://github.com/qqgjyx/AC-209A_VolForecast/actions/workflows/ci.yml/badge.svg)](https://github.com/qqgjyx/AC-209A_VolForecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org)

Course project 12 for Harvard CS1090A / AC209A, Fall 2026, Canvas group 38: Ashley Lee, Juntang Wang,
Shuo Yang, Simon Manivong and Terry Su. TF mentor: Daniel More Torres.

## 📌 Question

For each S&P 500 stock and week, forecast next week's log realized variance, measured by the
Garman-Klass range estimator from daily prices.

1. Out of sample (2019-2026), does a pooled model with cross-sectional features (ridge/lasso,
   gradient-boosted trees) beat per-stock HAR on QLIKE and MSE, under a Diebold-Mariano test at 5%?
2. Which feature blocks account for any gain, and does it hold in calm and in stress weeks?

The portfolio backtest of the original proposal is out of scope, per the staff notes.

## 📦 Installation

```bash
git clone https://github.com/qqgjyx/AC-209A_VolForecast.git
cd AC-209A_VolForecast
uv sync            # or: pip install -e ".[notebook]"
```

Optional keys go in `.env` (git-ignored): `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` from a free
Alpaca paper-trading account, and `SEC_USER_AGENT="Your Name your@email"` for SEC EDGAR. The CRSP
check (notebook section 3l) needs a personal WRDS account: `WRDS_USERNAME` in `.env`, the password in
`~/.pgpass`, and `uv sync --extra crsp`.

## 🚀 Notebook

[`ms2.ipynb`](ms2.ipynb) is the Milestone 2 deliverable: it gets, understands and cleans the data,
builds the stock-week modeling table and states the revised question. Run it locally or on Colab
with the badge above. **No raw data is shipped**: the notebook downloads every source into `data/`
on its first run (about 440 MB, most of it SEC filings; about 10 minutes) and reads the cache
afterwards, in about a minute.

## 🗂️ Repository Structure

```
AC-209A_VolForecast/
├── ms2.ipynb               # Milestone 2: get, understand, clean, question (Colab-runnable)
├── src/volforecast/
│   ├── data.py             # fetch + cache: membership, Yahoo, Alpaca, FRED, SEC, CRSP; RENAMES, ...
│   ├── clean.py            # index spells, cleaning rules with a log, price adjustment, coverage
│   ├── features.py         # range estimators, weekly table and target, HAR, market, earnings
│   ├── alpha158.py         # qlib's Alpha158 without qlib, ported from FToken
│   └── plots.py            # the figures in images/
├── scripts/
│   ├── smoke_test.py       # every stage on synthetic data; --notebooks executes ms2.ipynb
│   └── build_rv5.py        # 5-minute realized variance for every member since 2016 (hours)
├── tests/                  # one test file per module, incl. no-lookahead tests
├── deliverables/ms2/       # slides and revised proposal (LaTeX + PDF)
├── data/README.md          # sources, licences, known gaps (the data itself is git-ignored)
├── images/                 # figures for the notebook, slides and proposal
└── out/                    # regenerable outputs, e.g. weekly.parquet (git-ignored)
```

## 📊 Data

| Source | What | Licence / terms |
|---|---|---|
| [fja05680/sp500](https://github.com/fja05680/sp500) | point-in-time S&P 500 membership since 1996 | see repo |
| [Yahoo Finance](https://pypi.org/project/yfinance/) via yfinance | daily OHLCV, 2005-2026 | no redistribution |
| [Alpaca](https://docs.alpaca.markets/docs/about-market-data-api) | daily and 5-minute bars from 2016, delisted symbols included | free account, no redistribution |
| [FRED](https://fred.stlouisfed.org/) | VIX, 3-month VIX, BAA-Treasury spread, 10y-2y slope, T-bill | public |
| [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | earnings dates from 8-K Item 2.02, SIC codes | public domain |
| [CRSP](https://wrds-www.wharton.upenn.edu) via WRDS | every member's daily prices 2005-2025, for the survivorship check | per-user licence, no redistribution |

See [`data/README.md`](data/README.md) for pinned versions, cleaning decisions and known gaps.

## 🔁 Development

```bash
uv run pytest                                    # unit tests, incl. no-lookahead checks
uv run python scripts/smoke_test.py              # offline pipeline check
uv run python scripts/smoke_test.py --notebooks  # also executes ms2.ipynb in place
uv run ruff check . && uv run ruff format --check src tests scripts && uv run mypy
```

CI runs lint, types, the tests and the offline smoke test on every pull request; it does not
execute the notebook, which needs the network. Work on a branch and open a pull request into `main`.
