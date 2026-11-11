"""Download, cache and parse the raw sources.

Every `fetch_*` function hits the network once and then reads its cache under `data/`.
"""

import json
import logging
import os
import re
import tarfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# fja05680/sp500 pinned to its 2026-09-07 commit; the last snapshot in it is 2026-08-18.
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/a2430f2af0c79ddf0748e91de11bdeb1616ab5a7/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CANARY = "SPY"  # always listed on Yahoo, never an index member
PRICE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
# Old index ticker -> the Yahoo symbol that now carries the same company's full history.
# Each pair was checked: the new symbol's Yahoo series starts before the old ticker's index
# entry (or at the company's IPO). Chains end at the latest symbol (SYMC -> NLOK -> GEN).
RENAMES = {
    "ABC": "COR", "ADS": "BFH", "ANTM": "ELV", "ARNC": "HWM", "BBT": "TFC", "BHGE": "BKR",
    "BK": "BNY", "BLL": "BALL", "CTL": "LUMN", "DWDP": "DD", "FB": "META", "FI": "FISV",
    "FLT": "CPAY", "GPS": "GAP", "HCP": "DOC", "HFC": "DINO", "HRS": "LHX", "JEC": "J",
    "KORS": "CPRI", "LB": "BBWI", "MMC": "MRSH", "MYL": "VTRS", "NLOK": "GEN", "PEAK": "DOC",
    "PKI": "RVTY", "RE": "EG", "SYMC": "GEN", "TMK": "GL", "UTX": "RTX", "WLTW": "WTW",
}  # fmt: skip
# Yahoo series that sit inside an index spell but belong to another security. Found as the
# in-index median dollar volume below $5M a day (an S&P 500 stock trades $100M+), then checked
# by hand: the price level or the history does not match the index member.
WRONG_SERIES = {
    "CCU": "Clear Channel, private in 2008; Yahoo's CCU is a Chilean brewer's ADR",
    "CPWR": "Compuware, private in 2014; Yahoo's CPWR has zero volume",
    "EP": "El Paso, acquired in 2012; Yahoo's EP trades ~100 shares a day",
    "GENZ": "Genzyme, acquired in 2011; Yahoo's GENZ trades ~9,000 shares a day",
    "KG": "King Pharmaceuticals, acquired in 2010; Yahoo's KG is priced near 140 USD",
    "PARA": "Paramount, merged in 2025; Yahoo's PARA is priced near 100,000 USD",
    "SII": "Smith International, acquired in 2010; Yahoo's SII has zero volume",
    "SUN": "Sunoco, acquired in 2012; Yahoo's SUN is Sunoco LP",
    "WEN": "Wendy's International, acquired in 2008; Yahoo's WEN history is Triarc's",
}
# Series that change company or price scale on a date: rows before it are dropped. Found as
# overnight jumps larger than 2x after cleaning (the others are real: Fannie Mae, Freddie Mac
# and AIG in 2008, First Republic in 2023), each checked by hand.
BREAKS = {
    "EVHC": ("2016-12-02", "AmSurg took the Envision name; earlier rows are the old Envision"),
    "JCI": ("2016-09-02", "Yahoo's JCI before the Tyco merger is Tyco's history"),
    "TGNA": ("2017-06-01", "Alpaca's TGNA before the Cars.com spin-off is on another price scale"),
    "WRK": ("2016-05-16", "Alpaca's WRK before this date is on another price scale"),
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{}"
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_START = "2016-01-01"  # the free plan's history starts here


def parse_membership(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn `date, "T1,T2,..."` snapshot rows into one row per (date, ticker).

    A plain loop: `str.split` + `explode` peaks at 5 GB on the 2,720 snapshots, this at 0.3 GB.
    """
    dates = pd.to_datetime(raw["date"])
    pairs = [(d, t) for d, s in zip(dates, raw["tickers"], strict=True) for t in s.split(",")]
    return pd.DataFrame(pairs, columns=["date", "ticker"])


def to_current(membership: pd.DataFrame) -> pd.DataFrame:
    """Membership with each renamed ticker replaced by the symbol that now carries its history."""
    return membership.assign(ticker=membership["ticker"].map(RENAMES).fillna(membership["ticker"]))


def rv_sample(membership: pd.DataFrame, n: int = 24, seed: int = 0) -> list[str]:
    """A fixed random sample of the last snapshot's members, for validating the target."""
    last = membership.loc[membership["date"] == membership["date"].max(), "ticker"]
    return sorted(map(str, np.random.default_rng(seed).choice(sorted(set(last)), n, replace=False)))


def universe(membership: pd.DataFrame, start: str) -> list[str]:
    """Tickers in the index on `start` or at any time after it."""
    first = membership.loc[membership["date"] <= start, "date"].max()
    return sorted(membership.loc[membership["date"] >= first, "ticker"].unique())


def fetch_membership(cache: Path) -> pd.DataFrame:
    """Point-in-time S&P 500 membership, one row per (snapshot date, ticker)."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(MEMBERSHIP_URL).to_csv(cache, index=False)
    return parse_membership(pd.read_csv(cache))


def _cached_chunks(
    tickers: list[str],
    cache_dir: Path,
    request: dict,
    download: Callable[[list[str]], pd.DataFrame],
) -> pd.DataFrame:
    """Download `tickers` in chunks of `request["chunk"]`, caching each chunk as parquet.

    A rerun skips finished chunks. `request.json` records the request, so a changed ticker list,
    date range or chunk size cannot silently reuse the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "request.json"
    if manifest.exists() and json.loads(manifest.read_text()) != request:
        raise ValueError(f"{cache_dir} was built for another request; delete it to refetch")
    manifest.write_text(json.dumps(request))
    size, frames = request["chunk"], []
    for i in range(0, len(tickers), size):
        path = cache_dir / f"chunk_{i // size:03d}.parquet"
        if not path.exists():
            download(tickers[i : i + size]).to_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames).sort_index()[list(PRICE_COLUMNS.values())]


def fetch_prices(
    tickers: list[str], start: str, end: str, cache_dir: Path, chunk_size: int = 100
) -> pd.DataFrame:
    """Daily Yahoo prices by (date, ticker); OHLC split-adjusted, adj_close also for dividends.

    Each chunk also asks for SPY as a canary: if SPY is missing the failure is a rate limit or
    an outage, not a delisting, so the chunk raises instead of caching. `end` is exclusive.
    """
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # else one line per delisted ticker

    def get(symbols: list[str], threads: bool) -> pd.DataFrame:
        wide = yf.download(
            [*symbols, CANARY], start=start, end=end, auto_adjust=False, actions=False,
            progress=False, threads=threads,
        )  # fmt: skip
        long = wide.stack(level="Ticker", future_stack=True).dropna(how="all")
        if CANARY not in long.index.unique("Ticker"):
            raise RuntimeError(f"{CANARY} missing: rate-limited, retry later")
        return long if CANARY in symbols else long.drop(CANARY, level="Ticker")

    def download(chunk: list[str]) -> pd.DataFrame:
        batch = {t.replace(".", "-"): t for t in chunk}  # Yahoo spells BRK.B as BRK-B
        long = get(list(batch), threads=True)
        absent = sorted(set(batch) - set(long.index.unique("Ticker")))
        if absent:  # ask once more, without threads, so a transient failure is not cached
            long = pd.concat([long, get(absent, threads=False)])
        long = long.rename(columns=PRICE_COLUMNS).rename(index=batch, level="Ticker")
        long.index.names, long.columns.name = ["date", "ticker"], None
        return long

    request = {"tickers": tickers, "start": start, "end": end, "chunk": chunk_size}
    return _cached_chunks(tickers, cache_dir, request, download)


def fetch_fred(series: list[str], cache_dir: Path) -> pd.DataFrame:
    """Daily FRED series as columns on a date index; FRED's "." for no value becomes NaN."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in series:
        path = cache_dir / f"{name}.csv"
        if not path.exists():
            pd.read_csv(FRED_URL.format(name)).to_csv(path, index=False)
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True, na_values="."))
    out = pd.concat(frames, axis=1)
    out.index.name = "date"
    return out


def _sec_json(url: str, cache: Path) -> dict:
    """GET a SEC JSON file once, with the User-Agent SEC's fair-access policy requires."""
    if not cache.exists():
        agent = os.environ.get("SEC_USER_AGENT")
        if not agent:
            raise RuntimeError('set SEC_USER_AGENT="Your Name your@email" in .env (SEC rule)')
        resp = requests.get(url, headers={"User-Agent": agent}, timeout=60)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
        time.sleep(0.12)  # SEC allows at most 10 requests per second
    return json.loads(cache.read_text())


def parse_filings(block: dict) -> pd.DataFrame:
    """One row per filing from a submissions `recent` block or an older-filings file."""
    cols = ["form", "items", "acceptanceDateTime"]
    out = pd.DataFrame({c: block.get(c, []) for c in cols})
    out["accepted"] = pd.to_datetime(out.pop("acceptanceDateTime"), utc=True)
    return out


def earnings_times(filings: pd.DataFrame) -> pd.Series:
    """Acceptance times (US/Eastern) of 8-K filings with Item 2.02, results of operations."""
    mask = filings["form"].eq("8-K") & filings["items"].str.contains("2.02", regex=False)
    times = filings.loc[mask, "accepted"].dt.tz_convert("America/New_York")
    return times.sort_values().reset_index(drop=True)


def fetch_sec_events(tickers: list[str], cache_dir: Path) -> pd.DataFrame:
    """Earnings announcement times (8-K Item 2.02) and SIC industry code per current ticker.

    Needs `SEC_USER_AGENT` in the environment. Pass only tickers that trade today: SEC maps
    current symbols, so an old ticker since reused by another company would get that company's
    filings. Tickers without a CIK on SEC's list are skipped.
    """
    ciks = pd.DataFrame(_sec_json(SEC_TICKERS_URL, cache_dir / "company_tickers.json").values())
    ciks = ciks.assign(ticker=ciks["ticker"].str.replace("-", ".")).drop_duplicates("ticker")
    cik_of = ciks.set_index("ticker")["cik_str"]
    rows = []
    for ticker in tickers:
        if ticker not in cik_of.index:
            continue
        name = f"CIK{int(cik_of[ticker]):010d}.json"
        sub = _sec_json(SEC_SUBMISSIONS_URL.format(name), cache_dir / name)
        blocks = [sub["filings"]["recent"]]
        for older in sub["filings"].get("files", []):
            url = SEC_SUBMISSIONS_URL.format(older["name"])
            blocks.append(_sec_json(url, cache_dir / older["name"]))
        times = earnings_times(pd.concat(parse_filings(b) for b in blocks))
        rows.append(pd.DataFrame({"ticker": ticker, "sic": sub.get("sic"), "accepted": times}))
    return pd.concat(rows, ignore_index=True)


def _alpaca_bars(symbols: list[str], timeframe: str, start: str, end: str, adjustment: str) -> dict:
    """All bars for `symbols` from Alpaca's consolidated (SIP) feed, following page tokens.

    Needs `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` (a free paper-trading account).
    """
    headers = {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
    }
    params: dict[str, str | int] = {
        "symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
        "adjustment": adjustment, "feed": "sip", "limit": 10_000,
    }  # fmt: skip
    bars: dict[str, list] = {}
    while True:
        resp = requests.get(ALPACA_BARS_URL, headers=headers, params=params, timeout=60)
        if resp.status_code == 429:  # the free plan allows 200 requests a minute
            time.sleep(5)
            continue
        resp.raise_for_status()
        body = resp.json()
        for sym, rows in (body.get("bars") or {}).items():
            bars.setdefault(sym, []).extend(rows)
        if not body.get("next_page_token"):
            return bars
        params["page_token"] = body["next_page_token"]


def _bars_frame(bars: dict) -> pd.DataFrame:
    """Alpaca bars as a long table with `time` in US/Eastern and its trading `date`."""
    frames = [pd.DataFrame(rows).assign(ticker=sym) for sym, rows in bars.items() if rows]
    if not frames:
        return pd.DataFrame(columns=["time", "date", "ticker", "o", "h", "l", "c", "v"])
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out.pop("t"), utc=True).dt.tz_convert("America/New_York")
    out["date"] = out["time"].dt.tz_localize(None).dt.normalize()
    return out


def fetch_alpaca_daily(
    tickers: list[str], end: str, cache_dir: Path, chunk_size: int = 100
) -> pd.DataFrame:
    """Daily bars from 2016 in `fetch_prices`' format, for tickers Yahoo lacks.

    Alpaca serves a symbol as it traded on each date, delisted ones included (TWTR, ATVI), and
    files a renamed company under its current symbol. OHLC and volume are split-adjusted;
    adj_close is also dividend-adjusted. Alpaca's PARA is as wrong as Yahoo's.
    """

    def download(chunk: list[str]) -> pd.DataFrame:
        def day(adjustment: str) -> pd.DataFrame:
            bars = _bars_frame(_alpaca_bars(chunk, "1Day", ALPACA_START, end, adjustment))
            return bars.set_index(["date", "ticker"])

        out = day("split").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                           "v": "volume"})  # fmt: skip
        return out.assign(adj_close=day("all")["c"])[list(PRICE_COLUMNS.values())].astype(float)

    request = {"tickers": tickers, "start": ALPACA_START, "end": end, "chunk": chunk_size}
    return _cached_chunks(tickers, cache_dir, request, download)


def realized_variance(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily realized variance from intraday bars of the regular session (09:30 to 16:00 ET).

    The first return of a day is open to close of the first bar, then close to close, so the
    overnight gap is left out, as in the Garman-Klass target.
    """
    t = bars["time"]
    b = bars[(t.dt.hour * 60 + t.dt.minute).between(9 * 60 + 30, 16 * 60 - 1)]
    b = b.sort_values(["ticker", "time"])
    prev = b.groupby(["ticker", "date"])["c"].shift().fillna(b["o"])
    r2 = np.log(b["c"] / prev) ** 2
    return r2.groupby([b["ticker"], b["date"]]).agg(rv5="sum", n_bars="size").reset_index()


def fetch_alpaca_rv(
    tickers: list[str], end: str, cache_dir: Path, workers: int = 8
) -> pd.DataFrame:
    """Daily 5-minute realized variance from 2016, one cached file per ticker.

    About 50 pages of bars per ticker, so tickers are fetched `workers` at a time; eight stay
    under the free plan's 200 requests a minute.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(ticker: str) -> Path:
        path = cache_dir / f"{ticker}.parquet"
        if not path.exists():
            bars = _bars_frame(_alpaca_bars([ticker], "5Min", ALPACA_START, end, "split"))
            realized_variance(bars).to_parquet(path)
        return path

    with ThreadPoolExecutor(workers) as pool:
        paths = list(pool.map(one, tickers))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


WRDS_URL = "postgresql+psycopg://{}@wrds-pgdata.wharton.upenn.edu:9737/wrds?sslmode=require"
IN_WINDOW = "mbrenddt >= :start and mbrstartdt < :end"  # spells overlapping [start, end)
CRSP_SPELLS = f"""
select permno, mbrstartdt as start, mbrenddt as "end" from crsp.dsp500list_v2 where {IN_WINDOW}
"""
CRSP_ROWS = f"""
select permno, dlycaldt as date, dlyopen as open, dlyhigh as high, dlylow as low,
       dlyclose as close, dlyvol as volume, dlycumfacpr as cfacpr
from crsp.dsf_v2
where permno in (select permno from crsp.dsp500list_v2 where {IN_WINDOW})
  and dlycaldt >= :start and dlycaldt < :end
"""


def fetch_crsp(start: str, end: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S&P 500 spells (permno, start, end inclusive) and their CRSP daily rows, from WRDS.

    Spells are cut at `end` (exclusive). Needs the `crsp` extra, a WRDS account named by
    `WRDS_USERNAME` and its password in `~/.pgpass`. CRSP is licensed per user: the cache under
    `data/` is never committed; its `request.json` refetches when the range changes.
    """
    paths = cache_dir / "sp500_spells.parquet", cache_dir / "daily.parquet"
    manifest, request = cache_dir / "request.json", {"start": start, "end": end}
    cached = manifest.exists() and json.loads(manifest.read_text()) == request
    if not (cached and all(p.exists() for p in paths)):
        from sqlalchemy import create_engine, text  # optional dependency, first download only

        cache_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(WRDS_URL.format(os.environ["WRDS_USERNAME"]))
        with engine.connect() as con:
            spells = pd.read_sql(text(CRSP_SPELLS), con, params=request, parse_dates=["start"])
            rows = pd.read_sql(text(CRSP_ROWS), con, params=request, parse_dates=["date"])
        last = pd.Timestamp(end) - pd.Timedelta(days=1)
        spells["end"] = pd.to_datetime(spells["end"]).clip(upper=last)
        spells.to_parquet(paths[0])
        rows.to_parquet(paths[1])
        manifest.write_text(json.dumps(request))
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def from_crsp(spells: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CRSP in the free pipeline's shape: membership snapshots and daily rows, keyed by permno.

    A snapshot is taken on every date the membership changes, as in fja05680/sp500, and the
    permno goes in the `ticker` column, so `clean`, `coverage` and the features run unchanged.
    `adj_close` divides by CRSP's split factor, so prices are adjusted for splits only.
    """
    s = spells.assign(ticker=spells["permno"].astype(str))
    dates = pd.DataFrame({"date": np.union1d(s["start"], s["end"] + pd.Timedelta(days=1))})
    m = dates.merge(s, how="cross")
    membership = m.loc[m["start"].le(m["date"]) & m["end"].ge(m["date"]), ["date", "ticker"]]
    d = rows.dropna(subset=["close"])
    d = d.assign(ticker=d["permno"].astype(str), adj_close=d["close"] / d["cfacpr"])
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return membership.reset_index(drop=True), d[cols].reset_index(drop=True)


# Never in a snapshot: CRSP is licensed per user, the rest are tracked or local logs.
SNAPSHOT_SKIP = {"crsp", "README.md", ".gitkeep", "rv5_build.log"}


def pack_snapshot(data_dir: Path, archive: Path) -> None:
    """Write every cache under `data_dir` except SNAPSHOT_SKIP to a .tar.gz archive."""
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(data_dir.iterdir()):
            if p.name not in SNAPSHOT_SKIP:
                tar.add(p, arcname=p.name)


def unpack_snapshot(archive: Path, data_dir: Path) -> None:
    """Extract a `pack_snapshot` archive into `data_dir`; paths outside it are refused."""
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")


def fetch_snapshot(url: str, data_dir: Path) -> None:
    """Download our cache snapshot (a Google Drive share link) and unpack it into `data_dir`."""
    import gdown  # preinstalled on Colab

    file_id = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", url)
    if not file_id:
        raise ValueError(f"not a Google Drive share link: {url}")
    archive = data_dir.parent / "snapshot.tar.gz"
    gdown.download(id=file_id[1] or file_id[2], output=str(archive), quiet=False)  # gdown 5 and 6
    unpack_snapshot(archive, data_dir)
    archive.unlink()


def size_mb(path: Path) -> float:
    """Total size in MB of a file or of every file under a directory."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / 1e6
