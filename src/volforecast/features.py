"""Daily volatility measures and the stock-week modeling table.

One row of the weekly table is one stock in one week (weeks end on Friday). Every feature uses
data up to that Friday's close; the target is the log realized variance of the next week.
"""

import numpy as np
import pandas as pd

from volforecast.alpha158 import alpha158
from volforecast.clean import adjust_prices

LN2 = np.log(2)
VAR_FLOOR = 1e-6  # weekly variance floor (0.1% weekly vol) so that the log stays finite
MAX_GAP = pd.Timedelta(days=10)  # a longer gap in a ticker's rows starts a new run of lags
OHLCV = ["open", "high", "low", "close", "volume"]
WEEKLY = {  # weekly column: (daily column, aggregation)
    "rv": ("gk", "sum"), "rv_parkinson": ("parkinson", "sum"), "rv_rs": ("rs", "sum"),
    "rv_overnight": ("overnight2", "sum"), "rv_neg": ("neg2", "sum"), "ret": ("ret", "sum"),
    "last_gk": ("gk", "last"), "n_days": ("gk", "size"), "dollar_volume": ("dollar_volume", "mean"),
    "illiq": ("illiq", "mean"), "in_index": ("in_index", "last"),
}  # fmt: skip


def week_end(dates: pd.Series) -> pd.Series:
    """The Friday that ends each date's week: the key every weekly table is joined on."""
    return dates.dt.to_period("W-FRI").dt.end_time.dt.normalize()


def _runs(ticker: pd.Series, time: pd.Series, max_gap: pd.Timedelta) -> pd.Series:
    """Label runs of consecutive rows of one ticker; a new run starts after a gap > max_gap."""
    return (ticker.ne(ticker.shift()) | time.diff().gt(max_gap)).cumsum()


def daily_measures(d: pd.DataFrame) -> pd.DataFrame:
    """Add range-based daily variances, returns and liquidity to an adjusted stock-day table.

    gk: Garman-Klass (1980); parkinson: Parkinson (1980); rs: Rogers-Satchell (1991).
    overnight: log(open / previous close), which the range estimators ignore. Lags never cross
    a gap of more than `MAX_GAP` (a company that left the index and came back years later).
    """
    d = d.sort_values(["ticker", "date"]).copy()
    prev_close = d.groupby(_runs(d["ticker"], d["date"], MAX_GAP))["close"].shift()
    hl, co = np.log(d["high"] / d["low"]), np.log(d["close"] / d["open"])
    hc, ho = np.log(d["high"] / d["close"]), np.log(d["high"] / d["open"])
    lc, lo = np.log(d["low"] / d["close"]), np.log(d["low"] / d["open"])
    d["gk"] = 0.5 * hl**2 - (2 * LN2 - 1) * co**2
    d["parkinson"] = hl**2 / (4 * LN2)
    d["rs"] = hc * ho + lc * lo
    d["overnight"] = np.log(d["open"] / prev_close)
    d["ret"] = np.log(d["close"] / prev_close)
    d["dollar_volume"] = d["close"] * d["volume"]
    d["illiq"] = d["ret"].abs() / d["dollar_volume"] * 1e9  # Amihud (2002), per $1B traded
    return d


def weekly_panel(d: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a `daily_measures` table to one row per (ticker, week ending Friday)."""
    d = d.assign(week=week_end(d["date"]), overnight2=d["overnight"] ** 2,
                 neg2=d["ret"].clip(upper=0) ** 2)  # fmt: skip
    spec = {k: v for k, v in WEEKLY.items() if v[0] in d}  # in_index is optional
    w = d.groupby(["ticker", "week"]).agg(**spec).reset_index()
    run = _runs(w["ticker"], w["week"], pd.Timedelta(days=7))  # consecutive weeks only
    g = w.groupby(run)
    log_rv = np.log(w["rv"].clip(lower=VAR_FLOOR))
    w["target"] = log_rv.groupby(run).shift(-1).where(g["n_days"].shift(-1).ge(3))
    w["har_d"] = np.log((5 * w["last_gk"]).clip(lower=VAR_FLOOR))
    w["har_w"] = log_rv
    w["har_m"] = np.log(
        g["rv"].transform(lambda s: s.rolling(4, min_periods=3).mean()).clip(lower=VAR_FLOOR)
    )
    for col in ["rv_parkinson", "rv_rs", "rv_overnight", "rv_neg"]:
        w[f"log_{col}"] = np.log(w.pop(col).clip(lower=VAR_FLOOR))
    w["log_dollar_volume"] = np.log(w.pop("dollar_volume"))
    w["ret_4w"] = g["ret"].transform(lambda s: s.rolling(4, min_periods=3).sum())
    return w.drop(columns=["rv", "last_gk"])


def add_market(w: pd.DataFrame, market: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Join market-wide features known at each Friday close.

    market: weekly panel of SPY (from `weekly_panel`); fred: daily FRED columns.
    """
    spy = market.set_index("week")[["har_w", "ret"]].add_prefix("spy_")
    macro = fred.resample("W-FRI").last().ffill()
    return w.join(spy, on="week").join(macro, on="week")


def add_earnings(w: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add earnings timing without look-ahead.

    weeks_since_earnings: weeks since the last announcement filed before the Friday close
    (16:00 US/Eastern); one filed after the close counts for the next week.
    earnings_next_week: 1 if the last announcement plus the median gap of the previous four
    (about 91 days) falls in the target week; it uses only past announcement dates.
    """
    e = events.assign(time=events["accepted"].dt.tz_localize(None)).sort_values(["ticker", "time"])
    gap = e.groupby("ticker")["time"].diff().dt.days
    e["expected_next"] = e["time"].dt.normalize() + pd.to_timedelta(
        gap.groupby(e["ticker"]).transform(lambda s: s.rolling(4, min_periods=1).median()), "D"
    )
    out = pd.merge_asof(
        w.assign(close_time=w["week"] + pd.Timedelta(hours=16)).sort_values("close_time"),
        e[["ticker", "time", "expected_next"]].sort_values("time"),
        left_on="close_time", right_on="time", by="ticker", allow_exact_matches=False,
    )  # fmt: skip
    out["weeks_since_earnings"] = (out.pop("close_time") - out.pop("time")).dt.days / 7
    nxt = out.pop("expected_next")
    in_next = nxt.between(out["week"], out["week"] + pd.Timedelta(days=7), inclusive="right")
    out["earnings_next_week"] = in_next.astype(float).where(nxt.notna())
    return out.sort_values(["ticker", "week"]).reset_index(drop=True)


def alpha158_weekly(daily: pd.DataFrame, chunk: int = 25) -> pd.DataFrame:
    """Alpha158 at each ticker's last trading day of every week, one row per (ticker, week).

    Tickers go `chunk` at a time, ordered by first date so that each wide frame holds few empty
    cells; peak memory stays near 1 GB on the full panel.
    """
    tickers = daily.groupby("ticker")["date"].min().sort_values().index
    out = []
    for i in range(0, len(tickers), chunk):
        part = daily[daily["ticker"].isin(tickers[i : i + chunk])]
        wide = part.pivot(index="date", columns="ticker", values=OHLCV)
        values, names = alpha158({f: wide[f] for f in OHLCV})
        last = part.groupby(["ticker", week_end(part["date"]).rename("week")])["date"].max()
        rows = wide.index.get_indexer(last.to_numpy())
        cols = wide["close"].columns.get_indexer(last.index.get_level_values("ticker"))
        out.append(pd.DataFrame(values[rows, cols], columns=names, index=last.index))
    return pd.concat(out).reset_index()


def build_table(
    daily: pd.DataFrame, spy: pd.DataFrame, fred: pd.DataFrame, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The modeling table: stock-weeks in the index at the Friday close that have a target.

    daily: the clean, adjusted stock-day table with `in_index`; spy: SPY's raw daily bars;
    fred: daily FRED columns; events: SEC earnings times and SIC codes, if fetched.
    """
    market = weekly_panel(daily_measures(adjust_prices(spy)))
    w = add_market(weekly_panel(daily_measures(daily)), market, fred)
    if events is not None:
        w = add_earnings(w, events)
        w = w.join(events.groupby("ticker")["sic"].first().str[:2].rename("sic2"), on="ticker")
    w = w.merge(alpha158_weekly(daily), on=["ticker", "week"], how="left")
    keep = w["in_index"] & w["target"].notna()
    return w[keep].drop(columns="in_index").reset_index(drop=True)


def weekly_log_rv(rv: pd.DataFrame) -> pd.Series:
    """Log weekly sum of daily realized variance by (ticker, week), floored like the target."""
    weekly = rv.groupby(["ticker", week_end(rv["date"]).rename("week")])["rv5"].sum()
    return np.log(weekly.clip(lower=VAR_FLOOR)).rename("log_rv5")
