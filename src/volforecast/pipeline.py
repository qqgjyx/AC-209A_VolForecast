"""From the caches to forecasts: the tables Milestone 3 starts from, and the walk-forward loop.

`load_tables` reruns ms2.ipynb's steps 1a to 3e without the displays (about 20 seconds from the
caches) and adds the quarterly and yearly HAR lags. `walk_forward` is the one loop every model
runs through: refit on all earlier rows at the start of each period, predict that period.
"""

import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone

from volforecast.clean import clean
from volforecast.data import (
    BREAKS,
    WRONG_SERIES,
    fetch_alpaca_daily,
    fetch_fred,
    fetch_membership,
    fetch_prices,
    fetch_sec_events,
    to_current,
    universe,
)
from volforecast.diagnostics import SPLITS
from volforecast.features import build_table, daily_measures, long_memory

FRED_SERIES = ["VIXCLS", "VXVCLS", "BAA10Y", "T10Y2Y", "DTB3"]
NEXT_DAY = ["BAA10Y", "T10Y2Y", "DTB3"]  # FRED posts a day's H.15 rates the next business day
EMBARGO = pd.Timedelta(days=7)  # a row's target is the week after it, so it is known 7 days on


@dataclass
class Tables:
    """What every Milestone 3 step reads."""

    daily: pd.DataFrame  # clean stock-day table with daily measures (returns) and `in_index`
    table: pd.DataFrame  # one row per stock-week: target and every feature, incl. har_q, har_y


def load_tables(data_path: Path, start: str, end: str) -> Tables:
    """Rebuild the clean daily table and the stock-week table from the caches under `data_path`.

    Same calls as ms2.ipynb, so the table matches its `out/weekly.parquet` plus two columns,
    except that the interest rates in `NEXT_DAY` are lagged one trading day: a Friday row has
    Thursday's rates, the last ones published by Friday's close (VIX closes at 4:15 pm, before
    the next week opens). Alpaca and SEC are read only if cached or if their keys are set, as in
    ms2.ipynb.
    """
    membership = to_current(fetch_membership(data_path / "sp500_membership.csv"))
    tickers = universe(membership, start)
    prices = fetch_prices(tickers, start, end, data_path / "prices").assign(source="yahoo")
    missing = sorted(set(tickers) - set(prices.index.unique("ticker")))
    alpaca_keys = all(os.environ.get(k) for k in ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"])
    if (data_path / "alpaca_daily").exists() or alpaca_keys:
        alpaca = fetch_alpaca_daily(missing, end, data_path / "alpaca_daily")
        prices = pd.concat([prices, alpaca.assign(source="alpaca")]).sort_index()
    events = None
    if os.environ.get("SEC_USER_AGENT") or (data_path / "sec" / "company_tickers.json").exists():
        last_day = prices.reset_index().groupby("ticker")["date"].max()
        listed = sorted(last_day.index[last_day >= last_day.max() - pd.Timedelta(days=7)])
        events = fetch_sec_events(listed, data_path / "sec")
    breaks = {t: d for t, (d, _) in BREAKS.items()}
    daily, _ = clean(prices.reset_index(), membership, set(WRONG_SERIES), breaks, end)
    spy = fetch_prices(["SPY"], start, end, data_path / "market").reset_index()
    fred = fetch_fred(FRED_SERIES, data_path / "fred").loc[start:]
    table = build_table(daily, spy, fred, events, next_day=NEXT_DAY)
    table = table.merge(long_memory(daily), on=["ticker", "week"], how="left")
    return Tables(daily=daily_measures(daily), table=table)


def split_of(week: pd.Series) -> pd.Series:
    """'train', 'tune' or 'test' for each row: the split of its week (`diagnostics.SPLITS`).

    A row whose target week ends in the test years is a test row: the last Friday of 2018
    forecasts the first week of 2019, so validation stops a week early and no validation
    number reads a 2019 price.
    """
    out = pd.Series(pd.NA, index=week.index, dtype="string")
    for name, (first, last) in SPLITS.items():
        out[week.between(pd.Timestamp(first), pd.Timestamp(last))] = name
    out[week + EMBARGO >= pd.Timestamp(SPLITS["test"][0])] = "test"
    return out


def fit_rows(
    table: pd.DataFrame, start: pd.Timestamp, window_years: int | None = None
) -> pd.DataFrame:
    """The rows a model refit at `start` may use: those whose target week ended before it.

    That is week + 7 days < start (the embargo), all of them (expanding) or the last
    `window_years` (rolling).
    """
    rows = table[table["week"] + EMBARGO < start]
    if window_years is not None:
        rows = rows[rows["week"] >= start - pd.DateOffset(years=window_years)]
    return rows


def walk_forward(
    model: RegressorMixin,
    table: pd.DataFrame,
    first: str,
    last: str,
    freq: str = "YS",
    window_years: int | None = None,
) -> pd.DataFrame:
    """Refit a copy of `model` at the start of each period and forecast that period's rows.

    A period is [start, next start), with starts at `first` and every `freq` after it; rows
    with week in [`first`, `last`] are forecast. A model is fit on the rows whose target week
    ends before the period starts (week + 7 days < start), all of them (expanding) or the last
    `window_years`.
    Returns, per forecast row, the log forecast, the variance forecast exp(log) times Duan's
    (1983) smearing factor mean(exp(residual)) from the fit's own training rows, and the refit date.
    """
    starts = pd.DatetimeIndex([first]).union(pd.date_range(first, last, freq=freq))
    edges = [*starts, pd.Timestamp(last) + pd.Timedelta(days=1)]
    out = []
    for start, stop in pairwise(edges):
        rows = table[(table["week"] >= start) & (table["week"] < stop)]
        if rows.empty:
            continue
        train = fit_rows(table, start, window_years)
        fitted = clone(model).fit(train, train["target"])
        smear = float(np.mean(np.exp(train["target"] - fitted.predict(train))))
        log_pred = fitted.predict(rows)
        out.append(pd.DataFrame(
            {"log_pred": log_pred, "var_pred": np.exp(log_pred) * smear, "refit": start},
            index=rows.index,
        ))  # fmt: skip
    return pd.concat(out)
