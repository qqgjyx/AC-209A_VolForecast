"""Turn raw Yahoo rows into a clean stock-day table, logging what each step removes.

Every rule looks at one row, or at a static table built once by hand, never at later rows, so
cleaning cannot leak the future into a feature. The price adjustment follows qlib's Yahoo
normalizer.
"""

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 120  # rows kept before an index spell, for lagged features (calendar days)
LOOKAHEAD_DAYS = 14  # rows kept after a spell, so the last in-index week has a target


def spells(membership: pd.DataFrame, end: str) -> pd.DataFrame:
    """Index spells per company: (entity, start, end), `end` exclusive.

    A spell is a run of consecutive snapshots containing the ticker. Pass membership through
    `data.to_current` first, so that a rename does not break a spell.
    """
    snaps = np.append(np.sort(membership["date"].unique()), np.datetime64(pd.Timestamp(end)))
    m = membership.assign(k=np.searchsorted(snaps, membership["date"].to_numpy()))
    m = m.drop_duplicates(["ticker", "k"]).sort_values(["ticker", "k"])
    run = (m["ticker"].ne(m["ticker"].shift()) | m["k"].diff().ne(1)).cumsum()
    s = m.groupby(run).agg(entity=("ticker", "first"), a=("k", "first"), b=("k", "last"))
    return pd.DataFrame(
        {"entity": s["entity"].to_numpy(), "start": snaps[s["a"]], "end": snaps[s["b"] + 1]}
    )


def restrict_to_spells(daily: pd.DataFrame, spell_table: pd.DataFrame) -> pd.DataFrame:
    """Keep rows near an index spell of their ticker and flag the rows inside one.

    Yahoo reuses tickers, so a symbol's rows outside its spells may belong to another company.
    """
    pairs = (
        daily[["date", "ticker"]]
        .reset_index()
        .merge(spell_table, left_on="ticker", right_on="entity")
    )
    near = (pairs["date"] >= pairs["start"] - pd.Timedelta(days=LOOKBACK_DAYS)) & (
        pairs["date"] < pairs["end"] + pd.Timedelta(days=LOOKAHEAD_DAYS)
    )
    inside = (pairs["date"] >= pairs["start"]) & (pairs["date"] < pairs["end"])
    flags = pairs.assign(near=near, inside=inside).groupby("index")[["near", "inside"]].any()
    keep = flags.index[flags["near"]]
    return daily.loc[keep].assign(in_index=flags.loc[keep, "inside"].to_numpy())


def bad_ohlc(d: pd.DataFrame) -> pd.Series:
    """Impossible bars: a non-positive price or open/close outside [low, high]."""
    prices = d[["open", "high", "low", "close"]]
    return (
        (prices <= 0).any(axis=1)
        | prices.isna().any(axis=1)
        | (d["high"] < d[["open", "close"]].max(axis=1))
        | (d["low"] > d[["open", "close"]].min(axis=1))
    )


def no_trade(d: pd.DataFrame) -> pd.Series:
    """Days with zero volume: Yahoo carries the last price forward, so the bar is not real."""
    return d["volume"].eq(0)


def adjust_prices(d: pd.DataFrame) -> pd.DataFrame:
    """Apply qlib's Yahoo normalizer: prices times adj_close / close, volume divided by it.

    Yahoo's OHLC is adjusted for splits only; the factor adds dividends and spin-offs (Scripps
    in 2008, Keurig Dr Pepper's special dividend in 2018), so a price ratio across days is a
    total-return ratio, and price times volume stays the dollar volume traded. Within a day the
    factor cancels, so range estimators do not change. `adj_close` is dropped as redundant.
    """
    factor = d["adj_close"] / d["close"]
    prices = {c: d[c] * factor for c in ["open", "high", "low", "close"]}
    return d.assign(**prices, volume=d["volume"] / factor).drop(columns="adj_close")


def clean(
    daily: pd.DataFrame,
    membership: pd.DataFrame,
    wrong_series: set[str],
    breaks: dict[str, str],
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply every cleaning step in order; return the clean table and a log of the steps.

    membership: in current symbols (`data.to_current`), like the tickers of `daily`.
    wrong_series: tickers whose series is another security; breaks: ticker -> first good date.
    """
    d = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    log = [("raw rows", "", len(d))]
    d = restrict_to_spells(d, spells(membership, end))
    log.append(("index spells", "outside each ticker's index spells, incl. reused tickers", len(d)))
    rules = [
        ("wrong series", lambda x: x["ticker"].isin(wrong_series), "another security, same ticker"),
        (
            "series breaks",
            lambda x: x["date"] < pd.to_datetime(x["ticker"].map(breaks)),
            "before a change of company or price scale",
        ),
        ("impossible bars", bad_ohlc, "price <= 0 or open/close outside [low, high]"),
        ("no-trade days", no_trade, "zero volume, price carried forward"),
    ]
    for name, rule, reason in rules:
        d = d[~rule(d)].reset_index(drop=True)
        log.append((name, reason, len(d)))
    steps = pd.DataFrame(log, columns=["step", "removes", "rows_left"])
    steps["rows_removed"] = (-steps["rows_left"].diff()).fillna(0).astype(int)
    return adjust_prices(d), steps


def coverage(membership: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Monthly share of index members with at least one price row that month.

    rows: a (date, ticker) table in the same symbols as `membership`.
    """
    months = pd.period_range(rows["date"].min(), rows["date"].max(), freq="M")
    snaps = np.sort(membership["date"].unique())
    in_force = snaps[np.searchsorted(snaps, months.end_time.to_numpy(), side="right") - 1]
    members = pd.DataFrame({"month": months, "date": in_force}).merge(
        membership.drop_duplicates(["date", "ticker"]), on="date"
    )
    have = pd.MultiIndex.from_arrays([rows["date"].dt.to_period("M"), rows["ticker"]]).unique()
    members["has"] = pd.MultiIndex.from_frame(members[["month", "ticker"]]).isin(have)
    return members.groupby("month")["has"].mean().rename("coverage").to_timestamp()
