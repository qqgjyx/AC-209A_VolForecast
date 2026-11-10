"""Two checks behind the target and the stress label, run on the cleaned daily table.

1. How much of a stock's close-to-close variance the daily range (Garman-Klass) sees. The range
   covers the open to the close only, so the move from the previous close to the open (the
   overnight move, where after-close earnings news lands) is outside it.
2. How many stress weeks fall in each split. The test and tuning periods hold far fewer stress
   weeks than the stock-week count suggests, and they come in a few episodes.
"""

import numpy as np
import pandas as pd

# The project's splits (see the revised question): train, tune, then test.
SPLITS = {
    "train": ("2005-01-01", "2015-12-31"),
    "tune": ("2016-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2026-12-31"),
}
MIN_DAYS = 250  # about a year of usable days before a company's share counts


def follows_previous_session(d: pd.DataFrame) -> pd.Series:
    """True where a row's previous row (same ticker) is the previous trading day in the table.

    A row removed in cleaning, or a day with no bar, leaves a hole. The next row's overnight
    and close-to-close returns then span two sessions, so they are left out.
    """
    dates = np.sort(d["date"].unique())
    previous_session = pd.Series(dates[:-1], index=dates[1:])
    previous_row = d.groupby("ticker")["date"].shift()
    return previous_row.eq(d["date"].map(previous_session))


def variance_shares(
    measures: pd.DataFrame, start: str, end: str, min_days: int = MIN_DAYS
) -> pd.DataFrame:
    """Per company, shares of close-to-close variance: overnight, Garman-Klass, and both together.

    measures: `features.daily_measures` output with the `in_index` flag from cleaning, sorted by
    ticker and date. Uses days inside the company's index spell, between `start` and `end`
    (inclusive), that directly follow the previous session. Each share is a ratio of sums over
    those days, so a share of 1.0 means the measure adds up to the close-to-close variance, and
    high-variance days (2008-09, for example) weigh more than calm ones. Shares can pass 1.0: the
    range estimator is noisy, and overnight and daytime moves are not independent.
    """
    d = measures.sort_values(["ticker", "date"])
    keep = (
        d["in_index"].astype(bool)
        & d["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & follows_previous_session(d)
        & d[["ret", "overnight", "gk"]].notna().all(axis=1)
    )
    k = d.loc[keep]
    sums = (
        k.assign(cc=k["ret"] ** 2, on=k["overnight"] ** 2)
        .groupby("ticker")
        .agg(days=("cc", "size"), cc=("cc", "sum"), on=("on", "sum"), gk=("gk", "sum"))
    )
    sums = sums[(sums["days"] >= min_days) & (sums["cc"] > 0)]
    return pd.DataFrame(
        {
            "days": sums["days"],
            "overnight": sums["on"] / sums["cc"],
            "gk": sums["gk"] / sums["cc"],
            "gk_overnight": (sums["gk"] + sums["on"]) / sums["cc"],
        }
    )


def weekly_vix(vix: pd.Series) -> pd.Series:
    """VIX at the end of each week ending Friday: the last close in the week, carried forward.

    The same rule `features.add_market` uses, so a holiday Friday takes Thursday's close.
    """
    return vix.resample("W-FRI").last().ffill()


def stress_weeks_by_year(vix_weeks: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    """Number of weeks per calendar year with weekly VIX at or above each threshold."""
    return pd.DataFrame(
        {f"VIX >= {t:g}": vix_weeks.ge(t).groupby(vix_weeks.index.year).sum() for t in thresholds}
    ).rename_axis("year")


def episodes(stress: pd.Series) -> int:
    """Number of runs of consecutive stress weeks (a calm week ends a run)."""
    s = stress.astype(bool).to_numpy()
    return int(s[0] + np.sum(s[1:] & ~s[:-1])) if len(s) else 0


def stress_by_split(vix_weeks: pd.Series, threshold: float) -> pd.DataFrame:
    """Per split: weeks, stress weeks, and the episodes they come in."""
    rows = {}
    for name, (start, end) in SPLITS.items():
        v = vix_weeks.loc[start:end]
        stress = v.ge(threshold)
        rows[name] = {
            "weeks": len(v),
            "stress_weeks": int(stress.sum()),
            "episodes": episodes(stress),
        }
    return pd.DataFrame(rows).T
