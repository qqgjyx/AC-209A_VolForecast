import numpy as np
import pandas as pd
import pytest

from volforecast.alpha158 import alpha158
from volforecast.features import (
    OHLCV,
    add_earnings,
    add_market,
    alpha158_weekly,
    build_table,
    daily_measures,
    weekly_panel,
)


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    return weekly_panel(daily_measures(d.assign(in_index=True)))


def test_range_estimators_on_a_known_bar():
    bar = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]), "ticker": "A", "open": 100.0,
        "high": 100 * np.exp(0.05), "low": 100 * np.exp(-0.05), "close": 100.0,
        "adj_close": 100.0, "volume": 1e6,
    })  # fmt: skip
    m = daily_measures(bar).iloc[0]
    assert m["gk"] == pytest.approx(0.5 * 0.1**2)
    assert m["parkinson"] == pytest.approx(0.1**2 / (4 * np.log(2)))
    assert m["rs"] == pytest.approx(2 * 0.05**2)


def test_target_is_next_weeks_log_variance(daily):
    a = weekly(daily[daily["ticker"] == "AAA"])
    full_next = a["n_days"].shift(-1).ge(3)  # a target needs at least 3 days next week
    assert a["target"][full_next].to_numpy() == pytest.approx(a["har_w"].shift(-1)[full_next])
    assert a["target"][~full_next].isna().all()  # the last week, and one before a 2-day week


def test_no_target_across_a_missing_week(daily):
    """The target is next calendar week's variance, never the next observed week's."""
    hole = daily["date"].between("2020-03-09", "2020-03-13")  # drop one whole week
    a = weekly(daily[(daily["ticker"] == "AAA") & ~hole]).set_index("week")
    assert np.isnan(a.loc[pd.Timestamp("2020-03-06"), "target"])
    assert np.isnan(a.loc[pd.Timestamp("2020-03-20"), "har_m"])  # the 4-week mean restarts


def test_lags_restart_after_a_long_gap(daily):
    """A company that leaves and re-enters must not compare prices across the gap."""
    a = daily[(daily["ticker"] == "AAA") & ~daily["date"].between("2020-04-01", "2020-06-30")]
    m = daily_measures(a.assign(in_index=True)).set_index("date")
    first_back = m.loc["2020-07-01":].index[0]
    assert np.isnan(m.loc[first_back, "overnight"]) and np.isnan(m.loc[first_back, "ret"])


def test_features_ignore_later_data(daily):
    """Appending later weeks must not change any feature of an earlier week."""
    cutoff = pd.Timestamp("2020-06-26")  # a Friday, so no week is cut in half
    full = weekly(daily).set_index(["ticker", "week"])
    part = weekly(daily[daily["date"] <= cutoff]).set_index(["ticker", "week"])
    cols = full.columns.drop("target")
    pd.testing.assert_frame_equal(full.loc[part.index, cols], part[cols])


def test_earnings_after_the_close_count_for_next_week(daily):
    w = weekly(daily[daily["ticker"] == "AAA"])
    friday = pd.Timestamp("2020-03-13")
    events = pd.DataFrame({
        "ticker": "AAA", "sic": "1000",
        "accepted": pd.to_datetime(["2020-01-10 08:00", "2020-03-13 16:30"]).tz_localize(
            "America/New_York"
        ),
    })  # fmt: skip
    out = add_earnings(w, events).set_index("week")
    assert out.loc[friday, "weeks_since_earnings"] > 8  # still the January release
    assert out.loc[friday + pd.Timedelta(days=7), "weeks_since_earnings"] < 1


def test_market_features_join_on_the_friday(daily):
    w = weekly(daily)
    spy = weekly(daily[daily["ticker"] == "AAA"])
    days = pd.bdate_range("2020-01-01", periods=260)
    fred = pd.DataFrame({"VIXCLS": np.arange(len(days), dtype=float)}, index=days)
    out = add_market(w, spy, fred).set_index(["ticker", "week"])
    friday = pd.Timestamp("2020-03-13")
    assert out.loc[("BBB", friday), "VIXCLS"] == days.get_loc(friday)  # Friday's own value
    assert out.loc[("BBB", friday), "spy_har_w"] == spy.set_index("week").loc[friday, "har_w"]


def test_alpha158_weekly_keeps_each_weeks_last_day(daily):
    w = alpha158_weekly(daily, chunk=3)  # a chunk smaller than the universe
    wide = daily.pivot(index="date", columns="ticker", values=OHLCV)
    full, names = alpha158({f: wide[f] for f in OHLCV})
    friday = pd.Timestamp("2020-06-26")
    row = w[(w["ticker"] == "CCC") & (w["week"] == friday)]
    t, n = wide.index.get_loc(friday), wide["close"].columns.get_loc("CCC")
    np.testing.assert_allclose(row[names].to_numpy()[0], full[t, n], equal_nan=True)
    assert len(w) == daily.groupby(["ticker", daily["date"].dt.to_period("W-FRI")]).ngroups


def test_build_table_keeps_in_index_rows_with_a_target(daily):
    d = daily.drop(columns="adj_close").assign(in_index=daily["ticker"] != "DDD")
    spy = daily[daily["ticker"] == "AAA"].assign(ticker="SPY")
    fred = pd.DataFrame({"VIXCLS": 20.0}, index=pd.bdate_range("2020-01-01", periods=260))
    table = build_table(d, spy, fred)
    assert "DDD" not in set(table["ticker"]) and table["target"].notna().all()
    assert {"har_w", "spy_har_w", "VIXCLS", "KMID"} <= set(table.columns)
