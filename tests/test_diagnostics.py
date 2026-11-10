import numpy as np
import pandas as pd
import pytest

from volforecast import diagnostics as dg
from volforecast.features import add_market, daily_measures


def _measures(n: int = 300, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    """Hand-built daily measures: every day has ret^2 = 4, overnight^2 = 1 and gk = 2."""
    dates = pd.bdate_range("2010-01-04", periods=n)
    frames = [
        pd.DataFrame(
            {"date": dates, "ticker": t, "ret": 2.0, "overnight": -1.0, "gk": 2.0, "in_index": True}
        )
        for t in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def test_shares_are_ratios_of_sums_known_by_hand():
    """Overnight 1/4, Garman-Klass 2/4, both together 3/4 of close-to-close variance."""
    shares = dg.variance_shares(_measures(), "2010-01-01", "2018-12-31")
    assert shares["overnight"].tolist() == pytest.approx([0.25, 0.25])
    assert shares["gk"].tolist() == pytest.approx([0.5, 0.5])
    assert shares["gk_overnight"].tolist() == pytest.approx([0.75, 0.75])
    assert shares["days"].tolist() == [299, 299]  # the first day has no previous session


def test_rows_outside_the_window_or_the_index_spell_do_not_count():
    """Changing rows after `end` or outside the spell leaves every share unchanged."""
    m = _measures(n=600)
    before = dg.variance_shares(m, "2010-01-01", "2010-12-31")
    later = m["date"] > "2010-12-31"
    out_of_spell = (m["ticker"] == "AAA") & (m["date"] < "2010-03-01")
    changed = m.copy()
    changed.loc[later, ["ret", "overnight", "gk"]] = 99.0
    changed.loc[out_of_spell, "in_index"] = False
    changed.loc[out_of_spell, ["ret", "overnight", "gk"]] = 99.0
    after = dg.variance_shares(changed, "2010-01-01", "2010-12-31", min_days=1)
    assert after["overnight"].tolist() == pytest.approx(before["overnight"].tolist())


def test_a_day_after_a_missing_session_is_left_out():
    """When a session is missing for one ticker, its next row spans two sessions."""
    m = _measures(n=10)
    hole = m["date"] == m["date"].iloc[4]
    m = m[~(hole & (m["ticker"] == "AAA"))]
    follows = dg.follows_previous_session(m.sort_values(["ticker", "date"]))
    aaa = m.assign(f=follows)[m["ticker"] == "AAA"]
    assert aaa["f"].tolist() == [False, True, True, True, False, True, True, True, True]


def test_companies_with_too_few_days_are_dropped():
    m = _measures(n=100)
    assert dg.variance_shares(m, "2010-01-01", "2018-12-31").empty
    assert len(dg.variance_shares(m, "2010-01-01", "2018-12-31", min_days=50)) == 2


def test_shares_on_the_synthetic_panel_match_a_direct_computation(daily):
    """On cleaned-like bars, the GK share equals sum(gk) / sum(ret^2) computed by hand."""
    m = daily_measures(daily.assign(in_index=True))
    shares = dg.variance_shares(m, "2020-01-01", "2020-12-31", min_days=1)
    a = m[(m["ticker"] == "AAA") & m["ret"].notna()]
    a = a[a["date"].between("2020-01-01", "2020-12-31")]
    assert shares.loc["AAA", "gk"] == pytest.approx(a["gk"].sum() / (a["ret"] ** 2).sum())


def test_weekly_vix_matches_the_tables_rule_and_uses_thursday_on_a_holiday_friday():
    days = pd.bdate_range("2020-04-06", "2020-04-17")
    vix = pd.Series(np.arange(len(days), dtype=float), index=days)
    vix = vix.drop(pd.Timestamp("2020-04-10"))  # Good Friday: no close
    weeks = dg.weekly_vix(vix)
    assert weeks.loc["2020-04-10"] == vix.loc["2020-04-09"]
    w = pd.DataFrame({"ticker": "AAA", "week": weeks.index, "har_w": 0.0, "ret": 0.0})
    table = add_market(w, w.assign(ticker="SPY"), vix.to_frame("VIXCLS"))
    assert table["VIXCLS"].tolist() == weeks.tolist()


def test_stress_counts_per_split_and_episodes():
    """Two stress runs in test (3 weeks, then 1), one week in tune, none in train."""
    weeks = pd.date_range("2005-01-07", "2026-10-30", freq="W-FRI")
    vix = pd.Series(15.0, index=weeks)
    vix.loc["2018-02-09"] = 33.0
    vix.loc["2020-03-13":"2020-03-27"] = 60.0
    vix.loc["2022-03-04"] = 31.0
    table = dg.stress_by_split(vix, 30)
    assert table["stress_weeks"].tolist() == [0, 1, 4]
    assert table["episodes"].tolist() == [0, 1, 2]
    assert table["weeks"].sum() == len(weeks)
    by_year = dg.stress_weeks_by_year(vix, [30, 25])
    assert by_year.loc[2020, "VIX >= 30"] == 3
    assert by_year["VIX >= 25"].sum() == 5


def test_episodes_counts_runs():
    assert dg.episodes(pd.Series([False, True, True, False, True])) == 2
    assert dg.episodes(pd.Series([True])) == 1
    assert dg.episodes(pd.Series([], dtype=bool)) == 0
