import pandas as pd
import pytest

from volforecast.clean import (
    adjust_prices,
    bad_ohlc,
    clean,
    coverage,
    no_trade,
    restrict_to_spells,
    spells,
)

END = "2021-01-01"


def test_spells_split_on_exit_and_end_exclusive(membership):
    s = spells(membership, END).set_index("entity")
    assert s.loc["CCC", "start"] == pd.Timestamp("2020-01-01")
    assert s.loc["CCC", "end"] == pd.Timestamp("2020-06-01")  # first snapshot without it
    assert s.loc["AAA", "end"] == pd.Timestamp(END)


def test_rename_continues_the_same_spell():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-03-02", "2020-06-01"]),
        "ticker": ["OLD", "OLD", "NEW"],
    })  # fmt: skip
    s = spells(m.replace({"ticker": {"OLD": "NEW"}}), END)  # as data.to_current does
    assert len(s) == 1
    assert s.iloc[0].tolist() == ["NEW", pd.Timestamp("2020-01-01"), pd.Timestamp(END)]


def test_restrict_drops_rows_far_from_any_spell(daily, membership):
    kept = restrict_to_spells(daily, spells(membership, END))
    assert "DDD" not in set(kept["ticker"])  # never a member: a reused ticker looks like this
    ccc = kept[kept["ticker"] == "CCC"]
    assert ccc["date"].max() < pd.Timestamp("2020-06-15")  # spell end + 14-day lookahead
    assert not ccc.loc[ccc["date"] >= "2020-06-01", "in_index"].any()


def test_rules_flag_the_rows_they_name(daily):
    d = daily.copy()
    d.loc[0, "low"] = d.loc[0, ["open", "close"]].min() * 1.01  # low above the body
    d.loc[1, "volume"] = 0
    assert bad_ohlc(d).tolist()[:2] == [True, False]
    assert no_trade(d).sum() == 1


def test_adjust_prices_keeps_ranges_and_dollar_volume():
    d = pd.DataFrame({"open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0],
                      "adj_close": [5.5], "volume": [100.0]})  # fmt: skip
    a = adjust_prices(d).iloc[0]
    assert a["close"] == 5.5 and "adj_close" not in a
    assert a["high"] / a["low"] == pytest.approx(12 / 9)
    assert a["close"] * a["volume"] == pytest.approx(11 * 100)


def test_clean_log_adds_up_and_drops_wrong_series_and_breaks(daily, membership):
    out, steps = clean(daily, membership, {"BBB"}, {"AAA": "2020-03-02"}, END)
    assert "BBB" not in set(out["ticker"])
    assert out.loc[out["ticker"] == "AAA", "date"].min() == pd.Timestamp("2020-03-02")
    assert steps["rows_left"].iloc[-1] == len(out)
    assert steps["rows_removed"].sum() == len(daily) - len(out)


def test_coverage_counts_members_with_rows(daily, membership):
    cov = coverage(membership, daily[daily["ticker"] != "BBB"])
    assert cov.iloc[0] == 2 / 3  # AAA and CCC of {AAA, BBB, CCC}
    assert cov.iloc[-1] == 1 / 2  # AAA of {AAA, BBB}
