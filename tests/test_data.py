import numpy as np
import pandas as pd
import pytest

from volforecast.data import (
    RENAMES,
    WRONG_SERIES,
    earnings_times,
    from_crsp,
    pack_snapshot,
    parse_filings,
    parse_membership,
    realized_variance,
    rv_sample,
    to_current,
    universe,
    unpack_snapshot,
)

RAW = pd.DataFrame(
    {"date": ["2004-12-01", "2005-06-01", "2006-01-01"], "tickers": ["A,B", "A,C", "C,D.E"]}
)


def test_parse_membership_one_row_per_ticker():
    m = parse_membership(RAW)
    assert len(m) == 6
    assert m["date"].dtype.kind == "M"


def test_universe_starts_from_snapshot_in_force():
    assert universe(parse_membership(RAW), "2005-01-03") == ["A", "B", "C", "D.E"]
    assert universe(parse_membership(RAW), "2005-07-01") == ["A", "C", "D.E"]


def test_to_current_maps_renamed_tickers_only():
    m = pd.DataFrame({"date": pd.Timestamp("2020-01-01"), "ticker": ["BK", "AAPL"]})
    assert to_current(m)["ticker"].tolist() == ["BNY", "AAPL"]


def test_rv_sample_is_fixed_and_drawn_from_the_last_snapshot():
    m = parse_membership(RAW)
    assert rv_sample(m, n=2) == rv_sample(m, n=2)
    assert set(rv_sample(m, n=2)) == {"C", "D.E"}


def test_renames_end_at_the_current_symbol():
    """A chain must point straight to its end, or one lookup would stop half way."""
    assert not set(RENAMES.values()) & set(RENAMES)


def test_earnings_times_keep_only_item_202_8ks_in_eastern_time():
    block = {
        "form": ["8-K", "8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "5.02", "", "2.02"],
        "acceptanceDateTime": [
            "2024-01-25T21:30:00.000Z", "2024-02-01T12:00:00.000Z",
            "2024-04-30T20:00:00.000Z", "2024-04-25T11:00:00.000Z",
        ],
    }  # fmt: skip
    times = earnings_times(parse_filings(block))
    assert len(times) == 2
    assert str(times.iloc[0].tz) == "America/New_York"
    assert times.iloc[0].hour == 16  # 21:30 UTC is 16:30 in New York, after the close


def test_renamed_and_wrong_series_do_not_overlap():
    assert not set(RENAMES) & set(WRONG_SERIES)


def test_realized_variance_sums_squared_regular_session_returns():
    times = pd.to_datetime(["2024-01-02 09:00", "2024-01-02 09:30", "2024-01-02 09:35",
                            "2024-01-02 16:00"]).tz_localize("America/New_York")  # fmt: skip
    bars = pd.DataFrame({"time": times, "date": times.tz_localize(None).normalize(), "ticker": "A",
                         "o": [1.0, 10.0, 11.0, 50.0], "c": [2.0, 11.0, 12.1, 60.0]})  # fmt: skip
    rv = realized_variance(bars).iloc[0]  # 09:00 and 16:00 bars are outside the session
    assert rv["n_bars"] == 2
    assert rv["rv5"] == pytest.approx(np.log(1.1) ** 2 + np.log(1.1) ** 2)


def test_from_crsp_snapshots_follow_the_spells():
    spells = pd.DataFrame({"permno": [1, 2], "start": pd.to_datetime(["2005-01-03", "2005-02-01"]),
                           "end": pd.to_datetime(["2005-03-31", "2005-12-31"])})  # fmt: skip
    rows = pd.DataFrame({"permno": [1, 1], "date": pd.to_datetime(["2005-01-03", "2005-01-04"]),
                         "open": 10.0, "high": 11.0, "low": 9.0, "close": [10.0, None],
                         "volume": 100.0, "cfacpr": 2.0})  # fmt: skip
    m, d = from_crsp(spells, rows)
    members = m.groupby("date")["ticker"].apply(sorted).to_dict()
    assert members == {
        pd.Timestamp("2005-01-03"): ["1"],
        pd.Timestamp("2005-02-01"): ["1", "2"],
        pd.Timestamp("2005-04-01"): ["2"],
    }  # end is inclusive; the empty snapshot after the last spell has no rows
    assert d["ticker"].tolist() == ["1"] and d["adj_close"].tolist() == [5.0]


def test_snapshot_round_trip_skips_crsp(tmp_path):
    src = tmp_path / "data"
    (src / "prices").mkdir(parents=True)
    (src / "prices" / "chunk_000.parquet").write_bytes(b"x")
    (src / "crsp").mkdir()
    (src / "crsp" / "daily.parquet").write_bytes(b"licensed")
    (src / "README.md").write_text("tracked")
    pack_snapshot(src, tmp_path / "s.tar.gz")
    dst = tmp_path / "fresh"
    unpack_snapshot(tmp_path / "s.tar.gz", dst)
    assert [p.relative_to(dst).as_posix() for p in sorted(dst.rglob("*"))] == [
        "prices",
        "prices/chunk_000.parquet",
    ]
