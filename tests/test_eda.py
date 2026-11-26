import numpy as np
import pandas as pd
import pytest

from volforecast import eda
from volforecast.pipeline import split_of


def test_variance_decomposition_of_an_additive_panel():
    weeks = np.tile(np.arange(10), 5)
    stocks = np.repeat(np.arange(5), 10)
    y = pd.Series(stocks * 1.0 + weeks * 0.5)
    shares = eda.variance_decomposition(y, pd.Series(stocks), pd.Series(weeks))
    assert shares["stock + week"] == pytest.approx(1.0)
    assert shares["stock"] + shares["week"] == pytest.approx(1.0)  # balanced: the shares add up


def test_tails_flags_heavy_tails():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"normal": rng.normal(size=5000), "heavy": rng.standard_t(2, 5000)})
    out = eda.tails(f, ["normal", "heavy"])
    assert out.loc["heavy", "transform"] == "normal scores"
    assert out.loc["normal", "transform"] == "standardize"


def test_split_summary_counts_own_history_at_each_yearly_refit():
    weeks = pd.date_range("2014-01-03", "2017-06-30", freq="W-FRI")
    t = pd.DataFrame({"ticker": "AAA", "week": weeks})
    new = pd.DataFrame({"ticker": "NEW", "week": weeks[weeks >= "2016-01-01"]})
    t = pd.concat([t, new], ignore_index=True).assign(split=lambda d: split_of(d["week"]))
    out = eda.split_summary(t, pd.Series(False, index=t.index))
    # 2016: AAA has two years, NEW none; 2017: both have a year (53 + 26 + 26 of 158 rows)
    assert out.loc["tune", "own HAR fit"] == pytest.approx(105 / 158)
