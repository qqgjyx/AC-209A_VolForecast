import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from volforecast.pipeline import split_of, walk_forward


class Recorder(BaseEstimator, RegressorMixin):
    """Predicts the training mean and remembers the last training week it saw."""

    def fit(self, X, y):
        self.last_week_, self.mean_ = X["week"].max(), float(np.mean(y))
        LOG.append((X["week"].min(), self.last_week_))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


LOG: list[tuple[pd.Timestamp, pd.Timestamp]] = []


def panel(years=range(2010, 2014)) -> pd.DataFrame:
    weeks = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="W-FRI")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ticker": "AAA", "week": weeks, "target": rng.normal(-6, 1, len(weeks))})


def test_no_training_target_reaches_into_the_forecast_period():
    LOG.clear()
    t = panel()
    out = walk_forward(Recorder(), t, "2012-01-01", "2013-12-31")
    assert len(LOG) == 2 and out["refit"].nunique() == 2
    for (_, last), start in zip(LOG, ["2012-01-01", "2013-01-01"], strict=True):
        assert last + pd.Timedelta(days=7) < pd.Timestamp(start)  # its target week ended before
    assert out.index.equals(t.index[t["week"] >= "2012-01-01"])


def test_rolling_window_and_quarterly_refits():
    LOG.clear()
    walk_forward(Recorder(), panel(), "2013-01-01", "2013-12-31", window_years=1)
    assert LOG[0][0] >= pd.Timestamp("2012-01-01")
    LOG.clear()
    out = walk_forward(Recorder(), panel(), "2013-01-01", "2013-12-31", freq="QS")
    assert len(LOG) == 4 and out["refit"].nunique() == 4


def test_variance_forecast_carries_the_smearing_factor():
    t = panel()
    out = walk_forward(Recorder(), t, "2013-01-01", "2013-12-31")
    fit = t[t["week"] + pd.Timedelta(days=7) < pd.Timestamp("2013-01-01")]["target"]
    smear = np.mean(np.exp(fit - fit.mean()))
    assert out["var_pred"].iloc[0] == pytest.approx(np.exp(fit.mean()) * smear)
    assert smear > 1  # Jensen: the mean of exp exceeds exp of the mean


def test_split_of_labels_every_week():
    weeks = pd.Series(pd.to_datetime(["2005-01-07", "2016-01-08", "2026-10-30"]))
    assert split_of(weeks).tolist() == ["train", "tune", "test"]


def test_split_of_gives_the_last_2018_friday_to_the_test_years():
    weeks = pd.Series(pd.to_datetime(["2015-12-25", "2018-12-21", "2018-12-28"]))
    assert split_of(weeks).tolist() == ["train", "tune", "test"]  # 2018-12-28 forecasts 2019
