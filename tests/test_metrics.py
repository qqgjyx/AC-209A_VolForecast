import numpy as np
import pandas as pd
import pytest

from volforecast import metrics


def test_qlike_is_zero_at_the_truth_and_punishes_under_forecasts_more():
    target = pd.Series(np.log([0.01, 0.01]))
    assert metrics.qlike(target, pd.Series([0.01, 0.01])).tolist() == pytest.approx([0, 0])
    under, over = metrics.qlike(target, pd.Series([0.005, 0.02]))
    assert under > over > 0


def frame(n_weeks: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2016-01-01", periods=n_weeks, freq="W-FRI")
    idx = pd.MultiIndex.from_product([weeks, ["A", "B", "C"]]).to_frame(index=False)
    idx.columns = ["week", "ticker"]
    target = pd.Series(rng.normal(-6, 0.5, len(idx)))

    def forecast(noise):
        log_pred = target + rng.normal(0, noise, len(target))
        return pd.DataFrame({"log_pred": log_pred, "var_pred": np.exp(log_pred)})

    return idx, target, forecast


def test_compare_finds_the_better_model_and_scores_common_rows():
    idx, target, forecast = frame()
    good, bad = forecast(0.1), forecast(0.6).iloc[:-3]  # bad misses the last week
    out = metrics.compare(target, {"good": good, "bad": bad}, "bad", idx["week"])
    assert out.attrs["rows"] == len(bad)
    assert out.loc["good", "qlike vs bad"] < 0 and out.loc["good", "DM qlike"] < -2
    assert out.loc["good", "p mse"] < 0.01


def test_dm_test_is_symmetric():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(1, 0.1, 100)), pd.Series(rng.normal(1.1, 0.1, 100))
    s_ab, p_ab = metrics.dm_test(a, b)
    s_ba, p_ba = metrics.dm_test(b, a)
    assert s_ab == pytest.approx(-s_ba) and p_ab == pytest.approx(p_ba) and s_ab < 0


def test_mincer_zarnowitz_of_a_perfect_forecast():
    y = pd.Series(np.linspace(-8, -4, 50))
    mz = metrics.mincer_zarnowitz(y, y)
    assert mz["slope"] == pytest.approx(1) and mz["intercept"] == pytest.approx(0, abs=1e-9)


def test_by_group_means():
    idx, target, forecast = frame()
    f = forecast(0.2)
    groups = pd.Series(np.where(idx["ticker"] == "A", "a", "other"))
    out = metrics.by_group(target, {"m": f}, groups, "mse")
    assert out.loc["a", "m"] == pytest.approx(((target - f["log_pred"]) ** 2)[groups == "a"].mean())


def test_by_group_scores_the_rows_every_model_forecasts():
    _, target, forecast = frame()
    a, b = forecast(0.2), forecast(0.1).iloc[: len(target) // 2]
    out = metrics.by_group(target, {"a": a, "b": b}, pd.Series("all", index=target.index), "mse")
    assert out.loc["all", "a"] == pytest.approx(((target - a["log_pred"]) ** 2).loc[b.index].mean())
