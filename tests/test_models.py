import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from volforecast import models

HAR = models.HAR


def har_panel(n_weeks: int = 120, seed: int = 0) -> pd.DataFrame:
    """Two companies with their own HAR coefficients, and a third with only 20 weeks."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-01", periods=n_weeks, freq="W-FRI")
    rows = []
    for ticker, const, b, n in [("AAA", -1.0, 0.8, n_weeks), ("BBB", -3.0, 0.4, n_weeks),
                                ("CCC", -2.0, 0.6, 20)]:  # fmt: skip
        x = rng.normal(-6, 1, (n, 3))
        y = const + b * x[:, 1] + 0.1 * x[:, 2] + rng.normal(0, 0.01, n)
        rows.append(pd.DataFrame({"ticker": ticker, "week": weeks[:n], "har_d": x[:, 0],
                                  "har_w": x[:, 1], "har_m": x[:, 2], "target": y}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_per_stock_har_recovers_each_companys_coefficients():
    t = har_panel()
    m = models.PerStock().fit(t, t["target"])
    assert m.coef_.loc["AAA", "har_w"] == pytest.approx(0.8, abs=0.01)
    assert m.coef_.loc["BBB", "const"] == pytest.approx(-3.0, abs=0.1)
    assert "CCC" not in m.coef_.index  # 20 weeks: the pooled fit forecasts it
    new = t[t["ticker"] == "CCC"].assign(ticker="NEW")
    np.testing.assert_allclose(m.predict(new), m.pooled_.predict(new[HAR].to_numpy()))


def test_fixed_effects_share_slopes_and_keep_constants():
    t = har_panel()
    t = t.assign(target=t["har_w"] * 0.5 + t["ticker"].map({"AAA": 1.0, "BBB": -1.0, "CCC": 0.0}))
    m = models.FixedEffects().fit(t, t["target"])
    assert m.slopes_[1] == pytest.approx(0.5)
    assert m.const_["AAA"] - m.const_["BBB"] == pytest.approx(2.0)
    np.testing.assert_allclose(m.predict(t), t["target"], atol=1e-9)


def test_reference_models():
    t = har_panel()
    assert np.array_equal(models.LastWeek().fit(t, t["target"]).predict(t), t["har_w"].to_numpy())
    sm = models.StockMean().fit(t, t["target"])
    assert sm.predict(t.assign(ticker="NEW"))[0] == pytest.approx(t["target"].mean())


def test_linear_pipeline_handles_missing_values_and_unseen_industries():
    t = har_panel().assign(heavy=lambda d: np.exp(d["har_d"] + 6) ** 3, sic2="10")
    t.loc[::7, "har_m"] = np.nan
    pipe = models.linear(Ridge(alpha=1.0), HAR, ["heavy"], ["sic2"]).fit(t, t["target"])
    pred = pipe.predict(t.assign(sic2="99"))
    assert np.isfinite(pred).all()
    assert any("missingindicator_har_m" in n for n in pipe[0].get_feature_names_out())


def test_garch_forecast_uses_no_later_returns():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2010-01-01", periods=900)
    r = pd.Series(rng.standard_t(6, len(days)) * 0.01, index=days)
    returns = pd.concat({"AAA": r}, names=["ticker", "date"])
    fridays = pd.date_range("2010-01-08", days[-1], freq="W-FRI")
    t = pd.DataFrame({"ticker": "AAA", "week": fridays, "target": -7.0})
    fit_rows = t[t["week"] < "2012-06-01"]
    m = models.Garch(returns, min_days=200, n_jobs=1).fit(
        fit_rows, fit_rows["target"] + rng.normal(0, 0.1, len(fit_rows))
    )
    query = t[t["week"].between("2012-06-01", "2012-09-28")]
    cut = pd.concat({"AAA": r.loc[:"2012-09-28"]}, names=["ticker", "date"])
    short = models.Garch(cut, min_days=200, n_jobs=1)
    short.params_, short.default_, short.calibration_, short.fill_ = (
        m.params_, m.default_, m.calibration_, m.fill_)  # fmt: skip
    np.testing.assert_allclose(m.predict(query), short.predict(query))
    assert np.isfinite(m.predict(query)).all()


def test_garch_forecasts_in_the_first_75_days_use_no_later_returns():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2010-01-04", periods=120)
    r = pd.Series(rng.standard_t(6, len(days)), index=days)  # percent
    params = np.array([0.0, 0.05, 0.1, 0.85])
    fridays = pd.date_range("2010-01-29", days[-1], freq="W-FRI")
    full = models._garch_log_week_var(r, params, fridays)
    cut = [models._garch_log_week_var(r.loc[:f], params, pd.DatetimeIndex([f]))[0] for f in fridays]
    np.testing.assert_allclose(full, cut)
