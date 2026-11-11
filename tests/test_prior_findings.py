"""Tests for the six follow-up analyses in `prior_findings`."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from volforecast import prior_findings as pf


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opens."""
    yield
    plt.close("all")


def _panel(n_weeks: int = 300, n_stocks: int = 6, seed: int = 0) -> pd.DataFrame:
    """A stock-week table where target = c_i + 0.2 d + 0.3 w + 0.4 m + noise, c_i per stock."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2005-01-07", periods=n_weeks, freq="W-FRI")
    frames = []
    for i in range(n_stocks):
        x = rng.normal(0, 1, (n_weeks, 3))  # centered, so each constant is identified
        target = 0.5 * i + x @ [0.2, 0.3, 0.4] + rng.normal(0, 0.1, n_weeks)
        cols = {"har_d": x[:, 0], "har_w": x[:, 1], "har_m": x[:, 2], "target": target}
        frames.append(pd.DataFrame({"ticker": f"S{i}", "week": weeks, **cols}))
    return pd.concat(frames, ignore_index=True)


def test_train_rows_stop_at_the_end_of_2015():
    """Weeks after the training years never reach the analyses."""
    t = pd.DataFrame({"week": pd.to_datetime(["2015-12-25", "2016-01-01", "2004-12-31"])})
    assert pf.train_rows(t)["week"].tolist() == [pd.Timestamp("2015-12-25")]


def test_har_by_stock_recovers_known_slopes_and_constants():
    """Each company's fit finds its own constant and the shared slopes."""
    coefs = pf.har_by_stock(_panel())
    assert coefs[pf.HAR].mean().to_numpy() == pytest.approx([0.2, 0.3, 0.4], abs=0.02)
    assert coefs["const"].diff().dropna().to_numpy() == pytest.approx([0.5] * 5, abs=0.05)
    assert (coefs["weeks"] == 300).all()


def test_companies_with_too_few_weeks_get_no_fit():
    """Companies under the minimum number of weeks are left out."""
    t = _panel(n_weeks=50)
    assert pf.har_by_stock(t).empty
    assert len(pf.har_by_stock(t, min_weeks=10)) == 6


def test_a_constant_per_company_removes_the_level_gap_pooling_leaves():
    """Pooled HAR misses each company's level, a constant per company doesn't."""
    gain = pf.pooling_gain(_panel())
    assert gain["pooled_with_constant"] < 0.5 * gain["pooled"]
    assert gain["per_stock"] == pytest.approx(0.01, rel=0.2)  # the noise variance, 0.1^2
    assert gain["companies"] == 6
    big = pf.pooling_gain(_panel(n_weeks=2000))  # pooled slopes carry the level gaps as noise
    assert [big[f"pooled_{h}"] for h in pf.HAR] == pytest.approx([0.2, 0.3, 0.4], abs=0.03)


def test_miss_by_regime_reports_mean_bias_share_and_volatility_gap():
    """A stress miss of +0.4 in log variance is +22.1% in volatility; calm misses average 0."""
    resid = pd.Series([0.4, 0.4, -0.1, 0.1])
    stress = pd.Series([True, True, False, False])
    miss = pf.miss_by_regime(resid, stress)
    assert miss.loc["stress", "mean"] == pytest.approx(0.4)
    assert miss.loc["stress", "vol_pct"] == pytest.approx((np.exp(0.2) - 1) * 100)
    assert miss.loc["stress", "bias_share"] == pytest.approx(1.0)
    assert miss.loc["calm", "mean"] == pytest.approx(0.0)


def test_annual_vol_turns_weekly_log_variance_into_percent():
    """4% weekly volatility is 4% x sqrt(52), about 28.8% a year."""
    assert pf.annual_vol(pd.Series([np.log(0.04**2)])).iloc[0] == pytest.approx(
        0.04 * 52**0.5 * 100
    )


def test_target_by_regime_splits_on_this_weeks_stress_flag():
    """Calm and stress rows are split by this week's flag."""
    t = pd.DataFrame(
        {"target": np.log([0.02**2, 0.02**2, 0.05**2]), "stress": [False, False, True]}
    )
    out = pf.target_by_regime(t)
    assert out.loc["calm", "50%"] == pytest.approx(0.02 * 52**0.5 * 100)
    assert out.loc["stress", "count"] == 1


def test_miss_by_return_orders_deciles_from_worst_to_best_week():
    """Residuals that fall with the return show up as a positive miss in decile 1."""
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0, 0.03, 5000))
    resid = -2 * ret + rng.normal(0, 0.01, 5000)
    out = pf.miss_by_return(pd.DataFrame({"ret": ret}), resid)
    assert out.loc[1, "vol_pct"] > 0 > out.loc[10, "vol_pct"]
    assert out["weeks"].sum() == 5000


def test_memory_matches_an_ar1():
    """For x_t = 0.8 x_(t-1) + e, the lag-k autocorrelation is about 0.8^k."""
    rng = np.random.default_rng(2)
    weeks = pd.date_range("1980-01-04", periods=2000, freq="W-FRI")
    frames = []
    for i in range(8):
        x = np.zeros(2000)
        for t in range(1, 2000):
            x[t] = 0.8 * x[t - 1] + rng.normal()
        frames.append(pd.DataFrame({"ticker": f"S{i}", "week": weeks, "har_w": x}))
    t = pd.concat(frames, ignore_index=True)
    acf = pf.memory(t, max_lag=3)
    assert acf.to_numpy() == pytest.approx([0.8, 0.64, 0.512], abs=0.07)


def test_memory_never_pairs_two_weeks_across_a_gap():
    """Two runs of rising weeks with a 5-week hole: only the pairs inside a run count.

    Within each run x_t = x_(t-1) + 1, so lag 1 correlates perfectly. The one pair across the
    hole (0 after 9) is far off that line, so a version that shifts by rows instead of by
    calendar week lands well below 1.
    """
    weeks = pd.date_range("2010-01-01", periods=25, freq="W-FRI")
    keep = np.r_[0:10, 15:25]  # weeks 10 to 14 are missing
    x = np.r_[np.arange(10.0), np.arange(10.0)]
    t = pd.DataFrame({"ticker": "S0", "week": weeks[keep], "har_w": x})
    assert pf.memory(t, max_lag=1, min_weeks=1).iloc[0] == pytest.approx(1.0)
    by_rows = t["har_w"].corr(t["har_w"].shift(1))  # the pairing the function must avoid
    assert by_rows < 0.8  # 0.67 here


def test_comovement_is_higher_when_a_common_factor_dominates():
    """A strong common factor in stress weeks and a weak one in calm weeks."""
    rng = np.random.default_rng(3)
    weeks = pd.date_range("2005-01-07", periods=200, freq="W-FRI")
    stress = np.arange(200) >= 150
    factor = rng.normal(0, 1, 200) * np.where(stress, 3.0, 0.5)
    frames = [pd.DataFrame({"ticker": f"S{i}", "week": weeks, "stress": stress,
                            "ret": factor + rng.normal(0, 1, 200)}) for i in range(10)]  # fmt: skip
    co = pf.comovement(pd.concat(frames, ignore_index=True))
    assert co["stress"] > 0.8 > 0.3 > co["calm"]
    assert co["stress_weeks"] == 50


def test_every_figure_draws_and_saves(tmp_path):
    """Each analysis figure draws something and writes its file."""
    t = _panel().assign(
        target=lambda d: d["target"] - 7,  # about 20% annual volatility, inside the plot's range
        stress=lambda d: d.index % 7 == 0,
        ret=lambda d: d["har_d"] / 100,
    )
    coefs = pf.har_by_stock(t)
    resid = pf.har_residuals(t, coefs)
    figs = {
        "slopes": pf.plot_har_slopes(coefs, 0.4, tmp_path / "a.png"),
        "regime": pf.plot_miss_by_regime(pf.miss_by_regime(resid, t["stress"]), tmp_path / "b.png"),
        "target": pf.plot_target_by_regime(t, tmp_path / "c.png"),
        "return": pf.plot_miss_by_return(pf.miss_by_return(t, resid), tmp_path / "d.png"),
        "memory": pf.plot_memory(
            pf.memory(t.rename(columns={"har_d": "x"}), 4), tmp_path / "e.png"
        ),
        "comove": pf.plot_comovement({"calm": 0.3, "stress": 0.5}, tmp_path / "f.png"),
    }
    assert all(f.axes for f in figs.values())
    assert all((tmp_path / f"{n}.png").stat().st_size > 0 for n in "abcdef")


def test_findings_table_has_one_row_per_analysis_with_both_results():
    """Every analysis gets a row with the earlier result, the new one and why it matters."""
    md = pf.findings_markdown({k: f"new {k}" for k in range(1, 7)})
    rows = [line for line in md.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert len(rows) == 6
    assert all(f"new {k}" in rows[k - 1] and pf.EARLIER[k] in rows[k - 1] for k in range(1, 7))
