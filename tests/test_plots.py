import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from volforecast import plots

MONTHS = pd.date_range("2020-01-31", periods=12, freq="ME")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_coverage_plots_percent_and_saves(tmp_path):
    shares = pd.DataFrame({"before": np.linspace(0.5, 0.9, 12), "after": 1.0}, index=MONTHS)
    fig = plots.coverage(shares, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0
    assert fig.axes[0].lines[0].get_ydata()[-1] == pytest.approx(90)


def test_coverage_refuses_more_series_than_colors():
    with pytest.raises(ValueError):
        plots.coverage(pd.DataFrame(dict.fromkeys("abc", 1.0), index=MONTHS))


def test_cleaning_steps_skips_the_raw_row_and_sorts():
    steps = pd.DataFrame({"step": ["raw", "a", "b"], "rows_removed": [0, 5, 500]})
    labels = [t.get_text() for t in plots.cleaning_steps(steps).axes[0].get_yticklabels()]
    assert labels == ["a", "b"]


def test_volatility_over_time_annualizes():
    week = pd.Series(np.repeat(pd.date_range("2020-01-03", periods=3, freq="W-FRI"), 10))
    log_var = pd.Series(np.log(0.04**2), index=week.index)  # 4% weekly volatility
    median = plots.volatility_over_time(log_var, week).axes[0].lines[0].get_ydata()
    assert median == pytest.approx([0.04 * 52**0.5 * 100] * 3)


def test_target_hist_and_proxy_check_draw(tmp_path):
    x = pd.Series(np.random.default_rng(0).normal(-7, 1, 500))
    assert plots.target_hist(x).axes[0].patches
    assert plots.proxy_check(x, x + 0.1, tmp_path / "p.png").axes[0].collections
