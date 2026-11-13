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
        plots.coverage(pd.DataFrame(dict.fromkeys("abcd", 1.0), index=MONTHS))


def test_coverage_ends_a_series_that_stops_early():
    shares = pd.DataFrame({"full": 1.0, "short": [0.8] * 6 + [np.nan] * 6}, index=MONTHS)
    line = plots.coverage(shares).axes[0].lines[1]
    assert len(line.get_xdata()) == 6


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


def test_variance_coverage_marks_each_median(tmp_path):
    shares = pd.DataFrame({"GK": [0.5, 0.6, 0.7], "GK + overnight": [0.9, 1.0, 1.1]})
    fig = plots.variance_coverage(shares, tmp_path / "v.png")
    assert (tmp_path / "v.png").stat().st_size > 0
    medians = [line.get_xdata()[0] for line in fig.axes[0].lines[:2]]
    assert medians == pytest.approx([60, 100])


def test_stress_weeks_draws_a_bar_per_year_and_threshold():
    counts = pd.DataFrame({"VIX >= 30": [1, 16], "VIX >= 25": [3, 20]}, index=[2019, 2020])
    fig = plots.stress_weeks(counts, {"test": (2019, 2020)})
    heights = [p.get_height() for p in fig.axes[0].patches if p.get_width() < 1]
    assert sorted(heights) == [1, 3, 16, 20]


def test_new_plots_refuse_more_series_than_colors():
    too_many = pd.DataFrame({f"s{i}": [1.0] for i in range(len(plots.SERIES) + 1)}, index=[2020])
    with pytest.raises(ValueError):
        plots.variance_coverage(too_many)
    with pytest.raises(ValueError):
        plots.stress_weeks(too_many, {})
