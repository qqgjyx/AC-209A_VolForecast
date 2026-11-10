"""Smoke test: every stage runs on synthetic data; the numbers mean nothing.

Usage:
    uv run python scripts/smoke_test.py              # pipeline only, offline, seconds
    uv run python scripts/smoke_test.py --notebooks  # also executes ms2.ipynb in place
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast import diagnostics, plots
from volforecast.clean import clean, coverage
from volforecast.features import build_table, daily_measures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import make_daily  # noqa: E402  the tests' synthetic panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", action="store_true", help="also execute ms2.ipynb")
    args = parser.parse_args()

    daily = make_daily()
    membership = pd.DataFrame({"date": pd.Timestamp("2020-01-01"), "ticker": ["AAA", "BBB", "CCC"]})
    d, steps = clean(daily, membership, set(), {}, "2021-01-01")
    cov = coverage(membership, d)
    assert cov.eq(1).all()
    spy = daily[daily["ticker"] == "DDD"].assign(ticker="SPY")
    fred = pd.DataFrame({"VIXCLS": 20.0}, index=daily["date"].unique())
    times = pd.to_datetime(["2020-02-03 16:30", "2020-05-04 16:30"]).tz_localize("US/Eastern")
    events = pd.DataFrame({"ticker": "AAA", "sic": "3571", "accepted": times})
    table = build_table(d, spy, fred, events)
    assert np.isfinite(table["target"]).all() and table.shape[1] > 158
    for fig in [
        plots.coverage(cov.to_frame()),
        plots.cleaning_steps(steps),
        plots.target_hist(table["target"]),
        plots.volatility_over_time(table["har_w"], table["week"]),
        plots.proxy_check(table["har_w"], table["log_rv_parkinson"]),
    ]:
        assert fig.axes
    shares = diagnostics.variance_shares(daily_measures(d), "2020-01-01", "2020-12-31", min_days=1)
    assert shares["gk"].between(0, 5).all()
    vix_weeks = diagnostics.weekly_vix(fred["VIXCLS"])
    by_year = diagnostics.stress_weeks_by_year(vix_weeks, [30, 25])
    assert diagnostics.stress_by_split(vix_weeks, 30)["weeks"].sum() == len(vix_weeks)
    for fig in [
        plots.variance_coverage(shares[["gk", "gk_overnight"]]),
        plots.stress_weeks(by_year, {"test": (2020, 2020)}),
    ]:
        assert fig.axes
    print(f"pipeline: ok ({len(steps)} cleaning steps, {table.shape[0]} x {table.shape[1]} table)")

    if args.notebooks:
        cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "ms2.ipynb"]
        subprocess.run([sys.executable, "-m", *cmd], check=True, cwd=ROOT)
        print("ms2.ipynb: ok")


if __name__ == "__main__":
    main()
