"""Smoke test: every stage runs on synthetic data; the numbers mean nothing.

Usage:
    uv run python scripts/smoke_test.py              # pipeline only, offline, seconds
    uv run python scripts/smoke_test.py --notebooks  # also executes ms2.ipynb and ms3.ipynb
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from volforecast import diagnostics, eda, factors, metrics, models, plots
from volforecast.clean import clean, coverage
from volforecast.features import build_table, daily_measures, long_memory
from volforecast.pipeline import walk_forward

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import make_daily  # noqa: E402  the tests' synthetic panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", action="store_true", help="also execute the notebooks")
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

    # Milestone 3: baselines and a ridge through the walk-forward loop, factors, EDA
    table = table.merge(long_memory(d), on=["ticker", "week"], how="left").dropna(subset=models.HAR)
    returns = daily_measures(d).set_index(["ticker", "date"])["ret"].sort_index()
    numeric = [c for c in table.columns if c not in ("ticker", "week", "target", "sic2")]
    heavy = [c for c in numeric if c not in models.HAR]  # every other feature, as normal scores
    candidates = {
        "per-stock HAR": models.PerStock(min_weeks=10), "pooled HAR": models.pooled_ols(models.HAR),
        "FE": models.FixedEffects(), "GARCH": models.Garch(returns, min_days=50, n_jobs=1),
        "ridge": models.linear(Ridge(alpha=10.0), models.HAR, heavy, ["sic2"]),
    }  # fmt: skip
    fc = {
        name: walk_forward(m, table, "2020-07-01", "2020-12-31", freq="QS")
        for name, m in candidates.items()
    }
    scores = metrics.compare(table["target"], fc, "per-stock HAR", table["week"])
    assert np.isfinite(scores[["qlike", "mse"]].to_numpy()).all()
    ic = factors.ic_summary(factors.rank_ic(table, ["har_w", "har_q"], "target"))
    shares = eda.variance_decomposition(table["target"], table["ticker"], table["week"])
    assert ic["hit"].between(0, 1).all() and shares.between(0, 1).all()
    for fig in [
        plots.model_scores(scores, "qlike", "per-stock HAR"),
        plots.variance_decomposition(shares),
    ]:
        assert fig.axes
    print(f"models: ok ({len(fc)} models, {scores.attrs['rows']} forecast rows)")

    if args.notebooks:
        for notebook in ["ms2.ipynb", "ms3.ipynb"]:
            cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", notebook]
            subprocess.run([sys.executable, "-m", *cmd], check=True, cwd=ROOT)
            print(f"{notebook}: ok")


if __name__ == "__main__":
    main()
