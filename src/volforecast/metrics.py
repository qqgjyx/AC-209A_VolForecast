"""Forecast losses and the tests that compare them.

QLIKE (Patton, 2011) scores the variance forecast against the realized variance and ranks
forecasts correctly even though the target is a noisy proxy; MSE scores the log forecast against
the log target. A Diebold-Mariano (1995) test compares two models on the weekly cross-sectional
mean of the loss difference, so the dependence between stocks in a week is respected.
"""

from functools import reduce

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import diebold_mariano_test

LOSSES = ["qlike", "mse"]


def qlike(target: pd.Series, var_pred: pd.Series) -> pd.Series:
    """QLIKE of a variance forecast: ratio - log(ratio) - 1 with ratio = realized / forecast.

    target: log realized variance; zero for a perfect forecast, and under-forecasts cost more.
    """
    ratio = np.exp(target) / var_pred
    return ratio - np.log(ratio) - 1


def losses(target: pd.Series, forecast: pd.DataFrame) -> pd.DataFrame:
    """Per-row QLIKE and squared error of a `pipeline.walk_forward` forecast."""
    return pd.DataFrame({
        "qlike": qlike(target, forecast["var_pred"]),
        "mse": (target - forecast["log_pred"]) ** 2,
    })  # fmt: skip


def weekly_mean(loss: pd.DataFrame, week: pd.Series) -> pd.DataFrame:
    """Cross-sectional mean of each loss per week: the series a Diebold-Mariano test runs on."""
    return loss.groupby(week.reindex(loss.index)).mean()


def dm_test(weekly_a: pd.Series, weekly_b: pd.Series) -> tuple[float, float]:
    """Diebold-Mariano test of equal mean weekly loss; statistic < 0 means A has lower loss.

    statsmodels' test with Newey-West errors (ceil(T^(1/3)) lags) and the Harvey, Leybourne and
    Newbold (1997) small-sample correction. The losses are passed as the forecasts, with the
    identity as the criterion, so the loss differential is weekly_a - weekly_b.
    """
    res = diebold_mariano_test(np.zeros(len(weekly_a)), weekly_a.to_numpy(), weekly_b.to_numpy(),
                               criterion=lambda _, loss: loss, harvey_adj=True)  # fmt: skip
    return float(res.statistic), float(res.pvalue)


def _common_losses(
    target: pd.Series, forecasts: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Per-row losses of each model on the rows all of them forecast: a like-for-like score."""
    common = reduce(pd.Index.intersection, [f.index for f in forecasts.values()])
    return {name: losses(target.loc[common], f.loc[common]) for name, f in forecasts.items()}


def compare(target: pd.Series, forecasts: dict[str, pd.DataFrame], benchmark: str,
            week: pd.Series) -> pd.DataFrame:  # fmt: skip
    """Mean QLIKE and MSE per model, the change against `benchmark`, and a DM test of each.

    Every model is scored on the rows all of them forecast, so the comparison is like for like.
    """
    per_model = _common_losses(target, forecasts)
    weekly = {name: weekly_mean(loss, week) for name, loss in per_model.items()}
    base = {k: per_model[benchmark][k].mean() for k in LOSSES}
    rows = {}
    for name, loss in per_model.items():
        row = {k: loss[k].mean() for k in LOSSES}
        for k in LOSSES:
            row[f"{k} vs {benchmark}"] = row[k] / base[k] - 1
            if name != benchmark:
                row[f"DM {k}"], row[f"p {k}"] = dm_test(weekly[name][k], weekly[benchmark][k])
        rows[name] = row
    out = pd.DataFrame(rows).T
    out.attrs["rows"] = len(per_model[benchmark])
    return out


def by_group(target: pd.Series, forecasts: dict[str, pd.DataFrame], groups: pd.Series,
             loss: str = "qlike") -> pd.DataFrame:  # fmt: skip
    """Mean `loss` of each model within each group (a regime, a year, a volatility decile).

    As in `compare`, every model is scored on the rows all of them forecast.
    """
    return pd.DataFrame({
        name: rows[loss].groupby(groups.reindex(rows.index)).mean()
        for name, rows in _common_losses(target, forecasts).items()
    })  # fmt: skip


def mincer_zarnowitz(target: pd.Series, log_pred: pd.Series) -> dict[str, float]:
    """Regression of the target on the log forecast: intercept, slope and R^2.

    An unbiased forecast has intercept 0 and slope 1; a slope below 1 means the forecasts move
    too much (they should be shrunk toward the mean).
    """
    x, y = log_pred.to_numpy(), target.loc[log_pred.index].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return {"intercept": float(intercept), "slope": float(slope), "r2": float(r2)}
