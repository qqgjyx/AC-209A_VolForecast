"""Summary statistics behind the Milestone 3 EDA, each tied to a modeling decision.

Every function here reads the training years only when it informs a choice (a transform, a
feature, a model); ms3.ipynb passes them the training rows. The figures are in `plots`.
"""

import numpy as np
import pandas as pd

from volforecast.models import MIN_WEEKS
from volforecast.pipeline import fit_rows

HEAVY_KURTOSIS = 10  # excess kurtosis above which a feature is mapped to normal scores


def split_summary(table: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    """Per split (`table["split"]`): stock-weeks, companies, weeks, stress weeks, own-HAR share.

    The own-HAR share is the share of the split's stock-weeks whose company has at least
    `MIN_WEEKS` rows that the January refit forecasting them may use (`pipeline.fit_rows`), so
    per-stock HAR fits it on its own rows. Not defined for the training years, which are not
    forecast.
    """
    own = pd.Series(False, index=table.index)
    for year, index in table.groupby(table["week"].dt.year).groups.items():
        history = fit_rows(table, pd.Timestamp(year, 1, 1))["ticker"].value_counts()
        own[index] = table.loc[index, "ticker"].map(history).ge(MIN_WEEKS)
    return pd.DataFrame({
        name: {
            "stock-weeks": len(part), "companies": part["ticker"].nunique(),
            "weeks": part["week"].nunique(),
            "stress weeks": part.loc[stress[part.index], "week"].nunique(),
            "own HAR fit": float(own[part.index].mean()) if name != "train" else np.nan,
        }
        for name, part in table.groupby("split", sort=False)
    }).T  # fmt: skip


def tails(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Skewness, excess kurtosis and missing share per column, and the transform it gets."""
    out = pd.DataFrame({"skew": frame[columns].skew(), "kurtosis": frame[columns].kurt(),
                        "missing": frame[columns].isna().mean()})  # fmt: skip
    out["transform"] = np.where(out["kurtosis"] > HEAVY_KURTOSIS, "normal scores", "standardize")
    return out.sort_values("kurtosis", ascending=False)


def variance_decomposition(y: pd.Series, stock: pd.Series, week: pd.Series,
                           iters: int = 200, tol: float = 1e-8) -> pd.Series:  # fmt: skip
    """Share of the target's variance explained by a stock effect, a week effect, and both.

    The two-way fit on an unbalanced panel is found by alternating projections: remove the
    stock means, then the week means, until no mean moves by more than `tol`.
    """
    total = float(((y - y.mean()) ** 2).sum())

    def share(resid: pd.Series) -> float:
        return 1 - float((resid**2).sum()) / total

    by_stock = y - y.groupby(stock).transform("mean")
    by_week = y - y.groupby(week).transform("mean")
    resid = y - y.mean()
    for _ in range(iters):
        by_s = resid.groupby(stock).transform("mean")
        resid = resid - by_s
        by_w = resid.groupby(week).transform("mean")
        resid = resid - by_w
        if max(by_s.abs().max(), by_w.abs().max()) < tol:
            break
    else:
        raise RuntimeError(f"the two-way fit did not converge in {iters} passes")
    return pd.Series(
        {"stock": share(by_stock), "week": share(by_week), "stock + week": share(resid)}
    )


def binned_means(
    frame: pd.DataFrame, columns: list[str], target: str, bins: int = 20
) -> pd.DataFrame:
    """Mean of `target` within each ventile of each column: a check that the relation is linear."""
    parts = []
    for c in columns:
        b = pd.qcut(frame[c], bins, labels=False, duplicates="drop")
        g = frame.groupby(b).agg(x=(c, "mean"), y=(target, "mean"))
        parts.append(g.assign(feature=c))
    return pd.concat(parts, ignore_index=True)


def common_miss(resid: pd.Series, week: pd.Series, market: pd.DataFrame) -> pd.DataFrame:
    """The weekly cross-sectional mean of HAR's miss next to market variables of that week.

    market: one row per week (the table's market columns). Adds this week's change in log VIX.
    """
    weekly = resid.groupby(week).mean().rename("mean miss")
    m = market.groupby(week).first()
    m = m.assign(log_vix=np.log(m["VIXCLS"]), d_log_vix=np.log(m["VIXCLS"]).diff())
    return pd.concat([weekly, m], axis=1)
