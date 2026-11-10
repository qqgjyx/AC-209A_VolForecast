"""Six follow-up analyses on the stock-week table, rerun from earlier work on 25 stocks.

Each one relates something known at a Friday close to next week's target, so all of them use
the training years only (2005-2015): the tuning and test years must not shape a model choice.
The notebook `prior_findings.ipynb` in this folder runs them and presents the results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volforecast import plots  # the shared figure style
from volforecast.diagnostics import weekly_vix
from volforecast.features import daily_measures, weekly_panel

TRAIN = ("2005-01-01", "2015-12-31")
HAR = ["har_d", "har_w", "har_m"]
STRESS_VIX = 30.0  # the project's stress line (VIX at the week's last close)
MIN_WEEKS = 104  # two years of weeks before a company gets its own HAR fit
CALM, STRESS = plots.SERIES  # blue for calm, orange for stress


def repo_root(start: Path | None = None) -> Path:
    """The folder holding pyproject.toml, found by walking up from `start` (or this file)."""
    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("no pyproject.toml above " + str(here))


def weekly_table(daily: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Stock-weeks in the index with a target, as in `features.build_table`, plus weekly VIX.

    Only the columns these analyses need: the target, the HAR inputs and the week's return.
    """
    w = weekly_panel(daily_measures(daily))
    w = w[w["in_index"].astype(bool) & w["target"].notna()].drop(columns="in_index")
    w = w.join(weekly_vix(vix).rename("vix"), on="week")
    return w.assign(stress=w["vix"].ge(STRESS_VIX)).reset_index(drop=True)


def train_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows whose week falls in the training years."""
    return table[table["week"].between(*map(pd.Timestamp, TRAIN))]


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of y on [1, x]."""
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def har_by_stock(table: pd.DataFrame, min_weeks: int = MIN_WEEKS) -> pd.DataFrame:
    """HAR fitted to each company separately: constant, three slopes, weeks used, in-sample MSE."""
    rows = {}
    for ticker, g in table.dropna(subset=[*HAR, "target"]).groupby("ticker"):
        if len(g) < min_weeks:
            continue
        x, y = g[HAR].to_numpy(), g["target"].to_numpy()
        coef = _ols(x, y)
        resid = y - np.column_stack([np.ones(len(x)), x]) @ coef
        rows[ticker] = [*coef, len(g), float(np.mean(resid**2))]
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["const", *HAR, "weeks", "mse"]
    ).rename_axis("ticker")


def har_residuals(table: pd.DataFrame, coefs: pd.DataFrame) -> pd.Series:
    """Target minus each company's own HAR fit, for companies in `coefs` (NaN elsewhere)."""
    c = coefs.reindex(table["ticker"]).to_numpy()
    fitted = c[:, 0] + np.einsum("ij,ij->i", c[:, 1:4], table[HAR].to_numpy())
    return pd.Series(table["target"].to_numpy() - fitted, index=table.index, name="resid")


def pooling_gain(table: pd.DataFrame, min_weeks: int = MIN_WEEKS) -> dict[str, float]:
    """In-sample MSE of one pooled HAR, the same with a constant per company, and per-company HAR.

    All three use the same rows (companies with at least `min_weeks` weeks). The constant per
    company is fitted by removing each company's mean from the target and the inputs first.
    """
    keep = table.dropna(subset=[*HAR, "target"])
    keep = keep[keep.groupby("ticker")["target"].transform("size") >= min_weeks]
    x, y = keep[HAR].to_numpy(), keep["target"].to_numpy()
    pooled_coef = _ols(x, y)
    pooled = y - np.column_stack([np.ones(len(x)), x]) @ pooled_coef
    demeaned = keep[[*HAR, "target"]] - keep.groupby("ticker")[[*HAR, "target"]].transform("mean")
    xd, yd = demeaned[HAR].to_numpy(), demeaned["target"].to_numpy()
    slopes = np.linalg.lstsq(xd, yd, rcond=None)[0]
    with_constant = yd - xd @ slopes
    per_stock = har_residuals(keep, har_by_stock(keep, min_weeks))
    return {
        "pooled": float(np.mean(pooled**2)),
        "pooled_with_constant": float(np.mean(with_constant**2)),
        "per_stock": float(np.mean(per_stock**2)),
        **{f"pooled_{name}": float(c) for name, c in zip(HAR, pooled_coef[1:], strict=True)},
        "companies": float(keep["ticker"].nunique()),
        "weeks": float(len(keep)),
    }


def vol_change(log_variance_gap: pd.Series) -> pd.Series:
    """A gap in log variance as a percent change in volatility: (exp(gap / 2) - 1) * 100."""
    return (np.exp(log_variance_gap / 2) - 1) * 100


def miss_by_regime(resid: pd.Series, stress: pd.Series) -> pd.DataFrame:
    """Per regime: weeks, mean miss (log variance and volatility %), MSE, and the bias share.

    A positive mean miss means realized variance ran above the forecast. The bias share is
    mean^2 / MSE: the part of the squared error a level shift alone would remove.
    """
    frame = pd.DataFrame({"resid": resid, "regime": np.where(stress, "stress", "calm")})
    out = (
        frame.dropna()
        .groupby("regime")["resid"]
        .agg(weeks="size", mean="mean", mse=lambda r: float(np.mean(r**2)))
    )
    out["vol_pct"] = vol_change(out["mean"])
    out["bias_share"] = out["mean"] ** 2 / out["mse"]
    return out


def annual_vol(log_weekly_variance: pd.Series) -> pd.Series:
    """Weekly log variance as annualized volatility in percent (52 weeks)."""
    return np.sqrt(np.exp(log_weekly_variance) * 52) * 100


def target_by_regime(table: pd.DataFrame) -> pd.DataFrame:
    """Next-week annualized volatility by this week's regime: weeks and quartiles."""
    vol = annual_vol(table["target"]).rename("vol")
    regime = pd.Series(np.where(table["stress"], "stress", "calm"), index=table.index)
    return vol.groupby(regime).describe()[["count", "25%", "50%", "75%"]]


def miss_by_return(table: pd.DataFrame, resid: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean HAR miss by decile of this week's return, in volatility %."""
    frame = pd.DataFrame({"ret": table["ret"], "resid": resid}).dropna()
    decile = pd.qcut(frame["ret"], bins, labels=False) + 1
    out = frame.groupby(decile).agg(weeks=("resid", "size"), ret=("ret", "median"),
                                    mean=("resid", "mean"))  # fmt: skip
    out["vol_pct"] = vol_change(out["mean"])
    return out.rename_axis("decile")


def memory(table: pd.DataFrame, max_lag: int = 52, min_weeks: int = MIN_WEEKS) -> pd.Series:
    """Median across companies of the autocorrelation of weekly log variance, by lag in weeks.

    Each company's weeks sit on one Friday calendar, so a gap in its rows is a gap in time and
    never pairs two weeks that aren't `lag` weeks apart.
    """
    wide = table.pivot(index="week", columns="ticker", values="har_w")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="W-FRI"))
    wide = wide.loc[:, wide.notna().sum() >= min_weeks]
    acf = {lag: wide.corrwith(wide.shift(lag)).median() for lag in range(1, max_lag + 1)}
    return pd.Series(acf, name="acf").rename_axis("lag")


def mean_pair_correlation(returns: pd.DataFrame, min_coverage: float = 0.9) -> float:
    """Average correlation over all pairs of companies with at least `min_coverage` of rows."""
    r = returns.loc[:, returns.notna().mean() >= min_coverage]
    c = r.corr().to_numpy()
    return float(c[np.triu_indices_from(c, k=1)].mean())


def comovement(table: pd.DataFrame) -> dict[str, float]:
    """Average pairwise correlation of weekly returns over calm weeks and over stress weeks."""
    wide = table.pivot(index="week", columns="ticker", values="ret")
    stress = table.groupby("week")["stress"].first().reindex(wide.index)
    return {
        "calm": mean_pair_correlation(wide[~stress]),
        "stress": mean_pair_correlation(wide[stress]),
        "calm_weeks": float((~stress).sum()),
        "stress_weeks": float(stress.sum()),
    }


# The summary table, written from the computed values.

EARLIER = {  # the same analyses on 25 current stocks, daily rows, 2010 onward (from the record)
    1: "monthly slope 0.12 to 0.61; a constant per stock cut in-sample MSE 3.9%",
    2: "realized vol about 20% above HAR's forecast on stress days, out of sample 2019-2026",
    3: "median next-week vol 27.8% after stress days, 16.7% after calm days (VIX >= 25)",
    4: "+9.3% after the biggest down days, -5.0% after the biggest up days",
    5: "absolute daily returns: autocorrelation 0.22 at 1 day, about 0.05 at 60 days",
    6: "daily return correlation 0.28 calm, 0.47 stress (VIX >= 25)",
}
WHY = {
    1: "A second baseline, HAR pooled with a constant per company, separates the gain from "
    "pooling from the gain from features.",
    2: "A model can look better in stress weeks just by knowing VIX. A HAR+VIX baseline shows "
    "how much of a stress-week gain that is.",
    3: "The size of the regime gap Q2 is about.",
    4: "Evidence for the returns and downside-variance features before Q2's block ablation.",
    5: "Memory past HAR's 4-week window is what a longer lag or a model could still use.",
    6: "Evidence for the market-wide features (SPY variance, VIX).",
}


def findings_markdown(now: dict[int, str]) -> str:
    """The markdown table of the six analyses: earlier result, result on this table, and why."""
    titles = {
        1: "HAR fitted company by company",
        2: "HAR's miss in calm and stress weeks",
        3: "Next week's volatility after calm and stress weeks",
        4: "HAR's miss by this week's return",
        5: "How long volatility remembers",
        6: "How much stocks move together",
    }
    lines = [
        "<!-- Written by prior_findings.ipynb; do not edit. -->",
        "| # | Analysis | Earlier (25 stocks, daily) | Now (this table, weekly, 2005-2015) | "
        "Why it matters |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {k} | {titles[k]} | {EARLIER[k]} | {now[k]} | {WHY[k]} |" for k in titles]
    return "\n".join(lines) + "\n"


# Figures: one per analysis, in the shared style.


def _save(fig: plt.Figure, path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_har_slopes(coefs: pd.DataFrame, pooled_m: float, path: Path | None = None) -> plt.Figure:
    """Histogram of each company's monthly HAR slope, with the pooled slope marked."""
    fig, ax = plt.subplots()
    ax.hist(coefs["har_m"], bins=40, color=CALM, edgecolor="white", linewidth=0.3)
    ax.axvline(pooled_m, color=STRESS, linewidth=1.5, label=f"pooled HAR: {pooled_m:.2f}")
    ax.set(title="HAR's monthly slope, fitted company by company (2005-2015)",
           xlabel="Slope on the 4-week lag (har_m)", ylabel="Companies")  # fmt: skip
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def plot_miss_by_regime(miss: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Mean HAR miss in calm and stress weeks, as a percent of volatility."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(miss.index, miss["vol_pct"], color=[CALM, STRESS], width=0.5)
    ax.bar_label(bars, labels=[f"{v:+.1f}%" for v in miss["vol_pct"]], padding=3)
    ax.axhline(0, color=plots.MUTED, linewidth=1)
    ax.set(title="Realized vs HAR forecast, by regime (2005-2015)",
           ylabel="Realized vol above forecast (%)")  # fmt: skip
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_target_by_regime(table: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Distribution of next-week annualized volatility after calm and after stress weeks."""
    vol = annual_vol(table["target"])
    bins = list(np.geomspace(5, 200, 60))  # 5% to 200% a year, even steps on a log axis
    fig, ax = plt.subplots()
    for name, mask, color in [
        ("calm", ~table["stress"], CALM),
        ("stress", table["stress"], STRESS),
    ]:
        v = vol[mask]
        ax.hist(v, bins=bins, color=color, alpha=0.55, density=True,
                label=f"{name} week: median {v.median():.0f}%")  # fmt: skip
    ax.set(title="Next week's volatility after calm and stress weeks (2005-2015)",
           xlabel="Annualized volatility (%)", ylabel="Density", xscale="log")  # fmt: skip
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, path)


def plot_miss_by_return(miss: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Mean HAR miss by decile of this week's return."""
    fig, ax = plt.subplots()
    colors = [STRESS if v > 0 else CALM for v in miss["vol_pct"]]
    ax.bar(miss.index, miss["vol_pct"], color=colors, width=0.7)
    ax.axhline(0, color=plots.MUTED, linewidth=1)
    ax.set(title="HAR's miss by this week's return (2005-2015)",
           xlabel="Decile of this week's return (1 = worst)", xticks=list(miss.index),
           ylabel="Next week's realized vol above forecast (%)")  # fmt: skip
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_memory(acf: pd.Series, path: Path | None = None) -> plt.Figure:
    """Median autocorrelation of weekly log variance by lag, with HAR's 4-week window shaded."""
    fig, ax = plt.subplots()
    ax.axvspan(0.5, 4.5, color=plots.GRID, alpha=0.8, label="covered by HAR's lags (1-4 weeks)")
    ax.plot(acf.index, acf.to_numpy(), color=CALM, marker="o", markersize=3)
    ax.set(title="How long volatility remembers (2005-2015)", xlabel="Lag (weeks)",
           ylabel="Median autocorrelation of log variance", ylim=(0, 1))  # fmt: skip
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, path)


def plot_comovement(co: dict[str, float], path: Path | None = None) -> plt.Figure:
    """Average pairwise correlation of weekly returns in calm and in stress weeks."""
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [co["calm"], co["stress"]]
    bars = ax.bar(["calm", "stress"], values, color=[CALM, STRESS], width=0.5)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    ax.set(title="How much stocks move together (2005-2015)",
           ylabel="Average pairwise correlation of weekly returns", ylim=(0, 1))  # fmt: skip
    ax.grid(axis="x", visible=False)
    return _save(fig, path)
