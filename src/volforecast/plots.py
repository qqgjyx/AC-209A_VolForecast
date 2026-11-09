"""The few figures the notebook and slides use; one style for all of them."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.figsize": (8, 4), "figure.dpi": 120, "font.size": 12, "axes.titlesize": 14,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})  # fmt: skip


def _save(fig: plt.Figure, path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def coverage(shares: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Share of index members with usable prices over time, one line per column."""
    if shares.shape[1] > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series, got {shares.shape[1]}")
    fig, ax = plt.subplots()
    for color, (name, s) in zip(SERIES, shares.items(), strict=False):
        ax.plot(s.index, s * 100, color=color, label=name)
        ax.annotate(f"{s.iloc[-1] * 100:.0f}%", (s.index[-1], s.iloc[-1] * 100),
                    xytext=(4, 0), textcoords="offset points", va="center", color=INK)  # fmt: skip
    ax.set(title="Survivorship: index members with price data", xlabel="Date",
           ylabel="Members with data (%)", ylim=(50, 101))  # fmt: skip
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def target_hist(target: pd.Series, path: Path | None = None) -> plt.Figure:
    """Distribution of the response, the log next-week realized variance."""
    fig, ax = plt.subplots()
    ax.hist(target.dropna(), bins=120, color=SERIES[0], edgecolor="white", linewidth=0.3)
    ax.set(title="Response: log next-week Garman-Klass variance", xlabel="log weekly variance",
           ylabel="Stock-weeks")  # fmt: skip
    return _save(fig, path)


def cleaning_steps(steps: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Rows removed by each cleaning step, largest first."""
    s = steps.set_index("step")["rows_removed"].iloc[1:].sort_values()
    fig, ax = plt.subplots()
    bars = ax.barh(s.index, s.to_numpy(), color=SERIES[0], height=0.6)
    ax.bar_label(bars, labels=[f"{v:,}" for v in s], padding=4)
    ax.set(title="Stock-days removed by each cleaning step", xlabel="Rows removed (log scale)",
           xscale="log", xlim=(0.8, s.max() * 8))  # fmt: skip
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def volatility_over_time(
    log_var: pd.Series, week: pd.Series, path: Path | None = None
) -> plt.Figure:
    """Cross-sectional median and 10-90% band of annualized volatility, week by week."""
    vol = (np.exp(log_var) * 52) ** 0.5 * 100
    q = vol.groupby(week).quantile([0.1, 0.5, 0.9]).unstack()
    fig, ax = plt.subplots()
    ax.fill_between(q.index, q[0.1], q[0.9], color=SERIES[0], alpha=0.2, linewidth=0,
                    label="10th-90th percentile")  # fmt: skip
    ax.plot(q.index, q[0.5], color=SERIES[0], linewidth=1.5, label="Median stock")
    ax.set(title="Weekly realized volatility across S&P 500 stocks", xlabel="Week",
           ylabel="Annualized volatility (%)", yscale="log")  # fmt: skip
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def proxy_check(reference: pd.Series, proxy: pd.Series, path: Path | None = None) -> plt.Figure:
    """Hexbin of a volatility proxy against a reference measure, with the 45-degree line."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hexbin(reference, proxy, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    lo, hi = np.nanpercentile(np.r_[reference, proxy], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color=SERIES[1], linewidth=1.5, label="45-degree line")
    ax.set(title="Daily-range target vs 5-minute realized variance", xlim=(lo, hi), ylim=(lo, hi))
    ax.set(xlabel="log weekly 5-minute RV", ylabel="log weekly Garman-Klass variance")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)
