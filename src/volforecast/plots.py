"""The few figures the notebook and slides use; one style for all of them."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette
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


def _colors(n: int, palette: list[str]) -> list[str]:
    """The first `n` colors of `palette`; more series than colors is an error, not a lost line."""
    if n > len(palette):
        raise ValueError(f"at most {len(palette)} series, got {n}")
    return palette[:n]


def _lines_vs_45(ax: plt.Axes, frames: dict[str, pd.DataFrame], palette: list[str]) -> None:
    """One line of `y` against `x` per frame (binned means), the 45-degree line, and a legend."""
    for color, (name, g) in zip(_colors(len(frames), palette), frames.items(), strict=True):
        ax.plot(g["x"], g["y"], marker="o", markersize=4, color=color, label=name)
    lo = min(g[["x", "y"]].min().min() for g in frames.values())
    hi = max(g[["x", "y"]].max().max() for g in frames.values())
    ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.5, linestyle="--", label="45-degree line")
    ax.legend(frameon=False, loc="upper left")


def coverage(shares: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Share of index members with usable prices over time, one line per column."""
    fig, ax = plt.subplots()
    for color, (name, s) in zip(_colors(shares.shape[1], SERIES), shares.items(), strict=True):
        s = s.dropna()  # a source may end early (CRSP's last vintage)
        ax.plot(s.index, s * 100, color=color, label=name)
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


def variance_coverage(shares: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Per company, the share of close-to-close variance two measures capture, with medians."""
    fig, ax = plt.subplots()
    bins = list(np.linspace(0, 160, 65))  # 2.5-point bins; shares above 160% go in the last
    palette = _colors(shares.shape[1], SERIES)
    for row, (color, (name, s)) in enumerate(zip(palette, shares.items(), strict=True)):
        pct = s.dropna() * 100
        ax.hist(pct.clip(upper=bins[-1]), bins=bins, color=color, alpha=0.55, label=name)
        ax.axvline(pct.median(), color=color, linewidth=1.5)
        side = (-4, "right") if row == 0 else (4, "left")  # first label left of its line
        ax.annotate(f"median {pct.median():.0f}%", (pct.median(), 1), color=color,
                    xycoords=("data", "axes fraction"), xytext=(side[0], -14),
                    textcoords="offset points", ha=side[1])  # fmt: skip
    ax.axvline(100, color=MUTED, linewidth=1, linestyle="--")  # 100%: all of the variance
    ax.set(title="How much of close-to-close variance each measure captures",
           xlabel="Share of close-to-close variance, per company (%)",
           ylabel="Companies")  # fmt: skip
    ax.legend(frameon=False, loc="center left")
    return _save(fig, path)


def stress_weeks(
    counts: pd.DataFrame, spans: dict[str, tuple[int, int]], path: Path | None = None
) -> plt.Figure:
    """Stress weeks per year for each VIX threshold, with labeled spans (tuning, test) shaded."""
    fig, ax = plt.subplots()
    width = 0.8 / counts.shape[1]
    palette = _colors(counts.shape[1], SERIES)
    for i, (color, (name, s)) in enumerate(zip(palette, counts.items(), strict=True)):
        ax.bar(s.index + (i - (counts.shape[1] - 1) / 2) * width, s.to_numpy(), width=width,
               color=color, label=name)  # fmt: skip
    for label, (first, last) in spans.items():
        ax.axvspan(first - 0.5, last + 0.5, color=GRID, alpha=0.6, zorder=0)
        ax.annotate(label, ((first + last) / 2, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -14), textcoords="offset points", ha="center",
                    color=MUTED)  # fmt: skip
    ax.set(title="Stress weeks per year (VIX at the end of the week)", xlabel="Year",
           ylabel="Weeks")  # fmt: skip
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


# Milestone 3: EDA and baseline figures.


def target_scales(target: pd.Series, path: Path | None = None) -> plt.Figure:
    """The response on the variance scale and on the log scale, with each one's skewness."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label, top in [
        (axes[0], np.exp(target), "weekly variance", 0.995),
        (axes[1], target, "log weekly variance", 1.0),
    ]:
        v = values.dropna()
        ax.hist(v.clip(upper=v.quantile(top)), bins=100, color=SERIES[0], edgecolor="white",
                linewidth=0.2)  # fmt: skip
        ax.set(title=f"{label}: skewness {v.skew():.1f}", xlabel=label, ylabel="Stock-weeks")
    axes[0].annotate("clipped at the 99.5th percentile", (0.98, 0.9), xycoords="axes fraction",
                     ha="right", color=MUTED, fontsize=10)  # fmt: skip
    fig.suptitle("The response is right-skewed as a variance and close to normal as a log")
    return _save(fig, path)


def variance_decomposition(shares: pd.Series, path: Path | None = None) -> plt.Figure:
    """Share of the target's variance explained by stock effects, week effects and both."""
    fig, ax = plt.subplots(figsize=(7, 3))
    s = shares.iloc[::-1]  # first row on top
    bars = ax.barh(s.index, s.to_numpy() * 100, color=SERIES[0], height=0.5)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in s], padding=4)
    ax.set(title="Where the response varies (2005-2015)", xlabel="Variance explained (%)",
           xlim=(0, 100))  # fmt: skip
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def binned(means: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Mean target by ventile of each feature, one line per feature, with the 45-degree line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _lines_vs_45(ax, dict(tuple(means.groupby("feature", sort=False))), [*SERIES, MUTED])
    ax.set(title="Next week's log variance by ventile of each lag", xlabel="Lag (log variance)",
           ylabel="Mean next-week log variance")  # fmt: skip
    return _save(fig, path)


def corr_heatmap(corr: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    """A correlation matrix as a heatmap on [-1, 1], labeled on both axes."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.5, 0.42 * n + 1.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=10)
    ax.set_yticks(range(n), corr.index, fontsize=10)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Correlation")
    ax.set_title(title)
    return _save(fig, path)


def ic_bars(raw: pd.Series, resid: pd.Series, path: Path | None = None) -> plt.Figure:
    """Mean rank IC of the top factors, against the target (left) and against HAR's miss (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    for ax, ic, title in [(axes[0], raw, "against next week's log variance"),
                          (axes[1], resid, "against per-stock HAR's miss")]:  # fmt: skip
        s = ic.iloc[::-1]
        ax.barh(s.index, s.to_numpy(), color=[SERIES[1] if v > 0 else SERIES[0] for v in s],
                height=0.65)  # fmt: skip
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set(title=title, xlabel="Mean weekly rank IC")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Rank IC of the top factors, 2005-2015 (orange: positive, blue: negative)")
    return _save(fig, path)


def ic_heatmap(ic: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Mean rank IC by row (a year or a regime) for each factor (columns)."""
    fig, ax = plt.subplots(figsize=(0.45 * ic.shape[1] + 2.2, 0.32 * ic.shape[0] + 1.8))
    lim = float(np.nanmax(np.abs(ic.to_numpy())))
    im = ax.imshow(ic.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(ic.shape[1]), ic.columns, rotation=60, ha="right", fontsize=11)
    ax.set_yticks(range(ic.shape[0]), ic.index, fontsize=11)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean rank IC vs HAR's miss")
    ax.set_title("Each factor's IC by year and regime")
    return _save(fig, path)


def common_miss(weekly: pd.DataFrame, by_flag: pd.Series, path: Path | None = None) -> plt.Figure:
    """Left: weekly mean HAR miss against the week's change in log VIX. Right: mean miss by flag."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.scatter(weekly["d_log_vix"], weekly["mean miss"], s=10, color=SERIES[0], alpha=0.6)
    ax.axhline(0, color=MUTED, linewidth=1)
    r = weekly[["d_log_vix", "mean miss"]].corr().iloc[0, 1]
    ax.set(title=f"The week's common miss tracks VIX (r = {r:.2f})",
           xlabel="This week's change in log VIX", ylabel="Mean HAR miss next week")  # fmt: skip
    ax = axes[1]
    bars = ax.bar(by_flag.index, by_flag.to_numpy(), color=[SERIES[0], SERIES[1]], width=0.5)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in by_flag], padding=3)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(title="Earnings expected next week?", ylabel="Mean HAR miss (log variance)")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def model_scores(scores: pd.DataFrame, loss: str, benchmark: str,
                 path: Path | None = None) -> plt.Figure:  # fmt: skip
    """Change in mean loss against the benchmark per model; hatched bars are not significant."""
    s = scores[f"{loss} vs {benchmark}"].drop(benchmark).sort_values(ascending=False) * 100
    p = scores.loc[s.index, f"p {loss}"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.5))
    colors = [SERIES[0] if v < 0 else SERIES[1] for v in s]
    bars = ax.barh(s.index, s.to_numpy(), color=colors, height=0.6)
    for bar, q in zip(bars, p, strict=True):
        bar.set_hatch("" if q < 0.05 else "//")
    ax.bar_label(bars, labels=[f"{v:+.1f}%" for v in s], padding=4, fontsize=10)
    ax.axvline(0, color=MUTED, linewidth=1)
    lo, hi = min(s.min(), 0), max(s.max(), 0)
    pad = 0.16 * (hi - lo)  # room for the value labels on both sides
    ax.set(xlim=(lo - pad, hi + pad), xlabel=f"Mean {loss.upper()} vs {benchmark} (%)",
           title=f"Validation 2016-2018: {loss.upper()} against {benchmark}")  # fmt: skip
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def cumulative_gain(weekly: pd.DataFrame, benchmark: str, path: Path | None = None) -> plt.Figure:
    """Running sum over weeks of each model's weekly mean QLIKE minus the benchmark's.

    weekly: one column per model, one row per week. A line that falls steadily means a steady
    gain; a step means the gain came from a few weeks.
    """
    gap = weekly.drop(columns=benchmark).sub(weekly[benchmark], axis=0).cumsum()
    fig, ax = plt.subplots(figsize=(9, 4))
    for color, (name, s) in zip(_colors(gap.shape[1], SERIES), gap.items(), strict=True):
        ax.plot(s.index, s.to_numpy(), color=color, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(title=f"Cumulative QLIKE gap to {benchmark}, validation 2016-2018", xlabel="Week",
           ylabel="Running sum of weekly gap")  # fmt: skip
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, path)


def ablation(steps: pd.Series, path: Path | None = None) -> plt.Figure:
    """Change in QLIKE against the benchmark as feature blocks are added one at a time."""
    s = steps * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(s)), s.to_numpy(), color=SERIES[0], marker="o")
    for i, v in enumerate(s):
        ax.annotate(f"{v:+.1f}%", (i, v), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=10)  # fmt: skip
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(s)), s.index, rotation=30, ha="right")
    lo, hi = min(s.min(), 0), max(s.max(), 0)
    pad = 0.15 * (hi - lo) + 1  # room for the value labels
    ax.set(
        title="Ridge: adding one feature block at a time (validation 2016-2018)",
        ylabel="Mean QLIKE vs per-stock HAR (%)",
        ylim=(lo - pad, hi + pad),
    )
    return _save(fig, path)


def loss_by_group(losses: pd.DataFrame, title: str, xlabel: str,
                  path: Path | None = None) -> plt.Figure:  # fmt: skip
    """Mean loss per group (rows) for each model (columns), as grouped bars."""
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / losses.shape[1]
    x = np.arange(len(losses))
    palette = _colors(losses.shape[1], [MUTED, *SERIES])
    for i, (color, (name, s)) in enumerate(zip(palette, losses.items(), strict=True)):
        ax.bar(x + (i - (losses.shape[1] - 1) / 2) * width, s.to_numpy(), width=width, color=color,
               label=name)  # fmt: skip
    ax.set_xticks(x, [str(i) for i in losses.index])
    ax.set(title=title, xlabel=xlabel, ylabel="Mean QLIKE", ylim=(0, losses.max().max() * 1.3))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=losses.shape[1])
    return _save(fig, path)


def calibration(frames: dict[str, pd.DataFrame], path: Path | None = None) -> plt.Figure:
    """Mean realized against mean forecast log variance by ventile of the forecast, per model."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _lines_vs_45(ax, frames, [MUTED, *SERIES])
    ax.set(title="Calibration, validation 2016-2018", xlabel="Mean forecast (log variance)",
           ylabel="Mean realized (log variance)")  # fmt: skip
    return _save(fig, path)
