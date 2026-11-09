"""Alpha158 recomputed from OHLCV, ported from FToken (`Q9gJYx/FToken`).

`alpha158` reimplements qlib's published Alpha158 expressions on wide price frames with no qlib
dependency: 9 candlestick ratios, 4 price levels relative to close, and 29 rolling operators at
five window lengths, 158 columns in all. Every operator looks strictly backward, so a value on
date t never moves when later dates arrive. One change from FToken: `IMAX`/`IMIN` count
positions from 1, as qlib's `IdxMax`/`IdxMin` do. `features.alpha158_weekly` samples it weekly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ALPHA158_WINDOWS", "alpha158", "alpha158_names"]

ALPHA158_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)
_EPS = 1e-12

_ROLLING_OPS: tuple[str, ...] = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
_KBAR_NAMES: tuple[str, ...] = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
)
_PRICE_NAMES: tuple[str, ...] = ("OPEN0", "HIGH0", "LOW0", "VWAP0")


def alpha158_names() -> list[str]:
    """The 158 factor names, in the order :func:`alpha158` returns them."""
    names = list(_KBAR_NAMES) + list(_PRICE_NAMES)
    names += [f"{op}{d}" for op in _ROLLING_OPS for d in ALPHA158_WINDOWS]
    return names


def _rolling_arg_extreme(frame: pd.DataFrame, window: int, largest: bool) -> pd.DataFrame:
    """Position of the window's extreme value, 1 for the oldest observation (qlib's `IdxMax`)."""
    values = frame.to_numpy(dtype=np.float64)
    out = np.full(values.shape, np.nan)
    if values.shape[0] >= window:
        # Sentinel-fill so argmax never sees an all-NaN slice, then restore those windows.
        sentinel = -np.inf if largest else np.inf
        filled = np.where(np.isnan(values), sentinel, values)
        windows = np.lib.stride_tricks.sliding_window_view(filled, window, axis=0)
        picked = (np.argmax if largest else np.argmin)(windows, axis=-1).astype(np.float64) + 1
        empty = np.lib.stride_tricks.sliding_window_view(np.isnan(values), window, axis=0).all(
            axis=-1
        )
        picked[empty] = np.nan
        out[window - 1 :] = picked
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _rolling_regression(frame: pd.DataFrame, window: int) -> tuple[pd.DataFrame, ...]:
    """Slope, R-squared, and current-point residual of a rolling OLS fit against time."""
    n = float(window)
    sx = window * (window - 1) / 2.0
    sxx = (window - 1) * window * (2 * window - 1) / 6.0
    roll = frame.rolling(window)
    sy = roll.sum()
    syy = (frame**2).rolling(window).sum()

    # sum_j j * y_{t-window+1+j}, written as shifts so it stays vectorized across columns.
    sxy = sum((window - 1 - k) * frame.shift(k) for k in range(window))

    denom = n * sxx - sx**2
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    ss_tot = syy - sy**2 / n
    ss_reg = slope**2 * (sxx - sx**2 / n)
    rsquare = (ss_reg / ss_tot.where(ss_tot.abs() > _EPS)).clip(0.0, 1.0)
    residual = frame - (intercept + slope * (window - 1))
    return slope, rsquare, residual


def alpha158(ohlcv: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    """Compute the 158 factors from wide OHLCV frames.

    The window lengths are fixed at :data:`ALPHA158_WINDOWS` because the factor set is a
    published definition rather than a tunable feature space; 158 is the count those five
    windows produce.

    Args:
        ohlcv: mapping with keys ``open``, ``high``, ``low``, ``close``, ``volume`` and
            optionally ``vwap``, each a ``[T, N]`` frame indexed by date with instruments as
            columns. When ``vwap`` is absent the typical price ``(high + low + close) / 3``
            stands in for it, because Yahoo does not distribute a volume-weighted average price.

    Returns:
        ``(values, names)`` where ``values`` is ``[T, N, 158]`` float32 with ``NaN`` during
        each operator's warmup, and ``names`` matches :func:`alpha158_names`.
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [k for k in required if k not in ohlcv]
    if missing:
        raise ValueError(f"missing price fields: {missing}")

    open_, high, low = ohlcv["open"], ohlcv["high"], ohlcv["low"]
    close, volume = ohlcv["close"], ohlcv["volume"]
    vwap = ohlcv.get("vwap", (high + low + close) / 3.0)

    shape = close.shape
    for name, frame in ohlcv.items():
        if frame.shape != shape:
            raise ValueError(f"field {name!r} has shape {frame.shape}, expected {shape}")

    bar = (high - low).abs() + _EPS
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)

    features: dict[str, pd.DataFrame] = {
        "KMID": (close - open_) / open_,
        "KLEN": (high - low) / open_,
        "KMID2": (close - open_) / bar,
        "KUP": (high - upper) / open_,
        "KUP2": (high - upper) / bar,
        "KLOW": (lower - low) / open_,
        "KLOW2": (lower - low) / bar,
        "KSFT": (2 * close - high - low) / open_,
        "KSFT2": (2 * close - high - low) / bar,
        "OPEN0": open_ / close,
        "HIGH0": high / close,
        "LOW0": low / close,
        "VWAP0": vwap / close,
    }

    prev_close = close.shift(1)
    close_change = close - prev_close
    abs_change = close_change.abs()
    gain = close_change.clip(lower=0.0)
    loss = (-close_change).clip(lower=0.0)
    up_day = (close > prev_close).astype(float).where(prev_close.notna())
    down_day = (close < prev_close).astype(float).where(prev_close.notna())

    prev_volume = volume.shift(1)
    volume_change = volume - prev_volume
    abs_volume_change = volume_change.abs()
    volume_gain = volume_change.clip(lower=0.0)
    volume_loss = (-volume_change).clip(lower=0.0)

    log_volume = np.log(volume + 1.0)
    log_volume_change = np.log(volume / prev_volume + 1.0)
    close_ratio = close / prev_close
    flow = (close_ratio - 1.0).abs() * volume

    for d in ALPHA158_WINDOWS:
        roll_close = close.rolling(d)
        window_high = high.rolling(d).max()
        window_low = low.rolling(d).min()
        slope, rsquare, residual = _rolling_regression(close, d)

        features[f"ROC{d}"] = close.shift(d) / close
        features[f"MA{d}"] = roll_close.mean() / close
        features[f"STD{d}"] = roll_close.std() / close
        features[f"BETA{d}"] = slope / close
        features[f"RSQR{d}"] = rsquare
        features[f"RESI{d}"] = residual / close
        features[f"MAX{d}"] = window_high / close
        features[f"MIN{d}"] = window_low / close
        features[f"QTLU{d}"] = roll_close.quantile(0.8) / close
        features[f"QTLD{d}"] = roll_close.quantile(0.2) / close
        features[f"RANK{d}"] = roll_close.rank(pct=True)
        features[f"RSV{d}"] = (close - window_low) / (window_high - window_low + _EPS)
        features[f"IMAX{d}"] = _rolling_arg_extreme(high, d, largest=True) / d
        features[f"IMIN{d}"] = _rolling_arg_extreme(low, d, largest=False) / d
        features[f"IMXD{d}"] = features[f"IMAX{d}"] - features[f"IMIN{d}"]
        features[f"CORR{d}"] = close.rolling(d).corr(log_volume)
        features[f"CORD{d}"] = close_ratio.rolling(d).corr(log_volume_change)
        features[f"CNTP{d}"] = up_day.rolling(d).mean()
        features[f"CNTN{d}"] = down_day.rolling(d).mean()
        features[f"CNTD{d}"] = features[f"CNTP{d}"] - features[f"CNTN{d}"]

        abs_sum = abs_change.rolling(d).sum() + _EPS
        gain_sum = gain.rolling(d).sum()
        loss_sum = loss.rolling(d).sum()
        features[f"SUMP{d}"] = gain_sum / abs_sum
        features[f"SUMN{d}"] = loss_sum / abs_sum
        features[f"SUMD{d}"] = (gain_sum - loss_sum) / abs_sum

        features[f"VMA{d}"] = volume.rolling(d).mean() / (volume + _EPS)
        features[f"VSTD{d}"] = volume.rolling(d).std() / (volume + _EPS)
        features[f"WVMA{d}"] = flow.rolling(d).std() / (flow.rolling(d).mean() + _EPS)

        volume_abs_sum = abs_volume_change.rolling(d).sum() + _EPS
        volume_gain_sum = volume_gain.rolling(d).sum()
        volume_loss_sum = volume_loss.rolling(d).sum()
        features[f"VSUMP{d}"] = volume_gain_sum / volume_abs_sum
        features[f"VSUMN{d}"] = volume_loss_sum / volume_abs_sum
        features[f"VSUMD{d}"] = (volume_gain_sum - volume_loss_sum) / volume_abs_sum

    names = alpha158_names()
    # Fill in place while dropping each float64 frame as it is consumed: materializing all 158
    # of them as a list alongside the dict would roughly double peak memory on a full universe.
    stacked = np.empty((*close.shape, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        stacked[:, :, i] = features.pop(name).to_numpy(dtype=np.float32, na_value=np.nan)
    stacked[~np.isfinite(stacked)] = np.nan
    return stacked, names
