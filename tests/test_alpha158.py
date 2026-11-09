"""Ported from FToken's tests: composition, no lookahead, blowups, and qlib's IdxMax convention."""

import numpy as np
import pandas as pd
import pytest

from volforecast.alpha158 import ALPHA158_WINDOWS, alpha158, alpha158_names

FIELDS = ["open", "high", "low", "close", "volume"]


@pytest.fixture
def ohlcv(daily) -> dict[str, pd.DataFrame]:
    wide = daily.pivot(index="date", columns="ticker", values=FIELDS)
    return {f: wide[f] for f in FIELDS}


def test_factor_set_is_158_unique_names():
    names = alpha158_names()
    assert len(names) == len(set(names)) == 158


def test_no_lookahead(ohlcv):
    """A factor value on date t must not change when later dates are appended."""
    full, _ = alpha158(ohlcv)
    truncated, _ = alpha158({k: v.iloc[:-10] for k, v in ohlcv.items()})
    warmup = max(ALPHA158_WINDOWS)
    a, b = full[warmup : truncated.shape[0]], truncated[warmup:]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.99
    np.testing.assert_allclose(a[both], b[both], rtol=1e-5, atol=1e-6)


def test_division_blowups_become_nan_not_infinity(ohlcv):
    broken = dict(ohlcv, open=ohlcv["open"].copy())
    broken["open"].iloc[5:9] = 0.0
    values, names = alpha158(broken)
    assert not np.isinf(values).any()
    assert np.isnan(values[5:9, :, names.index("KMID")]).all()


def test_imax_counts_from_one_like_qlib(ohlcv):
    """With a rising high the latest bar is the window high: qlib's IdxMax gives d, so 1.0."""
    rising = dict(ohlcv, high=ohlcv["high"] * 0 + np.arange(len(ohlcv["high"]))[:, None] + 100.0)
    values, names = alpha158(rising)
    assert np.nanmax(values[:, :, names.index("IMAX5")]) == 1.0
    assert np.nanmin(values[10:, :, names.index("IMAX5")]) == 1.0
