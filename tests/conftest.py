"""Shared fixtures: a small synthetic stock-day panel shaped like the cleaned Yahoo table."""

import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def make_daily(n_days: int = 260, tickers: list[str] = TICKERS, seed: int = 0) -> pd.DataFrame:
    """Geometric random walks with consistent OHLC bars and positive volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for ticker in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        open_ = close * np.exp(rng.normal(0, 0.01, n_days))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n_days)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n_days)))
        bars = {"open": open_, "high": high, "low": low, "close": close, "adj_close": close}
        volume = rng.integers(100_000, 10_000_000, n_days)
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, **bars, "volume": volume}))
    return pd.concat(frames, ignore_index=True).astype({"volume": float})


@pytest.fixture
def daily() -> pd.DataFrame:
    return make_daily()


@pytest.fixture
def membership() -> pd.DataFrame:
    """AAA and BBB throughout; CCC leaves mid-sample; DDD is never a member."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-06-01"] * 2),
        "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB"],
    })  # fmt: skip
