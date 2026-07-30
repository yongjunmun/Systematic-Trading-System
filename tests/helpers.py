"""Shared helpers for building synthetic bar data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(
    closes: list[float] | np.ndarray,
    start: str = "2024-01-02 09:30:00",
    freq: str = "1D",
    volume: float | list[float] = 1_000_000.0,
    spread: float = 0.005,
) -> pd.DataFrame:
    """Build an OHLCV frame from a close series.

    open is the previous close, and the high/low straddle the bar by `spread`,
    which is enough structure for indicators and stop logic to be meaningful.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    vols = np.full(n, volume, dtype=float) if np.isscalar(volume) else np.asarray(volume, float)
    return pd.DataFrame(
        {
            "time_key": pd.date_range(start=start, periods=n, freq=freq),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def trending(n: int = 300, start_price: float = 100.0, drift: float = 0.002,
             noise: float = 0.0, seed: int = 7) -> np.ndarray:
    """Geometric drift with optional reproducible noise."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, noise, n) if noise else np.zeros(n)
    return start_price * np.cumprod(1 + drift + shocks)


def v_shape(down: int = 60, up: int = 60, start_price: float = 100.0,
            rate: float = 0.01) -> np.ndarray:
    """A clean decline followed by a clean recovery - triggers reversion logic."""
    falling = start_price * np.cumprod(np.full(down, 1 - rate))
    rising = falling[-1] * np.cumprod(np.full(up, 1 + rate))
    return np.concatenate([falling, rising])
