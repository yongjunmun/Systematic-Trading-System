"""Technical indicators.

Everything is implemented on pandas Series so the same functions are used by
both the backtester and the live engine - a signal can never disagree between
the two because of a different indicator implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (standard 2/(n+1) smoothing)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing - the 1/n variant used by RSI, ATR and ADX."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_ema(gain, period)
    avg_loss = wilder_ema(loss, period)
    # avg_loss == 0 means an unbroken run of up bars -> RSI is 100 by definition.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return wilder_ema(true_range(high, low, close), period)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    mid = sma(close, period)
    # ddof=0 matches the population stdev convention used by most charting tools.
    dev = close.rolling(window=period, min_periods=period).std(ddof=0)
    return mid + num_std * dev, mid, mid - num_std * dev


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel over the *previous* `period` bars.

    Shifted by one bar so a breakout test never peeks at the bar it is
    evaluating - that lookahead bug is the classic way to fake a good backtest.
    """
    upper = high.rolling(window=period, min_periods=period).max().shift(1)
    lower = low.rolling(window=period, min_periods=period).min().shift(1)
    return upper, lower


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of a series."""
    mean = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def realised_volatility(close: pd.Series, period: int = 20, bars_per_year: int = 252) -> pd.Series:
    """Annualised standard deviation of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=period, min_periods=period).std(ddof=0) * np.sqrt(bars_per_year)


def crossed_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar where `fast` crosses from at-or-below to above `slow`."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossed_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar where `fast` crosses from at-or-above to below `slow`."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))
