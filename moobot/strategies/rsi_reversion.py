"""Mean reversion: buy oversold RSI, exit when it recovers."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, rsi, sma
from .base import Action, Signal, Strategy, register


@register
class RsiReversion(Strategy):
    """Buy dips inside an uptrend, sell back into strength.

    Mean reversion works badly against a downtrend, so entries require price
    above a long moving average unless the trend filter is disabled.
    """

    name = "rsi_reversion"

    def configure(
        self,
        period: int = 14,
        oversold: float = 30.0,
        exit_level: float = 55.0,
        overbought: float = 75.0,
        trend: int = 100,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.5,
        **_: object,
    ) -> None:
        if not 0 < oversold < exit_level < 100:
            raise ValueError("rsi_reversion needs 0 < oversold < exit_level < 100")
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.overbought = overbought
        self.trend = trend
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = max(period * 3, trend if trend else 0, atr_period) + 5

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        f = bars.copy()
        f["rsi"] = rsi(f["close"], self.period)
        f["ma_trend"] = sma(f["close"], self.trend) if self.trend else 0.0
        f["atr"] = atr(f["high"], f["low"], f["close"], self.atr_period)
        return f

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        row = frame.iloc[i]
        value = row["rsi"]
        if pd.isna(value):
            return Signal(reason="RSI not ready")
        prev = frame["rsi"].iloc[i - 1] if i > 0 else float("nan")

        if value >= self.overbought:
            return Signal(Action.SELL, reason=f"RSI {value:.1f} >= overbought {self.overbought}")
        # Exit on the bar RSI recovers through the exit level.
        if not pd.isna(prev) and prev < self.exit_level <= value:
            return Signal(Action.SELL, reason=f"RSI recovered to {value:.1f}")

        if value <= self.oversold:
            if self.trend and not pd.isna(row["ma_trend"]) and row["close"] < row["ma_trend"]:
                return Signal(reason=f"oversold but below MA{self.trend} - not catching a knife")
            stop = None
            if not pd.isna(row["atr"]) and self.atr_stop_multiple > 0:
                stop = float(row["close"] - self.atr_stop_multiple * row["atr"])
            depth = (self.oversold - value) / max(self.oversold, 1e-9)
            return Signal(
                Action.BUY,
                strength=min(1.0, 0.5 + depth * 2),
                reason=f"RSI {value:.1f} <= oversold {self.oversold}",
                stop_price=stop,
            )

        return Signal(reason=f"RSI {value:.1f} neutral")
