"""Momentum: MACD histogram turning up while price holds its trend."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, crossed_above, crossed_below, ema, macd
from .base import Action, Signal, Strategy, register


@register
class MacdMomentum(Strategy):
    """Buy MACD line crossing above its signal line above the trend EMA."""

    name = "macd_momentum"

    def configure(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        trend: int = 100,
        require_positive: bool = False,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.0,
        **_: object,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"macd_momentum needs fast < slow, got {fast} >= {slow}")
        self.fast, self.slow, self.signal_period = fast, slow, signal
        self.trend = trend
        self.require_positive = require_positive
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = max(slow + signal, trend if trend else 0, atr_period) + 5

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        f = bars.copy()
        line, sig, hist = macd(f["close"], self.fast, self.slow, self.signal_period)
        f["macd"], f["macd_signal"], f["macd_hist"] = line, sig, hist
        f["ma_trend"] = ema(f["close"], self.trend) if self.trend else 0.0
        f["atr"] = atr(f["high"], f["low"], f["close"], self.atr_period)
        f["cross_up"] = crossed_above(line, sig)
        f["cross_dn"] = crossed_below(line, sig)
        return f

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        row = frame.iloc[i]
        if pd.isna(row["macd_signal"]):
            return Signal(reason="MACD not ready")

        if bool(row["cross_dn"]):
            return Signal(Action.SELL, reason="MACD crossed below its signal line")

        if bool(row["cross_up"]):
            if self.require_positive and row["macd"] < 0:
                return Signal(reason="MACD cross up but still below zero")
            if self.trend and not pd.isna(row["ma_trend"]) and row["close"] < row["ma_trend"]:
                return Signal(reason=f"MACD cross up below EMA{self.trend} trend filter")
            stop = None
            if not pd.isna(row["atr"]) and self.atr_stop_multiple > 0:
                stop = float(row["close"] - self.atr_stop_multiple * row["atr"])
            hist_scale = abs(row["macd_hist"]) / max(row["close"] * 0.01, 1e-9)
            return Signal(
                Action.BUY,
                strength=min(1.0, 0.5 + hist_scale),
                reason="MACD crossed above its signal line",
                stop_price=stop,
            )

        return Signal(reason="no MACD cross")
