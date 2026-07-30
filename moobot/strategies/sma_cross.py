"""Trend-following: fast/slow moving-average crossover with a trend filter."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, crossed_above, crossed_below, ema, sma
from .base import Action, Signal, Strategy, register


@register
class SmaCross(Strategy):
    """Buy when the fast average crosses above the slow one, exit on the reverse.

    An optional long-term trend filter suppresses longs below the trend average,
    which is what stops this strategy bleeding out in downtrends.
    """

    name = "sma_cross"

    def configure(
        self,
        fast: int = 20,
        slow: int = 50,
        trend: int = 200,
        use_ema: bool = False,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.0,
        **_: object,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"sma_cross needs fast < slow, got fast={fast} slow={slow}")
        self.fast, self.slow, self.trend = fast, slow, trend
        self.ma = ema if use_ema else sma
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = max(slow, trend if trend else 0, atr_period) + 5

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        f = bars.copy()
        f["ma_fast"] = self.ma(f["close"], self.fast)
        f["ma_slow"] = self.ma(f["close"], self.slow)
        f["ma_trend"] = self.ma(f["close"], self.trend) if self.trend else 0.0
        f["atr"] = atr(f["high"], f["low"], f["close"], self.atr_period)
        f["cross_up"] = crossed_above(f["ma_fast"], f["ma_slow"])
        f["cross_dn"] = crossed_below(f["ma_fast"], f["ma_slow"])
        return f

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        row = frame.iloc[i]
        if pd.isna(row["ma_slow"]):
            return Signal(reason="indicators not ready")

        if bool(row["cross_dn"]):
            return Signal(
                Action.SELL,
                reason=f"MA{self.fast} crossed below MA{self.slow}",
            )

        if bool(row["cross_up"]):
            if self.trend and not pd.isna(row["ma_trend"]) and row["close"] < row["ma_trend"]:
                return Signal(reason=f"cross up ignored: price below MA{self.trend} trend filter")
            stop = None
            if not pd.isna(row["atr"]) and self.atr_stop_multiple > 0:
                stop = float(row["close"] - self.atr_stop_multiple * row["atr"])
            # Wider separation of the averages = stronger trend = bigger size.
            spread = abs(row["ma_fast"] - row["ma_slow"]) / max(row["close"], 1e-9)
            return Signal(
                Action.BUY,
                strength=min(1.0, 0.5 + spread * 40),
                reason=f"MA{self.fast} crossed above MA{self.slow}",
                stop_price=stop,
            )

        return Signal(reason="no cross")
