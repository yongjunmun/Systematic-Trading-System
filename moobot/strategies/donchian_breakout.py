"""Breakout: Donchian channel with a volume confirmation filter."""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, donchian, sma
from .base import Action, Signal, Strategy, register


@register
class DonchianBreakout(Strategy):
    """Buy a break of the N-bar high, exit on a break of the M-bar low.

    The channel is shifted one bar back (see indicators.donchian) so the
    breakout bar itself is never part of the range it is breaking.
    """

    name = "donchian_breakout"

    def configure(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        volume_period: int = 20,
        min_volume_ratio: float = 1.2,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.0,
        **_: object,
    ) -> None:
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.volume_period = volume_period
        self.min_volume_ratio = min_volume_ratio
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = max(entry_period, exit_period, volume_period, atr_period) + 5

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        f = bars.copy()
        f["chan_high"], _ = donchian(f["high"], f["low"], self.entry_period)
        _, f["chan_low"] = donchian(f["high"], f["low"], self.exit_period)
        f["vol_avg"] = sma(f["volume"], self.volume_period)
        f["atr"] = atr(f["high"], f["low"], f["close"], self.atr_period)
        return f

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        row = frame.iloc[i]
        if pd.isna(row["chan_high"]) or pd.isna(row["chan_low"]):
            return Signal(reason="channel not ready")

        if row["close"] < row["chan_low"]:
            return Signal(
                Action.SELL,
                reason=f"broke {self.exit_period}-bar low {row['chan_low']:.2f}",
            )

        if row["close"] > row["chan_high"]:
            vol_avg = row["vol_avg"]
            ratio = row["volume"] / vol_avg if vol_avg and not pd.isna(vol_avg) and vol_avg > 0 else 0.0
            if self.min_volume_ratio > 0 and ratio < self.min_volume_ratio:
                return Signal(
                    reason=f"breakout on weak volume ({ratio:.2f}x < {self.min_volume_ratio}x)"
                )
            stop = None
            if not pd.isna(row["atr"]) and self.atr_stop_multiple > 0:
                stop = float(row["close"] - self.atr_stop_multiple * row["atr"])
            return Signal(
                Action.BUY,
                strength=min(1.0, 0.5 + max(0.0, ratio - 1.0) / 2),
                reason=f"broke {self.entry_period}-bar high {row['chan_high']:.2f} on {ratio:.2f}x volume",
                stop_price=stop,
            )

        return Signal(reason="inside channel")
