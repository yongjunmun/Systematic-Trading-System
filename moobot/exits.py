"""The exit ladder: scale out, move to breakeven, then trail.

Everything here is expressed in **R** - multiples of the initial risk taken on
the trade (entry price minus initial stop). Managing in R rather than in percent
is what makes exits comparable across a $4 stock and a $400 one, and it is what
lets a strategy with a sub-50% win rate still be profitable.

The same object drives the backtester and the live engine. The backtester feeds
it real OHLC bars; the live engine feeds it a single snapshot price as a
degenerate bar. Neither can drift from the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .settings import ExitsConfig


@dataclass
class Bar:
    """One price observation. Live snapshots use the same price for all four."""

    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_price(cls, price: float) -> Bar:
        return cls(price, price, price, price)


@dataclass
class Position:
    """Mutable state of one open long position."""

    entry_price: float
    initial_stop: float
    qty: int
    stop_price: float = 0.0
    target_price: float = 0.0  # absolute target, used by `simple` mode only
    high_water: float = 0.0
    partial_done: bool = False
    breakeven_done: bool = False
    realised_pnl: float = 0.0
    scale_outs: int = 0
    original_qty: int = 0

    def __post_init__(self) -> None:
        if self.stop_price <= 0:
            self.stop_price = self.initial_stop
        if self.high_water <= 0:
            self.high_water = self.entry_price
        if self.original_qty <= 0:
            self.original_qty = self.qty

    @property
    def risk_per_share(self) -> float:
        """The '1R' denominator. Fixed at entry - it never moves with the stop."""
        return max(self.entry_price - self.initial_stop, 1e-9)

    def r_at(self, price: float) -> float:
        return (price - self.entry_price) / self.risk_per_share

    def price_at_r(self, r: float) -> float:
        return self.entry_price + r * self.risk_per_share


@dataclass
class ExitDecision:
    """What to do with the position on this bar."""

    exit_qty: int = 0
    exit_price: float = 0.0
    reason: str = ""
    is_full_exit: bool = False
    new_stop: float | None = None
    events: list[str] = field(default_factory=list)

    @property
    def should_exit(self) -> bool:
        return self.exit_qty > 0


class ExitLadder:
    """Applies [exits] to an open position, one bar at a time."""

    def __init__(self, cfg: ExitsConfig) -> None:
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return self.cfg.mode == "ladder"

    def evaluate(
        self,
        pos: Position,
        bar: Bar,
        atr: float | None = None,
        swing_low: float | None = None,
    ) -> ExitDecision:
        """Decide this bar's action. Mutates `pos` state flags and the stop.

        Checks run worst-case-first: if a bar's range covers both the stop and a
        profit level, the stop is assumed to have been hit. Anything else
        flatters the backtest.
        """
        decision = ExitDecision()
        pos.high_water = max(pos.high_water, bar.high)

        # 1. Stop. A gap through it fills at the open, not at the stop price.
        if pos.stop_price > 0 and bar.low <= pos.stop_price:
            fill = min(pos.stop_price, bar.open) if bar.open < pos.stop_price else pos.stop_price
            decision.exit_qty = pos.qty
            decision.exit_price = fill
            decision.is_full_exit = True
            decision.reason = f"stop {pos.stop_price:.4f} ({pos.r_at(fill):+.2f}R)"
            return decision

        if self.enabled:
            # 2. Final target expressed in R.
            if self.cfg.target_r > 0:
                target = pos.price_at_r(self.cfg.target_r)
                if bar.high >= target:
                    decision.exit_qty = pos.qty
                    decision.exit_price = max(target, bar.open)
                    decision.is_full_exit = True
                    decision.reason = f"target {self.cfg.target_r:.2f}R at {target:.4f}"
                    return decision

            # 3. Partial scale-out. This and breakeven can both fire on one bar.
            if not pos.partial_done and self.cfg.partial_at_r > 0 and pos.qty > 1:
                level = pos.price_at_r(self.cfg.partial_at_r)
                if bar.high >= level:
                    qty = min(pos.qty - 1,
                              max(1, math.ceil(pos.qty * self.cfg.partial_fraction)))
                    decision.exit_qty = qty
                    decision.exit_price = max(level, bar.open)
                    decision.reason = f"partial {self.cfg.partial_at_r:.2f}R"
                    pos.partial_done = True
                    decision.events.append(
                        f"scaled out {qty}/{pos.qty} at {self.cfg.partial_at_r:.2f}R"
                    )

            # 4. Breakeven flip.
            if not pos.breakeven_done and self.cfg.breakeven_at_r > 0:
                level = pos.price_at_r(self.cfg.breakeven_at_r)
                if bar.high >= level:
                    pos.breakeven_done = True
                    if pos.entry_price > pos.stop_price:
                        decision.new_stop = pos.entry_price
                        decision.events.append(
                            f"stop to breakeven {pos.entry_price:.4f} "
                            f"at {self.cfg.breakeven_at_r:.2f}R"
                        )
        else:
            # `simple` mode: the flat target fixed at entry.
            if pos.target_price > 0 and bar.high >= pos.target_price:
                decision.exit_qty = pos.qty
                decision.exit_price = max(pos.target_price, bar.open)
                decision.is_full_exit = True
                decision.reason = f"target {pos.target_price:.4f}"
                return decision

        # 5. Trail. Ratchets up only - a stop that can move down is not a stop.
        trail = self._trail_candidate(pos, atr, swing_low)
        if trail is not None:
            best = max(trail, decision.new_stop or 0.0, pos.stop_price)
            if best > pos.stop_price:
                decision.new_stop = best
                decision.events.append(f"trail stop -> {best:.4f}")

        if decision.new_stop is not None:
            pos.stop_price = decision.new_stop
        return decision

    def _trail_candidate(
        self, pos: Position, atr: float | None, swing_low: float | None
    ) -> float | None:
        if self.cfg.trail_method == "none":
            return None
        # Waiting for breakeven before trailing stops the trail from strangling
        # a trade that has not yet proved itself. Only meaningful in ladder mode.
        if self.enabled and self.cfg.trail_only_after_breakeven and not pos.breakeven_done:
            return None

        if self.cfg.trail_method == "atr":
            if not atr or atr <= 0:
                return None
            return pos.high_water - self.cfg.trail_atr_multiple * atr
        if self.cfg.trail_method == "percent":
            return pos.high_water * (1 - self.cfg.trail_percent / 100.0)
        if self.cfg.trail_method == "swing_low":
            if swing_low is None or swing_low <= 0:
                return None
            # A tick below the swing so a retest does not stop you out.
            return swing_low * 0.999
        return None


def swing_lows(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Most recent *confirmed* swing low as of each bar.

    A swing low is a bar whose low is below the `left` bars before it and the
    `right` bars after it. Confirmation needs `right` future bars, so the value
    is shifted forward by `right` - without that shift this would be lookahead
    and the trailing stop would appear psychic in a backtest.
    """
    lows = low.to_numpy(dtype=float)
    n = len(lows)
    pivot = pd.Series([float("nan")] * n, index=low.index)
    for i in range(left, n - right):
        window_left = lows[i - left : i]
        window_right = lows[i + 1 : i + 1 + right]
        if lows[i] < window_left.min() and lows[i] < window_right.min():
            pivot.iloc[i] = lows[i]
    # Publish each pivot only once it could actually have been observed.
    return pivot.shift(right).ffill()
