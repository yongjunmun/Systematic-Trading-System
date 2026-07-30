"""Strategy interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"      # close a long (or open a short when shorts are allowed)
    HOLD = "HOLD"


@dataclass
class Signal:
    action: Action = Action.HOLD
    # 0.0-1.0. Scales the risk budget for the trade.
    strength: float = 1.0
    reason: str = ""
    # Optional absolute price levels the strategy wants to use instead of the
    # percentage defaults in [risk].
    stop_price: float | None = None
    target_price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, float(self.strength)))


class Strategy(ABC):
    """Base class for all strategies.

    A strategy is a pure function of the bar history. It must not place orders,
    read account state, or hold mutable per-symbol state - that keeps the
    backtest and the live engine provably consistent.
    """

    name = "base"
    # Bars required before the strategy can emit a non-HOLD signal.
    warmup = 50

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.configure(**params)

    def configure(self, **params: Any) -> None:
        """Hook for subclasses to read their parameters."""

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Attach indicator columns to a copy of `bars` and return it.

        `bars` has columns: time_key, open, high, low, close, volume.
        """

    @abstractmethod
    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        """Decide what to do at bar index `i` of a computed frame."""

    def latest_signal(self, bars: pd.DataFrame) -> Signal:
        """Signal for the most recent bar - what the live engine calls."""
        if len(bars) < self.warmup:
            return Signal(reason=f"warming up ({len(bars)}/{self.warmup} bars)")
        frame = self.compute(bars)
        return self.signal_at(frame, len(frame) - 1)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{type(self).__name__}({args})"


_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a strategy to the registry."""
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **params: Any) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)
