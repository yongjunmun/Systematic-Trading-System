"""Strategy package. Importing it registers every built-in strategy."""

from __future__ import annotations

from .base import Action, Signal, Strategy, available, get_strategy, register
from .donchian_breakout import DonchianBreakout
from .macd_momentum import MacdMomentum
from .rsi_reversion import RsiReversion
from .sma_cross import SmaCross

__all__ = [
    "Action",
    "Signal",
    "Strategy",
    "available",
    "get_strategy",
    "register",
    "DonchianBreakout",
    "MacdMomentum",
    "RsiReversion",
    "SmaCross",
]
