"""Intraday session gates.

Two things kill unattended intraday bots: entering into the opening auction
where spreads are widest and prices mean nothing, and holding into the close so
an overnight gap decides the trade instead of the strategy. This module blocks
both.

All comparisons happen in the exchange's own timezone, so the bot behaves the
same whether the machine running it sits in Hong Kong or California.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

from .settings import SessionConfig, parse_hhmm


class Phase(str, Enum):
    CLOSED = "closed"            # weekend or outside the trading day
    WARMUP = "warmup"            # market open but too early to enter
    OPEN = "open"                # entries and management allowed
    MANAGE_ONLY = "manage_only"  # too late to enter, still managing exits
    FORCE_CLOSE = "force_close"  # flatten everything now

    @property
    def allows_entry(self) -> bool:
        return self is Phase.OPEN

    @property
    def allows_management(self) -> bool:
        return self in (Phase.WARMUP, Phase.OPEN, Phase.MANAGE_ONLY, Phase.FORCE_CLOSE)


@dataclass
class SessionStatus:
    phase: Phase
    local_time: str
    timezone: str
    reason: str

    def __str__(self) -> str:
        return f"{self.phase.value} ({self.local_time} {self.timezone}): {self.reason}"


class SessionGate:
    """Answers 'may the bot act right now?' from the [session] settings."""

    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        self.zone = ZoneInfo(cfg.timezone) if cfg.enabled else None
        if cfg.enabled:
            self.earliest = parse_hhmm(cfg.earliest_entry, "[session].earliest_entry")
            self.latest = parse_hhmm(cfg.latest_entry, "[session].latest_entry")
            self.force = parse_hhmm(cfg.force_close, "[session].force_close")

    def status(self, now: datetime | None = None) -> SessionStatus:
        if not self.cfg.enabled:
            return SessionStatus(Phase.OPEN, "-", "-", "session gating disabled")

        local = (now or datetime.now(tz=self.zone)).astimezone(self.zone)
        stamp = local.strftime("%H:%M:%S")
        clock = local.time()

        if local.weekday() >= 5:
            return SessionStatus(Phase.CLOSED, stamp, self.cfg.timezone, "weekend")
        if clock >= self.force:
            # After force_close there is still a window before the bell; keep
            # flattening rather than falling straight through to CLOSED.
            return SessionStatus(
                Phase.FORCE_CLOSE, stamp, self.cfg.timezone,
                f"at or past force_close {self.cfg.force_close} - flatten everything",
            )
        if clock < self.earliest:
            return SessionStatus(
                Phase.WARMUP, stamp, self.cfg.timezone,
                f"before earliest_entry {self.cfg.earliest_entry} - no new positions",
            )
        if clock > self.latest:
            return SessionStatus(
                Phase.MANAGE_ONLY, stamp, self.cfg.timezone,
                f"past latest_entry {self.cfg.latest_entry} - managing existing positions only",
            )
        return SessionStatus(
            Phase.OPEN, stamp, self.cfg.timezone,
            f"inside the entry window {self.cfg.earliest_entry}-{self.cfg.latest_entry}",
        )
