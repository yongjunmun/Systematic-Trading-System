"""Position sizing and the safety limits that stop a bad day becoming a bad week."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .settings import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class SizingResult:
    qty: int
    stop_price: float
    target_price: float
    notional: float
    reason: str
    vol_scale: float = 1.0

    @property
    def ok(self) -> bool:
        return self.qty > 0


@dataclass
class RiskState:
    """Mutable per-session risk tracking."""

    day: date = field(default_factory=date.today)
    start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    entries_today: int = 0

    def roll_day(self, equity: float) -> None:
        today = date.today()
        if today != self.day or self.start_equity <= 0:
            self.day = today
            self.start_equity = equity
            self.peak_equity = equity
            self.halted = False
            self.halt_reason = ""
            self.entries_today = 0


class RiskManager:
    """Applies the [risk] section of the config to every proposed trade."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.state = RiskState()

    # ------------------------------------------------------------ kill switch

    def update_equity(self, equity: float) -> None:
        self.state.roll_day(equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)

        if self.state.start_equity <= 0:
            return
        drawdown_pct = (self.state.start_equity - equity) / self.state.start_equity * 100.0
        if drawdown_pct >= self.cfg.max_daily_loss_pct and not self.state.halted:
            self.state.halted = True
            self.state.halt_reason = (
                f"daily loss {drawdown_pct:.2f}% hit the "
                f"{self.cfg.max_daily_loss_pct:.2f}% limit"
            )
            log.warning("TRADING HALTED: %s", self.state.halt_reason)

    @property
    def halted(self) -> bool:
        return self.state.halted

    def daily_pnl_pct(self, equity: float) -> float:
        if self.state.start_equity <= 0:
            return 0.0
        return (equity - self.state.start_equity) / self.state.start_equity * 100.0

    # --------------------------------------------------------------- entries

    def can_open(self, open_positions: int, cash: float, equity: float) -> tuple[bool, str]:
        if self.state.halted:
            return False, f"trading halted ({self.state.halt_reason})"
        if open_positions >= self.cfg.max_positions:
            return False, f"already holding max_positions ({self.cfg.max_positions})"
        min_cash = equity * self.cfg.min_cash_buffer_pct / 100.0
        if cash <= min_cash:
            return False, (
                f"cash {cash:,.2f} is at or below the "
                f"{self.cfg.min_cash_buffer_pct:.1f}% buffer ({min_cash:,.2f})"
            )
        return True, "ok"

    def record_entry(self) -> None:
        self.state.entries_today += 1

    def daily_trade_limit_reached(self, limit: int) -> bool:
        return limit > 0 and self.state.entries_today >= limit

    def correlation_ok(
        self, candidate: str, held: list[str], closes: dict[str, "pd.Series"]
    ) -> tuple[bool, str]:
        """Reject a candidate that simply repeats a position you already hold.

        Five names at 0.95 correlation is one position with five sets of fees
        and five times the gap risk. Needs `correlation_lookback` overlapping
        bars; with less overlap the check abstains rather than guessing.
        """
        limit = self.cfg.max_correlation
        if limit <= 0 or not held or candidate not in closes:
            return True, ""

        lookback = self.cfg.correlation_lookback
        base = closes[candidate].pct_change().dropna().iloc[-lookback:]
        if len(base) < max(10, lookback // 2):
            return True, ""

        for code in held:
            other = closes.get(code)
            if other is None:
                continue
            peer = other.pct_change().dropna().iloc[-lookback:]
            joined = pd.concat([base, peer], axis=1, join="inner").dropna()
            if len(joined) < max(10, lookback // 2):
                continue
            rho = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if not math.isnan(rho) and abs(rho) >= limit:
                return False, (
                    f"{abs(rho):.2f} correlated with open position {code} "
                    f"(limit {limit:.2f}) - that is the same bet twice"
                )
        return True, ""

    def size_position(
        self,
        equity: float,
        cash: float,
        price: float,
        strength: float = 1.0,
        stop_price: float | None = None,
        atr_value: float | None = None,
        lot_size: int = 1,
        contract_multiplier: int = 1,
        realised_vol: float | None = None,
    ) -> SizingResult:
        """Fixed-fractional sizing off the distance to the stop.

        Risk budget = equity * risk_per_trade_pct * strength, optionally scaled
        so the position carries the same volatility regardless of regime.
        Quantity = risk budget / (per-unit loss if the stop is hit), then capped
        by max_position_pct and by available cash.
        """
        if price <= 0:
            return SizingResult(0, 0.0, 0.0, 0.0, "invalid price")

        stop = self._resolve_stop(price, stop_price, atr_value)
        risk_per_unit = (price - stop) * contract_multiplier
        if risk_per_unit <= 0:
            return SizingResult(0, 0.0, 0.0, 0.0, "stop is at or above entry price")

        vol_scale = self.volatility_scale(realised_vol)
        budget = (
            equity
            * (self.cfg.risk_per_trade_pct / 100.0)
            * max(0.0, min(1.0, strength))
            * vol_scale
        )
        qty = budget / risk_per_unit

        unit_cost = price * contract_multiplier
        max_notional = equity * (self.cfg.max_position_pct / 100.0)
        qty = min(qty, max_notional / unit_cost)

        spendable = cash - equity * (self.cfg.min_cash_buffer_pct / 100.0)
        qty = min(qty, spendable / unit_cost) if spendable > 0 else 0.0

        qty = int(math.floor(max(qty, 0.0) / lot_size) * lot_size)
        if qty <= 0:
            return SizingResult(
                0, stop, 0.0, 0.0,
                f"sized to zero (risk budget {budget:,.2f}, cost per unit {unit_cost:,.2f})",
                vol_scale,
            )

        target = price * (1 + self.cfg.take_profit_pct / 100.0)
        notional = qty * unit_cost
        scale_note = f", vol scale {vol_scale:.2f}x" if vol_scale != 1.0 else ""
        return SizingResult(
            qty=qty,
            stop_price=round(stop, 4),
            target_price=round(target, 4),
            notional=notional,
            reason=(
                f"risk {budget:,.2f} / {risk_per_unit:,.4f} per unit -> {qty} "
                f"(notional {notional:,.2f}, {notional / equity * 100:.1f}% of equity"
                f"{scale_note})"
            ),
            vol_scale=vol_scale,
        )

    def volatility_scale(self, realised_vol: float | None) -> float:
        """Multiplier that pushes position volatility toward the target.

        Halving size when volatility doubles keeps the risk you actually take
        roughly constant. Scaling up in calm markets is capped, because calm
        markets end abruptly and a 4x position is how a quiet week becomes a
        catastrophic one.
        """
        target = self.cfg.target_volatility_pct
        if target <= 0 or not realised_vol or realised_vol <= 0:
            return 1.0
        return float(min(target / realised_vol, self.cfg.max_vol_scale))

    def volatility_ok(self, realised_vol: float | None) -> tuple[bool, str]:
        """Refuse new entries when the symbol is more volatile than the ceiling."""
        ceiling = self.cfg.max_volatility_pct
        if ceiling <= 0 or not realised_vol or realised_vol <= 0:
            return True, ""
        if realised_vol > ceiling:
            return False, (
                f"realised volatility {realised_vol:.1f}% is above the "
                f"{ceiling:.1f}% ceiling"
            )
        return True, ""

    def _resolve_stop(
        self, price: float, stop_price: float | None, atr_value: float | None
    ) -> float:
        """Pick the stop: ATR-based, strategy-supplied, or the percentage default."""
        if self.cfg.use_atr_stops and atr_value and atr_value > 0:
            return price - self.cfg.atr_stop_multiple * atr_value
        if stop_price is not None and 0 < stop_price < price:
            return stop_price
        return price * (1 - self.cfg.stop_loss_pct / 100.0)
