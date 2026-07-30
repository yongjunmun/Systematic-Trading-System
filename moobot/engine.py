"""The live paper-trading loop.

Each cycle:

1. Check the session phase. Outside trading hours this returns almost
   immediately, which is what makes a short poll interval cheap.
2. Read the account, update the risk manager, check the daily kill switch.
3. Reconcile the local holdings table against the broker's real positions.
4. Manage every open position through the same `ExitLadder` the backtester uses.
5. If the phase allows entries, look for new ones.

Stops are enforced by this loop rather than resting on the exchange, because
moomoo paper trading only accepts limit and market orders. They are therefore
only as tight as ``poll_seconds``, and nothing protects a position while the bot
is not running.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from .broker import Broker, BrokerError, OrderResult
from .datafeed import DataError, DataFeed
from .exits import Bar, ExitLadder, Position
from .indicators import realised_volatility
from .journal import Holding, Journal
from .notify import Notifier
from .options import OptionQuote, contracts_for_budget, select_contract
from .risk import RiskManager
from .scanner import read_watchlist
from .session import Phase, SessionGate
from .settings import Config
from .strategies import Action, Signal, get_strategy

log = logging.getLogger(__name__)


@dataclass
class CycleReport:
    phase: str
    equity: float
    cash: float
    open_positions: int
    daily_pnl_pct: float
    halted: bool
    actions: list[str]


class TradingEngine:
    def __init__(self, cfg: Config, dry_run: bool | None = None) -> None:
        self.cfg = cfg
        self.feed = DataFeed(
            host=cfg.connection.host,
            port=cfg.connection.port,
            security_firm=cfg.connection.security_firm,
            cache_dir=cfg.data_dir / "cache",
        )
        self.broker = Broker(cfg, dry_run=dry_run)
        self.risk = RiskManager(cfg.risk)
        self.journal = Journal(cfg.data_dir / "journal.db")
        self.strategy = get_strategy(cfg.strategy.name, **cfg.strategy.params)
        self.ladder = ExitLadder(cfg.exits)
        self.gate = SessionGate(cfg.session)
        self.notifier = Notifier(cfg.notifications)
        self._market_open_cache: dict[str, tuple[float, bool]] = {}
        self._closes: dict[str, pd.Series] = {}

    def close(self) -> None:
        self.feed.close()
        self.broker.close()
        self.journal.close()

    def __enter__(self) -> TradingEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ symbols

    def active_codes(self) -> list[str]:
        """What to trade this cycle: the watchlist if enabled, else the config."""
        if not self.cfg.trading.use_watchlist:
            return self.cfg.trading.codes
        codes = read_watchlist(self.cfg.scanner.watchlist_path)
        if not codes:
            log.warning(
                "use_watchlist is on but %s is empty - run `scan` first. "
                "Falling back to [trading].codes.",
                self.cfg.scanner.watchlist_path,
            )
            return self.cfg.trading.codes
        return codes

    # ------------------------------------------------------------- main loop

    def run(self, max_cycles: int | None = None) -> None:
        mode = "DRY RUN (no orders sent)" if self.broker.dry_run else "LIVE PAPER ORDERS"
        log.info(
            "Starting %s | env=%s market=%s strategy=%s exits=%s poll=%ss | alerts: %s",
            mode, self.broker.trd_env, self.broker.market, self.strategy,
            self.cfg.exits.mode, self.cfg.trading.poll_seconds, self.notifier.describe(),
        )
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                started = time.monotonic()
                try:
                    report = self.run_cycle()
                    log.info(
                        "cycle %d [%s] | equity %s | cash %s | positions %d | day P&L %+.2f%%%s",
                        cycle, report.phase, f"{report.equity:,.2f}", f"{report.cash:,.2f}",
                        report.open_positions, report.daily_pnl_pct,
                        "  [HALTED]" if report.halted else "",
                    )
                    for action in report.actions:
                        log.info("    %s", action)
                except (BrokerError, DataError) as exc:
                    log.error("cycle %d failed: %s", cycle, exc)
                    self.notifier.error(f"cycle {cycle}: {exc}")

                if max_cycles is not None and cycle >= max_cycles:
                    break
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, self.cfg.trading.poll_seconds - elapsed))
        except KeyboardInterrupt:
            log.info("Interrupted by user after %d cycle(s). Shutting down.", cycle)

    def run_cycle(self) -> CycleReport:
        status = self.gate.status()
        if status.phase is Phase.CLOSED:
            # Cheapest possible path: no API calls at all when the market is shut.
            return CycleReport(status.phase.value, 0.0, 0.0, 0, 0.0, False, [status.reason])

        account = self.broker.account()
        was_halted = self.risk.halted
        self.risk.update_equity(account.equity)
        if self.risk.halted and not was_halted:
            self.notifier.halt(self.risk.state.halt_reason)

        positions = self.broker.positions()
        open_codes = self._held_codes(positions)
        self.journal.log_equity(
            account.equity, account.cash, account.market_value, len(open_codes)
        )
        self._reconcile(open_codes, positions)

        actions: list[str] = []
        codes = self.active_codes()

        if self.risk.halted:
            actions.append(f"halted: {self.risk.state.halt_reason}")
            actions.extend(self._flatten_all(positions, "daily loss limit"))
        elif status.phase is Phase.FORCE_CLOSE:
            actions.append(status.reason)
            actions.extend(self._flatten_all(positions, "end of session"))
        else:
            for code in codes:
                try:
                    actions.extend(self._manage(code))
                except (BrokerError, DataError) as exc:
                    log.error("%s manage: %s", code, exc)
                    actions.append(f"{code}: manage error - {exc}")

            if status.phase.allows_entry:
                actions.extend(self._look_for_entries(codes, account, open_codes))
            else:
                actions.append(status.reason)

        return CycleReport(
            phase=status.phase.value,
            equity=account.equity,
            cash=account.cash,
            open_positions=len(open_codes),
            daily_pnl_pct=self.risk.daily_pnl_pct(account.equity),
            halted=self.risk.halted,
            actions=actions,
        )

    # ------------------------------------------------------------------ entries

    def _look_for_entries(self, codes, account, open_codes) -> list[str]:
        limit = self.cfg.session.max_trades_per_day if self.cfg.session.enabled else 0
        if limit:
            taken = self.journal.entries_on(date.today().isoformat())
            if taken >= limit:
                return [f"daily trade limit reached ({taken}/{limit})"]

        # The session gate flattens intraday, so every entry taken now becomes a
        # day trade. Respect whatever the broker says is left rather than
        # tripping a restriction on the account.
        if self.cfg.session.enabled and account.day_trades_exhausted:
            return ["broker reports no day trades remaining - not opening new positions"]

        actions: list[str] = []
        for code in codes:
            try:
                actions.extend(self._maybe_enter(code, account, open_codes))
            except (BrokerError, DataError) as exc:
                log.error("%s entry: %s", code, exc)
                actions.append(f"{code}: entry error - {exc}")
        return actions

    def _maybe_enter(self, code: str, account, open_codes: dict[str, float]) -> list[str]:
        if self.cfg.trading.only_trade_market_hours and not self._market_open(code):
            return []
        if code in open_codes or self.journal.get_holding(code) is not None:
            return []
        if self.cfg.options.enabled and self._option_open_for(code):
            return []

        bars = self._bars(code)
        if len(bars) < self.strategy.warmup:
            return [f"{code}: only {len(bars)} bars, need {self.strategy.warmup}"]
        self._closes[code] = bars.set_index("time_key")["close"]

        price = float(bars.iloc[-1]["close"])
        signal = self.strategy.latest_signal(bars)
        self.journal.log_signal(
            code, self.strategy.name, signal.action.value, signal.strength, price, signal.reason
        )
        if signal.action is not Action.BUY:
            return []

        allowed, why = self.risk.can_open(len(open_codes), account.cash, account.equity)
        if not allowed:
            return [f"{code}: entry blocked - {why}"]

        correlated_ok, corr_why = self.risk.correlation_ok(
            code, [c for c in open_codes if c != code], self._closes
        )
        if not correlated_ok:
            return [f"{code}: entry blocked - {corr_why}"]

        vol = self._realised_vol(bars)
        vol_ok, vol_why = self.risk.volatility_ok(vol)
        if not vol_ok:
            return [f"{code}: entry blocked - {vol_why}"]

        if self.cfg.options.enabled:
            return self._open_option(code, signal, account)
        return self._open_stock(code, price, signal, bars, vol)

    def _realised_vol(self, bars) -> float | None:
        """Annualised realised volatility as a percentage, or None when unused."""
        if self.cfg.risk.target_volatility_pct <= 0 and self.cfg.risk.max_volatility_pct <= 0:
            return None
        series = realised_volatility(
            bars["close"], self.cfg.risk.vol_lookback, self.cfg.bars_per_year
        )
        latest = series.iloc[-1]
        return float(latest) * 100.0 if pd.notna(latest) else None

    def _open_stock(self, code, price, signal: Signal, bars, vol=None) -> list[str]:
        account = self.broker.account()
        frame = self.strategy.compute(bars)
        atr_value = (
            float(frame.iloc[-1]["atr"])
            if "atr" in frame.columns and pd.notna(frame.iloc[-1]["atr"])
            else None
        )

        sized = self.risk.size_position(
            equity=account.equity, cash=account.cash, price=price,
            strength=signal.strength, stop_price=signal.stop_price, atr_value=atr_value,
            realised_vol=vol,
        )
        if not sized.ok:
            return [f"{code}: not sized - {sized.reason}"]

        result = self._send(code, "BUY", sized.qty, price, signal.reason[:60])
        if not result.accepted:
            return [f"{code}: buy rejected - {result.message}"]

        self.risk.record_entry()
        holding = Holding(
            code=code,
            opened_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            entry_price=result.price,
            qty=sized.qty,
            stop_price=sized.stop_price,
            target_price=sized.target_price,
            high_water=result.price,
            strategy=self.strategy.name,
            meta={"kind": "stock", "reason": signal.reason},
            initial_stop=sized.stop_price,
            original_qty=sized.qty,
        )
        self.journal.upsert_holding(holding)
        self.notifier.entry(code, sized.qty, result.price, sized.stop_price, signal.reason)
        return [
            f"{code}: BUY {sized.qty} @ ~{result.price:.2f}, stop {sized.stop_price:.2f} "
            f"({sized.reason})"
        ]

    def _open_option(self, underlying, signal: Signal, account) -> list[str]:
        style = self.cfg.options.style
        quote = select_contract(self.feed, underlying, "BUY", self.cfg.options)
        if quote is None:
            return [f"{underlying}: no option contract passed the filters"]

        if style == "long":
            qty = contracts_for_budget(quote, self.cfg.options, account.cash)
            if qty <= 0:
                return [
                    f"{underlying}: cannot afford {quote.code} "
                    f"({quote.cost_per_contract:,.2f} per contract)"
                ]
            side = "BUY"
            stop = quote.mid * (1 - self.cfg.options.stop_loss_pct / 100.0)
            target = quote.mid * (1 + self.cfg.options.take_profit_pct / 100.0)
        else:
            qty = self._short_option_capacity(underlying, quote, account)
            if qty <= 0:
                need = (
                    "100 shares of the underlying" if style == "covered_call"
                    else f"{quote.strike * quote.contract_size:,.0f} in cash"
                )
                return [f"{underlying}: {style} needs {need} - skipping"]
            side = "SELL"
            target = quote.mid * (1 - self.cfg.options.short_profit_target_pct / 100.0)
            stop = quote.mid * self.cfg.options.short_stop_multiple

        result = self._send(quote.code, side, qty, quote.mid, f"{style} {signal.reason}"[:60])
        if not result.accepted:
            return [f"{quote.code}: {side} rejected - {result.message}"]

        self.risk.record_entry()
        self.journal.upsert_holding(
            Holding(
                code=quote.code,
                opened_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                entry_price=result.price,
                qty=qty if side == "BUY" else -qty,
                stop_price=round(stop, 4),
                target_price=round(target, 4),
                high_water=result.price,
                strategy=self.strategy.name,
                initial_stop=round(stop, 4),
                original_qty=qty,
                meta={
                    "kind": "option",
                    "style": style,
                    "underlying": underlying,
                    "expiry": quote.expiry,
                    "strike": quote.strike,
                    "delta": quote.delta,
                    "contract_size": quote.contract_size,
                },
            )
        )
        self.notifier.entry(quote.code, qty, result.price, stop, f"{style}: {signal.reason}")
        return [f"{underlying}: {side} {qty}x {quote.code} @ ~{result.price:.2f} ({quote})"]

    def _short_option_capacity(self, underlying: str, quote: OptionQuote, account) -> int:
        if self.cfg.options.style == "covered_call":
            shares = self.broker.position_qty(underlying)
            return min(self.cfg.options.max_contracts, int(shares // quote.contract_size))
        collateral = quote.strike * quote.contract_size
        if collateral <= 0:
            return 0
        spendable = account.cash - account.equity * self.cfg.risk.min_cash_buffer_pct / 100.0
        return max(0, min(self.cfg.options.max_contracts, int(spendable // collateral)))

    # ------------------------------------------------------- managing positions

    def _manage(self, code: str) -> list[str]:
        """Run the ladder and the strategy exit over anything held on `code`."""
        actions: list[str] = []
        for held_code, holding in self.journal.all_holdings().items():
            meta = holding.meta or {}
            is_option = meta.get("kind") == "option"
            owner = meta.get("underlying", held_code) if is_option else held_code
            if owner != code:
                continue

            try:
                mark = self.feed.last_price(held_code)
            except DataError:
                actions.append(f"{held_code}: no mark price, holding")
                continue

            if holding.qty < 0:
                actions.extend(self._manage_short_option(holding, mark))
            elif is_option:
                actions.extend(self._manage_long_option(holding, mark))
            else:
                actions.extend(self._manage_stock(holding, mark, code))
        return actions

    def _manage_stock(self, holding: Holding, mark: float, code: str) -> list[str]:
        position = Position(
            entry_price=holding.entry_price,
            initial_stop=holding.initial_stop,
            qty=int(holding.qty),
            stop_price=holding.stop_price,
            target_price=holding.target_price,
            high_water=holding.high_water,
            partial_done=holding.partial_done,
            breakeven_done=holding.breakeven_done,
            original_qty=int(holding.original_qty),
        )
        atr_value, swing = self._exit_inputs(code)
        decision = self.ladder.evaluate(
            position, Bar.from_price(mark), atr=atr_value, swing_low=swing
        )

        actions: list[str] = []
        if not decision.should_exit:
            # A strategy SELL closes the rest even if no protective level was hit.
            signal = self._latest_signal(code)
            if signal is not None and signal.action is Action.SELL:
                decision.exit_qty = position.qty
                decision.is_full_exit = True
                decision.exit_price = mark
                decision.reason = f"strategy exit: {signal.reason}"

        for event in decision.events:
            actions.append(f"{holding.code}: {event}")

        holding.stop_price = position.stop_price
        holding.high_water = position.high_water
        holding.partial_done = position.partial_done
        holding.breakeven_done = position.breakeven_done

        if not decision.should_exit:
            self.journal.upsert_holding(holding)
            return actions

        qty = min(int(decision.exit_qty), int(holding.qty))
        result = self._send(holding.code, "SELL", qty, mark, decision.reason[:60])
        if not result.accepted:
            actions.append(f"{holding.code}: close rejected - {result.message}")
            self.journal.upsert_holding(holding)
            return actions

        pnl = (result.price - holding.entry_price) * qty
        holding.realised_pnl += pnl
        holding.qty -= qty
        r_now = holding.r_at(result.price)

        if holding.qty <= 0:
            self.journal.log_closed_trade(
                holding, result.price, holding.original_qty, holding.realised_pnl,
                decision.reason,
            )
            self.journal.drop_holding(holding.code)
            self.notifier.exit(
                holding.code, holding.original_qty, result.price, decision.reason,
                holding.realised_pnl,
            )
            actions.append(
                f"{holding.code}: CLOSED {holding.original_qty:g} @ ~{result.price:.2f} "
                f"({decision.reason}) P&L {holding.realised_pnl:+,.2f} ({r_now:+.2f}R)"
            )
        else:
            self.journal.upsert_holding(holding)
            actions.append(
                f"{holding.code}: SCALED OUT {qty} @ ~{result.price:.2f} "
                f"({decision.reason}), {holding.qty:g} left"
            )
        return actions

    def _manage_long_option(self, holding: Holding, mark: float) -> list[str]:
        if mark >= holding.target_price:
            reason = f"premium {mark:.2f} reached target {holding.target_price:.2f}"
        elif mark <= holding.stop_price:
            reason = f"premium {mark:.2f} fell to stop {holding.stop_price:.2f}"
        else:
            holding.high_water = max(holding.high_water, mark)
            self.journal.upsert_holding(holding)
            return []
        return self._close_option(holding, mark, reason, "SELL")

    def _manage_short_option(self, holding: Holding, mark: float) -> list[str]:
        if mark <= holding.target_price:
            reason = f"premium decayed to {mark:.2f} (target {holding.target_price:.2f})"
        elif mark >= holding.stop_price:
            reason = f"premium expanded to {mark:.2f} (stop {holding.stop_price:.2f})"
        else:
            return []
        return self._close_option(holding, mark, reason, "BUY")

    def _close_option(self, holding: Holding, mark: float, reason: str, side: str) -> list[str]:
        qty = abs(holding.qty)
        result = self._send(holding.code, side, qty, mark, reason[:60])
        if not result.accepted:
            return [f"{holding.code}: close rejected - {result.message}"]
        direction = -1 if holding.qty < 0 else 1
        size = int((holding.meta or {}).get("contract_size", 100))
        pnl = (result.price - holding.entry_price) * direction * qty * size
        self.journal.log_closed_trade(holding, result.price, qty, pnl, reason)
        self.journal.drop_holding(holding.code)
        self.notifier.exit(holding.code, qty, result.price, reason, pnl)
        return [
            f"{holding.code}: CLOSED {qty} @ ~{result.price:.2f} ({reason}) "
            f"est P&L {pnl:+,.2f}"
        ]

    def _flatten_all(self, positions: pd.DataFrame, why: str) -> list[str]:
        """Cancel working orders and close everything."""
        actions: list[str] = []
        self.broker.cancel_all()
        for code, qty in self._held_codes(positions).items():
            if qty == 0:
                continue
            try:
                mark = self.feed.last_price(code)
            except DataError:
                mark = 0.0
            if mark <= 0:
                actions.append(f"{code}: CANNOT FLATTEN - no price available")
                log.error("cannot flatten %s: no price available", code)
                continue
            side = "SELL" if qty > 0 else "BUY"
            result = self._send(code, side, abs(qty), mark, f"flatten: {why}"[:60])
            holding = self.journal.get_holding(code)
            if result.accepted and holding is not None:
                pnl = (result.price - holding.entry_price) * qty
                self.journal.log_closed_trade(holding, result.price, abs(qty), pnl, why)
                self.notifier.exit(code, abs(qty), result.price, why, pnl)
            self.journal.drop_holding(code)
            actions.append(f"{code}: FLATTENED {abs(qty):g} @ ~{result.price:.2f} ({why})")
        return actions

    # ------------------------------------------------------------------ helpers

    def _exit_inputs(self, code: str) -> tuple[float | None, float | None]:
        """ATR and the latest confirmed swing low, for the trailing stop."""
        if self.cfg.exits.trail_method not in {"atr", "swing_low"}:
            return None, None
        try:
            bars = self._bars(code)
        except DataError:
            return None, None
        from .exits import swing_lows
        from .indicators import atr as atr_indicator

        atr_series = atr_indicator(
            bars["high"], bars["low"], bars["close"], self.cfg.risk.atr_period
        )
        atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None
        swing = None
        if self.cfg.exits.trail_method == "swing_low":
            series = swing_lows(
                bars["low"], self.cfg.exits.swing_left, self.cfg.exits.swing_right
            )
            swing = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else None
        return atr_value, swing

    def _latest_signal(self, code: str) -> Signal | None:
        try:
            return self.strategy.latest_signal(self._bars(code))
        except DataError:
            return None

    def _send(self, code, side, qty, price, remark) -> OrderResult:
        result = self.broker.place(code, side, qty, price, remark=remark)
        self.journal.log_order(result)
        return result

    def _bars(self, code: str) -> pd.DataFrame:
        """Latest bars, preferring the live subscription, falling back to history."""
        try:
            return self.feed.recent_bars(
                code, self.cfg.trading.bar_type, self.cfg.trading.history_bars
            )
        except DataError as exc:
            log.debug("live kline unavailable for %s (%s), using history", code, exc)
            return self.feed.history(
                code, bar_type=self.cfg.trading.bar_type,
                max_bars=self.cfg.trading.history_bars, refresh=True,
            )

    @staticmethod
    def _held_codes(positions: pd.DataFrame) -> dict[str, float]:
        if positions is None or positions.empty or "code" not in positions.columns:
            return {}
        out: dict[str, float] = {}
        for _, row in positions.iterrows():
            qty = float(row.get("qty", 0) or 0)
            if qty != 0:
                out[str(row["code"])] = qty
        return out

    def _option_open_for(self, underlying: str) -> bool:
        return any(
            (h.meta or {}).get("underlying") == underlying
            for h in self.journal.all_holdings().values()
        )

    def _reconcile(self, open_codes: dict[str, float], positions: pd.DataFrame) -> None:
        """Keep the local holdings table honest about what the broker really holds."""
        tracked = self.journal.all_holdings()

        for code in tracked:
            if code not in open_codes and not self.broker.dry_run:
                log.info("%s is no longer held at the broker - dropping local record", code)
                self.journal.drop_holding(code)

        if positions is None or positions.empty:
            return
        for _, row in positions.iterrows():
            code = str(row.get("code", ""))
            qty = float(row.get("qty", 0) or 0)
            if not code or qty == 0 or code in tracked:
                continue
            cost = float(row.get("cost_price", 0) or 0) or float(row.get("nominal_price", 0) or 0)
            if cost <= 0:
                continue
            stop = round(cost * (1 - self.cfg.risk.stop_loss_pct / 100.0), 4)
            log.info("adopting untracked broker position %s x%g @ %.4f", code, qty, cost)
            self.journal.upsert_holding(
                Holding(
                    code=code,
                    opened_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    entry_price=cost,
                    qty=qty,
                    stop_price=stop,
                    target_price=round(cost * (1 + self.cfg.risk.take_profit_pct / 100.0), 4),
                    high_water=cost,
                    strategy="adopted",
                    initial_stop=stop,
                    original_qty=qty,
                    meta={"kind": "stock", "reason": "existing broker position"},
                )
            )

    def _market_open(self, code: str) -> bool:
        """Cached market-state lookup - called once per symbol per cycle."""
        now = time.monotonic()
        cached = self._market_open_cache.get(code)
        if cached and now - cached[0] < 60:
            return cached[1]
        state = self.feed.is_market_open(code)
        self._market_open_cache[code] = (now, state)
        return state
