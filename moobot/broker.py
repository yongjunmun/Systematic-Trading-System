"""Trading account access and order placement.

This is the only module in the project that can send an order. It refuses to
initialise against a live-money account unless the caller has jumped through
the explicit hoops in ``config.assert_paper_trading``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .datafeed import probe_opend
from .settings import Config, assert_paper_trading

log = logging.getLogger(__name__)

# moomoo returns different column spellings across account types; try in order.
_CASH_COLUMNS = ("cash", "us_cash", "hk_cash", "available_funds", "avl_withdrawal_cash")
_EQUITY_COLUMNS = ("total_assets", "net_assets", "market_val")
_POWER_COLUMNS = ("power", "max_power_short", "net_cash_power", "available_funds")


class BrokerError(RuntimeError):
    """Raised when the trade API rejects a request."""


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    market_value: float
    currency: str
    raw: dict[str, Any]
    # Reported by the broker, not computed here. FINRA replaced the old pattern
    # day trader rules in June 2026 with a transition running to October 2027,
    # so whatever the firm actually enforces is the only reliable source.
    is_pattern_day_trader: bool = False
    day_trades_left: int | None = None

    @property
    def day_trades_exhausted(self) -> bool:
        return self.day_trades_left is not None and self.day_trades_left <= 0


@dataclass
class OrderResult:
    accepted: bool
    order_id: str
    code: str
    side: str
    qty: float
    price: float
    dry_run: bool
    message: str = ""

    def __str__(self) -> str:
        tag = "DRY-RUN" if self.dry_run else ("OK" if self.accepted else "REJECTED")
        return (
            f"[{tag}] {self.side} {self.qty:g} {self.code} @ {self.price:.4f}"
            f"{' - ' + self.message if self.message else ''}"
        )


def _first_number(row: Any, names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return default


def _day_trades_left(row: Any) -> int | None:
    """moomoo returns this as a free-form string, e.g. '3' or 'Unlimited'."""
    if "pdt_seq" not in row or pd.isna(row["pdt_seq"]):
        return None
    text = str(row["pdt_seq"]).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


class Broker:
    """Paper-trading gateway around ``moomoo.OpenSecTradeContext``."""

    def __init__(self, cfg: Config, dry_run: bool | None = None) -> None:
        # Hard gate. Raises unless the account is SIMULATE (or an override that
        # requires both a config flag and an environment acknowledgement).
        assert_paper_trading(cfg)

        self.cfg = cfg
        self.trd_env = cfg.account.trd_env.upper()
        self.market = cfg.account.market.upper()
        self.currency = cfg.account.currency.upper()
        self.dry_run = cfg.trading.dry_run if dry_run is None else dry_run
        self.acc_id = int(cfg.account.acc_id)
        self._ctx: Any | None = None
        self._unlocked = False

    # ---------------------------------------------------------------- context

    @property
    def ctx(self) -> Any:
        if self._ctx is None:
            problem = probe_opend(self.cfg.connection.host, self.cfg.connection.port)
            if problem:
                raise BrokerError(problem)
            try:
                import moomoo
            except ImportError as exc:  # pragma: no cover - env problem
                raise BrokerError(
                    "The 'moomoo' package is not installed. Run: pip install moomoo-api"
                ) from exc
            try:
                self._ctx = moomoo.OpenSecTradeContext(
                    filter_trdmarket=self.market,
                    host=self.cfg.connection.host,
                    port=self.cfg.connection.port,
                    security_firm=self.cfg.connection.security_firm,
                )
            except Exception as exc:
                raise BrokerError(
                    f"Could not open a trade context on {self.cfg.connection.host}:"
                    f"{self.cfg.connection.port}. Is OpenD running and logged in? "
                    f"Original error: {exc}"
                ) from exc
            if self.acc_id == 0:
                self.acc_id = self._auto_select_account()
            self._maybe_unlock()
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def __enter__(self) -> Broker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _check(self, result: tuple, what: str) -> Any:
        import moomoo

        ret, data = result[0], result[1]
        if ret != moomoo.RET_OK:
            raise BrokerError(f"{what} failed: {data}")
        return data

    # --------------------------------------------------------------- accounts

    def account_list(self) -> pd.DataFrame:
        return self._check(self.ctx.get_acc_list(), "get_acc_list")

    def _auto_select_account(self) -> int:
        accounts = self._check(self._ctx.get_acc_list(), "get_acc_list")
        if accounts is None or accounts.empty:
            raise BrokerError("OpenD returned no trading accounts for this login.")

        candidates = accounts
        if "trd_env" in accounts.columns:
            candidates = accounts[
                accounts["trd_env"].astype(str).str.upper() == self.trd_env
            ]
        if candidates.empty:
            raise BrokerError(
                f"No {self.trd_env} account found. Open the moomoo app and enable "
                "paper trading for this market first.\nAccounts visible to OpenD:\n"
                f"{accounts.to_string(index=False)}"
            )

        # Prefer an account that can trade options when options are enabled.
        if self.cfg.options.enabled and "sim_acc_type" in candidates.columns:
            option_accounts = candidates[
                candidates["sim_acc_type"].astype(str).str.upper().isin(
                    {"OPTION", "STOCK_AND_OPTION"}
                )
            ]
            if not option_accounts.empty:
                candidates = option_accounts
            else:
                log.warning(
                    "[options].enabled is true but no OPTION paper account was found; "
                    "option orders will be rejected by moomoo."
                )

        acc_id = int(candidates.iloc[0]["acc_id"])
        log.info("Auto-selected %s account acc_id=%s", self.trd_env, acc_id)
        return acc_id

    def _maybe_unlock(self) -> None:
        """Unlock trading. Paper accounts do not need this; live ones do."""
        if self.trd_env == "SIMULATE" or self._unlocked:
            return
        password_md5 = os.environ.get("MOOBOT_TRADE_PASSWORD_MD5", "")
        if not password_md5:
            raise BrokerError(
                "Live trading needs the MD5 of your trade password in the "
                "MOOBOT_TRADE_PASSWORD_MD5 environment variable."
            )
        self._check(self._ctx.unlock_trade(password_md5=password_md5), "unlock_trade")
        self._unlocked = True

    def account(self) -> AccountSnapshot:
        data = self._check(
            self.ctx.accinfo_query(
                trd_env=self.trd_env, acc_id=self.acc_id, currency=self.currency,
                refresh_cache=True,
            ),
            "accinfo_query",
        )
        if data is None or data.empty:
            raise BrokerError("accinfo_query returned no rows")
        row = data.iloc[0]
        return AccountSnapshot(
            equity=_first_number(row, _EQUITY_COLUMNS),
            cash=_first_number(row, _CASH_COLUMNS),
            buying_power=_first_number(row, _POWER_COLUMNS),
            market_value=_first_number(row, ("market_val", "long_mv")),
            currency=self.currency,
            raw=row.to_dict(),
            is_pattern_day_trader=bool(row["is_pdt"])
            if "is_pdt" in row and pd.notna(row["is_pdt"]) else False,
            day_trades_left=_day_trades_left(row),
        )

    def positions(self) -> pd.DataFrame:
        data = self._check(
            self.ctx.position_list_query(
                trd_env=self.trd_env, acc_id=self.acc_id, refresh_cache=True
            ),
            "position_list_query",
        )
        return data if data is not None else pd.DataFrame()

    def position_qty(self, code: str) -> float:
        pos = self.positions()
        if pos.empty or "code" not in pos.columns:
            return 0.0
        match = pos[pos["code"] == code]
        if match.empty:
            return 0.0
        return _first_number(match.iloc[0], ("qty",))

    def open_orders(self) -> pd.DataFrame:
        import moomoo

        data = self._check(
            self.ctx.order_list_query(
                status_filter_list=[
                    moomoo.OrderStatus.SUBMITTED,
                    moomoo.OrderStatus.WAITING_SUBMIT,
                    moomoo.OrderStatus.SUBMITTING,
                    moomoo.OrderStatus.FILLED_PART,
                ],
                trd_env=self.trd_env,
                acc_id=self.acc_id,
                refresh_cache=True,
            ),
            "order_list_query",
        )
        return data if data is not None else pd.DataFrame()

    def todays_fills(self) -> pd.DataFrame:
        data = self._check(
            self.ctx.deal_list_query(trd_env=self.trd_env, acc_id=self.acc_id),
            "deal_list_query",
        )
        return data if data is not None else pd.DataFrame()

    # ----------------------------------------------------------------- orders

    def limit_price(self, reference_price: float, side: str) -> float:
        """Marketable limit price: cross the book by the configured slippage."""
        bps = self.cfg.trading.limit_slippage_bps / 10_000.0
        price = reference_price * (1 + bps) if side == "BUY" else reference_price * (1 - bps)
        return round(max(price, 0.01), 2)

    def place(
        self,
        code: str,
        side: str,
        qty: float,
        reference_price: float,
        order_type: str | None = None,
        remark: str = "",
    ) -> OrderResult:
        """Send an order. Honours `dry_run` by logging instead of sending."""
        import moomoo

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if qty <= 0:
            return OrderResult(False, "", code, side, 0, 0, self.dry_run, "quantity is zero")
        if reference_price <= 0:
            return OrderResult(False, "", code, side, qty, 0, self.dry_run, "no reference price")

        order_type = order_type or self.cfg.trading.order_type
        # Shares and option contracts are both whole units.
        qty = float(int(qty))
        price = self.limit_price(reference_price, side)

        if self.dry_run:
            result = OrderResult(
                True, "", code, side, qty, price, True, remark or "dry run - not sent"
            )
            log.info("%s", result)
            return result

        try:
            ret, data = self.ctx.place_order(
                price=price,
                qty=qty,
                code=code,
                trd_side=moomoo.TrdSide.BUY if side == "BUY" else moomoo.TrdSide.SELL,
                order_type=order_type,
                trd_env=self.trd_env,
                acc_id=self.acc_id,
                remark=(remark or "moobot")[:64],
            )
        except Exception as exc:
            return OrderResult(False, "", code, side, qty, price, False, f"exception: {exc}")

        if ret != moomoo.RET_OK:
            result = OrderResult(False, "", code, side, qty, price, False, str(data))
            log.error("%s", result)
            return result

        order_id = ""
        if data is not None and not data.empty and "order_id" in data.columns:
            order_id = str(data.iloc[0]["order_id"])
        result = OrderResult(True, order_id, code, side, qty, price, False, "submitted")
        log.info("%s", result)
        return result

    def cancel_order(self, order_id: str) -> bool:
        import moomoo

        if self.dry_run:
            log.info("[DRY-RUN] cancel order %s", order_id)
            return True
        ret, data = self.ctx.modify_order(
            modify_order_op=moomoo.ModifyOrderOp.CANCEL,
            order_id=order_id,
            qty=0,
            price=0,
            trd_env=self.trd_env,
            acc_id=self.acc_id,
        )
        if ret != moomoo.RET_OK:
            log.error("cancel of %s failed: %s", order_id, data)
            return False
        return True

    def cancel_all(self) -> int:
        orders = self.open_orders()
        if orders.empty or "order_id" not in orders.columns:
            return 0
        return sum(self.cancel_order(str(oid)) for oid in orders["order_id"])

    def max_affordable_qty(self, code: str, price: float, lot_size: int = 1) -> int:
        """Shares affordable with current cash, rounded down to a whole lot."""
        if price <= 0:
            return 0
        cash = self.account().cash
        buffer = 1 - self.cfg.risk.min_cash_buffer_pct / 100.0
        raw = (cash * buffer) / price
        return int(math.floor(raw / lot_size) * lot_size)
