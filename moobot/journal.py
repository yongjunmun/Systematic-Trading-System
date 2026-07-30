"""SQLite trade journal.

Every signal, order and equity reading is written here so a run can be audited
after the fact. Also holds the engine's open-position bookkeeping (stop, target
and high-water mark) so a restart does not forget where the stops were.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    code TEXT NOT NULL,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,
    strength REAL NOT NULL,
    price REAL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    code TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    order_id TEXT,
    accepted INTEGER NOT NULL,
    dry_run INTEGER NOT NULL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    market_value REAL NOT NULL,
    open_positions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS holdings (
    code TEXT PRIMARY KEY,
    opened_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    qty REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    high_water REAL NOT NULL,
    strategy TEXT,
    meta TEXT,
    initial_stop REAL NOT NULL DEFAULT 0,
    original_qty REAL NOT NULL DEFAULT 0,
    partial_done INTEGER NOT NULL DEFAULT 0,
    breakeven_done INTEGER NOT NULL DEFAULT 0,
    realised_pnl REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    code TEXT NOT NULL,
    opened_ts TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    qty REAL NOT NULL,
    initial_stop REAL NOT NULL,
    pnl REAL NOT NULL,
    r_multiple REAL,
    strategy TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts);
CREATE INDEX IF NOT EXISTS idx_closed_ts ON closed_trades(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Holding:
    """Engine-side bookkeeping for one open position."""

    code: str
    opened_ts: str
    entry_price: float
    qty: float
    stop_price: float
    target_price: float
    high_water: float
    strategy: str = ""
    meta: dict[str, Any] | None = None
    # Ladder state. initial_stop is frozen at entry - it is the 1R denominator
    # and must not move when the stop trails up.
    initial_stop: float = 0.0
    original_qty: float = 0.0
    partial_done: bool = False
    breakeven_done: bool = False
    realised_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_stop <= 0:
            self.initial_stop = self.stop_price
        if self.original_qty <= 0:
            self.original_qty = self.qty

    @property
    def risk_per_share(self) -> float:
        return max(self.entry_price - self.initial_stop, 1e-9)

    def r_at(self, price: float) -> float:
        return (price - self.entry_price) / self.risk_per_share


class Journal:
    def __init__(self, path: str | Path = "data/journal.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        with closing(self.conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------- write side

    def log_signal(
        self, code: str, strategy: str, action: str, strength: float,
        price: float | None, reason: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO signals (ts, code, strategy, action, strength, price, reason)"
            " VALUES (?,?,?,?,?,?,?)",
            (_now(), code, strategy, action, strength, price, reason),
        )
        self.conn.commit()

    def log_order(self, result: Any) -> None:
        self.conn.execute(
            "INSERT INTO orders (ts, code, side, qty, price, order_id, accepted, dry_run, message)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                _now(), result.code, result.side, result.qty, result.price,
                result.order_id, int(result.accepted), int(result.dry_run), result.message,
            ),
        )
        self.conn.commit()

    def log_equity(
        self, equity: float, cash: float, market_value: float, open_positions: int
    ) -> None:
        self.conn.execute(
            "INSERT INTO equity (ts, equity, cash, market_value, open_positions)"
            " VALUES (?,?,?,?,?)",
            (_now(), equity, cash, market_value, open_positions),
        )
        self.conn.commit()

    # ---------------------------------------------------------------- holdings

    def upsert_holding(self, holding: Holding) -> None:
        self.conn.execute(
            "INSERT INTO holdings"
            " (code, opened_ts, entry_price, qty, stop_price, target_price, high_water,"
            "  strategy, meta, initial_stop, original_qty, partial_done, breakeven_done,"
            "  realised_pnl)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(code) DO UPDATE SET"
            "  entry_price=excluded.entry_price, qty=excluded.qty,"
            "  stop_price=excluded.stop_price, target_price=excluded.target_price,"
            "  high_water=excluded.high_water, strategy=excluded.strategy,"
            "  meta=excluded.meta, initial_stop=excluded.initial_stop,"
            "  original_qty=excluded.original_qty, partial_done=excluded.partial_done,"
            "  breakeven_done=excluded.breakeven_done, realised_pnl=excluded.realised_pnl",
            (
                holding.code, holding.opened_ts, holding.entry_price, holding.qty,
                holding.stop_price, holding.target_price, holding.high_water,
                holding.strategy, json.dumps(holding.meta or {}), holding.initial_stop,
                holding.original_qty, int(holding.partial_done), int(holding.breakeven_done),
                holding.realised_pnl,
            ),
        )
        self.conn.commit()

    def get_holding(self, code: str) -> Holding | None:
        row = self.conn.execute(
            "SELECT * FROM holdings WHERE code = ?", (code,)
        ).fetchone()
        return self._to_holding(row) if row else None

    def all_holdings(self) -> dict[str, Holding]:
        rows = self.conn.execute("SELECT * FROM holdings").fetchall()
        return {row["code"]: self._to_holding(row) for row in rows}

    def drop_holding(self, code: str) -> None:
        self.conn.execute("DELETE FROM holdings WHERE code = ?", (code,))
        self.conn.commit()

    @staticmethod
    def _to_holding(row: sqlite3.Row) -> Holding:
        keys = row.keys()
        return Holding(
            code=row["code"],
            opened_ts=row["opened_ts"],
            entry_price=row["entry_price"],
            qty=row["qty"],
            stop_price=row["stop_price"],
            target_price=row["target_price"],
            high_water=row["high_water"],
            strategy=row["strategy"] or "",
            meta=json.loads(row["meta"] or "{}"),
            initial_stop=row["initial_stop"] if "initial_stop" in keys else 0.0,
            original_qty=row["original_qty"] if "original_qty" in keys else 0.0,
            partial_done=bool(row["partial_done"]) if "partial_done" in keys else False,
            breakeven_done=bool(row["breakeven_done"]) if "breakeven_done" in keys else False,
            realised_pnl=row["realised_pnl"] if "realised_pnl" in keys else 0.0,
        )

    # ---------------------------------------------------------- closed trades

    def log_closed_trade(
        self, holding: Holding, exit_price: float, qty: float, pnl: float, reason: str
    ) -> None:
        risk = holding.risk_per_share * (holding.original_qty or qty)
        self.conn.execute(
            "INSERT INTO closed_trades"
            " (ts, code, opened_ts, entry_price, exit_price, qty, initial_stop, pnl,"
            "  r_multiple, strategy, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                _now(), holding.code, holding.opened_ts, holding.entry_price, exit_price,
                qty, holding.initial_stop, pnl,
                (pnl / risk) if risk > 0 else None, holding.strategy, reason,
            ),
        )
        self.conn.commit()

    def closed_trades(self, since: str | None = None) -> list[dict[str, Any]]:
        if since:
            rows = self.conn.execute(
                "SELECT * FROM closed_trades WHERE ts >= ? ORDER BY id", (since,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM closed_trades ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def entries_on(self, day: str) -> int:
        """Accepted BUY orders logged on an ISO date - drives max_trades_per_day."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM orders"
            " WHERE side = 'BUY' AND accepted = 1 AND substr(ts, 1, 10) = ?",
            (day,),
        ).fetchone()
        return int(row["n"]) if row else 0

    # -------------------------------------------------------------- read side

    def recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_signals(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def equity_curve(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM equity ORDER BY id").fetchall()
        return [dict(r) for r in rows]
