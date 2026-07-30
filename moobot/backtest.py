"""Event-driven backtester.

Deliberately pessimistic so the numbers do not flatter the strategy:

* A signal computed from bar *i*'s close is filled at bar *i+1*'s **open**.
  Filling at the signal bar's own close is the single most common way a
  backtest invents profit that does not exist.
* Slippage is applied against you on both sides, plus per-share commission.
* Intrabar stop and target hits come from each bar's high/low range.
* When a bar's range covers both the stop and a profit level, the stop is
  assumed to have been hit first.
* Every exit is priced through the same `ExitLadder` the live engine uses, so
  the two cannot silently disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exits import Bar, ExitLadder, Position, swing_lows
from .indicators import atr as atr_indicator
from .indicators import realised_volatility
from .risk import RiskManager
from .settings import BARS_PER_YEAR, ExitsConfig
from .strategies.base import Action, Strategy


@dataclass
class Trade:
    """One complete round trip, including any scale-outs."""

    code: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float          # quantity-weighted average across all exit legs
    qty: int                   # original size
    pnl: float
    return_pct: float
    bars_held: int
    entry_reason: str
    exit_reason: str
    initial_stop: float = 0.0
    r_multiple: float | None = None
    scale_outs: int = 0
    max_favourable_r: float = 0.0


@dataclass
class BacktestResult:
    code: str
    strategy: str
    equity: pd.Series = field(default_factory=pd.Series)
    trades: list[Trade] = field(default_factory=list)
    initial_cash: float = 0.0
    bars_per_year: int = 252
    bars_in_market: int = 0
    total_bars: int = 0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_cash

    @property
    def r_multiples(self) -> list[float]:
        return [t.r_multiple for t in self.trades if t.r_multiple is not None]

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "code", "entry_time", "entry_price", "exit_time", "exit_price",
                    "qty", "pnl", "return_pct", "bars_held", "entry_reason",
                    "exit_reason", "initial_stop", "r_multiple", "scale_outs",
                    "max_favourable_r",
                ]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _commission(qty: int, per_share: float, minimum: float) -> float:
    if qty <= 0:
        return 0.0
    return max(minimum, qty * per_share)


class _OpenTrade:
    """Accumulates the legs of one round trip while it is still open."""

    def __init__(self, position: Position, index: int, time_key, reason: str, fee: float) -> None:
        self.position = position
        self.entry_index = index
        self.entry_time = time_key
        self.entry_reason = reason
        self.entry_fee = fee
        self.exit_value = 0.0
        self.exit_qty = 0
        self.exit_fees = 0.0
        self.last_reason = ""
        self.max_favourable_r = 0.0

    def add_exit(self, qty: int, price: float, fee: float, reason: str) -> None:
        self.exit_value += qty * price
        self.exit_qty += qty
        self.exit_fees += fee
        self.last_reason = reason

    def finalise(self, code: str, index: int, time_key) -> Trade:
        pos = self.position
        original = pos.original_qty
        avg_exit = self.exit_value / self.exit_qty if self.exit_qty else pos.entry_price
        cost = original * pos.entry_price + self.entry_fee
        pnl = self.exit_value - self.exit_fees - cost
        risk_total = pos.risk_per_share * original
        return Trade(
            code=code,
            entry_time=self.entry_time,
            entry_price=round(pos.entry_price, 4),
            exit_time=time_key,
            exit_price=round(avg_exit, 4),
            qty=original,
            pnl=round(pnl, 2),
            return_pct=round(pnl / cost * 100.0, 4) if cost else 0.0,
            bars_held=index - self.entry_index,
            entry_reason=self.entry_reason,
            exit_reason=self.last_reason,
            initial_stop=round(pos.initial_stop, 4),
            # R is measured on the money actually risked, so position sizing
            # cannot flatter or hide the underlying edge.
            r_multiple=round(pnl / risk_total, 4) if risk_total > 0 else None,
            scale_outs=pos.scale_outs,
            max_favourable_r=round(self.max_favourable_r, 4),
        )


def backtest_symbol(
    bars: pd.DataFrame,
    strategy: Strategy,
    risk: RiskManager,
    code: str = "?",
    initial_cash: float = 100_000.0,
    commission_per_share: float = 0.005,
    min_commission: float = 1.0,
    slippage_bps: float = 5.0,
    bar_type: str = "K_DAY",
    allow_shorts: bool = False,
    exits: ExitsConfig | None = None,
) -> BacktestResult:
    """Run one strategy over one symbol's bars."""
    if len(bars) <= strategy.warmup + 2:
        raise ValueError(
            f"{code}: need more than {strategy.warmup + 2} bars for "
            f"{strategy.name}, got {len(bars)}"
        )

    exits = exits or ExitsConfig(mode="simple", trail_method="none")
    ladder = ExitLadder(exits)
    bars_per_year = BARS_PER_YEAR.get(bar_type, 252)

    frame = strategy.compute(bars).reset_index(drop=True)
    if "atr" not in frame.columns:
        frame["atr"] = atr_indicator(
            frame["high"], frame["low"], frame["close"], risk.cfg.atr_period
        )
    if exits.trail_method == "swing_low":
        frame["swing_low"] = swing_lows(frame["low"], exits.swing_left, exits.swing_right)
    if risk.cfg.target_volatility_pct > 0 or risk.cfg.max_volatility_pct > 0:
        frame["realised_vol"] = realised_volatility(
            frame["close"], risk.cfg.vol_lookback, bars_per_year
        ) * 100.0

    slip = slippage_bps / 10_000.0
    cash = initial_cash
    open_trade: _OpenTrade | None = None
    trades: list[Trade] = []
    equity_points: list[float] = []
    bars_in_market = 0
    pending: tuple[str, float, float, str] | None = None

    for i in range(len(frame)):
        row = frame.iloc[i]
        bar = Bar(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
        ts = row["time_key"]
        atr_value = float(row["atr"]) if pd.notna(row["atr"]) else None
        swing = (
            float(row["swing_low"])
            if "swing_low" in frame.columns and pd.notna(row["swing_low"])
            else None
        )
        vol = (
            float(row["realised_vol"])
            if "realised_vol" in frame.columns and pd.notna(row["realised_vol"])
            else None
        )

        # 1. Execute whatever the previous bar decided, at this bar's open.
        if pending is not None:
            side, stop_hint, strength, reason = pending
            pending = None
            if side == "BUY" and open_trade is None:
                fill = bar.open * (1 + slip)
                sized = risk.size_position(
                    equity=cash, cash=cash, price=fill, strength=strength,
                    stop_price=stop_hint or None, atr_value=atr_value,
                    realised_vol=vol,
                )
                fee = _commission(sized.qty, commission_per_share, min_commission)
                if sized.ok and sized.qty * fill + fee <= cash:
                    position = Position(
                        entry_price=fill,
                        initial_stop=sized.stop_price,
                        qty=sized.qty,
                        target_price=sized.target_price if exits.mode == "simple" else 0.0,
                    )
                    open_trade = _OpenTrade(position, i, ts, reason, fee)
                    cash -= sized.qty * fill + fee
            elif side == "SELL" and open_trade is not None:
                cash, open_trade = _close_out(
                    open_trade, open_trade.position.qty, bar.open * (1 - slip), reason,
                    cash, trades, code, i, ts, commission_per_share, min_commission,
                )

        # 2. Protective exits, priced from this bar's range.
        if open_trade is not None:
            position = open_trade.position
            open_trade.max_favourable_r = max(
                open_trade.max_favourable_r, position.r_at(bar.high)
            )
            decision = ladder.evaluate(position, bar, atr=atr_value, swing_low=swing)
            if decision.should_exit:
                if not decision.is_full_exit:
                    position.scale_outs += 1
                cash, open_trade = _close_out(
                    open_trade, decision.exit_qty, decision.exit_price * (1 - slip),
                    decision.reason, cash, trades, code, i, ts,
                    commission_per_share, min_commission,
                )

        # 3. Ask the strategy for the next bar's instruction.
        if i >= strategy.warmup:
            signal = strategy.signal_at(frame, i)
            if signal.action is Action.BUY and open_trade is None:
                if risk.volatility_ok(vol)[0]:
                    pending = ("BUY", signal.stop_price or 0.0, signal.strength, signal.reason)
            elif signal.action is Action.SELL and open_trade is not None:
                pending = ("SELL", 0.0, signal.strength, signal.reason)

        held = open_trade.position.qty if open_trade else 0
        if held > 0:
            bars_in_market += 1
        equity_points.append(cash + held * bar.close)

    # Close anything still open at the last close so the curve is comparable.
    if open_trade is not None:
        last = frame.iloc[-1]
        cash, open_trade = _close_out(
            open_trade, open_trade.position.qty, float(last["close"]) * (1 - slip),
            "end of backtest", cash, trades, code, len(frame) - 1, last["time_key"],
            commission_per_share, min_commission,
        )
        equity_points[-1] = cash

    return BacktestResult(
        code=code,
        strategy=strategy.name,
        equity=pd.Series(equity_points, index=pd.to_datetime(frame["time_key"])),
        trades=trades,
        initial_cash=initial_cash,
        bars_per_year=bars_per_year,
        bars_in_market=bars_in_market,
        total_bars=len(frame),
    )


def _close_out(
    open_trade: _OpenTrade, qty: int, price: float, reason: str, cash: float,
    trades: list[Trade], code: str, index: int, time_key,
    per_share: float, min_commission: float,
) -> tuple[float, _OpenTrade | None]:
    """Sell `qty` units, appending to `trades` once the position reaches zero."""
    qty = min(qty, open_trade.position.qty)
    if qty <= 0:
        return cash, open_trade
    fee = _commission(qty, per_share, min_commission)
    open_trade.add_exit(qty, price, fee, reason)
    open_trade.position.qty -= qty
    cash += qty * price - fee
    if open_trade.position.qty <= 0:
        trades.append(open_trade.finalise(code, index, time_key))
        return cash, None
    return cash, open_trade


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    """Standard performance statistics for an equity curve plus its trades."""
    equity = result.equity.dropna()
    out: dict[str, float] = {
        "initial_cash": result.initial_cash,
        "final_equity": result.final_equity,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "annual_vol_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar": 0.0,
        "num_trades": float(len(result.trades)),
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
        "expectancy_r": 0.0,
        "avg_win_r": 0.0,
        "avg_loss_r": 0.0,
        "exposure_pct": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "scale_outs": 0.0,
    }
    if len(equity) < 2 or result.initial_cash <= 0:
        return out

    out["total_return_pct"] = (result.final_equity / result.initial_cash - 1) * 100.0

    returns = equity.pct_change().dropna()
    ppy = result.bars_per_year
    if len(returns) > 1:
        std = float(returns.std(ddof=1))
        mean = float(returns.mean())
        out["annual_vol_pct"] = std * math.sqrt(ppy) * 100.0
        if std > 0:
            out["sharpe"] = mean / std * math.sqrt(ppy)
        downside = returns[returns < 0]
        d_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        if d_std > 0:
            out["sortino"] = mean / d_std * math.sqrt(ppy)

    years = len(equity) / ppy
    if years > 0 and result.final_equity > 0:
        out["cagr_pct"] = ((result.final_equity / result.initial_cash) ** (1 / years) - 1) * 100.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    out["max_drawdown_pct"] = float(drawdown.min()) * 100.0
    if out["max_drawdown_pct"] < 0:
        out["calmar"] = out["cagr_pct"] / abs(out["max_drawdown_pct"])

    if result.total_bars:
        out["exposure_pct"] = result.bars_in_market / result.total_bars * 100.0

    pnls = np.array([t.pnl for t in result.trades], dtype=float)
    if pnls.size:
        wins, losses = pnls[pnls > 0], pnls[pnls < 0]
        out["win_rate_pct"] = wins.size / pnls.size * 100.0
        out["avg_win"] = float(wins.mean()) if wins.size else 0.0
        out["avg_loss"] = float(losses.mean()) if losses.size else 0.0
        out["expectancy"] = float(pnls.mean())
        gross_loss = float(-losses.sum())
        out["profit_factor"] = (
            float(wins.sum()) / gross_loss if gross_loss > 0
            else (float("inf") if wins.size else 0.0)
        )
        out["best_trade"] = float(pnls.max())
        out["worst_trade"] = float(pnls.min())
        out["scale_outs"] = float(sum(t.scale_outs for t in result.trades))

    r_values = np.array(result.r_multiples, dtype=float)
    if r_values.size:
        out["expectancy_r"] = float(r_values.mean())
        r_wins, r_losses = r_values[r_values > 0], r_values[r_values < 0]
        out["avg_win_r"] = float(r_wins.mean()) if r_wins.size else 0.0
        out["avg_loss_r"] = float(r_losses.mean()) if r_losses.size else 0.0
    return out


def combine(results: list[BacktestResult]) -> BacktestResult:
    """Merge per-symbol runs into one portfolio curve.

    Capital is split evenly across symbols up front and never rebalanced, so
    each sleeve is independent. Curves are forward-filled onto the union of all
    timestamps before being summed.
    """
    if not results:
        raise ValueError("nothing to combine")
    if len(results) == 1:
        return results[0]

    curves = [r.equity for r in results if len(r.equity)]
    merged = pd.concat(curves, axis=1).sort_index().ffill().bfill()
    portfolio = merged.sum(axis=1)

    return BacktestResult(
        code="PORTFOLIO(" + ", ".join(r.code for r in results) + ")",
        strategy=results[0].strategy,
        equity=portfolio,
        trades=[t for r in results for t in r.trades],
        initial_cash=sum(r.initial_cash for r in results),
        bars_per_year=results[0].bars_per_year,
        bars_in_market=sum(r.bars_in_market for r in results),
        total_bars=sum(r.total_bars for r in results),
    )


def r_histogram(r_multiples: list[float], width: int = 30) -> str:
    """Text histogram of R outcomes - the shape of the edge at a glance."""
    if not r_multiples:
        return "  (no closed trades)"
    buckets = [
        ("<= -2R", lambda r: r <= -2),
        ("-2R..-1R", lambda r: -2 < r <= -1),
        ("-1R..0R", lambda r: -1 < r < 0),
        ("0R..+1R", lambda r: 0 <= r < 1),
        ("+1R..+2R", lambda r: 1 <= r < 2),
        ("+2R..+3R", lambda r: 2 <= r < 3),
        (">= +3R", lambda r: r >= 3),
    ]
    counts = [(label, sum(1 for r in r_multiples if test(r))) for label, test in buckets]
    peak = max((c for _, c in counts), default=0) or 1
    lines = []
    for label, count in counts:
        bar = "#" * int(round(count / peak * width))
        lines.append(f"  {label:>9} | {bar:<{width}} {count}")
    return "\n".join(lines)


def format_metrics(result: BacktestResult) -> str:
    m = result.metrics()
    pf = m["profit_factor"]
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    return "\n".join(
        [
            f"  Symbol                {result.code}",
            f"  Strategy              {result.strategy}",
            f"  Bars                  {result.total_bars:,}",
            f"  Start / final equity  {m['initial_cash']:,.2f} -> {m['final_equity']:,.2f}",
            f"  Total return          {m['total_return_pct']:+.2f}%",
            f"  CAGR                  {m['cagr_pct']:+.2f}%",
            f"  Annualised volatility {m['annual_vol_pct']:.2f}%",
            f"  Sharpe / Sortino      {m['sharpe']:.2f} / {m['sortino']:.2f}",
            f"  Max drawdown          {m['max_drawdown_pct']:.2f}%",
            f"  Calmar                {m['calmar']:.2f}",
            f"  Time in market        {m['exposure_pct']:.1f}%",
            f"  Trades                {int(m['num_trades'])}"
            f"  ({int(m['scale_outs'])} scale-outs)",
            f"  Win rate              {m['win_rate_pct']:.1f}%",
            f"  Profit factor         {pf_text}",
            f"  Expectancy            {m['expectancy']:,.2f} per trade"
            f"  ({m['expectancy_r']:+.3f}R)",
            f"  Avg win / avg loss    {m['avg_win']:,.2f} / {m['avg_loss']:,.2f}"
            f"  ({m['avg_win_r']:+.2f}R / {m['avg_loss_r']:+.2f}R)",
            f"  Best / worst trade    {m['best_trade']:,.2f} / {m['worst_trade']:,.2f}",
        ]
    )
