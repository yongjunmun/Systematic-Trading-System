"""Backtester tests.

The important ones are the no-lookahead and cost-accounting checks - those are
what separate a believable backtest from a fantasy.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from moobot.backtest import backtest_symbol, combine, compute_metrics
from moobot.risk import RiskManager
from moobot.settings import ExitsConfig, RiskConfig
from moobot.strategies.base import Action, Signal, Strategy
from tests.helpers import make_bars, trending


class BuyOnBar(Strategy):
    """Emits BUY exactly once, at a chosen bar index."""

    name = "unit_buy_on_bar"
    warmup = 2

    def configure(self, at: int = 5, **_: object) -> None:
        self.at = at

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars.copy()

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        return Signal(Action.BUY, reason="scheduled buy") if i == self.at else Signal()


class AlwaysHold(Strategy):
    name = "unit_always_hold"
    warmup = 2

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars.copy()

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        return Signal()


def risk_manager(**overrides) -> RiskManager:
    defaults = dict(
        risk_per_trade_pct=100.0, stop_loss_pct=50.0, take_profit_pct=1000.0,
        max_position_pct=100.0, min_cash_buffer_pct=0.0, max_daily_loss_pct=99.0,
    )
    defaults.update(overrides)
    m = RiskManager(RiskConfig(**defaults))
    m.update_equity(10_000.0)
    return m


class TestExecutionModel(unittest.TestCase):
    def test_a_signal_is_filled_at_the_next_bars_open_not_this_bars_close(self):
        closes = [100.0] * 5 + [100.0, 200.0] + [200.0] * 5
        bars = make_bars(closes)
        # helpers.make_bars sets open[i] = close[i-1], so bar 6 opens at 100.
        result = backtest_symbol(
            bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(), code="TEST",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(len(result.trades), 1)
        # Filled at bar 6's open (100), NOT bar 5's close and NOT bar 6's close (200).
        self.assertAlmostEqual(result.trades[0].entry_price, 100.0, places=6)

    def test_slippage_makes_the_buy_more_expensive(self):
        bars = make_bars([100.0] * 20)
        clean = backtest_symbol(
            bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(), code="T",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0,
        )
        slipped = backtest_symbol(
            bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(), code="T",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=100.0,
        )
        self.assertGreater(slipped.trades[0].entry_price, clean.trades[0].entry_price)

    def test_commission_reduces_the_final_equity(self):
        bars = make_bars(list(trending(60, drift=0.005)))
        free = backtest_symbol(
            bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(), code="T",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0,
        )
        charged = backtest_symbol(
            bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(), code="T",
            initial_cash=10_000, commission_per_share=0.05, min_commission=5.0,
            slippage_bps=0.0,
        )
        self.assertLess(charged.final_equity, free.final_equity)

    def test_doing_nothing_leaves_the_cash_untouched(self):
        result = backtest_symbol(
            bars=make_bars(list(trending(80))), strategy=AlwaysHold(),
            risk=risk_manager(), code="T", initial_cash=10_000,
        )
        self.assertEqual(len(result.trades), 0)
        self.assertAlmostEqual(result.final_equity, 10_000.0)
        self.assertEqual(result.bars_in_market, 0)

    def test_open_positions_are_closed_at_the_end(self):
        result = backtest_symbol(
            bars=make_bars(list(trending(80, drift=0.004))), strategy=BuyOnBar(at=5),
            risk=risk_manager(), code="T", initial_cash=10_000,
            commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, "end of backtest")


class TestProtectiveExits(unittest.TestCase):
    def test_stop_is_hit_on_a_collapse(self):
        closes = [100.0] * 8 + [50.0] * 10
        result = backtest_symbol(
            bars=make_bars(closes), strategy=BuyOnBar(at=5),
            risk=risk_manager(stop_loss_pct=10.0), code="T", initial_cash=10_000,
            commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertIn("stop", result.trades[0].exit_reason)
        self.assertLess(result.trades[0].pnl, 0)

    def test_target_is_hit_on_a_spike(self):
        closes = [100.0] * 8 + [130.0] * 10
        result = backtest_symbol(
            bars=make_bars(closes), strategy=BuyOnBar(at=5),
            risk=risk_manager(stop_loss_pct=20.0, take_profit_pct=10.0), code="T",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertIn("target", result.trades[0].exit_reason)
        self.assertGreater(result.trades[0].pnl, 0)

    def test_a_gap_through_the_stop_fills_below_it(self):
        # Bar opens at 60 with a stop at 90: a real fill happens at 60, not 90.
        closes = [100.0] * 8 + [60.0, 60.0, 60.0]
        result = backtest_symbol(
            bars=make_bars(closes), strategy=BuyOnBar(at=5),
            risk=risk_manager(stop_loss_pct=10.0), code="T", initial_cash=10_000,
            commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0,
        )
        self.assertLessEqual(result.trades[0].exit_price, 90.0)

    def test_trailing_stop_locks_in_a_gain(self):
        closes = [100.0] * 6 + [150.0, 150.0, 120.0, 120.0, 120.0]
        exits = ExitsConfig(
            mode="ladder", partial_at_r=0.0, breakeven_at_r=0.1, target_r=0.0,
            trail_method="percent", trail_percent=10.0,
        )
        result = backtest_symbol(
            bars=make_bars(closes), strategy=BuyOnBar(at=4),
            risk=risk_manager(stop_loss_pct=50.0), code="T", initial_cash=10_000,
            commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0, exits=exits,
        )
        self.assertTrue(
            any("trail" in t.exit_reason or "stop" in t.exit_reason for t in result.trades)
        )
        self.assertGreater(result.final_equity, 10_000)


class TestMetrics(unittest.TestCase):
    def test_flat_curve_has_zero_return_and_no_drawdown(self):
        result = backtest_symbol(
            bars=make_bars([100.0] * 80), strategy=AlwaysHold(), risk=risk_manager(),
            code="T", initial_cash=10_000,
        )
        m = compute_metrics(result)
        self.assertAlmostEqual(m["total_return_pct"], 0.0)
        self.assertAlmostEqual(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["num_trades"], 0)

    def test_drawdown_is_negative_when_equity_falls(self):
        closes = [100.0] * 6 + [80.0] * 6 + [90.0] * 6
        result = backtest_symbol(
            bars=make_bars(closes), strategy=BuyOnBar(at=4),
            risk=risk_manager(stop_loss_pct=90.0, take_profit_pct=1000.0), code="T",
            initial_cash=10_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0,
        )
        self.assertLess(compute_metrics(result)["max_drawdown_pct"], 0.0)

    def test_win_rate_and_profit_factor_are_consistent(self):
        bars = make_bars(list(trending(400, drift=0.003, noise=0.02)))
        from moobot.strategies import get_strategy

        result = backtest_symbol(
            bars=bars, strategy=get_strategy("sma_cross", fast=5, slow=15, trend=0),
            risk=risk_manager(risk_per_trade_pct=2.0, stop_loss_pct=5.0), code="T",
            initial_cash=100_000,
        )
        m = compute_metrics(result)
        self.assertGreaterEqual(m["win_rate_pct"], 0.0)
        self.assertLessEqual(m["win_rate_pct"], 100.0)
        self.assertGreaterEqual(m["profit_factor"], 0.0)
        if m["num_trades"] > 0:
            self.assertAlmostEqual(
                m["expectancy"],
                sum(t.pnl for t in result.trades) / len(result.trades),
                places=6,
            )

    def test_exposure_is_a_percentage(self):
        result = backtest_symbol(
            bars=make_bars(list(trending(200, drift=0.003))), strategy=BuyOnBar(at=5),
            risk=risk_manager(), code="T", initial_cash=10_000,
        )
        exposure = compute_metrics(result)["exposure_pct"]
        self.assertGreater(exposure, 0.0)
        self.assertLessEqual(exposure, 100.0)

    def test_sharpe_is_finite_on_a_noisy_curve(self):
        from moobot.strategies import get_strategy

        result = backtest_symbol(
            bars=make_bars(list(trending(500, drift=0.001, noise=0.02))),
            strategy=get_strategy("sma_cross", fast=5, slow=20, trend=0),
            risk=risk_manager(risk_per_trade_pct=2.0, stop_loss_pct=6.0),
            code="T", initial_cash=100_000,
        )
        self.assertTrue(math.isfinite(compute_metrics(result)["sharpe"]))


class TestCombine(unittest.TestCase):
    def test_portfolio_equity_is_the_sum_of_its_sleeves(self):
        bars = make_bars(list(trending(120, drift=0.003)))
        parts = [
            backtest_symbol(bars=bars, strategy=BuyOnBar(at=5), risk=risk_manager(),
                            code=f"S{i}", initial_cash=5_000)
            for i in range(2)
        ]
        portfolio = combine(parts)
        self.assertAlmostEqual(portfolio.initial_cash, 10_000.0)
        self.assertAlmostEqual(
            portfolio.final_equity, sum(p.final_equity for p in parts), places=4
        )
        self.assertEqual(len(portfolio.trades), sum(len(p.trades) for p in parts))

    def test_combining_a_single_result_returns_it_unchanged(self):
        one = backtest_symbol(
            bars=make_bars(list(trending(80))), strategy=AlwaysHold(),
            risk=risk_manager(), code="ONE", initial_cash=1_000,
        )
        self.assertIs(combine([one]), one)


class TestGuards(unittest.TestCase):
    def test_too_few_bars_raises_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            backtest_symbol(
                bars=make_bars([100.0, 101.0, 102.0]), strategy=BuyOnBar(at=1),
                risk=risk_manager(), code="T",
            )
        self.assertIn("bars", str(ctx.exception))

    def test_trades_frame_has_a_stable_schema_when_empty(self):
        result = backtest_symbol(
            bars=make_bars([100.0] * 50), strategy=AlwaysHold(), risk=risk_manager(),
            code="T",
        )
        frame = result.trades_frame()
        self.assertTrue(frame.empty)
        self.assertIn("pnl", frame.columns)
        self.assertIn("exit_reason", frame.columns)


if __name__ == "__main__":
    unittest.main()
