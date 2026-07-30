"""Strategy behaviour tests on constructed price paths."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from moobot.strategies import Action, available, get_strategy
from moobot.strategies.base import Signal, Strategy
from tests.helpers import make_bars, trending, v_shape


class TestRegistry(unittest.TestCase):
    def test_every_builtin_is_registered(self):
        for name in ("sma_cross", "rsi_reversion", "donchian_breakout", "macd_momentum"):
            self.assertIn(name, available())

    def test_unknown_strategy_lists_the_valid_ones(self):
        with self.assertRaises(KeyError) as ctx:
            get_strategy("does_not_exist")
        self.assertIn("sma_cross", str(ctx.exception))

    def test_all_strategies_return_hold_before_warmup(self):
        bars = make_bars(trending(20))
        for name in available():
            strategy = get_strategy(name)
            self.assertIs(strategy.latest_signal(bars).action, Action.HOLD, name)

    def test_all_strategies_produce_a_signal_on_real_length_data(self):
        bars = make_bars(trending(600, noise=0.01))
        for name in available():
            signal = get_strategy(name).latest_signal(bars)
            self.assertIsInstance(signal, Signal, name)
            self.assertIn(signal.action, list(Action), name)
            self.assertTrue(signal.reason, f"{name} gave no reason")


class TestSignalObject(unittest.TestCase):
    def test_strength_is_clamped(self):
        self.assertEqual(Signal(strength=5.0).strength, 1.0)
        self.assertEqual(Signal(strength=-2.0).strength, 0.0)


class TestSmaCross(unittest.TestCase):
    def test_rejects_fast_slower_than_slow(self):
        with self.assertRaises(ValueError):
            get_strategy("sma_cross", fast=50, slow=20)

    def test_buys_when_a_downtrend_turns_up(self):
        # Long decline then a sustained rally forces a golden cross.
        decline = 100 * np.cumprod(np.full(220, 0.998))
        rally = decline[-1] * np.cumprod(np.full(120, 1.004))
        bars = make_bars(np.concatenate([decline, rally]))
        strategy = get_strategy("sma_cross", fast=10, slow=30, trend=0)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.BUY, actions)

    def test_trend_filter_suppresses_buys_below_the_long_average(self):
        closes = 100 * np.cumprod(np.full(400, 0.997))
        # A short bounce inside a downtrend: cross up happens, trend blocks it.
        closes = np.concatenate([closes, closes[-1] * np.cumprod(np.full(40, 1.01))])
        bars = make_bars(closes)
        filtered = get_strategy("sma_cross", fast=5, slow=15, trend=200)
        unfiltered = get_strategy("sma_cross", fast=5, slow=15, trend=0)

        def buys(strategy):
            frame = strategy.compute(bars)
            return sum(
                strategy.signal_at(frame, i).action is Action.BUY
                for i in range(strategy.warmup, len(frame))
            )

        self.assertLess(buys(filtered), buys(unfiltered))

    def test_sells_on_a_death_cross(self):
        up = 100 * np.cumprod(np.full(150, 1.004))
        down = up[-1] * np.cumprod(np.full(120, 0.995))
        bars = make_bars(np.concatenate([up, down]))
        strategy = get_strategy("sma_cross", fast=10, slow=30, trend=0)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.SELL, actions)

    def test_buy_signal_carries_an_atr_stop_below_price(self):
        decline = 100 * np.cumprod(np.full(120, 0.998))
        path = np.concatenate([decline, decline[-1] * np.cumprod(np.full(100, 1.005))])
        bars = make_bars(path)
        strategy = get_strategy("sma_cross", fast=10, slow=30, trend=0)
        frame = strategy.compute(bars)
        for i in range(strategy.warmup, len(frame)):
            signal = strategy.signal_at(frame, i)
            if signal.action is Action.BUY and signal.stop_price:
                self.assertLess(signal.stop_price, float(frame.iloc[i]["close"]))
                return
        self.skipTest("no BUY with an ATR stop was generated on this path")


class TestRsiReversion(unittest.TestCase):
    def test_rejects_inverted_thresholds(self):
        with self.assertRaises(ValueError):
            get_strategy("rsi_reversion", oversold=70, exit_level=30)

    def test_buys_the_dip_inside_an_uptrend(self):
        closes = np.concatenate([
            trending(220, drift=0.004),                 # establish the uptrend
            trending(220, drift=0.004)[-1] * np.cumprod(np.full(12, 0.985)),  # sharp dip
        ])
        bars = make_bars(closes)
        strategy = get_strategy("rsi_reversion", period=14, oversold=35, trend=100)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.BUY, actions)

    def test_will_not_catch_a_falling_knife(self):
        bars = make_bars(100 * np.cumprod(np.full(400, 0.99)))
        strategy = get_strategy("rsi_reversion", period=14, oversold=30, trend=100)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertNotIn(Action.BUY, actions)

    def test_sells_when_overbought(self):
        bars = make_bars(v_shape(down=80, up=200, rate=0.012))
        strategy = get_strategy("rsi_reversion", period=14, overbought=70, trend=0)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.SELL, actions)


class TestDonchianBreakout(unittest.TestCase):
    def test_breakout_needs_confirming_volume(self):
        closes = np.concatenate([np.full(80, 100.0), np.array([115.0]), np.full(10, 116.0)])
        quiet = make_bars(closes, volume=[1_000_000.0] * len(closes))
        strategy = get_strategy("donchian_breakout", entry_period=20, exit_period=10,
                                min_volume_ratio=3.0)
        frame = strategy.compute(quiet)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertNotIn(Action.BUY, actions)

    def test_breakout_on_heavy_volume_is_taken(self):
        closes = np.concatenate([np.full(80, 100.0), np.array([115.0]), np.full(10, 116.0)])
        volumes = [1_000_000.0] * 80 + [9_000_000.0] * 11
        bars = make_bars(closes, volume=volumes)
        strategy = get_strategy("donchian_breakout", entry_period=20, exit_period=10,
                                min_volume_ratio=1.2)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.BUY, actions)

    def test_breakdown_produces_a_sell(self):
        closes = np.concatenate([np.full(80, 100.0), np.full(10, 70.0)])
        bars = make_bars(closes)
        strategy = get_strategy("donchian_breakout", entry_period=20, exit_period=10)
        frame = strategy.compute(bars)
        actions = [strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))]
        self.assertIn(Action.SELL, actions)


class TestMacdMomentum(unittest.TestCase):
    def test_rejects_fast_slower_than_slow(self):
        with self.assertRaises(ValueError):
            get_strategy("macd_momentum", fast=26, slow=12)

    def test_crosses_generate_both_directions(self):
        # Up, down, up - a plain V only ever produces the upward cross.
        leg1 = 100 * np.cumprod(np.full(150, 1.006))
        leg2 = leg1[-1] * np.cumprod(np.full(150, 0.994))
        leg3 = leg2[-1] * np.cumprod(np.full(150, 1.006))
        bars = make_bars(np.concatenate([leg1, leg2, leg3]))
        strategy = get_strategy("macd_momentum", trend=0)
        frame = strategy.compute(bars)
        actions = {strategy.signal_at(frame, i).action for i in range(strategy.warmup, len(frame))}
        self.assertIn(Action.BUY, actions)
        self.assertIn(Action.SELL, actions)


class TestDeterminism(unittest.TestCase):
    def test_same_input_gives_the_same_signal(self):
        bars = make_bars(trending(400, noise=0.01))
        for name in available():
            a = get_strategy(name).latest_signal(bars)
            b = get_strategy(name).latest_signal(bars.copy())
            self.assertEqual((a.action, a.reason), (b.action, b.reason), name)

    def test_a_strategy_never_sees_future_bars(self):
        """Truncating the data must not change the signal at the truncation point."""
        full = make_bars(trending(500, noise=0.012))
        cut = 400
        for name in available():
            strategy = get_strategy(name)
            whole = strategy.signal_at(strategy.compute(full), cut - 1)
            partial = get_strategy(name)
            truncated = partial.signal_at(partial.compute(full.iloc[:cut].copy()), cut - 1)
            self.assertEqual(whole.action, truncated.action, f"{name} peeked into the future")


class CustomStrategy(Strategy):
    name = "unit_test_custom"
    warmup = 3

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars.copy()

    def signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        return Signal(Action.BUY, reason="always")


class TestExtensibility(unittest.TestCase):
    def test_a_user_strategy_plugs_into_the_same_interface(self):
        bars = make_bars([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIs(CustomStrategy().latest_signal(bars).action, Action.BUY)


if __name__ == "__main__":
    unittest.main()
