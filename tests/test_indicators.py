"""Indicator correctness tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from moobot import indicators as ind


class TestMovingAverages(unittest.TestCase):
    def test_sma_matches_hand_calculation(self):
        s = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
        out = ind.sma(s, 3)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertAlmostEqual(out.iloc[2], 2.0)
        self.assertAlmostEqual(out.iloc[5], 5.0)

    def test_ema_uses_two_over_n_plus_one(self):
        s = pd.Series([10.0, 11.0, 12.0, 13.0])
        out = ind.ema(s, 3)
        # pandas seeds the recursion at the first observation rather than with
        # an SMA, so alpha = 2/(3+1) is applied from bar 1 onwards:
        #   10 -> 10.5 -> 11.25 -> 12.125
        # min_periods only masks the first two values.
        self.assertAlmostEqual(out.iloc[2], 11.25, places=6)
        self.assertAlmostEqual(out.iloc[3], 11.25 + 0.5 * (13.0 - 11.25), places=6)

    def test_sma_needs_full_window(self):
        out = ind.sma(pd.Series(range(10), dtype=float), 5)
        self.assertEqual(int(out.notna().sum()), 6)


class TestRsi(unittest.TestCase):
    def test_all_gains_pins_to_100(self):
        s = pd.Series(np.arange(1, 40, dtype=float))
        out = ind.rsi(s, 14).dropna()
        self.assertTrue((out == 100.0).all(), f"expected 100, got {out.unique()}")

    def test_all_losses_pins_to_zero(self):
        s = pd.Series(np.arange(60, 20, -1, dtype=float))
        out = ind.rsi(s, 14).dropna()
        self.assertTrue((out < 1e-9).all(), f"expected ~0, got {out.unique()}")

    def test_stays_within_bounds(self):
        rng = np.random.default_rng(3)
        s = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 500)))
        out = ind.rsi(s, 14).dropna()
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 100.0)

    def test_flat_series_is_neutral_or_undefined(self):
        out = ind.rsi(pd.Series([50.0] * 40), 14).dropna()
        # No gains and no losses: implementation defines this as 100.
        self.assertTrue(((out == 100.0) | out.isna()).all())


class TestAtr(unittest.TestCase):
    def test_true_range_includes_gaps(self):
        high = pd.Series([10.0, 20.0])
        low = pd.Series([9.0, 19.0])
        close = pd.Series([9.5, 19.5])
        tr = ind.true_range(high, low, close)
        # Second bar gapped up from 9.5 to a 19-20 range: TR = 20 - 9.5.
        self.assertAlmostEqual(tr.iloc[1], 10.5)

    def test_atr_of_constant_range_equals_that_range(self):
        n = 60
        close = pd.Series([100.0] * n)
        high = pd.Series([101.0] * n)
        low = pd.Series([99.0] * n)
        out = ind.atr(high, low, close, 14).dropna()
        self.assertAlmostEqual(out.iloc[-1], 2.0, places=6)

    def test_atr_is_never_negative(self):
        rng = np.random.default_rng(11)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
        out = ind.atr(close * 1.01, close * 0.99, close, 14).dropna()
        self.assertGreaterEqual(out.min(), 0.0)


class TestChannelsAndCrosses(unittest.TestCase):
    def test_donchian_does_not_peek_at_the_current_bar(self):
        # Bar 5 makes a huge new high. The channel evaluated ON bar 5 must not
        # already contain it, otherwise a breakout can never be detected.
        high = pd.Series([10, 10, 10, 10, 10, 99, 10], dtype=float)
        low = pd.Series([9] * 7, dtype=float)
        upper, _ = ind.donchian(high, low, 3)
        self.assertAlmostEqual(upper.iloc[5], 10.0)
        self.assertAlmostEqual(upper.iloc[6], 99.0)

    def test_crossed_above_fires_once(self):
        fast = pd.Series([1.0, 1.0, 3.0, 4.0, 5.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
        out = ind.crossed_above(fast, slow)
        self.assertEqual(list(out), [False, False, True, False, False])

    def test_crossed_below_fires_once(self):
        fast = pd.Series([5.0, 5.0, 1.0, 0.5, 0.2])
        slow = pd.Series([2.0] * 5)
        out = ind.crossed_below(fast, slow)
        self.assertEqual(list(out), [False, False, True, False, False])

    def test_touching_without_crossing_is_not_a_cross(self):
        fast = pd.Series([1.0, 2.0, 1.0])
        slow = pd.Series([2.0, 2.0, 2.0])
        self.assertFalse(ind.crossed_above(fast, slow).any())


class TestBollingerAndZScore(unittest.TestCase):
    def test_bands_straddle_the_middle(self):
        rng = np.random.default_rng(5)
        s = pd.Series(100 + rng.normal(0, 2, 200))
        upper, mid, lower = ind.bollinger(s, 20, 2.0)
        valid = mid.notna()
        self.assertTrue((upper[valid] >= mid[valid]).all())
        self.assertTrue((lower[valid] <= mid[valid]).all())

    def test_zscore_of_constant_series_is_undefined(self):
        out = ind.zscore(pd.Series([5.0] * 50), 20).dropna()
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
