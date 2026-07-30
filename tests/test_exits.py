"""Exit ladder tests: scale-out, breakeven, trailing and the R arithmetic."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from moobot.exits import Bar, ExitLadder, Position, swing_lows
from moobot.settings import ExitsConfig


def ladder(**overrides) -> ExitLadder:
    defaults = dict(
        mode="ladder", partial_at_r=0.75, partial_fraction=0.5,
        breakeven_at_r=1.0, target_r=3.0, trail_method="none",
    )
    defaults.update(overrides)
    return ExitLadder(ExitsConfig(**defaults))


def position(entry=100.0, stop=90.0, qty=100) -> Position:
    return Position(entry_price=entry, initial_stop=stop, qty=qty)


class TestRArithmetic(unittest.TestCase):
    def test_one_r_is_the_distance_to_the_initial_stop(self):
        pos = position(entry=100.0, stop=90.0)
        self.assertAlmostEqual(pos.risk_per_share, 10.0)
        self.assertAlmostEqual(pos.r_at(110.0), 1.0)
        self.assertAlmostEqual(pos.r_at(90.0), -1.0)
        self.assertAlmostEqual(pos.r_at(130.0), 3.0)

    def test_price_at_r_is_the_inverse(self):
        pos = position(entry=50.0, stop=45.0)
        self.assertAlmostEqual(pos.price_at_r(2.0), 60.0)
        self.assertAlmostEqual(pos.r_at(pos.price_at_r(1.7)), 1.7)

    def test_r_denominator_does_not_move_when_the_stop_trails(self):
        pos = position(entry=100.0, stop=90.0)
        pos.stop_price = 105.0  # trailed well above entry
        self.assertAlmostEqual(pos.risk_per_share, 10.0)
        self.assertAlmostEqual(pos.r_at(120.0), 2.0)

    def test_defaults_fill_themselves_in(self):
        pos = Position(entry_price=10.0, initial_stop=9.0, qty=50)
        self.assertEqual(pos.stop_price, 9.0)
        self.assertEqual(pos.high_water, 10.0)
        self.assertEqual(pos.original_qty, 50)


class TestStop(unittest.TestCase):
    def test_stop_exits_the_whole_position(self):
        pos = position()
        decision = ladder().evaluate(pos, Bar(100, 100, 89, 89))
        self.assertTrue(decision.is_full_exit)
        self.assertEqual(decision.exit_qty, 100)
        self.assertAlmostEqual(decision.exit_price, 90.0)

    def test_a_gap_below_the_stop_fills_at_the_open(self):
        pos = position()
        decision = ladder().evaluate(pos, Bar(70, 72, 68, 71))
        self.assertAlmostEqual(decision.exit_price, 70.0)
        self.assertIn("-3.00R", decision.reason)

    def test_stop_wins_when_a_bar_covers_both_stop_and_target(self):
        pos = position()
        # This bar reaches the 3R target (130) and the stop (90).
        decision = ladder().evaluate(pos, Bar(100, 135, 88, 130))
        self.assertIn("stop", decision.reason)
        self.assertTrue(decision.is_full_exit)

    def test_no_action_inside_the_range(self):
        self.assertFalse(ladder().evaluate(position(), Bar(100, 103, 98, 101)).should_exit)


class TestScaleOut(unittest.TestCase):
    def test_partial_fires_at_the_configured_r(self):
        pos = position(qty=100)
        decision = ladder(partial_at_r=0.75, partial_fraction=0.5).evaluate(
            pos, Bar(100, 108, 99, 107)  # 0.8R
        )
        self.assertTrue(decision.should_exit)
        self.assertFalse(decision.is_full_exit)
        self.assertEqual(decision.exit_qty, 50)
        self.assertAlmostEqual(decision.exit_price, 107.5)

    def test_partial_only_happens_once(self):
        engine = ladder()
        pos = position(qty=100)
        engine.evaluate(pos, Bar(100, 108, 99, 107))
        pos.qty -= 50
        second = engine.evaluate(pos, Bar(107, 109, 106, 108))
        self.assertFalse(second.should_exit)

    def test_partial_never_sells_the_last_share(self):
        # fraction 0.99 of 2 shares rounds up to 2, which would flatten the
        # position on what is supposed to be a partial. It must cap at qty - 1.
        pos = position(qty=2)
        decision = ladder(partial_fraction=0.99).evaluate(pos, Bar(100, 108, 99, 107))
        self.assertEqual(decision.exit_qty, 1)
        self.assertFalse(decision.is_full_exit)

    def test_a_single_share_position_cannot_scale_out(self):
        decision = ladder().evaluate(position(qty=1), Bar(100, 108, 99, 107))
        self.assertFalse(decision.should_exit)

    def test_partial_disabled_when_set_to_zero(self):
        decision = ladder(partial_at_r=0.0).evaluate(position(), Bar(100, 108, 99, 107))
        self.assertFalse(decision.should_exit)


class TestBreakeven(unittest.TestCase):
    def test_stop_moves_to_entry_at_one_r(self):
        pos = position()
        decision = ladder().evaluate(pos, Bar(100, 111, 99, 110))
        self.assertAlmostEqual(pos.stop_price, 100.0)
        self.assertTrue(pos.breakeven_done)
        self.assertTrue(any("breakeven" in e for e in decision.events))

    def test_breakeven_and_partial_can_fire_on_the_same_bar(self):
        pos = position(qty=100)
        decision = ladder().evaluate(pos, Bar(100, 115, 99, 114))
        self.assertGreater(decision.exit_qty, 0)
        self.assertTrue(pos.breakeven_done)
        self.assertAlmostEqual(pos.stop_price, 100.0)

    def test_a_breakeven_stop_then_protects_the_trade(self):
        engine = ladder()
        pos = position()
        engine.evaluate(pos, Bar(100, 111, 99, 110))     # flips to breakeven
        decision = engine.evaluate(pos, Bar(108, 108, 99, 99))  # comes back down
        self.assertTrue(decision.is_full_exit)
        self.assertAlmostEqual(decision.exit_price, 100.0)

    def test_disabled_when_set_to_zero(self):
        pos = position()
        ladder(breakeven_at_r=0.0, partial_at_r=0.0).evaluate(pos, Bar(100, 111, 99, 110))
        self.assertAlmostEqual(pos.stop_price, 90.0)


class TestTarget(unittest.TestCase):
    def test_target_closes_everything(self):
        decision = ladder(target_r=3.0).evaluate(position(), Bar(100, 131, 99, 130))
        self.assertTrue(decision.is_full_exit)
        self.assertAlmostEqual(decision.exit_price, 130.0)

    def test_no_target_means_the_trade_runs(self):
        decision = ladder(target_r=0.0, partial_at_r=0.0).evaluate(
            position(), Bar(100, 200, 99, 195)
        )
        self.assertFalse(decision.should_exit)

    def test_simple_mode_uses_the_absolute_target(self):
        engine = ExitLadder(ExitsConfig(mode="simple", trail_method="none"))
        pos = Position(entry_price=100, initial_stop=95, qty=10, target_price=110)
        decision = engine.evaluate(pos, Bar(100, 112, 99, 111))
        self.assertTrue(decision.is_full_exit)
        self.assertAlmostEqual(decision.exit_price, 110.0)

    def test_simple_mode_never_scales_out(self):
        engine = ExitLadder(ExitsConfig(mode="simple", trail_method="none"))
        pos = Position(entry_price=100, initial_stop=90, qty=100, target_price=500)
        self.assertFalse(engine.evaluate(pos, Bar(100, 108, 99, 107)).should_exit)


class TestTrailing(unittest.TestCase):
    def test_atr_trail_follows_the_high_water_mark(self):
        engine = ladder(trail_method="atr", trail_atr_multiple=2.0, target_r=0.0)
        pos = position()
        engine.evaluate(pos, Bar(100, 120, 99, 119), atr=3.0)
        self.assertAlmostEqual(pos.stop_price, 114.0)

    def test_percent_trail(self):
        engine = ladder(trail_method="percent", trail_percent=10.0, target_r=0.0)
        pos = position()
        engine.evaluate(pos, Bar(100, 120, 99, 119))
        self.assertAlmostEqual(pos.stop_price, 108.0)

    def test_the_stop_only_ever_ratchets_up(self):
        engine = ladder(trail_method="percent", trail_percent=10.0, target_r=0.0)
        pos = position()
        engine.evaluate(pos, Bar(100, 120, 99, 119))
        raised = pos.stop_price
        engine.evaluate(pos, Bar(119, 119, 112, 113))  # pulls back but no new high
        self.assertAlmostEqual(pos.stop_price, raised)

    def test_trailing_waits_for_breakeven_by_default(self):
        engine = ladder(trail_method="percent", trail_percent=1.0,
                        breakeven_at_r=5.0, target_r=0.0, partial_at_r=0.0)
        pos = position()
        engine.evaluate(pos, Bar(100, 105, 99, 104))
        self.assertAlmostEqual(pos.stop_price, 90.0)

    def test_trailing_can_start_immediately(self):
        engine = ladder(trail_method="percent", trail_percent=1.0, breakeven_at_r=5.0,
                        target_r=0.0, partial_at_r=0.0, trail_only_after_breakeven=False)
        pos = position()
        engine.evaluate(pos, Bar(100, 105, 99, 104))
        self.assertAlmostEqual(pos.stop_price, 103.95)

    def test_swing_low_trail(self):
        engine = ladder(trail_method="swing_low", target_r=0.0, breakeven_at_r=0.1)
        pos = position()
        engine.evaluate(pos, Bar(100, 120, 99, 119), swing_low=110.0)
        self.assertAlmostEqual(pos.stop_price, 110.0 * 0.999)

    def test_atr_trail_abstains_without_an_atr(self):
        engine = ladder(trail_method="atr", target_r=0.0, breakeven_at_r=0.1)
        pos = position()
        engine.evaluate(pos, Bar(100, 120, 99, 119), atr=None)
        self.assertAlmostEqual(pos.stop_price, 100.0)  # breakeven only

    def test_none_disables_trailing(self):
        engine = ladder(trail_method="none", target_r=0.0, breakeven_at_r=0.0,
                        partial_at_r=0.0)
        pos = position()
        engine.evaluate(pos, Bar(100, 150, 99, 149))
        self.assertAlmostEqual(pos.stop_price, 90.0)


class TestSwingLows(unittest.TestCase):
    def test_identifies_a_pivot(self):
        lows = pd.Series([10, 9, 5, 9, 10, 11, 12], dtype=float)
        out = swing_lows(lows, left=2, right=2)
        # The pivot at index 2 is only confirmed two bars later.
        self.assertTrue(pd.isna(out.iloc[2]))
        self.assertAlmostEqual(out.iloc[4], 5.0)

    def test_confirmation_lag_prevents_lookahead(self):
        """Truncating after the pivot must not reveal it early."""
        lows = pd.Series([10, 9, 5, 9, 10, 11, 12], dtype=float)
        full = swing_lows(lows, left=2, right=2)
        partial = swing_lows(lows.iloc[:4], left=2, right=2)
        for i in range(4):
            a, b = full.iloc[i], partial.iloc[i]
            self.assertTrue((pd.isna(a) and pd.isna(b)) or a == b)

    def test_forward_fills_until_a_new_pivot(self):
        lows = pd.Series([10, 9, 5, 9, 10, 11, 12, 13], dtype=float)
        out = swing_lows(lows, left=2, right=2)
        self.assertAlmostEqual(out.iloc[-1], 5.0)

    def test_no_pivot_gives_all_nan(self):
        out = swing_lows(pd.Series(np.arange(20, dtype=float)), left=2, right=2)
        self.assertEqual(int(out.notna().sum()), 0)


class TestSequence(unittest.TestCase):
    def test_a_full_winning_lifecycle(self):
        """Partial at 0.75R, breakeven at 1R, trail up, stopped out in profit."""
        engine = ladder(trail_method="percent", trail_percent=10.0, target_r=0.0)
        pos = position(qty=100)

        first = engine.evaluate(pos, Bar(100, 108, 99, 107))
        self.assertEqual(first.exit_qty, 50)
        pos.qty -= first.exit_qty

        engine.evaluate(pos, Bar(107, 130, 106, 129))
        self.assertTrue(pos.breakeven_done)
        self.assertAlmostEqual(pos.stop_price, 117.0)  # 10% below the 130 high

        final = engine.evaluate(pos, Bar(129, 129, 110, 112))
        self.assertTrue(final.is_full_exit)
        self.assertEqual(final.exit_qty, 50)
        self.assertGreater(final.exit_price, pos.entry_price)

    def test_a_losing_trade_stops_at_minus_one_r(self):
        pos = position(qty=100)
        decision = ladder().evaluate(pos, Bar(99, 99, 89, 90))
        self.assertAlmostEqual(pos.r_at(decision.exit_price), -1.0)


if __name__ == "__main__":
    unittest.main()
