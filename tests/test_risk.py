"""Risk manager tests - sizing arithmetic, the kill switch and exit levels."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from moobot.risk import RiskManager
from moobot.settings import RiskConfig


def manager(**overrides) -> RiskManager:
    cfg = RiskConfig(**overrides)
    m = RiskManager(cfg)
    m.update_equity(100_000.0)
    return m


class TestSizing(unittest.TestCase):
    def test_quantity_follows_the_distance_to_the_stop(self):
        # 1% of 100k = 1000 risk budget. Stop 5% below a 100 price = 5 per share.
        m = manager(risk_per_trade_pct=1.0, stop_loss_pct=5.0, max_position_pct=100.0)
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0)
        self.assertEqual(sized.qty, 200)
        self.assertAlmostEqual(sized.stop_price, 95.0)

    def test_a_tighter_stop_buys_more_shares(self):
        wide = manager(risk_per_trade_pct=1.0, stop_loss_pct=10.0, max_position_pct=100.0)
        tight = manager(risk_per_trade_pct=1.0, stop_loss_pct=2.0, max_position_pct=100.0)
        w = wide.size_position(equity=100_000, cash=100_000, price=100.0)
        t = tight.size_position(equity=100_000, cash=100_000, price=100.0)
        self.assertGreater(t.qty, w.qty)

    def test_max_position_pct_caps_the_notional(self):
        m = manager(risk_per_trade_pct=5.0, stop_loss_pct=1.0, max_position_pct=10.0)
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0)
        self.assertLessEqual(sized.qty * 100.0, 100_000 * 0.10 + 1e-9)

    def test_cash_buffer_is_never_spent(self):
        m = manager(risk_per_trade_pct=50.0, stop_loss_pct=1.0, max_position_pct=100.0,
                    min_cash_buffer_pct=20.0)
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0)
        self.assertLessEqual(sized.qty * 100.0, 80_000 + 1e-9)

    def test_signal_strength_scales_the_risk_budget(self):
        m = manager(risk_per_trade_pct=1.0, stop_loss_pct=5.0, max_position_pct=100.0)
        full = m.size_position(equity=100_000, cash=100_000, price=100.0, strength=1.0)
        half = m.size_position(equity=100_000, cash=100_000, price=100.0, strength=0.5)
        self.assertEqual(half.qty, full.qty // 2)

    def test_strategy_stop_overrides_the_default_percentage(self):
        m = manager(risk_per_trade_pct=1.0, stop_loss_pct=5.0, max_position_pct=100.0)
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0, stop_price=98.0)
        self.assertAlmostEqual(sized.stop_price, 98.0)
        self.assertEqual(sized.qty, 500)

    def test_atr_stops_take_priority_when_enabled(self):
        m = manager(use_atr_stops=True, atr_stop_multiple=2.0, risk_per_trade_pct=1.0,
                    max_position_pct=100.0)
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0,
                                stop_price=99.0, atr_value=3.0)
        self.assertAlmostEqual(sized.stop_price, 94.0)

    def test_option_multiplier_reduces_contract_count(self):
        m = manager(risk_per_trade_pct=1.0, stop_loss_pct=50.0, max_position_pct=100.0)
        shares = m.size_position(equity=100_000, cash=100_000, price=5.0,
                                 contract_multiplier=1)
        contracts = m.size_position(equity=100_000, cash=100_000, price=5.0,
                                    contract_multiplier=100)
        self.assertEqual(contracts.qty * 100, shares.qty)

    def test_a_stop_above_entry_is_rejected(self):
        m = manager()
        sized = m.size_position(equity=100_000, cash=100_000, price=100.0, stop_price=105.0)
        # 105 is not below 100, so the percentage default is used instead.
        self.assertTrue(sized.ok)
        self.assertLess(sized.stop_price, 100.0)

    def test_zero_price_is_refused(self):
        self.assertFalse(manager().size_position(equity=100_000, cash=100_000, price=0.0).ok)

    def test_no_cash_means_no_position(self):
        m = manager()
        self.assertFalse(m.size_position(equity=100_000, cash=0.0, price=100.0).ok)


class TestKillSwitch(unittest.TestCase):
    def test_halts_once_the_daily_loss_limit_is_reached(self):
        m = manager(max_daily_loss_pct=3.0)
        m.update_equity(98_000.0)
        self.assertFalse(m.halted)
        m.update_equity(96_900.0)
        self.assertTrue(m.halted)
        self.assertIn("daily loss", m.state.halt_reason)

    def test_halted_manager_refuses_new_entries(self):
        m = manager(max_daily_loss_pct=1.0)
        m.update_equity(90_000.0)
        allowed, why = m.can_open(open_positions=0, cash=90_000, equity=90_000)
        self.assertFalse(allowed)
        self.assertIn("halted", why)

    def test_a_new_day_resets_the_halt(self):
        m = manager(max_daily_loss_pct=1.0)
        m.update_equity(50_000.0)
        self.assertTrue(m.halted)
        m.state.day = date.today() - timedelta(days=1)
        m.update_equity(50_000.0)
        self.assertFalse(m.halted)

    def test_daily_pnl_percentage(self):
        m = manager()
        self.assertAlmostEqual(m.daily_pnl_pct(105_000.0), 5.0)
        self.assertAlmostEqual(m.daily_pnl_pct(95_000.0), -5.0)


class TestEntryGates(unittest.TestCase):
    def test_max_positions_is_enforced(self):
        m = manager(max_positions=3)
        self.assertTrue(m.can_open(2, 50_000, 100_000)[0])
        self.assertFalse(m.can_open(3, 50_000, 100_000)[0])

    def test_cash_buffer_blocks_entry(self):
        m = manager(min_cash_buffer_pct=10.0)
        self.assertFalse(m.can_open(0, 9_000, 100_000)[0])
        self.assertTrue(m.can_open(0, 11_000, 100_000)[0])

    def test_daily_trade_limit(self):
        m = manager()
        self.assertFalse(m.daily_trade_limit_reached(3))
        for _ in range(3):
            m.record_entry()
        self.assertTrue(m.daily_trade_limit_reached(3))
        self.assertFalse(m.daily_trade_limit_reached(0))  # 0 means no limit


class TestVolatilityManagement(unittest.TestCase):
    def test_targeting_is_off_by_default(self):
        self.assertEqual(manager().volatility_scale(45.0), 1.0)

    def test_high_volatility_shrinks_the_position(self):
        m = manager(target_volatility_pct=20.0)
        # Twice the target volatility should halve the size.
        self.assertAlmostEqual(m.volatility_scale(40.0), 0.5)

    def test_low_volatility_grows_the_position_up_to_the_cap(self):
        m = manager(target_volatility_pct=20.0, max_vol_scale=1.5)
        self.assertAlmostEqual(m.volatility_scale(10.0), 1.5)
        self.assertAlmostEqual(m.volatility_scale(16.0), 1.25)

    def test_missing_volatility_leaves_size_untouched(self):
        m = manager(target_volatility_pct=20.0)
        self.assertEqual(m.volatility_scale(None), 1.0)
        self.assertEqual(m.volatility_scale(0.0), 1.0)

    def test_scaling_flows_through_to_quantity(self):
        m = manager(target_volatility_pct=20.0, risk_per_trade_pct=1.0,
                    stop_loss_pct=5.0, max_position_pct=100.0)
        calm = m.size_position(equity=100_000, cash=100_000, price=100.0, realised_vol=20.0)
        wild = m.size_position(equity=100_000, cash=100_000, price=100.0, realised_vol=40.0)
        self.assertEqual(calm.qty, 200)
        self.assertEqual(wild.qty, 100)
        self.assertAlmostEqual(wild.vol_scale, 0.5)

    def test_volatility_ceiling_is_off_by_default(self):
        self.assertTrue(manager().volatility_ok(500.0)[0])

    def test_volatility_ceiling_blocks_a_turbulent_symbol(self):
        m = manager(max_volatility_pct=60.0)
        ok, why = m.volatility_ok(90.0)
        self.assertFalse(ok)
        self.assertIn("ceiling", why)
        self.assertTrue(m.volatility_ok(30.0)[0])

    def test_ceiling_abstains_without_a_reading(self):
        self.assertTrue(manager(max_volatility_pct=60.0).volatility_ok(None)[0])


class TestCorrelationGate(unittest.TestCase):
    @staticmethod
    def series(values):
        return pd.Series(values, index=pd.date_range("2024-01-02", periods=len(values)))

    def setUp(self):
        rng = np.random.default_rng(4)
        base = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        self.closes = {
            "US.A": self.series(base),
            # Same underlying path -> perfectly correlated returns.
            "US.CLONE": self.series(base * 2.5),
            "US.OTHER": self.series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200))),
        }

    def test_disabled_by_default(self):
        m = manager(max_correlation=0.0)
        ok, _ = m.correlation_ok("US.CLONE", ["US.A"], self.closes)
        self.assertTrue(ok)

    def test_blocks_a_duplicate_of_an_open_position(self):
        m = manager(max_correlation=0.85, correlation_lookback=60)
        ok, why = m.correlation_ok("US.CLONE", ["US.A"], self.closes)
        self.assertFalse(ok)
        self.assertIn("correlated", why)

    def test_allows_an_unrelated_name(self):
        m = manager(max_correlation=0.85, correlation_lookback=60)
        ok, _ = m.correlation_ok("US.OTHER", ["US.A"], self.closes)
        self.assertTrue(ok)

    def test_abstains_when_there_is_no_history(self):
        m = manager(max_correlation=0.85)
        ok, _ = m.correlation_ok("US.UNKNOWN", ["US.A"], self.closes)
        self.assertTrue(ok)

    def test_nothing_held_means_nothing_to_correlate_with(self):
        m = manager(max_correlation=0.85)
        self.assertTrue(m.correlation_ok("US.CLONE", [], self.closes)[0])


if __name__ == "__main__":
    unittest.main()
