"""Settings validation and, most importantly, the live-money guard."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moobot.settings import (
    REAL_MONEY_ACK,
    Config,
    ConfigError,
    assert_paper_trading,
    load_config,
)

MINIMAL = """
[trading]
codes = ["US.AAPL"]
"""


def write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.paths: list[Path] = []

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def load(self, text: str) -> Config:
        path = write(text)
        self.paths.append(path)
        return load_config(path)

    def test_defaults_are_paper_and_dry_run(self):
        cfg = self.load(MINIMAL)
        self.assertTrue(cfg.is_paper)
        self.assertTrue(cfg.trading.dry_run)
        self.assertFalse(cfg.account.allow_real_money)
        self.assertFalse(cfg.options.enabled)

    def test_missing_file_is_reported_clearly(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config("definitely-not-here.toml")
        self.assertIn("not found", str(ctx.exception))

    def test_unknown_section_is_rejected(self):
        with self.assertRaises(ConfigError):
            self.load("[nonsense]\nfoo = 1\n")

    def test_typo_in_a_key_is_rejected_with_a_hint(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load('[trading]\ncodes = ["US.AAPL"]\npol_seconds = 30\n')
        self.assertIn("pol_seconds", str(ctx.exception))
        self.assertIn("poll_seconds", str(ctx.exception))

    def test_bad_bar_type_is_rejected(self):
        with self.assertRaises(ConfigError):
            self.load('[trading]\ncodes = ["US.AAPL"]\nbar_type = "K_7M"\n')

    def test_code_without_a_market_prefix_is_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load('[trading]\ncodes = ["AAPL"]\n')
        self.assertIn("MARKET.SYMBOL", str(ctx.exception))

    def test_unsupported_order_type_is_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load('[trading]\ncodes = ["US.AAPL"]\norder_type = "STOP"\n')
        self.assertIn("paper trading", str(ctx.exception))

    def test_risk_percentages_must_be_sane(self):
        with self.assertRaises(ConfigError):
            self.load('[trading]\ncodes = ["US.AAPL"]\n[risk]\nrisk_per_trade_pct = 0\n')
        with self.assertRaises(ConfigError):
            self.load('[trading]\ncodes = ["US.AAPL"]\n[risk]\nmax_daily_loss_pct = 150\n')

    def test_option_delta_must_be_a_fraction(self):
        with self.assertRaises(ConfigError):
            self.load(
                '[trading]\ncodes = ["US.AAPL"]\n'
                '[options]\nenabled = true\ntarget_delta = 35\n'
            )

    def test_option_expiry_window_must_be_ordered(self):
        with self.assertRaises(ConfigError):
            self.load(
                '[trading]\ncodes = ["US.AAPL"]\n'
                '[options]\nenabled = true\nmin_dte = 45\nmax_dte = 14\n'
            )

    def test_strategy_params_pass_through_untouched(self):
        cfg = self.load(
            '[trading]\ncodes = ["US.AAPL"]\n'
            '[strategy]\nname = "rsi_reversion"\n'
            "[strategy.params]\nperiod = 21\noversold = 25.0\n"
        )
        self.assertEqual(cfg.strategy.name, "rsi_reversion")
        self.assertEqual(cfg.strategy.params["period"], 21)
        self.assertEqual(cfg.strategy.params["oversold"], 25.0)

    def test_ints_are_promoted_to_floats_where_expected(self):
        cfg = self.load('[trading]\ncodes = ["US.AAPL"]\n[risk]\nstop_loss_pct = 4\n')
        self.assertIsInstance(cfg.risk.stop_loss_pct, float)

    def test_real_env_still_loads_but_is_flagged_not_paper(self):
        cfg = self.load('[trading]\ncodes = ["US.AAPL"]\n[account]\ntrd_env = "REAL"\n')
        self.assertFalse(cfg.is_paper)


class TestLiveMoneyGuard(unittest.TestCase):
    """The whole point of the project: real money must be hard to reach."""

    @staticmethod
    def real_config(allow: bool = False) -> Config:
        cfg = Config()
        cfg.account.trd_env = "REAL"
        cfg.account.allow_real_money = allow
        return cfg

    def test_paper_passes(self):
        assert_paper_trading(Config())  # must not raise

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_real_is_blocked_by_default(self):
        with self.assertRaises(ConfigError) as ctx:
            assert_paper_trading(self.real_config())
        self.assertIn("REFUSING TO RUN", str(ctx.exception))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_config_flag_alone_is_not_enough(self):
        with self.assertRaises(ConfigError):
            assert_paper_trading(self.real_config(allow=True))

    @mock.patch.dict(os.environ, {"MOOBOT_ALLOW_REAL": REAL_MONEY_ACK}, clear=True)
    def test_env_var_alone_is_not_enough(self):
        with self.assertRaises(ConfigError):
            assert_paper_trading(self.real_config(allow=False))

    @mock.patch.dict(os.environ, {"MOOBOT_ALLOW_REAL": "yes"}, clear=True)
    def test_a_near_miss_acknowledgement_is_not_enough(self):
        with self.assertRaises(ConfigError):
            assert_paper_trading(self.real_config(allow=True))

    @mock.patch.dict(os.environ, {"MOOBOT_ALLOW_REAL": REAL_MONEY_ACK}, clear=True)
    def test_all_three_together_are_accepted(self):
        assert_paper_trading(self.real_config(allow=True))  # must not raise


class TestBrokerRefusesLiveMoney(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_broker_cannot_even_be_constructed_against_a_real_account(self):
        from moobot.broker import Broker

        cfg = Config()
        cfg.account.trd_env = "REAL"
        with self.assertRaises(ConfigError):
            Broker(cfg)


if __name__ == "__main__":
    unittest.main()
