"""Validation tests: benchmark, folds, bootstrap, Monte Carlo and warnings."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from moobot.backtest import backtest_symbol
from moobot.risk import RiskManager
from moobot.settings import Config
from moobot.strategies import get_strategy
from moobot.validation import (
    bootstrap_expectancy,
    build_run_card,
    buy_and_hold,
    collect_warnings,
    fold_stability,
    monte_carlo_sequences,
    walk_forward,
)
from tests.helpers import make_bars, trending


def config() -> Config:
    cfg = Config()
    cfg.backtest.initial_cash = 100_000.0
    cfg.backtest.commission_per_share = 0.0
    cfg.backtest.min_commission = 0.0
    cfg.backtest.slippage_bps = 0.0
    cfg.trading.bar_type = "K_DAY"
    cfg.risk.risk_per_trade_pct = 2.0
    return cfg


class TestBuyAndHold(unittest.TestCase):
    def test_a_rising_market_gives_a_positive_return(self):
        bars = make_bars(list(trending(300, drift=0.003)))
        bench = buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0)
        self.assertGreater(bench.total_return_pct, 0)
        self.assertGreater(bench.final_equity, 100_000)

    def test_a_falling_market_gives_a_negative_return(self):
        bars = make_bars(list(100 * np.cumprod(np.full(300, 0.997))))
        self.assertLess(buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0).total_return_pct, 0)

    def test_costs_reduce_the_benchmark_too(self):
        bars = make_bars(list(trending(300, drift=0.003)))
        free = buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0)
        charged = buy_and_hold(bars, 100_000, "K_DAY", 0.02, 5.0, 50.0)
        self.assertLess(charged.total_return_pct, free.total_return_pct)

    def test_drawdown_is_never_positive(self):
        bars = make_bars(list(trending(300, drift=0.001, noise=0.02)))
        self.assertLessEqual(buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0)
                             .max_drawdown_pct, 0.0)

    def test_comparison_reports_excess_return(self):
        bars = make_bars(list(trending(300, drift=0.003)))
        bench = buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0)
        delta = bench.compare({
            "total_return_pct": bench.total_return_pct + 5.0,
            "cagr_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": bench.sharpe + 1,
        })
        self.assertAlmostEqual(delta["excess_return_pct"], 5.0)
        self.assertAlmostEqual(delta["sharpe_delta"], 1.0)

    def test_too_few_bars_is_handled(self):
        bars = make_bars([100.0])
        self.assertEqual(buy_and_hold(bars, 100_000, "K_DAY").total_return_pct, 0.0)


class TestFolds(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.bars = make_bars(list(trending(1400, drift=0.0015, noise=0.02)))
        self.risk = RiskManager(self.cfg.risk)
        self.risk.update_equity(100_000)

    def make(self, **kw):
        return get_strategy("sma_cross", **{"fast": 10, "slow": 30, "trend": 0, **kw})

    def test_produces_one_entry_per_fold(self):
        report = fold_stability(self.bars, self.make, self.risk, self.cfg, folds=4, code="T")
        self.assertEqual(len(report.folds), 4)
        self.assertFalse(report.refitted)
        self.assertEqual([f.index for f in report.folds], [1, 2, 3, 4])

    def test_folds_do_not_overlap_and_cover_the_data(self):
        report = fold_stability(self.bars, self.make, self.risk, self.cfg, folds=4, code="T")
        self.assertEqual(sum(f.bars for f in report.folds), len(self.bars))

    def test_summary_reports_consistency(self):
        report = fold_stability(self.bars, self.make, self.risk, self.cfg, folds=4, code="T")
        summary = report.summary()
        self.assertEqual(summary["folds"], 4)
        self.assertGreaterEqual(summary["consistency_pct"], 0.0)
        self.assertLessEqual(summary["consistency_pct"], 100.0)
        self.assertLessEqual(summary["worst_fold_pct"], summary["best_fold_pct"])

    def test_too_many_folds_auto_reduces_rather_than_failing(self):
        # Asking for 60 folds of 1400 bars leaves too few for warmup, so the
        # split degrades instead of throwing away the diagnostic entirely.
        report = fold_stability(self.bars, self.make, self.risk, self.cfg, folds=60, code="T")
        self.assertLess(len(report.folds), 60)
        self.assertGreaterEqual(len(report.folds), 2)
        self.assertEqual(sum(f.bars for f in report.folds), len(self.bars))

    def test_it_still_refuses_when_even_two_folds_will_not_fit(self):
        short = make_bars(list(trending(100, drift=0.002, noise=0.02)))
        with self.assertRaises(ValueError) as ctx:
            fold_stability(short, self.make, self.risk, self.cfg, folds=4, code="T")
        self.assertIn("2 folds", str(ctx.exception))

    def test_walk_forward_skips_the_first_fold_and_records_params(self):
        report = walk_forward(
            self.bars, self.make, self.risk, self.cfg,
            grid={"fast": [5, 10], "slow": [30, 50]}, folds=4, code="T",
        )
        self.assertTrue(report.refitted)
        self.assertEqual(len(report.folds), 3)  # fold 1 has nothing to fit on
        for fold in report.folds:
            self.assertIn("fast", fold.params)
            self.assertIn(fold.params["fast"], (5, 10))

    def test_walk_forward_without_a_grid_degrades_to_fold_stability(self):
        report = walk_forward(
            self.bars, self.make, self.risk, self.cfg, grid={}, folds=3, code="T"
        )
        self.assertFalse(report.refitted)
        self.assertEqual(len(report.folds), 3)

    def test_embargo_shortens_the_training_window(self):
        """A trade opened at the end of training must close before the fold starts."""
        from moobot.validation import _embargo_bars

        self.assertEqual(_embargo_bars(1000, 2.0), 20)
        self.assertEqual(_embargo_bars(1000, 0.0), 0)

    def test_embargo_changes_the_fitted_parameters_or_leaves_folds_intact(self):
        grid = {"fast": [5, 10], "slow": [30, 50]}
        self.cfg.validation.embargo_pct = 0.0
        none = walk_forward(self.bars, self.make, self.risk, self.cfg, grid, 4, "T")
        self.cfg.validation.embargo_pct = 10.0
        wide = walk_forward(self.bars, self.make, self.risk, self.cfg, grid, 4, "T")
        # Both must still produce usable folds; the embargo only trims training.
        self.assertGreaterEqual(len(none.folds), 1)
        self.assertGreaterEqual(len(wide.folds), 1)


class TestGridEvaluation(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.bars = make_bars(list(trending(900, drift=0.0015, noise=0.02)))
        self.risk = RiskManager(self.cfg.risk)
        self.risk.update_equity(100_000)

    def make(self, **kw):
        return get_strategy("sma_cross", **{"fast": 10, "slow": 30, "trend": 0, **kw})

    def test_one_sharpe_per_combination(self):
        from moobot.validation import evaluate_grid

        sharpes = evaluate_grid(
            self.bars, self.make, self.risk, self.cfg,
            {"fast": [5, 10, 15], "slow": [30, 50]}, "T",
        )
        self.assertEqual(len(sharpes), 6)
        self.assertTrue(all(isinstance(s, float) for s in sharpes))

    def test_an_empty_grid_evaluates_nothing(self):
        from moobot.validation import evaluate_grid

        self.assertEqual(evaluate_grid(self.bars, self.make, self.risk, self.cfg, {}, "T"), [])


class TestBootstrap(unittest.TestCase):
    def test_a_clear_edge_is_significant(self):
        r = [1.5] * 30 + [-1.0] * 20
        report = bootstrap_expectancy(r, samples=1000, seed=1)
        self.assertIsNotNone(report)
        self.assertTrue(report.significant)
        self.assertGreater(report.probability_positive, 95)
        self.assertLess(report.lower_r, report.observed_expectancy_r)
        self.assertGreater(report.upper_r, report.observed_expectancy_r)

    def test_a_coin_flip_is_not_significant(self):
        r = [1.0, -1.0] * 25
        report = bootstrap_expectancy(r, samples=1000, seed=1)
        self.assertFalse(report.significant)
        self.assertLess(report.probability_positive, 95)

    def test_a_losing_system_is_not_significant(self):
        report = bootstrap_expectancy([-1.0] * 20 + [0.5] * 10, samples=1000, seed=1)
        self.assertFalse(report.significant)
        self.assertLess(report.observed_expectancy_r, 0)

    def test_observed_expectancy_matches_the_mean(self):
        r = [2.0, -1.0, 0.5, -1.0, 3.0, -1.0]
        report = bootstrap_expectancy(r, samples=500, seed=1)
        self.assertAlmostEqual(report.observed_expectancy_r, float(np.mean(r)), places=9)

    def test_too_few_trades_returns_none(self):
        self.assertIsNone(bootstrap_expectancy([1.0, -1.0], samples=500))

    def test_it_is_reproducible(self):
        r = [1.2, -1.0, 0.8, -1.0, 2.5, -1.0, 0.3]
        a = bootstrap_expectancy(r, samples=800, seed=99)
        b = bootstrap_expectancy(r, samples=800, seed=99)
        self.assertEqual(a.lower_r, b.lower_r)
        self.assertEqual(a.probability_positive, b.probability_positive)

    def test_a_wider_confidence_level_widens_the_interval(self):
        r = [1.5, -1.0, 2.0, -1.0, 0.5, -1.0, 3.0, -1.0]
        narrow = bootstrap_expectancy(r, samples=2000, confidence_pct=80, seed=7)
        wide = bootstrap_expectancy(r, samples=2000, confidence_pct=99, seed=7)
        self.assertLess(wide.lower_r, narrow.lower_r)
        self.assertGreater(wide.upper_r, narrow.upper_r)


class TestMonteCarlo(unittest.TestCase):
    def test_a_winning_system_usually_ends_up(self):
        report = monte_carlo_sequences([1.5] * 30 + [-1.0] * 20, runs=1000,
                                       risk_per_trade_pct=1.0, seed=3)
        self.assertIsNotNone(report)
        self.assertGreater(report.median_return_pct, 0)
        self.assertLess(report.probability_of_loss_pct, 30)

    def test_drawdown_is_never_positive(self):
        report = monte_carlo_sequences([1.0, -1.0] * 20, runs=500, seed=3)
        self.assertLessEqual(report.median_max_drawdown_pct, 0.0)
        self.assertLessEqual(report.worst_max_drawdown_pct, report.median_max_drawdown_pct)

    def test_percentiles_are_ordered(self):
        report = monte_carlo_sequences([2.0, -1.0, 0.5, -1.0, 1.0] * 6, runs=1000, seed=3)
        self.assertLessEqual(report.return_p5_pct, report.median_return_pct)
        self.assertLessEqual(report.median_return_pct, report.return_p95_pct)

    def test_higher_risk_per_trade_produces_deeper_drawdowns(self):
        r = [1.5, -1.0, -1.0, 2.0, -1.0] * 6
        small = monte_carlo_sequences(r, runs=800, risk_per_trade_pct=0.5, seed=5)
        large = monte_carlo_sequences(r, runs=800, risk_per_trade_pct=5.0, seed=5)
        self.assertLess(large.median_max_drawdown_pct, small.median_max_drawdown_pct)

    def test_a_losing_system_almost_always_loses(self):
        report = monte_carlo_sequences([-1.0] * 25 + [0.5] * 10, runs=800,
                                       risk_per_trade_pct=2.0, seed=3)
        self.assertGreater(report.probability_of_loss_pct, 90)

    def test_equity_cannot_go_negative(self):
        # 100% risk per trade with full losses must floor rather than go below zero.
        report = monte_carlo_sequences([-1.0] * 20, runs=200, risk_per_trade_pct=100.0, seed=3)
        self.assertGreaterEqual(report.median_return_pct, -100.0)

    def test_too_few_trades_returns_none(self):
        self.assertIsNone(monte_carlo_sequences([1.0, -1.0], runs=500))


class TestWarnings(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.risk = RiskManager(self.cfg.risk)
        self.risk.update_equity(100_000)

    def run_backtest(self, bars):
        return backtest_symbol(
            bars=bars,
            strategy=get_strategy("sma_cross", fast=10, slow=30, trend=0),
            risk=self.risk, code="T", initial_cash=100_000,
            commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0,
            bar_type="K_DAY", exits=self.cfg.exits,
        )

    def test_no_trades_is_reported_plainly(self):
        bars = make_bars([100.0] * 400)
        notes = collect_warnings(self.run_backtest(bars), bars)
        self.assertEqual(len(notes), 1)
        self.assertIn("No trades", notes[0])

    def test_small_sample_is_flagged(self):
        bars = make_bars(list(trending(400, drift=0.002, noise=0.02)))
        notes = collect_warnings(self.run_backtest(bars), bars, min_trades=1000)
        self.assertTrue(any("trades" in n for n in notes))

    def test_short_history_is_flagged(self):
        bars = make_bars(list(trending(200, drift=0.002, noise=0.02)))
        notes = collect_warnings(self.run_backtest(bars), bars, min_trades=0)
        self.assertTrue(any("years" in n for n in notes))

    def test_duplicate_timestamps_are_flagged(self):
        bars = make_bars(list(trending(900, drift=0.002, noise=0.02)))
        result = self.run_backtest(bars)
        bars.loc[5, "time_key"] = bars.loc[4, "time_key"]
        notes = collect_warnings(result, bars, min_trades=0)
        self.assertTrue(any("duplicate" in n for n in notes))

    def test_an_insignificant_bootstrap_is_surfaced(self):
        bars = make_bars(list(trending(900, drift=0.002, noise=0.02)))
        result = self.run_backtest(bars)
        boot = bootstrap_expectancy([1.0, -1.0] * 25, samples=800, seed=2)
        notes = collect_warnings(result, bars, min_trades=0, bootstrap=boot)
        self.assertTrue(any("luck" in n for n in notes))


class TestRunCard(unittest.TestCase):
    def test_it_round_trips_through_json(self):
        cfg = config()
        risk = RiskManager(cfg.risk)
        risk.update_equity(100_000)
        bars = make_bars(list(trending(900, drift=0.002, noise=0.02)))
        strategy = get_strategy("sma_cross", fast=10, slow=30, trend=0)
        result = backtest_symbol(
            bars=bars, strategy=strategy, risk=risk, code="US.TEST",
            initial_cash=100_000, commission_per_share=0.0, min_commission=0.0,
            slippage_bps=0.0, bar_type="K_DAY", exits=cfg.exits,
        )
        bench = buy_and_hold(bars, 100_000, "K_DAY", 0.0, 0.0, 0.0)
        card = build_run_card(result, bars, cfg, strategy, bench, None, None, None, ["note"])

        with tempfile.TemporaryDirectory() as tmp:
            path = card.save(tmp)
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["code"], "US.TEST")
        self.assertEqual(data["strategy"], "sma_cross")
        self.assertEqual(data["strategy_params"]["fast"], 10)
        self.assertEqual(data["bars"], len(bars))
        self.assertIn("total_return_pct", data["metrics"])
        self.assertIn("buy_and_hold_return_pct", data["benchmark"])
        self.assertEqual(data["warnings"], ["note"])

    def test_filename_is_filesystem_safe(self):
        cfg = config()
        risk = RiskManager(cfg.risk)
        risk.update_equity(100_000)
        bars = make_bars(list(trending(900, drift=0.002, noise=0.02)))
        strategy = get_strategy("sma_cross", fast=10, slow=30, trend=0)
        result = backtest_symbol(
            bars=bars, strategy=strategy, risk=risk, code="US.AAPL",
            initial_cash=100_000, bar_type="K_DAY", exits=cfg.exits,
        )
        bench = buy_and_hold(bars, 100_000, "K_DAY")
        card = build_run_card(result, bars, cfg, strategy, bench, None, None, None, [])
        with tempfile.TemporaryDirectory() as tmp:
            path = card.save(tmp)
            self.assertNotIn(".", Path(path).stem)


if __name__ == "__main__":
    unittest.main()
