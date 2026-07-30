"""Tests for the significance and bet-sizing statistics.

Properties matter more than reference values here: each measure must move in
the right direction as the input gets better or worse, and degenerate inputs
must return something safe rather than a plausible-looking number.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from moobot.stats import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    describe_returns,
    expected_max_sharpe,
    kelly_fraction,
    kelly_report,
    probabilistic_sharpe_ratio,
)


def normal_returns(n=500, mean=0.001, sd=0.01, seed=1) -> list[float]:
    return list(np.random.default_rng(seed).normal(mean, sd, n))


class TestDescribeReturns(unittest.TestCase):
    def test_sharpe_is_mean_over_stdev(self):
        values = np.array([0.01, 0.02, 0.03, 0.04])
        stats = describe_returns(values)
        self.assertAlmostEqual(stats.sharpe, values.mean() / values.std(ddof=1), places=9)

    def test_normal_data_has_kurtosis_near_three(self):
        stats = describe_returns(normal_returns(20_000, seed=5))
        self.assertAlmostEqual(stats.kurtosis, 3.0, delta=0.2)
        self.assertAlmostEqual(stats.skew, 0.0, delta=0.1)

    def test_negative_skew_is_detected(self):
        # Many small gains and a few large losses - the classic backtest darling.
        self.assertLess(describe_returns([0.01] * 100 + [-0.20, -0.25, -0.30]).skew, -1.0)

    def test_annualisation_scales_by_root_periods(self):
        stats = describe_returns(normal_returns(1000))
        self.assertAlmostEqual(stats.annualised(252), stats.sharpe * math.sqrt(252), places=9)

    def test_too_few_points_returns_none(self):
        self.assertIsNone(describe_returns([0.01, 0.02]))

    def test_constant_returns_have_no_defined_sharpe(self):
        self.assertIsNone(describe_returns([0.01] * 50))

    def test_non_finite_values_are_dropped(self):
        stats = describe_returns([0.01, float("nan"), 0.02, 0.03])
        self.assertEqual(stats.observations, 3)


class TestProbabilisticSharpe(unittest.TestCase):
    def test_more_observations_raise_confidence(self):
        short = describe_returns(normal_returns(60, seed=3))
        long = describe_returns(normal_returns(2000, seed=3))
        self.assertGreater(
            probabilistic_sharpe_ratio(long), probabilistic_sharpe_ratio(short)
        )

    def test_result_is_a_percentage(self):
        psr = probabilistic_sharpe_ratio(describe_returns(normal_returns(500)))
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 100.0)

    def test_a_losing_strategy_scores_below_fifty(self):
        stats = describe_returns(normal_returns(500, mean=-0.001))
        self.assertLess(probabilistic_sharpe_ratio(stats), 50.0)

    def test_a_higher_benchmark_lowers_the_score(self):
        stats = describe_returns(normal_returns(1000))
        easy = probabilistic_sharpe_ratio(stats, 0.0)
        hard = probabilistic_sharpe_ratio(stats, stats.sharpe * 0.9)
        self.assertLess(hard, easy)

    def test_a_benchmark_equal_to_the_estimate_gives_fifty_percent(self):
        stats = describe_returns(normal_returns(1000))
        self.assertAlmostEqual(probabilistic_sharpe_ratio(stats, stats.sharpe), 50.0, places=6)


class TestExpectedMaxSharpe(unittest.TestCase):
    def test_more_trials_raise_the_luck_threshold(self):
        self.assertGreater(expected_max_sharpe(500, 0.01), expected_max_sharpe(5, 0.01))

    def test_a_single_trial_has_no_threshold(self):
        self.assertEqual(expected_max_sharpe(1, 0.01), 0.0)

    def test_no_spread_between_trials_means_no_threshold(self):
        self.assertEqual(expected_max_sharpe(50, 0.0), 0.0)

    def test_threshold_scales_with_the_spread(self):
        narrow = expected_max_sharpe(50, 0.01)
        wide = expected_max_sharpe(50, 0.04)
        self.assertAlmostEqual(wide / narrow, 2.0, places=6)

    def test_euler_mascheroni_constant_is_right(self):
        self.assertAlmostEqual(EULER_MASCHERONI, 0.5772156649, places=9)


class TestDeflatedSharpe(unittest.TestCase):
    def test_one_trial_leaves_the_probabilistic_sharpe_alone(self):
        report = deflated_sharpe_ratio(normal_returns(600), 252)
        self.assertEqual(report.trials, 1)
        self.assertEqual(report.threshold_sharpe_annual, 0.0)
        self.assertAlmostEqual(
            report.deflated_sharpe_pct, report.probabilistic_sharpe_pct, places=9
        )

    def test_searching_a_grid_lowers_the_score(self):
        returns = normal_returns(600)
        alone = deflated_sharpe_ratio(returns, 252)
        searched = deflated_sharpe_ratio(returns, 252, [0.2, 0.9, 1.4, -0.3, 0.6, 1.1])
        self.assertGreater(searched.trials, 1)
        self.assertGreater(searched.threshold_sharpe_annual, 0.0)
        self.assertLess(searched.deflated_sharpe_pct, alone.deflated_sharpe_pct)

    def test_a_wider_spread_of_trials_deflates_harder(self):
        returns = normal_returns(600)
        tight = deflated_sharpe_ratio(returns, 252, [0.9, 1.0, 1.1, 1.0, 0.95, 1.05])
        loose = deflated_sharpe_ratio(returns, 252, [-2.0, 3.0, 0.1, 1.9, -1.4, 2.6])
        self.assertLess(loose.deflated_sharpe_pct, tight.deflated_sharpe_pct)

    def test_more_trials_at_the_same_spread_deflate_harder(self):
        returns = normal_returns(600)
        few = deflated_sharpe_ratio(returns, 252, [0.5, 1.0, 1.5, 1.0])
        many = deflated_sharpe_ratio(returns, 252, [0.5, 1.0, 1.5, 1.0] * 20)
        self.assertLess(many.deflated_sharpe_pct, few.deflated_sharpe_pct)

    def test_a_strong_long_sample_still_survives_a_small_search(self):
        returns = normal_returns(4000, mean=0.002, sd=0.008, seed=2)
        self.assertTrue(deflated_sharpe_ratio(returns, 252, [1.0, 1.2, 1.1, 0.9]).survives)

    def test_observed_sharpe_is_annualised(self):
        returns = normal_returns(1000)
        expected = describe_returns(returns).sharpe * math.sqrt(252)
        self.assertAlmostEqual(
            deflated_sharpe_ratio(returns, 252).observed_sharpe_annual, expected, places=9
        )

    def test_too_few_returns_gives_none(self):
        self.assertIsNone(deflated_sharpe_ratio([0.01, 0.02], 252))

    def test_survives_tracks_the_ninety_five_percent_bar(self):
        report = deflated_sharpe_ratio(normal_returns(600, mean=0.0, seed=9), 252)
        self.assertEqual(report.survives, report.deflated_sharpe_pct >= 95.0)


class TestKelly(unittest.TestCase):
    def test_no_edge_means_no_bet(self):
        self.assertEqual(kelly_fraction([1.0, -1.0] * 20), 0.0)

    def test_a_losing_system_means_no_bet(self):
        self.assertEqual(kelly_fraction([-1.0] * 20 + [0.5] * 5), 0.0)

    def test_a_strong_edge_gives_a_positive_fraction(self):
        self.assertGreater(kelly_fraction([2.0] * 6 + [-1.0] * 4), 0.0)

    def test_a_bigger_edge_justifies_a_bigger_bet(self):
        small = kelly_fraction([1.1] * 5 + [-1.0] * 5)
        large = kelly_fraction([4.0] * 5 + [-1.0] * 5)
        self.assertGreater(large, small)

    def test_the_fraction_never_risks_certain_ruin(self):
        """f must stay below 1/|worst loss| or one bad trade ends the account."""
        self.assertLess(kelly_fraction([3.0] * 8 + [-2.0] * 2) * 2.0, 1.0)

    def test_it_beats_neighbouring_fractions(self):
        r = [2.0] * 6 + [-1.0] * 4
        f = kelly_fraction(r)

        def growth(x: float) -> float:
            return float(np.mean(np.log(1 + x * np.array(r))))

        self.assertGreaterEqual(growth(f) + 1e-9, growth(f * 0.5))
        self.assertGreaterEqual(growth(f) + 1e-9, growth(min(f * 1.5, 0.49)))

    def test_report_flags_overbetting(self):
        r = [1.2] * 6 + [-1.0] * 4
        report = kelly_report(r, configured_risk_pct=kelly_fraction(r) * 100 + 5)
        self.assertTrue(report.overbetting)
        self.assertIn("above full Kelly", report.verdict)

    def test_report_accepts_conservative_sizing(self):
        report = kelly_report([2.0] * 6 + [-1.0] * 4, configured_risk_pct=1.0)
        self.assertFalse(report.overbetting)
        self.assertIn("half Kelly", report.verdict)

    def test_report_needs_a_sample(self):
        self.assertIsNone(kelly_report([1.0, -1.0], 1.0))

    def test_no_edge_report_says_do_not_bet(self):
        self.assertIn("do not bet", kelly_report([1.0, -1.0] * 10, 1.0).verdict)


if __name__ == "__main__":
    unittest.main()
