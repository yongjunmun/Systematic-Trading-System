"""Statistics for judging whether a backtest result is real.

Three ideas, all from Bailey & Lopez de Prado:

* **Probabilistic Sharpe Ratio** - the probability that the true Sharpe is above
  a threshold, given how many observations you have and how skewed and
  fat-tailed the returns are. A Sharpe of 1.5 from 40 bars of a negatively
  skewed strategy is worth much less than the same number from 4,000.
* **Deflated Sharpe Ratio** - the same question, but with the threshold raised
  to the Sharpe you would *expect* to see from the best of N random trials.
  Search a 12-point parameter grid and the winner looks good by construction;
  this is what prices that in.
* **Kelly fraction** - the bet size that maximises long-run growth, derived
  from the observed R distribution. Useful mainly as a ceiling: betting above
  full Kelly lowers growth *and* raises risk, which is the worst of both.

Uses ``statistics.NormalDist`` from the standard library, so no SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

# Euler-Mascheroni constant, used for the expected maximum of N normal draws.
EULER_MASCHERONI = 0.5772156649015328606
_NORMAL = NormalDist()


@dataclass
class SharpeStats:
    """Moments of a return series, at the frequency the returns were sampled."""

    observations: int
    sharpe: float          # per observation, NOT annualised
    skew: float
    kurtosis: float        # non-excess: a normal distribution scores 3.0
    mean: float
    stdev: float

    def annualised(self, periods_per_year: int) -> float:
        return self.sharpe * math.sqrt(periods_per_year)


def describe_returns(returns: Sequence[float]) -> SharpeStats | None:
    """Sharpe and higher moments of a return series. None if too short."""
    values = np.asarray([r for r in returns if r is not None and np.isfinite(r)], dtype=float)
    if values.size < 3:
        return None

    mean = float(values.mean())
    stdev = float(values.std(ddof=1))
    if stdev <= 0:
        return None

    centred = values - mean
    # Population moments, matching the convention in the PSR paper.
    m2 = float((centred ** 2).mean())
    m3 = float((centred ** 3).mean())
    m4 = float((centred ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurtosis = m4 / m2 ** 2 if m2 > 0 else 3.0

    return SharpeStats(
        observations=int(values.size),
        sharpe=mean / stdev,
        skew=skew,
        kurtosis=kurtosis,
        mean=mean,
        stdev=stdev,
    )


def probabilistic_sharpe_ratio(stats: SharpeStats, benchmark_sharpe: float = 0.0) -> float:
    """P(true Sharpe > benchmark), as a percentage.

    `benchmark_sharpe` is per observation, same units as `stats.sharpe`.
    Negative skew and fat tails both widen the estimator's error bars, so a
    strategy that wins small and often but loses big is penalised - which is
    exactly the profile that looks best in a naive backtest.
    """
    sr = stats.sharpe
    variance = 1.0 - stats.skew * sr + (stats.kurtosis - 1.0) / 4.0 * sr ** 2
    if variance <= 0 or stats.observations < 2:
        return float("nan")
    z = (sr - benchmark_sharpe) * math.sqrt(stats.observations - 1) / math.sqrt(variance)
    return _NORMAL.cdf(z) * 100.0


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """The Sharpe you should expect from the best of `trials` worthless strategies.

    This is the null hypothesis that matters when you optimise: even with no
    edge at all, the best of N tries looks good. Returns a per-observation
    Sharpe in the same units as `sharpe_variance`.
    """
    if trials < 2 or sharpe_variance <= 0:
        return 0.0
    spread = math.sqrt(sharpe_variance)
    a = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return spread * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


@dataclass
class DeflatedSharpeReport:
    trials: int
    observations: int
    observed_sharpe_annual: float
    threshold_sharpe_annual: float
    probabilistic_sharpe_pct: float
    deflated_sharpe_pct: float
    skew: float
    kurtosis: float

    @property
    def survives(self) -> bool:
        """Better than 95% confident the edge is not an artefact of searching."""
        return self.deflated_sharpe_pct >= 95.0


def deflated_sharpe_ratio(
    returns: Sequence[float],
    periods_per_year: int,
    trial_sharpes_annual: Sequence[float] | None = None,
) -> DeflatedSharpeReport | None:
    """Deflate an observed Sharpe by the number of configurations that were tried.

    `trial_sharpes_annual` should be the annualised Sharpe of every parameter
    combination evaluated. With one trial (no search) the deflated ratio equals
    the probabilistic one, which is the correct behaviour: nothing was cherry
    picked, so nothing needs deflating.
    """
    stats = describe_returns(returns)
    if stats is None:
        return None

    trials = len(trial_sharpes_annual) if trial_sharpes_annual else 1
    threshold = 0.0
    if trials > 1:
        # Convert the trial Sharpes to per-observation units so they are
        # comparable with stats.sharpe before taking their variance.
        scale = math.sqrt(periods_per_year)
        per_obs = np.asarray(trial_sharpes_annual, dtype=float) / scale
        threshold = expected_max_sharpe(trials, float(np.var(per_obs, ddof=1)))

    return DeflatedSharpeReport(
        trials=trials,
        observations=stats.observations,
        observed_sharpe_annual=stats.annualised(periods_per_year),
        threshold_sharpe_annual=threshold * math.sqrt(periods_per_year),
        probabilistic_sharpe_pct=probabilistic_sharpe_ratio(stats, 0.0),
        deflated_sharpe_pct=probabilistic_sharpe_ratio(stats, threshold),
        skew=stats.skew,
        kurtosis=stats.kurtosis,
    )


@dataclass
class KellyReport:
    full_kelly_pct: float     # % of equity to risk per trade
    half_kelly_pct: float
    configured_pct: float
    expectancy_r: float

    @property
    def overbetting(self) -> bool:
        return self.configured_pct > self.full_kelly_pct

    @property
    def verdict(self) -> str:
        if self.full_kelly_pct <= 0:
            return "no positive edge in the sample - Kelly says do not bet at all"
        if self.overbetting:
            return (
                f"configured risk ({self.configured_pct:.2f}%) is above full Kelly "
                f"({self.full_kelly_pct:.2f}%) - this lowers long-run growth AND "
                "raises risk"
            )
        if self.configured_pct > self.half_kelly_pct:
            return (
                f"configured risk sits between half and full Kelly - aggressive but "
                "not self-defeating"
            )
        return "configured risk is at or below half Kelly, the usual practical ceiling"


def kelly_fraction(r_multiples: Sequence[float], max_fraction: float = 1.0) -> float:
    """Growth-optimal fraction of equity to risk per trade, from the R distribution.

    Maximises E[log(1 + f * R)] by ternary search - the objective is concave in
    f, and this avoids a SciPy dependency. Returns 0 when the sample has no
    positive expectancy.
    """
    values = np.asarray([r for r in r_multiples if r is not None], dtype=float)
    if values.size < 3 or values.mean() <= 0:
        return 0.0

    worst = float(values.min())
    # Beyond f = 1/|worst| a single worst-case loss wipes the account out.
    ceiling = max_fraction if worst >= 0 else min(max_fraction, 0.99 / abs(worst))
    if ceiling <= 0:
        return 0.0

    def growth(f: float) -> float:
        wealth = 1.0 + f * values
        if np.any(wealth <= 0):
            return -np.inf
        return float(np.log(wealth).mean())

    low, high = 0.0, ceiling
    for _ in range(200):
        third = (high - low) / 3.0
        a, b = low + third, high - third
        if growth(a) < growth(b):
            low = a
        else:
            high = b
    best = (low + high) / 2.0
    return best if growth(best) > 0 else 0.0


def kelly_report(
    r_multiples: Sequence[float], configured_risk_pct: float
) -> KellyReport | None:
    """Compare the configured risk per trade against the growth-optimal one."""
    values = [r for r in r_multiples if r is not None]
    if len(values) < 5:
        return None
    full = kelly_fraction(values) * 100.0
    return KellyReport(
        full_kelly_pct=full,
        half_kelly_pct=full / 2.0,
        configured_pct=configured_risk_pct,
        expectancy_r=float(np.mean(values)),
    )
