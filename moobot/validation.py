"""Backtest validation: the difference between a number and evidence.

A single equity curve tells you what happened once. These tools ask harder
questions:

* **Fold stability** - did the edge exist in every period, or does one lucky
  quarter carry the whole result?
* **Walk-forward** - when parameters are re-fitted only on past data, does the
  strategy still work on the data it has never seen?
* **Bootstrap** - resampling the trades, how likely is it that the expectancy is
  above zero rather than noise?
* **Monte Carlo** - the trades happened in one particular order. In a different
  order, how deep could the drawdown have been?
* **Benchmark** - did it beat simply buying and holding, after costs?
* **Warnings** - the specific ways this particular result might be lying.

None of this makes a strategy profitable. It tells you how much to believe.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest_symbol, compute_metrics
from .risk import RiskManager
from .settings import BARS_PER_YEAR, Config
from .strategies.base import Strategy

log = logging.getLogger(__name__)

StrategyFactory = Callable[..., Strategy]


# --------------------------------------------------------------------- benchmark


@dataclass
class BenchmarkResult:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    final_equity: float

    def compare(self, strategy_metrics: dict[str, float]) -> dict[str, float]:
        # Drawdowns are negative numbers, so 'saved' is strategy minus benchmark:
        # -2% strategy against -42% benchmark saved 40 points, not -40.
        return {
            "excess_return_pct": strategy_metrics["total_return_pct"] - self.total_return_pct,
            "excess_cagr_pct": strategy_metrics["cagr_pct"] - self.cagr_pct,
            "drawdown_saved_pct": strategy_metrics["max_drawdown_pct"] - self.max_drawdown_pct,
            "sharpe_delta": strategy_metrics["sharpe"] - self.sharpe,
        }


def buy_and_hold(
    bars: pd.DataFrame,
    initial_cash: float,
    bar_type: str = "K_DAY",
    commission_per_share: float = 0.005,
    min_commission: float = 1.0,
    slippage_bps: float = 5.0,
) -> BenchmarkResult:
    """Buy at the first bar's open, hold to the last close. Same costs applied."""
    if len(bars) < 2:
        return BenchmarkResult(0.0, 0.0, 0.0, 0.0, initial_cash)

    slip = slippage_bps / 10_000.0
    entry = float(bars.iloc[0]["open"]) * (1 + slip)
    qty = int(initial_cash // entry)
    if qty <= 0:
        return BenchmarkResult(0.0, 0.0, 0.0, 0.0, initial_cash)

    fees = max(min_commission, qty * commission_per_share) * 2
    cash = initial_cash - qty * entry
    equity = cash + qty * bars["close"].to_numpy(dtype=float)
    equity[-1] = cash + qty * float(bars.iloc[-1]["close"]) * (1 - slip) - fees

    series = pd.Series(equity, index=pd.to_datetime(bars["time_key"]))
    final = float(series.iloc[-1])
    ppy = BARS_PER_YEAR.get(bar_type, 252)
    years = len(series) / ppy

    returns = series.pct_change().dropna()
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean()) / std * math.sqrt(ppy) if std > 0 else 0.0
    drawdown = float(((series - series.cummax()) / series.cummax()).min()) * 100.0
    cagr = ((final / initial_cash) ** (1 / years) - 1) * 100.0 if years > 0 and final > 0 else 0.0

    return BenchmarkResult(
        total_return_pct=(final / initial_cash - 1) * 100.0,
        cagr_pct=cagr,
        max_drawdown_pct=drawdown,
        sharpe=sharpe,
        final_equity=final,
    )


# ----------------------------------------------------------------- fold stability


@dataclass
class Fold:
    index: int
    start: str
    end: str
    bars: int
    trades: int
    return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy_r: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldReport:
    folds: list[Fold]
    refitted: bool

    @property
    def profitable_folds(self) -> int:
        return sum(1 for f in self.folds if f.return_pct > 0)

    @property
    def consistency_pct(self) -> float:
        return self.profitable_folds / len(self.folds) * 100.0 if self.folds else 0.0

    def summary(self) -> dict[str, float]:
        returns = [f.return_pct for f in self.folds]
        if not returns:
            return {}
        return {
            "folds": float(len(returns)),
            "profitable_folds": float(self.profitable_folds),
            "consistency_pct": self.consistency_pct,
            "mean_return_pct": float(np.mean(returns)),
            "median_return_pct": float(np.median(returns)),
            "std_return_pct": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "worst_fold_pct": float(np.min(returns)),
            "best_fold_pct": float(np.max(returns)),
        }


def fold_stability(
    bars: pd.DataFrame,
    make_strategy: StrategyFactory,
    risk: RiskManager,
    cfg: Config,
    folds: int = 4,
    code: str = "?",
) -> FoldReport:
    """Split the history into contiguous slices and backtest each independently.

    This does NOT re-fit anything - it answers 'is the edge present throughout,
    or concentrated in one regime?'. Use `walk_forward` for out-of-sample proof.
    """
    slices = _contiguous_slices(bars, folds, minimum=make_strategy().warmup + 30)
    results: list[Fold] = []
    for i, chunk in enumerate(slices, start=1):
        outcome = _safe_backtest(chunk, make_strategy(), risk, cfg, code)
        results.append(_to_fold(i, chunk, outcome))
    return FoldReport(folds=results, refitted=False)


def walk_forward(
    bars: pd.DataFrame,
    make_strategy: StrategyFactory,
    risk: RiskManager,
    cfg: Config,
    grid: dict[str, Sequence[Any]],
    folds: int = 4,
    code: str = "?",
) -> FoldReport:
    """Anchored walk-forward: fit on everything before a fold, trade the fold.

    Fold 1 has no history to fit on and is skipped. Each later fold is traded
    with the parameter set that scored best on the data available *before* it,
    which is the only honest way to ask 'would this have worked live?'.

    An embargo gap is cut from the end of every training window. Without it a
    trade opened in the last training bars is still open once the test period
    begins, so the fit is scored partly on data it is meant to be blind to.
    """
    slices = _contiguous_slices(bars, folds, minimum=make_strategy().warmup + 30)
    combos = _expand_grid(grid)
    if not combos:
        return fold_stability(bars, make_strategy, risk, cfg, folds, code)

    embargo = _embargo_bars(len(bars), cfg.validation.embargo_pct)
    results: list[Fold] = []
    for i, chunk in enumerate(slices, start=1):
        if i == 1:
            continue  # nothing to fit on yet
        in_sample = bars[bars["time_key"] < chunk.iloc[0]["time_key"]]
        if embargo:
            in_sample = in_sample.iloc[:-embargo]
        if len(in_sample) < make_strategy().warmup + 30:
            continue
        best = _best_params(in_sample, make_strategy, risk, cfg, combos, code)
        outcome = _safe_backtest(chunk, make_strategy(**best), risk, cfg, code)
        fold = _to_fold(i, chunk, outcome)
        fold.params = best
        results.append(fold)
    return FoldReport(folds=results, refitted=True)


def _embargo_bars(total_bars: int, embargo_pct: float) -> int:
    return int(total_bars * embargo_pct / 100.0) if embargo_pct > 0 else 0


def evaluate_grid(
    bars: pd.DataFrame,
    make_strategy: StrategyFactory,
    risk: RiskManager,
    cfg: Config,
    grid: dict[str, Sequence[Any]],
    code: str = "?",
) -> list[float]:
    """Annualised Sharpe of every parameter combination, for Sharpe deflation.

    The spread of these is what says how much of the winner's edge is just the
    luck of having tried several configurations.
    """
    combos = _expand_grid(grid)
    sharpes: list[float] = []
    for params in combos:
        try:
            outcome = _safe_backtest(bars, make_strategy(**params), risk, cfg, code)
        except (ValueError, TypeError):
            continue
        if outcome is not None:
            sharpes.append(compute_metrics(outcome)["sharpe"])
    return sharpes


def _expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return []
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _best_params(
    in_sample: pd.DataFrame,
    make_strategy: StrategyFactory,
    risk: RiskManager,
    cfg: Config,
    combos: list[dict[str, Any]],
    code: str,
) -> dict[str, Any]:
    """Pick the parameter set with the best in-sample return per unit drawdown."""
    best_params: dict[str, Any] = combos[0]
    best_score = -float("inf")
    for params in combos:
        try:
            outcome = _safe_backtest(in_sample, make_strategy(**params), risk, cfg, code)
        except (ValueError, TypeError):
            continue
        if outcome is None:
            continue
        m = compute_metrics(outcome)
        if m["num_trades"] < 3:
            continue
        drawdown = abs(m["max_drawdown_pct"]) or 1.0
        score = m["total_return_pct"] / drawdown
        if score > best_score:
            best_score, best_params = score, params
    return best_params


def _contiguous_slices(bars: pd.DataFrame, folds: int, minimum: int) -> list[pd.DataFrame]:
    """Split into `folds` equal slices, reducing the count if history is short.

    Silently returning nothing would hide the most useful diagnostic in the
    suite, so this degrades to fewer folds rather than refusing outright.
    """
    usable = folds
    while usable >= 2 and len(bars) // usable < minimum:
        usable -= 1
    if usable < 2:
        raise ValueError(
            f"{len(bars)} bars cannot be split into even 2 folds of {minimum} bars "
            f"(the strategy's warmup plus a margin). Fetch more history."
        )
    if usable != folds:
        log.info("reduced folds from %d to %d so each has enough warmup bars", folds, usable)
    size = len(bars) // usable
    return [
        bars.iloc[i * size : (i + 1) * size if i < usable - 1 else len(bars)]
        .reset_index(drop=True)
        for i in range(usable)
    ]


def _safe_backtest(
    chunk: pd.DataFrame, strategy: Strategy, risk: RiskManager, cfg: Config, code: str
) -> BacktestResult | None:
    try:
        return backtest_symbol(
            bars=chunk,
            strategy=strategy,
            risk=risk,
            code=code,
            initial_cash=cfg.backtest.initial_cash,
            commission_per_share=cfg.backtest.commission_per_share,
            min_commission=cfg.backtest.min_commission,
            slippage_bps=cfg.backtest.slippage_bps,
            bar_type=cfg.trading.bar_type,
            exits=cfg.exits,
        )
    except ValueError:
        return None


def _to_fold(index: int, chunk: pd.DataFrame, outcome: BacktestResult | None) -> Fold:
    start = str(chunk.iloc[0]["time_key"])[:16]
    end = str(chunk.iloc[-1]["time_key"])[:16]
    if outcome is None:
        return Fold(index, start, end, len(chunk), 0, 0.0, 0.0, 0.0, 0.0)
    m = compute_metrics(outcome)
    r_values = [t.r_multiple for t in outcome.trades if t.r_multiple is not None]
    return Fold(
        index=index,
        start=start,
        end=end,
        bars=len(chunk),
        trades=int(m["num_trades"]),
        return_pct=m["total_return_pct"],
        max_drawdown_pct=m["max_drawdown_pct"],
        profit_factor=m["profit_factor"],
        expectancy_r=float(np.mean(r_values)) if r_values else 0.0,
    )


# --------------------------------------------------------------------- bootstrap


@dataclass
class BootstrapReport:
    samples: int
    confidence_pct: float
    observed_expectancy_r: float
    lower_r: float
    upper_r: float
    probability_positive: float
    observed_profit_factor: float
    profit_factor_lower: float

    @property
    def significant(self) -> bool:
        """The interval excludes zero - the edge is unlikely to be pure noise."""
        return self.lower_r > 0


def bootstrap_expectancy(
    r_multiples: Sequence[float],
    samples: int = 2000,
    confidence_pct: float = 95.0,
    seed: int = 20260728,
) -> BootstrapReport | None:
    """Resample trades with replacement to put an interval around the edge.

    Works on R-multiples rather than dollars so position sizing cannot distort
    the answer.
    """
    values = np.asarray([r for r in r_multiples if r is not None], dtype=float)
    if values.size < 5:
        return None

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)

    wins = np.where(draws > 0, draws, 0.0).sum(axis=1)
    losses = np.where(draws < 0, -draws, 0.0).sum(axis=1)
    factors = np.divide(wins, losses, out=np.full_like(wins, np.inf), where=losses > 0)

    tail = (100.0 - confidence_pct) / 2.0
    observed_wins = values[values > 0].sum()
    observed_losses = -values[values < 0].sum()

    return BootstrapReport(
        samples=samples,
        confidence_pct=confidence_pct,
        observed_expectancy_r=float(values.mean()),
        lower_r=float(np.percentile(means, tail)),
        upper_r=float(np.percentile(means, 100.0 - tail)),
        probability_positive=float((means > 0).mean() * 100.0),
        observed_profit_factor=(
            float(observed_wins / observed_losses) if observed_losses > 0 else float("inf")
        ),
        profit_factor_lower=float(np.percentile(factors[np.isfinite(factors)], tail))
        if np.isfinite(factors).any()
        else float("inf"),
    )


# ------------------------------------------------------------------ monte carlo


@dataclass
class MonteCarloReport:
    runs: int
    risk_per_trade_pct: float
    trades_per_run: int
    median_return_pct: float
    return_p5_pct: float
    return_p95_pct: float
    median_max_drawdown_pct: float
    worst_max_drawdown_pct: float
    drawdown_p95_pct: float
    probability_of_loss_pct: float
    probability_ruin_pct: float
    ruin_threshold_pct: float


def monte_carlo_sequences(
    r_multiples: Sequence[float],
    runs: int = 2000,
    risk_per_trade_pct: float = 1.0,
    ruin_threshold_pct: float = 30.0,
    seed: int = 20260728,
) -> MonteCarloReport | None:
    """Reshuffle the trade sequence to see what luck contributed.

    The trades you got happened in one specific order. Ten losers in a row is
    perfectly possible with the same set of trades, and that ordering is what
    actually blows accounts up. Each sampled R is compounded at the configured
    risk fraction.
    """
    values = np.asarray([r for r in r_multiples if r is not None], dtype=float)
    if values.size < 5:
        return None

    rng = np.random.default_rng(seed)
    fraction = risk_per_trade_pct / 100.0
    draws = rng.choice(values, size=(runs, values.size), replace=True)

    # Compound: each trade risks `fraction` of current equity and returns R x that.
    growth = 1.0 + fraction * draws
    growth = np.maximum(growth, 1e-6)  # a trade cannot lose more than everything
    curves = np.cumprod(growth, axis=1)
    curves = np.concatenate([np.ones((runs, 1)), curves], axis=1)

    peaks = np.maximum.accumulate(curves, axis=1)
    drawdowns = (curves - peaks) / peaks
    worst_per_run = drawdowns.min(axis=1) * 100.0
    finals = (curves[:, -1] - 1.0) * 100.0

    return MonteCarloReport(
        runs=runs,
        risk_per_trade_pct=risk_per_trade_pct,
        trades_per_run=int(values.size),
        median_return_pct=float(np.median(finals)),
        return_p5_pct=float(np.percentile(finals, 5)),
        return_p95_pct=float(np.percentile(finals, 95)),
        median_max_drawdown_pct=float(np.median(worst_per_run)),
        worst_max_drawdown_pct=float(worst_per_run.min()),
        drawdown_p95_pct=float(np.percentile(worst_per_run, 5)),
        probability_of_loss_pct=float((finals < 0).mean() * 100.0),
        probability_ruin_pct=float((worst_per_run <= -ruin_threshold_pct).mean() * 100.0),
        ruin_threshold_pct=ruin_threshold_pct,
    )


# ---------------------------------------------------------------------- warnings


def collect_warnings(
    result: BacktestResult,
    bars: pd.DataFrame,
    min_trades: int = 30,
    bootstrap: BootstrapReport | None = None,
    deflated: Any | None = None,
) -> list[str]:
    """Concrete reasons to distrust this particular backtest."""
    notes: list[str] = []
    metrics = compute_metrics(result)
    trades = result.trades
    count = len(trades)

    if count == 0:
        return ["No trades were taken - this result says nothing about the strategy."]

    if count < min_trades:
        notes.append(
            f"Only {count} trades (want >= {min_trades}). At this sample size the "
            "statistics are close to meaningless."
        )

    years = result.total_bars / max(result.bars_per_year, 1)
    if years < 1:
        notes.append(
            f"Sample covers about {years:.2f} years - too short to have seen more "
            "than one market regime."
        )

    pnls = np.array([t.pnl for t in trades], dtype=float)
    total_profit = pnls[pnls > 0].sum()
    if total_profit > 0 and pnls.max() / total_profit > 0.5:
        notes.append(
            f"The single best trade ({pnls.max():,.2f}) is over half of all gross "
            "profit. Remove it and the edge probably disappears."
        )

    if metrics["win_rate_pct"] >= 95:
        notes.append(
            f"Win rate of {metrics['win_rate_pct']:.0f}% is a classic symptom of a "
            "lookahead bug or an unreachable limit price, not of skill."
        )

    if math.isinf(metrics["profit_factor"]):
        notes.append("No losing trades at all. Verify the exit logic before believing this.")

    if metrics["exposure_pct"] < 5:
        notes.append(
            f"In the market only {metrics['exposure_pct']:.1f}% of the time - the "
            "annualised figures are extrapolated from very little activity."
        )

    if metrics["max_drawdown_pct"] > -0.5 and metrics["total_return_pct"] > 0:
        notes.append(
            "Essentially no drawdown. Real strategies breathe; check that exits are "
            "actually being simulated."
        )

    if bootstrap is not None and not bootstrap.significant:
        notes.append(
            f"Bootstrap {bootstrap.confidence_pct:.0f}% interval for expectancy "
            f"({bootstrap.lower_r:+.3f}R to {bootstrap.upper_r:+.3f}R) includes zero - "
            "the edge is not distinguishable from luck."
        )

    if deflated is not None and deflated.trials > 1 and not deflated.survives:
        notes.append(
            f"Deflated Sharpe is {deflated.deflated_sharpe_pct:.1f}% after accounting "
            f"for {deflated.trials} parameter combinations tried. The best of that "
            f"many random strategies would be expected to score about "
            f"{deflated.threshold_sharpe_annual:.2f} Sharpe on its own."
        )

    duplicated = int(bars["time_key"].duplicated().sum())
    if duplicated:
        notes.append(f"{duplicated} duplicate timestamps in the input bars.")

    gaps = pd.to_datetime(bars["time_key"]).diff().dropna()
    if len(gaps) > 10:
        typical = gaps.median()
        big = int((gaps > typical * 20).sum())
        if big > len(gaps) * 0.02:
            notes.append(
                f"{big} unusually large gaps between bars - the history may be "
                "incomplete, which flatters trend strategies."
            )

    return notes


# ---------------------------------------------------------------------- run card


@dataclass
class RunCard:
    """A reproducible record of one validation run."""

    created_utc: str
    code: str
    strategy: str
    strategy_params: dict[str, Any]
    bar_type: str
    data_start: str
    data_end: str
    bars: int
    costs: dict[str, float]
    exits: dict[str, Any]
    risk: dict[str, Any]
    metrics: dict[str, float]
    benchmark: dict[str, float]
    fold_summary: dict[str, float]
    bootstrap: dict[str, Any] | None
    monte_carlo: dict[str, Any] | None
    warnings: list[str]
    deflated_sharpe: dict[str, Any] | None = None
    kelly: dict[str, Any] | None = None

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = "".join(ch for ch in self.created_utc if ch.isalnum())
        safe_code = "".join(ch if ch.isalnum() else "_" for ch in self.code)
        path = directory / f"runcard_{safe_code}_{stamp}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path


def build_run_card(
    result: BacktestResult,
    bars: pd.DataFrame,
    cfg: Config,
    strategy: Strategy,
    benchmark: BenchmarkResult,
    folds: FoldReport | None,
    bootstrap: BootstrapReport | None,
    monte_carlo: MonteCarloReport | None,
    warnings: list[str],
    deflated: Any | None = None,
    kelly: Any | None = None,
) -> RunCard:
    metrics = compute_metrics(result)
    return RunCard(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code=result.code,
        strategy=strategy.name,
        strategy_params=dict(strategy.params),
        bar_type=cfg.trading.bar_type,
        data_start=str(bars.iloc[0]["time_key"]),
        data_end=str(bars.iloc[-1]["time_key"]),
        bars=len(bars),
        costs={
            "commission_per_share": cfg.backtest.commission_per_share,
            "min_commission": cfg.backtest.min_commission,
            "slippage_bps": cfg.backtest.slippage_bps,
        },
        exits=asdict(cfg.exits),
        risk=asdict(cfg.risk),
        metrics=metrics,
        benchmark={
            "buy_and_hold_return_pct": benchmark.total_return_pct,
            "buy_and_hold_max_drawdown_pct": benchmark.max_drawdown_pct,
            "buy_and_hold_sharpe": benchmark.sharpe,
            **benchmark.compare(metrics),
        },
        fold_summary=folds.summary() if folds else {},
        bootstrap=asdict(bootstrap) if bootstrap else None,
        monte_carlo=asdict(monte_carlo) if monte_carlo else None,
        warnings=warnings,
        deflated_sharpe=asdict(deflated) if deflated else None,
        kelly=asdict(kelly) if kelly else None,
    )
