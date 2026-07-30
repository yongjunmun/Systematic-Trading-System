"""Command line interface.

    python -m moobot.cli <command> [options]

Commands
    check      Validate settings and probe the OpenD connection
    account    Show paper-account balances
    positions  Show open positions and the bot's tracked stops
    signals    Print the current signal for every configured symbol (no orders)
    scan       Rank a universe by gap and relative volume, write a watchlist
    chain      Show the option contract the bot would pick for a symbol
    backtest   Run the strategy over history and print performance statistics
    validate   Stress the backtest: benchmark, folds, bootstrap, Monte Carlo
    report     Summarise the trade journal (P&L, win rate, R distribution)
    run        Start the live paper-trading loop
    strategies List the available strategies and their parameters
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import backtest_symbol, combine, format_metrics, r_histogram
from .broker import Broker, BrokerError
from .datafeed import DataError, DataFeed, normalise_bars
from .journal import Journal
from .notify import Notifier
from .options import select_contract
from .risk import RiskManager
from .scanner import read_watchlist, scan, write_watchlist
from .session import SessionGate
from .settings import VALID_BAR_TYPES, Config, ConfigError, load_config
from .stats import deflated_sharpe_ratio, kelly_report
from .strategies import available, get_strategy
from .validation import (
    bootstrap_expectancy,
    build_run_card,
    buy_and_hold,
    collect_warnings,
    evaluate_grid,
    fold_stability,
    monte_carlo_sequences,
    walk_forward,
)

DEFAULT_SETTINGS = "settings.toml"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The SDK is chatty about protocol internals at INFO level.
    logging.getLogger("moomoo").setLevel(logging.WARNING)


def banner(cfg: Config) -> str:
    env = cfg.account.trd_env.upper()
    tag = "PAPER MONEY" if env == "SIMULATE" else "*** REAL MONEY ***"
    return (
        f"moobot {__version__} | {env} ({tag}) | market {cfg.account.market} | "
        f"strategy {cfg.strategy.name} | dry_run={cfg.trading.dry_run}"
    )


# --------------------------------------------------------------------- commands


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    print(banner(cfg))
    print(f"Symbols   : {', '.join(cfg.trading.codes)}")
    print(f"Bar type  : {cfg.trading.bar_type}   poll every {cfg.trading.poll_seconds}s")
    print(f"Strategy  : {get_strategy(cfg.strategy.name, **cfg.strategy.params)}")
    print(
        f"Risk      : {cfg.risk.risk_per_trade_pct}%/trade, max {cfg.risk.max_positions} "
        f"positions, stop {cfg.risk.stop_loss_pct}%, daily halt at "
        f"-{cfg.risk.max_daily_loss_pct}%"
    )
    if cfg.exits.mode == "ladder":
        print(
            f"Exits     : ladder - scale {cfg.exits.partial_fraction:.0%} at "
            f"{cfg.exits.partial_at_r}R, breakeven at {cfg.exits.breakeven_at_r}R, "
            f"target {cfg.exits.target_r}R, trail by {cfg.exits.trail_method}"
        )
    else:
        print(f"Exits     : simple stop/target, trail by {cfg.exits.trail_method}")
    if cfg.session.enabled:
        status = SessionGate(cfg.session).status()
        print(
            f"Session   : {cfg.session.earliest_entry}-{cfg.session.latest_entry} "
            f"{cfg.session.timezone}, force close {cfg.session.force_close}, "
            f"max {cfg.session.max_trades_per_day} trades/day"
        )
        print(f"            now: {status}")
    if cfg.risk.max_correlation > 0:
        print(
            f"Correlation: reject a new name above {cfg.risk.max_correlation:.2f} "
            f"correlation over {cfg.risk.correlation_lookback} bars"
        )
    if cfg.trading.use_watchlist:
        codes = read_watchlist(cfg.scanner.watchlist_path)
        state = f"{len(codes)} symbols" if codes else "EMPTY - run `scan` first"
        print(f"Watchlist : {cfg.scanner.watchlist_path} ({state})")
    print(f"Alerts    : {Notifier(cfg.notifications).describe()}")
    if cfg.options.enabled:
        print(
            f"Options   : {cfg.options.style}, {cfg.options.min_dte}-{cfg.options.max_dte} DTE, "
            f"target delta {cfg.options.target_delta}, max {cfg.options.max_contracts} contracts"
        )
    print("Settings look valid.\n")

    print(f"Connecting to OpenD at {cfg.connection.host}:{cfg.connection.port} ...")
    ok = True
    with DataFeed(
        cfg.connection.host, cfg.connection.port, cfg.connection.security_firm,
        cfg.data_dir / "cache",
    ) as feed:
        try:
            state = feed.global_state()
            print(f"  OpenD OK. Server ver {state.get('server_ver', '?')}, "
                  f"logged in: {state.get('qot_logined', '?')}")
            print(f"  Market states: US={state.get('market_us', '?')} "
                  f"HK={state.get('market_hk', '?')}")
        except DataError as exc:
            print(f"  QUOTE FAILED: {exc}")
            ok = False

        try:
            with Broker(cfg) as broker:
                accounts = broker.account_list()
                print(f"\n  Trading accounts visible to OpenD ({len(accounts)}):")
                cols = [c for c in ("acc_id", "trd_env", "acc_type", "sim_acc_type",
                                    "trdmarket_auth", "acc_status") if c in accounts.columns]
                print("    " + accounts[cols].to_string(index=False).replace("\n", "\n    "))
                acc = broker.account()
                print(f"\n  Selected acc_id={broker.acc_id} equity={acc.equity:,.2f} "
                      f"{acc.currency} cash={acc.cash:,.2f}")
        except (BrokerError, ConfigError) as exc:
            print(f"\n  TRADE FAILED: {exc}")
            ok = False

    print("\nAll good - you can run a backtest or start the loop."
          if ok else "\nFix the errors above before trading.")
    return 0 if ok else 1


def cmd_account(cfg: Config, args: argparse.Namespace) -> int:
    with Broker(cfg) as broker:
        acc = broker.account()
        print(banner(cfg))
        print(f"acc_id        {broker.acc_id}")
        print(f"equity        {acc.equity:,.2f} {acc.currency}")
        print(f"cash          {acc.cash:,.2f}")
        print(f"buying power  {acc.buying_power:,.2f}")
        print(f"market value  {acc.market_value:,.2f}")
        if args.raw:
            print("\nraw fields:")
            for key, value in sorted(acc.raw.items()):
                print(f"  {key:<28} {value}")
    return 0


def cmd_positions(cfg: Config, args: argparse.Namespace) -> int:
    with Broker(cfg) as broker:
        positions = broker.positions()
        print(banner(cfg))
        if positions is None or positions.empty:
            print("\nNo open positions.")
        else:
            cols = [c for c in ("code", "stock_name", "qty", "can_sell_qty", "cost_price",
                                "nominal_price", "pl_val", "pl_ratio", "market_val")
                    if c in positions.columns]
            print("\nBroker positions:")
            print(positions[cols].to_string(index=False))

        orders = broker.open_orders()
        if orders is not None and not orders.empty:
            cols = [c for c in ("order_id", "code", "trd_side", "order_type", "qty",
                                "price", "order_status") if c in orders.columns]
            print("\nWorking orders:")
            print(orders[cols].to_string(index=False))

    with Journal(cfg.data_dir / "journal.db") as journal:
        holdings = journal.all_holdings()
        if holdings:
            print("\nBot-tracked stops and targets:")
            rows = [
                {
                    "code": h.code, "qty": h.qty, "entry": round(h.entry_price, 4),
                    "stop": round(h.stop_price, 4), "target": round(h.target_price, 4),
                    "peak": round(h.high_water, 4), "opened": h.opened_ts,
                }
                for h in holdings.values()
            ]
            print(pd.DataFrame(rows).to_string(index=False))
        else:
            print("\nBot is not tracking any positions.")
    return 0


def cmd_signals(cfg: Config, args: argparse.Namespace) -> int:
    strategy = get_strategy(cfg.strategy.name, **cfg.strategy.params)
    print(banner(cfg))
    print(f"\n{strategy}\n")
    rows = []
    with DataFeed(
        cfg.connection.host, cfg.connection.port, cfg.connection.security_firm,
        cfg.data_dir / "cache",
    ) as feed:
        for code in cfg.trading.codes:
            try:
                bars = feed.history(
                    code, bar_type=cfg.trading.bar_type,
                    max_bars=cfg.trading.history_bars, refresh=args.refresh,
                )
                signal = strategy.latest_signal(bars)
                rows.append({
                    "code": code,
                    "last_bar": str(bars.iloc[-1]["time_key"]),
                    "close": round(float(bars.iloc[-1]["close"]), 4),
                    "action": signal.action.value,
                    "strength": round(signal.strength, 2),
                    "stop": round(signal.stop_price, 4) if signal.stop_price else "",
                    "reason": signal.reason,
                })
            except DataError as exc:
                rows.append({"code": code, "action": "ERROR", "reason": str(exc)[:80]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n(No orders were placed. This command is read-only.)")
    return 0


def cmd_chain(cfg: Config, args: argparse.Namespace) -> int:
    code = args.code or cfg.trading.codes[0]
    with DataFeed(
        cfg.connection.host, cfg.connection.port, cfg.connection.security_firm,
        cfg.data_dir / "cache",
    ) as feed:
        print(f"Option chain scan for {code} "
              f"({cfg.options.min_dte}-{cfg.options.max_dte} DTE, "
              f"target delta {cfg.options.target_delta})")
        expirations = feed.option_expirations(code)
        if expirations is not None and not expirations.empty:
            print("\nAvailable expirations:")
            print(expirations.to_string(index=False))
        quote = select_contract(feed, code, args.direction, cfg.options)
        if quote is None:
            print("\nNo contract passed the liquidity and premium filters.")
            return 1
        print(f"\nSelected: {quote}")
        print(f"  bid/ask       {quote.bid:.2f} / {quote.ask:.2f}  (mid {quote.mid:.2f})")
        print(f"  cost/contract {quote.cost_per_contract:,.2f}")
        print(f"  implied vol   {quote.implied_vol:.2f}")
    return 0


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    strategy = get_strategy(cfg.strategy.name, **cfg.strategy.params)
    risk = RiskManager(cfg.risk)
    codes = args.codes or cfg.trading.codes
    bar_type = args.bar_type or cfg.trading.bar_type
    per_symbol_cash = cfg.backtest.initial_cash / len(codes)

    print(banner(cfg))
    print(f"\nBacktesting {strategy} on {', '.join(codes)} "
          f"({bar_type}, {cfg.backtest.initial_cash:,.0f} split "
          f"{per_symbol_cash:,.0f} per symbol)\n")

    results = []
    for code in codes:
        try:
            bars = _load_bars(cfg, code, args)
            result = backtest_symbol(
                bars=bars,
                strategy=get_strategy(cfg.strategy.name, **cfg.strategy.params),
                risk=risk,
                code=code,
                initial_cash=per_symbol_cash,
                commission_per_share=cfg.backtest.commission_per_share,
                min_commission=cfg.backtest.min_commission,
                slippage_bps=cfg.backtest.slippage_bps,
                bar_type=bar_type,
                allow_shorts=cfg.trading.allow_shorts,
                exits=cfg.exits,
            )
        except (ValueError, DataError) as exc:
            print(f"{code}: skipped - {exc}\n")
            continue
        results.append(result)
        print(format_metrics(result))
        print()

    if not results:
        print("No symbol produced a usable backtest.")
        return 1

    if len(results) > 1:
        print("=" * 62)
        print("PORTFOLIO")
        print("=" * 62)
        print(format_metrics(combine(results)))
        print()

    all_r = [r for result in results for r in result.r_multiples]
    if all_r:
        print("R-multiple distribution (outcome relative to money risked):")
        print(r_histogram(all_r))
        print()

    if args.trades:
        frame = pd.concat([r.trades_frame() for r in results], ignore_index=True)
        if frame.empty:
            print("No trades were taken.")
        else:
            print("Trades:")
            print(frame.to_string(index=False))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([r.trades_frame() for r in results], ignore_index=True).to_csv(out, index=False)
        print(f"\nTrades written to {out}")
    return 0


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    """Ask how much the backtest should be believed, not just what it returned."""
    code = args.code or cfg.trading.codes[0]
    bar_type = args.bar_type or cfg.trading.bar_type
    v = cfg.validation
    risk = RiskManager(cfg.risk)

    bars = _load_bars(cfg, code, args)
    strategy = get_strategy(cfg.strategy.name, **cfg.strategy.params)

    print(banner(cfg))
    print(f"\nValidating {strategy} on {code} ({bar_type}, {len(bars):,} bars: "
          f"{str(bars.iloc[0]['time_key'])[:10]} to {str(bars.iloc[-1]['time_key'])[:10]})\n")

    result = backtest_symbol(
        bars=bars, strategy=strategy, risk=risk, code=code,
        initial_cash=cfg.backtest.initial_cash,
        commission_per_share=cfg.backtest.commission_per_share,
        min_commission=cfg.backtest.min_commission,
        slippage_bps=cfg.backtest.slippage_bps,
        bar_type=bar_type, exits=cfg.exits,
    )
    metrics = result.metrics()
    print("BASELINE")
    print(format_metrics(result))

    print("\nR-MULTIPLE DISTRIBUTION")
    print(r_histogram(result.r_multiples))

    print("\nBENCHMARK  (buy and hold, same costs)")
    bench = buy_and_hold(
        bars, cfg.backtest.initial_cash, bar_type,
        cfg.backtest.commission_per_share, cfg.backtest.min_commission,
        cfg.backtest.slippage_bps,
    )
    delta = bench.compare(metrics)
    print(f"  Buy and hold return   {bench.total_return_pct:+.2f}% "
          f"(max drawdown {bench.max_drawdown_pct:.2f}%, Sharpe {bench.sharpe:.2f})")
    print(f"  Strategy excess       {delta['excess_return_pct']:+.2f}%")
    print(f"  Drawdown saved        {delta['drawdown_saved_pct']:+.2f} points")
    print(f"  Sharpe delta          {delta['sharpe_delta']:+.2f}")
    # Sharpe alone is a poor arbiter here: a strategy in the market 5% of the
    # time is not comparable to one that is always invested. Judge on money.
    beat_return = delta["excess_return_pct"] > 0
    beat_risk = delta["drawdown_saved_pct"] > 0
    if beat_return and beat_risk:
        verdict = "beat buy-and-hold on both return and drawdown"
    elif beat_return:
        verdict = "beat buy-and-hold on return, but took more drawdown doing it"
    elif beat_risk:
        verdict = "lagged buy-and-hold on return but was far less exposed"
    else:
        verdict = "lost to buy-and-hold on both counts - holding would have been better"
    print(f"  -> {verdict}")
    if metrics["exposure_pct"] < 25:
        print(f"     (in the market only {metrics['exposure_pct']:.1f}% of the time, so "
              "Sharpe is not directly comparable)")

    folds = None
    label = "FOLD STABILITY  (no re-fitting - is the edge present throughout?)"
    try:
        if v.grid:
            folds = walk_forward(
                bars, lambda **kw: get_strategy(cfg.strategy.name, **{**cfg.strategy.params, **kw}),
                risk, cfg, v.grid, v.walk_forward_folds, code,
            )
            label = "WALK-FORWARD  (parameters fitted only on earlier data)"
        else:
            folds = fold_stability(
                bars, lambda **kw: get_strategy(cfg.strategy.name, **{**cfg.strategy.params, **kw}),
                risk, cfg, v.walk_forward_folds, code,
            )
        print(f"\n{label}")
        rows = [
            {
                "fold": f.index, "from": f.start[:10], "to": f.end[:10], "bars": f.bars,
                "trades": f.trades, "return%": round(f.return_pct, 2),
                "maxDD%": round(f.max_drawdown_pct, 2),
                "expectancy_R": round(f.expectancy_r, 3),
                **({"params": f.params} if f.params else {}),
            }
            for f in folds.folds
        ]
        print(pd.DataFrame(rows).to_string(index=False))
        if len(folds.folds) < v.walk_forward_folds and not folds.refitted:
            print(f"  (reduced from {v.walk_forward_folds} folds so each keeps enough "
                  "warmup bars)")
        summary = folds.summary()
        if summary:
            print(f"  {int(summary['profitable_folds'])}/{int(summary['folds'])} folds "
                  f"profitable ({summary['consistency_pct']:.0f}% consistency), "
                  f"worst {summary['worst_fold_pct']:+.2f}%, best {summary['best_fold_pct']:+.2f}%")
            if summary["profitable_folds"] == 0:
                print("  -> no period was profitable; the strategy is not working here at all")
            elif summary["consistency_pct"] < 60:
                print("  -> the edge is concentrated in a minority of periods; treat with "
                      "suspicion")
            else:
                print("  -> the edge shows up across periods rather than in one lucky run")
    except ValueError as exc:
        print(f"\nFOLD STABILITY  skipped: {exc}")

    boot = bootstrap_expectancy(
        result.r_multiples, v.bootstrap_samples, v.confidence_pct, v.random_seed
    )
    if boot:
        print(f"\nBOOTSTRAP  ({boot.samples:,} resamples of the {len(result.trades)} trades)")
        print(f"  Observed expectancy   {boot.observed_expectancy_r:+.3f}R per trade")
        print(f"  {boot.confidence_pct:.0f}% interval        "
              f"{boot.lower_r:+.3f}R to {boot.upper_r:+.3f}R")
        print(f"  P(edge > 0)           {boot.probability_positive:.1f}%")
        print(f"  -> {'edge is statistically distinguishable from luck' if boot.significant else 'CANNOT rule out that this is luck'}")
    else:
        print("\nBOOTSTRAP  skipped: fewer than 5 closed trades")

    mc = monte_carlo_sequences(
        result.r_multiples, v.monte_carlo_runs, cfg.risk.risk_per_trade_pct,
        seed=v.random_seed,
    )
    if mc:
        print(f"\nMONTE CARLO  ({mc.runs:,} reshuffles at "
              f"{mc.risk_per_trade_pct:g}% risk per trade)")
        print(f"  Return  p5 / median / p95   {mc.return_p5_pct:+.1f}% / "
              f"{mc.median_return_pct:+.1f}% / {mc.return_p95_pct:+.1f}%")
        print(f"  Max drawdown median         {mc.median_max_drawdown_pct:.1f}%")
        print(f"  Max drawdown 95th worst     {mc.drawdown_p95_pct:.1f}%")
        print(f"  Deepest seen                {mc.worst_max_drawdown_pct:.1f}%")
        print(f"  P(losing overall)           {mc.probability_of_loss_pct:.1f}%")
        print(f"  P(drawdown worse than {mc.ruin_threshold_pct:.0f}%) {mc.probability_ruin_pct:.1f}%")
        print("  -> the same trades in a different order could realistically have "
              f"drawn down {abs(mc.drawdown_p95_pct):.0f}%. Size for that, not for the median.")
    else:
        print("\nMONTE CARLO  skipped: fewer than 5 closed trades")

    # Every parameter combination tried is another lottery ticket. Deflating
    # prices that in; with no grid the trial count is 1 and nothing is deflated.
    trial_sharpes = None
    if v.grid:
        trial_sharpes = evaluate_grid(
            bars,
            lambda **kw: get_strategy(cfg.strategy.name, **{**cfg.strategy.params, **kw}),
            risk, cfg, v.grid, code,
        )
    returns = result.equity.pct_change().dropna().tolist()
    deflated = deflated_sharpe_ratio(returns, result.bars_per_year, trial_sharpes)
    if deflated:
        print(f"\nSHARPE DEFLATION  ({deflated.trials} parameter "
              f"combination{'s' if deflated.trials != 1 else ''} tried)")
        print(f"  Observed Sharpe       {deflated.observed_sharpe_annual:.2f}")
        print(f"  Skew / kurtosis       {deflated.skew:+.2f} / {deflated.kurtosis:.2f}")
        print(f"  Probabilistic Sharpe  {deflated.probabilistic_sharpe_pct:.1f}% "
              "confident the true Sharpe is above 0")
        if deflated.trials > 1:
            print(f"  Luck threshold        {deflated.threshold_sharpe_annual:.2f} Sharpe "
                  f"expected from the best of {deflated.trials} worthless strategies")
            print(f"  Deflated Sharpe       {deflated.deflated_sharpe_pct:.1f}%")
            print("  -> " + ("clears the multiple-testing bar" if deflated.survives
                             else "does NOT clear the bar once the search is accounted for"))
        else:
            print("  -> no parameter search was run, so there is nothing to deflate")
        if metrics["exposure_pct"] < 25:
            print(f"     (flat {100 - metrics['exposure_pct']:.0f}% of the time, so the "
                  "return series is mostly zeros and these Sharpe figures are strained)")

    kelly = kelly_report(result.r_multiples, cfg.risk.risk_per_trade_pct)
    if kelly:
        print("\nBET SIZING  (growth-optimal fraction from the observed R distribution)")
        print(f"  Full Kelly            {kelly.full_kelly_pct:.2f}% of equity per trade")
        print(f"  Half Kelly            {kelly.half_kelly_pct:.2f}%")
        print(f"  Configured            {kelly.configured_pct:.2f}%")
        print(f"  -> {kelly.verdict}")

    notes = collect_warnings(result, bars, v.min_trades_for_significance, boot, deflated)
    print("\nWARNINGS")
    if notes:
        for note in notes:
            print(f"  ! {note}")
    else:
        print("  none - but that is not the same as 'this will make money'")

    card = build_run_card(result, bars, cfg, strategy, bench, folds, boot, mc, notes,
                          deflated, kelly)
    path = card.save(cfg.data_dir / "runcards")
    print(f"\nRun card written to {path}")
    return 0


def cmd_scan(cfg: Config, args: argparse.Namespace) -> int:
    universe = args.codes or cfg.scanner.universe or cfg.trading.codes
    print(banner(cfg))
    print(f"\nScanning {len(universe)} symbols: price >= {cfg.scanner.min_price:g}, "
          f"|gap| >= {cfg.scanner.min_gap_pct:g}%, rvol >= {cfg.scanner.min_rvol:g}x\n")

    report = scan(_feed(cfg), universe, cfg.scanner, refresh=args.refresh)
    print(report.summary())
    if not report.survivors:
        print("\nNothing passed the filters. On a quiet day that is the correct answer.")
        return 0

    print("\nSurvivors, best first:")
    for i, candidate in enumerate(report.survivors, start=1):
        print(f"  {i:>2}. {candidate}")

    if args.write:
        path = write_watchlist(report, cfg.scanner.watchlist_path, cfg.scanner)
        print(f"\nWatchlist written to {path}")
        print("Set [trading].use_watchlist = true to have the engine trade it.")
    else:
        print("\n(Pass --write to save this as the watchlist.)")
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    since = None
    if args.days:
        since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(
            timespec="seconds"
        )

    with Journal(cfg.data_dir / "journal.db") as journal:
        trades = journal.closed_trades(since)
        curve = journal.equity_curve()
        holdings = journal.all_holdings()

    window = f"last {args.days} days" if args.days else "all time"
    print(banner(cfg))
    print(f"\nTRADE REPORT ({window})\n")

    if not trades:
        print("No closed trades recorded yet.")
    else:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        r_values = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
        gross_loss = -sum(losses)
        pf = sum(wins) / gross_loss if gross_loss > 0 else float("inf")

        print(f"  Closed trades   {len(trades)}")
        print(f"  Win rate        {len(wins) / len(trades) * 100:.1f}% "
              f"({len(wins)}W / {len(losses)}L)")
        print(f"  Net P&L         {sum(pnls):+,.2f}")
        print(f"  Profit factor   {'inf' if pf == float('inf') else f'{pf:.2f}'}")
        if r_values:
            print(f"  Expectancy      {sum(r_values) / len(r_values):+.3f}R per trade")
        best = max(trades, key=lambda t: t["pnl"])
        worst = min(trades, key=lambda t: t["pnl"])
        print(f"  Best / worst    {best['code']} {best['pnl']:+,.2f} / "
              f"{worst['code']} {worst['pnl']:+,.2f}")
        if r_values:
            print("\n  R-multiple distribution:")
            print(r_histogram(r_values))
        if args.trades:
            frame = pd.DataFrame(trades)[
                ["ts", "code", "entry_price", "exit_price", "qty", "pnl", "r_multiple", "reason"]
            ]
            print("\n  Trades:")
            print(frame.to_string(index=False))

    if curve:
        first, last = curve[0]["equity"], curve[-1]["equity"]
        peak = max(point["equity"] for point in curve)
        drawdown = (last - peak) / peak * 100 if peak else 0.0
        print(f"\n  Equity          {first:,.2f} -> {last:,.2f} "
              f"({(last / first - 1) * 100:+.2f}%)" if first else "")
        print(f"  From peak       {drawdown:+.2f}%")
        print(f"  Samples logged  {len(curve)}")

    if holdings:
        print(f"\n  Open positions ({len(holdings)}):")
        for h in holdings.values():
            print(f"    {h.code:<14} {h.qty:>8g} @ {h.entry_price:.4f}  "
                  f"stop {h.stop_price:.4f}  initial stop {h.initial_stop:.4f}"
                  f"{'  [partial taken]' if h.partial_done else ''}"
                  f"{'  [at breakeven]' if h.breakeven_done else ''}")
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from .engine import TradingEngine

    dry_run = cfg.trading.dry_run
    if args.live_orders:
        dry_run = False
    if args.dry_run:
        dry_run = True

    print(banner(cfg))
    if not dry_run:
        print("\nOrders WILL be sent to the paper account.")
    else:
        print("\nDry run: signals and sizing are computed but no orders are sent.")
    print()

    with TradingEngine(cfg, dry_run=dry_run) as engine:
        engine.run(max_cycles=args.cycles)
    return 0


def cmd_strategies(cfg: Config, args: argparse.Namespace) -> int:
    import inspect

    print("Available strategies:\n")
    for name in available():
        strategy = get_strategy(name)
        sig = inspect.signature(type(strategy).configure)
        defaults = {
            p.name: p.default for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty and p.name != "_"
        }
        doc = (type(strategy).__doc__ or "").strip().split("\n")[0]
        print(f"  {name}")
        print(f"      {doc}")
        print(f"      warmup: {strategy.warmup} bars")
        print(f"      params: {defaults}")
        print()
    print("Set one in settings.toml:\n\n  [strategy]\n  name = \"sma_cross\"\n"
          "  [strategy.params]\n  fast = 20\n  slow = 50\n")
    return 0


# ---------------------------------------------------------------------- helpers


def _feed(cfg: Config) -> DataFeed:
    """One shared quote connection per process."""
    if "feed" not in _FEED_CACHE:
        _FEED_CACHE["feed"] = DataFeed(
            cfg.connection.host, cfg.connection.port, cfg.connection.security_firm,
            cfg.data_dir / "cache",
        )
    return _FEED_CACHE["feed"]


_FEED_CACHE: dict[str, DataFeed] = {}


def _load_bars(cfg: Config, code: str, args: argparse.Namespace) -> pd.DataFrame:
    """Bars for a backtest: from a CSV if given, otherwise moomoo history."""
    if args.csv:
        path = Path(args.csv)
        if path.is_dir():
            path = path / f"{code.replace('.', '_')}.csv"
        if not path.is_file():
            raise SystemExit(f"CSV not found for {code}: {path}")
        return normalise_bars(pd.read_csv(path))

    return _feed(cfg).history(
        code,
        bar_type=args.bar_type or cfg.trading.bar_type,
        start=cfg.backtest.start or None,
        end=cfg.backtest.end or None,
        max_bars=args.max_bars,
        use_cache=cfg.backtest.use_cache,
        refresh=args.refresh,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moobot",
        description="Paper-money trading bot for moomoo / Futu OpenAPI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--settings", default=DEFAULT_SETTINGS,
                        help=f"path to the TOML settings file (default: {DEFAULT_SETTINGS})")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"moobot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate settings and probe OpenD").set_defaults(func=cmd_check)

    p = sub.add_parser("account", help="show paper account balances")
    p.add_argument("--raw", action="store_true", help="print every field moomoo returns")
    p.set_defaults(func=cmd_account)

    sub.add_parser("positions", help="show positions, orders and tracked stops").set_defaults(
        func=cmd_positions
    )

    p = sub.add_parser("signals", help="print current signals without trading")
    p.add_argument("--refresh", action="store_true", help="bypass the local bar cache")
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("chain", help="show the option contract that would be chosen")
    p.add_argument("code", nargs="?", help="underlying, e.g. US.AAPL")
    p.add_argument("--direction", default="BUY", choices=["BUY", "SELL"],
                   help="BUY selects calls, SELL selects puts")
    p.set_defaults(func=cmd_chain)

    p = sub.add_parser("backtest", help="run the strategy over history")
    p.add_argument("codes", nargs="*", help="override the symbols in settings")
    p.add_argument("--bar-type", choices=sorted(VALID_BAR_TYPES),
                   help="override [trading].bar_type (set this when using --csv)")
    p.add_argument("--max-bars", type=int, default=5000, help="cap on bars per symbol")
    p.add_argument("--refresh", action="store_true", help="bypass the local bar cache")
    p.add_argument("--csv", help="CSV file or directory to backtest instead of moomoo data")
    p.add_argument("--trades", action="store_true", help="print the trade list")
    p.add_argument("--save", help="write the trade list to this CSV")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser(
        "validate",
        help="stress the backtest: benchmark, fold stability, bootstrap, Monte Carlo",
    )
    p.add_argument("code", nargs="?", help="symbol to validate (default: the first configured)")
    p.add_argument("--bar-type", choices=sorted(VALID_BAR_TYPES),
                   help="override [trading].bar_type (set this when using --csv)")
    p.add_argument("--max-bars", type=int, default=5000, help="cap on bars")
    p.add_argument("--refresh", action="store_true", help="bypass the local bar cache")
    p.add_argument("--csv", help="CSV file or directory to use instead of moomoo data")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("scan", help="rank a universe by gap and relative volume")
    p.add_argument("codes", nargs="*", help="symbols to scan (default: [scanner].universe)")
    p.add_argument("--write", action="store_true", help="save the survivors as the watchlist")
    p.add_argument("--refresh", action="store_true", help="bypass the local bar cache")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("report", help="summarise the trade journal")
    p.add_argument("--days", type=int, help="only include the last N days")
    p.add_argument("--trades", action="store_true", help="list every closed trade")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", help="start the paper-trading loop")
    p.add_argument("--cycles", type=int, help="stop after this many cycles (default: forever)")
    p.add_argument("--live-orders", action="store_true",
                   help="actually send orders to the paper account")
    p.add_argument("--dry-run", action="store_true", help="force dry run, overriding settings")
    p.set_defaults(func=cmd_run)

    sub.add_parser("strategies", help="list strategies and their parameters").set_defaults(
        func=cmd_strategies
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = load_config(args.settings)
    except ConfigError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        return 2

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        return args.func(cfg, args)
    except (BrokerError, DataError, ConfigError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        for feed in _FEED_CACHE.values():
            feed.close()
        _FEED_CACHE.clear()


if __name__ == "__main__":
    raise SystemExit(main())
