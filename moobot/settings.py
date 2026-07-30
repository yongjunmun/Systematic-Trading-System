"""Settings loading and validation.

Settings live in a TOML file (stdlib ``tomllib``, no extra dependency).
No secrets belong in it - the trade-unlock password is read from the
``MOOBOT_TRADE_PASSWORD_MD5`` environment variable only.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time
from pathlib import Path
from typing import Any, get_type_hints
from zoneinfo import ZoneInfo

# The only value that lets the bot touch a live-money account.
REAL_MONEY_ACK = "I_ACCEPT_THE_RISK"

VALID_BAR_TYPES = {
    "K_1M", "K_3M", "K_5M", "K_10M", "K_15M", "K_30M", "K_60M",
    "K_120M", "K_180M", "K_240M", "K_DAY", "K_WEEK", "K_MON",
}

# Bars per trading year, used to annualise backtest metrics.
BARS_PER_YEAR = {
    "K_1M": 252 * 390, "K_3M": 252 * 130, "K_5M": 252 * 78,
    "K_10M": 252 * 39, "K_15M": 252 * 26, "K_30M": 252 * 13,
    "K_60M": 252 * 7, "K_120M": 252 * 4, "K_180M": 252 * 3,
    "K_240M": 252 * 2, "K_DAY": 252, "K_WEEK": 52, "K_MON": 12,
}


class ConfigError(ValueError):
    """Raised when the settings file is missing or internally inconsistent."""


@dataclass
class ConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 11111
    # FUTUINC=moomoo US, FUTUSG, FUTUAU, FUTUCA, FUTUMY, FUTUJP,
    # FUTUSECURITIES=Futu HK.
    security_firm: str = "FUTUINC"
    timeout_seconds: int = 20


@dataclass
class AccountConfig:
    trd_env: str = "SIMULATE"
    market: str = "US"
    # 0 means "auto-select the first paper account that matches the market".
    acc_id: int = 0
    currency: str = "USD"
    # Both this AND the env var must be set to trade real money.
    allow_real_money: bool = False


@dataclass
class TradingConfig:
    codes: list[str] = field(default_factory=lambda: ["US.AAPL"])
    bar_type: str = "K_15M"
    history_bars: int = 400
    poll_seconds: int = 60
    order_type: str = "NORMAL"  # NORMAL = limit, MARKET = market
    limit_slippage_bps: float = 10.0
    allow_shorts: bool = False
    dry_run: bool = True
    only_trade_market_hours: bool = True
    # When true the engine trades whatever the scanner wrote to the watchlist
    # instead of the static `codes` list.
    use_watchlist: bool = False


@dataclass
class RiskConfig:
    max_positions: int = 5
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_daily_loss_pct: float = 3.0
    min_cash_buffer_pct: float = 5.0
    use_atr_stops: bool = False
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    # Concentration control. Four positions at 0.95 correlation are one position
    # with four sets of fees. 0 disables the check.
    max_correlation: float = 0.0
    correlation_lookback: int = 60
    # Volatility management. Fixed-fractional sizing quietly takes far more risk
    # in a turbulent market than a calm one because the stop is wider in both
    # cases but the chance of gapping through it is not. 0 disables each.
    target_volatility_pct: float = 0.0   # annualised; scales size toward this
    max_volatility_pct: float = 0.0      # annualised; blocks entry above this
    vol_lookback: int = 20
    max_vol_scale: float = 1.5           # cap on scaling UP in a quiet market


@dataclass
class ExitsConfig:
    """How an open position is managed after entry.

    `simple` uses the flat stop/target percentages from [risk].
    `ladder` scales out at a multiple of the initial risk (R), moves the stop to
    breakeven, then trails. Managing in R rather than in percent is what lets a
    sub-50% win rate still be profitable.
    """

    mode: str = "ladder"  # simple | ladder
    partial_at_r: float = 0.75      # scale out here (0 disables)
    partial_fraction: float = 0.34  # fraction of the position to sell
    breakeven_at_r: float = 1.0     # move the stop to entry here (0 disables)
    target_r: float = 3.0           # final target in R (0 = ride the trail out)
    trail_method: str = "atr"       # atr | swing_low | percent | none
    trail_atr_multiple: float = 2.0
    trail_percent: float = 5.0
    swing_left: int = 2             # bars either side that confirm a swing low
    swing_right: int = 2
    trail_only_after_breakeven: bool = True


@dataclass
class SessionConfig:
    """Intraday time gates: skip the opening auction, flatten before the close so
    the bot never carries overnight gap risk it did not choose."""

    enabled: bool = False
    timezone: str = "America/New_York"
    earliest_entry: str = "10:05"
    latest_entry: str = "15:30"
    force_close: str = "15:51"
    max_trades_per_day: int = 5


@dataclass
class ScannerConfig:
    enabled: bool = False
    universe: list[str] = field(default_factory=list)  # empty = [trading].codes
    min_price: float = 3.0
    min_gap_pct: float = 3.0
    min_rvol: float = 1.5
    rvol_lookback: int = 14
    max_symbols: int = 20
    watchlist_path: str = "data/watchlist.txt"


@dataclass
class ValidationConfig:
    walk_forward_folds: int = 4
    bootstrap_samples: int = 2000
    monte_carlo_runs: int = 2000
    confidence_pct: float = 95.0
    min_trades_for_significance: int = 30
    random_seed: int = 20260728
    # Gap between the end of each training window and the start of the fold it
    # is tested on. Without it, a trade opened at the end of training is still
    # open when the test period starts, so its outcome depends on test data.
    embargo_pct: float = 2.0
    # Optional parameter grid. When present, `validate` runs a true anchored
    # walk-forward: fit on the past, trade the fold it has never seen.
    grid: dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationsConfig:
    """Push alerts. Credentials come from the environment, never from this file:
    MOOBOT_TELEGRAM_TOKEN, MOOBOT_TELEGRAM_CHAT_ID, MOOBOT_WEBHOOK_URL."""

    enabled: bool = False
    on_entry: bool = True
    on_exit: bool = True
    on_halt: bool = True
    on_error: bool = True
    timeout_seconds: float = 5.0


@dataclass
class OptionsConfig:
    enabled: bool = False
    style: str = "long"  # long | covered_call | cash_secured_put
    min_dte: int = 14
    max_dte: int = 45
    target_delta: float = 0.35
    max_contracts: int = 1
    max_premium_per_trade: float = 500.0
    min_open_interest: int = 100
    max_spread_pct: float = 15.0
    # Options move several times harder than the underlying, so they get their
    # own exit levels instead of reusing the [risk] stock percentages.
    stop_loss_pct: float = 50.0            # long: close after losing this % of premium
    take_profit_pct: float = 100.0         # long: close after this % gain
    short_profit_target_pct: float = 50.0  # short: buy back once this % of credit is kept
    short_stop_multiple: float = 2.0       # short: buy back if premium reaches this x credit


@dataclass
class StrategyConfig:
    name: str = "sma_cross"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    start: str = ""
    end: str = ""
    initial_cash: float = 100_000.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5.0
    use_cache: bool = True


@dataclass
class Config:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exits: ExitsConfig = field(default_factory=ExitsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def is_paper(self) -> bool:
        return self.account.trd_env.upper() == "SIMULATE"

    @property
    def bars_per_year(self) -> int:
        return BARS_PER_YEAR.get(self.trading.bar_type, 252)


def _build(cls, raw: dict[str, Any], section: str):
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(
            f"[{section}] has unknown option(s): {', '.join(sorted(unknown))}. "
            f"Valid options: {', '.join(sorted(known))}"
        )
    # `from __future__ import annotations` makes field.type a string, so the
    # real types have to be resolved before they can be compared.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        target = hints.get(key)
        # dict-valued fields (strategy params, the walk-forward grid) are
        # free-form and pass through untouched.
        if isinstance(value, dict):
            kwargs[key] = value
            continue
        if target is int and isinstance(value, bool):
            raise ConfigError(f"[{section}].{key} must be a number, got a boolean")
        if target is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str]) -> Config:
    """Read and validate a TOML settings file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Settings file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    sections = {f.name: f for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown settings section(s): {', '.join(sorted(unknown))}")

    cfg = Config()
    for name, f in sections.items():
        if name not in raw:
            continue
        if name == "data_dir":
            cfg.data_dir = Path(raw[name])
            continue
        section_cls = f.default_factory()  # type: ignore[misc]
        if not is_dataclass(section_cls):
            continue
        setattr(cfg, name, _build(type(section_cls), raw[name], name))

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    t, r, o, a = cfg.trading, cfg.risk, cfg.options, cfg.account

    if a.trd_env.upper() not in {"SIMULATE", "REAL"}:
        raise ConfigError("[account].trd_env must be 'SIMULATE' or 'REAL'")
    if t.bar_type not in VALID_BAR_TYPES:
        raise ConfigError(
            f"[trading].bar_type '{t.bar_type}' is not valid. "
            f"Choose one of: {', '.join(sorted(VALID_BAR_TYPES))}"
        )
    if t.order_type not in {"NORMAL", "MARKET"}:
        raise ConfigError(
            "[trading].order_type must be 'NORMAL' (limit) or 'MARKET'. "
            "moomoo paper trading supports no other type."
        )
    if not t.codes:
        raise ConfigError("[trading].codes is empty - nothing to trade")
    bad = [c for c in t.codes if "." not in c]
    if bad:
        raise ConfigError(
            f"[trading].codes entries must be 'MARKET.SYMBOL' (e.g. 'US.AAPL'). Bad: {bad}"
        )
    if t.history_bars < 60:
        raise ConfigError("[trading].history_bars should be >= 60 for indicators to warm up")
    if t.poll_seconds < 5:
        raise ConfigError("[trading].poll_seconds must be >= 5 to stay within API rate limits")

    for name, value in (
        ("risk_per_trade_pct", r.risk_per_trade_pct),
        ("max_position_pct", r.max_position_pct),
        ("max_daily_loss_pct", r.max_daily_loss_pct),
        ("stop_loss_pct", r.stop_loss_pct),
    ):
        if not 0 < value <= 100:
            raise ConfigError(f"[risk].{name} must be between 0 and 100, got {value}")
    if r.max_positions < 1:
        raise ConfigError("[risk].max_positions must be >= 1")

    if not 0 <= r.max_correlation <= 1:
        raise ConfigError("[risk].max_correlation must be between 0 and 1 (0 disables)")
    if r.target_volatility_pct < 0 or r.max_volatility_pct < 0:
        raise ConfigError("[risk] volatility limits cannot be negative (use 0 to disable)")
    if r.max_vol_scale < 1:
        raise ConfigError(
            "[risk].max_vol_scale must be >= 1 - it caps how far volatility "
            "targeting may scale a position UP"
        )
    if r.vol_lookback < 5:
        raise ConfigError("[risk].vol_lookback must be >= 5 to estimate volatility")
    if (
        r.max_volatility_pct
        and r.target_volatility_pct
        and r.max_volatility_pct < r.target_volatility_pct
    ):
        raise ConfigError(
            f"[risk].max_volatility_pct ({r.max_volatility_pct}) is below "
            f"target_volatility_pct ({r.target_volatility_pct}), so every trade that "
            "passes the target would be blocked by the ceiling"
        )

    _validate_exits(cfg.exits)
    _validate_session(cfg.session)
    _validate_scanner(cfg.scanner)
    _validate_validation(cfg.validation)

    if o.enabled:
        if o.style not in {"long", "covered_call", "cash_secured_put"}:
            raise ConfigError(
                "[options].style must be 'long', 'covered_call' or 'cash_secured_put'"
            )
        if not 0 < o.target_delta < 1:
            raise ConfigError("[options].target_delta must be between 0 and 1 (absolute value)")
        if o.min_dte < 0 or o.max_dte < o.min_dte:
            raise ConfigError("[options] requires 0 <= min_dte <= max_dte")
        if o.max_contracts < 1:
            raise ConfigError("[options].max_contracts must be >= 1")
        if not 0 < o.stop_loss_pct <= 100:
            raise ConfigError("[options].stop_loss_pct must be between 0 and 100")
        if not 0 < o.short_profit_target_pct <= 100:
            raise ConfigError("[options].short_profit_target_pct must be between 0 and 100")
        if o.short_stop_multiple <= 1:
            raise ConfigError("[options].short_stop_multiple must be greater than 1")


def parse_hhmm(value: str, label: str) -> time:
    """Parse a 'HH:MM' string from the settings file."""
    try:
        hour, minute = value.split(":")
        return time(int(hour), int(minute))
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"{label} must look like '09:30', got {value!r}") from exc


def _validate_exits(e: ExitsConfig) -> None:
    if e.mode not in {"simple", "ladder"}:
        raise ConfigError("[exits].mode must be 'simple' or 'ladder'")
    if e.trail_method not in {"atr", "swing_low", "percent", "none"}:
        raise ConfigError(
            "[exits].trail_method must be 'atr', 'swing_low', 'percent' or 'none'"
        )
    if e.mode != "ladder":
        return
    if e.partial_at_r and not 0 < e.partial_fraction < 1:
        raise ConfigError(
            "[exits].partial_fraction must be between 0 and 1 exclusive - "
            "selling the whole position is what target_r is for"
        )
    for name, value in (("partial_at_r", e.partial_at_r),
                        ("breakeven_at_r", e.breakeven_at_r),
                        ("target_r", e.target_r)):
        if value < 0:
            raise ConfigError(f"[exits].{name} cannot be negative (use 0 to disable)")
    if e.target_r and e.partial_at_r and e.partial_at_r >= e.target_r:
        raise ConfigError(
            f"[exits].partial_at_r ({e.partial_at_r}) must be below target_r "
            f"({e.target_r}), otherwise the whole position exits at the target first"
        )
    if e.swing_left < 1 or e.swing_right < 1:
        raise ConfigError("[exits].swing_left and swing_right must both be >= 1")


def _validate_session(s: SessionConfig) -> None:
    if not s.enabled:
        return
    try:
        ZoneInfo(s.timezone)
    except Exception as exc:  # ZoneInfoNotFoundError and friends
        raise ConfigError(
            f"[session].timezone '{s.timezone}' is not a known IANA zone "
            "(e.g. 'America/New_York', 'Asia/Hong_Kong')"
        ) from exc
    earliest = parse_hhmm(s.earliest_entry, "[session].earliest_entry")
    latest = parse_hhmm(s.latest_entry, "[session].latest_entry")
    force = parse_hhmm(s.force_close, "[session].force_close")
    if not earliest < latest:
        raise ConfigError("[session] requires earliest_entry < latest_entry")
    if force < latest:
        raise ConfigError(
            "[session].force_close must be at or after latest_entry, otherwise the "
            "bot opens trades it immediately flattens"
        )
    if s.max_trades_per_day < 1:
        raise ConfigError("[session].max_trades_per_day must be >= 1")


def _validate_scanner(s: ScannerConfig) -> None:
    if not s.enabled:
        return
    bad = [c for c in s.universe if "." not in c]
    if bad:
        raise ConfigError(f"[scanner].universe entries must be 'MARKET.SYMBOL'. Bad: {bad}")
    if s.max_symbols < 1:
        raise ConfigError("[scanner].max_symbols must be >= 1")
    if s.rvol_lookback < 2:
        raise ConfigError("[scanner].rvol_lookback must be >= 2")
    if s.min_price < 0 or s.min_rvol < 0:
        raise ConfigError("[scanner].min_price and min_rvol cannot be negative")


def _validate_validation(v: ValidationConfig) -> None:
    if v.walk_forward_folds < 2:
        raise ConfigError("[validation].walk_forward_folds must be >= 2 to compare folds")
    if v.bootstrap_samples < 100 or v.monte_carlo_runs < 100:
        raise ConfigError(
            "[validation] resampling counts must be >= 100 or the intervals are noise"
        )
    if not 50 < v.confidence_pct < 100:
        raise ConfigError("[validation].confidence_pct must be between 50 and 100")
    if not 0 <= v.embargo_pct < 50:
        raise ConfigError("[validation].embargo_pct must be between 0 and 50")
    for key, values in v.grid.items():
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"[validation.grid].{key} must be a non-empty list of values to try, "
                f"e.g. {key} = [10, 20, 30]"
            )


def assert_paper_trading(cfg: Config) -> None:
    """Gate that blocks live-money trading unless it is unambiguously requested.

    Requires all three of: trd_env=REAL in settings, allow_real_money=true in
    settings, and MOOBOT_ALLOW_REAL set to the exact acknowledgement string.
    """
    if cfg.is_paper:
        return
    ack = os.environ.get("MOOBOT_ALLOW_REAL", "")
    if not cfg.account.allow_real_money or ack != REAL_MONEY_ACK:
        raise ConfigError(
            "REFUSING TO RUN: [account].trd_env is 'REAL' (live money).\n"
            "This bot is built for paper trading. To override you must set BOTH\n"
            "  [account] allow_real_money = true\n"
            f"  and the environment variable MOOBOT_ALLOW_REAL={REAL_MONEY_ACK}\n"
            "Set trd_env = \"SIMULATE\" to trade paper money instead."
        )
