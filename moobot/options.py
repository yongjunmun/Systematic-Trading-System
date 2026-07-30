"""Option contract selection.

Turns a directional stock signal into a specific option contract by filtering
the chain on days-to-expiry, then scoring the survivors on how close their
delta is to the configured target.

Greeks are not in the option chain response - they come from a market
snapshot of the contract codes, which is what ``select_contract`` does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .datafeed import DataFeed, DataError
from .settings import OptionsConfig

log = logging.getLogger(__name__)

# US equity options are 100 shares per contract; the snapshot confirms it.
DEFAULT_CONTRACT_SIZE = 100


@dataclass
class OptionQuote:
    code: str
    underlying: str
    option_type: str  # CALL or PUT
    strike: float
    expiry: str
    dte: int
    bid: float
    ask: float
    last: float
    delta: float
    implied_vol: float
    open_interest: int
    contract_size: int

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a percentage of the mid. Wide = expensive to trade."""
        mid = self.mid
        if mid <= 0 or self.bid <= 0 or self.ask <= 0:
            return 100.0
        return (self.ask - self.bid) / mid * 100.0

    @property
    def cost_per_contract(self) -> float:
        return self.mid * self.contract_size

    def __str__(self) -> str:
        return (
            f"{self.code} {self.option_type} {self.strike:g} exp {self.expiry} "
            f"({self.dte}d) delta {self.delta:+.3f} mid {self.mid:.2f} "
            f"spread {self.spread_pct:.1f}% OI {self.open_interest}"
        )


def _days_to_expiry(strike_time: str) -> int:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return (datetime.strptime(str(strike_time)[:19], fmt).date() - date.today()).days
        except ValueError:
            continue
    return -1


def _num(row: pd.Series, name: str, default: float = 0.0) -> float:
    if name in row and pd.notna(row[name]):
        try:
            return float(row[name])
        except (TypeError, ValueError):
            return default
    return default


def select_contract(
    feed: DataFeed,
    underlying: str,
    direction: str,
    cfg: OptionsConfig,
    max_candidates: int = 60,
) -> OptionQuote | None:
    """Pick the best contract for a `direction` of 'BUY' (call) or 'SELL' (put).

    Returns None when nothing in the chain clears the liquidity filters - which
    is a legitimate outcome, not an error. Trading an illiquid option is a
    faster way to lose money than being wrong about direction.
    """
    want_call = direction.upper() == "BUY"
    if cfg.style == "covered_call":
        want_call = True
    elif cfg.style == "cash_secured_put":
        want_call = False
    option_type = "CALL" if want_call else "PUT"

    start, end = feed.expiry_window(cfg.min_dte, cfg.max_dte)
    try:
        chain = feed.option_chain(underlying, start=start, end=end, option_type=option_type)
    except DataError as exc:
        log.warning("option chain for %s unavailable: %s", underlying, exc)
        return None

    if chain is None or chain.empty:
        log.info("no %s contracts for %s between %s and %s", option_type, underlying, start, end)
        return None

    chain = chain.copy()
    if "suspension" in chain.columns:
        chain = chain[~chain["suspension"].astype(bool)]
    chain["dte"] = chain["strike_time"].map(_days_to_expiry)
    chain = chain[(chain["dte"] >= cfg.min_dte) & (chain["dte"] <= cfg.max_dte)]
    if chain.empty:
        return None

    # Snapshot only the strikes nearest the money - the endpoint is metered and
    # far-OTM contracts are never going to win the delta score anyway.
    try:
        spot = feed.last_price(underlying)
    except DataError:
        spot = float(chain["strike_price"].median())
    chain["moneyness"] = (chain["strike_price"] - spot).abs()
    chain = chain.nsmallest(max_candidates, "moneyness")

    try:
        snap = feed.snapshot(chain["code"].tolist())
    except DataError as exc:
        log.warning("option snapshot failed for %s: %s", underlying, exc)
        return None
    if snap is None or snap.empty:
        return None

    quotes = [q for q in (_to_quote(row, underlying) for _, row in snap.iterrows()) if q]
    return _best(quotes, cfg, option_type)


def _to_quote(row: pd.Series, underlying: str) -> OptionQuote | None:
    code = str(row.get("code", ""))
    if not code:
        return None
    if "option_valid" in row and pd.notna(row["option_valid"]) and not bool(row["option_valid"]):
        return None
    return OptionQuote(
        code=code,
        underlying=str(row.get("stock_owner", underlying)),
        option_type=str(row.get("option_type", "")).upper(),
        strike=_num(row, "option_strike_price"),
        expiry=str(row.get("strike_time", ""))[:10],
        dte=int(_num(row, "option_expiry_date_distance", -1)),
        bid=_num(row, "bid_price"),
        ask=_num(row, "ask_price"),
        last=_num(row, "last_price"),
        delta=_num(row, "option_delta"),
        implied_vol=_num(row, "option_implied_volatility"),
        open_interest=int(_num(row, "option_open_interest")),
        contract_size=int(_num(row, "option_contract_size", DEFAULT_CONTRACT_SIZE)) or
        DEFAULT_CONTRACT_SIZE,
    )


def _best(
    quotes: list[OptionQuote], cfg: OptionsConfig, option_type: str
) -> OptionQuote | None:
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    viable: list[OptionQuote] = []
    for q in quotes:
        if option_type and q.option_type and q.option_type != option_type:
            reject("wrong type")
        elif q.mid <= 0:
            reject("no price")
        elif q.open_interest < cfg.min_open_interest:
            reject(f"open interest < {cfg.min_open_interest}")
        elif q.spread_pct > cfg.max_spread_pct:
            reject(f"spread > {cfg.max_spread_pct}%")
        elif q.delta == 0:
            reject("no delta")
        elif cfg.style == "long" and q.cost_per_contract > cfg.max_premium_per_trade:
            reject(f"premium > {cfg.max_premium_per_trade}")
        else:
            viable.append(q)

    if not viable:
        log.info(
            "no contract passed the filters%s",
            f" ({', '.join(f'{v}x {k}' for k, v in rejected.items())})" if rejected else "",
        )
        return None

    # Closest |delta| to target; break ties on the tighter spread.
    viable.sort(key=lambda q: (abs(abs(q.delta) - cfg.target_delta), q.spread_pct))
    best = viable[0]
    log.info("selected %s (from %d viable contracts)", best, len(viable))
    return best


def contracts_for_budget(quote: OptionQuote, cfg: OptionsConfig, cash: float) -> int:
    """How many contracts to buy, respecting both the premium cap and cash."""
    cost = quote.cost_per_contract
    if cost <= 0:
        return 0
    by_premium = int(cfg.max_premium_per_trade // cost)
    by_cash = int(cash // cost)
    return max(0, min(cfg.max_contracts, by_premium, by_cash))
