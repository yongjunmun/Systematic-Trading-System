"""Universe scanner.

Trading a hardcoded list means you only ever catch a move if it happens to
occur in one of your four names. This ranks a wider universe by the conditions
that actually precede a tradable move - a gap, unusual volume, and enough price
and range to be worth the commission - and writes the survivors to a watchlist
the engine can pick up.

The scan is deliberately cheap: one daily-bar request per symbol, cached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .datafeed import DataError, DataFeed
from .settings import ScannerConfig

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    code: str
    price: float
    gap_pct: float
    rvol: float
    atr_pct: float
    prev_close: float

    @property
    def score(self) -> float:
        """Rank by gap size scaled by how unusual the volume is.

        Volume is the confirmation: a 5% gap on average volume is usually noise,
        the same gap on 4x volume is somebody with information.
        """
        return abs(self.gap_pct) * min(self.rvol, 5.0)

    def __str__(self) -> str:
        return (
            f"{self.code:<12} gap {self.gap_pct:+6.2f}%  rvol {self.rvol:4.2f}x  "
            f"atr {self.atr_pct:5.2f}%  ${self.price:.2f}"
        )


@dataclass
class ScanReport:
    scanned: int
    survivors: list[Candidate]
    rejected: dict[str, int]
    failed: list[str]
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"Scanned {self.scanned} symbols in {self.elapsed_seconds:.1f}s "
            f"-> {len(self.survivors)} survivors"
        ]
        if self.rejected:
            detail = ", ".join(f"{count} {why}" for why, count in sorted(self.rejected.items()))
            lines.append(f"Rejected: {detail}")
        if self.failed:
            shown = ", ".join(self.failed[:8])
            more = f" (+{len(self.failed) - 8} more)" if len(self.failed) > 8 else ""
            lines.append(f"No data for {len(self.failed)}: {shown}{more}")
        return "\n".join(lines)


def scan(
    feed: DataFeed,
    codes: list[str],
    cfg: ScannerConfig,
    refresh: bool = False,
) -> ScanReport:
    """Rank `codes` by gap and relative volume. One symbol failing never aborts."""
    started = datetime.now(timezone.utc)
    survivors: list[Candidate] = []
    rejected: dict[str, int] = {}
    failed: list[str] = []

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    need = cfg.rvol_lookback + 2
    for code in codes:
        try:
            bars = feed.history(
                code, bar_type="K_DAY", max_bars=max(need, 30), refresh=refresh
            )
        except DataError as exc:
            log.debug("scanner: no data for %s (%s)", code, exc)
            failed.append(code)
            continue

        if len(bars) < need:
            failed.append(code)
            continue

        candidate = _measure(code, bars, cfg.rvol_lookback)
        if candidate is None:
            failed.append(code)
        elif candidate.price < cfg.min_price:
            reject(f"below ${cfg.min_price:g}")
        elif abs(candidate.gap_pct) < cfg.min_gap_pct:
            reject(f"gap under {cfg.min_gap_pct:g}%")
        elif candidate.rvol < cfg.min_rvol:
            reject(f"rvol under {cfg.min_rvol:g}x")
        else:
            survivors.append(candidate)

    survivors.sort(key=lambda c: c.score, reverse=True)
    return ScanReport(
        scanned=len(codes),
        survivors=survivors[: cfg.max_symbols],
        rejected=rejected,
        failed=failed,
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )


def _measure(code: str, bars: pd.DataFrame, lookback: int) -> Candidate | None:
    latest = bars.iloc[-1]
    prev_close = float(bars.iloc[-2]["close"])
    price = float(latest["close"])
    if prev_close <= 0 or price <= 0:
        return None

    # Exclude the current bar from its own baseline, or an outlier hides itself.
    baseline = bars["volume"].iloc[-(lookback + 1) : -1]
    avg_volume = float(baseline.mean())
    rvol = float(latest["volume"]) / avg_volume if avg_volume > 0 else 0.0

    ranges = (bars["high"] - bars["low"]).iloc[-(lookback + 1) : -1]
    atr_pct = float(ranges.mean()) / price * 100.0 if price > 0 else 0.0

    return Candidate(
        code=code,
        price=price,
        gap_pct=(price - prev_close) / prev_close * 100.0,
        rvol=rvol,
        atr_pct=atr_pct,
        prev_close=prev_close,
    )


def write_watchlist(report: ScanReport, path: str | Path, cfg: ScannerConfig) -> Path:
    """Write survivors to a commented text file the engine can read back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# moobot watchlist generated {stamp}",
        f"# filters: price >= {cfg.min_price:g}, |gap| >= {cfg.min_gap_pct:g}%, "
        f"rvol >= {cfg.min_rvol:g}x",
        f"# {len(report.survivors)} survivors of {report.scanned} scanned",
        "#",
    ]
    for c in report.survivors:
        lines.append(
            f"{c.code}  # gap {c.gap_pct:+.2f}%  rvol {c.rvol:.2f}x  "
            f"atr {c.atr_pct:.2f}%  ${c.price:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_watchlist(path: str | Path) -> list[str]:
    """Read symbols back, ignoring comments and inline annotations."""
    path = Path(path)
    if not path.is_file():
        return []
    codes: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            codes.append(line)
    return codes
