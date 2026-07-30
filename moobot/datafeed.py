"""Market data access: historical bars, snapshots and option chains.

Wraps ``moomoo.OpenQuoteContext``. All history requests go through an on-disk
CSV cache because moomoo meters historical-kline requests per account.
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

BAR_COLUMNS = ["time_key", "open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Raised when the quote API returns an error or unusable data."""


def probe_opend(host: str, port: int, timeout: float = 3.0) -> str | None:
    """Check that something is listening on the OpenD port.

    The moomoo SDK retries a refused connection forever without raising, so a
    stopped gateway looks like a hang. Returns None when reachable, otherwise a
    ready-to-print explanation.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return (
            f"Nothing is listening on {host}:{port} ({exc.__class__.__name__}: {exc}).\n"
            "The OpenD gateway must be running and logged in before the bot can "
            "do anything.\n"
            "  1. Download OpenD from https://www.moomoo.com/download/OpenAPI\n"
            "  2. Start it and log in with your moomoo ID\n"
            "  3. Confirm the API port matches [connection].port in settings.toml"
        )


def normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a moomoo kline frame into the canonical OHLCV shape."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"kline data is missing column(s): {missing}")
    out = df.loc[:, BAR_COLUMNS].copy()
    out["time_key"] = pd.to_datetime(out["time_key"])
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset="time_key", keep="last")
    return out.sort_values("time_key").reset_index(drop=True)


class DataFeed:
    """Read-only market data. Safe to use with no trading account at all."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        security_firm: str = "FUTUINC",
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.host, self.port, self.security_firm = host, port, security_firm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ctx: Any | None = None

    # ---------------------------------------------------------------- context

    @property
    def ctx(self) -> Any:
        if self._ctx is None:
            problem = probe_opend(self.host, self.port)
            if problem:
                raise DataError(problem)
            try:
                import moomoo
            except ImportError as exc:  # pragma: no cover - env problem
                raise DataError(
                    "The 'moomoo' package is not installed. Run: pip install moomoo-api"
                ) from exc
            try:
                self._ctx = moomoo.OpenQuoteContext(
                    host=self.host, port=self.port, security_firm=self.security_firm
                )
            except Exception as exc:
                raise DataError(
                    f"Could not open a quote context on {self.host}:{self.port}. "
                    f"Is OpenD logged in? Original error: {exc}"
                ) from exc
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def __enter__(self) -> DataFeed:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _check(result: tuple) -> Any:
        """Unwrap a moomoo (ret, data) tuple, raising on failure."""
        import moomoo

        ret, data = result[0], result[1]
        if ret != moomoo.RET_OK:
            raise DataError(str(data))
        return data

    # ------------------------------------------------------------------ state

    def global_state(self) -> dict[str, Any]:
        return dict(self._check(self.ctx.get_global_state()))

    def market_state(self, codes: list[str]) -> pd.DataFrame:
        return self._check(self.ctx.get_market_state(codes))

    def is_market_open(self, code: str) -> bool:
        """True when `code`'s market is in a regular or extended session."""
        try:
            df = self.market_state([code])
        except DataError as exc:
            log.warning("market state lookup failed for %s: %s", code, exc)
            return False
        if df is None or df.empty or "market_state" not in df.columns:
            return False
        state = str(df.iloc[0]["market_state"]).upper()
        return any(tag in state for tag in ("MORNING", "AFTERNOON", "PRE_MARKET",
                                            "AFTER_HOURS", "NIGHT", "OPEN"))

    # ------------------------------------------------------------- price data

    def _cache_path(self, code: str, bar_type: str) -> Path:
        safe = code.replace(".", "_").replace("/", "_")
        return self.cache_dir / f"{safe}__{bar_type}.csv"

    def load_cache(self, code: str, bar_type: str) -> pd.DataFrame | None:
        path = self._cache_path(code, bar_type)
        if not path.is_file():
            return None
        try:
            return normalise_bars(pd.read_csv(path))
        except (DataError, ValueError, pd.errors.ParserError) as exc:
            log.warning("ignoring unreadable cache %s: %s", path, exc)
            return None

    def save_cache(self, code: str, bar_type: str, bars: pd.DataFrame) -> None:
        bars.to_csv(self._cache_path(code, bar_type), index=False)

    def history(
        self,
        code: str,
        bar_type: str = "K_DAY",
        start: str | None = None,
        end: str | None = None,
        max_bars: int = 1000,
        autype: str = "qfq",
        use_cache: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Historical candles, newest last.

        Paginates through moomoo's 1000-bar page limit. When `use_cache` is on
        and a cached file already covers the request, no API call is made -
        historical kline requests are quota-metered.
        """
        if use_cache and not refresh:
            cached = self.load_cache(code, bar_type)
            if cached is not None and self._cache_covers(cached, start, end, max_bars):
                return self._slice(cached, start, end, max_bars)

        import moomoo

        frames: list[pd.DataFrame] = []
        page_key = None
        while True:
            ret, data, page_key = self.ctx.request_history_kline(
                code=code,
                start=start,
                end=end,
                ktype=bar_type,
                autype=autype,
                max_count=1000,
                page_req_key=page_key,
            )
            if ret != moomoo.RET_OK:
                raise DataError(f"history_kline({code}, {bar_type}) failed: {data}")
            if data is not None and len(data):
                frames.append(data)
            if page_key is None:
                break

        if not frames:
            raise DataError(
                f"No historical bars returned for {code} at {bar_type}. "
                "Check the code format ('US.AAPL') and your market-data permissions."
            )

        bars = normalise_bars(pd.concat(frames, ignore_index=True))
        if use_cache:
            merged = bars
            cached = self.load_cache(code, bar_type)
            if cached is not None:
                merged = normalise_bars(pd.concat([cached, bars], ignore_index=True))
            self.save_cache(code, bar_type, merged)
        return self._slice(bars, start, end, max_bars)

    @staticmethod
    def _cache_covers(
        cached: pd.DataFrame, start: str | None, end: str | None, max_bars: int
    ) -> bool:
        if cached.empty:
            return False
        if start and cached["time_key"].min() > pd.Timestamp(start):
            return False
        if end and cached["time_key"].max() < pd.Timestamp(end):
            return False
        if not start and not end and len(cached) < max_bars:
            return False
        return True

    @staticmethod
    def _slice(
        bars: pd.DataFrame, start: str | None, end: str | None, max_bars: int
    ) -> pd.DataFrame:
        out = bars
        if start:
            out = out[out["time_key"] >= pd.Timestamp(start)]
        if end:
            out = out[out["time_key"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
        if max_bars and len(out) > max_bars:
            out = out.iloc[-max_bars:]
        return out.reset_index(drop=True)

    def recent_bars(self, code: str, bar_type: str, count: int) -> pd.DataFrame:
        """The last `count` bars including the forming one - used by the live loop.

        Requires an active subscription; `subscribe` is called for you.
        """
        import moomoo

        self.subscribe([code], [bar_type])
        ret, data = self.ctx.get_cur_kline(code=code, num=min(count, 1000), ktype=bar_type)
        if ret != moomoo.RET_OK:
            raise DataError(f"get_cur_kline({code}, {bar_type}) failed: {data}")
        return normalise_bars(data)

    def subscribe(self, codes: list[str], subtypes: list[str]) -> None:
        import moomoo

        ret, data = self.ctx.subscribe(codes, subtypes, is_first_push=False)
        if ret != moomoo.RET_OK:
            raise DataError(f"subscribe({codes}, {subtypes}) failed: {data}")

    def snapshot(self, codes: list[str]) -> pd.DataFrame:
        """Full market snapshot. For options this is where the greeks live."""
        import moomoo

        if not codes:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []
        # The snapshot endpoint accepts at most 400 codes per call.
        for i in range(0, len(codes), 400):
            ret, data = self.ctx.get_market_snapshot(codes[i : i + 400])
            if ret != moomoo.RET_OK:
                raise DataError(f"get_market_snapshot failed: {data}")
            rows.append(data)
        return pd.concat(rows, ignore_index=True)

    def last_price(self, code: str) -> float:
        snap = self.snapshot([code])
        if snap.empty or "last_price" not in snap.columns:
            raise DataError(f"no snapshot price available for {code}")
        return float(snap.iloc[0]["last_price"])

    # ---------------------------------------------------------------- options

    def option_expirations(self, underlying: str) -> pd.DataFrame:
        return self._check(self.ctx.get_option_expiration_date(code=underlying))

    def option_chain(
        self,
        underlying: str,
        start: str | None = None,
        end: str | None = None,
        option_type: str = "ALL",
    ) -> pd.DataFrame:
        """Contracts for `underlying` expiring between `start` and `end` (YYYY-MM-DD)."""
        return self._check(
            self.ctx.get_option_chain(
                code=underlying, start=start, end=end, option_type=option_type
            )
        )

    def expiry_window(self, min_dte: int, max_dte: int) -> tuple[str, str]:
        today = datetime.now().date()
        return (
            (today + timedelta(days=min_dte)).isoformat(),
            (today + timedelta(days=max_dte)).isoformat(),
        )
