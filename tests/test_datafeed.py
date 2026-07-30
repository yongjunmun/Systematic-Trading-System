"""Data feed tests: bar normalisation, cache round-trips and the OpenD probe."""

from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

from moobot.datafeed import BAR_COLUMNS, DataError, DataFeed, normalise_bars, probe_opend


class TestNormaliseBars(unittest.TestCase):
    def frame(self, **overrides) -> pd.DataFrame:
        data = {
            "time_key": ["2024-01-03 09:30:00", "2024-01-02 09:30:00"],
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
            "volume": [1000, 2000],
            "turnover": [1, 2],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_keeps_only_the_canonical_columns_in_order(self):
        out = normalise_bars(self.frame())
        self.assertEqual(list(out.columns), BAR_COLUMNS)

    def test_sorts_oldest_first(self):
        out = normalise_bars(self.frame())
        self.assertTrue(out["time_key"].is_monotonic_increasing)
        self.assertEqual(out.iloc[0]["close"], 100.5)

    def test_parses_timestamps(self):
        out = normalise_bars(self.frame())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["time_key"]))

    def test_duplicate_timestamps_keep_the_last_value(self):
        raw = pd.DataFrame({
            "time_key": ["2024-01-02", "2024-01-02"],
            "open": [1.0, 1.0], "high": [2.0, 2.0], "low": [0.5, 0.5],
            "close": [1.5, 9.9], "volume": [10, 20],
        })
        out = normalise_bars(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["close"], 9.9)

    def test_rows_with_unparseable_prices_are_dropped(self):
        raw = self.frame(close=["oops", 100.5])
        out = normalise_bars(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["close"], 100.5)

    def test_missing_column_is_reported_by_name(self):
        raw = self.frame().drop(columns=["volume"])
        with self.assertRaises(DataError) as ctx:
            normalise_bars(raw)
        self.assertIn("volume", str(ctx.exception))

    def test_string_numbers_are_coerced(self):
        out = normalise_bars(self.frame(open=["101.0", "100.0"]))
        self.assertEqual(out["open"].dtype.kind, "f")


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.feed = DataFeed(cache_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        bars = normalise_bars(pd.DataFrame({
            "time_key": pd.date_range("2024-01-02", periods=5),
            "open": [1.0] * 5, "high": [2.0] * 5, "low": [0.5] * 5,
            "close": [1.5] * 5, "volume": [100] * 5,
        }))
        self.feed.save_cache("US.TEST", "K_DAY", bars)
        loaded = self.feed.load_cache("US.TEST", "K_DAY")
        self.assertIsNotNone(loaded)
        pd.testing.assert_frame_equal(bars, loaded)

    def test_missing_cache_returns_none(self):
        self.assertIsNone(self.feed.load_cache("US.NOPE", "K_DAY"))

    def test_corrupt_cache_is_ignored_rather_than_crashing(self):
        path = Path(self.tmp.name) / "US_BAD__K_DAY.csv"
        path.write_text("this is not a csv of bars\n", encoding="utf-8")
        self.assertIsNone(self.feed.load_cache("US.BAD", "K_DAY"))

    def test_cache_path_is_filesystem_safe(self):
        self.assertNotIn(".", self.feed._cache_path("US.AAPL", "K_DAY").stem)


class TestExpiryWindow(unittest.TestCase):
    def test_window_is_ordered_and_iso_formatted(self):
        start, end = DataFeed().expiry_window(14, 45)
        self.assertLess(start, end)
        self.assertRegex(start, r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(end, r"^\d{4}-\d{2}-\d{2}$")


class TestOpendProbe(unittest.TestCase):
    def test_a_closed_port_produces_actionable_guidance(self):
        # Port 1 is reserved and never listening.
        problem = probe_opend("127.0.0.1", 1, timeout=1.0)
        self.assertIsNotNone(problem)
        self.assertIn("OpenD", problem)
        self.assertIn("settings.toml", problem)

    def test_an_open_port_probes_clean(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_once() -> None:
            try:
                conn, _ = server.accept()
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=accept_once, daemon=True)
        thread.start()
        try:
            self.assertIsNone(probe_opend("127.0.0.1", port, timeout=2.0))
        finally:
            thread.join(timeout=2.0)
            server.close()


if __name__ == "__main__":
    unittest.main()
