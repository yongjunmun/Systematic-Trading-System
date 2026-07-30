"""Session gate, scanner and notifier tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

from moobot.notify import Notifier
from moobot.scanner import Candidate, ScanReport, read_watchlist, write_watchlist
from moobot.session import Phase, SessionGate
from moobot.settings import ConfigError, NotificationsConfig, ScannerConfig, SessionConfig

NY = ZoneInfo("America/New_York")


def at(year=2026, month=7, day=28, hour=12, minute=0) -> datetime:
    """A Tuesday by default."""
    return datetime(year, month, day, hour, minute, tzinfo=NY)


def gate(**overrides) -> SessionGate:
    defaults = dict(
        enabled=True, timezone="America/New_York", earliest_entry="10:05",
        latest_entry="15:30", force_close="15:51", max_trades_per_day=5,
    )
    defaults.update(overrides)
    return SessionGate(SessionConfig(**defaults))


class TestSessionGate(unittest.TestCase):
    def test_disabled_always_allows_trading(self):
        status = SessionGate(SessionConfig(enabled=False)).status(at(hour=3))
        self.assertIs(status.phase, Phase.OPEN)
        self.assertTrue(status.phase.allows_entry)

    def test_weekend_is_closed(self):
        # 2026-08-01 is a Saturday.
        self.assertIs(gate().status(at(month=8, day=1, hour=12)).phase, Phase.CLOSED)
        self.assertIs(gate().status(at(month=8, day=2, hour=12)).phase, Phase.CLOSED)

    def test_before_the_entry_window_is_warmup(self):
        status = gate().status(at(hour=9, minute=45))
        self.assertIs(status.phase, Phase.WARMUP)
        self.assertFalse(status.phase.allows_entry)
        self.assertTrue(status.phase.allows_management)

    def test_inside_the_window_is_open(self):
        self.assertIs(gate().status(at(hour=11)).phase, Phase.OPEN)
        self.assertIs(gate().status(at(hour=10, minute=5)).phase, Phase.OPEN)
        self.assertIs(gate().status(at(hour=15, minute=30)).phase, Phase.OPEN)

    def test_after_latest_entry_is_manage_only(self):
        status = gate().status(at(hour=15, minute=40))
        self.assertIs(status.phase, Phase.MANAGE_ONLY)
        self.assertFalse(status.phase.allows_entry)
        self.assertTrue(status.phase.allows_management)

    def test_force_close_window(self):
        status = gate().status(at(hour=15, minute=55))
        self.assertIs(status.phase, Phase.FORCE_CLOSE)
        self.assertFalse(status.phase.allows_entry)

    def test_late_evening_is_still_force_close_not_a_silent_hold(self):
        # Never leave a position unmanaged just because the clock rolled on.
        self.assertIs(gate().status(at(hour=20)).phase, Phase.FORCE_CLOSE)

    def test_gating_uses_the_exchange_timezone_not_the_machine(self):
        hk = datetime(2026, 7, 28, 23, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        # 23:00 in Hong Kong is 11:00 the same morning in New York.
        self.assertIs(gate().status(hk).phase, Phase.OPEN)

    def test_status_renders_readably(self):
        text = str(gate().status(at(hour=11)))
        self.assertIn("open", text)
        self.assertIn("America/New_York", text)


class TestSessionValidation(unittest.TestCase):
    def test_bad_timezone_is_rejected(self):
        from moobot.settings import _validate_session

        with self.assertRaises(ConfigError) as ctx:
            _validate_session(SessionConfig(enabled=True, timezone="Mars/Olympus"))
        self.assertIn("IANA", str(ctx.exception))

    def test_inverted_window_is_rejected(self):
        from moobot.settings import _validate_session

        with self.assertRaises(ConfigError):
            _validate_session(
                SessionConfig(enabled=True, earliest_entry="15:00", latest_entry="10:00")
            )

    def test_force_close_before_latest_entry_is_rejected(self):
        from moobot.settings import _validate_session

        with self.assertRaises(ConfigError) as ctx:
            _validate_session(
                SessionConfig(enabled=True, latest_entry="15:30", force_close="12:00")
            )
        self.assertIn("force_close", str(ctx.exception))


class TestScannerScoring(unittest.TestCase):
    def test_volume_confirmation_raises_the_score(self):
        quiet = Candidate("US.A", 10, 5.0, 1.0, 2.0, 9.5)
        busy = Candidate("US.B", 10, 5.0, 4.0, 2.0, 9.5)
        self.assertGreater(busy.score, quiet.score)

    def test_a_bigger_gap_raises_the_score(self):
        small = Candidate("US.A", 10, 3.0, 2.0, 2.0, 9.7)
        large = Candidate("US.B", 10, 9.0, 2.0, 2.0, 9.2)
        self.assertGreater(large.score, small.score)

    def test_a_down_gap_scores_on_magnitude(self):
        down = Candidate("US.A", 10, -8.0, 3.0, 2.0, 10.9)
        self.assertGreater(down.score, 0)

    def test_extreme_volume_is_capped(self):
        sane = Candidate("US.A", 10, 4.0, 5.0, 2.0, 9.6)
        absurd = Candidate("US.B", 10, 4.0, 500.0, 2.0, 9.6)
        self.assertEqual(sane.score, absurd.score)


class TestWatchlist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "watchlist.txt"
        self.cfg = ScannerConfig(enabled=True)

    def tearDown(self):
        self.tmp.cleanup()

    def report(self) -> ScanReport:
        return ScanReport(
            scanned=50,
            survivors=[
                Candidate("US.AAPL", 190.0, 4.2, 3.1, 2.0, 182.0),
                Candidate("US.NVDA", 900.0, -6.0, 2.4, 3.0, 957.0),
            ],
            rejected={"gap under 3%": 40},
            failed=["US.BAD"],
            elapsed_seconds=1.2,
        )

    def test_round_trip(self):
        write_watchlist(self.report(), self.path, self.cfg)
        self.assertEqual(read_watchlist(self.path), ["US.AAPL", "US.NVDA"])

    def test_comments_and_annotations_are_stripped(self):
        write_watchlist(self.report(), self.path, self.cfg)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("# moobot watchlist", text)
        self.assertIn("gap +4.20%", text)
        self.assertNotIn("#", "".join(read_watchlist(self.path)))

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(read_watchlist(self.path.parent / "nope.txt"), [])

    def test_blank_and_comment_only_lines_are_ignored(self):
        self.path.write_text("# header\n\n   \nUS.MSFT  # note\n", encoding="utf-8")
        self.assertEqual(read_watchlist(self.path), ["US.MSFT"])

    def test_summary_mentions_rejections_and_failures(self):
        summary = self.report().summary()
        self.assertIn("2 survivors", summary)
        self.assertIn("gap under 3%", summary)
        self.assertIn("US.BAD", summary)


class TestScannerMeasurement(unittest.TestCase):
    def test_relative_volume_excludes_the_current_bar(self):
        from moobot.scanner import _measure

        bars = pd.DataFrame({
            "time_key": pd.date_range("2024-01-02", periods=16),
            "open": [100.0] * 16, "high": [101.0] * 16, "low": [99.0] * 16,
            "close": [100.0] * 15 + [110.0],
            "volume": [1_000_000.0] * 15 + [5_000_000.0],
        })
        candidate = _measure("US.T", bars, lookback=14)
        # Baseline is the flat 1M bars, so today is exactly 5x.
        self.assertAlmostEqual(candidate.rvol, 5.0, places=6)
        self.assertAlmostEqual(candidate.gap_pct, 10.0, places=6)

    def test_zero_previous_close_is_rejected(self):
        from moobot.scanner import _measure

        bars = pd.DataFrame({
            "time_key": pd.date_range("2024-01-02", periods=16),
            "open": [1.0] * 16, "high": [1.0] * 16, "low": [1.0] * 16,
            "close": [0.0] * 16, "volume": [1.0] * 16,
        })
        self.assertIsNone(_measure("US.T", bars, lookback=14))


class TestNotifier(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_unconfigured_is_inactive(self):
        notifier = Notifier(NotificationsConfig(enabled=True))
        self.assertFalse(notifier.configured)
        self.assertFalse(notifier.active)
        self.assertIn("no credentials", notifier.describe())

    @mock.patch.dict(os.environ, {"MOOBOT_TELEGRAM_TOKEN": "t",
                                  "MOOBOT_TELEGRAM_CHAT_ID": "c"}, clear=True)
    def test_disabled_in_settings_beats_present_credentials(self):
        notifier = Notifier(NotificationsConfig(enabled=False))
        self.assertTrue(notifier.configured)
        self.assertFalse(notifier.active)
        self.assertEqual(notifier.describe(), "disabled in settings")

    @mock.patch.dict(os.environ, {"MOOBOT_TELEGRAM_TOKEN": "t",
                                  "MOOBOT_TELEGRAM_CHAT_ID": "c"}, clear=True)
    def test_send_is_attempted_when_active(self):
        notifier = Notifier(NotificationsConfig(enabled=True))
        with mock.patch.object(notifier, "_post", return_value=True) as post:
            self.assertTrue(notifier.send("t", "b"))
        post.assert_called_once()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_sending_while_inactive_is_a_no_op(self):
        notifier = Notifier(NotificationsConfig(enabled=True))
        with mock.patch.object(notifier, "_post") as post:
            self.assertFalse(notifier.send("t", "b"))
        post.assert_not_called()

    @mock.patch.dict(os.environ, {"MOOBOT_WEBHOOK_URL": "http://insecure.example"}, clear=True)
    def test_non_https_webhook_is_refused(self):
        notifier = Notifier(NotificationsConfig(enabled=True))
        self.assertFalse(notifier.send("t", "b"))

    @mock.patch.dict(os.environ, {"MOOBOT_WEBHOOK_URL": "https://example.invalid/hook"},
                     clear=True)
    def test_a_network_failure_never_raises(self):
        """An alerting failure must not be able to kill the trading loop."""
        notifier = Notifier(NotificationsConfig(enabled=True, timeout_seconds=0.01))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            self.assertFalse(notifier.send("t", "b"))

    @mock.patch.dict(os.environ, {"MOOBOT_WEBHOOK_URL": "https://example.invalid/hook"},
                     clear=True)
    def test_event_hooks_respect_their_switches(self):
        notifier = Notifier(NotificationsConfig(enabled=True, on_entry=False, on_exit=True))
        with mock.patch.object(notifier, "send") as send:
            notifier.entry("US.T", 10, 5.0, 4.0, "because")
            send.assert_not_called()
            notifier.exit("US.T", 10, 6.0, "target", 10.0)
            send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
