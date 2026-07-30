"""Push notifications - Telegram and generic webhooks.

Fire and forget: a notifier that can throw is a notifier that can kill a trading
loop over a network blip, so every failure here is swallowed and logged.

Credentials come only from the environment, never from the settings file:

    MOOBOT_TELEGRAM_TOKEN     bot token from @BotFather
    MOOBOT_TELEGRAM_CHAT_ID   your chat id from @userinfobot
    MOOBOT_WEBHOOK_URL        any URL that accepts a POST body (e.g. ntfy.sh)

Uses urllib from the standard library, so it adds no dependency.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from .settings import NotificationsConfig

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Sends short alerts. Silently does nothing when unconfigured."""

    def __init__(self, cfg: NotificationsConfig) -> None:
        self.cfg = cfg
        self.token = os.environ.get("MOOBOT_TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.environ.get("MOOBOT_TELEGRAM_CHAT_ID", "").strip()
        self.webhook = os.environ.get("MOOBOT_WEBHOOK_URL", "").strip()

    @property
    def configured(self) -> bool:
        return bool((self.token and self.chat_id) or self.webhook)

    @property
    def active(self) -> bool:
        return self.cfg.enabled and self.configured

    def describe(self) -> str:
        if not self.cfg.enabled:
            return "disabled in settings"
        channels = []
        if self.token and self.chat_id:
            channels.append("telegram")
        if self.webhook:
            channels.append("webhook")
        if not channels:
            return (
                "enabled but no credentials found - set MOOBOT_TELEGRAM_TOKEN + "
                "MOOBOT_TELEGRAM_CHAT_ID, or MOOBOT_WEBHOOK_URL"
            )
        return ", ".join(channels)

    # -------------------------------------------------------------- event hooks

    def entry(self, code: str, qty: float, price: float, stop: float, reason: str) -> None:
        if self.cfg.on_entry:
            risk = (price - stop) * qty if stop else 0.0
            self.send(
                f"BUY {code}",
                f"{qty:g} @ {price:.4f}\nstop {stop:.4f} (risking {risk:,.2f})\n{reason}",
            )

    def exit(self, code: str, qty: float, price: float, reason: str, pnl: float) -> None:
        if self.cfg.on_exit:
            self.send(
                f"{'WIN' if pnl >= 0 else 'LOSS'} {code}",
                f"closed {qty:g} @ {price:.4f}\nP&L {pnl:+,.2f}\n{reason}",
            )

    def halt(self, reason: str) -> None:
        if self.cfg.on_halt:
            self.send("TRADING HALTED", reason)

    def error(self, message: str) -> None:
        if self.cfg.on_error:
            self.send("BOT ERROR", message[:500])

    def summary(self, title: str, body: str) -> None:
        self.send(title, body)

    # ------------------------------------------------------------------ transport

    def send(self, title: str, body: str) -> bool:
        """Deliver to every configured channel. Never raises."""
        if not self.active:
            return False
        delivered = False
        if self.token and self.chat_id:
            delivered |= self._telegram(title, body)
        if self.webhook:
            delivered |= self._webhook(title, body)
        return delivered

    def _telegram(self, title: str, body: str) -> bool:
        payload = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": f"{title}\n{body}", "disable_notification": "false"}
        ).encode()
        return self._post(TELEGRAM_API.format(token=self.token), payload, {}, "telegram")

    def _webhook(self, title: str, body: str) -> bool:
        payload = json.dumps({"title": title, "message": body}).encode()
        headers = {"Content-Type": "application/json", "Title": title}
        return self._post(self.webhook, payload, headers, "webhook")

    def _post(self, url: str, payload: bytes, headers: dict[str, str], label: str) -> bool:
        # Only ever talk HTTPS to an absolute URL, so a mangled env var cannot
        # turn into a file:// or local-scheme read.
        if not url.lower().startswith("https://"):
            log.warning("%s notification URL must be https, refusing to send", label)
            return False
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme checked above
                request, timeout=self.cfg.timeout_seconds
            ) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Never let an alert failure propagate into the trading loop.
            log.warning("%s notification failed: %s", label, exc)
            return False
