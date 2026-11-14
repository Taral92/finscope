"""Send a check's decision to Telegram. Plain code, no LLM.

The message follows decide.py's rule: only an alert carries the agent's text. A review entry
shows its fixed reason, and digest entries are only counted.

The bot token is part of every Telegram URL, and httpx puts the URL in its error messages.
So no httpx error is ever passed on: every failure becomes a NotifyError whose message we
write ourselves. (httpx also logs request URLs at INFO level; FinScope configures no logging,
so nothing is logged. Keep the "httpx" logger above INFO if that ever changes.)
"""

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import httpx

from finscope.config import Settings
from finscope.detect import CheckResult

TELEGRAM_API = "https://api.telegram.org"
# Telegram's limit for one message, counted in UTF-16 code units.
MAX_MESSAGE_CHARS = 4096
# Room for the "(12/12) " part number put in front of each part of a split message.
_PART_PREFIX_RESERVE = 16


class NotifyError(Exception):
    """Sending failed. The message is ours and never contains the bot token."""


def telegram_configured(settings: Settings) -> bool:
    return settings.telegram_bot_token is not None and settings.telegram_chat_id is not None


def format_alert_message(result: CheckResult) -> str | None:
    """The plain-text message for one check result, or None when there is nothing to send."""
    decision = result.decision
    alerts, reviews = decision.tier("alert"), decision.tier("review")
    if not alerts and not reviews:
        return None

    entry, filing = result.entry, result.new_filing
    counts = [f"{len(alerts)} alert{'' if len(alerts) == 1 else 's'}"] if alerts else []
    counts += [f"{len(reviews)} to review"] if reviews else []
    blocks = [
        f"FinScope: {entry.ticker} {entry.form}, {', '.join(counts)}\n"
        f"{result.report.section_title}\n"
        f"Filed {filing.filing_date.isoformat()}, accession {filing.accession_number}"
    ]
    blocks += [
        f"ALERT [{alert.id}]\n"
        f"WHAT CHANGED: {alert.what_changed}\n"
        f"WHY IT MATTERS: {alert.why_it_matters}"
        for alert in alerts
    ]
    blocks += [f"REVIEW [{review.id}]: {review.reason}" for review in reviews]
    blocks.append(f"Also: {len(decision.tier('digest'))} digest")
    return "\n\n".join(blocks)


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split at blank lines (entry boundaries) into parts Telegram accepts, numbered "(1/2)".

    A single entry longer than a whole message is cut, the only case that splits an entry.
    """
    if _length(text) <= limit:
        return [text]
    budget = limit - _PART_PREFIX_RESERVE
    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        for piece in _cut(block, budget):
            joined = f"{current}\n\n{piece}" if current else piece
            if current and _length(joined) > budget:
                parts.append(current)
                current = piece
            else:
                current = joined
    parts.append(current)
    return [f"({number}/{len(parts)}) {part}" for number, part in enumerate(parts, 1)]


def _length(text: str) -> int:
    """Length as Telegram counts it: UTF-16 code units, so an emoji counts as 2."""
    return len(text.encode("utf-16-le")) // 2


def _cut(block: str, budget: int) -> list[str]:
    """`block` in pieces of at most `budget` UTF-16 units (just [block] when it fits)."""
    pieces = []
    start = used = 0
    for index, char in enumerate(block):
        size = 2 if ord(char) > 0xFFFF else 1
        if used + size > budget:
            pieces.append(block[start:index])
            start, used = index, 0
        used += size
    pieces.append(block[start:])
    return pieces


class TelegramNotifier:
    """A Telegram bot that sends plain-text messages. Needs FINSCOPE_TELEGRAM_BOT_TOKEN."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        if settings.telegram_bot_token is None:
            raise NotifyError("Telegram not configured: set FINSCOPE_TELEGRAM_BOT_TOKEN")
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._client = httpx.Client(timeout=30.0, transport=transport)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def send(self, text: str) -> None:
        """Send `text` to the configured chat, split into several messages if it's too long."""
        if self._chat_id is None:
            raise NotifyError("Telegram chat not configured: set FINSCOPE_TELEGRAM_CHAT_ID")
        parts = split_message(text)
        for number, part in enumerate(parts, 1):
            where = f" (message {number} of {len(parts)})" if len(parts) > 1 else ""
            # Plain text, no parse_mode: nothing in a filing can break Markdown or HTML parsing.
            self._call("sendMessage", {"chat_id": self._chat_id, "text": part}, where)

    def chats(self) -> list[tuple[str, str]]:
        """(chat ID, name) of every chat in the bot's recent updates, each chat once."""
        updates = self._call("getUpdates", {}, "")
        found: dict[str, str] = {}
        for update in updates if isinstance(updates, list) else []:
            # The chat sits in whichever kind of update it is: message, channel_post, ...
            for value in update.values() if isinstance(update, dict) else []:
                chat = value.get("chat") if isinstance(value, dict) else None
                if isinstance(chat, dict) and "id" in chat:
                    found.setdefault(str(chat["id"]), _chat_name(chat))
        return list(found.items())

    def _call(self, method: str, payload: Mapping[str, Any], where: str) -> Any:
        """POST one Bot API method and return its "result". Raises NotifyError on any failure.

        The NotifyError is raised outside the `except` block on purpose, so it has no
        __context__ either: a traceback can't show the httpx error and its URL.
        """
        token = self._token.get_secret_value()
        error_name = None
        try:
            response = self._client.post(f"{TELEGRAM_API}/bot{token}/{method}", json=payload)
        except Exception as error:  # any httpx error; its text may hold the URL
            error_name = type(error).__name__
        if error_name is not None:
            raise NotifyError(f"Telegram {method} failed{where}: {error_name}")

        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("ok") is not True:
            description = body.get("description") if isinstance(body, dict) else None
            # Telegram's own words, minus the token in case it ever echoes it back.
            safe = str(description or "no description").replace(token, "<token>")
            raise NotifyError(
                f"Telegram {method} failed{where}: HTTP {response.status_code}: {safe}"
            )
        return body.get("result")


def _chat_name(chat: Mapping[str, Any]) -> str:
    if title := chat.get("title"):
        return f"{title} ({chat.get('type', 'chat')})"
    name = " ".join(str(chat[key]) for key in ("first_name", "last_name") if chat.get(key))
    if username := chat.get("username"):
        name = f"{name} @{username}".strip()
    return f"{name or 'no name'} ({chat.get('type', 'chat')})"
