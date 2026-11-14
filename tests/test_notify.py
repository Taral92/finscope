import json
import traceback
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

from finscope.__main__ import _notify_test
from finscope.config import Settings
from finscope.decide import Decision, DecisionEntry, Tier
from finscope.detect import CheckResult, WatchEntry
from finscope.edgar import Filing
from finscope.notify import (
    MAX_MESSAGE_CHARS,
    NotifyError,
    TelegramNotifier,
    format_alert_message,
    split_message,
)
from finscope.report import SectionReport

TOKEN = "123456:SECRET-test-token"
SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    telegram_bot_token=SecretStr(TOKEN),
    telegram_chat_id="42",
)
Handler = Callable[[httpx.Request], httpx.Response]


def filing(accession: str, day: date) -> Filing:
    return Filing(
        cik="0001318605",
        accession_number=accession,
        form="10-K",
        filing_date=day,
        report_date=None,
        primary_document="tsla.htm",
    )


def entry(change_id: str, tier: Tier, reason: str = "fixed reason") -> DecisionEntry:
    """An alert carries agent text, as decide.py makes it; digest and review entries don't."""
    if tier == "alert":
        return DecisionEntry(
            id=change_id,
            tier=tier,
            reason=reason,
            triage_reason=f"TRIAGE-ONLY {change_id}",
            what_changed=f"Change {change_id} added.",
            why_it_matters=f"It matters because of {change_id}.",
        )
    return DecisionEntry(
        id=change_id, tier=tier, reason=reason, triage_reason=f"TRIAGE-ONLY {change_id}"
    )


def check_result(*entries: DecisionEntry) -> CheckResult:
    report = SectionReport(
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        change_count=len(entries),
        high_count=len(entries),
        medium_count=0,
        low_count=0,
        high=(),
        not_explained=(),
        medium=(),
        triage_runs=1,
        triage_succeeded=1,
        triage_errors=(),
        input_tokens=0,
        output_tokens=0,
        cost_usd=Decimal("0"),
        usage_complete=True,
    )
    return CheckResult(
        checked_at=datetime(2026, 11, 11, tzinfo=UTC),
        entry=WatchEntry(ticker="TSLA", form="10-K", item="1A", part="I"),
        old_filing=filing("0001628280-25-000002", date(2025, 1, 30)),
        new_filing=filing("0001628280-26-000003", date(2026, 1, 29)),
        report=report,
        decision=Decision(entries=entries),
    )


class Telegram:
    """Stand-in for the Bot API: records each request and answers with `response`."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response or httpx.Response(200, json={"ok": True, "result": {}})

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    def texts(self) -> list[str]:
        return [json.loads(request.content)["text"] for request in self.requests]


def notifier(handler: Handler, settings: Settings = SETTINGS) -> TelegramNotifier:
    return TelegramNotifier(settings, transport=httpx.MockTransport(handler))


def test_message_has_header_alerts_reviews_and_digest_count() -> None:
    result = check_result(
        entry("+1", "alert"),
        entry("+2", "digest", "agent rated medium"),
        entry("~3", "review", "fact-check failed: quote not found"),
    )

    assert format_alert_message(result) == (
        "FinScope: TSLA 10-K, 1 alert, 1 to review\n"
        "Part I Item 1A. RISK FACTORS\n"
        "Filed 2026-01-29, accession 0001628280-26-000003\n"
        "\n"
        "ALERT [+1]\n"
        "WHAT CHANGED: Change +1 added.\n"
        "WHY IT MATTERS: It matters because of +1.\n"
        "\n"
        "REVIEW [~3]: fact-check failed: quote not found\n"
        "\n"
        "Also: 1 digest"
    )


def test_review_only_message_shows_the_fixed_reason_and_no_other_text() -> None:
    message = format_alert_message(check_result(entry("~3", "review", "agent failed")))

    assert message is not None
    assert message.startswith("FinScope: TSLA 10-K, 1 to review\n")
    assert "REVIEW [~3]: agent failed" in message
    assert "WHAT CHANGED" not in message
    assert "TRIAGE-ONLY" not in message  # triage reasons stay in the saved report
    assert message.endswith("Also: 0 digest")


@pytest.mark.parametrize("entries", [(), (entry("+2", "digest"), entry("+4", "digest"))])
def test_nothing_to_send_without_an_alert_or_review(entries: tuple[DecisionEntry, ...]) -> None:
    assert format_alert_message(check_result(*entries)) is None


def test_a_short_message_is_not_split() -> None:
    assert split_message("one\n\ntwo") == ["one\n\ntwo"]


def test_a_long_message_splits_at_entry_boundaries_and_numbers_the_parts() -> None:
    entries = [f"ALERT [+{n}]\n" + "x" * 1000 for n in range(10)]  # ~10,000 characters

    parts = split_message("\n\n".join(entries))

    assert len(parts) == 3
    assert all(len(part) <= MAX_MESSAGE_CHARS for part in parts)
    assert [part[: len("(1/3) ")] for part in parts] == ["(1/3) ", "(2/3) ", "(3/3) "]
    # Every entry arrives whole, in order.
    rejoined = "\n\n".join(part[len("(1/3) ") :] for part in parts)
    assert rejoined == "\n\n".join(entries)


def test_length_is_counted_as_telegram_counts_it() -> None:
    # An emoji is one Python character but two UTF-16 units: 3,000 of them don't fit in one.
    parts = split_message("\N{ROCKET}" * 1500 + "\n\n" + "\N{ROCKET}" * 1500)

    assert len(parts) == 2


def test_an_entry_longer_than_a_message_is_cut() -> None:
    parts = split_message("y" * 10_000)

    assert len(parts) == 3
    assert all(len(part) <= MAX_MESSAGE_CHARS for part in parts)
    assert "".join(part[len("(1/3) ") :] for part in parts) == "y" * 10_000


def test_send_posts_plain_text_to_the_configured_chat() -> None:
    telegram = Telegram()

    with notifier(telegram.handle) as bot:
        bot.send("hello")

    [request] = telegram.requests
    assert request.method == "POST"
    assert str(request.url) == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert json.loads(request.content) == {"chat_id": "42", "text": "hello"}  # no parse_mode


def test_send_splits_a_long_message_into_several() -> None:
    telegram = Telegram()

    with notifier(telegram.handle) as bot:
        bot.send("a" * 3000 + "\n\n" + "b" * 3000)

    assert telegram.texts() == ["(1/2) " + "a" * 3000, "(2/2) " + "b" * 3000]


def assert_token_hidden(error: NotifyError) -> None:
    shown = "".join(traceback.format_exception(error))
    assert TOKEN not in shown
    assert "SECRET" not in shown
    # Nothing chained either: the httpx error (whose text has the URL) isn't kept.
    assert error.__cause__ is None
    assert error.__context__ is None


def test_a_network_error_never_shows_the_token() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused for {request.url}", request=request)

    with notifier(refuse) as bot, pytest.raises(NotifyError) as raised:
        bot.send("hello")

    assert str(raised.value) == "Telegram sendMessage failed: ConnectError"
    assert_token_hidden(raised.value)


def test_a_telegram_error_says_why_without_the_token() -> None:
    answer = {"ok": False, "error_code": 400, "description": f"Bad Request: bot{TOKEN} said no"}
    telegram = Telegram(httpx.Response(400, json=answer))

    with notifier(telegram.handle) as bot, pytest.raises(NotifyError) as raised:
        bot.send("hello")

    assert str(raised.value) == (
        "Telegram sendMessage failed: HTTP 400: Bad Request: bot<token> said no"
    )
    assert_token_hidden(raised.value)


def test_a_response_that_is_not_json_is_a_notify_error() -> None:
    telegram = Telegram(httpx.Response(502, text="Bad Gateway"))

    with notifier(telegram.handle) as bot, pytest.raises(NotifyError) as raised:
        bot.send("hello")

    assert str(raised.value) == "Telegram sendMessage failed: HTTP 502: no description"
    assert_token_hidden(raised.value)


def test_a_failed_part_says_which_one() -> None:
    telegram = Telegram(httpx.Response(429, json={"ok": False, "description": "Too Many"}))

    with notifier(telegram.handle) as bot, pytest.raises(NotifyError, match="message 1 of 2"):
        bot.send("a" * 3000 + "\n\n" + "b" * 3000)


def test_not_configured() -> None:
    no_token = SETTINGS.model_copy(update={"telegram_bot_token": None})
    no_chat = SETTINGS.model_copy(update={"telegram_chat_id": None})
    telegram = Telegram()

    with pytest.raises(NotifyError, match="FINSCOPE_TELEGRAM_BOT_TOKEN"):
        TelegramNotifier(no_token)
    with notifier(telegram.handle, no_chat) as bot, pytest.raises(NotifyError, match="CHAT_ID"):
        bot.send("hello")
    assert telegram.requests == []


def test_notify_test_without_configuration_sends_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _notify_test(SETTINGS.model_copy(update={"telegram_chat_id": None})) == 1

    assert "Telegram not configured" in capsys.readouterr().err


def test_chats_lists_each_chat_that_messaged_the_bot_once() -> None:
    me = {"id": 42, "type": "private", "first_name": "Jane", "username": "jane"}
    group = {"id": -100, "type": "group", "title": "Investors"}
    updates = [
        {"update_id": 1, "message": {"chat": me, "text": "hi"}},
        {"update_id": 2, "message": {"chat": me, "text": "again"}},
        {"update_id": 3, "my_chat_member": {"chat": group}},
    ]
    telegram = Telegram(httpx.Response(200, json={"ok": True, "result": updates}))

    with notifier(telegram.handle) as bot:
        chats = bot.chats()

    assert chats == [("42", "Jane @jane (private)"), ("-100", "Investors (group)")]
    assert str(telegram.requests[0].url).endswith("/getUpdates")
