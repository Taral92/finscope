import json
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

import finscope.__main__ as cli
from finscope.__main__ import _run_check
from finscope.config import Settings
from finscope.detect import (
    CheckError,
    SeenFiling,
    WatchEntry,
    WatchlistError,
    load_state,
    load_watchlist,
    plan_entry,
    reports_dir,
    save_state,
    state_path,
)
from finscope.edgar import Filing, SecClient
from finscope.llm import (
    AgentStep,
    ConversationState,
    LLMError,
    LLMProvider,
    LLMResult,
    ToolResult,
    ToolSpec,
)
from finscope.notify import TelegramNotifier

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    sec_max_requests_per_second=10,
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)

TSLA = WatchEntry(ticker="TSLA", form="10-K", item="1A", part="I")
AAPL = WatchEntry(ticker="AAPL", form="10-K", item="1A", part="I")
CIKS = {"TSLA": "0001318605", "AAPL": "0000320193"}

BASE = ["Our costs may rise because of supply chain problems.", "We depend on our CEO."]
NEW_RISK = "We are launching a robotaxi service in new cities."


def filing_html(*paragraphs: str) -> str:
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        f"<html><body><p>PART I</p><p>Item 1A. Risk Factors</p>{body}"
        "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p></body></html>"
    )


class FakeEdgar:
    """Stand-in for SEC's servers. Each ticker gets 10-K filings, newest first: (accession,
    filing date, HTML). A 10-K/A is also listed as the newest filing, to show amendments are
    never compared."""

    def __init__(self) -> None:
        self.filings: dict[str, list[tuple[str, date, str]]] = {}

    def set_filings(self, ticker: str, *filings: tuple[str, date, str]) -> None:
        self.filings[ticker] = list(filings)

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://www.sec.gov/files/company_tickers.json":
            rows = {
                str(n): {"cik_str": int(cik), "ticker": ticker, "title": ticker}
                for n, (ticker, cik) in enumerate(CIKS.items())
            }
            return httpx.Response(200, json=rows)
        for ticker, cik in CIKS.items():
            if url == f"https://data.sec.gov/submissions/CIK{cik}.json":
                return httpx.Response(200, json=self._submissions(ticker))
            if url == f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json":
                return httpx.Response(200, json={"facts": {}})
        for filings in self.filings.values():
            for accession, _, html in filings:
                if url.endswith(f"/{accession}.htm"):
                    return httpx.Response(200, text=html)
        return httpx.Response(404)

    def _submissions(self, ticker: str) -> dict[str, Any]:
        filings = self.filings.get(ticker, [])
        amendment = ("0009999999-26-999999", date(2026, 6, 1), "10-K/A")
        rows = [amendment] + [(accession, day, "10-K") for accession, day, _ in filings]
        return {
            "filings": {
                "recent": {
                    "accessionNumber": [row[0] for row in rows],
                    "form": [row[2] for row in rows],
                    "filingDate": [row[1].isoformat() for row in rows],
                    "reportDate": ["" for _ in rows],
                    "primaryDocument": [f"{row[0]}.htm" for row in rows],
                }
            }
        }


NEW = ("0001628280-26-000003", date(2026, 1, 29), filing_html(*BASE, NEW_RISK))
OLD = ("0001628280-25-000002", date(2025, 1, 30), filing_html(*BASE))
OLDER = ("0001628280-24-000001", date(2024, 1, 29), filing_html(*BASE))


class FakeTriage:
    """complete_json: rates the single added change (+1) high, or raises the next error."""

    def __init__(self, errors: list[LLMError] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls = 0

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        items = [{"id": "+1", "importance": "high", "reason": "new business line"}]
        return LLMResult(
            text=json.dumps({"items": items}), model="fake", input_tokens=1000, output_tokens=200
        )


class FakeAgent:
    """start: answers right away, rating the change high and quoting it."""

    def __init__(self) -> None:
        self.calls = 0

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        self.calls += 1
        text = (
            "WHAT CHANGED: Robotaxi expansion risk added.\n"
            "WHY IT MATTERS: A new business line brings new risks.\n"
            "IMPORTANCE: high - new business line\n"
            'EVIDENCE: "robotaxi service in new cities"'
        )
        return AgentStep(text=text, tool_calls=(), input_tokens=300, output_tokens=50), (
            ConversationState()
        )

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        raise AssertionError("the fake agent answers in one step")


class Providers:
    """The make_providers callable: counts how often the real thing would have been built."""

    def __init__(self, triage: FakeTriage | None = None) -> None:
        self.triage = triage or FakeTriage()
        self.agent = FakeAgent()
        self.made = 0

    def __call__(self) -> tuple[LLMProvider, LLMProvider]:
        self.made += 1
        return self.triage, self.agent  # type: ignore[return-value]


@pytest.fixture
def edgar() -> FakeEdgar:
    return FakeEdgar()


@pytest.fixture
def sec(edgar: FakeEdgar, tmp_path: Path) -> Iterator[SecClient]:
    settings = SETTINGS.model_copy(update={"data_dir": tmp_path})
    with SecClient(settings=settings, transport=httpx.MockTransport(edgar.handle)) as client:
        yield client


def check(
    sec: SecClient,
    providers: Providers,
    entries: tuple[WatchEntry, ...] = (TSLA,),
    *,
    include_unseen: bool = False,
    dry_run: bool = False,
    triage_runs: int = 1,
    notify: bool = False,
    notifier: TelegramNotifier | None = None,
) -> int:
    return _run_check(
        entries,
        sec=sec,
        settings=SETTINGS,
        make_providers=providers,
        include_unseen=include_unseen,
        dry_run=dry_run,
        max_explain=5,
        triage_runs=triage_runs,
        notify=notify,
        notifier=notifier,
    )


def remember(sec: SecClient, entry: WatchEntry, filing: tuple[str, date, str]) -> None:
    seen = SeenFiling(
        accession_number=filing[0], filing_date=filing[1], processed_at=datetime.now(UTC)
    )
    state = load_state(state_path(sec.data_dir))
    save_state({**state, entry.key: seen}, state_path(sec.data_dir))


def remembered(sec: SecClient, entry: WatchEntry) -> str | None:
    seen = load_state(state_path(sec.data_dir)).get(entry.key)
    return seen.accession_number if seen else None


def saved_reports(sec: SecClient) -> list[Path]:
    return sorted(reports_dir(sec.data_dir).glob("*"))


def test_first_sight_records_the_start_without_ai_calls(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    providers = Providers()

    assert check(sec, providers) == 0

    assert remembered(sec, TSLA) == NEW[0]  # the 10-K/A listed above it is ignored
    assert providers.made == 0
    assert saved_reports(sec) == []
    assert f"now watching, starting from 2026-01-29 ({NEW[0]})" in capsys.readouterr().out


def test_include_unseen_runs_the_report_on_first_sight(sec: SecClient, edgar: FakeEdgar) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    providers = Providers()

    assert check(sec, providers, include_unseen=True) == 0

    assert providers.triage.calls == 1
    assert [path.name for path in saved_reports(sec)] == [f"TSLA_10-K_{NEW[0]}_I-1A.json"]
    assert remembered(sec, TSLA) == NEW[0]


def test_same_accession_does_nothing(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, NEW)
    before = state_path(sec.data_dir).read_text(encoding="utf-8")
    providers = Providers()

    assert check(sec, providers) == 0

    assert providers.made == 0
    assert saved_reports(sec) == []
    assert state_path(sec.data_dir).read_text(encoding="utf-8") == before
    assert "no new filing" in capsys.readouterr().out


def test_new_accession_runs_saves_the_report_then_updates_state(
    sec: SecClient,
    edgar: FakeEdgar,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)
    providers = Providers()
    report_path = reports_dir(sec.data_dir) / f"TSLA_10-K_{NEW[0]}_I-1A.json"
    real_save_state = cli.save_state

    def save_state_after_report(state: dict[str, SeenFiling], path: Path) -> None:
        assert report_path.exists(), "state must only be updated after the report is saved"
        real_save_state(state, path)

    monkeypatch.setattr(cli, "save_state", save_state_after_report)

    assert check(sec, providers) == 0

    assert (providers.triage.calls, providers.agent.calls) == (1, 1)
    assert remembered(sec, TSLA) == NEW[0]
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["old_filing"]["accession_number"] == OLD[0]
    assert saved["new_filing"]["accession_number"] == NEW[0]
    assert saved["report"]["high_count"] == 1
    assert [entry["tier"] for entry in saved["decision"]["entries"]] == ["alert"]
    out = capsys.readouterr().out
    assert "1 change(s): 1 high, 0 medium, 0 low" in out
    assert "[+1] Robotaxi expansion risk added." in out
    assert "DIGEST (0), REVIEW (0)" in out
    assert f"saved: {report_path}" in out
    assert "more than one new" not in out


def test_triage_runs_are_passed_through_and_votes_saved(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)
    providers = Providers()

    assert check(sec, providers, triage_runs=3) == 0

    assert (providers.triage.calls, providers.agent.calls) == (3, 1)
    [path] = saved_reports(sec)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["report"]["triage_runs"] == 3
    assert saved["report"]["high"][0]["triage_votes"] == ["high", "high", "high"]
    assert saved["decision"]["entries"][0]["triage_votes"] == ["high", "high", "high"]
    assert "triage: 3/3 runs succeeded" in capsys.readouterr().out


def test_dry_run_counts_the_triage_runs(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)

    assert check(sec, Providers(), dry_run=True, triage_runs=3) == 0

    assert "3 triage run(s) + up to 5 agent run(s)" in capsys.readouterr().out


@pytest.mark.parametrize(("flag", "expected"), [([], 1), (["--triage-runs", "2"], 2)])
def test_check_command_passes_triage_runs_through(
    monkeypatch: pytest.MonkeyPatch, flag: list[str], expected: int
) -> None:
    seen: dict[str, Any] = {}

    def fake_check(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_check", fake_check)

    assert cli.main(["check", *flag]) == 0
    assert seen["triage_runs"] == expected


def test_more_than_one_new_filing_is_noted(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD, OLDER)
    remember(sec, TSLA, OLDER)

    assert check(sec, Providers()) == 0

    out = capsys.readouterr().out
    assert f"more than one new 10-K since 2024-01-29 ({OLDER[0]})" in out
    assert remembered(sec, TSLA) == NEW[0]


def test_a_failure_keeps_its_state_and_the_other_entries_still_run(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    edgar.set_filings("AAPL", NEW, OLD)
    remember(sec, TSLA, OLD)
    remember(sec, AAPL, OLD)
    providers = Providers(FakeTriage(errors=[LLMError("rate limited")]))  # first triage fails

    assert check(sec, providers, (TSLA, AAPL)) == 1

    assert remembered(sec, TSLA) == OLD[0]  # retried on the next check
    assert remembered(sec, AAPL) == NEW[0]
    assert [path.name for path in saved_reports(sec)] == [f"AAPL_10-K_{NEW[0]}_I-1A.json"]
    captured = capsys.readouterr()
    assert "error: TSLA 10-K Part I Item 1A: LLMError: rate limited" in captured.err
    assert "1 failed: TSLA 10-K Part I Item 1A" in captured.out


def test_an_sec_failure_is_reported_like_any_other(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    unknown = WatchEntry(ticker="NOPE", form="10-K", item="1A", part="I")

    assert check(sec, Providers(), (unknown, TSLA)) == 1

    assert "error: NOPE 10-K Part I Item 1A: TickerNotFoundError" in capsys.readouterr().err
    assert remembered(sec, TSLA) == NEW[0]


def test_dry_run_changes_nothing(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)  # never seen
    edgar.set_filings("AAPL", NEW, OLD)  # has a new filing
    remember(sec, AAPL, OLD)
    before = state_path(sec.data_dir).read_text(encoding="utf-8")
    providers = Providers()

    assert check(sec, providers, (TSLA, AAPL), dry_run=True) == 0

    assert providers.made == 0
    assert state_path(sec.data_dir).read_text(encoding="utf-8") == before
    # Nothing else was written either: no report, not even a downloaded filing.
    assert [p for p in sec.data_dir.rglob("*") if p.is_file()] == [state_path(sec.data_dir)]
    out = capsys.readouterr().out
    assert f"would start watching from 2026-01-29 ({NEW[0]})" in out
    assert f"would compare 2026-01-29 ({NEW[0]}) with 2025-01-30 ({OLD[0]})" in out


def test_dry_run_with_include_unseen_would_compare(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)

    assert check(sec, Providers(), include_unseen=True, dry_run=True) == 0

    assert "would compare" in capsys.readouterr().out
    assert not state_path(sec.data_dir).exists()


def test_a_failed_report_write_leaves_no_file_and_keeps_state(
    sec: SecClient, edgar: FakeEdgar, monkeypatch: pytest.MonkeyPatch
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)
    real_replace = Path.replace

    def crash_in_reports(self: Path, target: Path) -> Path:
        if self.parent.name == "reports":
            raise OSError("disk full")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", crash_in_reports)

    assert check(sec, Providers()) == 1

    # Only the temp file is left: the real report path never holds half a file.
    assert [path.name for path in saved_reports(sec)] == [f"TSLA_10-K_{NEW[0]}_I-1A.json.partial"]
    assert remembered(sec, TSLA) == OLD[0]


def test_a_failed_state_write_keeps_the_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = state_path(tmp_path)
    seen = SeenFiling(accession_number=OLD[0], filing_date=OLD[1], processed_at=datetime.now(UTC))
    save_state({TSLA.key: seen}, path)
    before = path.read_text(encoding="utf-8")

    def crash(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", crash)
    newer = seen.model_copy(update={"accession_number": NEW[0]})
    with pytest.raises(OSError):
        save_state({TSLA.key: newer}, path)

    assert path.read_text(encoding="utf-8") == before


def test_a_bad_state_file_stops_the_check_and_is_not_overwritten(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    path = state_path(sec.data_dir)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    assert check(sec, Providers()) == 1

    assert path.read_text(encoding="utf-8") == "{not json"
    assert "fix or delete it" in capsys.readouterr().err


def test_only_one_filing_cannot_be_compared() -> None:
    newest = Filing(
        cik="0001318605",
        accession_number=NEW[0],
        form="10-K",
        filing_date=NEW[1],
        report_date=None,
        primary_document="x.htm",
    )
    assert plan_entry([newest], None).action == "start"  # first sight needs no comparison
    with pytest.raises(CheckError, match="need two"):
        plan_entry([newest], None, include_unseen=True)
    with pytest.raises(CheckError, match="no filing"):
        plan_entry([], None)


def test_the_repo_watchlist_loads() -> None:
    entries = load_watchlist(Path(__file__).parents[1] / "watchlist.json")
    assert entries == (TSLA,)
    assert entries[0].key == "TSLA|10-K|I|1A"


def test_watchlist_entries_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "watchlist.json"
    path.write_text('[{"ticker": " tsla ", "form": "10-Q", "item": "2"}]', encoding="utf-8")
    (entry,) = load_watchlist(path)
    assert (entry.ticker, entry.item, entry.part, entry.key) == ("TSLA", "2", None, "TSLA|10-Q|-|2")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not a valid watchlist"),
        ("[]", "no entries"),
        ('[{"ticker": "TSLA", "form": "10-K"}]', "item"),
        ('[{"ticker": "TSLA", "form": "8-K", "item": "1A"}]', "form"),
        ('[{"ticker": " ", "form": "10-K", "item": "1A"}]', "must not be blank"),
        ('[{"ticker": "TSLA", "form": "10-K", "item": "1A", "itme": "7"}]', "itme"),
        (
            '[{"ticker": "TSLA", "form": "10-K", "item": "1A", "part": "I"},'
            ' {"ticker": "tsla", "form": "10-K", "item": "1a", "part": "i"}]',
            "duplicated entries: TSLA|10-K|I|1A",
        ),
    ],
)
def test_a_bad_watchlist_fails_clearly(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "watchlist.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(WatchlistError, match=message):
        load_watchlist(path)


def test_a_missing_watchlist_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(WatchlistError, match="can't read"):
        load_watchlist(tmp_path / "missing.json")


TELEGRAM_TOKEN = "123456:SECRET-test-token"
TELEGRAM_SETTINGS = SETTINGS.model_copy(
    update={"telegram_bot_token": SecretStr(TELEGRAM_TOKEN), "telegram_chat_id": "42"}
)


class Telegram:
    """Stand-in for the Bot API. Runs `on_send` first, so a test can look at disk then."""

    def __init__(
        self,
        response: httpx.Response | None = None,
        on_send: Callable[[], None] | None = None,
    ) -> None:
        self.texts: list[str] = []
        self.response = response or httpx.Response(200, json={"ok": True, "result": {}})
        self.on_send = on_send

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.on_send is not None:
            self.on_send()
        self.texts.append(json.loads(request.content)["text"])
        return self.response

    def notifier(self) -> TelegramNotifier:
        return TelegramNotifier(TELEGRAM_SETTINGS, transport=httpx.MockTransport(self.handle))


def test_check_sends_the_alert_after_saving_the_report_and_state(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)
    report_path = reports_dir(sec.data_dir) / f"TSLA_10-K_{NEW[0]}_I-1A.json"

    def saved_first() -> None:
        assert report_path.exists(), "the report must be saved before sending"
        assert remembered(sec, TSLA) == NEW[0], "the state must be updated before sending"

    telegram = Telegram(on_send=saved_first)

    assert check(sec, Providers(), notify=True, notifier=telegram.notifier()) == 0

    [text] = telegram.texts
    assert text.startswith("FinScope: TSLA 10-K, 1 alert\n")
    assert f"Filed 2026-01-29, accession {NEW[0]}" in text
    assert "ALERT [+1]\nWHAT CHANGED: Robotaxi expansion risk added." in text
    assert "alerts sent to Telegram" in capsys.readouterr().out


def test_a_failed_send_warns_but_keeps_the_report_and_state(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused for {request.url}", request=request)

    bot = TelegramNotifier(TELEGRAM_SETTINGS, transport=httpx.MockTransport(refuse))

    assert check(sec, Providers(), notify=True, notifier=bot) == 0

    assert remembered(sec, TSLA) == NEW[0]
    assert [path.name for path in saved_reports(sec)] == [f"TSLA_10-K_{NEW[0]}_I-1A.json"]
    captured = capsys.readouterr()
    assert "warning: alerts not sent: Telegram sendMessage failed: ConnectError" in captured.err
    assert "none failed" in captured.out
    assert TELEGRAM_TOKEN not in captured.out + captured.err


def test_check_without_telegram_configured_says_so(
    sec: SecClient, edgar: FakeEdgar, capsys: pytest.CaptureFixture[str]
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)

    assert check(sec, Providers(), notify=True, notifier=None) == 0

    assert "alerts not sent: Telegram not configured" in capsys.readouterr().out
    assert remembered(sec, TSLA) == NEW[0]


@pytest.mark.parametrize(
    ("dry_run", "notify"),
    [(True, True), (False, False)],  # --dry-run; --no-notify
)
def test_dry_run_and_no_notify_send_nothing(
    sec: SecClient, edgar: FakeEdgar, dry_run: bool, notify: bool
) -> None:
    edgar.set_filings("TSLA", NEW, OLD)
    remember(sec, TSLA, OLD)
    telegram = Telegram()
    bot = telegram.notifier()

    assert check(sec, Providers(), dry_run=dry_run, notify=notify, notifier=bot) == 0

    assert telegram.texts == []


def test_nothing_is_sent_when_nothing_was_compared(sec: SecClient, edgar: FakeEdgar) -> None:
    edgar.set_filings("TSLA", NEW, OLD)  # first sight: only remembered
    telegram = Telegram()

    assert check(sec, Providers(), notify=True, notifier=telegram.notifier()) == 0

    assert telegram.texts == []


@pytest.mark.parametrize(("flag", "expected"), [([], False), (["--no-notify"], True)])
def test_check_command_passes_no_notify_through(
    monkeypatch: pytest.MonkeyPatch, flag: list[str], expected: bool
) -> None:
    seen: dict[str, Any] = {}

    def fake_check(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_check", fake_check)

    assert cli.main(["check", *flag]) == 0
    assert seen["no_notify"] is expected
