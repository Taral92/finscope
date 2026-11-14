import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.__main__ import _feedback
from finscope.decide import Decision, DecisionEntry, Tier
from finscope.detect import CheckResult, WatchEntry, reports_dir, save_check_result
from finscope.edgar import Filing
from finscope.feedback import (
    Feedback,
    FeedbackError,
    feedback_dir,
    load_feedback,
    load_report,
    newest_report,
    save_feedback,
)
from finscope.report import SectionReport

CHECKED = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)
NEW_ACCESSION = "0001628280-26-000003"


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


def check_result(
    *entries: DecisionEntry, accession: str = NEW_ACCESSION, checked_at: datetime = CHECKED
) -> CheckResult:
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
        checked_at=checked_at,
        entry=WatchEntry(ticker="TSLA", form="10-K", item="1A", part="I"),
        old_filing=filing("0001628280-25-000002", date(2025, 1, 30)),
        new_filing=filing(accession, date(2026, 1, 29)),
        report=report,
        decision=Decision(entries=entries),
    )


def save_report(data_dir: Path, result: CheckResult) -> Path:
    return save_check_result(result, reports_dir(data_dir))


def feedback_of(data_dir: Path, result: CheckResult) -> list[tuple[str, str, str]]:
    """(id, tier, verdict) of every saved answer for the result's report."""
    answers = load_feedback(feedback_dir(data_dir) / result.file_name)
    assert all(answer.report == result.file_name for answer in answers)
    assert all(answer.checked_at == result.checked_at for answer in answers)
    assert all(answer.answered_at.tzinfo is not None for answer in answers)
    return [(answer.id, answer.tier, answer.verdict) for answer in answers]


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """What the user types, in order; set the list's contents in the test.

    Runs out loudly: a question too many is a bug, not an empty answer.
    """
    answers: list[str] = []
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        if not answers:
            raise AssertionError(f"asked more than expected: {prompt!r} (seen: {prompts})")
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    yield answers
    assert answers == [], f"answers never asked for: {answers}"


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    answers = (
        Feedback(
            report="TSLA_10-K_x_I-1A.json",
            checked_at=CHECKED,
            id="+1",
            tier="alert",
            verdict="useful",
            answered_at=datetime(2026, 11, 12, 9, 30, tzinfo=UTC),
        ),
        Feedback(
            report="TSLA_10-K_x_I-1A.json",
            checked_at=CHECKED,
            id="+2",
            tier="digest",
            verdict="useful",
            answered_at=datetime(2026, 11, 12, 9, 31, tzinfo=UTC),
        ),
    )
    path = tmp_path / "feedback" / "TSLA_10-K_x_I-1A.json"

    save_feedback(answers, path)

    assert load_feedback(path) == answers
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no .partial left behind


def test_bad_feedback_file_is_a_feedback_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"id": "+1", "verdict": "maybe"}]', encoding="utf-8")

    with pytest.raises(FeedbackError, match="fix or delete it"):
        load_feedback(path)
    with pytest.raises(FeedbackError, match="can't read"):
        load_feedback(tmp_path / "missing.json")


def test_alert_and_review_answers_are_saved_with_the_tier(
    tmp_path: Path, typed: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+1", "alert"), entry("~3", "review", "agent failed"))
    save_report(tmp_path, result)
    typed += ["u", "w"]

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful"), ("~3", "review", "wrong")]
    out = capsys.readouterr().out
    assert "=== ALERT 1/1: [+1] ===\nWHAT CHANGED: Change +1 added.\n" in out
    assert "WHY IT MATTERS: It matters because of +1." in out
    assert "=== REVIEW 1/1: [~3] ===\nreason: agent failed" in out
    assert "TRIAGE-ONLY" not in out  # the triage reason stays in the saved report
    assert "2 answer(s): 1 useful, 0 not_useful, 1 wrong" in out
    assert f"Saved: {feedback_dir(tmp_path) / result.file_name}" in out


def test_full_words_are_accepted_too(tmp_path: Path, typed: list[str]) -> None:
    result = check_result(entry("+1", "alert"), entry("+2", "alert"))
    save_report(tmp_path, result)
    typed += ["not_useful", "WRONG"]

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [
        ("+1", "alert", "not_useful"),
        ("+2", "alert", "wrong"),
    ]


def test_missed_alert_ids_become_useful_digest_feedback(
    tmp_path: Path, typed: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(
        entry("+1", "alert"), entry("+2", "digest"), entry("+4", "digest"), entry("~5", "digest")
    )
    save_report(tmp_path, result)
    typed += ["n", " ~5, +2 ,~5"]  # spaces and a repeat are fine; each id counts once

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [
        ("+1", "alert", "not_useful"),
        ("~5", "digest", "useful"),
        ("+2", "digest", "useful"),
    ]
    out = capsys.readouterr().out
    assert "=== DIGEST (3): +2, +4, ~5 ===" in out
    assert "incl. 2 missed alerts from the digest" in out


def test_enter_at_the_digest_question_records_no_miss(tmp_path: Path, typed: list[str]) -> None:
    result = check_result(entry("+1", "alert"), entry("+2", "digest"))
    save_report(tmp_path, result)
    typed += ["u", ""]

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful")]


def test_bad_key_twice_then_accepted(
    tmp_path: Path, typed: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+1", "alert"))
    save_report(tmp_path, result)
    typed += ["x", "yes", "u"]

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful")]
    out = capsys.readouterr().out
    assert out.count("Unknown key") == 2
    assert "Unknown key 'yes': u = useful, n = not_useful, w = wrong" in out


def test_bad_digest_id_twice_then_accepted(
    tmp_path: Path, typed: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+1", "alert"), entry("+2", "digest"))
    save_report(tmp_path, result)
    typed += ["u", "+9", "+1,+2", "+2"]  # +1 is the alert, not a digest entry

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful"), ("+2", "digest", "useful")]
    out = capsys.readouterr().out
    assert "Unknown id(s) +9; the digest ids are +2" in out
    assert "Unknown id(s) +1; the digest ids are +2" in out


def test_every_answer_is_saved_before_the_next_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+1", "alert"), entry("+2", "alert"))
    save_report(tmp_path, result)
    path = feedback_dir(tmp_path) / result.file_name
    asked = 0

    def fake_input(prompt: str = "") -> str:
        nonlocal asked
        asked += 1
        if asked == 2:  # the first answer must already be on disk
            assert feedback_of(tmp_path, result) == [("+1", "alert", "useful")]
            raise KeyboardInterrupt  # like Ctrl+C
        return "u"

    monkeypatch.setattr("builtins.input", fake_input)

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert asked == 2
    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful")]
    assert f"Stopped. Every answer so far is saved in {path}" in capsys.readouterr().out


def test_ctrl_c_before_the_first_answer_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+1", "alert"))
    save_report(tmp_path, result)

    def interrupt(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    assert _feedback(tmp_path, report=None, redo=False) == 0

    assert not feedback_dir(tmp_path).exists()
    assert "Stopped before the first answer; nothing saved" in capsys.readouterr().out


def test_newest_unanswered_report_is_picked(tmp_path: Path, typed: list[str]) -> None:
    def report(accession: str, day: int) -> CheckResult:
        return check_result(
            entry("+1", "alert"),
            accession=accession,
            checked_at=datetime(2026, 11, day, tzinfo=UTC),
        )

    oldest = report("0001628280-24-000001", 1)
    middle = report("0001628280-26-000003", 2)  # the newest without feedback
    newest = report("0001628280-25-000002", 3)  # the accession doesn't order them
    for result in (oldest, middle, newest):
        save_report(tmp_path, result)
    save_feedback((), feedback_dir(tmp_path) / newest.file_name)

    assert newest_report(reports_dir(tmp_path), feedback_dir(tmp_path)) == (
        reports_dir(tmp_path) / middle.file_name
    )
    assert newest_report(reports_dir(tmp_path), feedback_dir(tmp_path), include_answered=True) == (
        reports_dir(tmp_path) / newest.file_name
    )

    typed += ["u"]
    assert _feedback(tmp_path, report=None, redo=False) == 0
    assert feedback_of(tmp_path, middle) == [("+1", "alert", "useful")]
    assert not (feedback_dir(tmp_path) / oldest.file_name).exists()


def test_nothing_to_rate_when_every_report_has_feedback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _feedback(tmp_path, report=None, redo=False) == 0
    assert "no reports in" in capsys.readouterr().out

    result = check_result(entry("+1", "alert"))
    save_report(tmp_path, result)
    save_feedback((), feedback_dir(tmp_path) / result.file_name)

    assert _feedback(tmp_path, report=None, redo=False) == 0  # input() is never called
    assert "every report in" in capsys.readouterr().out


def test_redo_asks_again_and_replaces_the_old_answers(tmp_path: Path, typed: list[str]) -> None:
    result = check_result(entry("+1", "alert"))
    save_report(tmp_path, result)
    typed += ["u"]
    assert _feedback(tmp_path, report=None, redo=False) == 0
    assert feedback_of(tmp_path, result) == [("+1", "alert", "useful")]

    typed += ["w"]
    assert _feedback(tmp_path, report=None, redo=True) == 0

    assert feedback_of(tmp_path, result) == [("+1", "alert", "wrong")]


def test_report_option_picks_one_and_refuses_an_answered_one_without_redo(
    tmp_path: Path, typed: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    older = check_result(
        entry("+1", "alert"),
        accession="0001628280-25-000002",
        checked_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    newer = check_result(entry("+1", "alert"))
    save_report(tmp_path, older)
    save_report(tmp_path, newer)
    name = older.file_name.removesuffix(".json")  # with or without .json

    typed += ["n"]
    assert _feedback(tmp_path, report=name, redo=False) == 0
    assert feedback_of(tmp_path, older) == [("+1", "alert", "not_useful")]
    assert not (feedback_dir(tmp_path) / newer.file_name).exists()

    assert _feedback(tmp_path, report=older.file_name, redo=False) == 1
    assert "already has feedback" in capsys.readouterr().err

    typed += ["u"]
    assert _feedback(tmp_path, report=name, redo=True) == 0
    assert feedback_of(tmp_path, older) == [("+1", "alert", "useful")]


def test_a_report_saved_before_triage_runs_existed_still_loads(tmp_path: Path) -> None:
    result = check_result(entry("+1", "alert"))
    path = save_report(tmp_path, result)
    old_format = json.loads(path.read_text(encoding="utf-8"))
    for field in ("triage_runs", "triage_succeeded", "triage_errors"):
        del old_format["report"][field]
    path.write_text(json.dumps(old_format), encoding="utf-8")

    loaded = load_report(path)

    assert loaded == result  # the defaults say: one run, it worked
    assert (loaded.report.triage_runs, loaded.report.triage_succeeded) == (1, 1)


def test_a_missing_or_bad_report_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _feedback(tmp_path, report="nope.json", redo=False) == 1
    assert "can't read report" in capsys.readouterr().err

    bad = reports_dir(tmp_path) / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    assert _feedback(tmp_path, report=None, redo=False) == 1  # found while picking the newest
    assert "can't read report" in capsys.readouterr().err


def test_no_alerts_and_no_reviews_says_so_and_marks_the_report_done(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_result(entry("+2", "digest"))
    save_report(tmp_path, result)

    assert _feedback(tmp_path, report=None, redo=False) == 0  # input() is never called

    assert feedback_of(tmp_path, result) == []
    out = capsys.readouterr().out
    assert "0 alert(s), 0 to review, 1 digest" in out
    assert "nothing to rate: no alerts and nothing to review" in out
    assert _feedback(tmp_path, report=None, redo=False) == 0
    assert "every report in" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["feedback"], {"report": None, "redo": False}),
        (["feedback", "--report", "x.json", "--redo"], {"report": "x.json", "redo": True}),
    ],
)
def test_feedback_command_passes_its_options_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: dict[str, Any]
) -> None:
    seen: dict[str, Any] = {}

    def fake_feedback(data_dir: Path, **kwargs: Any) -> int:
        seen["data_dir"] = data_dir
        seen.update(kwargs)
        return 0

    class FakeSettings:
        data_dir = tmp_path

    monkeypatch.setattr(cli, "_feedback", fake_feedback)
    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())

    assert cli.main(argv) == 0
    assert seen == {"data_dir": tmp_path, **expected}
