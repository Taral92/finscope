"""Your feedback on a check's alerts, saved so an eval can score alert precision. Plain code.

One JSON file per report, data/feedback/<the report's file name>: a list of Feedback, one per
answer. Each answer is self-contained (it names its report), so an eval can pool every file.
A report with a feedback file is never asked about again unless you say so (--redo).
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from finscope.decide import DecisionEntry, Tier
from finscope.detect import CheckResult

Verdict = Literal["useful", "not_useful", "wrong"]
# The letter or the full word, either is accepted.
KEYS: dict[str, Verdict] = {
    "u": "useful",
    "useful": "useful",
    "n": "not_useful",
    "not_useful": "not_useful",
    "w": "wrong",
    "wrong": "wrong",
}

GUIDE = """\
Rate each alert and review:
  u  useful      glad I was told
  n  not_useful  true, but not worth telling me
  w  wrong       the explanation is false or misleading (not just unimportant)
Every answer is saved right away."""
PROMPT = "[u/n/w] > "
MISSED_PROMPT = "any of these you'd have wanted as an alert? (ids, comma separated, or enter) > "


class FeedbackError(Exception):
    """A report or feedback file can't be used; the message says which and why."""


class Feedback(BaseModel):
    """One answer about one decision entry of one report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: str  # the report's file name, e.g. TSLA_10-K_0001628280-26-003952_I-1A.json
    checked_at: datetime  # copied from the report, so a re-run report is told apart
    id: str
    tier: Tier  # where the entry landed; tier "digest" with verdict "useful" is a missed alert
    verdict: Verdict
    answered_at: datetime


_FEEDBACK = TypeAdapter(tuple[Feedback, ...])


def feedback_dir(data_dir: Path) -> Path:
    return data_dir / "feedback"


def load_report(path: Path) -> CheckResult:
    """A saved check result, or a FeedbackError saying why the file can't be used."""
    try:
        return CheckResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise FeedbackError(f"can't read report {path}:\n{error}") from error


def load_feedback(path: Path) -> tuple[Feedback, ...]:
    """Every answer in the file, or a FeedbackError; never repaired silently."""
    try:
        return _FEEDBACK.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise FeedbackError(f"can't read feedback {path}; fix or delete it:\n{error}") from error


def save_feedback(answers: Sequence[Feedback], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same temp-file-then-rename as the reports, so a crash never leaves half a file.
    partial = path.with_name(path.name + ".partial")
    text = _FEEDBACK.dump_json(tuple(answers), indent=2).decode() + "\n"
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def newest_report(reports: Path, feedback: Path, *, include_answered: bool = False) -> Path | None:
    """The report checked last among those without a feedback file; None if there is none.

    `include_answered=True` considers every report. "Newest" is by the report's checked_at,
    not the file's timestamp, so copying files around changes nothing.
    """
    candidates = [
        path
        for path in sorted(reports.glob("*.json"))
        if include_answered or not (feedback / path.name).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: load_report(path).checked_at)


def ask_feedback(
    result: CheckResult,
    path: Path,
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
) -> tuple[Feedback, ...]:
    """Ask about every alert and review, then which digests should have been alerts.

    Saved to `path` after every answer, so an interrupted session loses nothing. Any
    earlier answers in `path` are replaced. `ask` and `show` stand in for input and print.
    """
    decision = result.decision
    alerts, reviews, digests = (
        decision.tier("alert"),
        decision.tier("review"),
        decision.tier("digest"),
    )
    answers: list[Feedback] = []

    def record(entry: DecisionEntry, verdict: Verdict) -> None:
        answers.append(
            Feedback(
                report=path.name,
                checked_at=result.checked_at,
                id=entry.id,
                tier=entry.tier,
                verdict=verdict,
                answered_at=datetime.now(UTC),
            )
        )

    show(GUIDE)
    for number, entry in enumerate(alerts, start=1):
        show(f"\n=== ALERT {number}/{len(alerts)}: [{entry.id}] ===")
        show(f"WHAT CHANGED: {entry.what_changed}")
        show(f"WHY IT MATTERS: {entry.why_it_matters}")
        record(entry, _ask_verdict(ask, show))
        save_feedback(answers, path)
    for number, entry in enumerate(reviews, start=1):
        show(f"\n=== REVIEW {number}/{len(reviews)}: [{entry.id}] ===")
        show(f"reason: {entry.reason}")
        record(entry, _ask_verdict(ask, show))
        save_feedback(answers, path)
    if digests:
        by_id = {entry.id: entry for entry in digests}
        show(f"\n=== DIGEST ({len(digests)}): {', '.join(by_id)} ===")
        for change_id in _ask_missed(ask, show, list(by_id)):
            record(by_id[change_id], "useful")  # a digest you wanted as an alert: a miss
    save_feedback(answers, path)  # also when nothing was recorded: the report is done
    return tuple(answers)


def _ask_verdict(ask: Callable[[str], str], show: Callable[[str], None]) -> Verdict:
    while True:
        key = ask(PROMPT).strip().lower()
        if key in KEYS:
            return KEYS[key]
        show(f"Unknown key {key!r}: u = useful, n = not_useful, w = wrong")


def _ask_missed(
    ask: Callable[[str], str], show: Callable[[str], None], ids: list[str]
) -> list[str]:
    """Digest ids typed at the missed-alert question, each once; [] for an empty answer."""
    while True:
        chosen = [part.strip() for part in ask(MISSED_PROMPT).split(",") if part.strip()]
        unknown = [change_id for change_id in chosen if change_id not in ids]
        if not unknown:
            return list(dict.fromkeys(chosen))
        show(f"Unknown id(s) {', '.join(unknown)}; the digest ids are {', '.join(ids)}")
