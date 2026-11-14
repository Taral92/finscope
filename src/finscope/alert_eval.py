"""Score the alerts against your feedback: how many were worth it? Plain code, no LLM.

Reads every file in data/feedback/ (one per rated report) and counts, per report and in
total, how the alerts were rated. The two numbers that matter:
- alert precision: useful alerts / rated alerts. Low means the system alerts too eagerly.
- wrong rate: wrong alerts / rated alerts. These passed the fact-check and still misled you,
  which a quote-and-number check can't catch: the explanation itself is the problem.
Missed alerts (digest entries you wanted as alerts) are listed too; they are the recall side.
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from finscope.evals import write_json_atomic
from finscope.feedback import Feedback, load_feedback


class ReportScore(BaseModel):
    """Your verdicts on one report, by id, so a bad alert can be looked up again."""

    model_config = ConfigDict(frozen=True)

    report: str
    checked_at: datetime
    useful: tuple[str, ...]  # alert ids
    not_useful: tuple[str, ...]
    wrong: tuple[str, ...]
    reviews_rated: int
    reviews_useful: int
    missed: tuple[str, ...]  # digest ids you'd have wanted as alerts

    @property
    def alerts_rated(self) -> int:
        return len(self.useful) + len(self.not_useful) + len(self.wrong)

    @property
    def precision(self) -> float | None:
        """useful / rated; None when no alert was rated."""
        return len(self.useful) / self.alerts_rated if self.alerts_rated else None

    @property
    def wrong_rate(self) -> float | None:
        return len(self.wrong) / self.alerts_rated if self.alerts_rated else None


class AlertEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    reports: tuple[ReportScore, ...]  # oldest check first
    alerts_rated: int
    useful: int
    not_useful: int
    wrong: int
    reviews_rated: int
    reviews_useful: int
    missed: int
    precision: float | None
    wrong_rate: float | None

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_alerts.json"


def load_all_feedback(directory: Path) -> list[Feedback]:
    """Every answer in every feedback file; a bad file is a FeedbackError, not skipped."""
    answers: list[Feedback] = []
    for path in sorted(directory.glob("*.json")):
        answers.extend(load_feedback(path))
    return answers


def score_reports(answers: Sequence[Feedback]) -> tuple[ReportScore, ...]:
    """One score per report, oldest check first."""
    by_report: dict[str, list[Feedback]] = defaultdict(list)
    for answer in answers:
        by_report[answer.report].append(answer)
    scores = [_score_report(name, group) for name, group in by_report.items()]
    return tuple(sorted(scores, key=lambda score: (score.checked_at, score.report)))


def summarize_alerts(scores: Sequence[ReportScore], created_at: datetime) -> AlertEvalResult:
    alerts_rated = sum(score.alerts_rated for score in scores)
    useful = sum(len(score.useful) for score in scores)
    wrong = sum(len(score.wrong) for score in scores)
    return AlertEvalResult(
        created_at=created_at,
        reports=tuple(scores),
        alerts_rated=alerts_rated,
        useful=useful,
        not_useful=sum(len(score.not_useful) for score in scores),
        wrong=wrong,
        reviews_rated=sum(score.reviews_rated for score in scores),
        reviews_useful=sum(score.reviews_useful for score in scores),
        missed=sum(len(score.missed) for score in scores),
        precision=useful / alerts_rated if alerts_rated else None,
        wrong_rate=wrong / alerts_rated if alerts_rated else None,
    )


def save_alert_eval(result: AlertEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _score_report(name: str, answers: Sequence[Feedback]) -> ReportScore:
    alerts = [answer for answer in answers if answer.tier == "alert"]
    reviews = [answer for answer in answers if answer.tier == "review"]
    return ReportScore(
        report=name,
        checked_at=answers[0].checked_at,
        useful=tuple(a.id for a in alerts if a.verdict == "useful"),
        not_useful=tuple(a.id for a in alerts if a.verdict == "not_useful"),
        wrong=tuple(a.id for a in alerts if a.verdict == "wrong"),
        reviews_rated=len(reviews),
        reviews_useful=sum(1 for a in reviews if a.verdict == "useful"),
        # A digest entry gets feedback only when you named it as a missed alert.
        missed=tuple(a.id for a in answers if a.tier == "digest" and a.verdict == "useful"),
    )
