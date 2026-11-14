from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.__main__ import _eval_alerts
from finscope.alert_eval import (
    AlertEvalResult,
    load_all_feedback,
    score_reports,
    summarize_alerts,
)
from finscope.decide import Tier
from finscope.feedback import Feedback, FeedbackError, Verdict, save_feedback

REPORT_A = "TSLA_10-K_0001628280-26-000003_I-1A.json"  # checked 2026-11-07
REPORT_B = "AAPL_10-K_0000320193-26-000001_I-1A.json"  # checked 2026-11-10
CHECKED = {
    REPORT_A: datetime(2026, 11, 7, tzinfo=UTC),
    REPORT_B: datetime(2026, 11, 10, tzinfo=UTC),
}
ANSWERED = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def answer(report: str, change_id: str, tier: Tier, verdict: Verdict) -> Feedback:
    return Feedback(
        report=report,
        checked_at=CHECKED[report],
        id=change_id,
        tier=tier,
        verdict=verdict,
        answered_at=ANSWERED,
    )


A = [
    answer(REPORT_A, "~1", "alert", "useful"),
    answer(REPORT_A, "~5", "alert", "not_useful"),
    answer(REPORT_A, "~19", "alert", "wrong"),
    answer(REPORT_A, "+7", "review", "useful"),
    answer(REPORT_A, "+3", "digest", "useful"),  # a missed alert
]
B = [
    answer(REPORT_B, "+1", "alert", "useful"),
    answer(REPORT_B, "+2", "alert", "useful"),
    answer(REPORT_B, "~4", "review", "not_useful"),
]


def write_feedback(directory: Path, *groups: list[Feedback]) -> None:
    for group in groups:
        save_feedback(group, directory / group[0].report)


def test_scores_are_per_report_oldest_check_first() -> None:
    # B is written first, but A was checked earlier, so A comes first.
    scores = score_reports(B + A)

    assert [score.report for score in scores] == [REPORT_A, REPORT_B]
    a, b = scores
    assert (a.useful, a.not_useful, a.wrong) == (("~1",), ("~5",), ("~19",))
    assert (a.alerts_rated, a.precision, a.wrong_rate) == (
        3,
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
    )
    assert (a.reviews_rated, a.reviews_useful, a.missed) == (1, 1, ("+3",))
    assert (b.useful, b.not_useful, b.wrong) == (("+1", "+2"), (), ())
    assert (b.precision, b.wrong_rate, b.reviews_useful, b.missed) == (1.0, 0.0, 0, ())


def test_summary_sums_every_report() -> None:
    result = summarize_alerts(score_reports(A + B), ANSWERED)

    assert (result.alerts_rated, result.useful, result.not_useful, result.wrong) == (5, 3, 1, 1)
    assert (result.reviews_rated, result.reviews_useful, result.missed) == (2, 1, 1)
    assert result.precision == pytest.approx(3 / 5)
    assert result.wrong_rate == pytest.approx(1 / 5)
    assert result.file_name == "20261114T120000Z_alerts.json"


def test_no_rated_alert_gives_no_precision() -> None:
    only_review = [answer(REPORT_A, "+7", "review", "useful")]

    result = summarize_alerts(score_reports(only_review), ANSWERED)

    assert result.alerts_rated == 0
    assert result.precision is None
    assert result.wrong_rate is None
    assert result.reports[0].precision is None


def test_load_all_feedback_reads_every_file_and_fails_on_a_bad_one(tmp_path: Path) -> None:
    write_feedback(tmp_path, A, B)

    assert len(load_all_feedback(tmp_path)) == len(A) + len(B)

    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FeedbackError, match=r"bad\.json"):
        load_all_feedback(tmp_path)


def test_command_prints_per_report_and_totals_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback, results = tmp_path / "feedback", tmp_path / "results"
    write_feedback(feedback, A, B)

    assert _eval_alerts(feedback, results) == 0

    [saved] = list(results.iterdir())
    assert saved.name.endswith("_alerts.json")
    result = AlertEvalResult.model_validate_json(saved.read_text(encoding="utf-8"))
    assert (result.alerts_rated, result.useful) == (5, 3)
    out = capsys.readouterr().out
    assert "=== PER REPORT (2) ===" in out
    assert f"{REPORT_A}  checked 2026-11-07" in out
    assert "alerts 3: 1 useful, 1 not_useful, 1 wrong   precision 33%" in out
    assert "  wrong:  ~19" in out
    assert "  missed: +3" in out
    assert "alert precision:  60%" in out
    assert "wrong:            20%" in out
    assert "reviews useful:   1/2" in out
    assert "missed alerts:    1" in out
    assert f"saved:  {saved}" in out


def test_command_without_feedback_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _eval_alerts(tmp_path / "feedback", tmp_path / "results") == 1

    assert "no feedback in" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_command_with_a_bad_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.json").write_text("[]x", encoding="utf-8")

    assert _eval_alerts(tmp_path, tmp_path / "results") == 1

    assert "can't read feedback" in capsys.readouterr().err


def test_eval_alerts_command_uses_the_data_dir_and_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_eval(feedback: Path, results_dir: Path) -> int:
        seen.update(feedback=feedback, results_dir=results_dir)
        return 0

    class FakeSettings:
        data_dir = tmp_path

    monkeypatch.setattr(cli, "_eval_alerts", fake_eval)
    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())

    assert cli.main(["eval-alerts"]) == 0
    assert seen == {"feedback": tmp_path / "feedback", "results_dir": cli.RESULTS_DIR}
