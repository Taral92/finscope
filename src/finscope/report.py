"""One run over a section: triage every change, explain the high ones, fact-check each answer.

Triage runs `triage_runs` times and a change counts as high if any run rated it high, so a
borderline change still reaches the agent; each item keeps every run's rating (its votes).

Plain orchestration code: no prompts of its own. It chains triage_section,
explain_change_with_agent and the fact-checker, and returns plain data a later alert step
can use.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from finscope.agent import AgentError
from finscope.agent_tools import ExplainContext
from finscope.config import Settings
from finscope.diff import SectionDiff, changes_by_id
from finscope.explain import explain_change_with_agent
from finscope.llm import LLMError, LLMProvider, cost_for_tokens
from finscope.sections import Section
from finscope.triage import Importance, triage_section_votes
from finscope.verify import (
    ParsedExplanation,
    Verification,
    parse_explanation,
    sources_for,
    verify_explanation,
)
from finscope.xbrl import MetricValue


class HighItem(BaseModel):
    """One high change the agent was asked to explain: its answer and fact-check, or an error.

    An answer that doesn't follow the format is not an error: it is kept, and the problems
    show up as format issues in a failed verification.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    triage_reason: str
    explanation: str | None  # None when the agent failed
    error: str | None  # why the agent failed; None when it answered
    parsed: ParsedExplanation | None  # the answer split into its labeled lines; None on failure
    verification: Verification | None  # None when there is no answer to check
    # Every model call of this run; None when a call failed and its usage isn't known.
    input_tokens: int | None
    output_tokens: int | None
    # Every triage run's rating of this change; empty when it wasn't picked by triage.
    triage_votes: tuple[Importance, ...] = ()


class RatedItem(BaseModel):
    """A change with only its triage rating: a medium one, or a high one over the cap."""

    model_config = ConfigDict(frozen=True)

    id: str
    triage_reason: str
    triage_votes: tuple[Importance, ...] = ()  # every triage run's rating; empty if not triaged


class SectionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    company: str
    form: str
    section_title: str
    change_count: int
    high_count: int
    medium_count: int
    low_count: int
    high: tuple[HighItem, ...]  # explained (or attempted) high changes, in diff order
    not_explained: tuple[RatedItem, ...]  # high changes over the max_explain cap, in diff order
    medium: tuple[RatedItem, ...]
    # Defaults describe a report saved before several triage runs existed: one run, it worked.
    triage_runs: int = 1  # triage runs attempted
    triage_succeeded: int = 1
    triage_errors: tuple[str, ...] = ()  # one per failed triage run
    # Totals over every call: each triage run plus each agent run, failed runs included.
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None  # None when no prices are configured
    usage_complete: bool  # False when a failed call's usage is unknown (totals undercount)


def build_section_report(
    diff: SectionDiff,
    *,
    old_section: Section,
    new_section: Section,
    metrics: tuple[MetricValue, ...],
    company: str,
    form: str,
    section_title: str,
    triage_provider: LLMProvider,
    agent_provider: LLMProvider,
    settings: Settings,
    max_explain: int = 20,
    triage_runs: int = 1,
) -> SectionReport:
    """Triage the section, then explain and fact-check up to `max_explain` high changes.

    Triage runs `triage_runs` times; a change's rating is the highest any run gave. A failed
    triage run is recorded and skipped; if every run fails, its error (LLMError, TriageError)
    is raised: without ratings there is nothing to report on. A failed agent run is recorded
    on its item and the next change still runs.
    """
    if max_explain < 0:
        raise ValueError("max_explain must be 0 or more")
    triage = triage_section_votes(
        diff,
        company=company,
        form=form,
        section_title=section_title,
        provider=triage_provider,
        settings=settings,
        runs=triage_runs,
    )
    changes = changes_by_id(diff)
    high = [item for item in triage.items if item.importance == "high"]
    medium = [item for item in triage.items if item.importance == "medium"]
    low_count = sum(1 for item in triage.items if item.importance == "low")

    explained = []
    for item in high[:max_explain]:
        context = ExplainContext(
            change=changes[item.id],
            old_section=old_section,
            new_section=new_section,
            metrics=metrics,
        )
        explained.append(
            explain_and_check(
                item.id,
                item.reason,
                context,
                company=company,
                form=form,
                section_title=section_title,
                provider=agent_provider,
                settings=settings,
            ).model_copy(update={"triage_votes": item.votes})
        )

    input_tokens = triage.input_tokens + sum(item.input_tokens or 0 for item in explained)
    output_tokens = triage.output_tokens + sum(item.output_tokens or 0 for item in explained)
    return SectionReport(
        company=company,
        form=form,
        section_title=section_title,
        change_count=len(triage.items),
        high_count=len(high),
        medium_count=len(medium),
        low_count=low_count,
        high=tuple(explained),
        not_explained=tuple(
            RatedItem(id=item.id, triage_reason=item.reason, triage_votes=item.votes)
            for item in high[max_explain:]
        ),
        medium=tuple(
            RatedItem(id=item.id, triage_reason=item.reason, triage_votes=item.votes)
            for item in medium
        ),
        triage_runs=triage.runs,
        triage_succeeded=triage.succeeded,
        triage_errors=triage.errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Price is per token, so the cost of the total is the sum of each call's cost.
        cost_usd=cost_for_tokens(input_tokens, output_tokens, settings),
        # A failed triage run's usage is lost with it, like a failed agent call's.
        usage_complete=triage.succeeded == triage.runs
        and all(item.input_tokens is not None for item in explained),
    )


def explain_and_check(
    change_id: str,
    triage_reason: str,
    context: ExplainContext,
    *,
    company: str,
    form: str,
    section_title: str,
    provider: LLMProvider,
    settings: Settings,
) -> HighItem:
    """One agent run on one change, fact-checked. Agent and model-call failures are recorded.

    Used by build_section_report for each high change, and by the agent eval for each run.
    """
    try:
        explanation = explain_change_with_agent(
            context,
            company=company,
            form=form,
            section_title=section_title,
            provider=provider,
            settings=settings,
        )
    except AgentError as error:
        # The trace says what the steps so far used, so this run's usage is still known.
        return HighItem(
            id=change_id,
            triage_reason=triage_reason,
            explanation=None,
            error=f"agent failed: {error}",
            parsed=None,
            verification=None,
            input_tokens=sum(step.input_tokens for step in error.trace),
            output_tokens=sum(step.output_tokens for step in error.trace),
        )
    except LLMError as error:
        # A model call failed inside the loop: what the earlier steps used is lost with it.
        return HighItem(
            id=change_id,
            triage_reason=triage_reason,
            explanation=None,
            error=f"model call failed: {error}",
            parsed=None,
            verification=None,
            input_tokens=None,
            output_tokens=None,
        )

    run = explanation.run
    parsed = parse_explanation(run.text, expect_evidence=True)
    return HighItem(
        id=change_id,
        triage_reason=triage_reason,
        explanation=run.text,
        error=None,
        parsed=parsed,
        verification=verify_explanation(parsed, sources_for(context.change, run.tool_outputs)),
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )
