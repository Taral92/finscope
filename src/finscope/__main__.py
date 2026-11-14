"""Command-line entry point.

python -m finscope filings TSLA                     list recent 10-K, 10-Q and 8-K filings
python -m finscope read TSLA [--form F]             download the latest filing as plain text
python -m finscope sections TSLA [--form F]         list the latest filing's Items
python -m finscope sections TSLA --item 1A [--part I]   print one Item's text
python -m finscope facts TSLA [--form F]            key financial numbers from XBRL
python -m finscope diff TSLA [--form F] [--item 1A] [--part I]   compare with the previous filing
python -m finscope explain TSLA --item 1A --kind added --n 4      AI explanation of one change
python -m finscope explain TSLA --item 1A --kind added --n 4 --agent   same, model uses tools first
python -m finscope triage TSLA --item 1A [--part I]     AI rates every change low/medium/high
python -m finscope report TSLA --item 1A [--max-explain 20] [--triage-runs 1]  triage, explain,
                                                    fact-check, decide
python -m finscope label TSLA --item 1A [--part I]      you rate every change (eval labels)
python -m finscope eval-triage TSLA --item 1A [--runs 3]   score AI triage against your labels
python -m finscope eval-known [--runs 3]                score AI triage on made-up known cases
python -m finscope eval-agent TSLA --item 1A [--runs 3] [--ids '~4,~11']  agent stability eval
python -m finscope eval-known-agent [--runs 2]          score the agent's importance on known cases
python -m finscope check [--dry-run] [--include-unseen] [--triage-runs 1] [--no-notify]
                                                    report on newly filed watched sections
python -m finscope feedback [--report NAME] [--redo]   rate a report's alerts (feeds evals)
python -m finscope telegram-chat-id                 chats that messaged your Telegram bot
python -m finscope notify-test                      send a test message to your Telegram chat
"""

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from functools import cache
from pathlib import Path

from finscope.agent import AgentError, TraceStep
from finscope.agent_eval import (
    AgentEvalError,
    AgentEvalResult,
    ChangeStability,
    StabilityRun,
    agent_prompt_version,
    run_stability,
    save_agent_eval,
    select_changes,
    summarize,
)
from finscope.agent_tools import ExplainContext
from finscope.config import Settings, get_settings
from finscope.decide import Decision, DecisionEntry, Tier, decide
from finscope.detect import (
    WATCHLIST_PATH,
    CheckError,
    CheckResult,
    SeenFiling,
    StateError,
    WatchEntry,
    WatchlistError,
    load_state,
    load_watchlist,
    plan_entry,
    reports_dir,
    save_check_result,
    save_state,
    state_path,
)
from finscope.diff import (
    ParagraphChange,
    SectionDiff,
    change_id,
    change_text,
    changes_by_id,
    diff_sections,
    numbered_changes,
)
from finscope.edgar import Filing, SecClient, TickerNotFoundError, filing_path
from finscope.evals import (
    LEVELS,
    RESULTS_DIR,
    ChangeRatings,
    EvalResult,
    EvalScore,
    Importance,
    RatedChange,
    RunRecord,
    coverage_of,
    prompt_version,
    save_result,
    score_triage,
)
from finscope.explain import explain_change, explain_change_with_agent
from finscope.feedback import (
    FeedbackError,
    ask_feedback,
    feedback_dir,
    load_report,
    newest_report,
    save_feedback,
)
from finscope.filings import load_filing_text
from finscope.known_agent import (
    AgentCaseResult,
    KnownAgentEvalResult,
    build_result,
    mismatch_reasons,
    run_known_cases,
    save_known_agent_eval,
)
from finscope.known_cases import (
    AGENT_CASES_PATH,
    COMPANY,
    FORM,
    KNOWN_CASES_PATH,
    KnownCasesError,
    build_known_diff,
    case_results,
    load_agent_cases,
    load_known_cases,
)
from finscope.labels import LABELS_DIR, LabelsError, SectionLabels, label_changes, load_labels
from finscope.llm import LLMError, LLMProvider, OpenAIProvider
from finscope.notify import (
    NotifyError,
    TelegramNotifier,
    format_alert_message,
    telegram_configured,
)
from finscope.report import HighItem, SectionReport, build_section_report
from finscope.sections import Section, split_sections
from finscope.triage import (
    SCHEMA,
    SYSTEM_PROMPT,
    TriageError,
    TriageReport,
    is_borderline,
    triage_section,
)
from finscope.verify import parse_explanation, sources_for, verify_explanation
from finscope.xbrl import (
    METRICS,
    MetricValue,
    format_amount,
    format_period,
    key_metrics,
    parse_facts,
)

LISTED_FORMS = {"10-K", "10-Q", "8-K"}
PREVIEW_LINES = 40
TITLE_WIDTH = 60
SNIPPET_CHARS = 600
TRIAGE_PREVIEW_CHARS = 100
# Triage answers for a whole section in one call: a long section needs more room than one
# explanation. This is only a ceiling (reasoning tokens included); you pay for what is used.
TRIAGE_MAX_OUTPUT_TOKENS = 16_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finscope", description="Explore SEC EDGAR filings.")
    commands = parser.add_subparsers(dest="command", required=True)

    filings_parser = commands.add_parser("filings", help="list recent 10-K, 10-Q and 8-K filings")
    filings_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")

    read_parser = commands.add_parser("read", help="download the latest filing as plain text")
    read_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    read_parser.add_argument("--form", default="10-Q", help="form type to read (default: 10-Q)")

    sections_parser = commands.add_parser("sections", help="split the latest filing into Items")
    sections_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    sections_parser.add_argument("--form", default="10-Q", help="form type (default: 10-Q)")
    sections_parser.add_argument("--item", help="print this Item's full text, e.g. 1A")
    sections_parser.add_argument("--part", help="with --item: which Part, e.g. II")

    facts_parser = commands.add_parser("facts", help="key financial numbers from XBRL data")
    facts_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    facts_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])

    diff_parser = commands.add_parser("diff", help="compare with the previous filing")
    diff_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    diff_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    diff_parser.add_argument("--item", help="show paragraph changes for this Item, e.g. 1A")
    diff_parser.add_argument("--part", help="with --item: which Part, e.g. I")

    explain_parser = commands.add_parser("explain", help="AI explanation of one diff change")
    explain_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    explain_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    explain_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    explain_parser.add_argument("--part", help="which Part, if the Item exists in several")
    explain_parser.add_argument("--kind", required=True, choices=["added", "removed", "modified"])
    explain_parser.add_argument("--n", type=int, required=True, help="change number from `diff`")
    explain_parser.add_argument(
        "--agent", action="store_true", help="let the model gather context with tools first"
    )

    triage_parser = commands.add_parser("triage", help="AI rates every change in a section")
    triage_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    triage_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    triage_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    triage_parser.add_argument("--part", help="which Part, if the Item exists in several")

    report_parser = commands.add_parser(
        "report", help="triage a section, then explain and fact-check its high changes"
    )
    report_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    report_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    report_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    report_parser.add_argument("--part", help="which Part, if the Item exists in several")
    report_parser.add_argument(
        "--max-explain",
        type=_non_negative_int,
        default=20,
        help="explain at most this many high changes, one agent run each (default 20)",
    )
    report_parser.add_argument(
        "--triage-runs",
        type=_positive_int,
        default=1,
        help="triage runs, one API call each; high in any run counts as high (default 1)",
    )

    label_parser = commands.add_parser("label", help="you rate every change (eval labels)")
    label_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    label_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    label_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    label_parser.add_argument("--part", help="which Part, if the Item exists in several")

    eval_parser = commands.add_parser("eval-triage", help="score AI triage against your labels")
    eval_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    eval_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    eval_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    eval_parser.add_argument("--part", help="which Part, if the Item exists in several")
    eval_parser.add_argument(
        "--runs", type=_positive_int, default=3, help="triage runs, one API call each (default 3)"
    )

    agent_eval_parser = commands.add_parser(
        "eval-agent", help="run the agent several times per change: does its answer hold?"
    )
    agent_eval_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    agent_eval_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    agent_eval_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    agent_eval_parser.add_argument("--part", help="which Part, if the Item exists in several")
    agent_eval_parser.add_argument(
        "--runs", type=_positive_int, default=3, help="agent runs per change (default 3)"
    )
    agent_eval_parser.add_argument(
        "--ids",
        type=_change_ids,
        help="exactly these changes, no triage, e.g. '~4,~11' (quote it: zsh expands ~4)",
    )

    known_parser = commands.add_parser(
        "eval-known", help="score AI triage on made-up changes with obvious answers"
    )
    known_parser.add_argument(
        "--runs", type=_positive_int, default=3, help="triage runs, one API call each (default 3)"
    )

    known_agent_parser = commands.add_parser(
        "eval-known-agent",
        help="score the explain agent's importance on made-up changes with obvious answers",
    )
    known_agent_parser.add_argument(
        "--runs", type=_positive_int, default=2, help="agent runs per case (default 2)"
    )

    check_parser = commands.add_parser(
        "check", help="run the report on every watched section that has a new filing"
    )
    check_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only say what would happen: no AI calls, no state changes, no files written",
    )
    check_parser.add_argument(
        "--include-unseen",
        action="store_true",
        help="on first sight of an entry, run the report instead of only remembering the start",
    )
    check_parser.add_argument(
        "--max-explain",
        type=_non_negative_int,
        default=20,
        help="per section: explain at most this many high changes (default 20)",
    )
    check_parser.add_argument(
        "--triage-runs",
        type=_positive_int,
        default=1,
        help="per section: triage runs; high in any run counts as high (default 1)",
    )
    check_parser.add_argument(
        "--no-notify", action="store_true", help="don't send alerts to Telegram"
    )

    feedback_parser = commands.add_parser(
        "feedback", help="rate a report's alerts useful, not_useful or wrong (feeds evals)"
    )
    feedback_parser.add_argument(
        "--report", help="a file name in data/reports (default: the newest without feedback)"
    )
    feedback_parser.add_argument(
        "--redo", action="store_true", help="ask again even if the report already has feedback"
    )

    commands.add_parser(
        "telegram-chat-id", help="list the chats that messaged your Telegram bot, with their IDs"
    )
    commands.add_parser("notify-test", help="send a test message to your Telegram chat")

    args = parser.parse_args(argv)

    # Needs no ticker and no SEC data: the cases are made up and stored in the repo.
    if args.command == "eval-known":
        return _eval_known(args.runs)
    if args.command == "eval-known-agent":
        return _eval_known_agent(args.runs)
    # Takes its tickers from the watchlist, not the command line.
    if args.command == "check":
        return _check(
            dry_run=args.dry_run,
            include_unseen=args.include_unseen,
            max_explain=args.max_explain,
            triage_runs=args.triage_runs,
            no_notify=args.no_notify,
        )
    # Reads saved reports only: no SEC data, no AI calls.
    if args.command == "feedback":
        return _feedback(get_settings().data_dir, report=args.report, redo=args.redo)
    if args.command == "telegram-chat-id":
        return _telegram_chat_id(get_settings())
    if args.command == "notify-test":
        return _notify_test(get_settings())

    with SecClient() as sec:
        try:
            cik = sec.get_cik(args.ticker)
        except TickerNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if args.command == "filings":
            return _list_filings(sec, cik)

        # The form filter is an exact match, so amendments (e.g. 10-K/A) are excluded.
        filings = sec.get_recent_filings(cik, forms={args.form}, limit=2)
        if not filings:
            print(f"error: no recent {args.form} filing found", file=sys.stderr)
            return 1
        if args.command in (
            "diff",
            "explain",
            "triage",
            "report",
            "label",
            "eval-triage",
            "eval-agent",
        ):
            if len(filings) < 2:
                print(f"error: need two {args.form} filings to compare", file=sys.stderr)
                return 1
            old, new = filings[1], filings[0]
            if args.command == "explain":
                return _explain(
                    sec, args.ticker, old, new, args.item, args.part, args.kind, args.n, args.agent
                )
            if args.command == "triage":
                return _triage(sec, args.ticker, old, new, args.item, args.part)
            if args.command == "report":
                return _report(
                    sec,
                    args.ticker,
                    old,
                    new,
                    args.item,
                    args.part,
                    max_explain=args.max_explain,
                    triage_runs=args.triage_runs,
                )
            if args.command == "label":
                return _label_changes(sec, args.ticker, old, new, args.item, args.part)
            if args.command == "eval-triage":
                return _eval_triage(sec, args.ticker, old, new, args.item, args.part, args.runs)
            if args.command == "eval-agent":
                return _eval_agent(
                    sec, args.ticker, old, new, args.item, args.part, args.runs, args.ids
                )
            return _show_diff(sec, old=old, new=new, item=args.item, part=args.part)
        if args.command == "read":
            return _read_filing(sec, filings[0])
        if args.command == "facts":
            return _show_facts(sec, filings[0])
        return _show_sections(sec, filings[0], args.item, args.part)


def _list_filings(sec: SecClient, cik: str) -> int:
    filings = sec.get_recent_filings(cik, forms=LISTED_FORMS, limit=10)
    print(f"{'DATE':<10}  {'FORM':<5}  {'ACCESSION NUMBER':<20}  URL")
    for filing in filings:
        print(
            f"{filing.filing_date.isoformat():<10}  {filing.form:<5}  "
            f"{filing.accession_number:<20}  {filing.url}"
        )
    return 0


def _read_filing(sec: SecClient, filing: Filing) -> int:
    # Always re-convert, so `read` also refreshes the saved text after html_to_text changes.
    text = load_filing_text(sec, filing, refresh=True)
    path = filing_path(sec.data_dir, filing, ".txt")

    lines = text.splitlines()
    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    print(f"Text:  {len(text):,} characters, {len(lines):,} lines")
    print(f"Saved: {path}")
    print()
    for line in [line for line in lines if line][:PREVIEW_LINES]:
        print(line)
    return 0


def _show_sections(sec: SecClient, filing: Filing, item: str | None, part: str | None) -> int:
    sections = split_sections(load_filing_text(sec, filing))

    if item is None:
        print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
        print()
        print(f"{'PART':<4}  {'ITEM':<5}  {'TITLE':<{TITLE_WIDTH}}  {'CHARS':>7}")
        for section in sections:
            print(
                f"{section.part or '-':<4}  {section.item:<5}  "
                f"{_shorten(section.title, TITLE_WIDTH):<{TITLE_WIDTH}}  {len(section.text):>7,}"
            )
        return 0

    chosen = _pick_section(sections, item, part)
    if chosen is None:
        return 1
    print(chosen.text)
    return 0


def _pick_section(sections: list[Section], item: str, part: str | None) -> Section | None:
    """The one section matching item (and part); prints why and returns None otherwise."""
    matches = [
        section
        for section in sections
        if section.item == item.upper() and (part is None or section.part == part.upper())
    ]
    if not matches:
        available = ", ".join(_label(section) for section in sections)
        print(f"error: Item {item} not found. Available: {available}", file=sys.stderr)
        return None
    if len(matches) > 1:
        options = " or ".join(f"--part {section.part}" for section in matches)
        print(f"error: Item {item} appears in more than one Part; add {options}", file=sys.stderr)
        return None
    return matches[0]


def _show_facts(sec: SecClient, filing: Filing) -> int:
    found = {value.name: value for value in _load_metrics(sec, filing)}

    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    if not found:
        # SEC usually updates companyfacts within minutes of a filing, but not instantly.
        print("No XBRL facts for this filing yet.")
        return 0
    print()
    print(f"{'METRIC':<16}  {'PERIOD':<24}  {'VALUE':>10}  {'PRIOR':>10}  {'CHANGE':>7}  CONCEPT")
    for metric in METRICS:
        value = found.get(metric.name)
        if value is None:
            print(f"{metric.name:<16}  not reported")
            continue
        prior = format_amount(value.prior.value, value.unit) if value.prior else "-"
        change = f"{value.change_pct:+.1f}%" if value.change_pct is not None else "-"
        print(
            f"{metric.name:<16}  {format_period(value):<24}  "
            f"{format_amount(value.current.value, value.unit):>10}  {prior:>10}  {change:>7}  "
            f"{value.concept}"
        )
    return 0


def _load_metrics(sec: SecClient, filing: Filing) -> list[MetricValue]:
    concepts = {concept for metric in METRICS for concept in metric.concepts}
    facts = parse_facts(sec.get_company_facts(filing.cik), concepts)
    return key_metrics(facts, filing.accession_number, filing.form)


def _show_diff(
    sec: SecClient, *, old: Filing, new: Filing, item: str | None, part: str | None
) -> int:
    _print_filing_pair(old, new)
    if item is not None:
        diff = _section_diff(sec, old, new, item, part)
        if diff is None:
            return 1
        _print_section_changes(diff)
        return 0

    old_sections = split_sections(load_filing_text(sec, old))
    new_sections = split_sections(load_filing_text(sec, new))
    old_by_key = {(s.part, s.item): s for s in old_sections}
    new_by_key = {(s.part, s.item): s for s in new_sections}
    print(
        f"{'PART':<4}  {'ITEM':<5}  {'TITLE':<40}  "
        f"{'UNCHANGED':>9}  {'MODIFIED':>8}  {'ADDED':>5}  {'REMOVED':>7}"
    )
    for key, new_section in new_by_key.items():
        if key not in old_by_key:
            continue
        diff = diff_sections(old_by_key[key], new_section)
        print(
            f"{diff.part or '-':<4}  {diff.item:<5}  {_shorten(diff.title, 40):<40}  "
            f"{diff.unchanged:>9}  {len(diff.of_kind('modified')):>8}  "
            f"{len(diff.of_kind('added')):>5}  {len(diff.of_kind('removed')):>7}"
        )
    only_new = [_label(s) for key, s in new_by_key.items() if key not in old_by_key]
    only_old = [_label(s) for key, s in old_by_key.items() if key not in new_by_key]
    if only_new or only_old:
        print()
        print(f"Only in the new filing: {', '.join(only_new) or '-'}")
        print(f"Only in the old filing: {', '.join(only_old) or '-'}")
    return 0


def _explain(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    kind: str,
    number: int,
    agent: bool,
) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings)  # fails fast without an API key
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    sections = _section_pair(sec, old, new, item, part)
    if sections is None:
        return 1
    old_section, new_section = sections
    diff = diff_sections(old_section, new_section)
    changes = numbered_changes(diff)[kind]
    if not 1 <= number <= len(changes):
        print(
            f"error: --n {number} is out of range; {_diff_label(diff)} has "
            f"{len(changes)} {kind} change(s)",
            file=sys.stderr,
        )
        return 1
    change = changes[number - 1]
    print(f"{_diff_label(diff)}, {kind} #{number}:")
    print(_snippet(change_text(change)))
    print()

    if agent:
        context = ExplainContext(
            change=change,
            old_section=old_section,
            new_section=new_section,
            metrics=tuple(_load_metrics(sec, new)),  # loaded now: tools never touch the network
        )
        return _explain_with_agent(context, ticker, new, _diff_label(diff), provider)

    try:
        explanation = explain_change(
            change,
            company=ticker.upper(),
            form=new.form,
            section_title=_diff_label(diff),
            provider=provider,
            settings=settings,
        )
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    result = explanation.result
    cost = "unknown (no prices configured)"
    if explanation.cost_usd is not None:
        cost = f"${explanation.cost_usd:.4f}"
    print(result.text)
    print()
    _print_fact_check(result.text, sources_for(change), expect_evidence=False)
    print()
    print(f"model:      {result.model}")
    print(f"tokens:     {result.input_tokens:,} input, {result.output_tokens:,} output")
    print(f"cost:       {cost}")
    print(f"request id: {result.request_id or '-'}")
    return 0


def _explain_with_agent(
    context: ExplainContext, ticker: str, filing: Filing, section: str, provider: OpenAIProvider
) -> int:
    settings = get_settings()
    try:
        explanation = explain_change_with_agent(
            context,
            company=ticker.upper(),
            form=filing.form,
            section_title=section,
            provider=provider,
            settings=settings,
        )
    except AgentError as error:
        _print_trace(error.trace)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    run = explanation.run
    _print_trace(run.trace)
    print()
    print(run.text)
    print()
    sources = sources_for(context.change, run.tool_outputs)
    _print_fact_check(run.text, sources, expect_evidence=True)
    print()
    cost = "unknown (no prices configured)"
    if explanation.cost_usd is not None:
        cost = f"${explanation.cost_usd:.4f}"
    print(f"steps:  {len(run.trace)}")
    print(f"tokens: {run.input_tokens:,} input, {run.output_tokens:,} output")
    print(f"cost:   {cost}")
    return 0


def _triage(
    sec: SecClient, ticker: str, old: Filing, new: Filing, item: str, part: str | None
) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings, max_output_tokens=TRIAGE_MAX_OUTPUT_TOKENS)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    diff = _section_diff(sec, old, new, item, part)
    if diff is None:
        return 1
    try:
        report = triage_section(
            diff,
            company=ticker.upper(),
            form=new.form,
            section_title=_diff_label(diff),
            provider=provider,
            settings=settings,
        )
    except (LLMError, TriageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"{_diff_label(diff)}: {len(report.items)} change(s) rated")
    _print_triage(report, changes_by_id(diff))
    return 0


def _print_triage(report: TriageReport, changes: dict[str, ParagraphChange]) -> None:
    for level in ("high", "medium"):
        items = [item for item in report.items if item.importance == level]
        print(f"\n=== {level.upper()} ({len(items)}) ===")
        for item in items:
            change = changes[item.id]
            paragraph = " ".join((change.new_text or change.old_text or "").split())
            print(f"\n[{item.id}] {item.reason}")
            print(f"     {_shorten(paragraph, TRIAGE_PREVIEW_CHARS)}")
    # LOW is shown too, one line each, so a change the model under-rated is still visible.
    low = [item for item in report.items if item.importance == "low"]
    print(f"\n=== LOW ({len(low)}) ===")
    for item in low:
        print(f"[{item.id}] {item.reason}")

    cost = "unknown (no prices configured)"
    if report.cost_usd is not None:
        cost = f"${report.cost_usd:.4f}"
    print()
    print(f"rated:  {len(report.items)} change(s)")
    print(f"model:  {report.model or '- (nothing to rate, no call made)'}")
    print(f"tokens: {report.input_tokens:,} input, {report.output_tokens:,} output")
    print(f"cost:   {cost}")


def _report(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    *,
    max_explain: int,
    triage_runs: int,
) -> int:
    settings = get_settings()
    try:
        triage_provider, agent_provider = _report_providers(settings)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    try:
        built = _build_report(
            sec,
            ticker,
            old,
            new,
            item,
            part,
            triage_provider=triage_provider,
            agent_provider=agent_provider,
            settings=settings,
            max_explain=max_explain,
            triage_runs=triage_runs,
        )
    except (LLMError, TriageError) as error:
        print(f"error: triage failed, nothing to report: {error}", file=sys.stderr)
        return 1
    if built is None:
        return 1
    report, _ = built
    _print_report(report)
    return 0


def _report_providers(settings: Settings) -> tuple[LLMProvider, LLMProvider]:
    """Triage and agent providers: the ones triage and explain --agent use.

    Each fails fast (LLMError) without an API key.
    """
    triage_provider = OpenAIProvider(settings, max_output_tokens=TRIAGE_MAX_OUTPUT_TOKENS)
    return triage_provider, OpenAIProvider(settings)


def _build_report(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    *,
    triage_provider: LLMProvider,
    agent_provider: LLMProvider,
    settings: Settings,
    max_explain: int,
    triage_runs: int,
) -> tuple[SectionReport, Decision] | None:
    """The report for one section and what to do with it; shared by `report` and `check`.

    None when the Item isn't found (the error is already printed). When every triage run
    fails, LLMError or TriageError is raised, as build_section_report does.
    """
    sections = _section_pair(sec, old, new, item, part)
    if sections is None:
        return None
    old_section, new_section = sections
    diff = diff_sections(old_section, new_section)
    report = build_section_report(
        diff,
        old_section=old_section,
        new_section=new_section,
        metrics=tuple(_load_metrics(sec, new)),  # loaded now: tools never touch the network
        company=ticker.upper(),
        form=new.form,
        section_title=_diff_label(diff),
        triage_provider=triage_provider,
        agent_provider=agent_provider,
        settings=settings,
        max_explain=max_explain,
        triage_runs=triage_runs,
    )
    return report, decide(report)


def _print_report(report: SectionReport) -> None:
    print(f"{report.company} {report.form}, {report.section_title}")
    print(
        f"{report.change_count} change(s): {report.high_count} high, "
        f"{report.medium_count} medium, {report.low_count} low"
    )
    _print_triage_runs(report)
    _print_decision(decide(report))

    print(f"\n=== HIGH ({report.high_count}), {len(report.high)} explained ===")
    for item in report.high:
        print(f"\n[{item.id}] triage: {_votes(item.triage_votes)}{item.triage_reason}")
        if item.explanation is None:
            print(f"AGENT FAILED: {item.error}")
            continue
        print(item.explanation)
        check = item.verification
        if check is None:  # an answer is always checked; never let a missing check pass
            print("FACT-CHECK: FAILED (not run)")
        elif check.passed:
            quotes = _count(check.quotes_checked, "quote")
            numbers = _count(check.numbers_checked, "number")
            print(f"FACT-CHECK: passed ({quotes}, {numbers})")
        else:
            print("FACT-CHECK: FAILED")
            for issue in check.issues:
                print(f"  {issue.kind}: {issue.message}")
    if report.not_explained:
        print(
            f"\n{len(report.not_explained)} more high change(s) not explained: "
            f"{', '.join(item.id for item in report.not_explained)} (raise --max-explain)"
        )

    print(f"\n=== MEDIUM ({report.medium_count}) ===")
    for medium in report.medium:
        print(f"[{medium.id}] {medium.triage_reason}")

    print(f"\n=== LOW ({report.low_count}) ===")

    cost = "unknown (no prices configured)"
    if report.cost_usd is not None:
        cost = f"${report.cost_usd:.4f}"
    print()
    print(f"tokens: {report.input_tokens:,} input, {report.output_tokens:,} output")
    print(f"cost:   {cost}")
    if not report.usage_complete:
        print("        (a failed model call isn't included: its token usage isn't known)")


def _print_triage_runs(report: SectionReport) -> None:
    """Only when triage ran more than once, so a single run prints exactly as before."""
    if report.triage_runs == 1:
        return
    print(
        f"triage: {report.triage_succeeded}/{report.triage_runs} runs succeeded; "
        "a change is high if any run rated it high"
    )
    for error in report.triage_errors:
        print(f"  FAILED {error}")


def _votes(votes: tuple[Importance, ...]) -> str:
    """'high in 1/3 runs | ', or nothing when there was a single vote (or none)."""
    if len(votes) <= 1:
        return ""
    return f"high in {votes.count('high')}/{len(votes)} runs | "


def _borderline(entry: DecisionEntry) -> str:
    return "(borderline) " if is_borderline(entry.triage_votes) else ""


def _print_decision(decision: Decision) -> None:
    alerts = decision.tier("alert")
    print(f"\n=== ALERT ({len(alerts)}) ===")
    for entry in alerts:
        print(f"[{entry.id}] {_borderline(entry)}{entry.what_changed}")
        print(f"{' ' * (len(entry.id) + 3)}WHY IT MATTERS: {entry.why_it_matters}")
    if not alerts:
        print("(none)")

    headings: tuple[tuple[Tier, str], ...] = (
        ("digest", "agent rated lower than triage"),
        ("review", "read these yourself"),
    )
    for tier, heading in headings:
        entries = decision.tier(tier)
        print(f"\n=== {tier.upper()} ({len(entries)}): {heading} ===")
        for entry in entries:
            print(
                f"[{entry.id}] {_borderline(entry)}{entry.reason} | triage: {entry.triage_reason}"
            )
        if not entries:
            print("(none)")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _check(
    *, dry_run: bool, include_unseen: bool, max_explain: int, triage_runs: int, no_notify: bool
) -> int:
    try:
        entries = load_watchlist(WATCHLIST_PATH)
    except WatchlistError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    settings = get_settings()
    notify = not dry_run and not no_notify
    notifier = TelegramNotifier(settings) if notify and telegram_configured(settings) else None
    try:
        with SecClient(settings) as sec:
            return _run_check(
                entries,
                sec=sec,
                settings=settings,
                make_providers=lambda: _report_providers(settings),
                include_unseen=include_unseen,
                dry_run=dry_run,
                max_explain=max_explain,
                triage_runs=triage_runs,
                notify=notify,
                notifier=notifier,
            )
    finally:
        if notifier is not None:
            notifier.close()


def _run_check(
    entries: Sequence[WatchEntry],
    *,
    sec: SecClient,
    settings: Settings,
    make_providers: Callable[[], tuple[LLMProvider, LLMProvider]],
    include_unseen: bool,
    dry_run: bool,
    max_explain: int,
    triage_runs: int,
    notify: bool,
    notifier: TelegramNotifier | None,
) -> int:
    """Check every entry; one failing entry doesn't stop the others. 1 if any failed.

    With `notify`, each saved report's alerts go to `notifier` (None: Telegram not configured).
    """
    path = state_path(sec.data_dir)
    try:
        state = load_state(path)
    except StateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    # Made on the first report only: a check with nothing new needs no API key.
    providers = cache(make_providers)

    failed = []
    for entry in entries:
        print(f"\n=== {entry.label} ===")
        try:
            seen, result = _check_entry(
                entry,
                state.get(entry.key),
                sec=sec,
                settings=settings,
                providers=providers,
                include_unseen=include_unseen,
                dry_run=dry_run,
                max_explain=max_explain,
                triage_runs=triage_runs,
            )
            if seen is not None:
                # A new dict, saved before it's kept: if the save fails, a later entry's save
                # must not write this entry's update after all.
                updated = {**state, entry.key: seen}
                save_state(updated, path)
                state = updated
            # Sent last, once the report is saved and the state updated: a failed send loses
            # nothing, the report is on disk.
            if result is not None and notify and not dry_run:
                _send_alert(result, notifier, reports_dir(sec.data_dir) / result.file_name)
        except Exception as error:  # any failure: report it, keep the old state, go on
            failed.append(entry.label)
            print(f"error: {entry.label}: {type(error).__name__}: {error}", file=sys.stderr)

    print(f"\nchecked {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}", end="")
    print(f", {len(failed)} failed: {', '.join(failed)}" if failed else ", none failed")
    if dry_run:
        print("(dry run: no AI calls, no state changes, no files written)")
    return 1 if failed else 0


def _check_entry(
    entry: WatchEntry,
    seen: SeenFiling | None,
    *,
    sec: SecClient,
    settings: Settings,
    providers: Callable[[], tuple[LLMProvider, LLMProvider]],
    include_unseen: bool,
    dry_run: bool,
    max_explain: int,
    triage_runs: int,
) -> tuple[SeenFiling | None, CheckResult | None]:
    """Check one entry. Returns what to remember now (None: leave the state as it is) and the
    saved result, if a report was run."""
    cik = sec.get_cik(entry.ticker)
    # The form filter is an exact match, so amendments (e.g. 10-K/A) are excluded.
    filings = sec.get_recent_filings(cik, forms={entry.form}, limit=2)
    plan = plan_entry(filings, seen, include_unseen=include_unseen)
    new = plan.new
    if plan.action == "unchanged":
        print(f"no new filing (newest is still {_filing_label(new)})")
        return None, None
    if plan.action == "start":
        if dry_run:
            print(f"would start watching from {_filing_label(new)}; no AI calls")
            return None, None
        print(f"now watching, starting from {_filing_label(new)}")
        return _seen(new), None

    old = plan.old
    if old is None:  # a compare plan always has both filings
        raise CheckError("nothing to compare the new filing with")
    if plan.skipped and seen is not None:
        print(
            f"note: more than one new {entry.form} since {seen.filing_date.isoformat()} "
            f"({seen.accession_number}); only the newest two are compared"
        )
    if dry_run:
        print(
            f"would compare {_filing_label(new)} with {_filing_label(old)}: "
            f"{triage_runs} triage run(s) + up to {max_explain} agent run(s)"
        )
        return None, None

    triage_provider, agent_provider = providers()
    _print_filing_pair(old, new)
    built = _build_report(
        sec,
        entry.ticker,
        old,
        new,
        entry.item,
        entry.part,
        triage_provider=triage_provider,
        agent_provider=agent_provider,
        settings=settings,
        max_explain=max_explain,
        triage_runs=triage_runs,
    )
    if built is None:
        raise CheckError("section not found (see the error above)")
    report, decision = built
    result = CheckResult(
        checked_at=datetime.now(UTC),
        entry=entry,
        old_filing=old,
        new_filing=new,
        report=report,
        decision=decision,
    )
    saved = save_check_result(result, reports_dir(sec.data_dir))
    _print_check_summary(report, decision, saved)
    # Remembered only now, after the report is saved: a failure above means a retry next time.
    return _seen(new), result


def _send_alert(result: CheckResult, notifier: TelegramNotifier | None, saved: Path) -> None:
    """Send the result's alerts, if any. A failure only warns: the report is already saved."""
    message = format_alert_message(result)
    if message is None:
        return
    if notifier is None:
        print("alerts not sent: Telegram not configured")
        return
    try:
        notifier.send(message)
    except NotifyError as error:
        print(f"warning: alerts not sent: {error} (the report is saved: {saved})", file=sys.stderr)
        return
    print("alerts sent to Telegram")


def _feedback(data_dir: Path, *, report: str | None, redo: bool) -> int:
    reports, answers = reports_dir(data_dir), feedback_dir(data_dir)
    try:
        if report is None:
            path = newest_report(reports, answers, include_answered=redo)
            if path is None:
                if any(reports.glob("*.json")):
                    print(f"nothing to rate: every report in {reports} has feedback (see --redo)")
                else:
                    print(f"nothing to rate: no reports in {reports}; run `check` first")
                return 0
        else:
            path = reports / (report if report.endswith(".json") else f"{report}.json")
        saved = answers / path.name
        if saved.exists() and not redo:
            print(
                f"error: {path.name} already has feedback: {saved} (--redo asks again)",
                file=sys.stderr,
            )
            return 1
        result = load_report(path)
    except FeedbackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    decision = result.decision
    alerts, reviews, digests = (
        decision.tier("alert"),
        decision.tier("review"),
        decision.tier("digest"),
    )
    filing = result.new_filing
    print(f"{result.entry.label}: filed {_filing_label(filing)}")
    print(f"report: {path} (checked {result.checked_at.isoformat(timespec='minutes')})")
    print(f"{len(alerts)} alert(s), {len(reviews)} to review, {len(digests)} digest\n")
    if not alerts and not reviews:
        # Marked as done with an empty file, so the next run moves on to another report.
        save_feedback((), saved)
        print(f"nothing to rate: no alerts and nothing to review. Saved (no answers): {saved}")
        return 0

    try:
        # input and print are passed here, not defaulted in feedback.py, so the command
        # reads the terminal and the tests can replace input().
        feedback = ask_feedback(result, saved, ask=input, show=print)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C or Ctrl+D: every answer is already saved
        if saved.exists():
            print(f"\nStopped. Every answer so far is saved in {saved}")
        else:
            print("\nStopped before the first answer; nothing saved")
        return 0
    counts = Counter(answer.verdict for answer in feedback)
    missed = sum(1 for answer in feedback if answer.tier == "digest")
    print(
        f"\n{len(feedback)} answer(s): {counts['useful']} useful, "
        f"{counts['not_useful']} not_useful, {counts['wrong']} wrong "
        f"(incl. {missed} missed alert{'' if missed == 1 else 's'} from the digest)"
    )
    print(f"Saved: {saved}")
    return 0


def _telegram_chat_id(settings: Settings) -> int:
    try:
        with TelegramNotifier(settings) as notifier:
            chats = notifier.chats()
    except NotifyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not chats:
        print("no chats yet: send your bot a message in Telegram, then run this again")
        return 0
    for chat_id, name in chats:
        print(f"{chat_id}  {name}")
    print("put yours in .env as FINSCOPE_TELEGRAM_CHAT_ID")
    return 0


def _notify_test(settings: Settings) -> int:
    if not telegram_configured(settings):
        print(
            "error: Telegram not configured: set FINSCOPE_TELEGRAM_BOT_TOKEN and "
            "FINSCOPE_TELEGRAM_CHAT_ID",
            file=sys.stderr,
        )
        return 1
    try:
        with TelegramNotifier(settings) as notifier:
            notifier.send("FinScope test message")
    except NotifyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("test message sent")
    return 0


def _seen(filing: Filing) -> SeenFiling:
    return SeenFiling(
        accession_number=filing.accession_number,
        filing_date=filing.filing_date,
        processed_at=datetime.now(UTC),
    )


def _print_check_summary(report: SectionReport, decision: Decision, saved: Path) -> None:
    print(
        f"{report.change_count} change(s): {report.high_count} high, "
        f"{report.medium_count} medium, {report.low_count} low"
    )
    _print_triage_runs(report)
    alerts = decision.tier("alert")
    print(f"ALERT ({len(alerts)})")
    for entry in alerts:
        print(f"  [{entry.id}] {_borderline(entry)}{entry.what_changed}")
    print(f"DIGEST ({len(decision.tier('digest'))}), REVIEW ({len(decision.tier('review'))})")
    cost = "unknown (no prices configured)"
    if report.cost_usd is not None:
        cost = f"${report.cost_usd:.4f}"
    print(f"cost:  {cost}")
    print(f"saved: {saved}")


def _filing_label(filing: Filing) -> str:
    return f"{filing.filing_date.isoformat()} ({filing.accession_number})"


def _label_changes(
    sec: SecClient, ticker: str, old: Filing, new: Filing, item: str, part: str | None
) -> int:
    _print_filing_pair(old, new)
    diff = _section_diff(sec, old, new, item, part)
    if diff is None:
        return 1
    labels = _no_labels(ticker, old, new, diff)
    path = LABELS_DIR / labels.file_name
    if path.exists():
        try:
            labels = load_labels(path, diff)
        except LabelsError as error:
            print(f"error: {path}: {error}", file=sys.stderr)
            return 1
    total = len(changes_by_id(diff))
    print(f"{_diff_label(diff)}: {total} change(s), {len(labels.labels)} already labeled\n")
    try:
        label_changes(diff, labels, LABELS_DIR)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C or Ctrl+D: every answer is already saved
        print(f"\nStopped. Every answer so far is saved in {path}")
    return 0


def _no_labels(ticker: str, old: Filing, new: Filing, diff: SectionDiff) -> SectionLabels:
    """An empty label set for this section; its file_name says where its labels are saved."""
    return SectionLabels(
        company=ticker.upper(),
        form=new.form,
        part=diff.part,
        item=diff.item,
        new_accession=new.accession_number,
        old_accession=old.accession_number,
        labels=(),
    )


def _eval_triage(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    runs: int,
) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings, max_output_tokens=TRIAGE_MAX_OUTPUT_TOKENS)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    diff = _section_diff(sec, old, new, item, part)
    if diff is None:
        return 1
    path = LABELS_DIR / _no_labels(ticker, old, new, diff).file_name
    if not path.exists():
        print(f"error: no labels at {path}; run `label` first", file=sys.stderr)
        return 1
    try:
        labels = load_labels(path, diff)
    except LabelsError as error:
        print(f"error: {path}: {error}", file=sys.stderr)
        return 1
    return _run_triage_eval(
        diff, labels, path, provider=provider, settings=settings, runs=runs, results_dir=RESULTS_DIR
    )


def _run_triage_eval(
    diff: SectionDiff,
    labels: SectionLabels,
    labels_path: Path,
    *,
    provider: LLMProvider,
    settings: Settings,
    runs: int,
    results_dir: Path,
) -> int:
    """Run triage `runs` times, score each run against the labels, print and save the result."""
    change_ids = list(changes_by_id(diff))
    human = {label.id: label.label for label in labels.labels}
    coverage = coverage_of(change_ids, human)
    print(f"{_diff_label(diff)}: {coverage.describe()}")
    if coverage.scored == 0:
        print("error: nothing to score yet; label some changes h/m/l first", file=sys.stderr)
        return 1
    print(f"Running triage {runs} time(s): {runs} API call(s).\n")

    records = _triage_runs(
        diff,
        company=labels.company,
        form=labels.form,
        provider=provider,
        settings=settings,
        runs=runs,
    )
    succeeded = [record for record in records if record.error is None]
    result = EvalResult(
        created_at=datetime.now(UTC),
        company=labels.company,
        form=labels.form,
        part=labels.part,
        item=labels.item,
        new_accession=labels.new_accession,
        old_accession=labels.old_accession,
        labels_file=labels_path.name,
        model=settings.llm_model,
        prompt_version=prompt_version(SYSTEM_PROMPT, SCHEMA),
        runs=tuple(records),
        score=score_triage(change_ids, human, _run_ratings(records)),
        input_tokens=sum(record.input_tokens for record in succeeded),
        output_tokens=sum(record.output_tokens for record in succeeded),
        cost_usd=_total_cost(succeeded),
    )
    saved = save_result(result, results_dir, labels_path.stem)
    _print_eval(result)
    print(f"saved:  {saved}")
    return 0 if succeeded else 1


def _triage_runs(
    diff: SectionDiff,
    *,
    company: str,
    form: str,
    provider: LLMProvider,
    settings: Settings,
    runs: int,
) -> list[RunRecord]:
    """Triage the same section `runs` times, one API call each; failed runs are kept."""
    records = []
    for number in range(1, runs + 1):
        try:
            report = triage_section(
                diff,
                company=company,
                form=form,
                section_title=_diff_label(diff),
                provider=provider,
                settings=settings,
            )
        except (LLMError, TriageError) as error:
            # Recorded, not fatal: a failed run says something about the model too.
            print(f"run {number}/{runs}: FAILED: {error}")
            records.append(
                RunRecord(
                    number=number,
                    error=str(error),
                    model=None,
                    items=(),
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                )
            )
            continue
        print(f"run {number}/{runs}: done")
        records.append(
            RunRecord(
                number=number,
                error=None,
                model=report.model,
                items=tuple(
                    RatedChange(id=item.id, importance=item.importance, reason=item.reason)
                    for item in report.items
                ),
                input_tokens=report.input_tokens,
                output_tokens=report.output_tokens,
                cost_usd=report.cost_usd,
            )
        )
    return records


def _run_ratings(records: list[RunRecord]) -> list[dict[str, Importance] | None]:
    """What score_triage takes: each run's ratings, None for a failed run."""
    return [record.ratings if record.error is None else None for record in records]


def _total_cost(succeeded: list[RunRecord]) -> Decimal | None:
    costs = [record.cost_usd for record in succeeded]
    return None if None in costs else sum((c for c in costs if c is not None), Decimal(0))


def _eval_agent(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    runs: int,
    ids: list[str] | None,
) -> int:
    settings = get_settings()
    try:
        triage_provider, agent_provider = _report_providers(settings)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    sections = _section_pair(sec, old, new, item, part)
    if sections is None:
        return 1
    old_section, new_section = sections
    return _run_agent_eval(
        diff_sections(old_section, new_section),
        old_section=old_section,
        new_section=new_section,
        metrics=tuple(_load_metrics(sec, new)),  # loaded now: tools never touch the network
        company=ticker.upper(),
        form=new.form,
        new_accession=new.accession_number,
        old_accession=old.accession_number,
        triage_provider=triage_provider,
        agent_provider=agent_provider,
        settings=settings,
        runs=runs,
        ids=ids,
        results_dir=RESULTS_DIR,
    )


def _run_agent_eval(
    diff: SectionDiff,
    *,
    old_section: Section,
    new_section: Section,
    metrics: tuple[MetricValue, ...],
    company: str,
    form: str,
    new_accession: str,
    old_accession: str,
    triage_provider: LLMProvider,
    agent_provider: LLMProvider,
    settings: Settings,
    runs: int,
    ids: list[str] | None,
    results_dir: Path,
) -> int:
    """Pick the changes, run the agent `runs` times on each, print and save the result."""
    section_title = _diff_label(diff)
    try:
        selection = select_changes(
            diff,
            ids=ids,
            company=company,
            form=form,
            section_title=section_title,
            triage_provider=triage_provider,
            settings=settings,
        )
    except AgentEvalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (LLMError, TriageError) as error:
        print(f"error: triage failed, nothing to evaluate: {error}", file=sys.stderr)
        return 1
    if not selection.changes:
        print("error: triage rated no change high; pick changes with --ids", file=sys.stderr)
        return 1

    picked = ", ".join(change.id for change in selection.changes)
    source = "triage high" if selection.picked_by == "triage" else "--ids"
    print(f"{section_title}: {_count(len(selection.changes), 'change')} ({source}): {picked}")
    total = len(selection.changes) * runs
    print(f"Running the agent {runs} time(s) per change: {total} agent run(s).\n")

    def progress(change_id: str, run: StabilityRun) -> None:
        print(f"[{change_id}] run {run.number}/{runs}: {run.tier} ({run.tier_reason})")

    changes = run_stability(
        selection,
        diff,
        old_section=old_section,
        new_section=new_section,
        metrics=metrics,
        company=company,
        form=form,
        section_title=section_title,
        agent_provider=agent_provider,
        settings=settings,
        runs=runs,
        on_run=progress,
    )
    summary = summarize(changes, selection, settings)
    result = AgentEvalResult(
        created_at=datetime.now(UTC),
        company=company,
        form=form,
        part=diff.part,
        item=diff.item,
        new_accession=new_accession,
        old_accession=old_accession,
        model=settings.llm_model,
        prompt_version=agent_prompt_version(),
        runs=runs,
        picked_by=selection.picked_by,
        changes=changes,
        summary=summary,
    )
    saved = save_agent_eval(result, results_dir)
    _print_agent_eval(result)
    print(f"saved:  {saved}")
    return 0 if summary.agent_failures < summary.total_runs else 1


def _print_agent_eval(result: AgentEvalResult) -> None:
    print(f"\n=== STABILITY: {result.runs} run(s) per change ===")
    rows = [(change, _importances(change), _tiers(change)) for change in result.changes]
    id_width = max(len(change.id) for change in result.changes) + 2
    importance_width = max(len(importances) for _, importances, _ in rows)
    tier_width = max(len(tiers) for _, _, tiers in rows)
    for change, importances, tiers in rows:
        print(
            f"{'[' + change.id + ']':<{id_width}} importance: {importances:<{importance_width}}"
            f"   tier: {tiers:<{tier_width}}   {'stable' if change.stable else 'UNSTABLE'}"
        )

    summary = result.summary
    print("\n=== SUMMARY ===")
    print(f"stable importance:   {summary.importance_stable}/{summary.changes} change(s)")
    print(f"tier flips:          {summary.tier_flips}/{summary.changes} change(s)")
    print(f"fact-check failures: {summary.fact_check_failures}/{summary.total_runs} run(s)")
    print(f"agent failures:      {summary.agent_failures}/{summary.total_runs} run(s)")

    cost = "unknown (no prices configured)"
    if summary.cost_usd is not None:
        cost = f"${summary.cost_usd:.4f}"
    calls = f"{summary.total_runs} agent run(s)"
    if result.picked_by == "triage":
        calls = f"triage + {calls}"
    print(f"\nmodel {result.model}, agent prompt version {result.prompt_version}")
    print(f"tokens: {summary.input_tokens:,} input, {summary.output_tokens:,} output ({calls})")
    print(f"cost:   {cost}")
    if not summary.usage_complete:
        print("        (a failed model call isn't included: its token usage isn't known)")


def _importances(change: ChangeStability) -> str:
    return ", ".join(_agent_word(run.item) for run in change.runs)


def _tiers(change: ChangeStability) -> str:
    return ", ".join(run.tier for run in change.runs)


def _eval_known(runs: int) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings, max_output_tokens=TRIAGE_MAX_OUTPUT_TOKENS)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return _run_known_eval(KNOWN_CASES_PATH, provider=provider, settings=settings, runs=runs)


def _run_known_eval(
    cases_path: Path, *, provider: LLMProvider, settings: Settings, runs: int
) -> int:
    """Triage the known cases `runs` times and print PASS/FAIL per case plus the score."""
    try:
        cases = load_known_cases(cases_path)
    except KnownCasesError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    known = build_known_diff(cases)
    print(f"{len(cases)} known case(s) from {cases_path} ({COMPANY}, a made-up company)")
    print(f"Running triage {runs} time(s): {runs} API call(s).\n")

    records = _triage_runs(
        known.diff, company=COMPANY, form=FORM, provider=provider, settings=settings, runs=runs
    )
    ratings = _run_ratings(records)
    results = case_results(known, ratings)

    print("\n=== CASES: PASS only if every run gave the expected rating ===")
    width = max(len(result.name) for result in results)
    for result in results:
        got = ", ".join(rating or "failed" for rating in result.ratings)
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.matches}/{len(result.ratings)}  "
            f"{result.name:<{width}}  expected {result.expected:<4}  got {got}"
        )
    passed = sum(result.passed for result in results)
    print(f"\npassed: {passed}/{len(results)} case(s)")

    succeeded = [record for record in records if record.error is None]
    print(f"\nmodel {settings.llm_model}, prompt version {prompt_version(SYSTEM_PROMPT, SCHEMA)}")
    score = score_triage(list(known.names), known.expected, ratings)
    # Case names instead of bare IDs: "+1" means nothing for a made-up section.
    _print_score(score, records, names={i: f"{i} {name}" for i, name in known.names.items()})
    _print_usage(
        sum(record.input_tokens for record in succeeded),
        sum(record.output_tokens for record in succeeded),
        _total_cost(succeeded),
        all_succeeded=len(succeeded) == len(records),
    )
    return 0 if succeeded else 1


def _eval_known_agent(runs: int) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings)  # the same provider explain --agent and report use
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return _run_known_agent_eval(
        AGENT_CASES_PATH, provider=provider, settings=settings, runs=runs, results_dir=RESULTS_DIR
    )


def _run_known_agent_eval(
    cases_path: Path, *, provider: LLMProvider, settings: Settings, runs: int, results_dir: Path
) -> int:
    """Explain every known case `runs` times; print PASS/FAIL per case and save the result."""
    try:
        cases = load_agent_cases(cases_path)
    except KnownCasesError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{len(cases)} known agent case(s) from {cases_path} ({COMPANY}, a made-up company)")
    print(f"Running the agent {runs} time(s) per case: {len(cases) * runs} agent run(s).\n")

    def progress(name: str, number: int, item: HighItem) -> None:
        print(f"[{name}] run {number}/{runs}: {_agent_word(item)}")

    results = run_known_cases(
        cases, provider=provider, settings=settings, runs=runs, on_run=progress
    )
    result = build_result(
        results,
        cases_file=cases_path,
        runs=runs,
        settings=settings,
        created_at=datetime.now(UTC),
    )
    saved = save_known_agent_eval(result, results_dir)
    _print_known_agent_eval(result)
    print(f"saved:  {saved}")
    return 0 if result.agent_failures < result.total_runs else 1


def _print_known_agent_eval(result: KnownAgentEvalResult) -> None:
    print("\n=== CASES: PASS only if every run gave the expected importance ===")
    name_width = max(len(case.name) for case in result.cases)
    for case in result.cases:
        print(
            f"{'PASS' if case.passed else 'FAIL'} {case.matches}/{len(case.ratings)}  "
            f"{case.name:<{name_width}} expected {case.expected:<6}  got {_agent_words(case)}"
        )
        if not case.passed:
            indent = " " * len("PASS 0/0  ")
            for reason in mismatch_reasons(case):
                print(f"{indent}reason: {reason}")

    print(f"\npassed: {result.passed}/{len(result.cases)} case(s)")
    print(f"fact-check failures: {result.fact_check_failures}/{result.total_runs} run(s)")
    print(f"agent failures:      {result.agent_failures}/{result.total_runs} run(s)")
    print(f"\nmodel {result.model}, agent prompt version {result.prompt_version}")
    _print_usage(
        result.input_tokens,
        result.output_tokens,
        result.cost_usd,
        all_succeeded=result.usage_complete,
    )


def _agent_words(case: AgentCaseResult) -> str:
    return ", ".join(_agent_word(item) for item in case.runs)


def _agent_word(item: HighItem) -> str:
    """The agent's level, "failed" (no answer) or "none" (an answer without a readable level)."""
    if item.parsed is not None and item.parsed.importance is not None:
        return item.parsed.importance
    return "failed" if item.error is not None else "none"


def _print_eval(result: EvalResult) -> None:
    print(f"\nmodel {result.model}, prompt version {result.prompt_version}")
    _print_score(result.score, result.runs)
    _print_usage(
        result.input_tokens,
        result.output_tokens,
        result.cost_usd,
        all_succeeded=all(run is not None for run in result.score.runs),
    )


def _print_score(
    score: EvalScore, records: Sequence[RunRecord], names: Mapping[str, str] | None = None
) -> None:
    """The score block shared by eval-triage and eval-known; `names` relabels change IDs."""
    print(f"coverage: {score.coverage.describe()}")

    print("\n=== RUNS ===")
    for record, run in zip(records, score.runs, strict=True):
        if run is None:
            print(f"run {record.number}: FAILED  {record.error}")
        else:
            print(
                f"run {record.number}: agreement {_pct(run.agreement)}  "
                f"high recall {_pct(run.high_recall)}  high precision {_pct(run.high_precision)}"
            )
    metrics = {
        "agreement": score.agreement,
        "high recall": score.high_recall,
        "high precision": score.high_precision,
    }
    print(
        "mean:  "
        + "  ".join(
            f"{name} {_pct(spread.mean)} ({_pct(spread.min)}-{_pct(spread.max)})"
            for name, spread in metrics.items()
        )
    )

    succeeded = sum(1 for run in score.runs if run is not None)
    print(f"\n=== CONFUSION: human rows, model columns, {succeeded} run(s) summed ===")
    print(f"{'human/model':<12}" + "".join(f"{level:>8}" for level in LEVELS))
    for row in LEVELS:
        print(f"{row:<12}" + "".join(f"{score.confusion[row][col]:>8}" for col in LEVELS))

    print(f"\n=== CRITICAL MISSES: human high, model low ({len(score.critical_misses)}) ===")
    for change in score.critical_misses:
        print(_ratings_line(change, names))
    print(f"\n=== UNSTABLE: runs disagree ({len(score.unstable)}) ===")
    for change in score.unstable:
        print(_ratings_line(change, names))


def _print_usage(
    input_tokens: int, output_tokens: int, cost_usd: Decimal | None, *, all_succeeded: bool
) -> None:
    cost = "unknown (no prices configured)"
    if cost_usd is not None:
        cost = f"${cost_usd:.4f}"
    print()
    print(f"tokens: {input_tokens:,} input, {output_tokens:,} output")
    print(f"cost:   {cost}")
    if not all_succeeded:
        print("        (failed runs aren't included: their token usage isn't known)")


def _ratings_line(change: ChangeRatings, names: Mapping[str, str] | None = None) -> str:
    ratings = ", ".join(rating or "failed" for rating in change.ratings)
    label = names.get(change.id, change.id) if names else change.id
    return f"[{label}] {ratings}   (human: {change.human or 'unlabeled'})"


def _pct(share: float | None) -> str:
    return "n/a" if share is None else f"{share:.0%}"


def _positive_int(text: str) -> int:
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _change_ids(text: str) -> list[str]:
    ids = [part.strip() for part in text.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("give change IDs from `diff`, e.g. '~4,+2'")
    return ids


def _non_negative_int(text: str) -> int:
    number = int(text)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def _print_fact_check(answer: str, sources: list[str], *, expect_evidence: bool) -> None:
    """Check the answer's quotes and numbers against what the model saw. Shown, never fatal."""
    check = verify_explanation(parse_explanation(answer, expect_evidence=expect_evidence), sources)
    quotes = f"{check.quotes_checked} quote{'' if check.quotes_checked == 1 else 's'}"
    numbers = f"{check.numbers_checked} number{'' if check.numbers_checked == 1 else 's'}"
    print(f"FACT-CHECK: {'PASSED' if check.passed else 'FAILED'} ({quotes}, {numbers} checked)")
    for issue in check.issues:
        print(f"  {issue.kind}: {issue.message}")


def _print_trace(trace: tuple[TraceStep, ...]) -> None:
    print("TRACE")
    for step in trace:
        calls = "; ".join(
            f"{call.name} {call.arguments} -> "
            + (f"ERROR: {call.error}" if call.is_error else f"{call.result_chars:,} chars")
            for call in step.tool_calls
        )
        action = calls if step.tool_calls else "answer"
        print(f"  {step.number}  {action}  ({step.input_tokens:,} in / {step.output_tokens:,} out)")


def _print_filing_pair(old: Filing, new: Filing) -> None:
    print(f"{new.form}  new: {new.filing_date.isoformat()} ({new.accession_number})")
    print(f"{' ' * len(new.form)}  old: {old.filing_date.isoformat()} ({old.accession_number})")
    print()


def _section_diff(
    sec: SecClient, old: Filing, new: Filing, item: str, part: str | None
) -> SectionDiff | None:
    sections = _section_pair(sec, old, new, item, part)
    return None if sections is None else diff_sections(*sections)


def _section_pair(
    sec: SecClient, old: Filing, new: Filing, item: str, part: str | None
) -> tuple[Section, Section] | None:
    """The same Item in the old and the new filing (errors are printed by _pick_section)."""
    old_section = _pick_section(split_sections(load_filing_text(sec, old)), item, part)
    new_section = _pick_section(split_sections(load_filing_text(sec, new)), item, part)
    if old_section is None or new_section is None:
        return None
    return old_section, new_section


def _print_section_changes(diff: SectionDiff) -> None:
    numbered = numbered_changes(diff)
    added, removed, modified = numbered["added"], numbered["removed"], numbered["modified"]
    print(
        f"{_diff_label(diff)}: {diff.unchanged} unchanged, {len(modified)} modified, "
        f"{len(added)} added, {len(removed)} removed"
    )

    print(f"\n=== ADDED ({len(added)}) ===")
    for n, change in enumerate(added, start=1):
        print(f"\n[{change_id('added', n)}] new @ {change.new_start:,}")
        print(_snippet(change_text(change)))

    print(f"\n=== REMOVED ({len(removed)}) ===")
    for n, change in enumerate(removed, start=1):
        print(f"\n[{change_id('removed', n)}] old @ {change.old_start:,}")
        print(_snippet(change_text(change)))

    print(f"\n=== MODIFIED ({len(modified)}), least similar first ===")
    for n, change in enumerate(modified, start=1):
        print(
            f"\n[{change_id('modified', n)}] {change.similarity or 0:.2f} similar  "
            f"old @ {change.old_start:,}  new @ {change.new_start:,}"
        )
        print(_snippet(change_text(change)))


def _diff_label(diff: SectionDiff) -> str:
    label = f"Part {diff.part} Item {diff.item}" if diff.part else f"Item {diff.item}"
    return f"{label}. {diff.title}"


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …"


def _label(section: Section) -> str:
    return f"{section.part} {section.item}" if section.part else section.item


def _shorten(title: str, width: int) -> str:
    return title if len(title) <= width else title[: width - 1] + "…"


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.stdout.flush()  # surface a closed pipe here, not during interpreter shutdown
    except BrokenPipeError:
        # The reader stopped early (e.g. `| head`). Point stdout at devnull so the flush at
        # exit doesn't raise again; recipe from the Python docs (signal module, SIGPIPE).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        exit_code = 1
    raise SystemExit(exit_code)
