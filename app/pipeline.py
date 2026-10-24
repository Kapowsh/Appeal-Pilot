"""Runs the agents in order and yields {"step", "status", "data"} progress events for SSE."""
import asyncio
import logging
from typing import AsyncIterator

from app.agents import intake, matcher, policy, reviewer, writer
from app.checks import deterministic_checks, merge_flags
from app.config import Settings
from app.retrieval import PolicyIndex
from app.schemas import Case, Review
from app.tools import tavily_search

log = logging.getLogger(__name__)

STEPS = ["preprocess", "intake", "policy", "match", "draft", "review", "checks"]


def _dump(items) -> list[dict]:
    return [i.model_dump() for i in items]


async def run_pipeline(case: Case, llm, settings: Settings, index: PolicyIndex) -> AsyncIterator[dict]:
    step = STEPS[0]
    case.status = "running"
    try:
        tavily = None
        if settings.enable_tavily and settings.tavily_api_key:
            tavily = lambda q: tavily_search.search(q, settings.tavily_api_key)  # noqa: E731

        async def stage(name, fn, *args):
            nonlocal step
            step = name
            result = await asyncio.to_thread(fn, *args)
            case.usage = list(llm.usage)
            return result

        # 1. Preprocess: text extraction and line splitting happen at upload; report the citable lines.
        yield {"step": "preprocess", "status": "running", "data": None}
        yield {"step": "preprocess", "status": "done", "data": {"chart_lines": _dump(case.chart_lines)}}

        yield {"step": "intake", "status": "running", "data": None}
        case.denial = await stage("intake", intake.run, llm, settings, case.denial_text)
        yield {"step": "intake", "status": "done", "data": case.denial.model_dump()}

        yield {"step": "policy", "status": "running", "data": None}
        case.criteria, case.external_refs = await stage("policy", policy.run, llm, settings, index, case.denial, tavily,
                                                        case.denial_text)
        if not case.criteria:
            raise ValueError(f"No matching coverage criteria found for {case.denial.payer} / {case.denial.service}.")
        yield {"step": "policy", "status": "done",
               "data": {"criteria": _dump(case.criteria), "external_refs": case.external_refs}}

        yield {"step": "match", "status": "running", "data": None}
        case.matches = await stage("match", matcher.run, llm, settings, case.criteria, case.chart_lines)
        yield {"step": "match", "status": "done", "data": {"matches": _dump(case.matches)}}

        yield {"step": "draft", "status": "running", "data": None}
        case.draft = await stage("draft", writer.run, llm, settings, case.denial, case.criteria, case.matches,
                                 case.chart_lines)
        yield {"step": "draft", "status": "done", "data": case.draft.model_dump()}

        yield {"step": "review", "status": "running", "data": None}
        review = await stage("review", reviewer.run, llm, settings, case.draft, case.chart_lines, case.criteria)
        yield {"step": "review", "status": "done", "data": review.model_dump()}

        step = "checks"
        yield {"step": "checks", "status": "running", "data": None}
        det = deterministic_checks(case.draft, case.chart_lines, case.criteria, case.denial_text)
        case.review = Review(flags=merge_flags(review.flags, det), summary=review.summary)
        yield {"step": "checks", "status": "done", "data": case.review.model_dump()}

        case.status = "needs_review"
        yield {"step": "complete", "status": "needs_review", "data": {"case_id": case.id}}
    except (asyncio.CancelledError, GeneratorExit):
        case.status = "error"  # client went away mid-run: leave the case re-runnable, not stuck in "running"
        raise
    except Exception as e:
        log.exception("Pipeline failed at %s", step)
        case.status = "error"
        yield {"step": step, "status": "error", "data": {"message": str(e) or e.__class__.__name__}}
