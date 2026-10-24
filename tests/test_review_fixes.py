"""Regression tests for the final whole-branch review findings."""
import asyncio
from pathlib import Path

import pytest

from app.agents import policy, writer
from app.checks import deterministic_checks, is_boilerplate
from app.config import Settings
import app.main as main
from app.pipeline import run_pipeline
from app.retrieval import PolicyIndex
from app.schemas import (AppealDraft, Case, ChartLine, CriteriaList, Criterion, CriterionMatch, Denial, LetterSentence)
from eval.run_eval import inject_fabrications
from tests.fakes import ScriptedLLM

S = Settings(_env_file=None, model_fast="fast", model_reason="reason")
INDEX = PolicyIndex.build(Settings(_env_file=None), llm=None)
LINES = [ChartLine(id="L1", text="Completed physical therapy for 6 weeks."),
         ChartLine(id="L2", text="Pain for 8 weeks.")]
CRITERIA = [Criterion(id="C1", text="At least 6 weeks of conservative therapy", source_doc="AH-RAD-012",
                      source_section="2.2")]


def denial(**over):
    base = dict(payer="Acme Health", member_id="AH-1", patient_name="Jo Doe", service="MRI lumbar spine",
                cpt_or_hcpcs="72148", icd10=["M54.16"], denial_reason="Insufficient conservative therapy",
                denial_date=None, appeal_deadline=None)
    return Denial(**{**base, **over})


def check(text, cites, denial_text=""):
    draft = AppealDraft(sentences=[LetterSentence(id="S1", text=text, cites=cites)])
    return deterministic_checks(draft, LINES, CRITERIA, denial_text)


# Important 1: wrong-policy criteria ----------------------------------------------------------

def test_unknown_payer_returns_no_chunks():
    assert INDEX.search("Initech Health bariatric surgery", k=4, payer="Initech Health") == []


def test_payer_name_variant_still_filters():
    assert {c.payer for c in INDEX.search("MRI lumbar spine", k=4, payer="Acme Health Plan")} == {"Acme Health"}


def test_bm25_drops_zero_score_chunks():
    assert INDEX.search("zzqx wqvv", k=4, payer="Acme Health") == []


def test_policy_restricted_to_cited_policy_id():
    llm = ScriptedLLM({CriteriaList: CriteriaList(criteria=[
        Criterion(id="2.1", text="a", source_doc="AH-SURG-022", source_section="2.1")])})
    policy.run(llm, S, INDEX, denial(service="Bariatric surgery"),
               denial_text="denied under Acme Health Medical Policy AH-SURG-022.")
    user = llm.calls[0]["user"]
    assert "source_doc=AH-SURG-022" in user and "source_doc=AH-RAD-012" not in user


def test_policy_drops_criteria_from_other_docs():
    llm = ScriptedLLM({CriteriaList: CriteriaList(criteria=[
        Criterion(id="2.1", text="real", source_doc="AH-RAD-012", source_section="2.1"),
        Criterion(id="9.9", text="invented", source_doc="FAKE-001", source_section="9.9")])})
    criteria, _ = policy.run(llm, S, INDEX, denial(), denial_text="Medical Policy AH-RAD-012")
    assert [c.text for c in criteria] == ["real"]


def test_policy_keeps_single_policy():
    llm = ScriptedLLM({CriteriaList: CriteriaList(criteria=[
        Criterion(id="2.1", text="a", source_doc="AH-RAD-012", source_section="2.1"),
        Criterion(id="2.2", text="b", source_doc="AH-RAD-012", source_section="2.2"),
        Criterion(id="2.1", text="other", source_doc="AH-ORTH-031", source_section="2.1")])})
    criteria, _ = policy.run(llm, S, INDEX, denial(denial_reason="knee and back"))
    assert {c.source_doc for c in criteria} == {"AH-RAD-012"}


def test_policy_no_chunks_skips_llm():
    llm = ScriptedLLM({})
    criteria, _ = policy.run(llm, S, INDEX, denial(payer="Initech Health"))
    assert criteria == [] and llm.calls == []
    assert "empty list" in policy.SYSTEM


# Important 2: disconnect leaves the case "running" ---------------------------------------------

def _case():
    return Case(id="c", denial_text="d", chart_lines=LINES)


async def test_pipeline_aclose_midrun_sets_error():
    c = _case()
    gen = run_pipeline(c, ScriptedLLM({Denial: denial()}), S, INDEX)
    for _ in range(3):
        await gen.__anext__()
    await gen.aclose()
    assert c.status == "error"


async def test_stream_disconnect_sets_error_and_saves():
    c = _case()
    c.status = "running"
    saved = []

    async def source():
        yield {"step": "intake", "status": "running", "data": None}
        await asyncio.sleep(10)

    gen = main.stream_events(source(), c, saved.append, heartbeat_s=5)
    assert (await gen.__anext__()).startswith("data: ")
    await gen.aclose()
    assert c.status == "error" and saved == [c]


async def test_stream_sends_heartbeat_while_idle():
    c = _case()

    async def source():
        await asyncio.sleep(0.2)
        yield {"step": "complete", "status": "needs_review", "data": {}}

    out = [chunk async for chunk in main.stream_events(source(), c, lambda _: None, heartbeat_s=0.05)]
    assert out[0] == ": ping\n\n" and out[-1].startswith("data: ")


# Important 3: "Re:" subject lines ------------------------------------------------------------

def test_re_subject_line_is_boilerplate():
    assert is_boilerplate("Re: Appeal of prior authorization denial for MRI lumbar spine")
    assert is_boilerplate("cc: Medical records department, Lakeside Family Medicine")


# Important 4: false-positive flags -----------------------------------------------------------

def test_inline_id_markers_not_numbers():
    assert check("The patient completed physical therapy for 6 weeks (L1).", ["L1"]) == []


def test_criterion_section_reference_not_a_number():
    assert check("This satisfies criterion 2.2 because therapy lasted 6 weeks.", ["L1"]) == []


def test_denial_pseudo_id_citable():
    denial_text = "Member ID: AH-48213907. Date: September 18, 2026."
    assert check("This appeals the denial dated September 18, 2026 for member AH-48213907.", ["DENIAL"],
                 denial_text) == []
    assert check("This appeals the denial dated September 19, 2026.", ["DENIAL"], denial_text)


def test_writer_prompt_mentions_denial_id_and_no_inline_ids():
    assert '"DENIAL"' in writer.SYSTEM and "inside the sentence text" in writer.SYSTEM


# Important 5: writer cites normalized --------------------------------------------------------

def test_writer_normalizes_cites():
    llm = ScriptedLLM({AppealDraft: AppealDraft(sentences=[
        LetterSentence(id="a", text="Pain for 8 weeks.", cites=[" l2", "L2", "c1 ", " "])])})
    matches = [CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L2"], rationale="")]
    out = writer.run(llm, S, denial(), CRITERIA, matches, LINES)
    assert out.sentences[0].cites == ["L2", "C1"]


# Important 6: X-Forwarded-For spoofing --------------------------------------------------------

class _Req:
    def __init__(self, xff, host="10.0.0.9"):
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client = type("C", (), {"host": host})()


def test_xff_ignored_without_trusted_proxy():
    assert main._client_ip(_Req("1.2.3.4"), trust_proxy=False) == "10.0.0.9"


def test_xff_rightmost_hop_with_trusted_proxy():
    assert main._client_ip(_Req("6.6.6.6, 203.0.113.7"), trust_proxy=True) == "203.0.113.7"


# Re-graded: eval fabrications cite a real chart line ------------------------------------------

def test_fabrications_do_not_cite_banner_line():
    lines = [ChartLine(id="L1", text="SYNTHETIC RECORD - FOR DEMONSTRATION ONLY")] + \
            [ChartLine(id=f"L{i}", text=f"Clinical line {i}.") for i in range(2, 21)]
    draft, _ = inject_fabrications(AppealDraft(sentences=[]), lines)
    assert all(s.cites != ["L1"] for s in draft.sentences)


# Re-graded: external reference links limited to http(s) ---------------------------------------

def test_external_links_scheme_checked():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "/^https?:\\/\\//i.test(" in html


def test_iso_date_matches_spelled_out_denial_date():
    assert check("The denial is dated 2026-09-18.", ["DENIAL"], "Date: September 18, 2026") == []
