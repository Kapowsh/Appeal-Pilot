import pytest

from app.agents import intake, matcher, policy, reviewer, writer
from app.config import Settings
from app.retrieval import PolicyIndex
from app.schemas import (AppealDraft, ChartLine, CriteriaList, Criterion, CriterionMatch, Denial, LetterSentence,
                         MatchList, Review)
from app.tools import tavily_search
from tests.fakes import ScriptedLLM

S = Settings(_env_file=None, model_fast="fast-model", model_reason="reason-model")
DENIAL = Denial(payer="Acme Health", member_id="AH-1", patient_name="Jo Doe", service="MRI lumbar spine",
                cpt_or_hcpcs="72148", icd10=["M54.16"], denial_reason="Insufficient conservative therapy",
                denial_date=None, appeal_deadline=None)
LINES = [ChartLine(id="L1", text="Back pain 8 weeks."), ChartLine(id="L2", text="PT 6 weeks."),
         ChartLine(id="L3", text="Positive SLR left.")]
CRITERIA = [Criterion(id="C1", text="6 weeks pain", source_doc="AH-RAD-012", source_section="2.1"),
            Criterion(id="C2", text="6 weeks therapy", source_doc="AH-RAD-012", source_section="2.2")]


def test_intake_uses_fast_model():
    llm = ScriptedLLM({Denial: DENIAL})
    assert intake.run(llm, S, "DENIAL TEXT HERE") == DENIAL
    assert llm.calls[0]["model"] == "fast-model" and "DENIAL TEXT HERE" in llm.calls[0]["user"]


def test_policy_query_and_renumber():
    index = PolicyIndex.build(Settings(_env_file=None), llm=None)
    llm = ScriptedLLM({CriteriaList: CriteriaList(criteria=[
        Criterion(id="2.1", text="a", source_doc="AH-RAD-012", source_section="Section 2.1"),
        Criterion(id="2.2", text="b", source_doc="", source_section="2.2")])})
    criteria, refs = policy.run(llm, S, index, DENIAL)
    assert [c.id for c in criteria] == ["C1", "C2"]
    assert [c.source_section for c in criteria] == ["2.1", "2.2"]
    assert criteria[1].source_doc == "AH-RAD-012"
    assert "AH-RAD-012" in llm.calls[0]["user"] and llm.calls[0]["model"] == "fast-model"
    assert refs == []


def test_policy_tavily_kept_separate():
    index = PolicyIndex.build(Settings(_env_file=None), llm=None)
    llm = ScriptedLLM({CriteriaList: CriteriaList(criteria=[
        Criterion(id="2.1", text="a", source_doc="AH-RAD-012", source_section="2.1")])})
    hits = [{"title": "T", "url": "u", "snippet": "s"}]
    criteria, refs = policy.run(llm, S, index, DENIAL, tavily=lambda q: hits)
    assert refs == hits
    assert all("T" not in c.text for c in criteria)
    assert "u" not in llm.calls[0]["user"].split()


def test_matcher_drops_bad_ids_and_downgrades():
    llm = ScriptedLLM({MatchList: MatchList(matches=[
        CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=[" l2", "L99"], rationale="r"),
        CriterionMatch(criterion_id="c2 ", status="met", evidence_line_ids=["L99"], rationale="r")])})
    m = matcher.run(llm, S, CRITERIA, LINES)
    assert (m[0].status, m[0].evidence_line_ids) == ("met", ["L2"])
    assert (m[1].criterion_id, m[1].status, m[1].evidence_line_ids) == ("C2", "unclear", [])
    assert llm.calls[0]["model"] == "reason-model" and "L1: Back pain 8 weeks." in llm.calls[0]["user"]


def test_matcher_fills_missing():
    llm = ScriptedLLM({MatchList: MatchList(matches=[
        CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L1"], rationale="r")])})
    m = matcher.run(llm, S, CRITERIA, LINES)
    assert [x.criterion_id for x in m] == ["C1", "C2"]
    assert m[1].status == "unclear" and m[1].rationale == "No determination returned."


def test_writer_rules_and_renumber():
    draft = AppealDraft(sentences=[LetterSentence(id="a", text="Dear Committee,", cites=[]),
                                   LetterSentence(id="b", text="Pain for 8 weeks.", cites=["L1"])])
    llm = ScriptedLLM({AppealDraft: draft})
    matches = [CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L1"], rationale="r"),
               CriterionMatch(criterion_id="C2", status="unclear", evidence_line_ids=[], rationale="r")]
    out = writer.run(llm, S, DENIAL, CRITERIA, matches, LINES)
    assert "peer-to-peer" in writer.SYSTEM
    assert llm.calls[0]["model"] == "reason-model"
    assert [s.id for s in out.sentences] == ["S1", "S2"]


def test_reviewer_gets_no_writer_reasoning():
    draft = AppealDraft(sentences=[LetterSentence(id="S1", text="Pain for 8 weeks.", cites=["L1"])])
    llm = ScriptedLLM({Review: Review(flags=[], summary="ok")})
    reviewer.run(llm, S, draft, LINES, CRITERIA)
    call = llm.calls[-1]
    assert "SECRET-RATIONALE" not in call["system"] + call["user"]
    assert call["model"] == "reason-model" and "Pain for 8 weeks." in call["user"]


def test_reviewer_drops_unknown_sentence_flags():
    from app.schemas import ReviewFlag
    draft = AppealDraft(sentences=[LetterSentence(id="S1", text="x", cites=[])])
    llm = ScriptedLLM({Review: Review(flags=[ReviewFlag(sentence_id="S9", issue="i", action="remove"),
                                             ReviewFlag(sentence_id="s1", issue="i", action="revise")], summary="s")})
    assert [f.sentence_id for f in reviewer.run(llm, S, draft, LINES, CRITERIA).flags] == ["S1"]


def test_tavily_search_swallows_errors(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("down")
    monkeypatch.setattr(tavily_search, "TavilyClient", Boom)
    assert tavily_search.search("q", "key") == []
