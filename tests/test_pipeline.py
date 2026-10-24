from app.config import Settings
from app.llm import LLMError
from app.pipeline import STEPS, run_pipeline
from app.retrieval import PolicyIndex
from app.schemas import (AppealDraft, Case, ChartLine, CriteriaList, Criterion, CriterionMatch, Denial,
                         LetterSentence, MatchList, Review)
from tests.fakes import ScriptedLLM

S = Settings(_env_file=None, model_fast="fast", model_reason="reason")
DENIAL = Denial(payer="Acme Health", member_id="AH-1", patient_name="Jo Doe", service="MRI lumbar spine",
                cpt_or_hcpcs="72148", icd10=["M54.16"], denial_reason="Insufficient conservative therapy",
                denial_date=None, appeal_deadline=None)
LINES = [ChartLine(id="L1", text="Low back pain for 8 weeks."), ChartLine(id="L2", text="Completed 6 weeks of PT.")]


def script(over=None):
    by_schema = {
        Denial: DENIAL,
        CriteriaList: CriteriaList(criteria=[
            Criterion(id="2.1", text="Pain for 6 weeks", source_doc="AH-RAD-012", source_section="2.1"),
            Criterion(id="2.2", text="6 weeks therapy", source_doc="AH-RAD-012", source_section="2.2")]),
        MatchList: MatchList(matches=[
            CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L1"], rationale="r"),
            CriterionMatch(criterion_id="C2", status="met", evidence_line_ids=["L2"], rationale="r")]),
        AppealDraft: AppealDraft(sentences=[
            LetterSentence(id="x", text="Dear Appeals Committee,", cites=[]),
            LetterSentence(id="y", text="The patient has had low back pain for 8 weeks.", cites=["L1"])]),
        Review: Review(flags=[], summary="ok"),
    }
    by_schema.update(over or {})
    return ScriptedLLM(by_schema)


def index():
    return PolicyIndex.build(Settings(_env_file=None), llm=None)


def case():
    return Case(id="c1", denial_text="denial", chart_lines=LINES)


async def collect(c, llm):
    return [e async for e in run_pipeline(c, llm, S, index())]


async def test_event_order():
    events = await collect(case(), script())
    expected = [(s, st) for s in STEPS for st in ("running", "done")] + [("complete", "needs_review")]
    assert STEPS == ["preprocess", "intake", "policy", "match", "draft", "review", "checks"]
    assert [(e["step"], e["status"]) for e in events] == expected


async def test_case_populated():
    c = case()
    await collect(c, script())
    assert c.status == "needs_review"
    assert c.denial and c.criteria and c.matches and c.draft and c.review is not None
    assert len(c.usage) >= 5


async def test_deterministic_flags_merged():
    c = case()
    bad = AppealDraft(sentences=[LetterSentence(id="x", text="Pain was confirmed.", cites=["L99"])])
    await collect(c, script({AppealDraft: bad}))
    assert [f.sentence_id for f in c.review.flags] == ["S1"]


async def test_error_event():
    c = case()
    events = await collect(c, script({Denial: LLMError("boom")}))
    assert events[-1] == {"step": "intake", "status": "error", "data": {"message": "boom"}}
    assert c.status == "error"


async def test_no_criteria_error():
    c = case()
    events = await collect(c, script({CriteriaList: CriteriaList(criteria=[])}))
    assert events[-1]["status"] == "error"
    assert "No matching coverage criteria" in events[-1]["data"]["message"]
    assert c.status == "error"
