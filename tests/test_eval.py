from eval.run_eval import (catch_rate, citation_validity, criterion_scores, extraction_accuracy, inject_fabrications,
                           to_markdown)
from app.checks import deterministic_checks
from app.schemas import AppealDraft, ChartLine, Criterion, CriterionMatch, Denial, LetterSentence, ReviewFlag

EXPECTED = {"payer": "Acme Health", "member_id": "AH-1", "patient_name": "Jo Doe", "service": "MRI lumbar spine",
            "cpt_or_hcpcs": "72148", "icd10": ["M54.16", "M51.26"],
            "denial_reason": "Documentation does not show failure of at least 6 weeks of conservative therapy.",
            "denial_date": "2026-09-18", "appeal_deadline": "2026-11-17"}
LINES = [ChartLine(id="L1", text="SYNTHETIC RECORD - FOR DEMONSTRATION ONLY"),
         ChartLine(id="L2", text="Low back pain for 8 weeks.")]


def denial(**over):
    return Denial(**{**EXPECTED, **over})


def test_extraction_accuracy_normalizes():
    pred = denial(payer="ACME health", icd10=["m51.26", "M54.16"], service="MRI of the lumbar spine",
                  denial_reason="Failure of at least 6 weeks of conservative therapy is not documented.")
    assert extraction_accuracy(pred, EXPECTED) == 1.0
    assert extraction_accuracy(denial(member_id="AH-2"), EXPECTED) == 8 / 9


def test_extraction_free_text_mismatch_counts_wrong():
    assert extraction_accuracy(denial(service="Knee arthroscopy"), EXPECTED) == 8 / 9


def test_criterion_scores():
    criteria = [Criterion(id="C1", text="a", source_doc="X", source_section="2.1")]
    matches = [CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L1", "L5"], rationale="")]
    expected = [{"section": "2.1", "status": "met", "evidence": ["L1"]},
                {"section": "2.2", "status": "met", "evidence": ["L2"]}]
    assert criterion_scores(criteria, matches, expected) == {"status_acc": 0.5, "evidence_precision": 0.5}


def test_criterion_scores_honor_accept():
    criteria = [Criterion(id="C1", text="a", source_doc="X", source_section="2.2")]
    matches = [CriterionMatch(criterion_id="C1", status="not_met", evidence_line_ids=[], rationale="")]
    expected = [{"section": "2.2", "status": "unclear", "evidence": [], "accept": ["unclear", "not_met"]}]
    assert criterion_scores(criteria, matches, expected) == {"status_acc": 1.0, "evidence_precision": 1.0}


def test_citation_validity():
    draft = AppealDraft(sentences=[LetterSentence(id="S1", text="Dear Committee,", cites=[]),
                                   LetterSentence(id="S2", text="Pain for 8 weeks.", cites=["L2"]),
                                   LetterSentence(id="S3", text="Pain for 9 weeks.", cites=["L2"])])
    assert citation_validity(draft, LINES, [], "") == 0.5


def test_injected_date_caught_by_checks():
    draft, ids = inject_fabrications(AppealDraft(sentences=[]), LINES)
    assert ids == ["F1", "F2"]
    assert "F1" in {f.sentence_id for f in deterministic_checks(draft, LINES, [])}


def test_catch_rate():
    assert catch_rate([ReviewFlag(sentence_id="F1", issue="x", action="remove")], ["F1", "F2"]) == 0.5


def test_to_markdown_has_mean_row():
    row = {"case": "case_01", "extraction": 1.0, "status_acc": 0.75, "evidence_precision": 0.5,
           "citation_validity": 1.0, "catch_rate": 1.0, "latency_s": 12.3, "tokens": 9000}
    md = to_markdown([row, {**row, "case": "case_02", "tokens": 11000}])
    assert "| Case |" in md and "| case_01 |" in md
    assert "| **Mean** |" in md and "10000" in md


async def test_run_case_end_to_end_offline():
    from pathlib import Path
    from app.config import Settings
    from app.retrieval import PolicyIndex
    from app.schemas import CriteriaList, MatchList, Review
    from eval.run_eval import run_case
    from tests.fakes import ScriptedLLM
    llm = ScriptedLLM({
        Denial: denial(),
        CriteriaList: CriteriaList(criteria=[Criterion(id="x", text="a", source_doc="AH-RAD-012", source_section="2.1")]),
        MatchList: MatchList(matches=[CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L5"], rationale="")]),
        AppealDraft: AppealDraft(sentences=[LetterSentence(id="a", text="Pain has persisted for 7 weeks.", cites=["L5"])]),
        Review: Review(flags=[], summary="ok"),
    })
    s = Settings(_env_file=None, model_fast="f", model_reason="r")
    row = await run_case(Path("data/cases/case_01"), s, llm, PolicyIndex.build(s, None))
    assert row["case"] == "case_01" and "error" not in row
    assert row["status_acc"] == 0.25 and row["evidence_precision"] == 1.0
    assert row["citation_validity"] == 1.0 and row["catch_rate"] == 0.5
