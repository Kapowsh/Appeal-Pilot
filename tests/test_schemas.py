import pytest
from pydantic import ValidationError

from app.schemas import Case, ChartLine, CriterionMatch, Denial, ReviewFlag


def test_match_rejects_bad_status():
    with pytest.raises(ValidationError):
        CriterionMatch(criterion_id="C1", status="maybe", evidence_line_ids=[], rationale="")


def test_flag_action_literal():
    with pytest.raises(ValidationError):
        ReviewFlag(sentence_id="S1", issue="x", action="delete")


def test_case_defaults_and_roundtrip():
    c = Case(id="a", denial_text="d", chart_lines=[ChartLine(id="L1", text="t")])
    assert c.status == "new" and c.flag_resolutions == {} and c.usage == []
    assert Case.model_validate_json(c.model_dump_json()) == c


def test_denial_optional_fields():
    d = Denial(payer="Acme Health", member_id=None, patient_name="Jo Doe", service="MRI",
               cpt_or_hcpcs=None, icd10=["M54.16"], denial_reason="r", denial_date=None, appeal_deadline=None)
    assert d.icd10 == ["M54.16"]
