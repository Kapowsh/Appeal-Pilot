import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import (AppealDraft, CriteriaList, Criterion, CriterionMatch, Denial, LetterSentence, MatchList,
                         Review)
from tests.fakes import ScriptedLLM

DENIAL = Denial(payer="Acme Health", member_id="AH-1", patient_name="Jo Doe", service="MRI lumbar spine",
                cpt_or_hcpcs="72148", icd10=["M54.16"], denial_reason="Insufficient conservative therapy",
                denial_date=None, appeal_deadline=None)
S2 = "The patient has had low back pain for 8 weeks."
S3 = "An MRI in 2019 was normal."


def scripted():
    return ScriptedLLM({
        Denial: DENIAL,
        CriteriaList: CriteriaList(criteria=[
            Criterion(id="2.1", text="Pain for 6 weeks", source_doc="AH-RAD-012", source_section="2.1")]),
        MatchList: MatchList(matches=[
            CriterionMatch(criterion_id="C1", status="met", evidence_line_ids=["L1"], rationale="r")]),
        AppealDraft: AppealDraft(sentences=[
            LetterSentence(id="a", text="Dear Appeals Committee,", cites=[]),
            LetterSentence(id="b", text=S2, cites=["L1"]),
            LetterSentence(id="c", text=S3, cites=["L99"])]),
        Review: Review(flags=[], summary="ok"),
    })


@pytest.fixture
def client(tmp_path):
    data = tmp_path / "data"
    shutil.copytree("data/policies", data / "policies")
    sample = data / "cases" / "case_01"
    sample.mkdir(parents=True)
    (sample / "denial.txt").write_text("Acme Health denial for MRI lumbar spine.", encoding="utf-8")
    (sample / "chart.txt").write_text("Low back pain for 8 weeks.\nCompleted 6 weeks of PT.\n", encoding="utf-8")
    (sample / "expected.json").write_text(json.dumps({"title": "Acme Health · Lumbar MRI"}), encoding="utf-8")
    settings = Settings(_env_file=None, data_dir=data, cache_dir=tmp_path / "cache", max_runs_per_ip_per_hour=1)
    with TestClient(create_app(settings, llm_factory=scripted)) as c:
        yield c


def new_case(client) -> str:
    r = client.post("/api/cases", json={"sample_id": "case_01"})
    assert r.status_code == 200, r.text
    return r.json()["case_id"]


def run(client, case_id) -> list[dict]:
    r = client.get(f"/api/cases/{case_id}/run")
    assert r.status_code == 200, r.text
    return [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_samples_and_create_from_sample(client):
    assert client.get("/api/samples").json() == [{"id": "case_01", "title": "Acme Health · Lumbar MRI"}]
    case_id = new_case(client)
    assert client.get(f"/api/cases/{case_id}").json()["status"] == "new"


def test_unknown_sample_404(client):
    assert client.post("/api/cases", json={"sample_id": "case_99"}).status_code == 404
    assert client.post("/api/cases", json={"sample_id": "../x"}).status_code == 404


def test_upload_txt(client):
    r = client.post("/api/cases", files={"denial": ("d.txt", b"Denied."), "chart": ("c.txt", b"Line one.\nLine two.")})
    assert r.status_code == 200
    case = client.get(f"/api/cases/{r.json()['case_id']}").json()
    assert [l["id"] for l in case["chart_lines"]] == ["L1", "L2"]


def test_upload_blank_pdf_400(client, tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "b.pdf"
    c = canvas.Canvas(str(p))
    c.showPage()
    c.save()
    r = client.post("/api/cases", files={"denial": ("d.txt", b"Denied."), "chart": ("b.pdf", p.read_bytes())})
    assert r.status_code == 400 and "No extractable text" in r.json()["detail"]


def test_upload_too_large_413(client):
    big = b"x" * (5 * 1024 * 1024 + 1)
    r = client.post("/api/cases", files={"denial": ("d.txt", b"Denied."), "chart": ("c.txt", big)})
    assert r.status_code == 413


def test_run_streams_to_needs_review(client):
    case_id = new_case(client)
    events = run(client, case_id)
    assert events[-1]["status"] == "needs_review"
    flags = client.get(f"/api/cases/{case_id}").json()["review"]["flags"]
    assert "S3" in [f["sentence_id"] for f in flags]


def test_rerun_conflict_409(client):
    case_id = new_case(client)
    run(client, case_id)
    assert client.get(f"/api/cases/{case_id}/run").status_code == 409


def test_rate_limit_429(client):
    first, second = new_case(client), new_case(client)
    run(client, first)
    assert client.get(f"/api/cases/{second}/run").status_code == 429


def test_approve_blocked_until_resolved(client):
    case_id = new_case(client)
    run(client, case_id)
    r = client.post(f"/api/cases/{case_id}/approve")
    assert r.status_code == 409 and "Resolve all flags" in r.json()["detail"]
    assert client.patch(f"/api/cases/{case_id}/letter", json={"sentence_id": "S3", "flag_action": "remove"}).status_code == 200
    r = client.post(f"/api/cases/{case_id}/approve")
    assert r.status_code == 200
    letter = r.json()["final_letter"]
    assert S3 not in letter and S2 in letter and r.json()["status"] == "approved"


def test_patch_edits_text(client):
    case_id = new_case(client)
    run(client, case_id)
    r = client.patch(f"/api/cases/{case_id}/letter", json={"sentence_id": "S2", "text": "Edited sentence here."})
    assert r.status_code == 200
    assert r.json()["draft"]["sentences"][1]["text"] == "Edited sentence here."
    assert client.patch(f"/api/cases/{case_id}/letter", json={"sentence_id": "S9", "text": "x"}).status_code == 404


def test_export_requires_approved(client):
    case_id = new_case(client)
    run(client, case_id)
    assert client.get(f"/api/cases/{case_id}/export").status_code == 409
    client.patch(f"/api/cases/{case_id}/letter", json={"sentence_id": "S3", "flag_action": "keep"})
    client.post(f"/api/cases/{case_id}/approve")
    r = client.get(f"/api/cases/{case_id}/export?format=md")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"] and f"appeal-{case_id}.md" in r.headers["content-disposition"]
    assert S2 in r.text and S3 in r.text


def test_approved_case_untouched_by_run(client):
    case_id = new_case(client)
    run(client, case_id)
    client.patch(f"/api/cases/{case_id}/letter", json={"sentence_id": "S3", "flag_action": "remove"})
    letter = client.post(f"/api/cases/{case_id}/approve").json()["final_letter"]
    assert client.get(f"/api/cases/{case_id}/run").status_code == 409
    assert client.get(f"/api/cases/{case_id}").json()["final_letter"] == letter


def test_unknown_case_404(client):
    assert client.get("/api/cases/nope").status_code == 404
    assert client.get("/api/cases/nope/run").status_code == 404
