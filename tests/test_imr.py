from pathlib import Path

from scripts.imr_patterns import match_service, render, summarize


def row(summary, determination="Overturned Decision of Health Plan", year="2015", findings=""):
    return {"Report Year": year, "Determination": determination, "Type": "Medical Necessity",
            "Findings": f"Nature of Statutory Criteria/Case Summary: {summary} Findings: {findings}"}


def test_match_service_uses_case_summary_only():
    assert match_service(row("An enrollee has requested an MRI of the lumbar spine.")) == "MRI lumbar spine"
    assert match_service(row("The patient requested gastric bypass surgery.")) == "Bariatric surgery"
    assert match_service(row("The patient requested Harvoni.", findings="prior lumbar MRI was normal")) is None


def test_summarize_counts_overturns_and_themes():
    rows = [row("Requested MRI of the lumbar spine.", findings="no conservative therapy was tried"),
            row("Requested lumbar MRI.", determination="Upheld Decision of Health Plan"),
            row("Requested Harvoni."),
            row("Requested lumbar MRI.", year="the plan upheld")]
    s = summarize(rows)["MRI lumbar spine"]
    assert s["n"] == 2 and s["overturned"] == 1 and s["years"] == (2015, 2015)
    assert s["themes"]["Conservative therapy"] == 1


def test_render_has_table_and_source():
    md = render(summarize([row("Requested lumbar MRI.")]), "https://example.org/imr.csv")
    assert "| MRI lumbar spine | 1 | 100% |" in md and "https://example.org/imr.csv" in md


def test_committed_patterns_file_cited_in_readme():
    assert "| Bariatric surgery |" in Path("data/imr_patterns.md").read_text(encoding="utf-8")
    assert "data/imr_patterns.md" in Path("README.md").read_text(encoding="utf-8")


def test_denials_use_medical_necessity_and_external_review_language():
    for p in sorted(Path("data/cases").glob("case_*/denial.txt")):
        text = p.read_text(encoding="utf-8")
        assert "not medically necessary" in text and "independent external review" in text, p
