import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.retrieval import load_policies
from app.schemas import Denial
from app.store import list_samples
from app.tools.pdf import extract_text, split_chart_lines

CASES = sorted(Path("data/cases").glob("case_*"))
POLICIES = {d.policy_id: d for d in load_policies(Path("data/policies"))}


def expected(d: Path) -> dict:
    return json.loads((d / "expected.json").read_text(encoding="utf-8"))


def test_ten_cases_and_mix():
    assert len(list_samples(Path("data"))) == 10
    assert Counter(expected(d)["category"] for d in CASES) == {"supported": 7, "missing": 2, "contradicted": 1}
    for d in CASES:
        e = expected(d)
        assert e["appealable"] is (e["category"] == "supported"), d.name


@pytest.mark.parametrize("d", CASES, ids=lambda d: d.name)
def test_chart_length_and_evidence_ids_exist(d):
    lines = split_chart_lines((d / "chart.txt").read_text(encoding="utf-8"))
    assert 20 <= len(lines) <= 40
    ids = {l.id for l in lines}
    for c in expected(d)["criteria"]:
        assert set(c["evidence"]) <= ids, (d.name, c)


@pytest.mark.parametrize("d", CASES, ids=lambda d: d.name)
def test_policy_and_sections_exist(d):
    e = expected(d)
    doc = POLICIES[e["policy_id"]]
    sections = set(re.findall(r"^(\d+\.\d+) ", doc.body, re.MULTILINE))
    assert {c["section"] for c in e["criteria"]} <= sections
    assert all(c["section"].startswith("2.") for c in e["criteria"])
    assert e["policy_id"] in (d / "denial.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("d", CASES, ids=lambda d: d.name)
def test_expected_denial_validates(d):
    denial = Denial.model_validate(expected(d)["denial"])
    assert denial.payer in {"Acme Health", "Northwind Care"}


@pytest.mark.parametrize("name", ["case_01", "case_02", "case_03"])
def test_demo_pdfs_extract(name):
    d = Path("data/cases") / name
    last_name = expected(d)["denial"]["patient_name"].split()[-1]
    chart_pdf = extract_text((d / "chart.pdf").read_bytes(), "chart.pdf")
    assert last_name in chart_pdf
    assert last_name in extract_text((d / "denial.pdf").read_bytes(), "denial.pdf")
    assert len(split_chart_lines(chart_pdf)) == len(split_chart_lines((d / "chart.txt").read_text(encoding="utf-8")))
