import re
from pathlib import Path

import pytest

from app.retrieval import load_policies, parse_policy

SAMPLE = """---
policy_id: AH-RAD-012
payer: Acme Health
service: MRI lumbar spine
codes: 72148, 72149
---
# Acme Health Medical Policy AH-RAD-012: MRI of the Lumbar Spine
## 1. Scope
Text.
"""


def test_parse_front_matter():
    d = parse_policy(SAMPLE, "x.md")
    assert d.policy_id == "AH-RAD-012" and d.payer == "Acme Health"
    assert d.codes == ["72148", "72149"]
    assert d.title.startswith("Acme Health Medical Policy")


def test_parse_missing_key_raises():
    with pytest.raises(ValueError):
        parse_policy(SAMPLE.replace("payer: Acme Health\n", ""), "x.md")


def criteria_lines(body: str) -> list[str]:
    section = re.search(r"## \d+\. Coverage Criteria\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
    assert section, "no Coverage Criteria section"
    return re.findall(r"^\d+\.\d+ .+$", section.group(1), re.MULTILINE)


def test_corpus_shape():
    docs = load_policies(Path("data/policies"))
    assert len(docs) == 15
    assert {d.payer for d in docs} == {"Acme Health", "Northwind Care"}
    assert len({d.policy_id for d in docs}) == 15
    for d in docs:
        assert len(criteria_lines(d.body)) >= 3, d.filename


def test_every_policy_cites_a_public_source():
    for d in load_policies(Path("data/policies")):
        assert d.source, d.filename
        assert d.source_url.startswith("https://"), d.filename
        assert "Adapted from" in d.body, d.filename
