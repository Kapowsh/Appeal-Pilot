from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

BANNER = "Synthetic data only · Drafting aid, not medical advice · Clinician sign-off required before use"
IDS = ["banner", "sample-select", "denial-file", "chart-file", "run-btn", "step-tracker", "denial-card",
       "criteria-table", "external-refs", "chart-panel", "letter", "approve-btn", "download-btn", "usage-footer"]
STEPS = ["intake", "policy", "match", "draft", "review"]


@pytest.fixture(scope="module")
def html():
    with TestClient(create_app(Settings(_env_file=None))) as c:
        r = c.get("/")
        assert r.status_code == 200
        return r.text


def test_index_served_with_banner(html):
    assert BANNER in html


def test_required_ids_present(html):
    for i in IDS:
        assert f'id="{i}"' in html, i
    for s in STEPS:
        assert f'data-step="{s}"' in html, s


def test_no_innerhtml():
    assert "innerHTML" not in Path("static/index.html").read_text(encoding="utf-8")
