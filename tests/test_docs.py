import re
import tomllib
from pathlib import Path

HEADINGS = ["## What it does", "## Hackathon requirements", "## Run it", "## Disclaimer", "## License"]


def test_license_is_mit():
    assert Path("LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_readme_sections():
    readme = Path("README.md").read_text(encoding="utf-8")
    for h in HEADINGS:
        assert h in readme, h
    assert "```mermaid" in readme and "api.tokenfactory.nebius.com" in readme


def test_requirements_in_sync():
    deps = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    reqs = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for dep in deps:
        name = re.split(r"[\[<>=]", dep)[0].strip().lower()
        assert re.search(rf"^{re.escape(name)}==", reqs, re.MULTILINE), name


def test_dockerfile_cmd():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim" in dockerfile and "app.main:app" in dockerfile


def test_every_policy_cites_its_public_source():
    from app.retrieval import load_policies
    for d in load_policies(Path("data/policies")):
        assert d.source and d.source_url.startswith("https://"), d.policy_id


def test_readme_documents_requirements():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B" in readme and "nvidia/nemotron-3-super-120b-a12b" in readme
    for phrase in ("MIT", "no login", "Synthetic data only", "TRUST_PROXY=true"):
        assert phrase in readme, phrase
