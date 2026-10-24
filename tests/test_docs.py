import re
import tomllib
from pathlib import Path

HEADINGS = ["## What it is", "## Architecture", "## How Nebius and NVIDIA are used", "## Eval results", "## Setup",
            "## Disclaimer", "## License"]


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


def test_readme_documents_models_and_deploy():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B" in readme and "nvidia/nemotron-3-super-120b-a12b" in readme
    assert "## Deploy" in readme and "TRUST_PROXY=true" in readme and "CODE_GUIDE.md" in readme
    assert "TODO (run" not in readme


def test_readme_lists_policy_sources():
    from app.retrieval import load_policies
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Policy sources" in readme
    for d in load_policies(Path("data/policies")):
        assert d.policy_id in readme and d.source_url in readme, d.policy_id
