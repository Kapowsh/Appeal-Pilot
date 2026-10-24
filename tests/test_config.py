import socket
from pathlib import Path

import pytest

from app.config import Settings


def test_defaults(monkeypatch):
    for k in ["MODEL_FAST", "MODEL_REASON", "MODEL_EMBED", "ENABLE_TAVILY", "MAX_RUNS_PER_IP_PER_HOUR"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.nebius_base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert s.model_embed == "" and s.enable_tavily is False and s.max_runs_per_ip_per_hour == 10


def test_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_REASON", "x/y")
    monkeypatch.setenv("ENABLE_TAVILY", "true")
    s = Settings(_env_file=None)
    assert s.model_reason == "x/y" and s.enable_tavily is True


def test_no_hardcoded_model_ids():
    for p in Path("app").rglob("*.py"):
        assert "nemotron" not in p.read_text(encoding="utf-8").lower(), p


def test_network_blocked():
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("example.com", 80))
