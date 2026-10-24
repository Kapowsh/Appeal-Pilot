import socket

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
