import socket

import pytest

from app.config import get_settings

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guarded_connect(self, address, *args, **kwargs):
        # asyncio on Windows builds its self-pipe with a loopback connect; allow only that.
        host = address[0] if isinstance(address, tuple) else address
        if host in _LOOPBACK:
            return _real_connect(self, address, *args, **kwargs)
        raise RuntimeError("network disabled in tests")

    def blocked(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", blocked)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
