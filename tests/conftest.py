import socket
import tempfile
from urllib.parse import urlparse

import pytest

from app.config import settings


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _redis_reachable() -> bool:
    parsed = urlparse(settings.storage.redis.url)
    return _port_open(parsed.hostname or "localhost", parsed.port or 6379)


requires_redis = pytest.mark.skipif(
    not _redis_reachable(), reason="Redis must be reachable at storage.redis.url"
)


@pytest.fixture(autouse=True)
def _isolated_duckdb_per_test(monkeypatch):
    """Every test gets its own throwaway DuckDB file so tests never see each
    other's rows, and never touch a real ~/.mimir/memories.db."""
    monkeypatch.setattr(settings.storage.duckdb, "path", tempfile.mktemp(suffix=".duckdb"))

    from app.db import duckdb_client, l1_store

    monkeypatch.setattr(duckdb_client, "_conn", None)
    monkeypatch.setattr(l1_store, "_fts_state", {"available": None, "dirty": True})
    yield
    monkeypatch.setattr(duckdb_client, "_conn", None)


@pytest.fixture(autouse=True)
def _isolated_qdrant(tmp_path, monkeypatch):
    """Embedded Qdrant holds a file lock per path — every test gets a fresh
    directory and a fresh client so tests can't collide or leak points."""
    monkeypatch.setattr(settings.storage.qdrant, "path", str(tmp_path / "qdrant"))

    from app.db import vector_store

    monkeypatch.setattr(vector_store, "_client", None)
    yield
    if vector_store._client is not None:
        vector_store._client.close()
    monkeypatch.setattr(vector_store, "_client", None)
