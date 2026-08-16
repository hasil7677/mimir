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


_TEST_REDIS_DB = 15


@pytest.fixture(autouse=True)
def _isolated_redis(monkeypatch):
    """Point tests at a dedicated Redis logical DB and empty it per test.

    DuckDB and Qdrant get a fresh path per test; Redis had no equivalent. That
    was invisible for as long as no Redis was running — every cache lookup
    missed, so nothing leaked. On a machine where Redis IS up, the semantic
    cache keys on nothing but (tenant, user, normalized query), all of which
    tests reuse freely: a response cached by an earlier test run gets served
    to a later one, carrying fact ids pointing into a DuckDB file that no
    longer exists. Two tests failed exactly that way, while passing in CI.

    Using db 15 rather than flushing the default keeps a developer's own
    `~/.mimir` hot turns and cached queries intact while the suite runs.
    """
    parsed = urlparse(settings.storage.redis.url)
    test_url = f"redis://{parsed.hostname or 'localhost'}:{parsed.port or 6379}/{_TEST_REDIS_DB}"
    monkeypatch.setattr(settings.storage.redis, "url", test_url)

    from app.db import redis_client

    monkeypatch.setattr(redis_client, "_client", None)

    def _flush():
        try:
            redis_client.get_redis().flushdb()
        except Exception:
            pass  # no Redis reachable: nothing was cached, nothing to clear

    _flush()
    yield
    _flush()
    monkeypatch.setattr(redis_client, "_client", None)


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
