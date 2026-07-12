"""Semantic cache against fakeredis — exercises real Redis command paths
(hset/hgetall/expire/delete) without needing a server."""

import fakeredis
import pytest

from app.config import settings
from app.core import semantic_cache
from app.db import redis_client


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "_client", fake)
    yield
    monkeypatch.setattr(redis_client, "_client", None)


def test_exact_hit_without_any_embeddings():
    response = {"context_string": "ctx", "memories": []}
    semantic_cache.put("t1", "u1", "What is my training split?", None, response)

    hit = semantic_cache.get("t1", "u1", "what is  my training split?")  # case/space normalized
    assert hit == response


def test_semantic_hit_above_threshold():
    response = {"context_string": "ctx", "memories": []}
    semantic_cache.put("t1", "u1", "best async error handling", [1.0, 0.0], response)

    near = [0.98, 0.19]  # cosine ~0.982 > 0.92 default
    assert semantic_cache.get("t1", "u1", "handling errors in async code", near) == response

    far = [0.3, 0.95]
    assert semantic_cache.get("t1", "u1", "unrelated question", far) is None


def test_cache_scoped_per_tenant_and_user():
    semantic_cache.put("t1", "u1", "my question", None, {"context_string": "private", "memories": []})

    assert semantic_cache.get("t2", "u1", "my question") is None
    assert semantic_cache.get("t1", "u2", "my question") is None


def test_invalidate_clears_users_cache():
    semantic_cache.put("t1", "u1", "q", None, {"context_string": "stale", "memories": []})
    semantic_cache.invalidate("t1", "u1")
    assert semantic_cache.get("t1", "u1", "q") is None


def test_redis_down_means_miss_not_error(monkeypatch):
    class Dead:
        def __getattr__(self, name):
            raise ConnectionError("down")

    monkeypatch.setattr(redis_client, "_client", Dead())
    assert semantic_cache.get("t1", "u1", "q") is None
    semantic_cache.put("t1", "u1", "q", None, {"context_string": "x", "memories": []})  # must not raise
    semantic_cache.invalidate("t1", "u1")  # must not raise


def test_recall_pipeline_serves_second_identical_query_from_cache(tmp_path, monkeypatch):
    import uuid
    from datetime import datetime, timezone

    from fastapi.testclient import TestClient

    from app.db import l1_store
    from app.main import app

    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))
    monkeypatch.setattr(settings.embedding, "provider", "none")

    l1_store.insert_fact(
        str(uuid.uuid4()), "u1", "local", "User trains at Iron Temple gym",
        "episodic", 60, "gym", "s0", [], datetime.now(timezone.utc),
    )

    with TestClient(app) as client:
        first = client.post("/v1/recall", json={"user_id": "u1", "query": "where does the user train"})
        second = client.post("/v1/recall", json={"user_id": "u1", "query": "where does the user train"})

    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert second.json()["context_string"] == first.json()["context_string"]
