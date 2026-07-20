"""Qdrant-native hybrid search (fastembed provider): dense+sparse upsert,
prefetch+RRF query, and the pipeline/recall wiring around them.

Most tests here inject deterministic fake dense/sparse vectors (same pattern
as test_recall_e2e's fake_embed) so the fusion/branching logic is testable
without downloading or running real models. One test (marked real) exercises
actual FastEmbed models end-to-end and is skipped when the optional
`fastembed` dependency isn't installed.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core import pipeline
from app.db import l1_store, vector_store
from app.main import app


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


@pytest.fixture(autouse=True)
def _force_offline_llm(monkeypatch):
    monkeypatch.setattr(settings.llm, "base_url", "http://127.0.0.1:1")
    monkeypatch.setattr(settings.llm, "timeout_ms", 200)
    monkeypatch.setattr(settings.embedding, "provider", "none")


def _seed_fact(tenant, user, content, **kw):
    fact_id = str(uuid.uuid4())
    l1_store.insert_fact(
        fact_id, user, tenant, content, kw.get("fact_type", "episodic"),
        kw.get("priority", 60), "seed", "s0", [], datetime.now(timezone.utc),
    )
    return fact_id


def _fake_hybrid(dense_by_text: dict[str, list[float]], sparse_by_text: dict[str, dict]):
    def fake_embed_hybrid(texts):
        return [{"dense": dense_by_text[t], "sparse": sparse_by_text[t]} for t in texts]
    return fake_embed_hybrid


def test_upsert_and_search_hybrid_finds_semantic_match(monkeypatch):
    """Mirrors test_recall_e2e's dense-only semantic test, but through the
    hybrid collection and Qdrant's own prefetch+RRF fusion instead of our
    RRF merge."""
    dense = {
        "User is into powerlifting and strength training": [1.0, 0.0, 0.0],
        "User's cat is named Whiskers": [0.0, 1.0, 0.0],
        "what protein powder should I buy": [0.9, 0.1, 0.0],
    }
    sparse = {
        "User is into powerlifting and strength training": {"indices": [1, 2], "values": [1.0, 1.0]},
        "User's cat is named Whiskers": {"indices": [3, 4], "values": [1.0, 1.0]},
        "what protein powder should I buy": {"indices": [1, 5], "values": [1.0, 1.0]},
    }
    fake_embed_hybrid = _fake_hybrid(dense, sparse)

    monkeypatch.setattr(settings.embedding, "provider", "fastembed")
    lifting = _seed_fact("local", "u1", "User is into powerlifting and strength training")
    cat = _seed_fact("local", "u1", "User's cat is named Whiskers")

    facts = l1_store.get_facts_by_ids("local", "u1", [lifting, cat])
    with patch("app.core.recall.embed_hybrid", side_effect=fake_embed_hybrid):
        embeddings = fake_embed_hybrid([f["content"] for f in facts])
        vector_store.upsert_facts_hybrid("local", "u1", facts, embeddings)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/recall", json={"user_id": "u1", "query": "what protein powder should I buy"}
            )

    payload = resp.json()
    assert payload["vector_used"] is True
    ids = [m["id"] for m in payload["memories"]]
    assert lifting in ids, "semantic+sparse match must surface despite zero keyword overlap"
    assert ids[0] == lifting
    assert 0.0 <= payload["memories"][0]["semantic_score"] <= 1.0, "raw RRF score must be normalized"


def test_search_hybrid_returns_empty_when_collection_missing():
    assert vector_store.search_hybrid("local", "u1", [0.1, 0.2], {"indices": [1], "values": [1.0]}) == []


def test_recall_falls_back_to_keyword_when_hybrid_index_empty(monkeypatch):
    """fastembed configured but nothing's been upserted into the hybrid
    collection yet — recall must still find the fact via DuckDB keyword
    search rather than returning nothing."""
    fake_embed_hybrid = _fake_hybrid(
        {"bench press gym PR": [1.0, 0.0]}, {"bench press gym PR": {"indices": [1], "values": [1.0]}}
    )
    monkeypatch.setattr(settings.embedding, "provider", "fastembed")
    hit = _seed_fact("local", "u1", "User benched 100kg at Iron Temple gym, a new PR")

    with patch("app.core.recall.embed_hybrid", side_effect=fake_embed_hybrid):
        with TestClient(app) as client:
            resp = client.post("/v1/recall", json={"user_id": "u1", "query": "bench press gym PR"})

    payload = resp.json()
    assert payload["memories"], "keyword fallback must still return results"
    assert payload["memories"][0]["id"] == hit


def test_pipeline_flush_session_indexes_via_hybrid_collection(monkeypatch):
    fake_embed_hybrid = _fake_hybrid(
        {"User trains legs on Mondays": [1.0, 0.0]},
        {"User trains legs on Mondays": {"indices": [1], "values": [1.0]}},
    )
    monkeypatch.setattr(settings.embedding, "provider", "fastembed")
    monkeypatch.setattr(
        "app.core.extraction.extract_facts",
        lambda turns: ([{"content": "User trains legs on Mondays", "type": "episodic", "priority": 60, "scene_name": "seed"}], "offline"),
    )

    with patch("app.core.pipeline.embed_hybrid", side_effect=fake_embed_hybrid):
        pipeline.capture("local", "u1", "s1", [{"role": "user", "content": "I train legs on Mondays"}])
        result = pipeline.flush_session("local", "u1", "s1")

    assert result["vector_indexed"] is True
    hits = vector_store.search_hybrid(
        "local", "u1", [1.0, 0.0], {"indices": [1], "values": [1.0]}, top_k=5
    )
    assert hits, "fact must be discoverable in the hybrid collection after flush"


def test_erase_user_clears_both_legacy_and_hybrid_collections():
    fact = {"id": str(uuid.uuid4()), "content": "erase me", "type": "episodic"}
    vector_store.upsert_facts("local", "u1", [fact], [[0.1, 0.2, 0.3]])
    vector_store.upsert_facts_hybrid(
        "local", "u1", [fact], [{"dense": [0.1, 0.2], "sparse": {"indices": [1], "values": [1.0]}}]
    )

    vector_store.erase_user("local", "u1")

    assert vector_store.search("local", "u1", [0.1, 0.2, 0.3]) == []
    assert vector_store.search_hybrid("local", "u1", [0.1, 0.2], {"indices": [1], "values": [1.0]}) == []


def test_real_fastembed_hybrid_round_trip(monkeypatch, tmp_path):
    """Opt-in end-to-end check with actual FastEmbed models (no mocks): dense
    384-dim + real BM25-style sparse output, indexed and queried through
    Qdrant's native fusion. Skipped when the optional `fastembed` package
    isn't installed — this is the only test in the suite that downloads or
    runs a real model."""
    pytest.importorskip("fastembed")
    from app.core import embeddings

    monkeypatch.setattr(settings.embedding, "provider", "fastembed")
    monkeypatch.setattr(settings.embedding, "fastembed_cache_dir", str(tmp_path / "models"))
    embeddings._fastembed_models.cache_clear()

    lifting = _seed_fact("local", "u1", "User is into powerlifting and deadlifts heavy")
    cat = _seed_fact("local", "u1", "User adopted a kitten named Momo")
    facts = l1_store.get_facts_by_ids("local", "u1", [lifting, cat])

    embedded = embeddings.embed_hybrid([f["content"] for f in facts])
    assert len(embedded[0]["dense"]) == 384
    assert embedded[0]["sparse"]["indices"], "real BM25 sparse output must be non-empty for real text"

    vector_store.upsert_facts_hybrid("local", "u1", facts, embedded)

    query = embeddings.embed_hybrid(["what does the user lift at the gym"])[0]
    hits = vector_store.search_hybrid("local", "u1", query["dense"], query["sparse"], top_k=5)

    assert hits
    assert hits[0]["id"] == lifting, "real embeddings must rank the lifting fact above the unrelated cat fact"

    embeddings._fastembed_models.cache_clear()
