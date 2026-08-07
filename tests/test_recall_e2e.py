"""Recall pipeline end-to-end. Offline path needs nothing external; the
vector path injects a deterministic fake embedder so cosine behavior is
testable without any embedding endpoint."""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.embeddings import EmbeddingsUnavailable
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


def test_offline_recall_returns_relevant_memory_in_context_string():
    hit = _seed_fact("local", "u1", "User benched 100kg at Iron Temple gym, a new PR")
    _seed_fact("local", "u1", "User prefers dark roast coffee in the morning")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "bench press gym PR"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["vector_used"] is False
    assert payload["memories"], "keyword-only recall must still return results"
    assert payload["memories"][0]["id"] == hit
    assert "[MEMORY CONTEXT]" in payload["context_string"]
    assert "Iron Temple" in payload["context_string"]
    assert "dark roast" not in payload["context_string"]


def test_recall_bumps_access_count_reinforcing_future_ranking():
    fact_id = _seed_fact("local", "u1", "User plays guitar every weekend")

    with TestClient(app) as client:
        client.post("/v1/recall", json={"user_id": "u1", "query": "guitar weekend"})
        client.post("/v1/recall", json={"user_id": "u1", "query": "guitar weekend"})

    facts = l1_store.get_facts_by_ids("local", "u1", [fact_id])
    assert facts[0]["access_count"] == 2


def test_recall_is_tenant_scoped_even_with_same_user_id():
    _seed_fact("other-tenant", "u1", "secret fact about quantum computing")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "quantum computing"})

    assert resp.json()["memories"] == []


def test_recall_includes_linked_vault_notes():
    from app.core import vault

    _seed_fact("local", "u1", "User trains with coach Vikram twice a week")
    note = vault.ensure_entity_stub("local", "u1", "Vikram")
    note.write_text("# Vikram\n\nPowerlifting coach, very data-driven.", encoding="utf-8")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "coach Vikram training"})

    context = resp.json()["context_string"]
    assert "LINKED NOTES:" in context
    assert "data-driven" in context


def test_graph_hops_reaches_co_mentioned_entities_via_scene_backlink():
    """The graph signal only means anything if hop traversal actually
    reaches something: entity -> scene it was mentioned in (hop 1) ->
    other entities mentioned in that same scene (hop 2)."""
    from app.core import vault
    from app.core.recall import _graph_hops

    vault.write_scene(
        "local", "u1", "sc1", "s1", "Gym session with Sarah",
        body="Sarah and the user trained at Iron Temple today.",
        entities=["Sarah", "Iron Temple"], source_ids=[],
    )

    hops = _graph_hops("local", "u1", "training with Sarah")

    assert hops.get("sarah") == 0  # named directly in the query
    assert hops.get("iron temple") == 2, (
        "a multi-word entity co-mentioned in the same scene must land at hop 2, "
        "keyed by its natural display name (not its slug) so it can substring-match "
        "real fact prose"
    )


def test_graph_score_boosts_fact_about_co_mentioned_entity(monkeypatch):
    """End-to-end: a fact about an entity that never appears in the query
    should still get a nonzero graph_score if it's vault-connected to an
    entity the query does name."""
    from app.core import vault

    vault.write_scene(
        "local", "u1", "sc1", "s1", "Gym session with Sarah",
        body="Sarah and the user trained at Iron Temple today.",
        entities=["Sarah", "Iron Temple"], source_ids=[],
    )
    # Shares "training" with the query so keyword search retrieves it at all
    # — graph_score augments an already-retrieved candidate, it isn't an
    # independent retrieval path. Deliberately never says "Sarah" — the point
    # is to isolate the hop-2 graph signal, not a direct hop-0 keyword hit.
    fact_id = _seed_fact("local", "u1", "Iron Temple has a strict no-phone policy during training sessions")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "training with Sarah"})

    memories = {m["id"]: m for m in resp.json()["memories"]}
    assert fact_id in memories
    assert memories[fact_id]["graph_score"] == 0.25


def test_vector_path_finds_semantic_match_keyword_search_misses(monkeypatch):
    """The reason the vector leg exists: 'protein powder' should surface the
    powerlifting fact despite zero keyword overlap."""
    fake_vectors = {
        "User is into powerlifting and strength training": [1.0, 0.0, 0.0],
        "User's cat is named Whiskers": [0.0, 1.0, 0.0],
        "what protein powder should I buy": [0.9, 0.1, 0.0],  # near the lifting vector
    }

    def fake_embed(texts):
        return [fake_vectors[t] for t in texts]

    monkeypatch.setattr(settings.embedding, "provider", "openai")
    lifting = _seed_fact("local", "u1", "User is into powerlifting and strength training")
    cat = _seed_fact("local", "u1", "User's cat is named Whiskers")

    facts = l1_store.get_facts_by_ids("local", "u1", [lifting, cat])
    with patch("app.core.recall.embed", side_effect=fake_embed):
        vector_store.upsert_facts(
            "local", "u1", facts, fake_embed([f["content"] for f in facts])
        )
        with TestClient(app) as client:
            resp = client.post(
                "/v1/recall", json={"user_id": "u1", "query": "what protein powder should I buy"}
            )

    payload = resp.json()
    assert payload["vector_used"] is True
    ids = [m["id"] for m in payload["memories"]]
    assert lifting in ids, "semantic match must surface despite zero keyword overlap"
    assert ids[0] == lifting


def test_keyword_only_hit_keeps_rrf_relevance_instead_of_zero(monkeypatch):
    """When the vector leg runs, a fact BM25 finds but the vector leg's
    top-20 doesn't (pushed out by 20 more-similar fillers) used to be forced
    to semantic=0.0 even though its RRF-derived keyword relevance was sitting
    right there — cutting its score via weights.semantic for no reason."""
    monkeypatch.setattr(settings.embedding, "provider", "openai")

    filler_ids = [_seed_fact("local", "u1", f"Filler fact number {i} about the weather") for i in range(20)]
    target_id = _seed_fact("local", "u1", "User's favorite made-up word is zephyrion")

    facts = l1_store.get_facts_by_ids("local", "u1", filler_ids + [target_id])
    fake_vectors = {f["content"]: ([0.0, 1.0, 0.0] if f["id"] == target_id else [1.0, 0.0, 0.0]) for f in facts}
    fake_vectors["zephyrion"] = [1.0, 0.0, 0.0]  # query vector: close to fillers, far from target

    def fake_embed(texts):
        return [fake_vectors[t] for t in texts]

    with patch("app.core.recall.embed", side_effect=fake_embed):
        vector_store.upsert_facts("local", "u1", facts, fake_embed([f["content"] for f in facts]))
        with TestClient(app) as client:
            resp = client.post("/v1/recall", json={"user_id": "u1", "query": "zephyrion"})

    payload = resp.json()
    assert payload["vector_used"] is True
    hit = next((m for m in payload["memories"] if m["id"] == target_id), None)
    assert hit is not None, "the exact keyword hit must still surface even when vector search misses it"
    assert hit["semantic_score"] > 0.0, "must keep RRF keyword relevance instead of collapsing to 0.0"


def test_context_string_respects_char_budget(monkeypatch):
    monkeypatch.setattr(settings.recall, "max_context_chars", 300)
    for i in range(8):
        _seed_fact("local", "u1", f"User fact number {i} about the gym routine and training split")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "gym routine training"})

    assert len(resp.json()["context_string"]) <= 300


def test_recall_with_no_matches_returns_empty_but_valid_context():
    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "nothing stored about this"})

    payload = resp.json()
    assert payload["memories"] == []
    assert "[MEMORY CONTEXT]" in payload["context_string"]


def test_recall_surfaces_superseded_history_regardless_of_query_phrasing():
    """No query-intent gating on purpose — a plain-sounding statement like
    'organizing panel discussions with industry experts' is exactly the kind
    of query PersonaMem-style evolution questions actually use, not anything
    phrased with "used to" / "changed". History has to surface unprompted."""
    old_id = _seed_fact("local", "u1", "User disliked researching retirement plans")
    new_id = _seed_fact("local", "u1", "User feels a renewed interest in retirement planning")
    l1_store.supersede(old_id, new_id)

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "retirement planning"})

    context = resp.json()["context_string"]
    assert "history:" in context
    assert "disliked researching retirement plans" in context
    assert "renewed interest in retirement planning" in context


def test_recall_omits_history_for_facts_that_were_never_superseded():
    _seed_fact("local", "u1", "User benched 100kg at Iron Temple gym, a new PR")

    with TestClient(app) as client:
        resp = client.post("/v1/recall", json={"user_id": "u1", "query": "gym PR"})

    assert "history:" not in resp.json()["context_string"]
