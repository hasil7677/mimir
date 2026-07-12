"""GDPR erasure + export through the real HTTP layer, offline."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(settings.llm, "base_url", "http://127.0.0.1:1")
    monkeypatch.setattr(settings.llm, "timeout_ms", 200)
    monkeypatch.setattr(settings.embedding, "provider", "none")


def _populate(client, user_id):
    client.post("/v1/capture", json={
        "user_id": user_id, "session_id": "s1",
        "messages": [{"role": "user", "content": "I train with coach Vikram at Iron Temple"}],
    })
    client.post("/v1/session/end", json={"user_id": user_id, "session_id": "s1"})


def test_export_contains_l0_l1_and_vault():
    with TestClient(app) as client:
        _populate(client, "u1")
        export = client.get("/v1/export/u1").json()

    assert len(export["l0_conversations"]) == 1
    assert len(export["l1_memories"]) == 1
    assert any(path.startswith("scenes/") for path in export["vault"])
    assert any("Vikram" in content for content in export["vault"].values())


def test_erasure_wipes_every_store_and_recall_finds_nothing():
    with TestClient(app) as client:
        _populate(client, "u1")

        receipt = client.delete("/v1/user/u1").json()
        assert receipt["l0_deleted"] == 1
        assert receipt["l1_deleted"] == 1
        assert receipt["vault_notes_deleted"] >= 1

        after = client.get("/v1/export/u1").json()
        recall = client.post("/v1/recall", json={"user_id": "u1", "query": "Vikram Iron Temple"}).json()

    assert after["l0_conversations"] == []
    assert after["l1_memories"] == []
    assert after["vault"] == {}
    assert recall["memories"] == []


def test_erasure_does_not_touch_other_users():
    with TestClient(app) as client:
        _populate(client, "u1")
        _populate(client, "u2")

        client.delete("/v1/user/u1")
        export_u2 = client.get("/v1/export/u2").json()

    assert len(export_u2["l1_memories"]) == 1
    assert export_u2["vault"] != {}


def test_erasure_leaves_an_audit_receipt():
    from app.db.duckdb_client import get_connection

    with TestClient(app) as client:
        _populate(client, "u1")
        client.delete("/v1/user/u1")

    rows = get_connection().execute(
        "SELECT action FROM audit_log WHERE user_id = 'u1' AND action = 'erasure'"
    ).fetchall()
    assert rows, "the erasure event itself must be provable after the data is gone"
