"""End-to-end through the real HTTP layer, fully offline: no LLM, no Redis
(the hot push degrades gracefully), just DuckDB + the filesystem vault.
This is the offline-brain guarantee under test."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


@pytest.fixture(autouse=True)
def _force_offline_synthesis(monkeypatch):
    # point the LLM at a dead port so tests always exercise the digest path
    monkeypatch.setattr(settings.llm, "base_url", "http://127.0.0.1:1")
    monkeypatch.setattr(settings.llm, "timeout_ms", 200)


def test_capture_then_session_end_produces_an_obsidian_note(tmp_path):
    with TestClient(app) as client:
        resp = client.post("/v1/capture", json={
            "user_id": "u1", "session_id": "s1",
            "messages": [
                {"role": "user", "content": "I fixed the deadlock in Project X today"},
                {"role": "assistant", "content": "Great — was it the asyncio lock issue?"},
                {"role": "user", "content": "Yes, asyncio.Lock around the shared dict"},
            ],
        })
        assert resp.status_code == 200
        assert len(resp.json()["message_ids"]) == 3

        end = client.post("/v1/session/end", json={"user_id": "u1", "session_id": "s1"})
        assert end.status_code == 200
        payload = end.json()
        assert payload["synthesis"] == "digest"  # offline path
        assert "Project X" in payload["entities"]

        notes = client.get("/v1/vault/notes", params={"user_id": "u1"}).json()["notes"]
    assert any(n.startswith("scenes") for n in notes)
    assert any(n.startswith("entities") for n in notes), "entity stubs should exist"

    # the note is a real markdown file a human can open in Obsidian
    vault_root = tmp_path / "vault" / "local" / "u1"
    scene_files = list((vault_root / "scenes").glob("*.md"))
    assert len(scene_files) == 1
    raw = scene_files[0].read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "[[Project X]]" in raw


def test_vault_note_endpoint_returns_note_with_linked_context():
    with TestClient(app) as client:
        client.post("/v1/capture", json={
            "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "content": "My teammate Sarah runs Project X at Acme Corp"}],
        })
        client.post("/v1/session/end", json={"user_id": "u1", "session_id": "s1"})

        notes = client.get("/v1/vault/notes", params={"user_id": "u1"}).json()["notes"]
        scene_target = next(n for n in notes if n.startswith("scenes")).split("/")[-1].removesuffix(".md")

        note = client.get(f"/v1/vault/note/{scene_target}", params={"user_id": "u1"})
    assert note.status_code == 200
    body = note.json()
    assert body["frontmatter"]["type"] == "scene"
    assert "Sarah" in body["linked"], "1-hop enrichment should pull the Sarah entity stub"


def test_session_end_with_no_messages_is_404():
    with TestClient(app) as client:
        resp = client.post("/v1/session/end", json={"user_id": "u1", "session_id": "never-captured"})
    assert resp.status_code == 404


def test_auth_required_when_api_key_configured(monkeypatch):
    monkeypatch.setattr(settings.server, "api_key", "sekrit")
    with TestClient(app) as client:
        no_auth = client.post("/v1/capture", json={
            "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "content": "x"}],
        })
        wrong = client.post("/v1/capture", headers={"Authorization": "Bearer nope"}, json={
            "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "content": "x"}],
        })
        right = client.post("/v1/capture", headers={"Authorization": "Bearer sekrit"}, json={
            "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "content": "x"}],
        })
    assert no_auth.status_code == 401
    assert wrong.status_code == 401
    assert right.status_code == 200


def test_hand_edited_note_is_seen_on_next_read(tmp_path):
    """The white-box guarantee: the user edits a note in Obsidian, the agent
    sees the edit — no sync step, no database to invalidate."""
    with TestClient(app) as client:
        client.post("/v1/capture", json={
            "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "content": "I met Sarah for coffee"}],
        })
        client.post("/v1/session/end", json={"user_id": "u1", "session_id": "s1"})

        stub = tmp_path / "vault" / "local" / "u1" / "entities" / "sarah.md"
        assert stub.exists()
        stub.write_text("# Sarah\n\nCORRECTION: her name is spelled Sara.", encoding="utf-8")

        note = client.get("/v1/vault/note/sarah", params={"user_id": "u1"})
    assert "CORRECTION" in note.json()["body"]
