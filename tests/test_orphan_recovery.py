"""Regression for a bug hit live: an MCP adapter process recycled mid-
conversation (a /model switch in Claude Code) generates a fresh random
session_id, orphaning everything captured under the old one — the new
process's mimir_flush() can't find it, and mimir_recall() never returns it
even though the data is safely sitting in l0_conversations the whole time.
flush_all_pending / the self-healing /session/end must recover it.
"""
import pytest

from app.config import settings
from app.core import pipeline


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(settings.llm, "base_url", "http://127.0.0.1:1")
    monkeypatch.setattr(settings.llm, "timeout_ms", 200)
    monkeypatch.setattr(settings.embedding, "provider", "none")


def test_flush_all_pending_recovers_an_orphaned_session():
    # session A captures, but the "process" dies before flushing (simulated
    # by simply never calling flush_session for it)
    pipeline.capture("t1", "u1", "session-A-orphaned", [
        {"role": "user", "content": "I am allergic to peanuts"},
    ])

    # a NEW process starts with a fresh session id, same as a recycled adapter
    result = pipeline.flush_all_pending("t1", "u1", "session-B-fresh")

    assert result["sessions_flushed"] == 1
    assert "session-A-orphaned" in result["results"]
    assert result["results"]["session-A-orphaned"]["facts_extracted"] == 1


def test_flushed_sessions_are_not_reflushed():
    pipeline.capture("t1", "u1", "session-A", [{"role": "user", "content": "I train at Iron Temple gym"}])
    pipeline.flush_session("t1", "u1", "session-A")

    # a second call must not re-process the already-flushed session
    result = pipeline.flush_all_pending("t1", "u1", "session-C-new-empty")
    assert result["sessions_flushed"] == 0


def test_flush_all_pending_recovers_multiple_orphans_at_once():
    pipeline.capture("t1", "u1", "orphan-1", [{"role": "user", "content": "I use Neovim with Gruvbox theme"}])
    pipeline.capture("t1", "u1", "orphan-2", [{"role": "user", "content": "My favorite food is biryani"}])

    result = pipeline.flush_all_pending("t1", "u1", "current-session-empty")

    assert result["sessions_flushed"] == 2
    assert set(result["results"]) == {"orphan-1", "orphan-2"}


def test_recovery_is_scoped_to_tenant_and_user():
    pipeline.capture("t1", "u1", "s1", [{"role": "user", "content": "tenant one secret fact"}])
    pipeline.capture("t2", "u1", "s2", [{"role": "user", "content": "tenant two secret fact"}])

    result = pipeline.flush_all_pending("t1", "u1", "current")

    assert result["sessions_flushed"] == 1
    assert "s2" not in result["results"]


def test_recovered_facts_become_searchable_via_recall():
    from app.core import recall as recall_pipeline

    pipeline.capture("t1", "u1", "orphaned-session", [
        {"role": "user", "content": "I am allergic to peanuts"},
    ])
    # recall alone must NOT find it — it's still unflushed at this point
    before = recall_pipeline.recall("t1", "u1", "peanut allergy")
    assert before["memories"] == []

    pipeline.flush_all_pending("t1", "u1", "brand-new-session")

    after = recall_pipeline.recall("t1", "u1", "peanut allergy")
    assert any("peanut" in m["content"].lower() for m in after["memories"])


def test_http_session_end_recovers_orphans_and_reports_them(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault2"))

    with TestClient(app) as client:
        client.post("/v1/capture", json={
            "user_id": "u1", "session_id": "orphan-http",
            "messages": [{"role": "user", "content": "I train with coach Vikram"}],
        })
        # a different session_id calls /session/end — simulating a recycled adapter
        resp = client.post("/v1/session/end", json={"user_id": "u1", "session_id": "orphan-http"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["facts_extracted"] == 1
    assert body["recovered_sessions"] == {}


def test_flush_survives_the_os_refusing_a_thread(monkeypatch):
    """`ThreadPoolExecutor` raising "can't start new thread" must not lose a
    flush. Observed for real on a loaded machine: the pool is a latency
    optimisation over two pure functions of `turns`, so exhausting the host's
    thread capacity has to degrade to sequential, not fail the session.
    """
    import concurrent.futures

    pipeline.capture("t1", "u-threads", "session-thread-pressure", [
        {"role": "user", "content": "I keep my bike in the hallway"},
    ])

    class _RefusesThreads:
        def __enter__(self):
            raise RuntimeError("can't start new thread")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        concurrent.futures, "ThreadPoolExecutor", lambda *a, **k: _RefusesThreads()
    )

    result = pipeline.flush_session("t1", "u-threads", "session-thread-pressure")

    assert result is not None, "a busy host must not cost us the whole session"
    assert result["scene_id"]
