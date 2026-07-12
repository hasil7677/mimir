import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from app.core import extraction
from app.core.llm import LlmUnavailable
from app.db import l1_store


def _insert(tenant, user, content, **kw):
    fact_id = str(uuid.uuid4())
    l1_store.insert_fact(
        fact_id, user, tenant, content,
        kw.get("fact_type", "episodic"), kw.get("priority", 50),
        kw.get("scene_name", "test"), "s1", [], datetime.now(timezone.utc),
    )
    return fact_id


def test_keyword_search_ranks_matching_facts():
    hit = _insert("t1", "u1", "User benched 100kg at Iron Temple gym")
    _insert("t1", "u1", "User prefers dark roast coffee")

    results = l1_store.search_keyword("t1", "u1", "bench press gym")
    assert results, "should find at least the gym fact"
    assert results[0]["id"] == hit


def test_keyword_search_is_tenant_and_user_scoped():
    _insert("t1", "u1", "secret tenant one gym routine")
    _insert("t2", "u1", "secret tenant two gym routine")

    results = l1_store.search_keyword("t1", "u1", "gym routine")
    assert len(results) == 1
    assert "tenant one" in results[0]["content"]


def test_sql_fallback_when_fts_unavailable(monkeypatch):
    monkeypatch.setitem(l1_store._fts_state, "available", False)
    hit = _insert("t1", "u1", "User is training for a powerlifting meet in December")
    _insert("t1", "u1", "User has a golden retriever named Biscuit")

    results = l1_store.search_keyword("t1", "u1", "powerlifting training december")
    assert results[0]["id"] == hit
    assert all(r["score"] > 0 for r in results)


def test_search_sees_facts_inserted_after_previous_search():
    _insert("t1", "u1", "old fact about cooking pasta")
    l1_store.search_keyword("t1", "u1", "pasta")  # builds the FTS index

    new = _insert("t1", "u1", "new fact about climbing mountains")
    results = l1_store.search_keyword("t1", "u1", "climbing mountains")

    assert any(r["id"] == new for r in results), "lazy index rebuild must pick up new facts"


def test_timestamps_survive_roundtrip_as_utc():
    """Regression: tz-aware datetimes handed to DuckDB get converted to LOCAL
    time before the tz is dropped, shifting every timestamp by the machine's
    UTC offset — surfaced live as facts aged '-1 days ago' on an IST box."""
    from app.core import scoring

    fact_id = _insert("t1", "u1", "timestamp roundtrip fact")
    fact = l1_store.get_facts_by_ids("t1", "u1", [fact_id])[0]

    # read back (naive, interpreted as UTC) must score as brand new
    assert scoring.recency_score(fact["created_at"]) > 0.99


def test_bump_access_increments_count():
    fact_id = _insert("t1", "u1", "frequently accessed fact about guitars")
    l1_store.bump_access([fact_id])
    l1_store.bump_access([fact_id])

    facts = l1_store.get_facts_by_ids("t1", "u1", [fact_id])
    assert facts[0]["access_count"] == 2


def test_extract_facts_verbatim_fallback_keeps_substantive_user_turns():
    turns = [
        {"role": "user", "content": "I am training for a powerlifting competition in December"},
        {"role": "assistant", "content": "That is exciting, tell me more about your program"},
        {"role": "user", "content": "ok"},
    ]
    with patch("app.core.extraction.chat", side_effect=LlmUnavailable("down")):
        facts, mode = extraction.extract_facts(turns)

    assert mode == "verbatim"
    assert len(facts) == 1  # assistant turn and "ok" both excluded
    assert facts[0]["type"] == "episodic"
    assert "powerlifting" in facts[0]["content"]


def test_extract_facts_llm_path_filters_low_priority():
    reply = """[
      {"content": "User is allergic to peanuts", "type": "persona", "priority": 95, "scene_name": "health"},
      {"content": "User said hello", "type": "episodic", "priority": 5, "scene_name": "greeting"}
    ]"""
    with patch("app.core.extraction.chat", return_value=reply):
        facts, mode = extraction.extract_facts([{"role": "user", "content": "x"}])

    assert mode == "llm"
    assert len(facts) == 1
    assert facts[0]["content"] == "User is allergic to peanuts"
