import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.config import settings
from app.core import consolidation, persona, vault
from app.core.llm import LlmUnavailable
from app.db import l1_store


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


def _seed(tenant, user, content, **kw):
    fact_id = str(uuid.uuid4())
    l1_store.insert_fact(
        fact_id, user, tenant, content, kw.get("fact_type", "episodic"),
        kw.get("priority", 60), "seed", "s0", [], datetime.now(timezone.utc),
    )
    return fact_id


def _fact(content, **kw):
    return {"content": content, "type": kw.get("type", "episodic"),
            "priority": kw.get("priority", 60), "scene_name": "t", "extraction": "test"}


def test_offline_consolidation_skips_exact_duplicates_keeps_rest():
    _seed("t1", "u1", "User lives in Berlin")
    new = [_fact("user lives in Berlin!"), _fact("User adopted a dog named Biscuit")]

    with patch("app.core.consolidation.chat", side_effect=LlmUnavailable("down")):
        kept = consolidation.consolidate("t1", "u1", new)

    assert len(kept) == 1
    assert "Biscuit" in kept[0]["content"]


def test_offline_consolidation_dedupes_within_the_new_batch():
    new = [_fact("User trains at Iron Temple"), _fact("user trains at IRON temple")]
    with patch("app.core.consolidation.chat", side_effect=LlmUnavailable("down")):
        kept = consolidation.consolidate("t1", "u1", new)
    assert len(kept) == 1


def test_llm_update_decision_maps_to_supersede():
    old_id = _seed("t1", "u1", "User lives in San Francisco")
    new = [_fact("User lives in Berlin now")]

    reply = f'[{{"index": 0, "decision": "update", "target_id": "{old_id}"}}]'
    with patch("app.core.consolidation.chat", return_value=reply):
        kept = consolidation.consolidate("t1", "u1", new)

    assert kept[0]["_decision"] == "update"
    assert kept[0]["_supersedes"] == old_id


def test_llm_skip_decision_drops_fact_and_contradiction_is_flagged():
    existing = _seed("t1", "u1", "User is vegetarian")
    # paraphrase (not exact dup) so it survives the offline layer and reaches the LLM
    new = [_fact("User follows a vegetarian diet"), _fact("User loves steak dinners")]

    reply = (
        f'[{{"index": 0, "decision": "skip"}},'
        f' {{"index": 1, "decision": "store", "contradicts_id": "{existing}"}}]'
    )
    with patch("app.core.consolidation.chat", return_value=reply):
        kept = consolidation.consolidate("t1", "u1", new)

    assert len(kept) == 1
    assert "steak" in kept[0]["content"]
    assert kept[0]["_contradicts"] == existing


def test_exact_duplicate_never_reaches_the_llm():
    _seed("t1", "u1", "User is vegetarian")
    new = [_fact("User is vegetarian")]

    with patch("app.core.consolidation.chat") as mock_chat:
        kept = consolidation.consolidate("t1", "u1", new)

    assert kept == []
    mock_chat.assert_not_called(), "offline dedup layer should short-circuit before any LLM spend"


def test_llm_cannot_supersede_ids_it_wasnt_shown():
    _seed("t1", "u1", "User lives in San Francisco")
    new = [_fact("User lives in Berlin now")]

    reply = '[{"index": 0, "decision": "update", "target_id": "forged-id-123"}]'
    with patch("app.core.consolidation.chat", return_value=reply):
        kept = consolidation.consolidate("t1", "u1", new)

    assert kept[0]["_supersedes"] is None, "hallucinated/forged target ids must be ignored"


def test_supersede_removes_old_fact_from_search():
    old_id = _seed("t1", "u1", "User lives in San Francisco")
    new_id = _seed("t1", "u1", "User lives in Berlin now")
    l1_store.supersede(old_id, new_id)

    results = l1_store.search_keyword("t1", "u1", "where does user live San Francisco Berlin")
    ids = [r["id"] for r in results]
    assert new_id in ids
    assert old_id not in ids


def test_persona_synthesis_offline_digest_and_recall_injection():
    _seed("t1", "u1", "User prefers TypeScript for frontend work", fact_type="persona", priority=90)
    _seed("t1", "u1", "Always answer in British English", fact_type="instruction", priority=95)

    with patch("app.core.persona.chat", side_effect=LlmUnavailable("down")):
        mode = persona.maybe_synthesize("t1", "u1")

    assert mode == "digest"
    note = vault.read_note("t1", "u1", "persona")
    assert note is not None
    assert note[0]["fact_count"] == 2
    assert "TypeScript" in note[1]
    assert "British English" in note[1]

    lines = persona.profile_lines("t1", "u1")
    assert any("TypeScript" in ln for ln in lines)


def test_persona_not_resynthesized_until_enough_new_facts(monkeypatch):
    monkeypatch.setattr(settings.pipeline, "l3_every_n_memories", 50)
    _seed("t1", "u1", "User plays guitar", fact_type="persona", priority=80)

    with patch("app.core.persona.chat", side_effect=LlmUnavailable("down")):
        first = persona.maybe_synthesize("t1", "u1")   # no persona yet -> runs
        second = persona.maybe_synthesize("t1", "u1")  # 0 new facts since -> skipped

    assert first == "digest"
    assert second is None
