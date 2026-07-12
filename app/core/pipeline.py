"""Capture and session-flush as plain functions — the one implementation
behind both front doors: the HTTP gateway (app.api.routes) and the embedded
MCP adapter (adapters/mcp_embedded.py), which imports the engine in-process
so nothing has to be running when no session is open.
"""

import logging
import uuid
from datetime import datetime, timezone

from app.core import consolidation, extraction, persona, semantic_cache, synthesis, vault
from app.core.embeddings import EmbeddingsUnavailable, embed
from app.db import duckdb_client, l1_store, redis_client, vector_store

logger = logging.getLogger(__name__)


def capture(tenant_id: str, user_id: str, session_id: str, messages: list[dict]) -> list[str]:
    """L0 ingest. DuckDB is ground truth and must succeed; the Redis hot push
    degrades (logged) instead of failing the capture."""
    now = datetime.now(timezone.utc)
    next_index = len(duckdb_client.get_l0_messages(user_id, tenant_id, session_id))

    message_ids: list[str] = []
    for offset, message in enumerate(messages):
        message_id = str(uuid.uuid4())
        duckdb_client.insert_l0_message(
            message_id, user_id, tenant_id, session_id,
            message.get("role", "user"), message["content"], next_index + offset, now,
        )
        message_ids.append(message_id)
        try:
            redis_client.push_turn(tenant_id, user_id, session_id, message.get("role", "user"), message["content"])
        except Exception:
            logger.warning("hot memory push failed (redis unreachable?) — capture continues")

    duckdb_client.log_audit(str(uuid.uuid4()), user_id, tenant_id, "capture", message_ids, now)
    return message_ids


def flush_session(tenant_id: str, user_id: str, session_id: str) -> dict | None:
    """Scene note -> L1 extraction -> L1.5 consolidation -> vector index ->
    cache invalidation -> L3 persona. Returns None when nothing was captured."""
    turns = duckdb_client.get_l0_messages(user_id, tenant_id, session_id)
    if not turns:
        return None

    title, note_body, entities, mode = synthesis.synthesize_scene(turns)
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    note_path = vault.write_scene(
        tenant_id, user_id, scene_id, session_id,
        title, note_body, entities, source_ids=[t["id"] for t in turns],
    )

    extracted, extraction_mode = extraction.extract_facts(turns)
    facts = consolidation.consolidate(tenant_id, user_id, extracted)
    now = datetime.now(timezone.utc)
    fact_ids: list[str] = []
    contradictions = 0
    for fact in facts:
        fact_id = str(uuid.uuid4())
        l1_store.insert_fact(
            fact_id, user_id, tenant_id, fact["content"], fact["type"],
            fact["priority"], fact["scene_name"], session_id,
            source_ids=[t["id"] for t in turns], created_at=now,
        )
        fact["id"] = fact_id
        fact_ids.append(fact_id)
        if fact.get("_supersedes"):
            l1_store.supersede(fact["_supersedes"], fact_id)
        if fact.get("_contradicts"):
            l1_store.record_contradiction(str(uuid.uuid4()), fact_id, fact["_contradicts"], now)
            contradictions += 1

    vector_indexed = False
    if facts:
        try:
            vectors = embed([f["content"] for f in facts])
            vector_store.upsert_facts(tenant_id, user_id, facts, vectors)
            vector_indexed = True
        except EmbeddingsUnavailable:
            logger.info("embeddings unavailable — facts stored keyword-searchable only")

    semantic_cache.invalidate(tenant_id, user_id)
    persona_mode = persona.maybe_synthesize(tenant_id, user_id)

    duckdb_client.log_audit(
        str(uuid.uuid4()), user_id, tenant_id, "scene_write",
        [scene_id, *fact_ids], now, {"mode": mode, "extraction": extraction_mode},
    )
    return {
        "scene_id": scene_id, "note": note_path.name, "synthesis": mode,
        "entities": entities, "facts_extracted": len(fact_ids),
        "extraction": extraction_mode, "vector_indexed": vector_indexed,
        "contradictions_flagged": contradictions, "persona": persona_mode,
    }


def flush_all_pending(tenant_id: str, user_id: str, current_session_id: str | None = None) -> dict:
    """Recovery path: flushes `current_session_id` (if given) plus every OTHER
    session that has captured turns but no scene note yet.

    Exists because session identity lives in adapter-process memory
    (mcp_embedded.py's `_SESSION_ID`), not in DuckDB — if that process gets
    recycled (a Claude Code /model switch restarts the MCP subprocess, a
    crash, anything) mid-conversation, its session_id is lost and a plain
    `flush_session(new_session_id)` finds nothing, even though the turns are
    sitting safely in l0_conversations under the OLD session_id. This walks
    every session in the vault's own bookkeeping (write_scene always stamps
    `session:` in frontmatter) to find what's actually still unflushed,
    so no captured turn is ever permanently orphaned.
    """
    all_sessions = duckdb_client.get_all_session_ids(user_id, tenant_id)
    already_flushed = vault.flushed_session_ids(tenant_id, user_id)

    to_flush = [s for s in all_sessions if s not in already_flushed]
    if current_session_id and current_session_id not in to_flush:
        to_flush.insert(0, current_session_id)

    results = {}
    for session_id in to_flush:
        result = flush_session(tenant_id, user_id, session_id)
        if result is not None:
            results[session_id] = result
    return {"sessions_flushed": len(results), "results": results}
