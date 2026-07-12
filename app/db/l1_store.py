"""l1_memories: the fact store inside the DuckDB cold archive.

Keyword search is real BM25 via DuckDB's fts extension when it loads
(cached locally after first install), else a plain SQL term-frequency
fallback — offline-first means search degrades, never disappears.

The FTS index is static in DuckDB, so writes mark it dirty and the next
search rebuilds it lazily — cheap at local scale, and always consistent.
"""

import json
import logging
import re

from app.db.duckdb_client import get_connection, to_utc_naive

logger = logging.getLogger(__name__)

_fts_state = {"available": None, "dirty": True}


def insert_fact(
    fact_id: str,
    user_id: str,
    tenant_id: str,
    content: str,
    fact_type: str,
    priority: int,
    scene_name: str,
    session_id: str,
    source_ids: list[str],
    created_at,
) -> None:
    get_connection().execute(
        """
        INSERT INTO l1_memories
            (id, user_id, tenant_id, content, type, priority, scene_name, session_id,
             source_ids, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [fact_id, user_id, tenant_id, content, fact_type, priority, scene_name,
         session_id, json.dumps(source_ids), to_utc_naive(created_at), to_utc_naive(created_at)],
    )
    _fts_state["dirty"] = True


def _fts_ready() -> bool:
    conn = get_connection()
    if _fts_state["available"] is None:
        try:
            conn.execute("INSTALL fts; LOAD fts;")
            _fts_state["available"] = True
        except Exception:
            logger.warning("duckdb fts extension unavailable — falling back to SQL term scoring")
            _fts_state["available"] = False
    if _fts_state["available"] and _fts_state["dirty"]:
        conn.execute("PRAGMA create_fts_index('l1_memories', 'id', 'content', overwrite=1)")
        _fts_state["dirty"] = False
    return bool(_fts_state["available"])


_ROW_COLUMNS = ["id", "content", "type", "priority", "scene_name", "created_at", "access_count", "score"]


def search_keyword(tenant_id: str, user_id: str, query: str, top_k: int = 20) -> list[dict]:
    """Active facts for this tenant/user ranked by keyword relevance.
    Score scale differs between BM25 and the fallback — callers must treat it
    as a ranking signal only, which is all RRF needs."""
    conn = get_connection()
    if _fts_ready():
        rows = conn.execute(
            """
            SELECT id, content, type, priority, scene_name, created_at, access_count,
                   fts_main_l1_memories.match_bm25(id, ?) AS score
            FROM l1_memories
            WHERE tenant_id = ? AND user_id = ? AND is_active
            ORDER BY score DESC NULLS LAST
            LIMIT ?
            """,
            [query, tenant_id, user_id, top_k],
        ).fetchall()
        return [dict(zip(_ROW_COLUMNS, r)) for r in rows if r[-1] is not None]

    # Fallback: count how many query terms appear in each fact (case-insensitive).
    terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not terms:
        return []
    match_expr = " + ".join("CAST(contains(lower(content), ?) AS INTEGER)" for _ in terms)
    rows = conn.execute(
        f"""
        SELECT id, content, type, priority, scene_name, created_at, access_count,
               ({match_expr}) AS score
        FROM l1_memories
        WHERE tenant_id = ? AND user_id = ? AND is_active
        ORDER BY score DESC, created_at DESC
        LIMIT ?
        """,
        [*terms, tenant_id, user_id, top_k],
    ).fetchall()
    return [dict(zip(_ROW_COLUMNS, r)) for r in rows if r[-1] and r[-1] > 0]


def get_facts_by_ids(tenant_id: str, user_id: str, fact_ids: list[str]) -> list[dict]:
    if not fact_ids:
        return []
    placeholders = ", ".join("?" for _ in fact_ids)
    rows = get_connection().execute(
        f"""
        SELECT id, content, type, priority, scene_name, created_at, access_count, 0.0 AS score
        FROM l1_memories
        WHERE tenant_id = ? AND user_id = ? AND is_active AND id IN ({placeholders})
        """,
        [tenant_id, user_id, *fact_ids],
    ).fetchall()
    return [dict(zip(_ROW_COLUMNS, r)) for r in rows]


def supersede(old_id: str, new_id: str) -> None:
    """Marks the old fact replaced — never deleted. Same stance as everywhere:
    is_active=false keeps it out of search, superseded_by keeps the lineage."""
    get_connection().execute(
        "UPDATE l1_memories SET is_active = FALSE, superseded_by = ?, updated_at = now() WHERE id = ?",
        [new_id, old_id],
    )
    _fts_state["dirty"] = True


def record_contradiction(contradiction_id: str, memory_id_a: str, memory_id_b: str, detected_at) -> None:
    """Flag-and-log, never auto-resolve — surfacing a conflict for review beats
    silently deciding which of the user's own statements to believe."""
    get_connection().execute(
        "INSERT INTO l1_contradictions (id, memory_id_a, memory_id_b, detected_at) VALUES (?, ?, ?, ?)",
        [contradiction_id, memory_id_a, memory_id_b, to_utc_naive(detected_at)],
    )


def count_active_facts(tenant_id: str, user_id: str) -> int:
    row = get_connection().execute(
        "SELECT count(*) FROM l1_memories WHERE tenant_id = ? AND user_id = ? AND is_active",
        [tenant_id, user_id],
    ).fetchone()
    return row[0]


def get_top_facts(tenant_id: str, user_id: str, limit: int = 30, types: list[str] | None = None) -> list[dict]:
    """Highest-priority active facts, for persona synthesis."""
    type_clause = ""
    params: list = [tenant_id, user_id]
    if types:
        type_clause = f"AND type IN ({', '.join('?' for _ in types)})"
        params.extend(types)
    params.append(limit)
    rows = get_connection().execute(
        f"""
        SELECT id, content, type, priority, scene_name, created_at, access_count, 0.0 AS score
        FROM l1_memories
        WHERE tenant_id = ? AND user_id = ? AND is_active {type_clause}
        ORDER BY priority DESC, access_count DESC, created_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(zip(_ROW_COLUMNS, r)) for r in rows]


def bump_access(fact_ids: list[str]) -> None:
    if not fact_ids:
        return
    placeholders = ", ".join("?" for _ in fact_ids)
    get_connection().execute(
        f"UPDATE l1_memories SET access_count = access_count + 1, last_accessed = now() "
        f"WHERE id IN ({placeholders})",
        fact_ids,
    )
