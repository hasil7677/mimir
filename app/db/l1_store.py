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


def fts_mode() -> str:
    """Which keyword-search implementation is actually live: "bm25" once
    DuckDB's fts extension has loaded, "sql_fallback" if it failed to, or
    "unknown" before the first search has forced the question.

    Deliberately read-only — it never runs the INSTALL/LOAD probe itself, so
    calling it from a diagnostics path can't decide which mode a later search
    ends up taking. The distinction matters because the fallback is a plain
    term-presence count, not a relevance ranking: tuning retrieval while
    silently on it means tuning the wrong thing entirely.
    """
    if _fts_state["available"] is None:
        return "unknown"
    return "bm25" if _fts_state["available"] else "sql_fallback"


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
            # ERROR, not WARNING: this silently swaps real BM25 ranking for a
            # term-presence count for the entire life of the process. It reads
            # like a minor degradation in a log and behaves like a retrieval
            # rewrite — it has to be impossible to scroll past, and it's
            # mirrored into every recall result as fts_mode for the same reason.
            logger.error(
                "duckdb fts extension failed to load — keyword search has DEGRADED to "
                "SQL term counting, not BM25. Retrieval quality is affected for the rest "
                "of this process; recall results will report fts_mode='sql_fallback'.",
                exc_info=True,
            )
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


def get_predecessor_chains(tenant_id: str, user_id: str, fact_ids: list[str]) -> dict[str, list[dict]]:
    """For each given (currently active) fact id, walks `superseded_by`
    backward to find every fact it replaced, oldest-first.

    Every other query in this module filters `is_active` — a fact's history
    is invisible by default, which is right for "what's true now" queries
    but wrong for "how did this change" queries, where the old, hidden
    version is exactly what's being asked about. This is the explicit,
    opt-in read path for that: nothing here mutates is_active or changes
    default recall behavior, it only surfaces history when a caller asks
    for it by id.
    """
    if not fact_ids:
        return {}
    conn = get_connection()
    chains: dict[str, list[dict]] = {fid: [] for fid in fact_ids}
    frontier = [(fid, fid) for fid in fact_ids]
    seen_ids = set(fact_ids)

    for _ in range(5):  # cap chain depth — sane bound, not expected to matter in practice
        if not frontier:
            break
        current_ids = list({cid for _, cid in frontier})
        placeholders = ", ".join("?" for _ in current_ids)
        rows = conn.execute(
            f"""
            SELECT id, content, type, superseded_by
            FROM l1_memories
            WHERE tenant_id = ? AND user_id = ? AND superseded_by IN ({placeholders})
            """,
            [tenant_id, user_id, *current_ids],
        ).fetchall()

        by_superseded_by: dict[str, list[tuple]] = {}
        for row in rows:
            by_superseded_by.setdefault(row[3], []).append(row)

        next_frontier = []
        for root_id, current_id in frontier:
            for fid, content, fact_type, _ in by_superseded_by.get(current_id, []):
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                chains[root_id].insert(0, {"id": fid, "content": content, "type": fact_type})
                next_frontier.append((root_id, fid))
        frontier = next_frontier

    return {k: v for k, v in chains.items() if v}


def bump_access(fact_ids: list[str]) -> None:
    if not fact_ids:
        return
    placeholders = ", ".join("?" for _ in fact_ids)
    get_connection().execute(
        f"UPDATE l1_memories SET access_count = access_count + 1, last_accessed = now() "
        f"WHERE id IN ({placeholders})",
        fact_ids,
    )
