from datetime import datetime, timezone
from pathlib import Path

import duckdb

from app.config import settings

_conn: duckdb.DuckDBPyConnection | None = None


def to_utc_naive(ts: datetime) -> datetime:
    """DuckDB TIMESTAMP is timezone-naive, and handing it a tz-aware datetime
    makes it convert to *local* time first — silently shifting every stored
    timestamp by the machine's UTC offset (surfaced live as facts aged
    '-1 days ago' on an IST machine). All writes normalize through this;
    every reader interprets naive as UTC."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts

_SCHEMA = """
CREATE TABLE IF NOT EXISTS l0_conversations (
    id VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL, session_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL, content TEXT NOT NULL,
    turn_index INTEGER NOT NULL, recorded_at TIMESTAMP NOT NULL,
    is_archived BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS l1_memories (
    id VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL, content TEXT NOT NULL,
    type VARCHAR NOT NULL, priority INTEGER NOT NULL,
    scene_name VARCHAR, session_id VARCHAR, source_ids JSON,
    created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0, last_accessed TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE, superseded_by VARCHAR
);

CREATE TABLE IF NOT EXISTS l1_contradictions (
    id VARCHAR PRIMARY KEY, memory_id_a VARCHAR NOT NULL,
    memory_id_b VARCHAR NOT NULL, detected_at TIMESTAMP NOT NULL,
    resolved BOOLEAN DEFAULT FALSE, resolution VARCHAR
);

CREATE TABLE IF NOT EXISTS audit_log (
    id VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL, action VARCHAR NOT NULL,
    target_ids JSON, performed_at TIMESTAMP NOT NULL, metadata JSON
);
"""


def get_connection() -> duckdb.DuckDBPyConnection:
    """Single shared connection — DuckDB serializes access across threads on
    one connection object, matching how the rest of this codebase treats its
    embedded/local stores (one client, lazily created, reused everywhere).
    """
    global _conn
    if _conn is None:
        db_path = Path(settings.storage.duckdb.path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(db_path))
        _conn.execute(_SCHEMA)
    return _conn


def ensure_schema() -> None:
    """Idempotent — safe to call on every startup."""
    get_connection()


def insert_l0_message(
    id: str,
    user_id: str,
    tenant_id: str,
    session_id: str,
    role: str,
    content: str,
    turn_index: int,
    recorded_at,
) -> None:
    get_connection().execute(
        """
        INSERT INTO l0_conversations
            (id, user_id, tenant_id, session_id, role, content, turn_index, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [id, user_id, tenant_id, session_id, role, content, turn_index, to_utc_naive(recorded_at)],
    )


def get_l0_messages(user_id: str, tenant_id: str, session_id: str) -> list[dict]:
    rows = get_connection().execute(
        """
        SELECT id, role, content, turn_index, recorded_at FROM l0_conversations
        WHERE user_id = ? AND tenant_id = ? AND session_id = ?
        ORDER BY turn_index ASC
        """,
        [user_id, tenant_id, session_id],
    ).fetchall()
    columns = ["id", "role", "content", "turn_index", "recorded_at"]
    return [dict(zip(columns, row)) for row in rows]


def get_all_session_ids(user_id: str, tenant_id: str) -> list[str]:
    """Every session_id that has captured turns, oldest first — used to find
    sessions an adapter process captured but never flushed."""
    rows = get_connection().execute(
        "SELECT DISTINCT session_id, min(recorded_at) AS first_turn FROM l0_conversations "
        "WHERE user_id = ? AND tenant_id = ? GROUP BY session_id ORDER BY first_turn",
        [user_id, tenant_id],
    ).fetchall()
    return [r[0] for r in rows]


def get_all_l0(user_id: str, tenant_id: str) -> list[dict]:
    rows = get_connection().execute(
        "SELECT id, session_id, role, content, turn_index, recorded_at FROM l0_conversations "
        "WHERE user_id = ? AND tenant_id = ? ORDER BY recorded_at, turn_index",
        [user_id, tenant_id],
    ).fetchall()
    cols = ["id", "session_id", "role", "content", "turn_index", "recorded_at"]
    return [dict(zip(cols, r)) for r in rows]


def get_all_l1(user_id: str, tenant_id: str) -> list[dict]:
    rows = get_connection().execute(
        "SELECT id, content, type, priority, scene_name, session_id, created_at, "
        "access_count, is_active, superseded_by FROM l1_memories "
        "WHERE user_id = ? AND tenant_id = ? ORDER BY created_at",
        [user_id, tenant_id],
    ).fetchall()
    cols = ["id", "content", "type", "priority", "scene_name", "session_id",
            "created_at", "access_count", "is_active", "superseded_by"]
    return [dict(zip(cols, r)) for r in rows]


def erase_user(user_id: str, tenant_id: str) -> dict:
    """Deletes the user's content rows. The audit log is intentionally NOT
    erased — it holds ids and actions, not content, and the erasure itself
    must remain provable. Contradiction rows referencing the user's facts go
    first (they'd dangle otherwise)."""
    conn = get_connection()
    fact_ids = [r[0] for r in conn.execute(
        "SELECT id FROM l1_memories WHERE user_id = ? AND tenant_id = ?", [user_id, tenant_id]
    ).fetchall()]
    if fact_ids:
        placeholders = ", ".join("?" for _ in fact_ids)
        conn.execute(
            f"DELETE FROM l1_contradictions WHERE memory_id_a IN ({placeholders}) "
            f"OR memory_id_b IN ({placeholders})",
            [*fact_ids, *fact_ids],
        )
    # execute() returns the connection itself, so each DELETE's count must be
    # fetched before the next statement runs — deferring the fetch reads the
    # wrong statement's count (found the hard way).
    l0_deleted = _rowcount(conn.execute(
        "DELETE FROM l0_conversations WHERE user_id = ? AND tenant_id = ?", [user_id, tenant_id]
    ))
    l1_deleted = _rowcount(conn.execute(
        "DELETE FROM l1_memories WHERE user_id = ? AND tenant_id = ?", [user_id, tenant_id]
    ))
    return {"l0_deleted": l0_deleted, "l1_deleted": l1_deleted, "fact_ids": len(fact_ids)}


def _rowcount(cursor) -> int:
    try:
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def log_audit(
    id: str,
    user_id: str,
    tenant_id: str,
    action: str,
    target_ids: list[str],
    performed_at,
    metadata: dict | None = None,
) -> None:
    import json

    get_connection().execute(
        """
        INSERT INTO audit_log (id, user_id, tenant_id, action, target_ids, performed_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [id, user_id, tenant_id, action, json.dumps(target_ids), to_utc_naive(performed_at), json.dumps(metadata or {})],
    )
