from datetime import datetime, timezone

from app.db import duckdb_client


def test_schema_creates_all_four_tables():
    conn = duckdb_client.get_connection()
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {"l0_conversations", "l1_memories", "l1_contradictions", "audit_log"}


def test_insert_and_read_back_l0_messages_ordered_by_turn_index():
    now = datetime.now(timezone.utc)
    duckdb_client.insert_l0_message("m2", "u1", "t1", "s1", "assistant", "hi there", 1, now)
    duckdb_client.insert_l0_message("m1", "u1", "t1", "s1", "user", "hello world", 0, now)

    messages = duckdb_client.get_l0_messages("u1", "t1", "s1")

    assert [m["id"] for m in messages] == ["m1", "m2"], "must come back in turn_index order, not insert order"


def test_l0_messages_scoped_to_session():
    now = datetime.now(timezone.utc)
    duckdb_client.insert_l0_message("m1", "u1", "t1", "s1", "user", "session one", 0, now)
    duckdb_client.insert_l0_message("m2", "u1", "t1", "s2", "user", "session two", 0, now)

    assert [m["content"] for m in duckdb_client.get_l0_messages("u1", "t1", "s1")] == ["session one"]


def test_log_audit_records_action_and_target_ids():
    now = datetime.now(timezone.utc)
    duckdb_client.log_audit("a1", "u1", "t1", "capture", ["m1", "m2"], now, {"source": "test"})

    row = duckdb_client.get_connection().execute(
        "SELECT user_id, tenant_id, action, target_ids FROM audit_log WHERE id = 'a1'"
    ).fetchone()

    assert row[0] == "u1"
    assert row[1] == "t1"
    assert row[2] == "capture"
    assert row[3] == '["m1", "m2"]'
