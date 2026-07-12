import uuid

from app.db import redis_client

from .conftest import requires_redis


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@requires_redis
def test_recent_turns_come_back_in_chronological_order():
    tenant, user, session = _id("tenant"), _id("user"), _id("session")

    redis_client.push_turn(tenant, user, session, "user", "first message")
    redis_client.push_turn(tenant, user, session, "assistant", "second message")
    redis_client.push_turn(tenant, user, session, "user", "third message")

    turns = redis_client.get_recent_turns(tenant, user, session)

    assert [t["content"] for t in turns] == ["first message", "second message", "third message"]


@requires_redis
def test_recent_turns_respects_limit():
    tenant, user, session = _id("tenant"), _id("user"), _id("session")

    for i in range(5):
        redis_client.push_turn(tenant, user, session, "user", f"message {i}")

    turns = redis_client.get_recent_turns(tenant, user, session, limit=2)

    assert [t["content"] for t in turns] == ["message 3", "message 4"]


@requires_redis
def test_hot_memory_isolated_across_tenants_with_same_user_and_session_id():
    user, session = _id("user"), _id("session")
    tenant_a, tenant_b = _id("tenant"), _id("tenant")

    redis_client.push_turn(tenant_a, user, session, "user", "tenant A's message")
    redis_client.push_turn(tenant_b, user, session, "user", "tenant B's message")

    turns_a = redis_client.get_recent_turns(tenant_a, user, session)
    turns_b = redis_client.get_recent_turns(tenant_b, user, session)

    assert [t["content"] for t in turns_a] == ["tenant A's message"]
    assert [t["content"] for t in turns_b] == ["tenant B's message"]
