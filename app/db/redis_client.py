import json
from datetime import datetime, timezone

import redis

from app.config import settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.storage.redis.url, decode_responses=True)
    return _client


def _hot_key(tenant_id: str, user_id: str, session_id: str) -> str:
    # Spec section 5's key pattern table lists `hot:{user_id}:{session_id}`,
    # but section 12 requires every record in every store to carry both
    # tenant_id and user_id so all queries filter on both — omitting
    # tenant_id here would violate that isolation guarantee if the same
    # user_id+session_id pair ever existed under two tenants. Including it.
    return f"hot:{tenant_id}:{user_id}:{session_id}"


def push_turn(tenant_id: str, user_id: str, session_id: str, role: str, content: str) -> None:
    """Level 1 hot memory: appends a turn to the session's list, TTL refreshed
    on every push (config: storage.redis.hot_ttl_hours)."""
    r = get_redis()
    key = _hot_key(tenant_id, user_id, session_id)
    entry = json.dumps({"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()})
    r.rpush(key, entry)
    r.expire(key, settings.storage.redis.hot_ttl_hours * 3600)


def get_recent_turns(tenant_id: str, user_id: str, session_id: str, limit: int = 10) -> list[dict]:
    """Last `limit` turns, oldest -> newest (chronological, ready to render
    straight into a transcript — no reversal needed at the call site)."""
    r = get_redis()
    raw = r.lrange(_hot_key(tenant_id, user_id, session_id), -limit, -1)
    return [json.loads(x) for x in raw]


def erase_user(tenant_id: str, user_id: str) -> int:
    """Deletes hot-memory and semantic-cache keys for the user (SCAN, not
    KEYS — never blocks a shared Redis)."""
    r = get_redis()
    deleted = 0
    for pattern in (f"hot:{tenant_id}:{user_id}:*", f"semcache:{tenant_id}:{user_id}"):
        for key in r.scan_iter(match=pattern):
            deleted += r.delete(key)
    return deleted
