"""The semantic cache: intercept repeat questions before the pipeline runs.

Two hit paths, degrading in order:
  1. exact — sha256 of the normalized query. Free, works with zero embeddings.
  2. semantic — cosine against cached query vectors at recall.cache_threshold
     (default 0.92). Only when an embedding for the incoming query exists.

Storage is one Redis hash per (tenant, user): field = query hash, value =
JSON {vector, response}. The whole hash carries the TTL, and any new memory
write for the user invalidates it outright — a stale "no, you don't have a
meeting" is worse than a re-run pipeline, so correctness beats hit rate.
Redis being down means every lookup is a miss, never an error.
"""

import hashlib
import json
import logging
import math

from app.config import settings
from app.db.redis_client import get_redis

logger = logging.getLogger(__name__)


def _key(tenant_id: str, user_id: str) -> str:
    return f"semcache:{tenant_id}:{user_id}"


def _query_hash(query: str) -> str:
    return hashlib.sha256(" ".join(query.lower().split()).encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def get(tenant_id: str, user_id: str, query: str, query_vector: list[float] | None = None) -> dict | None:
    try:
        entries = get_redis().hgetall(_key(tenant_id, user_id))
    except Exception:
        return None
    if not entries:
        return None

    exact = entries.get(_query_hash(query))
    if exact:
        return json.loads(exact)["response"]

    if query_vector is None:
        return None
    best, best_sim = None, 0.0
    for raw in entries.values():
        entry = json.loads(raw)
        vector = entry.get("vector")
        if not vector:
            continue
        sim = _cosine(query_vector, vector)
        if sim > best_sim:
            best, best_sim = entry, sim
    if best is not None and best_sim >= settings.recall.cache_threshold:
        return best["response"]
    return None


def put(tenant_id: str, user_id: str, query: str, query_vector: list[float] | None, response: dict) -> None:
    try:
        r = get_redis()
        key = _key(tenant_id, user_id)
        r.hset(key, _query_hash(query), json.dumps({"vector": query_vector, "response": response}))
        r.expire(key, settings.storage.redis.cache_ttl_hours * 3600)
    except Exception:
        logger.info("semantic cache write skipped — redis unavailable")


def invalidate(tenant_id: str, user_id: str) -> None:
    """Called whenever new memories land for the user. Blunt by design."""
    try:
        get_redis().delete(_key(tenant_id, user_id))
    except Exception:
        pass
