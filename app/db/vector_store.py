"""Qdrant in embedded/local mode: a directory on disk, no server, no Docker.

Collection is created lazily on first write because its dimension comes from
whatever embedding model is configured — creating it at startup would force
an embedding call (or a guess) before one is ever needed.
"""

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.storage.qdrant.mode == "server":
            _client = QdrantClient(url=settings.storage.qdrant.path)
        else:
            path = Path(settings.storage.qdrant.path).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(path))
    return _client


def _ensure_collection(dim: int) -> None:
    client = get_client()
    name = settings.storage.qdrant.collection_name
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def _both_ids_filter(tenant_id: str, user_id: str) -> qm.Filter:
    """Every query filters on tenant AND user — same non-negotiable as
    everywhere else in this codebase."""
    return qm.Filter(
        must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        ]
    )


def upsert_facts(tenant_id: str, user_id: str, facts: list[dict], vectors: list[list[float]]) -> None:
    """facts: dicts with at least id/content/type. Payload duplicates the
    scoping ids so filtering never depends on a join back to DuckDB."""
    if not facts:
        return
    _ensure_collection(dim=len(vectors[0]))
    points = [
        qm.PointStruct(
            id=fact["id"],
            vector=vector,
            payload={"tenant_id": tenant_id, "user_id": user_id,
                     "content": fact["content"], "type": fact["type"]},
        )
        for fact, vector in zip(facts, vectors)
    ]
    get_client().upsert(collection_name=settings.storage.qdrant.collection_name, points=points)


def erase_user(tenant_id: str, user_id: str) -> None:
    """Deletes every point for this tenant/user; a missing collection means
    nothing was ever embedded — a no-op, not an error."""
    client = get_client()
    name = settings.storage.qdrant.collection_name
    if not client.collection_exists(name):
        return
    client.delete(collection_name=name, points_selector=qm.FilterSelector(filter=_both_ids_filter(tenant_id, user_id)))


def search(tenant_id: str, user_id: str, vector: list[float], top_k: int = 20) -> list[dict]:
    """-> [{id, semantic_score}] or [] if the collection doesn't exist yet
    (nothing embedded so far — a valid state, not an error)."""
    client = get_client()
    if not client.collection_exists(settings.storage.qdrant.collection_name):
        return []
    hits = client.query_points(
        collection_name=settings.storage.qdrant.collection_name,
        query=vector,
        query_filter=_both_ids_filter(tenant_id, user_id),
        limit=top_k,
        with_payload=False,
    ).points
    return [{"id": str(h.id), "semantic_score": h.score} for h in hits]
