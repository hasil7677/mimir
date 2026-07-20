"""Qdrant in embedded/local mode: a directory on disk, no server, no Docker.

Two independent schemas live side by side in the same local Qdrant path,
under different collection names, so switching embedding providers can never
produce a "wrong vector config" error on an existing collection:

  - `{collection_name}`         — single unnamed dense vector. Used by the
    OpenAI-compatible provider (dense-only; no sparse vectors available).
  - `{collection_name}_hybrid`  — named "dense" + "sparse" vectors. Used by
    the fastembed provider, queried via Qdrant's native prefetch + RRF
    fusion instead of us hand-rolling BM25 + our own rrf_merge.

Both are created lazily on first write, since dimension/schema depends on
whatever embedding model ends up configured — nothing is guessed at startup.
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


def _hybrid_collection_name() -> str:
    return f"{settings.storage.qdrant.collection_name}_hybrid"


def _ensure_collection(dim: int) -> None:
    """Legacy dense-only schema (OpenAI-compatible provider)."""
    client = get_client()
    name = settings.storage.qdrant.collection_name
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def _ensure_hybrid_collection(dense_dim: int) -> None:
    """Named dense+sparse schema (fastembed provider) — what makes Qdrant's
    native prefetch/fusion query possible."""
    client = get_client()
    name = _hybrid_collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
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
    """Dense-only path (OpenAI-compatible provider). facts: dicts with at
    least id/content/type. Payload duplicates the scoping ids so filtering
    never depends on a join back to DuckDB."""
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


def upsert_facts_hybrid(tenant_id: str, user_id: str, facts: list[dict], embeddings: list[dict]) -> None:
    """embeddings: [{"dense": [...], "sparse": {"indices": [...], "values": [...]}}],
    one per fact, from embeddings.embed_hybrid()."""
    if not facts:
        return
    _ensure_hybrid_collection(dense_dim=len(embeddings[0]["dense"]))
    points = [
        qm.PointStruct(
            id=fact["id"],
            vector={
                "dense": emb["dense"],
                "sparse": qm.SparseVector(indices=emb["sparse"]["indices"], values=emb["sparse"]["values"]),
            },
            payload={"tenant_id": tenant_id, "user_id": user_id,
                     "content": fact["content"], "type": fact["type"]},
        )
        for fact, emb in zip(facts, embeddings)
    ]
    get_client().upsert(collection_name=_hybrid_collection_name(), points=points)


def erase_user(tenant_id: str, user_id: str) -> None:
    """Deletes every point for this tenant/user from BOTH schemas — a user
    may have switched embedding providers at some point, so full erasure has
    to reach whichever collection(s) actually exist. A missing collection
    means nothing was ever embedded there — a no-op, not an error."""
    client = get_client()
    for name in (settings.storage.qdrant.collection_name, _hybrid_collection_name()):
        if client.collection_exists(name):
            client.delete(collection_name=name, points_selector=qm.FilterSelector(filter=_both_ids_filter(tenant_id, user_id)))


def search(tenant_id: str, user_id: str, vector: list[float], top_k: int = 20) -> list[dict]:
    """Dense-only search (OpenAI-compatible provider). -> [{id, semantic_score}]
    or [] if the collection doesn't exist yet (nothing embedded so far — a
    valid state, not an error)."""
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


def search_hybrid(tenant_id: str, user_id: str, dense: list[float], sparse: dict, top_k: int = 20) -> list[dict]:
    """Qdrant's native hybrid search: one prefetch over dense, one over
    sparse (BM25), fused server-side with RRF — no DuckDB-fts, no our own
    rrf_merge, Qdrant does the fusion. sparse: {"indices": [...], "values": [...]}.
    -> [{id, semantic_score}] (the fused RRF score) or [] if nothing indexed yet.
    """
    client = get_client()
    name = _hybrid_collection_name()
    if not client.collection_exists(name):
        return []
    ids_filter = _both_ids_filter(tenant_id, user_id)
    hits = client.query_points(
        collection_name=name,
        prefetch=[
            qm.Prefetch(query=dense, using="dense", filter=ids_filter, limit=top_k),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                using="sparse", filter=ids_filter, limit=top_k,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=ids_filter,
        limit=top_k,
        with_payload=False,
    ).points
    return [{"id": str(h.id), "semantic_score": h.score} for h in hits]
