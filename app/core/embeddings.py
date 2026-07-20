"""Embeddings, two ways.

"openai" — any OpenAI-compatible /embeddings endpoint (Ollama included,
keyless). Dense vectors only; a real network call every time.

"fastembed" — runs fully in-process (ONNX runtime), no server, no API key,
no per-call network traffic. Produces BOTH a dense vector (semantic search)
and a sparse/BM25 vector (Qdrant's own sparse model) per text, which is what
lets vector_store hand both to Qdrant's native hybrid fusion instead of us
hand-rolling BM25 + RRF ourselves. Model weights download once on first use
(a one-time, anonymous file fetch — no user content sent) and are cached
under fastembed_cache_dir; every call after that is pure local compute.

provider "none" or an unreachable endpoint both surface as
EmbeddingsUnavailable — the recall pipeline treats that as "skip the vector
leg, keyword search still works", never as a failure.
"""

from functools import lru_cache
from pathlib import Path

import httpx

from app.config import settings


class EmbeddingsUnavailable(Exception):
    pass


def embed(texts: list[str]) -> list[list[float]]:
    """Dense vectors only — used by the semantic cache (cosine similarity)
    and as the fallback path for providers without sparse support."""
    if settings.embedding.provider == "none" or not texts:
        raise EmbeddingsUnavailable("embedding provider disabled")

    if settings.embedding.provider == "fastembed":
        return [v["dense"] for v in embed_hybrid(texts)]

    headers = {}
    if settings.embedding.api_key:
        headers["Authorization"] = f"Bearer {settings.embedding.api_key}"

    try:
        response = httpx.post(
            f"{settings.embedding.base_url.rstrip('/')}/embeddings",
            headers=headers,
            json={"model": settings.embedding.model, "input": texts},
            timeout=30,
        )
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        raise EmbeddingsUnavailable(str(exc)) from exc

    data = sorted(response.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def supports_hybrid() -> bool:
    """True only for providers that can produce a sparse vector alongside
    the dense one — that's what unlocks Qdrant's native fusion path in
    vector_store instead of the DuckDB-BM25 + hand-rolled RRF fallback."""
    return settings.embedding.provider == "fastembed"


@lru_cache(maxsize=1)
def _fastembed_models():
    # Imported lazily so a missing/unconfigured fastembed install never
    # breaks providers that don't use it (openai, none).
    from fastembed import SparseTextEmbedding, TextEmbedding

    cache_dir = str(Path(settings.embedding.fastembed_cache_dir).expanduser())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dense = TextEmbedding(settings.embedding.fastembed_dense_model, cache_dir=cache_dir)
    sparse = SparseTextEmbedding(settings.embedding.fastembed_sparse_model, cache_dir=cache_dir)
    return dense, sparse


def embed_hybrid(texts: list[str]) -> list[dict]:
    """-> [{"dense": [float, ...], "sparse": {"indices": [...], "values": [...]}}]
    one per text, in order. Raises EmbeddingsUnavailable if fastembed isn't
    the configured provider or the models fail to load (e.g. no network for
    the one-time download, out of disk, etc.) — same degrade contract as embed().
    """
    if settings.embedding.provider != "fastembed" or not texts:
        raise EmbeddingsUnavailable("fastembed not configured as the embedding provider")

    try:
        dense_model, sparse_model = _fastembed_models()
        dense_vecs = list(dense_model.embed(texts))
        sparse_vecs = list(sparse_model.embed(texts))
    except Exception as exc:  # model load/inference failure — degrade, don't crash the pipeline
        raise EmbeddingsUnavailable(str(exc)) from exc

    return [
        {
            "dense": dense_vecs[i].tolist(),
            "sparse": {"indices": sparse_vecs[i].indices.tolist(), "values": sparse_vecs[i].values.tolist()},
        }
        for i in range(len(texts))
    ]
