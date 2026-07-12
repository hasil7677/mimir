"""Embeddings via any OpenAI-compatible /embeddings endpoint (Ollama included,
keyless). provider "none" or an unreachable endpoint both surface as
EmbeddingsUnavailable — the recall pipeline treats that as "skip the vector
leg, keyword search still works", never as a failure.
"""

import httpx

from app.config import settings


class EmbeddingsUnavailable(Exception):
    pass


def embed(texts: list[str]) -> list[list[float]]:
    if settings.embedding.provider == "none" or not texts:
        raise EmbeddingsUnavailable("embedding provider disabled")

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
