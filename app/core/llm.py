"""LLM access via any OpenAI-compatible /chat/completions endpoint.

Local-first consequence: Ollama (http://localhost:11434/v1) is a first-class
target and needs no API key, so a missing key is not an error here — the
request is simply sent without an Authorization header. Callers that can
degrade gracefully (scene synthesis) catch LlmUnavailable and fall back.
"""

import httpx

from app.config import settings


class LlmUnavailable(Exception):
    """Raised when no LLM endpoint is reachable/usable — signals callers to
    take their offline fallback path rather than fail the pipeline."""


def chat(prompt: str, max_tokens: int = 512) -> str:
    headers = {}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key}"

    try:
        response = httpx.post(
            f"{settings.llm.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": settings.llm.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=settings.llm.timeout_ms / 1000,
        )
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        raise LlmUnavailable(str(exc)) from exc

    return response.json()["choices"][0]["message"]["content"].strip()
