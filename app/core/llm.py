"""LLM access via any OpenAI-compatible /chat/completions endpoint.

Local-first consequence: Ollama (http://localhost:11434/v1) is a first-class
target and needs no API key, so a missing key is not an error here — the
request is simply sent without an Authorization header. Callers that can
degrade gracefully (scene synthesis) catch LlmUnavailable and fall back.
"""

import time

import httpx

from app.config import settings


class LlmUnavailable(Exception):
    """Raised when no LLM endpoint is reachable/usable — signals callers to
    take their offline fallback path rather than fail the pipeline."""


# A rate limit (429) or a transient 5xx is not "the endpoint is unusable" —
# it's "try again in a moment". Treating either as immediate LlmUnavailable
# meant a single 429 permanently dropped a call to the offline/verbatim
# fallback, silently degrading extraction quality even though the same
# request would very likely succeed a few seconds later. Observed live
# against a rate-limited free-tier endpoint: sustained 429s turned what
# should be real LLM extraction into mostly-offline degradation for the
# whole run, not a one-off blip.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_BACKOFF_SECONDS = 1.0


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float) -> httpx.Response:
    last_error: str = "no attempt made"
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except (httpx.HTTPError, OSError) as exc:
            # connection-level failure (timeout, refused, DNS, ...) — retryable
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_BACKOFF_SECONDS * (2**attempt))
            continue

        if response.status_code in _RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code} (retryable)"
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_BACKOFF_SECONDS * (2**attempt))
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # a real, non-retryable failure (bad key, malformed request, ...) —
            # fail immediately, retrying would just waste time on the same error
            raise LlmUnavailable(str(exc)) from exc

        return response

    raise LlmUnavailable(last_error)


def _chat_vertex(prompt: str, max_tokens: int) -> str:
    # Vertex AI Express Mode: API-key auth via x-goog-api-key (not Bearer),
    # no project/location in the URL, and the native generateContent request/
    # response shape — not OpenAI's chat/completions. See engine/HANDOFF.md
    # for why an OpenAI-compatible key can't be pointed at this endpoint.
    headers = {"x-goog-api-key": settings.llm.api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
    }
    url = f"{settings.llm.base_url.rstrip('/')}/v1/publishers/google/models/{settings.llm.model}:generateContent"
    response = _post_with_retry(url, headers, payload, settings.llm.timeout_ms / 1000)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _chat_openai_compatible(prompt: str, max_tokens: int) -> str:
    headers = {}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key}"

    payload = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    url = f"{settings.llm.base_url.rstrip('/')}/chat/completions"
    response = _post_with_retry(url, headers, payload, settings.llm.timeout_ms / 1000)
    return response.json()["choices"][0]["message"]["content"].strip()


def chat(prompt: str, max_tokens: int = 512) -> str:
    if settings.llm.provider == "vertex":
        return _chat_vertex(prompt, max_tokens)
    return _chat_openai_compatible(prompt, max_tokens)
