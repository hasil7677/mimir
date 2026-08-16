"""LLM access via any OpenAI-compatible /chat/completions endpoint.

Local-first consequence: Ollama (http://localhost:11434/v1) is a first-class
target and needs no API key, so a missing key is not an error here — the
request is simply sent without an Authorization header. Callers that can
degrade gracefully (scene synthesis) catch LlmUnavailable and fall back.
"""

import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


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
    generation_config: dict = {"maxOutputTokens": max_tokens, "temperature": 0}
    if settings.llm.thinking_budget >= 0:
        # See LlmConfig.thinking_budget: without this, a thinking model spends
        # nearly the whole output budget reasoning and the real response gets
        # truncated mid-token. Negative means "don't send it, let the model
        # decide" — for non-thinking models that never needed the field.
        generation_config["thinkingConfig"] = {"thinkingBudget": settings.llm.thinking_budget}

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": generation_config}
    url = f"{settings.llm.base_url.rstrip('/')}/v1/publishers/google/models/{settings.llm.model}:generateContent"
    response = _post_with_retry(url, headers, payload, settings.llm.timeout_ms / 1000)
    return _extract_vertex_text(response.json(), max_tokens)


def _extract_vertex_text(body: dict, max_tokens: int) -> str:
    """Pull the text out of a generateContent response, and say so loudly when
    the model ran out of room.

    A truncated response is not an error at the HTTP level — it comes back 200
    with `finishReason: MAX_TOKENS` and whatever fragment fits. Reading only
    parts[0].text silently accepts that fragment, which is how a 44-fact
    extraction became a 1-fact extraction across an entire benchmark run with
    nothing in the logs. A model that thinks past its budget can also return a
    candidate with no parts at all, which used to raise KeyError from inside a
    function whose whole contract is to raise LlmUnavailable instead.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        raise LlmUnavailable(f"vertex returned no candidates: {str(body)[:200]}")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if finish_reason == "MAX_TOKENS":
        thoughts = (body.get("usageMetadata") or {}).get("thoughtsTokenCount") or 0
        logger.warning(
            "vertex response truncated at maxOutputTokens=%d (%d spent on thinking, "
            "%d chars of usable output). Raise the caller's max_tokens or lower "
            "llm.thinking_budget — output past this point was silently discarded.",
            max_tokens, thoughts, len(text),
        )
    if not text:
        raise LlmUnavailable(
            f"vertex returned no text (finishReason={finish_reason}) — "
            f"the entire output budget was consumed before any response was emitted"
        )
    return text


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
    return _extract_openai_text(response.json(), max_tokens)


def _extract_openai_text(body: dict, max_tokens: int) -> str:
    """Same contract as _extract_vertex_text, for the OpenAI-compatible shape.

    This path had exactly the blind spot that cost the Vertex path a 44-fact
    extraction: a response truncated at the token limit returns 200 with
    `finish_reason: "length"` and a fragment, and reading straight through to
    choices[0].message.content accepts that fragment as if it were complete.
    Extraction hands the fragment to a parser designed to salvage partial JSON,
    so the failure surfaces as quietly-fewer-facts rather than as an error.
    It is the default provider, so this is the path most deployments run.
    """
    choices = body.get("choices") or []
    if not choices:
        raise LlmUnavailable(f"llm returned no choices: {str(body)[:200]}")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    text = ((choice.get("message") or {}).get("content") or "").strip()

    if finish_reason == "length":
        # Reasoning models on OpenAI-compatible endpoints bill their reasoning
        # against the same completion budget, the same way Vertex bills
        # thinking tokens — so report it when the endpoint tells us.
        details = (body.get("usage") or {}).get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") or 0
        logger.warning(
            "llm response truncated at max_tokens=%d (%d spent on reasoning, "
            "%d chars of usable output). Raise the caller's max_tokens — "
            "output past this point was silently discarded.",
            max_tokens, reasoning, len(text),
        )
    if not text:
        raise LlmUnavailable(
            f"llm returned no text (finish_reason={finish_reason}) — "
            f"the entire output budget was consumed before any response was emitted"
        )
    return text


def chat(prompt: str, max_tokens: int = 512) -> str:
    if settings.llm.provider == "vertex":
        return _chat_vertex(prompt, max_tokens)
    return _chat_openai_compatible(prompt, max_tokens)
