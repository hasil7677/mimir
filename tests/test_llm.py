"""app.core.llm.chat: retry-with-backoff on rate limits and transient 5xx,
no retry (fail fast) on non-retryable errors like a bad API key."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.config import settings
from app.core.llm import LlmUnavailable, chat


def _response(status_code: int, content: str | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if content is not None:
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def test_retries_on_429_then_succeeds():
    responses = [_response(429), _response(429), _response(200, "hi there")]
    with patch("app.core.llm.httpx.post", side_effect=responses) as mock_post, \
         patch("app.core.llm.time.sleep") as mock_sleep:
        result = chat("hello")

    assert result == "hi there"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2  # backoff before retry 2 and retry 3


def test_exhausts_retries_and_raises_llm_unavailable():
    responses = [_response(429)] * 10  # more than _MAX_RETRIES + 1
    with patch("app.core.llm.httpx.post", side_effect=responses), \
         patch("app.core.llm.time.sleep"):
        with pytest.raises(LlmUnavailable):
            chat("hello")


def test_non_retryable_error_fails_immediately_without_retrying():
    with patch("app.core.llm.httpx.post", return_value=_response(401)) as mock_post, \
         patch("app.core.llm.time.sleep") as mock_sleep:
        with pytest.raises(LlmUnavailable):
            chat("hello")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_connection_error_retries_then_succeeds():
    with patch(
        "app.core.llm.httpx.post",
        side_effect=[httpx.ConnectError("refused"), _response(200, "recovered")],
    ) as mock_post, patch("app.core.llm.time.sleep"):
        result = chat("hello")

    assert result == "recovered"
    assert mock_post.call_count == 2


def _vertex_response(status_code: int, content: str | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if content is not None:
        resp.json.return_value = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": content}]}}]
        }
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def test_vertex_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    responses = [_vertex_response(429), _vertex_response(429), _vertex_response(200, "hi there")]
    with patch("app.core.llm.httpx.post", side_effect=responses) as mock_post, \
         patch("app.core.llm.time.sleep") as mock_sleep:
        result = chat("hello")

    assert result == "hi there"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2


def test_vertex_exhausts_retries_and_raises_llm_unavailable(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    responses = [_vertex_response(429)] * 10
    with patch("app.core.llm.httpx.post", side_effect=responses), \
         patch("app.core.llm.time.sleep"):
        with pytest.raises(LlmUnavailable):
            chat("hello")


def test_vertex_non_retryable_error_fails_immediately_without_retrying(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    with patch("app.core.llm.httpx.post", return_value=_vertex_response(403)) as mock_post, \
         patch("app.core.llm.time.sleep") as mock_sleep:
        with pytest.raises(LlmUnavailable):
            chat("hello")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_vertex_uses_x_goog_api_key_header_not_bearer(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    monkeypatch.setattr(settings.llm, "api_key", "test-express-key")
    monkeypatch.setattr(settings.llm, "base_url", "https://aiplatform.googleapis.com")
    monkeypatch.setattr(settings.llm, "model", "gemini-2.5-flash-lite")
    with patch(
        "app.core.llm.httpx.post", return_value=_vertex_response(200, "ok")
    ) as mock_post:
        chat("hello")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {
        "x-goog-api-key": "test-express-key",
        "Content-Type": "application/json",
    }
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["json"]["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert mock_post.call_args[0][0] == (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        "gemini-2.5-flash-lite:generateContent"
    )


# --- truncation: a 200 that quietly contains only part of the answer -------
#
# This is the failure that cost a whole benchmark run. Gemini 2.5 thinks by
# default and Vertex bills those thinking tokens against maxOutputTokens, so
# a 2000-token budget spent 1918 on thinking and returned 78 tokens of a JSON
# array that should have held 44 objects. HTTP 200, finishReason: MAX_TOKENS,
# and the engine read parts[0].text straight through. Extraction's parser is
# built to salvage partial JSON, so it dutifully salvaged one fact and the
# run looked healthy. These tests exist so that silence can't come back.


def _vertex_truncated(thoughts: int = 1918, text: str | None = "[{\"a\": 1}") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    parts = [{"text": text}] if text is not None else []
    resp.json.return_value = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"thoughtsTokenCount": thoughts},
    }
    return resp


def test_vertex_truncation_warns_but_returns_what_arrived(monkeypatch, caplog):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    with patch("app.core.llm.httpx.post", return_value=_vertex_truncated()):
        with caplog.at_level("WARNING"):
            result = chat("hello", max_tokens=2000)

    assert result == '[{"a": 1}'  # partial output is still the best available
    assert "truncated" in caplog.text
    assert "1918" in caplog.text, "the thinking-token spend is the actionable detail"


def test_vertex_all_budget_consumed_by_thinking_raises_rather_than_returning_empty(monkeypatch):
    """No parts at all: the old code raised KeyError out of a function whose
    entire contract is to raise LlmUnavailable so callers take their offline
    path."""
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    with patch("app.core.llm.httpx.post", return_value=_vertex_truncated(text=None)):
        with pytest.raises(LlmUnavailable, match="no text"):
            chat("hello", max_tokens=700)


def test_vertex_sends_thinking_config_by_default(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    monkeypatch.setattr(settings.llm, "thinking_budget", 0)
    with patch("app.core.llm.httpx.post", return_value=_vertex_response(200, "ok")) as mock_post:
        chat("hello", max_tokens=1234)

    cfg = mock_post.call_args[1]["json"]["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingBudget": 0}
    assert cfg["maxOutputTokens"] == 1234


def test_vertex_negative_thinking_budget_omits_the_field(monkeypatch):
    """Escape hatch for non-thinking models that reject the field outright."""
    monkeypatch.setattr(settings.llm, "provider", "vertex")
    monkeypatch.setattr(settings.llm, "thinking_budget", -1)
    with patch("app.core.llm.httpx.post", return_value=_vertex_response(200, "ok")) as mock_post:
        chat("hello")

    assert "thinkingConfig" not in mock_post.call_args[1]["json"]["generationConfig"]


def _openai_truncated(reasoning: int = 900, content: str | None = "[{\"a\": 1}") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "length"}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": reasoning}},
    }
    return resp


def test_openai_truncation_warns(monkeypatch, caplog):
    """The default provider had the identical blind spot — finish_reason
    'length' read straight through as if it were a complete response."""
    monkeypatch.setattr(settings.llm, "provider", "openai")
    with patch("app.core.llm.httpx.post", return_value=_openai_truncated()):
        with caplog.at_level("WARNING"):
            result = chat("hello", max_tokens=512)

    assert result == '[{"a": 1}'
    assert "truncated" in caplog.text
    assert "900" in caplog.text


def test_openai_empty_content_raises_llm_unavailable(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "openai")
    with patch("app.core.llm.httpx.post", return_value=_openai_truncated(content="")):
        with pytest.raises(LlmUnavailable, match="no text"):
            chat("hello")


def test_openai_missing_choices_raises_llm_unavailable(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "openai")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"error": {"message": "quota"}}
    with patch("app.core.llm.httpx.post", return_value=resp):
        with pytest.raises(LlmUnavailable, match="no choices"):
            chat("hello")


def test_complete_response_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(settings.llm, "provider", "openai")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": "all done"}, "finish_reason": "stop"}]}
    with patch("app.core.llm.httpx.post", return_value=resp):
        with caplog.at_level("WARNING"):
            assert chat("hello") == "all done"

    assert "truncated" not in caplog.text
