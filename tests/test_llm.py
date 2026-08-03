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
