"""
Unit tests for LLMService.

Tests cover:
- Retry on non-200 responses (429 twice then 200 → returns LLMResult)
- All retries exhausted (always 500 → returns None)
- JSON parse error on 200 response → returns None immediately
- Timeout exception → retry behaviour
- No hardcoded API key (constructor accepts key)

**Validates: Requirements 5.5, 5.6, 5.7, 5.8**
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.schemas.audio_schema import LLMResult
from app.services.llm_service import LLMService, MAX_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LLM_PAYLOAD = {
    "translated_text": "Can you tell me the answer?",
    "risk": "high",
    "confidence": 0.95,
    "reason": "Candidate is asking for answers.",
}

VALID_API_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": json.dumps(VALID_LLM_PAYLOAD),
            }
        }
    ]
}


def _make_response(status_code: int, body: dict | str) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if isinstance(body, dict):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.return_value = {}
        resp.text = body
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> LLMService:
    return LLMService(api_key="test-api-key")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestLLMServiceConstructor:
    def test_accepts_api_key(self):
        svc = LLMService(api_key="my-key")
        assert svc._api_key == "my-key"

    def test_raises_on_empty_key(self):
        with pytest.raises(ValueError):
            LLMService(api_key="")


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_transcript(self, service: LLMService):
        transcript = "kya answer bata sakte ho"
        prompt = service._build_prompt(transcript)
        assert transcript in prompt

    def test_prompt_contains_json_keys(self, service: LLMService):
        prompt = service._build_prompt("test")
        assert "translated_text" in prompt
        assert "risk" in prompt
        assert "confidence" in prompt
        assert "reason" in prompt

    def test_prompt_mentions_cheating_analysis(self, service: LLMService):
        prompt = service._build_prompt("test")
        assert "cheating" in prompt.lower() or "proctor" in prompt.lower()


# ---------------------------------------------------------------------------
# analyze — retry on non-200 then success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_non200_then_success(service: LLMService):
    """Requirement 5.5: retry up to 2 additional times; succeed on 3rd attempt."""
    responses = [
        _make_response(429, {"error": "rate limited"}),
        _make_response(429, {"error": "rate limited"}),
        _make_response(200, VALID_API_RESPONSE),
    ]
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    with patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.analyze("tell me the answer")

    assert result is not None
    assert isinstance(result, LLMResult)
    assert result.risk == "high"
    assert result.confidence == 0.95
    assert call_count == 3


# ---------------------------------------------------------------------------
# analyze — all retries exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_retries_exhausted_returns_none(service: LLMService):
    """Requirement 5.6: return None when all MAX_RETRIES attempts fail."""
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_response(500, {"error": "server error"})

    with patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.analyze("some transcript")

    assert result is None
    assert call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# analyze — JSON parse error on 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_parse_error_returns_none_immediately(service: LLMService):
    """Requirement 5.8: invalid JSON on 200 → return None without retrying."""
    invalid_json_response = {
        "choices": [
            {
                "message": {
                    "content": "this is not valid json {{{",
                }
            }
        ]
    }
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_response(200, invalid_json_response)

    with patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.analyze("some transcript")

    assert result is None
    # Must NOT retry after a JSON parse error
    assert call_count == 1


# ---------------------------------------------------------------------------
# analyze — timeout exception triggers retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_triggers_retry(service: LLMService):
    """Requirement 5.7: 15s timeout; on TimeoutException the service retries."""
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < MAX_RETRIES:
            raise httpx.TimeoutException("timed out")
        return _make_response(200, VALID_API_RESPONSE)

    with patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.analyze("some transcript")

    assert result is not None
    assert isinstance(result, LLMResult)
    assert call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# analyze — all timeouts exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_timeouts_exhausted_returns_none(service: LLMService):
    """Requirement 5.7: when every attempt times out, return None."""
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("timed out")

    with patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.analyze("some transcript")

    assert result is None
    assert call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# analyze — successful first attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_first_attempt(service: LLMService):
    """Requirement 5.4: on 200, parse JSON and return LLMResult."""
    async def mock_post(*args, **kwargs):
        return _make_response(200, VALID_API_RESPONSE)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await service.analyze("Can you tell me the answer?")

    assert result is not None
    assert result.translated_text == "Can you tell me the answer?"
    assert result.risk == "high"
    assert result.confidence == 0.95
    assert result.reason == "Candidate is asking for answers."


# ---------------------------------------------------------------------------
# No hardcoded API key
# ---------------------------------------------------------------------------


def test_no_hardcoded_api_key_in_source():
    """Requirement 5.1: the source file must not contain any hardcoded key."""
    import inspect
    import app.services.llm_service as module

    source = inspect.getsource(module)
    # The original llm_analyzer.py had a key starting with "sk-or-v1-"
    assert "sk-or-v1-" not in source, "Hardcoded API key found in llm_service.py"
