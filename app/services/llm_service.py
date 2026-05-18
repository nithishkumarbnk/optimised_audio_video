"""
LLM Analysis Service for the AI Proctoring Backend.

Wraps the OpenRouter API (GPT-4o-mini) to translate transcripts and analyze
them for cheating intent. Implements an async retry loop with structured
logging and a 15-second per-request timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import httpx

from app.schemas.audio_schema import LLMResult

logger = logging.getLogger(__name__)

# OpenRouter endpoint
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Maximum number of attempts (1 initial + 2 retries = 3 total)
MAX_RETRIES = 3


class LLMService:
    """Async service that calls OpenRouter to analyze exam transcripts.

    The service translates the transcript to English and evaluates it for
    cheating-related intent using GPT-4o-mini.  All network I/O is
    non-blocking (``httpx.AsyncClient``).

    Parameters
    ----------
    api_key:
        The OpenRouter API key.  The caller is responsible for reading this
        from ``Config.OPENROUTER_API_KEY``; no key is ever hardcoded here.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string.")
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, transcript: str) -> str:
        """Construct the multilingual cheating-analysis prompt.

        The prompt instructs the model to:

        1. Translate the transcript to English (if it is not already).
        2. Analyze the translated text for cheating intent, asking for
           answers, requesting external help, and suspicious conversation
           patterns.
        3. Return a strict JSON object with the four required fields.

        Parameters
        ----------
        transcript:
            The raw transcribed text in any supported language.

        Returns
        -------
        str
            The fully formatted prompt string ready to be sent as the user
            message.
        """
        return (
            "You are an AI exam proctor system.\n\n"
            "Step 1: Translate the transcript to English (if not already).\n"
            "Step 2: Analyze for:\n"
            "- cheating intent\n"
            "- asking for answers\n"
            "- external help\n"
            "- suspicious conversation\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Return STRICT JSON ONLY:\n"
            "{\n"
            '  "translated_text": "...",\n'
            '  "risk": "low | medium | high",\n'
            '  "confidence": 0-1,\n'
            '  "reason": "clear explanation"\n'
            "}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, transcript: str) -> Optional[LLMResult]:
        """Analyze a transcript for cheating intent via the OpenRouter API.

        Sends the transcript to GPT-4o-mini with a structured JSON response
        format.  Retries up to ``MAX_RETRIES`` times on non-200 responses,
        with a 1-second delay between attempts.

        Parameters
        ----------
        transcript:
            The transcribed speech text to analyze.

        Returns
        -------
        LLMResult
            Parsed result containing ``translated_text``, ``risk``,
            ``confidence``, and ``reason`` on success.
        None
            Returned when:
            - All retry attempts are exhausted (non-200 responses).
            - The API response body cannot be parsed as valid JSON.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict and precise exam proctor.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(transcript),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(1, MAX_RETRIES + 1):
            t_start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(
                        _OPENROUTER_URL, headers=headers, json=payload
                    )

                elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

                if response.status_code != 200:
                    # Log the failure and retry (unless this was the last attempt).
                    logger.warning(
                        "LLM API non-200 response",
                        extra={
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "elapsed_ms": elapsed_ms,
                            "error": response.text[:500],  # truncate large bodies
                        },
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)
                    continue

                # --- Successful HTTP response ---
                result_json = response.json()
                content: str = result_json["choices"][0]["message"]["content"]

                try:
                    parsed = json.loads(content)
                    return LLMResult(**parsed)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "LLM response JSON parse error",
                        extra={
                            "attempt": attempt,
                            "elapsed_ms": elapsed_ms,
                            "error": str(exc),
                            "raw_content": content[:500],
                        },
                    )
                    # JSON parse errors are not retried — return immediately.
                    return None

            except httpx.TimeoutException as exc:
                elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
                logger.warning(
                    "LLM API request timed out",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                        "error": str(exc),
                    },
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)

            except Exception as exc:  # noqa: BLE001
                elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
                logger.error(
                    "LLM API unexpected error",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                        "error": str(exc),
                    },
                    exc_info=True,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)

        # All retries exhausted.
        logger.error(
            "LLM API all retries exhausted",
            extra={"max_retries": MAX_RETRIES},
        )
        return None
