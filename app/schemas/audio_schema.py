"""
Pydantic schemas for the audio analysis pipeline.

This module defines the request and response models used by the
``POST /api/v1/audio/analyze`` endpoint and the LLM service.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    """Structured output returned by the LLM cheating-analysis service.

    Attributes:
        translated_text: English translation of the transcribed speech.
        risk: Qualitative risk label assigned by the LLM.
            Expected values: ``"low"``, ``"medium"``, or ``"high"``.
        confidence: Model confidence in the risk assessment, in the
            range ``[0.0, 1.0]``.
        reason: Human-readable explanation of why the risk label was
            assigned.
    """

    translated_text: str = Field(
        ...,
        description="English translation of the transcribed speech.",
    )
    risk: str = Field(
        ...,
        description='Qualitative risk label: "low", "medium", or "high".',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the risk assessment (0.0 – 1.0).",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation for the assigned risk label.",
    )


class AudioAnalysisResponse(BaseModel):
    """Response model for the audio analysis endpoint.

    Returned by ``POST /api/v1/audio/analyze`` after the full
    STT → Rule Engine → LLM → Risk Engine pipeline has been executed.

    Attributes:
        session_id: Unique identifier for the proctoring session.
        timestamp: UTC timestamp at which the analysis was completed.
        native_text: Raw transcription in the detected source language.
            ``None`` when the STT service could not produce a transcript.
        translated_text: English translation of the transcription.
            ``None`` when the LLM service was not invoked or failed.
        rule_flags: List of rule-engine flags triggered by the transcript
            (e.g. ``"keyword_detected"``, ``"long_speech"``).
        llm_result: Structured LLM analysis result.
            ``None`` when the LLM service was not invoked or failed.
        risk_score: Numeric risk score computed by the Risk Engine.
        risk_level: Categorical risk level derived from ``risk_score``.
            Expected values: ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
    """

    session_id: str = Field(
        ...,
        description="Unique identifier for the proctoring session.",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp at which the analysis was completed.",
    )
    native_text: Optional[str] = Field(
        default=None,
        description=(
            "Raw transcription in the detected source language. "
            "None when STT produced no output."
        ),
    )
    translated_text: Optional[str] = Field(
        default=None,
        description=(
            "English translation of the transcription. "
            "None when the LLM service was not invoked or failed."
        ),
    )
    rule_flags: List[str] = Field(
        default_factory=list,
        description=(
            'Rule-engine flags triggered by the transcript '
            '(e.g. "keyword_detected", "long_speech").'
        ),
    )
    llm_result: Optional[LLMResult] = Field(
        default=None,
        description=(
            "Structured LLM analysis result. "
            "None when the LLM service was not invoked or failed."
        ),
    )
    risk_score: int = Field(
        ...,
        ge=0,
        description="Numeric risk score computed by the Risk Engine.",
    )
    risk_level: str = Field(
        ...,
        description='Categorical risk level: "LOW", "MEDIUM", or "HIGH".',
    )
