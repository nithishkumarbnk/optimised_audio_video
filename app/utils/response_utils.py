"""
response_utils.py — Helpers for building consistent API response dicts.

These functions return plain dicts whose keys match the Pydantic schema field
names defined in:
  - app/schemas/audio_schema.py  (AudioAnalysisResponse, LLMResult)
  - app/schemas/video_schema.py  (VideoAnalysisResponse, Detection)

Using plain dicts (rather than schema instances) keeps this module free of
circular-import issues: the schemas may not yet exist when this module is
imported, and routers can still validate/serialize the returned dict through
their own response_model declarations.

Requirements addressed: 3.6, 7.5, 14.7
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # Only imported for type-checking tools; never executed at runtime.
    from app.schemas.audio_schema import LLMResult  # noqa: F401
    from app.schemas.video_schema import Detection  # noqa: F401


def make_audio_response(
    session_id: str,
    native_text: Optional[str],
    translated_text: Optional[str],
    rule_flags: List[str],
    llm_result: Optional[Any],
    risk_score: int,
    risk_level: str,
) -> Dict[str, Any]:
    """Build a response dict matching ``AudioAnalysisResponse`` field names.

    The ``timestamp`` field is set to the current UTC time at the moment this
    function is called, formatted as an ISO-8601 string with timezone info so
    that downstream serializers (Pydantic, ``json.dumps``) handle it correctly.

    Args:
        session_id:       Unique identifier for the proctoring session.
        native_text:      Raw transcript returned by the STT service, or
                          ``None`` when transcription produced no output.
        translated_text:  English translation produced by the LLM service, or
                          ``None`` when LLM analysis was skipped or failed.
        rule_flags:       List of rule-engine flag strings (e.g.
                          ``["keyword_detected", "long_speech"]``).
        llm_result:       ``LLMResult`` instance (or equivalent dict) returned
                          by the LLM service, or ``None``.
        risk_score:       Numeric risk score computed by the Risk Engine.
        risk_level:       Categorical risk label — one of ``"LOW"``,
                          ``"MEDIUM"``, or ``"HIGH"``.

    Returns:
        A ``dict`` whose keys match every field of ``AudioAnalysisResponse``:
        ``session_id``, ``timestamp``, ``native_text``, ``translated_text``,
        ``rule_flags``, ``llm_result``, ``risk_score``, ``risk_level``.

    Requirements: 3.6
    """
    return {
        "session_id": session_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "native_text": native_text,
        "translated_text": translated_text,
        "rule_flags": rule_flags,
        "llm_result": llm_result,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def make_video_response(
    session_id: str,
    events: List[str],
    person_count: int,
    risk_score: int,
    risk_level: str,
    detections: List[Any],
) -> Dict[str, Any]:
    """Build a response dict matching ``VideoAnalysisResponse`` field names.

    The ``timestamp`` field is set to the current UTC time at the moment this
    function is called.

    Args:
        session_id:    Unique identifier for the proctoring session.
        events:        List of event name strings detected in the frame (e.g.
                       ``["no_face", "looking_away"]``).
        person_count:  Number of persons detected by the YOLO service.
        risk_score:    Numeric risk score computed by the Risk Engine.
        risk_level:    Categorical risk label — one of ``"LOW"``,
                       ``"MEDIUM"``, or ``"HIGH"``.
        detections:    List of ``Detection`` instances (or equivalent dicts)
                       returned by the YOLO service.

    Returns:
        A ``dict`` whose keys match every field of ``VideoAnalysisResponse``:
        ``session_id``, ``timestamp``, ``events``, ``person_count``,
        ``risk_score``, ``risk_level``, ``detections``.

    Requirements: 7.5
    """
    return {
        "session_id": session_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "events": events,
        "person_count": person_count,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detections": detections,
    }


def make_error_response(code: str, message: str) -> Dict[str, str]:
    """Build a structured error response dict.

    Internal stack traces are intentionally excluded from this dict so that
    they are never exposed to API clients (Requirement 14.7).  Full tracebacks
    must be written to server logs by the caller before invoking this helper.

    Args:
        code:    A short machine-readable error code string (e.g.
                 ``"UNSUPPORTED_MIME_TYPE"`` or ``"INTERNAL_ERROR"``).
        message: A human-readable description of the error suitable for
                 returning to the API client.

    Returns:
        ``{"error": code, "message": message}``

    Requirements: 14.7
    """
    return {"error": code, "message": message}
