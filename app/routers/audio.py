"""
Audio analysis router for the AI Proctoring Backend.

Exposes a single endpoint:

    POST /api/v1/audio/analyze

The endpoint accepts a multipart upload containing an audio file and a
session identifier, runs the full STT → Rule Engine → LLM → Risk Engine
pipeline, and returns a structured :class:`~app.schemas.audio_schema.AudioAnalysisResponse`.

Requirements addressed: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
                        14.2, 14.4, 14.6, 14.7, 15.3
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional, Set

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_config
from app.core.security import sanitize_filename
from app.schemas.audio_schema import AudioAnalysisResponse
from app.services.llm_service import LLMService
from app.services.risk_engine import classify, compute_score
from app.services.rule_engine import check_rules
from app.services.stt_service import STTService
from app.utils.response_utils import make_audio_response, make_error_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MIME types accepted by the audio analysis endpoint (Requirement 3.2).
_ALLOWED_MIME_TYPES: Set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
}

#: Directory where temporary audio uploads are stored.
#: Resolves to <repo>/project/uploads/ relative to this file's location.
#: parents[0] = routers/, parents[1] = app/, parents[2] = project/
_UPLOADS_DIR: Path = Path(__file__).resolve().parents[2] / "uploads"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AudioAnalysisResponse,
    summary="Analyze an audio file for cheating signals",
    description=(
        "Accepts a multipart audio upload and a session identifier. "
        "Runs the STT → Rule Engine → LLM → Risk Engine pipeline and "
        "returns a structured risk assessment."
    ),
    responses={
        200: {"description": "Analysis completed successfully."},
        413: {"description": "Audio file exceeds the configured size limit."},
        422: {"description": "Unsupported MIME type or missing form fields."},
        500: {"description": "Unexpected internal server error."},
    },
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze."),
    session_id: str = Form(..., description="Unique identifier for the proctoring session."),
) -> JSONResponse:
    """Analyze an uploaded audio file for exam-cheating signals.

    Processing pipeline
    -------------------
    1. Validate the MIME type against the allowed set.
    2. Read the file bytes and check the size against ``MAX_AUDIO_SIZE_MB``.
    3. Sanitize the filename to prevent directory traversal.
    4. Persist the file to ``uploads/<uuid>_<safe_name>``.
    5. Run STT transcription.
    6. If STT returns ``None``, return an early LOW-risk response.
    7. Run the Rule Engine on the transcript.
    8. Run the LLM analysis service.
    9. Compute a weighted risk score and classify the risk level.
    10. Log structured audit fields.
    11. Delete the temporary file (always, in a ``finally`` block).

    Parameters
    ----------
    file:
        The uploaded audio file.  Must have a MIME type in
        ``{"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4"}``.
    session_id:
        Unique identifier for the proctoring session.  Used for
        structured logging and session-state recording.

    Returns
    -------
    JSONResponse
        A JSON body matching :class:`~app.schemas.audio_schema.AudioAnalysisResponse`.

    Raises
    ------
    HTTPException(422)
        When the uploaded file's content type is not in the allowed set.
    HTTPException(413)
        When the uploaded file exceeds ``MAX_AUDIO_SIZE_MB``.
    HTTPException(500)
        On any unhandled exception during pipeline execution.
    """
    config = get_config()
    temp_path: Optional[Path] = None

    try:
        # ------------------------------------------------------------------
        # 1. MIME type validation (Requirement 3.2, 14.2)
        # ------------------------------------------------------------------
        content_type: str = file.content_type or ""
        if content_type not in _ALLOWED_MIME_TYPES:
            logger.warning(
                "Audio upload rejected: unsupported MIME type",
                extra={
                    "session_id": session_id,
                    "content_type": content_type,
                    "allowed": sorted(_ALLOWED_MIME_TYPES),
                },
            )
            return JSONResponse(
                status_code=422,
                content=make_error_response(
                    "UNSUPPORTED_MIME_TYPE",
                    f"Unsupported audio MIME type: '{content_type}'. "
                    f"Allowed types: {sorted(_ALLOWED_MIME_TYPES)}.",
                ),
            )

        # ------------------------------------------------------------------
        # 2. Read file bytes and size check (Requirement 3.3, 14.4)
        # ------------------------------------------------------------------
        audio_bytes: bytes = await file.read()
        file_size_bytes: int = len(audio_bytes)
        max_bytes: int = config.MAX_AUDIO_SIZE_MB * 1024 * 1024

        if file_size_bytes > max_bytes:
            logger.warning(
                "Audio upload rejected: file too large",
                extra={
                    "session_id": session_id,
                    "file_size_bytes": file_size_bytes,
                    "max_bytes": max_bytes,
                    "max_mb": config.MAX_AUDIO_SIZE_MB,
                },
            )
            return JSONResponse(
                status_code=413,
                content=make_error_response(
                    "FILE_TOO_LARGE",
                    f"Audio file size ({file_size_bytes} bytes) exceeds the "
                    f"maximum allowed size of {config.MAX_AUDIO_SIZE_MB} MB.",
                ),
            )

        # ------------------------------------------------------------------
        # 3. Sanitize filename (Requirement 14.6)
        # ------------------------------------------------------------------
        raw_name: str = file.filename or "upload.bin"
        safe_name: str = sanitize_filename(raw_name)

        # ------------------------------------------------------------------
        # 4. Persist to uploads/<uuid>_<safe_name> (Requirement 3.4)
        # ------------------------------------------------------------------
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        unique_prefix: str = uuid.uuid4().hex
        temp_path = _UPLOADS_DIR / f"{unique_prefix}_{safe_name}"
        temp_path.write_bytes(audio_bytes)

        logger.info(
            "Audio file saved for processing",
            extra={
                "session_id": session_id,
                "file_size_bytes": file_size_bytes,
                "temp_path": str(temp_path),
            },
        )

        # ------------------------------------------------------------------
        # 5. STT transcription (Requirement 3.5)
        # ------------------------------------------------------------------
        stt_start: float = time.monotonic()
        native_text: Optional[str] = STTService.transcribe(str(temp_path))
        stt_elapsed_ms: float = round((time.monotonic() - stt_start) * 1000, 1)

        # ------------------------------------------------------------------
        # 6. Early exit when STT produces no transcript (Requirement 3.7)
        # ------------------------------------------------------------------
        if native_text is None:
            logger.info(
                "STT produced no transcript; returning LOW-risk response",
                extra={
                    "session_id": session_id,
                    "file_size_bytes": file_size_bytes,
                    "stt_elapsed_ms": stt_elapsed_ms,
                    "risk_score": 0,
                    "risk_level": "LOW",
                },
            )
            return JSONResponse(
                content=make_audio_response(
                    session_id=session_id,
                    native_text=None,
                    translated_text=None,
                    rule_flags=[],
                    llm_result=None,
                    risk_score=0,
                    risk_level="LOW",
                ),
            )

        # ------------------------------------------------------------------
        # 7. Rule Engine (Requirement 3.5)
        # ------------------------------------------------------------------
        rule_flags = check_rules(native_text)

        # ------------------------------------------------------------------
        # 8. LLM analysis (Requirement 3.5, 15.3)
        # ------------------------------------------------------------------
        llm_service = LLMService(get_config().OPENROUTER_API_KEY)
        llm_start: float = time.monotonic()
        llm_result = await llm_service.analyze(native_text)
        llm_elapsed_ms: float = round((time.monotonic() - llm_start) * 1000, 1)

        # ------------------------------------------------------------------
        # 9. Risk scoring (Requirement 3.5)
        # ------------------------------------------------------------------
        # Combine rule flags with any LLM-derived risk signal.
        events_for_scoring = list(rule_flags)
        if llm_result and llm_result.risk in ("medium", "high"):
            events_for_scoring.append("suspicious_transcript")

        risk_score: int = compute_score(events_for_scoring)
        risk_level: str = classify(risk_score)

        # Derive translated_text from llm_result for the response.
        translated_text: Optional[str] = (
            llm_result.translated_text if llm_result else None
        )

        # ------------------------------------------------------------------
        # 10. Structured audit log (Requirement 3.8, 14.7)
        # ------------------------------------------------------------------
        logger.info(
            "Audio analysis complete",
            extra={
                "session_id": session_id,
                "file_size_bytes": file_size_bytes,
                "stt_elapsed_ms": stt_elapsed_ms,
                "llm_elapsed_ms": llm_elapsed_ms,
                "risk_score": risk_score,
                "risk_level": risk_level,
            },
        )

        return JSONResponse(
            content=make_audio_response(
                session_id=session_id,
                native_text=native_text,
                translated_text=translated_text,
                rule_flags=rule_flags,
                llm_result=llm_result.model_dump() if llm_result else None,
                risk_score=risk_score,
                risk_level=risk_level,
            ),
        )

    except Exception:
        # ------------------------------------------------------------------
        # Unhandled exception handler (Requirement 14.7)
        # ------------------------------------------------------------------
        logger.error(
            "Unhandled exception in audio analysis endpoint",
            extra={"session_id": session_id},
            exc_info=True,
        )
        # Log full traceback to server logs; never expose it to the client.
        traceback.format_exc()  # already captured by exc_info=True above

        return JSONResponse(
            status_code=500,
            content=make_error_response(
                "INTERNAL_ERROR",
                "An unexpected error occurred while processing the audio file.",
            ),
        )

    finally:
        # ------------------------------------------------------------------
        # Always delete the temporary file (Requirement 3.4)
        # ------------------------------------------------------------------
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(
                    "Temporary audio file deleted",
                    extra={"temp_path": str(temp_path)},
                )
            except OSError as exc:
                logger.warning(
                    "Failed to delete temporary audio file",
                    extra={"temp_path": str(temp_path), "error": str(exc)},
                )
