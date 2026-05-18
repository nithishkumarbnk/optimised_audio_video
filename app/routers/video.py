"""
Video frame analysis router for the AI Proctoring Backend.

Exposes a single REST endpoint:

    POST /api/v1/video/analyze-frame

The endpoint accepts a multipart form upload containing a raw image frame
and a session identifier.  It runs the frame through the full three-stage
inference pipeline (YOLO → Headset → Proctor), aggregates behavioural
events, computes a risk score, records the event in the Session Manager,
and returns a structured :class:`~app.schemas.video_schema.VideoAnalysisResponse`.

Requirements addressed: 7.1, 7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 14.3, 15.4
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.video_schema import VideoAnalysisResponse
from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService
from app.services.risk_engine import classify, compute_score
from app.services.session_manager import SessionEvent, SessionManager
from app.services.yolo_service import YOLOService
from app.utils import image_utils
from app.utils.response_utils import make_video_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed MIME types for uploaded frames (Requirement 7.2)
# ---------------------------------------------------------------------------

_ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})

# ---------------------------------------------------------------------------
# Module-level SessionManager singleton
# ---------------------------------------------------------------------------

_session_manager: SessionManager = SessionManager()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/video", tags=["video"])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/analyze-frame",
    response_model=VideoAnalysisResponse,
    summary="Analyse a single video frame for proctoring events",
    description=(
        "Accepts a JPEG or PNG image frame together with a session identifier. "
        "Runs the frame through the YOLO object-detection, Headset classification, "
        "and Proctor behavioural-analysis pipelines in sequence.  Aggregates all "
        "detected events, computes a weighted risk score, records the result in the "
        "Session Manager, and returns a structured VideoAnalysisResponse."
    ),
    status_code=200,
)
async def analyze_frame(
    frame: UploadFile = File(..., description="JPEG or PNG image frame to analyse."),
    session_id: str = Form(..., description="Unique identifier for the proctoring session."),
) -> JSONResponse:
    """Analyse a single video frame and return a proctoring risk assessment.

    Pipeline
    --------
    1. **MIME validation** — reject non-JPEG/PNG uploads with HTTP 422.
    2. **Decode** — convert raw bytes to a BGR numpy array via
       :func:`~app.utils.image_utils.decode_image`; return HTTP 422 on failure.
    3. **YOLO inference** — detect objects and count persons.
    4. **Headset inference** — classify whether a headset is present.
    5. **Proctor inference** — detect behavioural events.
    6. **Event aggregation** — combine all signals into a single event list.
    7. **Risk scoring** — compute numeric score and categorical level.
    8. **Session recording** — persist the event in the Session Manager.
    9. **Structured logging** — emit a single INFO log line with key metrics.

    Args:
        frame:      Multipart file upload containing the raw image bytes.
        session_id: Unique identifier for the proctoring session this frame
                    belongs to.

    Returns:
        :class:`~app.schemas.video_schema.VideoAnalysisResponse` containing
        the detected events, person count, risk score, risk level, and the
        full list of YOLO detections.

    Raises:
        HTTPException (422): If the uploaded file's MIME type is not
            ``image/jpeg`` or ``image/png``, or if the image bytes cannot
            be decoded by OpenCV.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.9, 7.10, 14.3, 15.4
    """
    # ------------------------------------------------------------------
    # 1. MIME type validation (Requirement 7.2)
    # ------------------------------------------------------------------
    content_type: str = frame.content_type or ""
    if content_type not in _ALLOWED_MIME_TYPES:
        logger.warning(
            "Rejected frame upload: unsupported MIME type",
            extra={"session_id": session_id, "content_type": content_type},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "UNSUPPORTED_MIME_TYPE",
                "message": (
                    f"Unsupported content type '{content_type}'. "
                    "Only 'image/jpeg' and 'image/png' are accepted."
                ),
            },
        )

    # ------------------------------------------------------------------
    # 2. Read raw bytes and decode image (Requirement 7.3)
    # ------------------------------------------------------------------
    raw_bytes: bytes = await frame.read()
    frame_size_bytes: int = len(raw_bytes)

    try:
        import numpy as np  # local import keeps top-level imports clean
        bgr_frame: np.ndarray = image_utils.decode_image(raw_bytes)
    except ValueError as exc:
        logger.warning(
            "Failed to decode uploaded frame",
            extra={"session_id": session_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "IMAGE_DECODE_FAILED",
                "message": f"Could not decode the uploaded image: {exc}",
            },
        ) from exc

    # ------------------------------------------------------------------
    # 3. YOLO inference — detect objects, count persons (Requirement 7.4)
    # ------------------------------------------------------------------
    yolo_start: float = time.monotonic()
    detections = YOLOService.detect(bgr_frame)
    yolo_elapsed_ms: float = (time.monotonic() - yolo_start) * 1_000

    person_count: int = sum(
        1 for d in detections if d.class_name == "person"
    )

    # ------------------------------------------------------------------
    # 4. Headset inference (Requirement 7.4)
    # ------------------------------------------------------------------
    headset_detected: bool = HeadsetService.classify(bgr_frame)

    # ------------------------------------------------------------------
    # 5. Proctor inference (Requirement 7.4)
    # ------------------------------------------------------------------
    proctor_start: float = time.monotonic()
    proctor_events: List[str] = ProctorService.analyze(bgr_frame)
    proctor_elapsed_ms: float = (time.monotonic() - proctor_start) * 1_000

    # ------------------------------------------------------------------
    # 6. Event aggregation (Requirement 7.4)
    # ------------------------------------------------------------------
    events: List[str] = []

    if person_count == 0:
        events.append("no_face")

    if person_count > 1:
        events.append("multiple_persons")

    if headset_detected:
        events.append("headset_detected")

    events.extend(proctor_events)

    # ------------------------------------------------------------------
    # 7. Risk scoring (Requirement 7.9)
    # ------------------------------------------------------------------
    risk_score: int = compute_score(events)
    risk_level: str = classify(risk_score)

    # ------------------------------------------------------------------
    # 8. Session recording (Requirement 7.10)
    # ------------------------------------------------------------------
    session_event = SessionEvent(
        event_name=", ".join(events) if events else "clean_frame",
        timestamp=datetime.now(tz=timezone.utc),
        risk_score=risk_score,
        source="video",
    )
    await _session_manager.record_event(session_id, session_event, risk_level)

    # ------------------------------------------------------------------
    # 9. Structured logging (Requirement 15.4)
    # ------------------------------------------------------------------
    logger.info(
        "Video frame analysed",
        extra={
            "session_id": session_id,
            "frame_size_bytes": frame_size_bytes,
            "yolo_elapsed_ms": round(yolo_elapsed_ms, 2),
            "proctor_elapsed_ms": round(proctor_elapsed_ms, 2),
            "event_count": len(events),
            "risk_score": risk_score,
        },
    )

    # ------------------------------------------------------------------
    # 10. Build and return response (Requirement 7.5)
    # ------------------------------------------------------------------
    from fastapi.responses import JSONResponse as _JSONResponse
    from datetime import datetime as _dt, timezone as _tz
    response_dict = {
        "session_id": session_id,
        "timestamp": _dt.now(tz=_tz.utc).isoformat(),
        "events": events,
        "person_count": person_count,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bounding_box": {
                    "x1": d.bounding_box.x1,
                    "y1": d.bounding_box.y1,
                    "x2": d.bounding_box.x2,
                    "y2": d.bounding_box.y2,
                },
            }
            for d in detections
        ],
    }
    return _JSONResponse(content=response_dict)
