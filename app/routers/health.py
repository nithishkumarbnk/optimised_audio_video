"""
Health check router for the AI Proctoring Backend.

Exposes a single ``GET /health`` endpoint that returns the application
uptime and the load status of all four ML model singletons.

Example response::

    {
        "status": "ok",
        "uptime_seconds": 123.45,
        "models": {
            "yolo": true,
            "headset": true,
            "proctor": true,
            "stt": true
        }
    }
"""

from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.headset_service import HeadsetService
from app.services.proctor_service import ProctorService
from app.services.stt_service import STTService
from app.services.yolo_service import YOLOService

# ---------------------------------------------------------------------------
# Module-level start time — set once when this module is first imported.
# ---------------------------------------------------------------------------

#: Monotonic timestamp recorded at module import time; used to compute uptime.
_start_time: float = time.monotonic()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description=(
        "Returns HTTP 200 with the application uptime in seconds and the "
        "load status of each ML model singleton."
    ),
    response_class=JSONResponse,
    tags=["health"],
)
async def health_check() -> JSONResponse:
    """Return the current health status of the application.

    Computes the uptime by subtracting the module-level ``_start_time``
    (set at import) from the current :func:`time.monotonic` value.

    Inspects each service's class-level session/model attribute to
    determine whether the corresponding ML model has been loaded:

    * ``YOLOService._session is not None``
    * ``HeadsetService._session is not None``
    * ``ProctorService._session is not None``
    * ``STTService._model is not None``

    Returns
    -------
    JSONResponse
        HTTP 200 with a JSON body of the form::

            {
                "status": "ok",
                "uptime_seconds": 123.45,
                "models": {
                    "yolo": true,
                    "headset": true,
                    "proctor": true,
                    "stt": true
                }
            }
    """
    uptime_seconds: float = round(time.monotonic() - _start_time, 3)

    models: Dict[str, bool] = {
        "yolo": YOLOService._session is not None,
        "headset": HeadsetService._session is not None,
        "proctor": ProctorService._session is not None,
        "stt": STTService._model is not None,
    }

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "models": models,
        },
    )
