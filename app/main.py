"""
FastAPI application entry point for the AI Proctoring Backend.

Responsibilities
----------------
* Create the :class:`fastapi.FastAPI` application instance.
* Register :class:`BodySizeLimitMiddleware` to reject oversized request bodies.
* Register :class:`starlette.middleware.cors.CORSMiddleware` for cross-origin
  access control.
* Define startup and shutdown lifecycle hooks that initialise all singleton
  services, create required directories, and start the worker pool.
* Mount all routers at their versioned prefixes.
* Register a global exception handler that logs full tracebacks without
  leaking them to clients.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.8, 2.9, 14.1, 14.7
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import get_config
from app.core.logger import configure_logging
from app.routers.audio import router as audio_router
from app.routers.health import router as health_router
from app.routers.video import router as video_router
from app.routers.websocket import router as ws_router
from app.routers.websocket import set_managers
from app.services.headset_service import HeadsetService
from app.services.llm_service import LLMService
from app.services.proctor_service import ProctorService
from app.services.session_manager import SessionManager
from app.services.stt_service import STTService
from app.services.websocket_manager import WebSocketManager
from app.services.worker_manager import WorkerManager
from app.services.yolo_service import YOLOService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths — resolved relative to this file's parent directory (app/)
# ---------------------------------------------------------------------------

_APP_DIR: Path = Path(__file__).parent
_YOLO_MODEL_PATH: Path = _APP_DIR / "models" / "yolov8n.pt"
_HEADSET_MODEL_PATH: Path = _APP_DIR / "models" / "headset_model.pt"
_PROCTOR_MODEL_PATH: Path = _APP_DIR / "models" / "proctoring.pt"

# ---------------------------------------------------------------------------
# Module-level singletons (populated during startup)
# ---------------------------------------------------------------------------

session_manager: SessionManager | None = None
websocket_manager: WebSocketManager | None = None
llm_service: LLMService | None = None
worker_manager: WorkerManager | None = None


# ---------------------------------------------------------------------------
# Body-size limit middleware
# ---------------------------------------------------------------------------


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose ``Content-Length`` header exceeds the configured limit.

    Reads the ``Content-Length`` header from the incoming request and returns
    HTTP 413 (Request Entity Too Large) when the declared body size exceeds
    ``MAX_AUDIO_SIZE_MB * 1024 * 1024`` bytes.  Requests without a
    ``Content-Length`` header are passed through unchanged.

    Args:
        app:     The next ASGI application in the middleware stack.
        max_mb:  Maximum allowed body size in megabytes.
    """

    def __init__(self, app: ASGIApp, max_mb: int) -> None:
        super().__init__(app)
        self._max_bytes: int = max_mb * 1024 * 1024

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check ``Content-Length`` and either reject or forward the request.

        Args:
            request:   The incoming HTTP request.
            call_next: Callable that forwards the request to the next handler.

        Returns:
            HTTP 413 :class:`~fastapi.responses.JSONResponse` when the body
            exceeds the limit, otherwise the response from the downstream
            handler.
        """
        content_length_header: str | None = request.headers.get("content-length")
        if content_length_header is not None:
            try:
                content_length: int = int(content_length_header)
            except ValueError:
                content_length = 0

            if content_length > self._max_bytes:
                logger.warning(
                    "Request rejected: body too large",
                    extra={
                        "content_length": content_length,
                        "max_bytes": self._max_bytes,
                        "path": request.url.path,
                    },
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "REQUEST_ENTITY_TOO_LARGE",
                        "message": (
                            f"Request body ({content_length} bytes) exceeds the "
                            f"maximum allowed size of {self._max_bytes} bytes."
                        ),
                    },
                )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance with middleware
        registered and routers mounted.  Lifecycle hooks are attached via
        ``@app.on_event`` decorators defined at module level.
    """
    application = FastAPI(
        title="AI Proctoring Backend",
        version="1.0.0",
        description=(
            "Real-time AI-powered exam proctoring backend. "
            "Analyses webcam video frames and microphone audio to detect "
            "suspicious behaviour and compute risk scores."
        ),
    )

    config = get_config()

    # ------------------------------------------------------------------
    # Middleware — order matters: outermost middleware is applied first.
    # CORSMiddleware must wrap BodySizeLimitMiddleware so that CORS
    # pre-flight OPTIONS requests are handled before the size check.
    # ------------------------------------------------------------------

    # 1. Body-size limit (inner — applied after CORS)
    application.add_middleware(BodySizeLimitMiddleware, max_mb=config.MAX_AUDIO_SIZE_MB)

    # 2. CORS (outer — applied first)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    # Audio router already carries prefix /api/v1/audio
    application.include_router(audio_router)

    # Video router already carries prefix /api/v1/video
    application.include_router(video_router)

    # WebSocket router — no additional prefix
    application.include_router(ws_router)

    # Health router — no additional prefix
    application.include_router(health_router)

    # ------------------------------------------------------------------
    # Static frontend — serve frontend/ at /ui
    # ------------------------------------------------------------------
    _frontend_dir = Path(__file__).parent.parent / "frontend"
    if _frontend_dir.exists():
        application.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

    return application


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# Startup lifecycle hook
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    """Initialise all services and start background workers on application startup.

    Startup sequence
    ----------------
    a. Configure the root logger.
    b. Create ``uploads/`` and ``logs/`` directories.
    c. Initialise :class:`~app.services.stt_service.STTService`.
    d. Initialise :class:`~app.services.yolo_service.YOLOService`.
    e. Initialise :class:`~app.services.headset_service.HeadsetService`.
    f. Initialise :class:`~app.services.proctor_service.ProctorService`.
    g. Create :class:`~app.services.session_manager.SessionManager` and
       :class:`~app.services.websocket_manager.WebSocketManager` singletons.
    h. Create :class:`~app.services.llm_service.LLMService` instance.
    i. Create :class:`~app.services.worker_manager.WorkerManager` instance.
    j. Start the worker pool.
    k. Inject managers into the WebSocket router via :func:`set_managers`.
    l. Start the WebSocket heartbeat as an asyncio background task.
    """
    global session_manager, websocket_manager, llm_service, worker_manager

    config = get_config()

    # ------------------------------------------------------------------
    # a. Configure logging
    # ------------------------------------------------------------------
    configure_logging(config.LOG_LEVEL)
    logger.info("AI Proctoring Backend starting up")

    # ------------------------------------------------------------------
    # b. Create required directories
    # ------------------------------------------------------------------
    uploads_dir = Path("uploads")
    logs_dir = Path("logs")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Directories ensured",
        extra={"uploads": str(uploads_dir), "logs": str(logs_dir)},
    )

    # ------------------------------------------------------------------
    # c. Initialise STT service
    # ------------------------------------------------------------------
    logger.info("Initialising STTService")
    STTService.initialize()

    # ------------------------------------------------------------------
    # d. Initialise YOLO service
    # ------------------------------------------------------------------
    logger.info("Initialising YOLOService", extra={"model": str(_YOLO_MODEL_PATH)})
    YOLOService.initialize(str(_YOLO_MODEL_PATH), config.ENABLE_GPU)

    # ------------------------------------------------------------------
    # e. Initialise Headset service
    # ------------------------------------------------------------------
    logger.info(
        "Initialising HeadsetService", extra={"model": str(_HEADSET_MODEL_PATH)}
    )
    HeadsetService.initialize(str(_HEADSET_MODEL_PATH), config.ENABLE_GPU)

    # ------------------------------------------------------------------
    # f. Initialise Proctor service
    # ------------------------------------------------------------------
    logger.info(
        "Initialising ProctorService", extra={"model": str(_PROCTOR_MODEL_PATH)}
    )
    ProctorService.initialize(str(_PROCTOR_MODEL_PATH), config.ENABLE_GPU)

    # ------------------------------------------------------------------
    # g. Create SessionManager and WebSocketManager singletons
    # ------------------------------------------------------------------
    session_manager = SessionManager()
    websocket_manager = WebSocketManager()
    logger.info("SessionManager and WebSocketManager created")

    # ------------------------------------------------------------------
    # h. Create LLMService instance
    # ------------------------------------------------------------------
    llm_service = LLMService(config.OPENROUTER_API_KEY)
    logger.info("LLMService created")

    # ------------------------------------------------------------------
    # i. Create WorkerManager instance
    # ------------------------------------------------------------------
    worker_manager = WorkerManager(llm_service=llm_service)
    logger.info("WorkerManager created")

    # ------------------------------------------------------------------
    # j. Start the worker pool
    # ------------------------------------------------------------------
    worker_manager.start(config.WORKER_POOL_SIZE)
    logger.info(
        "Worker pool started",
        extra={"pool_size": config.WORKER_POOL_SIZE},
    )

    # ------------------------------------------------------------------
    # k. Inject managers into the WebSocket router
    # ------------------------------------------------------------------
    set_managers(websocket_manager, worker_manager)
    logger.info("Managers injected into WebSocket router")

    # ------------------------------------------------------------------
    # l. Start WebSocket heartbeat
    # ------------------------------------------------------------------
    asyncio.create_task(
        websocket_manager.start_heartbeat(),
        name="ws-heartbeat",
    )
    logger.info("WebSocket heartbeat task started")

    # ------------------------------------------------------------------
    # m. Start idle session cleanup (every 10 minutes)
    # ------------------------------------------------------------------
    asyncio.create_task(_idle_session_cleanup(), name="session-cleanup")
    logger.info("Idle session cleanup task started")

    logger.info("AI Proctoring Backend startup complete")


# ---------------------------------------------------------------------------
# Idle session cleanup background task
# ---------------------------------------------------------------------------

async def _idle_session_cleanup(idle_minutes: int = 30, interval_minutes: int = 10) -> None:
    """Periodically terminate sessions that have been idle for too long.

    Runs every ``interval_minutes`` minutes and terminates any session whose
    last event was recorded more than ``idle_minutes`` minutes ago.

    Args:
        idle_minutes:     Sessions idle longer than this are terminated.
        interval_minutes: How often to run the cleanup sweep.
    """
    from datetime import datetime, timezone, timedelta

    while True:
        await asyncio.sleep(interval_minutes * 60)
        if session_manager is None:
            continue
        try:
            session_ids = await session_manager.list_sessions()
            cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=idle_minutes)
            terminated = 0
            for sid in session_ids:
                sess = await session_manager.get_session(sid)
                if sess is None or sess.end_time is not None:
                    continue
                # Check last event time
                if sess.event_history:
                    last_event_time = sess.event_history[-1].timestamp
                    # Make timezone-aware if naive
                    if last_event_time.tzinfo is None:
                        last_event_time = last_event_time.replace(tzinfo=timezone.utc)
                    if last_event_time < cutoff:
                        await session_manager.terminate_session(sid)
                        terminated += 1
                else:
                    # No events — check start time
                    start = sess.start_time
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    if start < cutoff:
                        await session_manager.terminate_session(sid)
                        terminated += 1
            if terminated > 0:
                logger.info(
                    "Idle session cleanup complete",
                    extra={"terminated": terminated, "idle_minutes": idle_minutes},
                )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Idle session cleanup error", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# Shutdown lifecycle hook
# ---------------------------------------------------------------------------


@app.on_event("shutdown")
async def shutdown() -> None:
    """Gracefully stop all background workers on application shutdown.

    Cancels all audio and video worker tasks managed by
    :class:`~app.services.worker_manager.WorkerManager` and awaits their
    completion so that in-flight tasks can finish cleanly.
    """
    global worker_manager

    logger.info("AI Proctoring Backend shutting down")

    if worker_manager is not None:
        await worker_manager.stop()
        logger.info("WorkerManager stopped")

    logger.info("AI Proctoring Backend shutdown complete")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exception by logging the full traceback and returning HTTP 500.

    The full traceback is written to the server log so that operators can
    diagnose the root cause.  The client receives only a generic error
    message — no stack trace is ever included in the HTTP response.

    Args:
        request: The HTTP request that triggered the exception.
        exc:     The unhandled exception instance.

    Returns:
        :class:`~fastapi.responses.JSONResponse` with HTTP status 500 and
        body ``{"error": "Internal server error"}``.
    """
    logger.error(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc),
        },
        exc_info=True,
    )
    # Log the full traceback explicitly for log aggregators that may not
    # capture exc_info automatically.
    logger.debug("Full traceback:\n%s", traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )
