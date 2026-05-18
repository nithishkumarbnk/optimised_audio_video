"""
WebSocket router for the AI Proctoring Backend.

Exposes a single endpoint:

    WebSocket /ws/proctor/{session_id}

Each connection is assigned a unique ``connection_id`` (UUID4).  Incoming
JSON messages are dispatched to the appropriate worker queue based on their
``type`` field.  Results produced by the worker pool are forwarded back to
the client concurrently via an asyncio Task.

Circular-import avoidance
-------------------------
``websocket_manager`` and ``worker_manager`` are module-level variables
initialised to ``None``.  ``app.main`` calls :func:`set_managers` during the
FastAPI startup lifecycle event to inject the live instances.  The router
never imports from ``app.main`` directly.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.7
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_config
from app.schemas.websocket_schema import WSIncomingMessage
from app.utils.audio_utils import base64_to_audio
from app.utils.image_utils import base64_to_frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency injection — set by app.main at startup to avoid circular imports
# ---------------------------------------------------------------------------

# These are typed as Optional[object] here; the actual types are
# WebSocketManager and WorkerManager respectively.  Using TYPE_CHECKING
# imports would be cleaner but would still risk circular imports at runtime,
# so we keep them as plain module-level variables.
websocket_manager: Optional[object] = None  # WebSocketManager instance
worker_manager: Optional[object] = None     # WorkerManager instance


def set_managers(ws_manager: object, wk_manager: object) -> None:
    """Inject the live manager instances from ``app.main``.

    Called once during the FastAPI ``startup`` lifecycle event.  After this
    call the module-level ``websocket_manager`` and ``worker_manager``
    variables are set and the WebSocket endpoint is fully operational.

    Args:
        ws_manager: An initialised :class:`~app.services.websocket_manager.WebSocketManager`.
        wk_manager: An initialised :class:`~app.services.worker_manager.WorkerManager`.
    """
    global websocket_manager, worker_manager
    websocket_manager = ws_manager
    worker_manager = wk_manager
    logger.info("WebSocket router: managers injected")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()
# No prefix — the endpoint is mounted at /ws/proctor/{session_id} directly.


# ---------------------------------------------------------------------------
# Result-forwarding helper
# ---------------------------------------------------------------------------

async def _forward_results(
    websocket: WebSocket,
    connection_id: str,
) -> None:
    """Continuously read results from the worker queue and send them to the client.

    Runs as a background :class:`asyncio.Task` alongside the receive loop.
    Reads :class:`~app.services.worker_manager.WorkerResult` objects from the
    per-connection result queue and forwards their ``payload`` as JSON to the
    connected WebSocket client.

    The coroutine exits cleanly when cancelled (e.g. when the receive loop
    detects a disconnect and cancels this task).

    Args:
        websocket:     The active WebSocket connection to write results to.
        connection_id: Unique identifier for this connection; used to look up
                       the correct result queue in the worker manager.
    """
    try:
        result_queue: asyncio.Queue = worker_manager.get_result_queue(connection_id)  # type: ignore[union-attr]
        while True:
            result = await result_queue.get()
            try:
                if result.error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "connection_id": connection_id,
                            "error": result.error,
                        }
                    )
                else:
                    await websocket.send_json(result.payload)
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to forward worker result to client",
                    extra={
                        "connection_id": connection_id,
                        "error": str(send_exc),
                    },
                )
            finally:
                result_queue.task_done()
    except asyncio.CancelledError:
        logger.debug(
            "Result-forwarding task cancelled",
            extra={"connection_id": connection_id},
        )
        raise  # allow the task to be cancelled cleanly


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/proctor/{session_id}")
async def proctor_websocket(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """Handle a WebSocket connection for a proctoring session.

    Lifecycle
    ---------
    1. Extract the ``Origin`` header from the upgrade request.
    2. Call :meth:`~app.services.websocket_manager.WebSocketManager.connect`
       which validates the origin and accepts (or rejects) the connection.
    3. Spawn a background :class:`asyncio.Task` that forwards worker results
       to the client as they become available.
    4. Enter the receive loop:

       * Parse each incoming text frame as a
         :class:`~app.schemas.websocket_schema.WSIncomingMessage`.
       * Dispatch based on ``message.type``:

         - ``"video"`` — decode the base64 frame and call
           :meth:`~app.services.worker_manager.WorkerManager.enqueue_video`.
         - ``"audio"`` — decode the base64 audio and call
           :meth:`~app.services.worker_manager.WorkerManager.enqueue_audio`.
         - ``"pong"``  — call
           :meth:`~app.services.websocket_manager.WebSocketManager.record_pong`.
         - Unknown types are logged and ignored.

    5. On :class:`fastapi.WebSocketDisconnect`: cancel the forwarding task,
       call :meth:`~app.services.websocket_manager.WebSocketManager.disconnect`,
       and log the disconnection.

    Args:
        websocket:  The incoming FastAPI ``WebSocket`` instance (injected by
                    the framework).
        session_id: The proctoring session identifier extracted from the URL
                    path parameter ``{session_id}``.

    Raises:
        RuntimeError: If :func:`set_managers` has not been called before the
            first connection attempt (i.e. the managers are still ``None``).
    """
    if websocket_manager is None or worker_manager is None:
        # This should never happen in production — set_managers() is called
        # during startup.  Guard here to surface a clear error during testing.
        logger.error(
            "WebSocket endpoint called before managers were initialised",
            extra={"session_id": session_id},
        )
        await websocket.close(code=1011)  # Internal Error
        return

    # ------------------------------------------------------------------
    # 1. Generate a unique connection identifier
    # ------------------------------------------------------------------
    connection_id: str = str(uuid4())

    # ------------------------------------------------------------------
    # 2. Extract Origin header
    # ------------------------------------------------------------------
    # WebSocket upgrade headers are available via websocket.headers.
    origin: str = websocket.headers.get("origin", "")

    # ------------------------------------------------------------------
    # 3. Connect (origin validation + registration + accept)
    # ------------------------------------------------------------------
    config = get_config()
    await websocket_manager.connect(  # type: ignore[union-attr]
        websocket=websocket,
        session_id=session_id,
        connection_id=connection_id,
        origin=origin,
        allowed_origins=config.CORS_ORIGINS,
    )

    # If the origin was rejected, websocket_manager.connect() closes the
    # socket with code 4003 and returns without accepting.  We detect this
    # by checking whether the WebSocket state is still open.
    # FastAPI's WebSocket exposes the underlying Starlette state; a closed
    # socket will raise on any subsequent send/receive, so we guard with a
    # try/except in the receive loop.  However, we can also check the
    # client_state attribute to bail out early.
    try:
        # Starlette sets client_state to WebSocketState.CONNECTED after accept.
        # If the connection was rejected it will be DISCONNECTED.
        from starlette.websockets import WebSocketState  # local import to avoid top-level coupling
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.debug(
                "WebSocket not accepted (origin rejected); exiting handler",
                extra={"session_id": session_id, "connection_id": connection_id},
            )
            return
    except Exception:
        # If the state check itself fails for any reason, proceed and let the
        # receive loop surface the error naturally.
        pass

    logger.info(
        "WebSocket handler started",
        extra={
            "session_id": session_id,
            "connection_id": connection_id,
            "origin": origin,
        },
    )

    # ------------------------------------------------------------------
    # 4. Spawn result-forwarding task
    # ------------------------------------------------------------------
    forward_task: asyncio.Task = asyncio.create_task(
        _forward_results(websocket, connection_id),
        name=f"ws-forward-{connection_id}",
    )

    # ------------------------------------------------------------------
    # 5. Receive loop
    # ------------------------------------------------------------------
    try:
        while True:
            raw_text: str = await websocket.receive_text()

            # Parse the incoming JSON payload.
            try:
                data = json.loads(raw_text)
                message = WSIncomingMessage(**data)
            except (json.JSONDecodeError, ValidationError) as parse_exc:
                logger.warning(
                    "Received malformed WebSocket message; skipping",
                    extra={
                        "session_id": session_id,
                        "connection_id": connection_id,
                        "error": str(parse_exc),
                    },
                )
                continue

            # Dispatch based on message type.
            msg_type: str = message.type

            if msg_type == "video":
                if not message.frame:
                    logger.warning(
                        "Video message missing 'frame' field",
                        extra={
                            "session_id": session_id,
                            "connection_id": connection_id,
                        },
                    )
                    continue
                try:
                    frame = base64_to_frame(message.frame)
                    await worker_manager.enqueue_video(  # type: ignore[union-attr]
                        session_id=session_id,
                        connection_id=connection_id,
                        frame=frame,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to decode or enqueue video frame",
                        extra={
                            "session_id": session_id,
                            "connection_id": connection_id,
                            "error": str(exc),
                        },
                    )

            elif msg_type == "audio":
                if not message.audio:
                    logger.warning(
                        "Audio message missing 'audio' field",
                        extra={
                            "session_id": session_id,
                            "connection_id": connection_id,
                        },
                    )
                    continue
                try:
                    audio_bytes = base64_to_audio(message.audio)
                    await worker_manager.enqueue_audio(  # type: ignore[union-attr]
                        session_id=session_id,
                        connection_id=connection_id,
                        audio_bytes=audio_bytes,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to decode or enqueue audio chunk",
                        extra={
                            "session_id": session_id,
                            "connection_id": connection_id,
                            "error": str(exc),
                        },
                    )

            elif msg_type == "pong":
                websocket_manager.record_pong(connection_id)  # type: ignore[union-attr]
                logger.debug(
                    "Pong received",
                    extra={
                        "session_id": session_id,
                        "connection_id": connection_id,
                    },
                )

            else:
                logger.warning(
                    "Unknown WebSocket message type; ignoring",
                    extra={
                        "session_id": session_id,
                        "connection_id": connection_id,
                        "msg_type": msg_type,
                    },
                )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected",
            extra={
                "session_id": session_id,
                "connection_id": connection_id,
                "action": "disconnect",
            },
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected error in WebSocket receive loop",
            extra={
                "session_id": session_id,
                "connection_id": connection_id,
                "error": str(exc),
            },
        )

    finally:
        # ------------------------------------------------------------------
        # 6. Cleanup — cancel forwarding task and deregister connection
        # ------------------------------------------------------------------
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            pass  # expected — the task was cancelled above

        await websocket_manager.disconnect(connection_id)  # type: ignore[union-attr]

        logger.info(
            "WebSocket handler finished",
            extra={
                "session_id": session_id,
                "connection_id": connection_id,
            },
        )
