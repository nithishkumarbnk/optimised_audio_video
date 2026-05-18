"""
Session Manager for the AI Proctoring Backend.

Provides an in-memory, asyncio-safe session store. Each session tracks its
full event history, cumulative risk score, warning count, and a filtered
timeline of high-risk events.

Thread-safety is achieved through two levels of locking:

* ``_global_lock`` — serialises creation of new sessions so that concurrent
  callers for the same ``session_id`` never create duplicate entries.
* Per-session ``asyncio.Lock`` (stored in ``_locks``) — serialises mutations
  to an individual :class:`Session` so that concurrent event recordings do
  not produce race conditions.

Usage::

    manager = SessionManager()

    event = SessionEvent(
        event_name="mobile_detected",
        timestamp=datetime.utcnow(),
        risk_score=5,
        source="video",
    )
    await manager.record_event("session-abc", event, risk_level="HIGH")
    session = await manager.get_session("session-abc")
    print(session.cumulative_risk_score)  # 5
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# Valid source identifiers for a session event.
EventSource = Literal["audio", "video", "websocket"]


@dataclass
class SessionEvent:
    """A single proctoring event recorded during an exam session.

    Attributes
    ----------
    event_name:
        Human-readable name of the event (e.g. ``"mobile_detected"``).
    timestamp:
        UTC datetime at which the event was observed.
    risk_score:
        The risk-score contribution of this individual event.
    source:
        The subsystem that produced the event.  Must be one of
        ``"audio"``, ``"video"``, or ``"websocket"``.
    """

    event_name: str
    timestamp: datetime
    risk_score: int
    source: EventSource


@dataclass
class Session:
    """In-memory state for a single exam session.

    Attributes
    ----------
    session_id:
        Unique identifier for the session (typically a UUID string).
    start_time:
        UTC datetime when the session was first created.
    event_history:
        Ordered list of every :class:`SessionEvent` recorded for this
        session, regardless of risk level.
    cumulative_risk_score:
        Running total of all ``event.risk_score`` values recorded so far.
    warnings_count:
        Number of events that were classified as ``"HIGH"`` risk.
    suspicious_activity_timeline:
        Subset of ``event_history`` containing only high-risk events.
        Useful for generating a concise audit trail.
    end_time:
        UTC datetime when :meth:`SessionManager.terminate_session` was
        called, or ``None`` if the session is still active.
    """

    session_id: str
    start_time: datetime
    event_history: List[SessionEvent] = field(default_factory=list)
    cumulative_risk_score: int = 0
    warnings_count: int = 0
    suspicious_activity_timeline: List[SessionEvent] = field(default_factory=list)
    end_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Async-safe in-memory store for exam sessions.

    All public methods are coroutines so they integrate naturally with
    FastAPI's async request handlers and background workers.

    Attributes
    ----------
    _sessions:
        Mapping from ``session_id`` to the corresponding :class:`Session`.
    _locks:
        Per-session :class:`asyncio.Lock` instances used to serialise
        mutations to individual sessions.
    _global_lock:
        A single :class:`asyncio.Lock` that guards the creation of new
        entries in ``_sessions`` and ``_locks``.
    """

    def __init__(self) -> None:
        """Initialise an empty session store."""
        self._sessions: Dict[str, Session] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock: asyncio.Lock = asyncio.Lock()
        logger.info("SessionManager initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create(self, session_id: str) -> Session:
        """Return the existing session or create a new one.

        The creation step is performed under ``_global_lock`` to prevent
        duplicate entries when multiple coroutines race to create the same
        session simultaneously.

        Parameters
        ----------
        session_id:
            The unique identifier for the session.

        Returns
        -------
        Session
            The (possibly newly created) :class:`Session` object.
        """
        async with self._global_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(
                    session_id=session_id,
                    start_time=datetime.utcnow(),
                )
                self._locks[session_id] = asyncio.Lock()
                logger.info(
                    "New session created",
                    extra={"session_id": session_id},
                )
        return self._sessions[session_id]

    async def record_event(
        self,
        session_id: str,
        event: SessionEvent,
        risk_level: str,
    ) -> None:
        """Append an event to the session and update aggregate counters.

        Calls :meth:`get_or_create` to ensure the session exists, then
        acquires the per-session lock before mutating state.

        * ``event`` is always appended to ``event_history``.
        * ``event.risk_score`` is always added to ``cumulative_risk_score``.
        * When ``risk_level == "HIGH"``, ``warnings_count`` is incremented
          and ``event`` is also appended to ``suspicious_activity_timeline``.

        Parameters
        ----------
        session_id:
            The unique identifier for the session.
        event:
            The :class:`SessionEvent` to record.
        risk_level:
            The overall risk classification for this event.  Pass
            ``"HIGH"`` to trigger the warning counter and suspicious
            timeline update.
        """
        session = await self.get_or_create(session_id)
        async with self._locks[session_id]:
            session.event_history.append(event)
            session.cumulative_risk_score += event.risk_score

            if risk_level == "HIGH":
                session.warnings_count += 1
                session.suspicious_activity_timeline.append(event)

        logger.debug(
            "Event recorded",
            extra={
                "session_id": session_id,
                "event_name": event.event_name,
                "risk_score": event.risk_score,
                "risk_level": risk_level,
                "cumulative_risk_score": session.cumulative_risk_score,
            },
        )

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Return the session for the given ID, or ``None`` if not found.

        This method does **not** create a new session; use
        :meth:`get_or_create` when creation is desired.

        Parameters
        ----------
        session_id:
            The unique identifier for the session.

        Returns
        -------
        Optional[Session]
            The :class:`Session` if it exists, otherwise ``None``.
        """
        return self._sessions.get(session_id)

    async def list_sessions(self) -> List[str]:
        """Return a list of all known session IDs.

        Returns
        -------
        List[str]
            A snapshot of the current session IDs.  The list may be
            stale by the time the caller inspects it.
        """
        return list(self._sessions.keys())

    async def terminate_session(self, session_id: str) -> None:
        """Mark a session as terminated by setting its ``end_time``.

        If the session does not exist this method is a no-op (it logs a
        warning but does not raise).

        The mutation is performed under the per-session lock to avoid
        racing with concurrent :meth:`record_event` calls.

        Parameters
        ----------
        session_id:
            The unique identifier for the session to terminate.
        """
        if session_id not in self._sessions:
            logger.warning(
                "terminate_session called for unknown session",
                extra={"session_id": session_id},
            )
            return

        async with self._locks[session_id]:
            self._sessions[session_id].end_time = datetime.utcnow()

        logger.info(
            "Session terminated",
            extra={
                "session_id": session_id,
                "end_time": self._sessions[session_id].end_time.isoformat(),
            },
        )
