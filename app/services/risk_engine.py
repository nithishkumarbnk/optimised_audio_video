"""
Risk Engine for the AI Proctoring Backend.

Provides weighted event scoring, risk level classification, and rolling-window
score aggregation over a session's event history.

All functions are pure and stateless (except ``rolling_score``, which reads
session state but does not mutate it).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event weight table
# ---------------------------------------------------------------------------

#: Mapping of known event names to their integer risk weights.
#: Events not present in this table are assigned a weight of 1 and trigger
#: a WARNING log (Requirement 9.5).
EVENT_WEIGHTS: dict[str, int] = {
    "multiple_persons": 5,
    "suspicious_transcript": 4,
    "no_face": 3,
    "mobile_detected": 3,
    "looking_away": 2,
    "headset_detected": 2,
    "long_speech": 2,
    "keyword_detected": 2,
    "suspicious_movement": 2,
    "bad_posture": 1,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_score(events: List[str]) -> int:
    """Compute a weighted risk score for a list of detected events.

    Each event name is looked up in :data:`EVENT_WEIGHTS`.  If the event is
    not found, a weight of ``1`` is used and a WARNING is emitted so that
    unknown events are visible in the logs without crashing the pipeline.

    The total score is the arithmetic sum of all individual event weights,
    meaning the same event appearing multiple times in *events* contributes
    its weight each time (additive property — Requirement 9.6).

    Parameters
    ----------
    events:
        A list of event name strings produced by the detection pipeline
        (e.g. ``["no_face", "mobile_detected"]``).  May be empty.

    Returns
    -------
    int
        The total risk score (≥ 0).  Returns ``0`` for an empty list.

    Examples
    --------
    >>> compute_score(["no_face", "mobile_detected"])
    6
    >>> compute_score([])
    0
    >>> compute_score(["unknown_event"])  # logs a WARNING, returns 1
    1
    """
    total = 0
    for event in events:
        weight = EVENT_WEIGHTS.get(event)
        if weight is None:
            logger.warning(
                "Unknown event encountered in risk scoring; assigning default weight",
                extra={"event": event, "default_weight": 1},
            )
            weight = 1
        total += weight
    return total


def classify(score: int) -> str:
    """Classify a numeric risk score into a categorical risk level.

    Thresholds (Requirement 9.2):

    * ``score >= 8``  → ``"HIGH"``
    * ``score >= 4``  → ``"MEDIUM"``
    * otherwise       → ``"LOW"``

    Parameters
    ----------
    score:
        A non-negative integer risk score, typically produced by
        :func:`compute_score`.

    Returns
    -------
    str
        One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.

    Examples
    --------
    >>> classify(10)
    'HIGH'
    >>> classify(8)
    'HIGH'
    >>> classify(5)
    'MEDIUM'
    >>> classify(4)
    'MEDIUM'
    >>> classify(3)
    'LOW'
    >>> classify(0)
    'LOW'
    """
    if score >= 8:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def rolling_score(session: Any, window_seconds: int = 60) -> int:
    """Compute the cumulative risk score for events within a rolling time window.

    Filters ``session.event_history`` to only those
    :class:`~app.services.session_manager.SessionEvent` objects whose
    ``timestamp`` falls within the last *window_seconds* seconds relative to
    ``datetime.utcnow()``, then sums their ``risk_score`` fields.

    This implements Requirement 9.4: only recent events contribute to the
    rolling score, allowing the system to detect sustained suspicious activity
    within a configurable look-back window.

    Parameters
    ----------
    session:
        A :class:`~app.services.session_manager.Session` instance (typed as
        ``Any`` to avoid a circular import — ``Session`` is defined in
        ``session_manager.py`` which is not yet written at this stage).
        The object must expose an ``event_history`` attribute that is an
        iterable of objects each having a ``timestamp: datetime`` field and a
        ``risk_score: int`` field.
    window_seconds:
        The look-back window in seconds.  Only events whose ``timestamp`` is
        strictly within ``[now - window_seconds, now]`` are included.
        Defaults to ``60``.

    Returns
    -------
    int
        The sum of ``risk_score`` values for all qualifying events.
        Returns ``0`` if no events fall within the window.

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class FakeEvent:
    ...     timestamp: datetime
    ...     risk_score: int
    >>> @dataclass
    ... class FakeSession:
    ...     event_history: list
    >>> now = datetime.utcnow()
    >>> session = FakeSession(event_history=[
    ...     FakeEvent(timestamp=now - timedelta(seconds=30), risk_score=5),
    ...     FakeEvent(timestamp=now - timedelta(seconds=90), risk_score=3),
    ... ])
    >>> rolling_score(session)  # only the 30s-old event qualifies
    5
    """
    cutoff: datetime = datetime.utcnow() - timedelta(seconds=window_seconds)
    total = 0
    for event in session.event_history:
        if event.timestamp >= cutoff:
            total += event.risk_score
    return total
