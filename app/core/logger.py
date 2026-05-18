"""
Structured logging module for the AI Proctoring Backend.

Provides a JSON-line formatter and a ``configure_logging`` factory that
attaches both a stdout StreamHandler and a rotating file handler to the
root logger.  Every log record is serialised to a single JSON object so
that log-aggregation tools (Loki, CloudWatch, Datadog, …) can parse and
index fields without additional parsing rules.

Usage::

    from app.core.logger import configure_logging
    configure_logging("INFO")

    import logging
    logger = logging.getLogger(__name__)
    logger.info("session started", extra={"session_id": "abc123"})
    # → {"timestamp": "2024-01-01T00:00:00.000Z", "level": "INFO",
    #    "logger": "app.routers.audio", "message": "session started",
    #    "session_id": "abc123"}
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """Serialise a :class:`logging.LogRecord` to a single JSON line.

    The output always contains the four mandatory fields:

    * ``timestamp`` – ISO-8601 UTC timestamp with millisecond precision.
    * ``level``     – Upper-case log level name (e.g. ``"INFO"``).
    * ``logger``    – Dotted logger name (e.g. ``"app.services.llm_service"``).
    * ``message``   – The formatted log message.

    Any keyword arguments passed via the ``extra`` parameter of the logging
    call are merged into the JSON object at the top level, allowing callers
    to attach arbitrary structured fields::

        logger.info("request complete", extra={"session_id": "x", "ms": 42})

    If the record carries exception information it is serialised under the
    ``"exc_info"`` key as a formatted traceback string.

    Notes
    -----
    The formatter intentionally does *not* include Python-internal fields
    such as ``lineno``, ``pathname``, or ``thread`` in the default output to
    keep log lines compact.  Callers that need those fields can pass them via
    ``extra``.
    """

    # Fields that are part of every LogRecord but should NOT be forwarded as
    # extra structured fields (they are either already mapped to a canonical
    # key or are internal Python logging bookkeeping).
    _RESERVED_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Return a JSON-encoded string for *record*.

        Parameters
        ----------
        record:
            The log record to serialise.

        Returns
        -------
        str
            A single-line JSON string terminated by a newline character.
        """
        # Build the mandatory fields first.
        log_entry: Dict[str, Any] = {
            "timestamp": self._utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any caller-supplied ``extra`` fields that are not reserved.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS:
                log_entry[key] = value

        # Append formatted exception traceback when present.
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)

        # Append stack info when present (Python 3.2+).
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_entry, default=str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_iso(created: float) -> str:
        """Convert a POSIX timestamp to an ISO-8601 UTC string.

        Parameters
        ----------
        created:
            The ``LogRecord.created`` value (seconds since the epoch).

        Returns
        -------
        str
            A string of the form ``"2024-01-01T12:34:56.789Z"``.
        """
        dt = datetime.fromtimestamp(created, tz=timezone.utc)
        # strftime does not support sub-second precision directly; format
        # milliseconds manually.
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logger with structured JSON output.

    Attaches two handlers to the root logger (replacing any previously
    attached handlers to avoid duplicate output on repeated calls):

    1. **StreamHandler** writing to *stdout* — suitable for container
       environments where log collectors read from standard output.
    2. **RotatingFileHandler** writing to ``logs/app.log`` — provides a
       persistent on-disk record with automatic rotation.

    Both handlers use :class:`StructuredFormatter` so every log line is a
    valid JSON object regardless of destination.

    The ``logs/`` directory is expected to exist before this function is
    called.  :func:`app.main` creates it during the startup lifecycle event.

    Parameters
    ----------
    log_level:
        A standard Python logging level name (``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).  Case-insensitive.
        Defaults to ``"INFO"``.

    Raises
    ------
    ValueError
        If *log_level* is not a recognised logging level name.

    Examples
    --------
    >>> configure_logging("DEBUG")
    >>> import logging
    >>> logging.getLogger(__name__).debug("ready", extra={"env": "dev"})
    """
    numeric_level: int = getattr(logging, log_level.upper(), None)  # type: ignore[assignment]
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: {log_level!r}. "
            "Expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    formatter = StructuredFormatter()

    # --- stdout handler ---------------------------------------------------
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)

    # --- rotating file handler --------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MiB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    # --- root logger ------------------------------------------------------
    root_logger = logging.getLogger()
    # Remove any handlers that were attached by earlier calls or by the
    # logging module's default configuration so we do not emit duplicate
    # records.
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(numeric_level)
