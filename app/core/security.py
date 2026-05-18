"""
Security helper functions for the AI Proctoring Backend.

Provides input sanitization and validation utilities used by middleware,
routers, and the WebSocket manager to enforce security boundaries.
"""

from __future__ import annotations

import os
from typing import List


def sanitize_filename(filename: str) -> str:
    """Sanitize an uploaded filename to prevent directory traversal attacks.

    Strips all directory components using ``os.path.basename`` so that only
    the final filename segment is retained.  Any remaining ``..`` sequences
    (which could survive basename on some edge-case inputs) are then removed
    by replacing them with an empty string.

    Args:
        filename: The raw filename string supplied by the client.

    Returns:
        A safe filename containing no path separators or ``..`` sequences.
        If the result is empty after sanitization, an empty string is returned.

    Examples:
        >>> sanitize_filename("../../etc/passwd")
        'passwd'
        >>> sanitize_filename("../uploads/secret.wav")
        'secret.wav'
        >>> sanitize_filename("normal_file.wav")
        'normal_file.wav'
    """
    # Strip directory components (handles both / and \ separators)
    safe = os.path.basename(filename)
    # Remove any residual ".." sequences
    safe = safe.replace("..", "")
    return safe


def validate_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Validate a WebSocket or HTTP request origin against the allowed list.

    Returns ``True`` when the connection should be permitted, ``False`` when
    it should be rejected.

    The wildcard value ``"*"`` in *allowed_origins* permits all origins.
    Otherwise the *origin* must be an exact (case-sensitive) match against
    one of the entries in the list.

    Args:
        origin: The ``Origin`` header value from the incoming request.
        allowed_origins: The list of permitted origin strings, as configured
            via ``Config.CORS_ORIGINS``.  May contain ``"*"`` to allow all.

    Returns:
        ``True`` if the origin is allowed, ``False`` otherwise.

    Examples:
        >>> validate_origin("https://example.com", ["*"])
        True
        >>> validate_origin("https://example.com", ["https://example.com"])
        True
        >>> validate_origin("https://evil.com", ["https://example.com"])
        False
    """
    if "*" in allowed_origins:
        return True
    return origin in allowed_origins


def check_body_size(content_length: int, max_mb: int) -> bool:
    """Check whether a request body is within the permitted size limit.

    Compares *content_length* (in bytes) against *max_mb* converted to bytes.
    Returns ``True`` when the body is within the limit (i.e. the request
    should be accepted), ``False`` when it exceeds the limit.

    Args:
        content_length: The ``Content-Length`` header value in bytes.
        max_mb: The maximum permitted body size in megabytes, as configured
            via ``Config.MAX_AUDIO_SIZE_MB``.

    Returns:
        ``True`` if ``content_length <= max_mb * 1024 * 1024``, else ``False``.

    Examples:
        >>> check_body_size(1024, 20)
        True
        >>> check_body_size(20 * 1024 * 1024, 20)
        True
        >>> check_body_size(20 * 1024 * 1024 + 1, 20)
        False
    """
    max_bytes = max_mb * 1024 * 1024
    return content_length <= max_bytes
