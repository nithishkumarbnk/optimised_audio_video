"""
Rule Engine for the AI Proctoring Backend.

Applies fast, deterministic keyword and heuristic checks to transcribed
text.  Results are returned as a deduplicated list of flag strings that
downstream components (LLM_Service, Risk_Engine) can act on immediately
without waiting for heavier inference pipelines.

Usage::

    from app.services.rule_engine import check_rules

    flags = check_rules("Can you tell me the answer to question 3?")
    # → ["keyword_detected", "long_speech"]

    flags = check_rules(None)
    # → []
"""

from __future__ import annotations

import logging
from typing import List, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default keyword list checked against the lowercased transcript.
#: Any match causes ``"keyword_detected"`` to be appended to the flags list.
#: Requirements: 6.1, 6.2
DEFAULT_KEYWORDS: List[str] = [
    "answer",
    "google",
    "tell me",
    "search",
    "help me",
    "what is",
    "how to",
]

#: Word-count threshold above which ``"long_speech"`` is flagged.
#: Requirement: 6.3
_LONG_SPEECH_THRESHOLD: int = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_rules(text: Optional[str]) -> List[str]:
    """Apply keyword and heuristic rules to a transcript.

    Checks are performed in the following order:

    1. **Early exit** — if *text* is ``None`` or an empty string, return
       ``[]`` immediately without raising any exception (Requirement 6.4).
    2. **Keyword detection** — the lowercased transcript is scanned for each
       entry in :data:`DEFAULT_KEYWORDS`.  If *any* keyword is found,
       ``"keyword_detected"`` is appended to the flags list exactly once
       (Requirement 6.1, 6.2).
    3. **Long-speech detection** — if the word count of *text* exceeds
       :data:`_LONG_SPEECH_THRESHOLD` (8 words), ``"long_speech"`` is
       appended (Requirement 6.3).
    4. **Deduplication** — the list is deduplicated while preserving
       insertion order via ``list(dict.fromkeys(...))`` (Requirement 6.5).

    Parameters
    ----------
    text:
        The transcribed speech string to evaluate.  May be ``None`` or
        empty, in which case an empty list is returned.

    Returns
    -------
    List[str]
        A deduplicated list of flag strings.  Possible values are
        ``"keyword_detected"`` and ``"long_speech"``.  Returns ``[]``
        when no rules are triggered or when *text* is absent.

    Examples
    --------
    >>> check_rules(None)
    []
    >>> check_rules("")
    []
    >>> check_rules("Can you tell me the answer?")
    ['keyword_detected']
    >>> check_rules("Can you please tell me what is the answer to this very long question here")
    ['keyword_detected', 'long_speech']
    >>> check_rules("one two three four five six seven eight nine")
    ['long_speech']
    """
    # ------------------------------------------------------------------
    # 1. Early exit for None / empty input (Requirement 6.4)
    # ------------------------------------------------------------------
    if not text:
        logger.debug("check_rules called with empty or None text; returning []")
        return []

    flags: List[str] = []
    lowered: str = text.lower()

    # ------------------------------------------------------------------
    # 2. Keyword detection (Requirements 6.1, 6.2)
    # ------------------------------------------------------------------
    for keyword in DEFAULT_KEYWORDS:
        if keyword in lowered:
            logger.debug("Keyword match found", extra={"keyword": keyword})
            flags.append("keyword_detected")
            break  # append at most once

    # ------------------------------------------------------------------
    # 3. Long-speech detection (Requirement 6.3)
    # ------------------------------------------------------------------
    word_count: int = len(text.split())
    if word_count > _LONG_SPEECH_THRESHOLD:
        logger.debug(
            "Long speech detected",
            extra={"word_count": word_count, "threshold": _LONG_SPEECH_THRESHOLD},
        )
        flags.append("long_speech")

    # ------------------------------------------------------------------
    # 4. Deduplicate while preserving insertion order (Requirement 6.5)
    # ------------------------------------------------------------------
    result: List[str] = list(dict.fromkeys(flags))

    logger.info(
        "Rule check complete",
        extra={"word_count": word_count, "flags": result},
    )
    return result
