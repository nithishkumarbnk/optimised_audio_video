"""
Property-based and unit tests for the Rule Engine.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

Properties tested
-----------------
Property 3: Rule Engine Flag Uniqueness
    For any text input, the returned list must contain no duplicate flags.

Property 4: Rule Engine Keyword Detection (Case-Insensitive)
    For any keyword from DEFAULT_KEYWORDS embedded in a transcript (in any
    case), ``"keyword_detected"`` must appear in the result.

Property 5: Rule Engine Long Speech Detection
    For any list of 9+ non-empty words joined into a transcript,
    ``"long_speech"`` must appear in the result.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.rule_engine import DEFAULT_KEYWORDS, check_rules


# ---------------------------------------------------------------------------
# Unit tests — specific examples
# ---------------------------------------------------------------------------


class TestCheckRulesNoneAndEmpty:
    """Requirement 6.4 — None and empty string return []."""

    def test_none_returns_empty_list(self):
        assert check_rules(None) == []

    def test_empty_string_returns_empty_list(self):
        assert check_rules("") == []

    def test_whitespace_only_returns_empty_list(self):
        # A string of only spaces is falsy after strip, but Python's `not text`
        # treats a non-empty whitespace string as truthy.  The spec says
        # "None or empty"; whitespace-only is not explicitly covered, but
        # splitting it yields zero meaningful words so no flags fire.
        result = check_rules("   ")
        # No keywords, word count of "   ".split() == 0 → no long_speech
        assert "long_speech" not in result
        assert "keyword_detected" not in result


class TestKeywordDetection:
    """Requirements 6.1, 6.2 — keyword matching is case-insensitive."""

    @pytest.mark.parametrize("keyword", DEFAULT_KEYWORDS)
    def test_exact_keyword_triggers_flag(self, keyword: str):
        assert "keyword_detected" in check_rules(keyword)

    @pytest.mark.parametrize("keyword", DEFAULT_KEYWORDS)
    def test_uppercase_keyword_triggers_flag(self, keyword: str):
        assert "keyword_detected" in check_rules(keyword.upper())

    @pytest.mark.parametrize("keyword", DEFAULT_KEYWORDS)
    def test_mixed_case_keyword_triggers_flag(self, keyword: str):
        mixed = "".join(
            c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(keyword)
        )
        assert "keyword_detected" in check_rules(mixed)

    def test_no_keyword_no_flag(self):
        assert "keyword_detected" not in check_rules("the quick brown fox")

    def test_keyword_detected_appended_only_once(self):
        # Multiple keywords in the same text → still only one flag
        text = "google answer"
        result = check_rules(text)
        assert result.count("keyword_detected") == 1


class TestLongSpeechDetection:
    """Requirement 6.3 — word count > 8 triggers long_speech."""

    def test_exactly_8_words_no_flag(self):
        text = "one two three four five six seven eight"
        assert "long_speech" not in check_rules(text)

    def test_9_words_triggers_flag(self):
        text = "one two three four five six seven eight nine"
        assert "long_speech" in check_rules(text)

    def test_many_words_triggers_flag(self):
        text = " ".join(["word"] * 20)
        assert "long_speech" in check_rules(text)


class TestDeduplication:
    """Requirement 6.5 — returned list has no duplicate flags."""

    def test_no_duplicates_with_keyword_and_long_speech(self):
        # Both flags should appear exactly once
        text = "please tell me the answer to this very long question here"
        result = check_rules(text)
        assert len(result) == len(set(result))

    def test_no_duplicates_keyword_only(self):
        result = check_rules("google")
        assert len(result) == len(set(result))

    def test_no_duplicates_long_speech_only(self):
        result = check_rules("one two three four five six seven eight nine")
        assert len(result) == len(set(result))


class TestReturnOrder:
    """keyword_detected should precede long_speech when both are present."""

    def test_keyword_before_long_speech(self):
        text = "please tell me the answer to this very long question here"
        result = check_rules(text)
        if "keyword_detected" in result and "long_speech" in result:
            assert result.index("keyword_detected") < result.index("long_speech")


# ---------------------------------------------------------------------------
# Property 3: Rule Engine Flag Uniqueness
# **Validates: Requirements 6.5**
# ---------------------------------------------------------------------------


@given(st.text())
@settings(max_examples=200)
def test_property_3_flag_uniqueness(text: str):
    """Property 3: check_rules never returns duplicate flags for any input.

    **Validates: Requirements 6.5**
    """
    result = check_rules(text)
    assert len(result) == len(set(result)), (
        f"Duplicate flags found for text={text!r}: {result}"
    )


# ---------------------------------------------------------------------------
# Property 4: Rule Engine Keyword Detection (Case-Insensitive)
# **Validates: Requirements 6.1, 6.2**
# ---------------------------------------------------------------------------


@given(
    keyword=st.sampled_from(DEFAULT_KEYWORDS),
    uppercase=st.booleans(),
)
@settings(max_examples=200)
def test_property_4_keyword_detection_case_insensitive(
    keyword: str, uppercase: bool
) -> None:
    """Property 4: Any keyword from DEFAULT_KEYWORDS (in any case) triggers
    ``"keyword_detected"``.

    **Validates: Requirements 6.1, 6.2**
    """
    transcript = keyword.upper() if uppercase else keyword.lower()
    result = check_rules(transcript)
    assert "keyword_detected" in result, (
        f"Expected 'keyword_detected' for transcript={transcript!r}, got {result}"
    )


# ---------------------------------------------------------------------------
# Property 5: Rule Engine Long Speech Detection
# **Validates: Requirements 6.3**
# ---------------------------------------------------------------------------


@given(st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))), min_size=9))
@settings(max_examples=200)
def test_property_5_long_speech_detection(words: list) -> None:
    """Property 5: Joining 9+ non-empty words always triggers ``"long_speech"``.

    **Validates: Requirements 6.3**
    """
    transcript = " ".join(words)
    result = check_rules(transcript)
    assert "long_speech" in result, (
        f"Expected 'long_speech' for {len(words)}-word transcript, got {result!r}"
    )
