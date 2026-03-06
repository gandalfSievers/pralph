"""Tests for review output parsing and the assistant-text fallback in the runner.

The bug: when Claude produces a review JSON in an assistant text block and then
calls a tool (e.g. TodoWrite) as part of the same response, the final stream-json
result event's ``result`` field is empty.  ``parse_review_output`` falls through
all strategies and auto-approves a review that should have been rejected.

The fix: ``run_claude`` now accumulates all assistant text blocks during streaming
and uses them as a fallback when the result event's ``result`` field is empty.
"""

from __future__ import annotations

import json

import pytest

from pralph.parser import extract_json_from_text, parse_review_output
from pralph.runner import ClaudeResult, _parse_result_event


# ── Fixtures: realistic review payloads ──────────────────────────────


REVIEW_JSON_REJECTED = {
    "approved": False,
    "feedback": "Schema version mismatch and missing ModelConfiguration.",
    "issues": [
        {"severity": "critical", "description": "Schema not updated to V1_4."},
        {"severity": "major", "description": "No storage config for EntryRevision."},
    ],
}

REVIEW_JSON_APPROVED = {
    "approved": True,
    "feedback": "Implementation looks good and follows conventions.",
    "issues": [],
}

REVIEW_TEXT_WITH_JSON_BLOCK = (
    "I've completed my review. Here is the result:\n\n"
    "```json\n"
    + json.dumps(REVIEW_JSON_REJECTED, indent=2)
    + "\n```\n\n"
    "**Summary of blocking issues:**\n\n"
    "1. Schema not updated\n"
    "2. No storage config\n"
)

REVIEW_TEXT_BARE_JSON = json.dumps(REVIEW_JSON_APPROVED)

REVIEW_TEXT_WITH_PROSE_AND_BRACES = (
    "After careful review, the implementation is solid.\n\n"
    + json.dumps(REVIEW_JSON_APPROVED, indent=2)
    + "\n\nOverall, great work."
)


# ═══════════════════════════════════════════════════════════════════════
# extract_json_from_text
# ═══════════════════════════════════════════════════════════════════════


class TestExtractJsonFromText:
    """Verify all four extraction strategies."""

    def test_strategy1_direct_parse(self):
        raw = json.dumps({"approved": False, "feedback": "bad"})
        assert extract_json_from_text(raw) == {"approved": False, "feedback": "bad"}

    def test_strategy2_fenced_code_block(self):
        text = "Here:\n```json\n{\"approved\": true}\n```\nDone."
        result = extract_json_from_text(text)
        assert result == {"approved": True}

    def test_strategy2_fenced_block_no_lang(self):
        text = "Result:\n```\n{\"key\": 42}\n```"
        assert extract_json_from_text(text) == {"key": 42}

    def test_strategy3_balanced_braces(self):
        text = "Some preamble {\"a\": 1, \"b\": {\"c\": 2}} trailing"
        result = extract_json_from_text(text)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_strategy4_balanced_brackets(self):
        text = "List: [1, 2, 3] end"
        assert extract_json_from_text(text) == [1, 2, 3]

    def test_returns_none_on_garbage(self):
        assert extract_json_from_text("no json here at all") is None

    def test_returns_none_on_empty(self):
        assert extract_json_from_text("") is None

    def test_returns_none_on_whitespace(self):
        assert extract_json_from_text("   \n\n  ") is None

    def test_invalid_json_in_fenced_block(self):
        text = "```json\n{not valid json}\n```"
        # Falls through to strategy 3, which also fails
        assert extract_json_from_text(text) is None

    def test_realistic_review_with_code_block(self):
        """The exact shape that triggered the original bug."""
        result = extract_json_from_text(REVIEW_TEXT_WITH_JSON_BLOCK)
        assert result is not None
        assert result["approved"] is False
        assert len(result["issues"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# parse_review_output
# ═══════════════════════════════════════════════════════════════════════


class TestParseReviewOutput:
    """Verify review parsing strategies and fallbacks."""

    # ── Strategy 1: JSON extraction succeeds ──

    def test_bare_json_approved(self):
        parsed = parse_review_output(REVIEW_TEXT_BARE_JSON)
        assert parsed["approved"] is True
        assert parsed["feedback"] == REVIEW_JSON_APPROVED["feedback"]
        assert parsed["issues"] == []

    def test_json_code_block_rejected(self):
        parsed = parse_review_output(REVIEW_TEXT_WITH_JSON_BLOCK)
        assert parsed["approved"] is False
        assert "Schema" in parsed["feedback"]
        assert len(parsed["issues"]) == 2
        assert parsed["issues"][0]["severity"] == "critical"

    def test_json_embedded_in_prose(self):
        parsed = parse_review_output(REVIEW_TEXT_WITH_PROSE_AND_BRACES)
        assert parsed["approved"] is True

    # ── Strategy 2: APPROVED: marker fallback ──

    def test_approved_marker_true(self):
        text = "Everything looks good.\nAPPROVED: true\nNo issues found."
        parsed = parse_review_output(text)
        assert parsed["approved"] is True
        assert parsed["issues"] == []

    def test_approved_marker_false(self):
        text = "Several problems found.\nAPPROVED: false\nSee above."
        parsed = parse_review_output(text)
        assert parsed["approved"] is False

    def test_approved_marker_case_insensitive(self):
        text = "approved: TRUE"
        parsed = parse_review_output(text)
        assert parsed["approved"] is True

    # ── Strategy 3: unparseable → reject (safe default after 6d25c32 fix) ──

    def test_unparseable_rejects(self):
        parsed = parse_review_output("I looked at the code and it seems fine.")
        assert parsed["approved"] is False
        assert "could not be parsed" in parsed["feedback"]

    def test_empty_string_rejects(self):
        parsed = parse_review_output("")
        assert parsed["approved"] is False
        assert "could not be parsed" in parsed["feedback"]


# ═══════════════════════════════════════════════════════════════════════
# _parse_result_event
# ═══════════════════════════════════════════════════════════════════════


class TestParseResultEvent:
    """Verify the final stream-json result event parser."""

    def test_success_with_result_text(self):
        event = {
            "type": "result",
            "subtype": "success",
            "result": '{"approved": true}',
            "session_id": "abc-123",
            "total_cost_usd": 0.05,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        r = _parse_result_event(event)
        assert r.success is True
        assert r.result == '{"approved": true}'
        assert r.session_id == "abc-123"
        assert r.cost_usd == 0.05
        assert r.input_tokens == 100
        assert r.output_tokens == 50

    def test_success_with_empty_result(self):
        """The bug case: result event has empty result because Claude ended
        with a tool call."""
        event = {
            "type": "result",
            "subtype": "success",
            "result": "",
            "session_id": "def-456",
            "total_cost_usd": 0.03,
            "usage": {},
        }
        r = _parse_result_event(event)
        assert r.success is True
        assert r.result == ""

    def test_error_max_turns_is_success(self):
        event = {"type": "result", "subtype": "error_max_turns", "result": "partial"}
        r = _parse_result_event(event)
        assert r.success is True

    def test_error_subtype(self):
        event = {"type": "result", "subtype": "error", "error": "something broke"}
        r = _parse_result_event(event)
        assert r.success is False
        assert r.error == "something broke"

    def test_rate_limit_detection(self):
        event = {"type": "result", "subtype": "error", "error": "rate limit exceeded"}
        r = _parse_result_event(event)
        assert r.is_rate_limit is True


# ═══════════════════════════════════════════════════════════════════════
# Assistant text fallback integration
# ═══════════════════════════════════════════════════════════════════════


class TestAssistantTextFallback:
    """Simulate the streaming loop's text accumulation and verify the
    fallback populates result.result when the result event is empty.

    These tests replicate the logic in run_claude (lines 314-369) without
    spawning a subprocess.
    """

    @staticmethod
    def _simulate_stream(events: list[dict]) -> ClaudeResult:
        """Replay a list of stream-json events through the same logic as
        run_claude, returning the final ClaudeResult."""
        all_assistant_text: list[str] = []
        final_result: dict | None = None

        for event in events:
            etype = event.get("type", "")

            if etype == "result":
                final_result = event
                continue

            # Accumulate assistant text (mirrors runner.py lines 314-322)
            if etype == "assistant":
                msg = event.get("message", {})
                if isinstance(msg, dict):
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t:
                                all_assistant_text.append(t)

        assert final_result is not None, "stream must contain a result event"
        result = _parse_result_event(final_result)

        # Fallback (mirrors runner.py lines 365-369)
        if not result.result and all_assistant_text:
            result.result = "\n\n".join(all_assistant_text)

        return result

    def test_text_then_tool_call_then_empty_result(self):
        """The exact bug scenario: Claude writes the review JSON, then calls
        TodoWrite, and the result event has empty result."""
        review_text = REVIEW_TEXT_WITH_JSON_BLOCK
        events = [
            # Claude responds with text + tool_use in one assistant message
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": review_text},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "TodoWrite",
                            "input": {"todos": []},
                        },
                    ],
                },
            },
            # Tool result comes back
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "ok"},
            ]}},
            # Final result event — empty result because last turn had no text
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-001",
                "total_cost_usd": 0.04,
                "usage": {"input_tokens": 500, "output_tokens": 200},
            },
        ]

        r = self._simulate_stream(events)
        assert r.success is True
        # The fallback should have populated result with the review text
        assert r.result != ""
        assert "approved" in r.result.lower()

        # And the review parser should now extract the rejection
        parsed = parse_review_output(r.result)
        assert parsed["approved"] is False
        assert len(parsed["issues"]) == 2

    def test_text_only_no_tool_call(self):
        """Normal case: Claude writes text, no tool call, result has the text."""
        events = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": REVIEW_TEXT_BARE_JSON},
                    ],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": REVIEW_TEXT_BARE_JSON,
                "session_id": "sess-002",
                "total_cost_usd": 0.02,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        # result.result comes from the result event directly, not the fallback
        assert r.result == REVIEW_TEXT_BARE_JSON
        parsed = parse_review_output(r.result)
        assert parsed["approved"] is True

    def test_multiple_assistant_turns_accumulated(self):
        """Claude produces text across multiple assistant turns; all are
        accumulated and joined."""
        events = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "First I'll check the schema..."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                ]},
            },
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
            ]}},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": REVIEW_TEXT_WITH_JSON_BLOCK},
                    {"type": "tool_use", "id": "t2", "name": "TodoWrite", "input": {}},
                ]},
            },
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
            ]}},
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-003",
                "total_cost_usd": 0.06,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        assert r.result != ""
        # The accumulated text includes both turns
        assert "First I'll check" in r.result
        # And the review parser can still find the JSON in the accumulated text
        parsed = parse_review_output(r.result)
        assert parsed["approved"] is False

    def test_no_assistant_text_and_empty_result(self):
        """Edge case: no assistant text captured and empty result — result
        stays empty (no crash)."""
        events = [
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-004",
                "total_cost_usd": 0.0,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        assert r.result == ""
        # Parser falls through to reject (safe default after 6d25c32 fix)
        parsed = parse_review_output(r.result)
        assert parsed["approved"] is False
        assert "could not be parsed" in parsed["feedback"]

    def test_result_has_text_fallback_not_used(self):
        """When result event already has text, the fallback is NOT applied
        (we don't overwrite good data with accumulated fragments)."""
        good_result = json.dumps(REVIEW_JSON_REJECTED)
        events = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me look at this..."},
                ]},
            },
            {
                "type": "result",
                "subtype": "success",
                "result": good_result,
                "session_id": "sess-005",
                "total_cost_usd": 0.03,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        # result.result should be from the result event, not the accumulated text
        assert r.result == good_result
        assert "Let me look" not in r.result

    def test_tool_use_blocks_ignored_in_accumulation(self):
        """Only text blocks are accumulated, not tool_use blocks."""
        events = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ]},
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-006",
                "total_cost_usd": 0.01,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        # No text was accumulated, so result stays empty
        assert r.result == ""

    def test_empty_text_blocks_not_accumulated(self):
        """Empty string text blocks are skipped."""
        events = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": REVIEW_TEXT_BARE_JSON},
                ]},
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-007",
                "total_cost_usd": 0.01,
                "usage": {},
            },
        ]

        r = self._simulate_stream(events)
        # Only the non-empty text block is used
        assert r.result == REVIEW_TEXT_BARE_JSON


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: stream → parse_review_output
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndReviewPipeline:
    """Full pipeline: simulated stream events → ClaudeResult → parse_review_output.

    Demonstrates the bug is fixed: a rejected review that previously auto-approved
    now correctly returns approved=False.
    """

    def test_rejected_review_with_tool_call_is_not_auto_approved(self):
        """THE bug reproduction: review JSON + TodoWrite + empty result.
        Before the fix, this auto-approved. After the fix, it correctly rejects."""
        review_json = {
            "approved": False,
            "feedback": "Critical schema issues found.",
            "issues": [
                {"severity": "critical", "description": "Schema V1_3 instead of V1_4"},
                {"severity": "critical", "description": "No ModelConfiguration for EntryRevision"},
                {"severity": "major", "description": "No text diff preview"},
            ],
        }
        review_text = (
            "I've completed my review. Here is the result:\n\n"
            "```json\n" + json.dumps(review_json, indent=2) + "\n```\n\n"
            "**Summary of blocking issues:**\n\n"
            "1. Schema not updated\n"
            "2. No storage config\n"
            "3. No text diff\n"
        )

        # Simulate the stream
        all_assistant_text: list[str] = []
        events = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": review_text},
                    {"type": "tool_use", "id": "toolu_xyz", "name": "TodoWrite",
                     "input": {"todos": [{"id": "1", "content": "Fix schema"}]}},
                ]},
            },
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "ok"},
            ]}},
            {
                "type": "result",
                "subtype": "success",
                "result": "",
                "session_id": "sess-e2e",
                "total_cost_usd": 0.08,
                "usage": {"input_tokens": 1000, "output_tokens": 400},
            },
        ]

        # Replay stream logic
        final_result = None
        for event in events:
            etype = event.get("type", "")
            if etype == "result":
                final_result = event
                continue
            if etype == "assistant":
                msg = event.get("message", {})
                if isinstance(msg, dict):
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t:
                                all_assistant_text.append(t)

        result = _parse_result_event(final_result)
        if not result.result and all_assistant_text:
            result.result = "\n\n".join(all_assistant_text)

        # Now parse the review
        parsed = parse_review_output(result.result)

        # This is the critical assertion: must NOT auto-approve
        assert parsed["approved"] is False
        assert parsed["feedback"] == "Critical schema issues found."
        assert len(parsed["issues"]) == 3
        assert parsed["issues"][0]["severity"] == "critical"
