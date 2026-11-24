"""Tests for ``pralph implement --story-id``: one story, reworked until the review approves."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pralph.loop import run_implement_loop
from pralph.models import Story, StoryStatus
from pralph.runner import ClaudeResult
from pralph.state import StateManager


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)


@pytest.fixture
def state(tmp_path) -> StateManager:
    root = tmp_path / "shop"
    root.mkdir()
    st = StateManager(str(root))
    st.append_stories([
        Story(id="API-001", title="List products", content="As a shopper, I want to list products.",
              acceptance_criteria=["GET /products returns 200"], category="API"),
        Story(id="API-002", title="Product detail", content="As a shopper, I want details.", category="API"),
    ])
    return st


class FakeClaude:
    """run_with_retry stand-in that rejects the first ``rejections`` reviews."""

    def __init__(self, rejections: int = 0, review_error: bool = False) -> None:
        self.rejections = rejections
        self.review_error = review_error
        self.calls: list[str] = []          # "implement" / "review" / "compound"
        self.implement_prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs) -> ClaudeResult:
        if "# Code Review Mode" in prompt:
            self.calls.append("review")
            if self.review_error:
                return ClaudeResult(success=False, error="reviewer crashed")
            if self.rejections > 0:
                self.rejections -= 1
                return ClaudeResult(success=True, result=json.dumps({
                    "approved": False, "feedback": "Not there yet",
                    "issues": [{"severity": "major", "description": "Pagination is missing"}]}))
            return ClaudeResult(success=True, result='{"approved": true, "feedback": "ok", "issues": []}')
        if "# Compound Learning" in prompt:
            self.calls.append("compound")
            return ClaudeResult(success=True, result='{"learnings": []}')
        self.calls.append("implement")
        self.implement_prompts.append(prompt)
        return ClaudeResult(success=True, result=json.dumps(
            {"status": "implemented", "summary": "Built it", "files_changed": ["api.py"]}))


def run_single(state: StateManager, claude: FakeClaude, **kwargs):
    with patch("pralph.loop.run_with_retry", side_effect=claude):
        return run_implement_loop(state, story_id="API-001", cooldown=0, phase1=False, **kwargs)


def status_of(state: StateManager, story_id: str = "API-001") -> StoryStatus:
    return next(s.status for s in state.load_stories() if s.id == story_id)


def test_approved_story_is_implemented_once(state) -> None:
    claude = FakeClaude()
    ps = run_single(state, claude)
    assert claude.calls == ["implement", "review"]
    assert ps.completion_reason == "single_story_done"
    assert status_of(state) == StoryStatus.implemented
    assert status_of(state, "API-002") == StoryStatus.pending, "only the requested story is touched"


def test_rejected_story_is_reworked_until_the_review_approves(state) -> None:
    claude = FakeClaude(rejections=2)
    ps = run_single(state, claude)
    assert claude.calls == ["implement", "review"] * 3
    assert ps.completion_reason == "single_story_done"
    assert status_of(state) == StoryStatus.implemented
    assert "Reviewer Feedback" not in claude.implement_prompts[0]
    for prompt in claude.implement_prompts[1:]:
        assert "Pagination is missing" in prompt, "the rework sees what the review asked for"
    assert state.read_review_feedback("API-001") == "", "feedback is cleared once approved"


def test_rework_is_capped_by_max_iterations(state) -> None:
    claude = FakeClaude(rejections=99)
    ps = run_single(state, claude, max_iterations=3)
    assert claude.calls == ["implement", "review"] * 3
    assert ps.completion_reason == "review_rejected"
    assert status_of(state) == StoryStatus.rework
    assert "Pagination is missing" in state.read_review_feedback("API-001")


def test_next_run_picks_up_a_story_left_in_rework(state) -> None:
    run_single(state, FakeClaude(rejections=99), max_iterations=1)
    claude = FakeClaude()
    ps = run_single(state, claude)
    assert ps.completion_reason == "single_story_done"
    assert "Pagination is missing" in claude.implement_prompts[0]


def test_compound_runs_once_after_the_approval(state) -> None:
    claude = FakeClaude(rejections=1)
    run_single(state, claude, compound=True)
    assert claude.calls == ["implement", "review", "implement", "review", "compound"]

    rejected = FakeClaude(rejections=99)
    run_single(state, rejected, compound=True, max_iterations=2)
    assert "compound" not in rejected.calls, "nothing to learn from a story that is not done"


def test_without_review_there_is_one_round(state) -> None:
    claude = FakeClaude(rejections=99)
    ps = run_single(state, claude, review=False)
    assert claude.calls == ["implement"]
    assert ps.completion_reason == "single_story_done"


def test_review_error_does_not_trigger_rework(state) -> None:
    claude = FakeClaude(review_error=True)
    ps = run_single(state, claude)
    assert claude.calls == ["implement", "review"]
    assert ps.completion_reason == "single_story_done"
    assert status_of(state) == StoryStatus.implemented
