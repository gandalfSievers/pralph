"""A story that fails says why — in its metadata, in the terminal and in the viewer."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from pralph.loop import run_implement_loop
from pralph.models import Story, StoryStatus
from pralph.runner import ClaudeResult
from pralph.state import StateManager
from pralph.viewer import VIEWER_HTML, ViewerHandler, story_problem


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
        Story(id="API-001", title="List products", content="c", category="API"),
        Story(id="API-002", title="Product detail", content="c", category="API"),
    ])
    return st


def implement(state: StateManager, answer: ClaudeResult, **kwargs):
    with patch("pralph.loop.run_with_retry", side_effect=lambda prompt, **kw: answer):
        return run_implement_loop(state, cooldown=0, phase1=False, review=False, max_iterations=1, **kwargs)


def story(state: StateManager, story_id: str = "API-001") -> Story:
    return next(s for s in state.load_stories() if s.id == story_id)


# ── the reason is always recorded ─────────────────────────────────────


@pytest.mark.parametrize("single", [False, True], ids=["backlog", "--story-id"])
def test_reported_error_keeps_its_reason(state, capsys, single) -> None:
    answer = ClaudeResult(success=True, result=json.dumps(
        {"status": "error", "summary": "Stopped", "reason": "The products table does not exist.\nRun the migration first."}))
    implement(state, answer, **({"story_id": "API-001"} if single else {}))
    failed = story(state)
    assert failed.status == StoryStatus.error
    assert failed.metadata["error_reason"] == "The products table does not exist.\nRun the migration first."
    assert '"status": "error"' in failed.metadata["error_output"] and failed.metadata["error_at"]
    out = capsys.readouterr().out
    assert "✗ The products table does not exist." in out and "The full reason is in 'pralph viewer'." in out


@pytest.mark.parametrize("single", [False, True], ids=["backlog", "--story-id"])
def test_answer_that_cannot_be_read_is_quoted(state, single) -> None:
    answer = ClaudeResult(success=True, result="I looked around.\n\nThe build tool is not installed, so I stopped.")
    implement(state, answer, **({"story_id": "API-001"} if single else {}))
    reason = story(state).metadata["error_reason"]
    assert reason.startswith("The implementation ended without reporting a result.")
    assert "The build tool is not installed, so I stopped." in reason


def test_failed_run_keeps_its_error(state) -> None:
    implement(state, ClaudeResult(success=False, error="Timed out after 1800s", result="partial work"))
    failed = story(state)
    assert failed.status == StoryStatus.error and failed.metadata["error_reason"] == "Timed out after 1800s"
    assert failed.metadata["error_output"] == "partial work"


def test_error_without_any_words_still_has_a_reason(state) -> None:
    state.mark_story_status("API-001", StoryStatus.error)
    assert story(state).metadata["error_reason"] == "No reason was recorded."
    state.mark_story_status("API-002", StoryStatus.error, summary="gave up", extra={"reason": "no database"})
    assert story(state, "API-002").metadata["error_reason"] == "no database"


# ── the viewer shows it ───────────────────────────────────────────────


def test_problem_of_a_failed_story(state) -> None:
    assert story_problem(story(state)) is None
    state.mark_story_status("API-001", StoryStatus.error, error_reason="tests would not pass",
                            error_output="FAILED test_list.py::test_empty")
    problem = story_problem(story(state))
    assert problem["kind"] == "error" and problem["title"] == "Error"
    assert problem["reason"] == "tests would not pass" and problem["output"] == "FAILED test_list.py::test_empty"
    assert problem["at"] and problem["hint"] == "To retry: 'pralph reset-errors', then 'pralph implement'."


def test_problem_is_gone_once_the_story_is_through(state) -> None:
    state.mark_story_status("API-001", StoryStatus.error, error_reason="flaky")
    state.mark_story_status("API-001", StoryStatus.implemented, summary="second attempt")
    assert story_problem(story(state)) is None, "an old error is not shown on a story that made it"


def test_problem_of_a_paused_and_of_an_unshipped_story(state) -> None:
    def pause(s: Story) -> None:
        s.metadata["needs_human"] = {"reason": "Apply the migration", "at": "2026-01-02T10:00:00"}
    state.update_story("API-001", pause, notify=False)
    state.mark_story_status("API-001", StoryStatus.needs_human)
    problem = story_problem(story(state))
    assert problem["kind"] == "needs_human" and problem["reason"] == "Apply the migration"
    assert problem["hint"] == "When it is done: pralph unblock API-001"

    def unshipped(s: Story) -> None:
        s.metadata["git"] = {"branch": "api-002-product-detail", "base": "main", "ship_error": "permission denied"}
    state.update_story("API-002", unshipped, notify=False)
    state.mark_story_status("API-002", StoryStatus.implemented)
    problem = story_problem(story(state, "API-002"))
    assert problem["kind"] == "ship" and problem["reason"] == "permission denied"
    assert "pralph ship -s API-002" in problem["hint"] and "api-002-product-detail" in problem["hint"]


def test_viewer_serves_the_problem_with_the_story(state) -> None:
    state.mark_story_status("API-001", StoryStatus.error, error_reason='the "quoted" <reason>')
    server = HTTPServer(("127.0.0.1", 0), type("Handler", (ViewerHandler,), {"state": state}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/api/stories", timeout=5) as resp:
            served = {s["id"]: s for s in json.load(resp)}
        assert served["API-001"]["problem"]["reason"] == 'the "quoted" <reason>'
        assert served["API-002"]["problem"] is None

        # Saving an edit answers with the problem too, so the page keeps showing it.
        request = urllib.request.Request(
            f"{base}/api/stories/API-001", data=json.dumps({"title": "List all products"}).encode(),
            method="PUT", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as resp:
            saved = json.load(resp)
        assert saved["title"] == "List all products" and saved["problem"]["kind"] == "error"
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_page_renders_the_problem() -> None:
    assert "${renderProblem(s)}" in VIEWER_HTML and "function renderProblem(s)" in VIEWER_HTML
    assert 'class="sproblem"' in VIEWER_HTML and 'class="card-problem"' in VIEWER_HTML
    assert 'title="${esc(' not in VIEWER_HTML, "attribute values need escAttr: esc leaves quotes alone"
