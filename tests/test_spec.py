"""Tests for spec-driven development: in-repo specs, requirement tracing, tasks.md."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pralph import spec as spec_lib
from pralph.assembler import (
    assemble_add_prompt,
    assemble_implement_prompt,
    assemble_review_prompt,
    assemble_spec_prompt,
    assemble_stories_prompt,
)
from pralph.cli import main
from pralph.loop import run_refine, run_spec_loop, run_stories_loop
from pralph.models import PhaseState, Story, StoryStatus
from pralph.runner import ClaudeResult
from pralph.state import StateManager

SPEC_MD = """\
# Taskify — Specification

## Overview
A small task manager.

## Requirements

### REQ-001: Create tasks
Users can create a task with a title.

**Acceptance criteria**
- Given a title, when the user submits, then the task appears in the list
- Empty titles are rejected

### REQ-002: Complete tasks
Users can mark a task as done.

**Acceptance criteria**
- A done task is shown struck through

```markdown
### REQ-999: Not a requirement, just an example in a code block
```

## Non-functional requirements

### NFR-001: Fast startup
The app starts in under one second.

## Out of scope
Sharing.
"""


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep ~/.pralph (config, prompt overrides, global solutions) out of the tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)


@pytest.fixture
def project(tmp_path) -> Path:
    root = tmp_path / "taskify"
    root.mkdir()
    return root


def make_state(project: Path) -> StateManager:
    state = StateManager(str(project))
    spec_lib.attach(state)
    return state


def write_spec(project: Path, text: str = SPEC_MD, name: str = "taskify") -> Path:
    spec_dir = project / "specs" / name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "spec.md").write_text(text)
    (spec_dir / "design.md").write_text("# Taskify — Design\n\n## Architecture overview\nOne module.\n")
    return spec_dir


def story(sid: str, reqs: list[str] | None = None, *, spec: str | None = "taskify", **kw) -> Story:
    metadata = {}
    if reqs is not None:
        metadata["requirements"] = reqs
        if spec:
            metadata["spec"] = spec
    return Story(id=sid, title=kw.pop("title", f"Story {sid}"), content="As a user...",
                 category=sid.split("-")[0], metadata=metadata, **kw)


# ── requirement parsing ───────────────────────────────────────────────


class TestParseRequirements:
    def test_headings_are_parsed_with_their_sections(self) -> None:
        reqs = spec_lib.parse_requirements("taskify", SPEC_MD)
        assert [r.id for r in reqs] == ["REQ-001", "REQ-002", "NFR-001"]
        assert reqs[0].title == "Create tasks"
        assert "Empty titles are rejected" in reqs[0].body
        assert "REQ-002" not in reqs[0].body, "a requirement ends where the next one starts"

    def test_section_ends_at_a_higher_level_heading(self) -> None:
        reqs = {r.id: r for r in spec_lib.parse_requirements("taskify", SPEC_MD)}
        assert "Non-functional requirements" not in reqs["REQ-002"].body
        assert "Out of scope" not in reqs["NFR-001"].body

    def test_headings_inside_code_fences_are_ignored(self) -> None:
        reqs = spec_lib.parse_requirements("taskify", SPEC_MD)
        assert "REQ-999" not in [r.id for r in reqs]
        # ...but the fenced example stays part of the requirement it sits in
        assert "REQ-999" in next(r for r in reqs if r.id == "REQ-002").body

    def test_bullet_style_requirements(self) -> None:
        text = (
            "## Functional Requirements\n\n"
            "- **FR-001**: System MUST let users sign in\n"
            "  with an email address\n"
            "- **FR-002**: System MUST log sign-ins\n\n"
            "Unrelated paragraph.\n"
        )
        reqs = spec_lib.parse_requirements("auth", text)
        assert [r.id for r in reqs] == ["FR-001", "FR-002"]
        assert "with an email address" in reqs[0].body
        assert "Unrelated" not in reqs[1].body

    def test_first_occurrence_of_a_duplicate_id_wins(self) -> None:
        text = "### REQ-001: First\nA\n\n### REQ-001: Second\nB\n"
        reqs = spec_lib.parse_requirements("x", text)
        assert len(reqs) == 1 and reqs[0].title == "First"

    def test_other_dashed_headings_are_not_requirements(self) -> None:
        text = "### UTF-8 handling\nText\n\n### ISO-8601 dates\nText\n"
        assert spec_lib.parse_requirements("x", text) == []

    def test_digest_ignores_whitespace_but_not_wording(self) -> None:
        a = spec_lib.parse_requirements("x", "### REQ-001: T\nUsers can   create tasks.\n")[0]
        b = spec_lib.parse_requirements("x", "### REQ-001: T\n\nUsers can create\ntasks.\n\n")[0]
        c = spec_lib.parse_requirements("x", "### REQ-001: T\nUsers can delete tasks.\n")[0]
        assert a.digest == b.digest
        assert a.digest != c.digest


# ── spec mode / locating specs ────────────────────────────────────────


class TestSpecMode:
    def test_off_without_specs(self, project) -> None:
        assert make_state(project).spec_mode is False

    def test_on_as_soon_as_the_repo_has_a_spec(self, project) -> None:
        write_spec(project)
        assert make_state(project).spec_mode is True

    def test_a_specs_dir_without_spec_md_does_not_count(self, project) -> None:
        (project / "specs" / "openapi").mkdir(parents=True)
        (project / "specs" / "openapi" / "api.yaml").write_text("openapi: 3.1.0\n")
        assert make_state(project).spec_mode is False

    def test_config_overrides_detection(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        (state.state_dir / "config.json").write_text(json.dumps({"spec_driven": False}))
        assert state.spec_mode is False

    def test_spec_dir_is_configurable(self, project) -> None:
        state = make_state(project)
        (state.state_dir / "config.json").write_text(json.dumps({"spec_dir": "docs/specs"}))
        assert state.spec_dir == (project / "docs" / "specs").resolve()

    def test_spec_dir_cannot_escape_the_repo(self, project) -> None:
        state = make_state(project)
        (state.state_dir / "config.json").write_text(json.dumps({"spec_dir": "../elsewhere"}))
        assert state.spec_dir == project / "specs"

    def test_resolve_spec_name(self, project) -> None:
        state = make_state(project)
        assert spec_lib.resolve_spec_name(state, None) == "taskify", "defaults to the project dir name"
        assert spec_lib.resolve_spec_name(state, "User Auth!") == "user-auth"
        write_spec(project, name="payments")
        assert spec_lib.resolve_spec_name(state, None) == "payments", "the only spec is the default"
        write_spec(project, name="auth")
        with pytest.raises(ValueError, match="Several specs"):
            spec_lib.resolve_spec_name(state, None)

    def test_spec_name_cannot_be_a_path(self, project) -> None:
        state = make_state(project)
        assert spec_lib.get_spec(state, "../../etc").path.parent == state.spec_dir


# ── tracing, coverage, drift ──────────────────────────────────────────


class TestTracing:
    def test_story_trace_tolerates_llm_output(self, project) -> None:
        write_spec(project)
        specs = spec_lib.list_specs(make_state(project))
        assert spec_lib.story_trace(story("A-1", ["REQ-001", "req-002"]), specs) == ("taskify", ["REQ-001", "REQ-002"])
        as_string = Story(id="A-2", title="t", content="c", metadata={"requirements": "REQ-001, NFR-001"})
        assert spec_lib.story_trace(as_string, specs) == ("taskify", ["REQ-001", "NFR-001"]), \
            "a single spec is implied when the story doesn't name one"
        qualified = Story(id="A-3", title="t", content="c", metadata={"requirements": ["taskify:REQ-002"]})
        assert spec_lib.story_trace(qualified, specs) == ("taskify", ["REQ-002"])
        assert spec_lib.story_trace(Story(id="A-4", title="t", content="c"), specs) == (None, [])

    def test_uncovered_requirements(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        work = spec_lib.pending_work(state)
        assert [r.id for r in work.uncovered] == ["REQ-002", "NFR-001"]
        assert work.changed == []

    def test_editing_a_requirement_is_detected_as_change(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"]), story("TASK-002", ["REQ-002"])])
        assert spec_lib.pending_work(state).changed == []

        (spec_dir / "spec.md").write_text(SPEC_MD.replace(
            "Users can mark a task as done.", "Users can mark a task as done or archived."))
        changed = spec_lib.pending_work(state).changed
        assert [c.requirement.id for c in changed] == ["REQ-002"]
        assert [s.id for s in changed[0].stories] == ["TASK-002"]

    def test_reformatting_a_requirement_is_not_a_change(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        (spec_dir / "spec.md").write_text(SPEC_MD.replace(
            "Users can create a task with a title.", "Users can create a task\nwith a title.  "))
        assert spec_lib.pending_work(state).changed == []

    def test_acknowledge_clears_drift(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        (spec_dir / "spec.md").write_text(SPEC_MD.replace("with a title", "with a title and a due date"))
        assert spec_lib.pending_work(state).changed
        spec_lib.acknowledge(state)
        assert spec_lib.pending_work(state).changed == []

    def test_follow_up_story_acknowledges_the_changed_requirement(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        (spec_dir / "spec.md").write_text(SPEC_MD.replace("with a title", "with a title and a due date"))
        state.append_stories([story("TASK-002", ["REQ-001"], title="Add due dates")])
        assert spec_lib.pending_work(state).changed == []

    def test_stories_for_removed_requirements_are_reported(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-002"]), story("TASK-002", ["REQ-002"])])
        state.mark_story_status("TASK-002", StoryStatus.implemented)
        text = SPEC_MD[:SPEC_MD.index("### REQ-002")] + SPEC_MD[SPEC_MD.index("## Non-functional"):]
        (spec_dir / "spec.md").write_text(text)
        orphaned = spec_lib.pending_work(state).orphaned
        assert [(s.id, gone) for s, gone in orphaned] == [("TASK-001", ["REQ-002"])], \
            "only stories that would still be implemented are flagged"

    def test_untraced_stories_are_reported(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"]), story("MISC-001")])
        assert [s.id for s in spec_lib.pending_work(state).untraced] == ["MISC-001"]

    def test_requirement_state(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        state.append_stories([
            story("TASK-001", ["REQ-001"]), story("TASK-002", ["REQ-001"]),
            story("TASK-003", ["REQ-002"]),
        ])
        rows = {r.requirement.id: r for r in spec_lib.build_coverage(state)["taskify"]}
        assert rows["REQ-001"].state == "pending"
        assert rows["NFR-001"].state == "uncovered"

        state.mark_story_status("TASK-001", StoryStatus.implemented)
        state.mark_story_status("TASK-003", StoryStatus.skipped)
        rows = {r.requirement.id: r for r in spec_lib.build_coverage(state)["taskify"]}
        assert rows["REQ-001"].state == "in_progress"
        assert rows["REQ-002"].state == "dropped"

        state.mark_story_status("TASK-002", StoryStatus.implemented)
        rows = {r.requirement.id: r for r in spec_lib.build_coverage(state)["taskify"]}
        assert rows["REQ-001"].state == "done"

    def test_refined_stories_inherit_the_trace(self) -> None:
        originals = [story("A-1", ["REQ-001"]), story("A-2", ["REQ-002"])]
        new = [story("A-3"), story("A-4", ["NFR-001"])]
        spec_lib.inherit_trace(new, originals)
        assert new[0].metadata == {"spec": "taskify", "requirements": ["REQ-001", "REQ-002"]}
        assert new[1].metadata["requirements"] == ["NFR-001"], "an explicit trace is kept"


# ── tasks.md ──────────────────────────────────────────────────────────


class TestTasksFile:
    def test_written_when_stories_are_added(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([
            story("TASK-001", ["REQ-001"], title="Create task form", priority=1, complexity="small"),
            story("TASK-002", ["REQ-001", "REQ-002"], title="Task list", dependencies=["TASK-001"]),
            story("MISC-001"),
        ])
        tasks = (spec_dir / "tasks.md").read_text()
        assert "- [ ] **TASK-001** Create task form — REQ-001 · P1 · small" in tasks
        assert "- [ ] **TASK-002** Task list — REQ-001, REQ-002 · P3 · medium · after TASK-001" in tasks
        assert "MISC-001" not in tasks, "untraced stories are not part of a spec's task list"

    def test_follows_story_status(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"]), story("TASK-002", ["REQ-002"])])

        state.mark_story_status("TASK-001", StoryStatus.in_progress)
        assert "- [ ] **TASK-001**" in (spec_dir / "tasks.md").read_text()
        state.mark_story_status("TASK-001", StoryStatus.implemented)
        assert "- [x] **TASK-001**" in (spec_dir / "tasks.md").read_text()
        state.mark_story_status("TASK-001", StoryStatus.rework)
        assert "- [ ] **TASK-001**" in (spec_dir / "tasks.md").read_text(), "a rejected story is open again"
        state.mark_story_status("TASK-002", StoryStatus.skipped)
        assert "- [x] **TASK-002** ~~Story TASK-002~~ — REQ-002 · P3 · medium · skipped" in (spec_dir / "tasks.md").read_text()

    def test_implementer_ticking_the_box_matches_what_pralph_writes(self, project) -> None:
        """The implement prompt asks Claude to tick the box in its commit. pralph
        regenerating the file afterwards must not produce a diff."""
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"]), story("TASK-002", ["REQ-002"])])
        state.mark_story_status("TASK-001", StoryStatus.in_progress)

        tasks_path = spec_dir / "tasks.md"
        ticked_by_claude = tasks_path.read_text().replace("- [ ] **TASK-001**", "- [x] **TASK-001**")
        tasks_path.write_text(ticked_by_claude)
        mtime = tasks_path.stat().st_mtime_ns

        state.mark_story_status("TASK-001", StoryStatus.implemented)
        assert tasks_path.read_text() == ticked_by_claude
        assert tasks_path.stat().st_mtime_ns == mtime, "unchanged content is not rewritten"

    def test_a_tasks_file_pralph_did_not_write_is_left_alone(self, project) -> None:
        """A repo can have its own specs/ tree (other spec tooling uses the same
        layout). Spec mode switching on by itself must not cost anyone a file."""
        spec_dir = write_spec(project)
        tasks_path = spec_dir / "tasks.md"
        theirs = "# Tasks\n\n- [ ] T001 hand-written task\n"
        tasks_path.write_text(theirs)
        state = make_state(project)

        state.append_stories([story("MISC-001")])  # traces to nothing
        assert tasks_path.read_text() == theirs
        s = story("TASK-001", ["REQ-001"])
        state.append_stories([s])  # traces to this spec
        state.mark_story_status("TASK-001", StoryStatus.implemented)
        assert tasks_path.read_text() == theirs

        prompt = assemble_implement_prompt(state, s)
        assert "Tick this task" not in prompt, "nothing to tick in a file pralph doesn't maintain"

    def test_nothing_is_written_outside_spec_mode(self, project) -> None:
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        assert not (project / "specs").exists()
        assert not state.spec_baseline_path.exists()


# ── prompts ───────────────────────────────────────────────────────────


class TestPrompts:
    def test_first_spec_iteration_creates_the_files(self, project) -> None:
        state = make_state(project)
        spec = spec_lib.get_spec(state, "taskify")
        prompt = assemble_spec_prompt(state, spec, iteration=1, total=5, user_prompt="Build a {{task}} manager")
        assert "# Spec Authoring Mode" in prompt
        assert "`specs/taskify/spec.md`" in prompt and "`specs/taskify/design.md`" in prompt
        assert "guardrails.md" in prompt, "guardrails are created with the first spec"
        assert "### REQ-001: <Short title>" in prompt
        assert "{{" not in prompt.replace("{​{", ""), "no unresolved template variables"

    def test_existing_design_doc_is_migrated_into_the_spec(self, project) -> None:
        state = make_state(project)
        state.write_design_doc("# Old design\n")
        prompt = assemble_spec_prompt(state, spec_lib.get_spec(state, "taskify"), iteration=1, total=5, user_prompt="x")
        assert str(state.design_doc_path) in prompt

    def test_existing_spec_is_amended_not_recreated(self, project) -> None:
        write_spec(project)
        write_spec(project, name="payments")
        state = make_state(project)
        spec = spec_lib.get_spec(state, "taskify")
        prompt = assemble_spec_prompt(state, spec, iteration=1, total=5, user_prompt="Add due dates")
        assert "# Spec Refinement Mode" in prompt
        assert "Add due dates" in prompt
        assert "NEVER renumber or reuse an ID" in prompt
        assert "specs/payments/spec.md" in prompt, "other specs are listed for consistency"
        assert "{{" not in prompt

    def test_extract_prompt_lists_what_the_spec_asks_for(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"])])
        (spec_dir / "spec.md").write_text(SPEC_MD.replace("with a title", "with a title and a due date"))

        prompt = assemble_stories_prompt(state, mode="spec")
        new_section = prompt[prompt.index("### New"):prompt.index("### Changed")]
        changed_section = prompt[prompt.index("### Changed"):prompt.index("## Existing Stories")]
        assert "REQ-002: Complete tasks" in new_section and "NFR-001" in new_section
        assert "REQ-001" not in new_section
        assert "REQ-001: Create tasks (existing stories: TASK-001 [pending])" in changed_section
        assert "implements=taskify: REQ-001" in prompt
        assert "with a title and a due date" in prompt, "the spec text is inlined"
        assert '"spec": "taskify"' in prompt
        assert "{{" not in prompt

    def test_implement_prompt_carries_the_requirement_text(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        s = story("TASK-001", ["REQ-001"])
        state.append_stories([s])
        prompt = assemble_implement_prompt(state, s)
        assert "## Specification (source of truth)" in prompt
        assert "This story implements REQ-001 of spec `taskify`" in prompt
        assert "Empty titles are rejected" in prompt
        assert "Complete tasks" not in prompt, "only the story's own requirements are inlined"
        assert "`specs/taskify/design.md`" in prompt
        assert "`- [ ] **TASK-001**` to `- [x] **TASK-001**`" in prompt
        assert "untracked or have\n   uncommitted changes when you commit, include them" in prompt
        assert "{{" not in prompt

    def test_implement_prompt_flags_removed_requirements(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        s = story("TASK-009", ["REQ-001", "REQ-042"])
        state.append_stories([s])
        prompt = assemble_implement_prompt(state, s)
        assert "REQ-042 is referenced by this story but no longer exists" in prompt

    def test_implement_prompt_is_unchanged_outside_spec_mode(self, project) -> None:
        state = make_state(project)
        s = story("TASK-001")
        state.append_stories([s])
        prompt = assemble_implement_prompt(state, s)
        assert "Specification" not in prompt and "Linear" not in prompt

    def test_review_checks_conformance_to_the_spec(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        s = story("TASK-001", ["REQ-002"])
        state.append_stories([s])
        prompt = assemble_review_prompt(state, s)
        assert "## Specification Conformance" in prompt
        assert "A done task is shown struck through" in prompt
        assert "{{" not in prompt

    def test_other_phases_plan_from_the_spec(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        prompt = assemble_add_prompt(state, idea="dark mode", is_next=False)
        assert "### REQ-001: Create tasks" in prompt
        assert "(no design document)" not in prompt


# ── loops ─────────────────────────────────────────────────────────────


def claude_says(*outputs: str):
    """Patch run_with_retry to answer with the given outputs, one per call."""
    remaining = list(outputs)
    prompts: list[str] = []

    def fake(prompt: str, **kwargs) -> ClaudeResult:
        prompts.append(prompt)
        return ClaudeResult(success=True, result=remaining.pop(0))

    return patch("pralph.loop.run_with_retry", side_effect=fake), prompts


def stories_json(*stories: dict) -> str:
    return "```json\n" + json.dumps({"stories": list(stories)}) + "\n```"


DONE = '```json\n{"stories": []}\n```\n\n[GENERATION_COMPLETE]'


class TestStoriesLoop:
    def test_stories_are_derived_from_the_spec(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        patcher, prompts = claude_says(
            stories_json(
                {"id": "TASK-001", "title": "Create task", "content": "As a user...", "category": "TASK",
                 "spec": "taskify", "requirements": ["REQ-001"]},
                {"id": "TASK-002", "title": "Complete task", "content": "As a user...", "category": "TASK",
                 "requirements": "REQ-002, NFR-001"},
            ),
            DONE,
        )
        with patcher:
            ps = run_stories_loop(state, max_iterations=5, cooldown=0)

        assert ps.completion_reason == "generation_complete"
        assert "Story Extraction Mode (spec-driven)" in prompts[0]
        loaded = {s.id: s for s in state.load_stories()}
        assert loaded["TASK-001"].metadata["requirements"] == ["REQ-001"]
        assert loaded["TASK-002"].metadata == {"spec": "taskify", "requirements": ["REQ-002", "NFR-001"]}
        assert "**TASK-002**" in (spec_dir / "tasks.md").read_text()
        assert not spec_lib.pending_work(state).has_work

    def test_spec_mode_never_uses_research_mode(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        patcher, prompts = claude_says(DONE)
        with patcher:
            run_stories_loop(state, max_iterations=3, cooldown=0, extract_weight=0)
        assert "spec-driven" in prompts[0]

    def test_finished_extraction_reopens_when_the_spec_changes(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        covering = [
            {"id": f"TASK-00{i}", "title": "t", "content": "c", "category": "TASK", "requirements": [rid]}
            for i, rid in enumerate(["REQ-001", "REQ-002", "NFR-001"], start=1)
        ]
        patcher, _ = claude_says(stories_json(*covering), DONE)
        with patcher:
            run_stories_loop(state, max_iterations=5, cooldown=0)

        # Nothing changed: the phase stays done and Claude is not called.
        patcher, prompts = claude_says()
        with patcher:
            ps = run_stories_loop(state, max_iterations=5, cooldown=0)
        assert ps.completion_reason == "generation_complete" and prompts == []

        # Amend the spec: one new requirement, one changed.
        text = SPEC_MD.replace("with a title", "with a title and a due date").replace(
            "## Non-functional requirements",
            "### REQ-003: Delete tasks\nUsers can delete a task.\n\n## Non-functional requirements",
        )
        (spec_dir / "spec.md").write_text(text)
        follow_up = [
            {"id": "TASK-004", "title": "Delete task", "content": "c", "category": "TASK", "requirements": ["REQ-003"]},
            {"id": "TASK-005", "title": "Due dates", "content": "c", "category": "TASK", "requirements": ["REQ-001"]},
        ]
        patcher, prompts = claude_says(stories_json(*follow_up), DONE)
        with patcher:
            ps = run_stories_loop(state, max_iterations=5, cooldown=0)

        assert "REQ-003: Delete tasks" in prompts[0]
        assert "REQ-001: Create tasks (existing stories: TASK-001 [pending])" in prompts[0]
        assert ps.completion_reason == "generation_complete"
        assert len(state.load_stories()) == 5
        assert not spec_lib.pending_work(state).has_work

    def test_missing_spec_is_an_error_when_spec_mode_is_forced(self, project) -> None:
        state = make_state(project)
        (state.state_dir / "config.json").write_text(json.dumps({"spec_driven": True}))
        ps = run_stories_loop(state, max_iterations=1, cooldown=0)
        assert ps.completion_reason == "no_spec"

    def test_design_doc_flow_is_untouched(self, project) -> None:
        state = make_state(project)
        state.write_design_doc("# Design\n\nA thing.\n")
        patcher, prompts = claude_says(
            stories_json({"id": "API-001", "title": "t", "content": "c", "category": "API"}), DONE)
        with patcher:
            run_stories_loop(state, max_iterations=5, cooldown=0, extract_weight=100)
        assert "# Story Extraction Mode\n" in prompts[0]
        assert state.load_stories()[0].metadata == {}
        assert not (project / "specs").exists()


class TestSpecLoop:
    def test_each_invocation_is_a_new_round(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        state.save_phase_state(PhaseState(
            phase="spec", current_iteration=3, completed=True, completion_reason="planning_complete"))
        patcher, prompts = claude_says("<changes_summary>Added due dates</changes_summary>\n\n[PLANNING_COMPLETE]")
        with patcher:
            ps = run_spec_loop(state, spec_name="taskify", max_iterations=5, cooldown=0, user_prompt="Add due dates")
        assert len(prompts) == 1 and "Add due dates" in prompts[0]
        assert "iteration 1 of 5" in prompts[0]
        assert ps.completion_reason == "planning_complete"

    def test_creates_the_spec_directory(self, project) -> None:
        state = make_state(project)
        patcher, prompts = claude_says("[PLANNING_COMPLETE]")
        with patcher:
            run_spec_loop(state, spec_name="taskify", max_iterations=1, cooldown=0, user_prompt="A task app")
        assert (project / "specs" / "taskify").is_dir()
        assert "# Spec Authoring Mode" in prompts[0]


class TestRefine:
    def test_replacements_keep_the_requirement_coverage(self, project) -> None:
        write_spec(project)
        state = make_state(project)
        original = story("TASK-001", ["REQ-001"])
        state.append_stories([original])
        patcher, _ = claude_says(stories_json(
            {"id": "TASK-002", "title": "Form", "content": "c", "category": "TASK"},
            {"id": "TASK-003", "title": "Validation", "content": "c", "category": "TASK"},
        ))
        with patcher:
            run_refine(state, instruction="split", original_stories=[original])
        work = spec_lib.pending_work(state)
        assert "REQ-001" not in [r.id for r in work.uncovered]
        rows = {r.requirement.id: r for r in spec_lib.build_coverage(state)["taskify"]}
        assert rows["REQ-001"].state == "pending", "the skipped original doesn't make the requirement look dropped"


# ── CLI ───────────────────────────────────────────────────────────────


class TestCli:
    def test_spec_status(self, project) -> None:
        spec_dir = write_spec(project)
        state = make_state(project)
        state.append_stories([story("TASK-001", ["REQ-001"]), story("MISC-001")])
        state.mark_story_status("TASK-001", StoryStatus.implemented)
        (spec_dir / "spec.md").write_text(SPEC_MD.replace("with a title", "with a title and a due date"))

        result = CliRunner().invoke(main, ["--project-dir", str(project), "spec-status"])
        assert result.exit_code == 0, result.output
        assert "specs/taskify/spec.md — 3 requirements: 1 done, 2 uncovered" in result.output
        assert "(changed since stories were written)" in result.output
        assert "TASK-001 [implemented]" in result.output
        assert "1 stories do not trace to a requirement" in result.output
        assert "2 requirements without stories, 1 changed" in result.output

    def test_spec_status_without_specs(self, project) -> None:
        result = CliRunner().invoke(main, ["--project-dir", str(project), "spec-status"])
        assert result.exit_code == 0 and "No specs found" in result.output

    def test_spec_command_needs_a_name_when_several_specs_exist(self, project) -> None:
        write_spec(project, name="a")
        write_spec(project, name="b")
        result = CliRunner().invoke(main, ["--project-dir", str(project), "spec", "add x"])
        assert result.exit_code == 2 and "Several specs exist (a, b)" in result.output
