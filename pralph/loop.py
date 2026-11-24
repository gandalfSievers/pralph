from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import click

from pralph import ship as ship_lib
from pralph import spec as spec_lib
from pralph.assembler import (
    assemble_add_prompt,
    assemble_compound_prompt,
    assemble_ideate_prompt,
    assemble_implement_prompt,
    assemble_justloop_prompt,
    assemble_phase1_analyze_prompt,
    assemble_plan_prompt,
    assemble_refine_prompt,
    assemble_review_prompt,
    assemble_spec_prompt,
    assemble_stories_prompt,
    build_guardrails_system_prompt,
)
from pralph.linear import LinearError
from pralph.models import IterationResult, PhaseState, Story, StoryStatus
from pralph.parser import (
    detect_completion_signal,
    detect_ideation_complete,
    detect_loop_complete,
    extract_json_from_text,
    parse_compound_output,
    parse_implement_output,
    parse_plan_output,
    parse_review_output,
    parse_stories_output,
)
from pralph.runner import (
    ADD_TOOLS,
    COMPOUND_TOOLS,
    IDEATE_TOOLS,
    IMPLEMENT_TOOLS,
    JUSTLOOP_TOOLS,
    PLAN_TOOLS,
    REFINE_TOOLS,
    REVIEW_TOOLS,
    STORIES_TOOLS_EXTRACT,
    STORIES_TOOLS_RESEARCH,
    ClaudeResult,
    _stdin_from_tty,
    clear_loop_context,
    consume_truncate_request,
    resume_interactive,
    run_with_retry,
    set_loop_context,
)
from pralph.state import StateManager

# Foundation categories to prioritize in implementation
FOUNDATION_CATEGORIES = frozenset({
    "FND", "DBM", "SEC", "ARC", "ADM", "DAT", "DEP", "SYS", "INF",
    "INFRA", "ARCH", "DB", "AUTH", "SETUP", "FOUNDATION",
})


def _token_kwargs(cr: ClaudeResult) -> dict:
    """Extract token fields from a ClaudeResult as keyword arguments for IterationResult."""
    return {
        "input_tokens": cr.input_tokens,
        "output_tokens": cr.output_tokens,
        "cache_read_input_tokens": cr.cache_read_input_tokens,
        "cache_creation_input_tokens": cr.cache_creation_input_tokens,
    }


# ── session resume support ───────────────────────────────────────────


def _session_resume_prompt(ps: PhaseState) -> str:
    """Show resume prompt for interrupted session. Returns choice string."""
    with _stdin_from_tty():
        click.echo()
        click.echo(click.style("  \U0001f504 Found interrupted session", fg="yellow", bold=True))
        click.echo(f"   Phase: {ps.phase}", nl=False)
        if ps.active_story_id:
            click.echo(f", Story: {ps.active_story_id}", nl=False)
        click.echo()
        if ps.active_session_started:
            click.echo(f"   Started: {ps.active_session_started}")
        click.echo()
        click.echo("   [1] Resume headlessly  \u2014 continue automated session")
        click.echo("   [2] Resume interactive \u2014 open interactive Claude session")
        click.echo("   [3] Start fresh        \u2014 start a new session")
        click.echo("   [4] Abort")
        click.echo()
        choice = click.prompt(
            "   Choice", type=click.Choice(["1", "2", "3", "4"]), default="1",
        )
    return {"1": "headless", "2": "interactive", "3": "fresh", "4": "abort"}[choice]


def _clear_session_tracking(ps: PhaseState) -> None:
    """Clear active session tracking fields on PhaseState."""
    ps.active_session_id = ""
    ps.active_story_id = ""
    ps.active_session_started = ""


def _suggest_compact(state: StateManager) -> None:
    """Print a hint to run compact-index if solutions exist."""
    if state.has_solutions() or state.has_global_solutions():
        click.echo(click.style("\n  Tip:", dim=True) + click.style(" run ", dim=True)
                   + click.style("pralph compact-index", fg='cyan')
                   + click.style(" to deduplicate & prune solution indexes", dim=True))


# ── generic iteration loop ───────────────────────────────────────────


def _run_loop(
    phase: str,
    state: StateManager,
    max_iterations: int,
    cooldown: int,
    iteration_fn: Callable[[int, PhaseState], IterationResult],
    completion_fn: Callable[[IterationResult, PhaseState], bool],
    resume_fn: Callable[[str, PhaseState], IterationResult] | None = None,
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
) -> PhaseState:
    """Generic iteration loop shared by all phases."""
    ps = state.load_phase_state(phase)

    # Check for resumable session from previous crash
    if ps.active_session_id and resume_fn:
        # A story with a worktree of its own ran there, and so did its session.
        session_dir = ship_lib.story_workdir(state, ps.active_story_id) if ps.active_story_id else state.project_dir
        if not state.claude_session_exists(ps.active_session_id, session_dir):
            click.echo("  Previous session not found on disk, starting fresh")
            _clear_session_tracking(ps)
            state.save_phase_state(ps)
        else:
            choice = _session_resume_prompt(ps)
            if choice == "headless":
                result = resume_fn(ps.active_session_id, ps)
                state.log_iteration(result)
                ps.total_cost_usd += result.cost_usd
                _clear_session_tracking(ps)
                if result.success:
                    ps.consecutive_errors = 0
                if completion_fn(result, ps):
                    ps.completed = True
                    state.save_phase_state(ps)
                    return ps
                state.save_phase_state(ps)
            elif choice == "interactive":
                resume_interactive(ps.active_session_id, str(session_dir), dangerously_skip_permissions)
                _clear_session_tracking(ps)
                state.save_phase_state(ps)
            elif choice == "abort":
                ps.completed = True
                ps.completion_reason = "user_aborted"
                state.save_phase_state(ps)
                return ps
            else:  # "fresh"
                _clear_session_tracking(ps)
                state.save_phase_state(ps)
    elif ps.active_session_id:
        # No resume_fn but stale tracking — clear it
        _clear_session_tracking(ps)
        state.save_phase_state(ps)

    # Only truly "done" completions block re-running.
    # Everything else (errors, max_iterations) is resumable.
    DONE_REASONS = {"generation_complete", "planning_complete", "all_stories_done", "single_story_done", "loop_complete"}

    if ps.completed:
        if ps.completion_reason == "all_stories_done" and state.get_actionable_stories():
            # Stories were reset back to pending/rework — resume
            click.echo(f"  Phase '{phase}' resuming (stories reset to actionable)...")
            ps.completed = False
            ps.completion_reason = ""
        elif ps.completion_reason in DONE_REASONS:
            click.echo(f"  Phase '{phase}' already completed: {ps.completion_reason}")
            return ps
        else:
            # Resumable — reset and continue from where we left off
            click.echo(f"  Phase '{phase}' resuming (previously: {ps.completion_reason})...")
            ps.completed = False
            ps.consecutive_errors = 0
            ps.completion_reason = ""

    start_iter = ps.current_iteration + 1 if ps.current_iteration > 0 else 1
    last_iter = start_iter + max_iterations - 1

    # Clear any stale truncate flag from a previous run.
    consume_truncate_request()

    try:
        for i in range(start_iter, start_iter + max_iterations):
            ps.current_iteration = i
            set_loop_context(phase, i, last_iter)
            header = f"  [{phase}] iteration {i}"
            click.echo(f"\n{click.style('='*60, fg='cyan')}")
            click.echo(click.style(header, fg='cyan', bold=True))
            click.echo(click.style('='*60, fg='cyan'))

            t0 = time.time()
            result = iteration_fn(i, ps)
            result.duration = time.time() - t0

            state.log_iteration(result)
            ps.total_cost_usd += result.cost_usd

            # User abort — stop immediately
            if result.error == "aborted":
                ps.completed = True
                ps.completion_reason = "user_aborted"
                state.save_phase_state(ps)
                click.echo(click.style(f"\n  Phase '{phase}' aborted by user", fg='yellow'))
                return ps

            if result.success:
                ps.consecutive_errors = 0
                ps.last_error = ""
                ps.last_summary = result.raw_output[:500] if result.raw_output else result.impl_status
                click.echo(click.style(f"  ✓ success", fg='green', bold=True) + f" (${result.cost_usd:.4f}, {result.duration:.1f}s)")
            elif result.error == "interrupted":
                # User interrupted but chose to continue — don't count as error
                click.echo(click.style(f"  ⏭ skipped (interrupted)", fg='yellow'))
            else:
                ps.consecutive_errors += 1
                ps.last_error = result.error[:500]
                click.echo(click.style(f"  ✗ error:", fg='red', bold=True) + f" {result.error[:120]}")

            if completion_fn(result, ps):
                ps.completed = True
                state.save_phase_state(ps)
                click.echo(click.style(f"\n  Phase '{phase}' complete: {ps.completion_reason}", fg='green'))
                _suggest_compact(state)
                return ps

            # User asked to truncate max-iterations to current — exit cleanly.
            if consume_truncate_request():
                ps.completed = True
                ps.completion_reason = "user_stopped"
                state.save_phase_state(ps)
                click.echo(click.style(
                    f"\n  Phase '{phase}' stopped by user after iteration {i} "
                    f"(max-iterations truncated)", fg='yellow',
                ))
                _suggest_compact(state)
                return ps

            state.save_phase_state(ps)

            if i < last_iter:
                time.sleep(cooldown)

        ps.completed = True
        ps.completion_reason = "max_iterations"
        state.save_phase_state(ps)
        click.echo(click.style(f"\n  Phase '{phase}' complete: max iterations reached", fg='yellow'))
        _suggest_compact(state)
        return ps
    finally:
        clear_loop_context()


# ── Phase 1: Plan ────────────────────────────────────────────────────


def _run_authoring_loop(
    phase: str,
    state: StateManager,
    *,
    prompt_fn: Callable[[int, PhaseState], str],
    resume_prompt: str,
    model: str,
    max_iterations: int,
    cooldown: int,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
) -> PhaseState:
    """Create-then-refine loop for a planning document (design doc or spec).

    Runs until Claude signals ``[PLANNING_COMPLETE]``.
    """
    system_prompt = build_guardrails_system_prompt("plan", state)

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        prompt = prompt_fn(i, ps)

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=PLAN_TOOLS,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=900,
            verbose=verbose,
            project_dir=str(state.project_dir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            return IterationResult(
                iteration=i, phase=phase, mode="refine",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        parsed = parse_plan_output(result.result)
        summary = parsed.get("changes_summary", "")
        if summary:
            click.echo(click.style("  Changes:", fg='blue') + f" {summary[:200]}")

        return IterationResult(
            iteration=i, phase=phase, mode="create" if i == 1 else "refine",
            success=True, raw_output=result.result,
            cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        result = run_with_retry(
            resume_prompt,
            resume_session_id=session_id,
            timeout=900,
            project_dir=str(state.project_dir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )
        if not result.success:
            return IterationResult(
                iteration=ps.current_iteration, phase=phase, mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )
        parsed = parse_plan_output(result.result)
        summary = parsed.get("changes_summary", "")
        if summary:
            click.echo(click.style("  Changes:", fg='blue') + f" {summary[:200]}")
        return IterationResult(
            iteration=ps.current_iteration, phase=phase, mode="resume",
            success=True, raw_output=result.result,
            cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if result.raw_output and any(
            line.strip() == "[PLANNING_COMPLETE]" for line in result.raw_output.splitlines()
        ):
            ps.completion_reason = "planning_complete"
            return True
        if ps.consecutive_errors >= 5:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    return _run_loop(phase, state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)


def run_plan_loop(
    state: StateManager,
    *,
    model: str = "sonnet",
    max_iterations: int = 5,
    cooldown: int = 5,
    user_prompt: str = "",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Run the plan refinement loop."""
    total = max_iterations

    def prompt_fn(i: int, ps: PhaseState) -> str:
        return assemble_plan_prompt(state, iteration=i, total=total, user_prompt=user_prompt, phase_state=ps)

    return _run_authoring_loop(
        "plan", state,
        prompt_fn=prompt_fn,
        resume_prompt="Continue refining the design document.",
        model=model, max_iterations=max_iterations, cooldown=cooldown, verbose=verbose,
        dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
    )


# ── Phase 1 (spec-driven): Spec ──────────────────────────────────────


def run_spec_loop(
    state: StateManager,
    *,
    spec_name: str,
    model: str = "sonnet",
    max_iterations: int = 5,
    cooldown: int = 5,
    user_prompt: str = "",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Create, refine or amend an in-repo spec (``<spec_dir>/<name>/spec.md`` + ``design.md``)."""
    spec = spec_lib.get_spec(state, spec_name)
    spec.path.mkdir(parents=True, exist_ok=True)
    total = max_iterations

    # A finished spec run is not the end of the spec: every new invocation is a
    # new change request against it, so start a fresh round of iterations.
    previous = state.load_phase_state("spec")
    if previous.completed and previous.completion_reason == "planning_complete":
        state.save_phase_state(PhaseState(phase="spec"))

    def prompt_fn(i: int, ps: PhaseState) -> str:
        return assemble_spec_prompt(state, spec, iteration=i, total=total, user_prompt=user_prompt, phase_state=ps)

    ps = _run_authoring_loop(
        "spec", state,
        prompt_fn=prompt_fn,
        resume_prompt="Continue working on the specification.",
        model=model, max_iterations=max_iterations, cooldown=cooldown, verbose=verbose,
        dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
    )

    _report_spec(state, spec)
    return ps


def _report_spec(state: StateManager, spec: spec_lib.Spec) -> None:
    """Summarize the spec after authoring and point at the next step."""
    if not spec.exists():
        click.echo(click.style(f"\n  No spec was written to {spec_lib.rel_path(state, spec.spec_path)}", fg='red'))
        return
    requirements = spec.requirements()
    click.echo(f"\n  Spec: {click.style(spec_lib.rel_path(state, spec.spec_path), fg='blue')}"
               f" — {len(requirements)} requirements")
    if not requirements:
        click.echo(click.style(
            "  Warning: no requirement IDs found. Requirements need headings like "
            "'### REQ-001: Title' for stories to trace back to them.", fg='yellow',
        ))
        return
    work = spec_lib.pending_work(state)
    mine_new = [r for r in work.uncovered if r.spec == spec.name]
    mine_changed = [c for c in work.changed if c.requirement.spec == spec.name]
    if mine_new or mine_changed:
        click.echo(f"  {len(mine_new)} without stories, {len(mine_changed)} changed since their stories were written")
    # pralph leaves the commit to you: a spec change is worth a look before it
    # becomes the source of truth.
    click.echo(click.style("  Next:", dim=True) + " review and commit "
               + click.style(spec_lib.rel_path(state, spec.path) + "/", fg='blue')
               + (", then " + click.style("pralph stories", fg='cyan') if mine_new or mine_changed else ""))


# ── Phase 2: Stories ──────────────────────────────────────────────────


def run_stories_loop(
    state: StateManager,
    *,
    model: str = "sonnet",
    max_iterations: int = 50,
    cooldown: int = 5,
    extract_weight: int = 80,
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Run the story extraction loop.

    In spec mode stories are derived from the in-repo specs only (no research
    mode — everything has to trace back to a requirement), and a finished
    extraction resumes by itself when the spec has gained or changed requirements.
    """
    spec_mode = state.spec_mode
    if spec_mode:
        if not spec_lib.list_specs(state):
            click.echo(f"Error: No spec found in {state.spec_dir}. Run 'pralph spec' first.")
            return PhaseState(phase="stories", completed=True, completion_reason="no_spec")
        _prepare_spec_extraction(state)
    elif not state.has_design_doc():
        click.echo("Error: No design document found. Run 'pralph plan' (or 'pralph spec') first.")
        return PhaseState(phase="stories", completed=True, completion_reason="no_design_doc")

    system_prompt = build_guardrails_system_prompt("stories", state)

    def _pick_mode() -> str:
        if spec_mode:
            return "spec"
        return "extract" if random.randint(1, 100) <= extract_weight else "research"

    def _accept(stories: list[Story]) -> list[Story]:
        """Drop stories we already have; in spec mode also tidy up requirement tracing."""
        existing_ids = state.get_story_ids()
        new_stories = [s for s in stories if s.id not in existing_ids]
        if spec_mode and new_stories:
            for s in spec_lib.normalize_trace(state, new_stories):
                click.echo(click.style(f"    Warning: {s.id} does not trace to a spec requirement", fg='yellow'))
        return new_stories

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        mode = _pick_mode()
        prompt = assemble_stories_prompt(state, mode=mode, phase_state=ps)
        tools = STORIES_TOOLS_RESEARCH if mode == "research" else STORIES_TOOLS_EXTRACT

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=tools,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=900 if mode == "research" else 600,
            verbose=verbose,
            project_dir=str(state.project_dir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            return IterationResult(
                iteration=i, phase="stories", mode=mode,
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        stories, is_complete = parse_stories_output(result.result)

        # Deduplicate against existing
        new_stories = _accept(stories)

        if new_stories:
            state.append_stories(new_stories)
            click.echo(click.style(f"  +{len(new_stories)} stories", fg='green', bold=True) + f" (mode={mode})")
            for s in new_stories:
                click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
            ps.consecutive_empty = 0
        else:
            ps.consecutive_empty += 1
            click.echo(click.style(f"  0 new stories", fg='yellow') + f" (mode={mode}, consecutive_empty={ps.consecutive_empty})")

        return IterationResult(
            iteration=i, phase="stories", mode=mode,
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def _stories_resume_process(result: ClaudeResult) -> IterationResult:
        """Shared result processing for stories resume."""
        if not result.success:
            return IterationResult(
                iteration=0, phase="stories", mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )
        stories, _ = parse_stories_output(result.result)
        new_stories = _accept(stories)
        if new_stories:
            state.append_stories(new_stories)
            click.echo(click.style(f"  +{len(new_stories)} stories (resumed)", fg='green', bold=True))
            for s in new_stories:
                click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
        return IterationResult(
            iteration=0, phase="stories", mode="resume",
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        result = run_with_retry(
            "Continue extracting stories.",
            resume_session_id=session_id,
            timeout=900,
            project_dir=str(state.project_dir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )
        ir = _stories_resume_process(result)
        ir.iteration = ps.current_iteration
        return ir

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if detect_completion_signal(result.raw_output):
            ps.completion_reason = "generation_complete"
            return True
        if ps.consecutive_empty >= 3:
            ps.completion_reason = "consecutive_empty"
            return True
        if ps.consecutive_errors >= 5:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    ps = _run_loop("stories", state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)

    if spec_mode and ps.completion_reason in ("generation_complete", "consecutive_empty"):
        # The backlog now reflects the spec as it is today; only later edits are drift.
        spec_lib.acknowledge(state)
        spec_lib.sync_tasks(state)

    total = len(state.load_stories())
    click.echo(f"\n  Total stories: {total}")
    return ps


def _prepare_spec_extraction(state: StateManager) -> None:
    """Report what the spec asks for and reopen a finished extraction if it changed."""
    work = spec_lib.pending_work(state)
    specs = spec_lib.list_specs(state)
    total_reqs = sum(len(s.requirements()) for s in specs)
    click.echo(f"  spec-driven: {len(specs)} spec(s), {total_reqs} requirements — "
               f"{len(work.uncovered)} without stories, {len(work.changed)} changed")
    for story, gone in work.orphaned:
        click.echo(click.style(
            f"  Warning: {story.id} implements {', '.join(gone)} which no longer exists in the spec "
            f"— refine or skip it", fg='yellow',
        ))

    previous = state.load_phase_state("stories")
    finished = previous.completed and previous.completion_reason in ("generation_complete", "consecutive_empty")
    if finished and work.has_work:
        click.echo("  Spec changed since the last extraction — deriving stories for the difference")
        state.save_phase_state(PhaseState(phase="stories"))


# ── Add (single story) ────────────────────────────────────────────────


def run_add(
    state: StateManager,
    *,
    idea: str,
    is_next: bool = False,
    model: str = "sonnet",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> Story | None:
    """Create a single story from an idea. One Claude call, no loop."""
    system_prompt = build_guardrails_system_prompt("stories", state)
    prompt = assemble_add_prompt(state, idea=idea, is_next=is_next)

    result = run_with_retry(
        prompt,
        model=model,
        allowed_tools=ADD_TOOLS,
        system_prompt=system_prompt,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        timeout=600,
        verbose=verbose,
        project_dir=str(state.project_dir),
    )

    if not result.success:
        click.echo(click.style(f"  Error: {result.error[:200]}", fg='red'))
        state.log_iteration(IterationResult(
            iteration=1, phase="add", mode="add",
            success=False, error=result.error, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        ))
        return None

    stories, _ = parse_stories_output(result.result)
    if not stories:
        click.echo(click.style("  No story could be parsed from Claude's response", fg='red'))
        state.log_iteration(IterationResult(
            iteration=1, phase="add", mode="add",
            success=False, error="no stories parsed", cost_usd=result.cost_usd,
            **_token_kwargs(result),
        ))
        return None

    story = stories[0]
    story.source = "manual"
    if is_next:
        story.priority = 1

    # Dedup check
    existing_ids = state.get_story_ids()
    if story.id in existing_ids:
        # Append suffix to avoid collision
        base = story.id
        for suffix in range(2, 100):
            candidate = f"{base}b{suffix}"
            if candidate not in existing_ids:
                story.id = candidate
                break

    state.append_stories([story])
    state.log_iteration(IterationResult(
        iteration=1, phase="add", mode="add",
        success=True, stories_generated=1,
        raw_output=result.result, cost_usd=result.cost_usd,
        story_id=story.id,
        **_token_kwargs(result),
    ))

    return story


# ── Refine (replace stories) ──────────────────────────────────────────


def run_refine(
    state: StateManager,
    *,
    instruction: str,
    original_stories: list[Story],
    model: str = "sonnet",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> list[Story]:
    """Refine existing stories based on an instruction. One Claude call, no loop."""
    system_prompt = build_guardrails_system_prompt("stories", state)
    prompt = assemble_refine_prompt(state, instruction=instruction, original_stories=original_stories)

    result = run_with_retry(
        prompt,
        model=model,
        allowed_tools=REFINE_TOOLS,
        system_prompt=system_prompt,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        timeout=600,
        verbose=verbose,
        project_dir=str(state.project_dir),
    )

    if not result.success:
        click.echo(click.style(f"  Error: {result.error[:200]}", fg='red'))
        state.log_iteration(IterationResult(
            iteration=1, phase="refine", mode="refine",
            success=False, error=result.error, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        ))
        return []

    stories, _ = parse_stories_output(result.result)
    if not stories:
        click.echo(click.style("  No stories could be parsed from Claude's response", fg='red'))
        state.log_iteration(IterationResult(
            iteration=1, phase="refine", mode="refine",
            success=False, error="no stories parsed", cost_usd=result.cost_usd,
            **_token_kwargs(result),
        ))
        return []

    # Set source and refined_from metadata
    original_ids = [s.id for s in original_stories]
    for s in stories:
        s.source = "refine"
        s.metadata["refined_from"] = original_ids
    spec_lib.inherit_trace(stories, original_stories)

    # Dedup IDs against existing
    existing_ids = state.get_story_ids()
    for s in stories:
        if s.id in existing_ids:
            base = s.id
            for suffix in range(2, 100):
                candidate = f"{base}b{suffix}"
                if candidate not in existing_ids:
                    s.id = candidate
                    break
        existing_ids.add(s.id)

    # Append new stories
    state.append_stories(stories)

    # Mark originals as skipped
    new_ids = [s.id for s in stories]
    refined_into = ", ".join(new_ids)
    for orig in original_stories:
        state.mark_story_status(orig.id, StoryStatus.skipped, summary=f"Refined into: {refined_into}")

    state.log_iteration(IterationResult(
        iteration=1, phase="refine", mode="refine",
        success=True, stories_generated=len(stories),
        raw_output=result.result, cost_usd=result.cost_usd,
        **_token_kwargs(result),
    ))

    return stories


# ── Ideate (batch) ────────────────────────────────────────────────────


def run_ideate_loop(
    state: StateManager,
    *,
    ideas_text: str,
    model: str = "sonnet",
    max_iterations: int = 10,
    cooldown: int = 5,
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Process a batch of ideas into stories via the standard loop."""
    system_prompt = build_guardrails_system_prompt("stories", state)

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        prompt = assemble_ideate_prompt(state, ideas_text=ideas_text, phase_state=ps)

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=IDEATE_TOOLS,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=900,
            verbose=verbose,
            project_dir=str(state.project_dir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            return IterationResult(
                iteration=i, phase="ideate", mode="ideate",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        stories, _ = parse_stories_output(result.result)

        # Override source on all stories
        for s in stories:
            s.source = "ideate"

        # Deduplicate against existing
        existing_ids = state.get_story_ids()
        new_stories = [s for s in stories if s.id not in existing_ids]

        if new_stories:
            state.append_stories(new_stories)
            click.echo(click.style(f"  +{len(new_stories)} stories", fg='green', bold=True))
            for s in new_stories:
                click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
            ps.consecutive_empty = 0
        else:
            ps.consecutive_empty += 1
            click.echo(click.style(f"  0 new stories", fg='yellow') + f" (consecutive_empty={ps.consecutive_empty})")

        return IterationResult(
            iteration=i, phase="ideate", mode="ideate",
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        result = run_with_retry(
            "Continue processing ideas into stories.",
            resume_session_id=session_id,
            timeout=900,
            project_dir=str(state.project_dir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )
        if not result.success:
            return IterationResult(
                iteration=ps.current_iteration, phase="ideate", mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )
        stories, _ = parse_stories_output(result.result)
        for s in stories:
            s.source = "ideate"
        existing_ids = state.get_story_ids()
        new_stories = [s for s in stories if s.id not in existing_ids]
        if new_stories:
            state.append_stories(new_stories)
            click.echo(click.style(f"  +{len(new_stories)} stories (resumed)", fg='green', bold=True))
            for s in new_stories:
                click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
        return IterationResult(
            iteration=ps.current_iteration, phase="ideate", mode="resume",
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if detect_ideation_complete(result.raw_output):
            ps.completion_reason = "ideation_complete"
            return True
        if ps.consecutive_empty >= 2:
            ps.completion_reason = "consecutive_empty"
            return True
        if ps.consecutive_errors >= 3:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    ps = _run_loop("ideate", state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)

    total = len(state.load_stories())
    click.echo(f"\n  Total stories: {total}")
    return ps


# ── Phase 2b: Webgen ─────────────────────────────────────────────────


def run_webgen_loop(
    state: StateManager,
    *,
    model: str = "sonnet",
    max_iterations: int = 50,
    cooldown: int = 5,
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Run the web-gen requirements discovery loop."""
    if not state.has_design_doc() and not (state.spec_mode and spec_lib.list_specs(state)):
        click.echo("Error: No design document found. Run 'pralph plan' (or 'pralph spec') first.")
        return PhaseState(phase="webgen", completed=True, completion_reason="no_design_doc")

    system_prompt = build_guardrails_system_prompt("stories", state)

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        prompt = assemble_stories_prompt(state, mode="webgen", phase_state=ps)

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=STORIES_TOOLS_RESEARCH,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=900,
            verbose=verbose,
            project_dir=str(state.project_dir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            return IterationResult(
                iteration=i, phase="webgen", mode="webgen",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        stories, is_complete = parse_stories_output(result.result)

        existing_ids = state.get_story_ids()
        new_stories = [s for s in stories if s.id not in existing_ids]

        if new_stories:
            for s in new_stories:
                s.source = "webgen"
            state.append_stories(new_stories)
            click.echo(f"  +{len(new_stories)} webgen stories")
            for s in new_stories:
                click.echo(f"    {s.id}: {s.title}")
            ps.consecutive_empty = 0
        else:
            ps.consecutive_empty += 1
            click.echo(f"  0 new stories (consecutive_empty={ps.consecutive_empty})")

        return IterationResult(
            iteration=i, phase="webgen", mode="webgen",
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        result = run_with_retry(
            "Continue discovering web-gen requirements.",
            resume_session_id=session_id,
            timeout=900,
            project_dir=str(state.project_dir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )
        if not result.success:
            return IterationResult(
                iteration=ps.current_iteration, phase="webgen", mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )
        stories, _ = parse_stories_output(result.result)
        existing_ids = state.get_story_ids()
        new_stories = [s for s in stories if s.id not in existing_ids]
        if new_stories:
            for s in new_stories:
                s.source = "webgen"
            state.append_stories(new_stories)
            click.echo(click.style(f"  +{len(new_stories)} webgen stories (resumed)", fg='green', bold=True))
            for s in new_stories:
                click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
        return IterationResult(
            iteration=ps.current_iteration, phase="webgen", mode="resume",
            success=True, stories_generated=len(new_stories),
            raw_output=result.result, cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if detect_completion_signal(result.raw_output):
            ps.completion_reason = "generation_complete"
            return True
        if ps.consecutive_empty >= 3:
            ps.completion_reason = "consecutive_empty"
            return True
        if ps.consecutive_errors >= 5:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    ps = _run_loop("webgen", state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)

    total = len(state.load_stories())
    click.echo(f"\n  Total stories: {total}")
    return ps


# ── Phase 3: Implement ───────────────────────────────────────────────


def run_implement_loop(
    state: StateManager,
    *,
    model: str = "sonnet",
    max_iterations: int = 50,
    cooldown: int = 5,
    story_id: str | None = None,
    phase1: bool = True,
    review: bool = True,
    compound: bool = False,
    save_global: bool = False,
    user_prompt: str = "",
    extra_tools: str = "",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
    tracker=None,
    wait: bool = True,
) -> PhaseState:
    """Run the implementation loop.

    With a ``tracker`` (Linear) the tickets are the source of truth: the
    backlog is refreshed from the tracker before every story, so work that
    was re-prioritized, added, taken or closed there is respected.

    With a branch per story (see ``pralph.ship``) a story waits until the
    stories it depends on are merged. When nothing else can be worked on the
    loop waits for the open pull requests — or stops, with ``wait=False``.
    """
    flow = ship_lib.Flow(state)
    try:
        return _run_implement_loop(
            state, flow, model=model, max_iterations=max_iterations, cooldown=cooldown, story_id=story_id,
            phase1=phase1, review=review, compound=compound, save_global=save_global, user_prompt=user_prompt,
            extra_tools=extra_tools, verbose=verbose, dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd, tracker=tracker, wait=wait,
        )
    finally:
        flow.restore()


def _run_implement_loop(
    state: StateManager,
    flow: ship_lib.Flow,
    *,
    model: str,
    max_iterations: int,
    cooldown: int,
    story_id: str | None,
    phase1: bool,
    review: bool,
    compound: bool,
    save_global: bool,
    user_prompt: str,
    extra_tools: str,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
    tracker,
    wait: bool,
) -> PhaseState:
    # Recover any stories orphaned by a previous crash
    recovered = state.recover_orphaned_stories()
    if recovered:
        click.echo(click.style(f"  Recovered {len(recovered)} orphaned stories from previous crash:", fg='yellow', bold=True))
        for s in recovered:
            click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")

    if tracker is not None:
        _sync_tracker(tracker, push_missing=True)
    tracker_fresh = tracker is not None  # the first iteration doesn't need to pull again

    if not state.stories_path.exists():
        click.echo("Error: No stories found. Run 'pralph stories' first.")
        return PhaseState(phase="implement", completed=True, completion_reason="no_stories")

    system_prompt = build_guardrails_system_prompt("implement", state)

    # Build tools list with extras
    tools = IMPLEMENT_TOOLS
    if extra_tools:
        tools = tools + "," + extra_tools

    # If specific story requested, just implement it
    if story_id:
        if tracker is not None:
            # After the sync, so a ticket this checkout hasn't seen yet can be named.
            story_id = tracker.resolve_story_id(story_id)
        return _implement_single(state, story_id, model=model, system_prompt=system_prompt,
                                 tools=tools, user_prompt=user_prompt, review=review,
                                 compound=compound, save_global=save_global, verbose=verbose,
                                 dangerously_skip_permissions=dangerously_skip_permissions,
                                 max_budget_usd=max_budget_usd,
                                 max_rounds=max_iterations, tracker=tracker, flow=flow)

    # State-based mode selection (evaluated each iteration)
    def _pick_implement_mode() -> str:
        if phase1:
            has_analysis = state.phase1_analysis_path.exists()
            has_foundation = any(
                s.category.upper() in FOUNDATION_CATEGORIES
                for s in state.get_pending_stories()
            )
            if not has_analysis and has_foundation:
                return "phase1_analyze"
        return "implement"

    story_queue: list[Story] = []
    blocked: dict[str, list[str]] = {}  # actionable stories → the dependencies they wait for
    announced: tuple = ()

    def _refresh_queue() -> list[Story]:
        nonlocal blocked
        actionable, blocked = flow.split(state.get_actionable_stories())
        # Rework stories are already at the front from get_actionable_stories()
        rework = [s for s in actionable if s.status == StoryStatus.rework]
        pending = [s for s in actionable if s.status == StoryStatus.pending]

        analysis_path = state.phase1_analysis_path
        if analysis_path.exists():
            data = json.loads(analysis_path.read_text())
            impl_order = data.get("implementation_order", data.get("phase_1_group", []))
            pending_ids = {s.id for s in pending}
            ordered_ids = [sid for sid in impl_order if sid in pending_ids]
            ordered = [s for sid in ordered_ids for s in pending if s.id == sid]
            remaining = [s for s in pending if s.id not in set(ordered_ids)]
            return rework + ordered + _sort_stories(remaining)
        return rework + _sort_stories(pending)

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        nonlocal story_queue, tracker_fresh, announced
        if tracker is not None:
            if not tracker_fresh:
                _sync_tracker(tracker)
            tracker_fresh = False
            story_queue = []  # tickets may have been re-prioritized, added or closed
        if _report_flow_changes(flow.refresh()) or flow.enabled:
            story_queue = []  # a merge frees dependents; a new pull request holds them back
        mode = _pick_implement_mode()

        # ── phase1_analyze: identify foundation stories ──
        if mode == "phase1_analyze":
            click.echo(click.style("  Mode: phase1_analyze", fg='magenta', bold=True) + " — identifying foundation stories...")
            analyze_prompt = assemble_phase1_analyze_prompt(state)
            result = run_with_retry(
                analyze_prompt,
                model=model,
                allowed_tools="Read,Glob,Grep",
                system_prompt=system_prompt,
                dangerously_skip_permissions=dangerously_skip_permissions,
                max_budget_usd=max_budget_usd,
                timeout=600,
                verbose=verbose,
                project_dir=str(state.project_dir),
            )

            if not result.success:
                return IterationResult(
                    iteration=i, phase="implement", mode="phase1_analyze",
                    success=False, error=result.error, cost_usd=result.cost_usd,
                    **_token_kwargs(result),
                )

            data = extract_json_from_text(result.result)
            if not isinstance(data, dict) or "phase_1_group" not in data:
                return IterationResult(
                    iteration=i, phase="implement", mode="phase1_analyze",
                    success=False, error="Could not parse phase1 group from analysis",
                    cost_usd=result.cost_usd,
                    **_token_kwargs(result),
                )

            # Save analysis to file for the implement step
            state.phase1_analysis_path.write_text(json.dumps(data, indent=2) + "\n")

            group = data["phase_1_group"]
            reasoning = data.get("reasoning", {})
            click.echo(click.style(f"  Phase 1 group ({len(group)} stories):", fg='blue', bold=True) + f" {', '.join(group)}")
            if isinstance(reasoning, dict):
                for sid, reason in reasoning.items():
                    click.echo(f"    {click.style(sid, fg='blue')}: {reason[:80]}")
            elif reasoning:
                click.echo(f"    {str(reasoning)[:120]}")

            return IterationResult(
                iteration=i, phase="implement", mode="phase1_analyze",
                success=True, raw_output=result.result, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        # ── implement: one story per iteration ──
        if not story_queue:
            story_queue = _refresh_queue()

        while not story_queue:
            # Nothing to work on. Is anything still on its way (a pull request, a person)?
            waiting = flow.waiting()
            if not waiting:
                if blocked:
                    _report_blocked(state, blocked)
                    return IterationResult(
                        iteration=i, phase="implement", mode="implement", success=True, impl_status="blocked",
                    )
                return IterationResult(
                    iteration=i, phase="implement", mode="implement",
                    success=True, impl_status="all_done",
                )
            signature = tuple((s.id, s.status) for s in waiting)
            if signature != announced:
                announced = signature
                _report_waiting(waiting, blocked)
            if not wait:
                return IterationResult(
                    iteration=i, phase="implement", mode="implement", success=True, impl_status="waiting",
                )
            click.echo(click.style(
                f"  Looking again in {flow.settings.poll_interval}s (Ctrl-C to stop) ...", dim=True,
            ))
            flow.sleep()
            if tracker is not None:
                _sync_tracker(tracker)
            _report_flow_changes(flow.refresh())
            story_queue = _refresh_queue()

        story = story_queue.pop(0)
        try:
            workdir = flow.prepare(story)
        except ship_lib.GitError as e:
            # Not the story's fault, and the next one would fail the same way.
            click.echo(click.style(f"  Cannot set up the branch for {story.id}: {e}", fg='red'))
            story_queue.clear()
            return IterationResult(
                iteration=i, phase="implement", mode="implement", success=False,
                error=str(e), impl_status="workspace_error", story_id=story.id,
            )
        state.mark_story_status(story.id, StoryStatus.in_progress)
        click.echo(click.style(f"  Implementing: {story.id}", fg='yellow', bold=True) + f" — {story.title}")
        if flow.enabled:
            click.echo(click.style(f"  on {ship_lib.git_info(story).get('branch')} in {workdir}", dim=True))

        # Log iteration start
        state.log_iteration(IterationResult(
            iteration=i, phase="implement", mode="implement_started",
            success=True, story_id=story.id,
        ))

        prompt = assemble_implement_prompt(state, story, phase_state=ps, user_prompt=user_prompt)

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_story_id = story.id
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=tools,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=1800,
            verbose=verbose,
            project_dir=str(workdir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            if result.error in ("interrupted", "aborted"):
                state.mark_story_status(story.id, StoryStatus.pending, summary=f"User {result.error}")
                story_queue.clear()
                return IterationResult(
                    iteration=i, phase="implement", mode="implement",
                    success=result.error == "interrupted",
                    error=result.error if result.error == "aborted" else "",
                    impl_status=result.error,
                    cost_usd=result.cost_usd, story_id=story.id,
                    **_token_kwargs(result),
                )
            state.mark_story_status(
                story.id, StoryStatus.error, summary=result.error[:200],
                error_reason=result.error, error_output=result.result or "",
            )
            return IterationResult(
                iteration=i, phase="implement", mode="implement",
                success=False, error=result.error, cost_usd=result.cost_usd,
                story_id=story.id,
                **_token_kwargs(result),
            )

        parsed = parse_implement_output(result.result)
        status_str = parsed.get("status", "error")
        summary = parsed.get("summary", "")

        new_status = _reported_status(status_str, default=StoryStatus.error)
        _note_pause(state, story, new_status, parsed)
        state.mark_story_status(
            story.id, new_status, summary=summary, extra=parsed,
            error_reason=parsed.get("reason", "") if new_status == StoryStatus.error else "",
            error_output=result.result if new_status == StoryStatus.error else "",
            review_pending=review and new_status == StoryStatus.implemented,
        )
        status_color = 'green' if new_status == StoryStatus.implemented else 'yellow'
        click.echo(f"  → {story.id}: {click.style(new_status.value, fg=status_color)} — {summary[:120]}")

        total_cost = result.cost_usd
        total_input = result.input_tokens
        total_output = result.output_tokens
        total_cache_read = result.cache_read_input_tokens
        total_cache_create = result.cache_creation_input_tokens

        # Review step: run on fresh Claude instance after successful implementation
        if new_status == StoryStatus.implemented and review:
            review_result = _run_review(
                state, story,
                model=model,
                system_prompt=system_prompt,
                verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions,
                max_budget_usd=max_budget_usd,
                cwd=workdir,
            )
            if review_result is not None:
                total_cost += review_result.cost_usd
                total_input += review_result.input_tokens
                total_output += review_result.output_tokens
                total_cache_read += review_result.cache_read_input_tokens
                total_cache_create += review_result.cache_creation_input_tokens
                if not review_result.approved:
                    new_status = StoryStatus.rework
                    state.mark_story_status(story.id, StoryStatus.rework, summary="Review rejected")
                    story_queue.clear()  # Force refresh to pick up rework story

        if new_status == StoryStatus.needs_human:
            story_queue.clear()  # stories that depend on it wait

        # Ship step: hand the reviewed branch to the project's own way of opening a pull request
        if new_status == StoryStatus.implemented:
            ship_result = _run_ship(
                state, flow, story, workdir,
                model=model, system_prompt=system_prompt, verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
            )
            if ship_result is not None:
                total_cost += ship_result.cost_usd
                total_input += ship_result.input_tokens
                total_output += ship_result.output_tokens
                total_cache_read += ship_result.cache_read_input_tokens
                total_cache_create += ship_result.cache_creation_input_tokens

        # Compound learning: capture solutions after successful implementation
        if new_status == StoryStatus.implemented and compound:
            compound_result = _run_compound_capture(
                state, story,
                model=model,
                system_prompt=system_prompt,
                verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions,
                max_budget_usd=max_budget_usd,
                save_global=save_global,
                cwd=workdir,
            )
            total_cost += compound_result.cost_usd
            total_input += compound_result.input_tokens
            total_output += compound_result.output_tokens
            total_cache_read += compound_result.cache_read_input_tokens
            total_cache_create += compound_result.cache_creation_input_tokens

        # Clean up analysis file when all its stories are done
        if new_status == StoryStatus.implemented and state.phase1_analysis_path.exists():
            data = json.loads(state.phase1_analysis_path.read_text())
            group_ids = set(data.get("implementation_order", data.get("phase_1_group", [])))
            pending_ids = {s.id for s in state.get_pending_stories()}
            if not group_ids & pending_ids:
                state.phase1_analysis_path.unlink(missing_ok=True)
                story_queue.clear()
                click.echo(click.style("  Phase 1 analysis complete — switching to normal ordering", fg='green'))

        return IterationResult(
            iteration=i, phase="implement", mode="implement",
            success=new_status != StoryStatus.error, impl_status=new_status.value,
            error=parsed.get("reason", "") if new_status == StoryStatus.error else "",
            cost_usd=total_cost, story_id=story.id,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_input_tokens=total_cache_read,
            cache_creation_input_tokens=total_cache_create,
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        story = None
        if ps.active_story_id:
            story = next((s for s in state.load_stories() if s.id == ps.active_story_id), None)
        if not story:
            return IterationResult(
                iteration=ps.current_iteration, phase="implement", mode="resume",
                success=False, error=f"Story '{ps.active_story_id}' not found for resume",
            )

        workdir = ship_lib.story_workdir(state, story.id)
        result = run_with_retry(
            "Continue implementing the story.",
            resume_session_id=session_id,
            timeout=1800,
            project_dir=str(workdir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )

        if not result.success:
            state.mark_story_status(
                story.id, StoryStatus.error, summary=f"Resume failed: {result.error[:200]}",
                error_reason=f"Resume failed: {result.error}", error_output=result.result or "",
            )
            return IterationResult(
                iteration=ps.current_iteration, phase="implement", mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                story_id=story.id,
                **_token_kwargs(result),
            )

        parsed = parse_implement_output(result.result)
        status_str = parsed.get("status", "error")
        summary = parsed.get("summary", "")
        new_status = _reported_status(status_str, default=StoryStatus.error)
        _note_pause(state, story, new_status, parsed)
        state.mark_story_status(
            story.id, new_status, summary=summary, extra=parsed,
            error_reason=parsed.get("reason", "") if new_status == StoryStatus.error else "",
            error_output=result.result if new_status == StoryStatus.error else "",
            review_pending=review and new_status == StoryStatus.implemented,
        )
        status_color = 'green' if new_status == StoryStatus.implemented else 'yellow'
        click.echo(f"  \u2192 {story.id}: {click.style(new_status.value, fg=status_color)} \u2014 {summary[:120]}")

        total_cost = result.cost_usd
        total_input = result.input_tokens
        total_output = result.output_tokens
        total_cache_read = result.cache_read_input_tokens
        total_cache_create = result.cache_creation_input_tokens

        if new_status == StoryStatus.implemented and review:
            review_result = _run_review(
                state, story,
                model=model,
                system_prompt=system_prompt,
                verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions,
                max_budget_usd=max_budget_usd,
                cwd=workdir,
            )
            if review_result is not None:
                total_cost += review_result.cost_usd
                total_input += review_result.input_tokens
                total_output += review_result.output_tokens
                total_cache_read += review_result.cache_read_input_tokens
                total_cache_create += review_result.cache_creation_input_tokens
                if not review_result.approved:
                    new_status = StoryStatus.rework
                    state.mark_story_status(story.id, StoryStatus.rework, summary="Review rejected")

        if new_status == StoryStatus.implemented:
            ship_result = _run_ship(
                state, flow, story, workdir,
                model=model, system_prompt=system_prompt, verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
            )
            if ship_result is not None:
                total_cost += ship_result.cost_usd
                total_input += ship_result.input_tokens
                total_output += ship_result.output_tokens
                total_cache_read += ship_result.cache_read_input_tokens
                total_cache_create += ship_result.cache_creation_input_tokens

        return IterationResult(
            iteration=ps.current_iteration, phase="implement", mode="resume",
            success=new_status != StoryStatus.error, impl_status=new_status.value,
            error=parsed.get("reason", "") if new_status == StoryStatus.error else "",
            cost_usd=total_cost, story_id=story.id,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_input_tokens=total_cache_read,
            cache_creation_input_tokens=total_cache_create,
        )

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if result.impl_status == "all_done":
            ps.completion_reason = "all_stories_done"
            return True
        if result.impl_status in ("waiting", "blocked", "workspace_error"):
            # Not done: the next run picks up where the merges (or the person) left it.
            ps.completion_reason = result.impl_status
            return True
        if ps.consecutive_errors >= 5:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    return _run_loop("implement", state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)


def _sync_tracker(tracker, *, push_missing: bool = False) -> None:
    """Refresh the backlog from the tracker (Linear) and report what changed.

    A tracker that can't be reached doesn't stop the run — the loop carries on
    with the backlog as last synced, and says so.
    """
    try:
        if push_missing and tracker.unpushed():
            pushed = tracker.push()
            if pushed.issues_created:
                click.echo(click.style(f"  [linear] created {len(pushed.issues_created)} missing issues", fg='blue'))
        report = tracker.pull()
    except LinearError as e:
        click.echo(click.style(
            f"  [linear] sync failed: {e}\n  [linear] continuing with the backlog as last synced", fg='yellow',
        ))
        return

    for story in report.new:
        click.echo(click.style("  [linear] new: ", fg='blue') + f"{story.id} — {story.title}")
    for story_id, old, new, reason in report.status_changes:
        click.echo(click.style("  [linear] ", fg='blue') + f"{story_id}: {old} → {new}"
                   + (f" ({reason})" if reason else ""))
    if report.updated:
        click.echo(click.style("  [linear] ", fg='blue') + f"{report.updated} stories updated from their tickets")


def _reported_status(status_str: str, *, default: StoryStatus) -> StoryStatus:
    """The status an implementation reported. Shipping and merging are not its to claim."""
    try:
        status = StoryStatus(status_str)
    except ValueError:
        return StoryStatus.implemented if status_str == "completed" else default
    if status in (StoryStatus.shipped, StoryStatus.merged):
        return StoryStatus.implemented
    return status


def _note_pause(state: StateManager, story: Story, status: StoryStatus, parsed: dict) -> None:
    """Remember why a story waits for a person — and forget it once the story is through."""
    if status == StoryStatus.needs_human:
        reason = str(parsed.get("reason") or parsed.get("summary") or "").strip()
        pause = {"reason": reason, "at": datetime.now().isoformat()}

        def mutate(s: Story) -> None:
            s.metadata["needs_human"] = pause
            s.metadata.pop("human_note", None)

        state.update_story(story.id, mutate, notify=False)
        click.echo(click.style(f"  ⏸ {story.id} needs a person: ", fg='yellow', bold=True) + reason[:300])
        click.echo(f"    When that is done: pralph unblock {story.id}")
    elif status == StoryStatus.implemented and (story.metadata.get("needs_human") or story.metadata.get("human_note")):
        def clear(s: Story) -> None:
            s.metadata.pop("needs_human", None)
            s.metadata.pop("human_note", None)

        state.update_story(story.id, clear, notify=False)


def _report_flow_changes(changes: list[tuple[str, str, str, str]]) -> bool:
    for story_id, old, new, note in changes:
        color = 'green' if new == StoryStatus.merged.value else 'yellow'
        click.echo(click.style("  [git] ", fg='blue') + f"{story_id}: {old} → {click.style(new, fg=color)} ({note})")
    return bool(changes)


def _report_waiting(waiting: list[Story], blocked: dict[str, list[str]]) -> None:
    click.echo(click.style(f"  Waiting for {len(waiting)} {'story' if len(waiting) == 1 else 'stories'}:", bold=True))
    for story in waiting:
        click.echo(f"    {click.style(story.id, fg='blue')}: {ship_lib.waiting_reason(story)[:200]}")
    if blocked:
        held = ", ".join(f"{sid} (after {', '.join(deps)})" for sid, deps in sorted(blocked.items()))
        click.echo(f"    held back until then: {held}")


def _report_blocked(state: StateManager, blocked: dict[str, list[str]]) -> None:
    """Stories that cannot start, and nothing is on its way that would free them."""
    by_id = {s.id: s for s in state.load_stories()}
    click.echo(click.style("  Nothing left that can be worked on:", fg='yellow', bold=True))
    for story_id, deps in sorted(blocked.items()):
        why = ", ".join(f"{d} ({by_id[d].status.value})" if d in by_id else d for d in deps)
        click.echo(f"    {click.style(story_id, fg='blue')} waits for {why}")


def _run_ship(
    state: StateManager,
    flow: ship_lib.Flow,
    story: Story,
    workdir: Path,
    *,
    model: str,
    system_prompt: str,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
) -> ClaudeResult | None:
    """Run the project's ship step (.pralph/ship.md) for a story that is through its review.

    The step pushes the branch and opens the pull request; pralph only takes
    note of the URL and marks the story ``shipped``. Without a ship step
    nothing is pushed and the story stays ``implemented``. Returns the run
    (for its cost), or None if there was nothing to run.
    """
    if not flow.ships or not ship_lib.on_branch(story):
        return None
    branch = ship_lib.git_info(story)["branch"]
    click.echo(click.style(f"  🚢 Shipping: {story.id}", fg='magenta', bold=True) + f" — {branch}")

    result = run_with_retry(
        flow.ship_prompt(story, workdir),
        model=model,
        allowed_tools=flow.settings.ship_tools,
        system_prompt=system_prompt,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        timeout=flow.settings.ship_timeout,
        verbose=verbose,
        project_dir=str(workdir),
    )
    error = ""
    pr_url = ""
    if not result.success:
        error = result.error or "the ship step failed"
    else:
        pr_url = ship_lib.find_pr_url(result.result) or flow.pr_for_branch(workdir, branch)
        if not pr_url:
            error = "the ship step did not report a pull request"
    if error:
        flow.record_ship_failure(story, error)
        click.echo(click.style(f"  ⚠ Not shipped: {error[:200]}", fg='yellow'))
        click.echo(f"    {story.id} stays implemented on {branch} — try again with: pralph ship -s {story.id}")
        return result

    flow.record_shipped(story, pr_url)
    click.echo(click.style("  ✓ Shipped", fg='green', bold=True) + f" — {pr_url}")
    return result


def run_ship(
    state: StateManager,
    *,
    story_id: str,
    model: str = "sonnet",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Run the ship step for one story that is implemented on its branch."""
    flow = ship_lib.Flow(state)
    story = next((s for s in state.load_stories() if s.id == story_id), None)
    if not story:
        click.echo(f"Error: Story '{story_id}' not found")
        return PhaseState(phase="ship", completed=True, completion_reason="story_not_found")
    if not flow.ships:
        click.echo("Error: there is no ship step — write the prompt for it to .pralph/ship.md")
        return PhaseState(phase="ship", completed=True, completion_reason="no_ship_step")
    if story.status not in (StoryStatus.implemented, StoryStatus.shipped) or not ship_lib.on_branch(story):
        click.echo(f"Error: {story.id} is {story.status.value} — only a story implemented on its own branch can be shipped")
        return PhaseState(phase="ship", completed=True, completion_reason="not_implemented")

    result = None
    try:
        workdir = flow.prepare(story)
        result = _run_ship(
            state, flow, story, workdir,
            model=model, system_prompt=build_guardrails_system_prompt("implement", state), verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
        )
    except ship_lib.GitError as e:
        click.echo(f"Error: {e}")
        return PhaseState(phase="ship", completed=True, completion_reason="workspace_error")
    finally:
        flow.restore()

    shipped = next(s for s in state.load_stories() if s.id == story_id).status == StoryStatus.shipped
    return PhaseState(
        phase="ship", completed=True, total_cost_usd=result.cost_usd if result else 0.0,
        completion_reason="shipped" if shipped else "ship_failed",
    )


@dataclass
class _ReviewResult:
    approved: bool
    feedback: str
    issues: list
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def _run_review(
    state: StateManager,
    story: Story,
    *,
    model: str,
    system_prompt: str,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
    approve_on_error: bool = True,
    cwd: Path | None = None,
) -> _ReviewResult | None:
    """Run a review on a freshly implemented story. Returns None on review error.

    Inside the implementation loop a reviewer that fails must not hold the
    story back, so the story counts as approved (``approve_on_error``). A
    review that was asked for by itself reports the failure instead.
    """
    click.echo(click.style(f"  🔍 Reviewing: {story.id}", fg='magenta', bold=True))

    review_prompt = assemble_review_prompt(state, story)
    result = run_with_retry(
        review_prompt,
        model=model,
        allowed_tools=REVIEW_TOOLS,
        system_prompt=system_prompt,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        timeout=600,
        verbose=verbose,
        project_dir=str(cwd or state.project_dir),
    )

    if not result.success:
        click.echo(click.style(f"  ⚠ Review failed (error): {result.error[:120]}", fg='yellow'))
        if approve_on_error:
            click.echo(click.style("  → Auto-approving due to review error", fg='yellow'))
            state.notify_review_passed(story.id, error=result.error or "unknown error")
        return None

    parsed = parse_review_output(result.result)
    approved = parsed["approved"]
    feedback = parsed["feedback"]
    issues = parsed.get("issues", [])

    if approved:
        state.clear_review_feedback(story.id)
        state.notify_review_passed(story.id, feedback)
        click.echo(click.style(f"  ✓ Review approved", fg='green', bold=True) + f" — {feedback[:120]}")
    else:
        # Build feedback text for the rework file
        feedback_lines = [f"# Review Feedback for {story.id}\n", f"**Summary:** {feedback}\n"]
        for issue in issues:
            sev = issue.get("severity", "?")
            desc = issue.get("description", "")
            feedback_lines.append(f"- **[{sev}]** {desc}")
        feedback_text = "\n".join(feedback_lines)
        state.write_review_feedback(story.id, feedback_text)
        click.echo(click.style(f"  ✗ Review rejected", fg='red', bold=True) + f" — {feedback[:120]}")
        for issue in issues:
            sev = issue.get("severity", "?")
            desc = issue.get("description", "")
            color = 'red' if sev in ("critical", "major") else 'yellow'
            click.echo(f"    {click.style(f'[{sev}]', fg=color)} {desc[:100]}")

    return _ReviewResult(
        approved=approved,
        feedback=feedback,
        issues=issues,
        cost_usd=result.cost_usd,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
    )


def run_review(
    state: StateManager,
    *,
    story_id: str,
    model: str = "sonnet",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Review one story's implementation without implementing it.

    An approval makes a story that was waiting for rework ``implemented``; a
    rejection puts it into ``rework`` with the feedback, where the next
    ``implement`` picks it up.
    """
    story = next((s for s in state.load_stories() if s.id == story_id), None)
    if not story:
        click.echo(f"Error: Story '{story_id}' not found")
        return PhaseState(phase="review", completed=True, completion_reason="story_not_found")
    if story.status not in (StoryStatus.implemented, StoryStatus.rework):
        click.echo(f"Error: {story.id} is {story.status.value} — there is no implementation to review yet")
        return PhaseState(phase="review", completed=True, completion_reason="not_implemented")

    # A story on its own branch is reviewed there, not on whatever is checked out.
    flow = ship_lib.Flow(state)
    try:
        on_own_branch = flow.enabled and ship_lib.on_branch(story)
        cwd = flow.prepare(story) if on_own_branch else ship_lib.story_workdir(state, story.id)
        result = _run_review(
            state, story,
            model=model,
            system_prompt=build_guardrails_system_prompt("implement", state),
            verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            approve_on_error=False,
            cwd=cwd,
        )
    except ship_lib.GitError as e:
        click.echo(f"Error: {e}")
        return PhaseState(phase="review", completed=True, completion_reason="workspace_error")
    finally:
        flow.restore()
    if result is None:
        return PhaseState(phase="review", completed=True, completion_reason="review_error")

    ps = PhaseState(phase="review", completed=True, total_cost_usd=result.cost_usd)
    if result.approved:
        if story.status == StoryStatus.rework:
            state.mark_story_status(story.id, StoryStatus.implemented, summary=f"Review approved: {result.feedback}"[:200])
        ps.completion_reason = "review_approved"
    else:
        if story.status != StoryStatus.rework:
            state.mark_story_status(story.id, StoryStatus.rework, summary="Review rejected")
        ps.completion_reason = "review_rejected"
    return ps


def _implement_single(
    state: StateManager,
    story_id: str,
    *,
    model: str,
    system_prompt: str,
    tools: str = IMPLEMENT_TOOLS,
    user_prompt: str = "",
    review: bool = True,
    compound: bool = False,
    save_global: bool = False,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
    max_rounds: int = 50,
    tracker=None,
    flow: ship_lib.Flow | None = None,
) -> PhaseState:
    """Implement a single story by ID.

    Like the backlog run, a story the review rejects is implemented again with
    the review feedback, until the review approves it — at most ``max_rounds``
    implementations. Compound learning runs once, after the approval.
    """
    story = next((s for s in state.load_stories() if s.id == story_id), None)
    if not story:
        click.echo(f"Error: Story '{story_id}' not found")
        return PhaseState(phase="implement", completed=True, completion_reason="story_not_found")

    # If story was already in_progress (crash recovery via --story), add metadata
    if story.status == StoryStatus.in_progress:
        story.metadata["previous_attempt"] = {
            "was_in_progress": True,
            "recovered_at": datetime.now().isoformat(),
        }

    flow = flow or ship_lib.Flow(state)
    max_rounds = max(1, max_rounds)
    for round_no in range(1, max_rounds + 1):
        if round_no > 1:
            # Rework round: the ticket may have been edited, taken or closed meanwhile.
            if tracker is not None:
                _sync_tracker(tracker)
            story = next((s for s in state.load_stories() if s.id == story_id), None)
            if story is None or story.status != StoryStatus.rework:
                now = story.status.value if story is not None else "gone"
                click.echo(click.style(f"  {story_id} is no longer waiting for rework ({now}) — stopping", fg='yellow'))
                return PhaseState(phase="implement", completed=True, completion_reason="story_changed")
            click.echo(click.style(f"  Reworking: {story.id}", fg='yellow', bold=True)
                       + f" — round {round_no} of at most {max_rounds}")
        else:
            click.echo(f"  Implementing: {story.id} — {story.title}")

        try:
            workdir = flow.prepare(story)
        except ship_lib.GitError as e:
            click.echo(click.style(f"  Cannot set up the branch for {story.id}: {e}", fg='red'))
            return PhaseState(phase="implement", completed=True, completion_reason="workspace_error")
        if flow.enabled:
            click.echo(click.style(f"  on {ship_lib.git_info(story).get('branch')} in {workdir}", dim=True))

        # `story` keeps the status it was loaded with, so a story in rework
        # gets the review feedback in its prompt.
        state.mark_story_status(story.id, StoryStatus.in_progress)
        prompt = assemble_implement_prompt(state, story, user_prompt=user_prompt)
        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=tools,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=1800,
            verbose=verbose,
            project_dir=str(workdir),
        )

        if not result.success:
            if result.error in ("interrupted", "aborted"):
                state.mark_story_status(story.id, StoryStatus.pending, summary=f"User {result.error}")
                reason = "user_aborted" if result.error == "aborted" else "user_interrupted"
                return PhaseState(phase="implement", completed=True, completion_reason=reason)
            state.mark_story_status(
                story.id, StoryStatus.error, summary=result.error[:200],
                error_reason=result.error, error_output=result.result or "",
            )
            click.echo(f"  Error: {result.error[:200]}")
            return PhaseState(phase="implement", completed=True, completion_reason="error")

        parsed = parse_implement_output(result.result)
        status_str = parsed.get("status", "error")
        summary = parsed.get("summary", "")
        new_status = _reported_status(status_str, default=StoryStatus.implemented)
        _note_pause(state, story, new_status, parsed)
        state.mark_story_status(
            story.id, new_status, summary=summary, extra=parsed,
            review_pending=review and new_status == StoryStatus.implemented,
        )
        click.echo(f"  → {story.id}: {new_status.value} — {summary[:120]}")

        if new_status != StoryStatus.implemented or not review:
            break

        review_result = _run_review(
            state, story,
            model=model,
            system_prompt=system_prompt,
            verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            cwd=workdir,
        )
        if review_result is None or review_result.approved:
            break

        state.mark_story_status(story.id, StoryStatus.rework, summary="Review rejected")
        if round_no == max_rounds:
            click.echo(click.style(
                f"  Review still rejects {story.id} after {max_rounds} "
                f"{'round' if max_rounds == 1 else 'rounds'} — left in rework", fg='yellow',
            ))
            return PhaseState(phase="implement", completed=True, completion_reason="review_rejected")

    if new_status == StoryStatus.implemented:
        _run_ship(
            state, flow, story, workdir,
            model=model, system_prompt=system_prompt, verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions, max_budget_usd=max_budget_usd,
        )

    # Compound learning: capture solutions after successful implementation
    if new_status == StoryStatus.implemented and compound:
        _run_compound_capture(
            state, story,
            model=model,
            system_prompt=system_prompt,
            verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            save_global=save_global,
            cwd=workdir,
        )

    _suggest_compact(state)
    return PhaseState(phase="implement", completed=True, completion_reason="single_story_done")


def _slugify(text: str) -> str:
    """Generate a filename-safe slug from text."""
    import re as _re
    slug = text.lower().strip()
    slug = _re.sub(r"[^\w\s-]", "", slug)
    slug = _re.sub(r"[\s_]+", "-", slug)
    slug = _re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")


@dataclass
class _CompoundResult:
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def _run_compound_capture(
    state: StateManager,
    story: Story,
    *,
    model: str,
    system_prompt: str,
    verbose: bool,
    dangerously_skip_permissions: bool,
    max_budget_usd: float | None,
    save_global: bool = False,
    cwd: Path | None = None,
) -> _CompoundResult:
    """Run compound learning capture after a successful implementation. Returns result with cost and tokens."""
    click.echo(click.style(f"  Capturing learnings: {story.id}", fg='magenta', bold=True))

    prompt = assemble_compound_prompt(state, story)
    result = run_with_retry(
        prompt,
        model=model,
        allowed_tools=COMPOUND_TOOLS,
        system_prompt=system_prompt,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        timeout=300,
        verbose=verbose,
        project_dir=str(cwd or state.project_dir),
    )

    if not result.success:
        click.echo(click.style(f"  Compound capture failed: {result.error[:120]}", fg='yellow'))
        return _CompoundResult(cost_usd=result.cost_usd, **_token_kwargs(result))

    parsed = parse_compound_output(result.result)

    if not parsed["captured"]:
        click.echo(click.style(f"  Nothing notable: {parsed['reason'][:120]}", fg='yellow'))
        return _CompoundResult(cost_usd=result.cost_usd, **_token_kwargs(result))

    solutions = parsed.get("solutions", [])
    for sol in solutions:
        title = sol.get("title", "Untitled")
        category = sol.get("category", "logic-errors")
        tags = sol.get("tags", [])
        error_sig = sol.get("error_signature", "")
        content = sol.get("content", "")

        if not content:
            # Build content from fields if not provided as full doc
            parts = [f"# {title}\n"]
            if sol.get("problem"):
                parts.append(f"## Problem\n\n{sol['problem']}\n")
            if error_sig:
                parts.append(f"## Error Signature\n\n`{error_sig}`\n")
            if sol.get("solution"):
                parts.append(f"## Solution\n\n{sol['solution']}\n")
            if sol.get("prevention"):
                parts.append(f"## Prevention\n\n{sol['prevention']}\n")
            if sol.get("related_files"):
                files = "\n".join(f"- {f}" for f in sol["related_files"])
                parts.append(f"## Related Files\n\n{files}\n")
            content = "\n".join(parts)

        filename_slug = _slugify(title) + ".md"
        index_entry = {
            "filename": f"{category}/{filename_slug}",
            "category": category,
            "title": title,
            "tags": tags,
            "story_id": story.id,
            "created": datetime.now().isoformat(),
            "error_signature": error_sig,
            "related_files": sol.get("related_files", []),
        }

        if save_global:
            # Global learning on → write only to ~/.pralph/solutions/{domain}/,
            # not the project-local store. Fall back to local only if global
            # save produced nothing (no domains could be inferred), so the
            # solution isn't lost.
            global_paths = state.save_solution_global(category, filename_slug, content, index_entry)
            if global_paths:
                saved_domains = [p.parts[-3] for p in global_paths]
                click.echo(click.style(f"  ↑ {title}", fg='cyan') + f" [global: {', '.join(saved_domains)}]")
            else:
                path = state.save_solution(category, filename_slug, content, index_entry)
                click.echo(click.style(f"  ! {title}", fg='yellow')
                           + f" (no global domain inferred, saved local) → {path}")
        else:
            path = state.save_solution(category, filename_slug, content, index_entry)
            click.echo(click.style(f"  + {title}", fg='green') + f" → {path}")

    click.echo(click.style(f"  Captured {len(solutions)} solution(s)", fg='green', bold=True))
    return _CompoundResult(cost_usd=result.cost_usd, **_token_kwargs(result))


def run_compound(
    state: StateManager,
    *,
    story_id: str | None = None,
    description: str = "",
    model: str = "sonnet",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
    save_global: bool = False,
) -> float:
    """Standalone compound capture. Returns cost."""
    from pralph.assembler import build_guardrails_system_prompt

    system_prompt = build_guardrails_system_prompt("implement", state)

    if story_id:
        stories = state.load_stories()
        story = next((s for s in stories if s.id == story_id), None)
        if not story:
            click.echo(f"Error: Story '{story_id}' not found")
            return 0.0
    else:
        # Create a synthetic story for ad-hoc capture
        story = Story(
            id="COMPOUND",
            title=description or "Ad-hoc compound capture",
            content=description,
        )

    cr = _run_compound_capture(
        state, story,
        model=model,
        system_prompt=system_prompt,
        verbose=verbose,
        dangerously_skip_permissions=dangerously_skip_permissions,
        max_budget_usd=max_budget_usd,
        save_global=save_global,
    )
    return cr.cost_usd


def _sort_stories(stories: list[Story]) -> list[Story]:
    """Sort stories: foundation categories first → priority → dependency order."""

    def sort_key(s: Story) -> tuple[int, int, str]:
        is_foundation = 0 if s.category.upper() in FOUNDATION_CATEGORIES else 1
        return (is_foundation, s.priority, s.id)

    sorted_stories = sorted(stories, key=sort_key)

    # Simple topological adjustment: if A depends on B, B comes first
    id_to_idx = {s.id: i for i, s in enumerate(sorted_stories)}
    result: list[Story] = []
    visited: set[str] = set()

    def visit(story: Story) -> None:
        if story.id in visited:
            return
        visited.add(story.id)
        for dep_id in story.dependencies:
            if dep_id in id_to_idx and dep_id not in visited:
                dep_story = sorted_stories[id_to_idx[dep_id]]
                visit(dep_story)
        result.append(story)

    for s in sorted_stories:
        visit(s)

    return result


# ── Justloop ─────────────────────────────────────────────────────────


def run_justloop(
    state: StateManager,
    *,
    user_prompt: str,
    model: str = "sonnet",
    max_iterations: int = 50,
    cooldown: int = 5,
    extra_tools: str = "",
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
) -> PhaseState:
    """Run a simple prompt loop until completion."""
    system_prompt = build_guardrails_system_prompt("implement", state)
    tools = JUSTLOOP_TOOLS
    if extra_tools:
        tools = f"{tools},{extra_tools}"

    def iteration_fn(i: int, ps: PhaseState) -> IterationResult:
        prompt = assemble_justloop_prompt(state, user_prompt=user_prompt, phase_state=ps)

        sid = str(uuid.uuid4())
        ps.active_session_id = sid
        ps.active_session_started = datetime.now().isoformat()
        state.save_phase_state(ps)

        result = run_with_retry(
            prompt,
            model=model,
            allowed_tools=tools,
            system_prompt=system_prompt,
            dangerously_skip_permissions=dangerously_skip_permissions,
            max_budget_usd=max_budget_usd,
            timeout=1800,
            verbose=verbose,
            project_dir=str(state.project_dir),
            session_id=sid,
        )

        _clear_session_tracking(ps)

        if not result.success:
            return IterationResult(
                iteration=i, phase="justloop", mode="execute",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )

        return IterationResult(
            iteration=i, phase="justloop", mode="execute",
            success=True, raw_output=result.result,
            cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def resume_fn(session_id: str, ps: PhaseState) -> IterationResult:
        result = run_with_retry(
            "Continue where you left off.",
            resume_session_id=session_id,
            timeout=1800,
            project_dir=str(state.project_dir),
            dangerously_skip_permissions=dangerously_skip_permissions,
            verbose=verbose,
        )
        if not result.success:
            return IterationResult(
                iteration=ps.current_iteration, phase="justloop", mode="resume",
                success=False, error=result.error, cost_usd=result.cost_usd,
                **_token_kwargs(result),
            )
        return IterationResult(
            iteration=ps.current_iteration, phase="justloop", mode="resume",
            success=True, raw_output=result.result,
            cost_usd=result.cost_usd,
            **_token_kwargs(result),
        )

    def completion_fn(result: IterationResult, ps: PhaseState) -> bool:
        if result.raw_output and detect_loop_complete(result.raw_output):
            ps.completion_reason = "loop_complete"
            return True
        if ps.consecutive_errors >= 5:
            ps.completion_reason = "consecutive_errors"
            return True
        return False

    return _run_loop("justloop", state, max_iterations, cooldown, iteration_fn, completion_fn, resume_fn, verbose, dangerously_skip_permissions)
