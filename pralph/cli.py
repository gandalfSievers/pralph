from __future__ import annotations

import os
import sys

import click

from pralph import __version__
from pralph import linear_oauth
from pralph import linear_sync
from pralph import spec as spec_lib
from pralph.linear import LinearClient, LinearError
from pralph.loop import run_add, run_compound, run_ideate_loop, run_implement_loop, run_justloop, run_plan_loop, run_refine, run_review, run_spec_loop, run_stories_loop, run_webgen_loop
from pralph.viewer import run_viewer
from pralph.models import PhaseState, Story, StoryStatus
from pralph.state import StateManager


def _read_stdin() -> str | None:
    """Read from stdin if it's piped (not a TTY). Returns None if interactive."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip() or None
    return None


def _resolve_prompt(flag_value: str | None, interactive_label: str) -> str:
    """Resolve prompt: flag > stdin > interactive prompt."""
    if flag_value:
        return flag_value
    stdin = _read_stdin()
    if stdin:
        return stdin
    return click.prompt(interactive_label)


def _reset_phase(state: StateManager, phase: str) -> None:
    """Reset a phase's state so it runs from scratch."""
    state.save_phase_state(PhaseState(phase=phase))
    click.echo(f"  Reset phase '{phase}'")


def _state(ctx: click.Context, *, linear: str = "off") -> StateManager:
    """StateManager for the invocation, with integrations hooked in.

    Spec support (tasks.md, requirement baseline) is always attached — it is
    inert outside spec mode. ``linear`` controls the Linear sync for commands
    that change the backlog: ``"auto"`` syncs when the project is linked,
    ``"required"`` fails when it can't, ``"off"`` leaves Linear alone. The
    sync object, if any, ends up in ``ctx.obj["linear_sync"]``.
    """
    state = StateManager(ctx.obj["project_dir"], domains=ctx.obj.get("domains"))
    spec_lib.attach(state)
    ctx.obj["linear_sync"] = None
    if linear != "off":
        try:
            ctx.obj["linear_sync"] = linear_sync.attach(state, required=linear == "required")
        except LinearError as e:
            raise click.ClickException(str(e))
    return state


def _get_extra_tools(ctx: click.Context, state: StateManager) -> str:
    """Merge CLI --extra-tools with project-level .pralph/extra-tools.txt."""
    cli_tools = ctx.obj["extra_tools_cli"]
    project_tools = state.read_extra_tools()
    parts = [p for p in [project_tools, cli_tools] if p]
    return ",".join(parts)


def _resolve_story_ref(state: StateManager, story_id: str) -> str:
    """Accept a Linear identifier (ENG-123) wherever a story ID is expected."""
    for story in state.load_stories():
        link = linear_sync.story_link(story)
        if link and str(link.get("identifier", "")).upper() == story_id.upper():
            return story.id
    return story_id


class OrderedGroup(click.Group):
    """Click group that shows commands in defined order with section headers."""

    SECTIONS = [
        ("Workflow", ["plan", "spec", "stories", "webgen", "implement", "review"]),
        ("Replan", ["add", "ideate", "refine"]),
        ("Integration", ["linear"]),
        ("Tools", ["justloop", "compound", "compact-index", "spec-status", "reset-errors", "viewer"]),
    ]

    def list_commands(self, ctx):
        return list(self.commands)

    def format_commands(self, ctx, formatter):
        for section_name, cmd_names in self.SECTIONS:
            rows = []
            for name in cmd_names:
                cmd = self.commands.get(name)
                if cmd is None:
                    continue
                help_text = cmd.get_short_help_str(limit=formatter.width)
                rows.append((name, help_text))
            if rows:
                with formatter.section(f"{section_name} Commands"):
                    formatter.write_dl(rows)


@click.group(cls=OrderedGroup, invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", default="opus", help="Model alias or full name")
@click.option("--max-iterations", default=50, type=int, help="Max loop iterations")
@click.option("--max-budget-usd", default=None, type=float, help="Max $ per claude invocation")
@click.option("--cooldown", default=5, type=int, help="Seconds between iterations")
@click.option("--verbose", is_flag=True, help="Show full claude output")
@click.option("--project-dir", default=None, type=click.Path(exists=True), help="Target project dir [default: cwd]")
@click.option("--dangerously-skip-permissions", is_flag=True, help="Pass permission bypass to claude")
@click.option("--extra-tools", default=None, help="Additional tools to allow (comma-separated, e.g. mcp__db__query,mcp__api__call)")
@click.option("--domain", multiple=True, help="Override detected domains (repeatable, e.g. --domain rust --domain docker)")
@click.version_option(version=__version__)
@click.pass_context
def main(ctx, model, max_iterations, max_budget_usd, cooldown, verbose, project_dir, dangerously_skip_permissions, extra_tools, domain):
    """pralph — Planned Ralph: multi-phase development workflow powered by Claude Code."""
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["max_iterations"] = max_iterations
    ctx.obj["max_budget_usd"] = max_budget_usd
    ctx.obj["cooldown"] = cooldown
    ctx.obj["verbose"] = verbose
    ctx.obj["project_dir"] = project_dir or os.getcwd()
    ctx.obj["dangerously_skip_permissions"] = dangerously_skip_permissions
    ctx.obj["extra_tools_cli"] = extra_tools or ""
    ctx.obj["domains"] = list(domain) if domain else None

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option("--prompt", default=None, help="Guidance for design doc creation")
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def plan(ctx, prompt, reset):
    """Phase 1: Create/refine a design document."""
    state = _state(ctx)
    if reset:
        _reset_phase(state, "plan")
    prompt = _resolve_prompt(prompt, "Design prompt")
    click.echo(f"pralph plan — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")

    run_plan_loop(
        state,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        user_prompt=prompt,
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.argument("prompt_args", nargs=-1)
@click.option("--name", "-n", default=None, help="Spec name = directory under specs/ [default: the only spec, else the project name]")
@click.option("--prompt", default=None, help="What to specify, or the change to make to an existing spec")
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def spec(ctx, prompt_args, name, prompt, reset):
    """Phase 1 (spec-driven): Create or amend a spec stored in the repo.

    Writes specs/<name>/spec.md (requirements) and design.md. Run it again
    with a change request to amend the spec, then 'pralph stories' to derive
    stories for whatever the spec gained or changed.
    """
    state = _state(ctx)
    if reset:
        _reset_phase(state, "spec")
    try:
        spec_name = spec_lib.resolve_spec_name(state, name)
    except ValueError as e:
        raise click.UsageError(str(e))
    target = spec_lib.get_spec(state, spec_name)

    # Resolve prompt: positional args > --prompt > stdin > interactive.
    # An existing spec can be refined without a prompt.
    if prompt_args:
        user_prompt = " ".join(prompt_args)
    elif prompt:
        user_prompt = prompt
    else:
        user_prompt = _read_stdin() or ""
        if not user_prompt and not target.exists():
            user_prompt = click.prompt("What should be specified")

    click.echo(f"pralph spec — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  spec: {spec_lib.rel_path(state, target.spec_path)}"
               f" ({'amend' if target.exists() else 'new'})")

    run_spec_loop(
        state,
        spec_name=spec_name,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        user_prompt=user_prompt,
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.option("--extract-weight", default=80, type=int, help="Extract vs research weight (0-100)")
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def stories(ctx, extract_weight, reset):
    """Phase 2: Extract user stories from the design doc or spec."""
    state = _state(ctx, linear="auto")
    if reset:
        _reset_phase(state, "stories")
    click.echo(f"pralph stories — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    if state.spec_mode:
        click.echo(f"  source: specs in {spec_lib.rel_path(state, state.spec_dir)}/ (spec-driven)")
    else:
        click.echo(f"  extract_weight: {extract_weight}%")

    run_stories_loop(
        state,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        extract_weight=extract_weight,
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def webgen(ctx, reset):
    """Phase 2b: Discover missing requirements via web research."""
    state = _state(ctx, linear="auto")
    if reset:
        _reset_phase(state, "webgen")
    click.echo(f"pralph webgen — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")

    run_webgen_loop(
        state,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.option("--prompt", default=None, help="Brief idea to turn into a story (prompted if omitted)")
@click.option("--next", "is_next", is_flag=True, help="Priority 1 — implement next")
@click.option("--anytime", is_flag=True, default=False, help="Claude picks priority (default)")
@click.pass_context
def add(ctx, prompt, is_next, anytime):
    """Add a single story from an idea."""
    prompt = _resolve_prompt(prompt, "Idea")
    state = _state(ctx, linear="auto")
    click.echo(f"pralph add")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  priority: {'next (1)' if is_next else 'claude picks'}")
    click.echo(f"  idea: {prompt}")

    story = run_add(
        state,
        idea=prompt,
        is_next=is_next,
        model=ctx.obj["model"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )

    if story:
        click.echo()
        click.echo(click.style("  Story created:", fg='green', bold=True))
        click.echo(f"    ID:         {click.style(story.id, fg='blue')}")
        click.echo(f"    Title:      {story.title}")
        click.echo(f"    Priority:   {story.priority}")
        click.echo(f"    Category:   {story.category}")
        click.echo(f"    Complexity: {story.complexity}")
        click.echo(f"    Deps:       {', '.join(story.dependencies) or 'none'}")
        click.echo(f"    Criteria:   {len(story.acceptance_criteria)} items")
        for ac in story.acceptance_criteria:
            click.echo(f"      - {ac}")
    else:
        click.echo(click.style("\n  Failed to create story", fg='red'))


@main.command()
@click.argument("ideas_args", nargs=-1)
@click.option("--ideas-file", default=None, type=click.Path(), help="Path to ideas file [default: .pralph/ideas.md]")
@click.option("--prompt", default=None, help="Ideas as inline text")
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def ideate(ctx, ideas_args, ideas_file, prompt, reset):
    """Process a batch of ideas into stories.

    Ideas can be passed as arguments: pralph ideate "add dark mode" "CSV export"
    """
    state = _state(ctx, linear="auto")
    if reset:
        _reset_phase(state, "ideate")

    # Resolve ideas: args > --prompt > --ideas-file > default ideas.md > stdin > interactive
    if ideas_args:
        ideas_text = "\n".join(f"- {a}" for a in ideas_args)
    elif prompt:
        ideas_text = prompt
    elif ideas_file:
        from pathlib import Path
        p = Path(ideas_file)
        if not p.exists():
            click.echo(f"Error: Ideas file not found: {ideas_file}")
            return
        ideas_text = p.read_text()
    elif state.ideas_path.exists():
        ideas_text = state.ideas_path.read_text()
    else:
        stdin = _read_stdin()
        if stdin:
            ideas_text = stdin
        else:
            click.echo("Enter ideas (one per line, blank line to finish):")
            lines = []
            while True:
                line = click.prompt("", default="", prompt_suffix="  ", show_default=False)
                if not line:
                    break
                lines.append(line)
            if not lines:
                click.echo("Error: No ideas provided")
                return
            ideas_text = "\n".join(lines)

    ideas_text = ideas_text.strip()
    if not ideas_text:
        click.echo("Error: Ideas text is empty")
        return

    click.echo(f"pralph ideate — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  ideas: {len(ideas_text)} chars")

    run_ideate_loop(
        state,
        ideas_text=ideas_text,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.argument("instruction", required=False)
@click.option("--prompt", default=None, help="Refinement instruction")
@click.option("--story", "-s", "story_ids", multiple=True, help="Story ID(s) to refine")
@click.option("--pattern", "-p", "id_pattern", default=None, help="Glob pattern to match story IDs (e.g. 'I18N-*')")
@click.pass_context
def refine(ctx, instruction, prompt, story_ids, id_pattern):
    """Refine existing stories: split, merge, or rewrite."""
    import fnmatch

    state = _state(ctx, linear="auto")
    all_stories = state.load_stories()
    stories_by_id = {s.id: s for s in all_stories}

    # Resolve target stories from --story IDs + --pattern glob
    selected: list[Story] = []
    seen: set[str] = set()

    for sid in story_ids:
        if sid in stories_by_id:
            if sid not in seen:
                selected.append(stories_by_id[sid])
                seen.add(sid)
        else:
            click.echo(click.style(f"  Warning: story '{sid}' not found, skipping", fg='yellow'))

    if id_pattern:
        for s in all_stories:
            if fnmatch.fnmatch(s.id, id_pattern) and s.id not in seen:
                selected.append(s)
                seen.add(s.id)

    if not selected:
        click.echo(click.style("Error: no stories matched. Use -s STORY_ID or -p 'PATTERN-*'", fg='red'))
        return

    # Warn about non-actionable statuses
    for s in selected:
        if s.status not in (StoryStatus.pending, StoryStatus.rework, StoryStatus.error):
            click.echo(click.style(f"  Warning: {s.id} has status '{s.status.value}'", fg='yellow'))

    # Resolve instruction: positional arg > --prompt > stdin > interactive
    if not instruction:
        if prompt:
            instruction = prompt
        else:
            instruction = _read_stdin() or click.prompt("Refinement instruction")
    if not instruction.strip():
        click.echo(click.style("Error: instruction is empty", fg='red'))
        return

    click.echo(f"pralph refine")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  stories: {', '.join(s.id for s in selected)}")
    click.echo(f"  instruction: {instruction}")

    new_stories = run_refine(
        state,
        instruction=instruction,
        original_stories=selected,
        model=ctx.obj["model"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )

    if new_stories:
        click.echo()
        click.echo(click.style(f"  {len(new_stories)} replacement stories created:", fg='green', bold=True))
        for s in new_stories:
            click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
        click.echo()
        click.echo(click.style(f"  {len(selected)} original stories marked as skipped:", fg='yellow'))
        for s in selected:
            click.echo(f"    {click.style(s.id, dim=True)}: {s.title}")
    else:
        click.echo(click.style("\n  Failed to refine stories", fg='red'))


@main.command()
@click.option("--story-id", default=None, help="Implement a specific story")
@click.option("--phase1/--no-phase1", default=True, help="Architecture-first grouping")
@click.option("--review/--no-review", default=True, help="Run reviewer after each implementation")
@click.option("--compound/--no-compound", default=False, help="Capture learnings after each story (compound learning)")
@click.option("--prompt", default=None, help="Guidance for implementation (e.g. 'use FastAPI', 'use MCP for DB access')")
@click.option("--linear/--no-linear", "use_linear", default=None, help="Work off the linked Linear project [default: on when linked]")
@click.option("--reset", is_flag=True, help="Reset phase state and start fresh")
@click.pass_context
def implement(ctx, story_id, phase1, review, compound, prompt, use_linear, reset):
    """Phase 3: Implement stories from backlog."""
    linear_mode = {None: "auto", True: "required", False: "off"}[use_linear]
    if linear_mode == "auto" and linear_sync.LinearLink.load(
        StateManager(ctx.obj["project_dir"]).linear_link_path
    ):
        # Linked projects are executed off their tickets — running without
        # them has to be an explicit choice (--no-linear), not a missing key.
        linear_mode = "required"
    state = _state(ctx, linear=linear_mode)
    tracker = ctx.obj["linear_sync"]
    if reset:
        _reset_phase(state, "implement")
    prompt = prompt or _read_stdin() or ""
    save_global = state.global_compound
    click.echo(f"pralph implement — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  review: {'on' if review else 'off'}")
    click.echo(f"  compound: {'on' if compound else 'off'}")
    if compound and save_global:
        domains = state.detect_domains()
        click.echo(f"  global: on (domains: {', '.join(domains) or 'none detected'})")
    if tracker is not None:
        click.echo(f"  linear: {tracker.link.project_name} ({tracker.link.team_key}) {tracker.link.project_url}")
        if tracker.link.scope:
            click.echo(f"  linear scope: {tracker.scope_text()}")
        if tracker.link.read_mostly:
            click.echo(f"  linear mode: {_READ_MOSTLY}")
    if state.spec_mode:
        click.echo(f"  spec-driven: {', '.join(sp.name for sp in spec_lib.list_specs(state)) or 'no specs yet'}")
    if story_id:
        story_id = _resolve_story_ref(state, story_id)
        click.echo(f"  story: {story_id}")

    run_implement_loop(
        state,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        story_id=story_id,
        phase1=phase1,
        review=review,
        compound=compound,
        save_global=save_global,
        user_prompt=prompt,
        extra_tools=_get_extra_tools(ctx, state),
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
        tracker=tracker,
    )


@main.command()
@click.option("--story-id", "-s", required=True, help="Story to review (story ID or Linear identifier)")
@click.pass_context
def review(ctx, story_id):
    """Review an implemented story without implementing it again.

    Runs the same reviewer as 'implement'. An approval makes a story that was
    waiting for rework implemented; a rejection puts the story into rework
    with the feedback, where the next 'implement' picks it up. Exits with 1
    when the review rejects the story or cannot run.
    """
    state = _state(ctx, linear="auto")
    story_id = _resolve_story_ref(state, story_id)
    click.echo("pralph review")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  story: {story_id}")

    ps = run_review(
        state,
        story_id=story_id,
        model=ctx.obj["model"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )
    if ps.completion_reason == "review_rejected":
        click.echo(click.style("  Next:", dim=True) + click.style(f" pralph implement --story-id {story_id}", fg='cyan')
                   + click.style(" reworks it with this feedback", dim=True))
    if ps.completion_reason != "review_approved":
        sys.exit(1)


@main.command()
@click.argument("prompt_args", nargs=-1)
@click.option("--prompt", default=None, help="Task prompt (prompted if omitted)")
@click.pass_context
def justloop(ctx, prompt_args, prompt):
    """Run a prompt in a loop until complete."""
    state = _state(ctx)

    # Resolve prompt: positional args > --prompt > stdin > interactive
    if prompt_args:
        user_prompt = " ".join(prompt_args)
    elif prompt:
        user_prompt = prompt
    else:
        user_prompt = _resolve_prompt(None, "Task prompt")

    # Always start fresh — justloop is a standalone tool, not a resumable phase
    _reset_phase(state, "justloop")

    click.echo(f"pralph justloop — max {ctx.obj['max_iterations']} iterations")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    click.echo(f"  prompt: {user_prompt}")

    run_justloop(
        state,
        user_prompt=user_prompt,
        model=ctx.obj["model"],
        max_iterations=ctx.obj["max_iterations"],
        cooldown=ctx.obj["cooldown"],
        extra_tools=_get_extra_tools(ctx, state),
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
    )


@main.command()
@click.option("--story-id", default=None, help="Story ID to capture learnings from")
@click.option("--prompt", default=None, help="Description of what was done")
@click.pass_context
def compound(ctx, story_id, prompt):
    """Capture learnings from recent work (compound learning)."""
    prompt = _resolve_prompt(prompt, "Description of work done")
    state = _state(ctx)
    save_global = state.global_compound
    click.echo(f"pralph compound")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {ctx.obj['model']}")
    if save_global:
        domains = state.detect_domains()
        click.echo(f"  global: on (domains: {', '.join(domains) or 'none detected'})")
    if story_id:
        click.echo(f"  story: {story_id}")
    if prompt:
        click.echo(f"  description: {prompt[:80]}")

    cost = run_compound(
        state,
        story_id=story_id,
        description=prompt,
        model=ctx.obj["model"],
        verbose=ctx.obj["verbose"],
        dangerously_skip_permissions=ctx.obj["dangerously_skip_permissions"],
        max_budget_usd=ctx.obj["max_budget_usd"],
        save_global=save_global,
    )

    click.echo(f"\n  Cost: ${cost:.4f}")


@main.command("compact-index")
@click.option("--global-only", is_flag=True, help="Only compact global indexes")
@click.option("--local-only", is_flag=True, help="Only compact project-local index")
@click.option("--yes", "-y", is_flag=True, help="Skip the cost-estimate confirmation prompt")
@click.option("--full", is_flag=True, help="Re-check every cluster, not only those that gained entries since the last run")
@click.pass_context
def compact_index(ctx, global_only, local_only, yes, full):
    """Compact solution indexes: merge duplicates via Haiku and prune orphans."""
    state = _state(ctx)
    model = "haiku"
    verbose = ctx.obj.get("verbose", False)
    dangerously_skip_permissions = ctx.obj.get("dangerously_skip_permissions", False)
    click.echo(f"pralph compact-index")
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  model: {model}")

    # ── Pre-flight estimate ──────────────────────────────────────────────
    estimates: list[tuple[str, dict]] = []
    if not global_only and state.solutions_index_path.exists():
        estimates.append(("local", state.estimate_local_compaction(model=model, full=full)))
    if not local_only:
        for est in state.estimate_global_compactions(model=model, full=full):
            estimates.append((f"global/{est['domain']}", est))

    if not estimates:
        click.echo("  no indexes to compact")
        return

    click.echo("\n  Pre-flight estimate:")
    label_width = max(len(label) for label, _ in estimates)
    total_usd = 0.0
    total_calls = 0
    total_entries = 0
    for label, est in estimates:
        total_usd += est["usd"]
        total_calls += est["calls"]
        total_entries += est["entries"]
        click.echo(
            f"    {label.ljust(label_width)}  {est['entries']:>5} entries  "
            f"{est['calls']:>3} calls  ~${est['usd']:.4f}"
        )
    click.echo(f"    {'─' * (label_width + 36)}")
    click.echo(
        f"    {'total'.ljust(label_width)}  {total_entries:>5} entries  "
        f"{total_calls:>3} calls  ~${total_usd:.4f}"
    )

    if not yes:
        if not click.confirm("\n  Proceed?", default=False):
            click.echo("  aborted")
            return

    compact_kwargs = dict(
        model=model,
        verbose=verbose,
        dangerously_skip_permissions=dangerously_skip_permissions,
        full=full,
    )

    total_merged = 0
    total_removed = 0
    total_cost = 0.0

    # Local index
    if not global_only:
        if state.solutions_index_path.exists():
            stats = state.compact_local_index(**compact_kwargs)
            total_merged += stats["merged"]
            total_removed += stats["removed"]
            total_cost += stats.get("cost", 0.0)
            changed = stats["merged"] + stats["removed"]
            if changed:
                click.echo(click.style(f"  local: {stats['original']} → {stats['kept']}", fg='green')
                           + f" ({stats['merged']} merged, {stats['removed']} removed)")
            else:
                click.echo(f"  local: {stats['kept']} entries (clean)")
        else:
            click.echo("  local: no index")

    # Global indexes — independent of current project's detected domains
    if not local_only:
        results = state.compact_global_indexes(**compact_kwargs)
        if results:
            for stats in results:
                domain = stats["domain"]
                total_merged += stats["merged"]
                total_removed += stats["removed"]
                total_cost += stats.get("cost", 0.0)
                changed = stats["merged"] + stats["removed"]
                if changed:
                    click.echo(click.style(f"  global/{domain}: {stats['original']} → {stats['kept']}", fg='green')
                               + f" ({stats['merged']} merged, {stats['removed']} removed)")
                else:
                    click.echo(f"  global/{domain}: {stats['kept']} entries (clean)")
        else:
            click.echo("  global: no indexes found")

    total = total_merged + total_removed
    if total:
        click.echo(click.style(f"\n  Compacted: {total_merged} merged, {total_removed} removed", fg='green', bold=True))
    else:
        click.echo(click.style("\n  All indexes clean", fg='green'))
    if total_cost > 0:
        click.echo(f"  Cost: ${total_cost:.4f}")


_REQ_MARKS = {
    "done": ("\u2713", "green"),
    "in_progress": ("\u25d0", "yellow"),
    "pending": ("\u25cb", None),
    "error": ("\u2717", "red"),
    "dropped": ("\u2717", "red"),
    "uncovered": ("\u2717", "red"),
}


@main.command("spec-status")
@click.pass_context
def spec_status(ctx):
    """Show how the backlog covers the spec's requirements."""
    state = _state(ctx)
    specs = spec_lib.list_specs(state)
    if not specs:
        click.echo(f"No specs found in {state.spec_dir}. Run 'pralph spec' first.")
        return

    coverage = spec_lib.build_coverage(state)
    work = spec_lib.pending_work(state)
    for sp in specs:
        rows = coverage.get(sp.name, [])
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.state] = counts.get(row.state, 0) + 1
        summary = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items())) or "no requirements found"
        click.echo(f"\n  {click.style(spec_lib.rel_path(state, sp.spec_path), fg='blue', bold=True)}"
                   f" — {len(rows)} requirements: {summary}")
        width = max((len(row.requirement.id) for row in rows), default=0)
        for row in rows:
            mark, color = _REQ_MARKS.get(row.state, ("?", None))
            covering = ", ".join(f"{s.id} [{s.status.value}]" for s in row.stories) or "no stories"
            line = f"    {click.style(mark, fg=color)} {row.requirement.id.ljust(width)}  {row.requirement.title}"
            if row.changed:
                line += click.style("  (changed since stories were written)", fg='yellow')
            click.echo(line)
            click.echo(click.style(f"      {covering}", dim=True))

    if work.orphaned:
        click.echo(click.style("\n  Stories whose requirement is no longer in the spec:", fg='yellow'))
        for story, gone in work.orphaned:
            click.echo(f"    {click.style(story.id, fg='blue')}: {story.title} ({', '.join(gone)})")
    if work.untraced:
        click.echo(click.style(f"\n  {len(work.untraced)} stories do not trace to a requirement:", fg='yellow'))
        for story in work.untraced:
            click.echo(f"    {click.style(story.id, fg='blue')}: {story.title} [{story.status.value}]")
    if work.has_work:
        click.echo(f"\n  {len(work.uncovered)} requirements without stories, {len(work.changed)} changed"
                   " — run " + click.style("pralph stories", fg='cyan'))
    else:
        click.echo(click.style("\n  Every requirement is covered by the backlog", fg='green'))


# ── Linear ───────────────────────────────────────────────────────────


def _linear_sync(ctx: click.Context, *, need_key: bool = True) -> tuple[StateManager, linear_sync.LinearSync]:
    """State + sync object for the `pralph linear` commands (linked or not)."""
    state = _state(ctx)
    api_key, oauth = linear_oauth.credentials(state.get_config, state.home_dir)
    if need_key and not api_key and oauth is None:
        raise click.ClickException(
            "LINEAR_API_KEY is not set. Create a personal API key in Linear "
            "(Settings → Security & access) and export it, or log in with 'pralph linear oauth'."
        )
    link = linear_sync.LinearLink.load(state.linear_link_path)
    client = LinearClient("", oauth=oauth) if oauth is not None else LinearClient(api_key)
    return state, linear_sync.LinearSync(state, client, link)


def _echo_pull(report: linear_sync.PullReport) -> None:
    for story in report.new:
        click.echo(f"  {click.style('+', fg='green')} {click.style(story.id, fg='blue')}: {story.title}")
    for story_id, old, new, reason in report.status_changes:
        click.echo(f"  {click.style(story_id, fg='blue')}: {old} → {new}" + (f" ({reason})" if reason else ""))
    click.echo(f"  {report.total} issues: {len(report.new)} new, {report.updated} updated, "
               f"{len(report.status_changes)} status changes, {report.pushed} pushed back"
               + (f", {report.relations_created} blocking relations created" if report.relations_created else ""))
    if report.out_of_scope:
        click.echo(click.style(f"  {report.out_of_scope} issues outside the scope left alone", dim=True))


_READ_MOSTLY = "read-mostly (comments and In Progress only)"


def _scope_options(fn):
    """--milestone / --assignee / --label, shared by 'linear link' and 'linear scope'."""
    for decorator in reversed([
        click.option("--milestone", default=None, help="Only issues in this project milestone"),
        click.option("--assignee", default=None, help="Only issues assigned to this user: 'me', a name or an email"),
        click.option("--label", default=None, help="Only issues with this label"),
    ]):
        fn = decorator(fn)
    return fn


@main.group(cls=click.Group)
def linear():
    """Run the plan as a Linear project (push, pull, link, scope, status, oauth)."""


@linear.command("push")
@click.option("--team", default=None, help="Linear team key, e.g. ENG [default: config linear_team, or the only team]")
@click.option("--project", "project_name", default=None, help="Name for the Linear project [default: title of the plan]")
@click.option("--dry-run", is_flag=True, help="Show what would be created without calling Linear")
@click.pass_context
def linear_push(ctx, team, project_name, dry_run):
    """Turn the plan into a Linear project with sub-issues.

    Creates the project (design doc / spec as its description), a parent issue
    per story category, a sub-issue per story and blocking relations for
    dependencies. Safe to re-run: only what is missing gets created.
    """
    state, sync = _linear_sync(ctx, need_key=not dry_run)
    if not state.load_stories():
        raise click.ClickException("No stories to push. Run 'pralph stories' first.")
    click.echo("pralph linear push" + (" (dry run)" if dry_run else ""))
    try:
        report = sync.push(team_key=team, project_name=project_name, dry_run=dry_run)
    except LinearError as e:
        raise click.ClickException(str(e))

    verb = "would create" if dry_run else "created"
    project_note = f" ({verb})" if report.project_created else " (existing)"
    click.echo(f"  project: {report.project_name}{project_note} {report.project_url}")
    if report.epics_created:
        click.echo(f"  parent issues {verb}: {', '.join(report.epics_created)}")
    for story_id, identifier in report.issues_created:
        click.echo(f"    {click.style(story_id, fg='blue')}" + (f" → {identifier}" if identifier else ""))
    click.echo(f"  {verb}: {len(report.issues_created)} sub-issues, {report.relations_created} blocking relations"
               f" ({report.already_linked} stories already in Linear)")
    if not dry_run:
        click.echo(click.style("  Next:", dim=True) + click.style(" pralph implement", fg='cyan')
                   + click.style(" now works off the Linear project", dim=True))


@linear.command("pull")
@click.pass_context
def linear_pull(ctx):
    """Refresh the backlog from the Linear project (implement does this itself)."""
    state, sync = _linear_sync(ctx)
    click.echo("pralph linear pull")
    try:
        report = sync.pull()
    except LinearError as e:
        raise click.ClickException(str(e))
    _echo_pull(report)


@linear.command("link")
@click.argument("project")
@click.option("--team", default=None, help="Linear team key, if the project belongs to several teams")
@_scope_options
@click.option("--read-mostly", is_flag=True,
              help="Only comment and move issues to In Progress: never close or cancel them, never create issues")
@click.pass_context
def linear_link(ctx, project, team, milestone, assignee, label, read_mostly):
    """Attach this checkout to an existing Linear project (URL, ID or name).

    Use this on a fresh clone: the backlog is rebuilt from the tickets.

    If the project is shared with other repositories or other people, scope
    the link with --milestone, --assignee and/or --label: only matching issues
    become work, and nothing outside the scope is written to.

    With --read-mostly pralph only comments on issues and moves them to In
    Progress. Closing them is left to people, or to the merge.
    """
    state, sync = _linear_sync(ctx)
    if sync.linked:
        raise click.ClickException(
            f"Already linked to {sync.link.project_name} — remove {state.linear_link_path} to link elsewhere"
        )
    click.echo("pralph linear link")
    try:
        sync.link_project(project, team_key=team)
        click.echo(f"  project: {sync.link.project_name} ({sync.link.team_key}) {sync.link.project_url}")
        if milestone or assignee or label:
            try:
                sync.set_scope(milestone=milestone, assignee=assignee, label=label)
            except LinearError:
                # Don't leave an unscoped link behind: the next implement would take every issue.
                state.linear_link_path.unlink(missing_ok=True)
                raise
            click.echo(f"  scope: {sync.scope_text()}")
        if read_mostly:
            sync.set_read_mostly(True)
            click.echo(f"  mode: {_READ_MOSTLY}")
        report = sync.pull()
    except LinearError as e:
        raise click.ClickException(str(e))
    _echo_pull(report)


@linear.command("scope")
@_scope_options
@click.option("--clear", is_flag=True, help="Drop the current scope first (alone: back to the whole project)")
@click.option("--read-mostly/--read-write", "read_mostly", default=None,
              help="Read-mostly: only comment and move issues to In Progress. Read-write: the full sync")
@click.pass_context
def linear_scope(ctx, milestone, assignee, label, clear, read_mostly):
    """Show or change which issues of the linked project are this checkout's work.

    An issue is in scope when it matches every criterion. Issues outside the
    scope are never turned into stories and never written to. Criteria that
    are not given stay as they are; --clear drops them.

    --read-mostly limits what pralph writes to the issues that are in scope:
    comments, and the move to In Progress. It then never closes or cancels an
    issue and never creates issues, parent issues or relations.

    \b
      pralph linear scope --label pralph
      pralph linear scope --milestone "Base work" --assignee me
      pralph linear scope --clear
      pralph linear scope --read-mostly
    """
    changing_scope = bool(milestone or assignee or label or clear)
    changing = changing_scope or read_mostly is not None
    state, sync = _linear_sync(ctx, need_key=changing)
    if not sync.linked:
        raise click.ClickException("Not linked to a Linear project. Run 'pralph linear push' or 'pralph linear link'.")

    def show() -> None:
        click.echo(f"  scope: {sync.scope_text() or 'the whole project'}")
        click.echo(f"  mode: {_READ_MOSTLY if sync.link.read_mostly else 'read-write'}")

    if not changing:
        show()
        return
    click.echo("pralph linear scope")
    try:
        if changing_scope:
            sync.set_scope(milestone=milestone, assignee=assignee, label=label, clear=clear)
        if read_mostly is not None:
            sync.set_read_mostly(read_mostly)
        show()
        report = sync.pull()
    except LinearError as e:
        raise click.ClickException(str(e))
    _echo_pull(report)


@linear.command("oauth")
@click.option("--client-id", default=None,
              help="Client ID of your Linear OAuth application [default: LINEAR_OAUTH_CLIENT_ID, config, or the last login]")
@click.option("--port", default=linear_oauth.DEFAULT_PORT, show_default=True, type=int,
              help="Local port for the callback (http://localhost:PORT/oauth/callback)")
@click.option("--actor", type=click.Choice(["user", "app"]), default="user", show_default=True,
              help="Act as yourself, or as the application (comments and changes show the app's name)")
@click.option("--scope", "scopes", default=linear_oauth.DEFAULT_SCOPES, show_default=True,
              help="Comma-separated OAuth scopes to ask for")
@click.option("--no-browser", is_flag=True, help="Print the URL instead of opening a browser")
@click.option("--logout", is_flag=True, help="Revoke the stored login and delete it")
@click.pass_context
def linear_oauth_login(ctx, client_id, port, actor, scopes, no_browser, logout):
    """Log in to Linear with OAuth instead of a personal API key.

    Needs an OAuth application in your Linear workspace (Settings → API) whose
    callback URL is http://localhost:8765/oauth/callback.
    Only its client ID is used — the login is done with PKCE, no secret.

    The login is stored in ~/.pralph/linear-oauth.json and refreshed
    automatically. A LINEAR_API_KEY, if set, takes precedence over it.
    """
    state = _state(ctx)
    path = linear_oauth.token_path(state.home_dir)
    get = state.get_config
    click.echo("pralph linear oauth")

    if logout:
        session = linear_oauth.OAuthSession.load(path, client_secret=linear_oauth.client_secret(get))
        if session is None:
            click.echo("  Not logged in.")
            return
        revoked = session.revoke()
        click.echo("  Logged out" + ("" if revoked else " locally — Linear could not be reached to revoke the token"))
        return

    client = client_id or linear_oauth.client_id(get, state.home_dir)
    callback = linear_oauth.redirect_uri(port)
    if not client:
        raise click.ClickException(
            "No OAuth client ID.\n"
            f"  1. Create an OAuth application in Linear: {linear_oauth.NEW_APPLICATION_URL}\n"
            f"     with the callback URL {callback}\n"
            "  2. Run: pralph linear oauth --client-id <its client ID>\n"
            "     (or set LINEAR_OAUTH_CLIENT_ID, or linear_oauth_client_id in ~/.pralph/config.json)"
        )
    click.echo(f"  callback URL: {callback} " + click.style("(must be registered in the OAuth application)", dim=True))
    try:
        session = linear_oauth.login(
            client, path, port=port, scopes=scopes, actor=actor,
            secret=linear_oauth.client_secret(get),
            open_browser=None if no_browser else linear_oauth.webbrowser.open,
            echo=click.echo,
        )
        viewer = LinearClient("", oauth=session).viewer()
    except LinearError as e:
        raise click.ClickException(str(e))
    except KeyboardInterrupt:
        raise click.ClickException("Login canceled")

    workspace = (viewer.get("organization") or {}).get("name") or "?"
    click.echo(click.style("  Logged in", fg='green') + f" as {viewer.get('name') or '?'} ({workspace})"
               + (" — acting as the application" if actor == "app" else ""))
    click.echo(f"  scopes: {session.scope or scopes}")
    click.echo(f"  stored in {path} — refreshed automatically")
    if os.environ.get("LINEAR_API_KEY", "").strip() or get("linear_api_key", None):
        click.echo(click.style(
            "  A Linear API key is configured and takes precedence — unset LINEAR_API_KEY / linear_api_key to use this login",
            fg='yellow',
        ))


def _linear_auth_text(state: StateManager) -> str:
    api_key, oauth = linear_oauth.credentials(state.get_config, state.home_dir)
    if api_key:
        return "API key"
    if oauth is None:
        return "none — set LINEAR_API_KEY or run 'pralph linear oauth'"
    from datetime import datetime

    until = f", access token valid until {datetime.fromtimestamp(oauth.expires_at):%Y-%m-%d %H:%M}" if oauth.expires_at else ""
    return f"OAuth login ({oauth.actor} actor{until}, refreshed automatically)"


@linear.command("status")
@click.pass_context
def linear_status(ctx):
    """Show the Linear link and how the backlog maps onto it (as of the last sync)."""
    state, sync = _linear_sync(ctx, need_key=False)
    click.echo(f"  auth: {_linear_auth_text(state)}")
    if not sync.linked:
        click.echo("Not linked to a Linear project. Run 'pralph linear push' or 'pralph linear link'.")
        return
    click.echo(f"  project: {sync.link.project_name} {sync.link.project_url}")
    click.echo(f"  team: {sync.link.team_key}")
    if sync.link.scope:
        click.echo(f"  scope: {sync.scope_text()}")
    if sync.link.read_mostly:
        click.echo(f"  mode: {_READ_MOSTLY}")
    click.echo(f"  parent issues: {', '.join(sorted(sync.link.epics)) or 'none'}")
    by_state: dict[str, int] = {}
    for story in state.load_stories():
        link = linear_sync.story_link(story)
        if link and not linear_sync.is_outside(story):
            name = link.get("state") or link.get("state_type") or "?"
            by_state[name] = by_state.get(name, 0) + 1
    click.echo(f"  issues: {sum(by_state.values())}"
               + (" (" + ", ".join(f"{n} {k}" for k, n in sorted(by_state.items())) + ")" if by_state else ""))
    unpushed = sync.unpushed()
    if unpushed:
        click.echo(click.style(f"  {len(unpushed)} stories not in Linear yet", fg='yellow')
                   + " — run 'pralph linear push'")


@main.command("reset-errors")
@click.pass_context
def reset_errors(ctx):
    """Reset error stories to pending and clear error state."""
    state = _state(ctx, linear="auto")

    # Show error details before resetting
    error_stories = [s for s in state.load_stories() if s.status == StoryStatus.error]
    if error_stories:
        click.echo(click.style(f"  {len(error_stories)} stories in error state:", fg='red'))
        for s in error_stories:
            reason = s.metadata.get("error_reason", "(no reason captured)")
            error_at = s.metadata.get("error_at", "")
            click.echo(f"    {click.style(s.id, fg='blue')}: {s.title}")
            click.echo(f"      Reason: {reason[:200]}")
            if error_at:
                click.echo(f"      Error at: {error_at}")

    # Reset error stories back to pending
    reset_stories = state.reset_error_stories()

    # Clear error fields in whichever phase is currently stored
    if state.phase_state_path.exists():
        try:
            import json
            data = json.loads(state.phase_state_path.read_text())
            phase = data.get("phase", "implement")
        except (json.JSONDecodeError, KeyError):
            phase = "implement"
        ps = state.load_phase_state(phase)
        if ps.consecutive_errors > 0 or ps.last_error or ps.completion_reason in ("consecutive_errors", "error"):
            ps.consecutive_errors = 0
            ps.last_error = ""
            if ps.completion_reason in ("consecutive_errors", "error"):
                ps.completed = False
                ps.completion_reason = ""
            state.save_phase_state(ps)
            click.echo(f"  Cleared '{phase}' phase error state")

    if reset_stories:
        click.echo(click.style(f"  Reset {len(reset_stories)} stories to pending", fg='green'))
    else:
        click.echo("  No error stories found")


@main.command()
@click.option("--port", default=8411, type=int, help="Port to serve on")
@click.option("--no-open", is_flag=True, help="Don't auto-open browser")
@click.pass_context
def viewer(ctx, port, no_open):
    """Browse and review user stories in a web UI."""
    state = _state(ctx)
    stories = state.load_stories()
    if not stories:
        click.echo("No stories found. Run 'pralph stories' first.")
        return
    click.echo(f"  project: {ctx.obj['project_dir']}")
    click.echo(f"  stories: {len(stories)}")
    run_viewer(state, port=port, open_browser=not no_open)
