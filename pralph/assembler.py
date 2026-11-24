from __future__ import annotations

import json
from datetime import datetime

from pralph import spec as spec_lib
from pralph.models import PhaseState, Story
from pralph.prompts.compound import COMPOUND_CAPTURE_PROMPT, COMPOUND_RECALL_SECTION
from pralph.prompts.guardrails import CODE_GUARDRAILS, DEFAULT_GUARDRAILS, STORY_GUARDRAILS
from pralph.prompts.implement import (
    IMPLEMENTATION_IMPLEMENT_PROMPT,
    IMPLEMENTATION_PHASE1_ANALYZE_PROMPT,
    IMPLEMENTATION_PHASE1_IMPLEMENT_PROMPT,
)
from pralph.prompts.review import REVIEW_PROMPT
from pralph.prompts.plan import INITIAL_PLAN_PROMPT, ITERATION_PROMPT_TEMPLATE
from pralph.prompts.ideate import ADD_STORY_PROMPT, IDEATE_PROMPT, REFINE_PROMPT
from pralph.prompts.justloop import JUSTLOOP_PROMPT
from pralph.prompts.spec import (
    SPEC_EXTRACT_PROMPT,
    SPEC_IMPLEMENT_SECTION,
    SPEC_INITIAL_PROMPT,
    SPEC_ITERATION_PROMPT,
    SPEC_REVIEW_SECTION,
)
from pralph.prompts.stories import (
    PLANNING_EXTRACT_PROMPT,
    PLANNING_RESEARCH_PROMPT,
    WEBGEN_REQUIREMENTS_PROMPT,
)
from pralph.state import StateManager

# Size threshold for inlining design doc vs referencing by path
LARGE_DOC_THRESHOLD = 50_000


def _escape_template_vars(text: str) -> str:
    """Escape {{ in user content to prevent variable injection."""
    return text.replace("{{", "{\u200b{")


def _safe_sub(template: str, key: str, value: str) -> str:
    """Substitute {{key}} in template with value."""
    return template.replace("{{" + key + "}}", value)


def _build_crash_recovery_context(story: Story) -> str:
    """Build context block when retrying a story that was in_progress during a crash."""
    prev = story.metadata.get("previous_attempt")
    if not prev:
        return ""
    return (
        "\n## Crash Recovery Context\n\n"
        "**This story was previously in progress when the process crashed or was killed.**\n"
        "The previous implementation attempt may have left partial work. Before starting:\n\n"
        "1. Check `git status` and recent commits for any partial work from the previous attempt\n"
        "2. Review any partially modified files for incomplete changes\n"
        "3. Continue from where the previous attempt left off rather than starting over\n"
        "4. Fix any broken partial work rather than duplicating it\n"
        f"\nRecovered at: {prev.get('recovered_at', 'unknown')}\n"
    )


def _build_resume_context(ps: PhaseState | None) -> str:
    """Build a resume/context block from phase state."""
    if ps is None:
        return ""
    parts: list[str] = []
    if ps.current_iteration > 1:
        parts.append(f"This is a continuation. {ps.current_iteration - 1} iterations have already run.")
    if ps.last_error:
        parts.append(f"The previous iteration FAILED with: {ps.last_error}")
        parts.append("Please avoid the same failure. If it was a timeout, focus on making fewer/smaller changes this iteration.")
    if ps.last_summary:
        parts.append(f"Previous iteration summary: {ps.last_summary}")
    if not parts:
        return ""
    return "\n## Resume Context\n\n" + "\n".join(parts) + "\n"


def _planning_doc(state: StateManager) -> tuple[str, str]:
    """The document the backlog is planned from, and where it can be read.

    In spec mode that is the in-repo specs; otherwise the design document.
    """
    if state.spec_mode:
        content, sources = spec_lib.planning_context(state)
        if content:
            return content, sources
    return state.read_design_doc(), str(state.design_doc_path)


def _planning_doc_for_prompt(state: StateManager, *, empty: str = "(no design document)") -> str:
    """Planning document inlined for a prompt, or a pointer when it is too large."""
    content, sources = _planning_doc(state)
    if not content:
        return empty
    if len(content) > LARGE_DOC_THRESHOLD:
        return f"(Design doc too large to inline — read it from: {sources})"
    return _escape_template_vars(content)


# -- Plan prompts --


def assemble_plan_prompt(
    state: StateManager,
    *,
    iteration: int,
    total: int,
    user_prompt: str,
    phase_state: PhaseState | None = None,
) -> str:
    """Build the prompt for a plan iteration."""
    design_doc_file = str(state.design_doc_path)
    guardrails_file = str(state.guardrails_path)
    research_notes_file = str(state.research_notes_path)
    resume = _build_resume_context(phase_state)

    # Combine project-level prompt with CLI --user-prompt
    project_prompt = state.read_phase_prompt("plan")
    combined_prompt = "\n\n".join(p for p in [project_prompt, user_prompt] if p) or "Create a comprehensive design document."

    # Build solutions recall section for plan awareness (local + global)
    solutions_recall = ""
    if state.has_solutions() or state.has_global_solutions():
        solutions_ctx = _build_solutions_context(state)
        if solutions_ctx:
            recall = _safe_sub(COMPOUND_RECALL_SECTION, "solutions_context", _escape_template_vars(solutions_ctx))
            solutions_recall = "\n" + recall

    if iteration == 1 and not state.has_design_doc():
        template = state.resolve_prompt_template("plan-initial", INITIAL_PLAN_PROMPT)
        return template.format(
            user_prompt=combined_prompt,
            design_doc_file=design_doc_file,
            guardrails_file=guardrails_file,
            research_notes_file=research_notes_file,
        ) + solutions_recall + resume

    template = state.resolve_prompt_template("plan-iteration", ITERATION_PROMPT_TEMPLATE)
    return template.format(
        current=iteration,
        total=total,
        user_prompt=combined_prompt,
        design_doc_file=design_doc_file,
        research_notes_file=research_notes_file,
    ) + solutions_recall + resume


# -- Spec prompts --


def assemble_spec_prompt(
    state: StateManager,
    spec: spec_lib.Spec,
    *,
    iteration: int,
    total: int,
    user_prompt: str,
    phase_state: PhaseState | None = None,
) -> str:
    """Build the prompt for a spec authoring iteration (create, refine or amend)."""
    resume = _build_resume_context(phase_state)
    spec_file = spec_lib.rel_path(state, spec.spec_path)
    design_file = spec_lib.rel_path(state, spec.design_path)
    today = datetime.now().strftime("%Y-%m-%d")

    project_prompt = state.read_phase_prompt("spec")
    combined_prompt = "\n\n".join(p for p in [project_prompt, user_prompt] if p)

    solutions_recall = ""
    if state.has_solutions() or state.has_global_solutions():
        solutions_ctx = _build_solutions_context(state)
        if solutions_ctx:
            recall = _safe_sub(COMPOUND_RECALL_SECTION, "solutions_context", _escape_template_vars(solutions_ctx))
            solutions_recall = "\n" + recall

    if iteration == 1 and not spec.exists():
        guardrails_instruction = ""
        if not state.guardrails_path.exists():
            guardrails_instruction = (
                f"- `{state.guardrails_path}` — guardrails: coding standards, testing "
                "requirements, git practices, security requirements.\n"
            )
        existing_inputs = ""
        if state.has_design_doc():
            existing_inputs = (
                "\n## Existing design document\n\n"
                f"An earlier design document exists at `{state.design_doc_path}`. Read it and carry "
                "its content over into the spec and the design — nothing in it should get lost.\n"
            )
        prompt = state.resolve_prompt_template("spec-initial", SPEC_INITIAL_PROMPT)
        prompt = _safe_sub(prompt, "spec_file", spec_file)
        prompt = _safe_sub(prompt, "design_file", design_file)
        prompt = _safe_sub(prompt, "guardrails_instruction", guardrails_instruction)
        prompt = _safe_sub(prompt, "existing_inputs", existing_inputs)
        prompt = _safe_sub(prompt, "research_notes_file", str(state.research_notes_path))
        prompt = _safe_sub(prompt, "today", today)
        prompt = _safe_sub(
            prompt, "user_prompt",
            _escape_template_vars(combined_prompt or "Write a comprehensive specification for this project."),
        )
        return prompt + solutions_recall + resume

    others = [s for s in spec_lib.list_specs(state) if s.name != spec.name]
    other_specs = ""
    if others:
        listed = "\n".join(f"  - `{spec_lib.rel_path(state, o.spec_path)}`" for o in others)
        other_specs = (
            "- Other specs in this repository (keep this one consistent with them, "
            f"do not edit them):\n{listed}\n"
        )

    prompt = state.resolve_prompt_template("spec-iteration", SPEC_ITERATION_PROMPT)
    prompt = _safe_sub(prompt, "current", str(iteration))
    prompt = _safe_sub(prompt, "total", str(total))
    prompt = _safe_sub(prompt, "spec_file", spec_file)
    prompt = _safe_sub(prompt, "design_file", design_file)
    prompt = _safe_sub(prompt, "research_notes_file", str(state.research_notes_path))
    prompt = _safe_sub(prompt, "other_specs", other_specs)
    prompt = _safe_sub(prompt, "today", today)
    prompt = _safe_sub(
        prompt, "user_prompt",
        _escape_template_vars(combined_prompt or "Review the specification and improve its completeness and quality."),
    )
    return prompt + solutions_recall + resume


def _spec_files_list(state: StateManager, specs: list[spec_lib.Spec]) -> str:
    """Bullet list of spec documents with what each one is for."""
    roles = {
        spec_lib.SPEC_FILE: "requirements (WHAT / WHY)",
        spec_lib.DESIGN_FILE: "technical design (HOW)",
        spec_lib.TASKS_FILE: "task list",
    }
    lines: list[str] = []
    for spec in specs:
        for path in spec.documents():
            role = roles.get(path.name, "supporting document")
            lines.append(f"- `{spec_lib.rel_path(state, path)}` — {role}")
    return "\n".join(lines)


_REQUIREMENT_BODY_LIMIT = 4000


def _spec_story_context(state: StateManager, story: Story) -> tuple[str, str, spec_lib.Spec | None]:
    """(spec files list, requirement text block, spec) for a story in spec mode."""
    all_specs = spec_lib.list_specs(state)
    spec, requirements = spec_lib.story_requirements(state, story)
    spec_files = _spec_files_list(state, [spec] if spec else all_specs)

    _, traced_ids = spec_lib.story_trace(story, all_specs)
    found_ids = {r.id for r in requirements}
    missing = [rid for rid in traced_ids if rid not in found_ids]

    if requirements and spec is not None:
        bodies = []
        for r in requirements:
            body = r.body
            if len(body) > _REQUIREMENT_BODY_LIMIT:
                body = body[:_REQUIREMENT_BODY_LIMIT] + "\n\n(... truncated — read the full text in the spec)"
            bodies.append(body)
        ids = ", ".join(r.id for r in requirements)
        block = (
            f"This story implements {ids} of spec `{spec.name}`. Current text of the requirement(s):\n\n"
            + "\n\n".join(bodies)
        )
    else:
        block = (
            "This story is not traced to a specific requirement. Find the requirement(s) it "
            "relates to in the spec before you start. If the spec does not cover it, keep the "
            "change minimal and record the gap under \"Open questions\" in spec.md."
        )
    if missing:
        block += (
            f"\n\n**Note:** {', '.join(missing)} is referenced by this story but no longer exists "
            "in the spec. If that makes the story obsolete, do not implement it — return status "
            "`skipped` with the reason."
        )
    return spec_files, _escape_template_vars(block), spec if traced_ids else None


def _build_spec_implement_context(state: StateManager, story: Story) -> str:
    if not state.spec_mode or not spec_lib.list_specs(state):
        return ""
    spec_files, requirements_block, spec = _spec_story_context(state, story)
    tasks_rule = ""
    if spec is not None and spec_lib.tasks_managed(spec):
        tasks_file = spec_lib.rel_path(state, spec.tasks_path)
        tasks_rule = (
            f"6. Tick this task in `{tasks_file}` as part of your commit: change "
            f"`- [ ] **{story.id}**` to `- [x] **{story.id}**` and nothing else in that file"
        )
    section = state.resolve_prompt_template("spec-implement", SPEC_IMPLEMENT_SECTION)
    section = _safe_sub(section, "spec_files", spec_files)
    section = _safe_sub(section, "story_requirements", requirements_block)
    section = _safe_sub(section, "tasks_rule", _escape_template_vars(tasks_rule))
    return section


def _build_spec_review_context(state: StateManager, story: Story) -> str:
    if not state.spec_mode or not spec_lib.list_specs(state):
        return ""
    spec_files, requirements_block, _ = _spec_story_context(state, story)
    section = state.resolve_prompt_template("spec-review", SPEC_REVIEW_SECTION)
    section = _safe_sub(section, "spec_files", spec_files)
    section = _safe_sub(section, "story_requirements", requirements_block)
    return section


def _build_git_context(story: Story) -> str:
    """Tell the implementer which branch it is on — and that shipping is not its job."""
    info = story.metadata.get("git")
    if not isinstance(info, dict) or not info.get("branch"):
        return ""
    lines = [
        "\n## Branch\n",
        f"This story has its own branch, `{info['branch']}`, started from `{info.get('base') or 'the base branch'}`. "
        "It is checked out in your working directory.\n",
        "- Commit your work to this branch",
        "- Do not switch branches, push, or open a pull request — pralph ships the branch after its review",
    ]
    if info.get("pr_url"):
        lines.append(f"- A pull request for it is already open ({info['pr_url']}); your commits will be added to it")
    return "\n".join(lines) + "\n"


_PAUSE_SECTION = """
## When a person has to step in

If the story cannot be finished without something only a person can do — apply
infrastructure, check a deployment, make a decision — do everything that can
be done in code, commit it, and return:

```json
{"status": "needs_human", "summary": "What is done so far", "reason": "Exactly what the person has to do"}
```

pralph pauses the story, passes your `reason` on, and continues it once the
person says it is done. Use this only when you are truly stuck on a person.
"""


def _build_pause_context(story: Story) -> str:
    """A story that was paused for a person and is now continued."""
    pause = story.metadata.get("needs_human")
    note = str(story.metadata.get("human_note") or "").strip()
    if not isinstance(pause, dict) and not note:
        return ""
    lines = ["\n## Continuing after a pause\n", "You paused this story earlier because a person had to do something first:\n"]
    reason = str((pause or {}).get("reason") or "").strip() if isinstance(pause, dict) else ""
    lines.append(f"> {reason or '(no reason recorded)'}\n")
    lines.append(f"They report: {note}\n" if note else "They report that it is done.\n")
    lines.append("Your earlier work is committed. Check that what you asked for is in place, then finish the story.")
    return "\n".join(lines) + "\n"


def _build_linear_context(story: Story) -> str:
    """Tell the implementer which Linear issue it is working on."""
    link = story.metadata.get("linear")
    if not isinstance(link, dict) or not link.get("identifier"):
        return ""
    identifier = link["identifier"]
    url = f" ({link['url']})" if link.get("url") else ""
    return (
        "\n## Linear\n\n"
        f"This story is tracked in Linear as **{identifier}**{url}.\n\n"
        f"- Mention `{identifier}` in your commit message so Linear links the commit to the issue\n"
        "- Do not edit the issue yourself — pralph moves it through the workflow and posts your summary\n"
    )


# -- Stories prompts --


def assemble_stories_prompt(
    state: StateManager,
    *,
    mode: str = "extract",
    phase_state: PhaseState | None = None,
) -> str:
    """Build the prompt for a stories iteration."""
    resume = _build_resume_context(phase_state)
    if mode == "spec":
        return _assemble_spec_extract_prompt(state) + resume
    if mode == "research":
        return _assemble_research_prompt(state) + resume
    if mode == "webgen":
        return _assemble_webgen_prompt(state) + resume
    return _assemble_extract_prompt(state) + resume


def _assemble_extract_prompt(state: StateManager) -> str:
    design_doc = state.read_design_doc()
    if len(design_doc) > LARGE_DOC_THRESHOLD:
        design_doc_content = f"(Design doc too large to inline — read it from: {state.design_doc_path})"
    else:
        design_doc_content = _escape_template_vars(design_doc)

    total = len(state.load_stories())
    existing = _escape_template_vars(state.format_existing_stories_context())
    cat_stats = state.format_category_stats()
    inputs_list = f"- {state.design_doc_path}"
    guardrails = state.guardrails_path
    if guardrails.exists():
        inputs_list += f"\n- {guardrails}"

    prompt = state.resolve_prompt_template("stories-extract", PLANNING_EXTRACT_PROMPT)
    prompt = _safe_sub(prompt, "design_doc", design_doc_content)
    prompt = _safe_sub(prompt, "total_stories", str(total))
    prompt = _safe_sub(prompt, "existing_stories", existing)
    prompt = _safe_sub(prompt, "category_stats", cat_stats)
    prompt = _safe_sub(prompt, "inputs_list", inputs_list)
    return prompt


def _assemble_spec_extract_prompt(state: StateManager) -> str:
    """Story extraction from the in-repo specs, with requirement tracing."""
    specs = spec_lib.list_specs(state)
    coverage = spec_lib.build_coverage(state)
    work = spec_lib.pending_work(state)

    overview: list[str] = []
    for spec in specs:
        overview.append(f"### Spec `{spec.name}`\n")
        overview.append(_spec_files_list(state, [spec]))
        rows = coverage.get(spec.name, [])
        if rows:
            overview.append("\nRequirements:")
            for row in rows:
                n = len(row.stories)
                covered = f"{n} stor{'y' if n == 1 else 'ies'}, {row.state}" if n else "no stories"
                overview.append(f"- {row.requirement.id}: {row.requirement.title} ({covered})")
        else:
            overview.append(
                "\n(No requirement IDs found in this spec — requirements need headings "
                "like `### REQ-001: Title`.)"
            )
        overview.append("")

    inline = "\n\n---\n\n".join(
        f"<!-- {spec_lib.rel_path(state, spec.spec_path)} -->\n\n{spec.read().strip()}" for spec in specs
    )
    if len(inline) > LARGE_DOC_THRESHOLD:
        spec_content = "(Specs too large to inline — read them from the files listed above.)"
    else:
        spec_content = inline

    pending: list[str] = []
    if work.uncovered:
        pending.append("### New — no stories yet\n")
        pending.extend(f"- `{r.spec}` {r.id}: {r.title}" for r in work.uncovered)
        pending.append("")
    if work.changed:
        pending.append("### Changed — spec text was edited after these stories were written\n")
        for row in work.changed:
            existing = ", ".join(f"{s.id} [{s.status.value}]" for s in row.stories)
            r = row.requirement
            pending.append(f"- `{r.spec}` {r.id}: {r.title} (existing stories: {existing})")
        pending.append("")
    if not pending:
        pending.append(
            "(None — every requirement has at least one story and no requirement changed. "
            "Check coverage as described in step 3.)"
        )

    prompt = state.resolve_prompt_template("stories-spec", SPEC_EXTRACT_PROMPT)
    prompt = _safe_sub(prompt, "specs_overview", _escape_template_vars("\n".join(overview).strip()))
    prompt = _safe_sub(prompt, "spec_content", _escape_template_vars(spec_content))
    prompt = _safe_sub(prompt, "pending_requirements", _escape_template_vars("\n".join(pending).strip()))
    prompt = _safe_sub(prompt, "total_stories", str(len(state.load_stories())))
    prompt = _safe_sub(prompt, "existing_stories", _escape_template_vars(spec_lib.format_stories_with_trace(state)))
    prompt = _safe_sub(prompt, "category_stats", state.format_category_stats())
    prompt = _safe_sub(prompt, "example_spec", specs[0].name if specs else "main")
    return prompt


def _assemble_research_prompt(state: StateManager) -> str:
    return state.resolve_prompt_template("stories-research", PLANNING_RESEARCH_PROMPT)


def _assemble_webgen_prompt(state: StateManager) -> str:
    design_doc, _ = _planning_doc(state)
    # Use first 5000 chars as summary for webgen
    summary = design_doc[:5000] if design_doc else "(no design doc)"
    summary = _escape_template_vars(summary)

    total = len(state.load_stories())
    existing = _escape_template_vars(state.format_existing_stories_context())
    cat_stats = state.format_category_stats()
    year = str(datetime.now().year)

    prompt = state.resolve_prompt_template("stories-webgen", WEBGEN_REQUIREMENTS_PROMPT)
    prompt = _safe_sub(prompt, "design_doc_summary", summary)
    prompt = _safe_sub(prompt, "total_stories", str(total))
    prompt = _safe_sub(prompt, "existing_stories", existing)
    prompt = _safe_sub(prompt, "category_stats", cat_stats)
    prompt = _safe_sub(prompt, "current_year", year)
    return prompt


# -- Add / Ideate prompts --


def assemble_add_prompt(
    state: StateManager,
    *,
    idea: str,
    is_next: bool,
) -> str:
    """Build the prompt for adding a single story from an idea."""
    design_doc_content = _planning_doc_for_prompt(state)

    if is_next:
        priority_mode = "Priority is forced to 1 — this story should be implemented next."
    else:
        priority_mode = "Claude picks the appropriate priority (1-5) based on the idea and existing backlog."

    total = len(state.load_stories())
    existing = _escape_template_vars(state.format_existing_stories_context())
    cat_stats = state.format_category_stats()

    prompt = state.resolve_prompt_template("add", ADD_STORY_PROMPT)
    prompt = _safe_sub(prompt, "user_idea", _escape_template_vars(idea))
    prompt = _safe_sub(prompt, "priority_mode", priority_mode)
    prompt = _safe_sub(prompt, "design_doc", design_doc_content)
    prompt = _safe_sub(prompt, "total_stories", str(total))
    prompt = _safe_sub(prompt, "existing_stories", existing)
    prompt = _safe_sub(prompt, "category_stats", cat_stats)
    return prompt


def assemble_ideate_prompt(
    state: StateManager,
    *,
    ideas_text: str,
    phase_state: PhaseState | None = None,
) -> str:
    """Build the prompt for batch ideation."""
    resume = _build_resume_context(phase_state)

    design_doc_content = _planning_doc_for_prompt(state)

    total = len(state.load_stories())
    existing = _escape_template_vars(state.format_existing_stories_context())
    cat_stats = state.format_category_stats()

    prompt = state.resolve_prompt_template("ideate", IDEATE_PROMPT)
    prompt = _safe_sub(prompt, "ideas_text", _escape_template_vars(ideas_text))
    prompt = _safe_sub(prompt, "design_doc", design_doc_content)
    prompt = _safe_sub(prompt, "total_stories", str(total))
    prompt = _safe_sub(prompt, "existing_stories", existing)
    prompt = _safe_sub(prompt, "category_stats", cat_stats)
    return prompt + resume


def assemble_refine_prompt(
    state: StateManager,
    *,
    instruction: str,
    original_stories: list["Story"],
) -> str:
    """Build the prompt for refining existing stories."""
    design_doc_content = _planning_doc_for_prompt(state)

    original_ids = [s.id for s in original_stories]
    original_json = json.dumps([s.to_dict() for s in original_stories], indent=2)

    total = len(state.load_stories())
    existing = _escape_template_vars(state.format_existing_stories_context())
    cat_stats = state.format_category_stats()

    prompt = state.resolve_prompt_template("refine", REFINE_PROMPT)
    prompt = _safe_sub(prompt, "refinement_instruction", _escape_template_vars(instruction))
    prompt = _safe_sub(prompt, "original_stories_json", _escape_template_vars(original_json))
    prompt = _safe_sub(prompt, "original_story_ids", ", ".join(original_ids))
    prompt = _safe_sub(prompt, "design_doc", design_doc_content)
    prompt = _safe_sub(prompt, "total_stories", str(total))
    prompt = _safe_sub(prompt, "existing_stories", existing)
    prompt = _safe_sub(prompt, "category_stats", cat_stats)
    return prompt


# -- Implement prompts --


def assemble_implement_prompt(
    state: StateManager,
    story: Story,
    phase_state: PhaseState | None = None,
    user_prompt: str = "",
) -> str:
    """Build the prompt for implementing a single story."""
    metadata = _escape_template_vars(json.dumps(story.to_dict(), indent=2))
    content = _escape_template_vars(story.content)
    title = _escape_template_vars(story.title)
    impl_summary = state.get_implemented_summary()
    resume = _build_resume_context(phase_state)

    prompt = state.resolve_prompt_template("implement", IMPLEMENTATION_IMPLEMENT_PROMPT)
    prompt = _safe_sub(prompt, "input_item.title", title)
    prompt = _safe_sub(prompt, "input_item.content", content)
    prompt = _safe_sub(prompt, "input_item.metadata", metadata)
    prompt = _safe_sub(prompt, "implemented_summary", impl_summary)

    # Crash recovery context (before resume)
    crash_context = _build_crash_recovery_context(story)

    # Project-level prompt (.pralph/implement-prompt.md) + CLI --user-prompt
    project_prompt = state.read_phase_prompt("implement")
    guidance_parts = [p for p in [project_prompt, user_prompt] if p]
    if guidance_parts:
        prompt += "\n## Project context\n\n" + "\n\n".join(guidance_parts) + "\n"

    # Spec-driven: the in-repo spec and the requirements this story implements
    prompt += _build_spec_implement_context(state, story)

    # Linear: which issue this story is tracked as
    prompt += _build_linear_context(story)

    # Its own branch, if stories have one; and how to hand over to a person
    prompt += _build_git_context(story)
    prompt += _PAUSE_SECTION
    prompt += _build_pause_context(story)

    # Rework context: include reviewer feedback when retrying a rejected story
    rework_context = _build_rework_context(state, story)

    # Compound learning: inject relevant past solutions (local + global)
    solutions_recall = ""
    if state.has_solutions() or state.has_global_solutions():
        solutions_ctx = _build_solutions_context(state, story)
        if solutions_ctx:
            recall = _safe_sub(COMPOUND_RECALL_SECTION, "solutions_context", _escape_template_vars(solutions_ctx))
            solutions_recall = "\n" + recall

    return prompt + crash_context + rework_context + solutions_recall + resume


def _build_rework_context(state: StateManager, story: Story) -> str:
    """Build context block with reviewer feedback for stories in rework status."""
    from pralph.models import StoryStatus

    if story.status != StoryStatus.rework:
        return ""
    feedback = state.read_review_feedback(story.id)
    if not feedback:
        return ""
    return (
        "\n## Reviewer Feedback (Previous Attempt)\n\n"
        "**This story was previously implemented but rejected during review.**\n"
        "Address the following feedback before re-submitting:\n\n"
        f"{feedback}\n"
    )


def assemble_review_prompt(
    state: StateManager,
    story: Story,
) -> str:
    """Build the prompt for reviewing a story implementation."""
    title = _escape_template_vars(story.title)
    content = _escape_template_vars(story.content)

    ac_lines = story.acceptance_criteria
    if ac_lines:
        ac_text = "\n".join(f"- {ac}" for ac in ac_lines)
    else:
        ac_text = "(no explicit acceptance criteria)"
    ac_text = _escape_template_vars(ac_text)

    # Load project-specific review guidance if present
    review_prompt_path = state.state_dir / "review-prompt.md"
    if review_prompt_path.exists():
        guidance = review_prompt_path.read_text().strip()
        review_guidance = f"## Project Review Guidelines\n\n{_escape_template_vars(guidance)}"
    else:
        review_guidance = ""

    spec_review = _build_spec_review_context(state, story)
    if spec_review:
        review_guidance = "\n\n".join(p for p in [review_guidance, spec_review] if p)

    prompt = state.resolve_prompt_template("review", REVIEW_PROMPT)
    prompt = _safe_sub(prompt, "story_title", title)
    prompt = _safe_sub(prompt, "story_content", content)
    prompt = _safe_sub(prompt, "acceptance_criteria", ac_text)
    prompt = _safe_sub(prompt, "review_guidance", review_guidance)

    info = story.metadata.get("git")
    if isinstance(info, dict) and info.get("branch"):
        base = info.get("base") or "main"
        prompt += (
            "\n## Branch under review\n\n"
            f"The story's work is on the branch `{info['branch']}`, checked out in your working directory. "
            f"Everything on it since `{base}` belongs to the story — look at `git diff {base}...HEAD` "
            f"(or `origin/{base}...HEAD`) rather than at the last commit alone.\n"
        )
    return prompt


def _compact_story(s: "Story") -> dict:
    """Return a compact dict with only the fields needed for categorization."""
    d: dict = {
        "id": s.id,
        "title": s.title,
        "category": s.category,
        "priority": s.priority,
        "complexity": s.complexity,
        "dependencies": s.dependencies,
    }
    # Include a truncated content preview — enough for categorization
    if s.content:
        d["content_preview"] = s.content[:300] + ("..." if len(s.content) > 300 else "")
    return d


# Threshold for inlining stories JSON vs referencing by file path
_STORIES_INLINE_THRESHOLD = 40_000


def assemble_phase1_analyze_prompt(
    state: StateManager,
) -> str:
    """Build the Phase 1 analysis prompt."""
    pending = state.get_pending_stories()
    compact = [_compact_story(s) for s in pending]
    stories_json = json.dumps(compact, indent=2)

    if len(stories_json) > _STORIES_INLINE_THRESHOLD:
        # Too large to inline — write to file and reference it
        stories_file = state.phase1_analysis_path.with_name("phase1-stories-input.json")
        stories_file.write_text(stories_json)
        stories_json = (
            f"(Stories too large to inline — {len(pending)} stories. "
            f"Read them from: {stories_file})"
        )

    stories_json = _escape_template_vars(stories_json)

    prompt = state.resolve_prompt_template("implement-phase1-analyze", IMPLEMENTATION_PHASE1_ANALYZE_PROMPT)
    prompt = _safe_sub(prompt, "stories_json", stories_json)
    return prompt


def assemble_phase1_implement_prompt(
    state: StateManager,
    story_ids: list[str],
    implementation_order: list[str],
    architecture_context: str = "",
) -> str:
    """Build the Phase 1 implementation prompt."""
    all_stories = {s.id: s for s in state.load_stories()}
    phase1_stories = [all_stories[sid].to_dict() for sid in story_ids if sid in all_stories]

    prompt = state.resolve_prompt_template("implement-phase1", IMPLEMENTATION_PHASE1_IMPLEMENT_PROMPT)
    prompt = _safe_sub(prompt, "phase1_stories_json", _escape_template_vars(json.dumps(phase1_stories, indent=2)))
    prompt = _safe_sub(prompt, "implementation_order", _escape_template_vars(json.dumps(implementation_order)))
    prompt = _safe_sub(prompt, "architecture_context", _escape_template_vars(architecture_context))
    return prompt


# -- Compound learning prompts --


def _build_solutions_context(state: StateManager, story: "Story | None" = None) -> str:
    """Build solutions context for injection into other prompts.

    Merges both project-local and global (domain-matched) solutions.
    With story: searches using story title/category, includes up to 3 full docs.
    Without story (plan phase): includes summary of all solutions.
    """
    has_local = state.has_solutions()
    has_global = state.has_global_solutions()

    if not has_local and not has_global:
        return ""

    if story is not None:
        # Search mode: find relevant solutions for this story
        query_parts = [story.title]
        if story.category:
            query_parts.append(story.category)
        query = " ".join(query_parts)
        matches = state.search_all_solutions(query, max_results=3)

        if not matches:
            return ""

        parts: list[str] = []
        for match in matches:
            title = match.get("title", "?")
            source = match.get("_source", "local")
            source_label = " (global)" if source == "global" else ""
            content = state.read_any_solution(match)
            if content:
                if len(content) > 2000:
                    content = content[:2000] + "\n\n(... truncated)"
                parts.append(f"### {title}{source_label}\n\n{content}")
            else:
                tags = ", ".join(match.get("tags", []))
                parts.append(f"### {title}{source_label}\n\nTags: {tags}")

        return "\n\n---\n\n".join(parts)
    else:
        # Summary mode: compact overview of all solutions (local + global)
        summaries: list[str] = []
        local_summary = state.get_solutions_summary()
        if local_summary:
            summaries.append(local_summary)
        global_summary = state.get_global_solutions_summary()
        if global_summary:
            if summaries:
                summaries.append("\n### Global learnings\n")
            summaries.append(global_summary)
        return "\n".join(summaries)


def assemble_compound_prompt(
    state: StateManager,
    story: "Story",
) -> str:
    """Build the prompt for compound learning capture after implementation."""
    impl_summary = state.get_implemented_summary()

    prompt = state.resolve_prompt_template("compound", COMPOUND_CAPTURE_PROMPT)
    prompt = _safe_sub(prompt, "story_id", _escape_template_vars(story.id))
    prompt = _safe_sub(prompt, "story_title", _escape_template_vars(story.title))
    prompt = _safe_sub(prompt, "implementation_summary", _escape_template_vars(impl_summary))
    return prompt


# -- Guardrails system prompt --


def build_guardrails_system_prompt(phase: str, state: StateManager) -> str:
    """Combine project guardrails + phase-appropriate guardrails for --append-system-prompt."""
    parts: list[str] = []

    # Project guardrails (if they exist)
    project_guardrails = state.read_guardrails()
    if project_guardrails:
        parts.append(project_guardrails)

    # Phase guardrails
    if phase == "plan":
        parts.append(DEFAULT_GUARDRAILS)
    elif phase == "stories":
        parts.append(STORY_GUARDRAILS)
    elif phase == "implement":
        parts.append(CODE_GUARDRAILS)
    else:
        parts.append(DEFAULT_GUARDRAILS)

    return "\n\n".join(parts)


# -- Justloop prompts --


def assemble_justloop_prompt(
    state: StateManager,
    *,
    user_prompt: str,
    phase_state: PhaseState | None = None,
) -> str:
    """Build the prompt for a justloop iteration."""
    resume = _build_resume_context(phase_state)

    prompt = state.resolve_prompt_template("justloop", JUSTLOOP_PROMPT)
    prompt = _safe_sub(prompt, "user_prompt", _escape_template_vars(user_prompt))
    return prompt + resume
