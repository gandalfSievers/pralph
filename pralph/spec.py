"""Spec-driven development: specs that live in the repo and drive the backlog.

Layout (``<spec_dir>`` defaults to ``specs/``, see ``StateManager.spec_dir``)::

    specs/
      <name>/
        spec.md     requirements — WHAT and WHY (source of truth)
        design.md   technical design — HOW
        tasks.md    task list derived from the backlog (written by pralph)

Requirements are identified by stable IDs (``### REQ-001: Title``). Stories
record which requirements they implement (``metadata.spec`` and
``metadata.requirements``), which gives pralph three things for free:

* coverage   — which requirements have no stories yet
* drift      — which requirements were edited after their stories were written
               (digest of the requirement text vs. the recorded baseline)
* conformance — implement/review prompts carry the exact requirement text
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pralph.models import Story, StoryStatus
from pralph.state import StateManager

SPEC_FILE = "spec.md"
DESIGN_FILE = "design.md"
TASKS_FILE = "tasks.md"

# Requirement ID prefixes. REQ/NFR are what pralph's own template produces;
# FR is accepted so specs written in the common "FR-001" style work as-is.
_REQ_ID = r"(?:REQ|NFR|FR)-\d+[a-z]?"

# "### REQ-001: Title" (any heading level 2-6, optional bold, ':' / dash separator)
_REQ_HEADING_RE = re.compile(
    rf"^(?P<hashes>#{{2,6}})\s+\**(?P<id>{_REQ_ID})\**\s*[:—–-]?\s*(?P<title>.*?)\s*#*\s*$"
)
# "- **FR-001**: The system MUST ..." (bullet-style requirements)
_REQ_BULLET_RE = re.compile(
    rf"^\s{{0,3}}[-*+]\s+\*\*(?P<id>{_REQ_ID})\*\*\s*[:—–-]?\s*(?P<title>.*?)\s*$"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_REQ_ID_RE = re.compile(rf"^{_REQ_ID}$")


# ── model ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Requirement:
    spec: str
    id: str
    title: str
    body: str  # full text of the requirement, heading/bullet line included

    @property
    def digest(self) -> str:
        """Fingerprint of the requirement text, insensitive to whitespace changes."""
        normalized = " ".join(self.body.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Spec:
    name: str
    path: Path  # the spec's directory

    @property
    def spec_path(self) -> Path:
        return self.path / SPEC_FILE

    @property
    def design_path(self) -> Path:
        return self.path / DESIGN_FILE

    @property
    def tasks_path(self) -> Path:
        return self.path / TASKS_FILE

    def exists(self) -> bool:
        return self.spec_path.is_file()

    def read(self) -> str:
        return self.spec_path.read_text() if self.spec_path.is_file() else ""

    def read_design(self) -> str:
        return self.design_path.read_text() if self.design_path.is_file() else ""

    def requirements(self) -> list[Requirement]:
        return parse_requirements(self.name, self.read())

    def documents(self) -> list[Path]:
        """All markdown files of the spec, spec.md and design.md first."""
        if not self.path.is_dir():
            return []
        order = {SPEC_FILE: 0, DESIGN_FILE: 1, TASKS_FILE: 3}
        files = [p for p in self.path.glob("*.md") if p.is_file()]
        return sorted(files, key=lambda p: (order.get(p.name, 2), p.name))


@dataclass
class RequirementCoverage:
    requirement: Requirement
    stories: list[Story] = field(default_factory=list)
    changed: bool = False  # requirement text differs from the recorded baseline

    @property
    def state(self) -> str:
        """uncovered | dropped | pending | in_progress | error | done"""
        if not self.stories:
            return "uncovered"
        active = [s for s in self.stories if s.status not in (StoryStatus.skipped, StoryStatus.duplicate)]
        if not active:
            return "dropped"
        statuses = {s.status for s in active}
        if statuses == {StoryStatus.implemented}:
            return "done"
        if StoryStatus.error in statuses:
            return "error"
        if statuses & {StoryStatus.in_progress, StoryStatus.rework, StoryStatus.implemented}:
            return "in_progress"
        return "pending"


@dataclass
class SpecWork:
    """What the spec currently asks for that the backlog doesn't reflect."""

    uncovered: list[Requirement] = field(default_factory=list)
    changed: list[RequirementCoverage] = field(default_factory=list)
    # (story, requirement refs that no longer exist in the spec)
    orphaned: list[tuple[Story, list[str]]] = field(default_factory=list)
    untraced: list[Story] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.uncovered or self.changed)


# ── parsing ──────────────────────────────────────────────────────────


def parse_requirements(spec_name: str, text: str) -> list[Requirement]:
    """Extract requirements from spec markdown.

    A heading requirement (``### REQ-001: Title``) owns everything up to the
    next heading of the same or a higher level. A bullet requirement
    (``- **FR-001**: text``) owns the bullet and its indented continuation
    lines. Headings inside fenced code blocks are ignored, and the first
    occurrence of an ID wins.
    """
    lines = text.splitlines()
    in_fence = False
    fenced: list[bool] = []
    for line in lines:
        if _FENCE_RE.match(line):
            fenced.append(True)
            in_fence = not in_fence
        else:
            fenced.append(in_fence)

    requirements: list[Requirement] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        if fenced[i]:
            i += 1
            continue
        line = lines[i]

        m = _REQ_HEADING_RE.match(line)
        if m:
            level = len(m.group("hashes"))
            end = i + 1
            while end < len(lines):
                if not fenced[end]:
                    h = _HEADING_RE.match(lines[end])
                    if h and len(h.group(1)) <= level:
                        break
                end += 1
            _add_requirement(requirements, seen, spec_name, m, lines[i:end])
            i = end
            continue

        m = _REQ_BULLET_RE.match(line)
        if m:
            end = i + 1
            while end < len(lines) and lines[end].strip() and lines[end][:1] in (" ", "\t"):
                end += 1
            _add_requirement(requirements, seen, spec_name, m, lines[i:end])
            i = end
            continue

        i += 1
    return requirements


def _add_requirement(
    out: list[Requirement], seen: set[str], spec_name: str, m: re.Match, body_lines: list[str],
) -> None:
    req_id = m.group("id")
    if req_id in seen:
        return
    seen.add(req_id)
    title = m.group("title").strip().strip("*").strip()
    out.append(Requirement(
        spec=spec_name,
        id=req_id,
        title=title,
        body="\n".join(body_lines).strip(),
    ))


# ── locating specs ───────────────────────────────────────────────────


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60].strip("-")


def list_specs(state: StateManager) -> list[Spec]:
    spec_dir = state.spec_dir
    if not spec_dir.is_dir():
        return []
    return [
        Spec(name=p.name, path=p)
        for p in sorted(spec_dir.iterdir())
        if p.is_dir() and (p / SPEC_FILE).is_file()
    ]


def get_spec(state: StateManager, name: str) -> Spec:
    """Spec handle by name (the spec does not have to exist yet)."""
    for spec in list_specs(state):
        if spec.name == name:
            return spec
    slug = slugify(name)
    if not slug:
        raise ValueError(f"Invalid spec name: {name!r}")
    return Spec(name=slug, path=state.spec_dir / slug)


def resolve_spec_name(state: StateManager, name: str | None) -> str:
    """Pick the spec a command should work on.

    Explicit name wins. Without one: the only existing spec, or — when the
    repo has no spec yet — a name derived from the project directory.
    """
    if name:
        return get_spec(state, name).name
    specs = list_specs(state)
    if len(specs) == 1:
        return specs[0].name
    if not specs:
        return slugify(state.project_dir.resolve().name) or "main"
    names = ", ".join(s.name for s in specs)
    raise ValueError(f"Several specs exist ({names}) — pick one with --name")


def rel_path(state: StateManager, path: Path) -> str:
    """Path relative to the project root, for prompts and output."""
    try:
        return str(path.resolve().relative_to(state.project_dir.resolve()))
    except ValueError:
        return str(path)


# ── story ↔ requirement tracing ──────────────────────────────────────


def story_trace(story: Story, specs: list[Spec]) -> tuple[str | None, list[str]]:
    """Return (spec name, requirement IDs) recorded on a story.

    Tolerates what an LLM might produce: a comma-separated string instead of
    a list, ``spec:REQ-001`` qualified IDs, or a missing ``spec`` when the
    repo only has one.
    """
    spec_name = story.metadata.get("spec")
    raw = story.metadata.get("requirements") or []
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw)
    req_ids: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if ":" in item:
            prefix, _, item = item.rpartition(":")
            spec_name = spec_name or prefix.strip()
        normalized = _normalize_req_id(item)
        if normalized and normalized not in req_ids:
            req_ids.append(normalized)
    if not isinstance(spec_name, str) or not spec_name.strip():
        spec_name = specs[0].name if len(specs) == 1 and req_ids else None
    else:
        spec_name = spec_name.strip()
    return spec_name, req_ids


def _normalize_req_id(raw: str) -> str | None:
    """'req-001' → 'REQ-001'; returns None when it isn't a requirement ID."""
    m = re.match(r"^([A-Za-z]+)-(\d+)([A-Za-z]?)$", raw.strip())
    if not m:
        return None
    candidate = f"{m.group(1).upper()}-{m.group(2)}{m.group(3).lower()}"
    return candidate if _REQ_ID_RE.match(candidate) else None


def normalize_trace(state: StateManager, stories: list[Story]) -> list[Story]:
    """Clean up trace metadata on freshly parsed stories. Returns the untraced ones."""
    specs = list_specs(state)
    known = {s.name for s in specs}
    untraced: list[Story] = []
    for story in stories:
        spec_name, req_ids = story_trace(story, specs)
        if spec_name in known and req_ids:
            story.metadata["spec"] = spec_name
            story.metadata["requirements"] = req_ids
        else:
            untraced.append(story)
    return untraced


def inherit_trace(new_stories: list[Story], originals: list[Story]) -> None:
    """Carry spec tracing over to stories that replace others (refine).

    A replacement that doesn't say what it implements inherits the union of
    the originals' requirements, so refining never silently drops coverage.
    """
    spec_names = {o.metadata.get("spec") for o in originals if o.metadata.get("spec")}
    if len(spec_names) != 1:
        return
    spec_name = spec_names.pop()
    union: list[str] = []
    for o in originals:
        for r in o.metadata.get("requirements") or []:
            if isinstance(r, str) and r not in union:
                union.append(r)
    if not union:
        return
    for s in new_stories:
        if not s.metadata.get("requirements"):
            s.metadata["spec"] = spec_name
            s.metadata["requirements"] = list(union)


# ── baseline (requirement digests the backlog was generated from) ────


def load_baseline(state: StateManager) -> dict[str, dict[str, str]]:
    path = state.spec_baseline_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def save_baseline(state: StateManager, baseline: dict[str, dict[str, str]]) -> None:
    state.spec_baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")


def acknowledge(state: StateManager, spec_name: str | None = None, req_ids: list[str] | None = None) -> None:
    """Record the current text of requirements as "the backlog reflects this".

    Called when a story is created for a requirement, and for everything once
    story extraction completes — from then on only later edits count as drift.
    """
    baseline = load_baseline(state)
    for spec in list_specs(state):
        if spec_name is not None and spec.name != spec_name:
            continue
        current = {r.id: r.digest for r in spec.requirements()}
        if req_ids is None:
            baseline[spec.name] = current  # also forgets removed requirements
        else:
            entry = baseline.setdefault(spec.name, {})
            for rid in req_ids:
                if rid in current:
                    entry[rid] = current[rid]
    save_baseline(state, baseline)


# ── coverage ─────────────────────────────────────────────────────────


def build_coverage(state: StateManager) -> dict[str, list[RequirementCoverage]]:
    """Per spec: every requirement with the stories that implement it."""
    specs = list_specs(state)
    stories = state.load_stories()
    baseline = load_baseline(state)

    by_ref: dict[tuple[str, str], list[Story]] = {}
    for story in stories:
        spec_name, req_ids = story_trace(story, specs)
        if spec_name is None:
            continue
        for rid in req_ids:
            by_ref.setdefault((spec_name, rid), []).append(story)

    coverage: dict[str, list[RequirementCoverage]] = {}
    for spec in specs:
        known = baseline.get(spec.name, {})
        rows: list[RequirementCoverage] = []
        for req in spec.requirements():
            covering = by_ref.get((spec.name, req.id), [])
            recorded = known.get(req.id)
            rows.append(RequirementCoverage(
                requirement=req,
                stories=covering,
                # Only drift if we know what the stories were written against.
                changed=bool(covering) and recorded is not None and recorded != req.digest,
            ))
        coverage[spec.name] = rows
    return coverage


def pending_work(state: StateManager) -> SpecWork:
    """Compare spec and backlog: what needs stories, what drifted, what is stale."""
    specs = list_specs(state)
    coverage = build_coverage(state)
    work = SpecWork()
    for rows in coverage.values():
        for row in rows:
            if not row.stories:
                work.uncovered.append(row.requirement)
            elif row.changed:
                work.changed.append(row)

    existing = {spec.name: {r.requirement.id for r in coverage[spec.name]} for spec in specs}
    for story in state.load_stories():
        spec_name, req_ids = story_trace(story, specs)
        if spec_name is None or not req_ids:
            if story.status not in (StoryStatus.skipped, StoryStatus.duplicate):
                work.untraced.append(story)
            continue
        gone = [rid for rid in req_ids if rid not in existing.get(spec_name, set())]
        if gone and story.status in (StoryStatus.pending, StoryStatus.rework, StoryStatus.error):
            work.orphaned.append((story, gone))
    return work


def story_requirements(state: StateManager, story: Story) -> tuple[Spec | None, list[Requirement]]:
    """The spec and requirement texts a story implements (for prompts)."""
    specs = list_specs(state)
    spec_name, req_ids = story_trace(story, specs)
    spec = next((s for s in specs if s.name == spec_name), None)
    if spec is None:
        return None, []
    by_id = {r.id: r for r in spec.requirements()}
    return spec, [by_id[rid] for rid in req_ids if rid in by_id]


def format_stories_with_trace(state: StateManager) -> str:
    """Existing-stories context for the extraction prompt, including tracing."""
    stories = state.load_stories()
    if not stories:
        return "(none yet)"
    specs = list_specs(state)
    lines: list[str] = []
    for s in stories:
        deps = ", ".join(s.dependencies) if s.dependencies else "none"
        spec_name, req_ids = story_trace(s, specs)
        trace = f"{spec_name}: {', '.join(req_ids)}" if spec_name and req_ids else "untraced"
        lines.append(
            f"- {s.id}: {s.title} [priority={s.priority}, status={s.status.value}, "
            f"deps={deps}, implements={trace}]"
        )
    return "\n".join(lines)


# ── planning context (spec as the "design doc" for other phases) ─────


def planning_context(state: StateManager) -> tuple[str, str]:
    """Spec + design text of all specs, and where to read them from.

    Lets phases that were written against a single design document (add,
    ideate, refine, webgen) work from the in-repo specs instead.
    """
    parts: list[str] = []
    sources: list[str] = []
    for spec in list_specs(state):
        for path in (spec.spec_path, spec.design_path):
            if path.is_file():
                rel = rel_path(state, path)
                sources.append(rel)
                parts.append(f"<!-- {rel} -->\n\n{path.read_text().strip()}")
    return "\n\n---\n\n".join(parts), ", ".join(sources)


# ── tasks.md ─────────────────────────────────────────────────────────

# How pralph recognizes a tasks.md as its own. A file without it was written
# by a person or by other spec tooling and is never touched.
_TASKS_MARKER = "Generated by pralph"

_TASKS_HEADER = """\
# Tasks — {name}

<!-- Generated by pralph from the story backlog — do not add tasks here.
     Scope changes go into spec.md; run `pralph stories` to derive new tasks.
     Implementations tick their own box; pralph keeps the rest in sync. -->
"""

_TERMINAL_LABELS = {
    StoryStatus.skipped: "skipped",
    StoryStatus.duplicate: "duplicate",
    StoryStatus.external: "external",
}


def task_line(story: Story, req_ids: list[str]) -> str:
    """One checklist line. Only the checkbox changes when a story is implemented,
    so an implementer ticking the box produces exactly what pralph would write."""
    done = story.status == StoryStatus.implemented or story.status in _TERMINAL_LABELS
    details = [", ".join(req_ids), f"P{story.priority}", story.complexity or "medium"]
    if story.dependencies:
        details.append("after " + ", ".join(story.dependencies))
    title = " ".join(story.title.split())
    label = _TERMINAL_LABELS.get(story.status)
    if label:
        title = f"~~{title}~~"
        details.append(label)
    return f"- [{'x' if done else ' '}] **{story.id}** {title} — {' · '.join(d for d in details if d)}"


def render_tasks(spec: Spec, traced: list[tuple[Story, list[str]]]) -> str:
    lines = [_TASKS_HEADER.format(name=spec.name)]
    if not traced:
        lines.append("_No tasks yet — run `pralph stories`._")
    for story, req_ids in traced:
        lines.append(task_line(story, req_ids))
    return "\n".join(lines) + "\n"


def tasks_managed(spec: Spec) -> bool:
    """Whether the spec's ``tasks.md`` exists and was generated by pralph."""
    try:
        return _TASKS_MARKER in spec.tasks_path.read_text()
    except OSError:
        return False


def sync_tasks(state: StateManager) -> list[Path]:
    """Regenerate ``tasks.md`` of every spec from the backlog. Returns files that changed.

    A ``tasks.md`` that pralph did not generate is left alone: a repository
    can have a ``specs/`` tree of its own, and spec mode switches on by itself.
    """
    specs = list_specs(state)
    if not specs:
        return []
    stories = state.load_stories()
    per_spec: dict[str, list[tuple[Story, list[str]]]] = {s.name: [] for s in specs}
    for story in stories:
        spec_name, req_ids = story_trace(story, specs)
        if spec_name in per_spec and req_ids:
            per_spec[spec_name].append((story, req_ids))

    changed: list[Path] = []
    for spec in specs:
        traced = per_spec[spec.name]
        exists = spec.tasks_path.exists()
        if exists and not tasks_managed(spec):
            continue  # somebody else's task list
        if not traced and not exists:
            continue  # don't litter specs that have no stories yet
        content = render_tasks(spec, traced)
        if not exists or spec.tasks_path.read_text() != content:
            spec.tasks_path.write_text(content)
            changed.append(spec.tasks_path)
    return changed


# ── wiring ───────────────────────────────────────────────────────────


def attach(state: StateManager) -> None:
    """Keep tasks.md and the requirement baseline in step with the backlog."""
    if getattr(state, "_spec_attached", False):
        return
    state._spec_attached = True  # type: ignore[attr-defined]

    def on_story_event(event: str, story: Story | None, info: dict) -> None:
        if not state.spec_mode:
            return
        if event == "added" and story is not None:
            spec_name, req_ids = story_trace(story, list_specs(state))
            if spec_name and req_ids:
                acknowledge(state, spec_name, req_ids)
        sync_tasks(state)

    state.add_story_listener(on_story_event)
