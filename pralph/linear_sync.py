"""Linear as pralph's tracker.

``push`` turns the plan into a Linear project: one parent issue per story
category, one sub-issue per story, and "blocks" relations for dependencies.
From then on the tickets are the source of truth for *what* to work on:

* ``pull`` (run before every story) copies titles, descriptions, priorities,
  dependencies and workflow state from Linear into the local backlog, and
  picks up issues people created in Linear.
* story status changes are pushed back as workflow transitions and comments
  (wired through ``StateManager.add_story_listener``).

The link can be scoped to a milestone, an assignee and/or a label
(``LinearLink.scope``). Then only matching issues become work, and nothing
outside the scope is ever written to — the way to share a Linear project with
other repositories and other people.

A link can also be read-mostly (``LinearLink.read_mostly``), for projects that
people run: pralph then only comments and moves tickets to In Progress. It
never closes or cancels a ticket and never creates issues or relations.

State sync is three-way. Each story remembers the Linear state it last saw
(``metadata.linear.state_type``). If Linear differs from that, somebody moved
the ticket and Linear wins; if it doesn't, any local change (an update that
failed to reach Linear, an edit in the viewer) is pushed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import click

from pralph import spec as spec_lib
from pralph import linear_oauth
from pralph.linear import LinearClient, LinearError
from pralph.models import Story, StoryStatus
from pralph.state import StateManager

# Friendly names for the usual category codes, used in parent-issue titles.
EPIC_NAMES = {
    "ADM": "Administration", "API": "API", "ARC": "Architecture", "ARCH": "Architecture",
    "AUTH": "Authentication", "DAT": "Data", "DB": "Database", "DBM": "Database",
    "DEP": "Deployment", "DOC": "Documentation", "FEAT": "Features", "FND": "Foundation",
    "FOUNDATION": "Foundation", "INF": "Infrastructure", "INFRA": "Infrastructure",
    "OPS": "Operations", "PERF": "Performance", "SEC": "Security", "SETUP": "Setup",
    "SYS": "System", "TEST": "Testing", "UI": "User Interface", "UX": "User Experience",
}
_GENERAL = "GENERAL"

# Headings that introduce the acceptance criteria in an issue description.
# The first one is what pralph writes; all of them are read. More can be
# configured with ``linear_ac_heading`` (see ``ac_headings``).
DEFAULT_AC_HEADINGS = ("Acceptance criteria", "Done when", "Definition of done")
_FOOTER_MARK = "pralph story"
_PROJECT_CONTENT_LIMIT = 60_000

_TITLE_ID_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d+\w*):\s+(.+)$")
_EPIC_TITLE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s+[—–-]\s+\S")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
# A markdown heading, or a line that is nothing but bold text ("**Notes**").
_ANY_HEADING_RE = re.compile(r"^(?:#{1,6}\s+\S|\s*\*\*[^*]+\*\*:?\s*$)")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.*\S)\s*$")
_PROJECT_IDENTIFIER_RE = re.compile(r"^P-[A-Z][A-Z0-9]*-\d+$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
# The slug ID closes the project segment of a project URL, whatever follows:
# .../project/shop-1a2b3c4d5e6f, .../project/shop-1a2b3c4d5e6f/overview?x=1
_SLUG_ID_RE = re.compile(r"/project/(?:[^/?#]*-)?([0-9a-f]{12})(?=[/?#]|$)", re.IGNORECASE)

# What a local status means for the ticket's workflow state class.
_STATUS_CLASS = {
    StoryStatus.pending: "open",
    StoryStatus.in_progress: "started",
    StoryStatus.rework: "started",
    # A failed story stays "In Progress" with a comment: moving it back to Todo
    # in Linear is how a human re-queues it.
    StoryStatus.error: "started",
    StoryStatus.implemented: "completed",
    StoryStatus.skipped: "canceled",
    StoryStatus.duplicate: "canceled",
    StoryStatus.external: "canceled",
    # A branch per story (pralph.ship): the ticket stays in progress while the
    # pull request is open — Linear's GitHub integration moves it to In Review —
    # and is done once the work is merged.
    StoryStatus.shipped: "started",
    StoryStatus.needs_human: "started",
    StoryStatus.merged: "completed",
}
# Stories that were dropped from the plan are not worth a ticket.
_NOT_IN_PLAN = (StoryStatus.skipped, StoryStatus.duplicate)
_SCOPE_KINDS = ("milestone", "assignee", "label")
# Linear's default name for the workflow state of each type.
_STANDARD_STATE = {
    "backlog": "backlog", "unstarted": "todo", "started": "in progress",
    "completed": "done", "canceled": "canceled",
}
_ACTIONABLE = (StoryStatus.pending, StoryStatus.rework, StoryStatus.error, StoryStatus.in_progress)


# ── link file ────────────────────────────────────────────────────────


@dataclass
class LinearLink:
    """``.pralph/linear.json`` — which Linear project this backlog lives in."""

    team_id: str = ""
    team_key: str = ""
    project_id: str = ""
    project_name: str = ""
    project_url: str = ""
    # category → {id, identifier, url, state_type} of the parent issue
    epics: dict[str, dict] = field(default_factory=dict)
    # Which issues of the project are this checkout's work. Any of:
    #   milestone {id, name} · assignee {id, name} · label {name}
    # Empty: every issue of the project.
    scope: dict[str, dict] = field(default_factory=dict)
    # Read-mostly: comment and move to In Progress, nothing else. Tickets are
    # closed by people (or the GitHub integration, on merge), never by pralph.
    read_mostly: bool = False

    @classmethod
    def load(cls, path: Path) -> LinearLink | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict) or not data.get("project_id"):
            return None
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        link = cls(**known)
        if not isinstance(link.epics, dict):
            link.epics = {}
        if not isinstance(link.scope, dict):
            link.scope = {}
        link.scope = {k: v for k, v in link.scope.items() if k in _SCOPE_KINDS and isinstance(v, dict)}
        link.read_mostly = link.read_mostly is True
        return link

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass
class PushReport:
    dry_run: bool = False
    project_name: str = ""
    project_url: str = ""
    project_created: bool = False
    epics_created: list[str] = field(default_factory=list)
    issues_created: list[tuple[str, str]] = field(default_factory=list)  # (story id, Linear identifier)
    relations_created: int = 0
    already_linked: int = 0


@dataclass
class PullReport:
    new: list[Story] = field(default_factory=list)
    updated: int = 0
    # (story id, old status, new status, reason)
    status_changes: list[tuple[str, str, str, str]] = field(default_factory=list)
    pushed: int = 0
    relations_created: int = 0  # dependencies that had not made it to Linear yet
    total: int = 0
    out_of_scope: int = 0  # issues of the project that the scope leaves out


# ── pure helpers (story ↔ issue) ─────────────────────────────────────


def state_class(state_type: str | None) -> str:
    """Collapse Linear's workflow state types into open / started / completed / canceled."""
    if state_type in ("started", "completed", "canceled"):
        return state_type
    if state_type == "duplicate":
        return "canceled"  # closed as a duplicate of another issue
    return "open"  # triage, backlog, unstarted


def story_link(story: Story) -> dict | None:
    link = story.metadata.get("linear")
    return link if isinstance(link, dict) and link.get("id") else None


def desired_class(story: Story) -> str | None:
    """Workflow state class the story's status calls for; None = not pralph's ticket to move."""
    link = story_link(story)
    if link and (link.get("claimed") or link.get("container") or link.get("outside")):
        # Claimed: somebody else is working on it. Container: it was broken down
        # into sub-issues in Linear. Outside: it left the project or the scope.
        # Either way people own the ticket's state.
        return None
    return status_class(story)


def status_class(story: Story) -> str:
    """Workflow state class a story's status stands for."""
    if story.status == StoryStatus.implemented and _on_branch(story):
        return "started"  # implemented on its own branch: not done before it is merged
    return _STATUS_CLASS[story.status]


def _on_branch(story: Story) -> bool:
    info = story.metadata.get("git")
    return isinstance(info, dict) and bool(info.get("branch"))


def is_outside(story: Story) -> bool:
    """The story's ticket is not (or no longer) in the linked project and scope: hands off."""
    link = story_link(story)
    return bool(link and link.get("outside"))


def issue_title(story: Story) -> str:
    return f"{story.id}: {' '.join(story.title.split())}"


def split_title(title: str) -> tuple[str | None, str]:
    """'AUTH-001: Login form' → ('AUTH-001', 'Login form'); no ID prefix → (None, title)."""
    m = _TITLE_ID_RE.match(title.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return None, title.strip()


def priority_to_linear(priority: int) -> int:
    """pralph 1 (highest)…5 → Linear 1 urgent, 2 high, 3 normal, 4 low."""
    return max(1, min(4, int(priority)))


def priority_from_linear(linear_priority: int, current: int) -> int:
    """Linear → pralph. 0 ("no priority") is treated as normal."""
    if linear_priority <= 0:
        return current if current else 3
    if linear_priority >= 4 and current == 5:
        return 5  # Linear has no fifth level; don't flatten a deliberate 5
    return min(4, linear_priority)


def ac_headings(configured: object = None) -> tuple[str, ...]:
    """Acceptance-criteria headings: the configured ones, then the built-in ones.

    ``configured`` is the ``linear_ac_heading`` config value — one heading or
    a list. Markdown around it ("## Done when:") is ignored.
    """
    if isinstance(configured, str):
        configured = [configured]
    names: list[str] = []
    for raw in list(configured or []) + list(DEFAULT_AC_HEADINGS):
        name = " ".join(str(raw).strip().strip("#*").strip().rstrip(":").split())
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return tuple(names)


def _ac_heading_re(headings: tuple[str, ...]) -> re.Pattern[str]:
    """Matches a criteria heading as people write it: ``## Done when``, ``**Done when:**``, ``Done when:``."""
    names = "|".join(r"\s+".join(re.escape(word) for word in h.split()) for h in headings)
    return re.compile(
        rf"^\s*(?:#{{1,6}}\s*(?:{names})\s*:?\s*#*|\*\*(?:{names})\s*:?\s*\*\*\s*:?|(?:{names})\s*:)\s*$",
        re.IGNORECASE,
    )


def render_description(
    story: Story,
    trace: tuple[str | None, list[str]] | None = None,
    heading: str = DEFAULT_AC_HEADINGS[0],
) -> str:
    parts: list[str] = []
    if story.content.strip():
        parts.append(story.content.strip())
    if story.acceptance_criteria:
        parts.append(f"## {heading}\n\n" + "\n".join(f"- [ ] {c}" for c in story.acceptance_criteria))
    footer = [f"{_FOOTER_MARK} `{story.id}`"]
    if story.complexity:
        footer.append(f"complexity: {story.complexity}")
    if trace and trace[0] and trace[1]:
        footer.append(f"spec `{trace[0]}`: {', '.join(trace[1])}")
    parts.append("---\n\n" + " · ".join(footer))
    return "\n\n".join(parts)


def parse_description(
    text: str, headings: tuple[str, ...] = DEFAULT_AC_HEADINGS,
) -> tuple[str, list[str]]:
    """Split an issue description into (story content, acceptance criteria).

    Inverse of ``render_description`` but tolerant of what people (and
    Linear's editor) do to the markdown: any heading level, a bold line or a
    plain "Done when:" line for the criteria heading (any of ``headings``),
    ``*``/``-`` bullets, ticked boxes, extra sections. Text in the criteria
    section that is not a list item stays part of the content.
    """
    lines = (text or "").replace("\r\n", "\n").split("\n")

    # Drop pralph's own footer (a rule followed by the "pralph story" line).
    for i in range(len(lines) - 1, -1, -1):
        if _HR_RE.match(lines[i]):
            if _FOOTER_MARK in "\n".join(lines[i + 1:]):
                lines = lines[:i]
            break

    heading_re = _ac_heading_re(headings)
    heading_at = next((i for i, line in enumerate(lines) if heading_re.match(line)), None)
    if heading_at is None:
        return "\n".join(lines).strip(), []

    section_end = next(
        (j for j in range(heading_at + 1, len(lines)) if _ANY_HEADING_RE.match(lines[j])),
        len(lines),
    )
    criteria: list[str] = []
    prose: list[str] = []
    in_item = False
    for line in lines[heading_at + 1:section_end]:
        m = _BULLET_RE.match(line)
        if m:
            criteria.append(m.group(1).strip())
            in_item = True
        elif not line.strip():
            in_item = False
        elif in_item and line[:1] in (" ", "\t"):
            criteria[-1] += " " + line.strip()  # wrapped list item
        else:
            prose.append(line)
            in_item = False
    content = "\n".join(lines[:heading_at] + prose + ([""] if prose else []) + lines[section_end:]).strip()
    return content, criteria


def epic_title(category: str) -> str:
    name = EPIC_NAMES.get(category)
    if category == _GENERAL:
        return f"{_GENERAL} — Uncategorized work"
    return f"{category} — {name}" if name else f"{category} — {category.title()} stories"


def category_from_epic_title(title: str) -> str:
    """Category code for a parent issue: pralph's 'AUTH — …' format, else its first word."""
    m = _EPIC_TITLE_RE.match(title.strip())
    if m:
        return m.group(1)
    words = re.findall(r"[A-Za-z0-9]+", title)
    return (words[0].upper()[:12] if words else _GENERAL)


def story_category(story: Story) -> str:
    return story.category.upper() if story.category else _GENERAL


# ── sync ─────────────────────────────────────────────────────────────


class LinearSync:
    def __init__(
        self,
        state: StateManager,
        client: LinearClient,
        link: LinearLink | None = None,
        *,
        echo: Callable[[str], None] = click.echo,
    ) -> None:
        self.state = state
        self.client = client
        self.link = link or LinearLink()
        self._echo = echo
        self._states: list[dict] | None = None
        self._ac_headings = ac_headings(state.get_config("linear_ac_heading", None))
        # Implemented stories whose review verdict is still out: story id → status info.
        self._awaiting_review: dict[str, dict] = {}

    @property
    def linked(self) -> bool:
        return bool(self.link.project_id and self.link.team_id)

    def _save_link(self) -> None:
        self.link.save(self.state.linear_link_path)

    def _warn(self, message: str) -> None:
        self._echo(click.style(f"  [linear] {message}", fg="yellow"))

    # -- workflow states --

    def _workflow_states(self) -> list[dict]:
        if self._states is None:
            self._states = self.client.workflow_states(self.link.team_id)
        return self._states

    def _state_for(self, cls: str, status: StoryStatus | None = None) -> dict:
        """The team's workflow state to use for a state class."""
        states = self._workflow_states()

        def of_type(*types: str) -> list[dict]:
            # A team can have several states of a type (In Progress, In Review):
            # take the one with Linear's standard name if there is one.
            return [
                st for t in types
                for st in sorted(
                    (st for st in states if st.get("type") == t),
                    key=lambda st: str(st.get("name", "")).strip().lower() != _STANDARD_STATE.get(t),
                )
            ]

        if cls == "open":
            candidates = of_type("unstarted", "backlog")
        elif cls == "canceled":
            # Only "canceled" states: Linear refuses to move an issue into a
            # state of type "duplicate" unless it has a duplicate-of relation.
            candidates = of_type("canceled")
            is_dup = [st for st in candidates if str(st.get("name", "")).lower() == "duplicate"]
            not_dup = [st for st in candidates if st not in is_dup]
            candidates = (is_dup + not_dup) if status == StoryStatus.duplicate else (not_dup + is_dup)
        else:
            candidates = of_type(cls)
        if not candidates:
            raise LinearError(f"Team {self.link.team_key} has no '{cls}' workflow state")
        return candidates[0]

    # -- setup: team / project --

    def _ensure_team(self, team_key: str | None) -> None:
        if self.link.team_id:
            if team_key and team_key.upper() != self.link.team_key.upper():
                raise LinearError(
                    f"This project is linked to Linear team {self.link.team_key}, not {team_key}"
                )
            return
        wanted = team_key or self.state.get_config("linear_team", None)
        teams = self.client.teams()
        if not teams:
            raise LinearError("No Linear teams are visible to this login")
        if wanted:
            match = [t for t in teams if str(t.get("key", "")).upper() == str(wanted).upper()
                     or str(t.get("name", "")).lower() == str(wanted).lower()]
            if not match:
                keys = ", ".join(sorted(t.get("key", "?") for t in teams))
                raise LinearError(f"Linear team {wanted!r} not found (available: {keys})")
            team = match[0]
        elif len(teams) == 1:
            team = teams[0]
        else:
            keys = ", ".join(sorted(t.get("key", "?") for t in teams))
            raise LinearError(f"Several Linear teams — choose one with --team (available: {keys})")
        self.link.team_id = team["id"]
        self.link.team_key = team.get("key", "")

    def default_project_name(self) -> str:
        """First H1 of the planning document, else the project directory name."""
        doc, _ = self._planning_doc()
        for line in doc.splitlines():
            m = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
            if m:
                title = re.sub(r"\s+[—–-]\s+(Specification|Design|Design Document)\s*$", "", m.group(1))
                title = re.sub(r"^(Specification|Design Document|Design)\s*[:—–-]\s+", "", title)
                if title.strip():
                    return title.strip()[:80]
        return self.state.project_dir.resolve().name

    def _planning_doc(self) -> tuple[str, str]:
        if self.state.spec_mode:
            content, sources = spec_lib.planning_context(self.state)
            if content:
                return content, sources
        return self.state.read_design_doc(), str(self.state.design_doc_path)

    def _ensure_project(self, project_name: str | None) -> bool:
        """Create the Linear project if this backlog isn't linked to one yet."""
        if self.link.project_id:
            return False
        name = (project_name or self.default_project_name()).strip()
        content, sources = self._planning_doc()
        if self.state.spec_mode and content:
            content = (
                f"> The specification lives in the repository ({sources}) and is the source of "
                f"truth. This is a snapshot taken when the project was created.\n\n{content}"
            )
        if len(content) > _PROJECT_CONTENT_LIMIT:
            content = content[:_PROJECT_CONTENT_LIMIT] + "\n\n_(truncated — see the repository for the full document)_"
        project = self.client.create_project(
            name=name,
            team_id=self.link.team_id,
            description=f"Plan for {self.state.project_dir.resolve().name}, executed by pralph."[:255],
            content=content,
        )
        self.link.project_id = project["id"]
        self.link.project_name = project.get("name", name)
        self.link.project_url = project.get("url", "")
        self._save_link()  # persist right away: a crash must not create a second project
        return True

    def link_project(self, reference: str, team_key: str | None = None) -> None:
        """Attach this backlog to an existing Linear project (URL, ID or exact name)."""
        reference = reference.strip()
        projects: list[dict]
        if "linear.app" in reference and _SLUG_ID_RE.search(reference):
            found = self.client.project(_SLUG_ID_RE.search(reference).group(1))  # type: ignore[union-attr]
            projects = [found] if found else []
        elif _UUID_RE.match(reference) or _PROJECT_IDENTIFIER_RE.match(reference):
            found = self.client.project(reference)
            projects = [found] if found else []
        else:
            projects = self.client.find_projects(reference)
        if not projects:
            raise LinearError(f"No Linear project found for {reference!r}")
        if len(projects) > 1:
            raise LinearError(f"{len(projects)} Linear projects match {reference!r} — pass the project URL instead")
        project = projects[0]

        teams = list(project.get("teams") or [])
        wanted = team_key or self.state.get_config("linear_team", None)
        if wanted:
            teams = [t for t in teams if str(t.get("key", "")).upper() == str(wanted).upper()]
            if not teams:
                raise LinearError(f"Project {project['name']!r} does not belong to team {wanted}")
        if len(teams) != 1:
            keys = ", ".join(t.get("key", "?") for t in teams) or "none"
            raise LinearError(f"Project {project['name']!r} has several teams ({keys}) — choose one with --team")

        self.link = LinearLink(
            team_id=teams[0]["id"],
            team_key=teams[0].get("key", ""),
            project_id=project["id"],
            project_name=project.get("name", ""),
            project_url=project.get("url", ""),
        )
        self._states = None
        self._save_link()

    def _require_link(self) -> None:
        if not self.linked:
            raise LinearError("Not linked to a Linear project — run 'pralph linear push' or 'pralph linear link'")

    # -- scope --

    def set_scope(
        self,
        *,
        milestone: str | None = None,
        assignee: str | None = None,
        label: str | None = None,
        clear: bool = False,
    ) -> None:
        """Restrict the link to a milestone, an assignee and/or a label of the project.

        Criteria that are not given stay as they are, unless ``clear``. An
        issue is in scope when it matches all of them.
        """
        self._require_link()
        scope = {} if clear else dict(self.link.scope)
        if milestone:
            scope["milestone"] = self._resolve_milestone(milestone)
        if assignee:
            scope["assignee"] = self._resolve_assignee(assignee)
        if label:
            scope["label"] = self._resolve_label(label)
        self.link.scope = scope
        self._save_link()

    def set_read_mostly(self, read_mostly: bool) -> None:
        self._require_link()
        self.link.read_mostly = bool(read_mostly)
        self._save_link()

    def _resolve_milestone(self, name: str) -> dict:
        milestones = self.client.project_milestones(self.link.project_id)
        wanted = name.strip().lower()
        match = [m for m in milestones if str(m.get("name", "")).strip().lower() == wanted]
        if not match:
            names = ", ".join(repr(m.get("name", "")) for m in milestones) or "none"
            raise LinearError(f"Project {self.link.project_name!r} has no milestone {name!r} (available: {names})")
        return {"id": match[0]["id"], "name": match[0].get("name", name)}

    def _resolve_assignee(self, reference: str) -> dict:
        reference = reference.strip()
        if reference.lower() == "me":
            viewer = self.client.viewer()
            if not viewer.get("id"):
                raise LinearError("Linear did not say who this login belongs to")
            return {"id": viewer["id"], "name": viewer.get("name") or "me"}
        users = self.client.find_users(reference)
        if not users:
            raise LinearError(f"No Linear user with the name or email {reference!r}")
        if len(users) > 1:
            names = ", ".join(str(u.get("email") or u.get("name")) for u in users)
            raise LinearError(f"Several Linear users match {reference!r} ({names}) — use the email address")
        return {"id": users[0]["id"], "name": users[0].get("name") or reference}

    def _resolve_label(self, name: str) -> dict:
        # A label of this team, or a workspace label. Issues list their labels
        # by name, so the name is what the scope goes by.
        labels = self.client.find_labels(name.strip(), self.link.team_id)
        if not labels:
            raise LinearError(f"No label {name!r} in team {self.link.team_key} or the workspace")
        return {"name": labels[0]["name"] or name.strip()}

    def scope_text(self) -> str:
        """The scope in words ('' when the whole project is in scope)."""
        scope = self.link.scope
        parts = []
        if scope.get("milestone"):
            parts.append(f"milestone {scope['milestone'].get('name', '?')!r}")
        if scope.get("assignee"):
            parts.append(f"assigned to {scope['assignee'].get('name', '?')}")
        if scope.get("label"):
            parts.append(f"label {scope['label'].get('name', '?')!r}")
        return ", ".join(parts)

    def _in_scope(self, issue: dict) -> bool:
        scope = self.link.scope
        if scope.get("milestone") and issue.get("milestone_id") != scope["milestone"].get("id"):
            return False
        if scope.get("assignee") and issue.get("assignee_id") != scope["assignee"].get("id"):
            return False
        if scope.get("label"):
            wanted = str(scope["label"].get("name", "")).strip().lower()
            if wanted not in {str(name).strip().lower() for name in issue.get("labels") or ()}:
                return False
        return True

    def _scope_fields(self) -> dict:
        """What puts an issue pralph creates into the scope."""
        scope = self.link.scope
        return {
            "milestone_id": (scope.get("milestone") or {}).get("id"),
            "assignee_id": (scope.get("assignee") or {}).get("id"),
            "labels": [scope["label"]["name"]] if (scope.get("label") or {}).get("name") else None,
        }

    def resolve_story_id(self, reference: str) -> str:
        """Story ID for a story ID or Linear identifier (ENG-123), as of the last sync."""
        stories = self.state.load_stories()
        story = next((s for s in stories if s.id == reference), None)
        if story is None:
            story = next(
                (s for s in stories
                 if (link := story_link(s)) and str(link.get("identifier", "")).upper() == reference.upper()),
                None,
            )
        if story is None:
            return reference
        if is_outside(story):
            where = f"the scope ({self.scope_text()})" if self.link.scope else "the Linear project"
            self._warn(f"{story.id} is outside {where} — its ticket will not be updated")
        return story.id

    # -- creating issues --

    def _ensure_epic(self, category: str) -> dict:
        epic = self.link.epics.get(category)
        if epic and epic.get("id"):
            return epic
        issue = self.client.create_issue(
            team_id=self.link.team_id,
            title=epic_title(category),
            description=(
                f"Parent issue for the `{category}` stories of this plan. "
                "pralph works through the sub-issues and closes this one when they are done."
            ),
            project_id=self.link.project_id,
            **self._scope_fields(),
        )
        epic = {
            "id": issue["id"], "identifier": issue["identifier"],
            "url": issue["url"], "state_type": issue["state_type"], "in_scope": True,
        }
        self.link.epics[category] = epic
        self._save_link()
        return epic

    def _remember(self, story_id: str, issue: dict, **extra) -> None:
        """Store the Linear link + last seen workflow state on the story."""
        def mutate(s: Story) -> None:
            link = s.metadata.get("linear") if isinstance(s.metadata.get("linear"), dict) else {}
            link.update({
                "id": issue["id"],
                "identifier": issue["identifier"] or link.get("identifier", ""),
                "url": issue["url"] or link.get("url", ""),
                "state": issue["state_name"],
                "state_type": issue["state_type"],
            })
            if issue.get("branch"):
                link["branch"] = issue["branch"]
            link.update(extra)
            s.metadata["linear"] = link

        # notify=False: this is bookkeeping, not a backlog change.
        self.state.update_story(story_id, mutate, notify=False)

    def _create_issue(self, story: Story) -> dict:
        epic = self._ensure_epic(story_category(story))
        cls = desired_class(story) or "open"
        specs = spec_lib.list_specs(self.state) if self.state.spec_mode else []
        issue = self.client.create_issue(
            team_id=self.link.team_id,
            title=issue_title(story),
            description=render_description(
                story, spec_lib.story_trace(story, specs) if specs else None, self._ac_headings[0],
            ),
            project_id=self.link.project_id,
            parent_id=epic["id"],
            priority=priority_to_linear(story.priority),
            # Open stories take the team's default state for new issues.
            state_id=None if cls == "open" else self._state_for(cls, story.status)["id"],
            **self._scope_fields(),
        )
        self._remember(story.id, issue)
        return issue

    def _create_relations(
        self,
        stories: list[Story],
        existing: dict[str, set[str]],
        touching: set[str] | None = None,
    ) -> int:
        """Create missing "blocks" relations for story dependencies.

        ``existing`` maps issue ID → IDs of issues already blocking it.
        ``touching`` restricts the work to dependencies that involve one of
        those stories, on either side — a new ticket can be blocked by older
        ones and can itself be what an older one was waiting for.

        A relation Linear didn't take is remembered on the story
        (``metadata.linear.pending_deps``): until it exists there, the next
        pull must not read its absence as "removed in Linear", and tries again.
        """
        if self.link.read_mostly:
            return 0  # relations are the project owners' to draw
        # Not for tickets outside the project or scope: a relation writes to both ends.
        links = {s.id: None if is_outside(s) else story_link(s) for s in stories}
        created = 0
        pending_changes: dict[str, list[str]] = {}
        for story in stories:
            link = links.get(story.id)
            if not link:
                continue
            have = existing.setdefault(link["id"], set())
            for dep in story.dependencies:
                if touching is not None and story.id not in touching and dep not in touching:
                    continue
                blocker = links.get(dep)
                if not blocker or blocker["id"] in have or blocker["id"] == link["id"]:
                    continue
                pending = list(pending_changes.get(story.id, link.get("pending_deps") or []))
                try:
                    self.client.create_blocks_relation(blocker["id"], link["id"])
                    have.add(blocker["id"])
                    created += 1
                    if dep in pending:
                        pending.remove(dep)
                        pending_changes[story.id] = pending
                except LinearError as e:
                    self._warn(f"could not link {dep} → {story.id}: {e}")
                    if dep not in pending:
                        pending_changes[story.id] = pending + [dep]

        for story_id, pending in pending_changes.items():
            def mutate(s: Story, pending: list[str] = pending) -> None:
                link = story_link(s)
                if link is None:
                    return
                if pending:
                    link["pending_deps"] = pending
                else:
                    link.pop("pending_deps", None)

            self.state.update_story(story_id, mutate, notify=False)
        return created

    def unpushed(self) -> list[Story]:
        """Stories of the plan that have no Linear issue yet (and should get one)."""
        if self.link.read_mostly:
            return []  # they stay local: a read-mostly link creates nothing
        return [
            s for s in self.state.load_stories()
            if s.status not in _NOT_IN_PLAN and not story_link(s)
        ]

    # -- push: plan → Linear --

    def push(
        self,
        *,
        team_key: str | None = None,
        project_name: str | None = None,
        dry_run: bool = False,
    ) -> PushReport:
        """Create everything in Linear that the local plan has and Linear doesn't. Idempotent."""
        if self.link.read_mostly:
            raise LinearError(
                "This Linear link is read-mostly: pralph does not create issues. "
                "Run 'pralph linear scope --read-write' to let it."
            )
        plan = [s for s in self.state.load_stories() if s.status not in _NOT_IN_PLAN]
        todo = self.unpushed()
        report = PushReport(dry_run=dry_run, already_linked=len(plan) - len(todo))

        if dry_run:
            report.project_name = self.link.project_name or project_name or self.default_project_name()
            report.project_url = self.link.project_url
            report.project_created = not self.link.project_id
            categories = {story_category(s) for s in todo}
            report.epics_created = sorted(c for c in categories if c not in self.link.epics)
            report.issues_created = [(s.id, "") for s in todo]
            in_plan = {s.id for s in plan}
            report.relations_created = sum(
                1 for s in todo for dep in s.dependencies if dep in in_plan and dep != s.id
            )
            return report

        self._ensure_team(team_key)
        report.project_created = self._ensure_project(project_name)
        report.project_name = self.link.project_name
        report.project_url = self.link.project_url

        # What already exists (only possible if the project did): relations,
        # and tickets a previous push created without getting to record them.
        existing: dict[str, set[str]] = {}
        adopted: set[str] = set()
        if not report.project_created:
            remote = self.client.project_issues(self.link.project_id)
            adopted = self._adopt_existing(todo, remote)
            report.already_linked += len(adopted)
            # Relations are only drawn towards stories that have dependencies.
            now = self.state.load_stories()
            blocked = {link["id"] for s in now if s.dependencies and (link := story_link(s))}
            self._load_blockers(remote, [i for i in remote if i["id"] in blocked])
            existing = self._existing_relations(remote, {link["id"] for s in now if (link := story_link(s))})

        epics_before = set(self.link.epics)
        for story in todo:
            if story.id in adopted:
                continue
            issue = self._create_issue(story)
            report.issues_created.append((story.id, issue["identifier"]))
        report.epics_created = sorted(set(self.link.epics) - epics_before)

        stories = self.state.load_stories()
        report.relations_created = self._create_relations(
            [s for s in stories if s.status not in _NOT_IN_PLAN], existing,
        )
        self._rollup_epics(stories)
        return report

    def _load_blockers(self, issues: list[dict], wanted: list[dict]) -> None:
        """Read the blocking relations of ``wanted`` (a listing does not include them).

        One call per issue, so callers ask only for the ones they need.
        Blockers outside ``issues`` (another project) are left out. An issue
        whose relations could not be read keeps ``relations_truncated``, which
        makes the sync leave its dependencies as they are.
        """
        by_identifier = {i["identifier"]: i["id"] for i in issues}
        for issue in wanted:
            try:
                blockers = self.client.issue_blockers(issue["id"])
            except LinearError as e:
                self._warn(f"could not read the relations of {issue['identifier']}: {e}")
                continue
            issue["blocked_by"] = [by_identifier[b] for b in blockers if b in by_identifier]
            issue["relations_truncated"] = False

    @staticmethod
    def _existing_relations(issues: list[dict], linked_ids: set[str]) -> dict[str, set[str]]:
        """Issue ID → IDs of the issues known to block it, for ``_create_relations``.

        Where the relations are not known, every relation counts as existing:
        better to miss creating one (it is retried) than to draw it twice.
        """
        return {
            i["id"]: (set(linked_ids) if i["relations_truncated"] else set(i["blocked_by"])) for i in issues
        }

    def _adopt_existing(self, todo: list[Story], remote: list[dict]) -> set[str]:
        """Link stories and categories to tickets that already exist for them.

        A create can succeed in Linear without pralph learning about it (the
        response timed out, the process died). Its ticket carries the story ID
        in its title, so the next push picks it up instead of creating a twin.
        Returns the IDs of the stories that got linked.
        """
        taken = {link["id"] for s in self.state.load_stories() if (link := story_link(s))}
        taken |= {e.get("id") for e in self.link.epics.values()}
        parent_ids = {i["parent_id"] for i in remote if i["parent_id"]}
        wanted = {s.id: s for s in todo}
        missing_epics = {
            epic_title(c): c for c in {story_category(s) for s in todo} if c not in self.link.epics
        }

        adopted: set[str] = set()
        for issue in remote:
            if issue["id"] in taken:
                continue
            category = missing_epics.pop(issue["title"].strip(), None) if not issue["parent_id"] else None
            if category is not None:
                self.link.epics[category] = {
                    "id": issue["id"], "identifier": issue["identifier"],
                    "url": issue["url"], "state_type": issue["state_type"],
                }
                self._save_link()
                continue
            title_id, _ = split_title(issue["title"])
            if title_id in wanted and title_id not in adopted and issue["id"] not in parent_ids:
                self._remember(title_id, issue)
                adopted.add(title_id)
        return adopted

    # -- pull: Linear → backlog --

    def pull(self) -> PullReport:
        """Bring the local backlog in line with the Linear project."""
        self._require_link()
        issues = self.client.project_issues(self.link.project_id)
        report = PullReport()

        # Parent issues are containers, not work. So are pralph's own epics,
        # even while they have no children.
        parent_ids = {i["parent_id"] for i in issues if i["parent_id"]}
        epic_ids = {e.get("id") for e in self.link.epics.values()}
        containers = [i for i in issues if i["id"] in parent_ids or i["id"] in epic_ids]
        leaves = [i for i in issues if i["id"] not in parent_ids and i["id"] not in epic_ids]
        # Only what is in scope is work. The rest belongs to other people or
        # other repositories: no story, and never a write.
        work = [i for i in leaves if self._in_scope(i)]
        outside = {i["id"] for i in leaves} - {i["id"] for i in work}
        report.total = len(work)
        report.out_of_scope = len(outside)
        if outside and not work:
            self._warn(f"no issue of the project matches the scope ({self.scope_text()})")
        container_category = self._refresh_epics(containers, [i for i in leaves if i["id"] in outside])

        stories = self.state.load_stories()
        by_issue = {story_link(s)["id"]: s for s in stories if story_link(s)}  # type: ignore[index]
        by_id = {s.id: s for s in stories}

        resolved: list[tuple[dict, Story, bool]] = []
        for issue in work:
            story = by_issue.get(issue["id"])
            is_new = False
            if story is None:
                title_id, _ = split_title(issue["title"])
                candidate = by_id.get(title_id) if title_id else None
                if candidate is not None and not story_link(candidate):
                    story = candidate  # same story, pushed from another checkout
                else:
                    new_id = title_id if title_id and title_id not in by_id else issue["identifier"]
                    while new_id in by_id:
                        new_id += "b"
                    story = Story(id=new_id, title="", content="", source="linear")
                    stories.append(story)
                    by_id[new_id] = story
                    is_new = True
            resolved.append((issue, story, is_new))

        self._load_blockers(issues, [issue for issue, story, _ in resolved if _relations_stale(issue, story)])

        entries: list[dict] = []

        # A story ticket that somebody broke down into sub-issues is now a
        # container: its sub-issues are the work, and people own its state.
        for issue in containers:
            story = by_issue.get(issue["id"])
            if story is None:
                continue
            link = story_link(story) or {}
            link.update({"container": True, "state": issue["state_name"], "state_type": issue["state_type"]})
            link.pop("claimed", None)
            if story.status in _ACTIONABLE:
                reason = "Split into sub-issues in Linear"
                entries.append({
                    "story_id": story.id, "status": "skipped", "summary": reason, "source": "linear",
                })
                report.status_changes.append((story.id, story.status.value, "skipped", reason))
                story.status = StoryStatus.skipped

        story_for_issue = {issue["id"]: story for issue, story, _ in resolved}
        synced_ids = {story.id for _, story, _ in resolved}
        # Dependencies on stories that aren't in Linear can't be expressed there — keep them.
        local_only = {s.id for s in stories if s.id not in synced_ids and not story_link(s)}

        to_push: list[Story] = []
        for issue, story, is_new in resolved:
            before = _content(story)
            old_status = story.status
            reason = self._apply_issue(
                issue, story, is_new, story_for_issue, container_category, local_only, to_push,
            )
            if story.status != old_status:
                entries.append({
                    "story_id": story.id, "status": story.status.value,
                    "summary": reason, "source": "linear",
                })
                report.status_changes.append((story.id, old_status.value, story.status.value, reason))
            if is_new:
                report.new.append(story)
            elif _content(story) != before:
                report.updated += 1

        # Linked stories whose issue is gone (deleted, archived, moved to another
        # project) or outside the scope (reassigned, other milestone, label removed).
        # Not this checkout's work any more, and not pralph's ticket to write to.
        ours = {i["id"] for i in work} | {i["id"] for i in containers}
        for story in stories:
            link = story_link(story)
            if not link or link["id"] in ours:
                continue
            link["outside"] = True
            if story.status in _ACTIONABLE:
                reason = (
                    f"Issue is outside the Linear scope ({self.scope_text()})" if link["id"] in outside
                    else "Issue is no longer in the Linear project"
                )
                entries.append({
                    "story_id": story.id, "status": "skipped", "summary": reason, "source": "linear",
                })
                report.status_changes.append((story.id, story.status.value, "skipped", reason))
                story.status = StoryStatus.skipped
                link["state_type"] = "canceled"  # if it comes back open, that reads as a reopen
                link.pop("claimed", None)

        self.state.replace_stories(stories, entries)

        # Dependencies whose relation never made it to Linear (see _create_relations).
        # Where the relations were not read we can't tell what exists — leave those be.
        linked_ids = {link["id"] for s in stories if (link := story_link(s))}
        report.relations_created = self._create_relations(
            [s for s in stories if s.status not in _NOT_IN_PLAN], self._existing_relations(issues, linked_ids),
        )

        # Linear didn't move these, but the local status did (failed push, viewer edit).
        for story in to_push:
            try:
                report.pushed += self._push_state(story)
            except LinearError as e:
                self._warn(f"could not update {story.id} in Linear: {e}")

        self._rollup_epics(self.state.load_stories())
        return report

    def _refresh_epics(self, containers: list[dict], outside: list[dict]) -> dict[str, str]:
        """Learn about parent issues; returns {issue id: category}.

        ``outside``: the issues the scope leaves out. A parent that still has
        open sub-issues among them is not pralph's to close.
        """
        by_epic_id = {e.get("id"): cat for cat, e in self.link.epics.items()}
        open_outside: dict[str, int] = {}
        for issue in outside:
            if issue["parent_id"] and state_class(issue["state_type"]) in ("open", "started"):
                open_outside[issue["parent_id"]] = open_outside.get(issue["parent_id"], 0) + 1
        mapping: dict[str, str] = {}
        changed = False
        for issue in containers:
            category = by_epic_id.get(issue["id"])
            if category is None:
                category = category_from_epic_title(issue["title"])
                if category not in self.link.epics:
                    self.link.epics[category] = {
                        "id": issue["id"], "identifier": issue["identifier"],
                        "url": issue["url"], "state_type": issue["state_type"],
                    }
                    changed = True
            epic = self.link.epics.get(category, {})
            if epic.get("id") == issue["id"]:
                seen = {
                    "state_type": issue["state_type"],
                    "in_scope": self._in_scope(issue),
                    "open_outside": open_outside.get(issue["id"], 0),
                }
                if any(epic.get(k) != v for k, v in seen.items()):
                    epic.update(seen)
                    changed = True
            mapping[issue["id"]] = category
        if changed:
            self._save_link()
        return mapping

    def _apply_issue(
        self,
        issue: dict,
        story: Story,
        is_new: bool,
        story_for_issue: dict[str, Story],
        container_category: dict[str, str],
        local_only: set[str],
        to_push: list[Story],
    ) -> str:
        """Copy an issue onto its story. Returns the reason if the status changed."""
        title_id, bare_title = split_title(issue["title"])
        story.title = bare_title if title_id == story.id else issue["title"].strip()
        story.content, story.acceptance_criteria = parse_description(issue["description"], self._ac_headings)
        story.priority = priority_from_linear(int(issue["priority"] or 0), story.priority)

        if is_new or not story.category:
            category = container_category.get(issue["parent_id"] or "")
            if not category and "-" in story.id:
                category = story.id.rsplit("-", 1)[0]
            story.category = category or self.link.team_key or _GENERAL

        link = story_link(story) or {}
        blockers = [
            story_for_issue[b].id for b in issue["blocked_by"]
            if b in story_for_issue and story_for_issue[b] is not story
        ]
        # A dependency Linear doesn't list was removed there — unless Linear
        # never had it: the other story isn't a ticket, the relation is still
        # waiting to be created, or the list we got is incomplete.
        waiting = link.get("pending_deps") or []
        kept = [
            d for d in story.dependencies
            if d not in blockers and (d in local_only or d in waiting or issue["relations_truncated"])
        ]
        story.dependencies = list(dict.fromkeys(blockers + kept))
        pending_deps = [d for d in waiting if d in kept]

        known = link.get("state_type")
        remote_cls = state_class(issue["state_type"])
        claimed = bool(link.get("claimed"))
        reason = ""

        if is_new or not known or state_class(known) != remote_cls:
            # The ticket moved in Linear (or we see it for the first time): Linear wins.
            new_status, reason, claimed = _status_from_linear(
                remote_cls, story.status, claimed, duplicate=issue["state_type"] == "duplicate",
                on_branch=_on_branch(story),
            )
            if new_status != story.status:
                if new_status == StoryStatus.pending:
                    for key in ("error_reason", "error_output", "error_at"):
                        story.metadata.pop(key, None)
                story.status = new_status
        else:
            wanted = status_class(story) if not claimed else None
            if wanted is not None and wanted != remote_cls:
                to_push.append(story)

        new_link = {
            "id": issue["id"], "identifier": issue["identifier"], "url": issue["url"],
            "state": issue["state_name"], "state_type": issue["state_type"],
        }
        if issue.get("branch") or link.get("branch"):
            # Linear's name for the issue's branch: a pull request from it is linked to the issue.
            new_link["branch"] = issue.get("branch") or link["branch"]
        if claimed:
            new_link["claimed"] = True
        if pending_deps:
            new_link["pending_deps"] = pending_deps
        # Which version of the ticket the dependencies were read from (see _relations_stale).
        relations_at = link.get("relations_at") if issue["relations_truncated"] else issue["updated_at"]
        if relations_at:
            new_link["relations_at"] = relations_at
        story.metadata["linear"] = new_link
        return reason

    # -- push: status changes → Linear --

    def _push_state(self, story: Story) -> bool:
        """Move the story's issue to the workflow state its status calls for."""
        link = story_link(story)
        if not link:
            return False
        cls = desired_class(story)
        if cls is None or state_class(link.get("state_type")) == cls:
            return False
        if self.link.read_mostly and cls != "started":
            # Read-mostly: starting work is the one transition pralph makes.
            # Done, Canceled and back to Todo are for people (or the merge).
            return False
        target = self._state_for(cls, story.status)
        issue = self.client.update_issue(link["id"], state_id=target["id"])
        self._remember(story.id, issue)
        return True

    def on_story_event(self, event: str, story: Story | None, info: dict) -> None:
        """Story listener: mirror backlog changes to Linear."""
        if story is None or not self.linked:
            return
        try:
            if event == "added":
                if not story_link(story) and story.status not in _NOT_IN_PLAN and not self.link.read_mostly:
                    self._create_ticket(story)
            elif event == "status":
                self._on_status(story, info)
            elif event == "reviewed":
                held = self._awaiting_review.pop(story.id, None)
                if held is not None and story.status == StoryStatus.implemented:
                    self._sync_ticket(story, {**held, "review": info})
        except LinearError as e:
            self._warn(f"{story.id} not synced ({e}) — will be retried on the next pull")

    def _create_ticket(self, story: Story) -> None:
        """Ticket for a story that appeared after the push, with its dependencies."""
        self._create_issue(story)
        # The ticket is new, so none of its relations exist yet.
        self._create_relations(self.state.load_stories(), {}, touching={story.id})

    def _on_status(self, story: Story, info: dict) -> None:
        if not story_link(story):
            if story.status not in _NOT_IN_PLAN and not self.link.read_mostly:
                self._create_ticket(story)
            return
        if story.status == StoryStatus.implemented and info.get("review_pending"):
            # "Done" in Linear should mean reviewed. Hold the transition until
            # the verdict: `reviewed` completes it, `rework` replaces it. If the
            # run dies in between, the next pull pushes the status.
            self._awaiting_review[story.id] = info
            return
        self._awaiting_review.pop(story.id, None)
        self._sync_ticket(story, info)

    def _sync_ticket(self, story: Story, info: dict) -> None:
        """Move the ticket to where the story's status puts it, and say why."""
        if is_outside(story):
            return  # left the project or the scope: not pralph's ticket any more
        moved = self._push_state(story)
        # Say what happened to the ticket only if it did: "closed" would be
        # wrong on a ticket a read-mostly link (or a claim) left open.
        # (`story` is the listener's copy: its link still has the state from before the move.)
        now = desired_class(story) if moved else state_class((story_link(story) or {}).get("state_type"))
        body = _comment_for(self.state, story, info, closed=now in ("completed", "canceled"))
        if body:
            self.client.create_comment(story_link(story)["id"], body)  # type: ignore[index]
        self._rollup_epics(self.state.load_stories(), only=story_category(story))

    def _rollup_epics(self, stories: list[Story], only: str | None = None) -> None:
        """Start a parent issue when work on its stories starts; close it when all are done."""
        if self.link.read_mostly:
            return  # parent issues are run by people
        for category, epic in list(self.link.epics.items()):
            if only is not None and category != only:
                continue
            if self.link.scope and not epic.get("in_scope"):
                continue  # a parent outside the scope is somebody else's
            classes = [
                state_class(story_link(s)["state_type"])  # type: ignore[index]
                for s in stories if story_category(s) == category and story_link(s) and not is_outside(s)
            ]
            if not classes:
                continue
            done = all(c in ("completed", "canceled") for c in classes) and "completed" in classes
            if done and epic.get("open_outside"):
                # Our part is done, but sub-issues outside the scope are still
                # open: not ours to close — and if people closed it, not ours to reopen.
                if state_class(epic.get("state_type")) != "open":
                    continue
                target = "started"
            elif done:
                target = "completed"
            elif any(c in ("started", "completed") for c in classes):
                target = "started"
            else:
                continue  # never push a parent back to open
            if state_class(epic.get("state_type")) == target:
                continue
            try:
                issue = self.client.update_issue(epic["id"], state_id=self._state_for(target)["id"])
                epic["state_type"] = issue["state_type"]
                self._save_link()
            except LinearError as e:
                self._warn(f"could not update parent issue {epic.get('identifier', category)}: {e}")


def _content(story: Story) -> dict:
    """The story without sync bookkeeping — for telling whether a pull changed it."""
    data = story.to_dict()
    link = (data.get("metadata") or {}).get("linear")
    if isinstance(link, dict):
        data["metadata"] = {**data["metadata"], "linear": {k: v for k, v in link.items() if k != "relations_at"}}
    return data


def _relations_stale(issue: dict, story: Story) -> bool:
    """Do this ticket's blocking relations have to be read (again)?

    Reading them costs a call per ticket, so it is done only when they can
    have changed: Linear touches a ticket's ``updatedAt`` when a blocker is
    added or removed, and the story remembers the ``updatedAt`` its
    dependencies were read at. A dependency that still has to be created in
    Linear always needs the current list; a closed ticket's are not read.
    """
    link = story_link(story) or {}
    if link.get("pending_deps"):
        return True
    if state_class(issue["state_type"]) in ("completed", "canceled"):
        return False  # nothing waits on the blockers of a closed ticket
    return not issue["updated_at"] or link.get("relations_at") != issue["updated_at"]


def _status_from_linear(
    remote_cls: str, current: StoryStatus, claimed: bool, *, duplicate: bool = False, on_branch: bool = False,
) -> tuple[StoryStatus, str, bool]:
    """Local status for a ticket that moved in Linear → (status, reason, claimed elsewhere).

    ``duplicate``: the ticket was closed in Linear as a duplicate of another issue.
    ``on_branch``: the story's work is on a branch of its own.
    """
    if remote_cls == "completed":
        if current in (StoryStatus.shipped, StoryStatus.merged) or on_branch:
            # Done in Linear is what the merge of the pull request leaves behind.
            return StoryStatus.merged, "" if current == StoryStatus.merged else "Completed in Linear (merged)", False
        return StoryStatus.implemented, "Completed in Linear", False
    if remote_cls == "canceled":
        if current in (StoryStatus.skipped, StoryStatus.duplicate) or (current == StoryStatus.external and not claimed):
            return current, "", False
        if duplicate:
            return StoryStatus.duplicate, "Marked as duplicate in Linear", False
        return StoryStatus.skipped, "Canceled in Linear", False
    if remote_cls == "started":
        if current in (StoryStatus.in_progress, StoryStatus.rework, StoryStatus.error,
                       StoryStatus.shipped, StoryStatus.needs_human) or (
                current == StoryStatus.implemented and on_branch):
            return current, "", False
        # Someone took the ticket in Linear. Hands off until they finish or put it back.
        return StoryStatus.external, "In progress in Linear (picked up outside pralph)", True
    # open
    if current == StoryStatus.rework:
        return current, "", False  # still needs the rework; keep the review feedback
    return StoryStatus.pending, "" if current == StoryStatus.pending else "Reopened in Linear", False


def _comment_for(state: StateManager, story: Story, info: dict, *, closed: bool = True) -> str:
    """Comment to leave on the ticket for a status change ('' for none).

    ``closed``: the ticket is completed/canceled in Linear. When pralph does
    not close tickets (read-mostly), the comment must not claim it did.
    """
    summary = (info.get("summary") or "").strip()
    extra = info.get("extra") or {}
    reason = (info.get("error_reason") or extra.get("reason") or summary or "").strip()
    status = story.status

    git = story.metadata.get("git") if isinstance(story.metadata.get("git"), dict) else {}
    if status == StoryStatus.shipped:
        url = extra.get("pr_url") or git.get("pr_url") or ""
        if extra.get("pr_updated"):
            return f"**pralph** pushed the requested changes to the pull request: {url}"
        return f"**pralph** opened a pull request: {url}"
    if status == StoryStatus.merged:
        return "**pralph** saw this merged" + (f": {git['pr_url']}" if git.get("pr_url") else ".")
    if status == StoryStatus.needs_human:
        return (
            "**pralph** paused this story — it needs a person.\n\n"
            f"{reason[:1500] or 'No reason captured.'}\n\n"
            f"When that is done, run `pralph unblock {story.id}` (or move the issue back to Todo) "
            "and pralph continues the story."
        )
    if status == StoryStatus.implemented:
        lines = ["**pralph** implemented this story."]
        if git.get("branch"):
            lines[0] += f" It is on the branch `{git['branch']}`, not merged yet."
        if summary:
            lines += ["", summary]
        files = [f for f in extra.get("files_changed") or [] if isinstance(f, str)]
        if files:
            lines += ["", "Files changed:"] + [f"- `{f}`" for f in files[:40]]
        review = info.get("review")
        if isinstance(review, dict):
            if review.get("error"):
                lines += ["", f"Not reviewed: the reviewer failed ({str(review['error'])[:200]})."]
            else:
                feedback = str(review.get("feedback") or "").strip()
                lines += ["", "Review: approved" + (f" — {feedback[:500]}" if feedback else "") ]
        return "\n".join(lines)
    if status == StoryStatus.rework:
        if extra.get("pr_url"):
            # The feedback is on the pull request already; no need to repeat it here.
            return f"**pralph** will rework this story: changes were requested on {extra['pr_url']}"
        feedback = state.read_review_feedback(story.id)
        return "**pralph** review requested changes — the story will be reworked." + (
            f"\n\n{feedback}" if feedback else ""
        )
    if status == StoryStatus.error:
        return (
            "**pralph** could not implement this story.\n\n"
            f"{reason[:1500] or 'No reason captured.'}\n\n"
            "Move the issue back to Todo (or run `pralph reset-errors`) to retry."
        )
    if status == StoryStatus.duplicate:
        of = f" of {extra['duplicate_of']}" if extra.get("duplicate_of") else ""
        head = (
            f"**pralph** closed this as a duplicate{of}." if closed
            else f"**pralph** considers this a duplicate{of} and will not work on it. The issue was left open."
        )
        return head + (f"\n\n{reason}" if reason else "")
    if status == StoryStatus.external:
        system = extra.get("external_system")
        return (
            f"**pralph** cannot do this in code — it needs work in {system or 'an external system'}."
            + (f"\n\n{reason}" if reason else "")
        )
    if status == StoryStatus.skipped:
        head = "**pralph** skipped this story." if closed else (
            "**pralph** skipped this story and will not work on it. The issue was left open."
        )
        return head + (f"\n\n{reason}" if reason else "")
    return ""  # pending / in_progress: the state change says it all


# ── wiring ───────────────────────────────────────────────────────────


def attach(
    state: StateManager,
    *,
    required: bool = False,
    echo: Callable[[str], None] = click.echo,
) -> LinearSync | None:
    """Hook Linear sync into the backlog if this project is linked.

    With ``required`` a missing link or login is an error; otherwise the
    project simply runs without Linear (with a warning if it is linked but
    nobody is logged in — see ``pralph linear oauth``).
    """
    link = LinearLink.load(state.linear_link_path)
    if link is None:
        if required:
            raise LinearError("This project is not linked to Linear — run 'pralph linear push' first")
        return None
    oauth = linear_oauth.stored_login(state.home_dir)
    if oauth is None:
        if required:
            raise LinearError(
                "Not logged in to Linear — run 'pralph linear oauth' "
                "to work off the linked Linear project"
            )
        echo(click.style(
            "  [linear] project is linked to Linear but there is no Linear login — "
            "run 'pralph linear oauth'; changes will not be synced",
            fg="yellow",
        ))
        return None
    client = LinearClient(oauth=oauth)
    sync = LinearSync(state, client, link, echo=echo)
    state.add_story_listener(sync.on_story_event)
    return sync
