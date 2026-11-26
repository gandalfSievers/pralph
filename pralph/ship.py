"""Stories on their own branch: branches, worktrees, the ship step, waiting for the merge.

Off by default — pralph then commits to whatever branch is checked out, as it
always did. Any of these switches it on:

- ``.pralph/ship.md``         a prompt pralph runs once a story passed its review,
                              to push the branch and open a pull request
- ``worktree_dir`` (config)   every story's branch gets its own worktree there
- ``branch_per_story`` (config)  just the branch, in the checkout itself

A story then starts on a new branch from the up-to-date base branch, named
after the Linear issue (``gitBranchName``) so that Linear's GitHub integration
follows the pull request. After the review the ship step opens the pull
request and the story is ``shipped``; pralph itself never pushes. A story
counts as done for its dependents once it is ``merged`` — pralph watches the
pull request (``gh``) for that, turns requested changes into rework on the
same branch, and gives up on a pull request that was closed.

What a story's ``metadata["git"]`` holds: ``branch``, ``base``, ``start`` (the
commit a new branch started from), ``worktree`` (if it has one), ``pr_url``,
``pr_feedback_at`` (the newest review comment already turned into rework) and
``ship_error`` / ``ship_error_at`` (why the last ship step opened no pull request).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import click

from pralph.models import Story, StoryStatus
from pralph.state import StateManager

DEFAULT_SHIP_TOOLS = "Read,Glob,Grep,Bash"
DEFAULT_SHIP_TIMEOUT = 3600   # a ship step may run its own review rounds
DEFAULT_POLL_INTERVAL = 300   # seconds between looks at the open pull requests

# Dependencies that will never bring code: nothing to wait for.
_NO_CODE = (StoryStatus.skipped, StoryStatus.duplicate, StoryStatus.external)
_UNFINISHED = (StoryStatus.pending, StoryStatus.in_progress, StoryStatus.rework, StoryStatus.error)

PR_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+/(?:pull|merge_requests)/\d+")
_GITHUB_PR_RE = re.compile(r"^https?://([^/]+)/([^/]+)/([^/]+)/pull/(\d+)")


class GitError(Exception):
    """A git command pralph needed failed; the message says what to do about it."""


# ── settings ─────────────────────────────────────────────────────────


@dataclass
class Settings:
    branches: bool = False
    base: str = ""                    # "" = the remote's default branch
    worktree_dir: Path | None = None
    ship_prompt: str = ""             # "" = no ship step: pralph never pushes
    ship_tools: str = DEFAULT_SHIP_TOOLS
    ship_timeout: int = DEFAULT_SHIP_TIMEOUT
    poll_interval: int = DEFAULT_POLL_INTERVAL


def ship_prompt_path(state: StateManager) -> Path:
    return state.state_dir / "ship.md"


def ship_tools_path(state: StateManager) -> Path:
    return state.state_dir / "ship-tools.txt"


def load_settings(state: StateManager) -> Settings:
    prompt = ""
    if ship_prompt_path(state).exists():
        prompt = ship_prompt_path(state).read_text().strip()
    tools = DEFAULT_SHIP_TOOLS
    if ship_tools_path(state).exists():
        raw = ship_tools_path(state).read_text()
        listed = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip() and not t.strip().startswith("#")]
        tools = ",".join(listed) or DEFAULT_SHIP_TOOLS

    worktree_dir = None
    configured = str(state.get_config("worktree_dir", "") or "").strip()
    if configured:
        worktree_dir = Path(configured).expanduser()
        if not worktree_dir.is_absolute():
            worktree_dir = state.project_dir / worktree_dir
        worktree_dir = worktree_dir.resolve()

    return Settings(
        branches=bool(prompt or worktree_dir or state.get_config("branch_per_story", False)),
        base=str(state.get_config("base_branch", "") or "").strip(),
        worktree_dir=worktree_dir,
        ship_prompt=prompt,
        ship_tools=tools,
        ship_timeout=_positive(state.get_config("ship_timeout", None), DEFAULT_SHIP_TIMEOUT),
        poll_interval=_positive(state.get_config("merge_poll_interval", None), DEFAULT_POLL_INTERVAL),
    )


def _positive(value, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


# ── stories ──────────────────────────────────────────────────────────


def git_info(story: Story) -> dict:
    info = story.metadata.get("git")
    return info if isinstance(info, dict) else {}


def on_branch(story: Story) -> bool:
    """The story's work lives on a branch of its own (and is not in the base branch yet)."""
    return bool(git_info(story).get("branch"))


def branch_name(story: Story) -> str:
    """Linear's branch name for the issue, so its GitHub integration finds the pull request."""
    link = story.metadata.get("linear")
    if isinstance(link, dict) and str(link.get("branch") or "").strip():
        return str(link["branch"]).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", f"{story.id} {story.title}".lower()).strip("-")
    return slug[:60].rstrip("-") or "story"


def story_workdir(state: StateManager, story_id: str) -> Path:
    """Where a story's code is: its worktree if it has one, else the checkout."""
    story = next((s for s in state.load_stories() if s.id == story_id), None)
    path = git_info(story).get("worktree") if story is not None else None
    if path and Path(path).is_dir():
        return Path(path)
    return state.project_dir


def is_waiting(story: Story) -> bool:
    """Nothing for pralph to do on it right now, but it is not finished either."""
    if story.status in (StoryStatus.shipped, StoryStatus.needs_human):
        return True
    return story.status == StoryStatus.implemented and on_branch(story)


def waiting_reason(story: Story) -> str:
    info = git_info(story)
    if story.status == StoryStatus.needs_human:
        reason = (story.metadata.get("needs_human") or {}).get("reason") or "see the story"
        return f"needs a person: {reason}"
    if story.status == StoryStatus.shipped:
        return f"pull request open: {info.get('pr_url') or info.get('branch')}"
    return f"on branch {info.get('branch')}, not merged yet"


# ── processes ────────────────────────────────────────────────────────


def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def git(cwd: Path, *args: str, timeout: int = 120) -> str:
    try:
        proc = _run(["git", *args], cwd, timeout)
    except FileNotFoundError as e:
        raise GitError("git is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {args[0]} timed out") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise GitError(f"git {' '.join(args[:2])} failed: {detail[-1] if detail else proc.returncode}")
    return proc.stdout.strip()


def git_ok(cwd: Path, *args: str) -> bool:
    try:
        git(cwd, *args)
    except GitError:
        return False
    return True


def find_pr_url(text: str) -> str:
    """The pull request a ship step reported: the last such URL in its answer."""
    found = PR_URL_RE.findall(text or "")
    return found[-1].rstrip(".,;:") if found else ""


# ── the flow ─────────────────────────────────────────────────────────


class Flow:
    """Branch, worktree and pull request handling for one run."""

    def __init__(
        self,
        state: StateManager,
        *,
        echo: Callable[[str], None] = click.echo,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.state = state
        self.settings = load_settings(state)
        self.root = state.project_dir
        self._echo = echo
        self._sleep = sleep
        self._base: str | None = None
        self._original_branch: str | None = None
        self._fetched_at = 0.0
        self._warned: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.settings.branches

    @property
    def ships(self) -> bool:
        return bool(self.settings.ship_prompt)

    def describe(self) -> str:
        if not self.enabled:
            return ""
        where = f", worktrees in {self.settings.worktree_dir}" if self.settings.worktree_dir else ""
        ship = "ship step: .pralph/ship.md" if self.ships else "no ship step — merging is up to you"
        return f"a branch per story{where}; {ship}"

    def _warn(self, key: str, message: str) -> None:
        """Say it once per run: these come up on every look at every story."""
        if key not in self._warned:
            self._warned.add(key)
            self._echo(click.style(f"  [git] {message}", fg="yellow"))

    # -- base branch --

    def base_branch(self) -> str:
        if self._base is None:
            self._base = self.settings.base or self._default_branch()
        return self._base

    def _default_branch(self) -> str:
        try:
            head = git(self.root, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
            if head.startswith("origin/"):
                return head[len("origin/"):]
        except GitError:
            pass
        for name in ("main", "master"):
            if git_ok(self.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}") or git_ok(
                self.root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{name}",
            ):
                return name
        return git(self.root, "branch", "--show-current") or "main"

    def _fetch(self, base: str) -> None:
        """Bring origin's base branch up to date (at most every half minute)."""
        if time.time() - self._fetched_at < 30:
            return
        self._fetched_at = time.time()
        if "origin" not in git(self.root, "remote").split():
            return
        try:
            git(self.root, "fetch", "--quiet", "origin", base)
        except GitError as e:
            self._warn("fetch", f"could not fetch origin/{base} ({e}) — working from what is here")

    def _start_point(self, base: str) -> str:
        for ref in (f"refs/remotes/origin/{base}", f"refs/heads/{base}"):
            if git_ok(self.root, "rev-parse", "--verify", "--quiet", ref):
                return ref
        return "HEAD"

    # -- a story's branch and directory --

    def worktrees(self) -> dict[str, Path]:
        """Branch → directory of every worktree of the repository."""
        found: dict[str, Path] = {}
        path: Path | None = None
        for line in git(self.root, "worktree", "list", "--porcelain").splitlines():
            if line.startswith("worktree "):
                path = Path(line[len("worktree "):])
            elif line.startswith("branch refs/heads/") and path is not None:
                found[line[len("branch refs/heads/"):]] = path
        return found

    def worktree_path(self, branch: str) -> Path:
        assert self.settings.worktree_dir is not None
        # One directory per project, so a shared worktree_dir can serve several repositories.
        return self.settings.worktree_dir / self.root.resolve().name / branch.replace("/", "-")

    def _dirty(self, cwd: Path) -> bool:
        # pralph's own state changes all the time and is not the story's work.
        return bool(git(cwd, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude).pralph"))

    def prepare(self, story: Story) -> Path:
        """Put the story on its branch and return the directory to work in.

        A branch that exists is taken up again (rework, a second attempt); a
        new one starts from the freshly fetched base branch. Raises GitError.
        """
        if not self.enabled:
            return self.root
        if not git_ok(self.root, "rev-parse", "--git-dir"):
            raise GitError(f"{self.root} is not a git repository — a branch per story needs one")

        info = dict(git_info(story))
        branch = info.get("branch") or branch_name(story)
        base = info.get("base") or self.base_branch()
        self._fetch(base)
        start = self._start_point(base)

        exists = git_ok(self.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
        if not exists and not git_ok(self.root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"):
            info["start"] = git(self.root, "rev-parse", start)
        if not exists and git_ok(self.root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"):
            git(self.root, "branch", "--track", branch, f"origin/{branch}")  # somebody pushed it already
            exists = True

        if self.settings.worktree_dir is not None:
            workdir = self.worktrees().get(branch)
            if workdir is None:
                workdir = self.worktree_path(branch)
                workdir.parent.mkdir(parents=True, exist_ok=True)
                if exists:
                    git(self.root, "worktree", "add", str(workdir), branch)
                else:
                    # --no-track: the branch must not get the base branch as its upstream,
                    # or a plain `git push` would aim at the base branch.
                    git(self.root, "worktree", "add", "--no-track", "-b", branch, str(workdir), start)
            info["worktree"] = str(workdir)
        else:
            workdir = self.root
            current = git(self.root, "branch", "--show-current")
            if current != branch:
                if self._dirty(self.root):
                    raise GitError(
                        "the checkout has uncommitted changes — commit or stash them, "
                        "or set worktree_dir so every story gets a directory of its own"
                    )
                if self._original_branch is None:
                    self._original_branch = current
                if exists:
                    git(self.root, "switch", branch)
                else:
                    git(self.root, "switch", "--no-track", "-c", branch, start)
            info.pop("worktree", None)

        info.update(branch=branch, base=base)
        self._save_info(story, info)
        return workdir

    def _save_info(self, story: Story, info: dict) -> None:
        story.metadata["git"] = info  # the caller's copy

        def mutate(s: Story) -> None:
            s.metadata["git"] = dict(info)

        self.state.update_story(story.id, mutate, notify=False)

    def restore(self) -> None:
        """Without worktrees: put the checkout back on the branch the run started from."""
        if self._original_branch is None:
            return
        try:
            if git(self.root, "branch", "--show-current") != self._original_branch and not self._dirty(self.root):
                git(self.root, "switch", self._original_branch)
        except GitError as e:
            self._warn("restore", f"left the checkout where it is ({e})")
        self._original_branch = None

    # -- dependencies --

    def unmet(self, story: Story, by_id: dict[str, Story]) -> list[str]:
        """Dependencies whose work is not available to this story yet."""
        unmet: list[str] = []
        for dep in story.dependencies:
            other = by_id.get(dep)
            if other is None or other is story:
                continue
            if other.status == StoryStatus.merged or other.status in _NO_CODE:
                continue
            if other.status == StoryStatus.implemented and not on_branch(other):
                continue  # done in the checkout itself (or in the tracker): it is there
            if not self.enabled and other.status in _UNFINISHED:
                continue  # one shared branch: a dependency only orders the work
            unmet.append(dep)
        return unmet

    def split(self, actionable: list[Story]) -> tuple[list[Story], dict[str, list[str]]]:
        """Stories that can be worked on now, and what holds the others back."""
        by_id = {s.id: s for s in self.state.load_stories()}
        ready: list[Story] = []
        blocked: dict[str, list[str]] = {}
        for story in actionable:
            unmet = self.unmet(story, by_id)
            if unmet:
                blocked[story.id] = unmet
            else:
                ready.append(story)
        return ready, blocked

    def waiting(self) -> list[Story]:
        return [s for s in self.state.load_stories() if is_waiting(s)]

    # -- the ship step --

    def ship_prompt(self, story: Story, workdir: Path) -> str:
        info = git_info(story)
        link = story.metadata.get("linear") if isinstance(story.metadata.get("linear"), dict) else {}
        values = {
            "story_id": story.id,
            "story_title": story.title,
            "branch": info.get("branch", ""),
            "base_branch": info.get("base", ""),
            "linear_id": link.get("identifier", ""),
            "linear_url": link.get("url", ""),
            "pr_url": info.get("pr_url", ""),
            "worktree": str(workdir),
        }
        prompt = self.settings.ship_prompt
        for key, value in values.items():
            prompt = prompt.replace("{{" + key + "}}", str(value))

        context = [
            f"- Story: {story.id} — {story.title}",
            f"- Branch: `{values['branch']}` (from `{values['base_branch']}`), checked out in `{workdir}`",
        ]
        if values["linear_id"]:
            context.append(f"- Linear issue: {values['linear_id']} {values['linear_url']}".rstrip())
        if values["pr_url"]:
            context.append(
                f"- A pull request for this branch is already open: {values['pr_url']} — "
                "it was reworked after review comments; update that pull request, do not open another"
            )
        return (
            f"{prompt}\n\n## What is being shipped\n\n" + "\n".join(context) + "\n\n"
            "The story is implemented, committed on that branch and has passed pralph's review.\n"
            "Do not wait for reviewers or for the merge — pralph watches the pull request itself.\n\n"
            "End your answer with the URL of the pull request on a line of its own.\n"
        )

    def pr_for_branch(self, workdir: Path, branch: str) -> str:
        """The open pull request of a branch, if `gh` knows one."""
        pr = self._gh_json(["pr", "view", branch, "--json", "url,state"], workdir)
        return str(pr.get("url") or "") if isinstance(pr, dict) and pr.get("state") == "OPEN" else ""

    def record_shipped(self, story: Story, pr_url: str) -> None:
        info = dict(git_info(story))
        again = bool(info.get("pr_url"))
        info["pr_url"] = pr_url
        info.pop("ship_error", None)
        info.pop("ship_error_at", None)
        self._save_info(story, info)
        self.state.mark_story_status(
            story.id, StoryStatus.shipped, summary=f"Pull request: {pr_url}",
            extra={"pr_url": pr_url, "pr_updated": again},
        )

    def record_ship_failure(self, story: Story, error: str) -> None:
        info = dict(git_info(story))
        info["ship_error"] = error[:2000]
        info["ship_error_at"] = datetime.now().isoformat()
        self._save_info(story, info)
        self.state.notify_story_problem(story.id, "ship", info["ship_error"])

    # -- watching pull requests --

    def _gh(self, args: list[str], cwd: Path) -> str | None:
        if "gh-missing" in self._warned:
            return None
        try:
            proc = _run(["gh", *args], cwd, timeout=60)
        except FileNotFoundError:
            self._warn("gh-missing", "the GitHub CLI (gh) is not installed — pull requests cannot be watched")
            return None
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            self._warn("gh:" + " ".join(args[:2]), f"gh {' '.join(args[:2])} failed: {detail[-1] if detail else '?'}")
            return None
        return proc.stdout

    def _gh_json(self, args: list[str], cwd: Path):
        out = self._gh(args, cwd)
        if out is None:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    def _inline_comments(self, pr_url: str, since: str) -> list[dict]:
        match = _GITHUB_PR_RE.match(pr_url)
        if not match:
            return []
        host, owner, repo, number = match.groups()
        args = ["api", f"repos/{owner}/{repo}/pulls/{number}/comments", "--paginate",
                "--jq", ".[] | {path, line, body, user: .user.login, created_at}"]
        if host != "github.com":
            args += ["--hostname", host]
        out = self._gh(args, self.root) or ""
        comments = []
        for line in out.splitlines():
            try:
                comment = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(comment, dict) and str(comment.get("created_at") or "") > since:
                comments.append(comment)
        return comments

    def _requested_changes(self, pr: dict, pr_url: str, since: str) -> tuple[str, str]:
        """Review feedback newer than ``since`` → (text for the rework, newest timestamp).

        Only a review that asks for changes sends a story back; comments that
        came with it are passed along.
        """
        reviews = [r for r in pr.get("reviews") or [] if isinstance(r, dict) and str(r.get("submittedAt") or "") > since]
        if not any(r.get("state") == "CHANGES_REQUESTED" for r in reviews):
            return "", since
        newest = max(str(r["submittedAt"]) for r in reviews)
        lines = ["# Changes requested on the pull request", "", pr_url, ""]
        for review in reviews:
            body = str(review.get("body") or "").strip()
            if body and review.get("state") in ("CHANGES_REQUESTED", "COMMENTED"):
                who = (review.get("author") or {}).get("login") or "reviewer"
                lines += [f"**{who}** ({str(review['state']).lower().replace('_', ' ')}):", "", body, ""]
        comments = self._inline_comments(pr_url, since)
        if comments:
            lines += ["## Comments on the code", ""]
            for c in comments:
                where = f"`{c.get('path')}:{c.get('line')}`" if c.get("line") else f"`{c.get('path')}`"
                lines.append(f"- {where} — {c.get('user') or 'reviewer'}: {str(c.get('body') or '').strip()}")
            newest = max(newest, max(str(c["created_at"]) for c in comments))
        return "\n".join(lines).strip() + "\n", newest

    def _branch_merged(self, info: dict) -> bool:
        """Is everything on the story's branch in the base branch (also after a squash merge)?"""
        branch = info.get("branch") or ""
        if not branch or not git_ok(self.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"):
            return False
        worktree = info.get("worktree")
        try:
            if worktree and Path(worktree).is_dir():
                if self._dirty(Path(worktree)):
                    return False  # work that was never committed is not merged
            elif git(self.root, "branch", "--show-current") == branch and self._dirty(self.root):
                return False
            if info.get("start") and git(self.root, "rev-parse", branch) == info["start"]:
                # Nothing was ever committed on it: that is not a merge, it is missing work.
                self._warn(f"empty:{branch}", f"branch {branch} has no commits — its story was not committed")
                return False
            base = info.get("base") or self.base_branch()
            self._fetch(base)
            start = self._start_point(base)
            if git_ok(self.root, "merge-base", "--is-ancestor", branch, start):
                return True
            # Squash or rebase merge: merging the branch would change nothing.
            merged_tree = git(self.root, "merge-tree", "--write-tree", start, branch).splitlines()[0]
            return merged_tree == git(self.root, "rev-parse", f"{start}^{{tree}}")
        except (GitError, IndexError):
            return False

    def _retire(self, story: Story) -> None:
        """A merged story no longer needs its worktree."""
        worktree = git_info(story).get("worktree")
        if not worktree or not Path(worktree).is_dir():
            return
        try:
            git(self.root, "worktree", "remove", worktree)
        except GitError as e:
            self._echo(click.style(f"  [git] kept the worktree {worktree} ({e})", fg="yellow"))

    def refresh(self) -> list[tuple[str, str, str, str]]:
        """Look at every waiting story's pull request or branch and record what happened.

        Returns (story ID, old status, new status, note) for each change.
        """
        changes: list[tuple[str, str, str, str]] = []

        def mark(story: Story, status: StoryStatus, note: str, **extra) -> None:
            changes.append((story.id, story.status.value, status.value, note))
            self.state.mark_story_status(story.id, status, summary=note, extra=extra)

        for story in self.state.load_stories():
            info = git_info(story)
            pr_url = info.get("pr_url") or ""
            if story.status == StoryStatus.merged:
                self._retire(story)  # merged as far as the tracker knows: the worktree is done with
                continue
            if story.status == StoryStatus.shipped and pr_url:
                pr = self._gh_json(["pr", "view", pr_url, "--json", "state,reviews,url"], self.root)
                if isinstance(pr, dict):
                    if pr.get("state") == "MERGED":
                        mark(story, StoryStatus.merged, f"Pull request merged: {pr_url}", pr_url=pr_url)
                        self._retire(story)
                    elif pr.get("state") == "CLOSED":
                        mark(story, StoryStatus.skipped, f"Pull request closed without a merge: {pr_url}",
                             reason=f"The pull request {pr_url} was closed without being merged.")
                    else:
                        feedback, newest = self._requested_changes(pr, pr_url, str(info.get("pr_feedback_at") or ""))
                        if feedback:
                            self.state.write_review_feedback(story.id, feedback)
                            self._save_info(story, {**info, "pr_feedback_at": newest})
                            mark(story, StoryStatus.rework, f"Changes requested on {pr_url}", pr_url=pr_url)
                    continue
            # No pull request to ask (or no gh): the branch itself tells.
            if story.status in (StoryStatus.shipped, StoryStatus.implemented) and info.get("branch"):
                if self._branch_merged(info):
                    mark(story, StoryStatus.merged, f"Branch {info['branch']} is merged into {info.get('base') or self.base_branch()}")
                    self._retire(story)
        return changes

    def sleep(self) -> None:
        self._sleep(self.settings.poll_interval)

