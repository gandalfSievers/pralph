"""Tests for stories on their own branch: worktrees, the ship step, waiting for the merge.

git is real (throwaway repositories with a bare "origin"); Claude and the
GitHub CLI are played by ``Claude`` and ``GitHub``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pralph import ship as ship_lib
from pralph.cli import main
from pralph.loop import run_implement_loop, run_review, run_ship
from pralph.models import Story, StoryStatus
from pralph.runner import ClaudeResult
from pralph.state import StateManager


def sh(cwd: Path, *args: str) -> str:
    proc = subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True)
    assert proc.returncode == 0, f"{' '.join(args)}: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    # git must not pick up the developer's own configuration (signing, hooks, default branch).
    (tmp_path / "gitconfig").write_text("[user]\n\tname = Test\n\temail = test@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path) -> Path:
    """A clone of a bare origin with one commit on main."""
    origin = tmp_path / "origin.git"
    sh(tmp_path, "git", "init", "--quiet", "--bare", "-b", "main", str(origin))
    root = tmp_path / "shop"
    sh(tmp_path, "git", "clone", "--quiet", str(origin), str(root))
    sh(root, "git", "checkout", "--quiet", "-b", "main")
    (root / ".gitignore").write_text(".pralph/\n")
    (root / "README.md").write_text("shop\n")
    sh(root, "git", "add", ".")
    sh(root, "git", "commit", "--quiet", "-m", "init")
    sh(root, "git", "push", "--quiet", "-u", "origin", "main")
    sh(root, "git", "remote", "set-head", "origin", "main")
    return root


def make_state(repo: Path, stories: list[Story], **config) -> StateManager:
    state = StateManager(str(repo))
    if config:
        (state.state_dir / "config.json").write_text(json.dumps(config))
    state.append_stories(stories)
    return state


def two_stories() -> list[Story]:
    return [
        Story(id="DB-001", title="Schema", content="Tables.", category="DB"),
        Story(id="API-001", title="List products", content="Endpoint.", category="API", dependencies=["DB-001"]),
    ]


class Claude:
    """run_with_retry stand-in: an implementation commits a file named after the story."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []      # (kind, story ID)
        self.cwds: dict[tuple[str, str], Path] = {}
        self.branches: dict[tuple[str, str], str] = {}  # what was checked out there at the time
        self.prompts: dict[tuple[str, str], str] = {}
        self.kwargs: dict[tuple[str, str], dict] = {}
        self.answers: dict[str, list[dict]] = {}    # story ID → implementation answers, in order
        self.ship_answer = "Opened https://github.com/acme/shop/pull/{n}"
        self.titles: dict[str, str] = {}
        self.pulls = 0

    def _note(self, kind: str, story_id: str, prompt: str, kwargs: dict) -> None:
        key = (kind, story_id)
        self.calls.append(key)
        self.cwds[key] = Path(kwargs["project_dir"])
        self.branches[key] = sh(self.cwds[key], "git", "branch", "--show-current")
        self.prompts[key] = prompt
        self.kwargs[key] = kwargs

    def __call__(self, prompt: str, **kwargs) -> ClaudeResult:
        cwd = Path(kwargs["project_dir"])
        if "## What is being shipped" in prompt:
            story_id = re.search(r"- Story: (\S+)", prompt).group(1)
            self._note("ship", story_id, prompt, kwargs)
            if "already open" not in prompt:
                self.pulls += 1
            return ClaudeResult(success=True, result=self.ship_answer.format(n=self.pulls))
        if "# Code Review Mode" in prompt:
            title = re.search(r"\*\*Title:\*\* (.+)", prompt).group(1)
            self._note("review", self.titles[title], prompt, kwargs)
            return ClaudeResult(success=True, result='{"approved": true, "feedback": "ok", "issues": []}')
        story_id = re.search(r'"id": "([A-Z]+-\d+)"', prompt).group(1)
        self.titles[re.search(r"\*\*Title:\*\* (.+)", prompt).group(1)] = story_id
        self._note("implement", story_id, prompt, kwargs)
        answer = (self.answers.get(story_id) or [None]).pop(0) if self.answers.get(story_id) else None
        answer = answer or {"status": "implemented", "summary": f"Built {story_id}"}
        n = sum(1 for kind, sid in self.calls if (kind, sid) == ("implement", story_id))
        (cwd / f"{story_id}.txt").write_text(f"{story_id} round {n}\n")
        sh(cwd, "git", "add", f"{story_id}.txt")
        sh(cwd, "git", "commit", "--quiet", "-m", f"{story_id} round {n}")
        return ClaudeResult(success=True, result=json.dumps(answer))


class GitHub:
    """The `gh` CLI: pull requests by URL."""

    def __init__(self) -> None:
        self.pulls: dict[str, dict] = {}
        self.comments: dict[str, list[dict]] = {}
        self.calls: list[list[str]] = []

    def pull(self, url: str) -> dict:
        return self.pulls.setdefault(url, {"url": url, "state": "OPEN", "reviews": []})

    def run(self, cmd: list[str], cwd: Path, timeout: int = 120):
        if cmd[0] != "gh":
            return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        self.calls.append(cmd)
        if cmd[1:3] == ["pr", "view"]:
            pr = self.pulls.get(cmd[3])
            if pr is None:
                return subprocess.CompletedProcess(cmd, 1, "", "no pull requests found")
            return subprocess.CompletedProcess(cmd, 0, json.dumps(pr), "")
        if cmd[1] == "api":
            number = cmd[2].split("/")[4]
            url = next(u for u in self.pulls if u.endswith(f"/pull/{number}"))
            return subprocess.CompletedProcess(cmd, 0, "\n".join(json.dumps(c) for c in self.comments.get(url, [])), "")
        raise AssertionError(f"unexpected gh call: {cmd}")


@pytest.fixture
def github(monkeypatch) -> GitHub:
    gh = GitHub()
    monkeypatch.setattr(ship_lib, "_run", gh.run)
    return gh


def run(state: StateManager, claude: Claude, **kwargs):
    kwargs.setdefault("wait", False)
    with patch("pralph.loop.run_with_retry", side_effect=claude):
        return run_implement_loop(state, cooldown=0, phase1=False, **kwargs)


def story(state: StateManager, story_id: str) -> Story:
    return next(s for s in state.load_stories() if s.id == story_id)


def status(state: StateManager, story_id: str) -> StoryStatus:
    return story(state, story_id).status


def merge_into_main(repo: Path, branch: str) -> None:
    """What merging the pull request does on the remote (fast-forward)."""
    sh(repo, "git", "push", "--quiet", "origin", f"{branch}:main")


# ── off by default ────────────────────────────────────────────────────


def test_without_configuration_everything_stays_on_the_checked_out_branch(repo) -> None:
    state = make_state(repo, two_stories())
    claude = Claude()
    ps = run(state, claude)
    assert ps.completion_reason == "all_stories_done"
    assert [c for c in claude.calls if c[0] == "implement"] == [("implement", "DB-001"), ("implement", "API-001")]
    assert sh(repo, "git", "branch", "--format=%(refname:short)") == "main"
    assert (repo / "DB-001.txt").exists() and (repo / "API-001.txt").exists()
    assert status(state, "API-001") == StoryStatus.implemented and "git" not in story(state, "API-001").metadata
    assert "## Branch" not in claude.prompts[("implement", "DB-001")]


# ── branches and worktrees ────────────────────────────────────────────


def test_every_story_gets_a_worktree_on_its_own_branch(repo, tmp_path) -> None:
    stories = [Story(id="DB-001", title="Schema", content="c", category="DB"),
               Story(id="UI-001", title="Start page!", content="c", category="UI")]
    state = make_state(repo, stories, worktree_dir="../trees")
    claude = Claude()
    ps = run(state, claude)

    db, ui = tmp_path / "trees" / "shop" / "db-001-schema", tmp_path / "trees" / "shop" / "ui-001-start-page"
    assert claude.cwds[("implement", "DB-001")] == db and claude.cwds[("review", "DB-001")] == db
    assert claude.cwds[("implement", "UI-001")] == ui
    assert (db / "DB-001.txt").exists() and not (db / "UI-001.txt").exists(), "each branch starts from main"
    assert sh(db, "git", "branch", "--show-current") == "db-001-schema"
    assert sh(repo, "git", "branch", "--show-current") == "main", "the checkout is left alone"
    assert not (repo / "DB-001.txt").exists()
    assert "branch.db-001-schema.merge" not in sh(repo, "git", "config", "--list"), "no upstream: a push cannot aim at main"

    info = story(state, "DB-001").metadata["git"]
    assert info == {"branch": "db-001-schema", "base": "main", "worktree": str(db),
                    "start": sh(repo, "git", "rev-parse", "origin/main")}
    assert status(state, "DB-001") == StoryStatus.implemented
    assert ps.completion_reason == "waiting", "implemented, but not merged: the run is not finished"
    assert not any(kind == "ship" for kind, _ in claude.calls), "without a ship step nothing is shipped"

    prompt = claude.prompts[("implement", "DB-001")]
    assert "its own branch, `db-001-schema`, started from `main`" in prompt and "Do not switch branches, push" in prompt
    assert "git diff main...HEAD" in claude.prompts[("review", "DB-001")]


def test_branch_is_named_after_the_linear_issue(repo, tmp_path) -> None:
    ticket = Story(id="COM-554", title="Guard", content="c", category="COM", metadata={"linear": {
        "id": "uuid-1", "identifier": "COM-554", "url": "https://linear.app/acme/issue/COM-554",
        "branch": "sam/com-554-guard-in-comply"}})
    state = make_state(repo, [ticket], worktree_dir=str(tmp_path / "trees"))
    run(state, Claude())
    assert story(state, "COM-554").metadata["git"]["branch"] == "sam/com-554-guard-in-comply"
    assert (tmp_path / "trees" / "shop" / "sam-com-554-guard-in-comply" / "COM-554.txt").exists()


def test_new_branch_starts_from_the_up_to_date_base(repo, tmp_path) -> None:
    other = tmp_path / "other"
    sh(tmp_path, "git", "clone", "--quiet", str(tmp_path / "origin.git"), str(other))
    (other / "colleague.txt").write_text("x\n")
    sh(other, "git", "add", "."), sh(other, "git", "commit", "--quiet", "-m", "colleague")
    sh(other, "git", "push", "--quiet", "origin", "main")

    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    claude = Claude()
    run(state, claude)
    assert (claude.cwds[("implement", "DB-001")] / "colleague.txt").exists(), "main was fetched first"
    assert not (repo / "colleague.txt").exists(), "the checkout itself is not pulled"


def test_branch_in_the_checkout_itself(repo) -> None:
    state = make_state(repo, two_stories()[:1], branch_per_story=True)
    claude = Claude()
    ps = run(state, claude)
    assert claude.cwds[("implement", "DB-001")] == repo
    assert sh(repo, "git", "log", "--format=%s", "db-001-schema", "-1") == "DB-001 round 1"
    assert sh(repo, "git", "branch", "--show-current") == "main", "the run puts the checkout back"
    assert not (repo / "DB-001.txt").exists() and ps.completion_reason == "waiting"
    assert "worktree" not in story(state, "DB-001").metadata["git"]


def test_review_by_itself_looks_at_the_storys_branch(repo) -> None:
    state = make_state(repo, two_stories()[:1], branch_per_story=True)
    claude = Claude()
    run(state, claude)
    assert claude.branches[("review", "DB-001")] == "db-001-schema"
    assert sh(repo, "git", "branch", "--show-current") == "main"

    with patch("pralph.loop.run_with_retry", side_effect=claude):
        ps = run_review(state, story_id="DB-001")
    assert ps.completion_reason == "review_approved"
    assert claude.calls[-1] == ("review", "DB-001") and claude.branches[("review", "DB-001")] == "db-001-schema"
    assert sh(repo, "git", "branch", "--show-current") == "main", "and the checkout is put back"


def test_uncommitted_changes_stop_the_run_without_touching_the_story(repo) -> None:
    (repo / "README.md").write_text("half-done edit\n")
    state = make_state(repo, two_stories(), branch_per_story=True)
    claude = Claude()
    ps = run(state, claude)
    assert claude.calls == [] and ps.completion_reason == "workspace_error"
    assert status(state, "DB-001") == StoryStatus.pending
    assert (repo / "README.md").read_text() == "half-done edit\n"


# ── dependencies wait for the merge ───────────────────────────────────


def test_dependent_story_waits_until_its_dependency_is_merged(repo, tmp_path) -> None:
    state = make_state(repo, two_stories(), worktree_dir="../trees")
    claude = Claude()
    ps = run(state, claude)
    assert [c for c in claude.calls if c[0] == "implement"] == [("implement", "DB-001")]
    assert status(state, "API-001") == StoryStatus.pending and ps.completion_reason == "waiting"

    ps = run(state, claude)
    assert [c for c in claude.calls if c[0] == "implement"] == [("implement", "DB-001")], "still not merged"

    merge_into_main(repo, "db-001-schema")
    ps = run(state, claude)
    assert status(state, "DB-001") == StoryStatus.merged
    assert not (tmp_path / "trees" / "shop" / "db-001-schema").exists(), "the worktree of a merged story is removed"
    api = claude.cwds[("implement", "API-001")]
    assert (api / "DB-001.txt").exists(), "the dependent story builds on the merged work"
    assert status(state, "API-001") == StoryStatus.implemented and ps.completion_reason == "waiting"

    merge_into_main(repo, "api-001-list-products")
    assert run(state, claude).completion_reason == "all_stories_done"


def test_squash_merge_counts_as_merged(repo, tmp_path) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    run(state, Claude())
    other = tmp_path / "other"
    sh(tmp_path, "git", "clone", "--quiet", str(tmp_path / "origin.git"), str(other))
    (other / "DB-001.txt").write_text("DB-001 round 1\n")  # the same change as one new commit
    sh(other, "git", "add", "."), sh(other, "git", "commit", "--quiet", "-m", "DB-001 (#1)")
    sh(other, "git", "push", "--quiet", "origin", "main")

    assert ship_lib.Flow(state).refresh() == [
        ("DB-001", "implemented", "merged", "Branch db-001-schema is merged into main")]
    assert status(state, "DB-001") == StoryStatus.merged


def test_uncommitted_work_is_not_taken_for_merged(repo, tmp_path) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    claude = Claude()
    run(state, claude)
    merge_into_main(repo, "db-001-schema")
    (claude.cwds[("implement", "DB-001")] / "README.md").write_text("forgot to commit\n")
    assert ship_lib.Flow(state).refresh() == []
    assert status(state, "DB-001") == StoryStatus.implemented


def test_branch_without_commits_is_not_taken_for_merged(repo, capsys) -> None:
    """An implementation that forgot to commit leaves a branch identical to main."""
    state = make_state(repo, two_stories(), worktree_dir="../trees")
    flow = ship_lib.Flow(state)
    workdir = flow.prepare(story(state, "DB-001"))
    (workdir / "new-file.txt").write_text("never committed\n")
    state.mark_story_status("DB-001", StoryStatus.implemented)
    assert flow.refresh() == [] and status(state, "DB-001") == StoryStatus.implemented
    assert "branch db-001-schema has no commits" in capsys.readouterr().out


def test_story_that_was_done_before_branches_does_not_hold_others_back(repo) -> None:
    state = make_state(repo, two_stories(), worktree_dir="../trees")
    state.mark_story_status("DB-001", StoryStatus.implemented)  # e.g. completed in Linear, or done on main
    claude = Claude()
    run(state, claude)
    assert ("implement", "API-001") in claude.calls


def test_nothing_workable_is_reported_as_blocked(repo) -> None:
    state = make_state(repo, two_stories(), worktree_dir="../trees")
    state.mark_story_status("DB-001", StoryStatus.error, error_reason="boom")
    claude = Claude()
    ps = run(state, claude)
    assert claude.calls == [] and ps.completion_reason == "blocked"
    assert status(state, "API-001") == StoryStatus.pending


# ── the ship step ─────────────────────────────────────────────────────


def with_ship_step(state: StateManager, text: str = "Run /team:open-pr for {{branch}} ({{story_id}}, {{linear_id}}).") -> None:
    (state.state_dir / "ship.md").write_text(text)
    (state.state_dir / "ship-tools.txt").write_text("Skill\nBash, Read\n")
    cfg = state.state_dir / "config.json"
    config = json.loads(cfg.read_text()) if cfg.exists() else {}
    cfg.write_text(json.dumps({**config, "ship_timeout": 5400}))


def test_ship_step_runs_after_the_review_and_the_story_is_shipped(repo, tmp_path, github) -> None:
    ticket = Story(id="COM-554", title="Guard", content="c", category="COM", metadata={"linear": {
        "id": "u", "identifier": "COM-554", "url": "https://linear.app/acme/issue/COM-554", "branch": "com-554-guard"}})
    state = make_state(repo, [ticket], worktree_dir="../trees")
    with_ship_step(state)
    claude = Claude()
    ps = run(state, claude)

    assert claude.calls == [("implement", "COM-554"), ("review", "COM-554"), ("ship", "COM-554")]
    key = ("ship", "COM-554")
    assert claude.cwds[key] == tmp_path / "trees" / "shop" / "com-554-guard"
    assert claude.kwargs[key]["allowed_tools"] == "Skill,Bash,Read" and claude.kwargs[key]["timeout"] == 5400
    prompt = claude.prompts[key]
    assert prompt.startswith("Run /team:open-pr for com-554-guard (COM-554, COM-554).")
    assert "Linear issue: COM-554 https://linear.app/acme/issue/COM-554" in prompt
    assert "Do not wait for reviewers or for the merge" in prompt

    shipped = story(state, "COM-554")
    assert shipped.status == StoryStatus.shipped
    assert shipped.metadata["git"]["pr_url"] == "https://github.com/acme/shop/pull/1"
    assert ps.completion_reason == "waiting"
    assert sh(repo, "git", "ls-remote", "--heads", "origin", "com-554-guard") == "", "pralph itself pushes nothing"


def test_ship_step_alone_switches_branches_on(repo, github) -> None:
    state = make_state(repo, two_stories()[:1])
    with_ship_step(state, "Open the pull request.")
    claude = Claude()
    run(state, claude)
    assert status(state, "DB-001") == StoryStatus.shipped and claude.cwds[("ship", "DB-001")] == repo
    assert sh(repo, "git", "branch", "--show-current") == "main"


def test_ship_step_that_opens_nothing_leaves_the_story_implemented(repo, github) -> None:
    state = make_state(repo, two_stories(), worktree_dir="../trees")
    with_ship_step(state)
    claude = Claude()
    claude.ship_answer = "I could not push: permission denied."
    ps = run(state, claude)
    assert status(state, "DB-001") == StoryStatus.implemented
    assert "did not report a pull request" in story(state, "DB-001").metadata["git"]["ship_error"]
    assert story(state, "DB-001").metadata["git"]["ship_error_at"]
    assert ("implement", "API-001") not in claude.calls and ps.completion_reason == "waiting"

    # `pralph ship` tries again.
    claude.ship_answer = "Done: https://github.com/acme/shop/pull/7."
    with patch("pralph.loop.run_with_retry", side_effect=claude):
        assert run_ship(state, story_id="DB-001").completion_reason == "shipped"
    info = story(state, "DB-001").metadata["git"]
    assert info["pr_url"] == "https://github.com/acme/shop/pull/7"
    assert "ship_error" not in info and "ship_error_at" not in info


def test_pull_request_found_by_branch_when_the_step_does_not_print_it(repo, github) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    with_ship_step(state)
    github.pulls["db-001-schema"] = {"url": "https://github.com/acme/shop/pull/3", "state": "OPEN"}
    claude = Claude()
    claude.ship_answer = "Pushed and opened."
    run(state, claude)
    assert story(state, "DB-001").metadata["git"]["pr_url"] == "https://github.com/acme/shop/pull/3"


def test_find_pr_url() -> None:
    assert ship_lib.find_pr_url("see https://github.com/a/b/pull/1 and then\nhttps://github.com/a/b/pull/22.") == \
        "https://github.com/a/b/pull/22"
    assert ship_lib.find_pr_url("MR: https://gitlab.example.com/g/p/-/merge_requests/5") == \
        "https://gitlab.example.com/g/p/-/merge_requests/5"
    assert ship_lib.find_pr_url("https://github.com/a/b/issues/4") == ""


def test_cli_ship_needs_a_ship_step(repo) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    run(state, Claude())
    result = CliRunner().invoke(main, ["--project-dir", str(repo), "ship", "-s", "DB-001"])
    assert result.exit_code == 1 and "write the prompt for it to .pralph/ship.md" in result.output


# ── watching the pull request ─────────────────────────────────────────


def shipped(repo: Path, github: GitHub, stories: list[Story] | None = None) -> tuple[StateManager, Claude, str]:
    state = make_state(repo, stories or two_stories(), worktree_dir="../trees")
    with_ship_step(state)
    claude = Claude()
    run(state, claude)
    url = story(state, "DB-001").metadata["git"]["pr_url"]
    github.pull(url)
    return state, claude, url


def test_merged_pull_request_frees_the_dependent_story(repo, github) -> None:
    state, claude, url = shipped(repo, github)
    assert ("implement", "API-001") not in claude.calls, "a pull request that is open is not a dependency met"

    merge_into_main(repo, "db-001-schema")
    github.pull(url)["state"] = "MERGED"
    run(state, claude)
    assert status(state, "DB-001") == StoryStatus.merged
    assert ("implement", "API-001") in claude.calls and status(state, "API-001") == StoryStatus.shipped


def test_requested_changes_become_rework_on_the_same_branch(repo, github) -> None:
    state, claude, url = shipped(repo, github, two_stories()[:1])
    github.pull(url)["reviews"] = [
        {"author": {"login": "kim"}, "state": "COMMENTED", "body": "Looks close.", "submittedAt": "2026-01-02T10:00:00Z"},
        {"author": {"login": "kim"}, "state": "CHANGES_REQUESTED", "body": "Please add an index.",
         "submittedAt": "2026-01-02T10:05:00Z"},
    ]
    github.comments[url] = [{"path": "schema.sql", "line": 12, "body": "Wrong type here", "user": "kim",
                             "created_at": "2026-01-02T10:04:00Z"}]
    ps = run(state, claude)

    assert claude.calls[-3:] == [("implement", "DB-001"), ("review", "DB-001"), ("ship", "DB-001")]
    rework = claude.prompts[("implement", "DB-001")]
    assert "Please add an index." in rework and "`schema.sql:12` — kim: Wrong type here" in rework and "Looks close." in rework
    worktree = claude.cwds[("implement", "DB-001")]
    assert sh(worktree, "git", "log", "--format=%s", "db-001-schema", "-2").splitlines() == ["DB-001 round 2", "DB-001 round 1"]
    again = claude.prompts[("ship", "DB-001")]
    assert f"already open: {url}" in again and "do not open another" in again
    assert status(state, "DB-001") == StoryStatus.shipped and claude.pulls == 1
    assert ps.completion_reason == "waiting"

    # The same review is not acted on twice — only a newer one is.
    before = len(claude.calls)
    run(state, claude)
    assert len(claude.calls) == before and status(state, "DB-001") == StoryStatus.shipped
    github.pull(url)["reviews"].append(
        {"author": {"login": "kim"}, "state": "CHANGES_REQUESTED", "body": "One more thing.", "submittedAt": "2026-01-03T09:00:00Z"})
    run(state, claude)
    assert "One more thing." in claude.prompts[("implement", "DB-001")]
    assert "Please add an index." not in claude.prompts[("implement", "DB-001")]


def test_comments_alone_do_not_send_a_story_back(repo, github) -> None:
    state, claude, url = shipped(repo, github, two_stories()[:1])
    github.pull(url)["reviews"] = [
        {"author": {"login": "kim"}, "state": "COMMENTED", "body": "Nice.", "submittedAt": "2026-01-02T10:00:00Z"},
        {"author": {"login": "lee"}, "state": "APPROVED", "body": "", "submittedAt": "2026-01-02T11:00:00Z"},
    ]
    assert ship_lib.Flow(state).refresh() == [] and status(state, "DB-001") == StoryStatus.shipped


def test_closed_pull_request_skips_the_story_and_says_so(repo, github, capsys) -> None:
    state, claude, url = shipped(repo, github)
    github.pull(url)["state"] = "CLOSED"
    ps = run(state, claude)
    assert status(state, "DB-001") == StoryStatus.skipped
    assert f"DB-001: shipped → skipped (Pull request closed without a merge: {url})" in capsys.readouterr().out
    assert ("implement", "API-001") in claude.calls, "a skipped dependency brings no code: nothing to wait for"
    assert ps.completion_reason == "waiting"


def test_without_gh_the_branch_tells_when_it_is_merged(repo, monkeypatch) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    with_ship_step(state)
    claude = Claude()

    def no_gh(cmd, cwd, timeout=120):
        if cmd[0] == "gh":
            raise FileNotFoundError("gh")
        return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)

    monkeypatch.setattr(ship_lib, "_run", no_gh)
    run(state, claude)
    assert status(state, "DB-001") == StoryStatus.shipped
    assert ship_lib.Flow(state).refresh() == []
    merge_into_main(repo, "db-001-schema")
    run(state, claude)
    assert status(state, "DB-001") == StoryStatus.merged


def test_the_loop_waits_for_the_merge(repo, github, monkeypatch) -> None:
    state, claude, url = shipped(repo, github)
    naps: list[int] = []

    def nap(flow: ship_lib.Flow) -> None:
        naps.append(flow.settings.poll_interval)
        if len(naps) == 2:  # merged while pralph slept the second time
            merge_into_main(repo, "db-001-schema")
            github.pull(url)["state"] = "MERGED"
        if len(naps) == 4:
            merge_into_main(repo, "api-001-list-products")
            github.pull(story(state, "API-001").metadata["git"]["pr_url"])["state"] = "MERGED"
        assert len(naps) < 10, "the loop never came out of waiting"

    monkeypatch.setattr(ship_lib.Flow, "sleep", nap)
    cfg = state.state_dir / "config.json"
    cfg.write_text(json.dumps({**json.loads(cfg.read_text()), "merge_poll_interval": 900}))
    github.pull("https://github.com/acme/shop/pull/2")
    ps = run(state, claude, wait=True)

    assert naps == [900] * 4
    assert status(state, "DB-001") == StoryStatus.merged and status(state, "API-001") == StoryStatus.merged
    assert ps.completion_reason == "all_stories_done"


# ── a story that needs a person ───────────────────────────────────────


def test_story_that_needs_a_person_pauses_and_holds_its_dependents(repo) -> None:
    stories = two_stories() + [Story(id="UI-001", title="Start page", content="c", category="UI")]
    state = make_state(repo, stories)  # no branches: this works in a plain checkout too
    claude = Claude()
    claude.answers["DB-001"] = [{"status": "needs_human", "summary": "Migration written",
                                 "reason": "Apply the migration to staging and check the deploy"}]
    ps = run(state, claude)

    assert [sid for kind, sid in claude.calls if kind == "implement"] == ["DB-001", "UI-001"]
    assert ("review", "DB-001") not in claude.calls
    paused = story(state, "DB-001")
    assert paused.status == StoryStatus.needs_human
    assert paused.metadata["needs_human"]["reason"] == "Apply the migration to staging and check the deploy"
    assert status(state, "API-001") == StoryStatus.pending and ps.completion_reason == "waiting"
    assert "needs_human" in claude.prompts[("implement", "UI-001")], "the implementation is told how to pause"

    cli = ["--project-dir", str(repo)]
    wrong = CliRunner().invoke(main, [*cli, "unblock", "UI-001"])
    assert wrong.exit_code == 1 and "UI-001 is implemented — nothing to unblock" in wrong.output
    assert "DB-001: Apply the migration to staging" in wrong.output
    done = CliRunner().invoke(main, [*cli, "unblock", "DB-001", "-m", "Applied, deploy is green"])
    assert done.exit_code == 0 and status(state, "DB-001") == StoryStatus.pending

    ps = run(state, claude)
    prompt = claude.prompts[("implement", "DB-001")]
    assert "## Continuing after a pause" in prompt
    assert "> Apply the migration to staging and check the deploy" in prompt
    assert "They report: Applied, deploy is green" in prompt
    finished = story(state, "DB-001")
    assert finished.status == StoryStatus.implemented
    assert "needs_human" not in finished.metadata and "human_note" not in finished.metadata
    assert ("implement", "API-001") in claude.calls and ps.completion_reason == "all_stories_done"


def test_implementation_cannot_claim_to_have_shipped(repo) -> None:
    state = make_state(repo, two_stories()[:1])
    claude = Claude()
    claude.answers["DB-001"] = [{"status": "merged", "summary": "all done, trust me"}]
    run(state, claude)
    assert status(state, "DB-001") == StoryStatus.implemented


# ── CLI ───────────────────────────────────────────────────────────────


def test_cli_implement_says_how_stories_are_branched(repo) -> None:
    state = make_state(repo, two_stories()[:1], worktree_dir="../trees")
    claude = Claude()
    with patch("pralph.loop.run_with_retry", side_effect=claude):
        result = CliRunner().invoke(main, ["--project-dir", str(repo), "--cooldown", "0", "implement", "--no-wait", "--no-phase1"])
    assert result.exit_code == 0, result.output
    assert "git: a branch per story, worktrees in" in result.output and "no ship step" in result.output
    assert "Waiting for 1 story:" in result.output and "on branch db-001-schema, not merged yet" in result.output
    assert status(state, "DB-001") == StoryStatus.implemented
