"""Tests for the Linear mode: plan → Linear project, execution off the tickets.

Linear itself is replaced by ``FakeLinear``, an in-memory workspace that
answers the exact GraphQL documents ``pralph.linear`` sends.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pralph import linear as L
from pralph import linear_sync
from pralph import spec as spec_lib
from pralph.assembler import assemble_implement_prompt
from pralph.cli import main
from pralph.linear import LinearClient, LinearError
from pralph.linear_sync import LinearLink, LinearSync, parse_description, render_description
from pralph.loop import run_implement_loop
from pralph.models import Story, StoryStatus
from pralph.runner import ClaudeResult
from pralph.state import StateManager


# ── fake Linear workspace ─────────────────────────────────────────────


class FakeLinear:
    """In-memory Linear: one workspace, teams, workflow states, projects, issues."""

    def __init__(self, *, teams: list[str] = ("ENG",), page_size: int = 50) -> None:
        self.teams = [{"id": f"team-{k.lower()}", "key": k, "name": f"{k} team"} for k in teams]
        self.states = [
            {"id": "st-backlog", "name": "Backlog", "type": "backlog", "position": 0},
            {"id": "st-todo", "name": "Todo", "type": "unstarted", "position": 1},
            {"id": "st-progress", "name": "In Progress", "type": "started", "position": 2},
            {"id": "st-review", "name": "In Review", "type": "started", "position": 3},
            {"id": "st-done", "name": "Done", "type": "completed", "position": 4},
            # Its own type in Linear, and only reachable through a duplicate-of relation.
            {"id": "st-dup", "name": "Duplicate", "type": "duplicate", "position": 5},
            {"id": "st-canceled", "name": "Canceled", "type": "canceled", "position": 6},
        ]
        self.projects: dict[str, dict] = {}
        self.issues: dict[str, dict] = {}       # id → issue
        self.relations: list[tuple[str, str]] = []  # (blocker id, blocked id)
        self.comments: list[tuple[str, str]] = []   # (issue identifier, body)
        self.page_size = page_size
        self.calls: list[str] = []
        self.failures: list[dict | Exception] = []  # consumed one per call
        self.truncated: set[str] = set()  # issue ids whose relation list is cut off
        self._counter = 0

    # -- transport --

    def __call__(self, query: str, variables: dict) -> dict:
        name = next(k for k, doc in L.DOCUMENTS.items() if doc == query)
        self.calls.append(name)
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return failure
        return {"data": getattr(self, f"_{name}")(variables)}

    def mutations(self) -> list[str]:
        return [c for c in self.calls if c.endswith(("_create", "_update"))]

    # -- queries --

    def _viewer(self, v: dict) -> dict:
        return {"viewer": {"id": "u1", "name": "Dev", "email": "dev@example.com"},
                "organization": {"name": "Acme", "urlKey": "acme"}}

    def _teams(self, v: dict) -> dict:
        return {"teams": {"nodes": self.teams}}

    def _states(self, v: dict) -> dict:
        team = next(t for t in self.teams if t["id"] == v["teamId"])
        return {"team": {**team, "states": {"nodes": list(reversed(self.states))}}}

    def _project_node(self, p: dict) -> dict:
        return {**{k: p[k] for k in ("id", "name", "url", "slugId")},
                "teams": {"nodes": [t for t in self.teams if t["id"] in p["teamIds"]]}}

    def _project(self, v: dict) -> dict:
        p = self.projects.get(v["id"])
        return {"project": self._project_node(p) if p else None}

    def _projects(self, v: dict) -> dict:
        flt = v.get("filter") or {}
        found = list(self.projects.values())
        if "name" in flt:
            found = [p for p in found if p["name"].lower() == flt["name"]["eqIgnoreCase"].lower()]
        if "slugId" in flt:
            found = [p for p in found if p["slugId"] == flt["slugId"]["eq"]]
        return {"projects": {"nodes": [self._project_node(p) for p in found]}}

    def _issue_node(self, issue: dict) -> dict:
        state = next(s for s in self.states if s["id"] == issue["stateId"])
        return {
            "id": issue["id"], "identifier": issue["identifier"], "title": issue["title"],
            "description": issue["description"], "priority": issue["priority"], "url": issue["url"],
            "trashed": issue.get("trashed"),
            "state": {"id": state["id"], "name": state["name"], "type": state["type"]},
            "parent": {"id": issue["parentId"]} if issue.get("parentId") else None,
            "inverseRelations": {
                "nodes": [] if issue["id"] in self.truncated else [
                    {"type": "blocks", "issue": {"id": blocker}}
                    for blocker, blocked in self.relations if blocked == issue["id"]
                ],
                "pageInfo": {"hasNextPage": issue["id"] in self.truncated},
            },
        }

    def _project_issues(self, v: dict) -> dict:
        if v["id"] not in self.projects:
            return {"project": None}
        issues = [i for i in self.issues.values() if i["projectId"] == v["id"] and not i.get("archived")]
        start = int(v.get("after") or 0)
        page = issues[start:start + self.page_size]
        end = start + len(page)
        return {"project": {"issues": {
            "nodes": [self._issue_node(i) for i in page],
            "pageInfo": {"hasNextPage": end < len(issues), "endCursor": str(end)},
        }}}

    # -- mutations --

    def _project_create(self, v: dict) -> dict:
        inp = v["input"]
        self._counter += 1
        pid = f"00000000-0000-4000-8000-{self._counter:012d}"
        slug = f"{self._counter:012x}"
        self.projects[pid] = {
            "id": pid, "name": inp["name"], "slugId": slug, "teamIds": inp["teamIds"],
            "description": inp.get("description", ""), "content": inp.get("content", ""),
            "url": f"https://linear.app/acme/project/{inp['name'].lower().replace(' ', '-')}-{slug}",
        }
        return {"projectCreate": {"success": True, "project": self._project_node(self.projects[pid])}}

    def _issue_create(self, v: dict) -> dict:
        inp = v["input"]
        self._counter += 1
        team = next(t for t in self.teams if t["id"] == inp["teamId"])
        identifier = f"{team['key']}-{self._counter}"
        issue = {
            "id": f"issue-{self._counter}", "identifier": identifier, "title": inp["title"],
            "description": inp.get("description", ""), "priority": inp.get("priority", 0),
            "projectId": inp.get("projectId"), "parentId": inp.get("parentId"),
            "stateId": inp.get("stateId") or "st-backlog",  # team default for new issues
            "url": f"https://linear.app/acme/issue/{identifier}",
        }
        self.issues[issue["id"]] = issue
        return {"issueCreate": {"success": True, "issue": self._issue_node(issue)}}

    def _issue_update(self, v: dict) -> dict:
        issue = self.issues[v["id"]]
        if v["input"].get("stateId") == "st-dup":
            return_error = "Issues can only be moved to a duplicate state when a duplicate issue relation exists."
            raise LinearError(f"Linear API error: {return_error}")
        issue.update(v["input"])
        return {"issueUpdate": {"success": True, "issue": self._issue_node(issue)}}

    def _comment_create(self, v: dict) -> dict:
        issue = self.issues[v["input"]["issueId"]]
        self.comments.append((issue["identifier"], v["input"]["body"]))
        return {"commentCreate": {"success": True}}

    def _relation_create(self, v: dict) -> dict:
        inp = v["input"]
        assert inp["type"] == "blocks"
        self.relations.append((inp["issueId"], inp["relatedIssueId"]))
        return {"issueRelationCreate": {"success": True}}

    # -- what people do in the Linear UI --

    def find(self, title_part: str) -> dict:
        matches = [i for i in self.issues.values() if title_part in i["title"] or title_part == i["identifier"]]
        assert len(matches) == 1, f"{title_part!r} matches {[m['title'] for m in matches]}"
        return matches[0]

    def state_of(self, title_part: str) -> str:
        return next(s["name"] for s in self.states if s["id"] == self.find(title_part)["stateId"])

    def move(self, title_part: str, state_name: str) -> None:
        self.find(title_part)["stateId"] = next(s["id"] for s in self.states if s["name"] == state_name)

    def add_issue(self, title: str, *, description: str = "", parent: str | None = None,
                  state: str = "Todo", priority: int = 0) -> dict:
        project_id = next(iter(self.projects))
        self._issue_create({"input": {
            "teamId": self.teams[0]["id"], "title": title, "description": description,
            "projectId": project_id, "parentId": self.find(parent)["id"] if parent else None,
            "stateId": next(s["id"] for s in self.states if s["name"] == state), "priority": priority,
        }})
        return self.find(title)

    def comments_on(self, title_part: str) -> list[str]:
        identifier = self.find(title_part)["identifier"]
        return [body for ident, body in self.comments if ident == identifier]


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)


@pytest.fixture
def state(tmp_path) -> StateManager:
    root = tmp_path / "shop"
    root.mkdir()
    st = StateManager(str(root))
    spec_lib.attach(st)
    st.write_design_doc("# Shop — Design Document\n\nAn online shop.\n")
    return st


@pytest.fixture
def fake() -> FakeLinear:
    return FakeLinear()


def make_sync(state: StateManager, fake: FakeLinear, *, listen: bool = True) -> LinearSync:
    client = LinearClient("lin_api_test", transport=fake, sleep=lambda s: None)
    sync = LinearSync(state, client, LinearLink.load(state.linear_link_path), echo=lambda msg: None)
    if listen:
        state.add_story_listener(sync.on_story_event)
    return sync


def backlog() -> list[Story]:
    return [
        Story(id="DB-001", title="Product schema", content="As a dev, I want a schema.",
              acceptance_criteria=["Migration exists", "Indexes on sku"], priority=1, category="DB",
              complexity="small"),
        Story(id="API-001", title="List products", content="As a shopper, I want to list products.",
              acceptance_criteria=["GET /products returns 200"], priority=2, category="API",
              dependencies=["DB-001"]),
        Story(id="API-002", title="Product detail", content="As a shopper, I want product details.",
              priority=5, category="API", dependencies=["DB-001", "API-001"]),
        Story(id="UI-001", title="Old idea", content="Dropped.", category="UI", status=StoryStatus.skipped),
    ]


@pytest.fixture
def pushed(state, fake) -> LinearSync:
    """A backlog that has been pushed to Linear, with sync listening."""
    state.append_stories(backlog())
    sync = make_sync(state, fake)
    sync.push()
    fake.calls.clear()
    return sync


def stories_by_id(state: StateManager) -> dict[str, Story]:
    return {s.id: s for s in state.load_stories()}


# ── client ────────────────────────────────────────────────────────────


class TestClient:
    def test_personal_keys_are_sent_raw_and_oauth_tokens_as_bearer(self) -> None:
        assert LinearClient("lin_api_abc")._auth_header() == "lin_api_abc"
        assert LinearClient("lin_oauth_abc")._auth_header() == "Bearer lin_oauth_abc"
        assert LinearClient("Bearer xyz")._auth_header() == "Bearer xyz"

    def test_missing_key_fails_before_any_request(self, fake) -> None:
        with pytest.raises(LinearError, match="LINEAR_API_KEY"):
            LinearClient("", transport=fake).teams()
        assert fake.calls == []

    def test_key_resolution_prefers_the_environment(self, monkeypatch) -> None:
        config = {"linear_api_key": "from-config"}
        assert L.resolve_api_key(config.get) == "from-config"
        monkeypatch.setenv("LINEAR_API_KEY", "from-env")
        assert L.resolve_api_key(config.get) == "from-env"

    def test_rate_limits_are_retried(self, fake) -> None:
        fake.failures = [{"errors": [{"message": "slow down", "extensions": {"code": "RATELIMITED"}}]}] * 2
        slept: list[float] = []
        client = LinearClient("k", transport=fake, sleep=slept.append)
        assert [t["key"] for t in client.teams()] == ["ENG"]
        assert slept == [1.0, 2.0]

    def test_gives_up_after_max_retries(self, fake) -> None:
        fake.failures = [LinearError("down", retryable=True)] * 10
        client = LinearClient("k", transport=fake, sleep=lambda s: None, max_retries=2)
        with pytest.raises(LinearError, match="down"):
            client.teams()
        assert len(fake.calls) == 3

    def test_validation_errors_are_not_retried(self, fake) -> None:
        fake.failures = [{"errors": [{"message": "Argument Validation Error",
                                      "extensions": {"userPresentableMessage": "Title is too long"}}]}]
        with pytest.raises(LinearError, match="Title is too long"):
            LinearClient("k", transport=fake).teams()
        assert len(fake.calls) == 1

    def test_bad_key_gets_a_helpful_message(self, fake) -> None:
        fake.failures = [{"errors": [{"message": "nope", "extensions": {"code": "AUTHENTICATION_ERROR"}}]}]
        with pytest.raises(LinearError, match="rejected the API key"):
            LinearClient("k", transport=fake).teams()

    def test_http_transport_surfaces_graphql_errors_sent_with_4xx(self) -> None:
        body = json.dumps({"errors": [{"message": "bad", "extensions": {"code": "AUTHENTICATION_ERROR"}}]}).encode()
        err = urllib.error.HTTPError(L.API_URL, 401, "Unauthorized", {}, io.BytesIO(body))  # type: ignore[arg-type]
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(LinearError, match="rejected the API key"):
                LinearClient("k").teams()

    def test_http_transport_treats_network_failures_as_retryable(self) -> None:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")) as urlopen:
            with pytest.raises(LinearError, match="Could not reach Linear"):
                LinearClient("k", sleep=lambda s: None, max_retries=1).teams()
        assert urlopen.call_count == 2

    def test_mutations_with_an_unknown_outcome_are_not_repeated(self, fake) -> None:
        """A create that timed out may have gone through — sending it again makes a twin."""
        lost = LinearError("Could not reach Linear: timed out", retryable=True, outcome_unknown=True)
        client = LinearClient("k", transport=fake, sleep=lambda s: None)

        fake.failures = [lost]
        with pytest.raises(LinearError, match="timed out"):
            client.create_issue(team_id="team-eng", title="Once")
        assert fake.calls == ["issue_create"] and fake.issues == {}

        fake.calls.clear()
        fake.failures = [lost]
        assert [t["key"] for t in client.teams()] == ["ENG"], "reads are safe to repeat"
        assert fake.calls == ["teams", "teams"]

        fake.calls.clear()
        fake.failures = [{"errors": [{"message": "slow down", "extensions": {"code": "RATELIMITED"}}]}]
        client.create_issue(team_id="team-eng", title="Once")
        assert fake.calls == ["issue_create", "issue_create"], "a rejected mutation was not applied"
        assert len(fake.issues) == 1

    def test_http_transport_marks_lost_requests_as_outcome_unknown(self) -> None:
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")) as urlopen:
            with pytest.raises(LinearError) as caught:
                LinearClient("k", sleep=lambda s: None).create_comment("issue-1", "hello")
        assert caught.value.outcome_unknown and urlopen.call_count == 1

    def test_project_issues_follows_pagination(self, state) -> None:
        fake = FakeLinear(page_size=2)
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        sync.push()
        issues = sync.client.project_issues(sync.link.project_id)
        assert len(issues) == 5  # 2 parents + 3 stories
        assert fake.calls.count("project_issues") >= 3


# ── description round trip ────────────────────────────────────────────


class TestDescription:
    def test_round_trip(self) -> None:
        story = backlog()[0]
        text = render_description(story, ("shop", ["REQ-001"]))
        assert "pralph story `DB-001`" in text and "spec `shop`: REQ-001" in text
        assert parse_description(text) == (story.content, story.acceptance_criteria)

    def test_tolerates_edits_made_in_linear(self) -> None:
        text = (
            "As a dev, I want a schema.\n\n"
            "### Acceptance Criteria:\n\n"
            "* [x] Migration exists\n"
            "* Indexes on sku\n"
            "1. Rollback works\n\n"
            "## Notes\n\nTalk to ops first.\n\n"
            "***\n\npralph story `DB-001` · complexity: small\n"
        )
        content, criteria = parse_description(text)
        assert criteria == ["Migration exists", "Indexes on sku", "Rollback works"]
        assert content == "As a dev, I want a schema.\n\n## Notes\n\nTalk to ops first."

    def test_plain_description_written_by_a_person(self) -> None:
        assert parse_description("Fix the footer.\n\n---\n\nSee Figma.") == ("Fix the footer.\n\n---\n\nSee Figma.", [])
        assert parse_description("") == ("", [])
        assert parse_description(None) == ("", [])  # type: ignore[arg-type]


# ── push ──────────────────────────────────────────────────────────────


class TestPush:
    def test_plan_becomes_a_project_with_sub_issues(self, state, fake) -> None:
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        report = sync.push()

        project = next(iter(fake.projects.values()))
        assert project["name"] == "Shop", "named after the plan's title"
        assert "An online shop." in project["content"]
        assert len(project["description"]) <= 255
        assert report.project_created and report.epics_created == ["API", "DB"]

        db_epic, api_epic = fake.find("DB — Database"), fake.find("API — API")
        assert db_epic["parentId"] is None and db_epic["projectId"] == project["id"]
        schema, listing, detail = fake.find("DB-001"), fake.find("API-001"), fake.find("API-002")
        assert schema["title"] == "DB-001: Product schema"
        assert schema["parentId"] == db_epic["id"] and listing["parentId"] == api_epic["id"]
        assert schema["projectId"] == project["id"]
        assert (schema["priority"], listing["priority"], detail["priority"]) == (1, 2, 4)
        assert "- [ ] Migration exists" in schema["description"]
        assert fake.state_of("DB-001") == "Backlog", "open stories take the team's default state"

        assert set(fake.relations) == {
            (schema["id"], listing["id"]), (schema["id"], detail["id"]), (listing["id"], detail["id"]),
        }
        assert not [i for i in fake.issues.values() if "UI-001" in i["title"]], "dropped stories get no ticket"

    def test_stories_remember_their_ticket(self, state, fake) -> None:
        state.append_stories(backlog())
        make_sync(state, fake).push()
        link = stories_by_id(state)["API-001"].metadata["linear"]
        assert link["identifier"].startswith("ENG-") and link["url"].endswith(link["identifier"])
        assert link["state_type"] == "backlog"
        saved = LinearLink.load(state.linear_link_path)
        assert saved and saved.team_key == "ENG" and set(saved.epics) == {"API", "DB"}

    def test_push_is_idempotent(self, state, fake) -> None:
        state.append_stories(backlog())
        make_sync(state, fake).push()
        fake.calls.clear()
        report = make_sync(state, fake, listen=False).push()
        assert fake.mutations() == []
        assert not report.project_created and report.issues_created == [] and report.already_linked == 3

    def test_push_resumes_after_a_failure(self, state, fake) -> None:
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        # Fail in the middle: after the DB parent, DB-001 and the API parent exist.
        original = fake._issue_create
        created = {"n": 0}

        def flaky(v):
            created["n"] += 1
            if created["n"] == 4:
                raise LinearError("boom")
            return original(v)

        fake._issue_create = flaky  # type: ignore[method-assign]
        with pytest.raises(LinearError, match="boom"):
            sync.push()
        fake._issue_create = original  # type: ignore[method-assign]

        make_sync(state, fake, listen=False).push()
        assert len(fake.projects) == 1, "the project is not created twice"
        titles = sorted(i["title"] for i in fake.issues.values())
        assert titles == sorted(set(titles)), "no duplicate issues"
        assert len(titles) == 5 and len(fake.relations) == 3

    def test_push_adopts_tickets_it_created_without_learning_about_it(self, state, fake) -> None:
        """The create went through, the response got lost: the next push must not make twins."""
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        original = fake._issue_create
        created = {"n": 0}

        def lost_response(v):
            created["n"] += 1
            result = original(v)
            if created["n"] in (3, 4):  # the API parent, then API-001 on the retry
                raise LinearError("Could not reach Linear: timed out", retryable=True, outcome_unknown=True)
            return result

        fake._issue_create = lost_response  # type: ignore[method-assign]
        for _ in range(2):
            with pytest.raises(LinearError, match="timed out"):
                sync.push()
        fake._issue_create = original  # type: ignore[method-assign]

        report = make_sync(state, fake, listen=False).push()
        titles = sorted(i["title"] for i in fake.issues.values())
        assert titles == sorted(set(titles)) and len(titles) == 5, "no duplicate issues"
        assert [story_id for story_id, _ in report.issues_created] == ["API-002"]
        api_001 = fake.find("API-001")
        assert stories_by_id(state)["API-001"].metadata["linear"]["id"] == api_001["id"]
        assert api_001["parentId"] == fake.find("API — API")["id"]
        assert len(fake.relations) == 3

    def test_implemented_stories_are_created_as_done(self, state, fake) -> None:
        stories = backlog()
        stories[0].status = StoryStatus.implemented
        state.append_stories(stories)
        make_sync(state, fake).push()
        assert fake.state_of("DB-001") == "Done"
        assert fake.state_of("DB — Database") == "Done", "parent closes with its last sub-issue"
        assert fake.state_of("API — API") == "Backlog"

    def test_dry_run_makes_no_requests(self, state, fake) -> None:
        state.append_stories(backlog())
        report = make_sync(state, fake).push(dry_run=True)
        assert fake.calls == []
        assert report.project_name == "Shop" and report.project_created
        assert report.epics_created == ["API", "DB"]
        assert [sid for sid, _ in report.issues_created] == ["DB-001", "API-001", "API-002"]
        assert report.relations_created == 3
        assert not state.linear_link_path.exists()

    def test_team_must_be_chosen_when_there_are_several(self, state) -> None:
        fake = FakeLinear(teams=["ENG", "OPS"])
        state.append_stories(backlog())
        with pytest.raises(LinearError, match=r"choose one with --team \(available: ENG, OPS\)"):
            make_sync(state, fake).push()
        make_sync(state, fake).push(team_key="ops")
        assert fake.find("DB-001")["identifier"].startswith("OPS-")

    def test_unknown_team(self, state, fake) -> None:
        state.append_stories(backlog())
        with pytest.raises(LinearError, match="'NOPE' not found"):
            make_sync(state, fake).push(team_key="NOPE")

    def test_spec_mode_project_points_at_the_repo(self, state, fake) -> None:
        spec_dir = state.project_dir / "specs" / "shop"
        spec_dir.mkdir(parents=True)
        (spec_dir / "spec.md").write_text("# Shop — Specification\n\n### REQ-001: Browse\nUsers browse.\n")
        story = Story(id="UI-001", title="Browse page", content="c", category="UI",
                      metadata={"spec": "shop", "requirements": ["REQ-001"]})
        state.append_stories([story])
        make_sync(state, fake).push()
        project = next(iter(fake.projects.values()))
        assert project["name"] == "Shop"
        assert "specification lives in the repository (specs/shop/spec.md)" in project["content"]
        assert "spec `shop`: REQ-001" in fake.find("UI-001")["description"]


# ── status changes → Linear ───────────────────────────────────────────


class TestStatusSync:
    def test_story_lifecycle_moves_the_ticket(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        assert fake.state_of("DB-001") == "In Progress"
        assert fake.state_of("DB — Database") == "In Progress", "parent starts with its first sub-issue"
        assert fake.comments == []

        state.mark_story_status("DB-001", StoryStatus.implemented, summary="Added the migration",
                                extra={"files_changed": ["db/001.sql"]})
        assert fake.state_of("DB-001") == "Done"
        assert fake.state_of("DB — Database") == "Done"
        (comment,) = fake.comments_on("DB-001")
        assert "Added the migration" in comment and "`db/001.sql`" in comment
        assert stories_by_id(state)["DB-001"].metadata["linear"]["state_type"] == "completed"

    def test_done_means_reviewed(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.implemented, summary="Built it",
                                extra={"files_changed": ["api.py"]}, review_pending=True)
        assert fake.state_of("API-001") == "In Progress", "not Done before the review verdict"
        assert fake.comments == []

        state.notify_review_passed("API-001", "looks good")
        assert fake.state_of("API-001") == "Done"
        (comment,) = fake.comments_on("API-001")
        assert "Built it" in comment and "`api.py`" in comment

    def test_review_rejection_posts_the_feedback(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.implemented, summary="Done", review_pending=True)
        state.write_review_feedback("API-001", "- **[critical]** Pagination is missing")
        state.mark_story_status("API-001", StoryStatus.rework, summary="Review rejected")
        assert fake.state_of("API-001") == "In Progress"
        (comment,) = fake.comments_on("API-001")
        assert "Pagination is missing" in comment, "one comment: the ticket never claimed to be done"

        state.notify_review_passed("API-001")  # a stale verdict must not close a story in rework
        assert fake.state_of("API-001") == "In Progress"

    def test_crash_before_the_review_verdict_is_caught_up_by_pull(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.implemented, summary="Built it", review_pending=True)
        restarted = make_sync(StateManager(str(state.project_dir)), fake, listen=False)
        restarted.pull()
        assert fake.state_of("API-001") == "Done"

    def test_reopened_sub_issue_reopens_the_parent(self, state, fake, pushed) -> None:
        for sid in ("API-001", "API-002"):
            state.mark_story_status(sid, StoryStatus.implemented)
        assert fake.state_of("API — API") == "Done"
        state.mark_story_status("API-001", StoryStatus.rework, summary="Review rejected")
        assert fake.state_of("API — API") == "In Progress"

    def test_failure_stays_in_progress_with_a_comment(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.error, error_reason="tests would not pass")
        assert fake.state_of("API-001") == "In Progress"
        assert "tests would not pass" in fake.comments_on("API-001")[0]

        state.reset_error_stories()
        assert fake.state_of("API-001") == "Todo"

    def test_interrupted_story_goes_back_to_todo(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.pending, summary="User interrupted")
        assert fake.state_of("API-001") == "Todo"
        assert fake.comments == []

    def test_crash_recovery_releases_the_ticket(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.recover_orphaned_stories()
        assert fake.state_of("API-001") == "Todo"

    def test_duplicate_and_skipped_close_the_ticket(self, state, fake, pushed) -> None:
        state.mark_story_status("API-002", StoryStatus.duplicate, extra={"duplicate_of": "API-001"})
        # Not "Duplicate": Linear only allows that state together with a duplicate-of relation.
        assert fake.state_of("API-002") == "Canceled"
        assert "duplicate of API-001" in fake.comments_on("API-002")[0]
        state.mark_story_status("API-001", StoryStatus.skipped, summary="Refined into: API-003")
        assert fake.state_of("API-001") == "Canceled"
        assert "Refined into: API-003" in fake.comments_on("API-001")[0]

    def test_new_stories_get_a_ticket(self, state, fake, pushed) -> None:
        state.append_stories([Story(id="API-003", title="Search", content="c", category="API",
                                    dependencies=["API-001"])])
        search = fake.find("API-003")
        assert search["parentId"] == fake.find("API — API")["id"]
        assert (fake.find("API-001")["id"], search["id"]) in fake.relations
        assert len(fake.relations) == 4, "existing relations are not created again"

    def test_dependencies_within_a_new_batch_survive(self, state, fake, pushed) -> None:
        """A story can depend on one that is added after it in the same batch."""
        state.append_stories([
            Story(id="API-003", title="Search", content="c", category="API", dependencies=["API-004"]),
            Story(id="API-004", title="Search index", content="c", category="API"),
        ])
        assert (fake.find("API-004")["id"], fake.find("API-003")["id"]) in fake.relations
        assert len(fake.relations) == 4
        pushed.pull()
        assert stories_by_id(state)["API-003"].dependencies == ["API-004"]

    def test_new_category_gets_a_parent_issue(self, state, fake, pushed) -> None:
        state.append_stories([Story(id="PAY-001", title="Checkout", content="c", category="PAY")])
        assert fake.find("PAY-001")["parentId"] == fake.find("PAY — Pay stories")["id"]

    def test_linear_being_down_does_not_break_the_backlog(self, state, fake, pushed) -> None:
        fake.failures = [LinearError("offline", retryable=False)]
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        assert stories_by_id(state)["DB-001"].status == StoryStatus.in_progress
        assert fake.state_of("DB-001") == "Backlog"


# ── pull ──────────────────────────────────────────────────────────────


class TestPull:
    def test_no_changes_is_a_no_op(self, state, fake, pushed) -> None:
        before = state.stories_path.read_text()
        report = pushed.pull()
        assert state.stories_path.read_text() == before
        assert (report.total, report.new, report.updated, report.status_changes, report.pushed) == (3, [], 0, [], 0)
        assert fake.mutations() == []

    def test_ticket_edits_flow_into_the_story(self, state, fake, pushed) -> None:
        issue = fake.find("API-001")
        issue["title"] = "API-001: List products with paging"
        issue["priority"] = 1
        issue["description"] = "As a shopper, I want pages.\n\n## Acceptance criteria\n\n- [ ] 20 per page\n- [ ] Link header"
        fake.relations.remove((fake.find("DB-001")["id"], issue["id"]))

        report = pushed.pull()
        story = stories_by_id(state)["API-001"]
        assert story.title == "List products with paging"
        assert story.priority == 1
        assert story.content == "As a shopper, I want pages."
        assert story.acceptance_criteria == ["20 per page", "Link header"]
        assert story.dependencies == []
        assert report.updated == 1

    def test_priority_five_survives_linears_four_levels(self, state, fake, pushed) -> None:
        pushed.pull()
        assert stories_by_id(state)["API-002"].priority == 5

    def test_issues_created_in_linear_become_stories(self, state, fake, pushed) -> None:
        fake.add_issue("Rate limit the API", description="Too many requests.\n\n## Acceptance criteria\n- 429 after 100/min",
                       parent="API — API", priority=2)
        fake.add_issue("API-009: Health endpoint", parent="API — API")
        fake.add_issue("Write launch post")
        report = pushed.pull()

        assert len(report.new) == 3
        by_title = {s.title: s for s in state.load_stories()}
        rate = by_title["Rate limit the API"]
        assert rate.id == fake.find("Rate limit")["identifier"], "no pralph ID in the title → Linear's identifier"
        assert (rate.category, rate.priority, rate.source) == ("API", 2, "linear")
        assert rate.acceptance_criteria == ["429 after 100/min"]
        assert by_title["Health endpoint"].id == "API-009"
        assert by_title["Write launch post"].category == "ENG", "top-level issue: team key as category"
        assert all(s.status == StoryStatus.pending for s in report.new)

    def test_blocking_relations_become_dependencies(self, state, fake, pushed) -> None:
        new = fake.add_issue("Cache products", parent="API — API")
        fake.relations.append((fake.find("API-001")["id"], new["id"]))
        pushed.pull()
        assert stories_by_id(state)[new["identifier"]].dependencies == ["API-001"]

    def test_relation_removed_in_linear_removes_the_dependency(self, state, fake, pushed) -> None:
        fake.relations.remove((fake.find("API-001")["id"], fake.find("API-002")["id"]))
        pushed.pull()
        assert stories_by_id(state)["API-002"].dependencies == ["DB-001"]
        assert len(fake.relations) == 2, "and pralph does not put it back"

    def test_dependency_that_never_reached_linear_is_kept_and_retried(self, state, fake, pushed) -> None:
        original = fake._relation_create

        def refuse(v):
            raise LinearError("relation refused")

        fake._relation_create = refuse  # type: ignore[method-assign]
        state.append_stories([Story(id="API-003", title="Search", content="c", category="API",
                                    dependencies=["API-001"])])
        fake._relation_create = original  # type: ignore[method-assign]
        assert len(fake.relations) == 3, "the ticket exists, its relation does not"
        assert stories_by_id(state)["API-003"].metadata["linear"]["pending_deps"] == ["API-001"]

        report = pushed.pull()
        search = stories_by_id(state)["API-003"]
        assert search.dependencies == ["API-001"], "absent in Linear, but never there: not a removal"
        assert (fake.find("API-001")["id"], fake.find("API-003")["id"]) in fake.relations
        assert report.relations_created == 1 and "pending_deps" not in search.metadata["linear"]

        before = state.stories_path.read_text()
        pushed.pull()
        assert state.stories_path.read_text() == before and len(fake.relations) == 4

    def test_incomplete_relation_list_does_not_drop_dependencies(self, state, fake, pushed) -> None:
        fake.truncated.add(fake.find("API-002")["id"])
        pushed.pull()
        assert sorted(stories_by_id(state)["API-002"].dependencies) == ["API-001", "DB-001"]
        assert len(fake.relations) == 3, "nothing is created on a guess"

    def test_marked_as_duplicate_in_linear(self, state, fake, pushed) -> None:
        """Linear's "Duplicate" state has its own type — it is closed, not open."""
        fake.move("API-002", "Duplicate")
        report = pushed.pull()
        assert stories_by_id(state)["API-002"].status == StoryStatus.duplicate
        assert report.status_changes == [("API-002", "pending", "duplicate", "Marked as duplicate in Linear")]
        fake.calls.clear()
        pushed.pull()
        assert fake.mutations() == [], "nothing to push back"
        assert "API-002" not in {s.id for s in state.get_actionable_stories()}

    def test_canceled_and_completed_in_linear(self, state, fake, pushed) -> None:
        fake.move("API-002", "Canceled")
        fake.move("DB-001", "Done")
        report = pushed.pull()
        stories = stories_by_id(state)
        assert stories["API-002"].status == StoryStatus.skipped
        assert stories["DB-001"].status == StoryStatus.implemented
        assert ("API-002", "pending", "skipped", "Canceled in Linear") in report.status_changes
        assert fake.mutations() == ["issue_update"], "only the parent issue is rolled up"
        assert fake.state_of("DB — Database") == "Done"
        history = [json.loads(line) for line in state.status_path.read_text().splitlines()]
        assert {"story_id": "DB-001", "status": "implemented", "summary": "Completed in Linear", "source": "linear"} in history

    def test_ticket_taken_by_a_person_is_left_alone(self, state, fake, pushed) -> None:
        fake.move("API-001", "In Progress")
        pushed.pull()
        story = stories_by_id(state)["API-001"]
        assert story.status == StoryStatus.external and story.metadata["linear"]["claimed"] is True
        assert story.id not in [s.id for s in state.get_actionable_stories()]

        pushed.pull()
        assert fake.state_of("API-001") == "In Progress", "pralph does not fight over the ticket"

        fake.move("API-001", "Todo")  # they put it back
        pushed.pull()
        story = stories_by_id(state)["API-001"]
        assert story.status == StoryStatus.pending and "claimed" not in story.metadata["linear"]

    def test_person_finishing_a_claimed_ticket(self, state, fake, pushed) -> None:
        fake.move("API-001", "In Progress")
        pushed.pull()
        fake.move("API-001", "Done")
        pushed.pull()
        assert stories_by_id(state)["API-001"].status == StoryStatus.implemented

    def test_moving_a_failed_ticket_to_todo_requeues_it(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.error, error_reason="flaky")
        pushed.pull()
        assert stories_by_id(state)["API-001"].status == StoryStatus.error, "still failed until someone acts"

        fake.move("API-001", "Todo")
        pushed.pull()
        story = stories_by_id(state)["API-001"]
        assert story.status == StoryStatus.pending and "error_reason" not in story.metadata

    def test_reopening_a_done_ticket(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.implemented)
        fake.move("DB-001", "Todo")
        report = pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.pending
        assert ("DB-001", "implemented", "pending", "Reopened in Linear") in report.status_changes

    def test_rework_keeps_its_status_and_feedback(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.rework, summary="Review rejected")
        fake.move("DB-001", "In Review")  # same state class — not a change
        pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.rework

    def test_update_that_never_reached_linear_is_retried(self, state, fake, pushed) -> None:
        fake.failures = [LinearError("offline")]
        state.mark_story_status("DB-001", StoryStatus.implemented, summary="done")
        assert fake.state_of("DB-001") == "Backlog"

        report = pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.implemented, "local progress is not lost"
        assert fake.state_of("DB-001") == "Done" and report.pushed == 1

    def test_status_edited_in_the_viewer_is_pushed(self, state, fake, pushed) -> None:
        stories = state.load_stories()
        next(s for s in stories if s.id == "API-002").status = StoryStatus.skipped
        state.replace_stories(stories)  # what the viewer does
        pushed.pull()
        assert fake.state_of("API-002") == "Canceled"

    def test_issue_removed_from_the_project(self, state, fake, pushed) -> None:
        fake.find("API-002")["archived"] = True
        fake.find("API-001")["trashed"] = True
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.implemented)
        fake.find("DB-001")["archived"] = True

        pushed.pull()
        stories = stories_by_id(state)
        assert stories["API-002"].status == StoryStatus.skipped
        assert stories["API-001"].status == StoryStatus.skipped
        assert stories["DB-001"].status == StoryStatus.implemented, "finished work stays finished"

        fake.find("API-002")["archived"] = False  # restored
        pushed.pull()
        assert stories_by_id(state)["API-002"].status == StoryStatus.pending

    def test_story_split_into_sub_issues_in_linear(self, state, fake, pushed) -> None:
        fake.add_issue("Detail endpoint", parent="API-002")
        fake.add_issue("Detail caching", parent="API-002")
        pushed.pull()
        stories = stories_by_id(state)
        assert stories["API-002"].status == StoryStatus.skipped, "the sub-issues are the work now"
        assert stories["API-002"].metadata["linear"]["container"] is True
        assert {s.title for s in state.get_actionable_stories()} >= {"Detail endpoint", "Detail caching"}
        assert stories[fake.find("Detail caching")["identifier"]].category == "API"

        pushed.pull()
        assert fake.state_of("API-002") == "Backlog", "pralph leaves the split ticket's state to people"

    def test_fresh_checkout_rebuilds_the_backlog_from_linear(self, state, fake, pushed, tmp_path) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.implemented)
        url = next(iter(fake.projects.values()))["url"]

        clone = tmp_path / "clone"
        clone.mkdir()
        other = StateManager(str(clone))
        sync = make_sync(other, fake)
        sync.link_project(url)
        report = sync.pull()

        assert sync.link.team_key == "ENG" and set(sync.link.epics) == {"API", "DB"}
        rebuilt = stories_by_id(other)
        assert set(rebuilt) == {"DB-001", "API-001", "API-002"} and len(report.new) == 3
        assert rebuilt["DB-001"].status == StoryStatus.implemented
        assert rebuilt["API-002"].dependencies == ["DB-001", "API-001"]
        assert rebuilt["API-001"].acceptance_criteria == ["GET /products returns 200"]
        assert rebuilt["API-001"].category == "API"

    @pytest.mark.parametrize("suffix", ["", "/", "/overview", "/issues?layout=list", "#updates"])
    def test_link_by_url_as_copied_from_the_browser(self, state, fake, pushed, tmp_path, suffix) -> None:
        project = next(iter(fake.projects.values()))
        sync = LinearSync(StateManager(str(tmp_path / "clone")), pushed.client, echo=lambda m: None)
        sync.link_project(project["url"] + suffix)
        assert sync.link.project_id == project["id"]

    def test_link_by_name_and_by_id(self, state, fake, pushed, tmp_path) -> None:
        project = next(iter(fake.projects.values()))
        for reference in ("shop", project["id"]):
            sync = LinearSync(StateManager(str(tmp_path / reference[:4])), pushed.client, echo=lambda m: None)
            sync.link_project(reference)
            assert sync.link.project_id == project["id"]
        with pytest.raises(LinearError, match="No Linear project found"):
            LinearSync(StateManager(str(tmp_path / "x")), pushed.client).link_project("Nope")

    def test_pull_requires_a_link(self, state, fake) -> None:
        with pytest.raises(LinearError, match="Not linked"):
            make_sync(state, fake).pull()


# ── execution off Linear ──────────────────────────────────────────────


def fake_claude(on_story=None, reject_once: str | None = None):
    """run_with_retry stand-in: implements everything; approves it, except that a
    story whose title is ``reject_once`` fails its first review."""
    implemented: list[str] = []
    rejected: set[str] = set()

    def run(prompt: str, **kwargs) -> ClaudeResult:
        if "# Code Review Mode" in prompt:
            title = prompt.split("**Title:** ", 1)[1].splitlines()[0]
            if title == reject_once and title not in rejected:
                rejected.add(title)
                return ClaudeResult(success=True, result=json.dumps({
                    "approved": False, "feedback": "Not there yet",
                    "issues": [{"severity": "major", "description": "No tests"}]}))
            return ClaudeResult(success=True, result='{"approved": true, "feedback": "ok", "issues": []}')
        title = prompt.split("**Title:** ", 1)[1].splitlines()[0]
        implemented.append(title)
        if on_story:
            on_story(title)
        return ClaudeResult(success=True, result=json.dumps(
            {"status": "implemented", "summary": f"Built {title}", "files_changed": ["app.py"]}))

    return patch("pralph.loop.run_with_retry", side_effect=run), implemented


class TestImplementOffLinear:
    def test_works_through_the_tickets(self, state, fake, pushed) -> None:
        patcher, implemented = fake_claude()
        with patcher:
            ps = run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=pushed)
        assert ps.completion_reason == "all_stories_done"
        assert implemented == ["Product schema", "List products", "Product detail"]
        assert {fake.state_of(t) for t in ("DB-001", "API-001", "API-002")} == {"Done"}
        assert fake.state_of("API — API") == "Done"
        assert "Built List products" in fake.comments_on("API-001")[0]

    def test_rejected_story_is_reworked_before_the_ticket_closes(self, state, fake, pushed) -> None:
        seen: list[tuple[str, str]] = []
        patcher, implemented = fake_claude(
            on_story=lambda title: seen.append((title, fake.state_of("DB-001"))),
            reject_once="Product schema",
        )
        with patcher:
            run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=pushed)
        assert implemented[:2] == ["Product schema", "Product schema"], "rework comes first"
        assert seen[1] == ("Product schema", "In Progress"), "never Done between attempt and rework"
        comments = fake.comments_on("DB-001")
        assert len(comments) == 2 and "No tests" in comments[0] and "implemented this story" in comments[1]
        assert fake.state_of("DB-001") == "Done"

    def test_changes_made_in_linear_during_the_run_are_respected(self, state, fake, pushed) -> None:
        def meanwhile(title: str) -> None:
            if title == "Product schema":
                fake.move("API-002", "Canceled")                     # dropped by the PM
                fake.move("API-001", "In Progress")                  # a colleague takes this one
                fake.add_issue("Hotfix: broken images", priority=1)  # urgent new ticket
                fake.find("DB-001")["title"] = "DB-001: Product schema v2"

        patcher, implemented = fake_claude(meanwhile)
        with patcher:
            ps = run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=pushed)

        assert implemented == ["Product schema", "Hotfix: broken images"]
        assert ps.completion_reason == "all_stories_done"
        assert fake.state_of("API-001") == "In Progress" and fake.comments_on("API-001") == []
        assert fake.state_of("API-002") == "Canceled"
        assert fake.state_of("Hotfix") == "Done"
        stories = stories_by_id(state)
        assert stories["DB-001"].status == StoryStatus.implemented
        assert stories["DB-001"].title == "Product schema v2"

    def test_unpushed_stories_get_tickets_before_work_starts(self, state, fake) -> None:
        state.append_stories(backlog()[:1])
        sync = make_sync(state, fake)
        sync.push()
        # A story added while Linear was unreachable.
        fake.failures = [LinearError("offline")]
        state.append_stories([Story(id="DB-002", title="Seed data", content="c", category="DB")])
        assert not linear_sync.story_link(stories_by_id(state)["DB-002"])

        patcher, implemented = fake_claude()
        with patcher:
            run_implement_loop(state, max_iterations=5, cooldown=0, phase1=False, tracker=sync)
        assert implemented == ["Product schema", "Seed data"]
        assert fake.state_of("DB-002") == "Done"

    def test_linear_outage_does_not_stop_the_run(self, state, fake, pushed) -> None:
        patcher, implemented = fake_claude()
        fake.failures = [LinearError("offline")] * 3
        with patcher:
            ps = run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=pushed)
        assert len(implemented) == 3 and ps.completion_reason == "all_stories_done"
        assert {fake.state_of(t) for t in ("DB-001", "API-001", "API-002")} == {"Done"}, \
            "updates missed during the outage are caught up"

    def test_implement_prompt_names_the_ticket(self, state, fake, pushed) -> None:
        story = stories_by_id(state)["API-001"]
        identifier = story.metadata["linear"]["identifier"]
        prompt = assemble_implement_prompt(state, story)
        assert f"tracked in Linear as **{identifier}**" in prompt
        assert f"Mention `{identifier}` in your commit message" in prompt


# ── wiring / CLI ──────────────────────────────────────────────────────


class TestWiring:
    def test_attach_is_inert_without_a_link(self, state) -> None:
        assert linear_sync.attach(state) is None
        with pytest.raises(LinearError, match="not linked"):
            linear_sync.attach(state, required=True)

    def test_attach_needs_a_key_for_linked_projects(self, state, fake, pushed, monkeypatch) -> None:
        other = StateManager(str(state.project_dir))
        messages: list[str] = []
        assert linear_sync.attach(other, echo=messages.append) is None
        assert "LINEAR_API_KEY is not set" in messages[0]
        with pytest.raises(LinearError, match="LINEAR_API_KEY"):
            linear_sync.attach(other, required=True)

        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
        sync = linear_sync.attach(other)
        assert sync is not None and sync.link.team_key == "ENG"

    def test_cli_dry_run_needs_no_key(self, state) -> None:
        state.append_stories(backlog())
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "push", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "project: Shop (would create)" in result.output
        assert "would create: 3 sub-issues, 3 blocking relations" in result.output

    def test_cli_push_without_key_explains_what_to_do(self, state) -> None:
        state.append_stories(backlog())
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "push"])
        assert result.exit_code == 1 and "LINEAR_API_KEY is not set" in result.output

    def test_cli_implement_refuses_to_bypass_a_linked_project_silently(self, state, fake, pushed) -> None:
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "implement"])
        assert result.exit_code == 1 and "LINEAR_API_KEY" in result.output

    def test_cli_status(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "status"])
        assert result.exit_code == 0, result.output
        assert "project: Shop https://linear.app/acme/project/shop-" in result.output
        assert "issues: 3 (2 Backlog, 1 In Progress)" in result.output
