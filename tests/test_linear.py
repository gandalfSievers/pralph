"""Tests for the Linear mode: plan → Linear project, execution off the tickets.

Linear itself is replaced by ``FakeLinear``, an in-memory workspace that
answers the MCP tool calls ``pralph.linear`` makes.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pralph import linear as L
from pralph import linear_sync
from pralph import ship as ship_lib
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
    """In-memory Linear behind the MCP tools pralph calls.

    Answers in the shapes Linear's MCP server uses: issues are named by
    identifier (``id``) and carry their UUID as ``uuid``, labels come as bare
    names, parents as identifiers, and blocking relations only from get_issue.
    """

    OPERATIONS = {
        "whoami": "viewer", "list_teams": "teams", "list_issue_statuses": "states",
        "get_project": "project", "list_projects": "projects", "list_issues": "project_issues",
        "get_issue": "issue", "list_milestones": "milestones", "list_issue_labels": "labels",
        "list_users": "users", "save_project": "project_create", "save_comment": "comment_create",
    }

    def __init__(self, *, teams: list[str] = ("ENG",), page_size: int | None = None) -> None:
        self.teams = [{"id": f"team-{k.lower()}", "key": k, "name": f"{k} team"} for k in teams]
        # Listed in no particular order, like the real tool does.
        self.states = [
            {"id": "st-done", "name": "Done", "type": "completed"},
            # Its own type in Linear, and only reachable through a duplicate-of relation.
            {"id": "st-dup", "name": "Duplicate", "type": "duplicate"},
            {"id": "st-todo", "name": "Todo", "type": "unstarted"},
            {"id": "st-review", "name": "In Review", "type": "started"},
            {"id": "st-progress", "name": "In Progress", "type": "started"},
            {"id": "st-backlog", "name": "Backlog", "type": "backlog"},
            {"id": "st-canceled", "name": "Canceled", "type": "canceled"},
        ]
        self.projects: dict[str, dict] = {}
        self.milestones = [{"id": "ms-base", "name": "Base work"}, {"id": "ms-launch", "name": "Launch"}]
        self.labels = [
            {"id": "lb-shop", "name": "repo:shop", "team": None},
            {"id": "lb-shop-old", "name": "repo:shop-legacy", "team": None},
            {"id": "lb-ops", "name": "repo:ops", "team": "team-ops"},
            {"id": "lb-app", "name": "repo:app", "team": "team-eng"},
        ]
        self.users = [
            {"id": "u1", "name": "Dev", "displayName": "dev", "email": "dev@example.com"},
            {"id": "u2", "name": "Sam Lee", "displayName": "sam", "email": "sam@example.com"},
            {"id": "u3", "name": "Sam Lee", "displayName": "samuel", "email": "samuel@example.com"},
        ]
        self.issues: dict[str, dict] = {}       # id → issue
        self.relations: list[tuple[str, str]] = []  # (blocker id, blocked id)
        self.comments: list[tuple[str, str]] = []   # (issue identifier, body)
        self.page_size = page_size
        self.calls: list[str] = []
        self.mutation_log: list[tuple[str, dict]] = []
        self.failures: list[Exception] = []  # consumed one per call
        self.unreadable: set[str] = set()  # issue ids whose relations cannot be read
        self._counter = 0

    # -- transport --

    def __call__(self, tool: str, args: dict) -> object:
        unknown = set(args) - set(L.TOOLS[tool])
        assert not unknown, f"{tool} is called with arguments the client does not declare: {unknown}"
        if tool == "save_issue":
            name = "issue_create" if "id" not in args else "relation_create" if "blockedBy" in args else "issue_update"
        else:
            name = self.OPERATIONS[tool]
        self.calls.append(name)
        if name.endswith(("_create", "_update")):
            self.mutation_log.append((name, args))
        if self.failures:
            raise self.failures.pop(0)
        return getattr(self, f"_{name}")(args)

    def mutations(self) -> list[str]:
        return [c for c in self.calls if c.endswith(("_create", "_update"))]

    def _lookup(self, reference: str) -> dict:
        for issue in self.issues.values():
            if reference in (issue["id"], issue["identifier"]):
                return issue
        raise LinearError(f'Linear: Error: Could not find issue "{reference}"')

    # -- reads --

    def _viewer(self, a: dict) -> dict:
        return {"user": {"id": "u1", "name": "Dev", "email": "dev@example.com"},
                "workspace": {"id": "w1", "name": "Acme", "url": "https://linear.app/acme"}}

    def _teams(self, a: dict) -> dict:
        return {"teams": self.teams, "hasNextPage": False}

    def _states(self, a: dict) -> list:
        assert any(t["id"] == a["team"] for t in self.teams)
        return list(self.states)

    def _project_node(self, p: dict) -> dict:
        return {"id": p["identifier"], "uuid": p["id"], "name": p["name"], "url": p["url"],
                "summary": p["description"], "description": p["content"],
                "teams": [t for t in self.teams if t["id"] in p["teamIds"]]}

    def _find_project(self, reference: str) -> dict:
        for p in self.projects.values():
            if reference in (p["id"], p["identifier"], p["slugId"], p["url"].rsplit("/", 1)[-1], p["name"]):
                return p
        raise LinearError(f'Linear: Error: Could not find project "{reference}"')

    def _project(self, a: dict) -> dict:
        return self._project_node(self._find_project(a["query"]))

    def _projects(self, a: dict) -> dict:
        found = [p for p in self.projects.values() if a["query"].lower() in p["name"].lower()]
        return {"projects": [self._project_node(p) for p in found], "hasNextPage": False}

    def _issue_node(self, issue: dict) -> dict:
        state = next(s for s in self.states if s["id"] == issue["stateId"])
        milestone = next((m for m in self.milestones if m["id"] == issue.get("projectMilestoneId")), None)
        blockers = sorted(blocker for blocker, blocked in self.relations if blocked == issue["id"])
        return {
            "id": issue["identifier"], "uuid": issue["id"], "title": issue["title"],
            "description": issue["description"] or None,
            "priority": {"value": issue["priority"], "name": "?"}, "url": issue["url"],
            "status": state["name"], "statusType": state["type"],
            "labels": [lb["name"] for lb in self.labels if lb["id"] in (issue.get("labelIds") or [])],
            "assigneeId": issue.get("assigneeId"),
            "projectMilestone": dict(milestone) if milestone else None,
            "parentId": self.issues[issue["parentId"]]["identifier"] if issue.get("parentId") else None,
            "archivedAt": "2026-01-01T00:00:00.000Z" if issue.get("trashed") else None,
            "gitBranchName": f"dev/{issue['identifier'].lower()}",
            # Linear touches updatedAt on every change, relations included.
            "updatedAt": hashlib.sha1(json.dumps([issue, blockers], sort_keys=True).encode()).hexdigest()[:12],
        }

    def _project_issues(self, a: dict) -> dict:
        project = self._find_project(a["project"])
        assert set(a["fields"]) == set(L.ISSUE_FIELDS)
        issues = [i for i in self.issues.values() if i["projectId"] == project["id"] and not i.get("archived")]
        size = self.page_size or a["limit"]
        start = int(a.get("cursor") or 0)
        page = issues[start:start + size]
        end = start + len(page)
        answer = {"issues": [self._issue_node(i) for i in page], "hasNextPage": end < len(issues)}
        if answer["hasNextPage"]:
            answer["cursor"] = str(end)
        return answer

    def _issue(self, a: dict) -> dict:
        issue = self._lookup(a["id"])
        if issue["id"] in self.unreadable:
            raise LinearError("Linear: relations are not available right now")
        assert a["fields"] == ["id", "relations"]
        blockers = [self.issues[blocker] for blocker, blocked in self.relations if blocked == issue["id"]]
        return {"id": issue["identifier"], "relations": {
            "blocks": [], "relatedTo": [], "duplicateOf": None,
            "blockedBy": [{"id": b["identifier"], "title": b["title"]} for b in blockers],
        }}

    def _milestones(self, a: dict) -> dict:
        self._find_project(a["project"])
        return {"milestones": [dict(m, progress=0) for m in self.milestones]}

    def _labels(self, a: dict) -> dict:
        found = [
            {"id": lb["id"], "name": lb["name"], "archivedAt": None} for lb in self.labels
            if a["name"].lower() in lb["name"].lower() and lb["team"] in (None, a.get("team"))
        ]
        return {"labels": found, "hasNextPage": False}

    def _users(self, a: dict) -> dict:
        query = a["query"].lower()
        found = [u for u in self.users if any(query in str(u[k]).lower() for k in ("name", "displayName", "email"))]
        return {"users": found, "hasNextPage": False}

    # -- writes --

    def _project_create(self, a: dict) -> dict:
        self._counter += 1
        pid = f"00000000-0000-4000-8000-{self._counter:012d}"
        slug = f"{self._counter:012x}"
        self.projects[pid] = {
            "id": pid, "identifier": f"P-ENG-{self._counter}", "name": a["name"], "slugId": slug,
            "teamIds": a["addTeams"], "description": a.get("summary", ""), "content": a.get("description", ""),
            "url": f"https://linear.app/acme/project/{a['name'].lower().replace(' ', '-')}-{slug}",
        }
        return self._project_node(self.projects[pid])

    def _state_id(self, reference: str | None) -> str:
        if not reference:
            return "st-backlog"  # team default for new issues
        return next(s["id"] for s in self.states if reference in (s["id"], s["name"]))

    def _issue_create(self, a: dict) -> dict:
        self._counter += 1
        team = next(t for t in self.teams if t["id"] == a["team"])
        identifier = f"{team['key']}-{self._counter}"
        issue = {
            "id": f"issue-{self._counter}", "identifier": identifier, "title": a["title"],
            "description": a.get("description", ""), "priority": a.get("priority", 0),
            "projectId": self._find_project(a["project"])["id"] if a.get("project") else None,
            "parentId": self._lookup(a["parentId"])["id"] if a.get("parentId") else None,
            "assigneeId": a.get("assignee"), "projectMilestoneId": a.get("milestone"),
            "labelIds": [next(lb["id"] for lb in self.labels if ref in (lb["id"], lb["name"]))
                         for ref in a.get("labels") or []],
            "stateId": self._state_id(a.get("state")),
            "url": f"https://linear.app/acme/issue/{identifier}",
        }
        self.issues[issue["id"]] = issue
        return self._issue_node(issue)

    def _issue_update(self, a: dict) -> dict:
        issue = self._lookup(a["id"])
        if a.get("state") == "st-dup":
            raise LinearError("Linear: Issues can only be moved to a duplicate state when a duplicate issue relation exists.")
        issue["stateId"] = self._state_id(a["state"])
        return self._issue_node(issue)

    def _comment_create(self, a: dict) -> dict:
        self.comments.append((self._lookup(a["issueId"])["identifier"], a["body"]))
        return {"id": f"comment-{len(self.comments)}", "body": a["body"]}

    def _relation_create(self, a: dict) -> dict:
        blocked = self._lookup(a["id"])
        for reference in a["blockedBy"]:
            pair = (self._lookup(reference)["id"], blocked["id"])
            if pair not in self.relations:  # append-only, and saying it twice is fine
                self.relations.append(pair)
        return self._issue_node(blocked)

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
                  state: str = "Todo", priority: int = 0, assignee: str | None = None,
                  milestone: str | None = None, labels: tuple[str, ...] = ()) -> dict:
        self._issue_create({
            "team": self.teams[0]["id"], "title": title, "description": description,
            "project": next(iter(self.projects)), "parentId": self.find(parent)["id"] if parent else None,
            "state": state, "priority": priority,
            "assignee": assignee, "milestone": milestone, "labels": list(labels),
        })
        return self.find(title)

    def writes_to(self, title_part: str) -> list[str]:
        """Writes that touched an issue (by ID in the arguments), in order."""
        issue_id = self.find(title_part)["id"]
        return [name for name, arguments in self.mutation_log if issue_id in json.dumps(arguments)]

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


def log_in() -> None:
    """Leave a login in the (isolated) home directory, as 'pralph linear oauth' does."""
    path = Path.home() / ".pralph" / "linear-oauth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access_token": "token", "refresh_token": "refresh", "client_id": "c"}))


def make_sync(state: StateManager, fake: FakeLinear, *, listen: bool = True) -> LinearSync:
    client = LinearClient("token", transport=fake, sleep=lambda s: None)
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


class FakeMcpServer:
    """Stands in for ``urllib.request.urlopen``: an MCP server speaking Streamable HTTP."""

    def __init__(self, *, sse: bool = False, session: str | None = "sess-1") -> None:
        self.sse = sse
        self.session = session
        self.requests: list[tuple[dict, dict]] = []  # (headers, JSON-RPC message)
        self.status_for: list[int] = []               # HTTP statuses to answer with first
        self.tool_result: dict = {"content": [{"type": "text", "text": '{"teams": [], "hasNextPage": false}'}]}

    def methods(self) -> list[str]:
        return [message["method"] for _, message in self.requests]

    def __call__(self, request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        message = json.loads(request.data)
        self.requests.append((headers, message))
        if self.status_for:
            status = self.status_for.pop(0)
            if status >= 400:
                raise urllib.error.HTTPError(request.full_url, status, "x", {}, io.BytesIO(b'{"error":"nope"}'))  # type: ignore[arg-type]
        if "id" not in message:
            return _Response(202, {}, b"")
        if message["method"] == "initialize":
            result: dict = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "linear"}}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "whoami", "inputSchema": {"properties": {}}}]}
        else:
            result = self.tool_result
        answer = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        response_headers = {"Mcp-Session-Id": self.session} if self.session and message["method"] == "initialize" else {}
        if self.sse:
            # A server may send other messages on the stream before the answer.
            noise = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "working"}}
            body = f"event: message\ndata: {json.dumps(noise)}\n\nevent: message\ndata: {json.dumps(answer)}\n\n"
            return _Response(200, {**response_headers, "Content-Type": "text/event-stream"}, body.encode())
        return _Response(200, {**response_headers, "Content-Type": "application/json"}, json.dumps(answer).encode())


class _Response:
    def __init__(self, status: int, headers: dict, body: bytes) -> None:
        self.status, self.headers, self._body = status, headers, body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class TestMcpTransport:
    def call(self, server: FakeMcpServer, client: LinearClient | None = None, **kwargs):
        client = client or LinearClient("lin_api_abc", sleep=lambda s: None, **kwargs)
        with patch("urllib.request.urlopen", side_effect=server):
            return client, client.teams()

    def test_session_is_initialized_once_and_tools_are_called(self) -> None:
        server = FakeMcpServer()
        client, teams = self.call(server)
        assert teams == []
        assert server.methods() == ["initialize", "notifications/initialized", "tools/call"]
        headers, message = server.requests[-1]
        assert message["params"] == {"name": "list_teams", "arguments": {"limit": 250}}
        assert headers["authorization"] == "Bearer lin_api_abc"
        assert headers["mcp-session-id"] == "sess-1" and headers["mcp-protocol-version"] == "2025-06-18"
        assert "text/event-stream" in headers["accept"] and "application/json" in headers["accept"]
        assert "mcp-session-id" not in server.requests[0][0]

        self.call(server, client)
        assert server.methods().count("initialize") == 1

    def test_answer_as_an_event_stream(self) -> None:
        server = FakeMcpServer(sse=True, session=None)
        server.tool_result = {"content": [{"type": "text", "text": '{"teams": [{"id": "t", "key": "ENG"}]}'}]}
        _, teams = self.call(server)
        assert teams == [{"id": "t", "key": "ENG"}]
        assert "mcp-session-id" not in server.requests[-1][0], "a server without sessions gets no session header"

    def test_a_bearer_prefix_on_the_key_is_not_doubled(self) -> None:
        server = FakeMcpServer()
        self.call(server, LinearClient("Bearer xyz"))
        assert server.requests[-1][0]["authorization"] == "Bearer xyz"

    @pytest.mark.parametrize("text, message, retryable", [
        ('{"error":"invalid_request","message":"Title is too long","status":400}', "Linear: Title is too long", False),
        ('Error: Could not find project "x"', 'Could not find project "x"', False),
        ('{"error":"ratelimited","message":"Slow down","status":429}', "Slow down", True),
    ])
    def test_tool_errors_carry_linears_message(self, text, message, retryable) -> None:
        server = FakeMcpServer()
        server.tool_result = {"isError": True, "content": [{"type": "text", "text": text}]}
        with pytest.raises(LinearError, match=message) as caught:
            self.call(server, max_retries=0)
        assert caught.value.retryable is retryable

    def test_dropped_session_is_started_again(self) -> None:
        server = FakeMcpServer()
        client, _ = self.call(server)
        server.status_for = [404]  # the server forgot the session
        _, teams = self.call(server, client)
        assert teams == []
        assert server.methods()[3:] == ["tools/call", "initialize", "notifications/initialized", "tools/call"]

    def test_rejected_credentials(self) -> None:
        server = FakeMcpServer()
        server.status_for = [401]
        with pytest.raises(LinearError, match="Linear rejected the credentials") as caught:
            self.call(server)
        assert caught.value.auth_failed and len(server.requests) == 1

    def test_network_failures_are_retried(self) -> None:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")) as urlopen:
            with pytest.raises(LinearError, match="Could not reach Linear"):
                LinearClient("k", sleep=lambda s: None, max_retries=1).teams()
        assert urlopen.call_count == 2

    def test_lost_write_is_not_sent_again(self) -> None:
        server = FakeMcpServer()
        client, _ = self.call(server)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")) as urlopen:
            with pytest.raises(LinearError) as caught:
                client.create_comment("issue-1", "hello")
        assert caught.value.outcome_unknown and urlopen.call_count == 1

    def test_server_errors_on_a_write_are_not_retried(self) -> None:
        server = FakeMcpServer()
        client, _ = self.call(server)
        server.status_for = [502]
        with patch("urllib.request.urlopen", side_effect=server):
            with pytest.raises(LinearError, match="HTTP 502") as caught:
                client.create_comment("issue-1", "hello")
        assert caught.value.outcome_unknown and server.methods().count("tools/call") == 2

    def test_tools_can_be_listed(self) -> None:
        server = FakeMcpServer()
        client = LinearClient("k")
        with patch("urllib.request.urlopen", side_effect=server):
            assert [t["name"] for t in client._transport.list_tools()] == ["whoami"]


class TestClient:
    def test_missing_credentials_fail_before_any_request(self, fake) -> None:
        with pytest.raises(LinearError, match="Not logged in to Linear — run 'pralph linear oauth'"):
            LinearClient(transport=fake).teams()
        assert fake.calls == []

    def test_rate_limits_are_retried(self, fake) -> None:
        fake.failures = [LinearError("Linear rate limit reached", retryable=True)] * 2
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
        fake.failures = [LinearError("Linear: Title is too long")]
        with pytest.raises(LinearError, match="Title is too long"):
            LinearClient("k", transport=fake).teams()
        assert len(fake.calls) == 1

    def test_rejected_credentials_are_not_retried(self, fake) -> None:
        fake.failures = [LinearError("Linear rejected the credentials", auth_failed=True)]
        with pytest.raises(LinearError, match="Linear rejected the credentials") as caught:
            LinearClient("k", transport=fake).teams()
        assert caught.value.auth_failed and len(fake.calls) == 1

    def test_writes_with_an_unknown_outcome_are_not_repeated(self, fake) -> None:
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
        fake.failures = [LinearError("Linear rate limit reached", retryable=True)]
        client.create_issue(team_id="team-eng", title="Once")
        assert fake.calls == ["issue_create", "issue_create"], "a rejected write was not applied"
        assert len(fake.issues) == 1

    def test_arguments_that_are_not_set_are_left_out(self, fake) -> None:
        LinearClient("k", transport=fake).create_issue(team_id="team-eng", title="Bare")
        assert fake.mutation_log == [("issue_create", {"team": "team-eng", "title": "Bare"})]

    def test_project_issues_follows_pagination(self, state) -> None:
        fake = FakeLinear(page_size=2)
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        sync.push()
        issues = sync.client.project_issues(sync.link.project_id)
        assert len(issues) == 5  # 2 parents + 3 stories
        assert fake.calls.count("project_issues") >= 3
        parents = {i["id"] for i in issues if not i["parent_id"]}
        assert {i["parent_id"] for i in issues if i["parent_id"]} == parents, "parents are given by ID"

    def test_project_lookup(self, state, fake, pushed) -> None:
        project = next(iter(fake.projects.values()))
        client = pushed.client
        for reference in (project["id"], project["identifier"], project["slugId"]):
            assert client.project(reference)["id"] == project["id"]
        assert client.project("no-such-project") is None
        assert [p["id"] for p in client.find_projects("SHOP")] == [project["id"]]
        assert client.find_projects("Sho") == [], "part of a name is not the project"


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

    @pytest.mark.parametrize("heading", [
        "## Done when", "### Done when:", "**Done when**", "**Done when:**", "Done when:",
        "## Definition of Done", "## DONE  WHEN",
    ])
    def test_done_when_is_an_acceptance_criteria_heading(self, heading) -> None:
        text = f"Ship the export.\n\n{heading}\n\n- CSV downloads\n- [ ] Empty state is handled\n\n## Notes\n\nAsk Sam."
        content, criteria = parse_description(text)
        assert criteria == ["CSV downloads", "Empty state is handled"]
        assert content == "Ship the export.\n\n## Notes\n\nAsk Sam."

    def test_done_when_in_a_sentence_is_not_a_heading(self) -> None:
        text = "This is done when the export works.\n\n- not a criterion"
        assert parse_description(text) == (text, [])

    def test_criteria_section_ends_at_a_bold_heading(self) -> None:
        text = "**Done when**\n- Export works\n\n**Out of scope**\n- PDF export"
        assert parse_description(text) == ("**Out of scope**\n- PDF export", ["Export works"])

    def test_text_in_the_criteria_section_is_kept(self) -> None:
        text = (
            "Ship the export.\n\n## Done when\n\nAll of these hold on staging:\n\n"
            "- CSV downloads\n  with a header row\n- Empty state is handled\n\nAsk Sam about quotas."
        )
        content, criteria = parse_description(text)
        assert criteria == ["CSV downloads with a header row", "Empty state is handled"]
        assert content == "Ship the export.\n\nAll of these hold on staging:\nAsk Sam about quotas."

    def test_configured_heading(self) -> None:
        headings = linear_sync.ac_headings("## Fertig wenn:")
        assert headings == ("Fertig wenn", "Acceptance criteria", "Done when", "Definition of done")
        assert parse_description("Export.\n\n### Fertig wenn\n- CSV lädt", headings) == ("Export.", ["CSV lädt"])
        assert parse_description("Export.\n\n## Done when\n- CSV", headings)[1] == ["CSV"], "built-ins stay"
        assert parse_description("Export.\n\n### Fertig wenn\n- CSV lädt")[1] == []
        assert linear_sync.ac_headings(["Done when", "Checks"])[:2] == ("Done when", "Checks")
        assert "## Fertig wenn\n\n- [ ] Migration exists" in render_description(backlog()[0], heading=headings[0])

    def test_configured_heading_is_used_by_push_and_pull(self, state, fake) -> None:
        (state.state_dir / "config.json").write_text(json.dumps({"linear_ac_heading": "Done when"}))
        state.append_stories(backlog())
        sync = make_sync(state, fake)
        sync.push()
        assert "## Done when\n\n- [ ] Migration exists" in fake.find("DB-001")["description"]
        fake.find("API-002")["description"] = "Details.\n\n**Checks**\n- one\n\n**Done when**\n- Returns 404 for unknown ids"
        sync.pull()
        assert stories_by_id(state)["API-002"].acceptance_criteria == ["Returns 404 for unknown ids"]
        assert stories_by_id(state)["DB-001"].acceptance_criteria == ["Migration exists", "Indexes on sku"]


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

    def test_failure_reported_by_the_implementation_is_in_the_comment(self, state, fake, pushed) -> None:
        """As the implement loop records it: the reason comes with the implementation's answer."""
        state.mark_story_status("API-001", StoryStatus.in_progress)
        state.mark_story_status("API-001", StoryStatus.error, summary="Stopped",
                                extra={"status": "error", "summary": "Stopped", "reason": "The products table does not exist."})
        (comment,) = fake.comments_on("API-001")
        assert comment.startswith("**pralph** could not implement this story.")
        assert "The products table does not exist." in comment

    def test_failure_without_a_word_says_so(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.error)
        assert "No reason was recorded." in fake.comments_on("API-001")[0]

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
        assert "closed this as a duplicate of API-001" in fake.comments_on("API-002")[0]
        state.mark_story_status("API-001", StoryStatus.skipped, summary="Refined into: API-003")
        assert fake.state_of("API-001") == "Canceled"
        (comment,) = fake.comments_on("API-001")
        assert comment.startswith("**pralph** skipped this story.\n") and "Refined into: API-003" in comment

    def test_comment_carries_the_review_verdict(self, state, fake, pushed) -> None:
        state.mark_story_status("API-001", StoryStatus.implemented, summary="Built it", review_pending=True)
        state.notify_review_passed("API-001", "Clean and tested")
        assert "Review: approved — Clean and tested" in fake.comments_on("API-001")[0]

        # A reviewer that failed is not an approval — the ticket must say so.
        state.mark_story_status("API-002", StoryStatus.implemented, summary="Built it", review_pending=True)
        state.notify_review_passed("API-002", error="unknown slash command /dev:vet")
        (comment,) = fake.comments_on("API-002")
        assert "Not reviewed: the reviewer failed (unknown slash command /dev:vet)" in comment
        assert "approved" not in comment

        state.mark_story_status("DB-001", StoryStatus.implemented, summary="Built it")  # --no-review
        assert "Review" not in fake.comments_on("DB-001")[0] and "review" not in fake.comments_on("DB-001")[0]

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
        report = pushed.pull()  # reads the tickets' relations once
        assert (report.total, report.new, report.updated, report.status_changes, report.pushed) == (3, [], 0, [], 0)
        assert fake.calls == ["project_issues", "issue", "issue", "issue"] and fake.mutations() == []

        before = state.stories_path.read_text()
        fake.calls.clear()
        report = pushed.pull()
        assert state.stories_path.read_text() == before
        assert (report.total, report.new, report.updated, report.status_changes, report.pushed) == (3, [], 0, [], 0)
        assert fake.calls == ["project_issues"], "one call: nothing changed, so no relations are read again"

    def test_relations_are_read_only_for_tickets_that_changed(self, state, fake, pushed) -> None:
        pushed.pull()
        fake.calls.clear()
        fake.find("API-001")["title"] = "API-001: List all products"  # any edit touches updatedAt
        fake.move("DB-001", "Done")
        pushed.pull()
        assert fake.calls.count("issue") == 1, "API-001 changed and is open; DB-001 is closed: nothing waits on its blockers"

        fake.calls.clear()
        state.mark_story_status("API-002", StoryStatus.in_progress)  # pralph's own write touches it too
        pushed.pull()
        assert fake.calls.count("issue") == 1

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

        pushed.pull()  # the new relation touched the ticket: its relations are read once more
        before = state.stories_path.read_text()
        pushed.pull()
        assert state.stories_path.read_text() == before and len(fake.relations) == 4

    def test_unreadable_relations_do_not_drop_dependencies(self, state, fake, pushed) -> None:
        fake.unreadable.add(fake.find("API-002")["id"])
        fake.relations.remove((fake.find("API-001")["id"], fake.find("API-002")["id"]))
        pushed.pull()
        assert sorted(stories_by_id(state)["API-002"].dependencies) == ["API-001", "DB-001"]
        assert len(fake.relations) == 2, "nothing is created on a guess"

        fake.unreadable.clear()
        pushed.pull()
        assert stories_by_id(state)["API-002"].dependencies == ["DB-001"], "read as soon as it can be"

    def test_blocker_in_another_project_is_ignored(self, state, fake, pushed) -> None:
        other = fake._issue_create({"team": "team-eng", "title": "Elsewhere"})
        fake.relations.append((fake._lookup(other["id"])["id"], fake.find("API-001")["id"]))
        pushed.pull()
        assert stories_by_id(state)["API-001"].dependencies == ["DB-001"]

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


# ── scope: a project shared with other repositories and people ────────


@pytest.fixture
def shared(state, fake) -> LinearSync:
    """A project that people run in Linear: this repo's tickets carry a label,
    the rest belongs to other repositories."""
    sync = make_sync(state, fake)
    sync._ensure_team(None)
    sync._ensure_project("Platform")
    fake.add_issue("Checkout flow")                                          # parent, written by a PM
    fake.add_issue("Cart API", parent="Checkout flow", labels=("lb-shop",), assignee="u1",
                   milestone="ms-base", description="Carts.\n\n## Done when\n- POST /cart works")
    fake.add_issue("Cart page", parent="Checkout flow", labels=("lb-app",), assignee="u2", milestone="ms-base")
    fake.add_issue("Payment API", labels=("lb-shop", "lb-app"), assignee="u2", milestone="ms-launch")
    fake.add_issue("Push notifications", labels=("lb-app",), assignee="u1", milestone="ms-launch")
    fake.calls.clear()
    fake.mutation_log.clear()
    return sync


class TestScope:
    def titles(self, state) -> set[str]:
        return {s.title for s in state.load_stories() if s.status != StoryStatus.skipped}

    def test_unscoped_link_takes_every_issue(self, state, fake, shared) -> None:
        report = shared.pull()
        assert self.titles(state) == {"Cart API", "Cart page", "Payment API", "Push notifications"}
        assert report.out_of_scope == 0

    def test_label_scope(self, state, fake, shared) -> None:
        shared.set_scope(label="REPO:shop")
        assert shared.link.scope["label"] == {"name": "repo:shop"}, "as Linear spells it"
        fake.add_issue("Legacy import", labels=("lb-shop-old",))  # "repo:shop-legacy" is another label
        report = shared.pull()
        assert self.titles(state) == {"Cart API", "Payment API"}
        report.out_of_scope -= 1
        assert report.total == 2 and report.out_of_scope == 2
        assert stories_by_id(state)[fake.find("Cart API")["identifier"]].acceptance_criteria == ["POST /cart works"]

    def test_milestone_and_assignee_scope(self, state, fake, shared) -> None:
        shared.set_scope(milestone="base WORK")
        shared.pull()
        assert self.titles(state) == {"Cart API", "Cart page"}

        shared.set_scope(assignee="me")
        assert shared.scope_text() == "milestone 'Base work', assigned to Dev"
        shared.pull()
        assert self.titles(state) == {"Cart API"}, "criteria combine"

        shared.set_scope(assignee="sam@example.com", clear=True)
        shared.pull()
        assert self.titles(state) == {"Cart page", "Payment API"}

    def test_scope_survives_a_restart(self, state, fake, shared) -> None:
        shared.set_scope(label="repo:shop", milestone="Launch")
        again = make_sync(StateManager(str(state.project_dir)), fake, listen=False)
        assert again.scope_text() == "milestone 'Launch', label 'repo:shop'"

    def test_unknown_scope_values_are_refused(self, state, fake, shared) -> None:
        with pytest.raises(LinearError, match="no milestone 'Beta' .*'Base work', 'Launch'"):
            shared.set_scope(milestone="Beta")
        with pytest.raises(LinearError, match="No label 'nope'"):
            shared.set_scope(label="nope")
        with pytest.raises(LinearError, match="No label 'repo:ops' in team ENG"):
            shared.set_scope(label="repo:ops")  # another team's label
        with pytest.raises(LinearError, match="No label 'repo'"):
            shared.set_scope(label="repo")  # part of a name is not a label
        with pytest.raises(LinearError, match="No Linear user"):
            shared.set_scope(assignee="nobody")
        with pytest.raises(LinearError, match="Several Linear users match 'Sam Lee' .*use the email"):
            shared.set_scope(assignee="Sam Lee")
        assert shared.link.scope == {}

    def test_nothing_outside_the_scope_is_written_to(self, state, fake, shared) -> None:
        shared.set_scope(label="repo:shop")
        patcher, implemented = fake_claude()
        with patcher:
            ps = run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=shared)
        assert ps.completion_reason == "all_stories_done"
        assert sorted(implemented) == ["Cart API", "Payment API"]
        assert fake.state_of("Cart API") == "Done" and fake.state_of("Payment API") == "Done"
        for other in ("Cart page", "Push notifications", "Checkout flow"):
            assert fake.writes_to(other) == [], f"{other} is not this repo's ticket"
        assert fake.state_of("Checkout flow") == "Todo", "a parent outside the scope is left to its owners"
        assert set(fake.mutations()) == {"issue_update", "comment_create"}, "nothing was created"

    def test_issue_leaving_the_scope_is_dropped_and_left_alone(self, state, fake, shared) -> None:
        shared.set_scope(assignee="me")
        shared.pull()
        cart = fake.find("Cart API")
        state.mark_story_status(cart["identifier"], StoryStatus.in_progress)
        state.mark_story_status(cart["identifier"], StoryStatus.pending, summary="User interrupted")
        assert fake.state_of("Cart API") == "Todo"

        cart["assigneeId"] = "u2"  # reassigned to a colleague
        fake.mutation_log.clear()
        report = shared.pull()
        story = stories_by_id(state)[cart["identifier"]]
        assert story.status == StoryStatus.skipped
        assert (cart["identifier"], "pending", "skipped",
                "Issue is outside the Linear scope (assigned to Dev)") in report.status_changes
        assert fake.writes_to("Cart API") == [], "skipping it locally must not cancel the colleague's ticket"

        # Whatever happens to the story locally, the ticket is not pralph's any more.
        state.mark_story_status(story.id, StoryStatus.implemented, summary="done anyway")
        state.mark_story_status(story.id, StoryStatus.error, error_reason="boom")
        shared.pull()
        assert fake.writes_to("Cart API") == []

        cart["assigneeId"] = "u1"  # and back
        shared.pull()
        assert stories_by_id(state)[cart["identifier"]].status == StoryStatus.pending
        assert not linear_sync.is_outside(stories_by_id(state)[cart["identifier"]])

    def test_new_stories_are_created_inside_the_scope(self, state, fake, shared) -> None:
        shared.set_scope(label="repo:shop", milestone="Launch", assignee="me")
        shared.pull()
        state.append_stories([Story(id="PAY-001", title="Refunds", content="c", category="PAY")])
        for title in ("PAY-001", "PAY — Pay stories"):
            issue = fake.find(title)
            assert (issue["labelIds"], issue["projectMilestoneId"], issue["assigneeId"]) == (["lb-shop"], "ms-launch", "u1")
        shared.pull()
        assert stories_by_id(state)["PAY-001"].status == StoryStatus.pending, "its own ticket is in scope"
        state.mark_story_status("PAY-001", StoryStatus.implemented)
        assert fake.state_of("PAY — Pay stories") == "Done", "pralph's own parent issue is in scope too"

    def test_parent_with_open_work_outside_the_scope_is_not_closed(self, state, fake, shared) -> None:
        fake.find("Checkout flow")["labelIds"] = ["lb-shop"]  # the parent itself is ours
        shared.set_scope(label="repo:shop")
        shared.pull()
        cart = fake.find("Cart API")["identifier"]
        state.mark_story_status(cart, StoryStatus.in_progress)
        assert fake.state_of("Checkout flow") == "In Progress"
        state.mark_story_status(cart, StoryStatus.implemented)
        assert fake.state_of("Checkout flow") == "In Progress", "'Cart page' (another repo) is still open"

        fake.move("Cart page", "Done")
        shared.pull()
        assert fake.state_of("Checkout flow") == "Done"

    def test_relations_to_issues_outside_the_scope_are_not_created(self, state, fake, shared) -> None:
        shared.pull()  # unscoped: every issue is a story
        ids = {s.title: s.id for s in state.load_stories()}
        state.update_story(ids["Cart API"], lambda s: s.dependencies.append(ids["Cart page"]), notify=False)
        shared.set_scope(label="repo:shop")
        shared.pull()
        assert fake.relations == []

    def test_empty_scope_match_is_called_out(self, state, fake) -> None:
        messages: list[str] = []
        sync = make_sync(state, fake)
        sync._echo = messages.append
        sync._ensure_team(None)
        sync._ensure_project("Platform")
        fake.add_issue("Cart API", labels=("lb-app",))
        sync.set_scope(label="repo:shop")
        sync.pull()
        assert any("no issue of the project matches the scope (label 'repo:shop')" in m for m in messages)

    def test_story_id_names_a_ticket_this_checkout_has_not_seen(self, state, fake, shared) -> None:
        shared.set_scope(label="repo:shop")
        identifier = fake.find("Payment API")["identifier"]
        patcher, implemented = fake_claude()
        with patcher:
            ps = run_implement_loop(state, story_id=identifier.lower(), cooldown=0, phase1=False, tracker=shared)
        assert ps.completion_reason == "single_story_done" and implemented == ["Payment API"]
        assert fake.state_of("Payment API") == "Done" and fake.state_of("Cart API") == "Todo"

    def test_story_id_outside_the_scope_warns_and_does_not_write(self, state, fake, shared) -> None:
        shared.pull()
        shared.set_scope(label="repo:shop")
        messages: list[str] = []
        shared._echo = messages.append
        patcher, implemented = fake_claude()
        with patcher:
            run_implement_loop(state, story_id=fake.find("Cart page")["identifier"], cooldown=0,
                               phase1=False, tracker=shared)
        assert implemented == ["Cart page"], "an explicit request is carried out"
        assert any("outside the scope (label 'repo:shop') — its ticket will not be updated" in m for m in messages)
        assert fake.writes_to("Cart page") == []

    def test_rejected_single_story_is_reworked_with_the_ticket_in_progress(self, state, fake, shared) -> None:
        shared.set_scope(label="repo:shop")
        seen: list[str] = []
        patcher, implemented = fake_claude(on_story=lambda t: seen.append(fake.state_of("Cart API")),
                                           reject_once="Cart API")
        with patcher:
            ps = run_implement_loop(state, story_id=fake.find("Cart API")["identifier"], cooldown=0,
                                    phase1=False, tracker=shared)
        assert ps.completion_reason == "single_story_done" and implemented == ["Cart API", "Cart API"]
        assert seen == ["In Progress", "In Progress"] and fake.state_of("Cart API") == "Done"
        comments = fake.comments_on("Cart API")
        assert len(comments) == 2 and "No tests" in comments[0] and "implemented this story" in comments[1]


# ── read-mostly: a project that people run ────────────────────────────


@pytest.fixture
def guest(state, fake, shared) -> LinearSync:
    """Read-mostly link to the shared project, scoped to this repo's label."""
    shared.set_scope(label="repo:shop")
    shared.set_read_mostly(True)
    shared.pull()
    fake.calls.clear()
    fake.mutation_log.clear()
    return shared


class TestReadMostly:
    def cart(self, fake) -> str:
        return fake.find("Cart API")["identifier"]

    def state_writes(self, fake) -> set[str]:
        return {v["state"] for name, v in fake.mutation_log if name == "issue_update"}

    def test_work_starts_the_ticket_and_never_closes_it(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.in_progress)
        assert fake.state_of("Cart API") == "In Progress"

        state.mark_story_status(cart, StoryStatus.implemented, summary="Built it",
                                extra={"files_changed": ["cart.py"]}, review_pending=True)
        assert fake.comments == [], "the comment waits for the review, like Done does in a read-write link"
        state.notify_review_passed(cart, "looks good")
        assert fake.state_of("Cart API") == "In Progress", "Done is for the merge"
        (comment,) = fake.comments_on("Cart API")
        assert "implemented this story" in comment and "`cart.py`" in comment and "Review: approved" in comment

        guest.pull()
        assert fake.state_of("Cart API") == "In Progress", "the sync does not catch up on it either"
        assert self.state_writes(fake) == {"st-progress"}
        assert stories_by_id(state)[cart].status == StoryStatus.implemented

    def test_merge_closing_the_ticket_is_taken_as_done(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.in_progress)
        state.mark_story_status(cart, StoryStatus.implemented, summary="Built it")
        fake.move("Cart API", "In Review")   # the GitHub integration: PR opened …
        guest.pull()
        assert stories_by_id(state)[cart].status == StoryStatus.implemented
        fake.move("Cart API", "Done")        # … and merged
        report = guest.pull()
        assert report.status_changes == [] and stories_by_id(state)[cart].status == StoryStatus.implemented

    def test_rejected_in_linear_by_moving_it_back(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.in_progress)
        state.mark_story_status(cart, StoryStatus.implemented, summary="Built it")
        fake.move("Cart API", "Todo")
        guest.pull()
        assert stories_by_id(state)[cart].status == StoryStatus.pending

    def test_never_canceled(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.skipped, summary="Not needed after all")
        assert fake.state_of("Cart API") == "Todo"
        (comment,) = fake.comments_on("Cart API")
        assert "will not work on it. The issue was left open." in comment and "Not needed after all" in comment

        payment = fake.find("Payment API")["identifier"]
        state.mark_story_status(payment, StoryStatus.duplicate, extra={"duplicate_of": cart})
        assert fake.state_of("Payment API") == "Todo"
        assert f"considers this a duplicate of {cart}" in fake.comments_on("Payment API")[0]
        assert "closed" not in fake.comments_on("Payment API")[0]

        guest.pull()
        assert fake.state_of("Cart API") == "Todo" and self.state_writes(fake) == set()
        assert stories_by_id(state)[cart].status == StoryStatus.skipped, "the local decision stands"

    def test_interrupted_story_keeps_its_ticket_and_is_picked_up_again(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.in_progress)
        state.mark_story_status(cart, StoryStatus.pending, summary="User interrupted")
        assert fake.state_of("Cart API") == "In Progress", "no move back to Todo"
        guest.pull()
        assert stories_by_id(state)[cart].status == StoryStatus.pending, \
            "its own In Progress is not read as 'taken by someone else'"

    def test_failure_is_a_comment(self, state, fake, guest) -> None:
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.in_progress)
        state.mark_story_status(cart, StoryStatus.error, error_reason="tests would not pass")
        assert fake.state_of("Cart API") == "In Progress"
        assert "tests would not pass" in fake.comments_on("Cart API")[0]

    def test_nothing_is_created(self, state, fake, guest) -> None:
        state.append_stories([
            Story(id="LOCAL-001", title="Local chore", content="c", category="LOCAL",
                  dependencies=[self.cart(fake)]),
        ])
        state.mark_story_status("LOCAL-001", StoryStatus.in_progress)
        state.mark_story_status("LOCAL-001", StoryStatus.implemented)
        guest.pull()
        assert guest.unpushed() == []
        with pytest.raises(LinearError, match="read-mostly"):
            guest.push()
        with pytest.raises(LinearError, match="read-mostly"):
            guest.push(dry_run=True)
        assert fake.mutations() == [] and fake.relations == []
        assert len(fake.issues) == 5

    def test_parent_issues_are_not_moved(self, state, fake, guest) -> None:
        fake.find("Checkout flow")["labelIds"] = ["lb-shop"]  # even one that is in scope
        guest.pull()
        state.mark_story_status(self.cart(fake), StoryStatus.in_progress)
        assert fake.state_of("Checkout flow") == "Todo" and fake.writes_to("Checkout flow") == []

    def test_a_run_only_comments_and_starts(self, state, fake, guest) -> None:
        state.append_stories([Story(id="LOCAL-001", title="Local chore", content="c", category="LOCAL")])
        patcher, implemented = fake_claude(reject_once="Cart API")
        with patcher:
            ps = run_implement_loop(state, max_iterations=10, cooldown=0, phase1=False, tracker=guest)
        assert ps.completion_reason == "all_stories_done"
        assert sorted(implemented) == ["Cart API", "Cart API", "Local chore", "Payment API"]
        assert set(fake.mutations()) == {"issue_update", "comment_create"}
        assert self.state_writes(fake) == {"st-progress"}
        assert fake.state_of("Cart API") == "In Progress" and fake.state_of("Payment API") == "In Progress"
        for other in ("Cart page", "Push notifications", "Checkout flow"):
            assert fake.writes_to(other) == []
        assert len(fake.comments_on("Cart API")) == 2, "review feedback, then the result"
        assert len(fake.issues) == 5, "the local story stays local"

    def test_setting_survives_a_restart_and_can_be_lifted(self, state, fake, guest) -> None:
        again = make_sync(StateManager(str(state.project_dir)), fake, listen=False)
        assert again.link.read_mostly is True
        again.set_read_mostly(False)
        cart = self.cart(fake)
        state.mark_story_status(cart, StoryStatus.implemented)
        make_sync(StateManager(str(state.project_dir)), fake, listen=False).pull()
        assert fake.state_of("Cart API") == "Done"


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


# ── stories on their own branch (pralph.ship) ─────────────────────────


class TestBranchStates:
    PR = "https://github.com/acme/shop/pull/7"

    def on_branch(self, state, story_id: str = "DB-001") -> None:
        def mutate(s: Story) -> None:
            s.metadata["git"] = {"branch": s.metadata["linear"]["branch"], "base": "main"}
        state.update_story(story_id, mutate, notify=False)

    def test_the_ticket_carries_linears_branch_name(self, state, fake, pushed) -> None:
        identifier = fake.find("DB-001")["identifier"].lower()
        assert stories_by_id(state)["DB-001"].metadata["linear"]["branch"] == f"dev/{identifier}", "known from the push"
        fake.add_issue("Checkout", parent="API —")
        pushed.pull()
        new = next(s for s in state.load_stories() if s.title == "Checkout")
        assert new.metadata["linear"]["branch"] == f"dev/{new.id.lower()}"
        assert ship_lib.branch_name(new) == f"dev/{new.id.lower()}"

    def test_ticket_is_done_when_the_story_is_merged_not_when_it_is_implemented(self, state, fake, pushed) -> None:
        self.on_branch(state)
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.implemented, summary="Built it", review_pending=True)
        state.notify_review_passed("DB-001", "fine")
        assert fake.state_of("DB-001") == "In Progress", "on a branch: not done yet"
        assert "not merged yet" in fake.comments_on("DB-001")[-1]

        state.mark_story_status("DB-001", StoryStatus.shipped, extra={"pr_url": self.PR})
        assert fake.state_of("DB-001") == "In Progress"
        assert fake.comments_on("DB-001")[-1] == f"**pralph** opened a pull request: {self.PR}"
        pushed.pull()
        assert fake.state_of("DB-001") == "In Progress" and stories_by_id(state)["DB-001"].status == StoryStatus.shipped

        state.mark_story_status("DB-001", StoryStatus.rework, summary="Changes requested", extra={"pr_url": self.PR})
        state.mark_story_status("DB-001", StoryStatus.shipped, extra={"pr_url": self.PR, "pr_updated": True})
        assert "pushed the requested changes" in fake.comments_on("DB-001")[-1]

        state.mark_story_status("DB-001", StoryStatus.merged, summary="merged")
        assert fake.state_of("DB-001") == "Done"

    def test_merge_seen_in_linear_first(self, state, fake, pushed) -> None:
        """Linear's GitHub integration closes the ticket when the pull request is merged."""
        self.on_branch(state)
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.shipped, extra={"pr_url": self.PR})
        fake.move("DB-001", "In Review")
        pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.shipped, "In Review is still in progress"

        fake.move("DB-001", "Done")
        report = pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.merged
        assert ("DB-001", "shipped", "merged", "Completed in Linear (merged)") in report.status_changes

    def test_done_in_linear_without_a_branch_is_implemented_as_before(self, state, fake, pushed) -> None:
        fake.move("DB-001", "Done")
        pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.implemented

    def test_story_that_needs_a_person(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.needs_human, summary="Migration written",
                                extra={"reason": "Apply the migration to staging"})
        assert fake.state_of("DB-001") == "In Progress"
        comment = fake.comments_on("DB-001")[-1]
        assert "needs a person" in comment and "Apply the migration to staging" in comment
        assert "pralph unblock DB-001" in comment
        pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.needs_human, "it stays paused"

        fake.move("DB-001", "Todo")  # the person's way of saying: go on
        pushed.pull()
        assert stories_by_id(state)["DB-001"].status == StoryStatus.pending

    def test_failed_ship_step_is_a_comment(self, state, fake, pushed) -> None:
        self.on_branch(state)
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        state.mark_story_status("DB-001", StoryStatus.implemented, summary="Built it")
        fake.calls.clear()
        state.notify_story_problem("DB-001", "ship", "git push: permission denied")
        comment = fake.comments_on("DB-001")[-1]
        assert "could not open a pull request" in comment and "git push: permission denied" in comment
        assert "pralph ship -s DB-001" in comment and "`dev/" in comment
        assert fake.state_of("DB-001") == "In Progress", "the ticket itself does not move"

    def test_read_mostly_leaves_done_to_the_merge(self, state, fake, guest) -> None:
        cart = fake.find("Cart API")["identifier"]
        self.on_branch(state, cart)
        state.mark_story_status(cart, StoryStatus.in_progress)
        state.mark_story_status(cart, StoryStatus.shipped, extra={"pr_url": self.PR})
        state.mark_story_status(cart, StoryStatus.merged, summary="merged")
        guest.pull()
        assert fake.state_of("Cart API") == "In Progress", "pralph saw the merge; closing the ticket is GitHub's part"
        assert {v["state"] for name, v in fake.mutation_log if name == "issue_update"} == {"st-progress"}


# ── wiring / CLI ──────────────────────────────────────────────────────


class TestWiring:
    def test_attach_is_inert_without_a_link(self, state) -> None:
        assert linear_sync.attach(state) is None
        with pytest.raises(LinearError, match="not linked"):
            linear_sync.attach(state, required=True)

    def test_attach_needs_a_login_for_linked_projects(self, state, fake, pushed, monkeypatch) -> None:
        other = StateManager(str(state.project_dir))
        messages: list[str] = []
        assert linear_sync.attach(other, echo=messages.append) is None
        assert "there is no Linear login — run 'pralph linear oauth'" in messages[0]
        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")  # an API key is not a login
        with pytest.raises(LinearError, match="Not logged in to Linear — run 'pralph linear oauth'"):
            linear_sync.attach(other, required=True)

        log_in()
        sync = linear_sync.attach(other)
        assert sync is not None and sync.link.team_key == "ENG" and sync.client.oauth is not None

    def test_cli_dry_run_needs_no_login(self, state) -> None:
        state.append_stories(backlog())
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "push", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "project: Shop (would create)" in result.output
        assert "would create: 3 sub-issues, 3 blocking relations" in result.output

    def test_cli_push_without_a_login_explains_what_to_do(self, state) -> None:
        state.append_stories(backlog())
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "push"])
        assert result.exit_code == 1 and "Not logged in to Linear. Run 'pralph linear oauth'" in result.output

    def test_cli_implement_refuses_to_bypass_a_linked_project_silently(self, state, fake, pushed) -> None:
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "implement"])
        assert result.exit_code == 1 and "Not logged in to Linear — run 'pralph linear oauth'" in result.output

    def test_cli_link_with_scope_and_scope_command(self, state, fake, shared, tmp_path, monkeypatch) -> None:
        log_in()
        clone = tmp_path / "clone"
        clone.mkdir()
        url = next(iter(fake.projects.values()))["url"]
        cli = ["--project-dir", str(clone), "linear"]
        with patch("pralph.cli.LinearClient", lambda oauth: LinearClient(oauth=oauth, transport=fake)):
            bad = CliRunner().invoke(main, [*cli, "link", url, "--milestone", "Beta"])
            assert bad.exit_code == 1 and "no milestone 'Beta'" in bad.output
            assert not (clone / ".pralph" / "linear.json").exists(), "no unscoped link is left behind"

            result = CliRunner().invoke(main, [*cli, "link", url, "--label", "repo:shop", "--assignee", "me"])
            assert result.exit_code == 0, result.output
            assert "scope: assigned to Dev, label 'repo:shop'" in result.output
            assert "1 issues: 1 new" in result.output and "3 issues outside the scope left alone" in result.output

            shown = CliRunner().invoke(main, [*cli, "scope"])
            assert "scope: assigned to Dev, label 'repo:shop'" in shown.output

            widened = CliRunner().invoke(main, [*cli, "scope", "--clear", "--label", "repo:shop"])
            assert widened.exit_code == 0, widened.output
            assert "scope: label 'repo:shop'" in widened.output and "2 issues: 1 new" in widened.output

            status = CliRunner().invoke(main, [*cli, "status"])
            assert "scope: label 'repo:shop'" in status.output and "issues: 2" in status.output

            cleared = CliRunner().invoke(main, [*cli, "scope", "--clear"])
            assert "scope: the whole project" in cleared.output and "4 issues: 2 new" in cleared.output

    def test_cli_read_mostly(self, state, fake, shared, tmp_path, monkeypatch) -> None:
        log_in()
        clone = tmp_path / "clone"
        clone.mkdir()
        url = next(iter(fake.projects.values()))["url"]
        cli = ["--project-dir", str(clone), "linear"]
        with patch("pralph.cli.LinearClient", lambda oauth: LinearClient(oauth=oauth, transport=fake)):
            result = CliRunner().invoke(main, [*cli, "link", url, "--label", "repo:shop", "--read-mostly"])
            assert result.exit_code == 0, result.output
            assert "mode: read-mostly (comments and In Progress only)" in result.output
            assert LinearLink.load(clone / ".pralph" / "linear.json").read_mostly is True

            status = CliRunner().invoke(main, [*cli, "status"])
            assert "mode: read-mostly" in status.output and "not in Linear yet" not in status.output

            push = CliRunner().invoke(main, [*cli, "push"])
            assert push.exit_code == 1 and "read-mostly: pralph does not create issues" in push.output
            assert fake.mutations() == []

            lifted = CliRunner().invoke(main, [*cli, "scope", "--read-write"])
            assert lifted.exit_code == 0 and "mode: read-write" in lifted.output
            assert "scope: label 'repo:shop'" in lifted.output, "the scope is kept"
            assert LinearLink.load(clone / ".pralph" / "linear.json").read_mostly is False

    def test_cli_status(self, state, fake, pushed) -> None:
        state.mark_story_status("DB-001", StoryStatus.in_progress)
        result = CliRunner().invoke(main, ["--project-dir", str(state.project_dir), "linear", "status"])
        assert result.exit_code == 0, result.output
        assert "project: Shop https://linear.app/acme/project/shop-" in result.output
        assert "issues: 3 (2 Backlog, 1 In Progress)" in result.output
