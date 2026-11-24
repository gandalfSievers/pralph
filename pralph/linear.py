"""Linear client on top of Linear's MCP server (stdlib only).

pralph does not use Linear's GraphQL API. It calls the tools of the hosted
MCP server (https://mcp.linear.app/mcp) — the same interface AI clients use —
which is what lets it log in with OAuth and dynamic client registration, with
nothing to set up in Linear first (see ``pralph.linear_oauth``).

Covers exactly what pralph needs: teams and workflow states, creating a
project with issues / sub-issues / blocking relations, moving issues through
the workflow, commenting, and reading a project's issues back.

Authentication: the OAuth login of ``pralph linear oauth``, refreshed as
needed. API keys are not used.
See https://linear.app/docs/mcp
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from pralph import __version__

MCP_URL = "https://mcp.linear.app/mcp"
PROTOCOL_VERSION = "2025-06-18"

_PAGE_SIZE = 250

# transport(tool name, arguments) → the tool's result, parsed.
Transport = Callable[[str, dict], Any]

# What pralph asks for on every issue it lists (``fields`` of list_issues).
ISSUE_FIELDS = [
    "id", "uuid", "title", "description", "priority", "url", "status", "statusType",
    "labels", "assigneeId", "projectMilestone", "parentId", "updatedAt", "archivedAt",
]

# Every tool the client calls, with the arguments it passes — lets a check
# against the live server (tests/manual_linear_mcp.py) notice when a tool or
# an argument is renamed.
TOOLS: dict[str, list[str]] = {
    "whoami": [],
    "list_teams": ["limit", "cursor"],
    "list_issue_statuses": ["team"],
    "get_project": ["query"],
    "list_projects": ["query", "limit", "cursor", "fields"],
    "list_issues": ["project", "limit", "cursor", "fields", "orderBy"],
    "get_issue": ["id", "fields"],
    "list_milestones": ["project"],
    "list_issue_labels": ["name", "team", "limit", "cursor"],
    "list_users": ["query", "limit", "cursor"],
    "save_project": ["name", "addTeams", "summary", "description"],
    "save_issue": [
        "id", "team", "title", "description", "project", "parentId", "priority", "state",
        "assignee", "milestone", "labels", "blockedBy",
    ],
    "save_comment": ["issueId", "body"],
}


class LinearError(Exception):
    """A call to Linear failed (network, auth, validation or rate limit).

    ``retryable``: trying again may work. ``outcome_unknown``: the request may
    have reached Linear and been carried out anyway (timeout, dropped
    connection, 5xx) — repeating a write then risks doing it twice.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        outcome_unknown: bool = False,
        auth_failed: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown
        self.auth_failed = auth_failed  # Linear did not accept the credentials; nothing was done


def is_write(tool: str) -> bool:
    return tool.startswith(("save_", "create_", "delete_"))


# ── MCP transport ────────────────────────────────────────────────────


class McpTransport:
    """Calls tools on an MCP server over Streamable HTTP (JSON-RPC 2.0).

    One POST per message; the answer comes back as plain JSON or as a short
    event stream. The session is initialized on first use, and again if the
    server has dropped it.
    """

    def __init__(
        self,
        token: Callable[[], str],
        *,
        url: str = MCP_URL,
        timeout: int = 60,
    ) -> None:
        self._token = token
        self.url = url
        self.timeout = timeout
        self._session_id: str | None = None
        self._initialized = False
        self._protocol = PROTOCOL_VERSION
        self._next_id = 0

    # -- HTTP --

    def _post(self, message: dict) -> tuple[int, dict, str]:
        """Send one JSON-RPC message → (status, lower-cased headers, body)."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {_bare_token(self._token())}",
            "User-Agent": f"pralph/{__version__}",
        }
        if self._initialized:
            headers["MCP-Protocol-Version"] = self._protocol
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.url, data=json.dumps(message).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read().decode(errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LinearError(f"Could not reach Linear: {e}", retryable=True, outcome_unknown=True) from e

    def _rpc(self, method: str, params: dict | None = None, *, notification: bool = False) -> Any:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            self._next_id += 1
            message["id"] = self._next_id
        status, headers, body = self._post(message)

        if status == 401:
            raise LinearError("Linear rejected the credentials", auth_failed=True)
        if status == 403:
            raise LinearError(f"Linear does not allow this ({_error_text(body)})")
        if status == 404 and self._session_id:
            raise _SessionExpired()
        if status == 429:
            raise LinearError("Linear rate limit reached", retryable=True)
        if status >= 500:
            raise LinearError(f"HTTP {status} from Linear: {body[:200]}", retryable=True, outcome_unknown=True)
        if status >= 400:
            raise LinearError(f"HTTP {status} from Linear: {_error_text(body)}")
        if headers.get("mcp-session-id"):
            self._session_id = headers["mcp-session-id"]
        if notification:
            return None

        answer = _find_response(body, headers.get("content-type", ""), message["id"])
        if answer is None:
            raise LinearError("Linear's MCP server sent no answer", outcome_unknown=True)
        if answer.get("error"):
            error = answer["error"] if isinstance(answer["error"], dict) else {"message": str(answer["error"])}
            raise LinearError(f"Linear MCP error: {error.get('message') or error.get('code') or 'unknown error'}")
        return answer.get("result")

    def _ensure_session(self) -> None:
        if self._initialized:
            return
        result = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pralph", "version": __version__},
        }) or {}
        self._protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        self._initialized = True
        self._rpc("notifications/initialized", notification=True)

    def _with_session(self, method: str, params: dict) -> Any:
        self._ensure_session()
        try:
            return self._rpc(method, params)
        except _SessionExpired:
            # The server forgot the session (it did not act on the message): start a new one.
            self._session_id, self._initialized = None, False
            self._ensure_session()
            try:
                return self._rpc(method, params)
            except _SessionExpired:
                raise LinearError("Linear's MCP server keeps dropping the session", retryable=True) from None

    # -- tools --

    def __call__(self, tool: str, arguments: dict) -> Any:
        result = self._with_session("tools/call", {"name": tool, "arguments": arguments}) or {}
        return _tool_result(tool, result)

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor: str | None = None
        while True:
            result = self._with_session("tools/list", {"cursor": cursor} if cursor else {}) or {}
            tools += [t for t in result.get("tools") or [] if isinstance(t, dict)]
            cursor = result.get("nextCursor")
            if not cursor:
                return tools


class _SessionExpired(Exception):
    pass


def _bare_token(token: str) -> str:
    return token[7:].strip() if token.lower().startswith("bearer ") else token


def _error_text(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:200]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:200]
        return str(payload.get("error_description") or error or payload.get("message") or body)[:200]
    return body[:200]


def _find_response(body: str, content_type: str, request_id: int) -> dict | None:
    """The JSON-RPC response to ``request_id`` in a JSON or event-stream body."""
    messages: list[Any] = []
    if "text/event-stream" in content_type:
        data: list[str] = []
        for line in body.splitlines() + [""]:
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line.strip() and data:
                try:
                    messages.append(json.loads("\n".join(data)))
                except json.JSONDecodeError:
                    pass
                data = []
    else:
        try:
            messages.append(json.loads(body))
        except json.JSONDecodeError as e:
            raise LinearError(f"Invalid response from Linear: {e}", outcome_unknown=True) from e
    flat = [m for item in messages for m in (item if isinstance(item, list) else [item])]
    for message in flat:
        if isinstance(message, dict) and message.get("id") == request_id and ("result" in message or "error" in message):
            return message
    return None


def _tool_result(tool: str, result: dict) -> Any:
    """Parse a tools/call result: Linear's tools answer with JSON in a text block."""
    text = "\n".join(
        str(block.get("text", "")) for block in result.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if result.get("isError"):
        # Either a sentence ("Error: Could not find project …") or a JSON
        # object with the API's message and HTTP status.
        message, status = text, 0
        try:
            detail = json.loads(text)
        except json.JSONDecodeError:
            detail = None
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("error") or text)
            status = int(detail.get("status") or 0) if str(detail.get("status") or "").isdigit() else 0
        lowered = message.lower()
        raise LinearError(
            f"Linear: {message[:300] or f'{tool} failed'}",
            retryable=status == 429 or status >= 500 or "rate limit" in lowered or "ratelimit" in lowered,
            outcome_unknown=status >= 500,
            auth_failed=status == 401 or "not authenticated" in lowered,
        )
    if isinstance(result.get("structuredContent"), (dict, list)):
        return result["structuredContent"]
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


# ── client ───────────────────────────────────────────────────────────


class LinearClient:
    def __init__(
        self,
        token: str = "",
        *,
        oauth: Any = None,
        transport: Transport | None = None,
        url: str = MCP_URL,
        timeout: int = 60,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # The login (linear_oauth.OAuthSession) is asked for the access token
        # before every call, so it can refresh it. ``token`` is a fixed access
        # token for use without a login to refresh it from.
        self.token = token
        self.oauth = oauth
        self.max_retries = max_retries
        self._sleep = sleep
        self._transport = transport or McpTransport(lambda: self.token, url=url, timeout=timeout)

    # -- calling tools --

    def call(self, tool: str, **arguments: Any) -> Any:
        """Call a tool and return its parsed result. Raises LinearError on any failure.

        Failures that are worth another try are retried with backoff — except
        a write whose outcome is unknown: creating is not idempotent, so
        repeating a call that did go through would leave a second issue or
        comment behind. Those surface to the caller, which catches up on its
        next sync.

        With an OAuth login, a rejected access token is refreshed and the
        call made once more (a rejected call was not carried out).
        """
        arguments = {k: v for k, v in arguments.items() if v is not None}
        if self.oauth is None:
            return self._call(tool, arguments)
        self.token = self.oauth.access_token()
        try:
            return self._call(tool, arguments)
        except LinearError as e:
            if not e.auth_failed:
                raise
        self.token = self.oauth.access_token(force_refresh=True)
        try:
            return self._call(tool, arguments)
        except LinearError as e:
            if e.auth_failed:
                raise LinearError(
                    "Linear rejected the OAuth login — run 'pralph linear oauth' again", auth_failed=True,
                ) from e
            raise

    def _call(self, tool: str, arguments: dict) -> Any:
        if not self.token:
            raise LinearError("Not logged in to Linear — run 'pralph linear oauth'")
        last: LinearError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._transport(tool, arguments)
            except LinearError as e:
                last = e
                if not e.retryable or attempt >= self.max_retries or (is_write(tool) and e.outcome_unknown):
                    raise
                self._sleep(2.0 ** attempt)
        raise last or LinearError("Linear request failed")  # pragma: no cover

    def _pages(self, tool: str, key: str, **arguments: Any) -> list[dict]:
        """All items of a paginated list tool."""
        items: list[dict] = []
        cursor: str | None = None
        while True:
            page = self.call(tool, cursor=cursor, **arguments)
            if isinstance(page, list):  # a tool that answers with a bare list has one page
                return items + [n for n in page if isinstance(n, dict)]
            items += [n for n in (page or {}).get(key) or [] if isinstance(n, dict)]
            cursor = (page or {}).get("cursor")
            if not (page or {}).get("hasNextPage") or not cursor:
                return items

    # -- reads --

    def viewer(self) -> dict:
        who = self.call("whoami") or {}
        return {**(who.get("user") or {}), "organization": who.get("workspace") or {}}

    def teams(self) -> list[dict]:
        return self._pages("list_teams", "teams", limit=_PAGE_SIZE)

    def workflow_states(self, team_id: str) -> list[dict]:
        states = self.call("list_issue_statuses", team=team_id)
        if isinstance(states, dict):
            states = states.get("statuses") or states.get("states") or []
        return [st for st in states or [] if isinstance(st, dict)]

    def project(self, reference: str) -> dict | None:
        """A project by ID, identifier (P-ENG-12) or URL slug; None if there is none."""
        try:
            found = self.call("get_project", query=reference)
        except LinearError as e:
            if "could not find" in str(e).lower() or "not found" in str(e).lower():
                return None
            raise
        return _flatten_project(found) if isinstance(found, dict) and (found.get("uuid") or found.get("id")) else None

    def find_projects(self, name: str) -> list[dict]:
        """Projects with exactly this name (ignoring case)."""
        found = self._pages(
            "list_projects", "projects", query=name, limit=50, fields=["id", "uuid", "name", "url", "teams"],
        )
        wanted = name.strip().lower()
        return [_flatten_project(p) for p in found if str(p.get("name", "")).strip().lower() == wanted]

    def project_issues(self, project_id: str) -> list[dict]:
        """All (non-archived) issues of a project, flattened.

        Each issue carries ``parent_id``. Blocking relations are not part of a
        listing: ``blocked_by`` is empty and ``relations_truncated`` set until
        ``load_blockers`` has been asked for the issue.
        """
        nodes = self._pages(
            "list_issues", "issues", project=project_id, limit=_PAGE_SIZE, fields=ISSUE_FIELDS, orderBy="createdAt",
        )
        # Archived issues are not listed; one in the trash is archived too, in case it is.
        issues = [_flatten_issue(n) for n in nodes if not n.get("archivedAt")]
        # Parents are named by identifier (ENG-12); everything else here works with IDs.
        by_identifier = {i["identifier"]: i["id"] for i in issues}
        for issue in issues:
            if issue["parent_id"]:
                issue["parent_id"] = by_identifier.get(issue["parent_id"], issue["parent_id"])
        return issues

    def issue_blockers(self, issue_id: str) -> list[str]:
        """Identifiers (ENG-12) of the issues that block this one."""
        issue = self.call("get_issue", id=issue_id, fields=["id", "relations"]) or {}
        relations = issue.get("relations") if isinstance(issue, dict) else None
        blocked_by = (relations or {}).get("blockedBy") or []
        return [str(b["id"]) for b in blocked_by if isinstance(b, dict) and b.get("id")]

    def project_milestones(self, project_id: str) -> list[dict]:
        found = self.call("list_milestones", project=project_id)
        if isinstance(found, dict):
            found = found.get("milestones") or []
        return [m for m in found or [] if isinstance(m, dict) and m.get("id")]

    def find_labels(self, name: str, team_id: str | None = None) -> list[dict]:
        """Issue labels with exactly this name that the team can use (its own and the workspace's)."""
        found = self._pages("list_issue_labels", "labels", name=name, team=team_id, limit=_PAGE_SIZE)
        wanted = name.strip().lower()
        return [
            {"id": lb["id"], "name": lb.get("name", "")} for lb in found
            if lb.get("id") and str(lb.get("name", "")).strip().lower() == wanted and not lb.get("archivedAt")
        ]

    def find_users(self, reference: str) -> list[dict]:
        """Users whose name, display name or email is ``reference``."""
        wanted = reference.strip().lower()
        return [
            u for u in self._pages("list_users", "users", query=reference, limit=_PAGE_SIZE)
            if wanted in {str(u.get(k) or "").strip().lower() for k in ("name", "displayName", "email")}
        ]

    # -- writes --

    def create_project(self, *, name: str, team_id: str, description: str = "", content: str = "") -> dict:
        """``description``: the one-line summary. ``content``: the project's document (markdown)."""
        created = self.call(
            "save_project", name=name, addTeams=[team_id], summary=description or None, description=content or None,
        )
        project = _flatten_project(created) if isinstance(created, dict) else {}
        if not project.get("id"):
            raise LinearError(f"Linear did not create project {name!r}")
        return project

    def create_issue(
        self,
        *,
        team_id: str,
        title: str,
        description: str = "",
        project_id: str | None = None,
        parent_id: str | None = None,
        priority: int | None = None,
        state_id: str | None = None,
        assignee_id: str | None = None,
        milestone_id: str | None = None,
        labels: list[str] | None = None,
    ) -> dict:
        """``labels``: label names (or IDs)."""
        created = self.call(
            "save_issue", team=team_id, title=title, description=description or None, project=project_id,
            parentId=parent_id, priority=priority, state=state_id, assignee=assignee_id,
            milestone=milestone_id, labels=labels or None,
        )
        issue = self._saved_issue(created)
        if not issue.get("id"):
            raise LinearError(f"Linear did not create issue {title!r}")
        return issue

    def update_issue(self, issue_id: str, *, state_id: str) -> dict:
        """Move an issue to another workflow state."""
        issue = self._saved_issue(self.call("save_issue", id=issue_id, state=state_id))
        if not issue.get("id"):
            raise LinearError(f"Linear did not update issue {issue_id}")
        return issue

    def _saved_issue(self, saved: Any) -> dict:
        """The issue a save returned, with the fields pralph tracks."""
        node = saved.get("issue") if isinstance(saved, dict) and isinstance(saved.get("issue"), dict) else saved
        if not isinstance(node, dict) or not node.get("id"):
            return {}
        if not all(k in node for k in ("uuid", "status", "statusType", "url")):
            # The save answered with less than we track: read the rest.
            node = {**node, **(self.call("get_issue", id=node.get("uuid") or node["id"],
                                         fields=["id", "uuid", "title", "url", "status", "statusType"]) or {})}
        return _flatten_issue(node)

    def create_comment(self, issue_id: str, body: str) -> None:
        self.call("save_comment", issueId=issue_id, body=body)

    def create_blocks_relation(self, blocker_id: str, blocked_id: str) -> None:
        """Record that ``blocker_id`` blocks ``blocked_id``."""
        self.call("save_issue", id=blocked_id, blockedBy=[blocker_id])


# ── helpers ──────────────────────────────────────────────────────────


def _flatten_project(node: dict) -> dict:
    teams = node.get("teams") or []
    if isinstance(teams, dict):
        teams = teams.get("nodes") or []
    return {
        "id": node.get("uuid") or node.get("id", ""),
        "identifier": node.get("id", "") if node.get("uuid") else "",
        "name": node.get("name", "") or "",
        "url": node.get("url", "") or "",
        "teams": [t for t in teams if isinstance(t, dict) and t.get("id")],
    }


def _flatten_issue(node: dict) -> dict:
    priority = node.get("priority")
    if isinstance(priority, dict):
        priority = priority.get("value")
    milestone = node.get("projectMilestone")
    labels = node.get("labels") or []
    if isinstance(labels, dict):
        labels = labels.get("nodes") or []
    return {
        "id": node.get("uuid") or node.get("id", ""),
        "identifier": node.get("id", "") if node.get("uuid") else node.get("identifier", ""),
        "title": node.get("title", "") or "",
        "description": node.get("description") or "",
        "priority": int(priority or 0),
        "url": node.get("url", "") or "",
        "state_name": node.get("status") or "",
        "state_type": node.get("statusType") or "",
        "parent_id": node.get("parentId") or None,
        "assignee_id": node.get("assigneeId") or None,
        "milestone_id": (milestone.get("id") if isinstance(milestone, dict) else milestone) or None,
        # An issue's labels come as bare names.
        "labels": [str(lb.get("name", "")) if isinstance(lb, dict) else str(lb) for lb in labels],
        "updated_at": node.get("updatedAt") or "",
        "blocked_by": [],
        "relations_truncated": True,
    }
