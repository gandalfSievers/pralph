"""Minimal Linear GraphQL client (stdlib only).

Covers exactly what pralph needs: teams and workflow states, creating a
project with issues / sub-issues / blocking relations, moving issues through
the workflow, commenting, and reading a project's issues back.

Authentication: a personal API key (``lin_api_...``, sent as-is) or an OAuth
access token (sent as ``Bearer``). See https://linear.app/developers/graphql
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from pralph import __version__

API_URL = "https://api.linear.app/graphql"

# Issues per page / relations per issue. Linear rates queries by complexity
# (nested connections multiply), so keep the product modest. An issue with
# more relations than this is reported with ``relations_truncated``.
_PAGE_SIZE = 50
_RELATIONS_PER_ISSUE = 25

Transport = Callable[[str, dict], dict]


class LinearError(Exception):
    """A Linear API call failed (network, auth, validation or rate limit).

    ``retryable``: trying again may work. ``outcome_unknown``: the request may
    have reached Linear and been carried out anyway (timeout, dropped
    connection, 5xx) — repeating a mutation then risks doing it twice.
    """

    def __init__(self, message: str, *, retryable: bool = False, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


def resolve_api_key(config_get: Callable[[str, Any], Any] | None = None) -> str:
    """API key from ``LINEAR_API_KEY``, falling back to the ``linear_api_key`` config value."""
    key = os.environ.get("LINEAR_API_KEY", "").strip()
    if not key and config_get is not None:
        key = str(config_get("linear_api_key", "") or "").strip()
    return key


# ── documents ────────────────────────────────────────────────────────

_ISSUE_FIELDS = """
fragment IssueFields on Issue {
  id
  identifier
  title
  description
  priority
  url
  trashed
  state { id name type }
  parent { id }
  inverseRelations(first: %d) { nodes { type issue { id } } pageInfo { hasNextPage } }
}
""" % _RELATIONS_PER_ISSUE

_ISSUE_RESULT = "success issue { id identifier url title state { id name type } }"

Q_VIEWER = "query { viewer { id name email } organization { name urlKey } }"

Q_TEAMS = "query { teams(first: 100) { nodes { id key name } } }"

Q_STATES = """
query($teamId: String!) {
  team(id: $teamId) { id key name states(first: 100) { nodes { id name type position } } }
}
"""

Q_PROJECT = """
query($id: String!) {
  project(id: $id) { id name url slugId teams(first: 10) { nodes { id key name } } }
}
"""

Q_PROJECTS = """
query($filter: ProjectFilter) {
  projects(filter: $filter, first: 20) { nodes { id name url slugId teams(first: 10) { nodes { id key name } } } }
}
"""

Q_PROJECT_ISSUES = _ISSUE_FIELDS + """
query($id: String!, $after: String) {
  project(id: $id) {
    issues(first: %d, after: $after) {
      nodes { ...IssueFields }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""" % _PAGE_SIZE

M_PROJECT_CREATE = """
mutation($input: ProjectCreateInput!) {
  projectCreate(input: $input) { success project { id name url slugId } }
}
"""

M_ISSUE_CREATE = "mutation($input: IssueCreateInput!) { issueCreate(input: $input) { %s } }" % _ISSUE_RESULT

M_ISSUE_UPDATE = (
    "mutation($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { %s } }"
    % _ISSUE_RESULT
)

M_COMMENT_CREATE = "mutation($input: CommentCreateInput!) { commentCreate(input: $input) { success } }"

M_RELATION_CREATE = """
mutation($input: IssueRelationCreateInput!) {
  issueRelationCreate(input: $input) { success }
}
"""

# Every document the client can send — lets a schema check validate them all.
DOCUMENTS = {
    "viewer": Q_VIEWER,
    "teams": Q_TEAMS,
    "states": Q_STATES,
    "project": Q_PROJECT,
    "projects": Q_PROJECTS,
    "project_issues": Q_PROJECT_ISSUES,
    "project_create": M_PROJECT_CREATE,
    "issue_create": M_ISSUE_CREATE,
    "issue_update": M_ISSUE_UPDATE,
    "comment_create": M_COMMENT_CREATE,
    "relation_create": M_RELATION_CREATE,
}


# ── client ───────────────────────────────────────────────────────────


class LinearClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: Transport | None = None,
        url: str = API_URL,
        timeout: int = 30,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self._transport = transport or self._http_transport

    # -- transport --

    def _auth_header(self) -> str:
        key = self.api_key
        if key.lower().startswith("bearer "):
            return key
        if key.startswith("lin_oauth_"):
            return f"Bearer {key}"
        return key

    def _http_transport(self, query: str, variables: dict) -> dict:
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": self._auth_header(),
                "User-Agent": f"pralph/{__version__}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("errors"):
                return payload  # GraphQL errors arrive with 4xx — handled by graphql()
            raise LinearError(
                f"HTTP {e.code} from Linear: {raw[:200]}",
                retryable=e.code == 429 or e.code >= 500,
                outcome_unknown=e.code >= 500,
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LinearError(f"Could not reach Linear: {e}", retryable=True, outcome_unknown=True) from e
        except json.JSONDecodeError as e:
            raise LinearError(f"Invalid response from Linear: {e}", outcome_unknown=True) from e

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        """Run a document and return its ``data``. Raises LinearError on any failure.

        Failures that are worth another try are retried with backoff — except
        a mutation whose outcome is unknown: Linear's creates are not
        idempotent, so repeating one that did go through would leave a second
        issue or comment behind. Those surface to the caller, which catches up
        on its next sync.
        """
        if not self.api_key:
            raise LinearError("No Linear API key — set LINEAR_API_KEY (Linear → Settings → Security & access)")
        is_mutation = query.lstrip().startswith("mutation")
        last: LinearError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._transport(query, variables or {})
                errors = payload.get("errors") if isinstance(payload, dict) else None
                if errors:
                    raise _error_from_graphql(errors)
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, dict):
                    raise LinearError("Linear returned no data")
                return data
            except LinearError as e:
                last = e
                if not e.retryable or attempt >= self.max_retries or (is_mutation and e.outcome_unknown):
                    raise
                self._sleep(2.0 ** attempt)
        raise last or LinearError("Linear request failed")  # pragma: no cover

    # -- reads --

    def viewer(self) -> dict:
        data = self.graphql(Q_VIEWER)
        return {**(data.get("viewer") or {}), "organization": data.get("organization") or {}}

    def teams(self) -> list[dict]:
        return _nodes(self.graphql(Q_TEAMS).get("teams"))

    def workflow_states(self, team_id: str) -> list[dict]:
        team = self.graphql(Q_STATES, {"teamId": team_id}).get("team") or {}
        return sorted(_nodes(team.get("states")), key=lambda st: st.get("position", 0))

    def project(self, project_id: str) -> dict | None:
        try:
            return self.graphql(Q_PROJECT, {"id": project_id}).get("project")
        except LinearError as e:
            if "not found" in str(e).lower():
                return None
            raise

    def find_projects(self, *, name: str | None = None, slug_id: str | None = None) -> list[dict]:
        flt: dict[str, Any] = {}
        if name:
            flt["name"] = {"eqIgnoreCase": name}
        if slug_id:
            flt["slugId"] = {"eq": slug_id}
        return _nodes(self.graphql(Q_PROJECTS, {"filter": flt}).get("projects"))

    def project_issues(self, project_id: str) -> list[dict]:
        """All (non-archived, non-trashed) issues of a project, flattened.

        Each issue carries ``parent_id``, ``blocked_by`` (IDs of issues that
        block it) and ``relations_truncated`` (the issue has more relations
        than were fetched, so ``blocked_by`` may be incomplete).
        """
        issues: list[dict] = []
        after: str | None = None
        while True:
            project = self.graphql(Q_PROJECT_ISSUES, {"id": project_id, "after": after}).get("project")
            if project is None:
                raise LinearError(f"Linear project {project_id} not found")
            connection = project.get("issues") or {}
            for node in _nodes(connection):
                if node.get("trashed"):
                    continue
                issues.append(_flatten_issue(node))
            page = connection.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                return issues
            after = page["endCursor"]

    # -- writes --

    def create_project(self, *, name: str, team_id: str, description: str = "", content: str = "") -> dict:
        payload: dict[str, Any] = {"name": name, "teamIds": [team_id]}
        if description:
            payload["description"] = description
        if content:
            payload["content"] = content
        result = self.graphql(M_PROJECT_CREATE, {"input": payload}).get("projectCreate") or {}
        if not result.get("success") or not result.get("project"):
            raise LinearError(f"Linear did not create project {name!r}")
        return result["project"]

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
    ) -> dict:
        payload: dict[str, Any] = {"teamId": team_id, "title": title}
        if description:
            payload["description"] = description
        if project_id:
            payload["projectId"] = project_id
        if parent_id:
            payload["parentId"] = parent_id
        if priority is not None:
            payload["priority"] = priority
        if state_id:
            payload["stateId"] = state_id
        result = self.graphql(M_ISSUE_CREATE, {"input": payload}).get("issueCreate") or {}
        if not result.get("success") or not result.get("issue"):
            raise LinearError(f"Linear did not create issue {title!r}")
        return _flatten_issue(result["issue"])

    def update_issue(self, issue_id: str, **fields: Any) -> dict:
        """Update an issue. Field names are IssueUpdateInput keys (``stateId``, ``title`` ...)."""
        result = self.graphql(M_ISSUE_UPDATE, {"id": issue_id, "input": fields}).get("issueUpdate") or {}
        if not result.get("success") or not result.get("issue"):
            raise LinearError(f"Linear did not update issue {issue_id}")
        return _flatten_issue(result["issue"])

    def create_comment(self, issue_id: str, body: str) -> None:
        result = self.graphql(M_COMMENT_CREATE, {"input": {"issueId": issue_id, "body": body}})
        if not (result.get("commentCreate") or {}).get("success"):
            raise LinearError(f"Linear did not create the comment on {issue_id}")

    def create_blocks_relation(self, blocker_id: str, blocked_id: str) -> None:
        """Record that ``blocker_id`` blocks ``blocked_id``."""
        result = self.graphql(M_RELATION_CREATE, {
            "input": {"issueId": blocker_id, "relatedIssueId": blocked_id, "type": "blocks"},
        })
        if not (result.get("issueRelationCreate") or {}).get("success"):
            raise LinearError(f"Linear did not create the relation {blocker_id} blocks {blocked_id}")


# ── helpers ──────────────────────────────────────────────────────────


def _nodes(connection: Any) -> list[dict]:
    if not isinstance(connection, dict):
        return []
    return [n for n in connection.get("nodes") or [] if isinstance(n, dict)]


def _flatten_issue(node: dict) -> dict:
    state = node.get("state") or {}
    relations = node.get("inverseRelations")
    relations_page = (relations.get("pageInfo") or {}) if isinstance(relations, dict) else {}
    blocked_by = [
        rel["issue"]["id"]
        for rel in _nodes(node.get("inverseRelations"))
        if rel.get("type") == "blocks" and isinstance(rel.get("issue"), dict) and rel["issue"].get("id")
    ]
    return {
        "id": node.get("id", ""),
        "identifier": node.get("identifier", ""),
        "title": node.get("title", "") or "",
        "description": node.get("description") or "",
        "priority": node.get("priority") or 0,
        "url": node.get("url", "") or "",
        "state_id": state.get("id", ""),
        "state_name": state.get("name", ""),
        "state_type": state.get("type", ""),
        "parent_id": (node.get("parent") or {}).get("id"),
        "blocked_by": blocked_by,
        "relations_truncated": bool(relations_page.get("hasNextPage")),
    }


def _error_from_graphql(errors: list) -> LinearError:
    messages: list[str] = []
    retryable = False
    for err in errors:
        if not isinstance(err, dict):
            continue
        ext = err.get("extensions") or {}
        code = str(ext.get("code", ""))
        if code == "RATELIMITED":
            retryable = True
        if code == "AUTHENTICATION_ERROR":
            return LinearError(
                "Linear rejected the API key — check LINEAR_API_KEY "
                "(Linear → Settings → Security & access → Personal API keys)"
            )
        messages.append(str(ext.get("userPresentableMessage") or err.get("message") or code or "unknown error"))
    return LinearError("Linear API error: " + "; ".join(messages or ["unknown error"]), retryable=retryable)
