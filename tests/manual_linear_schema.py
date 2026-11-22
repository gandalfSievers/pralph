"""Check every GraphQL document pralph sends against Linear's live schema.

No API key needed: Linear validates a document before it checks
authentication, so an unauthenticated request answers the question —
AUTHENTICATION_ERROR means the document is valid, GRAPHQL_VALIDATION_FAILED
means the schema moved and ``pralph/linear.py`` needs to follow.

Not part of the test suite (it talks to api.linear.app). Run it by hand:

    python tests/manual_linear_schema.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pralph.linear import API_URL, DOCUMENTS  # noqa: E402

# Input fields pralph puts into variables — document validation doesn't cover those.
INPUT_FIELDS = {
    "ProjectCreateInput": ["name", "teamIds", "description", "content"],
    "IssueCreateInput": ["teamId", "title", "description", "projectId", "parentId", "priority", "stateId"],
    "IssueUpdateInput": ["stateId"],
    "CommentCreateInput": ["issueId", "body"],
    "IssueRelationCreateInput": ["issueId", "relatedIssueId", "type"],
    "ProjectFilter": ["name", "slugId"],
    "StringComparator": ["eq", "eqIgnoreCase"],
}


def post(query: str) -> dict:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query, "variables": {}}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "pralph-schema-check"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def main() -> int:
    failures = 0
    for name, document in DOCUMENTS.items():
        errors = post(document).get("errors") or []
        codes = {(e.get("extensions") or {}).get("code") for e in errors}
        ok = bool(errors) and codes <= {"AUTHENTICATION_ERROR"}
        print(f"{'ok  ' if ok else 'FAIL'} document {name}")
        if not ok:
            failures += 1
            for e in errors:
                print(f"       {e.get('message')}")

    for type_name, wanted in INPUT_FIELDS.items():
        data = post(f'query {{ __type(name: "{type_name}") {{ inputFields {{ name }} }} }}').get("data") or {}
        have = {f["name"] for f in (data.get("__type") or {}).get("inputFields") or []}
        missing = [f for f in wanted if f not in have]
        print(f"{'ok  ' if not missing else 'FAIL'} input {type_name}" + (f" — missing {missing}" if missing else ""))
        failures += bool(missing)

    enum = post('query { __type(name: "IssueRelationType") { enumValues { name } } }').get("data") or {}
    has_blocks = "blocks" in {v["name"] for v in (enum.get("__type") or {}).get("enumValues") or []}
    print(f"{'ok  ' if has_blocks else 'FAIL'} enum IssueRelationType.blocks")
    failures += not has_blocks
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
