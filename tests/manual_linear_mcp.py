"""Check the tools pralph calls against Linear's live MCP server.

Asks the server for its tool list and compares it with ``pralph.linear.TOOLS``:
every tool must exist and take every argument pralph passes, and the fields
pralph asks for must be on offer. A failure means Linear changed a tool and
``pralph/linear.py`` needs to follow. Read-only — it calls no tool.

Not part of the test suite (it talks to mcp.linear.app and needs the login
of ``pralph linear oauth``). Run it by hand:

    python tests/manual_linear_mcp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pralph import linear, linear_oauth  # noqa: E402

# Values pralph passes for a tool's ``fields`` argument.
FIELDS = {
    "list_issues": linear.ISSUE_FIELDS,
    "get_issue": ["id", "uuid", "title", "url", "status", "statusType", "relations"],
    "list_projects": ["id", "uuid", "name", "url", "teams"],
}


def main() -> int:
    oauth = linear_oauth.stored_login(Path.home() / ".pralph")
    if oauth is None:
        print("Not logged in to Linear — run 'pralph linear oauth'")
        return 2
    client = linear.LinearClient(oauth=oauth)
    try:
        client.viewer()  # logs in, refreshing the token if needed
        tools = {tool["name"]: tool for tool in client._transport.list_tools()}
    except linear.LinearError as e:
        print(f"Could not ask Linear: {e}")
        return 2

    failures = 0
    for name, arguments in linear.TOOLS.items():
        problems: list[str] = []
        if name not in tools:
            problems.append("no such tool")
        else:
            properties = (tools[name].get("inputSchema") or {}).get("properties") or {}
            problems += [f"no argument {a!r}" for a in arguments if a not in properties]
            offered = ((properties.get("fields") or {}).get("items") or {}).get("enum")
            if name in FIELDS and offered is not None:
                problems += [f"no field {f!r}" for f in FIELDS[name] if f not in offered]
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {name}" + (": " + ", ".join(problems) if problems else ""))

    print(f"\n{len(linear.TOOLS)} tools checked against {len(tools)} on the server, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
