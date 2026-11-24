"""OAuth login for Linear — ``pralph linear oauth`` (stdlib only).

pralph talks to Linear through Linear's MCP server, and logs in to it the way
other MCP clients do: OAuth 2.1 with dynamic client registration and PKCE.
Nothing has to be set up in Linear first — pralph registers itself, the
browser takes the user to Linear to approve, Linear redirects to a loopback
port pralph listens on, and the code is exchanged for an access token (valid
for about a day) plus a refresh token. The login is kept in
``~/.pralph/linear-oauth.json`` and refreshed when it runs out; refresh tokens
rotate, so every refresh is saved.

See https://linear.app/docs/mcp
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socketserver
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

from pralph import __version__
from pralph.linear import MCP_URL, LinearError

ISSUER = "https://mcp.linear.app"
METADATA_URL = f"{ISSUER}/.well-known/oauth-authorization-server"
# Used when the metadata document cannot be read.
DEFAULT_ENDPOINTS = {
    "authorization_endpoint": f"{ISSUER}/authorize",
    "token_endpoint": f"{ISSUER}/token",
    "registration_endpoint": f"{ISSUER}/register",
    "revocation_endpoint": f"{ISSUER}/token",
}

CLIENT_NAME = "pralph"
DEFAULT_PORT = 8765
CALLBACK_PATH = "/oauth/callback"
SCOPES = "read write"
TOKEN_FILE = "linear-oauth.json"

# Refresh this long before the access token runs out, so a request that is
# about to be sent doesn't arrive with a token that has just expired.
_EXPIRY_MARGIN = 120

# request(url, form=None, json_body=None) -> parsed JSON answer. GET without a body.
Request = Callable[..., dict]


def redirect_uri(port: int = DEFAULT_PORT) -> str:
    return f"http://localhost:{port}{CALLBACK_PATH}"


def request(url: str, *, form: dict | None = None, json_body: dict | None = None) -> dict:
    """Talk to one of the OAuth endpoints: GET, a form POST, or a JSON POST."""
    headers = {"Accept": "application/json", "User-Agent": f"pralph/{__version__}"}
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        detail = raw[:200]
        if isinstance(payload, dict):
            detail = payload.get("error_description") or payload.get("error") or detail
        raise LinearError(
            f"Linear refused the OAuth request (HTTP {e.code}): {detail}", retryable=e.code >= 500,
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LinearError(f"Could not reach Linear: {e}", retryable=True) from e
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        raise LinearError(f"Invalid OAuth response from Linear: {e}") from e
    return payload if isinstance(payload, dict) else {}


def endpoints(req: Request = request) -> dict:
    """The authorization server's endpoints, from its metadata document."""
    try:
        metadata = req(METADATA_URL)
    except LinearError:
        metadata = {}
    return {key: str(metadata.get(key) or default) for key, default in DEFAULT_ENDPOINTS.items()}


# ── stored login ─────────────────────────────────────────────────────


class OAuthSession:
    """A Linear OAuth login on disk: hands out a valid access token, refreshing as needed."""

    def __init__(
        self,
        path: Path,
        data: dict,
        *,
        req: Request | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.data = data
        self._req = req or request
        self._clock = clock

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> OAuthSession | None:
        data = _read_token_file(path)
        return cls(path, data, **kwargs) if data else None

    # -- what the login is --

    @property
    def scope(self) -> str:
        return str(self.data.get("scope") or "")

    @property
    def expires_at(self) -> float:
        return float(self.data.get("expires_at") or 0)

    def _expired(self) -> bool:
        # 0: Linear gave no lifetime — use the token until it is rejected.
        return bool(self.expires_at) and self.expires_at - _EXPIRY_MARGIN <= self._clock()

    def _endpoint(self, name: str) -> str:
        return str(self.data.get(name) or DEFAULT_ENDPOINTS[name])

    # -- tokens --

    def access_token(self, *, force_refresh: bool = False) -> str:
        """A valid access token. ``force_refresh``: Linear just rejected the current one."""
        if force_refresh or self._expired():
            self.refresh(rejected=self.data.get("access_token") if force_refresh else None)
        return str(self.data.get("access_token") or "")

    def refresh(self, *, rejected: str | None = None) -> None:
        # Another pralph may have refreshed already. Refresh tokens rotate, so
        # use what it saved rather than spending the old token a second time.
        on_disk = _read_token_file(self.path)
        if on_disk and on_disk.get("access_token") != self.data.get("access_token"):
            self.data = on_disk
            if not self._expired() and on_disk.get("access_token") != rejected:
                return

        if not self.data.get("refresh_token"):
            raise LinearError("The Linear login has no refresh token — run 'pralph linear oauth' again")
        form = {
            "grant_type": "refresh_token",
            "refresh_token": self.data["refresh_token"],
            "client_id": self.data.get("client_id", ""),
            "resource": self.data.get("resource") or MCP_URL,
        }
        try:
            answer = self._req(self._endpoint("token_endpoint"), form=form)
        except LinearError as e:
            if e.retryable:
                raise
            raise LinearError(f"The Linear login has expired ({e}) — run 'pralph linear oauth' again") from e
        self.data = {**self.data, **_token_fields(answer, self._clock())}
        self.save()

    def save(self) -> None:
        """Write the login atomically, readable by the user only."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(self.data, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def revoke(self) -> bool:
        """Revoke the login at Linear and delete it locally. False if Linear could not be told."""
        revoked = True
        for hint in ("refresh_token", "access_token"):
            if not self.data.get(hint):
                continue
            try:
                self._req(self._endpoint("revocation_endpoint"), form={
                    "token": self.data[hint], "token_type_hint": hint,
                    "client_id": self.data.get("client_id", ""),
                })
            except LinearError as e:
                # Refused means there is nothing left to revoke (expired, revoked in Linear).
                revoked = revoked and not e.retryable
        self.path.unlink(missing_ok=True)
        return revoked


def _read_token_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("access_token"):
        return None
    return data


def _token_fields(answer: dict, now: float) -> dict:
    """The parts of a token response worth keeping."""
    token = answer.get("access_token")
    if not token:
        raise LinearError(
            "Linear did not return an access token: "
            + str(answer.get("error_description") or answer.get("error") or "empty response")
        )
    scope = answer.get("scope")
    fields = {
        "access_token": token,
        "expires_at": now + float(answer["expires_in"]) if answer.get("expires_in") else 0,
        "scope": " ".join(scope) if isinstance(scope, list) else str(scope or ""),
    }
    if answer.get("refresh_token"):
        fields["refresh_token"] = answer["refresh_token"]
    return fields


def token_path(home_dir: Path) -> Path:
    return home_dir / TOKEN_FILE


def stored_login(home_dir: Path) -> OAuthSession | None:
    """The login of ``pralph linear oauth``, if there is one."""
    return OAuthSession.load(token_path(home_dir))


# ── login ────────────────────────────────────────────────────────────


def pkce_pair() -> tuple[str, str]:
    """(code verifier, S256 code challenge)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def register_client(registration_endpoint: str, redirect: str, req: Request = request) -> str:
    """Register pralph as a public OAuth client (RFC 7591) and return its client ID."""
    answer = req(registration_endpoint, json_body={
        "client_name": CLIENT_NAME,
        "redirect_uris": [redirect],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    if not answer.get("client_id"):
        raise LinearError("Linear did not register pralph as an OAuth client")
    return str(answer["client_id"])


def authorize_url(
    authorization_endpoint: str, client: str, *, redirect: str, state: str, challenge: str,
    scopes: str = SCOPES, resource: str = MCP_URL,
) -> str:
    params = {
        "client_id": client,
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": scopes,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    return f"{authorization_endpoint}?{urllib.parse.urlencode(params)}"


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pralph</title></head>
<body style="font-family: system-ui, sans-serif; margin: 4rem auto; max-width: 32rem">
<h2>{title}</h2><p>{text}</p></body></html>"""


class _LoopbackServer(HTTPServer):
    def server_bind(self) -> None:
        # HTTPServer looks up the host's FQDN here, which can hang for a long
        # time on a machine with slow or no DNS. Nobody needs the name.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


def wait_for_code(port: int, state: str, *, timeout: float = 300, ready: Callable[[], None] | None = None) -> str:
    """Listen on the loopback port for Linear's redirect and return the authorization code."""
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server API
            url = urllib.parse.urlsplit(self.path)
            if url.path != CALLBACK_PATH:
                self.send_error(404)
                return
            query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            if query.get("error"):
                result["error"] = query.get("error_description") or query["error"]
            elif not query.get("code"):
                result["error"] = "Linear sent no authorization code"
            elif not secrets.compare_digest(query.get("state", ""), state):
                result["error"] = "the answer does not belong to this login attempt (state mismatch)"
            else:
                result["code"] = query["code"]
            failed = "error" in result
            body = _PAGE.format(
                title="Login failed" if failed else "pralph is connected to Linear",
                text="See the terminal for details." if failed else "You can close this tab.",
            ).encode()
            self.send_response(400 if failed else 200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # keep the terminal clean
            return

    try:
        server = _LoopbackServer(("127.0.0.1", port), Handler)
    except OSError as e:
        raise LinearError(
            f"Cannot listen on port {port} ({e.strerror or e}) — choose another with --port"
        ) from e
    try:
        if ready is not None:
            ready()
        deadline = time.monotonic() + timeout
        while not result:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LinearError("Timed out waiting for the approval in the browser")
            server.timeout = min(remaining, 1.0)
            server.handle_request()
    finally:
        server.server_close()
    if "error" in result:
        raise LinearError(f"Linear login failed: {result['error']}")
    return result["code"]


def login(
    path: Path,
    *,
    port: int = DEFAULT_PORT,
    open_browser: Callable[[str], Any] | None = webbrowser.open,
    echo: Callable[[str], None] = print,
    req: Request | None = None,
    timeout: float = 300,
) -> OAuthSession:
    """Run the browser login and store the result at ``path``.

    ``open_browser=None`` only prints the URL (a machine without a browser:
    open it elsewhere, as long as the redirect reaches this port).
    """
    req = req or request
    found = endpoints(req)
    redirect = redirect_uri(port)

    # A registration is tied to its redirect URI: keep using ours while the
    # port stays the same, register again otherwise.
    previous = _read_token_file(path) or {}
    if previous.get("client_id") and previous.get("redirect_uri") == redirect:
        client = str(previous["client_id"])
    else:
        client = register_client(found["registration_endpoint"], redirect, req)

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(24)
    url = authorize_url(found["authorization_endpoint"], client, redirect=redirect, state=state, challenge=challenge)

    def ready() -> None:
        echo(f"  Open this URL to approve the access:\n  {url}")
        if open_browser is not None:
            open_browser(url)
        echo("  Waiting for the approval in the browser (Ctrl-C to cancel) ...")

    code = wait_for_code(port, state, timeout=timeout, ready=ready)
    answer = req(found["token_endpoint"], form={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect,
        "client_id": client,
        "code_verifier": verifier,
        "resource": MCP_URL,
    })
    session = OAuthSession(
        path,
        {
            "client_id": client, "redirect_uri": redirect, "resource": MCP_URL,
            "token_endpoint": found["token_endpoint"], "revocation_endpoint": found["revocation_endpoint"],
            **_token_fields(answer, time.time()),
        },
        req=req,
    )
    session.save()
    return session
