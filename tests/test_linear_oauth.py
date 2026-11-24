"""Tests for the Linear login (``pralph linear oauth``).

Linear's authorization server is played by ``FakeAuthServer`` (metadata,
client registration, token endpoint) and by a "browser" that calls the
loopback redirect the way Linear would. Nothing here reaches the network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

from pralph import linear as L
from pralph import linear_oauth as O
from pralph import linear_sync
from pralph.cli import main
from pralph.linear import LinearClient, LinearError
from pralph.state import StateManager

TOKEN = O.DEFAULT_ENDPOINTS["token_endpoint"]
REGISTER = O.DEFAULT_ENDPOINTS["registration_endpoint"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)

    # A test that forgets its fakes must not register a client with Linear,
    # open the developer's browser or call the real MCP server.
    def offline(*args, **kwargs):
        raise AssertionError("this test would have talked to Linear")

    monkeypatch.setattr(O, "request", offline)
    monkeypatch.setattr(O.webbrowser, "open", offline)
    monkeypatch.setattr(L.McpTransport, "_post", offline)


@pytest.fixture
def token_file(tmp_path) -> Path:
    return tmp_path / "home" / ".pralph" / O.TOKEN_FILE


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeAuthServer:
    """Linear's OAuth endpoints for the MCP server, as ``linear_oauth.request``."""

    def __init__(self) -> None:
        self.metadata: dict | None = {"issuer": O.ISSUER, **O.DEFAULT_ENDPOINTS}
        self.requests: list[tuple[str, dict]] = []  # (url, body) — without the metadata reads
        self.clients: dict[str, dict] = {}
        self.issued = 0
        self.valid_refresh: str | None = None
        self.challenge: str | None = None  # set by the "browser" from the authorize URL
        self.revoked: list[str] = []
        self.fail: LinearError | None = None
        self.metadata_reads = 0

    def __call__(self, url: str, *, form: dict | None = None, json_body: dict | None = None) -> dict:
        if url == O.METADATA_URL:
            assert form is None and json_body is None
            self.metadata_reads += 1
            if self.metadata is None:
                raise LinearError("Linear refused the OAuth request (HTTP 404): not found")
            return dict(self.metadata)
        self.requests.append((url, dict(form if form is not None else json_body or {})))
        if self.fail is not None:
            raise self.fail
        endpoints = {**O.DEFAULT_ENDPOINTS, **(self.metadata or {})}
        if url == endpoints["registration_endpoint"]:
            assert json_body is not None, "registration is a JSON request"
            client_id = f"client-{len(self.clients) + 1}"
            self.clients[client_id] = json_body
            return {"client_id": client_id, **json_body}
        assert url == endpoints["token_endpoint"] and form is not None
        assert form["client_id"] in self.clients, "an unknown client"
        if "grant_type" not in form:  # revocation shares the token endpoint
            self.revoked.append(form["token"])
            return {}
        assert form["resource"] == L.MCP_URL, "the token is asked for the MCP server"
        if form["grant_type"] == "authorization_code":
            assert form["code"] == "the-code"
            assert form["redirect_uri"] in self.clients[form["client_id"]]["redirect_uris"]
            digest = hashlib.sha256(form["code_verifier"].encode()).digest()
            assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == self.challenge, "PKCE"
        else:
            assert form["grant_type"] == "refresh_token"
            if form["refresh_token"] != self.valid_refresh:
                raise LinearError("Linear refused the OAuth request (HTTP 400): invalid_grant")
        self.issued += 1
        self.valid_refresh = f"refresh-{self.issued}"  # refresh tokens rotate
        return {"access_token": f"token-{self.issued}", "token_type": "Bearer", "expires_in": 86100,
                "scope": "read write", "refresh_token": self.valid_refresh}

    def of(self, url: str) -> list[dict]:
        return [body for u, body in self.requests if u == url]


def browser(server: FakeAuthServer, *, answer: dict | None = None, seen: list[str] | None = None):
    """Stand-in for the user's browser: 'approves' and follows Linear's redirect."""

    def open_url(url: str) -> None:
        query = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}
        server.challenge = query["code_challenge"]
        if seen is not None:
            seen.append(url)
        params = answer if answer is not None else {"code": "the-code", "state": query["state"]}

        def follow_redirect() -> None:
            try:
                urllib.request.urlopen(f"{query['redirect_uri']}?{urllib.parse.urlencode(params)}", timeout=5).read()
            except urllib.error.HTTPError:
                pass  # the callback page for a failed login is a 400

        threading.Thread(target=follow_redirect, daemon=True).start()

    return open_url


def do_login(token_file: Path, server: FakeAuthServer, **kwargs) -> O.OAuthSession:
    kwargs.setdefault("open_browser", browser(server))
    kwargs.setdefault("port", free_port())
    return O.login(token_file, req=server, echo=lambda m: None, timeout=5, **kwargs)


class Clock:
    def __init__(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now


# ── login ─────────────────────────────────────────────────────────────


class TestLogin:
    def test_authorize_url(self) -> None:
        verifier, challenge = O.pkce_pair()
        assert 43 <= len(verifier) <= 128
        url = O.authorize_url(O.DEFAULT_ENDPOINTS["authorization_endpoint"], "client-1",
                              redirect=O.redirect_uri(), state="st", challenge=challenge)
        split = urllib.parse.urlsplit(url)
        assert f"{split.scheme}://{split.netloc}{split.path}" == "https://mcp.linear.app/authorize"
        assert {k: v[0] for k, v in urllib.parse.parse_qs(split.query).items()} == {
            "client_id": "client-1", "redirect_uri": "http://localhost:8765/oauth/callback",
            "response_type": "code", "scope": "read write", "state": "st",
            "code_challenge": challenge, "code_challenge_method": "S256",
            "resource": "https://mcp.linear.app/mcp",
        }

    def test_login_registers_pralph_and_stores_the_tokens(self, token_file) -> None:
        server = FakeAuthServer()
        seen: list[str] = []
        session = do_login(token_file, server, open_browser=browser(server, seen=seen))

        (registration,) = server.of(REGISTER)
        redirect = registration["redirect_uris"][0]
        assert redirect.startswith("http://localhost:") and redirect.endswith("/oauth/callback")
        assert registration == {
            "client_name": "pralph", "redirect_uris": [redirect],
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }, "a public client: no secret, and nothing about the user or the project"
        assert "client_id=client-1" in seen[0]

        (exchange,) = server.of(TOKEN)
        assert exchange["grant_type"] == "authorization_code" and exchange["client_id"] == "client-1"
        assert exchange["redirect_uri"] == redirect and "client_secret" not in exchange

        assert session.access_token() == "token-1"
        stored = json.loads(token_file.read_text())
        assert stored["access_token"] == "token-1" and stored["refresh_token"] == "refresh-1"
        assert stored["client_id"] == "client-1" and stored["redirect_uri"] == redirect
        assert stored["scope"] == "read write" and stored["expires_at"] > time.time() + 80000
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600, "readable by the user only"

    def test_registration_is_reused_until_the_port_changes(self, token_file) -> None:
        server = FakeAuthServer()
        port = free_port()
        do_login(token_file, server, port=port)
        do_login(token_file, server, port=port)
        assert len(server.clients) == 1 and json.loads(token_file.read_text())["access_token"] == "token-2"

        do_login(token_file, server, port=free_port())
        assert len(server.clients) == 2, "a registration belongs to its redirect URI"
        assert json.loads(token_file.read_text())["client_id"] == "client-2"

    def test_endpoints_come_from_the_servers_metadata(self, token_file) -> None:
        server = FakeAuthServer()
        server.metadata.update(token_endpoint=f"{O.ISSUER}/v2/token", registration_endpoint=f"{O.ISSUER}/v2/register",
                               authorization_endpoint=f"{O.ISSUER}/v2/authorize",
                               revocation_endpoint=f"{O.ISSUER}/v2/token")
        seen: list[str] = []
        session = do_login(token_file, server, open_browser=browser(server, seen=seen))
        assert seen[0].startswith(f"{O.ISSUER}/v2/authorize?")
        assert [url for url, _ in server.requests] == [f"{O.ISSUER}/v2/register", f"{O.ISSUER}/v2/token"]

        # The stored login keeps using them — without reading the metadata again.
        loaded = O.OAuthSession.load(token_file, req=server)
        assert loaded.access_token(force_refresh=True) == "token-2" and session.path == loaded.path
        assert server.requests[-1][0] == f"{O.ISSUER}/v2/token" and server.metadata_reads == 1
        assert loaded.revoke() and server.revoked == ["refresh-2", "token-2"]

    def test_unreadable_metadata_falls_back_to_the_known_endpoints(self, token_file) -> None:
        server = FakeAuthServer()
        server.metadata = None
        do_login(token_file, server)
        assert [url for url, _ in server.requests] == [REGISTER, TOKEN]

    def test_denied_access(self, token_file) -> None:
        server = FakeAuthServer()
        with pytest.raises(LinearError, match="Linear login failed: access_denied"):
            do_login(token_file, server, open_browser=browser(server, answer={"error": "access_denied"}))
        assert server.of(TOKEN) == [] and not token_file.exists()

    def test_answer_for_another_login_attempt_is_refused(self, token_file) -> None:
        server = FakeAuthServer()
        forged = browser(server, answer={"code": "stolen", "state": "not-ours"})
        with pytest.raises(LinearError, match="state mismatch"):
            do_login(token_file, server, open_browser=forged)
        assert server.of(TOKEN) == [], "the code is not exchanged"

    def test_without_a_browser_the_url_is_printed(self, token_file) -> None:
        server = FakeAuthServer()
        printed: list[str] = []
        with pytest.raises(LinearError, match="Timed out"):
            O.login(token_file, port=free_port(), open_browser=None, echo=printed.append, timeout=0.2, req=server)
        assert "https://mcp.linear.app/authorize?client_id=client-1" in printed[0]

    def test_port_in_use(self, token_file) -> None:
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with pytest.raises(LinearError, match="Cannot listen on port .* --port"):
                O.login(token_file, port=busy.getsockname()[1], open_browser=None, echo=lambda m: None,
                        req=FakeAuthServer())

    def test_registration_refused(self, token_file) -> None:
        server = FakeAuthServer()
        server.fail = LinearError("Linear refused the OAuth request (HTTP 400): invalid_client_metadata")
        with pytest.raises(LinearError, match="invalid_client_metadata"):
            do_login(token_file, server)
        assert not token_file.exists()


# ── stored login ──────────────────────────────────────────────────────


class TestSession:
    def logged_in(self, token_file, server, clock) -> O.OAuthSession:
        do_login(token_file, server)
        session = O.OAuthSession.load(token_file, req=server, clock=clock)
        assert session is not None
        return session

    def test_valid_token_is_used_without_asking_linear(self, token_file) -> None:
        server = FakeAuthServer()
        session = self.logged_in(token_file, server, Clock())
        assert session.access_token() == "token-1" and len(server.of(TOKEN)) == 1

    def test_expired_token_is_refreshed_and_the_rotation_saved(self, token_file) -> None:
        server, clock = FakeAuthServer(), Clock()
        session = self.logged_in(token_file, server, clock)
        clock.now += 86100 - 60  # inside the safety margin before expiry
        assert session.access_token() == "token-2"
        assert server.of(TOKEN)[-1] == {
            "grant_type": "refresh_token", "refresh_token": "refresh-1", "client_id": "client-1",
            "resource": "https://mcp.linear.app/mcp",
        }
        stored = json.loads(token_file.read_text())
        assert stored["refresh_token"] == "refresh-2" and stored["client_id"] == "client-1"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

        clock.now += 86100
        assert session.access_token() == "token-3", "the rotated refresh token works"

    def test_token_refreshed_by_another_process_is_adopted(self, token_file) -> None:
        server, clock = FakeAuthServer(), Clock()
        mine = self.logged_in(token_file, server, clock)
        other = O.OAuthSession.load(token_file, req=server, clock=clock)
        clock.now += 86100
        assert other.access_token() == "token-2"
        before = len(server.requests)
        assert mine.access_token() == "token-2", "taken from disk — refresh-1 is spent"
        assert len(server.requests) == before

    def test_expired_login_says_how_to_fix_it(self, token_file) -> None:
        server, clock = FakeAuthServer(), Clock()
        session = self.logged_in(token_file, server, clock)
        server.valid_refresh = None  # the refresh token ran out, or access was revoked in Linear
        clock.now += 90000
        with pytest.raises(LinearError, match="login has expired .*run 'pralph linear oauth' again"):
            session.access_token()

    def test_network_trouble_during_refresh_is_not_an_expired_login(self, token_file) -> None:
        server, clock = FakeAuthServer(), Clock()
        session = self.logged_in(token_file, server, clock)
        server.fail = LinearError("Could not reach Linear: offline", retryable=True)
        clock.now += 90000
        with pytest.raises(LinearError, match="Could not reach Linear") as err:
            session.access_token()
        assert "expired" not in str(err.value)
        assert json.loads(token_file.read_text())["refresh_token"] == "refresh-1", "the login is kept"

    def test_logout_revokes_both_tokens_and_deletes_the_file(self, token_file) -> None:
        server = FakeAuthServer()
        session = self.logged_in(token_file, server, Clock())
        assert session.revoke() is True
        assert server.revoked == ["refresh-1", "token-1"]
        assert server.of(TOKEN)[-1] == {"token": "token-1", "token_type_hint": "access_token", "client_id": "client-1"}
        assert not token_file.exists()

    def test_logout_while_offline_still_forgets_the_login(self, token_file) -> None:
        server = FakeAuthServer()
        session = self.logged_in(token_file, server, Clock())
        server.fail = LinearError("Could not reach Linear: offline", retryable=True)
        assert session.revoke() is False and not token_file.exists()

    def test_missing_or_broken_file_is_no_login(self, token_file) -> None:
        assert O.OAuthSession.load(token_file) is None
        token_file.parent.mkdir(parents=True)
        token_file.write_text("{not json")
        assert O.OAuthSession.load(token_file) is None


# ── client ────────────────────────────────────────────────────────────


class TestClientWithOAuth:
    def test_calls_are_made_with_the_access_token(self, token_file) -> None:
        server = FakeAuthServer()
        session = do_login(token_file, server)
        seen: list[str] = []

        def transport(tool: str, arguments: dict) -> dict:
            seen.append(client.api_key)
            return {"teams": [], "hasNextPage": False}

        client = LinearClient("", transport=transport, oauth=session)
        client.teams()
        assert seen == ["token-1"]

    def test_rejected_token_is_refreshed_once_and_the_call_repeated(self, token_file) -> None:
        server = FakeAuthServer()
        session = do_login(token_file, server)
        tokens: list[str] = []

        def transport(tool: str, arguments: dict) -> dict:
            tokens.append(client.api_key)
            if client.api_key == "token-1":  # revoked early on Linear's side
                raise LinearError("Linear rejected the credentials", auth_failed=True)
            return {"teams": [{"id": "t", "key": "ENG", "name": "Eng"}], "hasNextPage": False}

        client = LinearClient("", transport=transport, oauth=session)
        assert [t["key"] for t in client.teams()] == ["ENG"]
        assert tokens == ["token-1", "token-2"]

    def test_login_that_linear_no_longer_accepts(self, token_file) -> None:
        server = FakeAuthServer()
        session = do_login(token_file, server)

        def transport(tool: str, arguments: dict) -> dict:
            raise LinearError("Linear rejected the credentials", auth_failed=True)

        client = LinearClient("", transport=transport, oauth=session)
        with pytest.raises(LinearError, match="rejected the OAuth login — run 'pralph linear oauth' again"):
            client.teams()

    def test_api_key_takes_precedence_over_the_login(self, token_file, monkeypatch, tmp_path) -> None:
        do_login(token_file, FakeAuthServer())
        home = tmp_path / "home" / ".pralph"
        key, session = O.credentials({}.get, home)
        assert key == "" and session is not None and session.access_token() == "token-1"
        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
        assert O.credentials({}.get, home) == ("lin_api_x", None)


# ── wiring / CLI ──────────────────────────────────────────────────────


class TestCli:
    def project(self, tmp_path) -> Path:
        root = tmp_path / "shop"
        root.mkdir(exist_ok=True)
        return root

    def test_login_status_and_logout(self, tmp_path, token_file, monkeypatch) -> None:
        server = FakeAuthServer()
        port = str(free_port())
        monkeypatch.setattr(O, "request", server)
        monkeypatch.setattr(O.webbrowser, "open", browser(server))
        tokens: list[str] = []

        def whoami(self, tool: str, arguments: dict) -> dict:
            assert tool == "whoami"
            tokens.append(self._token())
            return {"user": {"id": "u1", "name": "Dev", "email": "d@example.com"},
                    "workspace": {"id": "w1", "name": "Acme"}}

        monkeypatch.setattr(L.McpTransport, "__call__", whoami)
        cli = ["--project-dir", str(self.project(tmp_path)), "linear"]

        assert "auth: none — run 'pralph linear oauth'" in CliRunner().invoke(main, [*cli, "status"]).output

        result = CliRunner().invoke(main, [*cli, "oauth", "--port", port])
        assert result.exit_code == 0, result.output
        assert "https://mcp.linear.app/authorize?client_id=client-1" in result.output
        assert "Logged in as Dev (Acme)" in result.output
        assert tokens == ["token-1"], "the login is tried out before it is reported"
        assert json.loads(token_file.read_text())["access_token"] == "token-1"

        status = CliRunner().invoke(main, [*cli, "status"])
        assert "auth: OAuth login (access token valid until " in status.output

        again = CliRunner().invoke(main, [*cli, "oauth", "--port", port])
        assert again.exit_code == 0 and json.loads(token_file.read_text())["access_token"] == "token-2"
        assert len(server.clients) == 1, "the registration is reused"

        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
        assert "auth: API key" in CliRunner().invoke(main, [*cli, "status"]).output
        warned = CliRunner().invoke(main, [*cli, "oauth", "--port", port])
        assert "API key is configured and takes precedence" in warned.output
        monkeypatch.delenv("LINEAR_API_KEY")

        out = CliRunner().invoke(main, [*cli, "oauth", "--logout"])
        assert "Logged out" in out.output and "locally" not in out.output and not token_file.exists()
        assert server.revoked == ["refresh-3", "token-3"]
        assert "Not logged in" in CliRunner().invoke(main, [*cli, "oauth", "--logout"]).output
        assert "auth: none" in CliRunner().invoke(main, [*cli, "status"]).output

    def test_no_browser_prints_the_url_and_waits(self, tmp_path, monkeypatch) -> None:
        server = FakeAuthServer()
        monkeypatch.setattr(O, "request", server)
        monkeypatch.setattr(O.login, "__kwdefaults__", {**O.login.__kwdefaults__, "timeout": 0.2})
        result = CliRunner().invoke(main, ["--project-dir", str(self.project(tmp_path)), "linear", "oauth",
                                           "--no-browser", "--port", str(free_port())])
        assert result.exit_code == 1 and "Timed out waiting for the approval" in result.output
        assert "Open this URL to approve the access" in result.output

    def test_commands_that_need_linear_say_how_to_log_in(self, tmp_path) -> None:
        result = CliRunner().invoke(main, ["--project-dir", str(self.project(tmp_path)), "linear", "pull"])
        assert result.exit_code == 1 and "Not logged in to Linear. Run 'pralph linear oauth'" in result.output

    def test_linked_project_runs_on_the_oauth_login(self, tmp_path, token_file) -> None:
        state = StateManager(str(self.project(tmp_path)))
        linear_sync.LinearLink(team_id="t", team_key="ENG", project_id="p", project_name="Shop").save(
            state.linear_link_path)
        with pytest.raises(LinearError, match="Not logged in to Linear — run 'pralph linear oauth'"):
            linear_sync.attach(state, required=True)

        do_login(token_file, FakeAuthServer())
        sync = linear_sync.attach(state, required=True)
        assert sync is not None and sync.client.oauth is not None and sync.client.api_key == ""
