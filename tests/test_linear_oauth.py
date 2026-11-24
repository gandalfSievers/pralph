"""Tests for the Linear OAuth login (``pralph linear oauth``).

Linear's side is played by ``FakeTokenEndpoint`` (the form POSTs) and by a
"browser" that calls the loopback redirect the way Linear would.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import stat
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pralph import linear_oauth as O
from pralph import linear_sync
from pralph.cli import main
from pralph.linear import LinearClient, LinearError
from pralph.state import StateManager


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    for var in ("LINEAR_API_KEY", "LINEAR_OAUTH_CLIENT_ID", "LINEAR_OAUTH_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def token_file(tmp_path) -> Path:
    return tmp_path / "home" / ".pralph" / O.TOKEN_FILE


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeTokenEndpoint:
    """Linear's /oauth/token and /oauth/revoke."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.issued = 0
        self.valid_refresh: str | None = None
        self.challenge: str | None = None  # set by the "browser" from the authorize URL
        self.fail: LinearError | None = None

    def __call__(self, url: str, fields: dict) -> dict:
        self.requests.append((url, dict(fields)))
        if self.fail is not None:
            raise self.fail
        if url == O.REVOKE_URL:
            return {}
        assert url == O.TOKEN_URL
        if fields["grant_type"] == "authorization_code":
            assert fields["code"] == "the-code"
            digest = hashlib.sha256(fields["code_verifier"].encode()).digest()
            assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == self.challenge, "PKCE"
        else:
            assert fields["grant_type"] == "refresh_token"
            if fields["refresh_token"] != self.valid_refresh:
                raise LinearError("Linear refused the OAuth request (HTTP 400): invalid_grant")
        self.issued += 1
        self.valid_refresh = f"refresh-{self.issued}"  # refresh tokens rotate
        return {"access_token": f"lin_oauth_{self.issued}", "token_type": "Bearer", "expires_in": 86399,
                "scope": "read write", "refresh_token": self.valid_refresh}


def browser(endpoint: FakeTokenEndpoint, *, answer: dict | None = None, seen: list[str] | None = None):
    """Stand-in for the user's browser: 'approves' and follows Linear's redirect."""

    def open_url(url: str) -> None:
        query = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}
        endpoint.challenge = query["code_challenge"]
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


def do_login(token_file: Path, endpoint: FakeTokenEndpoint, **kwargs) -> O.OAuthSession:
    kwargs.setdefault("open_browser", browser(endpoint))
    return O.login("client-1", token_file, port=free_port(), post=endpoint, echo=lambda m: None,
                   timeout=5, **kwargs)


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ── login ─────────────────────────────────────────────────────────────


class TestLogin:
    def test_authorize_url(self) -> None:
        verifier, challenge = O.pkce_pair()
        assert 43 <= len(verifier) <= 128
        url = O.authorize_url("client-1", redirect=O.redirect_uri(), state="st", challenge=challenge)
        split = urllib.parse.urlsplit(url)
        assert f"{split.scheme}://{split.netloc}{split.path}" == "https://linear.app/oauth/authorize"
        assert {k: v[0] for k, v in urllib.parse.parse_qs(split.query).items()} == {
            "client_id": "client-1", "redirect_uri": "http://localhost:8765/oauth/callback",
            "response_type": "code", "scope": "read,write", "state": "st",
            "code_challenge": challenge, "code_challenge_method": "S256",
        }
        as_app = O.authorize_url("c", redirect="r", state="s", challenge="x", scopes="read comments:create", actor="app")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(as_app).query)
        assert query["actor"] == ["app"] and query["scope"] == ["read,comments:create"]

    def test_browser_login_stores_the_tokens(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        session = do_login(token_file, endpoint)
        (url, fields), = endpoint.requests
        assert url == O.TOKEN_URL and fields["grant_type"] == "authorization_code"
        assert fields["client_id"] == "client-1" and "client_secret" not in fields, "PKCE needs no secret"
        assert fields["redirect_uri"].startswith("http://localhost:") and fields["redirect_uri"].endswith("/oauth/callback")

        assert session.access_token() == "lin_oauth_1"
        stored = json.loads(token_file.read_text())
        assert stored["access_token"] == "lin_oauth_1" and stored["refresh_token"] == "refresh-1"
        assert stored["client_id"] == "client-1" and stored["scope"] == "read write" and stored["actor"] == "user"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600, "readable by the user only"

    def test_client_secret_is_sent_when_configured_but_never_stored(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        do_login(token_file, endpoint, secret="s3cret")
        assert endpoint.requests[0][1]["client_secret"] == "s3cret"
        assert "s3cret" not in token_file.read_text()

    def test_denied_access(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        with pytest.raises(LinearError, match="Linear login failed: access_denied"):
            do_login(token_file, endpoint, open_browser=browser(endpoint, answer={"error": "access_denied"}))
        assert endpoint.requests == [] and not token_file.exists()

    def test_answer_for_another_login_attempt_is_refused(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        forged = browser(endpoint, answer={"code": "stolen", "state": "not-ours"})
        with pytest.raises(LinearError, match="state mismatch"):
            do_login(token_file, endpoint, open_browser=forged)
        assert endpoint.requests == [], "the code is not exchanged"

    def test_times_out_when_nobody_approves(self, token_file) -> None:
        with pytest.raises(LinearError, match="Timed out"):
            O.login("c", token_file, port=free_port(), open_browser=None, echo=lambda m: None, timeout=0.2,
                    post=FakeTokenEndpoint())

    def test_port_in_use(self, token_file) -> None:
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with pytest.raises(LinearError, match="Cannot listen on port .* --port"):
                O.login("c", token_file, port=busy.getsockname()[1], open_browser=None, echo=lambda m: None,
                        post=FakeTokenEndpoint())


# ── stored login ──────────────────────────────────────────────────────


class TestSession:
    def logged_in(self, token_file, endpoint, clock) -> O.OAuthSession:
        do_login(token_file, endpoint)
        session = O.OAuthSession.load(token_file, post=endpoint, clock=clock)
        assert session is not None
        return session

    def test_valid_token_is_used_without_asking_linear(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        import time
        session = self.logged_in(token_file, endpoint, Clock(time.time()))
        assert session.access_token() == "lin_oauth_1" and len(endpoint.requests) == 1

    def test_expired_token_is_refreshed_and_the_rotation_saved(self, token_file) -> None:
        import time
        endpoint, clock = FakeTokenEndpoint(), Clock(time.time())
        session = self.logged_in(token_file, endpoint, clock)
        clock.now += 86399 - 60  # inside the safety margin before expiry
        assert session.access_token() == "lin_oauth_2"
        _, fields = endpoint.requests[-1]
        assert fields == {"grant_type": "refresh_token", "refresh_token": "refresh-1", "client_id": "client-1"}
        stored = json.loads(token_file.read_text())
        assert stored["refresh_token"] == "refresh-2" and stored["client_id"] == "client-1"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

        clock.now += 86399
        assert session.access_token() == "lin_oauth_3", "the rotated refresh token works"

    def test_token_refreshed_by_another_process_is_adopted(self, token_file) -> None:
        import time
        endpoint, clock = FakeTokenEndpoint(), Clock(time.time())
        mine = self.logged_in(token_file, endpoint, clock)
        other = O.OAuthSession.load(token_file, post=endpoint, clock=clock)
        clock.now += 86399
        assert other.access_token() == "lin_oauth_2"
        before = len(endpoint.requests)
        assert mine.access_token() == "lin_oauth_2", "taken from disk — refresh-1 is spent"
        assert len(endpoint.requests) == before

    def test_expired_login_says_how_to_fix_it(self, token_file) -> None:
        import time
        endpoint, clock = FakeTokenEndpoint(), Clock(time.time())
        session = self.logged_in(token_file, endpoint, clock)
        endpoint.valid_refresh = None  # 30 days passed, or access was revoked in Linear
        clock.now += 90000
        with pytest.raises(LinearError, match="login has expired .*run 'pralph linear oauth' again"):
            session.access_token()

    def test_network_trouble_during_refresh_is_not_an_expired_login(self, token_file) -> None:
        import time
        endpoint, clock = FakeTokenEndpoint(), Clock(time.time())
        session = self.logged_in(token_file, endpoint, clock)
        endpoint.fail = LinearError("Could not reach Linear: offline", retryable=True)
        clock.now += 90000
        with pytest.raises(LinearError, match="Could not reach Linear") as err:
            session.access_token()
        assert "expired" not in str(err.value)
        assert json.loads(token_file.read_text())["refresh_token"] == "refresh-1", "the login is kept"

    def test_logout_revokes_and_deletes(self, token_file) -> None:
        import time
        endpoint = FakeTokenEndpoint()
        session = self.logged_in(token_file, endpoint, Clock(time.time()))
        assert session.revoke() is True
        assert endpoint.requests[-1] == (O.REVOKE_URL, {"token": "refresh-1"})
        assert not token_file.exists()

    def test_missing_or_broken_file_is_no_login(self, token_file) -> None:
        assert O.OAuthSession.load(token_file) is None
        token_file.parent.mkdir(parents=True)
        token_file.write_text("{not json")
        assert O.OAuthSession.load(token_file) is None


# ── client ────────────────────────────────────────────────────────────


class TestClientWithOAuth:
    def test_requests_carry_a_bearer_token(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        session = do_login(token_file, endpoint)
        seen: list[str] = []

        def transport(query: str, variables: dict) -> dict:
            seen.append(client._auth_header())
            return {"data": {"teams": {"nodes": []}}}

        client = LinearClient("", transport=transport, oauth=session)
        client.teams()
        assert seen == ["Bearer lin_oauth_1"]

    def test_rejected_token_is_refreshed_once_and_the_request_repeated(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        session = do_login(token_file, endpoint)
        session._post = endpoint
        tokens: list[str] = []

        def transport(query: str, variables: dict) -> dict:
            tokens.append(client.api_key)
            if client.api_key == "lin_oauth_1":  # revoked early on Linear's side
                return {"errors": [{"message": "nope", "extensions": {"code": "AUTHENTICATION_ERROR"}}]}
            return {"data": {"teams": {"nodes": [{"id": "t", "key": "ENG", "name": "Eng"}]}}}

        client = LinearClient("", transport=transport, oauth=session)
        assert [t["key"] for t in client.teams()] == ["ENG"]
        assert tokens == ["lin_oauth_1", "lin_oauth_2"]

    def test_login_that_linear_no_longer_accepts(self, token_file) -> None:
        endpoint = FakeTokenEndpoint()
        session = do_login(token_file, endpoint)
        session._post = endpoint
        rejected = {"errors": [{"message": "nope", "extensions": {"code": "AUTHENTICATION_ERROR"}}]}
        client = LinearClient("", transport=lambda q, v: rejected, oauth=session)
        with pytest.raises(LinearError, match="rejected the OAuth login — run 'pralph linear oauth' again"):
            client.teams()

    def test_api_key_takes_precedence_over_the_login(self, token_file, monkeypatch, tmp_path) -> None:
        do_login(token_file, FakeTokenEndpoint())
        home = tmp_path / "home" / ".pralph"
        key, session = O.credentials({}.get, home)
        assert key == "" and session is not None and session.access_token() == "lin_oauth_1"
        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
        assert O.credentials({}.get, home) == ("lin_api_x", None)


# ── wiring / CLI ──────────────────────────────────────────────────────


class TestCli:
    def project(self, tmp_path) -> Path:
        root = tmp_path / "shop"
        root.mkdir(exist_ok=True)
        return root

    def test_without_a_client_id_it_explains_the_setup(self, tmp_path) -> None:
        result = CliRunner().invoke(main, ["--project-dir", str(self.project(tmp_path)), "linear", "oauth"])
        assert result.exit_code == 1
        assert "https://linear.app/settings/api/applications/new" in result.output
        assert "callback URL http://localhost:8765/oauth/callback" in result.output
        assert "--client-id" in result.output and "LINEAR_OAUTH_CLIENT_ID" in result.output

    def test_login_status_and_logout(self, tmp_path, token_file, monkeypatch) -> None:
        endpoint = FakeTokenEndpoint()
        port = str(free_port())
        monkeypatch.setattr(O, "post_form", endpoint)
        monkeypatch.setattr(O.login, "__kwdefaults__", {**O.login.__kwdefaults__, "post": endpoint})
        monkeypatch.setattr(O.OAuthSession.__init__, "__kwdefaults__",
                            {**O.OAuthSession.__init__.__kwdefaults__, "post": endpoint})
        monkeypatch.setattr(O.webbrowser, "open", browser(endpoint))
        viewer = {"data": {"viewer": {"id": "u1", "name": "Dev", "email": "d@example.com"},
                           "organization": {"name": "Acme", "urlKey": "acme"}}}
        monkeypatch.setattr(LinearClient, "_http_transport", lambda self, q, v: viewer)
        cli = ["--project-dir", str(self.project(tmp_path)), "linear"]

        result = CliRunner().invoke(main, [*cli, "oauth", "--client-id", "client-1", "--port", port])
        assert result.exit_code == 0, result.output
        assert f"callback URL: http://localhost:{port}/oauth/callback" in result.output
        assert "Logged in as Dev (Acme)" in result.output and "scopes: read write" in result.output
        assert json.loads(token_file.read_text())["access_token"] == "lin_oauth_1"

        status = CliRunner().invoke(main, [*cli, "status"])
        assert "auth: OAuth login (user actor, access token valid until " in status.output

        # The client ID is remembered: logging in again needs no option.
        again = CliRunner().invoke(main, [*cli, "oauth", "--port", port])
        assert again.exit_code == 0 and json.loads(token_file.read_text())["access_token"] == "lin_oauth_2"

        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
        assert "auth: API key" in CliRunner().invoke(main, [*cli, "status"]).output
        warned = CliRunner().invoke(main, [*cli, "oauth", "--port", port])
        assert "API key is configured and takes precedence" in warned.output
        monkeypatch.delenv("LINEAR_API_KEY")

        out = CliRunner().invoke(main, [*cli, "oauth", "--logout"])
        assert "Logged out" in out.output and not token_file.exists()
        assert endpoint.requests[-1][0] == O.REVOKE_URL
        assert "Not logged in" in CliRunner().invoke(main, [*cli, "oauth", "--logout"]).output
        assert "auth: none" in CliRunner().invoke(main, [*cli, "status"]).output

    def test_linked_project_runs_on_the_oauth_login(self, tmp_path, token_file) -> None:
        state = StateManager(str(self.project(tmp_path)))
        linear_sync.LinearLink(team_id="t", team_key="ENG", project_id="p", project_name="Shop").save(
            state.linear_link_path)
        with pytest.raises(LinearError, match="LINEAR_API_KEY is not set and there is no OAuth login"):
            linear_sync.attach(state, required=True)

        do_login(token_file, FakeTokenEndpoint())
        sync = linear_sync.attach(state, required=True)
        assert sync is not None and sync.client.oauth is not None and sync.client.api_key == ""
