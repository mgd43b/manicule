"""Signing in through a provider, end to end, and what the session cookie is and is not.

Driven through the production application, with the identity provider played by
:class:`~tests.api.identity_provider.FakeIdentityProvider` — a transport that answers as Google
and GitHub answer and records what it was sent. Three groups:

* **The protocol.** The login redirect carries a fresh ``state``, an S256 challenge and the
  registered ``redirect_uri``; the callback refuses anything that did not come back from the
  browser that left, and the code exchange sends the verifier whose challenge went out.
* **The cookie.** ``HttpOnly``, ``SameSite=Strict``, ``Secure`` when HTTPS is enforced; it
  authenticates the API and the pages and **not** the MCP mount; a forged or expired one is
  refused before anything is looked up; a sign-out revokes it on the server.
* **What cannot spend it.** A cross-site ``POST`` carrying the cookie is refused, sign-out
  included.

Every refusal is asserted against a control that succeeds, because a surface that refused every
sign-in would pass every negative assertion here.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
import pytest
from itsdangerous import TimestampSigner

from manicule.api.cookies import SESSION_COOKIE, SIGNIN_COOKIE
from manicule.web.rendering import UI_POLICY
from tests.api.identity_provider import (
    CODE,
    GITHUB_CALLBACK,
    GOOGLE_CALLBACK,
    FakeIdentityProvider,
    challenge_of,
    client_with,
    github,
    google,
    oauth_backend,
    set_cookie,
    sign_in,
    start,
)
from tests.api.support import envelope

OK = 200
BAD_REQUEST = 400
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404
SEE_OTHER = 303

MCP_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _signed_ago(seconds: int) -> Callable[[TimestampSigner], int]:
    """A replacement clock for the signer: every signature made under it is ``seconds`` old."""

    def timestamp(signer: TimestampSigner) -> int:
        del signer
        return int(time.time()) - seconds

    return timestamp


# --- the protocol ---------------------------------------------------------------------------------


def test_the_login_redirect_carries_state_an_s256_challenge_and_the_registered_callback() -> None:
    """Everything the provider needs, and the challenge for a verifier that never leaves here."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        parameters = start(client)
        again = start(client)

    assert parameters["_endpoint"] == "https://accounts.google.com/o/oauth2/v2/auth"
    assert parameters["response_type"] == "code"
    assert parameters["client_id"] == "google-client"
    assert parameters["redirect_uri"] == GOOGLE_CALLBACK
    assert parameters["scope"] == "openid email profile"
    assert parameters["code_challenge_method"] == "S256"
    assert len(parameters["code_challenge"]) == 43
    assert len(parameters["state"]) >= 32
    assert again["state"] != parameters["state"], "state is not fresh per sign-in"
    assert "google-secret" not in str(parameters), "the client secret went to the browser"


def test_a_sign_in_exchanges_the_code_with_the_verifier_whose_challenge_went_out() -> None:
    """PKCE checked end to end: the challenge on the URL is the S256 of what the exchange sent."""
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        parameters = start(client)
        landed = client.get(
            "/auth/callback/google",
            params={"code": CODE, "state": parameters["state"]},
            follow_redirects=False,
        )

    assert landed.status_code == OK, landed.text
    (exchange,) = provider.exchanges
    assert exchange["grant_type"] == "authorization_code"
    assert exchange["code"] == CODE
    assert exchange["redirect_uri"] == GOOGLE_CALLBACK
    assert exchange["client_id"] == "google-client"
    assert exchange["client_secret"] == "google-secret"  # noqa: S105 - the fixture's own
    assert challenge_of(exchange["code_verifier"]) == parameters["code_challenge"]


def test_a_callback_with_no_sign_in_cookie_is_refused() -> None:
    """A browser that never started a sign-in here has nothing to finish."""
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        refused = client.get("/auth/callback/google", params={"code": CODE, "state": "guessed"})
    assert refused.status_code == BAD_REQUEST
    assert "text/html" in refused.headers["content-type"]
    assert provider.exchanges == [], "a code was exchanged for a callback nobody started"


def test_a_callback_whose_state_does_not_match_is_refused_before_any_exchange() -> None:
    """The login CSRF this defends against: somebody else's code, delivered to this browser."""
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        parameters = start(client)
        refused = client.get(
            "/auth/callback/google", params={"code": CODE, "state": parameters["state"] + "x"}
        )
        assert refused.status_code == BAD_REQUEST
        assert provider.exchanges == []
        # And the transaction is spent: the browser was told to forget it.
        assert SIGNIN_COOKIE not in client.cookies


def test_a_sign_in_cookie_older_than_ten_minutes_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signed with a timestamp, so an old transaction is refused without being stored anywhere."""
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        with monkeypatch.context() as patch:
            patch.setattr(TimestampSigner, "get_timestamp", _signed_ago(601))
            parameters = start(client)
        refused = client.get(
            "/auth/callback/google", params={"code": CODE, "state": parameters["state"]}
        )
    assert refused.status_code == BAD_REQUEST
    assert provider.exchanges == []


def test_a_forged_sign_in_cookie_is_refused() -> None:
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        parameters = start(client)
        forged = client.cookies[SIGNIN_COOKIE][:-2] + "AA"
        client.cookies.set(SIGNIN_COOKIE, forged, path="/auth/callback")
        refused = client.get(
            "/auth/callback/google", params={"code": CODE, "state": parameters["state"]}
        )
    assert refused.status_code == BAD_REQUEST
    assert provider.exchanges == []


def test_a_provider_that_did_not_authorize_is_refused_without_an_exchange() -> None:
    provider = FakeIdentityProvider()
    with client_with(oauth_backend(), provider) as client:
        parameters = start(client)
        refused = client.get(
            "/auth/callback/google",
            params={"error": "access_denied", "state": parameters["state"]},
        )
    assert refused.status_code == FORBIDDEN
    assert "access_denied" not in refused.text
    assert provider.exchanges == []


def test_a_failed_code_exchange_is_refused_and_the_page_carries_no_code() -> None:
    """The provider said no; the person is told so, and nothing that crossed the wire is shown."""
    provider = FakeIdentityProvider(
        token_status=400,
        token_body={
            "error": "invalid_grant",
            "error_description": "Bad code the-authorization-code",
        },
    )
    with client_with(oauth_backend(), provider) as client:
        refused = sign_in(client)
        assert SESSION_COOKIE not in client.cookies
    assert refused.status_code == BAD_REQUEST
    assert CODE not in refused.text
    assert "Bad code" not in refused.text


def test_a_provider_that_cannot_be_reached_is_a_refusal_rather_than_an_error() -> None:
    """A dropped connection is a page for the person, naming the step, not a 500."""

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with client_with(
        oauth_backend(), FakeIdentityProvider(), transport=httpx.MockTransport(unreachable)
    ) as client:
        refused = sign_in(client)
    assert refused.status_code == BAD_REQUEST
    assert "exchanging the authorization code" in refused.text
    assert "ConnectError" in refused.text


def test_an_unknown_provider_is_a_page_that_says_so() -> None:
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        refused = client.get("/auth/login/okta", follow_redirects=False)
    assert refused.status_code == NOT_FOUND
    assert "text/html" in refused.headers["content-type"]


def test_a_provider_configured_for_another_workspace_is_not_offered_here() -> None:
    """The same 404 as a provider nobody configured, and absent from the provider list."""
    backend = oauth_backend(google(), github(workspace="elsewhere"))
    with client_with(backend, FakeIdentityProvider()) as client:
        refused = client.get("/auth/login/github", follow_redirects=False)
        listed = envelope(client.get("/auth/providers"))["data"]
    assert refused.status_code == NOT_FOUND
    assert listed["providers"] == ["google"]
    assert listed["login_paths"] == ["/auth/login/google"]


def test_a_github_account_without_a_verified_primary_address_is_not_verified() -> None:
    """GitHub marks each address verified or not; only the primary one is the person's."""
    provider = FakeIdentityProvider(
        github_emails=[
            {"email": "alice@example.org", "primary": True, "verified": False},
            {"email": "alice@example.org", "primary": False, "verified": True},
        ]
    )
    backend = oauth_backend(github())
    with client_with(backend, provider) as client:
        refused = sign_in(client, "github")
    assert refused.status_code == FORBIDDEN
    assert "verified" in refused.text
    assert backend.users_.people == {}, "somebody was recorded whose address was not verified"


def test_a_github_sign_in_is_identified_by_the_numeric_account_id() -> None:
    """``login`` can be renamed and reclaimed by somebody else; the numeric id cannot."""
    provider = FakeIdentityProvider()
    backend = oauth_backend(github())
    with client_with(backend, provider) as client:
        landed = sign_in(client, "github")
    assert landed.status_code == OK, landed.text
    (person,) = backend.users_.people.values()
    assert person["subject"] == "4242"
    assert person["email"] == "alice@example.org"
    (exchange,) = provider.exchanges
    assert exchange["redirect_uri"] == GITHUB_CALLBACK


def test_a_person_the_workspace_does_not_admit_is_refused_as_a_page() -> None:
    provider = FakeIdentityProvider(
        userinfo={"sub": "g-9", "email": "mallory@elsewhere.test", "email_verified": True}
    )
    with client_with(oauth_backend(), provider) as client:
        refused = sign_in(client)
        assert SESSION_COOKIE not in client.cookies
    assert refused.status_code == FORBIDDEN
    assert "not admitted" in refused.text
    assert "/ui/login" in refused.text


# --- the session cookie -------------------------------------------------------------------------


def test_a_sign_in_lands_on_a_page_that_moves_on_without_script() -> None:
    """A page, not a redirect: a Strict cookie is withheld from a redirect that ends a cross-site
    chain, so the person would arrive signed out. And no script, because the policy forbids it."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        landed = sign_in(client)
    assert landed.status_code == OK
    assert landed.headers["content-security-policy"] == UI_POLICY
    assert '<meta http-equiv="refresh" content="0;url=/">' in landed.text
    assert "<script" not in landed.text
    assert "Alice" in landed.text


def test_the_session_cookie_is_http_only_strict_and_spans_the_site() -> None:
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        landed = sign_in(client)
    cookie = set_cookie(landed, SESSION_COOKIE).lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/;" in cookie or cookie.endswith("path=/")
    assert "secure" not in cookie, "Secure was set with HTTPS not enforced"
    signin = set_cookie(landed, SIGNIN_COOKIE).lower()
    assert "max-age=0" in signin or "expires=" in signin, "the sign-in cookie was not cleared"


def test_both_cookies_are_secure_when_https_is_enforced() -> None:
    """Driven over https, because a Secure cookie is not sent back over anything else."""
    backend = oauth_backend(
        google(redirect_uri="https://manicule.example.org/auth/callback/google"),
        enforce_https=True,
    )
    with client_with(
        backend, FakeIdentityProvider(), base_url="https://manicule.example.org"
    ) as client:
        started = client.get("/auth/login/google", follow_redirects=False)
        landed = sign_in(client)
    signin = set_cookie(started, SIGNIN_COOKIE).lower()
    assert "secure" in signin
    assert "samesite=lax" in signin, "Strict would be withheld from the provider's return"
    assert "path=/auth/callback" in signin
    assert "secure" in set_cookie(landed, SESSION_COOKIE).lower()


def test_the_cookie_authenticates_the_api_and_the_pages() -> None:
    backend = oauth_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        assert client.get("/api/v1/documents").status_code == UNAUTHORIZED, "control failed"
        sign_in(client)
        assert client.get("/api/v1/documents").status_code == OK
        assert client.get("/ui/documents").status_code == OK
        identity = envelope(client.get("/auth/session"))["data"]
    assert identity["authenticated"] is True
    assert identity["via"] == "session"
    assert identity["user_email"] == "alice@example.org"
    assert identity["role"] == "member"


def test_the_cookie_does_not_authenticate_the_mcp_mount() -> None:
    """MCP is stateless by design: a client presents its key on every call.

    An assistant running in a browser tab must not inherit whoever is signed in there. The
    control is the same client reading the API with the same cookie in the same breath.
    """
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        assert client.get("/api/v1/documents").status_code == OK
        refused = client.post("/mcp/", json=MCP_LIST, headers=MCP_HEADERS)
    assert refused.status_code == UNAUTHORIZED
    assert envelope(refused)["op"] == "mcp"


def test_the_cookie_opens_the_chat_websocket_from_this_origin_and_no_other() -> None:
    """The handshake resolves the same way the routes do, after its own origin check.

    A browser applies no cross-origin policy to a websocket, so the origin check before the
    credential is looked at is what stops another site's page spending the cookie there.
    """
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415

    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/chat/ws"):
            pass  # the control: no cookie yet, so the handshake is refused
        sign_in(client)
        with client.websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_text('{"question": "does the client retry"}')
            assert socket.receive_json()["event"]
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/api/v1/chat/ws", headers={"Origin": "https://attacker.example.com"}
            ),
        ):
            pass


def test_a_header_credential_wins_over_the_cookie() -> None:
    """A program's key is never overridden by a browser's ambient cookie — even a bad key."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        refused = client.get("/api/v1/documents", headers={"X-API-Key": "mnk_not-a-key"})
    assert refused.status_code == UNAUTHORIZED


def test_a_tampered_cookie_is_refused_before_anything_is_looked_up() -> None:
    """The signature is checked first, so a forgery costs a hash rather than a query."""
    backend = oauth_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        assert client.get("/api/v1/documents").status_code == OK
        looked_up = backend.users_.resolves
        genuine = client.cookies[SESSION_COOKIE]
        forged = genuine[:-3] + ("AAA" if genuine[-3:] != "AAA" else "BBB")
        client.cookies.set(SESSION_COOKIE, forged)
        refused = client.get("/api/v1/documents")
    assert refused.status_code == UNAUTHORIZED
    assert backend.users_.resolves == looked_up, "a forged cookie reached the session store"


def test_a_cookie_whose_signature_has_outlived_the_session_is_refused_without_a_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signed long enough ago that the signature has expired, while the row has not.

    The session's own expiry is the store's, and is in the future here; what refuses is the
    signature's age, and it refuses before the store is asked.
    """
    backend = oauth_backend()
    max_age = backend.settings.security.auth.session_max_age_s
    with client_with(backend, FakeIdentityProvider()) as client:
        with monkeypatch.context() as patch:
            patch.setattr(TimestampSigner, "get_timestamp", _signed_ago(max_age + 5))
            landed = sign_in(client)
        assert landed.status_code == OK
        looked_up = backend.users_.resolves
        refused = client.get("/api/v1/documents")
    assert refused.status_code == UNAUTHORIZED
    assert backend.users_.resolves == looked_up


def test_signing_out_revokes_the_session_so_a_replayed_cookie_fails() -> None:
    """Revoked on the server, not only forgotten by the browser."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        kept = client.cookies[SESSION_COOKIE]
        out = client.post("/auth/logout", headers={"Accept": "application/json"})
        assert out.status_code == OK
        assert envelope(out)["data"] == {"ended": True}
        assert SESSION_COOKIE not in client.cookies, "the browser was not told to forget it"
        client.cookies.set(SESSION_COOKIE, kept)
        assert client.get("/api/v1/documents").status_code == UNAUTHORIZED


def test_signing_out_from_the_page_goes_on_to_the_sign_in_page() -> None:
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        out = client.post("/auth/logout", headers={"Accept": "text/html"}, follow_redirects=False)
    assert out.status_code == SEE_OTHER
    assert out.headers["location"] == "/ui/login"


def test_signing_out_with_no_session_succeeds_and_ends_nothing() -> None:
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        out = client.post("/auth/logout")
    assert out.status_code == OK
    assert envelope(out)["data"] == {"ended": False}


# --- what cannot spend the cookie ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/auth/logout", None),
        ("POST", "/api/v1/collections", {"name": "planted"}),
        ("PATCH", "/api/v1/auth/users/anybody", {"role": "admin"}),
        ("DELETE", "/api/v1/auth/keys/anything", None),
    ],
)
def test_a_cross_site_request_carrying_the_cookie_is_refused(
    method: str, path: str, body: dict[str, str] | None
) -> None:
    """``SameSite=Strict`` keeps the browser from attaching it; this is the half that holds even
    if a browser did. Sign-out included, so another site cannot sign a person out either."""
    backend = oauth_backend()
    with client_with(backend, FakeIdentityProvider()) as client:
        sign_in(client)
        refused = client.request(
            method,
            path,
            json=body,
            headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://attacker.example.com"},
        )
        still = client.get("/api/v1/documents")
    assert refused.status_code == FORBIDDEN
    assert envelope(refused)["error"]["type"] == "PolicyError"
    assert still.status_code == OK, "the cross-site request ended the session anyway"


def test_the_same_request_from_this_origin_is_admitted() -> None:
    """The control: the refusal above is about where the request came from, not the cookie."""
    with client_with(oauth_backend(), FakeIdentityProvider()) as client:
        sign_in(client)
        made = client.post(
            "/auth/logout",
            headers={"Sec-Fetch-Site": "same-origin", "Accept": "application/json"},
        )
    assert made.status_code == OK
    assert envelope(made)["data"] == {"ended": True}
