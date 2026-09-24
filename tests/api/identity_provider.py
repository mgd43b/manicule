"""A fake Google and a fake GitHub, and an installation that signs people in through them.

The sign-in routes reach a provider through an ``httpx`` transport the application holds, so
these suites replace it with :class:`FakeIdentityProvider` — an ``httpx.MockTransport`` that
answers the token, userinfo, user and emails endpoints the way the real ones do. No request
leaves this machine, and every request the routes make is recorded so a test can assert what
was *sent*: the PKCE verifier, the registered ``redirect_uri``, the client's credentials.

Everything else is the production application over the ordinary fake backend, exactly as
``tests/api/support.py`` builds it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi.testclient import TestClient

from manicule.api.oauth import GITHUB, GITHUB_EMAILS, GITHUB_USER, GOOGLE, GOOGLE_USERINFO
from tests.api.support import LOCAL_PEER, app_for, backend_with_a_document

if TYPE_CHECKING:
    from httpx2 import Response

    from tests.app.fakes import FakeBackend

SECRET = "a-signing-key-that-is-long-enough-to-serve"  # noqa: S105 - a fixture key
GOOGLE_CALLBACK = "http://127.0.0.1:8765/auth/callback/google"
GITHUB_CALLBACK = "http://127.0.0.1:8765/auth/callback/github"
ACCESS_TOKEN = "provider-access-token"  # noqa: S105 - what the fake provider issues
CODE = "the-authorization-code"


def google(**overrides: Any) -> dict[str, Any]:
    provider: dict[str, Any] = {
        "type": "google",
        "client_id": "google-client",
        "client_secret": "google-secret",
        "redirect_uri": GOOGLE_CALLBACK,
        "allowed_domains": ["example.org"],
    }
    provider.update(overrides)
    return provider


def github(**overrides: Any) -> dict[str, Any]:
    provider: dict[str, Any] = {
        "type": "github",
        "client_id": "github-client",
        "client_secret": "github-secret",
        "redirect_uri": GITHUB_CALLBACK,
        "allowed_domains": ["example.org"],
    }
    provider.update(overrides)
    return provider


def oauth_backend(
    *providers: dict[str, Any], enforce_https: bool = False, **overrides: Any
) -> FakeBackend:
    """A backend whose installation signs people in through ``providers``.

    ``enforce_https`` is off by default because the test client speaks plain http, and a cookie
    marked ``Secure`` is never sent back over it — which is what the attribute is for, and would
    make every test after the sign-in look like a sign-in that did not work.
    """
    backend, _ = backend_with_a_document(
        security={
            "auth": {
                "mode": "oauth",
                "session_secret": SECRET,
                "providers": list(providers or (google(),)),
            },
            "transport": {"enforce_https": enforce_https},
            "audit": {"enabled": True},
        },
        **overrides,
    )
    return backend


def challenge_of(verifier: str) -> str:
    """RFC 7636's S256, written out so the test does not ask the library it is checking."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass
class FakeIdentityProvider:
    """Both providers' endpoints, answering as the real ones do, and recording what they were sent.

    ``token_status`` and ``token_body`` let a test make the code exchange fail the way a provider
    fails it. ``userinfo``, ``github_user`` and ``github_emails`` are who the provider says signed
    in.
    """

    userinfo: dict[str, Any] = field(
        default_factory=lambda: {
            "sub": "google-account-1",
            "email": "alice@example.org",
            "email_verified": True,
            "name": "Alice",
        }
    )
    github_user: dict[str, Any] = field(
        default_factory=lambda: {"id": 4242, "login": "alice", "name": "Alice"}
    )
    github_emails: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"email": "alice@elsewhere.test", "primary": False, "verified": True},
            {"email": "alice@example.org", "primary": True, "verified": True},
        ]
    )
    token_status: int = 200
    token_body: dict[str, Any] = field(
        default_factory=lambda: {"access_token": ACCESS_TOKEN, "token_type": "bearer"}
    )
    exchanges: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    requests: list[str] = field(default_factory=list[str])

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(f"{request.method} {url}")
        if request.method == "POST" and url in {GOOGLE.token, GITHUB.token}:
            form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
            self.exchanges.append(form)
            return httpx.Response(self.token_status, json=self.token_body)
        if request.headers.get("authorization") != f"Bearer {ACCESS_TOKEN}":
            return httpx.Response(401, json={"message": "Bad credentials"})
        if url == GOOGLE_USERINFO:
            return httpx.Response(200, json=self.userinfo)
        if url == GITHUB_USER:
            return httpx.Response(200, json=self.github_user)
        if url == GITHUB_EMAILS:
            return httpx.Response(200, content=json.dumps(self.github_emails))
        return httpx.Response(404)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


def client_with(
    backend: FakeBackend,
    provider: FakeIdentityProvider,
    *,
    base_url: str = "http://testserver",
    transport: httpx.AsyncBaseTransport | None = None,
) -> TestClient:
    """The production application, reaching ``provider`` instead of the network.

    ``transport`` replaces the provider outright, for a test about a provider that answers
    nothing at all.
    """
    app = app_for(backend)
    app.state.oauth_transport = transport or provider.transport()
    return TestClient(app, base_url=base_url, client=(LOCAL_PEER, 41234))


def start(client: TestClient, provider: str = "google") -> dict[str, str]:
    """Begin a sign-in and return the authorization URL's parameters, one value each."""
    started = client.get(f"/auth/login/{provider}", follow_redirects=False)
    assert started.status_code == 302, started.text
    location = urlsplit(started.headers["location"])
    parameters = {key: values[0] for key, values in parse_qs(location.query).items()}
    parameters["_endpoint"] = f"{location.scheme}://{location.netloc}{location.path}"
    return parameters


def sign_in(client: TestClient, provider: str = "google") -> Response:
    """Begin and complete a sign-in, returning the callback's response."""
    parameters = start(client, provider)
    return client.get(
        f"/auth/callback/{provider}",
        params={"code": CODE, "state": parameters["state"]},
        follow_redirects=False,
    )


def set_cookie(response: Response, name: str) -> str:
    """The one ``Set-Cookie`` header for ``name``, asserting there is exactly one."""
    found = [
        header
        for header in response.headers.get_list("set-cookie")
        if header.startswith(f"{name}=")
    ]
    assert len(found) == 1, response.headers.get_list("set-cookie")
    return found[0]


__all__ = [
    "CODE",
    "GITHUB_CALLBACK",
    "GOOGLE_CALLBACK",
    "SECRET",
    "FakeIdentityProvider",
    "challenge_of",
    "client_with",
    "github",
    "google",
    "oauth_backend",
    "set_cookie",
    "sign_in",
    "start",
]
