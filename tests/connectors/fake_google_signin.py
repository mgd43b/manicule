"""Google's side of the installed-app sign-in, added to the token-endpoint fake.

:class:`~tests.connectors.fake_google_oauth.FakeGoogleOAuth` answers the refresh exchange and
``users.getProfile``. This adds the two things ``manicule connector login`` talks to on the way
to a refresh token: the **authorization endpoint**, which a person's browser visits (``consent``
plays that browser's part), and the **authorization-code grant** at the token endpoint, which
enforces PKCE the way Google does — the verifier must hash to the challenge the code was issued
against, and a code is redeemable once, by the client and redirect it was issued for.

The redirect that ``consent`` returns is the real thing: ``visit`` sends it to the loopback
socket the command opened, so the whole flow runs over a real connection to ``127.0.0.1``. Only
Google's own two hosts are replaced (by the mock transport).

Addresses are at ``example.org``.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import override
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from manicule.connectors.gmail_login import GMAIL_AUTH_URL, GMAIL_READONLY_SCOPE
from tests.connectors.fake_google_oauth import REFRESH_TOKEN, FakeGoogleOAuth

__all__ = ["FakeGoogleSignIn", "visit"]


@dataclass
class _Issued:
    client_id: str
    redirect_uri: str
    challenge: str
    redeemed: bool = False


class FakeGoogleSignIn(FakeGoogleOAuth):
    """The authorization endpoint and the code grant, on top of the refresh-token fake."""

    def __init__(self, *, address: str = "mail@example.org") -> None:
        super().__init__(address=address)
        self.authorizations: list[dict[str, str]] = []
        self.exchanges: list[dict[str, str]] = []
        self._codes: dict[str, _Issued] = {}
        self.granted_scope = GMAIL_READONLY_SCOPE
        self.issues_refresh_token = True

    def consent(
        self,
        auth_url: str,
        *,
        approve: bool = True,
        state: str | None = None,
    ) -> str:
        """What the browser is redirected to after the person has chosen, as an address.

        Validates the request as Google would for this client: the address, response type, the
        read-only scope, offline access with forced consent, an ``S256`` challenge and a loopback
        redirect. ``state`` replaces the one the request carried, to play a forged redirect.
        """
        parts = urlsplit(auth_url)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == GMAIL_AUTH_URL
        asked = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.authorizations.append(asked)
        assert asked["response_type"] == "code"
        assert asked["scope"] == GMAIL_READONLY_SCOPE
        assert asked["access_type"] == "offline"
        assert asked["prompt"] == "consent"
        assert asked["code_challenge_method"] == "S256"
        redirect = urlsplit(asked["redirect_uri"])
        assert redirect.scheme == "http"
        assert redirect.hostname == "127.0.0.1"
        assert redirect.port is not None
        sent_state = asked["state"] if state is None else state
        if not approve:
            return f"{asked['redirect_uri']}/?" + urlencode(
                {"error": "access_denied", "state": sent_state}
            )
        code = secrets.token_urlsafe(24)
        self._codes[code] = _Issued(
            client_id=asked["client_id"],
            redirect_uri=asked["redirect_uri"],
            challenge=asked["code_challenge"],
        )
        return f"{asked['redirect_uri']}/?" + urlencode({"code": code, "state": sent_state})

    @override
    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if form.get("grant_type") != "authorization_code":
            return super()._token(request)
        self.exchanges.append(form)
        issued = self._codes.get(form.get("code", ""))
        digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if (
            issued is None
            or issued.redeemed
            or issued.client_id != form.get("client_id")
            or issued.redirect_uri != form.get("redirect_uri")
            or issued.challenge != challenge
        ):
            return httpx.Response(400, json={"error": "invalid_grant"})
        issued.redeemed = True
        body: dict[str, object] = {
            "access_token": "access-token-from-the-code-grant",
            "token_type": "Bearer",
            "expires_in": 3599,
            "scope": self.granted_scope,
        }
        if self.issues_refresh_token:
            body["refresh_token"] = REFRESH_TOKEN
        return httpx.Response(200, json=body)


def visit(address: str) -> int:
    """GET an address as a browser would, and return the status the loopback socket answered."""
    try:
        with urllib.request.urlopen(address, timeout=10) as response:  # noqa: S310
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as error:
        with error:  # an error response holds the connection open until it is closed
            return int(error.code)
