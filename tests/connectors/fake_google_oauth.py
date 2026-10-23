"""A minimal stand-in for Google's token endpoint and ``users.getProfile``.

Only what the credential needs: exchange a refresh token for an access token, and answer the one
profile read the mailbox pin makes. The synthetic Gmail server (#459) is the fuller fake, with
the whole mailbox behind it; this one exists so the credential is tested without depending on
that one, and it speaks the same wire format so the two can be swapped.

Addresses are at ``example.org``.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import parse_qs

import httpx

from manicule.connectors.credentials import GMAIL_PROFILE_URL, GOOGLE_TOKEN_URL

__all__ = ["FakeGoogleOAuth"]

CLIENT_SECRET = "client-secret-sentinel-4821"  # noqa: S105 - a fixture value
REFRESH_TOKEN = "refresh-token-sentinel-7730"  # noqa: S105 - a fixture value
"""Values no assertion about leakage can mistake for ordinary text."""


class FakeGoogleOAuth:
    """Serves the two endpoints, records every request, and lets a test break either."""

    def __init__(
        self,
        *,
        address: str = "mail@example.org",
        expires_in: object = 3599,
        issue: Callable[[], str] | None = None,
    ) -> None:
        self.address = address
        self.expires_in = expires_in
        self.requests: list[httpx.Request] = []
        self.refreshes = 0
        self.token_failure: tuple[int, dict[str, object]] | None = None
        self.profile_status = 200
        self.profile_error: dict[str, object] = {"error": {"code": 403}}
        self._valid: set[str] = set()
        self.minted: list[str] = []
        self._issue = issue
        """Where tokens come from when another fake (the synthetic Gmail) accepts them."""

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def revoke_access_tokens(self) -> None:
        """Every access token minted so far stops working, as an early revocation does."""
        self._valid.clear()

    def fail_refresh(self, status: int, error: str) -> None:
        self.token_failure = (status, {"error": error, "error_description": "sentinel-prose"})

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url).split("?")[0]
        if url == GOOGLE_TOKEN_URL:
            return self._token(request)
        if url == GMAIL_PROFILE_URL:
            return self._profile(request)
        return httpx.Response(404, json={"error": "not_found"})

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if self.token_failure is not None:
            status, body = self.token_failure
            return httpx.Response(status, json=body)
        if (
            form.get("grant_type") != "refresh_token"
            or form.get("refresh_token") != REFRESH_TOKEN
            or form.get("client_secret") != CLIENT_SECRET
        ):
            return httpx.Response(400, json={"error": "invalid_grant"})
        self.refreshes += 1
        token = self._issue() if self._issue else f"access-token-sentinel-{self.refreshes}"
        self._valid.add(token)
        self.minted.append(token)
        body: dict[str, object] = {"access_token": token, "token_type": "Bearer"}
        if self.expires_in is not None:
            body["expires_in"] = self.expires_in
        return httpx.Response(200, json=body)

    def _profile(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if bearer not in self._valid:
            return httpx.Response(401, json={"error": {"code": 401}})
        if self.profile_status != 200:
            return httpx.Response(self.profile_status, json=self.profile_error)
        return httpx.Response(200, json={"emailAddress": self.address, "historyId": "1"})
