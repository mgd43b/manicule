"""Minting an Outlook refresh token: Entra's authorization code flow with PKCE, and where it goes.

``manicule connector login <name>`` for an Outlook instance has one job: hand a person a refresh
token for the mailbox the instance indexes, so they can put it where a headless server reads it
(``docs/connectors/outlook.md`` §1.2, §1.3). The flow is the one
:mod:`manicule.connectors.gmail_login` implements for Google, with Microsoft's parameters, and it
reuses that module's loopback listener and token sinks so there is one implementation of the parts
that must not differ: a listener bound to loopback only, ``state`` checked in constant time, PKCE
``S256``, a token that reaches only a ``0600`` file or a pipe, and a destination checked before the
browser opens.

What is Microsoft's:

- **A public client.** No client secret is sent or exists. The app registration must allow public
  client flows and register ``http://localhost`` as a "Mobile and desktop applications" redirect.
- **The scopes are the instance's and nothing else**: ``offline_access`` (without it no refresh
  token is issued), ``User.Read`` (the mailbox check), ``Mail.Read`` and ``Calendars.Read``
  as the instance indexes them. A token that reports any other scope than those and the OpenID
  ones is discarded: it is a larger secret than the connector needs, and no write scope is ever
  kept.
- **The token endpoint and the authorization endpoint are built from the tenant GUID** and a
  constant host, as the credential's is.

Nothing here imports an HTTP client at module level.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import unquote, urlencode

from pydantic import SecretStr

from manicule.connectors.errors import (
    ConnectorError,
    CredentialError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
)
from manicule.connectors.gmail_login import (
    DEFAULT_SIGN_IN_TIMEOUT_SECONDS,
    FileSink,
    Loopback,
    StreamSink,
    TokenSink,
    ignore_message,
    open_browser_quietly,
    pkce_challenge,
)
from manicule.connectors.outlook_auth import GRAPH_SCOPE_PREFIX, authorize_url, token_url
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from manicule.connectors.outlook_config import OutlookConfig

__all__ = [
    "DEFAULT_SIGN_IN_TIMEOUT_SECONDS",
    "FileSink",
    "StreamSink",
    "TokenSink",
    "granted_scopes_are_acceptable",
    "mint_refresh_token",
]

_ERROR_CODE: Final = re.compile(r"[a-z_]{1,40}")

_OPENID_SCOPES: Final = frozenset({"openid", "profile", "email", "offline_access"})
"""Scopes Entra adds on its own and that read nothing."""


def granted_scopes_are_acceptable(granted: str, wanted: tuple[str, ...]) -> bool:
    """Whether a token response's ``scope`` is exactly what was asked for, and nothing wider.

    Graph's scopes arrive prefixed (``https://graph.microsoft.com/Mail.Read``) and, in some
    documentation, percent-encoded; both are normalized, case is ignored, and the OpenID scopes
    Entra adds itself are allowed. Every scope that was **asked for and carries data** must be
    present (a user who declined ``Mail.Read`` has signed in for nothing), and anything else is
    refused: a ``.ReadWrite`` or ``.Send`` scope in the answer means the registration grants more
    than the connector asks for, and a token that can write is never kept.
    """
    seen = {
        unquote(scope).removeprefix(GRAPH_SCOPE_PREFIX).casefold()
        for scope in granted.split()
        if scope
    }
    asked = {scope.casefold() for scope in wanted}
    if not (seen - _OPENID_SCOPES) <= (asked - _OPENID_SCOPES):
        return False
    return (asked - _OPENID_SCOPES) <= seen


async def mint_refresh_token(
    config: OutlookConfig,
    *,
    announce: Callable[[str], None] = ignore_message,
    open_browser: Callable[[str], object] | None = None,
    timeout_seconds: float = DEFAULT_SIGN_IN_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SecretStr:
    """Run the authorization code flow once and return the refresh token Entra issued.

    The token is **not verified here**: that it belongs to the instance's mailbox is the caller's
    check, :meth:`~manicule.connectors.outlook_auth.OutlookCredential.confirm_mailbox`, made before
    it is written anywhere.

    Args:
        config: The instance's settings. ``account`` is offered to Entra as a hint for which
            account to choose; it decides nothing, because the mailbox is checked afterward.
        announce: Receives the lines a person needs while this waits, including the address to
            open when no browser could be started. None carries a secret.
        open_browser: Opens an address. ``None`` uses the system's default browser.
        timeout_seconds: How long to wait for the sign-in to finish.
        transport: Replaces the network for the code exchange.

    Raises:
        ConfigError: The HTTP client is not installed.
        CredentialError: The person declined, the redirect failed the ``state`` check, the wait
            ran out, or Entra refused the exchange or granted something other than what was
            asked. The message never carries the code, the token or Microsoft's free text.
    """
    import importlib.util  # noqa: PLC0415 - kept beside its only use

    if importlib.util.find_spec("httpx") is None:
        msg = (
            "signing in to Microsoft needs the HTTP client, which this installation lacks. "
            "Install it with `pip install 'manicule[connectors]'`."
        )
        raise ConfigError(msg)
    if open_browser is None:
        import webbrowser  # noqa: PLC0415 - only a person at a terminal needs a browser

        open_browser = webbrowser.open

    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(32)
    async with Loopback(state, host=config.login_redirect_host, provider="Microsoft") as loopback:
        url = f"{authorize_url(config.tenant_id)}?" + urlencode(
            {
                "client_id": config.client_id,
                "response_type": "code",
                "redirect_uri": loopback.redirect_uri,
                "response_mode": "query",
                "scope": " ".join(config.scopes),
                "code_challenge": pkce_challenge(verifier),
                "code_challenge_method": "S256",
                "state": state,
                "login_hint": config.account,
                "prompt": "select_account",
            }
        )
        announce(
            "Opening your browser to sign in. Choose the account this instance indexes. "
            f"If nothing opens, open this address yourself:\n  {url}"
        )
        # On a thread of its own, not waited for, and a daemon: a console browser blocks until the
        # person quits it, and neither the redirect, the timeout nor the exit may wait on that.
        threading.Thread(
            target=open_browser_quietly,
            args=(open_browser, url),
            name="outlook-login-browser",
            daemon=True,
        ).start()
        announce(f"Waiting up to {timeout_seconds:g} seconds for the sign-in to finish.")
        try:
            async with asyncio.timeout(timeout_seconds):
                code = await loopback.code()
        except TimeoutError:
            msg = (
                f"no sign-in arrived within {timeout_seconds:g} seconds, so nothing was "
                f"issued. Run the command again; `--timeout` allows longer."
            )
            raise CredentialError(msg) from None
        redirect_uri = loopback.redirect_uri

    return await _exchange(
        config, code=code, verifier=verifier, redirect_uri=redirect_uri, transport=transport
    )


async def _exchange(
    config: OutlookConfig,
    *,
    code: SecretStr,
    verifier: str,
    redirect_uri: str,
    transport: httpx.AsyncBaseTransport | None,
) -> SecretStr:
    """Redeem the authorization code for a refresh token, and check what it grants."""
    import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

    form = {
        "grant_type": "authorization_code",
        "code": code.get_secret_value(),
        "client_id": config.client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "scope": " ".join(config.scopes),
    }
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=config.request_timeout_seconds
        ) as http:
            response = await http.post(token_url(config.tenant_id), data=form)
    except httpx.TimeoutException as exc:
        msg = "Microsoft's token endpoint did not answer in time"
        raise RequestTimeoutError(msg) from exc
    except httpx.HTTPError as exc:
        msg = "Microsoft's token endpoint could not be reached"
        raise ConnectorError(msg) from exc

    body = _json_object(response)
    if response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED):
        raise _refused(body.get("error"))
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        msg = "Microsoft's token endpoint is throttling this client"
        raise RateLimitedError(msg)
    if response.status_code != HTTPStatus.OK:
        msg = f"Microsoft's token endpoint answered HTTP {response.status_code}"
        raise RemoteError(msg, response.status_code)

    granted = body.get("scope")
    if not isinstance(granted, str) or not granted_scopes_are_acceptable(granted, config.scopes):
        msg = (
            "Microsoft issued a token for something other than exactly the read-only scopes "
            "this instance asks for, so it was discarded. Nothing wider than read access to "
            "mail and calendar is ever kept."
        )
        raise CredentialError(msg)
    token = body.get("refresh_token")
    if not isinstance(token, str) or not token:
        msg = (
            "Microsoft signed the account in but issued no refresh token. Check that the sign-in "
            "asked for offline_access, and that the app registration allows public client flows."
        )
        raise CredentialError(msg)
    return SecretStr(token)


def _refused(error: object) -> CredentialError:
    """The typed error for a refused exchange, from Microsoft's error code and nothing else."""
    code = error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else "unknown"
    if code in {"invalid_client", "unauthorized_client"}:
        msg = (
            f"Microsoft rejected the app registration ({code}). Check that client_id and "
            f"tenant_id belong together, that the registration allows public client flows, and "
            f"that it registers a Mobile and desktop redirect of http://localhost."
        )
    elif code == "invalid_grant":
        msg = (
            "Microsoft refused the authorization code (invalid_grant). It is single use and "
            "short lived; run the command again and finish the sign-in in one go."
        )
    elif code in {"consent_required", "interaction_required", "invalid_scope"}:
        msg = (
            f"Microsoft would not issue the token without more consent or a different set of "
            f"permissions ({code}). Ask the directory's administrator to grant this app the "
            f"delegated permissions it requests, or to allow user consent for it."
        )
    else:
        msg = f"Microsoft refused the token exchange ({code})."
    return CredentialError(msg)


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
