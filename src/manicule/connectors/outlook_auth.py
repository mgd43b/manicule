"""The Outlook credential: a refresh token turned into Graph access tokens, for one instance.

``docs/connectors/outlook.md`` §1. The Microsoft counterpart of
:class:`~manicule.connectors.credentials.RefreshTokenCredential`, and different from it in the
ways that matter:

**A public client, so no client secret.** The Entra app registration is a public client; a
refresh is the token, the client id and the scopes, and nothing else is secret.

**Microsoft rotates refresh tokens, and says the old one is to be discarded.** Every refresh may
return a replacement. This credential adopts it in memory at once and hands it to ``persist``
(a file written readable only by its owner, when the instance names one), so the token that is
stored is always the newest. The old token is *not* revoked when it is used, so a replacement
that cannot be written costs the token's sliding lifetime and nothing else: the failure is logged
without the token and the run goes on.

**The token endpoint is built from the tenant GUID and nothing else.** The host is a constant
and the tenant is validated as a GUID (:class:`~manicule.connectors.outlook_config.OutlookConfig`),
so no setting can send a refresh token to another host.

**Nothing secret is reachable by accident.** Tokens are :class:`~pydantic.SecretStr`, the repr
names only the instance, the exchange is a POST form body and never a query string, and no error
carries a response body, only the short ``error`` code Microsoft defines.

No HTTP client is imported at module level, for the reason
:mod:`manicule.connectors.credentials` gives.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

from pydantic import SecretStr

from manicule.connectors.errors import (
    ConnectorError,
    CredentialError,
    GrantRevokedError,
    MailboxMismatchError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
    TokenRejectedError,
)
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    import httpx

    from manicule.connectors.outlook_config import OutlookConfig

__all__ = [
    "AUTHORITY",
    "GRAPH_BASE",
    "GRAPH_ME_URL",
    "GRAPH_SCOPE_PREFIX",
    "OutlookCredential",
    "authorize_url",
    "identity_mismatch",
    "outlook_credential",
    "token_url",
]

AUTHORITY: Final = "https://login.microsoftonline.com"
"""Microsoft Entra's public-cloud host. A constant for the reason the token endpoint is one."""

GRAPH_BASE: Final = "https://graph.microsoft.com/v1.0"
"""Microsoft Graph, public cloud, v1.0."""

GRAPH_ME_URL: Final = f"{GRAPH_BASE}/me?$select=id,userPrincipalName,mail"
"""The one read the mailbox pin makes."""

GRAPH_SCOPE_PREFIX: Final = "https://graph.microsoft.com/"
"""What Graph's scopes are prefixed with in a token response; stripped before comparing."""

REFRESH_MARGIN_SECONDS: Final = 60.0
FALLBACK_LIFETIME_SECONDS: Final = 300.0

_ERROR_CODE: Final = re.compile(r"[a-z_]{1,40}")

_REVOKED_CODES: Final = frozenset({"invalid_grant", "interaction_required", "consent_required"})
"""Answers a person fixes by signing in again: the grant is gone, expired, or now needs a step
(multi-factor, a conditional-access policy, consent) that only a browser can satisfy."""


def token_url(tenant_id: str) -> str:
    """The v2.0 token endpoint of one directory."""
    return f"{AUTHORITY}/{tenant_id}/oauth2/v2.0/token"


def authorize_url(tenant_id: str) -> str:
    """The v2.0 authorization endpoint of one directory."""
    return f"{AUTHORITY}/{tenant_id}/oauth2/v2.0/authorize"


def _require_httpx() -> None:
    if importlib.util.find_spec("httpx") is None:
        msg = (
            "the Outlook connector needs the HTTP client, which this installation lacks. Install "
            "it with `pip install 'manicule[connectors]'`."
        )
        raise ConfigError(msg)


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


def _usable_lifetime(expires_in: object) -> float:
    """Seconds an access token is offered for: its lifetime less the margin."""
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        return FALLBACK_LIFETIME_SECONDS
    if expires_in > 2 * REFRESH_MARGIN_SECONDS:
        return float(expires_in) - REFRESH_MARGIN_SECONDS
    return float(expires_in) / 2


def identity_mismatch(
    reported: tuple[str, ...], account: str, renewal: str
) -> MailboxMismatchError | None:
    """The refusal for a token that reads another mailbox than ``account``, or ``None``.

    One rule for every place that compares ``/me`` with the configured account: ``doctor``, the
    connector's setup and the start of each run. ``reported`` is the ``userPrincipalName`` and the
    ``mail`` Graph gave, either of which may be absent; the account matches if it equals one,
    without regard to case. Neither address is in the message.
    """
    wanted = account.strip().casefold()
    if any(value.strip().casefold() == wanted for value in reported if value):
        return None
    msg = (
        "the credential belongs to a different mailbox than this instance's configured "
        "account (an alias counts as different: configure the sign-in name or the primary "
        "address). Indexing another mailbox is a different instance under a new name, so the "
        "run did not begin. " + renewal
    ).rstrip()
    return MailboxMismatchError(msg)


@dataclass(eq=False, repr=False)
class OutlookCredential:
    """A refresh token turned into short-lived access tokens, for one Outlook instance.

    Cached until shortly before ``expires_in``, one refresh at a time: concurrent callers wait on
    a lock and share the token the first one minted. A 401 from Graph earns one forced refresh
    (:meth:`access_token` with ``refresh=True``) and one resend, which is the Graph client's loop
    and not this class's; this class types what a refresh can answer.
    """

    connector: str
    """The instance's name: what the renewal text tells a person to log in to."""

    tenant_id: str
    client_id: str
    refresh_token: SecretStr
    scopes: tuple[str, ...]

    persist: Callable[[SecretStr], None] | None = None
    """Where a rotated refresh token goes. ``None`` keeps it in memory only."""

    transport: httpx.AsyncBaseTransport | None = None
    """Replaces the network. ``None`` uses the real one; a test supplies a ``MockTransport``."""

    clock: Callable[[], float] = time.monotonic
    timeout_seconds: float = 30.0

    rotations: int = field(default=0, init=False)
    """How many times Microsoft replaced the refresh token during this process's life."""

    _access: SecretStr | None = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @override
    def __repr__(self) -> str:
        return f"OutlookCredential(connector={self.connector!r})"

    def renewal(self) -> str:
        return (
            f"Run `manicule connector login {self.connector}` to mint a new refresh token, and "
            f"put it where the configuration reads it."
        )

    def rejected(self) -> TokenRejectedError:
        """What to raise when Graph rejects a request again with a freshly minted access token."""
        return TokenRejectedError(
            "Microsoft Graph rejected the request with a freshly minted access token. The grant "
            "may have lost a scope, or a conditional-access policy now blocks it. " + self.renewal()
        )

    async def access_token(self, *, refresh: bool = False) -> str:
        """The bare access token, minted if none is fresh.

        ``refresh=True`` is for a caller that has just been answered 401: it mints a new token
        whatever the cache holds.

        Raises:
            GrantRevokedError: Microsoft no longer honors the refresh token.
            CredentialError: Microsoft rejected the app registration, or answered nonsense.
            RemoteError: The token endpoint failed; a later attempt may succeed.
        """
        if refresh:
            async with self._lock:
                self._access = None
                return (await self._refresh()).get_secret_value()
        cached = self._fresh()
        if cached is not None:
            return cached.get_secret_value()
        async with self._lock:
            # Re-checked under the lock: a caller ahead of this one has usually just minted it.
            return (self._fresh() or await self._refresh()).get_secret_value()

    async def confirm_mailbox(self, account: str) -> None:
        """Check that this credential reads the mailbox the instance is configured for.

        One refresh if none is cached and one ``GET /me``, compared with ``account``. Called by
        ``doctor`` and by ``manicule connector login`` before a token is written.

        Raises:
            MailboxMismatchError: The token belongs to another mailbox. Neither address is in the
                message.
            CredentialError: The credential cannot be used at all.
            RemoteError: Graph failed to answer.
        """
        _require_httpx()
        import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

        token = await self.access_token()
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout_seconds
            ) as http:
                response = await http.get(
                    GRAPH_ME_URL, headers={"Authorization": f"Bearer {token}"}
                )
        except httpx.TimeoutException as exc:
            msg = "Microsoft Graph did not answer GET /me in time"
            raise RequestTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Microsoft Graph could not be reached for GET /me"
            raise ConnectorError(msg) from exc
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            msg = "Microsoft Graph is throttling GET /me"
            raise RateLimitedError(msg)
        if response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            msg = (
                f"Microsoft Graph refused GET /me (HTTP {response.status_code}). The grant may "
                f"lack the User.Read scope, or a conditional-access policy blocks this client. "
                + self.renewal()
            )
            raise CredentialError(msg)
        if response.status_code != HTTPStatus.OK:
            msg = f"Microsoft Graph answered GET /me with HTTP {response.status_code}"
            raise RemoteError(msg, response.status_code)
        body = _json_object(response)
        reported = tuple(
            value.strip()
            for key in ("userPrincipalName", "mail")
            if isinstance(value := body.get(key), str) and value.strip()
        )
        if not reported:
            msg = (
                "Microsoft Graph answered GET /me without an address; the mailbox cannot be checked"
            )
            raise CredentialError(msg)
        mismatch = identity_mismatch(reported, account, self.renewal())
        if mismatch is not None:
            raise mismatch

    # -- the exchange ----------------------------------------------------------------

    def _fresh(self) -> SecretStr | None:
        if self._access is not None and self.clock() < self._expires_at:
            return self._access
        return None

    async def _refresh(self) -> SecretStr:
        """Exchange the refresh token for an access token. Called with the lock held."""
        _require_httpx()
        import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

        form = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token.get_secret_value(),
            "client_id": self.client_id,
            "scope": " ".join(self.scopes),
        }
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout_seconds
            ) as http:
                response = await http.post(token_url(self.tenant_id), data=form)
        except httpx.TimeoutException as exc:
            msg = "Microsoft's token endpoint did not answer in time"
            raise RequestTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Microsoft's token endpoint could not be reached"
            raise ConnectorError(msg) from exc

        body = _json_object(response)
        if response.status_code == HTTPStatus.OK:
            token = body.get("access_token")
            if not isinstance(token, str) or not token:
                msg = "Microsoft's token endpoint answered without an access token"
                raise CredentialError(msg)
            self._access = SecretStr(token)
            self._expires_at = self.clock() + _usable_lifetime(body.get("expires_in"))
            self._adopt(body.get("refresh_token"))
            return self._access
        if response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED):
            raise self._refused(body.get("error"))
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            msg = "Microsoft's token endpoint is throttling this client"
            raise RateLimitedError(msg)
        msg = f"Microsoft's token endpoint answered HTTP {response.status_code}"
        raise RemoteError(msg, response.status_code)

    def _adopt(self, replacement: object) -> None:
        """Keep a rotated refresh token: in memory at once, and where the instance says."""
        if not isinstance(replacement, str) or not replacement:
            return
        if replacement == self.refresh_token.get_secret_value():
            return
        self.refresh_token = SecretStr(replacement)
        self.rotations += 1
        if self.persist is None:
            return
        try:
            self.persist(self.refresh_token)
        except Exception as exc:  # noqa: BLE001 - the token is in memory; the old one still works
            # The type only: the message of an OSError names the path, and nothing here may name
            # the token. The old token is not revoked by use, so this costs lifetime, not access.
            logging.getLogger(__name__).warning(
                "Outlook instance %s could not store its rotated refresh token (%s); the previous "
                "token remains valid and the new one is held in memory",
                self.connector,
                type(exc).__name__,
            )

    def _refused(self, error: object) -> CredentialError:
        """The typed error for a refused exchange, from Microsoft's error code and nothing else.

        ``error_description`` is not repeated: it is free text that carries the account and a
        trace id, and the code says which of the few things a person can do applies.
        """
        code = error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else "unknown"
        if code in _REVOKED_CODES:
            msg = (
                f"Microsoft no longer honors this instance's refresh token ({code}): it was "
                f"revoked, it expired, or the directory now requires a sign-in step only a "
                f"browser can complete. " + self.renewal()
            )
            return GrantRevokedError(msg)
        if code in {"invalid_client", "unauthorized_client"}:
            msg = (
                f"Microsoft rejected the app registration ({code}). Check that client_id and "
                f"tenant_id belong together, and that the registration allows public client "
                f"flows (Authentication, Allow public client flows)."
            )
            return CredentialError(msg)
        return CredentialError(f"Microsoft refused the token exchange ({code}). {self.renewal()}")


def outlook_credential(
    config: OutlookConfig,
    *,
    instance: str,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> OutlookCredential:
    """The credential an Outlook instance mints access tokens with.

    Pure construction, no network: the factory is synchronous and the container calls it from
    inside the event loop, so the refresh and the mailbox check are
    :meth:`OutlookCredential.confirm_mailbox`'s, awaited before the connector is handed out. When
    the instance names a ``refresh_token_file``, a rotated token is written back to it.

    Raises:
        ConfigError: The refresh token is unset, which
            :func:`~manicule.connectors.outlook_config.resolve_outlook_credentials` fills in and
            refuses on; a caller building this some other way has to call it first.
    """
    if config.refresh_token is None:
        msg = (
            "the Outlook refresh token is not set. resolve_outlook_credentials() fills it from "
            "its file or the environment and refuses when it is absent; a caller building this "
            "some other way has to call it too."
        )
        raise ConfigError(msg)
    persist: Callable[[SecretStr], None] | None = None
    if config.refresh_token_file is not None:
        from manicule.connectors.gmail_login import FileSink  # noqa: PLC0415 - pure, kept local

        persist = FileSink(Path(config.refresh_token_file), overwrite=True).write
    return OutlookCredential(
        connector=instance,
        tenant_id=config.tenant_id,
        client_id=config.client_id,
        refresh_token=config.refresh_token,
        scopes=config.scopes,
        persist=persist,
        transport=transport,
        clock=clock,
        timeout_seconds=config.request_timeout_seconds,
    )
