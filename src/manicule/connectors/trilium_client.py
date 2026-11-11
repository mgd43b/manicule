"""The HTTP layer of the Trilium connector: ETAPI requests with a token, retries and a ceiling.

``docs/connectors/trilium.md`` §2 is the design. Separate from the connector so that the parts
that are hard to get right (backing off when the server says to, never sending a token anywhere
the configuration did not name, refusing a proxy's sign-in page that arrives as content) are
exercised on their own and not only through a sync. It is the Gmail client's shape, not the
Confluence client's: one static credential, no browser session, no pagination.

**Nothing here follows a redirect.** ETAPI does not redirect, so a redirect is always something
standing between the connector and Trilium: an identity-aware proxy answering an unauthenticated
request by sending the client to its sign-in. Following it would make that a ``200`` carrying a
page, and a note's content endpoint, unlike the JSON ones, returns ``text/html`` that a sign-in
page also is. So a redirect is refused where it is found, and
:meth:`TriliumClient.content` additionally refuses a body whose ``Content-Type`` is not the one
the note was swept with.

**Everything is driven through an injected transport in the suites**, so the real retry loop,
the real headers and the real query encoding run against a synthetic server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from manicule.connectors.config import TriliumConfig
from manicule.connectors.errors import (
    ConnectorError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    RemoteError,
    UntrustedLinkError,
)
from manicule.connectors.intercept import offsite
from manicule.connectors.pagination import origin_of
from manicule.connectors.retry import backoff, retry_after

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = ["NOTE_IS_PROTECTED", "Content", "NoteProtectedError", "TriliumClient"]

NOTE_IS_PROTECTED = "NOTE_IS_PROTECTED"
"""The ETAPI error code for a request that needs a protected note's decrypted content."""

type Params = Sequence[tuple[str, str]]
type Json = Mapping[str, object]

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class NoteProtectedError(NotFoundError):
    """The note is protected, so its content is unavailable without a password.

    A kind of "gone", because for this connector it is: a protected note is never indexed, and
    one that was ordinary when it was swept and protected before it was fetched is simply not
    there any more.
    """


class Content:
    """A note's bytes, and the media type the server said they were."""

    __slots__ = ("content", "media_type")

    def __init__(self, content: bytes, media_type: str) -> None:
        self.content = content
        self.media_type = media_type


def base_media_type(value: str) -> str:
    """``text/html; charset=utf-8`` as ``text/html``."""
    return value.split(";", 1)[0].strip().lower()


class TriliumClient:
    """One authenticated conversation with a Trilium server's ETAPI.

    Args:
        config: Already token-resolved: see
            :func:`~manicule.connectors.config.resolve_trilium_token`.
        transport: Injected in tests. ``None`` uses the network.
        sleep: How to wait between attempts, injected so a test asserting backoff does not
            serve it in real time.
    """

    def __init__(
        self,
        config: TriliumConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if config.token is None:
            msg = "the Trilium client needs a token; resolve_trilium_token fills it in"
            raise ConnectorError(msg)
        self._config = config
        self._token = config.token
        self._transport = transport
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._origin = origin_of(config.base_url)
        self._client: httpx.AsyncClient | None = None
        self.requests = 0
        """How many HTTP requests this client has made, retries included."""
        self.throttled = 0
        """How many of them came back throttled."""

    async def setup(self) -> None:
        """Open the underlying client.

        The ``httpx`` import is here so that registering the plugin, which happens in every
        process that starts, does not load an HTTP stack on a machine that never syncs.
        """
        if self._client is not None:
            return
        import httpx  # noqa: PLC0415 - see docstring

        self._client = httpx.AsyncClient(
            transport=self._transport,
            timeout=self._config.request_timeout_seconds,
            follow_redirects=False,
        )

    async def teardown(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # --- the two requests the connector makes --------------------------------------------

    async def search(self, expression: str, *, limit: int) -> list[Json]:
        """``GET /etapi/notes``: every note matching ``expression``, with its metadata.

        ETAPI has no cursor, so the whole result is one response and ``limit`` is the only
        bound. The caller decides what a response of exactly ``limit`` means.
        """
        payload = await self._json(
            "notes",
            (
                ("search", expression),
                ("limit", str(limit)),
                ("includeArchivedNotes", "true"),
            ),
        )
        results = payload.get("results")
        if not isinstance(results, list):
            msg = (
                f"{self._origin} answered the note search without a results list. Something "
                f"other than ETAPI answered, a proxy in front of Trilium most often"
            )
            raise ConnectorError(msg)
        notes: list[Json] = []
        for entry in cast("list[object]", results):
            if not isinstance(entry, dict):
                msg = "the note search returned an entry that is not an object"
                raise ConnectorError(msg)
            notes.append(cast("Json", entry))
        return notes

    async def content(self, note_id: str, *, expected_type: str, max_bytes: int) -> Content:
        """``GET /etapi/notes/{id}/content``, streamed, refusing a body that is not the note.

        Args:
            note_id: The note.
            expected_type: The ``mime`` the sweep reported for it. The response's own
                ``Content-Type`` must be that, because a proxy's sign-in page is valid
                ``text/html`` and nothing else about a ``200`` distinguishes it.
            max_bytes: The ceiling, enforced against the bytes that arrive and not against a
                declared length, since the declared length is the server's claim.

        Raises:
            NoteProtectedError: The note became protected after it was swept.
            NotFoundError: The note was deleted after it was swept.
            ConnectorError: The body is larger than ``max_bytes``, or is not the media type
                the note has.
        """
        client = self._require_client()
        url = self._url(f"notes/{note_id}/content")
        import httpx  # noqa: PLC0415 - see setup()

        for attempt in range(self._config.max_retries + 1):
            self.requests += 1
            try:
                async with client.stream("GET", url, headers=self._headers()) as response:
                    wait = self._retry_delay(response.status_code, response.headers, attempt)
                    if wait is not None:
                        await response.aread()
                        await self._sleep(wait)
                        continue
                    self._refuse_redirect(url, response.status_code, response.headers)
                    if response.status_code >= HTTPStatus.BAD_REQUEST:
                        await response.aread()
                        self._raise_for_status(response.status_code, url, response.content)
                    declared = base_media_type(response.headers.get("content-type", ""))
                    if declared != base_media_type(expected_type):
                        msg = (
                            f"the content of a note came back as {declared or 'no media type'} "
                            f"where the note is {expected_type}. An identity-aware proxy in "
                            f"front of Trilium answers an unauthenticated request with its "
                            f"sign-in page, which is a 200 of text/html; the body was refused "
                            f"rather than indexed as the note"
                        )
                        raise ConnectorError(msg)
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            msg = (
                                f"a note is larger than the {max_bytes}-byte fetch cap and was "
                                f"not read. Raise ingest.max_fetch_bytes to index it"
                            )
                            raise ConnectorError(msg)
                        chunks.append(chunk)
                    return Content(b"".join(chunks), declared)
            except httpx.TransportError as exc:
                await self._retry_transport(exc, attempt)
        raise self._exhausted(url)

    # --- requests ------------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """An ETAPI URL, on the configured origin by construction."""
        return f"{self._config.base_url.rstrip('/')}/etapi/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "Accept": "application/json, */*;q=0.5",
        }

    async def _json(self, path: str, params: Params) -> Json:
        url = self._url(path)
        response = await self._get(url, params)
        try:
            payload: object = response.json()
        except ValueError as exc:
            msg = (
                f"{url} answered with {response.headers.get('content-type', 'no content type')} "
                f"rather than JSON. A sign-in page from a proxy in front of Trilium is the usual "
                f"cause; check that {self._config.base_url} is reachable with only the token"
            )
            raise ConnectorError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"{url} answered with a JSON {type(payload).__name__}, expected an object"
            raise ConnectorError(msg)
        return cast("Json", payload)

    async def _get(self, url: str, params: Params) -> httpx.Response:
        client = self._require_client()
        import httpx  # noqa: PLC0415 - see setup()

        for attempt in range(self._config.max_retries + 1):
            self.requests += 1
            try:
                response = await client.get(url, params=list(params), headers=self._headers())
            except httpx.TransportError as exc:
                await self._retry_transport(exc, attempt)
                continue
            wait = self._retry_delay(response.status_code, response.headers, attempt)
            if wait is not None:
                await self._sleep(wait)
                continue
            self._refuse_redirect(url, response.status_code, response.headers)
            self._raise_for_status(response.status_code, url, response.content)
            return response
        raise self._exhausted(url)

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = (
                "the Trilium client was used before setup(). The container calls setup() on "
                "components that declare it; a caller constructing this directly must too."
            )
            raise ConnectorError(msg)
        return self._client

    # --- redirects, status and retry -----------------------------------------------------

    def _refuse_redirect(self, url: str, status: int, headers: Mapping[str, str]) -> None:
        """Refuse a redirect, naming whether it left the configured origin.

        Raises:
            UntrustedLinkError: It points at another host, which the token would travel to.
            ConnectorError: It points somewhere on this origin, which is still not ETAPI.
        """
        if status not in _REDIRECT_STATUSES:
            return
        import httpx  # noqa: PLC0415 - see setup()

        location = headers.get("location", "").strip()
        target = str(httpx.URL(url).join(location)) if location else ""
        elsewhere = offsite(target, origin=self._origin) if target else None
        if elsewhere is not None:
            raise UntrustedLinkError(elsewhere)
        msg = (
            f"{url} redirected{f' to {target}' if target else ''}. ETAPI does not redirect, so "
            f"something in front of Trilium is answering instead, most often a proxy that wants "
            f"a sign-in. manicule does not follow, because the token travels with the request"
        )
        raise ConnectorError(msg)

    def _retry_delay(
        self, status_code: int, headers: Mapping[str, str], attempt: int
    ) -> float | None:
        """How long to wait before trying again, or ``None`` if this status is not retryable.

        Raises:
            RateLimitedError: The server is throttling and either the retry budget is spent or
                it asked for a longer pause than configuration permits.
            RemoteError: A server-side failure that did not clear within the retry budget.
        """
        config = self._config
        if status_code == HTTPStatus.TOO_MANY_REQUESTS:
            self.throttled += 1
            requested = retry_after(headers)
            wait = requested if requested is not None else backoff(attempt)
            if wait > config.max_retry_after_seconds:
                msg = (
                    f"Trilium asked for a {wait:.0f}s pause, longer than the "
                    f"{config.max_retry_after_seconds:.0f}s ceiling (max_retry_after_seconds). "
                    f"The run stops here and the next one starts over"
                )
                raise RateLimitedError(msg, retry_after=wait)
            if attempt >= config.max_retries:
                msg = f"Trilium throttled {config.max_retries + 1} consecutive attempts"
                raise RateLimitedError(msg, retry_after=requested)
            return wait
        if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            if attempt >= config.max_retries:
                msg = (
                    f"Trilium returned {status_code} on every one of "
                    f"{config.max_retries + 1} attempts"
                )
                raise RemoteError(msg, status_code=status_code)
            return backoff(attempt)
        return None

    async def _retry_transport(self, exc: Exception, attempt: int) -> None:
        """Wait out a connection-level failure, or give up naming it."""
        if attempt >= self._config.max_retries:
            msg = (
                f"could not reach {self._origin} after {self._config.max_retries + 1} attempts: "
                f"{type(exc).__name__}. The run stops and the next one starts over"
            )
            raise ConnectorError(msg) from exc
        await self._sleep(backoff(attempt))

    def _raise_for_status(self, status: int, url: str, body: bytes) -> None:
        if status < HTTPStatus.BAD_REQUEST:
            return
        code = _error_code(body)
        if status == HTTPStatus.NOT_FOUND:
            msg = f"{url} does not exist"
            raise NotFoundError(msg)
        if status == HTTPStatus.BAD_REQUEST and code == NOTE_IS_PROTECTED:
            msg = f"{url} is a protected note"
            raise NoteProtectedError(msg)
        if status == HTTPStatus.UNAUTHORIZED:
            msg = (
                f"Trilium rejected the token for {self._origin}. Create a new one under "
                f"Options, then ETAPI, and set it where the connector reads it"
            )
            raise ConnectorError(msg)
        if status == HTTPStatus.FORBIDDEN:
            msg = f"the token is authenticated but not permitted to read {url}"
            raise PermissionDeniedError(msg)
        msg = f"{url} answered {status}" + (f" ({code})" if code else "")
        raise RemoteError(msg, status_code=status)

    def _exhausted(self, url: str) -> ConnectorError:
        """Reached only if the retry loop falls out without returning or raising."""
        return ConnectorError(f"gave up on {url} after {self._config.max_retries + 1} attempts")


def _error_code(body: bytes) -> str:
    """The ``code`` of an ETAPI error body, or ``""`` for anything that is not one."""
    import json  # noqa: PLC0415 - only on the error path

    try:
        payload: object = json.loads(body)
    except ValueError:
        return ""
    if isinstance(payload, dict):
        code = cast("dict[str, object]", payload).get("code")
        if isinstance(code, str):
            return code
    return ""
