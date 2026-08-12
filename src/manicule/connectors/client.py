"""The HTTP layer: authentication, throttling, pagination and downloads.

Separated from the connector so that the two things that are hard to get right — backing off
when the source says to, and following a cursor without corrupting it — are exercised on their
own rather than only through a sync.

**Everything here is driven through an injected transport in the suites.** ``httpx`` takes a
transport object, so the tests in ``tests/connectors`` run a synthetic Confluence over the
real client: the real retry loop, the real header building, the real query encoding. A mock of
the client itself would test the mock.

**Backoff is deterministic and has no jitter.** Jitter exists to keep a fleet of clients from
retrying in lockstep; a self-hosted index syncing one Confluence is not a fleet, and a
deterministic delay is one a test can assert on. Where a delay would be long enough to matter,
the connector stops instead (``max_retry_after_seconds``).

**The credential is consulted per request and redirects are followed by hand.** Both are the
same decision, and it is the decision browser-session authentication forces. A credential that
expires has to be asked for again before every request, because a sync outlives sessions; and a
redirect has to be a redirect at the moment it is judged, because the alternative — letting the
HTTP client follow it — converts "302 to the identity provider" into "200 carrying a sign-in
page", which is a successful response no downstream stage can tell from a document.
:mod:`manicule.connectors.credentials` holds the first half and
:mod:`manicule.connectors.intercept` the second.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.credentials import Credential, token_credential
from manicule.connectors.errors import (
    AttachmentTooLargeError,
    ConnectorError,
    CursorExpiredError,
    NotFoundError,
    RateLimitedError,
    RemoteError,
    SessionExpiredError,
    UntrustedLinkError,
)
from manicule.connectors.intercept import Answer, offsite, signed_out, signin_path
from manicule.connectors.pagination import next_page, origin_of

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = ["BACKOFF_BASE_SECONDS", "MAX_REDIRECTS", "ConfluenceClient", "Downloaded"]

BACKOFF_BASE_SECONDS = 0.5
"""First delay after a retryable failure. Doubles per attempt, capped by configuration."""

MAX_REDIRECTS = 5
"""How many redirects one request may take before the client calls it a loop."""

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

type Params = Sequence[tuple[str, str]]
type Json = Mapping[str, object]


class Downloaded:
    """Bytes from the source, and what the source said they were."""

    __slots__ = ("content", "media_type")

    def __init__(self, content: bytes, media_type: str) -> None:
        self.content = content
        self.media_type = media_type


class ConfluenceClient:
    """One authenticated conversation with a Confluence instance.

    Args:
        config: Already credential-resolved — see
            :func:`~manicule.connectors.config.resolve_credentials`.
        credential: What each request authenticates with. ``None`` builds the one this
            configuration's token implies, which is every mode except a browser session — that
            one is not derivable from configuration and the factory passes it in.
        transport: Injected in tests. ``None`` uses the network.
        sleep: How to wait between attempts. Injected so a test asserting the backoff does not
            have to serve it in real time.
        clock: Monotonic source, for the cursor-lifetime check.
    """

    def __init__(
        self,
        config: ConfluenceConfig,
        *,
        credential: Credential | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._credential = credential if credential is not None else token_credential(config)
        self._transport = transport
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._clock = clock
        self._client: httpx.AsyncClient | None = None
        self.requests = 0
        """How many HTTP requests this client has made, retries included. Reported as a metric."""

        self.throttled = 0
        """How many of them came back throttled. The signal that a sync is rate-limit bound."""

    # --- lifecycle -----------------------------------------------------------------------

    async def setup(self) -> None:
        """Open the underlying client.

        The ``httpx`` import lives here rather than at module scope so that registering the
        connector plugin — which happens in every process that starts, before configuration is
        read — does not load an HTTP stack on a machine that never syncs anything.
        """
        if self._client is not None:
            return
        import httpx  # noqa: PLC0415 - see docstring

        self._client = httpx.AsyncClient(
            transport=self._transport,
            headers={"Accept": "application/json"},
            timeout=self._config.request_timeout_seconds,
            follow_redirects=False,
        )

    async def teardown(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    def _headers(self) -> dict[str, str]:
        """The credential for the request about to be made.

        Built per request rather than once at :meth:`setup`, which is the difference a browser
        session forces. A token is the same string every time and could have been baked into
        the client; a session expires on the instance's schedule and is renewed out of band, so
        the only correct time to ask what the credential is — and whether there still is one —
        is immediately before using it.

        Raises:
            SessionExpiredError: The credential has a lifetime and has outlived it. Raised
                here, before the request, because a request made with a dead session comes back
                as a sign-in page rather than as an error.
        """
        return dict(self._credential.authorize().headers)

    # --- requests ------------------------------------------------------------------------

    def url(self, path: str) -> str:
        """An absolute URL for a path relative to the configured site root."""
        return f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _checked(self, url: str) -> str:
        """``url``, or a refusal if it is not on the configured origin.

        Most URLs this client is given ultimately come from a response — the next page's link,
        an attachment's download link — and every request carries the sync account's
        credential. Without this, a response decides who receives that credential. Checked at
        the one place requests are made rather than at each call site, so that a URL threaded
        in later cannot skip it.

        Raises:
            UntrustedLinkError: The URL points at another host.
        """
        expected = origin_of(self._config.base_url)
        found = origin_of(url)
        if found != expected:
            msg = (
                f"refusing to request {url!r}: it is on {found or 'no host'} and this connector "
                f"is configured for {expected}. Links like this come from responses, and "
                f"following one would send this account's Confluence credentials elsewhere."
            )
            raise UntrustedLinkError(msg)
        return url

    async def get_json(self, url: str, params: Params = ()) -> Json:
        """GET ``url`` and decode a JSON object from it.

        Raises:
            ConnectorError: The response was not a JSON object, which for these endpoints
                means something other than the API answered — a proxy login page, most often.
        """
        response = await self._get(url, params)
        try:
            payload: object = response.json()
        except ValueError as exc:
            msg = (
                f"{url} answered with {response.headers.get('content-type', 'no content type')} "
                f"rather than JSON. A sign-in page from a proxy in front of Confluence is the "
                f"usual cause — one manicule did not recognise as a sign-in page, since it "
                f"would have been refused already if it had. Check that "
                f"{self._config.base_url} is reachable without one. {self._credential.renewal()}"
            )
            raise ConnectorError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"{url} answered with a JSON {type(payload).__name__}, expected an object"
            raise ConnectorError(msg)
        return cast("Json", payload)

    async def paginate(self, url: str, params: Params = ()) -> AsyncIterator[Json]:
        """Yield each page of a cursor-paginated response, following ``_links.next``.

        Three failures are guarded here, and each is silent without a guard:

        - **A corrupted cursor.** Handled in :mod:`~manicule.connectors.pagination`: the link's
          query is decoded without form-encoding rules, so a ``+`` inside a cursor survives and
          is re-encoded as ``%2B`` on the way out.
        - **An expired cursor.** Cursors do not live forever, and a consumer that stalls
          mid-enumeration resumes onto one the server has forgotten. Refused before the request
          rather than after a fresh first page comes back looking like progress.
        - **A cursor that repeats.** A response pointing back at a page already walked is a
          loop, and a loop over a paginated search reads as a very large space.
        """
        current_url = url
        current_params: Params = params
        followed: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

        while True:
            payload = await self.get_json(current_url, current_params)
            received = self._clock()
            yield payload

            following = next_page(payload, base_url=self._config.base_url)
            if following is None:
                return

            held = self._clock() - received
            if held > self._config.cursor_lifetime_seconds:
                msg = (
                    f"a search cursor was held for {held:.0f}s, longer than the "
                    f"{self._config.cursor_lifetime_seconds:.0f}s this connector will trust "
                    f"one for. Confluence expires search cursors, and an expired cursor can be "
                    f"answered with a fresh first page rather than an error — which would "
                    f"enumerate the start of this space twice and its end never. Re-run the "
                    f"sync: the watermark was not advanced, so nothing is lost but the "
                    f"re-enumeration. If this recurs, the consumer is slower than the source's "
                    f"cursor lifetime and cursor_lifetime_seconds is not the setting to raise."
                )
                raise CursorExpiredError(msg)

            request = (following.url, following.params)
            if request in followed:
                msg = (
                    f"pagination was sent back to a page it has already followed "
                    f"(cursor {following.cursor[:24]!r}). Continuing would enumerate the same "
                    f"results forever while looking like an unusually large space. The whole "
                    f"request is compared rather than the cursor alone, so a link that repeats "
                    f"without one is caught too."
                )
                raise ConnectorError(msg)
            followed.add(request)
            current_url = following.url
            current_params = following.params

    async def download(self, url: str, *, max_bytes: int) -> Downloaded:
        """Stream a file, refusing one larger than ``max_bytes`` and one that is a sign-in page.

        The limit is enforced against the bytes that actually arrive, never against a declared
        ``Content-Length``: the declared length is the source's claim about the response, and
        the point of a ceiling is to survive a claim that turns out to be wrong.

        **The sign-in check matters more here than anywhere else.** A JSON endpoint answered
        with HTML fails to decode, so a sign-in page reaching one is caught by the decoder even
        without a check. An attachment download has no such backstop: whatever arrives is bytes
        with a media type, and a sign-in page arrives as perfectly good ``text/html`` that the
        parser chain would parse, chunk, embed and serve. So the same check runs on the headers
        and again on the first bytes.

        Raises:
            AttachmentTooLargeError: More than ``max_bytes`` arrived. Named as its own failure
                so the document records why it has no content rather than recording a size.
            SessionExpiredError: A sign-in answered instead of the file.
        """
        for _ in range(MAX_REDIRECTS + 1):
            following = await self._streamed(url, max_bytes=max_bytes)
            if isinstance(following, Downloaded):
                return following
            url = following
        raise self._looping(url)

    async def _streamed(self, url: str, *, max_bytes: int) -> Downloaded | str:
        """One download attempt: the file, or the URL a redirect points at."""
        client = self._require_client()
        url = self._checked(url)
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
                    redirect = self._redirect(url, response.status_code, response.headers)
                    if redirect is not None:
                        await response.aread()
                        return redirect
                    self._raise_for_status(response.status_code, url, response.headers)
                    self._verify(Answer(url, response.status_code, dict(response.headers)))
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        if not chunks:
                            self._verify(
                                Answer(url, response.status_code, dict(response.headers), chunk)
                            )
                        total += len(chunk)
                        if total > max_bytes:
                            msg = (
                                f"{url} is larger than the {max_bytes} byte ceiling "
                                f"(max_attachment_bytes) and was not downloaded. Raise the "
                                f"setting if this file is meant to be indexed."
                            )
                            raise AttachmentTooLargeError(msg)
                        chunks.append(chunk)
                    declared = response.headers.get("content-type", "")
                    return Downloaded(b"".join(chunks), declared.split(";")[0].strip())
            except httpx.TransportError as exc:
                await self._retry_transport(exc, url, attempt)
        raise self._exhausted(url)

    async def _get(self, url: str, params: Params) -> httpx.Response:
        for _ in range(MAX_REDIRECTS + 1):
            response = await self._attempt(url, params)
            redirect = self._redirect(url, response.status_code, response.headers)
            if redirect is None:
                self._verify(
                    Answer(url, response.status_code, dict(response.headers), response.content)
                )
                return response
            # The Location is a complete URL, so the query that was sent does not travel with
            # it; carrying the old parameters over would append them to whatever the redirect
            # already carries.
            url, params = redirect, ()
        raise self._looping(url)

    async def _attempt(self, url: str, params: Params) -> httpx.Response:
        """One URL's worth of requesting, including the retries. Redirects are the caller's."""
        client = self._require_client()
        url = self._checked(url)
        import httpx  # noqa: PLC0415 - see setup()

        for attempt in range(self._config.max_retries + 1):
            self.requests += 1
            try:
                response = await client.get(url, params=list(params), headers=self._headers())
            except httpx.TransportError as exc:
                await self._retry_transport(exc, url, attempt)
                continue
            wait = self._retry_delay(response.status_code, response.headers, attempt)
            if wait is not None:
                await self._sleep(wait)
                continue
            if response.status_code in _REDIRECT_STATUSES:
                return response
            self._raise_for_status(response.status_code, url, response.headers)
            return response
        raise self._exhausted(url)

    # --- redirects and sign-in pages -----------------------------------------------------

    def _redirect(self, url: str, status: int, headers: Mapping[str, str]) -> str | None:
        """Where this response says to go next, or ``None`` if it is an answer.

        Redirects are followed by this client rather than by ``httpx``, and that is the whole
        reason an expired session cannot be ingested as a page. Following automatically turns
        "302 to the identity provider" into "200 with a sign-in page", which is the one shape
        of failure no downstream stage can detect: it is a successful response of a plausible
        content type carrying real text. Following by hand means the redirect is still a
        redirect when the decision is made, and the decision is made against the configured
        origin — so nothing is ever requested from the provider and no sign-in page is fetched.

        Raises:
            UntrustedLinkError: The redirect leaves the configured origin, taking this
                account's credential with it.
            SessionExpiredError: It points at this instance's own sign-in.
        """
        if status not in _REDIRECT_STATUSES:
            return None
        location = headers.get("location", "").strip()
        if not location:
            msg = f"{url} answered {status} with no Location header, so there is nowhere to go"
            raise RemoteError(msg, status_code=status)
        import httpx  # noqa: PLC0415 - see setup()

        target = str(httpx.URL(url).join(location))
        elsewhere = offsite(target, origin=origin_of(self._config.base_url))
        if elsewhere is not None:
            raise UntrustedLinkError(f"{elsewhere}. {self._credential.renewal()}")
        signin = signin_path(target)
        if signin is not None:
            raise SessionExpiredError(f"{signin}. {self._credential.renewal()}")
        return target

    def _verify(self, answer: Answer) -> None:
        """Refuse a response that is a sign-in rather than an answer.

        Runs on every response this client returns, for every credential, because the thing in
        front of Confluence does not know or care which one was sent: a reverse proxy answers a
        personal access token with a sign-in page exactly as it answers a dead browser session.

        Raises:
            SessionExpiredError: It is a sign-in page.
        """
        reason = signed_out(answer, expected_account=self._credential.account())
        if reason is not None:
            raise SessionExpiredError(f"{reason}. {self._credential.renewal()}")

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = (
                "the Confluence client was used before setup(). The container calls setup() "
                "on components that declare it; a caller constructing this directly must too."
            )
            raise ConnectorError(msg)
        return self._client

    # --- retry policy --------------------------------------------------------------------

    def _retry_delay(
        self, status_code: int, headers: Mapping[str, str], attempt: int
    ) -> float | None:
        """How long to wait before trying again, or ``None`` if this status is not retryable.

        Raises:
            RateLimitedError: The source is throttling and either the retry budget is spent or
                it asked for a longer pause than configuration permits.
            RemoteError: A server-side failure that did not clear within the retry budget.
        """
        config = self._config
        if status_code == HTTPStatus.TOO_MANY_REQUESTS:
            self.throttled += 1
            requested = _retry_after(headers)
            wait = requested if requested is not None else _backoff(attempt)
            if wait > config.max_retry_after_seconds:
                msg = (
                    f"Confluence asked for a {wait:.0f}s pause, longer than the "
                    f"{config.max_retry_after_seconds:.0f}s ceiling (max_retry_after_seconds). "
                    f"A sync asleep that long is indistinguishable from a hung one, so it "
                    f"stops here; the watermark is not advanced and the next run resumes."
                )
                raise RateLimitedError(msg, retry_after=wait)
            if attempt >= config.max_retries:
                msg = (
                    f"Confluence throttled {config.max_retries + 1} consecutive attempts. "
                    f"A first sync of a large space is expected to hit the rate limit; if this "
                    f"is every run, lower the sync's concurrency rather than raising "
                    f"max_retries."
                )
                raise RateLimitedError(msg, retry_after=requested)
            return wait
        if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            if attempt >= config.max_retries:
                msg = (
                    f"Confluence returned {status_code} on every one of "
                    f"{config.max_retries + 1} attempts. Batch endpoints have been observed to "
                    f"fail this way when Atlassian Document Format is requested "
                    f"(docs/connectors/confluence.md §4); if this is one page rather than all "
                    f"of them, that page is the thing to look at."
                )
                raise RemoteError(msg, status_code=status_code)
            return _backoff(attempt)
        return None

    async def _retry_transport(self, exc: Exception, url: str, attempt: int) -> None:
        """Wait out a connection-level failure, or give up naming it.

        Raises:
            ConnectorError: The retry budget is spent.
        """
        if attempt >= self._config.max_retries:
            msg = (
                f"could not reach {url} after {self._config.max_retries + 1} attempts: {exc}. "
                f"The run stops without advancing the watermark, so re-running resumes."
            )
            raise ConnectorError(msg) from exc
        await self._sleep(_backoff(attempt))

    def _raise_for_status(self, status: int, url: str, headers: Mapping[str, str]) -> None:
        if status < HTTPStatus.BAD_REQUEST:
            return
        if status == HTTPStatus.NOT_FOUND:
            msg = f"{url} does not exist, or this account cannot see it"
            raise NotFoundError(msg)
        if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            raise ConnectorError(self._credential_message(status, url, headers))
        msg = f"{url} answered {status}"
        raise RemoteError(msg, status_code=status)

    def _credential_message(self, status: int, url: str, headers: Mapping[str, str]) -> str:
        """Why a rejected request was rejected, in terms of the thing that has to change.

        401 and 403 are different problems with the same shape, and conflating them sends
        somebody to rotate a working token. What the credential *is* comes from the credential
        rather than from a branch here, because "rotate the token" and "sign in again in your
        browser" are different acts and only one of them is available at a time.
        """
        config = self._config
        if status == HTTPStatus.UNAUTHORIZED:
            return (
                f"Confluence rejected the credential for {url}. This is configured as "
                f"{config.deployment.value} authenticating with "
                f"{self._credential.describe()}. {self._credential.renewal()}"
            )
        challenge = headers.get("x-authentication-denied-reason", "")
        detail = f" The source said: {challenge}." if challenge else ""
        return (
            f"the account is authenticated but not permitted to read {url}.{detail} The index "
            f"holds what this account can see and nothing else — restricted spaces stay "
            f"invisible unless the account is granted them."
        )

    def _exhausted(self, url: str) -> ConnectorError:
        """Reached only if the retry loop falls out without returning or raising."""
        return ConnectorError(f"gave up on {url} after {self._config.max_retries + 1} attempts")

    def _looping(self, url: str) -> ConnectorError:
        """More than :data:`MAX_REDIRECTS` hops, which is a redirect loop rather than a route."""
        return ConnectorError(
            f"{url} redirected more than {MAX_REDIRECTS} times without answering. A loop like "
            f"this is what an instance does when a sign-in it insists on cannot be completed by "
            f"a client that only holds cookies. {self._credential.renewal()}"
        )


def _backoff(attempt: int) -> float:
    """Exponential, from :data:`BACKOFF_BASE_SECONDS`, doubling per attempt."""
    return BACKOFF_BASE_SECONDS * (2**attempt)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The pause a ``Retry-After`` header asks for, in seconds.

    Both forms are accepted, because both are sent: a delay in seconds, and an HTTP-date. A
    client that reads only the first waits zero seconds when it gets the second, and retries
    straight into the same limit — which is how a throttled sync turns into a throttled sync
    that also looks like a hot loop.
    """
    raw: Any = headers.get("retry-after")
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())
