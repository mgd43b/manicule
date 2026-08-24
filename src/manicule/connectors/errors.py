"""What a connector raises, and what each one means for the run that saw it.

Every one descends from :class:`~manicule.core.errors.ManiculeError`, so the pipeline can tell
"the source said no" from "something unexpected broke". They live here rather than in
:mod:`manicule.core.errors` because they describe HTTP-shaped failures, and core carries no
HTTP.

The distinction that matters to the pipeline is **whether the enumeration was complete**.
``docs/ingest.md`` §11.1 refuses to diff a partial enumeration against the stored set, because
the ids seen so far are a prefix rather than the truth, and diffing a prefix soft-deletes
everything not yet reached. So a connector never swallows a mid-enumeration failure and never
returns what it has: it raises, and the pipeline discards the prefix.
"""

from __future__ import annotations

from manicule.core.errors import ConfigError, ManiculeError

__all__ = [
    "AttachmentTooLargeError",
    "BodyUnavailableError",
    "ConnectorError",
    "CrawlerAddressError",
    "CrawlerChallengeError",
    "CrawlerMediaTypeError",
    "CrawlerPolicyError",
    "CrawlerRedirectError",
    "CrawlerResponseTooLargeError",
    "CrawlerRobotsError",
    "CrawlerScopeError",
    "CrawlerThrottleError",
    "CrawlerTimeoutError",
    "CrawlerTransportError",
    "CursorExpiredError",
    "NotFoundError",
    "PermissionDeniedError",
    "ProviderRefusedError",
    "RateLimitedError",
    "RemoteError",
    "RequestTimeoutError",
    "SessionExpiredError",
    "SessionMissingError",
    "UntrustedLinkError",
]


class ConnectorError(ManiculeError):
    """A connector could not do what it was asked."""


class CrawlerPolicyError(ConnectorError):
    """A crawler input or response violated its closed outbound-request policy."""


class CrawlerScopeError(CrawlerPolicyError):
    """A normalized URL is outside the connector's declared publication scope."""


class CrawlerAddressError(CrawlerPolicyError):
    """A destination or connected peer is not a globally routable approved address."""


class CrawlerRedirectError(CrawlerPolicyError):
    """A redirect would weaken transport security or exceed its bounded policy."""


class CrawlerTransportError(CrawlerPolicyError):
    """A bounded crawler request could not produce a usable response."""


class CrawlerTimeoutError(CrawlerTransportError):
    """A crawler request exhausted its bounded timeout and retry budget."""


class CrawlerThrottleError(CrawlerTransportError):
    """A source throttle was malformed, excessive, or remained after bounded retries."""


class CrawlerResponseTooLargeError(CrawlerTransportError):
    """A response crossed its wire, decoded, header, or run-wide byte ceiling."""


class CrawlerRobotsError(CrawlerTransportError):
    """Robots policy refused a page or could not be established safely."""


class CrawlerChallengeError(CrawlerTransportError):
    """A login, challenge, or error template answered where page content was expected."""


class CrawlerMediaTypeError(CrawlerTransportError):
    """A response media type is outside the crawler's textual allowlist."""


class RemoteError(ConnectorError):
    """The source answered with a status the connector cannot use."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class NotFoundError(RemoteError):
    """The source no longer has the thing that was asked for.

    Ordinary during a sync: a page discovered ten minutes ago can be deleted before it is
    fetched. Its own class so a caller can treat it as "gone" rather than as a fault.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class PermissionDeniedError(RemoteError):
    """The authenticated account cannot read a specific Confluence resource."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=403)


class RateLimitedError(RemoteError):
    """The source is throttling and the connector has stopped waiting.

    Raised after the retry budget is spent, or when ``Retry-After`` asks for a longer pause
    than configuration permits. Waiting an hour inside one HTTP call is not resilience: it is a
    sync that appears to be running and is not.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class CursorExpiredError(ConnectorError):
    """A pagination cursor was held longer than the source keeps one alive.

    Search cursors expire (``docs/connectors/confluence.md`` §2), so a sync whose consumer
    stalls mid-enumeration resumes onto a cursor the server has forgotten. The server's own
    answer to that is an error or, worse, a fresh first page — which would enumerate the
    opening of the corpus twice and the tail never. Refusing before the request is sent makes
    the failure legible and keeps the run's incompleteness honest.
    """


class RequestTimeoutError(ConnectorError):
    """One request ran out of time while the source stayed reachable and healthy.

    Raised only when the caller asked for timeouts surfaced
    (``ConfluenceClient.get_json(..., surface_read_timeouts=True)``), for a read timeout or an
    explicitly classified transient gateway timeout (504). Everything else — a connection
    refused, a 500, a throttle — keeps the ordinary retry policy, because those are not
    statements about the *shape* of the request.

    Its own class because a timeout at a deep pagination offset is the one failure where
    repeating the same request is known to be useless and *changing* it is known to help: the
    source can often answer the same offset with a smaller page. The authoritative direct
    inventory catches exactly this type to shrink its requested page size at the same offset,
    and treats every other failure as final. A timed-out offset is therefore never deletion
    evidence and never an end-of-inventory marker — it is a request that needs a different
    size, or a typed incomplete run.
    """


class BodyUnavailableError(ConnectorError):
    """The page exists but has no usable value in the body format that was asked for.

    Narrower than :class:`ConnectorError` on purpose. Cloud answers an unavailable ADF body by
    asking for storage format; an unavailable storage body has no alternative and fails closed.
    Falling back on *any* error would retry through a second endpoint after a 429 or a rejected
    credential, doubling the load on a source that has just said stop and hiding which of the two
    problems it was.
    """


class UntrustedLinkError(ConnectorError):
    """A request was about to be sent somewhere other than the configured host.

    Responses are full of URLs — ``_links.next``, ``_links.base``, an attachment's
    ``_links.download`` — and every request carries the sync account's credential. Using one
    without checking where it points means a response can decide who receives that credential.
    So the origin of every URL is checked against the configured one before the request is
    made, not only for the pagination link where the hazard is most obvious.
    """


class AttachmentTooLargeError(ConnectorError):
    """An attachment exceeded the configured size ceiling and was not downloaded whole."""


class SessionExpiredError(ConnectorError):
    """The browser session this connector authenticates with is no longer usable.

    Its own class because it is the one credential failure a person answers by going back to a
    browser rather than by editing configuration, and because it does not arrive as a 401. An
    instance behind an identity provider answers an expired session with a redirect to that
    provider, so "signed out" reaches the client as a 302, or as a **200 whose body is a
    sign-in page** — a response that is successful by every measure a client usually applies.
    Indexing that page as content is the worst outcome available here: it is plausible text,
    it arrives for every request, and nothing downstream can tell it from a document.

    Raised in three places, all of which mean the same thing to the run: before the first
    request, when the stored session is older than this connector will trust; before any
    request, when the session ages out mid-sync; and on any response that shows the request
    was answered by a sign-in rather than by Confluence.

    The run stops rather than pausing. Renewal is an out-of-band act — a person signs in to a
    browser — and there is no interval a sync can usefully wait; the cursor it is holding
    would expire first (:class:`CursorExpiredError`). Stopping leaves the watermark
    unadvanced, so a re-run after a fresh sign-in resumes rather than starting over.
    """


class ProviderRefusedError(ConfigError):
    """The selected browser-session provider could not run, and nothing else was tried.

    **The class exists so that "it did not work" cannot quietly become "something else worked".**
    A login that fell back from an installed browser to bundled Chromium would authenticate the
    person against a browser their identity provider has never seen, and one that fell back to
    the paste prompt would ask for a cookie header from somebody who had just asked not to be —
    in both cases succeeding at something other than what was requested, which is the failure
    mode hardest to notice afterwards. So a provider that cannot run raises, and the raise is
    the end of the attempt.

    Every message names the alternatives that *are* available on this machine, because a refusal
    an operator cannot act on is a refusal that gets worked around. Naming them is not the same
    as taking one: the choice stays with the person, which is the whole distinction between a
    refusal and a fallback.

    A :class:`~manicule.core.errors.ConfigError`, because it is nearly always the arrangement —
    a browser that is not installed, a name that matches two of them, a profile directory that
    cannot be made safely — rather than a fault at the instance. No message carries a cookie, a
    profile's contents, or a path outside the configured private root.
    """


class SessionMissingError(ConfigError):
    """Nobody has signed in to this instance in this process's lifetime.

    Distinct from :class:`SessionExpiredError`, and the distinction is the point rather than a
    taxonomy. An expired session was captured and has aged out; **this one was never captured at
    all**, which after a restart is the ordinary state and not a fault — sessions live in the
    server's memory, so a crash, a logout or a reboot ends every one of them.

    A :class:`~manicule.core.errors.ConfigError` rather than a
    :class:`ConnectorError`, because that is what this was before it had a name of its own and
    because it is genuinely about the arrangement rather than about the source: nothing was asked
    of Confluence and Confluence said nothing. Every existing ``except ConfigError`` therefore
    still catches it.

    **What the separate class buys is the one thing a log line could not.** The scheduler's loop
    reports every refusal the same way, so "no session" and "the instance was unreachable" read
    identically at three in the morning — one needs a person at a browser and the other needs
    nothing at all. :class:`~manicule.app.served.Scheduler` matches on this type, records it as
    its own state, and says which of the two it was. Matching on the message text would be a
    guard that breaks when somebody improves the wording.
    """
