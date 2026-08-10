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

from manicule.core.errors import ManiculeError

__all__ = [
    "AttachmentTooLargeError",
    "BodyUnavailableError",
    "ConnectorError",
    "CursorExpiredError",
    "NotFoundError",
    "RateLimitedError",
    "RemoteError",
    "UntrustedLinkError",
]


class ConnectorError(ManiculeError):
    """A connector could not do what it was asked."""


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


class BodyUnavailableError(ConnectorError):
    """The page exists but did not come back in the body format that was asked for.

    Narrower than :class:`ConnectorError` on purpose, because it is the one failure the caller
    answers by asking for a different format. Falling back on *any* error would retry through a
    second endpoint after a 429 or a rejected credential, doubling the load on a source that
    has just said stop and hiding which of the two problems it was.
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
