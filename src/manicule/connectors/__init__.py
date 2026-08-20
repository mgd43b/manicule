"""Sources manicule ingests from.

A connector answers three questions about a source: what has changed since last time
(``discover``), what one document's bytes are (``fetch``), and what still exists
(``reconcile``). The third is the one that is easy to leave out and expensive to have left
out: incremental sync cannot detect a deletion, because a deleted document simply stops
appearing, so without a reconciliation pass the index serves removed documents forever.

Confluence is the source for v1. Everything specific to it — Atlassian Document Format, macro
expansion, CQL — lives behind those three methods rather than in the protocol, so the sources
that follow are additions rather than a rewrite.

Only the light half is importable from here. The connector and its HTTP client are imported by
the factory in :mod:`manicule.connectors.plugin`, so plugin discovery — which runs in every
process that starts — does not load an HTTP stack on a machine that never syncs anything.
"""

from __future__ import annotations

from manicule.connectors.config import (
    CONNECTOR_NAME,
    AuthMethod,
    ConfluenceConfig,
    Deployment,
    resolve_credentials,
)
from manicule.connectors.errors import (
    AttachmentTooLargeError,
    BodyUnavailableError,
    ConnectorError,
    CursorExpiredError,
    NotFoundError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
    SessionExpiredError,
    UntrustedLinkError,
)

__all__ = [
    "CONNECTOR_NAME",
    "AttachmentTooLargeError",
    "AuthMethod",
    "BodyUnavailableError",
    "ConfluenceConfig",
    "ConnectorError",
    "CursorExpiredError",
    "Deployment",
    "NotFoundError",
    "RateLimitedError",
    "RemoteError",
    "RequestTimeoutError",
    "SessionExpiredError",
    "UntrustedLinkError",
    "resolve_credentials",
]
