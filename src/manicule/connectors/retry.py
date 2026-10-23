"""The retry arithmetic every HTTP client in this package shares.

Two clients retry against a source that can say "slow down": the Confluence client and the Gmail
client. What a pause *means* differs (Confluence steers by it, Gmail paces ahead of it), but how
long a response asks for and how long to wait when it does not say are the same questions, and
the second copy of an answer is the one that drifts.

**Backoff is deterministic and has no jitter.** Jitter keeps a fleet of clients from retrying in
lockstep; one self-hosted index syncing one source is not a fleet, and a deterministic delay is
one a test can assert on.

The crawler's transport keeps its own, stricter parser on purpose: it refuses a malformed
header outright, where these fall back to :func:`backoff`.

**Nothing here imports an HTTP client**, so importing it costs no HTTP stack.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

__all__ = ["BACKOFF_BASE_SECONDS", "backoff", "retry_after"]

BACKOFF_BASE_SECONDS = 0.5
"""First delay after a retryable failure. Doubles per attempt, capped by configuration."""


def backoff(attempt: int) -> float:
    """Exponential, from :data:`BACKOFF_BASE_SECONDS`, doubling per attempt."""
    return BACKOFF_BASE_SECONDS * (2**attempt)


def retry_after(headers: Mapping[str, str]) -> float | None:
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
        delay = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delay = (when - datetime.now(tz=UTC)).total_seconds()
    # ``nan`` compares false with everything, so it would pass for a zero-second wait and retry
    # straight into the limit; a header that is not a number of seconds is no instruction.
    return max(0.0, delay) if math.isfinite(delay) else None
