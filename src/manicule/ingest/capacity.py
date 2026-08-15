"""Safe vocabulary for durable-ingest capacity decisions.

Capacity diagnostics cross every operational surface and are routinely copied into logs and
support requests. They therefore carry only a closed resource name and aggregate integers.
There is deliberately nowhere to attach a document title, source URL, opaque fetch reference,
credential, cookie, query parameter or arbitrary exception message.

This module does not perform storage admission. The durable journal owns the transaction that
reserves capacity; this vocabulary lets that implementation describe its decision without
coupling configuration or result surfaces to the journal schema.
"""

from __future__ import annotations

import errno
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import TYPE_CHECKING, cast

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any


class CapacityResource(StrEnum):
    """The bounded durable-ingest resources an operator may act on."""

    JOURNAL_RECORDS = "journal_records"
    JOURNAL_METADATA_BYTES = "journal_metadata_bytes"
    ACQUIRED_BLOB_BACKLOG_BYTES = "acquired_blob_backlog_bytes"
    DISK_HEADROOM_BYTES = "disk_headroom_bytes"


def _require_resource(value: object) -> None:
    if not isinstance(value, CapacityResource):
        raise TypeError("resource must be a CapacityResource")


def _require_integer(name: str, value: object) -> None:
    # Exact built-in integers only. An ``int`` subclass controls its own formatting and repr,
    # which would let source text escape when an aggregate is rendered or serialized.
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")


@dataclass(frozen=True, slots=True)
class CapacityDiagnostic:
    """An aggregate capacity refusal, safe for persistence and public result surfaces.

    ``used`` is the amount already reserved, ``requested`` is the pending reservation and
    ``limit`` is the configured upper bound. For disk headroom, ``used`` is space unavailable
    to new writes and ``limit`` is the largest usable amount after preserving the configured
    free-space floor. Callers calculate those figures from one filesystem snapshot inside the
    same admission decision.
    """

    resource: CapacityResource
    limit: int
    used: int
    requested: int

    def __post_init__(self) -> None:
        """Validate without retaining rejected inputs in a structured error object."""
        self._validate()

    def _validate(self) -> None:
        """Enforce the closed scalar types at every output boundary."""
        _require_resource(self.resource)
        _require_integer("limit", self.limit)
        _require_integer("used", self.used)
        _require_integer("requested", self.requested)
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.used < 0:
            raise ValueError("used must be non-negative")
        if self.requested < 1:
            raise ValueError("requested must be positive")
        if self.used + self.requested <= self.limit:
            raise ValueError("a capacity refusal requires used + requested to exceed limit")

    @property
    def over_by(self) -> int:
        """How much the requested reservation exceeds the configured bound."""
        return self.used + self.requested - self.limit

    def as_metadata(self) -> dict[str, str | int]:
        """The exact aggregate payload allowed into persistence and public surfaces.

        Written field by field rather than through dataclass reflection: a subclass may add
        fields, and reflection would faithfully serialize the source-shaped data this boundary
        exists to exclude.
        """
        self._validate()
        return {
            "resource": self.resource.value,
            "limit": self.limit,
            "used": self.used,
            "requested": self.requested,
        }


class CapacityRefusedError(ManiculeError):
    """A durable-ingest reservation was refused before acknowledgment.

    The message is derived solely from the closed aggregate diagnostic. Accepting no free-form
    detail is intentional: source URLs and credentials most often leak through an exception
    message somebody believed was merely diagnostic.
    """

    def __init__(self, diagnostic: CapacityDiagnostic) -> None:
        # Python value objects are subclassable, and a subclass may declare source-shaped fields
        # even though this base model forbids extras. Rebuild the boundary value field by
        # field instead of retaining the supplied object: no subclass state, custom dump or
        # later-added attribute can cross into persistence, a result envelope or a log.
        self.diagnostic = CapacityDiagnostic(
            resource=diagnostic.resource,
            limit=diagnostic.limit,
            used=diagnostic.used,
            requested=diagnostic.requested,
        )
        safe = self.diagnostic
        super().__init__(
            f"{safe.resource.value} capacity refused: "
            f"used={safe.used}, requested={safe.requested}, "
            f"limit={safe.limit}, over_by={safe.over_by}"
        )


def require_disk_headroom(*, free: int, requested: int, minimum: int) -> None:
    """Refuse a write using only filesystem aggregates."""
    usable = max(0, free - minimum)
    if requested > usable:
        raise CapacityRefusedError(
            CapacityDiagnostic(
                resource=CapacityResource.DISK_HEADROOM_BYTES,
                limit=usable,
                used=0,
                requested=max(1, requested),
            )
        )


def require_blob_backlog_capacity(*, used: int, requested: int, limit: int) -> None:
    """Refuse physical blob admission using only aggregate stored-byte counts."""
    if used + requested > limit:
        raise CapacityRefusedError(
            CapacityDiagnostic(
                resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
                limit=limit,
                used=used,
                requested=max(1, requested),
            )
        )


def storage_capacity_error(error: BaseException) -> bool:
    """Whether an exception chain is an aggregate-only disk-capacity failure."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in {errno.ENOSPC, errno.EDQUOT}:
            return True
        if getattr(current, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
            return True
        for related in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


def translate_storage_capacity_errors[F: Callable[..., Coroutine[Any, Any, Any]]](
    operation: F,
) -> F:
    """Map ENOSPC/EDQUOT/SQLITE_FULL without retaining their path- or SQL-shaped errors."""

    @wraps(operation)
    async def translated(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        diagnostic: CapacityDiagnostic | None = None
        try:
            return await operation(*args, **kwargs)
        except CapacityRefusedError:
            raise
        except Exception as error:
            if not storage_capacity_error(error):
                raise
            diagnostic = CapacityDiagnostic(
                resource=CapacityResource.DISK_HEADROOM_BYTES,
                limit=0,
                used=0,
                requested=1,
            )
        # Outside the source exception handler on purpose. ``raise ... from None`` suppresses
        # display but still retains the handled exception in ``__context__``; raising here
        # leaves neither a cause nor a context carrying its path, SQL or source-shaped detail.
        raise CapacityRefusedError(diagnostic) from None

    return cast("F", translated)


__all__ = [
    "CapacityDiagnostic",
    "CapacityRefusedError",
    "CapacityResource",
    "require_blob_backlog_capacity",
    "require_disk_headroom",
    "storage_capacity_error",
    "translate_storage_capacity_errors",
]
