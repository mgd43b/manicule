"""Column types with the conversions SQLite does not do.

SQLite has no date type and no boolean type. Everything is text, integer, real or blob, and
the interesting failures come from two writers disagreeing about which representation they
meant. These types make that disagreement impossible by giving each concept exactly one
encoding and one place that applies it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import override

from sqlalchemy import DateTime, Dialect, TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that is always timezone-aware UTC, in and out.

    **Timestamps are set in Python and never by a SQL default.** SQLite compares text
    lexicographically, and ``datetime('now')`` renders ``2026-08-10 12:00:00`` while an
    application writing ISO-8601 renders ``2026-08-10T12:00:00.123Z``. Space sorts before
    ``T``, so a column written by both orders every SQL-defaulted row ahead of every
    application-written one regardless of when either happened. One writer, one format, and
    ``ORDER BY created_at`` means what it says.

    Naive values are rejected rather than assumed to be UTC. A naive timestamp is missing
    information, and guessing which timezone it came from is how a corpus acquires an hour of
    skew nobody can find later.
    """

    impl = DateTime
    cache_ok = True

    @override
    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            msg = (
                "naive datetime rejected: pass an aware UTC value. A timestamp without a "
                "zone cannot be compared with one that has one."
            )
            raise ValueError(msg)
        return value.astimezone(UTC).replace(tzinfo=None)

    @override
    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


def utcnow() -> datetime:
    """Now, timezone-aware, in UTC. The only clock the storage layer reads."""
    return datetime.now(UTC)


def next_observation(previous: datetime | None, observed: datetime) -> datetime:
    """Turn a wall-clock observation into a strictly advancing source revision.

    Wall clocks can repeat or move backward. Reconciliation proposals compare this value, so
    every path that proves a document still exists must advance it even within one clock tick.
    """
    if observed.tzinfo is None:
        msg = "source observation time must be timezone-aware"
        raise ValueError(msg)
    if previous is not None and observed <= previous:
        return previous + timedelta(microseconds=1)
    return observed


__all__ = ["UtcDateTime", "next_observation", "utcnow"]
