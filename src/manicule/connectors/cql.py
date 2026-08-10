"""Building CQL queries, and the timestamps a watermark sync compares against.

Pure text and datetime work, deliberately separated from the client that sends it: the whole
point of the watermark (``docs/connectors/confluence.md`` §2) is that a sync costs what
changed rather than the whole corpus, and the query is where that is either true or not.

Two things here are less obvious than they look.

**Every value is quoted and escaped.** A space key or a page title reaches CQL as a string
literal, and a title containing a quote would otherwise end the literal and continue as query
syntax — which against a search endpoint is not a crash but a different, wider search. The
same hazard as SQL injection, with a result that looks like results.

**Timestamps keep the offset the source reported.** CQL's ``lastmodified`` is evaluated in the
Confluence instance's own timezone, which no API reliably states. Rather than guess it, the
connector reads ``version.when`` — which arrives as ISO-8601 *with* the instance's offset —
and formats the comparison in that same offset. Nothing is converted, so nothing depends on a
zone anybody had to know.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

__all__ = [
    "CQL_TIMESTAMP",
    "content_query",
    "cql_timestamp",
    "latest",
    "parse_when",
    "quote",
    "title_query",
]

CQL_TIMESTAMP = "%Y/%m/%d %H:%M"
"""The format CQL compares dates in. Minute granularity, which §2's overlap exists to absorb."""

_FORBIDDEN = frozenset({"\n", "\r", "\x00"})


def quote(value: str) -> str:
    """A CQL string literal for ``value``.

    Raises:
        ValueError: ``value`` contains a newline or a NUL. Both would terminate or truncate
            the literal in ways no escaping fixes, and a space key or page title containing
            one is not something to paper over — the query built from it would search for
            something other than what was asked.
    """
    if any(character in _FORBIDDEN for character in value):
        msg = (
            f"cannot build a CQL literal from {value!r}: it contains a line break or a NUL, "
            f"which would end the literal and continue as query syntax"
        )
        raise ValueError(msg)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def content_query(
    space: str,
    *,
    types: Sequence[str] = ("page",),
    since: str | None = None,
    ordered: bool = True,
) -> str:
    """The CQL for one space's content, optionally only what changed since ``since``.

    ``status = current`` is explicit rather than assumed. Reconciliation depends on a deleted
    page *not* being returned (``docs/connectors/confluence.md`` §3); a query that quietly
    included trashed content would report every deleted page as still present, and deletion
    detection would run, succeed, and find nothing forever.

    Args:
        space: Space key.
        types: Content types to enumerate, e.g. ``("page", "attachment")``.
        since: A :data:`CQL_TIMESTAMP`-formatted timestamp, or ``None`` for everything.
        ordered: Sort oldest first. Deterministic enumeration, so an interrupted run resumes
            over the same ground rather than a reshuffled corpus.
    """
    if not types:
        msg = "content_query needs at least one content type"
        raise ValueError(msg)
    kinds = ", ".join(types)
    clauses = [
        f"type in ({kinds})" if len(types) > 1 else f"type = {types[0]}",
        f"space = {quote(space)}",
        "status = current",
    ]
    if since is not None:
        clauses.append(f"lastmodified >= {quote(since)}")
    query = " AND ".join(clauses)
    return f"{query} order by lastmodified asc" if ordered else query


def title_query(space: str, title: str) -> str:
    """The CQL that finds one page by its title, which is how an ``include`` names its target."""
    return f"type = page AND space = {quote(space)} AND title = {quote(title)} AND status = current"


def parse_when(value: object) -> datetime | None:
    """The timestamp in a ``version.when`` field, with the offset the source reported.

    ``None`` for anything that is not a parseable ISO-8601 instant, because a watermark built
    from a misread timestamp is worse than no watermark: it advances past content it never saw.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def latest(candidates: Iterable[datetime | None]) -> datetime | None:
    """The newest of some timestamps, ignoring the ones that were not readable."""
    known = [candidate for candidate in candidates if candidate is not None]
    return max(known) if known else None


def cql_timestamp(when: datetime, overlap: timedelta = timedelta()) -> str:
    """``when``, minus an overlap, in the format CQL compares dates in.

    The offset is left exactly as it arrived. Converting to UTC here would compare a UTC
    instant against a field the instance evaluates in local time, which is correct for as long
    as the instance happens to be in UTC and silently skips or repeats an hour's content
    otherwise.
    """
    return (when - overlap).strftime(CQL_TIMESTAMP)
