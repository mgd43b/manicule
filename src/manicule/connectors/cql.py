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

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

__all__ = [
    "CQL_TIMESTAMP",
    "content_query",
    "cql_timestamp",
    "is_page_id",
    "latest",
    "parse_when",
    "quote",
    "status_clause",
    "subtree_clause",
    "title_query",
]

CQL_TIMESTAMP = "%Y/%m/%d %H:%M"
"""The format CQL compares dates in. Minute granularity, which §2's overlap exists to absorb."""

_FORBIDDEN = frozenset({"\n", "\r", "\x00"})

_PAGE_ID = re.compile(r"[0-9]+")


def is_page_id(value: str) -> bool:
    """Whether ``value`` is something this module will put in a query unquoted.

    Confluence content ids are decimal integers on both Cloud and Server/Data Center, and
    :func:`subtree_clause` emits them as bare numbers rather than as quoted literals. That is
    not a shortcut: ``ancestor`` and ``id`` compare against a content id, and a bare number has
    no way to end a string literal and continue as query syntax. Refusing anything that is not
    digits at the point configuration is read therefore makes the injection hazard
    :func:`quote` exists for structurally impossible here, instead of escaped.
    """
    return bool(value) and _PAGE_ID.fullmatch(value) is not None


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


def status_clause(*, current_only: bool) -> tuple[str, ...]:
    """``("status = current",)`` or nothing, and the choice is the caller's to state.

    **Two deployments disagree about whether this field exists.** Cloud's search accepts
    ``status`` and needs it: reconciliation depends on a deleted page *not* being returned
    (``docs/connectors/confluence.md`` §3), and a query that quietly included trashed content
    would report every deleted page as still present, so deletion detection would run, succeed,
    and find nothing forever. The standard Data Center content-search resource **rejects** the
    field outright and returns current content by default, so the same clause is an HTTP 400
    there rather than a safeguard.

    **No default, and keyword-only.** A default would have to be one deployment's answer, and
    whichever one it was would be silently wrong for the other every time somebody added a call
    site without thinking about it — which is exactly how a query builder comes to have seven
    callers and six of them correct. Requiring the word makes a forgotten call site a type
    error at the point it is written rather than an HTTP 400 against somebody's live wiki.

    A tuple rather than a string so that "no clause" is an empty sequence to splice, not an
    empty string a caller has to remember to filter out of a join.
    """
    return ("status = current",) if current_only else ()


def subtree_clause(roots: Sequence[str], *, include_roots: bool) -> str:
    """The CQL that selects one or more page trees, and nothing else in the space.

    ``ancestor`` is Confluence's own descendant predicate: it matches content at any depth
    below the named page, and it does **not** match that page. So the root itself is selected
    by a second disjunct on ``id``, and ``include_roots`` decides whether that disjunct is
    written at all.

    Writing the flag into the query rather than filtering afterwards is what makes it
    debuggable: the scope of a run is the query it sent, and a page the source was never asked
    for arriving anyway is then a fact about the deployment rather than something the connector
    quietly absorbed.

    Args:
        roots: Configured root page ids. Each must satisfy :func:`is_page_id`.
        include_roots: Whether the root pages themselves are in scope.

    Raises:
        ValueError: ``roots`` is empty, or one of them is not a content id. Both are refusals
            rather than repairs: an empty tree clause would widen the query to the whole space,
            and a non-numeric id is either a title somebody pasted or an attempt to write
            query syntax.
    """
    if not roots:
        msg = "subtree_clause needs at least one root page id"
        raise ValueError(msg)
    for root in roots:
        if not is_page_id(root):
            msg = (
                f"{root!r} is not a Confluence page id. Ids are decimal numbers — the "
                f"`pageId` in a page's URL, or the number at the end of a short link."
            )
            raise ValueError(msg)
    listed = ", ".join(roots)
    descendants = f"ancestor = {roots[0]}" if len(roots) == 1 else f"ancestor in ({listed})"
    if not include_roots:
        return descendants
    themselves = f"id = {roots[0]}" if len(roots) == 1 else f"id in ({listed})"
    return f"({descendants} OR {themselves})"


def content_query(
    space: str,
    *,
    current_only: bool,
    types: Sequence[str] = ("page",),
    since: str | None = None,
    ordered: bool = True,
    subtree: str = "",
) -> str:
    """The CQL for one space's content, optionally only what changed since ``since``.

    Args:
        space: Space key.
        current_only: Whether to write ``status = current``. **Required, and keyword-only.**
            See :func:`status_clause` for why it has no default.
        types: Content types to enumerate, e.g. ``("page", "attachment")``.
        since: A :data:`CQL_TIMESTAMP`-formatted timestamp, or ``None`` for everything.
        ordered: Sort oldest first. Deterministic enumeration, so an interrupted run resumes
            over the same ground rather than a reshuffled corpus.
        subtree: A :func:`subtree_clause`, or ``""`` for the whole space. Narrowing happens
            **at the source**: a client-side filter over a space-wide enumeration returns the
            same documents while paying for the space, which is the cost this exists to avoid.
    """
    if not types:
        msg = "content_query needs at least one content type"
        raise ValueError(msg)
    kinds = ", ".join(types)
    clauses = [
        f"type in ({kinds})" if len(types) > 1 else f"type = {types[0]}",
        f"space = {quote(space)}",
        *status_clause(current_only=current_only),
    ]
    if subtree:
        clauses.append(subtree)
    if since is not None:
        clauses.append(f"lastmodified >= {quote(since)}")
    query = " AND ".join(clauses)
    return f"{query} order by lastmodified asc" if ordered else query


def title_query(space: str, title: str, *, current_only: bool) -> str:
    """The CQL that finds one page by its title, which is how an ``include`` names its target."""
    clauses = [
        "type = page",
        f"space = {quote(space)}",
        f"title = {quote(title)}",
        *status_clause(current_only=current_only),
    ]
    return " AND ".join(clauses)


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
