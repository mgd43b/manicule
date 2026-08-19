"""Following ``_links.next``, and the one character that breaks it.

Search pagination is cursor-based; ``start`` no longer works reliably
(``docs/connectors/confluence.md`` §2). So a client reads the next page's link out of the
response and sends it back — and that is where the trap is.

**A cursor contains ``+``, and ``+`` in a query string means a space.** Every general-purpose
query parser in the standard library — :func:`urllib.parse.parse_qsl`,
:func:`urllib.parse.unquote_plus`, ``httpx.URL.params`` — decodes ``a+b`` to ``a b``, because
for form-encoded data that is exactly right. Re-encoding then sends a space where the cursor
had a plus, the server answers with a cursor it does not recognize, and the failure is not an
error: pagination stops, or restarts, part-way through a sync. Some pages are simply never
seen, and nothing anywhere says so.

The fix is one line and it is the whole reason this module exists: decode percent-escapes and
leave ``+`` alone (:func:`split_query`), then let the HTTP client percent-encode it back to
``%2B`` on the way out. A test that walks two pages of a well-behaved cursor certifies nothing
about this — the bug only appears when the cursor contains a ``+``, which is why the fixtures
in ``tests/connectors`` put one there.

Nothing here imports an HTTP client. It is string work, and string work is where this went
wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import cast
from urllib.parse import unquote, unquote_plus, urlsplit

from manicule.connectors.errors import UntrustedLinkError

__all__ = ["LITERAL_PLUS", "NextPage", "next_page", "origin_of", "split_query"]

LITERAL_PLUS = frozenset({"cursor"})
"""Parameters whose ``+`` is data rather than a space.

Exactly one, and narrowly so. Form-encoding rules are right for everything else in a
Confluence link: a ``cql`` parameter whose spaces arrived as ``+`` must decode back to spaces,
and treating *those* as literal would send a query the source reads differently. A cursor is
opaque base64-ish text where ``+`` is a character of the value and a space is impossible, so
the two cases do not overlap and the exception can be this small.
"""


def split_query(
    query: str, *, literal_plus: AbstractSet[str] = LITERAL_PLUS
) -> list[tuple[str, str]]:
    """Decode a raw query string into pairs, keeping ``+`` intact where it is data.

    :func:`urllib.parse.parse_qsl` applies form-encoding rules to everything, where ``+`` is a
    space. A Confluence search cursor contains literal ``+`` characters, so those rules corrupt
    it — and the corruption is silent, because the server answers an unrecognized cursor with
    results rather than an error. Percent-escapes are decoded either way; those are unambiguous.

    A pair with no ``=`` keeps an empty value rather than being dropped, so a flag-shaped
    parameter survives the round trip instead of quietly disappearing from the next request.
    """
    pairs: list[tuple[str, str]] = []
    for part in query.split("&"):
        if not part:
            continue
        raw_key, separator, raw_value = part.partition("=")
        key = unquote_plus(raw_key)
        if not separator:
            pairs.append((key, ""))
            continue
        decode = unquote if key in literal_plus else unquote_plus
        pairs.append((key, decode(raw_value)))
    return pairs


def origin_of(url: str) -> str:
    """Scheme, host and port of ``url``, lowercased. Empty when it has no host.

    **An IPv6 literal gets its brackets back**, because ``urlsplit`` strips them and the result
    is not merely ugly — it is ambiguous. ``https://[::1]:8443`` and ``https://[::1:8443]`` are
    different servers, one a host with a port and one a host without, and both render as
    ``https://::1:8443`` once the brackets are gone. Two origins that are not equal comparing
    equal is a same-origin check passing for somewhere else, and — since
    :func:`~manicule.connectors.sessions.authority_key` is built on this — one instance's held
    session being offered to another.
    """
    parsed = urlsplit(url)
    if not parsed.hostname:
        return ""
    host = parsed.hostname.lower()
    # A colon in a host is only ever an IPv6 literal; `urlsplit` has already taken the port off.
    host = f"[{host}]" if ":" in host else host
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme.lower()}://{host}{port}"


@dataclass(frozen=True, slots=True)
class NextPage:
    """Where the next page of results is, ready to be requested."""

    url: str
    """Absolute, with no query string of its own."""

    params: tuple[tuple[str, str], ...]
    """Decoded parameters. The client percent-encodes these, turning ``+`` back into ``%2B``."""

    @property
    def cursor(self) -> str:
        """The cursor this page is addressed by, or ``""`` when the link carries none."""
        return next((value for key, value in self.params if key == "cursor"), "")


def next_page(payload: Mapping[str, object], *, base_url: str) -> NextPage | None:
    """The next page's request, or ``None`` when the enumeration is complete.

    ``_links.next`` is relative to ``_links.base``, and the two are **concatenated** rather
    than joined by URL resolution: an instance served from a context path has that path in
    ``base`` and not in ``next``, and RFC 3986 resolution of a root-absolute reference would
    discard it — producing a URL that 404s on every deployment that is not at a domain root.

    Args:
        payload: A decoded search response.
        base_url: The configured site root. Used when the response carries no ``_links.base``,
            and always used to check the origin of the resolved link.

    Raises:
        UntrustedLinkError: The resolved link points at a different origin than the configured
            one. Following it would send the sync user's credentials wherever a response asked
            them to.
    """
    links = _mapping(payload.get("_links"))
    link = links.get("next")
    if not isinstance(link, str) or not link.strip():
        return None

    declared = links.get("base")
    base = declared.strip() if isinstance(declared, str) and declared.strip() else base_url
    resolved = _resolve(base, link.strip())

    expected = origin_of(base_url)
    found = origin_of(resolved)
    if found != expected:
        msg = (
            f"a paginated response asked the sync to continue at {found or resolved!r}, but "
            f"this connector is configured for {expected!r}. Following it would send this "
            f"account's Confluence credentials to another host, so the enumeration stops here "
            f"rather than completing against somewhere else."
        )
        raise UntrustedLinkError(msg)

    split = urlsplit(resolved)
    without_query = f"{split.scheme}://{split.netloc}{split.path}"
    return NextPage(url=without_query, params=tuple(split_query(split.query)))


def _mapping(value: object) -> Mapping[str, object]:
    """Narrow a decoded JSON value to an object. JSON keys are strings by construction."""
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _resolve(base: str, link: str) -> str:
    """``link`` against ``base``, keeping any context path in ``base``."""
    if origin_of(link):
        return link
    if link.startswith("/"):
        return base.rstrip("/") + link
    return f"{base.rstrip('/')}/{link}"
