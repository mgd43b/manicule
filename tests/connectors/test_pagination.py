"""Cursor pagination, and the one character that silently breaks it.

The trap in ``docs/connectors/confluence.md`` §2: a search cursor contains ``+``, every
form-decoding query parser reads that as a space, and the re-encoded cursor is one the server
does not recognise. Nothing raises. Pagination restarts or stops, and some pages of a space
are simply never seen.

So the fixtures put a ``+`` in every cursor, and the synthetic instance answers an unrecognised
one with the *first* page rather than an error — which is what makes the real failure invisible.
A suite that walked two pages of a well-behaved cursor would certify nothing at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast
from urllib.parse import parse_qsl, urlsplit

import pytest

from manicule.connectors import ConnectorError, CursorExpiredError, UntrustedLinkError
from manicule.connectors.pagination import NextPage, next_page, split_query
from manicule.testing import closing
from tests.connectors.fake_confluence import FakeConfluence, FakePage
from tests.connectors.support import Clock, cloud_config, connected, drain, ids


def _instance(count: int = 5) -> FakeConfluence:
    return FakeConfluence(
        pages=[
            FakePage(id=str(index), title=f"Page {index}", space="ENG", when=f"2026-08-0{index}")
            for index in range(1, count + 1)
        ],
        page_size=2,
    )


def test_a_cursor_keeps_its_plus_when_a_link_is_split() -> None:
    """The whole bug in one assertion.

    ``parse_qsl`` applies form-encoding rules, where ``+`` means space. A Confluence cursor is
    not form data, and the space it becomes is a cursor the server has never issued.
    """
    query = "cql=type%3Dpage&cursor=abc+def/ghi==&limit=100"

    assert dict(split_query(query))["cursor"] == "abc+def/ghi=="
    assert dict(parse_qsl(query))["cursor"] == "abc def/ghi=="


def test_everything_that_is_not_a_cursor_still_decodes_as_form_data() -> None:
    """The exception is one parameter wide, and it has to be.

    Confluence writes the rest of the link form-encoded, so a ``cql`` whose spaces arrived as
    ``+`` must decode back to spaces. Treating those as literal would send the source a query
    nobody wrote — the mirror image of the cursor bug, and just as quiet.
    """
    query = "cql=type%3Dpage+AND+space+%3D+%22ENG%22&cursor=a+b"
    decoded = dict(split_query(query))

    assert decoded["cql"] == 'type=page AND space = "ENG"'
    assert decoded["cursor"] == "a+b"


def test_a_parameter_without_a_value_survives_being_split() -> None:
    """A flag-shaped parameter dropped here would quietly change the next request's meaning."""
    assert split_query("expand&limit=5") == [("expand", ""), ("limit", "5")]


async def test_every_page_of_a_space_is_seen_exactly_once() -> None:
    """Five pages at a page size of two: three requests, five results, no repeats.

    The count is what matters. A cursor that stops working part-way through does not fail — it
    returns a shorter, entirely plausible corpus.
    """
    instance = _instance()
    connector = await connected(instance)
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == ["1", "2", "3", "4", "5"]


async def test_the_cursor_goes_back_percent_encoded() -> None:
    """``+`` must reach the wire as ``%2B``. Sent raw, the server reads it as a space."""
    instance = _instance()
    connector = await connected(instance)
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    sent = [query for query in instance.raw_queries() if "cursor=" in query]
    assert sent, "no page after the first was requested, so pagination was never exercised"
    assert all("cursor=cur%2B" in query for query in sent), sent
    assert not any("cursor=cur+" in query for query in sent)


async def test_form_decoding_the_cursor_stops_the_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proof that the check above is load-bearing rather than decorative.

    Swapping in the obvious implementation — split the link's query with ``parse_qsl`` — makes
    this exact fixture fail, because the corrupted cursor is answered with the first page again
    and the enumeration goes in a circle. Without the ``+`` in the fixture, this passes.
    """

    def naive(payload: Mapping[str, object], *, base_url: str) -> NextPage | None:
        links = cast("Mapping[str, object]", payload.get("_links") or {})
        link = links.get("next")
        if not isinstance(link, str):
            return None
        declared = links.get("base")
        base = declared if isinstance(declared, str) and declared else base_url
        split = urlsplit(base.rstrip("/") + link)
        return NextPage(
            url=f"{split.scheme}://{split.netloc}{split.path}",
            params=tuple(parse_qsl(split.query)),
        )

    monkeypatch.setattr("manicule.connectors.client.next_page", naive)

    instance = _instance()
    connector = await connected(instance)
    try:
        with pytest.raises(ConnectorError, match="already followed"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_link_to_another_host_is_not_followed() -> None:
    """``_links.base`` decides where the next request goes, and it comes from the response.

    Following it wherever it points would send this account's Confluence credentials to
    whatever host a response named.
    """
    payload: dict[str, object] = {
        "results": [],
        "_links": {"base": "https://elsewhere.example", "next": "/rest/api/content/search?x=1"},
    }
    with pytest.raises(UntrustedLinkError, match=r"elsewhere\.example"):
        next_page(payload, base_url="https://example.atlassian.net/wiki")


def test_a_context_path_survives_link_resolution() -> None:
    """Data Center is commonly served from ``/confluence``, and ``next`` does not repeat it.

    RFC 3986 resolution of a root-absolute reference discards the context path, producing a URL
    that 404s on every deployment that is not at a domain root.
    """
    payload: dict[str, object] = {
        "_links": {
            "base": "https://wiki.example.com/confluence",
            "next": "/rest/api/content/search?cursor=a%2Bb",
        }
    }
    following = next_page(payload, base_url="https://wiki.example.com/confluence")

    assert following is not None
    assert following.url == "https://wiki.example.com/confluence/rest/api/content/search"
    assert following.cursor == "a+b"


async def test_a_cursor_held_longer_than_its_lifetime_is_refused() -> None:
    """A stalled consumer must not resume onto a cursor the server has forgotten.

    An expired cursor can be answered with a fresh first page rather than an error, which
    enumerates the start of a space twice and its end never. Refusing before the request is
    sent turns that into a run that fails and is re-run against an unadvanced watermark.
    """
    instance = _instance()
    clock = Clock()
    connector = await connected(
        instance,
        cloud_config(base_url=instance.base_url, cursor_lifetime_seconds=60.0),
        clock=clock,
    )
    try:
        async with closing(connector.discover(None)) as stream:
            await anext(stream)
            await anext(stream)
            clock.advance(61.0)
            with pytest.raises(CursorExpiredError, match="cursor_lifetime_seconds"):
                await anext(stream)
    finally:
        await connector.teardown()
