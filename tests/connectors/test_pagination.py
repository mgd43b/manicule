"""Cursor pagination, and the one character that silently breaks it.

The trap in ``docs/connectors/confluence.md`` §2: a search cursor contains ``+``, every
form-decoding query parser reads that as a space, and the re-encoded cursor is one the server
does not recognize. Nothing raises. Pagination restarts or stops, and some pages of a space
are simply never seen.

So the fixtures put a ``+`` in every cursor, and the synthetic instance answers an unrecognized
one with the *first* page rather than an error — which is what makes the real failure invisible.
A suite that walked two pages of a well-behaved cursor would certify nothing at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest

import manicule.connectors.client as client_module
from manicule.connectors import ConnectorError, CursorExpiredError, UntrustedLinkError
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.confluence import ConfluenceConnector
from manicule.connectors.pagination import NextPage, next_page, origin_of, split_query
from manicule.testing import closing
from tests.connectors.fake_confluence import CLOUD_BASE, FakeConfluence, FakePage
from tests.connectors.support import (
    Clock,
    client_for,
    cloud_config,
    connected,
    drain,
    ids,
)


def _instance(count: int = 5) -> FakeConfluence:
    return FakeConfluence(
        pages=[
            FakePage(id=str(index), title=f"Page {index}", space="ENG", when=f"2026-08-0{index}")
            for index in range(1, count + 1)
        ],
        page_size=2,
    )


async def test_discovery_preserves_an_empty_filtered_source_page_boundary() -> None:
    """A page with no usable rows is still a cursor-lifetime and durability boundary."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space/ENG"):
            return httpx.Response(200, json={"key": "ENG"})
        if request.url.path.endswith("/rest/api/content/search"):
            if request.url.params.get("cursor") == "page+2":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "usable",
                                "type": "page",
                                "title": "Public synthetic page",
                                "space": {"key": "ENG"},
                                "version": {
                                    "number": 1,
                                    "when": "2026-08-09T14:30:00+00:00",
                                },
                                "_links": {"webui": "/spaces/ENG/pages/usable"},
                            }
                        ],
                        "_links": {"base": CLOUD_BASE},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "results": [{"id": "filtered", "type": "blogpost"}],
                    "_links": {
                        "base": CLOUD_BASE,
                        "next": "/rest/api/content/search?cursor=page%2B2",
                    },
                },
            )
        raise AssertionError(f"unexpected synthetic request path {request.url.path}")

    config = cloud_config(spaces=["ENG"], include_attachments=False, page_size=250)
    client = ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0)
    connector = ConfluenceConnector(config, client)
    await connector.setup()
    try:
        batches = [tuple(batch) async for batch in connector.discover_batches(None)]
    finally:
        await connector.teardown()

    assert [ids(batch) for batch in batches] == [[], ["usable"]]
    assert connector.watermark is not None


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


async def test_cursor_history_io_is_included_in_the_cursor_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled disk ledger cannot open an unchecked expiry window before request two."""
    instance = _instance()
    clock = Clock()
    original = client_module.BoundedHistory.add

    def delayed_add(history: Any, *parts: str) -> bool:
        added = original(history, *parts)
        clock.advance(61.0)
        return added

    monkeypatch.setattr("manicule.connectors.client.BoundedHistory.add", delayed_add)
    connector = await connected(
        instance,
        cloud_config(
            base_url=instance.base_url,
            spaces=("ENG",),
            cursor_lifetime_seconds=60.0,
        ),
        clock=clock,
    )
    try:
        async with closing(connector.discover(None)) as stream:
            await anext(stream)
            await anext(stream)
            with pytest.raises(CursorExpiredError, match="cursor_lifetime_seconds"):
                await anext(stream)
    finally:
        await connector.teardown()

    searches = [
        request for request in instance.requests if request.url.path.endswith("/content/search")
    ]
    assert len(searches) == 1, "the expired second cursor must not reach the transport"


async def test_a_next_link_that_repeats_without_a_cursor_stops_the_walk() -> None:
    """The loop guard compares the whole request, not the cursor alone.

    A link that keeps pointing at itself with no cursor at all would slip past a cursor-only
    check and enumerate the same results forever, which reads as a very large space rather
    than as a fault.
    """
    payload = {
        "results": [{"id": "1", "type": "page", "title": "A"}],
        "_links": {"base": CLOUD_BASE, "next": "/rest/api/content/search?limit=1"},
    }

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=payload)

    config = cloud_config()
    client = ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0)
    await client.setup()
    try:
        with pytest.raises(ConnectorError, match="already followed"):
            await drain(client.paginate(client.url("/rest/api/content/search")))
    finally:
        await client.teardown()


async def test_a_request_off_the_configured_origin_is_refused_before_it_is_sent() -> None:
    """Not only the pagination link: most URLs this client is handed come from a response.

    The check lives at the one place requests are made, so a URL threaded in later — a
    download link, an endpoint added next year — cannot skip it.
    """
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    _, client = client_for(instance)
    await client.setup()
    try:
        with pytest.raises(UntrustedLinkError, match=r"collector\.example"):
            await client.get_json("https://collector.example/rest/api/space")
    finally:
        await client.teardown()

    assert instance.requests == [], "the refusal must happen before anything is sent"


# --- one reading of "the origin" ------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "https://WIKI.example.test/confluence",
        "https://Wiki.Example.Test/confluence",
        "https://wiki.example.test:8443/confluence",
        "https://wiki.example.test/confluence",
    ],
    ids=["upper-host", "mixed-host", "non-default-port", "already-canonical"],
)
def test_a_configured_origin_reads_the_same_way_the_link_check_reads_it(base_url: str) -> None:
    """A regression, and the bug it is for was a hard failure against a healthy instance.

    ``ConfluenceConfig.origin`` used to build its answer by splitting on ``://``, which neither
    lowercased the host nor dropped userinfo. Every caller compares it against ``origin_of`` — a
    link base a response declared, a ``webui`` path joined onto one — and ``origin_of`` does
    both. So a ``base_url`` with a capital letter in the host made
    ``origin_of(base) != config.origin`` true for a link that *was* same-origin, and the
    connector refused its own instance with "declared an untrusted link base".

    Parameterized over the shapes that used to disagree plus one that never did, because a test
    of only the broken spellings would pass against an implementation that had stopped
    normalizing altogether.
    """
    assert ConfluenceConfig(base_url=base_url).origin == origin_of(base_url)


def test_a_configured_origin_still_separates_instances_that_are_genuinely_different() -> None:
    """Agreeing with ``origin_of`` must not become agreeing with everything.

    The fix removed a normalizer; it did not loosen one. Scheme, host and port still decide, so
    a link that really is elsewhere is still elsewhere.
    """
    configured = ConfluenceConfig(base_url="https://wiki.example.test/confluence").origin

    assert configured != origin_of("http://wiki.example.test/confluence")
    assert configured != origin_of("https://wiki.example.test:8443/confluence")
    assert configured != origin_of("https://wiki.example.test.evil.test/confluence")
    assert configured != origin_of("https://evil.test/wiki.example.test")


def test_origin_of_is_total_even_for_a_malformed_authority() -> None:
    """`urlsplit` defers the port, and `.port` is where it refuses.

    A response-supplied `Link:` header naming `https://host:99999/` or a non-numeric port
    parses fine and then raises a bare `ValueError` out of `origin_of` — a function whose
    docstring promises a total answer, and whose callers handle refusal as the typed
    `UntrustedLinkError`. So a header from a server this connector does not control could
    escape the refusal path entirely.

    A malformed authority names no server this can vouch for, which is the same answer as a
    missing host: the empty origin every caller already treats as a refusal.
    """
    assert origin_of("https://host:99999/x") == "", "a port out of range"
    assert origin_of("https://host:notaport/x") == "", "a port that is not a number"
    assert origin_of("not a url") == ""

    # And a well-formed one still resolves, so "refuse everything" would not pass.
    assert origin_of("https://host:8443/x") == "https://host:8443"
    assert origin_of("https://HOST/x") == "https://host"
