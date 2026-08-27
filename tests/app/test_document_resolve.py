"""Reading a cached document by a handle that came from somewhere other than manicule.

``document_resolve`` exists so another local program can treat this installation as the cache
of a source it does not want to call itself: hand it a Confluence page id, or a URL off
somebody's clipboard, and get back the bytes that were fetched. Everything here is about the
two ways that goes wrong quietly.

**A handle that is not an identity can match more than one document, and the wrong answer looks
exactly like the right one.** A URI is display data — a source may change it, and nothing makes
it unique — so the URI path is tested for what it does with zero matches and with two, not only
with one. Returning the first row would pass a happy-path test forever and depend on storage
order in production.

**"No content" has two causes and they are different facts about the installation.** Retention
switched off is a setting somebody can change; bytes reclaimed by the sweep is the retention
policy working as designed. A single ``content: null`` would send an operator hunting for a
misconfiguration that is not there, so the two are asserted to say different things.

Nothing here reaches a network, which is the property that makes ``stale`` worth being careful
about: this installation cannot know what the source has done since it last synced, so the
field answers only the question the caller asked and is ``None`` when they asked nothing.
"""

from __future__ import annotations

import asyncio
import json
from base64 import b64decode
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from manicule.app.service import ApplicationService
from manicule.app.tenancy import CrossWorkspaceError
from manicule.config.settings import Settings
from manicule.core.errors import AmbiguousHandleError, UnknownEntityError
from manicule.core.ids import content_hash, document_id
from manicule.mcp.server import build_server
from tests.app.fakes import FakeBackend, LeakyStore, make_document

WORKSPACE = "default"
PAGE = "123456"
CANONICAL = "https://wiki.example.test/wiki/spaces/ENG/pages/123456/Retry-policy"
BODY = b"<p>The client retries three times, then gives up.</p>"
VERSION = "47"
"""Confluence's ``version.number`` for the fixture page — a change token, not a credential."""
"""A Confluence storage-format fragment. Invented here; no real page, no real host."""


def _backend(
    *,
    body: bytes | None = BODY,
    uri: str = CANONICAL,
    indexed_at: datetime | None = None,
    reclaimed: bool = False,
    workspace: str = WORKSPACE,
) -> FakeBackend:
    """One indexed Confluence page, with whatever retention state the case needs.

    ``reclaimed`` is the state a bare ``body=None`` cannot express: the document still points
    at its bytes and the blob store no longer holds them, which is what the sweep leaves behind
    and is a different fact from never having retained anything.
    """
    backend = FakeBackend(settings=Settings(workspace=workspace))
    document = make_document(workspace, source="confluence", source_id=PAGE, indexed_at=indexed_at)
    reference = content_hash(body) if body is not None else None
    document = document.model_copy(
        update={
            "uri": uri,
            "media_type": "application/xhtml+xml;profile=confluence-storage",
            "version_token": VERSION,
            "original_ref": reference if not reclaimed else content_hash(b"gone"),
        }
    )
    backend.store.add(document)
    if body is not None and not reclaimed:
        backend.retained_.data[content_hash(body)] = body
    return backend


def _service(backend: FakeBackend) -> ApplicationService:
    return ApplicationService(backend)


# --- the three handles ----------------------------------------------------------------------


async def test_a_page_id_and_its_connector_resolve_to_the_document_and_its_bytes() -> None:
    """The headline case: a caller holding Confluence's own id, not manicule's.

    ``resolved_by`` is asserted as well as the content, because the three handles are not
    equally trustworthy and a caller that asked by identity should be able to see that it got
    an identity match rather than a URI one.
    """
    resolved = await _service(_backend()).document_resolve(source="confluence", source_id=PAGE)

    assert resolved.resolved_by == "source_id"
    assert resolved.document.source_id == PAGE
    assert resolved.content == BODY.decode()
    assert resolved.encoding == "utf-8"
    assert resolved.byte_count == len(BODY)
    assert resolved.version_token == VERSION


async def test_the_page_id_path_derives_the_id_rather_than_searching_for_it() -> None:
    """``(workspace, source, source_id)`` *is* the document id, so this is arithmetic.

    Asserted by computing the id independently and checking the resolved document carries it.
    The property that matters is not speed: deriving the id puts the workspace inside the key,
    so this path cannot reach another tenant's row even if every ``WHERE`` clause in storage
    were deleted — the same argument ``tests/app/test_tenancy.py`` makes about ``belongs_to``.
    """
    resolved = await _service(_backend()).document_resolve(source="confluence", source_id=PAGE)

    assert resolved.document.id == document_id(WORKSPACE, "confluence", PAGE)


async def test_manicules_own_id_resolves_the_same_document() -> None:
    service = _service(_backend())
    known = document_id(WORKSPACE, "confluence", PAGE)

    resolved = await service.document_resolve(document_id=known)

    assert resolved.resolved_by == "document_id"
    assert resolved.document.id == known
    assert resolved.content == BODY.decode()


async def test_a_url_off_a_clipboard_resolves_when_it_matches_what_was_stored() -> None:
    resolved = await _service(_backend()).document_resolve(uri=CANONICAL)

    assert resolved.resolved_by == "uri"
    assert resolved.document.source_id == PAGE


@pytest.mark.parametrize(
    "pasted",
    [
        f"{CANONICAL}/",
        CANONICAL.replace("wiki.example.test", "WIKI.EXAMPLE.TEST"),
        CANONICAL.replace("https://", "HTTPS://"),
        f"{CANONICAL}#heading",
    ],
    ids=["trailing-slash", "host-case", "scheme-case", "fragment"],
)
async def test_a_url_still_resolves_across_differences_it_cannot_carry_meaning_in(
    pasted: str,
) -> None:
    """Scheme case, host case and a fragment cannot distinguish two documents, so they fold.

    Each of these is a form a browser will happily hand somebody. Refusing them would make the
    feature fail on the exact input it exists to accept, and folding them is safe because RFC
    3986 says none of the three is significant.
    """
    resolved = await _service(_backend()).document_resolve(uri=pasted)

    assert resolved.document.source_id == PAGE


async def test_the_path_is_never_folded_because_a_page_title_lives_in_it() -> None:
    """The normalization stops exactly where guessing would start.

    A Confluence URL ends in the page's title, so trimming path segments to find a match would
    resolve one page's URL to a different page that happens to share a prefix — silently, and
    with a plausible document coming back. A renamed page is a miss, and the message says to
    use the page id instead.
    """
    service = _service(_backend())
    renamed = CANONICAL.replace("Retry-policy", "Retry-and-backoff-policy")

    with pytest.raises(UnknownEntityError) as caught:
        await service.document_resolve(uri=renamed)

    assert "source_id" in str(caught.value), "the refusal does not say what to do instead"


# --- when a handle names more than one thing ------------------------------------------------


async def test_a_uri_matching_two_documents_is_refused_and_both_are_named() -> None:
    """The failure this returns a sequence to make possible.

    A live Confluence instance and a snapshot of the same wiki both carry the page's URL. There
    is no answer to "which one did you mean", so the caller is asked — and given the handles
    that *are* identities, which is what makes the refusal actionable rather than a dead end.
    """
    backend = _backend()
    mirrored = make_document(WORKSPACE, source="confluence-snapshot", source_id=PAGE)
    backend.store.add(mirrored.model_copy(update={"uri": CANONICAL}))

    with pytest.raises(AmbiguousHandleError) as caught:
        await _service(backend).document_resolve(uri=CANONICAL)

    message = str(caught.value)
    assert "confluence" in message
    assert "confluence-snapshot" in message, "the alternatives are not named"


async def test_the_same_page_id_under_two_connectors_stays_two_documents() -> None:
    """Which is why ``source`` is required rather than inferred from the one connector that has it.

    Both documents exist and neither is more correct; inferring would return whichever the
    corpus happened to hold, and the answer would change as unrelated connectors were added.
    """
    backend = _backend()
    mirrored = make_document(WORKSPACE, source="confluence-snapshot", source_id=PAGE)
    backend.store.add(mirrored.model_copy(update={"uri": "file:///export/123456"}))
    service = _service(backend)

    live = await service.document_resolve(source="confluence", source_id=PAGE)
    snapshot = await service.document_resolve(source="confluence-snapshot", source_id=PAGE)

    assert live.document.id != snapshot.document.id


# --- handles that do not name anything ------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"document_id": "abc", "uri": CANONICAL},
        {"source": "confluence"},
        {"source_id": PAGE},
    ],
    ids=["none", "two-handles", "source-alone", "source-id-alone"],
)
async def test_a_request_that_names_no_document_exactly_once_is_refused(
    arguments: dict[str, Any],
) -> None:
    """Half a handle and two handles are both refused, and neither reaches the store.

    Two handles matter more than they look: answering the first silently would hide a caller
    whose two identifiers disagree, which is a bug in the caller that this is the only place
    positioned to notice.
    """
    with pytest.raises(ValueError, match="document"):
        await _service(_backend()).document_resolve(**arguments)


async def test_a_page_that_is_absent_and_one_that_is_another_tenants_refuse_identically() -> None:
    """The membership oracle :class:`UnknownEntityError` exists to close, on this path.

    A different message for the two would confirm, from outside a workspace, that a page id
    exists inside one — which is the whole of what an attacker holding a key to workspace A
    wants to learn about workspace B. The messages are compared for equality rather than
    inspected for a forbidden word, because the leak is any difference at all.
    """
    backend = FakeBackend(settings=Settings(workspace="acme"))
    theirs = make_document("globex", source="confluence", source_id="777", title="Globex payroll")
    backend.store.add(theirs)
    service = _service(backend)

    with pytest.raises(UnknownEntityError) as absent:
        await service.document_resolve(source="confluence", source_id="777")
    with pytest.raises(UnknownEntityError) as never_existed:
        await service.document_resolve(source="confluence", source_id="778")

    assert str(absent.value).replace("777", "N") == str(never_existed.value).replace("778", "N")
    assert "Globex payroll" not in str(absent.value)


# --- the bytes, and the two ways there are none ---------------------------------------------


async def test_retention_switched_off_and_bytes_reclaimed_are_reported_differently() -> None:
    """Both leave ``content`` null, and an operator needs to act differently on each."""
    never = await _service(_backend(body=None)).document_resolve(
        source="confluence", source_id=PAGE
    )
    reclaimed = await _service(_backend(reclaimed=True)).document_resolve(
        source="confluence", source_id=PAGE
    )

    assert never.content is None
    assert reclaimed.content is None
    assert never.unavailable_reason != reclaimed.unavailable_reason
    assert "retain_source_bytes" in (never.unavailable_reason or "")
    assert "reclaimed" in (reclaimed.unavailable_reason or "")


async def test_bytes_that_are_not_text_arrive_base64_and_say_so() -> None:
    """A PDF attachment, encoded rather than refused and labeled rather than left to guess.

    The label is a measurement, not a repetition of ``media_type``: an attachment mislabeled
    ``text/plain`` upstream would otherwise arrive as mojibake that reads like content.
    """
    binary = b"%PDF-1.7\n\xde\xad\xbe\xef"
    resolved = await _service(_backend(body=binary)).document_resolve(
        source="confluence", source_id=PAGE
    )

    assert resolved.encoding == "base64"
    assert resolved.content is not None
    assert b64decode(resolved.content) == binary
    assert resolved.byte_count == len(binary), "byte_count counts the bytes, not the encoding"


async def test_content_false_answers_without_reading_the_blob_store() -> None:
    """ "Is my copy current" is a common question and does not need a megabyte to answer.

    Asserted by emptying the blob store: the metadata still comes back, which it could not if
    the body were being read and its absence reported.
    """
    backend = _backend()
    backend.retained_.data.clear()

    resolved = await _service(backend).document_resolve(
        source="confluence", source_id=PAGE, content=False
    )

    assert resolved.content is None
    assert resolved.unavailable_reason is None, "not asking for bytes is not a failure to hold them"
    assert resolved.version_token == VERSION


# --- freshness, reported and never claimed --------------------------------------------------


async def test_staleness_is_measured_only_against_a_max_age_the_caller_named() -> None:
    old = datetime.now(UTC) - timedelta(hours=6)
    service = _service(_backend(indexed_at=old))

    unasked = await service.document_resolve(source="confluence", source_id=PAGE)
    asked = await service.document_resolve(source="confluence", source_id=PAGE, max_age_s=3600)
    tolerant = await service.document_resolve(source="confluence", source_id=PAGE, max_age_s=86400)

    assert unasked.stale is None, "no max_age was given, so there is no question to answer"
    assert asked.stale is True
    assert tolerant.stale is False
    assert asked.age_seconds is not None
    assert asked.age_seconds > 3600


async def test_a_copy_that_never_indexed_reports_no_age_rather_than_zero() -> None:
    """``None`` because there is nothing to measure. Zero would read as "just synced"."""
    resolved = await _service(_backend(indexed_at=None)).document_resolve(
        source="confluence", source_id=PAGE, max_age_s=60
    )

    assert resolved.age_seconds is None
    assert resolved.stale is None, "a freshness verdict built out of a gap is worse than none"


async def test_a_copy_stamped_in_the_future_is_clock_skew_rather_than_a_negative_age() -> None:
    """Clamped to zero, and the verdict is computed from the same clamped number.

    ``indexed_at`` is written by this installation and read against this installation's clock,
    so the two can disagree across an NTP correction or a suspended laptop. A negative
    ``age_seconds`` on the wire would read as a defect in manicule rather than as the clock
    hiccup it is, and a verdict computed from a different number than the one reported would be
    unexplainable from the response alone.
    """
    ahead = datetime.now(UTC) + timedelta(hours=2)
    resolved = await _service(_backend(indexed_at=ahead)).document_resolve(
        source="confluence", source_id=PAGE, max_age_s=60
    )

    assert resolved.age_seconds == 0.0
    assert resolved.stale is False, "a copy newer than now is not older than any max_age"


async def test_a_stale_copy_is_still_returned_in_full() -> None:
    """``max_age_s`` reports; it does not withhold.

    Refusing would leave a caller who wanted the bytes anyway with no way to get them, and a
    caller who did not care with no reason to have asked.
    """
    resolved = await _service(
        _backend(indexed_at=datetime.now(UTC) - timedelta(days=30))
    ).document_resolve(source="confluence", source_id=PAGE, max_age_s=1)

    assert resolved.stale is True
    assert resolved.content == BODY.decode()


# --- tenancy --------------------------------------------------------------------------------


async def test_the_uri_lookup_refuses_another_workspaces_document() -> None:
    """The URI path is a second way into a document, so it needs the guard the first one has.

    Run against :class:`~tests.app.fakes.LeakyStore`, which matches on the URI alone across
    every workspace — a store written without the ``WHERE`` clause. A guard written only into
    ``get_document``'s caller would leave this open, and the leak would be invisible: a
    plausible document, correctly formatted, belonging to somebody else.
    """
    store = LeakyStore(workspace_id="acme")
    ours = make_document("acme", source="confluence", source_id="1")
    store.add(ours.model_copy(update={"uri": "https://wiki.example.test/ours"}))
    theirs = make_document("globex", source="confluence", source_id="2", title="Globex payroll")
    store.add(theirs.model_copy(update={"uri": "https://wiki.example.test/theirs"}))
    service = ApplicationService(FakeBackend(settings=Settings(workspace="acme"), store=store))

    mine = await service.document_resolve(uri="https://wiki.example.test/ours")
    assert mine.document.source_id == "1"

    with pytest.raises(CrossWorkspaceError) as caught:
        await service.document_resolve(uri="https://wiki.example.test/theirs")
    assert "Globex payroll" not in str(caught.value), (
        "the refusal leaked the title it was defending"
    )


# --- the two surfaces are one service -------------------------------------------------------


def test_http_and_mcp_return_byte_identical_envelopes() -> None:
    """The parity claim ``tests/app/test_surface_parity.py`` makes, for the two surfaces this
    operation is on.

    It is not in that file's ``PAIRS`` table because every row there also names a command line
    invocation and this operation deliberately has none — so the property is asserted here
    rather than left unasserted. Compared as serialized JSON, not as objects: two payloads that
    are equal in Python and serialize differently would still break a client.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only this test needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI off the other paths

    arguments = {"source": "confluence", "source_id": PAGE}

    # Through the real server's `call_tool` rather than through the service, which is the
    # difference between asserting parity and asserting that one function equals itself: the
    # tool wrapper is where an argument could be renamed, dropped or defaulted differently, and
    # only a call that crosses it would notice.
    async def over_mcp() -> Any:
        server = build_server(_service(_backend()))
        result = await server.call_tool("document_resolve", arguments)
        return result.structured_content

    from_tool = asyncio.run(over_mcp())

    with TestClient(build_app(_service(_backend())), client=("127.0.0.1", 41234)) as client:
        from_http = client.get("/api/v1/documents/resolve", params=arguments).json()

    assert json.dumps(from_http, sort_keys=True) == json.dumps(from_tool, sort_keys=True)


def test_the_resolve_route_is_not_swallowed_by_the_document_id_route() -> None:
    """Declaration order, asserted rather than trusted.

    ``/documents/{document_id}`` is registered in the same router. Were ``resolve`` declared
    after it, Starlette would match it as an id and answer 404 for a route that exists — the
    trap ``/documents/trash`` already carries a comment about.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only this test needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI off the other paths

    with TestClient(build_app(_service(_backend())), client=("127.0.0.1", 41234)) as client:
        response = client.get(
            "/api/v1/documents/resolve", params={"source": "confluence", "source_id": PAGE}
        )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["document"]["source_id"] == PAGE
