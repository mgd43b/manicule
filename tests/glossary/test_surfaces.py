"""``explicit_definition`` as a public contract, asserted on every surface that publishes it.

The classification itself is retrieval's, and ``tests/glossary/test_retrieval.py`` is where its
three conditions are argued with. What is asserted here is the other half: that the boolean
reaches a caller *unchanged* — on the search payload, the answer payload, the streamed final
frame, the MCP tool, the HTTP route and the terminal — and that the number beside it is what it
was before the field existed.

**Every case runs the shipped retriever over the synthetic corpus**, wired into the real
:class:`~manicule.app.service.ApplicationService` through the same fakes the rest of the
application suite uses for the parts this feature does not touch. That is the point of the file.
A fake retriever handed a ``Confidence`` with the flag already set would prove the field is
copied and nothing else, and every ``false`` case below would pass by having been told the
answer. Here the states are produced by putting the retrieval genuinely in them: an ordinary use
of a defined token, a term two documents disagree about, a defining passage a filter refused, a
defining passage the context budget dropped, and a question the router answered without looking
at the corpus at all.

**Everything is async, and that is a constraint rather than a style.** The store is a migrated
SQLite database whose connections belong to the loop that opened them, so a synchronous test
calling ``asyncio.run`` around a fixture built in another loop fails inside the driver rather
than on anything this file is about. The MCP column therefore awaits the tool manager directly
and the HTTP column drives the production application over an ASGI transport in this loop.

The corpus is synthetic and invented for these tests. Nothing in it names a real organisation,
system or document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.app.results import AnswerResultPayload, SearchResult
from manicule.app.service import ApplicationService
from manicule.cli.render import EXPLICIT_DEFINITION, console, render_answer, render_search
from manicule.config.settings import RouterSettings
from manicule.core.content import BlockKind
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.core.retrieval import Filter, Query
from manicule.ingest.glossary import detect_entries
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.confidence import DEFINITION_CITED, NOTHING_RESEMBLES
from manicule.retrieval.router import QueryRouter
from manicule.retrieval.tokens import ContextTokenCounter
from manicule.retrieval.utility import handlers_for
from tests.app.fakes import FakeBackend
from tests.evaluation.fakes import BagOfWordsEmbedder
from tests.glossary import corpus, system
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.app.ports import DocumentSurface, Retrieving
    from manicule.core.content import Chunk, Document
    from manicule.retrieval.retriever import RetrievalResult, Retriever
    from manicule.storage.docstore import SqliteDocStore

LIMIT = 10

LOCATES_A_DEFINITION = ("document_id", "chunk_id", "title", "uri")
"""The four fields requirement 9 says an explicit definition must resolve to.

Named once rather than written out per assertion, so a payload that stopped reporting one of
them fails in every case that reads a definition rather than in whichever one remembered it.
"""

READS_AFTER_RETRIEVAL = 2
"""How many times ``search`` reads the document store once retrieval has returned.

The passage check and the provenance lookup, in that order. Asserted rather than assumed by
the case that puts a soft delete between them: if the sequence ever changes, that case is
simulating something else and should say so rather than keep passing.
"""

MINIMUM_CONTEXT_TOKENS = 256
"""The smallest context budget ``ProfileConfig`` will accept.

Named because the budget case below has to work *within* it rather than choose a convenient
number: the floor is a validation rule, so a test that wanted a budget of 90 tokens would be
testing a configuration manicule refuses to run.
"""

ORDINARY_USE_OF_A_DEFINED_TERM = "raise a ticket in NOVA today"
"""A query that names a defined term and does not ask what it means.

The same shape ``test_retrieval.py`` uses for the narrowest version of the rule, repeated here
because the surfaces have to report ``false`` for it — and a surface suite that only ever saw
the ``true`` case would pass just as happily on a field wired to "a glossary entry fired".
"""

TWO_TERMS_ONE_QUESTION = f"raise a ticket in {corpus.DESCRIBED_ACRONYM} — what is {corpus.ACRONYM}?"
"""Two defined terms, and only the second is asked about.

Written for the context-budget case, and it has to name two terms. A promoted definition leads
the ranking, so with one term the defining passage is at rank 1 — and rank 1 is the one passage
assembly will not drop: if it does not fit, assembly raises rather than returning a context
without it. With two, the passage for the term the query merely *uses* leads, and the passage
for the term it *asks about* is the one a budget can push out. That is the state requirement 5
names, and it is not reachable with a single term.
"""


MIRRORED_ACRONYM = "ZEBU"
MIRRORED_EXPANSION = "Zonal Event Buffer Unit"
MIRRORED_ENTRY = f"{MIRRORED_ACRONYM} — {MIRRORED_EXPANSION}, a queue for deferred events."
MIRRORED_QUERY = f"What is {MIRRORED_ACRONYM}?"
MIRRORED_TITLE = "Deferred event handling"
MIRRORED_URI = "https://docs.example.test/handbook/deferred-events"
MIRRORED_SOURCE_ID = "page-84213"
MIRRORED_VERSION = "7"
"""A term defined on a page that carries an authoritative source record.

Invented here rather than added to ``corpus.py``: every cosine that file documents is measured
against its two glossary pages, and a third term on a third page would change what those
numbers describe. It is a whole extra document, so it changes nothing about them.
"""


async def _with_a_published_source(store: SqliteDocStore) -> list[Chunk]:
    """Index one glossary page that came from somewhere with a canonical address.

    Written out rather than routed through :func:`tests.glossary.system.index`, because the
    difference *is* the metadata: that helper builds a plain synthetic file, which is the
    correct fixture for every other case here and the one thing this case cannot use.
    """
    record = Provenance(
        source=SourceMetadata(
            title=MIRRORED_TITLE,
            canonical_uri=MIRRORED_URI,
            source_id=MIRRORED_SOURCE_ID,
            version=MIRRORED_VERSION,
        )
    )
    document = make_document(
        source="fixture",
        source_id="mirrored",
        workspace_id=system.WORKSPACE,
        title="Glossary supplement: mirrored page",
        uri=MIRRORED_URI,
        body=MIRRORED_ENTRY.encode(),
    ).model_copy(update={"metadata": {PROVENANCE_KEY: record.as_metadata_value()}})
    await store.upsert_document(document)
    chunks = [make_chunk(document, 0, MIRRORED_ENTRY, heading_path=(document.title,))]
    await store.replace_chunks(document.id, chunks)
    entries = detect_entries(chunks, title=document.title)
    assert entries, "the fixture must actually detect a definition, or it tests nothing"
    await store.replace_glossary_entries(document.id, entries)
    return chunks


@dataclass
class _Wired(FakeBackend):
    """The application's backend with real retrieval and the real store behind it.

    Everything this feature does not touch stays a fake — the answerer, telemetry,
    conversations — so a failure here is about a classification reaching a payload rather than
    about a model or a schema. The two that are real are the two it travels through.
    """

    docstore: DocumentSurface | None = None
    retrieval: Retrieving | None = None

    @override
    async def documents(self) -> DocumentSurface:
        assert self.docstore is not None, "the fixture must wire a store"
        return self.docstore

    @override
    async def retriever(self) -> Retrieving:
        assert self.retrieval is not None, "the fixture must wire a retriever"
        return self.retrieval


@dataclass
class _Restricted:
    """A retriever that adds a chunk-level restriction to every query it is handed.

    **Not a stub for retrieval.** The shipped retriever still runs, still looks the term up,
    still fetches the defining passage by id and still refuses it — every decision under test is
    made by the real code. What this stands in for is a *caller* able to express a chunk-level
    restriction, and there is none: ``kinds`` and ``langs`` live on
    :class:`~manicule.core.retrieval.Filter` and on no route, tool or command, by design.
    Without something in this position the "found a definition and then kept it out of the
    context" state is unreachable from the application boundary, and a case that cannot be
    constructed cannot be asserted.
    """

    inner: Retriever
    kinds: frozenset[BlockKind]

    async def retrieve(self, query: Query) -> RetrievalResult:
        narrowed = query.filter.model_copy(update={"kinds": self.kinds})
        return await self.inner.retrieve(query.model_copy(update={"filter": narrowed}))


@dataclass
class _AfterRetrieval:
    """Where a store read falls relative to retrieval finishing, and to the passage check.

    The service reads the store twice after retrieval returns and they are separate round
    trips: the retrieved passages are proved to belong to this workspace, and then the glossary
    entries' documents are resolved for their provenance. This counts them, so a fixture can put
    a soft delete in the gap between the two rather than guessing at which lookup is which.
    """

    retrieved: bool = False
    after: int = 0

    def still_visible(self) -> bool:
        """Whether a read happening now should still see the document about to be deleted."""
        if not self.retrieved:
            return True
        self.after += 1
        return self.after == 1


@dataclass
class _Announcing:
    """The shipped retriever, which says when it has finished.

    Nothing about the retrieval changes — the result is returned untouched. What this buys is a
    point in time the fixture can hang a race off, which is otherwise inside a method.
    """

    inner: Retriever
    reads: _AfterRetrieval

    async def retrieve(self, query: Query) -> RetrievalResult:
        result = await self.inner.retrieve(query)
        self.reads.retrieved = True
        return result


@dataclass
class _Recorded:
    """One query's payloads, so a case states its question once and asserts on both surfaces."""

    search: SearchResult
    answer: AnswerResultPayload


@pytest.fixture
async def indexed(store: SqliteDocStore) -> list[Chunk]:
    return await system.build_corpus(store)


async def _service(
    store: SqliteDocStore,
    chunks: Sequence[Chunk],
    *,
    cache: L1QueryCache | None = None,
    router: QueryRouter | None = None,
    overrides: Mapping[str, object] | None = None,
    retrieval: Retrieving | None = None,
) -> ApplicationService:
    """The real service over the real retriever, or over one the caller has already wrapped."""
    if retrieval is None:
        retrieval = await _retriever(store, chunks, cache=cache, router=router, overrides=overrides)
    return ApplicationService(_Wired(docstore=store, retrieval=retrieval))


async def _retriever(
    store: SqliteDocStore,
    chunks: Sequence[Chunk],
    *,
    cache: L1QueryCache | None = None,
    router: QueryRouter | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Retriever:
    """The shipped retriever over this corpus, for a case that wants to wrap one."""
    return await system.retriever_over(
        store, BagOfWordsEmbedder(), chunks, cache=cache, router=router, overrides=overrides
    )


async def _both(service: ApplicationService, text: str) -> _Recorded:
    """The same question through ``search`` and through ``ask``, on one service."""
    found = await service.search(text, limit=LIMIT)
    answered = await service.ask(text, limit=LIMIT)
    return _Recorded(search=found, answer=answered)


async def _run(
    store: SqliteDocStore, chunks: Sequence[Chunk], text: str, **wiring: Any
) -> _Recorded:
    return await _both(await _service(store, chunks, **wiring), text)


def _json(payload: SearchResult | AnswerResultPayload) -> dict[str, Any]:
    """The payload as it is serialised, parsed back. Never the model's attributes.

    The requirement is about what a consumer reads out of JSON, and a boolean that is a boolean
    on the model and a string in the dump would satisfy an attribute assertion while failing the
    contract. Round-tripping through the serialiser is what makes the type part of the claim.
    """
    parsed: dict[str, Any] = json.loads(payload.model_dump_json())
    return parsed


def _flat(text: str) -> str:
    """Rendered output with its line breaks collapsed.

    The terminal wraps at whatever width the capture ran at, so a sentence asserted against raw
    output is a sentence that fails when somebody adds a word to the line before it. Collapsing
    whitespace asserts what was said rather than where it wrapped.
    """
    return " ".join(text.split())


# --- the supported case, on each surface ------------------------------------------------------


async def test_search_json_reports_the_classification_for_a_definitional_query(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 1 and requirement 5's ``true`` case, on the search payload.

    Every condition is asserted rather than assumed: the term expanded, the expansion names the
    document and passage the definition was read out of, that passage is among the hits, and the
    reason is the one that says a definition was cited rather than the one that says nothing in
    the corpus resembles the question.
    """
    recorded = await _run(store, indexed, corpus.QUERY_ACRONYM)
    body = _json(recorded.search)

    assert body["explicit_definition"] is True
    assert body["expansions"], "a cited definition with no provenance is not reportable"
    cited = body["expansions"][0]
    assert cited["acronym"] == corpus.ACRONYM
    assert cited["expansion"] == corpus.EXPANSION
    for located in LOCATES_A_DEFINITION:
        assert cited[located], f"an expansion with no {located} is one nobody can open"
    assert cited["chunk_id"] in {hit["chunk_id"] for hit in body["hits"]}
    assert body["confidence_reason"] == DEFINITION_CITED


async def test_an_expansion_carries_its_own_source_reference(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 9's fifth field: source identity, on the expansion rather than joined to it.

    Two definitions, and the difference between them is the point. One is defined on a mirrored
    page carrying an authoritative record, and its expansion reports the publisher's own
    ``source_id``, ``canonical_uri``, ``title`` and ``version``. The other is defined on an
    ordinary synthetic file, and reports ``null`` — the honest answer for a document with no
    such record, where an empty object would read as one that was looked for and came back
    blank.

    Both halves are needed. Without the populated one the field could be hard-wired to ``null``
    and nothing would notice; without the ``null`` one, a consumer would have no evidence that
    the ordinary case is a stated absence rather than a missing key.
    """
    mirrored = await _with_a_published_source(store)
    service = await _service(store, [*indexed, *mirrored])

    published = _json(await service.search(MIRRORED_QUERY, limit=LIMIT))["expansions"][0]
    local = _json(await service.search(corpus.QUERY_ACRONYM, limit=LIMIT))["expansions"][0]

    assert published["provenance"] is not None
    assert published["provenance"]["source_id"] == MIRRORED_SOURCE_ID
    assert published["provenance"]["canonical_uri"] == MIRRORED_URI
    assert published["provenance"]["title"] == MIRRORED_TITLE
    assert local["provenance"] is None, "an ordinary file has no record, and says so"
    for cited in (published, local):
        for located in LOCATES_A_DEFINITION:
            assert cited[located], f"the four location fields are populated either way: {located}"


async def test_answer_json_reports_the_same_classification_as_the_search(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 1 on the answer payload, compared with the search rather than to a literal.

    Both run the same retrieval over the same corpus, so the claim worth making is not that the
    answer says ``true`` — it is that the two surfaces cannot disagree about it or about the
    passage it points at.
    """
    recorded = await _run(store, indexed, corpus.QUERY_ACRONYM)

    assert _json(recorded.answer)["explicit_definition"] is True
    assert recorded.answer.explicit_definition == recorded.search.explicit_definition
    assert [item.chunk_id for item in recorded.answer.expansions] == [
        item.chunk_id for item in recorded.search.expansions
    ]


async def test_the_streamed_final_frame_carries_what_the_settled_answer_does(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 4: the field travels on the accumulator, so the stream cannot lose it.

    An async generator cannot also return a value, so a fact the accumulator does not carry
    reaches the streamed ``final`` frame through no path at all — it would be ``false`` on an
    SSE stream and ``true`` on a plain ``ask`` of the same question, which is the divergence
    that side channel exists to prevent. Driven through the HTTP surface's own frame builder,
    because that is what a browser receives.
    """
    from manicule.api.streaming import answer_frames  # noqa: PLC0415 - keeps the web extra out

    service = await _service(store, indexed)

    collected = [
        frame
        async for frame in answer_frames(
            service,
            question=corpus.QUERY_ACRONYM,
            profile=None,
            limit=LIMIT,
            sources=(),
            conversation_id=None,
        )
    ]

    name, final = collected[-1]
    assert name == "final"
    envelope = json.loads(json.dumps(final))
    assert envelope["ok"] is True
    assert envelope["data"]["explicit_definition"] is True
    assert envelope["data"]["expansions"], "the streamed frame lost the provenance"


async def test_the_mcp_tools_return_a_boolean_and_not_a_string(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 1 through the real FastMCP tool manager, and the *type* is the assertion.

    ``is True`` rather than a truthiness check, because ``"false"`` is truthy: a stringified
    boolean would take the wrong branch in every client on every negative result, and no
    assertion written as ``assert value`` would ever notice.
    """
    # Imported here so FastMCP stays off the import path of every other case in this file.
    from manicule.mcp.server import build_server  # noqa: PLC0415

    server = build_server(await _service(store, indexed))

    asking = corpus.QUERY_ACRONYM
    found = (await server.call_tool("search", {"query": asking})).structured_content
    answered = (await server.call_tool("ask", {"question": asking})).structured_content

    assert found is not None
    assert answered is not None
    assert found["data"]["explicit_definition"] is True
    assert answered["data"]["explicit_definition"] is True
    assert isinstance(found["data"]["explicit_definition"], bool)


async def test_the_http_routes_return_the_same_field_with_the_same_semantics(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 1 over the production application, on both routes and on both answers.

    The ordinary use goes through the same client as the definitional question, so the two are
    one comparison: a route reporting ``true`` unconditionally passes the first assertion alone.
    """
    import httpx  # noqa: PLC0415 - only this case needs an HTTP client

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the rest

    app = build_app(await _service(store, indexed))
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 41234))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        asked = (await client.get("/api/v1/search", params={"q": corpus.QUERY_ACRONYM})).json()
        used = (
            await client.get("/api/v1/search", params={"q": ORDINARY_USE_OF_A_DEFINED_TERM})
        ).json()
        answered = (
            await client.post("/api/v1/chat", json={"question": corpus.QUERY_ACRONYM})
        ).json()

    assert asked["data"]["explicit_definition"] is True
    assert used["data"]["explicit_definition"] is False
    assert answered["data"]["explicit_definition"] is True


async def test_the_terminal_says_a_definition_was_cited_and_leaves_the_number_alone(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The human surface, and the presentation rule the specification is explicit about.

    Two claims, and the second is the one worth the test: the indication appears, **and** the
    confidence figure printed beside it is the figure the payload carries, unrounded up and
    unrebanded. A terminal that quietly showed a better number when a definition was cited
    would be doing exactly what the classification exists to prevent, and it would look like an
    improvement.

    The renderers are called directly rather than through the command, and that is what the
    command does: ``manicule.cli.render`` decides nothing and computes nothing, so rendering the
    real payload through the real renderer *is* the command's output. Driving Typer here would
    add a second event loop around a database bound to this one.
    """
    recorded = await _run(store, indexed, corpus.QUERY_ACRONYM)
    assert recorded.search.confidence is not None

    out = console()
    with out.capture() as capture:
        render_search(out, recorded.search)
        render_answer(out, recorded.answer)
    printed = _flat(capture.get())

    assert printed.count(EXPLICIT_DEFINITION) == 2, "one indication per rendered payload"
    assert f"confidence {recorded.search.confidence:.2f}" in printed
    assert f"({recorded.search.confidence_band})" in printed
    assert _flat(DEFINITION_CITED) in printed, "the reader-facing sentence is still printed"


# --- the false cases, which are what a hollow implementation would get wrong -------------------


async def test_an_ordinary_use_of_a_defined_token_reports_false(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5: ``false`` for a use of the same token, with every other condition true.

    The term expands, the definition is fetched and promoted to rank 1, and the reader is
    looking at a glossary entry for the word they typed. The classification is still ``false``,
    because the question is an instruction rather than a request for a meaning — so a field
    populated from "an expansion fired", or from "the definition is on screen", is ``true``
    here on both surfaces and this is where it says so.
    """
    recorded = await _run(store, indexed, ORDINARY_USE_OF_A_DEFINED_TERM)

    assert recorded.search.expansions, "the fixture must expand, or it tests nothing"
    cited = recorded.search.expansions[0].chunk_id
    assert cited in {hit.chunk_id for hit in recorded.search.hits}, (
        "the defining passage must be in front of the reader, or the rule is untested"
    )
    assert _json(recorded.search)["explicit_definition"] is False
    assert _json(recorded.answer)["explicit_definition"] is False


async def test_a_contested_term_reports_false_and_still_shows_the_disagreement(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5: ``false`` for a conflict, and the conflict is reported rather than hidden.

    A second, disagreeing definition of the same term is indexed, so the term expands to nothing
    at all — expansion reports the disagreement instead of choosing between them. The two
    assertions belong together: a payload that had also gone quiet about the conflict would
    report ``false`` as well, and would be withholding the one thing the reader can act on.
    """
    _, second = await system.index(
        store,
        "second-glossary",
        "Glossary of terms, elsewhere",
        [f"{corpus.ACRONYM} — Nightly Operations Watchdog, a scheduled checker."],
    )
    recorded = await _run(store, [*indexed, *second], corpus.QUERY_ACRONYM)

    assert recorded.search.expansions == (), "a contested term expands to nothing"
    assert len(recorded.search.conflicts) == 1
    assert len(recorded.search.conflicts[0].candidates) == 2
    assert _json(recorded.search)["explicit_definition"] is False
    assert _json(recorded.answer)["explicit_definition"] is False


async def test_a_defining_passage_a_chunk_filter_refused_reports_false(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5: ``false`` when the definition was found and then kept out of the context.

    Restricted to source code, and every passage in this corpus is prose. The vocabulary lookup
    deliberately ignores ``kinds`` — it returns vocabulary rather than chunks — so the term
    still expands and the expansion is still reported with its provenance. What does not happen
    is the passage reaching the reader, and "we found a definition" is not "we are showing you
    one".

    The expansion surviving is what makes this different from an empty result: the payload has
    a document, a chunk, a title and a URI in hand and still reports ``false``, which is the
    only combination that distinguishes the classification from "the glossary said something".
    Asserted against the same query on an unrestricted service, so the difference between the
    two is the restriction and nothing else.
    """
    built = await _retriever(store, indexed)
    restricted = await _service(
        store, indexed, retrieval=_Restricted(built, frozenset({BlockKind.CODE}))
    )

    found = await restricted.search(corpus.QUERY_ACRONYM, limit=LIMIT)
    unrestricted = await (await _service(store, indexed)).search(corpus.QUERY_ACRONYM, limit=LIMIT)

    assert unrestricted.explicit_definition is True, "the fixture must fire without the filter"
    assert found.expansions, "the lookup ignores chunk-level restrictions, so this still fires"
    cited = found.expansions[0].chunk_id
    assert cited == unrestricted.expansions[0].chunk_id, "the same passage is being talked about"
    assert cited not in {hit.chunk_id for hit in found.hits}, (
        "the restriction has to actually keep the defining passage out of the results"
    )
    assert _json(found)["explicit_definition"] is False
    assert found.confidence_reason != DEFINITION_CITED


async def test_a_defining_passage_the_context_budget_dropped_reports_false(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5: ``false`` when assembly dropped the defining passage to fit the window.

    Two definitions are promoted and definitions lead the ranking, so the order into assembly is
    fixed: the passage for the term the query *uses* first, the passage for the term it *asks
    about* second. At the smallest budget a profile may declare, the first fits and the second
    does not — so what survives is a definition nobody asked for and what is dropped is the one
    that would have made the claim true.

    **The budget is the only difference between the two runs**, and that is how the case proves
    what it claims. ``hits`` are the ranking rather than the assembled context, so the defining
    passage is still among them either way; what changes is whether it reached the context, and
    the classification is the only field on this payload that reports the distinction. A single
    run asserting ``false`` would pass just as well against a corpus that had nothing to drop.

    The arithmetic is asserted rather than trusted, because the whole case rests on it: change a
    fixture entry and this stops being the situation it says it is.
    """
    entries = await system.entries_by_acronym(store, indexed)
    used, asked = entries[corpus.DESCRIBED_ACRONYM], entries[corpus.ACRONYM]
    sizes = {chunk.id: ContextTokenCounter().count_chunk(chunk) for chunk in indexed}
    assert sizes[used.chunk_id] <= MINIMUM_CONTEXT_TOKENS, (
        "the used term's definition must fit, or assembly refuses the context outright"
    )
    assert sizes[used.chunk_id] + sizes[asked.chunk_id] > MINIMUM_CONTEXT_TOKENS, (
        "both definitions fit at the floor, so nothing is dropped and this case tests nothing"
    )

    roomy = await (await _service(store, indexed)).search(TWO_TERMS_ONE_QUESTION, limit=LIMIT)
    cramped = await (
        await _service(store, indexed, overrides={"context_tokens": MINIMUM_CONTEXT_TOKENS})
    ).search(TWO_TERMS_ONE_QUESTION, limit=LIMIT)

    assert roomy.truncated is False
    assert roomy.explicit_definition is True, "the fixture must fire when everything fits"
    assert cramped.truncated is True, "a context that dropped a passage has to say so"
    assert _json(cramped)["explicit_definition"] is False
    assert {hit.chunk_id for hit in cramped.hits} >= {used.chunk_id, asked.chunk_id}, (
        "both definitions are still ranked — it is the context they did not both reach"
    )


async def test_a_query_the_router_answered_directly_reports_false(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5: ``false`` for a utility route that never consults the corpus.

    ``Confidence`` is **absent** on these rather than zero, so there is no classification to
    copy and the payload has to supply ``false`` itself. Asserted alongside ``confidence is
    None`` because that absence is what makes this the branch it is: a change that started
    scoring directly-routed queries would move the case somewhere else entirely, and this would
    say so rather than passing on.
    """
    service = await _service(
        store, indexed, router=QueryRouter(RouterSettings(), available=handlers_for(store))
    )

    found = await service.search("how many documents are indexed", limit=LIMIT)

    assert found.route == "utility", "the fixture must actually route away from the corpus"
    assert found.confidence is None
    assert found.expansions == ()
    assert _json(found)["explicit_definition"] is False


async def test_a_question_this_corpus_cannot_answer_stays_false_with_its_reason_intact(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 5's last case and requirement 7: no evidence, and the sentence that says so.

    Nothing in the corpus is about this and no term is named, so the reason has to remain the
    one reporting that the corpus holds nothing resembling the question. That sentence is only
    replaced when a definition really is on screen, and this is the case that notices if the
    replacement has quietly become unconditional.
    """
    recorded = await _run(store, indexed, corpus.QUERY_UNRELATED)

    assert _json(recorded.search)["explicit_definition"] is False
    assert recorded.search.confidence_band == "none"
    assert recorded.search.confidence_reason == NOTHING_RESEMBLES
    assert _json(recorded.answer)["explicit_definition"] is False
    assert recorded.answer.confidence_reason == NOTHING_RESEMBLES


async def test_a_definition_whose_document_vanished_withdraws_the_claim_rather_than_failing(
    store: SqliteDocStore, indexed: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one live path where requirement 9's invariant bites, and how the service resolves it.

    A search makes two scoped reads and they are separate round trips: the retrieved passages
    are proved to belong to this workspace, and *then* the glossary entries' documents are
    resolved for their provenance. A soft delete landing between the two is an ordinary race —
    the second read finds nothing, and the expansion is dropped rather than rendered with a
    blank source, which is a rule that predates this field.

    What is new is that retrieval classified the query a moment earlier, while the document was
    still there. So the payload would claim a citation it can no longer name, and
    :class:`~manicule.app.results.Glossed` refuses that combination outright — which means the
    alternative to withdrawing the claim is not "report it anyway" but "raise", turning a soft
    delete into a failed search.

    Sequenced rather than filtered on which document was asked for, and that is a correction:
    retrieval reads this store several times, once for the visibility of the passage it is
    about to promote, and hiding the document from *those* reads produces a different case
    entirely — the definition is never promoted, retrieval classifies ``false`` itself, and the
    payload has nothing to withdraw. The race being simulated happens after retrieval returns.

    Without this case the clause that withdraws the claim is unreachable code, and deleting it
    would break nothing that anybody runs.
    """
    entries = await system.entries_by_acronym(store, indexed)
    vanishing = entries[corpus.ACRONYM].document_id
    real = store.list_documents
    reads = _AfterRetrieval()

    async def races(
        filter: Filter | None = None,  # noqa: A002 - mirrors the method it stands in for
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        found = await real(filter, limit=limit, offset=offset)
        if reads.still_visible():
            return found
        return [document for document in found if document.id != vanishing]

    service = await _service(
        store, indexed, retrieval=_Announcing(await _retriever(store, indexed), reads)
    )
    monkeypatch.setattr(store, "list_documents", races)

    found = await service.search(corpus.QUERY_ACRONYM, limit=LIMIT)

    assert reads.after == READS_AFTER_RETRIEVAL, (
        "the sequence changed: this case is about a delete landing between the passage check "
        "and the provenance lookup, and there are no longer exactly those two reads"
    )

    assert found.hits, "the search still answers; a soft delete is not an outage"
    assert found.expansions == (), "an expansion with no readable document is dropped"
    assert _json(found)["explicit_definition"] is False
    assert found.confidence_reason == DEFINITION_CITED, (
        "retrieval's own reason is untouched — this withdraws a claim, it does not rescore"
    )


# --- the properties the field is not allowed to disturb ---------------------------------------


async def test_a_cache_hit_reports_everything_a_miss_did(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 8, over the whole payload rather than over the boolean.

    The classification is recomputed on a hit rather than stored in the cache entry, and that is
    a thing to check rather than assume: a field computed after the cache and a field read out
    of it behave identically right up until the cached ranking and the freshly resolved
    expansion disagree, and only one of them is right. Compared as whole payloads with the
    clocks removed, so a field that stopped surviving a hit fails here whether or not anybody
    thought of it — which is more than the specification's list of five asks for, and cheaper
    than five assertions that each name one field.
    """
    service = await _service(store, indexed, cache=L1QueryCache(entries=16))

    miss = await service.search(corpus.QUERY_ACRONYM, limit=LIMIT)
    hit = await service.search(corpus.QUERY_ACRONYM, limit=LIMIT)

    assert miss.cached is False
    assert hit.cached is True, "the fixture must actually hit the cache"
    assert hit.explicit_definition is True
    volatile = {"elapsed_ms", "cached"}
    assert {key: value for key, value in _json(hit).items() if key not in volatile} == {
        key: value for key, value in _json(miss).items() if key not in volatile
    }


async def test_surfacing_the_field_changed_no_number(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """Requirement 6, asserted against retrieval's own figures rather than against a recording.

    A test comparing today's score with a number written down last week proves only that
    somebody updated the number. What is compared here is each payload against the
    ``Confidence`` the retriever produced for the same query — on the query where the
    classification fires, on one where the term is merely used, and on one with no evidence at
    all — so the claim is that surfacing the field copies the arithmetic rather than joining in.

    The stronger evidence is structural and is in the diff: ``score``, ``band`` and ``reason``
    are read from exactly where they were read before, and ``explicit_definition`` is a fourth
    read of the same object. This is what would fail if that stopped being true.
    """
    service = await _service(store, indexed)
    retriever = await service.backend.retriever()

    for text in (corpus.QUERY_ACRONYM, ORDINARY_USE_OF_A_DEFINED_TERM, corpus.QUERY_UNRELATED):
        found = await service.search(text, limit=LIMIT)
        scored = await retriever.retrieve(
            Query(
                text=text,
                limit=LIMIT,
                profile=service.settings.rag.profile,
                filter=system.query_filter(),
            )
        )
        assert scored.confidence is not None
        assert found.confidence == scored.confidence.score
        assert found.confidence_band == scored.confidence.band.value
        assert found.confidence_reason == scored.confidence.reason
        assert found.explicit_definition is scored.confidence.explicit_definition


def test_a_payload_written_before_the_field_existed_still_parses() -> None:
    """Requirement 2: the default is ``false``, so a document stored before it deserialises.

    ``false`` is the honest default rather than a convenient one: a result produced before the
    classification existed made no claim about it, and "no claim" is what an unset boolean has
    always meant on these payloads.
    """
    stored = {"query": "what is it", "profile": "balanced", "count": 0}
    assert SearchResult.model_validate(stored).explicit_definition is False

    older = {"question": "what is it", "text": "no idea"}
    assert AnswerResultPayload.model_validate(older).explicit_definition is False


@pytest.mark.parametrize("model", [SearchResult, AnswerResultPayload])
def test_a_claimed_definition_with_nothing_to_cite_cannot_be_built(
    model: type[SearchResult] | type[AnswerResultPayload],
) -> None:
    """Requirement 9 as an invariant of the model rather than as an assertion about a run.

    ``explicit_definition=true`` with no expansion is a result telling a reader that a
    definition was cited and naming no document, chunk, title or URI for it. Refusing it in the
    model means no surface can emit that shape — including one nobody has written yet, which is
    the half a test cannot cover.

    The provenance half of the requirement reduces to this because
    :class:`~manicule.app.results.GlossaryExpansion` cannot be constructed without a document, a
    chunk, a title and a URI. If there is one, those fields are there.
    """
    required: dict[str, Any] = (
        {"query": "what is it", "profile": "balanced", "count": 0}
        if model is SearchResult
        else {"question": "what is it", "text": "no idea"}
    )

    with pytest.raises(ValueError, match="explicit_definition is true"):
        model.model_validate({**required, "explicit_definition": True})
