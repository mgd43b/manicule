"""A definition inside a Confluence macro, from storage XHTML to a public result.

**This suite begins at raw bytes on purpose.** Every other glossary test starts from chunk
text, which is the right fixture for a detector and the wrong one for the defect this file
exists for: the boundary that was lost is lost *between* the parser and the chunker, so a
regression handed ready-made chunks cannot see it. Here the shipped
:class:`~manicule.parsers.confluence.ConfluenceStorageParser` reads the fixture, the shipped
:class:`~manicule.chunking.StructuralChunker` chunks it at the shipped budget, the shipped
detector reads the chunks, and the shipped retriever and application service answer questions
about the result.

**What was wrong.** Consecutive paragraphs inside a rich-text macro were joined with a single
newline. Nothing downstream reads that as a paragraph boundary, so a macro body was one
paragraph however many ``<p>`` elements it held — and once past the chunk budget it was split
into sentences and repacked with spaces.

Measured by disabling the fix and running these cases: **every definition in the macro is lost,
not merely the ones after the first.** Detection matches its forms at the start of a line, so
after the repack the second and third definitions are not at one; and the first, which is,
takes the whole remaining line as its right-hand side, overruns
:data:`~manicule.ingest.glossary.MAX_EXPANSION_WORDS` and is refused as well. The store ends up
with no entries at all from a page carrying three definitions.

**Which of the fixture's three definitions are recorded, and why.** One: ``SaFeR``. This is
stated rather than left to be discovered, because the arithmetic is the whole reason the case
is sensitive to anything.

===========  ==========================  ==========  ============================================
Definition   Initials of its expansion   Confidence  Outcome
===========  ==========================  ==========  ============================================
``SaFeR``    ``SFR`` — agrees            0.80        recorded
``OLD``      ``ODL`` — disagrees         0.45        refused
``NEXT``     ``NEET`` — disagrees        0.45        refused
===========  ==========================  ==========  ============================================

A spaced hyphen is worth 0.45 and the threshold is 0.60, so a definition on a page that does
not announce itself as a glossary needs the 0.35 of initials agreement to clear it. ``SaFeR``
has it only through :func:`~manicule.ingest.glossary.initial_skeleton`: the normalized key is
``SAFER``, whose initials nothing spells, and it is the *display* spelling's skeleton ``SFR``
that ``Signal Fault Router`` matches. ``OLD`` and ``NEXT`` have no such route and are refused.

**The page is deliberately not titled like a glossary**, and that is what keeps this honest.
:data:`~manicule.ingest.glossary.GLOSSARY_CONTEXT_EVIDENCE` is 0.15, which is exactly enough to
carry a bare spaced hyphen over the threshold — on a page called "Routing terms" all three would
be recorded, and so would ``SaFeR`` with the skeleton machinery completely broken. The first
draft of the fixture was titled that, by accident. See
:data:`tests.corpus.confluence.MACRO_TITLE`.

The one case below that *does* want all three asserts them under a glossary title, and says so:
it is the positive form of "the adjacent definitions were not swallowed", which the rest of the
file asserts negatively on ``SaFeR``'s own expansion.

Everything here is synthetic and invented for this repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.app.service import ApplicationService
from manicule.chunking.sentences import paragraphs
from manicule.core.protocols import read_blocks
from manicule.core.retrieval import Query
from manicule.ingest.glossary import detect_entries
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE, ConfluenceConfig
from manicule.parsers.confluence import ConfluenceStorageParser
from manicule.retrieval.confidence import DEFINITION_CITED
from tests.app.fakes import FakeBackend
from tests.corpus.confluence import MACRO_BODY, MACRO_TITLE
from tests.evaluation.fakes import BagOfWordsEmbedder
from tests.glossary import system
from tests.parsers.support import make_chunker, raw_of
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.app.ports import DocumentSurface, Retrieving
    from manicule.core.content import Chunk, Document
    from manicule.core.glossary import GlossaryEntry
    from manicule.storage.docstore import SqliteDocStore

LIMIT = 10

TERM = "SaFeR"
KEY = "SAFER"
EXPANSION = "Signal Fault Router"
DESCRIPTION = "a service that routes operational signals"
"""The words that follow the expansion on the same line, and must not be part of it."""

NEIGHBORS = ("Older Delivery Layer", "Network Event Exchange Toolkit")
"""The definitions either side of the one under test.

Asserted absent from its expansion, which is the direct form of "not swallowed": under the
defect the whole macro body was one line, and a detector matching from the start of it read
every neighbor as part of the first definition's right-hand side — which is also why that
first definition was then refused for being too long to be an expansion.
"""

DEFINITION_QUERIES = ("What is SaFeR?", "what is safer?", "WHAT IS SAFER?")
"""One question in three spellings. The key is normalized; the display is not."""

USES_THE_TERM = "restart SaFeR after the maintenance window closes"
"""A query that names the term and does not ask what it means."""

GLOSSARY_TITLE = "Routing terms and abbreviations"
"""A title that announces the page as a glossary, used by exactly one case below."""

USAGE_PASSAGES = (
    "Restart the router before the window closes, then confirm the buffer has drained.",
    "The maintenance window closes at midnight and the on-call engineer confirms the drain.",
    "Operational signals are replayed from the buffer once the gateway reports itself ready.",
)
"""Ordinary passages that use the vocabulary without defining anything.

A one-document corpus makes every ranking assertion trivial: the only passage there is is the
one that would have to win. These give the retriever something to be wrong about.
"""


@dataclass
class _Wired(FakeBackend):
    """The application over the real store and the real retriever, everything else a fake.

    The same shape ``test_surfaces.py`` uses, and for the same reason: a fake retriever handed a
    ``Confidence`` with the flag already set would prove the field is copied and nothing about
    whether ingest produced a definition to copy it from.
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


async def _ingest(
    store: SqliteDocStore, *, title: str = MACRO_TITLE
) -> tuple[Document, list[Chunk]]:
    """Storage XHTML through the shipped parser and the shipped chunker, into the store.

    Not a shortcut around ingest so much as ingest's middle: the pipeline adds blob retention,
    embedding and lineage, none of which decides where a paragraph ends. What is deliberately
    *not* substituted is the parser, the chunker or its budget — those three between them are
    the defect.
    """
    raw = raw_of(MACRO_BODY, CONFLUENCE_MEDIA_TYPE, uri="macro-body.storage", title=title)
    blocks = await read_blocks(ConfluenceStorageParser(ConfluenceConfig()), raw)
    document = make_document(
        source="fixture",
        source_id="macro-body",
        workspace_id=system.WORKSPACE,
        title=title,
        uri="https://docs.example.test/pages/macro-body",
        media_type=CONFLUENCE_MEDIA_TYPE,
        body=MACRO_BODY.encode(),
    )
    chunks = make_chunker().chunk(document, blocks)
    await store.upsert_document(document)
    await store.replace_chunks(document.id, chunks)
    await store.replace_glossary_entries(
        document.id,
        detect_entries(chunks, title=title),
        fingerprint=glossary_fingerprint().canonical(),
    )
    return document, chunks


async def _corpus(store: SqliteDocStore, *, title: str = MACRO_TITLE) -> list[Chunk]:
    """The Confluence page plus a few passages that merely use its vocabulary."""
    _, chunks = await _ingest(store, title=title)
    usage = await _usage(store)
    return [*chunks, *usage]


async def _usage(store: SqliteDocStore) -> list[Chunk]:
    document = make_document(
        source="fixture",
        source_id="runbook",
        workspace_id=system.WORKSPACE,
        title="Gateway runbook",
        uri="https://docs.example.test/pages/runbook",
        body=b"runbook",
    )
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, position, text, heading_path=(document.title,))
        for position, text in enumerate(USAGE_PASSAGES)
    ]
    await store.replace_chunks(document.id, chunks)
    return chunks


@pytest.fixture
async def indexed(store: SqliteDocStore) -> list[Chunk]:
    return await _corpus(store)


@pytest.fixture
async def entry(store: SqliteDocStore, indexed: Sequence[Chunk]) -> GlossaryEntry:
    found = await system.entries_by_acronym(store, indexed)
    assert KEY in found, (
        f"nothing detected {TERM} from the fixture, so every assertion below would be about "
        f"the wrong thing. Recorded: {sorted(found)}"
    )
    return found[KEY]


def _ask(text: str) -> Query:
    return Query(text=text, limit=LIMIT, filter=system.query_filter())


async def _service(store: SqliteDocStore, chunks: Sequence[Chunk]) -> ApplicationService:
    retriever = await system.retriever_over(store, BagOfWordsEmbedder(), chunks)
    return ApplicationService(_Wired(docstore=store, retrieval=retriever))


# --- the boundary, through the chunker ---------------------------------------------------------


async def test_the_macro_body_reaches_the_chunker_as_more_than_one_paragraph(
    store: SqliteDocStore,
) -> None:
    """The fixture is big enough for the defect to bite, and the boundaries survive it.

    Both halves matter. A macro body under the chunk budget keeps its definitions at the start
    of a line whether it was joined with one newline or two, so a fixture that never reaches the
    splitter cannot demonstrate the fix. This one does reach it — the assertion on the block
    says so — and the definition still opens a line afterwards.
    """
    _, chunks = await _ingest(store)
    body = next(chunk for chunk in chunks if TERM in chunk.text)

    assert len(chunks) > 1, "the page split, so the paragraph splitter ran"
    assert any(line.startswith(TERM) for line in body.text.splitlines()), (
        "the definition opens a line of the chunk that carries it; under the defect the whole "
        "macro body was one line and only its first definition did"
    )
    assert TERM not in body.text.splitlines()[0], (
        "and it opens a line somewhere in the middle rather than being the chunk's first line, "
        "which is the position the defect left the first definition in and no other"
    )


# --- the entry the ingest produced -------------------------------------------------------------


async def test_the_stylized_term_is_detected_from_the_middle_paragraph(
    entry: GlossaryEntry,
) -> None:
    """The middle of three, which is the position the defect made unreachable.

    Under the old join this definition was not at the start of any line, because there was only
    one line. Disabling the fix and running this case alone reports ``Recorded: []``.
    """
    assert entry.acronym == KEY, "the normalized lookup key"
    assert entry.display == TERM, "the source's own spelling, stored verbatim"


async def test_the_core_expansion_is_extracted_without_its_trailing_description(
    entry: GlossaryEntry,
) -> None:
    """What the term stands for, not the sentence the page wrote about it."""
    assert entry.expansion == EXPANSION
    assert DESCRIPTION not in entry.expansion


async def test_the_adjacent_definitions_are_not_swallowed_into_the_expansion(
    entry: GlossaryEntry,
) -> None:
    """The requirement the boundary fix is really for.

    Under the defect the three paragraphs were one line, so the first definition's right-hand
    side ran to the end of the macro body and carried both of its neighbors with it.
    """
    for neighbor in NEIGHBORS:
        assert neighbor not in entry.expansion


async def test_the_defining_chunk_is_kept_as_the_entry_provenance(
    entry: GlossaryEntry, indexed: Sequence[Chunk]
) -> None:
    """An expansion whose source cannot be opened is an assertion rather than a citation.

    The location is asserted against the cited chunk's own breadcrumb rather than against the
    fixture's headings written out again. A second copy would keep passing if the entry started
    naming some *other* chunk's section, which is the failure worth catching.
    """
    cited = next(chunk for chunk in indexed if chunk.id == entry.chunk_id)

    assert EXPANSION in cited.text, "the cited chunk states the definition it is cited for"
    assert entry.location == " > ".join(cited.heading_path)
    assert entry.document_id == cited.document_id


# --- retrieval over what the ingest produced ---------------------------------------------------


@pytest.mark.parametrize("text", DEFINITION_QUERIES, ids=["display", "lower", "upper"])
async def test_a_definition_query_resolves_the_term_whatever_its_case(
    store: SqliteDocStore, indexed: list[Chunk], entry: GlossaryEntry, text: str
) -> None:
    """Case-insensitive because the key is normalized, and the display survives regardless."""
    retriever = await system.retriever_over(store, BagOfWordsEmbedder(), indexed)

    result = await retriever.retrieve(_ask(text))

    assert result.expansion is not None
    assert result.expansion.matches, f"no entry fired for {text!r}"
    fired = result.expansion.matches[0].entry
    assert fired.acronym == KEY
    assert fired.display == TERM
    assert fired.expansion == EXPANSION
    assert system.rank_of(result.context.passages, entry.chunk_id) == 1, (
        "the defining passage survives into the delivered context, at the top of it"
    )


async def test_the_public_result_reports_an_explicit_definition(
    store: SqliteDocStore, indexed: list[Chunk], entry: GlossaryEntry
) -> None:
    """The whole path, ending where a caller reads it: the serialized search payload."""
    service = await _service(store, indexed)

    payload = await service.search(DEFINITION_QUERIES[0], limit=LIMIT)
    body: dict[str, Any] = json.loads(payload.model_dump_json())

    assert body["explicit_definition"] is True
    assert body["confidence_reason"] == DEFINITION_CITED
    cited = body["expansions"][0]
    assert cited["acronym"] == KEY
    assert cited["expansion"] == EXPANSION
    assert cited["chunk_id"] == entry.chunk_id
    assert cited["chunk_id"] in {hit["chunk_id"] for hit in body["hits"]}


async def test_an_ordinary_use_of_the_term_is_not_an_explicit_definition(
    store: SqliteDocStore, indexed: list[Chunk]
) -> None:
    """The negative control, and the reason the flag means anything.

    The term is defined and the query names it, so a flag wired to "a glossary entry fired"
    would report ``true`` here. The question is not what the term means, so it is ``false``.
    """
    service = await _service(store, indexed)

    payload = await service.search(USES_THE_TERM, limit=LIMIT)
    body: dict[str, Any] = json.loads(payload.model_dump_json())

    assert body["explicit_definition"] is False


# --- the two definitions the page does not carry evidence for ----------------------------------


async def test_the_neighbors_are_refused_on_their_initials_rather_than_lost(
    store: SqliteDocStore,
) -> None:
    """Named, so nobody reads this suite's single entry as a boundary still being missed.

    ``OLD`` and ``NEXT`` reach the detector on their own lines — that is what the fix bought —
    and are then refused by the confidence gate, because their expansions' initials spell
    ``ODL`` and ``NEET``. Nothing about the paragraph rule would change that.
    """
    _, chunks = await _ingest(store)
    body = next(chunk for chunk in chunks if TERM in chunk.text)
    opening = [line.split(" - ")[0] for line in body.text.splitlines() if " - " in line]

    assert opening == ["OLD", TERM, "NEXT"], "all three reach detection, in source order"
    found = await system.entries_by_acronym(store, chunks)
    assert sorted(found) == [KEY], "and only the one with initials evidence is recorded"


async def test_a_page_that_says_it_is_a_glossary_records_all_three(
    store: SqliteDocStore,
) -> None:
    """The positive form of "not swallowed": three paragraphs, three separate records.

    **The title is doing the work here and that is the point of the case, not a flaw in it.**
    ``GLOSSARY_CONTEXT_EVIDENCE`` carries a bare spaced hyphen from 0.45 to 0.60, which is why
    every other case on this page runs under a title that says nothing. What this adds is the
    thing a single entry cannot show — that the boundaries produced three independently recorded
    definitions rather than one long one — and it is worth having only because it is labeled.
    """
    _, chunks = await _ingest(store, title=GLOSSARY_TITLE)
    found = await system.entries_by_acronym(store, chunks)

    assert sorted(found) == ["NEXT", "OLD", KEY]
    assert found["OLD"].expansion == "Older Delivery Layer"
    assert found[KEY].expansion == EXPANSION
    assert found["NEXT"].expansion == "Network Event Exchange Toolkit"


async def test_the_paragraph_the_definitions_sit_in_is_the_one_the_source_wrote(
    store: SqliteDocStore,
) -> None:
    """Order and separation together, on the chunk rather than on the block.

    The parser suite asserts this on the parsed block. Repeating it here is not duplication:
    between the two there is a chunker that repacks text, and this is the side of it the
    detector actually reads.
    """
    _, chunks = await _ingest(store)
    body = next(chunk for chunk in chunks if TERM in chunk.text)

    written = [part for part in paragraphs(body.text) if " - " in part]
    assert [part.split(" - ")[0] for part in written] == ["OLD", TERM, "NEXT"]
