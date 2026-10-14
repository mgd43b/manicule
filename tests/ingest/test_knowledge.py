"""A structured memory through the pipeline: when it is read, what is stored, and when it goes.

**Driven through** :class:`~tests.ingest.fakes.DictConnector`, for the reason
``test_provenance.py`` gives: a dictionary is about as far from a filesystem as a source gets, so
if these pass then the pipeline reads a memory from *any* connector. What the record is, and what
refuses it, is ``tests/test_knowledge.py``'s business; what is asserted here is only the
pipeline's part — which documents it reads, what it writes, and what it does about the previous
revision's record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.content import DocumentStatus
from manicule.core.knowledge import KNOWLEDGE_KEY, EvidenceStatus
from manicule.ingest.reindex import re_parse
from tests.fakes import MEDIA_TYPE
from tests.ingest import fakes
from tests.ingest.test_pipeline import build
from tests.knowledge_samples import EVIDENCE, LOCATOR, SECOND_SOURCE, memory

if TYPE_CHECKING:
    from manicule.core.content import Document

MARKDOWN = "text/markdown"
ID = "reference_vpn_gateway_failover.md"


def a_connector(text: str, *, media_type: str = MARKDOWN) -> fakes.DictConnector:
    connector = fakes.DictConnector({ID: text})
    connector.media_types[ID] = media_type
    return connector


async def stored(store: fakes.MemoryIngestStore) -> Document:
    document = await store.find_document("memory", ID)
    assert document is not None
    return document


async def test_a_declared_memory_is_stored_with_its_record() -> None:
    pipeline, store, _ = build()

    await pipeline.run(a_connector(memory(evidence=EVIDENCE + SECOND_SOURCE)))

    document = await stored(store)
    assert document.status is DocumentStatus.INDEXED
    assert document.knowledge is not None
    assert document.knowledge.record is not None
    assert document.knowledge.record.evidence_status is EvidenceStatus.CORROBORATED
    assert document.knowledge.record.evidence[0].locator == LOCATOR


async def test_a_refused_record_is_stored_with_its_reason_and_costs_the_document_nothing() -> None:
    """The memory is still a document: searchable, citable, and visibly missing its evidence.

    A refusal that failed the ingest would make the text of a memory unfindable over a malformed
    date in a field that has nothing to do with whether the text is useful.
    """
    pipeline, store, _ = build()

    await pipeline.run(a_connector(memory(knowledge_type="recipe")))

    document = await stored(store)
    assert document.status is DocumentStatus.INDEXED
    assert document.knowledge is not None
    assert document.knowledge.record is None
    assert "knowledge_type" in document.knowledge.unavailable_reason
    assert store.chunks[document.id], "the text was indexed"


async def test_an_ordinary_markdown_document_acquires_no_key() -> None:
    """A null under the key would be a stored value where there was none, so there is no key."""
    pipeline, store, _ = build()

    await pipeline.run(a_connector("---\nstatus: draft\nsource: x\n---\n\n# A page\n\nText.\n"))

    assert KNOWLEDGE_KEY not in (await stored(store)).metadata


async def test_only_markdown_is_read_for_a_record() -> None:
    """The same bytes under another media type are not front matter, whatever they look like."""
    pipeline, store, _ = build()

    await pipeline.run(a_connector(memory(), media_type=MEDIA_TYPE))

    assert KNOWLEDGE_KEY not in (await stored(store)).metadata


async def test_a_revision_that_drops_its_evidence_does_not_keep_the_old_citations() -> None:
    """The stored metadata merges *under* what was just fetched, so absence cannot remove a key.

    Without the explicit overwrite a memory edited to cite nothing would go on showing the
    previous revision's sources — a claim that reads as better supported than its own text says.
    """
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store)
    connector = a_connector(memory())
    await pipeline.run(connector)
    assert (await stored(store)).knowledge is not None

    connector.documents[ID] = "---\nname: x\n---\n\n# Plain now\n\nNo evidence any more.\n"
    await pipeline.run(connector)

    after = await stored(store)
    assert after.knowledge is None
    assert after.metadata[KNOWLEDGE_KEY] is None, "overwritten, not left behind"


async def test_a_revision_that_breaks_its_record_does_not_keep_the_valid_one() -> None:
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store)
    connector = a_connector(memory())
    await pipeline.run(connector)

    connector.documents[ID] = memory(knowledge_type="recipe")
    await pipeline.run(connector)

    after = await stored(store)
    assert after.knowledge is not None
    assert after.knowledge.record is None, "the sources it no longer validly cites are gone"


async def test_a_re_parse_rederives_the_record_from_the_retained_bytes() -> None:
    """The file is the record, so a stored value that was lost or damaged is repaired from it.

    The stored copy is deleted outright and a re-parse must put it back — which is the whole
    meaning of the record being *derived*: nothing in the database is the only copy.
    """
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(blobs=blobs)
    await pipeline.run(a_connector(memory()))
    document = await stored(store)
    before = document.knowledge
    assert before is not None
    damaged = document.model_copy(
        update={"metadata": {k: v for k, v in document.metadata.items() if k != KNOWLEDGE_KEY}}
    )
    assert damaged.knowledge is None

    await re_parse([damaged], pipeline=pipeline, blobs=blobs)

    assert (await stored(store)).knowledge == before


async def test_a_memory_replaced_by_non_markdown_loses_its_evidence() -> None:
    """The early return for other media types used to skip the removal."""
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store)
    connector = a_connector(memory())
    await pipeline.run(connector)
    assert (await stored(store)).knowledge is not None

    connector.documents[ID] = "plain text now\n"
    connector.media_types[ID] = MEDIA_TYPE
    await pipeline.run(connector)

    assert (await stored(store)).knowledge is None


async def test_a_non_markdown_document_that_never_had_a_record_gains_no_key() -> None:
    pipeline, store, _ = build()

    await pipeline.run(a_connector("plain\n", media_type=MEDIA_TYPE))

    assert KNOWLEDGE_KEY not in (await stored(store)).metadata
