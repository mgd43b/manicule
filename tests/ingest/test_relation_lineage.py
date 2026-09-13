"""What a full ingest records about the extractor that derived a document's chunk relations.

The stage is a plugin's, and the *lineage* is this repository's — so what is tested here is the
half that stays whether or not anybody installs an extractor: the fingerprint a chain produces,
where the pipeline stamps it, the three states that are easy to conflate, and the selector a
repair reads. ``packages/manicule-plugin-wikilinks/tests`` covers the extraction itself.

Three cases are easy to write and wrong, and they are the same three the glossary stage had:
a document that contains no links, a document ingested with no extractor configured, and a
document whose extractor raised. All three would work if lineage were recorded beside the edges,
and all three are the states in which there are no edges to record it beside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from manicule.core.errors import MiddlewareViolationError
from manicule.core.fingerprints import (
    EXTRACTION_DISABLED,
    RelationFingerprint,
    RelationRules,
)
from manicule.core.protocols import Middleware
from manicule.ingest import reindex
from manicule.ingest.middleware import MiddlewareRunner
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

if TYPE_CHECKING:
    from manicule.core.content import Document

PROSE = "The scheduler restarts nightly, which is fine."


class Extractor(Middleware):
    """A middleware that says it extracts relations, and records what it was asked to look at.

    It writes no edges. Everything asserted here is about the *lineage* — which is deliberately
    not the extractor's to write, so a stand-in that writes nothing exercises exactly the same
    code path the real one does.
    """

    def __init__(self, name: str = "links", rules: str = "v1") -> None:
        self.name = name
        self._rules = rules
        self.seen: list[str] = []

    def relation_rules(self) -> RelationRules:
        return RelationRules(extractor="test-extractor", rules=self._rules)

    @override
    async def after_store(self, document: Document) -> None:
        self.seen.append(document.id)


class FailingExtractor(Extractor):
    """An extractor that raises, which must cost the lineage and not the document."""

    @override
    async def after_store(self, document: Document) -> None:
        self.seen.append(document.id)
        msg = "the extractor could not read this document"
        raise RuntimeError(msg)


class Bystander(Middleware):
    """A configured hook that extracts nothing. It is still part of the identity."""

    name = "bystander"


# --- what a chain produces -----------------------------------------------------------------


def test_a_chain_with_no_extractor_records_a_state_rather_than_an_absence() -> None:
    """``disabled`` is a value somebody reads, and ``NULL`` means "nobody has looked".

    Collapsing the two would make installing an extractor a silent no-op on everything already
    stored: the corpus would carry no lineage either way, so the repair would have nothing to
    compare against and nothing to select.
    """
    lineage = MiddlewareRunner([Bystander()]).relation_lineage()

    assert lineage.extractor == EXTRACTION_DISABLED
    assert lineage.extracts is False
    assert lineage == RelationFingerprint.disabled()
    assert lineage.middleware == (), (
        "a chain that did not run must not be folded into the disabled state, or configuring an "
        "unrelated hook would churn the lineage of every document nobody extracted anything for"
    )


def test_an_extractors_identity_carries_the_whole_configured_chain() -> None:
    """Not only the extractor, and not only the hooks that declare they rewrite embedded text.

    An edge names a *chunk*, so which chunk holds a link is part of what is stored — and chunk
    boundaries follow from block metadata any hook may rewrite in ``after_parse``, carrying no
    declaration at all. Filtering on ``mutates_embedded_text`` would be a guard that looks right
    and covers the one field extraction never reads.
    """
    lineage = MiddlewareRunner([Bystander(), Extractor()]).relation_lineage()

    assert lineage.extractor == "test-extractor"
    assert lineage.rules == "v1"
    assert lineage.middleware == ("bystander@", "links@")


def test_reordering_configuration_does_not_invalidate_a_corpus() -> None:
    """Sorted, because listing two hooks the other way round changes not one edge."""
    one = MiddlewareRunner([Bystander(), Extractor()]).relation_lineage()
    other = MiddlewareRunner([Extractor(), Bystander()]).relation_lineage()

    assert one.canonical() == other.canonical()


def test_two_extractors_are_refused_rather_than_merged() -> None:
    """One column records which extractor produced a document's edges, so there can be one.

    Merged, a change to either would have to invalidate both and neither could be repaired
    without re-running the other. Refused at construction, where somebody is reading
    configuration, rather than per document where it would arrive as a corpus-sized failure.
    """
    with pytest.raises(MiddlewareViolationError, match="Configure one"):
        MiddlewareRunner([Extractor("first"), Extractor("second")]).relation_lineage()


def test_a_changed_rule_changes_the_identity() -> None:
    """The whole point of the digest: a corrected rule makes the corpus visibly stale."""
    before = MiddlewareRunner([Extractor(rules="v1")]).relation_lineage()
    after = MiddlewareRunner([Extractor(rules="v2")]).relation_lineage()

    assert before.canonical() != after.canonical()
    assert before.changed_fields(after) == {"rules"}


# --- where the pipeline stamps it ------------------------------------------------------------


async def _ingest(text: str, *, middleware: list[Middleware]) -> tuple[Document, str | None]:
    """Run one document through a pipeline with this chain, and read back its relation lineage."""
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store, middleware=middleware)
    await pipeline.run(fakes.DictConnector({"note": text}))
    document = await store.find_document("memory", "note")
    assert document is not None
    return document, store.relation_lineage_by_id.get(document.id)


async def test_an_installation_with_no_extractor_stamps_every_document_disabled() -> None:
    """Which is what makes installing one a repair rather than a silence.

    Without this the corpus would be indistinguishable from one nothing has scanned, and the
    first run with an extractor would have no way to tell "never looked" from "looked and found
    nothing" — so it would either re-scan everything for ever or nothing at all.
    """
    _, lineage = await _ingest(PROSE, middleware=[])

    assert lineage == RelationFingerprint.disabled().canonical()


async def test_a_document_with_no_links_records_the_extractor_that_read_it() -> None:
    """**An empty result is a derived result**, and this is the expensive one to get wrong.

    Most of a corpus contains no links. An implementation that stamped only documents that
    produced edges would leave almost everything unstamped, every repair would select almost
    everything, and every repair would end with exactly as much outstanding as it started with.
    """
    extractor = Extractor()
    document, lineage = await _ingest(PROSE, middleware=[extractor])

    assert extractor.seen == [document.id], "the hook ran and found nothing"
    assert lineage == MiddlewareRunner([extractor]).relation_lineage().canonical()


async def test_an_extractor_that_raises_leaves_the_lineage_unwritten() -> None:
    """The document stays published and stays selected, which is the pair that matters.

    Stamping regardless would claim an extractor had run over a document it raised on — the
    "reports itself current" failure the column exists to prevent — and the document would never
    be selected again. Found by writing the stamp before the hooks and watching this pass.
    """
    extractor = FailingExtractor()
    document, lineage = await _ingest(PROSE, middleware=[extractor])

    assert extractor.seen == [document.id]
    assert lineage is None
    assert document.status.value == "indexed", "a hook's failure is not the document's"


# --- what a repair selects -------------------------------------------------------------------


async def test_the_selector_finds_everything_a_different_extractor_left_behind() -> None:
    """Including every document recording nothing, which on a first enable is the corpus.

    There is no backfill command and this is why there needs to be none: the repair selector the
    other stages already have answers the question, and ``NULL`` falls inside the selection
    rather than outside it.
    """
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store, middleware=[])
    await pipeline.run(fakes.DictConnector({"one": PROSE, "two": PROSE}))

    installed = MiddlewareRunner([Extractor()]).relation_lineage()
    selected = await reindex.select(store, relation_fingerprint=installed)

    assert len(selected) == 2, "every document was built by something else"


async def test_a_document_the_current_extractor_has_scanned_is_not_selected_again() -> None:
    """The other half, and without it the test above passes on a selector that selects all.

    This is also the assertion that pins "an empty result is a derived result": the document
    below produced no edges at all, and it must still fall *outside* the next repair.
    """
    extractor = Extractor()
    store = fakes.MemoryIngestStore()
    pipeline, _, _ = build(store=store, middleware=[extractor])
    await pipeline.run(fakes.DictConnector({"one": PROSE}))

    installed = MiddlewareRunner([extractor]).relation_lineage()
    assert await reindex.select(store, relation_fingerprint=installed) == []

    corrected = MiddlewareRunner([Extractor(rules="v2")]).relation_lineage()
    assert len(await reindex.select(store, relation_fingerprint=corrected)) == 1, (
        "changing a rule must re-select the corpus it has moved past"
    )
