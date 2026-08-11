"""Corpus pinning: the two ways "same documents on both sides" quietly stops being true."""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.evaluation.corpus import CorpusVersion, corpus_version_of, digest_of
from tests.evaluation.pipeline import SCOPE, build_corpus
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from manicule.storage.docstore import SqliteDocStore


def a_version(**overrides: object) -> CorpusVersion:
    payload: dict[str, object] = {
        "label": "knowledge-base",
        "digest": "sha256:aaa",
        "document_count": 120,
    }
    payload.update(overrides)
    return CorpusVersion.model_validate(payload)


def test_two_sides_with_different_labels_may_not_be_compared() -> None:
    """A comparison across different content is a measurement of the content."""
    assert a_version().disagreement_with(a_version(label="other")) is not None


def test_the_same_label_over_different_content_is_caught_by_the_digest() -> None:
    """The label is the thing that does not change when the corpus does.

    Documents get added, the label still says ``knowledge-base``, and last month's win rate
    gets compared against this month's over different content. The digest is what notices.
    """
    moved = a_version().disagreement_with(a_version(digest="sha256:bbb"))

    assert moved is not None
    assert "stale" in moved


def test_a_missing_digest_is_absence_rather_than_agreement() -> None:
    """A system that cannot compute one is not thereby confirmed to match.

    Not a refusal, because one side may be a system manicule cannot introspect — but not
    silence either: the report prints that the corpus identity was asserted, not verified.
    """
    external = a_version(digest=None)

    assert a_version().disagreement_with(external) is None
    assert not a_version().agrees_verifiably_with(external)
    assert a_version().agrees_verifiably_with(a_version())


def test_chunk_counts_that_differ_are_recorded_and_not_refused() -> None:
    """Two systems chunk differently, and refusing on that would block the whole method."""
    assert a_version(chunk_count=900).disagreement_with(a_version(chunk_count=400)) is None


def test_a_digest_does_not_depend_on_the_order_documents_were_enumerated_in() -> None:
    """Two stores that page differently must agree, or every comparison across them fails."""
    entries = [("d2", "h2"), ("d1", "h1"), ("d3", "h3")]

    assert digest_of(entries) == digest_of(sorted(entries, reverse=True))


def test_a_corpus_whose_documents_were_all_edited_is_a_different_corpus() -> None:
    """Same ids, same count, different content. Ids alone would call it unchanged."""
    assert digest_of([("d1", "before")]) != digest_of([("d1", "after")])


async def test_a_version_read_from_a_store_describes_what_it_holds(
    store: SqliteDocStore,
) -> None:
    chunks = await build_corpus(store)

    version = await corpus_version_of(
        store, label="fixture", workspace_ids=SCOPE, embed_fingerprint="fake/embedder@1"
    )

    assert version.document_count == 60
    assert version.chunk_count == len(chunks)
    assert version.digest is not None
    assert version.embed_fingerprint == "fake/embedder@1"


async def test_a_store_that_gains_a_document_reports_a_different_version(
    store: SqliteDocStore,
) -> None:
    """This is what keeps results comparable as the corpus grows: it stops being silent."""
    await build_corpus(store)
    before = await corpus_version_of(store, label="fixture", workspace_ids=SCOPE)

    document = make_document(source="fixture", source_id="late-arrival", title="late arrival")
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "arrived later")])
    after = await corpus_version_of(store, label="fixture", workspace_ids=SCOPE)

    assert before.digest != after.digest
    assert before.disagreement_with(after) is not None
