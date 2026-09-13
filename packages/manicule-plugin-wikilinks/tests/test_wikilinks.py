"""The rules, and the two directions the middleware writes edges in.

Every assertion about a stored edge goes through a real store, so "an edge was written" means a
row that survived the foreign keys, the workspace check and the idempotence the schema and
:class:`~manicule.storage.relations.RelationsMixin` enforce — rather than a call that was made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from manicule_plugin_wikilinks import (
    EXTRACTOR,
    WikilinkConfig,
    WikilinkMiddleware,
    rules_digest,
    slug_of,
)
from manicule_plugin_wikilinks.links import Shape, links_in, normalize

from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.organization import ChunkRelationType
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = "/corpus/memory"


def a_document(slug: str) -> Document:
    """A markdown memory at ``<root>/<slug>.md``, identified the way the corpus identifies one."""
    source_id = f"{ROOT}/{slug}.md"
    return Document(
        id=document_id(DEFAULT_WORKSPACE, "memories", source_id),
        source="memories",
        source_id=source_id,
        uri=f"file://{source_id}",
        title=f"{slug}.md",
        content_hash=content_hash(source_id),
        media_type="text/markdown",
        status=DocumentStatus.INDEXED,
    )


def a_chunk(document: Document, position: int, text: str) -> Chunk:
    return Chunk(
        id=chunk_id(document.id, position, text),
        document_id=document.id,
        text=text,
        embed_text=text,
        anchor=HeadingAnchor(path=("Body",), fragment=None),
        heading_path=("Body",),
        kind=BlockKind.PROSE,
        position=position,
        token_count=max(1, len(text.split())),
    )


async def seed(store: SqliteDocStore, slug: str, *texts: str) -> Document:
    """Store one document and its chunks, as a completed ingest would leave them."""
    document = a_document(slug)
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id, [a_chunk(document, index, text) for index, text in enumerate(texts)]
    )
    return document


def middleware(store: SqliteDocStore, **config: object) -> WikilinkMiddleware:
    return WikilinkMiddleware(store, WikilinkConfig.model_validate(config))


async def edges(store: SqliteDocStore, chunk: Chunk) -> Sequence[tuple[str, str, str]]:
    """Every edge touching ``chunk``, as comparable tuples."""
    return [
        (edge.source_chunk_id, edge.target_chunk_id, edge.relation_type.value)
        for edge in await store.related(chunk.id)
    ]


# --- the rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("project_backoff", "project-backoff"),
        ("project-backoff", "project-backoff"),
        ("Project Backoff", "project-backoff"),
        ("project_backoff.md", "project-backoff"),
        ("PROJECT_BACKOFF.MD", "project-backoff"),
        ("  project__backoff  ", "project-backoff"),
        ("project_backoff|the policy", "project-backoff"),
        ("project_backoff#Retries", "project-backoff"),
    ],
)
def test_every_spelling_of_one_target_normalizes_to_the_same_slug(
    written: str, expected: str
) -> None:
    """The corpus is inconsistent between hyphens and underscores and both are in real use.

    Literal matching would silently disconnect roughly half the graph — silently, because an
    unresolved link is a legitimate state and looks exactly like a link nobody has written the
    target for yet.
    """
    assert normalize(written) == expected


@pytest.mark.parametrize(
    ("line", "shape"),
    [
        ("The client retries twice. See [[project_backoff]].", Shape.PROSE),
        ("- [[project_backoff]]", Shape.DECLARED),
        ("* [[project_backoff]]", Shape.DECLARED),
        ("1. [[project_backoff]]", Shape.DECLARED),
        ("relates_to [[project_backoff]]", Shape.DECLARED),
        ("Related: [[project_backoff]], [[project_metallb]]", Shape.DECLARED),
        ("A long sentence about retries that ends with [[project_backoff]]", Shape.PROSE),
    ],
)
def test_a_declared_link_and_a_mention_are_told_apart_by_the_line_they_are_on(
    line: str, shape: Shape
) -> None:
    """Two shapes, two claims, and they are deliberately not collapsed.

    A declared link is the author asserting a relationship; a mention is evidence that two
    documents are about related things. Ranking by one and ranking by the other give different
    answers on the same corpus, which is why the distinction has to survive into the store.
    """
    assert [found.shape for found in links_in(line)] == [shape] * line.count("[[")


def test_a_link_with_no_target_is_not_a_link() -> None:
    """``[[]]`` addresses nothing, and a target that normalizes to nothing would match nothing
    in a way that looks like matching everything else that also normalizes to nothing."""
    assert links_in("empty [[]] and [[ | alias ]] and [[#heading]]") == []


def test_the_extractor_states_a_name_and_a_digest_of_its_own_rules(store: SqliteDocStore) -> None:
    """The name tells two strategies apart; the digest tells two versions of one apart.

    A digest alone could not: two extractors hash differently, but so does one extractor after a
    typo fix, and a reader of the stored column would have no way to tell which kind of change
    they were looking at.
    """
    rules = middleware(store).relation_rules()
    assert rules.extractor == EXTRACTOR
    assert rules.rules == rules_digest()
    assert len(rules.rules) == 64, "a sha256 hex digest"


def test_a_document_is_found_by_its_filename_stem_rather_than_its_title() -> None:
    """Identity is the slug. A retitled document keeps every link into it."""
    document = a_document("project_backoff").model_copy(update={"title": "Something else"})
    assert slug_of(document) == "project-backoff"


# --- outgoing edges ------------------------------------------------------------------------------


async def test_a_link_becomes_an_edge_from_the_linking_chunk_to_the_targets_first(
    store: SqliteDocStore,
) -> None:
    """A link addresses a document and an edge addresses a chunk, so one had to be chosen.

    The first chunk: the document's opening, which for one self-contained fact per file is the
    fact, and the only choice that does not depend on how the link was worded — so two links to
    one document from two places agree about where they point.
    """
    target = await seed(store, "project_backoff", "Back off exponentially.", "A second chunk.")
    source = await seed(store, "project_retry", "See [[project_backoff]] for the schedule.")
    chunks = await store.document_chunks(source.id)
    target_chunks = await store.document_chunks(target.id)

    await middleware(store).after_store(source)

    assert await edges(store, chunks[0]) == [
        (chunks[0].id, target_chunks[0].id, ChunkRelationType.MENTIONS.value)
    ]


async def test_re_ingesting_the_same_document_produces_one_edge_rather_than_two(
    store: SqliteDocStore,
) -> None:
    """Idempotent, and asserted against the store rather than against the middleware.

    Re-ingest is the ordinary case — a connector sync touches every document it can see — so an
    extractor that appended would double the graph on the second sync and keep going.
    """
    await seed(store, "project_backoff", "Back off exponentially.")
    source = await seed(store, "project_retry", "- [[project_backoff]]")
    chunks = await store.document_chunks(source.id)

    hook = middleware(store)
    await hook.after_store(source)
    await hook.after_store(source)

    written = await edges(store, chunks[0])
    assert len(written) == 1
    assert written[0][2] == ChunkRelationType.LINKS_TO.value


async def test_a_document_linking_to_itself_writes_nothing(store: SqliteDocStore) -> None:
    """An edge from a document to itself says nothing a reader could use.

    Worth refusing here rather than leaving to the store: the store refuses a chunk related to
    *itself* and would happily write chunk two of a document to chunk one of the same document,
    which is a row that exists and means nothing.
    """
    document = await seed(store, "project_retry", "Header.", "See [[project_retry]] again.")
    chunks = await store.document_chunks(document.id)

    await middleware(store).after_store(document)

    assert await edges(store, chunks[1]) == []


async def test_a_link_to_a_document_that_does_not_exist_is_not_an_error(
    store: SqliteDocStore,
) -> None:
    """A forward reference is part of the writing convention, not a failure.

    The hook returning normally is the assertion that matters: a raise here would cost the
    document its relation lineage and leave it selected by every repair for ever, for doing
    exactly what the convention asks.
    """
    source = await seed(store, "project_retry", "See [[project_not_written_yet]].")
    chunks = await store.document_chunks(source.id)

    await middleware(store).after_store(source)

    assert await edges(store, chunks[0]) == []


# --- incoming edges ------------------------------------------------------------------------------


async def test_a_forward_reference_resolves_when_its_target_is_published(
    store: SqliteDocStore,
) -> None:
    """The half that makes an unresolved link *retained* rather than dropped.

    ``chunk_relations`` has foreign keys to ``chunks``, so an edge to a document that does not
    exist cannot be stored — there is nothing to point at. Of the two available designs, this is
    "re-resolve on each target publish": when the target arrives, the documents that already
    link to it get their edges.
    """
    source = await seed(store, "project_retry", "See [[project_backoff]] for the schedule.")
    await middleware(store).after_store(source)
    source_chunks = await store.document_chunks(source.id)
    assert await edges(store, source_chunks[0]) == [], "the target does not exist yet"

    target = await seed(store, "project_backoff", "Back off exponentially.")
    await middleware(store).after_store(target)

    target_chunks = await store.document_chunks(target.id)
    assert await edges(store, source_chunks[0]) == [
        (source_chunks[0].id, target_chunks[0].id, ChunkRelationType.MENTIONS.value)
    ]


async def test_the_inbound_pass_writes_nothing_for_a_chunk_that_merely_says_the_words(
    store: SqliteDocStore,
) -> None:
    """Every candidate is re-scanned with the real rules before an edge is written.

    The inbound search is lexical, so it matches a chunk containing the words rather than a chunk
    containing the link. Treating a hit as a link would invent edges out of prose that happened
    to mention a document's name — which, in a corpus whose slugs are made of ordinary words, is
    most of it.
    """
    source = await seed(store, "project_retry", "The project backoff schedule is documented.")
    await middleware(store).after_store(source)
    target = await seed(store, "project_backoff", "Back off exponentially.")

    await middleware(store).after_store(target)

    chunks = await store.document_chunks(source.id)
    assert await edges(store, chunks[0]) == []


async def test_the_inbound_pass_can_be_switched_off(store: SqliteDocStore) -> None:
    """``inbound_limit = 0`` leaves forward references to the repair, and is a real choice.

    Asserted because a bound whose zero did nothing would be a setting that reads like a switch
    and is not one.
    """
    source = await seed(store, "project_retry", "See [[project_backoff]].")
    await middleware(store, inbound_limit=0).after_store(source)
    target = await seed(store, "project_backoff", "Back off exponentially.")

    await middleware(store, inbound_limit=0).after_store(target)

    chunks = await store.document_chunks(source.id)
    assert await edges(store, chunks[0]) == []


async def test_a_hyphen_and_an_underscore_spelling_reach_the_same_target(
    store: SqliteDocStore,
) -> None:
    """Both spellings are in the corpus and both must connect, in both directions.

    Two links from two documents, written differently, to one target whose own filename uses the
    third spelling again — because normalization that only worked between the link and itself
    would look correct on a fixture where every name agreed.
    """
    hyphenated = await seed(store, "feedback_one", "- [[feedback-commit-everything]]")
    underscored = await seed(store, "feedback_two", "- [[feedback_commit_everything]]")
    hook = middleware(store)
    await hook.after_store(hyphenated)
    await hook.after_store(underscored)

    target = await seed(store, "feedback-commit-everything", "Commit everything.")
    await hook.after_store(target)

    target_chunks = await store.document_chunks(target.id)
    for source in (hyphenated, underscored):
        chunks = await store.document_chunks(source.id)
        assert await edges(store, chunks[0]) == [
            (chunks[0].id, target_chunks[0].id, ChunkRelationType.LINKS_TO.value)
        ]


async def test_a_source_outside_the_configured_scope_is_not_a_link_target(
    store: SqliteDocStore,
) -> None:
    """A slug is a filename stem, and stems are not unique across sources.

    Without the scope, a link written in a memory would resolve into whatever else the workspace
    indexes that happens to have a file of the same name — a **wrong** edge rather than a missing
    one, which is the direction that matters. Asserted by seeding the same stem under two sources
    and configuring only one.
    """
    elsewhere = Document(
        id=document_id(DEFAULT_WORKSPACE, "wiki", "/wiki/project_backoff.md"),
        source="wiki",
        source_id="/wiki/project_backoff.md",
        uri="file:///wiki/project_backoff.md",
        title="project_backoff.md",
        content_hash=content_hash("wiki"),
        media_type="text/markdown",
        status=DocumentStatus.INDEXED,
    )
    await store.upsert_document(elsewhere)
    await store.replace_chunks(elsewhere.id, [a_chunk(elsewhere, 0, "A wiki page.")])
    source = await seed(store, "project_retry", "See [[project_backoff]].")

    await middleware(store, sources=("memories",)).after_store(source)

    chunks = await store.document_chunks(source.id)
    assert await edges(store, chunks[0]) == [], "the only candidate was outside the scope"


async def test_a_document_outside_the_scope_does_not_become_a_target_by_arriving(
    store: SqliteDocStore,
) -> None:
    """The other half of the scope, and the half a single-direction check would miss.

    A document is registered as a link target when it is stored, and searched for inbound links
    at the same moment. If only one of those consulted the scope, a document outside it would be
    reachable by links written after it and not by links written before it — a graph whose shape
    depends on ingest order. Here the out-of-scope document arrives **last**, which is the case a
    build-time-only check passes.
    """
    source = await seed(store, "project_retry", "See [[project_backoff]].")
    hook = middleware(store, sources=("memories",))
    await hook.after_store(source)

    elsewhere = Document(
        id=document_id(DEFAULT_WORKSPACE, "wiki", "/wiki/project_backoff.md"),
        source="wiki",
        source_id="/wiki/project_backoff.md",
        uri="file:///wiki/project_backoff.md",
        title="project_backoff.md",
        content_hash=content_hash("wiki-later"),
        media_type="text/markdown",
        status=DocumentStatus.INDEXED,
    )
    await store.upsert_document(elsewhere)
    await store.replace_chunks(elsewhere.id, [a_chunk(elsewhere, 0, "A wiki page.")])
    await hook.after_store(elsewhere)

    chunks = await store.document_chunks(source.id)
    assert await edges(store, chunks[0]) == []
