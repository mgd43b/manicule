"""Scoped storage: one workspace's glossary, one collection's glossary, and no other's.

``bugs/bug2.md`` §4 makes contamination a correctness property rather than a convenience, so
every isolation claim here is demonstrated twice: once against the real store, and once against
a source written to ignore its scope entirely. Without the second, a test that passes proves
only that the fixture had nothing to leak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.content import DocumentStatus
from manicule.core.glossary import DefinitionForm, GlossaryEntry
from manicule.core.ids import glossary_entry_id
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.retrieval.expansion import ExpansionPolicy, resolve_expansion
from manicule.retrieval.ports import GlossarySource
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.scoped import CrossWorkspaceCollisionError
from tests.glossary import corpus, system

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.retrieval import Filter

EXPANSION = corpus.EXPANSION
OTHER = "Nightly Operations Watchdog"

pytestmark = pytest.mark.usefixtures("store")


class LeakySource:
    """A glossary source that ignores the filter **and** the limit. Deliberately broken.

    The same shape as :class:`~tests.app.fakes.LeakyStore`, and broken the same two ways for the
    same reason. Ignoring the filter is what a source written without its ``WHERE`` clause does.
    Ignoring the limit as well is what stops a caller's "some of what I asked for is missing"
    check catching a foreign entry by accident — leaving the *identity* check, the one that
    would still fire if every scoping clause in storage were deleted, actually exercised.

    A test using a correct source passes whether or not the caller checks anything.
    """

    def __init__(self, *entries: GlossaryEntry) -> None:
        self._entries = entries

    async def entries_for(
        self,
        keys: Sequence[str],
        filter: Filter,  # noqa: A002 - mirrors the protocol it breaks
    ) -> Sequence[GlossaryEntry]:
        del keys, filter
        return list(self._entries)


def an_entry(
    document_id: str,
    chunk_id: str,
    *,
    acronym: str = "NOW",
    expansion: str = EXPANSION,
    confidence: float = 0.95,
    aliases: tuple[str, ...] = (),
) -> GlossaryEntry:
    return GlossaryEntry(
        acronym=acronym,
        display=acronym,
        expansion=expansion,
        document_id=document_id,
        chunk_id=chunk_id,
        location="Glossary",
        form=DefinitionForm.EM_DASH,
        confidence=confidence,
        aliases=aliases,
    )


async def a_glossary(
    store: SqliteDocStore,
    source_id: str,
    line: str,
    *,
    workspace_id: str = system.WORKSPACE,
    title: str = "Glossary of terms",
) -> str:
    """Index a one-line glossary and return its document id."""
    document, _ = await system.index(store, source_id, title, [line], workspace_id=workspace_id)
    return document.id


# --- the basics ---------------------------------------------------------------------------


async def test_an_entry_round_trips_with_everything_needed_to_cite_it(
    store: SqliteDocStore,
) -> None:
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    entries = await store.glossary_entries(document_id)

    assert len(entries) == 1
    entry = entries[0]
    assert (entry.acronym, entry.expansion) == ("NOW", EXPANSION)
    assert entry.document_id == document_id
    assert entry.chunk_id, "an expansion without a chunk is one nobody can check"
    assert entry.location == "Glossary of terms"
    assert entry.form is DefinitionForm.EM_DASH


async def test_the_real_store_satisfies_the_port_retrieval_asks_for(
    store: SqliteDocStore,
) -> None:
    """Structural, so nothing declares it — which is exactly why it is asserted."""
    assert isinstance(store, GlossarySource)


async def test_re_ingesting_replaces_a_documents_entries_rather_than_adding_to_them(
    store: SqliteDocStore,
) -> None:
    """The failure this prevents is a definition that is wrong, confident and still cited."""
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    await system.index(store, "glossary", "Glossary of terms", [f"NOW — {OTHER}"])

    entries = await store.glossary_entries(document_id)

    assert [entry.expansion for entry in entries] == [OTHER]


async def test_a_glossary_that_stops_defining_anything_stops_answering(
    store: SqliteDocStore,
) -> None:
    """A write that only ever *added* would leave the old definition answering forever."""
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    await system.index(store, "glossary", "Glossary of terms", ["This page has been emptied."])

    assert await store.glossary_entries(document_id) == []
    assert await store.entries_for(["NOW"], system.query_filter()) == []


async def test_writing_an_empty_list_clears_a_document_without_touching_its_chunks(
    store: SqliteDocStore,
) -> None:
    """The method's own contract, exercised where the chunk cascade cannot do the work.

    Written after a mutation showed nothing covered it: on this store ``replace_chunks``
    already destroys the entries through the foreign key, so every other test passed with the
    explicit clear removed. A store whose chunks lived elsewhere would then keep a whole
    glossary for a page that no longer states any of it.
    """
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    assert await store.glossary_entries(document_id)

    await store.replace_glossary_entries(
        document_id, [], fingerprint=glossary_fingerprint().canonical()
    )

    assert await store.glossary_entries(document_id) == []
    assert await store.document_chunks(document_id), "the chunks are deliberately left alone"


async def test_entries_are_refused_when_they_name_another_document(
    store: SqliteDocStore,
) -> None:
    """Rewriting the attribution would file one document's vocabulary under another's scope."""
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    with pytest.raises(ValueError, match="name document"):
        await store.replace_glossary_entries(
            document_id,
            [an_entry("some-other-document", "chunk-x")],
            fingerprint=glossary_fingerprint().canonical(),
        )


async def test_deleting_a_documents_chunks_takes_its_definitions_with_them(
    store: SqliteDocStore,
) -> None:
    """The cascade, checked rather than assumed.

    An entry cites a chunk. ``replace_chunks`` rewrites chunks on every re-parse, so without
    the foreign key an edited glossary would keep citing a chunk id that no longer resolves.
    """
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    await store.replace_chunks(document_id, [])

    assert await store.glossary_entries(document_id) == []


async def test_a_soft_deleted_document_contributes_no_vocabulary(
    store: SqliteDocStore,
) -> None:
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    assert await store.entries_for(["NOW"], system.query_filter())

    await store.soft_delete_document(document_id)

    assert await store.entries_for(["NOW"], system.query_filter()) == []


async def test_a_document_still_being_ingested_contributes_no_vocabulary(
    store: SqliteDocStore,
) -> None:
    """Its chunks and its vectors need not agree yet, so a definition read out of it would
    cite a passage a search cannot return."""
    document_id = await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    await store.set_status(document_id, DocumentStatus.EMBEDDING)

    assert await store.entries_for(["NOW"], system.query_filter()) == []


async def test_an_alias_finds_its_entry_through_the_store(store: SqliteDocStore) -> None:
    await a_glossary(store, "glossary", f"The {EXPANSION} (NOW, NETOPS) holds the runbooks.")

    found = await store.entries_for(["NETOPS"], system.query_filter())

    assert [entry.expansion for entry in found] == [EXPANSION]
    assert found[0].aliases == ("NETOPS",)


async def test_an_entry_id_is_derived_rather_than_generated() -> None:
    """Re-ingesting an unchanged glossary must produce the same rows, not a second copy."""
    first = glossary_entry_id("chunk-1", "NOW", EXPANSION)
    again = glossary_entry_id("chunk-1", "NOW", EXPANSION)
    edited = glossary_entry_id("chunk-1", "NOW", OTHER)

    assert first == again
    assert first != edited


# --- workspace isolation ---------------------------------------------------------------------


async def test_one_workspaces_glossary_is_invisible_to_another(engine: AsyncEngine) -> None:
    """``bugs/bug2.md`` §4, the workspace half. Two handles, one database."""
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()

    await a_glossary(alpha, "glossary", f"NOW — {EXPANSION}", workspace_id="alpha")
    await a_glossary(beta, "glossary", f"NOW — {OTHER}", workspace_id="beta")

    assert [entry.expansion for entry in await alpha.entries_for(["NOW"], _scope("alpha"))] == [
        EXPANSION
    ]
    assert [entry.expansion for entry in await beta.entries_for(["NOW"], _scope("beta"))] == [OTHER]


async def test_a_filter_naming_another_workspace_is_refused_rather_than_answered(
    store: SqliteDocStore,
) -> None:
    """Answering the question that was not asked is the shape a cross-tenant leak takes."""
    await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    with pytest.raises(CrossWorkspaceCollisionError):
        await store.entries_for(["NOW"], _scope("somebody-else"))


async def test_two_workspaces_defining_one_term_differently_is_not_a_conflict(
    engine: AsyncEngine,
) -> None:
    """The point of isolation, stated as the thing that must *not* happen.

    A tenant whose glossary disagrees with another tenant's has no disagreement: it cannot see
    theirs. Reporting one would leak the existence — and the wording — of another tenant's
    definition.
    """
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()
    await a_glossary(alpha, "glossary", f"NOW — {EXPANSION}", workspace_id="alpha")
    await a_glossary(beta, "glossary", f"NOW — {OTHER}", workspace_id="beta")

    result = await _expand("What is NOW?", alpha, _scope("alpha"))

    assert result.fired
    assert result.conflicts == ()
    assert result.matches[0].entry.expansion == EXPANSION


# --- collection isolation ----------------------------------------------------------------------


async def test_a_search_scoped_to_a_collection_sees_only_that_collections_glossary(
    store: SqliteDocStore,
) -> None:
    """``bugs/bug2.md`` §4, the collection half, and its regression case for conflicts.

    The same workspace holds two disagreeing definitions. Scoped to either collection there is
    one answer and no conflict; scoped to the workspace there are two and neither is chosen.
    """
    left = await a_glossary(store, "left", f"NOW — {EXPANSION}", title="Left glossary")
    right = await a_glossary(store, "right", f"NOW — {OTHER}", title="Right glossary")
    inner = await store.create_collection("inner")
    outer = await store.create_collection("outer")
    await store.add_to_collection(inner.id, [left])
    await store.add_to_collection(outer.id, [right])

    scoped_in = await _expand(
        "What is NOW?", store, system.query_filter(collection_ids=frozenset({inner.id}))
    )
    scoped_out = await _expand(
        "What is NOW?", store, system.query_filter(collection_ids=frozenset({outer.id}))
    )
    unscoped = await _expand("What is NOW?", store, system.query_filter())

    assert scoped_in.matches[0].entry.expansion == EXPANSION
    assert scoped_in.conflicts == ()
    assert scoped_out.matches[0].entry.expansion == OTHER
    assert scoped_out.conflicts == ()
    assert not unscoped.fired
    assert set(unscoped.conflicts[0].expansions) == {EXPANSION, OTHER}


async def test_two_definitions_inside_one_collection_still_conflict(
    store: SqliteDocStore,
) -> None:
    """The regression case for conflicting expansions in the *same* collection.

    Narrowing the scope is not a tie-break. A collection holding both definitions holds a
    disagreement, and picking the more confident one there would be the silent choice under
    another name.
    """
    left = await a_glossary(store, "left", f"NOW — {EXPANSION}", title="Left glossary")
    right = await a_glossary(store, "right", f"NOW — {OTHER}", title="Right glossary")
    both = await store.create_collection("both")
    await store.add_to_collection(both.id, [left, right])

    result = await _expand(
        "What is NOW?", store, system.query_filter(collection_ids=frozenset({both.id}))
    )

    assert not result.fired
    assert set(result.conflicts[0].expansions) == {EXPANSION, OTHER}


async def test_a_scope_that_matches_no_document_consults_no_glossary(
    store: SqliteDocStore,
) -> None:
    """An empty collection is the narrowest request anybody can make.

    Collapsing "resolved to nothing" into "no restriction" would make it the widest — the whole
    workspace's vocabulary, consulted on behalf of a collection holding none of it.
    """
    await a_glossary(store, "glossary", f"NOW — {EXPANSION}")
    empty = await store.create_collection("empty")

    found = await store.entries_for(
        ["NOW"], system.query_filter(collection_ids=frozenset({empty.id}))
    )

    assert found == []


async def test_a_filter_by_source_restricts_the_vocabulary(store: SqliteDocStore) -> None:
    """Document-level fields are honored rather than dropped."""
    await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    assert await store.entries_for(["NOW"], system.query_filter(sources=frozenset({"fixture"})))
    assert (
        await store.entries_for(["NOW"], system.query_filter(sources=frozenset({"elsewhere"})))
        == []
    )


async def test_a_chunk_level_restriction_does_not_hide_the_vocabulary(
    store: SqliteDocStore,
) -> None:
    """``kinds`` restricts which passages come back, not what an acronym means.

    Refusing here would mean a query for table passages could not be told what a term in its own
    text stands for. The restriction is applied where it belongs — to the passages — and
    ``test_retrieval.py`` holds the promotion path to it.
    """
    from manicule.core.content import BlockKind  # noqa: PLC0415 - one test needs the enum

    await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    found = await store.entries_for(
        ["NOW"], system.query_filter(kinds=frozenset({BlockKind.TABLE}))
    )

    assert [entry.expansion for entry in found] == [EXPANSION]


# --- the leaky control ---------------------------------------------------------------------------


async def test_a_source_that_ignores_its_scope_is_what_the_guards_are_measured_against() -> None:
    """The control. It leaks, and it is supposed to.

    Every isolation test above passes against the real store. This one demonstrates that a
    source *can* fail those assertions — so the passes are the store's doing, not the fixture's.
    """
    leaky = LeakySource(an_entry("foreign-document", "foreign-chunk", expansion=OTHER))

    found = await leaky.entries_for(["ZZQX"], system.query_filter())

    assert [entry.expansion for entry in found] == [OTHER], (
        "the control must return an entry nothing asked for, or it is not a control"
    )


async def test_the_real_store_refuses_what_the_leaky_one_returns(
    store: SqliteDocStore,
) -> None:
    """The same call, the same arguments, against the store that applies its clauses."""
    await a_glossary(store, "glossary", f"NOW — {EXPANSION}")

    assert await store.entries_for(["ZZQX"], system.query_filter()) == []


async def test_an_expansion_built_on_a_leaky_source_still_carries_its_provenance() -> None:
    """Even wrong, an entry names where it came from — which is how the leak is *findable*.

    A feature that presented an expansion with no source would leak silently. This asserts the
    weaker but load-bearing property: whatever reaches a reader can be traced to a document.
    """
    leaky = LeakySource(an_entry("foreign-document", "foreign-chunk"))

    result = await _expand("What is NOW?", leaky, system.query_filter())

    assert result.fired
    assert result.matches[0].entry.document_id == "foreign-document"


# --- helpers ------------------------------------------------------------------------------------


def _scope(workspace_id: str) -> Filter:
    return system.query_filter(workspace_id=workspace_id)


async def _expand(text: str, source: GlossarySource, filter: Filter):  # noqa: A002, ANN202
    from manicule.core.retrieval import Query  # noqa: PLC0415 - a helper, not a module import

    return await resolve_expansion(Query(text=text, filter=filter), source, ExpansionPolicy())
