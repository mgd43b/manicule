"""Collections: one document, one identity, one set of embeddings.

The property the whole feature rests on is that membership is *metadata*. A document in two
collections is the same document with the same chunk ids and therefore the same vectors, and
nothing about grouping it is allowed to touch the index. Everything else here — rename,
deletion, reconciliation, concurrency — is a way of asking that same question from a different
side.

The corpus is synthetic and local: two project directories that deliberately overlap, so a
document belonging to both is the ordinary case rather than a contrived one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from manicule.core.errors import ManiculeError, NameInUseError
from manicule.core.organisation import CollectionRule
from manicule.core.retrieval import Filter
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore
from manicule.storage.organisation import resolve_filter, rule_clause
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Chunk, Document

pytestmark = pytest.mark.anyio

# Two project directories that share a document, which is the shape the spec is about: a
# personal corpus where "the runbooks" and "the platform notes" are not disjoint.
SHARED = "shared/rotation.md"
ALPHA_ONLY = "alpha/deploy.md"
BETA_ONLY = "beta/oncall.md"


async def _corpus(store: SqliteDocStore) -> dict[str, Document]:
    """Three documents across two overlapping project directories, each with two chunks."""
    made: dict[str, Document] = {}
    for index, path in enumerate((SHARED, ALPHA_ONLY, BETA_ONLY)):
        document = await store.upsert_document(
            make_document(source="fs", source_id=path, uri=f"file:///{path}")
        )
        await store.replace_chunks(
            document.id,
            [
                make_chunk(document, 0, f"rotation policy for {path}"),
                make_chunk(document, 1, f"escalation contacts for {path}"),
            ],
        )
        made[path] = document
        del index
    return made


async def _chunk_ids(store: SqliteDocStore, document: Document) -> set[str]:
    """The chunk ids a document currently has. A chunk id is derived from its content, so this
    set changing *is* the index moving — it is the observable form of "was this re-embedded"."""
    chunks: list[Chunk] = list(await store.document_chunks(document.id))
    return {chunk.id for chunk in chunks}


# --- one identity, one set of embeddings ------------------------------------------------------


async def test_a_document_in_two_collections_is_one_document_with_one_set_of_chunks(
    store: SqliteDocStore,
) -> None:
    """Overlapping membership, which is the case the spec opens with.

    Asserted on the *ids*, not on a count: two documents with two different ids and identical
    content would also give a count of two, and that is precisely the failure — a corpus that
    duplicated a document per collection would look right in every summary.
    """
    corpus = await _corpus(store)
    shared = corpus[SHARED]
    before = await _chunk_ids(store, shared)

    alpha = await store.create_collection("alpha")
    beta = await store.create_collection("beta")
    await store.add_to_collection(alpha.id, [shared.id, corpus[ALPHA_ONLY].id])
    await store.add_to_collection(beta.id, [shared.id, corpus[BETA_ONLY].id])

    in_alpha = {document.id for document in await store.collection_documents(alpha.id)}
    in_beta = {document.id for document in await store.collection_documents(beta.id)}
    assert shared.id in in_alpha
    assert shared.id in in_beta

    holding = {held.id for held in await store.collections_for(shared.id)}
    assert holding == {alpha.id, beta.id}

    assert await _chunk_ids(store, shared) == before, (
        "the document's chunk ids changed when it joined two collections. A chunk id is "
        "derived from its content, so a changed id is a new vector: the document was "
        "re-embedded for being grouped"
    )
    assert await store.count_chunks(shared.id) == len(before)


async def test_adding_and_removing_membership_never_touches_the_index(
    store: SqliteDocStore,
) -> None:
    """The headline promise, asserted over the whole workspace rather than one document.

    ``live_chunk_count`` is the total across the corpus, so a membership change that quietly
    re-chunked *anything* moves it. Checked after every step, because a change that added and
    then removed chunks would net to zero at the end.
    """
    corpus = await _corpus(store)
    before_total = await store.live_chunk_count()
    before_each = {path: await _chunk_ids(store, document) for path, document in corpus.items()}

    collection = await store.create_collection("alpha")
    assert await store.live_chunk_count() == before_total, "creating a collection changed chunks"

    await store.add_to_collection(collection.id, [document.id for document in corpus.values()])
    assert await store.live_chunk_count() == before_total, "adding membership re-chunked"

    await store.remove_from_collection(collection.id, [corpus[BETA_ONLY].id])
    assert await store.live_chunk_count() == before_total, "removing membership re-chunked"

    for path, document in corpus.items():
        assert await _chunk_ids(store, document) == before_each[path], (
            f"{path} has different chunk ids after its membership changed"
        )


async def test_adding_the_same_document_twice_is_one_membership(store: SqliteDocStore) -> None:
    """Membership is a set. The second add reports nothing changed rather than failing."""
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")

    first = await store.add_to_collection(collection.id, [corpus[SHARED].id])
    second = await store.add_to_collection(collection.id, [corpus[SHARED].id])

    assert (first, second) == (1, 0)
    members = [document.id for document in await store.collection_documents(collection.id)]
    assert members.count(corpus[SHARED].id) == 1


# --- rename -----------------------------------------------------------------------------------


async def test_renaming_a_collection_moves_no_document_and_re_indexes_nothing(
    store: SqliteDocStore,
) -> None:
    """A name is a label on a row.

    Membership is reached through the join table and the rule, neither of which mentions the
    name, so this is really a test that no *future* rename grows an index path. The chunk ids
    are the thing that would move if one did.
    """
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha", description="worked examples")
    await store.add_to_collection(collection.id, [corpus[SHARED].id, corpus[ALPHA_ONLY].id])
    before_total = await store.live_chunk_count()
    before_each = {path: await _chunk_ids(store, document) for path, document in corpus.items()}
    before_members = {document.id for document in await store.collection_documents(collection.id)}

    renamed = await store.rename_collection(collection.id, "  alpha runbooks  ")

    assert renamed.name == "alpha runbooks", "the new name was not normalised on the way in"
    assert renamed.id == collection.id, "renaming minted a new collection"
    assert renamed.description == "worked examples", "renaming dropped the description"
    assert {
        document.id for document in await store.collection_documents(collection.id)
    } == before_members, "renaming changed which documents are members"
    assert await store.live_chunk_count() == before_total, "renaming re-indexed the corpus"
    for path, document in corpus.items():
        assert await _chunk_ids(store, document) == before_each[path], f"{path} was re-embedded"


async def test_renaming_onto_a_name_already_in_use_is_refused(store: SqliteDocStore) -> None:
    """Two collections under one name merge two sets somebody kept apart."""
    await _corpus(store)
    alpha = await store.create_collection("alpha")
    await store.create_collection("beta")

    with pytest.raises(NameInUseError):
        await store.rename_collection(alpha.id, "beta")

    assert (await store.get_collection(alpha.id)) is not None
    survivor = await store.find_collection("alpha")
    assert survivor is not None, "the rename was refused and the old name stopped resolving"
    assert survivor.id == alpha.id, "the old name now resolves to a different collection"


# --- deletion ---------------------------------------------------------------------------------


async def test_deleting_a_collection_leaves_every_document_where_it_was(
    store: SqliteDocStore,
) -> None:
    """A collection groups documents; the cascade runs to the membership rows and stops."""
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")
    await store.add_to_collection(collection.id, [document.id for document in corpus.values()])
    before = await store.live_chunk_count()

    await store.delete_collection(collection.id)

    assert await store.get_collection(collection.id) is None
    for path, document in corpus.items():
        assert await store.get_document(document.id) is not None, (
            f"deleting a collection deleted {path}"
        )
        assert await store.count_chunks(document.id) == 2, f"{path} lost its chunks"
    assert await store.live_chunk_count() == before


async def test_deleting_one_collection_leaves_the_other_ones_membership_intact(
    store: SqliteDocStore,
) -> None:
    """The overlap case: a shared document must stay in the collection that survives."""
    corpus = await _corpus(store)
    alpha = await store.create_collection("alpha")
    beta = await store.create_collection("beta")
    await store.add_to_collection(alpha.id, [corpus[SHARED].id])
    await store.add_to_collection(beta.id, [corpus[SHARED].id])

    await store.delete_collection(alpha.id)

    assert {document.id for document in await store.collection_documents(beta.id)} == {
        corpus[SHARED].id
    }
    assert {held.id for held in await store.collections_for(corpus[SHARED].id)} == {beta.id}


# --- transactional membership -----------------------------------------------------------------


async def test_a_membership_update_that_names_an_unknown_document_writes_nothing(
    store: SqliteDocStore,
) -> None:
    """Transactional, which is what "interrupted halfway" reduces to for a single call.

    The interesting half is the *good* id in the same batch: a loop that inserted as it went
    would leave that one a member and report a failure, so the caller retries and the counts
    stop meaning anything. Nothing is written, and the retry is therefore safe.
    """
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")

    with pytest.raises(ManiculeError):
        await store.add_to_collection(collection.id, [corpus[SHARED].id, "not-a-document"])

    assert list(await store.collection_documents(collection.id)) == [], (
        "a refused batch left its valid half behind, so the write was not atomic"
    )
    assert await store.add_to_collection(collection.id, [corpus[SHARED].id]) == 1, (
        "the retry after a refused batch did not behave like a first attempt"
    )


async def test_concurrent_membership_updates_settle_on_one_set(store: SqliteDocStore) -> None:
    """Two writers, overlapping sets, and membership is still a set afterwards.

    Both name the shared document, so the interleaving that matters is two inserts of the same
    row. The assertion is on the final membership rather than on the two return values: which
    writer is told "1" and which "0" is a race, and only the resulting set is a promise.
    """
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")

    await asyncio.gather(
        store.add_to_collection(collection.id, [corpus[SHARED].id, corpus[ALPHA_ONLY].id]),
        store.add_to_collection(collection.id, [corpus[SHARED].id, corpus[BETA_ONLY].id]),
    )

    members = [document.id for document in await store.collection_documents(collection.id)]
    assert sorted(members) == sorted(document.id for document in corpus.values())
    assert len(members) == len(set(members)), "a document is in the collection twice"


# --- reconciliation from rules ----------------------------------------------------------------


async def test_a_rule_selects_documents_that_arrived_after_it_was_written(
    store: SqliteDocStore,
) -> None:
    """Why reconciliation has nothing to resume: membership is evaluated, never materialised.

    A rule stored and not evaluated is a saved query that answers about the day it was
    written. This is the test that would fail if anyone ever "optimised" it into a snapshot,
    and it is also the whole safety argument for interruption — there is no reconciliation
    pass to be interrupted, because there is no materialised membership to build.
    """
    await _corpus(store)
    ruled = await store.create_collection(
        "everything from fs", rule=CollectionRule(sources=frozenset({"fs"}))
    )
    before = {document.id for document in await store.collection_documents(ruled.id)}

    latecomer = await store.upsert_document(
        make_document(source="fs", source_id="alpha/new.md", uri="file:///alpha/new.md")
    )

    after = {document.id for document in await store.collection_documents(ruled.id)}
    assert after == before | {latecomer.id}, (
        "a document that the rule selects did not appear in the collection, so the rule was "
        "evaluated once and stored rather than evaluated on read"
    )
    assert ruled.id in {held.id for held in await store.collections_for(latecomer.id)}


async def test_applying_the_same_rule_twice_changes_nothing(store: SqliteDocStore) -> None:
    """Idempotent, and therefore safe to re-run after a crash of any kind."""
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")
    await store.add_to_collection(collection.id, [corpus[ALPHA_ONLY].id])
    rule = CollectionRule(sources=frozenset({"fs"}))

    first = await store.set_collection_rule(collection.id, rule)
    members_once = {document.id for document in await store.collection_documents(collection.id)}
    second = await store.set_collection_rule(collection.id, rule)
    members_twice = {document.id for document in await store.collection_documents(collection.id)}

    assert first.rule == second.rule == rule
    assert members_once == members_twice
    assert corpus[ALPHA_ONLY].id in members_twice, (
        "setting a rule dropped a document that had been added by hand. Membership is the "
        "union of the two, so a rule must never silently replace the manual half"
    )


async def test_clearing_a_rule_leaves_the_manual_members_behind(store: SqliteDocStore) -> None:
    """The union again, from the other end: removing the rule removes only what it selected."""
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")
    await store.add_to_collection(collection.id, [corpus[ALPHA_ONLY].id])
    await store.set_collection_rule(collection.id, CollectionRule(sources=frozenset({"fs"})))

    await store.set_collection_rule(collection.id, None)

    assert {document.id for document in await store.collection_documents(collection.id)} == {
        corpus[ALPHA_ONLY].id
    }


async def test_removing_a_document_a_rule_still_selects_leaves_it_a_member(
    store: SqliteDocStore,
) -> None:
    """Documented behaviour, pinned so it stays a decision rather than becoming a surprise.

    ``remove_from_collection`` drops *manual* memberships. A document the rule still selects
    is still selected, and the alternative — a tombstone that suppresses it — would be a
    second kind of membership with its own rules about when it expires.
    """
    corpus = await _corpus(store)
    collection = await store.create_collection(
        "everything from fs", rule=CollectionRule(sources=frozenset({"fs"}))
    )

    await store.remove_from_collection(collection.id, [corpus[SHARED].id])

    assert corpus[SHARED].id in {
        document.id for document in await store.collection_documents(collection.id)
    }


# --- what a rule cannot say -------------------------------------------------------------------


async def test_a_rule_cannot_name_a_path_or_carry_an_expression(store: SqliteDocStore) -> None:
    """Path traversal and expression execution are unreachable, not merely filtered.

    ``CollectionRule`` is a closed model of set-valued fields, and ``rule_clause`` is the one
    function that turns it into SQL. There is no free-text field to smuggle a path through and
    nothing is ever evaluated as code, so neither attack has a surface to land on — which is a
    stronger statement than "the validator rejects it" and the reason this test asserts on the
    *shape of the model* rather than on a list of blocked strings.
    """
    del store
    fields = set(CollectionRule.model_fields)
    assert fields == {
        "sources",
        "media_types",
        "tag_ids",
        "updated_after",
        "updated_before",
    }, (
        f"CollectionRule grew a field: {sorted(fields)}. Every field here is matched against "
        f"a column by equality or set membership. A field carrying a path or an expression "
        f"would need its own argument about traversal and evaluation, and this test is where "
        f"that argument gets written down."
    )

    for hostile in ("../../etc/passwd", "'; DROP TABLE documents; --", "__import__('os')"):
        rule = CollectionRule(sources=frozenset({hostile}))
        # It compiles to a bound parameter, so the string is data. Rendering the clause is how
        # a literal that had been interpolated rather than bound would show itself.
        rendered = str(rule_clause(rule))
        assert hostile not in rendered, (
            f"{hostile!r} was interpolated into the SQL text rather than bound as a parameter"
        )


async def test_an_empty_rule_is_refused_rather_than_matching_everything() -> None:
    """A rule that restricts nothing selects the whole workspace."""
    with pytest.raises(ValueError, match="must restrict something"):
        CollectionRule()


async def test_a_rule_selects_nothing_from_another_workspace(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The *evaluation* half of the workspace promise.

    That the stored rule cannot name a workspace is a property of the type, and
    ``tests/test_storage_organisation.py`` already pins it. This is the other half: a rule
    matching on ``source`` alone, evaluated by a handle that owns one workspace, against a
    corpus where another workspace has documents from the same source name. Source names are
    not tenant-unique — two people both call a connector ``fs`` — so this is the realistic
    shape of the leak rather than a contrived one.

    The foreign document is created through **its own store**, because ``upsert_document``
    writes the handle's workspace onto the row. Building a ``Document`` with someone else's
    workspace id and inserting it here would produce a row in *this* workspace wearing a
    foreign-looking id, and the test would fail against correct code.
    """
    corpus = await _corpus(store)
    other = SqliteDocStore(engine, workspace_id="other-tenant")
    await other.ensure_workspace()
    outsider = await other.upsert_document(
        make_document(
            source="fs",
            source_id="other/notes.md",
            workspace_id="other-tenant",
            uri="file:///other/notes.md",
        )
    )

    ruled = await store.create_collection(
        "everything from fs", rule=CollectionRule(sources=frozenset({"fs"}))
    )

    members = {document.id for document in await store.collection_documents(ruled.id)}
    assert outsider.id not in members, (
        "a rule-driven collection returned a document from another workspace. The rule "
        "restricts on source, and the workspace is supposed to come from the handle "
        "evaluating it rather than from the rule"
    )
    assert members == {document.id for document in corpus.values()}
    assert not await store.collections_for(outsider.id), (
        "this workspace's collections claim to hold another workspace's document"
    )


# --- filters combine predictably --------------------------------------------------------------


async def test_a_collection_and_a_source_keep_only_what_is_in_both(
    store: SqliteDocStore,
) -> None:
    """Conjunction between fields, disjunction within one — what ``Filter`` already promises."""
    corpus = await _corpus(store)
    other = await store.upsert_document(
        make_document(source="web", source_id="alpha/page.html", uri="https://example.invalid/a")
    )
    collection = await store.create_collection("alpha")
    await store.add_to_collection(collection.id, [corpus[ALPHA_ONLY].id, other.id])

    scope = Filter(
        workspace_ids=frozenset({DEFAULT_WORKSPACE}),
        collection_ids=frozenset({collection.id}),
        sources=frozenset({"fs"}),
    )
    resolved = await resolve_filter(scope, collections=store, tags=store)

    assert resolved is not None
    assert resolved.document_ids == frozenset({corpus[ALPHA_ONLY].id, other.id})
    assert resolved.sources == frozenset({"fs"}), (
        "resolution dropped the source restriction while turning the collection into ids, so "
        "the two fields stopped combining"
    )
    assert not resolved.collection_ids


async def test_no_membership_operation_can_reach_a_write_that_would_re_embed(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec's central promise, guarded by *reachability* rather than by state.

    The test above compares chunk ids before and after, which catches a re-embed that changed
    something. It would not catch one that re-derived identical chunks — the corpus would look
    untouched while every membership change quietly did the work of an ingest, and on a real
    model that is the difference between instant and minutes.

    So this asserts the stronger thing: from the six operations that change what a collection
    *is*, the two methods that rewrite a document or its chunks are not reachable at all. It
    holds today because membership is join rows and a rule is a column. It is here so that a
    later change which wires a re-index into `rename_collection` — a plausible, well-meant
    change — fails loudly instead of being discovered as a performance complaint.
    """
    corpus = await _corpus(store)
    collection = await store.create_collection("alpha")
    await store.add_to_collection(collection.id, [corpus[SHARED].id])

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        msg = (
            "a collection operation reached a document or chunk write. Membership is "
            "metadata: one document in two collections is one identity with one set of "
            "embeddings, and grouping it must never re-parse, re-chunk or re-embed it"
        )
        raise AssertionError(msg)

    monkeypatch.setattr(type(store), "replace_chunks", forbidden)
    monkeypatch.setattr(type(store), "upsert_document", forbidden)

    await store.add_to_collection(collection.id, [corpus[ALPHA_ONLY].id])
    await store.remove_from_collection(collection.id, [corpus[ALPHA_ONLY].id])
    await store.rename_collection(collection.id, "alpha handbook")
    await store.describe_collection(collection.id, "worked examples")
    await store.set_collection_rule(collection.id, CollectionRule(sources=frozenset({"fs"})))
    await store.delete_collection(collection.id)
