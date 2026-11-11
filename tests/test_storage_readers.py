"""Reader groups and source grants in SQLite, and the predicate that applies them.

Every predicate here is checked against the rows it must exclude, and each one that guards a
read is held to the same answer in both of its spellings — the clause SQLAlchemy builds and the
fragment the lexical leg's textual statement splices in — because two spellings of one rule
are a corpus that answers the same question two ways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select, text

from manicule.core.errors import NameInUseError, PolicyError, UnknownEntityError
from manicule.core.readers import (
    UNRESTRICTED,
    AccessGate,
    DocumentAccess,
    GateOrigin,
    admits,
    group_principal,
    user_principal,
)
from manicule.core.retrieval import Filter, Query
from manicule.retrieval.router import UtilityKind
from manicule.retrieval.utility import handlers_for
from manicule.storage import models
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore
from manicule.storage.readers import readable, readable_sql
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.content import Document

SCOPE = frozenset({DEFAULT_WORKSPACE})


async def _person(store: SqliteDocStore, user_id: str) -> str:
    async with store.sessions.begin() as session:
        session.add(models.User(id=user_id, provider="local", subject=user_id))
    return user_id


async def _corpus(store: SqliteDocStore) -> tuple[Document, Document]:
    """An open document and a restricted one, both live, indexed and searchable."""
    made: list[Document] = []
    for source in ("open", "restricted"):
        document = make_document(source=source, source_id=source, title=source)
        await store.upsert_document(document)
        await store.replace_chunks(
            document.id, [make_chunk(document, 0, f"authentication tokens {source}")]
        )
        made.append(document)
    return made[0], made[1]


async def _restrict(store: SqliteDocStore) -> str:
    """Grant ``restricted`` to a group and return the group's principal."""
    group = await store.create_reader_group("editors")
    await store.grant_source("restricted", [group.id])
    return group_principal(group.id)


# --- the rule ------------------------------------------------------------------------------


def test_none_and_the_empty_set_are_opposite_instructions() -> None:
    gated = DocumentAccess(gates=(AccessGate(origin=GateOrigin.SOURCE, readers=frozenset({"g"})),))

    assert admits(None, gated)
    assert not admits(frozenset(), gated)
    assert admits(frozenset(), UNRESTRICTED)


def test_a_document_whose_gates_were_never_read_is_refused_to_a_restricted_reader() -> None:
    """``None`` is "not read", and a re-check that passed it would be a re-check of nothing."""
    assert admits(None, None)
    assert not admits(frozenset({"anything"}), None)


def test_a_gate_with_no_principals_opens_for_nobody() -> None:
    """How "the readers could not be established" is recorded: closed, not open."""
    unknown = DocumentAccess(gates=(AccessGate(origin=GateOrigin.DOCUMENT),))

    assert not admits(frozenset({user_principal("u1"), group_principal("g1")}), unknown)


def test_every_gate_has_to_admit() -> None:
    both = DocumentAccess(
        gates=(
            AccessGate(origin=GateOrigin.SOURCE, readers=frozenset({"group:a"})),
            AccessGate(origin=GateOrigin.DOCUMENT, readers=frozenset({"user:u"})),
        )
    )

    assert admits(frozenset({"group:a", "user:u"}), both)
    assert not admits(frozenset({"group:a"}), both)
    assert not admits(frozenset({"user:u"}), both)


# --- the store -----------------------------------------------------------------------------


async def test_an_ungranted_workspace_has_no_gates_and_every_document_reads_unrestricted(
    store: SqliteDocStore,
) -> None:
    open_document, restricted = await _corpus(store)

    assert not await store.has_reader_gates()
    listed = await store.list_documents(Filter(workspace_ids=SCOPE))
    assert {document.id for document in listed} == {open_document.id, restricted.id}
    assert all(document.access == UNRESTRICTED for document in listed)


async def test_a_grant_restricts_its_source_and_only_its_source(store: SqliteDocStore) -> None:
    open_document, restricted = await _corpus(store)
    member = await _restrict(store)

    assert await store.has_reader_gates()
    nobody = await store.list_documents(Filter(workspace_ids=SCOPE, readers=frozenset()))
    granted = await store.list_documents(Filter(workspace_ids=SCOPE, readers=frozenset({member})))

    assert [document.id for document in nobody] == [open_document.id]
    assert {document.id for document in granted} == {open_document.id, restricted.id}


async def test_listed_documents_carry_the_gates_they_were_admitted_under(
    store: SqliteDocStore,
) -> None:
    """The re-check on the way out reads these, and nothing else."""
    _, restricted = await _corpus(store)
    member = await _restrict(store)

    unfiltered = {
        document.id: document
        for document in await store.list_documents(Filter(workspace_ids=SCOPE))
    }
    fetched = await store.get_document(restricted.id)

    assert fetched is not None
    for document in (unfiltered[restricted.id], fetched):
        assert document.access == DocumentAccess(
            gates=(AccessGate(origin=GateOrigin.SOURCE, readers=frozenset({member})),)
        )
        assert not admits(frozenset(), document.access)
        assert admits(frozenset({member}), document.access)


async def test_the_lexical_leg_applies_the_reader_set_before_its_limit(
    store: SqliteDocStore,
) -> None:
    open_document, restricted = await _corpus(store)
    member = await _restrict(store)

    nobody = await store.search_lexical(
        "authentication", 10, Filter(workspace_ids=SCOPE, readers=frozenset())
    )
    granted = await store.search_lexical(
        "authentication", 10, Filter(workspace_ids=SCOPE, readers=frozenset({member}))
    )
    # `k = 1` with the restricted chunk ranked above the open one is the top-k trap: filtering
    # after the limit would return nothing at all.
    first = await store.search_lexical(
        "restricted authentication", 1, Filter(workspace_ids=SCOPE, readers=frozenset())
    )

    assert {candidate.chunk.document_id for candidate in nobody} == {open_document.id}
    assert {candidate.chunk.document_id for candidate in granted} == {
        open_document.id,
        restricted.id,
    }
    assert [candidate.chunk.document_id for candidate in first] == [open_document.id]


@pytest.mark.parametrize("holding", ["nothing", "the group", "another group", "a person"])
async def test_the_two_spellings_of_the_predicate_agree(
    store: SqliteDocStore, holding: str
) -> None:
    """The clause and the textual fragment admit exactly the same documents."""
    await _corpus(store)
    member = await _restrict(store)
    other = await store.create_reader_group("others")
    readers = {
        "nothing": frozenset[str](),
        "the group": frozenset({member}),
        "another group": frozenset({group_principal(other.id)}),
        "a person": frozenset({user_principal("someone")}),
    }[holding]

    async with store.sessions() as session:
        built = set(
            (
                await session.execute(
                    select(models.Document.id).where(
                        models.Document.workspace_id == DEFAULT_WORKSPACE, readable(readers)
                    )
                )
            )
            .scalars()
            .all()
        )
        fragment, bound = readable_sql(readers)
        spliced = {
            str(row[0])
            for row in (
                await session.execute(
                    text(f"SELECT d.id FROM documents d WHERE d.workspace_id = :w AND {fragment}"),  # noqa: S608 - a fragment of fixed text and bound names
                    {"w": DEFAULT_WORKSPACE, **bound},
                )
            ).all()
        }

    assert built == spliced
    assert built, "a predicate that admits nothing agrees with any other one"


async def test_a_grant_to_a_group_of_another_workspace_is_refused(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    theirs = await other.create_reader_group("editors")

    with pytest.raises(UnknownEntityError):
        await store.grant_source("restricted", [theirs.id])
    assert not await store.has_reader_gates()


async def test_a_grant_naming_one_unknown_group_grants_nothing(store: SqliteDocStore) -> None:
    group = await store.create_reader_group("editors")

    with pytest.raises(UnknownEntityError):
        await store.grant_source("restricted", [group.id, "nobody-by-that-name"])
    assert await store.source_grants() == {}


async def test_a_group_still_granted_a_source_cannot_be_deleted(store: SqliteDocStore) -> None:
    """Deleting the last grant opens a source to everyone, so it is never a side effect."""
    await _corpus(store)
    await _restrict(store)

    with pytest.raises(PolicyError, match="Revoke them first"):
        await store.delete_reader_group("editors")
    assert await store.source_grants() == {"restricted": ("editors",)}

    await store.revoke_source("restricted", ["editors"])
    deleted = await store.delete_reader_group("editors")
    assert deleted.name == "editors"
    assert not await store.has_reader_gates()


async def test_the_database_refuses_to_delete_a_granted_group_even_when_asked_directly(
    store: SqliteDocStore,
) -> None:
    """The foreign key backs the refusal above for a statement that skipped it."""
    await _corpus(store)
    await _restrict(store)

    with pytest.raises(Exception, match="FOREIGN KEY"):
        async with store.sessions.begin() as session:
            await session.execute(text("DELETE FROM reader_groups"))
    assert await store.source_grants() == {"restricted": ("editors",)}


async def test_a_duplicate_group_name_is_refused(store: SqliteDocStore) -> None:
    await store.create_reader_group("editors")

    with pytest.raises(NameInUseError):
        await store.create_reader_group(" editors ")


async def test_a_person_holds_themself_and_their_groups_here_only(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    person = await _person(store, "u1")
    editors = await store.create_reader_group("editors")
    await store.create_reader_group("others")
    await store.add_reader_group_members("editors", [person])
    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    theirs = await other.create_reader_group("editors")
    await other.add_reader_group_members(theirs.id, [person])

    held = await store.reader_principals(person)

    assert held == frozenset({user_principal(person), group_principal(editors.id)})


async def test_removing_a_member_narrows_what_they_hold(store: SqliteDocStore) -> None:
    person = await _person(store, "u1")
    editors = await store.create_reader_group("editors")
    await store.add_reader_group_members("editors", [person, person])

    assert (await store.find_reader_group("editors")) is not None
    assert group_principal(editors.id) in await store.reader_principals(person)
    await store.remove_reader_group_members(editors.id, [person])
    assert await store.reader_principals(person) == frozenset({user_principal(person)})


async def test_a_grant_moves_the_generation_and_a_membership_change_does_not(
    store: SqliteDocStore,
) -> None:
    """A grant changes what one reader set reads, so a cached ranking must not survive it.

    A membership change does not: it changes *which* reader set a caller holds, and the reader
    set is part of the cache key.
    """
    person = await _person(store, "u1")
    group = await store.create_reader_group("editors")

    before = store.generation
    await store.add_reader_group_members(group.id, [person])
    assert store.generation == before

    await store.grant_source("restricted", [group.id])
    after_grant = store.generation
    assert after_grant > before

    await store.revoke_source("restricted", [group.id])
    assert store.generation > after_grant


async def test_a_source_is_readable_by_the_same_rule_its_documents_are(
    store: SqliteDocStore,
) -> None:
    member = await _restrict(store)

    assert await store.may_read_source("open", frozenset())
    assert not await store.may_read_source("restricted", frozenset())
    assert await store.may_read_source("restricted", frozenset({member}))


async def test_readable_totals_count_only_what_the_reader_set_may_read(
    store: SqliteDocStore,
) -> None:
    open_document, _ = await _corpus(store)
    member = await _restrict(store)

    nobody = await store.readable_totals(frozenset())
    granted = await store.readable_totals(frozenset({member}))

    assert (nobody.documents, nobody.chunks) == (1, 1)
    assert nobody.by_source == {open_document.source: 1}
    assert (granted.documents, granted.chunks) == (2, 2)


async def test_readable_ids_cover_the_trash_too(store: SqliteDocStore) -> None:
    open_document, restricted = await _corpus(store)
    member = await _restrict(store)
    await store.soft_delete_document(restricted.id)

    asked = [open_document.id, restricted.id, "not-a-document"]
    assert await store.readable_document_ids(asked, frozenset()) == {open_document.id}
    assert await store.readable_document_ids(asked, frozenset({member})) == {
        open_document.id,
        restricted.id,
    }


async def test_a_collection_lists_only_the_members_the_reader_set_may_read(
    store: SqliteDocStore,
) -> None:
    open_document, restricted = await _corpus(store)
    member = await _restrict(store)
    collection = await store.create_collection("everything")
    await store.add_to_collection(collection.id, [open_document.id, restricted.id])

    nobody = await store.collection_documents(collection.id, readers=frozenset())
    granted = await store.collection_documents(collection.id, readers=frozenset({member}))
    everyone = await store.collection_documents(collection.id)

    assert [document.id for document in nobody] == [open_document.id]
    assert {document.id for document in granted} == {open_document.id, restricted.id}
    assert {document.id for document in everyone} == {open_document.id, restricted.id}
    assert all(document.access is not None for document in everyone)


async def test_the_utility_routes_count_only_what_the_reader_set_may_read(
    store: SqliteDocStore,
) -> None:
    """ "How many documents", answered with the workspace's total, says the rest exist."""
    await _corpus(store)
    member = await _restrict(store)
    handlers = handlers_for(store)

    def asked(readers: frozenset[str] | None) -> Query:
        return Query(text="how many documents", filter=Filter(workspace_ids=SCOPE, readers=readers))

    nobody = await handlers[UtilityKind.DOCUMENT_COUNT](asked(frozenset()))
    granted = await handlers[UtilityKind.DOCUMENT_COUNT](asked(frozenset({member})))
    everyone = await handlers[UtilityKind.DOCUMENT_COUNT](asked(None))
    listed = await handlers[UtilityKind.DOCUMENT_LIST](asked(frozenset()))
    status = await handlers[UtilityKind.INDEX_STATUS](asked(frozenset()))

    assert (nobody.data["documents"], granted.data["documents"]) == (1, 2)
    assert everyone.data["documents"] == 2
    assert listed.data["documents"] == [
        {"id": document.id, "title": document.title, "uri": document.uri}
        for document in await store.list_documents(Filter(workspace_ids=SCOPE, readers=frozenset()))
    ]
    assert status.data["chunks"] == 1
