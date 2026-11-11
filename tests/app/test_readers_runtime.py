"""Readers, end to end: a source granted to a group, and what every surface then returns.

The shipped runtime, the real ingest pipeline, SQLite, the embedded vector store and the L1
cache, with :class:`~manicule.app.service.ApplicationService` in front and a caller set the way
a network surface sets one. Only the embedder, the chunker and the model are doubles.

Two sources are indexed with the same words, so that every search reaches both and only the
reader set can tell their documents apart. ``restricted`` is granted to ``editors``; Erin is in
the group and Vic is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from manicule.app.caller import Caller, acting_as
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.app.tenancy import UnreadableDocumentError
from manicule.config.settings import Role, Settings
from manicule.container import keys
from manicule.core.errors import PolicyError, UnknownEntityError
from manicule.core.ids import document_id
from manicule.plugins.registry import discover
from manicule.storage import docstore as docstore_module
from tests.fakes import HashEmbedder
from tests.generation.fakes import ScriptedGenerator

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

WORKSPACE = "alpha"
QUESTION = "orchard irrigation"
ANSWER = "It runs at dawn [[cite:1]] and costs money [[cite:2]]."
"""Cites both passages a reader holding everything retrieves, so that the restricted one, when
it is in the context, is cited."""
OPEN = {"schedule": "The orchard irrigation schedule runs at dawn."}
RESTRICTED = {"budget": "The orchard irrigation budget is confidential."}
OPEN_ID = document_id(WORKSPACE, "open", "schedule")
RESTRICTED_ID = document_id(WORKSPACE, "restricted", "budget")


def _runtime(environment: Path) -> Runtime:
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    bound.add(
        keys.GENERATOR.named("scripted"),
        lambda _: ScriptedGenerator(script=(ANSWER,)),
    )
    settings = Settings(
        data_dir=environment / "data",
        workspace=WORKSPACE,
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        llm={"generator": "scripted"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
        security={"sharing": {"enabled": True}},  # pyright: ignore[reportArgumentType]
    )
    assert settings.rag.cache.enabled, "the cache is part of what is under test"
    return Runtime(settings, discovery=found)


async def _index(opened: Runtime, name: str, passages: Mapping[str, str]) -> None:
    from tests.ingest.fakes import DictConnector  # noqa: PLC0415 - fakes, local to this harness

    source = DictConnector(dict(passages), name=name)
    for source_id in passages:
        source.media_types[source_id] = "text/plain"
    report = await (await opened.pipeline()).run(source)
    assert report.indexed == len(passages)


class People:
    """The two readers, by user id."""

    def __init__(self, erin: str, vic: str) -> None:
        self.erin = erin
        self.vic = vic


def viewer(user_id: str | None) -> Caller:
    return Caller(role=Role.VIEWER, user_id=user_id)


def _ignores_the_reader_set(readers: frozenset[str]) -> tuple[str, dict[str, object]]:
    """The lexical leg's reader predicate with its body deleted: it admits everything."""
    del readers
    return "1 = 1", {}


@pytest.fixture
async def opened(manicule_environment: Path) -> AsyncIterator[Runtime]:
    async with _runtime(manicule_environment) as running:
        await _index(running, "open", OPEN)
        await _index(running, "restricted", RESTRICTED)
        yield running


@pytest.fixture
def service(opened: Runtime) -> ApplicationService:
    return ApplicationService(opened)


@pytest.fixture
async def people(service: ApplicationService) -> People:
    erin = await service.user_create("erin", role="viewer")
    vic = await service.user_create("vic", role="viewer")
    return People(erin.user.id, vic.user.id)


@pytest.fixture
async def restricted(service: ApplicationService, people: People) -> People:
    """``restricted`` granted to ``editors``, with Erin in it."""
    await service.reader_group_create("editors")
    await service.reader_group_add("editors", ["erin"])
    granted = await service.source_grant("restricted", ["editors"])
    assert granted.restricted
    return people


def _documents(result: Any) -> set[str]:
    return {hit.document_id for hit in result.hits}


# --- nothing configured --------------------------------------------------------------------


async def test_with_no_grant_every_reader_reads_everything(
    service: ApplicationService, people: People
) -> None:
    """An installation that configured no readers reads exactly as before they existed."""
    with acting_as(viewer(people.vic)):
        found = await service.search(QUESTION, limit=5)
        listed = await service.document_list()

    assert _documents(found) == {OPEN_ID, RESTRICTED_ID}
    assert {document.id for document in listed.documents} == {OPEN_ID, RESTRICTED_ID}


# --- search, both legs, and the cache ------------------------------------------------------


async def test_a_reader_outside_the_group_never_sees_the_restricted_source(
    service: ApplicationService, restricted: People
) -> None:
    with acting_as(viewer(restricted.vic)):
        found = await service.search(QUESTION, limit=5)
    with acting_as(viewer(restricted.erin)):
        granted = await service.search(QUESTION, limit=5)

    assert _documents(found) == {OPEN_ID}
    assert _documents(granted) == {OPEN_ID, RESTRICTED_ID}
    assert all(set(hit.scores) >= {"dense", "lexical"} for hit in granted.hits)


async def test_the_operator_at_this_machine_reads_everything(
    service: ApplicationService, restricted: People
) -> None:
    del restricted
    found = await service.search(QUESTION, limit=5)

    assert _documents(found) == {OPEN_ID, RESTRICTED_ID}


async def test_a_key_with_no_person_reads_only_what_carries_no_gate(
    service: ApplicationService, restricted: People
) -> None:
    del restricted
    with acting_as(Caller(role=Role.ADMIN, key_id="k1")):
        found = await service.search(QUESTION, limit=5)

    assert _documents(found) == {OPEN_ID}


async def test_one_readers_cached_ranking_is_never_served_to_another(
    service: ApplicationService, restricted: People
) -> None:
    """Erin's ranking holds the restricted passage. Vic's identical search must miss.

    Served to Vic, the cached entry would either hand him the passage or — had the rehydrating
    join dropped it — a ranking, an order and a confidence computed over a document he may not
    read. The reader set is in the key, so it is neither.
    """
    with acting_as(viewer(restricted.erin)):
        first = await service.search(QUESTION, limit=5)
        again = await service.search(QUESTION, limit=5)
    with acting_as(viewer(restricted.vic)):
        other = await service.search(QUESTION, limit=5)

    assert again.cached is True, "the fixture must reach the cache, or this proves nothing"
    assert RESTRICTED_ID in _documents(first)
    assert other.cached is False, "a ranking cached for one reader set was served to another"
    assert _documents(other) == {OPEN_ID}


async def test_a_narrower_readers_cached_ranking_is_never_served_to_a_wider_one(
    service: ApplicationService, restricted: People
) -> None:
    """The other direction, which the rehydrating join cannot catch.

    Vic's ranking holds nothing restricted, so every document in it is one Erin may read and
    the join would serve it to her whole — a ranking short of the passage she may read, with a
    confidence computed without it. Only the reader set in the key keeps the two apart.
    """
    with acting_as(viewer(restricted.vic)):
        await service.search(QUESTION, limit=5)
        again = await service.search(QUESTION, limit=5)
    with acting_as(viewer(restricted.erin)):
        wider = await service.search(QUESTION, limit=5)

    assert again.cached is True, "the fixture must reach the cache, or this proves nothing"
    assert wider.cached is False, "a ranking cached for one reader set was served to another"
    assert RESTRICTED_ID in _documents(wider)


async def test_a_grant_invalidates_a_ranking_cached_before_it(
    service: ApplicationService, people: People
) -> None:
    with acting_as(viewer(people.vic)):
        before = await service.search(QUESTION, limit=5)
    await service.reader_group_create("editors")
    await service.source_grant("restricted", ["editors"])
    with acting_as(viewer(people.vic)):
        after = await service.search(QUESTION, limit=5)

    assert RESTRICTED_ID in _documents(before)
    assert after.cached is False
    assert _documents(after) == {OPEN_ID}


async def test_a_revocation_reopens_the_source(
    service: ApplicationService, restricted: People
) -> None:
    revoked = await service.source_revoke("restricted", ["editors"])
    with acting_as(viewer(restricted.vic)):
        found = await service.search(QUESTION, limit=5)

    assert not revoked.restricted
    assert _documents(found) == {OPEN_ID, RESTRICTED_ID}


async def test_the_outbound_check_refuses_what_a_store_that_ignored_the_reader_set_returned(
    service: ApplicationService, restricted: People, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the lexical leg's reader predicate and the passage is caught on the way out.

    The re-check reads the gates the document came back carrying and nothing else, so it fires
    with the store's ``WHERE`` clause gone — and the refusal names no document.
    """
    monkeypatch.setattr(docstore_module, "readable_sql", _ignores_the_reader_set)

    with acting_as(viewer(restricted.vic)), pytest.raises(UnreadableDocumentError) as refused:
        await service.search(QUESTION, limit=5)

    assert RESTRICTED_ID not in str(refused.value)
    assert "budget" not in str(refused.value)


# --- the answer path -----------------------------------------------------------------------


async def test_an_answer_is_built_only_from_what_the_reader_may_read(
    service: ApplicationService, restricted: People
) -> None:
    with acting_as(viewer(restricted.vic)):
        answered = await service.ask(QUESTION, limit=5)

    assert answered.citations, "the scripted answer must cite something, or this proves nothing"
    assert {citation.document_id for citation in answered.citations} <= {OPEN_ID}


# --- reads by id ---------------------------------------------------------------------------


async def test_a_document_the_reader_may_not_read_is_refused_as_if_absent(
    service: ApplicationService, restricted: People
) -> None:
    with acting_as(viewer(restricted.vic)):
        with pytest.raises(UnknownEntityError) as refused:
            await service.document_get(RESTRICTED_ID)
        with pytest.raises(UnknownEntityError) as absent:
            await service.document_get("not-a-document")
        with pytest.raises(UnknownEntityError):
            await service.document_resolve(source="restricted", source_id="budget")
        with pytest.raises(UnknownEntityError):
            await service.workbench(RESTRICTED_ID)
        listed = await service.document_list()
        found = await service.document_get(OPEN_ID)

    assert str(refused.value).replace(RESTRICTED_ID, "ID") == str(absent.value).replace(
        "not-a-document", "ID"
    ), "a refusal worded differently from absence says the document exists"
    assert {document.id for document in listed.documents} == {OPEN_ID}
    assert found.document.id == OPEN_ID
    with acting_as(viewer(restricted.erin)):
        assert (await service.document_get(RESTRICTED_ID)).document.id == RESTRICTED_ID


async def test_a_reader_cannot_file_tag_or_delete_what_they_may_not_read(
    service: ApplicationService, restricted: People
) -> None:
    collection = await service.collection_create("everything")
    tag = await service.tag_create("budget")
    with acting_as(Caller(role=Role.MEMBER, user_id=restricted.vic)):
        with pytest.raises(UnknownEntityError):
            await service.collection_add(collection.id, [RESTRICTED_ID])
        with pytest.raises(UnknownEntityError):
            await service.document_tag(RESTRICTED_ID, [tag.id])
        with pytest.raises(UnknownEntityError):
            await service.document_delete(RESTRICTED_ID)
        added = await service.collection_add(collection.id, [OPEN_ID])

    assert added.changed == 1
    assert (await service.document_get(RESTRICTED_ID)).document.id == RESTRICTED_ID


# --- counts --------------------------------------------------------------------------------


async def test_counts_are_of_what_the_reader_may_read(
    service: ApplicationService, restricted: People
) -> None:
    collection = await service.collection_create("everything")
    await service.collection_add(collection.id, [OPEN_ID, RESTRICTED_ID])

    with acting_as(viewer(restricted.vic)):
        stats = await service.stats()
        counts = await service.collection_counts(collection.id)
        members = await service.collection_documents(collection.id)
        with pytest.raises(PolicyError, match="administrator"):
            await service.index_status()
    with acting_as(Caller(role=Role.ADMIN, user_id=restricted.vic)):
        totals = await service.stats()
        status = await service.index_status()

    assert (stats.documents, stats.by_source) == (1, {"open": 1})
    assert counts.documents == 1
    assert [document.id for document in members.documents] == [OPEN_ID]
    assert totals.documents == status.documents == 2


async def test_two_readers_with_different_counts_do_not_share_a_counts_scope(
    service: ApplicationService, restricted: People
) -> None:
    with acting_as(viewer(restricted.vic)):
        vic = await service.counts_scope()
    with acting_as(viewer(restricted.erin)):
        erin = await service.counts_scope()
    with acting_as(Caller(role=Role.ADMIN, user_id=restricted.vic)):
        admin = await service.counts_scope()

    assert vic != erin
    assert admin == ""
    assert await service.counts_scope() == ""


# --- conversations -------------------------------------------------------------------------


async def test_a_restricted_reader_opens_only_their_own_conversations(
    service: ApplicationService, restricted: People
) -> None:
    with acting_as(viewer(restricted.erin)):
        mine = await service.conversation_create(title="budget")
        await service.ask(QUESTION, limit=5, conversation_id=mine.id)
    with acting_as(viewer(restricted.vic)):
        listed = await service.conversation_list()
        with pytest.raises(UnknownEntityError):
            await service.conversation_messages(mine.id)
        with pytest.raises(UnknownEntityError):
            await service.ask(QUESTION, limit=5, conversation_id=mine.id)
    with acting_as(viewer(restricted.erin)):
        own = await service.conversation_messages(mine.id)

    assert listed.conversations == ()
    assert own.count >= 1


async def test_a_turn_citing_a_document_since_restricted_is_withheld_from_its_owner(
    service: ApplicationService, people: People
) -> None:
    with acting_as(viewer(people.vic)):
        mine = await service.conversation_create(title="budget")
        answered = await service.ask(QUESTION, limit=5, conversation_id=mine.id)
    assert RESTRICTED_ID in {citation.document_id for citation in answered.citations}, (
        "the turn must cite the document about to be restricted, or this proves nothing"
    )
    await service.reader_group_create("editors")
    await service.source_grant("restricted", ["editors"])
    with acting_as(viewer(people.vic)):
        turns = await service.conversation_messages(mine.id)

    cited = {citation.document_id for turn in turns.turns for citation in turn.citations}
    assert RESTRICTED_ID not in cited
    assert all(turn.role != "assistant" for turn in turns.turns)


async def test_sharing_is_refused_while_any_source_is_restricted(
    service: ApplicationService, people: People
) -> None:
    """A link is read by whoever holds it, so it can honor no reader set.

    And a link minted before the restriction stops serving once it is in force.
    """
    conversation = await service.conversation_create(title="anything")
    with acting_as(viewer(people.vic)):
        await service.ask(QUESTION, limit=5, conversation_id=conversation.id)
    link = await service.conversation_share(conversation.id)
    assert (await service.shared_conversation(link.token)).count >= 1
    await service.reader_group_create("editors")
    await service.source_grant("restricted", ["editors"])

    with pytest.raises(PolicyError, match="cannot be shared"):
        await service.conversation_share(conversation.id)
    assert (await service.shared_conversation(link.token)).count == 0


# --- administration ------------------------------------------------------------------------


async def test_only_an_administrator_changes_who_may_read_what(
    service: ApplicationService, people: People
) -> None:
    with acting_as(Caller(role=Role.MEMBER, user_id=people.vic)):
        with pytest.raises(PolicyError, match="admin"):
            await service.reader_group_create("editors")
        with pytest.raises(PolicyError, match="admin"):
            await service.source_grant("restricted", ["editors"])

    listed = await service.reader_group_list()
    assert listed.count == 0
    assert listed.grants == {}


async def test_a_group_is_listed_with_its_members_and_its_sources(
    service: ApplicationService, restricted: People
) -> None:
    listed = await service.reader_group_list()

    assert [group.name for group in listed.groups] == ["editors"]
    assert listed.groups[0].members == (restricted.erin,)
    assert listed.groups[0].sources == ("restricted",)
    assert listed.grants == {"restricted": ("editors",)}
