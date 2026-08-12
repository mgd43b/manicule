"""Nothing crosses a workspace boundary, and this is what proves the check fires.

Every assertion here is written so it cannot be satisfied by accident, and the three devices
that make that true are worth stating because each one closes a different escape:

**Ids are derived, never written down.** Every document in these tests gets its id from
:func:`manicule.core.ids.document_id`, the same function the store uses. There is no literal
to adjust until a comparison passes; the only way to make a refusal stop happening is to make
the identity genuinely belong to the workspace.

**The store under test is deliberately broken.** :class:`~tests.app.fakes.LeakyStore` ignores
its workspace scope entirely, which is what a store written without the ``WHERE`` clause does.
A test run against a correct store would pass whether or not the surface checked anything, and
would keep passing after the check was deleted.

**Every refusal is paired with a positive control.** A service that refused everything would
satisfy half of these on its own, so each refusal is accompanied by the same call succeeding
for a document this workspace does own.

The last assertion in the leak tests is over the **serialised envelope**, not over an
exception type. A guard that raised the right error while putting the foreign document's title
into the message would have leaked precisely what it was defending.
"""

from __future__ import annotations

import json

import pytest

from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.app.tenancy import CrossWorkspaceError, belongs_to, require_owned
from manicule.config.settings import Settings
from manicule.core.errors import UnknownEntityError
from manicule.core.ids import document_id
from manicule.core.retrieval import Candidate
from tests.app.fakes import (
    FakeBackend,
    LeakyOrganisation,
    LeakyStore,
    make_chunk,
    make_document,
)

OURS = "acme"
THEIRS = "globex"
THEIR_TITLE = "Globex compensation review"
THEIR_TEXT = "Everyone at Globex earns exactly four hundred thousand"
"""Content only the other tenant may see. Distinctive, so a leak is greppable."""


def _service(store: LeakyStore) -> ApplicationService:
    backend = FakeBackend(settings=Settings(workspace=OURS), store=store)
    return ApplicationService(backend)


def _leaky_with_a_foreign_document() -> tuple[LeakyStore, str]:
    """A store holding one of ours and one of theirs, and the foreign id."""
    store = LeakyStore(workspace_id=OURS)
    store.add(make_document(OURS, source_id="ours.md", title="Our runbook"))
    foreign = make_document(THEIRS, source="hr", source_id="comp.md", title=THEIR_TITLE)
    store.add(foreign, make_chunk(foreign, text=THEIR_TEXT))
    return store, foreign.id


# --- the identity check itself ------------------------------------------------------------


def test_an_id_minted_for_another_workspace_does_not_belong_to_this_one() -> None:
    """The whole check, in one line, and it is arithmetic rather than a lookup.

    ``document_id`` digests the workspace first, so recomputing it from a document's own
    ``source`` and ``source_id`` proves which tenant minted it. Nothing here consults a store,
    which is why the check would still fire if every ``WHERE`` clause in storage were deleted.
    """
    theirs = make_document(THEIRS, source="hr", source_id="comp.md")
    assert belongs_to(THEIRS, theirs)
    assert not belongs_to(OURS, theirs)


def test_a_foreign_document_is_refused_as_a_batch_rather_than_filtered_out() -> None:
    """Refused whole, because a shortened list is a success report for a partial answer."""
    ours = make_document(OURS)
    theirs = make_document(THEIRS, source="hr", source_id="comp.md", title=THEIR_TITLE)
    assert list(require_owned(OURS, [ours])) == [ours]
    with pytest.raises(CrossWorkspaceError) as caught:
        require_owned(OURS, [ours, theirs])
    assert THEIR_TITLE not in str(caught.value)


# --- through the service, against a store that ignores its scope ---------------------------


async def test_document_get_refuses_a_document_from_another_workspace() -> None:
    """A store that hands over a foreign row does not get that row returned to a caller."""
    store, foreign = _leaky_with_a_foreign_document()
    service = _service(store)

    ours = await service.document_get(document_id(OURS, "local", "ours.md"))
    assert ours.document.title == "Our runbook"

    with pytest.raises(CrossWorkspaceError):
        await service.document_get(foreign)


async def test_document_list_refuses_a_page_containing_another_workspace_s_document() -> None:
    """The listing is refused whole. Silently dropping the row would still have retrieved it."""
    store, _ = _leaky_with_a_foreign_document()
    service = _service(store)
    with pytest.raises(CrossWorkspaceError):
        await service.document_list()

    # The positive control: with the foreign document gone, the same call succeeds. Without
    # this, a service that raised unconditionally would pass the assertion above.
    store.documents = {
        key: value for key, value in store.documents.items() if belongs_to(OURS, value)
    }
    listed = await service.document_list()
    assert [document.title for document in listed.documents] == ["Our runbook"]


async def test_search_refuses_a_hit_pointing_into_another_workspace() -> None:
    """A retriever that returns a foreign chunk is a leak the surface has to catch.

    Retrieval enforces the boundary in its dense leg. This asserts the *second*, independent
    check: the surface resolves every hit's document through the scoped store and proves the
    identity. The two cannot fail the same way, which is the only reason having both is worth
    anything.
    """
    store, foreign_id = _leaky_with_a_foreign_document()
    foreign = store.documents[foreign_id]
    service = _service(store)
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(foreign, text=THEIR_TEXT), score=1.0)
    ]

    with pytest.raises(CrossWorkspaceError):
        await service.search("compensation")

    ours = store.documents[document_id(OURS, "local", "ours.md")]
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(ours), score=1.0)]
    found = await service.search("runbook")
    assert found.count == 1


async def test_ask_refuses_before_the_model_is_called() -> None:
    """The refusal happens **before** generation, so nothing foreign reaches a provider.

    Asserted by the answerer's own call log rather than by the exception: an implementation
    that refused after streaming would raise exactly the same error, having already sent the
    passages.
    """
    store, foreign_id = _leaky_with_a_foreign_document()
    foreign = store.documents[foreign_id]
    service = _service(store)
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(foreign, text=THEIR_TEXT), score=1.0)
    ]

    with pytest.raises(CrossWorkspaceError):
        await service.ask("what is the compensation")
    assert backend.answerer_.calls == [], "the model was called with another tenant's passages"

    ours = store.documents[document_id(OURS, "local", "ours.md")]
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(ours), score=1.0)]
    answered = await service.ask("what is the retry policy")
    assert answered.text
    assert len(backend.answerer_.calls) == 1


# --- nothing of theirs appears in what a caller receives -----------------------------------


@pytest.mark.parametrize("operation", ["document_get", "document_list", "search"])
async def test_no_byte_of_the_other_workspace_reaches_the_serialised_result(
    operation: str,
) -> None:
    """The strongest form of the claim, and the one a hand-edited assertion cannot satisfy.

    The envelope is serialised to JSON and searched for the foreign document's title, its
    text, its id and its workspace name. An error message that named any of them would fail
    here even though the refusal itself worked — which is the failure mode a test asserting
    only ``pytest.raises`` would miss.
    """
    store, foreign_id = _leaky_with_a_foreign_document()
    foreign = store.documents[foreign_id]
    service = _service(store)
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(foreign, text=THEIR_TEXT), score=1.0)
    ]

    calls = {
        "document_get": lambda: service.document_get(foreign_id),
        "document_list": service.document_list,
        "search": lambda: service.search("compensation"),
    }
    envelope = await run_op(operation, service.workspace, calls[operation])
    body = json.dumps(envelope.as_json())

    assert envelope.ok is False
    assert envelope.workspace == OURS
    for forbidden in (THEIR_TITLE, THEIR_TEXT, THEIRS, foreign.source_id, foreign.uri):
        assert forbidden not in body, f"{forbidden!r} leaked into a {operation} result"


async def test_deleting_another_workspace_s_document_is_refused_and_writes_nothing() -> None:
    """A refusal that had already written is not a refusal.

    The store records every delete it is asked to perform, so this asserts the *absence* of a
    write rather than the presence of an exception.
    """
    store, foreign_id = _leaky_with_a_foreign_document()
    service = _service(store)

    with pytest.raises(CrossWorkspaceError):
        await service.document_delete(foreign_id)
    assert store.deleted == []

    ours = document_id(OURS, "local", "ours.md")
    removed = await service.document_delete(ours)
    assert removed.deleted
    assert store.deleted == [(ours, "soft")]


async def test_a_document_this_workspace_has_never_seen_is_not_reported_as_existing_elsewhere() -> (
    None
):
    """ "No such document here" never becomes "it belongs to somebody else".

    Answering the second question is itself a cross-tenant disclosure: it tells a caller that
    an id they guessed is real somewhere.
    """
    store = LeakyStore(workspace_id=OURS)
    service = _service(store)
    with pytest.raises(UnknownEntityError) as caught:
        await service.document_get(document_id(THEIRS, "hr", "comp.md"))
    assert THEIRS not in str(caught.value)


# --- collections ----------------------------------------------------------------------------
#
# Collections are a *join*: membership is the set a search resolves into document ids, so a
# foreign member is a cross-tenant result with a perfectly ordinary explanation. Both reads
# added for collections go through `require_owned` for that reason, and both are driven here
# by a deliberately broken store so the surface's own check is what is being observed.


def _leaky_organisation() -> tuple[LeakyOrganisation, LeakyStore, str]:
    """An organisation store and a document store, each holding one of ours and one of theirs."""
    store = LeakyStore(workspace_id=OURS)
    ours = make_document(OURS, source_id="ours.md", title="Our runbook")
    foreign = make_document(THEIRS, source="hr", source_id="comp.md", title=THEIR_TITLE)
    store.add(ours)
    store.add(foreign, make_chunk(foreign, text=THEIR_TEXT))

    organisation = LeakyOrganisation(workspace_id=OURS)
    organisation.documents[ours.id] = ours
    organisation.documents[foreign.id] = foreign
    return organisation, store, foreign.id


def _service_with(organisation: LeakyOrganisation, store: LeakyStore) -> ApplicationService:
    return ApplicationService(
        FakeBackend(settings=Settings(workspace=OURS), store=store, organisation_=organisation)
    )


async def test_counting_a_collection_refuses_a_foreign_member_rather_than_counting_it() -> None:
    """A count is a read, and this one walks the membership to produce it.

    A leaked member would inflate a number rather than print a title, which is the quiet
    version of the same disclosure: it tells a caller how much another tenant has.
    """
    organisation, store, foreign_id = _leaky_organisation()
    collection = await organisation.create_collection("mixed")
    ours = document_id(OURS, "local", "ours.md")
    organisation.members[collection.id] = [ours, foreign_id]
    service = _service_with(organisation, store)

    with pytest.raises(CrossWorkspaceError):
        await service.collection_counts(collection.id)

    organisation.members[collection.id] = [ours]
    counted = await service.collection_counts(collection.id)
    assert counted.documents == 1


async def test_the_orphan_sweep_refuses_a_foreign_document_rather_than_trashing_it() -> None:
    """The most dangerous read here, because what follows it is a delete.

    ``LeakyStore.list_documents`` ignores the filter *and* the limit, so the sweep is offered
    another tenant's document as an ordinary row. Refusing before anything is written is the
    property: a guard that raised after the loop had already soft-deleted would have destroyed
    somebody else's corpus and then reported an error about it.
    """
    organisation, store, foreign_id = _leaky_organisation()
    service = _service_with(organisation, store)

    with pytest.raises(CrossWorkspaceError):
        await service.collection_orphans(delete=True)
    assert store.deleted == [], "a foreign document was trashed before the check fired"
    del foreign_id


async def test_a_search_scoped_to_a_collection_that_does_not_exist_is_refused() -> None:
    """Refused, never widened.

    An unknown name resolving to "no collections" would leave the filter unrestricted, and an
    empty field restricts nothing — so a misspelt collection would return the whole workspace,
    ranked and plausible. The refusal is what stops a typo becoming a wider search than the
    one anybody asked for.
    """
    organisation, store, _ = _leaky_organisation()
    service = _service_with(organisation, store)

    with pytest.raises(UnknownEntityError):
        await service.search("anything", collections=["no-such-collection"])
