"""No route reads across workspaces, and the check that stops it is this surface's own.

The method is the one ``tests/app/test_tenancy.py`` established: drive the routes against a
store that **ignores its workspace scope entirely**, and watch the refusal happen. A test run
against a correct store would pass whether or not any check existed, which is the failure this
project has been bitten by more than once.

:class:`~tests.app.fakes.LeakyStore` and :class:`~tests.app.fakes.LeakyOrganisation` ignore the
filter *and* the limit. Ignoring the limit matters: a leaky store that still truncated would
let the "some of what I asked for came back missing" check catch a foreign document by
accident, and the identity check — the arithmetic that would still fire if every ``WHERE``
clause in storage were deleted — would never be exercised.

Every test here also asserts the **control**: the same route against a correct store returns
the tenant's own document. Without that, a surface that refused everything would pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.ids import document_id
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import LeakyOrganisation, LeakyStore, make_chunk, make_document

if TYPE_CHECKING:
    from tests.app.fakes import FakeBackend

OTHER = "another-tenant"
SERVER_ERROR = 500


def _leaky(backend: FakeBackend) -> None:
    """Replace this backend's stores with ones that ignore their scope, keeping the contents."""
    leaky = LeakyStore(workspace_id=backend.settings.workspace)
    leaky.documents = dict(backend.store.documents)
    leaky.chunks = dict(backend.store.chunks)
    backend.store = leaky
    organisation = LeakyOrganisation(workspace_id=backend.settings.workspace)
    organisation.documents = dict(backend.organisation_.documents)
    organisation.collections = dict(backend.organisation_.collections)
    organisation.members = dict(backend.organisation_.members)
    organisation.trash = dict(backend.organisation_.trash)
    backend.organisation_ = organisation


def test_a_document_listing_refuses_a_foreign_row_rather_than_filtering_it() -> None:
    """A shortened list is a success report for a partial answer, so the whole read is refused.

    The refusal never quotes the foreign document's title or URI: a leak reported by quoting
    what leaked is still a leak.
    """
    backend, mine = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title="Their private notes")
    backend.store.add(theirs)
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    body = envelope(response)
    assert response.status_code == SERVER_ERROR
    assert body["ok"] is False
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert "Their private notes" not in response.text
    assert "secret.md" not in response.text
    assert mine.title not in response.text


def test_the_same_listing_against_a_correct_store_returns_the_tenant_s_document() -> None:
    """The control. Without it the test above proves only that the surface refuses everything."""
    backend, mine = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/api/v1/documents")
    body = envelope(response)
    assert body["ok"] is True
    assert [document["id"] for document in body["data"]["documents"]] == [mine.id]


def test_reading_another_tenants_document_by_id_is_a_refusal() -> None:
    """A store that hands back any id it holds is caught by the identity arithmetic.

    ``document_id`` is a digest of ``(workspace, source, source_id)``, so a document minted
    for another workspace cannot satisfy the check — there is no id it could carry that would.
    """
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md")
    backend.store.add(theirs)
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get(f"/api/v1/documents/{theirs.id}")
    body = envelope(response)
    assert body["ok"] is False, "another tenant's document was returned"
    assert body["error"]["type"] == "CrossWorkspaceError"


def test_deleting_another_tenants_document_is_refused_before_anything_is_written() -> None:
    """The refusal has to happen *before* the delete, not be reported after it."""
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md")
    backend.store.add(theirs)
    _leaky(backend)
    with client_for(backend) as client:
        response = client.delete(f"/api/v1/documents/{theirs.id}")
    body = envelope(response)
    assert body["ok"] is False, "a foreign document was deleted rather than refused"
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert backend.store.deleted == [], "a foreign document was deleted before the refusal"


def test_a_collection_listing_refuses_a_foreign_document() -> None:
    """Collections are a second route to documents, and it gets the same check.

    Built through the API first, so the collection and its membership are whatever the real
    routes produce; the store is swapped for the leaky one afterwards, carrying that state
    across. A test that hand-built the collection could get the membership shape wrong and
    then pass on a 404 while proving nothing about tenancy.
    """
    backend, mine = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title="Their private notes")
    backend.organisation_.documents[theirs.id] = theirs
    with client_for(backend) as client:
        created = envelope(client.post("/api/v1/collections", json={"name": "Everything"}))
        collection = str(created["data"]["id"])
        assert client.post(f"/api/v1/collections/{collection}/documents/{mine.id}").status_code
        assert client.post(f"/api/v1/collections/{collection}/documents/{theirs.id}").status_code
        both = envelope(client.get(f"/api/v1/collections/{collection}/documents"))
        # The control, in the same test: a correct store returns this tenant's document and
        # silently omits the other one, which is the *store's* scope doing its job.
        assert [item["id"] for item in both["data"]["documents"]] == [mine.id]
        _leaky(backend)
        response = client.get(f"/api/v1/collections/{collection}/documents")
    assert response.status_code == SERVER_ERROR
    assert envelope(response)["error"]["type"] == "CrossWorkspaceError"
    assert "Their private notes" not in response.text


def test_the_trash_refuses_a_foreign_document() -> None:
    """The trash is a listing of documents like any other, and is checked like one."""
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title="Their private notes")
    backend.organisation_.trash[theirs.id] = theirs
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get("/api/v1/documents/trash")
    body = envelope(response)
    # `ok` first, so a surface that stopped refusing fails on the claim rather than on a
    # `TypeError` from subscripting a null error.
    assert body["ok"] is False, "the trash returned a foreign document"
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert "Their private notes" not in response.text


def test_the_trash_against_a_correct_store_lists_this_tenants_document() -> None:
    """The control for the trash."""
    backend, mine = backend_with_a_document()
    backend.organisation_.trash[mine.id] = mine
    with client_for(backend) as client:
        body = envelope(client.get("/api/v1/documents/trash"))
    assert body["ok"] is True
    assert [entry["document"]["id"] for entry in body["data"]["documents"]] == [mine.id]


def test_the_workbench_refuses_a_foreign_document() -> None:
    """The workbench returns full passage text, so it is the worst route to leak from."""
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md")
    backend.store.add(theirs, make_chunk(theirs, text="their private passage"))
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get("/api/v1/workbench", params={"document_id": theirs.id})
    body = envelope(response)
    assert body["ok"] is False, "the workbench rendered another tenant's passages"
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert "their private passage" not in response.text


def test_a_search_refuses_when_retrieval_returns_another_tenants_chunk() -> None:
    """Refused whole, and before the passages are rendered.

    The retriever is told to return a chunk of a document this workspace cannot see. The
    ranking itself is a disclosure — it says a document exists and matches — so nothing comes
    back at all.
    """
    from manicule.core.retrieval import Candidate  # noqa: PLC0415 - only this test builds one

    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md")
    chunk = make_chunk(theirs, text="their private passage")
    backend.retriever_.candidates = [Candidate(chunk=chunk, score=0.9)]
    with client_for(backend) as client:
        response = client.get("/api/v1/search", params={"q": "retry"})
    body = envelope(response)
    assert body["ok"] is False, "a ranking over another tenant's chunk was returned"
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert "their private passage" not in response.text


def test_asking_a_question_refuses_before_the_model_is_called() -> None:
    """Asserted through the answerer's call log, not through the exception.

    An implementation that refused *after* streaming would raise exactly the same error having
    already sent another tenant's passages to a provider — which is the disclosure the check
    exists to prevent.
    """
    from manicule.core.retrieval import Candidate  # noqa: PLC0415 - only this test builds one

    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md")
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(theirs, text="their private passage"), score=0.9)
    ]
    with client_for(backend) as client:
        response = client.post("/api/v1/chat", json={"question": "what is it"})
    body = envelope(response)
    assert body["ok"] is False, "an answer was produced over another tenant's passages"
    assert body["error"]["type"] == "CrossWorkspaceError"
    assert backend.answerer_.calls == [], "the model was called with another tenant's passages"


def test_a_workspace_is_derived_rather_than_written_down() -> None:
    """The suite's own scaffolding, asserted.

    Every test above depends on ``make_document`` deriving an id the way the real code does.
    If it wrote one down, a tenancy test could be made to pass by editing a literal.
    """
    theirs = make_document(OTHER, source_id="secret.md")
    assert theirs.id == document_id(OTHER, "local", "secret.md")
    assert theirs.id != document_id("default", "local", "secret.md")
