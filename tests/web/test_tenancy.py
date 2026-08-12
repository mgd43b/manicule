"""No page reads across workspaces, and the check that stops it is not this surface's own.

The method is the one ``tests/app/test_tenancy.py`` established and ``tests/api/test_tenancy.py``
repeated: drive the pages against stores that **ignore their workspace scope entirely** and
watch the refusal happen. A page driven against a correct store would render correctly whether
or not any check existed.

:class:`~tests.app.fakes.LeakyStore` and :class:`~tests.app.fakes.LeakyOrganisation` ignore the
filter *and* the limit, which is what makes them useful: a leaky store that still truncated
would let a "some of what I asked for came back missing" check catch a foreign row by accident,
and the identity arithmetic — which would still fire if every ``WHERE`` clause in storage were
deleted — would never be exercised.

Every test here also asserts the **control**: the same page against a correct store renders this
tenant's own document. Without it, a surface that refused everything would pass.

The assertion this file adds over the API's is that the refusal is checked **against the
rendered HTML**. A page is where a leak would actually be read, and a title that never reached
the payload could still reach the frame — through a heading, a link title, a page title.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.ids import document_id
from tests.api.support import backend_with_a_document, client_for
from tests.app.fakes import LeakyOrganisation, LeakyStore, make_chunk, make_document
from tests.web.support import CONVERSATION, backend_with_hostile_text

if TYPE_CHECKING:
    from tests.app.fakes import FakeBackend

OTHER = "another-tenant"
SERVER_ERROR = 500
FOREIGN_TITLE = "Their private notes"
FOREIGN_TEXT = "their private passage"


def _leaky(backend: FakeBackend) -> None:
    """Replace this backend's stores with ones that ignore their scope, keeping the contents.

    The same substitution ``tests/api/test_tenancy.py`` makes, and deliberately so: the two
    suites have to be driving the *same* broken components, or a refusal seen in one is not
    evidence about the other.
    """
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


def test_the_document_page_refuses_a_foreign_row_rather_than_filtering_it() -> None:
    """A shortened list is a success report for a partial answer, so the read is refused whole.

    The rendered page is what is checked: neither the foreign title nor its identifier appears
    anywhere in the HTML, including in the error the page shows. A leak reported by quoting
    what leaked is still a leak.
    """
    backend, mine = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title=FOREIGN_TITLE)
    backend.store.add(theirs)
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get("/ui/documents")
    assert response.status_code == SERVER_ERROR
    assert "CrossWorkspaceError" in response.text
    assert FOREIGN_TITLE not in response.text
    assert "secret.md" not in response.text
    assert mine.title not in response.text


def test_the_same_page_against_a_correct_store_lists_this_tenants_document() -> None:
    """The control. Without it the test above proves only that the page refuses everything."""
    backend, mine = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/ui/documents")
    assert response.status_code == 200
    assert mine.title in response.text
    assert mine.id in response.text


def test_a_foreign_document_page_is_refused_and_quotes_nothing() -> None:
    """The detail page renders full passage text, so it is the worst page to leak from."""
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title=FOREIGN_TITLE)
    backend.store.add(theirs, make_chunk(theirs, text=FOREIGN_TEXT))
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get(f"/ui/documents/{theirs.id}")
    assert response.status_code == SERVER_ERROR
    assert "CrossWorkspaceError" in response.text
    assert FOREIGN_TEXT not in response.text
    assert FOREIGN_TITLE not in response.text


def test_the_document_page_against_a_correct_store_renders_its_blocks() -> None:
    """The control for the detail page: the tenant's own passage is rendered."""
    backend, mine = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(f"/ui/documents/{mine.id}")
    assert response.status_code == 200
    assert "the client retries twice" in response.text


def test_the_trash_page_refuses_a_foreign_document() -> None:
    """The trash is a listing of documents like any other, and is checked like one."""
    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title=FOREIGN_TITLE)
    backend.organisation_.trash[theirs.id] = theirs
    _leaky(backend)
    with client_for(backend) as client:
        response = client.get("/ui/documents/trash")
    assert response.status_code == SERVER_ERROR
    assert FOREIGN_TITLE not in response.text


def test_the_trash_page_against_a_correct_store_lists_this_tenants_document() -> None:
    """The control for the trash."""
    backend, mine = backend_with_a_document()
    backend.organisation_.trash[mine.id] = mine
    with client_for(backend) as client:
        response = client.get("/ui/documents/trash")
    assert response.status_code == 200
    assert mine.title in response.text


def test_the_search_page_refuses_a_ranking_over_another_tenants_chunk() -> None:
    """The ranking is itself a disclosure — it says a document exists and matches."""
    from manicule.core.retrieval import Candidate  # noqa: PLC0415 - only this test builds one

    backend, _ = backend_with_a_document()
    theirs = make_document(OTHER, source_id="secret.md", title=FOREIGN_TITLE)
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(theirs, text=FOREIGN_TEXT), score=0.9)
    ]
    with client_for(backend) as client:
        response = client.get("/ui/search", params={"q": "retry"})
    assert response.status_code == SERVER_ERROR
    assert "CrossWorkspaceError" in response.text
    assert FOREIGN_TEXT not in response.text


def test_the_search_page_against_a_correct_store_renders_the_passage() -> None:
    """The control for search, and it has to plant a candidate to be one.

    The retriever fake returns nothing unless told to, so a control that merely searched would
    render an empty page and pass — proving that the refusal above is not simply "this page
    never shows a passage".
    """
    from manicule.core.retrieval import Candidate  # noqa: PLC0415 - only this test builds one

    backend, mine = backend_with_a_document()
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(mine), score=0.9)]
    with client_for(backend) as client:
        response = client.get("/ui/search", params={"q": "retry"})
    assert response.status_code == 200
    assert "the client retries twice" in response.text


def test_a_shared_page_is_still_scoped_to_the_conversation_the_token_names() -> None:
    """A token resolves one conversation, not the store.

    The control is the positive case in the same test: the right token renders the turn, and a
    token that is one character different renders nothing at all.
    """
    backend, _ = backend_with_hostile_text()
    from tests.web.support import SHARE_TOKEN  # noqa: PLC0415 - read beside its near-miss

    with client_for(backend) as client:
        good = client.get(f"/ui/shared/{SHARE_TOKEN}")
        bad = client.get(f"/ui/shared/{SHARE_TOKEN}x")
    assert "assistant" in good.text
    assert "does not resolve" in bad.text
    assert CONVERSATION not in bad.text


def test_a_workspace_is_derived_rather_than_written_down() -> None:
    """This suite's own scaffolding, asserted.

    Every test above depends on ``make_document`` deriving an id the way the real code does. If
    it wrote one down, a tenancy test could be made to pass by editing a literal.
    """
    theirs = make_document(OTHER, source_id="secret.md")
    assert theirs.id == document_id(OTHER, "local", "secret.md")
    assert theirs.id != document_id("default", "local", "secret.md")
