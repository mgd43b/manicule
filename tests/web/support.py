"""Scaffolding for the browser surface's suites.

Built on ``tests.api.support`` rather than beside it: the browser surface is mounted on the
same application, resolves the same principal and runs through the same middleware, so a second
way of building one would be testing something that is not what runs.

The one thing added here is a corpus that is **hostile on purpose**. Every string in
:data:`MARKUP` is a place where text somebody else wrote reaches HTML, and each is planted in
the field a real attacker would use: a document title comes from a file, a heading path comes
from a parser reading that file, an answer body is a model writing about it, and a citation
label is the document's own words travelling back out under manicule's name.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind
from manicule.core.provenance import LocalSnapshot, Provenance, SourceMetadata
from manicule.core.retrieval import Candidate
from manicule.generation.answers import Citation, Verification
from manicule.generation.history import Turn
from manicule.generation.sharing import hash_token
from tests.api.support import backend_with_a_document, client_for
from tests.app.fakes import make_chunk, make_document

if TYPE_CHECKING:
    from manicule.core.content import Document
    from tests.app.fakes import FakeBackend

__all__ = [
    "CONVERSATION",
    "MARKUP",
    "PAGES",
    "SHARE_TOKEN",
    "backend_with_a_document",
    "backend_with_hostile_text",
    "client_for",
    "hostile_provenance",
    "pages_of",
]

MARKUP: dict[str, str] = {
    "title": "<script>window.__title=1</script>Notes",
    "heading": "<img src=x onerror=window.__heading=1>Retry policy",
    "answer": "The client retries twice.<script>window.__answer=1</script>",
    "quote": "<svg onload=window.__quote=1>the client retries twice",
    "canonical": "<script>window.__canonical=1</script>Retry policy",
    "section": "<iframe srcdoc='<script>window.__section=1</script>'>Runbooks",
}
"""One hostile string per place model output or corpus content reaches HTML.

Each carries a distinct payload so a test can say *which* field escaped rather than only that
something did. They are deliberately different shapes — an element, an attribute handler on an
image, a trailing script, an SVG — because a template that escaped one and not another is the
realistic failure, not one that escaped nothing.

``canonical`` is the **authoritative source title**, and it belongs here for exactly the reason
``title`` does: it is a string somebody else wrote that arrives on this surface. It is not a
duplicate of ``title``. A sidecar manifest is a file in the corpus, so anybody who can get a
document indexed can supply one — and unlike a filename, which a filesystem constrains to no
slashes and a length limit, a manifest field is arbitrary JSON text. It reaches the same pages by
a different route: ``title`` is what the connector discovered, ``canonical`` is what the manifest
declared and the pipeline preferred over it. A test that only planted ``title`` would be
exercising the route that is now the *fallback*.
"""

CONVERSATION = "conv-hostile"
SHARE_TOKEN = "a-token-that-resolves"  # noqa: S105 - a share token, and a fixture one


def hostile_provenance() -> Provenance:
    """A source record whose every free-text field is hostile.

    The canonical URI is **not** hostile, and cannot be: the interface restricts it to ``http``
    and ``https`` before it can be stored, and re-validates on every read, so there is no way to
    get a ``javascript:`` address into a record that a page will render. That refusal is asserted
    in ``tests/test_provenance.py``; what is left for this surface to prove is the fields that
    are legitimately arbitrary text — a title and a hierarchy somebody wrote in a JSON file.
    """
    return Provenance(
        source=SourceMetadata(
            title=MARKUP["canonical"],
            canonical_uri="https://docs.example.test/pages/123456/retry-policy",
            source_id="123456",
            version="7",
            section_path=(MARKUP["section"],),
        ),
        snapshot=LocalSnapshot(path="mirror/123456.html"),
    )


def _hostile_citation(document: Document) -> Citation:
    return Citation(
        slot=1,
        document_id=document.id,
        uri=document.uri,
        title=MARKUP["title"],
        heading_path=(MARKUP["heading"],),
        anchor=HeadingAnchor(path=(MARKUP["heading"],)),
        chunk_id=make_chunk(document).id,
        kind=BlockKind.PROSE,
        quote=MARKUP["quote"],
        verification=Verification.RESOLVED,
    )


def backend_with_hostile_text() -> tuple[FakeBackend, Document]:
    """A backend whose document, conversation, share link, search hits and trash are hostile.

    The document is indexed with markup in its title and in its chunk's heading path; a
    conversation holds an answer with markup in the body and a citation with markup in its
    label and quote; and that conversation is shared, so the anonymous page renders the same
    citation through the redaction rather than through a second path.

    **The retrieval candidate and the trash entry are seeded here deliberately.** Without them
    the fake retriever returns nothing and the trash is empty, so ``/ui/search`` renders *no
    hits* and ``/ui/documents/trash`` renders *the trash is empty* — and every assertion about
    those two pages walks past the markup it means to check. Both render a **document title**,
    which is the field an attacker controls by naming a file, so leaving them on their empty
    branch left the two cheapest routes to a title on this surface unexercised.
    """
    backend, document = backend_with_a_document()
    hostile = make_document(
        backend.settings.workspace,
        source_id="hostile.md",
        title=MARKUP["title"],
        # A sidecar manifest is a file in the corpus, so its fields are attacker-controlled text
        # arriving by a *different* route from the filename — and unlike a filename, arbitrary
        # JSON rather than something a filesystem constrains. The title here is deliberately not
        # the same string as `title` above so that a page which escaped one and not the other
        # says which.
        provenance=hostile_provenance(),
    )
    chunk = make_chunk(hostile, text=MARKUP["quote"]).model_copy(
        update={"heading_path": (MARKUP["heading"],)}
    )
    backend.store.add(hostile, chunk)
    backend.organisation_.documents[hostile.id] = hostile

    # What `/ui/search` ranks. The same chunk the store holds, so the hit's title, heading path
    # and text are the hostile document's own rather than a second fixture that could drift.
    backend.retriever_.candidates = [Candidate(chunk=chunk, score=0.5)]
    # What `/ui/documents/trash` lists.
    backend.organisation_.trash[hostile.id] = hostile

    turn = Turn(role="assistant", content=MARKUP["answer"], citations=(_hostile_citation(hostile),))
    backend.conversations_.seed(CONVERSATION, Turn(role="user", content="what happens"), turn)
    moment = datetime.now(UTC)
    backend.conversations_.shares[CONVERSATION] = (
        hash_token(SHARE_TOKEN),
        moment + timedelta(hours=1),
        moment + timedelta(hours=1),
    )
    del document
    return backend, hostile


def pages_of(document_id: str) -> tuple[str, ...]:
    """Every page of the browser surface, for a suite that has to visit all of them."""
    return (
        "/ui",
        "/ui/chat",
        f"/ui/chat/{CONVERSATION}",
        "/ui/documents",
        "/ui/documents/trash",
        f"/ui/documents/{document_id}",
        "/ui/search?q=retry",
        "/ui/collections",
        "/ui/connectors",
        "/ui/health",
        "/ui/plugins",
        "/ui/settings",
        "/ui/workspaces",
        "/ui/admin",
        "/ui/auth",
        f"/ui/shared/{SHARE_TOKEN}",
    )


PAGES = pages_of
