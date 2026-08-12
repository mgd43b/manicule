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
    "pages_of",
]

MARKUP: dict[str, str] = {
    "title": "<script>window.__title=1</script>Notes",
    "heading": "<img src=x onerror=window.__heading=1>Retry policy",
    "answer": "The client retries twice.<script>window.__answer=1</script>",
    "quote": "<svg onload=window.__quote=1>the client retries twice",
}
"""One hostile string per place model output or corpus content reaches HTML.

Each carries a distinct payload so a test can say *which* field escaped rather than only that
something did. They are deliberately different shapes — an element, an attribute handler on an
image, a trailing script, an SVG — because a template that escaped one and not another is the
realistic failure, not one that escaped nothing.
"""

CONVERSATION = "conv-hostile"
SHARE_TOKEN = "a-token-that-resolves"  # noqa: S105 - a share token, and a fixture one


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
    """A backend whose one document, one conversation and one share link are all hostile.

    The document is indexed with markup in its title and in its chunk's heading path; a
    conversation holds an answer with markup in the body and a citation with markup in its
    label and quote; and that conversation is shared, so the anonymous page renders the same
    citation through the redaction rather than through a second path.
    """
    backend, document = backend_with_a_document()
    hostile = make_document(
        backend.settings.workspace, source_id="hostile.md", title=MARKUP["title"]
    )
    chunk = make_chunk(hostile, text=MARKUP["quote"])
    backend.store.add(hostile, chunk.model_copy(update={"heading_path": (MARKUP["heading"],)}))
    backend.organisation_.documents[hostile.id] = hostile

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
