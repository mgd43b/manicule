"""Documents: listing, reading, searching, the trash, and one delete.

Two decisions in this group are worth reading before the code.

**The delete is soft, and there is no hard one.** ``document_delete --hard`` exists on the
command line and is not reachable from here. A hard delete removes the document, its chunks
and its vectors with no restore path, and this surface is the one an unattended caller — a
script, a widget, an assistant holding a key — reaches. A soft delete is reversible through
``POST /documents/{id}/restore``, which is what makes the destructive version's absence a
recoverable inconvenience rather than a missing feature.

**There is no upload.** ``POST /api/v1/documents/upload`` is in the capability list and is not
here: accepting bytes over HTTP and writing them into the corpus is an ingest path with a
different threat model from every other one — no filesystem permission check, no path the
operator chose — and ``index_path`` over a directory the operator named is the ingest this
build offers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.security import MemberPrincipal, ViewerPrincipal

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.get("/documents", name="document_list", summary="A page of this workspace's documents.")
async def list_documents(
    service: Service,
    caller: ViewerPrincipal,
    *,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    source: Annotated[str | None, Query()] = None,
    media_type: Annotated[str | None, Query()] = None,
) -> Response:
    """Newest first, scoped to this workspace and checked again on the way out."""
    del caller
    return await respond(
        "document_list",
        service,
        lambda: service.document_list(
            limit=limit, offset=offset, source=source, media_type=media_type
        ),
    )


@router.get("/documents/trash", name="document_trash", summary="What is in the trash.")
async def trash(
    service: Service,
    caller: ViewerPrincipal,
    *,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """Longest-deleted first — the order the sweep will take them in.

    Declared **above** ``/documents/{document_id}``. Starlette matches routes in declaration
    order, so the parameterised one would otherwise swallow ``trash`` as an id and answer 404
    for a route that exists.
    """
    del caller
    return await respond(
        "document_trash", service, lambda: service.document_trash(limit=limit, offset=offset)
    )


@router.get(
    "/documents/{document_id}",
    name="document_get",
    summary="One document, optionally with its chunks.",
)
async def get_document(
    service: Service,
    caller: ViewerPrincipal,
    document_id: str,
    *,
    chunks: Annotated[bool, Query()] = False,
) -> Response:
    """One document of **this** workspace.

    A document belonging to another tenant is a 404 with the same message an absent one gets.
    Saying "it exists but is not yours" is itself a cross-tenant disclosure.
    """
    del caller
    return await respond(
        "document_get", service, lambda: service.document_get(document_id, chunks=chunks)
    )


@router.delete(
    "/documents/{document_id}", name="document_delete", summary="Move a document to the trash."
)
async def delete_document(service: Service, caller: MemberPrincipal, document_id: str) -> Response:
    """Soft delete, always.

    There is no ``hard`` parameter, deliberately. A hard delete is unrecoverable and this is
    the surface reachable by an unattended caller; the command line keeps that one.
    """
    del caller
    return await respond(
        "document_delete", service, lambda: service.document_delete(document_id, hard=False)
    )


@router.post(
    "/documents/{document_id}/restore",
    name="document_restore",
    summary="Take a document out of the trash.",
)
async def restore_document(service: Service, caller: MemberPrincipal, document_id: str) -> Response:
    """Restore, and say what that achieved.

    Inside the grace period it costs nothing; after the sweep has purged the content the
    document comes back empty and needs a re-parse. The payload says which happened, because
    those need different follow-ups and only one of them is finished.
    """
    del caller
    return await respond("document_restore", service, lambda: service.document_restore(document_id))


@router.post(
    "/documents/{document_id}/reindex", name="document_reindex", summary="Re-parse one document."
)
async def reindex_document(service: Service, caller: MemberPrincipal, document_id: str) -> Response:
    """Re-parse from the bytes ingest retained. Touches no network.

    Chunk ids are derived from content, so a chunk that survives unchanged keeps its vector.
    """
    del caller
    return await respond("document_reindex", service, lambda: service.document_reindex(document_id))


@router.get("/search", name="search", summary="Rank passages without asking a model anything.")
async def search(
    service: Service,
    caller: ViewerPrincipal,
    q: Annotated[str, Query(min_length=1, description="What to search for.")],
    *,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    profile: Annotated[str | None, Query()] = None,
    sources: Annotated[list[str] | None, Query()] = None,
    media_types: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """The cheap half of ``ask``: ranked passages, each with the score every stage gave it."""
    del caller
    return await respond(
        "search",
        service,
        lambda: service.search(
            q,
            limit=limit,
            profile=profile,
            sources=tuple(sources or ()),
            media_types=tuple(media_types or ()),
        ),
    )


__all__ = ["router"]
