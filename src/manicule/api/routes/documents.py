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

**``POST /documents`` is not that upload**, and the sentence above is the test it has to pass.
It takes no path and no filename: a slug becomes ``<collection>/<slug>.md`` beneath the root of
a filesystem source an operator configured for authoring, in one of a configured set of
collections, and an installation that configured neither refuses the call. So the two things the
upload lacked — a filesystem boundary somebody chose, and a path that is not the caller's — are
exactly what this has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.models import DocumentBody
from manicule.api.security import MemberPrincipal, ViewerPrincipal, require
from manicule.config.settings import Role

if TYPE_CHECKING:
    from manicule.app.results import Payload

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
    order, so the parameterized one would otherwise swallow ``trash`` as an id and answer 404
    for a route that exists.
    """
    del caller
    return await respond(
        "document_trash", service, lambda: service.document_trash(limit=limit, offset=offset)
    )


@router.get(
    "/documents/resolve",
    name="document_resolve",
    summary="One cached document by page id, URI or document id, with its retained bytes.",
)
async def resolve_document(
    service: Service,
    caller: ViewerPrincipal,
    *,
    document_id: Annotated[str | None, Query(min_length=1)] = None,
    source: Annotated[str | None, Query(min_length=1)] = None,
    source_id: Annotated[str | None, Query(min_length=1)] = None,
    uri: Annotated[str | None, Query(min_length=1)] = None,
    max_age_s: Annotated[float | None, Query(gt=0)] = None,
    content: Annotated[bool, Query()] = True,
) -> Response:
    """Read a document out of the cache without knowing manicule's own id for it.

    This is the endpoint another local program calls when it wants the Confluence page behind
    an id or a URL and does not want to hold Confluence credentials of its own. It serves the
    bytes the connector fetched, and it reaches no network: ``max_age_s`` is reported against,
    never enforced, because this installation cannot know what the source has done since.

    Declared **above** ``/documents/{document_id}`` for the reason ``/documents/trash`` is:
    Starlette matches in declaration order, so the parameterized route would otherwise take
    ``resolve`` for an id and 404 a route that exists.
    """
    del caller
    return await respond(
        "document_resolve",
        service,
        lambda: service.document_resolve(
            document_id=document_id,
            source=source,
            source_id=source_id,
            uri=uri,
            max_age_s=max_age_s,
            content=content,
        ),
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


@router.post("/documents", name="document_create", summary="Author a document into the corpus.")
async def create_document(
    service: Service, caller: MemberPrincipal, body: DocumentBody
) -> Response:
    """Write a markdown file into the configured authoring source and index it.

    **This is not the upload the module docstring says is absent**, and the difference is the
    whole of why it is here. An upload takes bytes and a path from the caller; this takes a slug
    and derives ``<collection>/<slug>.md`` beneath a root an operator configured, into one of a
    configured set of collections, so nothing the caller sends decides where anything is written.
    An installation that has configured no authoring source refuses every call to it.

    A write that could not be indexed answers ``ok: false`` with the payload still attached, so
    the path of the file that was kept is in the response rather than only in a log.
    """
    del caller
    return await respond(
        "document_create",
        service,
        lambda: service.document_create(
            collection=body.collection,
            slug=body.slug,
            body=body.body,
            overwrite=body.overwrite,
        ),
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
    workspaces: Annotated[
        list[str] | None,
        Query(
            description="Search these workspaces together, merged into one ranking whose hits "
            "each name their workspace. An administrator's search: naming any workspace but "
            "this one needs the admin role."
        ),
    ] = None,
) -> Response:
    """The cheap half of ``ask``: ranked passages, each with the score every stage gave it.

    **Naming other workspaces raises the floor to admin**, here as well as in the service. The
    service's check is the rule, and it holds on every surface; this one refuses with the
    route's own ``ForbiddenError`` before a single workspace is opened, whatever identity the
    service has been told it is acting for. It asks the service's own question —
    :meth:`~manicule.app.service.ApplicationService.crosses_workspaces` — so the two cannot
    disagree about which requests span workspaces, and it asks it inside the dispatched call, so
    a refusal is the ordinary envelope at the ordinary 403.
    """
    spanned = tuple(workspaces) if workspaces else None

    async def searched() -> Payload:
        if service.crosses_workspaces(spanned):
            require(caller, Role.ADMIN)
        return await service.search(
            q,
            limit=limit,
            profile=profile,
            sources=tuple(sources or ()),
            media_types=tuple(media_types or ()),
            workspaces=spanned,
        )

    return await respond("search", service, searched)


__all__ = ["router"]
