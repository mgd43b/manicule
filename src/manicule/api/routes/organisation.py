"""Collections and tags: the two ways a person groups a corpus.

Two route groups in one module because they are one decision seen twice. What separates them
is what a duplicate name means, and the storage layer already encodes it: creating a
collection that already exists is refused — a collection is a deliberate object, and handing
back somebody else's under the same name merges two people's sets — while applying a tag that
already exists is the normal case, so ``POST /tags`` is idempotent.

Both are workspace-scoped, and a collection's documents are checked on the way out by the same
identity arithmetic every other listing uses.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.models import (
    CollectionBody,
    CollectionNameBody,
    CollectionUpdateBody,
    TagBody,
)
from manicule.api.security import MemberPrincipal, ViewerPrincipal

router = APIRouter(prefix="/api/v1", tags=["collections", "tags"])


@router.get("/collections", name="collection_list", summary="Every collection in this workspace.")
async def list_collections(service: Service, caller: ViewerPrincipal) -> Response:
    """Name, description and — where membership is rule-driven — the rule itself."""
    del caller
    return await respond("collection_list", service, service.collection_list)


@router.post("/collections", name="collection_create", summary="Create a collection.")
async def create_collection(
    service: Service, caller: MemberPrincipal, body: CollectionBody
) -> Response:
    """A duplicate name is a 409, not a merge."""
    del caller
    return await respond(
        "collection_create",
        service,
        lambda: service.collection_create(body.name, description=body.description),
    )


@router.get(
    "/collections/{collection_id}/documents",
    name="collection_documents",
    summary="A collection's documents.",
)
async def collection_documents(
    service: Service,
    caller: ViewerPrincipal,
    collection_id: str,
    *,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """A page, checked on the way out like every other document listing.

    Rule-driven membership is **evaluated**, not materialised, so what comes back is what the
    rule selects now rather than what it selected when somebody last saved it.
    """
    del caller
    return await respond(
        "collection_documents",
        service,
        lambda: service.collection_documents(collection_id, limit=limit, offset=offset),
    )


@router.get(
    "/collections/{collection_id}/counts",
    name="collection_counts",
    summary="A collection's document and chunk counts.",
)
async def collection_counts(
    service: Service, caller: ViewerPrincipal, collection_id: str
) -> Response:
    """Counted on the call, not remembered.

    A rule-driven collection has no materialised membership to keep a total for, so a stored
    number would be the answer to the day it was written.
    """
    del caller
    return await respond(
        "collection_counts", service, lambda: service.collection_counts(collection_id)
    )


@router.patch(
    "/collections/{collection_id}",
    name="collection_update",
    summary="Change a collection's description.",
)
async def update_collection(
    service: Service, caller: MemberPrincipal, collection_id: str, body: CollectionUpdateBody
) -> Response:
    """Membership is untouched, and nothing is re-indexed."""
    del caller
    return await respond(
        "collection_update",
        service,
        lambda: service.collection_update(collection_id, description=body.description),
    )


@router.post(
    "/collections/{collection_id}/name",
    name="collection_rename",
    summary="Rename a collection.",
)
async def rename_collection(
    service: Service, caller: MemberPrincipal, collection_id: str, body: CollectionNameBody
) -> Response:
    """A name is a label on a row: no document moves and nothing is re-embedded.

    Its own route rather than a field on the ``PATCH`` above, because renaming can fail in a
    way describing cannot — the name may already be in use — and one route returning either
    a 409 or a 200 depending on which field was present is a route a caller cannot reason
    about from its status code.
    """
    del caller
    return await respond(
        "collection_rename", service, lambda: service.collection_rename(collection_id, body.name)
    )


@router.post(
    "/collections/{collection_id}/documents/{document_id}",
    name="collection_add",
    summary="Add a document to a collection.",
)
async def add_to_collection(
    service: Service, caller: MemberPrincipal, collection_id: str, document_id: str
) -> Response:
    """Idempotent: adding a document already in the collection changes nothing and says so."""
    del caller
    return await respond(
        "collection_add", service, lambda: service.collection_add(collection_id, [document_id])
    )


@router.delete(
    "/collections/{collection_id}/documents/{document_id}",
    name="collection_remove",
    summary="Remove a document from a collection.",
)
async def remove_from_collection(
    service: Service, caller: MemberPrincipal, collection_id: str, document_id: str
) -> Response:
    """The document itself is untouched."""
    del caller
    return await respond(
        "collection_remove",
        service,
        lambda: service.collection_remove(collection_id, [document_id]),
    )


@router.delete(
    "/collections/{collection_id}", name="collection_delete", summary="Delete a collection."
)
async def delete_collection(
    service: Service, caller: MemberPrincipal, collection_id: str
) -> Response:
    """The documents in it are untouched. A collection is a grouping, not a container."""
    del caller
    return await respond(
        "collection_delete", service, lambda: service.collection_delete(collection_id)
    )


@router.get("/tags", name="tag_list", summary="Every tag in this workspace.")
async def list_tags(service: Service, caller: ViewerPrincipal) -> Response:
    """Names are case-sensitive and normalised to NFKC, so two keyboards produce one tag."""
    del caller
    return await respond("tag_list", service, service.tag_list)


@router.post(
    "/tags", name="tag_create", summary="Create a tag, or return the existing one of that name."
)
async def create_tag(service: Service, caller: MemberPrincipal, body: TagBody) -> Response:
    """Idempotent by design. There is no strict variant to reach for by mistake."""
    del caller
    return await respond(
        "tag_create", service, lambda: service.tag_create(body.name, color=body.color)
    )


@router.delete("/tags/{tag_id}", name="tag_delete", summary="Delete a tag.")
async def delete_tag(service: Service, caller: MemberPrincipal, tag_id: str) -> Response:
    """Documents keep their other tags."""
    del caller
    return await respond("tag_delete", service, lambda: service.tag_delete(tag_id))


@router.post(
    "/documents/{document_id}/tags/{tag_id}",
    name="document_tag",
    summary="Apply a tag to a document.",
)
async def tag_document(
    service: Service, caller: MemberPrincipal, document_id: str, tag_id: str
) -> Response:
    """Both the document and the tag must belong to this workspace."""
    del caller
    return await respond(
        "document_tag", service, lambda: service.document_tag(document_id, [tag_id])
    )


@router.delete(
    "/documents/{document_id}/tags/{tag_id}", name="document_untag", summary="Remove a tag."
)
async def untag_document(
    service: Service, caller: MemberPrincipal, document_id: str, tag_id: str
) -> Response:
    """Removing a tag the document does not carry changes nothing and says so."""
    del caller
    return await respond(
        "document_untag", service, lambda: service.document_untag(document_id, [tag_id])
    )


__all__ = ["router"]
