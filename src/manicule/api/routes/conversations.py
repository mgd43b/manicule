"""Conversations, and the one unauthenticated route in the whole surface.

``GET /shared/{token}`` is the highest-risk endpoint here by a distance: no credential, and it
returns conversation content. Everything that makes it safe is somewhere else, deliberately —
:mod:`manicule.generation.sharing` mints and hashes the token,
:meth:`~manicule.storage.conversations.SqliteConversationStore.shared_conversation` resolves
it in one statement with expiry, revocation, soft-delete and the snapshot boundary as
predicates of that statement, and the anonymous projection to citation *labels* happens in
storage. This module carries a path parameter to the service and renders what comes back.

That is the point. A route that assembled the answer itself would be a second path to
conversation data, and the second path is the one where a predicate gets left out.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.models import ConversationBody, ConversationPatch, ShareBody
from manicule.api.security import AnonymousPrincipal, MemberPrincipal, ViewerPrincipal

router = APIRouter(tags=["conversations"])


@router.get("/api/v1/conversations", summary="This workspace's conversations.")
async def list_conversations(
    service: Service,
    caller: ViewerPrincipal,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """Most recently touched first, with whether each is shared and until when.

    Never the share token. The stored form is a hash precisely so that a listing cannot hand
    out a bearer capability, and a field for it here would put one into every log and cache in
    front of this route.
    """
    del caller
    return await respond(
        "conversation_list",
        service,
        lambda: service.conversation_list(limit=limit, offset=offset),
    )


@router.post("/api/v1/conversations", summary="Start a conversation.")
async def create_conversation(
    service: Service, caller: MemberPrincipal, body: ConversationBody
) -> Response:
    """A new, empty conversation in this workspace."""
    del caller
    return await respond(
        "conversation_create", service, lambda: service.conversation_create(title=body.title)
    )


@router.get("/api/v1/conversations/{conversation_id}/messages", summary="A conversation's turns.")
async def messages(
    service: Service,
    caller: ViewerPrincipal,
    conversation_id: str,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Response:
    """Oldest first, with the **full** citations each answer carried.

    This is the owner's view: the reader holds a key for this workspace and could have
    retrieved those passages themselves. The anonymous view is a different route over a
    different query returning a different type.
    """
    del caller
    return await respond(
        "conversation_messages",
        service,
        lambda: service.conversation_messages(conversation_id, limit=limit),
    )


@router.patch("/api/v1/conversations/{conversation_id}", summary="Retitle a conversation.")
async def patch_conversation(
    service: Service, caller: MemberPrincipal, conversation_id: str, body: ConversationPatch
) -> Response:
    """The title is the only changeable field. Turns are a record, not a draft."""
    del caller
    return await respond(
        "conversation_rename",
        service,
        lambda: service.conversation_rename(conversation_id, body.title),
    )


@router.delete("/api/v1/conversations/{conversation_id}", summary="Delete a conversation.")
async def delete_conversation(
    service: Service, caller: MemberPrincipal, conversation_id: str
) -> Response:
    """Soft delete, which also revokes any share link over it.

    The revocation is not a courtesy. A public read that lacked the soft-delete predicate is
    exactly how deleting a conversation leaves its contents readable to anyone holding the URL.
    """
    del caller
    return await respond(
        "conversation_delete", service, lambda: service.conversation_delete(conversation_id)
    )


@router.post("/api/v1/conversations/{conversation_id}/share", summary="Mint a share link.")
async def share_conversation(
    service: Service, caller: MemberPrincipal, conversation_id: str, body: ShareBody
) -> Response:
    """Return the only copy of a share token.

    The token is in the response and nowhere else — the database holds a digest. Minting
    replaces any previous link, so re-sharing invalidates the old token and produces a new
    snapshot. A requested lifetime is clamped to ``security.sharing.link_ttl_s``, and the
    store refuses one past the ceiling outright.
    """
    del caller
    return await respond(
        "conversation_share",
        service,
        lambda: service.conversation_share(conversation_id, ttl_s=body.ttl_s),
    )


@router.delete("/api/v1/conversations/{conversation_id}/share", summary="Revoke a share link.")
async def unshare_conversation(
    service: Service, caller: MemberPrincipal, conversation_id: str
) -> Response:
    """Revocation clears the stored hash, so the link stops resolving rather than looking
    revoked beside a token that still works."""
    del caller
    return await respond(
        "conversation_unshare", service, lambda: service.conversation_unshare(conversation_id)
    )


@router.get("/shared/{token}", summary="Read a shared conversation. No credential.")
async def shared(service: Service, caller: AnonymousPrincipal, token: str) -> Response:
    """The anonymous view of a shared conversation.

    Citation **labels** only: a title, a breadcrumb where the block kind permits one, a page
    number where there is one, and whether the claim verified. No passage text, no document or
    chunk id, no URI, no anchor — and no conversation id either, so holding the link does not
    become a way to address the conversation anywhere else.

    An unknown token, an expired link, a revoked link, a deleted conversation and sharing
    being switched off all return the same empty result. Distinguishing them for an
    unauthenticated caller tells them which of their guesses was closest.
    """
    del caller
    return await respond("shared_conversation", service, lambda: service.shared_conversation(token))


__all__ = ["router"]
