"""Identity: who this request is, and the keys that make one.

**Minting a key is an admin operation and there is no interactive login.** OAuth is #13's, and
the honest thing to do until it exists is to say so rather than to offer ``/auth/login/{provider}``
routes that would refuse every attempt — an endpoint that exists and always fails is worse
documentation than one that is absent, because it reads as a bug rather than as a decision.

``GET /auth/session`` is what a client uses to find out whether its credential works. It is
deliberately not a route that *creates* a session: there is no session cookie in this build, a
key is presented on every request, and a signed cookie would be a second credential type with
its own expiry, revocation and CSRF story.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from manicule.api.context import Service
from manicule.api.envelopes import as_response
from manicule.api.models import KeyBody
from manicule.api.security import AdminPrincipal, AnonymousPrincipal
from manicule.app.dispatch import run_op

router = APIRouter(tags=["auth"])


@router.get(
    "/auth/providers", name="auth_providers", summary="Which identity providers are configured."
)
async def providers(service: Service, caller: AnonymousPrincipal) -> Response:
    """Names and types only, never a client secret.

    Unauthenticated on purpose: a client has to be able to find out *how* to authenticate
    before it has authenticated. It discloses which of two or three well-known provider types
    an operator configured, and nothing about the installation's contents.
    """
    del caller
    return as_response(await run_op("auth_providers", service.workspace, service.auth_providers))


@router.get("/auth/session", name="auth_session", summary="Who this request is.")
async def session(service: Service, caller: AnonymousPrincipal) -> Response:
    """The caller's identity, as this installation resolved it.

    Answers for an unauthenticated caller too — ``authenticated: false`` with the configured
    mode — because "your key is not working" and "this server wants no key" are different
    problems and a 401 conflates them.

    The workspace on the envelope is the one this process serves; the workspace *in the
    payload* is the one the key was minted for. On a valid key they are the same, and they are
    both reported because that is the property a caller most wants to be sure of.
    """
    from manicule.app.results import succeeded  # noqa: PLC0415 - only this route renders one

    del service
    return as_response(
        succeeded("auth_session", caller.identity.workspace or "unknown", caller.identity)
    )


@router.post("/api/v1/auth/keys", name="api_key_create", summary="Mint an API key.")
async def create_key(service: Service, caller: AdminPrincipal, body: KeyBody) -> Response:
    """Return the only copy of a key's secret.

    Admin-only, and audited. A key is an identity: anything that can mint one can mint one
    with a role it does not itself have unless something stops it, and being an admin is what
    stops it here.
    """
    del caller
    return as_response(
        await run_op(
            "api_key_create",
            service.workspace,
            lambda: service.api_key_create(
                body.name, role=body.role, expires_days=body.expires_days
            ),
        )
    )


@router.get("/api/v1/auth/keys", name="api_key_list", summary="Every key in this workspace.")
async def list_keys(service: Service, caller: AdminPrincipal) -> Response:
    """Records, never secrets. Only digests are stored, so there is no secret to return."""
    del caller
    return as_response(await run_op("api_key_list", service.workspace, service.api_key_list))


@router.delete("/api/v1/auth/keys/{name_or_id}", name="api_key_revoke", summary="Revoke a key.")
async def revoke_key(service: Service, caller: AdminPrincipal, name_or_id: str) -> Response:
    """Immediate, and scoped to this workspace.

    A revoke that could reach another tenant's key would be a denial of service across the
    boundary the whole design exists to hold, so the lookup is workspace-scoped in the store.
    """
    del caller
    return as_response(
        await run_op(
            "api_key_revoke", service.workspace, lambda: service.api_key_revoke(name_or_id)
        )
    )


__all__ = ["router"]
