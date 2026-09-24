"""Identity: who this request is, and the keys that make one.

**There is no interactive login.** OAuth is #13's, and the honest thing to do until it exists
is to say so rather than to offer ``/auth/login/{provider}`` routes that would refuse every
attempt — an endpoint that exists and always fails is worse documentation than one that is
absent, because it reads as a bug rather than as a decision.

**Minting, listing and revoking a key are viewer-floor routes, and the service decides the
rest.** That reads oddly next to "an authenticated caller may administer keys" until the
ownership rule is in view: :meth:`~manicule.app.service.ApplicationService.api_key_create`
refuses a caller with no ``user_id`` and no admin authority outright, caps a non-admin's
requested role at their own, and scopes what
:meth:`~manicule.app.service.ApplicationService.api_key_list` and
:meth:`~manicule.app.service.ApplicationService.api_key_revoke` show or touch to keys that
caller owns. Asking for more than viewer here would be a second, looser copy of that rule — the
floor a route asks for is "is there a caller at all", and the service is where "what may *this*
caller do" is decided once.

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
from manicule.api.security import AnonymousPrincipal, ViewerPrincipal
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
async def create_key(service: Service, caller: ViewerPrincipal, body: KeyBody) -> Response:
    """Return the only copy of a key's secret.

    Audited, and the ownership and role-cap rule the module docstring describes is enforced by
    the service against :func:`~manicule.app.caller.current` — not by this route's floor, which
    only asks that a caller exists at all.
    """
    del caller
    return as_response(
        await run_op(
            "api_key_create",
            service.workspace,
            lambda: service.api_key_create(
                body.name,
                role=body.role,
                expires_days=body.expires_days,
                allowed_ips=body.allowed_ips,
                rate_limit=body.rate_limit,
            ),
        )
    )


@router.get("/api/v1/auth/keys", name="api_key_list", summary="This caller's keys.")
async def list_keys(service: Service, caller: ViewerPrincipal) -> Response:
    """Records, never secrets. Only digests are stored, so there is no secret to return.

    Every key in the workspace for an admin or the local operator; only the caller's own
    otherwise — see :meth:`~manicule.app.service.ApplicationService.api_key_list`.
    """
    del caller
    return as_response(await run_op("api_key_list", service.workspace, service.api_key_list))


@router.delete("/api/v1/auth/keys/{name_or_id}", name="api_key_revoke", summary="Revoke a key.")
async def revoke_key(service: Service, caller: ViewerPrincipal, name_or_id: str) -> Response:
    """Immediate, and scoped to this workspace and — for a non-admin caller — to keys they own.

    A revoke that could reach another tenant's key, or another person's, would be a denial of
    service across a boundary the whole design exists to hold, so both scopes are enforced in
    the store's own lookup.
    """
    del caller
    return as_response(
        await run_op(
            "api_key_revoke", service.workspace, lambda: service.api_key_revoke(name_or_id)
        )
    )


__all__ = ["router"]
