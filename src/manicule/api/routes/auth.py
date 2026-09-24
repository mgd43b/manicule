"""Identity: who this request is, how a browser signs in, and the people and keys that make one.

**A program presents a key; a person at a browser signs in.** A key is minted at the command
line or by a signed-in person, and presented on every request in a header. A browser cannot
attach a header to a page load, so under ``security.auth.mode = 'oauth'`` a person signs in
through an identity provider instead, and the browser holds a session cookie from then on. Both
arrive at the same principal through :func:`~manicule.api.security.identify`, and every route
below that takes a floor is held to it whichever credential was presented.

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

**The sign-in is three routes and nothing decides anything in them.** ``GET /auth/login/{type}``
sends the browser to the provider with a fresh ``state`` and a PKCE challenge, carrying both in
a short-lived signed cookie; ``GET /auth/callback/{type}`` checks the browser brought them back,
exchanges the code, and asks the service whether the person the provider named is admitted;
``POST /auth/logout`` ends the session on the server as well as in the browser. Which provider
applies, whether a person is admitted and in what role are the service's —
:meth:`~manicule.app.service.ApplicationService.sign_in` — and the protocol is
:mod:`manicule.api.oauth`'s. :mod:`manicule.api.cookies` says what each cookie carries and why
each attribute is what it is.

**What a person is shown is a page, and never a value that crossed the wire.** The callback is
reached by a browser, so every outcome is HTML — a refusal at the status the API would have
used, or a short page that sends the browser on. No authorization code, token or provider error
description appears in either.

``GET /auth/session`` is what any client uses to find out whether its credential works, and it
reports how the caller authenticated and, for a person, who they are.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import RedirectResponse

from manicule.api import oauth
from manicule.api.context import Service
from manicule.api.cookies import (
    SESSION_COOKIE,
    SIGNIN_COOKIE,
    clear_session,
    clear_signin,
    session_token,
    set_session,
    set_signin,
    signin_transaction,
)
from manicule.api.envelopes import BAD_REQUEST, FORBIDDEN, NOT_FOUND, as_response, status_for
from manicule.api.models import KeyBody, UserPatch
from manicule.api.security import AdminPrincipal, AnonymousPrincipal, ViewerPrincipal
from manicule.app.dispatch import run_op
from manicule.core.errors import UnknownEntityError

router = APIRouter(tags=["auth"])

SEE_OTHER = 303
FOUND = 302

SIGNED_OUT_PAGE = "/ui/login"
"""Where a browser that signed out through the page's form is sent."""

START_AGAIN = (
    "this sign-in cannot be completed: it was not started from this browser, it took longer "
    "than ten minutes, or it has already been used. Start again from the sign-in page."
)
"""One refusal for every way the sign-in cookie can fail to match the callback.

Deliberately not three. Whether the cookie was missing, expired or carried a different
``state`` is the difference between a stale tab and a forged callback, and a page that said
which would be telling whoever forged it how close they came.
"""


@router.get(
    "/auth/providers", name="auth_providers", summary="Which identity providers are configured."
)
async def providers(service: Service, caller: AnonymousPrincipal) -> Response:
    """Types and login paths only, never a client secret.

    Unauthenticated on purpose: a client has to be able to find out *how* to authenticate
    before it has authenticated. It discloses which of two well-known provider types an
    operator configured for this workspace, and nothing about the installation's contents.
    """
    del caller
    return as_response(await run_op("auth_providers", service.workspace, service.auth_providers))


@router.get("/auth/session", name="auth_session", summary="Who this request is.")
async def session(service: Service, caller: AnonymousPrincipal) -> Response:
    """The caller's identity, as this installation resolved it.

    Answers for an unauthenticated caller too — ``authenticated: false`` with the configured
    mode — because "your key is not working" and "this server wants no key" are different
    problems and a 401 conflates them. ``via`` says whether a key or a browser session was
    presented, and for a signed-in person the payload names them.

    The workspace on the envelope is the one this process serves; the workspace *in the
    payload* is the one the credential belongs to. On a valid one they are the same, and they
    are both reported because that is the property a caller most wants to be sure of.
    """
    from manicule.app.results import succeeded  # noqa: PLC0415 - only this route renders one

    del service
    return as_response(
        succeeded("auth_session", caller.identity.workspace or "unknown", caller.identity)
    )


# --- signing in -------------------------------------------------------------------------------


@router.get(
    "/auth/login/{provider}",
    name="sign_in",
    summary="Start signing in through an identity provider.",
)
async def login(
    request: Request, service: Service, caller: AnonymousPrincipal, provider: str
) -> Response:
    """Send the browser to the provider, carrying a fresh ``state`` and PKCE verifier with it.

    Anonymous, because it is how a person without a credential gets one. A provider that does
    not apply to this workspace — or sign-in not being the configured mode — is a 404 page:
    there is no such sign-in here, and which of the two it was is not the caller's business.
    """
    from manicule.web.security import sign_in_refused  # noqa: PLC0415 - avoids a cycle

    del caller
    try:
        configured = service.sign_in_provider(provider)
    except UnknownEntityError as exc:
        return sign_in_refused(request, str(exc), status=NOT_FOUND)
    url, transaction = await oauth.begin(configured)
    response = RedirectResponse(url, status_code=FOUND)
    set_signin(response, service.settings, transaction)
    return response


@router.get(
    "/auth/callback/{provider}",
    name="sign_in",
    summary="Complete a browser sign-in the provider has sent back.",
)
async def callback(
    request: Request,
    service: Service,
    caller: AnonymousPrincipal,
    provider: str,
    *,
    code: Annotated[str, Query(max_length=2048)] = "",
    state: Annotated[str, Query(max_length=512)] = "",
    error: Annotated[str, Query(max_length=256)] = "",
) -> Response:
    """Check the browser brought back what it left with, and ask the service who this is.

    **The sign-in cookie is deleted on every outcome**, success or not: a transaction is used
    once, and a second callback with the same one is a replay.

    **On success the answer is a page, not a redirect**, and the reason is the session cookie's
    ``SameSite=Strict``. This response is the end of a chain of redirects that began on the
    provider's site, so the browser treats the chain as cross-site — and it withholds a Strict
    cookie from any further redirect in that chain, including one to this installation's own
    pages. The person would arrive signed out. A page ends the chain; the navigation it starts
    — a ``<meta http-equiv="refresh">``, with an ordinary link beside it for a browser that
    ignores that — begins on this origin, and the cookie goes with it. There is no script on
    the page to do it instead: the browser surface's policy forbids inline script, and this
    page is held to that policy like every other.
    """
    from manicule.web.security import sign_in_refused, signed_in_page  # noqa: PLC0415

    del caller
    settings = service.settings
    transaction = signin_transaction(settings, request.cookies.get(SIGNIN_COOKIE))
    response: Response
    try:
        configured = service.sign_in_provider(provider)
    except UnknownEntityError as exc:
        response = sign_in_refused(request, str(exc), status=NOT_FOUND)
        clear_signin(response, settings)
        return response
    if (
        transaction is None
        or transaction.provider != provider
        or not state
        # Constant time, on bytes. A comparison that stopped at the first differing
        # character would let somebody find a valid state a character at a time.
        or not hmac.compare_digest(state.encode("utf-8"), transaction.state.encode("utf-8"))
    ):
        response = sign_in_refused(request, START_AGAIN, status=BAD_REQUEST)
    elif error:
        # The provider's own error code is not echoed: it is text another site chose, and
        # "access_denied" tells the person nothing they did not just click.
        response = sign_in_refused(
            request,
            f"the {provider} sign-in was not completed: the identity provider did not "
            f"authorize it. Start again from the sign-in page if that was not what you meant.",
            status=FORBIDDEN,
        )
    elif not code:
        response = sign_in_refused(request, START_AGAIN, status=BAD_REQUEST)
    else:
        try:
            profile = await oauth.complete(
                configured,
                transaction,
                code,
                transport=getattr(request.app.state, "oauth_transport", None),
            )
        except oauth.SignInFailedError as exc:
            response = sign_in_refused(request, f"{exc}. Start again.", status=BAD_REQUEST)
        else:
            envelope = await run_op(
                "sign_in", service.workspace, lambda: service.sign_in(provider, profile)
            )
            if not envelope.ok or envelope.data is None:
                message = envelope.error.message if envelope.error else "the sign-in failed"
                response = sign_in_refused(request, message, status=status_for(envelope))
            else:
                response = signed_in_page(request, envelope.data)
                set_session(response, settings, str(envelope.data.get("token", "")))
    clear_signin(response, settings)
    return response


@router.post("/auth/logout", name="sign_out", summary="End this browser's session.")
async def logout(request: Request, service: Service, caller: AnonymousPrincipal) -> Response:
    """End the session on the server, and tell the browser to forget it.

    **Revoked on the server, not only forgotten by the browser.** A copy of the cookie taken
    before this — from a backup of a browser profile, from a machine somebody walked away
    from — stops working at the same moment.

    Anonymous and idempotent: signing out with no session, or twice, succeeds and does nothing.
    A cross-site ``POST`` here is refused before routing like every other unsafe method, so
    another site cannot sign a person out either.

    The browser surface's sign-out button is a plain form, so a request that asks for HTML is
    sent on to the sign-in page with a ``303``; a program gets the envelope.
    """
    del caller
    settings = service.settings
    token = session_token(settings, request.cookies.get(SESSION_COOKIE))
    envelope = await run_op("sign_out", service.workspace, lambda: service.sign_out(token))
    response: Response
    if envelope.ok and "text/html" in request.headers.get("accept", ""):
        response = RedirectResponse(SIGNED_OUT_PAGE, status_code=SEE_OTHER)
    else:
        response = as_response(envelope)
    clear_session(response, settings)
    return response


# --- keys -------------------------------------------------------------------------------------


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


# --- people -----------------------------------------------------------------------------------


@router.get("/api/v1/auth/users", name="user_list", summary="Every member of this workspace.")
async def list_users(service: Service, caller: AdminPrincipal) -> Response:
    """Members, their roles and standing, and how many browsers each is signed in with.

    Admin-only, like the key list: it is the workspace's identities.
    """
    del caller
    return as_response(await run_op("user_list", service.workspace, service.user_list))


@router.patch(
    "/api/v1/auth/users/{user}",
    name="user_update",
    summary="Change a member's role, their standing, or both.",
)
async def update_user(
    service: Service, caller: AdminPrincipal, user: str, body: UserPatch
) -> Response:
    """One change, held to the rules the service states — the last administrator stays one.

    ``user`` is a member's id, or an address that names exactly one member of this workspace.
    Disabling ends every session the person holds here and revokes every key they minted, in
    the same transaction.
    """
    del caller
    return as_response(
        await run_op(
            "user_update",
            service.workspace,
            lambda: service.user_update(user, role=body.role, disabled=body.disabled),
        )
    )


@router.post(
    "/api/v1/auth/users/{user}/sign-out",
    name="user_sign_out",
    summary="End every browser session a member holds in this workspace.",
)
async def sign_out_user(service: Service, caller: AdminPrincipal, user: str) -> Response:
    """For a lost laptop or a shared machine. The person stays a member and may sign in again."""
    del caller
    return as_response(
        await run_op("user_sign_out", service.workspace, lambda: service.user_sign_out(user))
    )


__all__ = ["router"]
