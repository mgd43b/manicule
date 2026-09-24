"""People: who may sign in, what keeps a workspace administrable, and whether sign-in can work.

The protocol — authorization URLs, PKCE, the code exchange, asking a provider who somebody is —
lives in :mod:`manicule.api.oauth`, because it needs an HTTP client and this package may not
import one. What lives here is everything that is a *decision* rather than a conversation, so
that the service can make it and no surface has to:

* **Which providers apply to this workspace.** A provider names the workspace it admits people
  to, or none, meaning whichever one the process serves. A process serving ``beta`` offers
  nothing configured for ``alpha``.
* **Whether a person is admitted.** :func:`admission` is the whole rule, and it is re-checked on
  every sign-in rather than only the first, so an address removed from an allowlist stops that
  person at their next sign-in instead of never.
* **That a workspace keeps an administrator.** :data:`LAST_ADMIN` is what the service and the
  people store both say when a change would leave a workspace that has one with none.
* **Whether a served installation could complete a sign-in at all.** :func:`serving_problems`
  lists what stops it, and both ``build_app`` and ``doctor`` read it — one refuses to build an
  application that would fail every sign-in, the other says so without serving.

**Why serving problems are not** :meth:`~manicule.config.settings.Settings.policy_problems`.
That method is consulted by every command, including the ones that open no socket and ask
nobody to sign in, and its docstring holds it to problems that are problems for *any* command.
A missing ``session_secret`` does not stop ``manicule index``; it stops a browser from being
handed a cookie. So the rule is enforced where a browser could arrive, exactly as the rule about
an unauthenticated wide bind is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from manicule.app.bind import is_loopback
from manicule.config.settings import AuthMode

if TYPE_CHECKING:
    from manicule.config.settings import OAuthProvider, Settings

LOGIN_PATH = "/auth/login/{provider}"
"""Where a browser starts signing in through one provider."""

CALLBACK_PATH = "/auth/callback/{provider}"
"""Where a provider sends the browser back, and what every ``redirect_uri`` must end with."""

MIN_SESSION_SECRET = 32
"""The shortest ``security.auth.session_secret`` a served installation accepts.

A signing key an attacker can enumerate is a cookie an attacker can forge — and the session
cookie is the credential. Thirty-two characters is well past guessing and well within what
``secrets.token_urlsafe(32)`` prints.
"""


@dataclass(frozen=True, slots=True)
class Profile:
    """Who an identity provider says has just signed in.

    The provider's own words, reduced to the five facts admission and the people store need.
    ``subject`` is the provider's stable account id and is the identity; ``email`` is display
    and admission only, and is trusted for admission only when ``email_verified`` says the
    provider checked it.
    """

    provider: str
    subject: str
    email: str = ""
    email_verified: bool = False
    name: str = ""

    @property
    def verified_email(self) -> str:
        """The address, lower-cased, when the provider verified it; empty otherwise.

        An unverified address is somebody's claim about themselves, so it is neither compared
        against an allowlist nor stored as the address a person is known by.
        """
        return self.email.strip().lower() if self.email_verified else ""


class Refusal(StrEnum):
    """Why a sign-in was refused, as the three categories a person may be told.

    Deliberately coarse. ``not_admitted`` covers every allowlist miss — the address, the domain,
    a provider that admits nobody — because telling a stranger *which* check they failed tells
    them what to try next.
    """

    EMAIL_NOT_VERIFIED = "email_not_verified"
    NOT_ADMITTED = "not_admitted"
    DISABLED = "disabled"


REFUSALS: dict[Refusal, str] = {
    Refusal.EMAIL_NOT_VERIFIED: (
        "the identity provider did not report a verified email address for this account, and "
        "this workspace admits people by verified address. Verify the address with the "
        "provider and sign in again."
    ),
    Refusal.NOT_ADMITTED: (
        "this account is not admitted to this workspace. Ask one of its administrators to add you."
    ),
    Refusal.DISABLED: (
        "your membership of this workspace has been disabled. Ask one of its administrators to "
        "enable it again."
    ),
}
"""What a refused person is told, one sentence per category and nothing more specific."""

LAST_ADMIN = (
    "{who} is the last enabled administrator of workspace {workspace!r}, and this change would "
    "leave it with none: nobody could then change a role, disable a member or mint a key from a "
    "browser. Make somebody else an administrator first — `manicule auth set-role <user> admin` "
    "at the command line, or the people page — and then make this change."
)
"""The refusal for a change that would leave a workspace with no enabled administrator.

Said by the service, which checks before it asks the store, and by the store, which checks again
inside the transaction that makes the change — so two administrators demoting each other at the
same moment cannot both succeed. The command line gets it too: the operator at a terminal can
always promote somebody first, so there is nothing an override would let them do that they
cannot do in the right order.
"""


def applicable_providers(settings: Settings) -> tuple[OAuthProvider, ...]:
    """The providers a process serving ``settings.workspace`` offers, in configured order.

    Empty unless ``security.auth.mode`` is ``oauth``: a provider list under another mode is
    configuration nobody is reading, and offering a sign-in from it would be a login route the
    rest of the installation does not honor.
    """
    auth = settings.security.auth
    if auth.mode is not AuthMode.OAUTH:
        return ()
    return tuple(
        provider
        for provider in auth.providers
        if provider.workspace is None or provider.workspace == settings.workspace
    )


def provider_for(settings: Settings, provider_type: str) -> OAuthProvider | None:
    """The one applicable provider of ``provider_type``, or ``None``.

    One, because :func:`serving_problems` refuses two applicable providers of the same type — a
    callback path names a type, and two configurations behind one path would leave the second
    unreachable while looking configured.
    """
    for provider in applicable_providers(settings):
        if provider.type == provider_type:
            return provider
    return None


def admission(provider: OAuthProvider, profile: Profile) -> Refusal | None:
    """Whether ``profile`` may sign in through ``provider``. ``None`` means admitted.

    ``allow_any_user`` admits every account the provider authenticates, verified address or
    not, because that is what it says. Otherwise the address has to be verified and then match
    an entry of ``allowed_emails`` exactly, or have its part after the ``@`` equal an entry of
    ``allowed_domains`` exactly — ``team.example.org`` is not ``example.org``, and a suffix
    match would admit whoever registers ``evil-example.org``.

    Both lists are already lower-cased by the settings model; the address is folded here, so the
    comparison is between two normalized forms rather than one comparison remembering to fold.
    """
    if provider.allow_any_user:
        return None
    address = profile.verified_email
    if not address:
        return Refusal.EMAIL_NOT_VERIFIED
    if address in provider.allowed_emails:
        return None
    local, at, domain = address.rpartition("@")
    if local and at and domain in provider.allowed_domains:
        return None
    return Refusal.NOT_ADMITTED


def serving_problems(settings: Settings) -> list[str]:
    """What stops a served installation from completing a sign-in. Empty when nothing does.

    Only consulted when ``security.auth.mode`` is ``oauth``; every other mode offers no sign-in
    and has nothing here to get wrong. Each problem names the setting and what to write.
    """
    auth = settings.security.auth
    if auth.mode is not AuthMode.OAUTH:
        return []
    problems: list[str] = []
    secret = auth.session_secret.get_secret_value() if auth.session_secret is not None else ""
    if len(secret) < MIN_SESSION_SECRET:
        problems.append(
            f"security.auth.session_secret must be set to at least {MIN_SESSION_SECRET} "
            f"characters when security.auth.mode is 'oauth': it signs the session cookie, and "
            f"a key that can be guessed is a cookie that can be forged. Set "
            f"MANICULE_SECURITY__AUTH__SESSION_SECRET to the output of "
            f"`python -c 'import secrets; print(secrets.token_urlsafe(32))'`."
        )
    applicable = applicable_providers(settings)
    if not applicable:
        problems.append(
            f"security.auth.mode is 'oauth' but no OAuth provider applies to workspace "
            f"{settings.workspace!r}. Add one under security.auth.providers with no `workspace`, "
            f"or with `workspace = {settings.workspace!r}`."
        )
    seen: set[str] = set()
    for provider in applicable:
        label = f"the {provider.type} provider"
        if provider.type in seen:
            problems.append(
                f"two {provider.type} providers apply to workspace {settings.workspace!r}. The "
                f"callback path names only the type, so the second could never be reached; "
                f"give each a distinct `workspace` or remove one."
            )
        seen.add(provider.type)
        problems.extend(
            _redirect_problems(
                provider, label, enforce_https=settings.security.transport.enforce_https
            )
        )
        if not provider.admits_anybody:
            problems.append(
                f"{label} admits nobody: set allowed_emails, allowed_domains, or "
                f"allow_any_user = true. An allowlist left empty is refused rather than read "
                f"as 'everyone', because for GitHub everyone is anybody on the internet."
            )
    return problems


def _redirect_problems(provider: OAuthProvider, label: str, *, enforce_https: bool) -> list[str]:
    """What is wrong with one provider's ``redirect_uri``.

    Configured rather than derived from the request, because deriving it would let a ``Host``
    header choose where a provider sends the authorization code.
    """
    expected = CALLBACK_PATH.format(provider=provider.type)
    uri = (provider.redirect_uri or "").strip()
    if not uri:
        return [
            f"{label} has no redirect_uri. Set it to the callback registered with the provider, "
            f"https://<this host>{expected}."
        ]
    parts = urlsplit(uri)
    host = parts.hostname or ""
    if parts.scheme not in {"http", "https"} or not host:
        return [f"{label}'s redirect_uri {uri!r} is not an absolute http(s) URL."]
    problems: list[str] = []
    if not parts.path.endswith(expected):
        problems.append(
            f"{label}'s redirect_uri {uri!r} does not end with {expected!r}, which is where "
            f"this installation completes a sign-in through that provider."
        )
    if enforce_https and parts.scheme != "https" and not is_loopback(host):
        problems.append(
            f"{label}'s redirect_uri {uri!r} is plain http on a host that is not loopback. The "
            f"authorization code and then the session cookie would cross the network in the "
            f"clear. Use https, or set security.transport.enforce_https = false on a network "
            f"you own."
        )
    return problems


__all__ = [
    "CALLBACK_PATH",
    "LAST_ADMIN",
    "LOGIN_PATH",
    "MIN_SESSION_SECRET",
    "REFUSALS",
    "Profile",
    "Refusal",
    "admission",
    "applicable_providers",
    "provider_for",
    "serving_problems",
]
