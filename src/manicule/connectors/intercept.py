"""Telling an answer from a sign-in page.

An instance behind an identity provider does not answer an expired session with a 401. It
answers with a redirect to the provider, and what comes back at the end of that redirect is a
**sign-in page with status 200**: a successful response, of a plausible content type, carrying
several kilobytes of real text. Every ordinary check a client makes passes.

That is the failure this module exists to make impossible. A sync that mistook it for content
would index one copy of the sign-in page per page it tried to fetch — plausible documents,
retrievable, citable, and wrong, with nothing downstream able to tell them from the corpus. It
would also look, from every metric, like a sync that worked.

Two layers catch it, and they are independent on purpose:

- **The redirect itself.** The client does not follow a redirect blindly
  (:mod:`manicule.connectors.client`), so an instance that redirects to its identity provider
  is caught before the provider is ever asked — no request is made off-origin, and no sign-in
  page is fetched at all, let alone parsed. :func:`offsite` and :func:`signin_path` decide that.
- **The response.** Some deployments serve the sign-in page inline with 200 and no redirect —
  a filter in front of the application rather than a redirector. :func:`signed_out` reads what
  came back and refuses it.

Nothing here is specific to browser sessions. A reverse proxy in front of a Server instance
answers a request carrying a personal access token exactly the same way, and a check that ran
only for one credential would be a check that ran for the wrong one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import unquote

from manicule.connectors.pagination import origin_of

__all__ = ["Answer", "offsite", "signed_out", "signin_path"]

_ANONYMOUS = "anonymous"

SEARCHED_BYTES = 64 * 1024
"""How much of a body is read looking for sign-in markers.

A sign-in page announces itself in its form, which is in the first few kilobytes of every one
of them. Reading further would only mean scanning an entire attachment for the words in a login
form, which is how a legitimate document gets refused.
"""

SIGNIN_PATHS: tuple[str, ...] = (
    "/login.action",
    "/dologin.action",
    "/authenticate.action",
    "/plugins/servlet/samlsso",
    "/plugins/servlet/samlconsumer",
    "/plugins/servlet/oidc",
    "/oauth2/authorize",
    "/idp/",
    "/adfs/ls",
    "/saml2/",
)
"""Paths that mean "sign in here" on a Confluence Server or Data Center instance, or on the
identity provider in front of one.

Matched as a prefix on the path, case-insensitively. The list is not a guess about one vendor:
the first four are Seraph's own, which every Server instance has whatever sits in front of it,
and the rest are the standard servlet paths the SAML and OIDC plugins mount. An instance whose
provider uses a path not listed here is still caught — by its origin, by ``X-AUSERNAME``, or by
the markers in what it serves — because no one of these checks is load-bearing alone.
"""

SIGNIN_MARKERS: tuple[bytes, ...] = (
    b'name="os_username"',
    b'name="os_password"',
    b'name="j_username"',
    b'id="login-form"',
    b'id="loginform"',
    b'name="samlrequest"',
    b'action="/dologin.action"',
    b"<title>log in",
)
"""Byte sequences that appear in a sign-in page and not in a document.

Deliberately narrow. A broad marker — the word "login", a ``<form>`` element — would refuse an
attached HTML file that happens to document how to sign in to something, and refusing a real
document is a smaller failure than indexing a sign-in page but it is still a failure. These are
the name attributes of Seraph's own login form and of a SAML request, which a document does not
contain by accident. Compared against a lower-cased body, so attribute casing does not matter.
"""


@dataclass(frozen=True, slots=True)
class Answer:
    """What came back, in the terms this check reads.

    A plain record rather than an ``httpx.Response`` so that the rules can be exercised without
    an HTTP stack, and so that the streaming path — which has headers long before it has a
    body — can ask the same question twice with what it has.
    """

    url: str
    """The URL that produced it, after any redirect the client chose to follow."""

    status: int
    headers: Mapping[str, str]
    body: bytes = b""
    """As much of the body as has arrived. Empty means "headers only, so far"."""


def offsite(url: str, *, origin: str) -> str | None:
    """Why ``url`` is somewhere else, or ``None`` if it is on ``origin``.

    Unconditional, and it is the first thing asked of a redirect. The client's credential is
    attached to whatever it is asked to fetch (``docs/connectors/confluence.md`` §2), so a
    redirect to another host is the same hazard as a link to one — and when the credential is a
    browser session, what would travel is the sync account's entire identity at that company
    rather than a scoped token. It is reported as an untrusted link rather than as a dead
    session, because a redirect off the configured origin is also what a wrong ``base_url``
    looks like, and naming the wrong cause sends somebody to re-authenticate a working session.
    """
    found = origin_of(url)
    if found == origin:
        return None
    return (
        f"a response redirected the sync to {found or url!r}, and this connector is configured "
        f"for {origin}. manicule does not follow, because the credential travels with the "
        f"request: an instance behind an identity provider answers an expired session by "
        f"sending the client to that provider, and what comes back from there is a sign-in "
        f"page rather than content"
    )


def signin_path(url: str) -> str | None:
    """Why ``url`` is a sign-in page on this instance, or ``None`` if it is not.

    Asked of a same-origin redirect. Confluence's own authentication filter redirects an
    unauthenticated request to ``/login.action``, and the SAML and OIDC plugins mount their own
    servlets; a request that ends at one of them was answered by a sign-in rather than by
    Confluence, whatever status the sign-in page itself carries.
    """
    path = _path_of(url).lower()
    matched = next((candidate for candidate in SIGNIN_PATHS if path.startswith(candidate)), None)
    if matched is None:
        return None
    return (
        f"a response redirected the sync to {path}, which is a sign-in page rather than an "
        f"answer. The session is no longer signed in"
    )


def signed_out(answer: Answer, *, expected_account: str = "") -> str | None:
    """Why ``answer`` is a sign-in rather than a reply, or ``None`` if it is a reply.

    Three independent signals, strongest first. Any one is enough, which is the point: an
    instance that omits the headers is caught by its body, and one that serves a body this does
    not recognize is caught by its headers.

    Args:
        answer: What came back. ``body`` may be empty when only headers have arrived.
        expected_account: Who the credential believes it is. When the instance names somebody
            else, the session is not this account's any more — which is a correctness problem
            and not only an authentication one, because the index holds whatever the sync
            account can see.
    """
    reason = _seraph_reason(answer.headers)
    if reason is not None:
        return reason
    reason = _named_user(answer.headers, expected_account)
    if reason is not None:
        return reason
    return _signin_body(answer)


def _seraph_reason(headers: Mapping[str, str]) -> str | None:
    """Seraph — the authentication filter every Confluence Server runs — says so in a header."""
    stated = headers.get("x-seraph-loginreason", "").strip()
    if not stated or stated.upper() == "OK":
        return None
    return (
        f"the instance answered with X-Seraph-LoginReason: {stated}, which is Confluence's own "
        f"authentication filter reporting that this request was not signed in"
    )


def _named_user(headers: Mapping[str, str], expected_account: str) -> str | None:
    """Confluence names the authenticated user on every REST response, whatever the status."""
    named = unquote(headers.get("x-ausername", "").strip())
    if not named:
        return None
    if named.lower() == _ANONYMOUS:
        return (
            "the instance answered as anonymous (X-AUSERNAME), so the request reached "
            "Confluence without being signed in. Anything it returned is what a signed-out "
            "reader can see, which is not what this index is meant to hold"
        )
    if expected_account and named != expected_account:
        return (
            f"the instance answered as {named!r} (X-AUSERNAME) where this credential was "
            f"captured as {expected_account!r}. The index holds whatever the sync account can "
            f"see, so syncing as a different account would quietly change what is in it"
        )
    return None


def _signin_body(answer: Answer) -> str | None:
    """The last resort: what came back looks like the sign-in form itself."""
    if not answer.body:
        return None
    media_type = answer.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type not in {"text/html", "application/xhtml+xml"}:
        return None
    haystack = answer.body[:SEARCHED_BYTES].lower()
    found = next((marker for marker in SIGNIN_MARKERS if marker in haystack), None)
    if found is None:
        return None
    return (
        f"{answer.url} answered {answer.status} with a sign-in page rather than an answer "
        f"(it contains {found.decode()!r}). A sign-in page is never content: indexing one "
        f"would put a plausible, retrievable, citable document in place of every page this "
        f"sync tried to read"
    )


def _path_of(url: str) -> str:
    """The path of ``url``, without a scheme or query, and never empty."""
    without_scheme = url.split("://", 1)[-1]
    slash = without_scheme.find("/")
    if slash == -1:
        return "/"
    return without_scheme[slash:].split("?", 1)[0].split("#", 1)[0] or "/"
