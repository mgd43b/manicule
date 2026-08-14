"""Which cross-site requests may change something, and why the question only arises now.

Until there was a browser surface, nothing pointed a browser at this origin, and cross-site
request forgery needs exactly that: a browser that will attach *ambient* authority to a request
some other page caused. manicule's ambient case is real and is the one it ships as —
``security.auth.mode = none`` on loopback, where there is no credential at all and the caller is
whoever can reach the port. A page on the internet cannot read the response (there is no CORS
header to let it), but with a "simple" request it does not need to: ``POST`` from a form, or
``fetch`` with no unusual header, is **sent** and its effect happens.

So an unsafe method that a browser says came from somewhere else is refused unless configuration
named that somewhere.

**Two signals, and the first one cannot be forged.** ``Sec-Fetch-Site`` is a forbidden header
name: page script cannot set it, and every current browser sends it. ``Origin`` is the fallback
for anything older, and is compared against the host the request was addressed to.

**A request with neither header is allowed**, and that is deliberate rather than a gap. `curl`,
a script, an assistant holding a key — none of them send either, and none of them has ambient
authority to abuse: they present a credential explicitly or they get whatever an unauthenticated
caller gets. Refusing them would break every non-browser client to defend against a threat only
browsers create.

Scheme is deliberately **not** compared. Behind a TLS-terminating proxy the request arrives as
``http`` while the browser's ``Origin`` says ``https``, and a check that failed there would be a
policy operators disable rather than one that holds.

**The websocket is checked too, and it is the worse case.** A browser applies no cross-origin
policy to ``WebSocket`` at all: there is no preflight, the connection is made, and the page can
**read** every frame that comes back. So a cross-origin socket to an installation with no
credential is not a write it cannot see the answer to — it is the corpus, answering questions,
to a page the operator merely visited. :func:`handshake_permitted` is the same decision applied
where the handshake is, because middleware never sees a websocket scope.
"""

from __future__ import annotations

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
"""Methods that can change something. ``GET`` and ``HEAD`` are not on this list on purpose.

A ``GET`` that changed state would be the defect, and putting reads behind this check would
break every ordinary link into the browser surface.
"""

FETCH_SITE = "sec-fetch-site"
ORIGIN = "origin"
"""The two headers this decision reads, named once so a rename cannot leave a reader behind."""

SAME_SITE_VALUES = frozenset({"same-origin", "none"})
"""``Sec-Fetch-Site`` values that mean "this did not come from another site".

``none`` is a user-initiated navigation — a typed URL or a bookmark — which is the case a
person reaching a page has. ``same-site`` is deliberately absent: a sibling subdomain is a
different origin, and on a shared domain it is exactly the neighbor this check is for.
"""


def host_of(origin: str) -> str:
    """The host and port of an origin, or empty if it is not one.

    Written out rather than parsed with ``urlsplit``, because what is wanted is the strict
    reading: ``scheme://host[:port]`` and nothing else. A value with a path, a query or no
    scheme is not an origin a browser sent, and returning empty makes it fail the comparison
    rather than match something by accident.
    """
    scheme, separator, remainder = origin.partition("://")
    if not separator or not scheme or not remainder:
        return ""
    if any(character in remainder for character in "/?#"):
        return ""
    return remainder.lower()


def from_this_site(origin: str | None, host: str | None, allowed_origins: tuple[str, ...]) -> bool:
    """Whether an ``Origin`` header is this installation's own, or one an operator listed.

    No ``Origin`` at all means a client that is not a browser, which is admitted — see this
    module's docstring for why that is a decision rather than a gap.

    Args:
        origin: The ``Origin`` header, if the client sent one.
        host: The ``Host`` the request was addressed to.
        allowed_origins: ``security.transport.allowed_origins``.
    """
    if not origin:
        return True
    if origin in allowed_origins:
        return True
    theirs = host_of(origin)
    return theirs != "" and theirs == (host or "").strip().lower()


def permitted(
    method: str,
    *,
    fetch_site: str | None,
    origin: str | None,
    host: str | None,
    allowed_origins: tuple[str, ...],
) -> bool:
    """Whether this request may proceed.

    Args:
        method: The HTTP method. Safe methods are always permitted.
        fetch_site: The ``Sec-Fetch-Site`` header, if the client sent one.
        host: The ``Host`` header the request was addressed to, which is what an ``Origin``
            is compared against.
        origin: The ``Origin`` header, if the client sent one.
        allowed_origins: ``security.transport.allowed_origins`` — the origins an operator has
            already declared may talk to this installation cross-origin. The widget's
            embedding page is the reason that list exists, and a widget asks questions, which
            is a ``POST``.

    Returns:
        True when the request may proceed.
    """
    if method.upper() not in UNSAFE_METHODS:
        return True
    if fetch_site is not None:
        if fetch_site.strip().lower() in SAME_SITE_VALUES:
            return True
        return bool(origin) and origin in allowed_origins
    return from_this_site(origin, host, allowed_origins)


def handshake_permitted(
    *, origin: str | None, host: str | None, allowed_origins: tuple[str, ...]
) -> bool:
    """Whether a websocket handshake may proceed.

    The same decision as :func:`permitted`, over the one header a websocket handshake carries.
    ``Sec-Fetch-Site`` is not sent on one, and there is no preflight and no CORS to fall back
    on — a browser makes the connection and the page reads every frame. So the ``Origin``
    comparison is not a second line of defense here; it is the only one.

    Named separately rather than reached by calling ``permitted("POST", ...)``, because a
    handshake is not a POST and a call site that pretended otherwise would read as a mistake.
    """
    return from_this_site(origin, host, allowed_origins)


REFUSAL = (
    "refusing a cross-site {method} request from {origin!r}. A page on another origin caused "
    "this, and this installation has not listed that origin in "
    "security.transport.allowed_origins. Requests from a program — which send neither an "
    "Origin nor a Sec-Fetch-Site header — are unaffected."
)
"""What the refusal says. It names the method and the origin, because the two together are what
an operator needs to decide whether to add an entry or to go and look at a page."""


HANDSHAKE_REFUSAL = (
    "refusing a websocket handshake from {origin!r}. A browser applies no cross-origin policy "
    "to a websocket, so this connection would have read every answer it asked for. List the "
    "origin in security.transport.allowed_origins if it is yours."
)
"""What a refused handshake is closed with. Short: a close reason is capped at 123 bytes."""


__all__ = [
    "FETCH_SITE",
    "HANDSHAKE_REFUSAL",
    "ORIGIN",
    "REFUSAL",
    "SAME_SITE_VALUES",
    "UNSAFE_METHODS",
    "from_this_site",
    "handshake_permitted",
    "host_of",
    "permitted",
]
