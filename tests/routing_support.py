"""Walking the mounted route table, and asking what a request would actually reach.

Two suites assert that an operation is **deliberately absent**, and both used to do it by
sending the request and requiring 404 or 405 back. That is not the same question. A status code
answers "what came back"; absence is a statement about the route table, and the two agree only
by luck. Three ways they come apart, all of them live in this project:

* a literal path segment is swallowed by a **placeholder** — ``/documents/upload`` matches
  ``/documents/{document_id}`` — so the 405 is "wrong verb on a path that exists", and the day
  somebody adds that verb the request starts *executing* with ``document_id='upload'``;
* the path is genuinely published for another operation under another verb, which is a
  different and much more stable reason for the same 405;
* the route matches outright and **runs**, returning 404 from inside the handler because the
  entity it looked up does not exist. Create the entity and the same request returns 200.

So the classification is done with Starlette's own matcher, against the walked route table, and
each declared absence says which of those it is. :func:`classify` is the whole of it;
:class:`Reach` is the vocabulary.

**The walk proves it ran before it reports what it found.** ``app.routes`` is not a flat list on
every FastAPI: from 0.13x an included router appears as a wrapper object, and a walk that only
recognised :class:`~fastapi.routing.APIRoute` therefore found **nothing at all** — and reported
success, because every assertion built on it is a statement about every route and there were
none. :func:`walk_routes` descends into whatever shape the framework used, and
:data:`MINIMUM_ROUTES` is a floor below which the walk is assumed to have collapsed rather than
the surface to have shrunk. That has caught a real regression once already; it is not
hypothetical.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, NamedTuple, cast

from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.routing import Match
from starlette.types import Scope

from manicule.api.app import build_app
from manicule.app.service import ApplicationService
from tests.api.support import backend_with_a_document

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

MINIMUM_ROUTES = 40
"""A floor on how many routes the walk must find.

Far below the real count, and present to catch a walk that collapsed rather than to track the
size of the surface. It has caught one: on FastAPI 0.141 an included router is a wrapper object
rather than its routes, and the previous walk found **zero** — with every assertion built on it
passing, because each is a statement about every route and there were none.
"""


def _descend(routes: Iterable[object]) -> Iterator[APIRoute | APIWebSocketRoute]:
    """Every route, whichever shape this FastAPI put them in.

    A router included with ``include_router`` may appear on ``app.routes`` as its routes or as
    one object standing for them, depending on the version. Both are followed, so the suites
    keep enumerating the surface across an upgrade instead of quietly enumerating none of it.

    Only leaves are yielded. A wrapper object matches every path beneath its prefix, so
    including one would make :func:`classify` report that *something* matches ``/ui/anything``
    and the classification would mean nothing.
    """
    for route in routes:
        if isinstance(route, (APIRoute, APIWebSocketRoute)):
            yield route
            continue
        inner = getattr(route, "routes", None)
        if inner is None:
            inner = getattr(getattr(route, "original_router", None), "routes", None)
        if inner:
            yield from _descend(cast("Iterable[object]", inner))


def walk_routes() -> list[APIRoute | APIWebSocketRoute]:
    """Every leaf route the production application mounts, or a failure saying the walk broke."""
    backend, _ = backend_with_a_document()
    found = list(_descend(build_app(ApplicationService(backend)).routes))
    assert len(found) >= MINIMUM_ROUTES, (
        f"the walk found {len(found)} route(s), below the floor of {MINIMUM_ROUTES}. Assertions "
        f"built on this walk are statements about *every* route, so a walk that found none "
        f"passes them all. Fix the walk rather than the floor."
    )
    return found


class Reach(enum.Enum):
    """What a request for a given method and path actually reaches in the route table."""

    UNROUTED = "unrouted"
    """No route matches the path at all. The operation is absent structurally."""

    SIBLING = "sibling"
    """The same **literal** path is published for other methods, and not for this one.

    A stable 405: the path exists because a different operation lives there, and adding this
    method to it would be adding *this* operation — which is the change the assertion wants to
    fail on.
    """

    SHADOWED = "shadowed"
    """Only a **placeholder** route matches, and it does not declare this method.

    A 405 by coincidence. The literal segment was swallowed by a path parameter, so the absence
    holds only for as long as nobody adds that verb to the parameterised route. This is a latent
    defect wherever it appears, and declaring it is how it stops being a silent one.
    """

    EXECUTES = "executes"
    """A route matches and its handler **runs**.

    Never a demonstration of absence. If such a request 404s it is the handler saying the entity
    was not found, which a request that names an entity that does exist will not do.
    """


class Reached(NamedTuple):
    """The classification, and the route responsible for it."""

    reach: Reach
    route: str
    """``name path`` of the matched route, or ``-`` when nothing matched."""


def classify(method: str, path: str, routes: Iterable[APIRoute | APIWebSocketRoute]) -> Reached:
    """What ``method path`` reaches, asked of the route table rather than of a response.

    Starlette's matcher is used rather than a reimplementation of it, because a second opinion
    about what a path matches is exactly the thing that would be wrong.
    """
    headers: list[tuple[bytes, bytes]] = []
    scope: Scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "headers": headers,
        "query_string": b"",
    }
    siblings: list[str] = []
    shadows: list[str] = []
    for route in routes:
        match, _ = route.matches(scope)
        found = f"{route.name} {route.path}"
        if match is Match.FULL:
            return Reached(Reach.EXECUTES, found)
        if match is Match.PARTIAL:
            # The literal path being published for another verb is a different fact from a
            # placeholder having swallowed a segment, and only the first is stable.
            (siblings if route.path == path else shadows).append(found)
    # A shadow is reported ahead of a sibling when both exist. Starlette would answer 405 from
    # whichever it saw first, but the question here is whether the absence is *stable*, and one
    # parameterised route that could gain this verb is enough to make it not.
    if shadows:
        return Reached(Reach.SHADOWED, shadows[0])
    if siblings:
        return Reached(Reach.SIBLING, siblings[0])
    return Reached(Reach.UNROUTED, "-")
