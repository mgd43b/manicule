"""What the browser surface does not do, asserted by name.

Absence is the easiest property to lose by accident and the hardest to notice, and a browser
surface is exactly where it gets lost: a checklist asks for drag-and-drop upload and a settings
screen, and the shortest way to both is a new route. #11 decided that neither operation has one
— an upload is an ingest path with no filesystem permission check and no path an operator chose,
and reading or writing configuration over the network is how an installation gets repointed at a
different data directory — and a route added from a *different package* would undo that
silently, with the assertion that protects it still passing because it only knows about
``/api``.

So this file asserts three things:

* the paths those operations would have are still absent, now checked under ``/ui`` as well
  and checked against the **route table** rather than against a status code;
* this package never calls ``config_get`` or ``config_set``, read from the source tree rather
  than from behavior, because a call added later would otherwise only fail if somebody wrote a
  test for that page;
* every request the served script makes is to a route the HTTP API already publishes, so the
  browser surface adds no operation by way of JavaScript either.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from manicule.web import pages, rendering, security
from tests.routing_support import Reach, classify, walk_routes
from tests.web.support import backend_with_a_document, client_for

SOURCE = Path(rendering.HERE)

ABSENT: tuple[tuple[str, str, Reach, str], ...] = (
    (
        "POST",
        "/ui/documents/upload",
        Reach.SHADOWED,
        "an ingest path with no filesystem permission check",
    ),
    ("POST", "/ui/documents", Reach.SIBLING, "the same, under the listing's own path"),
    (
        "POST",
        "/ui/settings",
        Reach.SIBLING,
        "writing configuration over the network repoints an installation",
    ),
    ("POST", "/ui/config", Reach.UNROUTED, "the same operation under its own name"),
    (
        "POST",
        "/ui/index",
        Reach.UNROUTED,
        "indexing a path a browser named makes the server read any file",
    ),
    (
        "POST",
        "/ui/connectors",
        Reach.SIBLING,
        "declaring a connector points the index somewhere new",
    ),
    ("POST", "/ui/plugins", Reach.SIBLING, "installing a plugin fetches and executes code"),
    (
        "POST",
        "/ui/workspaces",
        Reach.SIBLING,
        "switching a workspace is a command-line operation",
    ),
)
"""Operations a browser UI is usually expected to have, and does not have here.

Each with the reason, and with **why the request reaches no operation**, which is a fact about
the route table rather than about a status code. ``/ui/index`` is worth reading twice: it is the
obvious *replacement* for an upload — "point the server at a path" — and it is worse, because it
turns a browser into a reader of every file the process can open.

Most of these are :attr:`~tests.routing_support.Reach.SIBLING`, and on this surface that is the
*expected* shape rather than a weakness: the page itself is published at that path for ``GET``,
so a ``POST`` appearing there would be this very operation arriving, which is the change to
fail on. The two :attr:`~tests.routing_support.Reach.UNROUTED` entries are the load-bearing
ones, because they are what proves the 404 handler described in this module's docstring is
still a handler and not a catch-all *route*: a catch-all would match, and they would classify
as :attr:`~tests.routing_support.Reach.EXECUTES` instead.

``POST /ui/documents/upload`` is the one to watch. It is refused only because
``/ui/documents/{document_id}`` declares no ``POST`` today — nothing about upload is checked at
all — so it is declared :attr:`~tests.routing_support.Reach.SHADOWED` and will fail here the day
that route gains a write verb.
"""


@pytest.mark.parametrize(("method", "path", "expected", "why"), ABSENT)
def test_the_browser_surface_adds_no_write_route(
    method: str, path: str, expected: Reach, why: str
) -> None:
    """Asserted over the route table, not by sending the request and reading the status.

    404 and 405 are what an absent operation returns and also what several present ones return,
    so a probe cannot distinguish "there is no such operation" from "a handler ran and did not
    find what you named". The route table can, and it is the thing the property is about.
    """
    reached = classify(method, path, walk_routes())
    assert reached.reach is not Reach.EXECUTES, (
        f"{method} {path} runs {reached.route}. It is deliberately absent because {why}, but "
        f"this request reaches a handler rather than nothing."
    )
    assert reached.reach is expected, (
        f"{method} {path} is {reached.reach.value} ({reached.route}), not {expected.value}. The "
        f"operation is deliberately absent because {why}; what changed is *why*."
    )


def test_every_page_route_is_a_read() -> None:
    """The browser surface is GET-only, and that is asserted over the router rather than read.

    Every mutation the pages offer goes to a route the HTTP API already publishes, from the
    browser, with a JSON content type. That is not a style choice: it means the browser surface
    introduces no new write path at all, so #11's boundary is the whole boundary.
    """
    methods = {method for route in pages.router.routes for method in getattr(route, "methods", ())}
    assert methods <= {"GET", "HEAD"}, f"the browser surface declares write methods: {methods}"
    assert len(pages.router.routes) >= 15, "the router is empty; this assertion proves nothing"


def _modules() -> list[Path]:
    found = sorted(SOURCE.rglob("*.py"))
    assert len(found) >= 5, (
        f"the scan read {len(found)} module(s) under {SOURCE}; it is walking the wrong tree, "
        f"and an empty walk reports success."
    )
    assert {path.name for path in found} >= {"pages.py", "rendering.py", "security.py"}, (
        "the scan did not read this package's own modules, so whatever it walked was not it"
    )
    return found


def test_no_module_here_reads_or_writes_configuration() -> None:
    """``config_get`` and ``config_set`` are not called from this package.

    Read from the source rather than inferred from behavior: a settings page that started
    calling ``config_get`` would render perfectly, and the only thing that would notice is a
    check like this one. The settings area shows the installation's posture from ``doctor`` and
    the index report instead — live facts about the running system, not the contents of a file.
    """
    offenders: dict[str, list[str]] = {}
    for module in _modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        named = sorted(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in {"config_get", "config_set"}
        )
        if named:
            offenders[module.name] = named
    assert offenders == {}, (
        f"these modules reach configuration: {offenders}. Reading and writing configuration "
        f"over the network is how an installation gets repointed at a different data "
        f"directory; it stays on the command line and the MCP tool."
    )


def test_the_served_script_builds_dom_and_never_markup() -> None:
    """Asserted against **what the route returned**, not against the source file.

    An answer is model output over indexed documents. Treating it as markup would make any
    document in the corpus a script-injection vector into this page, so the script writes text
    through ``textContent`` and nothing else.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/ui/static/manicule.js").text
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in served, f"the script uses {forbidden}"
    assert "textContent" in served, "the script renders nothing, so this test proves nothing"


def test_the_served_script_calls_only_routes_the_api_already_publishes() -> None:
    """No operation arrives by way of JavaScript.

    Every path the script fetches is matched against the routes the application actually
    mounted. A script that called ``/api/v1/documents/upload`` would fail here even though no
    Python route exists — which is the failure mode of "the UI added an endpoint" seen from the
    other end.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/ui/static/manicule.js").text
        paths = client.get("/api/openapi.json").json()["paths"]
    called = set(re.findall(r'"(/api/v1/[a-z/-]*)"', served))
    assert called, "no API call was found in the script, so this test proves nothing"
    # A call with a path parameter appears in the script as the literal up to that parameter —
    # `"/api/v1/documents/" + id + "/reindex"` — so a published path is compared as the part
    # before its first placeholder as well as whole.
    known = set(paths) | {path.split("{")[0] for path in paths}
    unknown = sorted(path for path in called if path not in known)
    assert unknown == [], f"the script calls routes the API does not publish: {unknown}"


def test_the_script_stores_nothing_but_a_theme() -> None:
    """A theme is not a credential, and nothing else here is written to the browser's storage.

    The widget stores nothing at all because what it holds *is* a credential. This page holds
    none — a key never reaches it, because a page load cannot carry one — so the one thing it
    persists is a color scheme, and this is what keeps that true.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        served = client.get("/ui/static/manicule.js").text
    assert "document.cookie" not in served
    assert "sessionStorage" not in served
    stored = set(re.findall(r"localStorage\.\w+\((\w+)", served))
    assert stored == {"THEME_KEY"}, f"the script stores something other than a theme: {stored}"


def test_the_refusal_page_is_this_surfaces_and_the_decision_is_not() -> None:
    """Authorization is decided once, in the API's own module, and rendered twice.

    A second implementation of "does this principal clear this floor" for the browser is the
    shape of bug where a rule holds on one surface and not on the one somebody is using.
    """
    source = Path(security.__file__).read_text(encoding="utf-8")
    assert "from manicule.api.security import" in source
    assert "require(principal, floor)" in source
    assert "_RANK" not in source, "the browser surface has its own idea of what a role outranks"
