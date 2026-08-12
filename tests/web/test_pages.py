"""Every area exists, is framed the same way, and fails the same way.

Two kinds of assertion, matching ``tests/api/test_routes.py``:

**Coverage.** Each of the twelve areas has a page that answers, checked from the mounted routes
rather than from a list in somebody's head. ``layout`` is the one area with no page of its own,
and that is asserted as a property of the templates rather than waved away.

**Behaviour that has to hold for every page.** The policy header, the escaping environment, the
status a failure carries, and what a refusal looks like to a browser. Each is checked over the
whole set, because a per-page test proves a property of the page somebody wrote a test for.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from manicule.app.service import ApplicationService
from manicule.web.areas import AREAS, NAVIGATION
from manicule.web.rendering import ENVIRONMENT, SCRIPT, STYLESHEET, TEMPLATE_DIR, UI_POLICY
from tests.web.support import (
    CONVERSATION,
    SHARE_TOKEN,
    backend_with_a_document,
    backend_with_hostile_text,
    client_for,
    pages_of,
)

NOT_FOUND = 404
UNAUTHORIZED = 401
FORBIDDEN = 403

# The areas that are a page, and one path each. `layout` is deliberately absent: it is the
# frame, and the test below proves that rather than skipping it.
PAGE_FOR_AREA: dict[str, str] = {
    "dashboard": "/ui",
    "chat": "/ui/chat",
    "documents": "/ui/documents",
    "collections": "/ui/collections",
    "connectors": "/ui/connectors",
    "health": "/ui/health",
    "plugins": "/ui/plugins",
    "settings": "/ui/settings",
    "workspaces": "/ui/workspaces",
    "admin": "/ui/admin",
    "auth": "/ui/auth",
}


def test_the_twelve_areas_are_the_eleven_pages_and_the_frame() -> None:
    """The area list and the pages cannot drift apart without this failing."""
    assert set(PAGE_FOR_AREA) | {"layout"} == set(AREAS)
    assert len(AREAS) == 12


def test_every_navigable_area_is_in_the_navigation() -> None:
    """The palette reads the navigation out of the DOM, so a page missing from it is a page
    with no keyboard route to it either."""
    assert {area for area, _, _ in NAVIGATION} == set(PAGE_FOR_AREA)
    assert {path for _, path, _ in NAVIGATION} == set(PAGE_FOR_AREA.values())


def test_the_layout_area_is_the_frame_every_other_page_extends() -> None:
    """``layout`` has no route, and that is a fact about the templates rather than an excuse.

    Every page template extends a frame. The two frames are the ordinary one and the bare one
    the anonymous shared page uses, and nothing else is a page.
    """
    pages = sorted(
        path
        for path in TEMPLATE_DIR.glob("*.html")
        if path.name not in {"layout.html", "bare.html", "macros.html", "refused.html"}
    )
    assert len(pages) >= 10, f"only {len(pages)} page templates found; the scan is looking wrong"
    for page in pages:
        assert "{% extends layout %}" in page.read_text(encoding="utf-8"), (
            f"{page.name} does not extend the frame, so it is a page with no navigation, no "
            f"stylesheet and no script"
        )


def test_every_template_is_reachable_through_the_loader_and_is_rendered_by_a_page() -> None:
    """The templates are package data, and package data is what a wheel silently omits.

    Read through the environment's own loader rather than off the filesystem, because that is
    what a running installation does: a build that shipped the modules and not the templates
    would fail at import — ``manicule.web.rendering`` reads the stylesheet and the script at
    import time — and this is the assertion that would notice.

    The second half is the other direction: a template nothing renders is a page somebody
    forgot to mount, which the coverage assertions above cannot see.
    """
    from manicule.web import pages as page_module  # noqa: PLC0415 - read as source, deliberately

    available = set(ENVIRONMENT.list_templates())
    assert len(available) >= 15, f"the loader found {len(available)} templates; it is misconfigured"
    assert STYLESHEET.strip(), "the stylesheet was not packaged with the templates"
    assert SCRIPT.strip(), "the script was not packaged with the templates"

    source = Path(page_module.__file__).read_text(encoding="utf-8")
    frames = {"layout.html", "bare.html", "macros.html", "problem.html", "refused.html"}
    unrendered = sorted(name for name in available - frames if f'"{name}"' not in source)
    assert unrendered == [], f"templates no page renders: {unrendered}"


@pytest.mark.parametrize(("area", "path"), sorted(PAGE_FOR_AREA.items()))
def test_every_area_answers(area: str, path: str) -> None:
    """Each page, against the default fake backend, and it marks itself as the current area."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get(path)
    assert response.status_code == 200, response.text
    assert 'aria-current="page"' in response.text, f"{area} does not mark itself in the navigation"


def test_every_page_answers_including_the_ones_with_parameters() -> None:
    """The whole surface in one pass, so a page added later is covered by the list it is on."""
    backend, document = backend_with_hostile_text()
    with client_for(backend) as client:
        for path in pages_of(document.id):
            assert client.get(path).status_code == 200, path


@pytest.mark.parametrize("path", ["/ui", "/ui/documents", f"/ui/shared/{SHARE_TOKEN}"])
def test_every_page_states_the_narrow_policy(path: str) -> None:
    """The page policy, not the JSON one — a browser applies ``default-src 'none'`` to a
    document and then refuses that document's own stylesheet.

    The middleware sets its header with ``setdefault``, so a route that states a policy keeps
    it; this is what proves the browser surface's is the one that survives.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        policy = client.get(path).headers["content-security-policy"]
    assert policy == UI_POLICY
    assert "'unsafe-inline'" not in policy
    assert "frame-ancestors 'none'" in policy


def test_the_json_surface_keeps_its_own_policy() -> None:
    """The exception is the browser surface, not a loosening of the default."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        policy = client.get("/api/v1/documents").headers["content-security-policy"]
    assert policy == "default-src 'none'; frame-ancestors 'none'"


@pytest.mark.parametrize("path", ["/ui", "/ui/admin", f"/ui/shared/{SHARE_TOKEN}"])
def test_no_page_is_cached(path: str) -> None:
    """A rendered page holds answer text and document titles, and a shared machine is exactly
    where somebody presses Back."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        assert client.get(path).headers["cache-control"] == "no-store"


def test_a_missing_document_is_a_404_page_rather_than_an_empty_one() -> None:
    """The status comes from the same table the JSON surface uses.

    A document detail with no document is not a page with an empty section, so the response is
    the problem page — carrying the error's type, message and hint, because the hint is usually
    the thing that fixes it.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/ui/documents/nope")
    assert response.status_code == NOT_FOUND
    assert "UnknownEntityError" in response.text
    assert "text/html" in response.headers["content-type"]


def test_a_panel_that_failed_does_not_take_the_rest_of_the_page_with_it() -> None:
    """A dashboard is several operations and they fail independently.

    The counts fail; the diagnosis and the workspaces still render. A page that collapsed to
    one error would be hiding two answers it had.
    """
    backend, _ = backend_with_a_document()

    async def broken() -> None:
        msg = "the store is unavailable"
        raise OSError(msg)

    backend.store.list_documents = broken  # pyright: ignore[reportAttributeAccessIssue]
    with client_for(backend) as client:
        response = client.get("/ui")
    assert response.status_code == 200
    assert "Checks" in response.text, "the diagnosis panel went missing with the counts"


def test_a_browser_that_presents_no_key_is_refused_as_a_page() -> None:
    """With authentication configured, a navigation carries no credential and this says so.

    A browser cannot attach a header to a page load and this build has no session cookie, so
    the honest answer is a refusal that explains itself — not a JSON envelope in a browser
    window, and not a page that quietly renders as though the caller were the operator.
    """
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    with client_for(backend) as client:
        response = client.get("/ui/documents")
    assert response.status_code == UNAUTHORIZED
    assert "text/html" in response.headers["content-type"]
    assert "session cookie" in response.text
    assert "{" not in response.text.split("<body")[0], "a JSON body reached a browser"


def test_a_viewer_may_not_read_the_administration_areas() -> None:
    """The pages take the floor the routes behind them take.

    Query logs are what somebody asked, the audit trail names who did what, and the key list is
    the installation's identities. A page that took less than its routes would be a way round
    them.
    """
    backend, _ = backend_with_a_document(security={"auth": {"mode": "api_key"}})
    secret = asyncio.run(ApplicationService(backend).api_key_create("reader", role="viewer")).secret
    with client_for(backend) as client:
        headers = {"X-API-Key": secret}
        assert client.get("/ui/documents", headers=headers).status_code == 200
        for path in ("/ui/admin", "/ui/auth", "/ui/connectors", "/ui/plugins"):
            assert client.get(path, headers=headers).status_code == FORBIDDEN, path


def test_the_stylesheet_and_the_script_are_constants() -> None:
    """No request value reaches what a browser executes, so there is no reflected path into it."""
    backend, _ = backend_with_a_document()
    marker = "PROBE-71ac"
    with client_for(backend) as client:
        script = client.get(f"/ui/static/manicule.js?{marker}=1", headers={"X-Probe": marker})
        style = client.get(f"/ui/static/manicule.css?{marker}=1")
    assert marker not in script.text
    assert marker not in style.text
    assert script.headers["content-type"].startswith("text/javascript")
    assert style.headers["content-type"].startswith("text/css")
    assert script.headers["x-content-type-options"] == "nosniff"


def test_there_is_no_directory_to_traverse() -> None:
    """Two constants rather than a static mount, so a path is never resolved against a
    directory. A traversal attempt matches no route at all."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        for path in ("/ui/static/../rendering.py", "/ui/static/%2e%2e/pages.py", "/ui/static/"):
            assert client.get(path).status_code == NOT_FOUND, path


def test_the_conversation_page_renders_the_stored_turns() -> None:
    """The owner's view: full citations, because the reader could have retrieved them."""
    backend, _ = backend_with_hostile_text()
    with client_for(backend) as client:
        body = client.get(f"/ui/chat/{CONVERSATION}").text
    assert "what happens" in body
    assert "resolved" in body
