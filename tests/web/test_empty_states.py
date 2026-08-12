"""The states a reader actually lands in: no query, no hits, and no such document.

These are the least-trodden paths on the surface and the ones a new reader hits first, and each
of them was reached by a single ordinary action rather than by anything unusual.

The search box in the frame carries no ``required`` attribute and submits on Enter, so
``GET /ui/search?q=`` is one keystroke from every page here — and it answered with the raw JSON
envelope of a validation error, in a browser window, carrying a traceback frame that named this
repository's path on the server. ``/ui/search`` with no query at all is what a bookmark
produces, and it did the same thing. Neither is an error; both are the state the page is in
before anybody has typed.
"""

from __future__ import annotations

import re

import pytest

from manicule.core.retrieval import Candidate
from tests.app.fakes import make_chunk
from tests.web.support import backend_with_a_document, client_for

OK = 200
NOT_FOUND = 404


@pytest.mark.parametrize(
    "path",
    ["/ui/search", "/ui/search?q=", "/ui/search?q=%20%20"],
    ids=["no query at all", "the empty submission", "whitespace only"],
)
def test_a_blank_search_is_a_page_rather_than_a_validation_error(path: str) -> None:
    """Blank renders the search page. It used to answer 422 with a JSON body."""
    backend, _ = backend_with_a_document()
    response = client_for(backend).get(path)
    assert response.status_code == OK, (
        f"{path} answered {response.status_code}. A reader who pressed Enter in an empty "
        "search box is not making a malformed request."
    )
    assert "text/html" in response.headers["content-type"], (
        "a browser navigation was answered with something that is not a page"
    )


@pytest.mark.parametrize("path", ["/ui/search", "/ui/search?q="])
def test_a_blank_search_does_not_disclose_the_server_source_tree(path: str) -> None:
    """The validation envelope carried a source path and a line number into the browser.

    ``File "/…/src/manicule/web/pages.py", line 259, in search`` was in the body of a response
    that any visitor could produce with one keystroke.
    """
    backend, _ = backend_with_a_document()
    body = client_for(backend).get(path).text
    for leak in ("src/manicule", "pages.py", "RequestValidationError"):
        assert leak not in body, (
            f"a blank search disclosed {leak!r} to the browser. The empty state is a page, "
            "not a stack frame."
        )


def test_a_blank_search_says_what_to_do_and_runs_nothing() -> None:
    """An empty state that does not say what it wants is a blank screen with a form on it."""
    backend, _ = backend_with_a_document()
    body = client_for(backend).get("/ui/search").text
    assert "Type a query" in body, "the empty search page offers no instruction"
    assert "passages ·" not in body, "a blank query ran a search it had no query for"


def test_scores_are_not_rendered_at_full_float_precision() -> None:
    """A relevance score is shown to four places, as ``manicule search`` shows it.

    The raw value reached the page as ``0.032266458495966696`` — sixteen digits of apparent
    precision on an uncalibrated number, and a different rendering of the same value from the
    one the command line gives for the same query.

    **The candidate is seeded here on purpose.** ``backend_with_a_document`` leaves the fake
    retriever's candidate list empty, so ``/ui/search?q=…`` renders the *no hits* branch and
    every existing test of this page walks past the hit markup without entering it. A test that
    asserted on the formatting without seeding one would pass against either rendering.
    """
    backend, document = backend_with_a_document()
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(document), score=0.032266458495966696)
    ]
    body = client_for(backend).get("/ui/search?q=retry").text
    scores = re.findall(r'<span class="score">([^<]*)</span>', body)
    assert scores, "the hit branch did not render, so this asserts nothing"
    assert scores == ["0.0323"], (
        f"the score rendered as {scores}. `manicule search` prints the same number as 0.0323; "
        "the raw double claims precision the score does not have."
    )


def test_a_missing_document_offers_a_way_back() -> None:
    """The refusal page is where a reader is most likely to be stuck.

    It rendered the bare operation name — a reader who asked for a document that is not there
    was shown the word "workbench" — and the only navigation was the frame's list of eleven
    areas.
    """
    backend, _ = backend_with_a_document()
    response = client_for(backend).get("/ui/documents/not-a-real-id")
    assert response.status_code == NOT_FOUND
    assert "Back to Documents" in response.text, (
        "the refusal page offers no route back to the area the reader came from"
    )
    assert "while running" in response.text, (
        "the operation name is presented as a heading rather than as the operation"
    )
