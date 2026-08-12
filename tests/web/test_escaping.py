"""Model output and corpus content reach HTML here, and they reach it inert.

This is the security property the browser surface introduces, so it is asserted in the way that
can fail: hostile strings are planted in **four different fields**, each on the path a real
attacker would use, and every page that renders them is checked against what the route actually
returned rather than against a template read off disk.

- a **document title** — whatever was in the file that got indexed;
- a **heading path** — whatever the parser found inside it;
- an **answer body** — a model writing about that document;
- a **citation quote and label** — the document's own words, travelling back out under
  manicule's name.

The last test is the one that makes the rest mean anything. It switches autoescaping off and
asserts the same page then *does* carry raw markup — so the assertions above are known to be
testing the escaping rather than testing that the fixture never had any markup in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment
from markupsafe import escape

from manicule.web import rendering
from tests.web.support import (
    CONVERSATION,
    MARKUP,
    SHARE_TOKEN,
    backend_with_hostile_text,
    client_for,
)

# Each hostile string with the field it was planted in and the pages that render it. Written
# out rather than discovered, so a page that stops rendering a field fails here instead of
# silently reducing what this file covers.
PLANTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "title",
        (
            "/ui/documents",
            "/ui/documents/{document}",
            f"/ui/chat/{CONVERSATION}",
            f"/ui/shared/{SHARE_TOKEN}",
        ),
    ),
    (
        "heading",
        ("/ui/documents/{document}", f"/ui/chat/{CONVERSATION}", f"/ui/shared/{SHARE_TOKEN}"),
    ),
    ("answer", (f"/ui/chat/{CONVERSATION}", f"/ui/shared/{SHARE_TOKEN}")),
    ("quote", ("/ui/documents/{document}", f"/ui/chat/{CONVERSATION}")),
)

SENTINELS = ("window.__title", "window.__heading", "window.__answer", "window.__quote")
"""What each hostile string would set if it ever executed.

Named so an assertion can say which field escaped. They are checked for their *raw* forms
only — the page legitimately contains a ``<script src>`` of its own, so "the page contains no
script tag" would be a test of the wrong thing.
"""


def _pages() -> list[tuple[str, str]]:
    return [(field, path) for field, paths in PLANTED for path in paths]


@pytest.mark.parametrize(("field", "path"), _pages())
def test_hostile_text_renders_inert(field: str, path: str) -> None:
    """The markup is present as text and absent as markup.

    Both halves matter. "The page does not contain ``<script>``" passes for a page that does
    not render the field at all, so the escaped form has to be found as well — that is what
    makes this a test of escaping rather than of omission.
    """
    backend, document = backend_with_hostile_text()
    with client_for(backend) as client:
        body = client.get(path.format(document=document.id)).text
    assert MARKUP[field] not in body, f"the {field} reached the page unescaped"
    # The escaped form, computed with the escaper the templates actually use rather than
    # approximated — an assertion that guessed the entity spelling would pass or fail for a
    # reason that has nothing to do with the property.
    assert str(escape(MARKUP[field])) in body, (
        f"the {field} does not appear on this page at all, escaped or not, so this test is "
        f"asserting nothing about it"
    )
    for sentinel in SENTINELS:
        raw = "<script>" + sentinel
        assert raw not in body, f"{sentinel} reached the page as executable markup"


def test_no_template_marks_anything_safe() -> None:
    """``|safe`` and ``{% autoescape false %}`` do not appear in this package.

    Autoescaping is only a property of the whole surface if nothing opts out of it, and an opt-
    out is one line that nobody reviewing a template diff would necessarily read as a security
    change. Asserted over the directory, so a template added later is covered.
    """
    templates = sorted(Path(rendering.TEMPLATE_DIR).glob("*.html"))
    assert len(templates) >= 10, (
        f"found {len(templates)} templates under {rendering.TEMPLATE_DIR}; the scan is looking "
        f"in the wrong place, and an empty scan reports success."
    )
    offenders = {
        template.name: marker
        for template in templates
        for marker in ("|safe", "| safe", "autoescape false", "autoescape False")
        if marker in template.read_text(encoding="utf-8")
    }
    assert offenders == {}, f"templates that opt out of escaping: {offenders}"


def test_the_environment_escapes_and_refuses_undefined_names() -> None:
    """Asserted on the environment as well, because the default is the whole property."""
    assert rendering.ENVIRONMENT.autoescape is True
    assert rendering.ENVIRONMENT.undefined.__name__ == "StrictUndefined"


def test_switching_the_escaping_off_lets_the_markup_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evidence that the assertions above are load-bearing.

    The same page, rendered by an identical environment with ``autoescape`` off, carries the
    script tag verbatim. If this test ever passes with the escaping *on*, every assertion in
    this file is worthless and this is where that shows up.

    Deliberately a monkeypatched environment rather than a switch on
    :func:`~manicule.web.rendering.build_environment`. An off-switch that ships is an off-switch
    somebody can reach; this one exists only for the length of this test.
    """
    unsafe = Environment(
        loader=rendering.ENVIRONMENT.loader,
        autoescape=False,  # noqa: S701 - an unescaped environment is exactly what this proves
        undefined=rendering.ENVIRONMENT.undefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    monkeypatch.setattr(rendering, "ENVIRONMENT", unsafe)
    backend, document = backend_with_hostile_text()
    with client_for(backend) as client:
        body = client.get(f"/ui/documents/{document.id}").text
    assert MARKUP["title"] in body, (
        "the escaping was switched off and the page *still* did not carry raw markup, so the "
        "tests above are not testing the escaping. Find out what else is stripping it before "
        "trusting them."
    )
    assert "<script>" in body
