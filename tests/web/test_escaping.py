"""Model output and corpus content reach HTML here, and they reach it inert.

This is the security property the browser surface introduces, so it is asserted in the way that
can fail: hostile strings are planted in **six different fields**, each on the path a real
attacker would use, and every page that renders them is checked against what the route actually
returned rather than against a template read off disk.

- a **document title** — whatever was in the file that got indexed;
- a **heading path** — whatever the parser found inside it;
- an **answer body** — a model writing about that document;
- a **citation quote and label** — the document's own words, travelling back out under
  manicule's name;
- an **authoritative source title** and **source hierarchy** — whatever an adjacent sidecar
  manifest declared, and preferred over the filename.

The last two arrive by a different route from the first, which is why they are separate entries
rather than covered by it. A filename is attacker-controlled but a filesystem constrains its
length and its character set; a manifest field is arbitrary JSON text in a file anybody who can
get a document indexed can also write. And since the pipeline *prefers* the manifest's title, the
filename is now the fallback — so a suite that planted only in the filename would be exercising
the path that is taken second.

A refused manifest's diagnostic is checked too, on its own, and it is the easiest of these to
overlook: an error message feels like manicule's own words, and it is not — it quotes a filename
and a validation failure straight out of the file.

The last test is the one that makes the rest mean anything. It switches autoescaping off and
asserts the same page then *does* carry raw markup — so the assertions above are known to be
testing the escaping rather than testing that the fixture never had any markup in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment
from markupsafe import escape

from manicule.core.provenance import PROVENANCE_KEY, LocalSnapshot, Provenance
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
            # Both of these render a document title and neither was listed here. The fixture
            # left the retriever with no candidates and the trash empty, so both pages rendered
            # their *empty* branch — `Nothing matched` and `The trash is empty` — and the
            # markup they would have shown was never produced by any test. A title is whatever
            # was in the file that got indexed, which makes these the two cheapest routes to
            # attacker-controlled text on this surface.
            "/ui/search?q=retry",
            "/ui/documents/trash",
        ),
    ),
    (
        "heading",
        (
            "/ui/documents/{document}",
            f"/ui/chat/{CONVERSATION}",
            f"/ui/shared/{SHARE_TOKEN}",
            "/ui/search?q=retry",
        ),
    ),
    ("answer", (f"/ui/chat/{CONVERSATION}", f"/ui/shared/{SHARE_TOKEN}")),
    # The passage text a search hit shows is the chunk's own bytes, which is the same string
    # the citation quote carries.
    ("quote", ("/ui/documents/{document}", f"/ui/chat/{CONVERSATION}", "/ui/search?q=retry")),
    # The authoritative source title and hierarchy, out of a sidecar manifest. A manifest is a
    # file in the corpus, so these are attacker-controlled on the same terms as a filename — and
    # arbitrary JSON text rather than something a filesystem constrains in length or character
    # set. They render on the document page's source panel, which is the only place either
    # appears, so this pair is the whole of their coverage.
    ("canonical", ("/ui/documents/{document}",)),
    ("section", ("/ui/documents/{document}",)),
)

SENTINELS = (
    "window.__title",
    "window.__heading",
    "window.__answer",
    "window.__quote",
    "window.__canonical",
    "window.__section",
)
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


def test_a_refused_manifests_reason_reaches_the_page_inert() -> None:
    """The diagnostic quotes the manifest, so the diagnostic is attacker-controlled too.

    A separate test rather than a sixth row in :data:`PLANTED`, because a record carries either a
    source or a reason and never both — one fixture cannot plant in both fields, and a row that
    silently rendered nothing would be the exact failure the ``escape`` assertion above exists to
    catch.

    Worth having at all because it is the easiest of these to overlook: the source title is
    obviously corpus text, while an error message feels like manicule's own words. It is not. It
    quotes a filename and pydantic's report of what was wrong with a field, and both come
    straight out of the file.
    """
    hostile = "<script>window.__reason=1</script>bad manifest"
    backend, document = backend_with_hostile_text()
    refused = document.model_copy(
        update={
            "metadata": {
                PROVENANCE_KEY: Provenance(
                    snapshot=LocalSnapshot(path="mirror/123456.html"),
                    unavailable_reason=hostile,
                ).as_metadata_value()
            }
        }
    )
    backend.store.documents[refused.id] = refused
    backend.organisation_.documents[refused.id] = refused

    with client_for(backend) as client:
        body = client.get(f"/ui/documents/{refused.id}").text

    assert hostile not in body, "the refusal reason reached the page unescaped"
    assert str(escape(hostile)) in body, (
        "the refusal reason does not appear on this page at all, so this test asserts nothing — "
        "an operator with a broken manifest would have no way to find out why"
    )
    assert "<script>window.__reason" not in body


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


@pytest.mark.parametrize(
    "path",
    ["/ui/documents/{document}", "/ui/search?q=retry", "/ui/documents/trash"],
    ids=["document", "search hit", "trash entry"],
)
def test_switching_the_escaping_off_lets_the_markup_through(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """The evidence that the assertions above are load-bearing.

    The same page, rendered by an identical environment with ``autoescape`` off, carries the
    script tag verbatim. If this test ever passes with the escaping *on*, every assertion in
    this file is worthless and this is where that shows up.

    Deliberately a monkeypatched environment rather than a switch on
    :func:`~manicule.web.rendering.build_environment`. An off-switch that ships is an off-switch
    somebody can reach; this one exists only for the length of this test.

    **Parametrised over the two pages that were added to** :data:`PLANTED`. Proving the control
    on one page and then trusting it for two more that render through different templates is
    exactly the assumption that left those two untested in the first place — a page whose
    fixture gives it nothing to render also carries no raw markup with the escaping off, and
    would look identical to a page that escaped correctly.
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
        body = client.get(path.format(document=document.id)).text
    assert MARKUP["title"] in body, (
        f"the escaping was switched off and {path} *still* did not carry raw markup, so the "
        "assertions above are not testing the escaping on this page. Find out what else is "
        "stripping it — or whether the fixture leaves this page with nothing to render."
    )
    assert "<script>" in body
