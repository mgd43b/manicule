"""A memory's evidence reaches HTML here, and it reaches it inert.

The evidence on a memory is text somebody else wrote: the quote is a message copied out of a chat,
the author is a display name, and the locator is whatever address they pasted. All three arrive in
a file anybody who can get a document indexed can also write, so on this surface they are
attacker-controlled on the same terms as a title — and unlike a title, nothing in a filesystem
constrains them.

``tests/web/test_escaping.py`` is the model, and these follow its method: plant a distinct
payload in each field, check the page against what the route actually returned, assert both that
the raw markup is **absent** and that the escaped form is **present** (the second half is what
makes it a test of escaping rather than of omission), and switch autoescaping off to prove the
assertions are load-bearing.

**The kind and the relationship target are not planted**, and cannot be: both are validated to a
token before they can be stored, and re-validated on every read. What is left for this surface to
prove is the free text.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from jinja2 import Environment
from markupsafe import escape

from manicule.core.knowledge import (
    KNOWLEDGE_KEY,
    EvidenceSource,
    EvidenceStatus,
    Knowledge,
    KnowledgeRecord,
    KnowledgeType,
)
from manicule.web import rendering
from tests.web.support import backend_with_hostile_text, client_for

HOSTILE: dict[str, str] = {
    "quote": "<svg onload=window.__quote=1>failover is manual",
    "author": "<script>window.__author=1</script>Jane Doe",
    "locator": "https://example.test/archives/<script>window.__locator=1</script>",
    "reason": "<img src=x onerror=window.__reason=1>knowledge_type: not valid",
}

PAGES = ("/ui/search?q=retry", "/ui/documents/{document}")


def _hostile_record() -> Knowledge:
    return Knowledge(
        record=KnowledgeRecord(
            knowledge_type=KnowledgeType.CONSTRAINT,
            evidence_status=EvidenceStatus.SINGLE_SOURCE,
            evidence=(
                EvidenceSource(
                    kind="slack",
                    locator=HOSTILE["locator"],
                    author=HOSTILE["author"],
                    quote=HOSTILE["quote"],
                    said_at=datetime(2026, 3, 2, 14, 5, tzinfo=UTC),
                ),
            ),
        )
    )


def _backend_holding(knowledge: Knowledge):  # noqa: ANN202 - the fixture's own pair, inferred
    backend, document = backend_with_hostile_text()
    held = document.model_copy(
        update={"metadata": {**document.metadata, KNOWLEDGE_KEY: knowledge.as_metadata_value()}}
    )
    backend.store.documents[held.id] = held
    backend.organization_.documents[held.id] = held
    return backend, held


@pytest.mark.parametrize("field", ["quote", "author", "locator"])
@pytest.mark.parametrize("path", PAGES, ids=["search hit", "document page"])
def test_a_memorys_evidence_renders_inert(field: str, path: str) -> None:
    backend, document = _backend_holding(_hostile_record())

    with client_for(backend) as client:
        body = client.get(path.format(document=document.id)).text

    assert HOSTILE[field] not in body, f"the evidence {field} reached the page unescaped"
    assert str(escape(HOSTILE[field])) in body, (
        f"the evidence {field} does not appear on this page at all, escaped or not, so this "
        f"test is asserting nothing about it"
    )
    for sentinel in ("__quote", "__author", "__locator"):
        assert f"<script>window.{sentinel}" not in body


@pytest.mark.parametrize("path", PAGES, ids=["search hit", "document page"])
def test_a_refusal_reason_renders_inert(path: str) -> None:
    """The reason quotes the author's own value — the unknown type they wrote, echoed back.

    Easy to overlook for the reason a refused manifest's reason is: it reads as manicule's words,
    and part of it is the file's.
    """
    backend, document = _backend_holding(Knowledge(unavailable_reason=HOSTILE["reason"]))

    with client_for(backend) as client:
        body = client.get(path.format(document=document.id)).text

    assert HOSTILE["reason"] not in body
    assert str(escape(HOSTILE["reason"])) in body


def test_evidence_is_never_a_link() -> None:
    """The locator is shown as text, so there is no ``href`` for a bad address to reach.

    It is restricted to http and https before it is stored, so a link would be safe. An address
    that is never an ``href`` has no sink to get wrong at all, which is the stronger property and
    the one the document page's source panel already holds.
    """
    backend, document = _backend_holding(_hostile_record())

    with client_for(backend) as client:
        body = client.get(f"/ui/documents/{document.id}").text

    assert 'href="https://example.test' not in body


@pytest.mark.parametrize("path", PAGES, ids=["search hit", "document page"])
def test_switching_the_escaping_off_lets_the_evidence_through(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """The control: the same page with ``autoescape`` off carries the markup verbatim.

    If this passes with the escaping *on*, every assertion above is worthless — so it is
    parametrized over both pages, because proving the control on one and trusting it on the other
    is how a page that renders nothing comes to look like a page that escaped correctly.
    """
    unsafe = Environment(
        loader=rendering.ENVIRONMENT.loader,
        autoescape=False,  # noqa: S701 - an unescaped environment is exactly what this proves
        undefined=rendering.ENVIRONMENT.undefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    monkeypatch.setattr(rendering, "ENVIRONMENT", unsafe)
    backend, document = _backend_holding(_hostile_record())

    with client_for(backend) as client:
        body = client.get(path.format(document=document.id)).text

    assert HOSTILE["quote"] in body, (
        f"the escaping was switched off and {path} still did not carry the raw evidence, so the "
        f"assertions above are not testing the escaping on this page"
    )
