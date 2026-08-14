"""The one page with no credential, and what it may not disclose.

``GET /ui/shared/{token}`` is the highest-risk page on this surface by a distance: no
credential, and it renders conversation content. Everything that makes it safe is somewhere
else, deliberately — :mod:`manicule.generation.sharing` mints and hashes the token, the store
resolves it in one statement with expiry, revocation, soft-delete and the snapshot boundary as
predicates of that statement, and the projection to citation **labels** happens in storage.

So what this file asserts is that the page adds no second path: the redaction that already
exists is the one it renders, and nothing an owner sees leaks into what a guest sees.
"""

from __future__ import annotations

import pytest

from manicule.generation.answers import Verification
from manicule.generation.sharing import (
    anonymous_location,
    anonymous_trail,
    is_live,
    new_share,
    redact_for_anonymous,
    tokens_match,
)
from tests.web.support import (
    CONVERSATION,
    MARKUP,
    SHARE_TOKEN,
    backend_with_hostile_text,
    client_for,
)


def test_the_shared_page_renders_labels_and_never_the_passage() -> None:
    """The guest gets a title, a breadcrumb and a verdict. Not the quote, not any identifier.

    The owner's page for the same conversation renders the quote, which is the control: without
    it this test would pass for a page that renders nothing.
    """
    backend, document = backend_with_hostile_text()
    with client_for(backend) as client:
        guest = client.get(f"/ui/shared/{SHARE_TOKEN}").text
        owner = client.get(f"/ui/chat/{CONVERSATION}").text
    assert "resolved" in guest, "the verdict is part of the attestation and must survive"
    assert MARKUP["quote"] not in guest
    assert "the client retries twice" not in guest
    assert document.id not in guest
    assert document.uri not in guest
    assert CONVERSATION not in guest
    # The control: the same citation, for a reader who holds a key for this workspace.
    assert document.id in owner


def test_the_shared_page_offers_no_navigation_into_the_installation() -> None:
    """A guest holds a bearer URL and nothing else.

    A navigation listing every area would be a list of what exists here, disclosed to somebody
    who can reach none of it. The page uses a smaller frame, and this is what keeps that true.
    """
    backend, _ = backend_with_hostile_text()
    with client_for(backend) as client:
        guest = client.get(f"/ui/shared/{SHARE_TOKEN}").text
    assert "/ui/documents" not in guest
    assert "/ui/admin" not in guest
    assert "Workspace" not in guest


@pytest.mark.parametrize(
    "token", ["", "not-a-token", SHARE_TOKEN.upper(), SHARE_TOKEN + "x", SHARE_TOKEN[:-1]]
)
def test_every_way_of_being_wrong_looks_identical(token: str) -> None:
    """An unknown token, an expired one, a revoked one and sharing being off all render the same.

    Distinguishing them for an unauthenticated reader tells them which of their guesses was
    closest, which is the only feedback a guessing attack needs.
    """
    backend, _ = backend_with_hostile_text()
    with client_for(backend) as client:
        body = client.get(f"/ui/shared/{token or 'x'}").text
    assert "does not resolve" in body
    assert MARKUP["answer"] not in body


def test_sharing_switched_off_closes_a_link_that_already_resolved() -> None:
    """The read path consults configuration, not only the minting path.

    An operator switching sharing off has decided the disclosure already made is the problem,
    so a link minted yesterday stops resolving today.
    """
    backend, _ = backend_with_hostile_text()
    backend.settings = backend.settings.model_copy(
        update={
            "security": backend.settings.security.model_copy(
                update={
                    "sharing": backend.settings.security.sharing.model_copy(
                        update={"enabled": False}
                    )
                }
            )
        }
    )
    with client_for(backend) as client:
        body = client.get(f"/ui/shared/{SHARE_TOKEN}").text
    assert "does not resolve" in body


def test_the_page_uses_the_redaction_that_already_exists() -> None:
    """Named here so the dependency is a test rather than a convention.

    Every one of these is what makes the anonymous view safe, and a page that reimplemented any
    of them would be the second path this module's docstring exists to refuse. They are
    exercised for their own behavior in ``tests/api/test_sharing.py``; what is asserted here is
    that they are the functions the page's data came through.
    """
    from manicule.core.anchors import PageAnchor  # noqa: PLC0415 - only this test builds one
    from manicule.core.content import BlockKind  # noqa: PLC0415
    from manicule.generation.answers import Citation  # noqa: PLC0415

    citation = Citation(
        slot=1,
        document_id="doc",
        uri="file:///corpus/secret.md",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=PageAnchor(page=4),
        chunk_id="chunk",
        kind=BlockKind.PROSE,
        quote="the passage nobody outside the workspace may read",
        verification=Verification.RESOLVED,
    )
    label = redact_for_anonymous(citation)
    assert label.title == "Deploy runbook"
    assert label.location == anonymous_location(citation.anchor) == "page 4"
    assert label.heading_path == anonymous_trail(citation.kind, citation.heading_path)
    assert not hasattr(label, "quote")
    assert not hasattr(label, "document_id")


def test_a_share_link_cannot_be_minted_without_a_ceiling() -> None:
    """``maximum_ttl_s`` is required, and that is the whole of the clamp.

    As an optional argument it was a ceiling nobody passed: the clamp existed and never ran, and
    a hundred-year link minted cleanly. Asserted here because the browser surface renders share
    state, and a surface that displays a policy is a surface people trust it from.
    """
    with pytest.raises(TypeError):
        new_share("conv", ttl_s=60)  # pyright: ignore[reportCallIssue] - the point of the test
    link = new_share("conv", ttl_s=10**6, maximum_ttl_s=60)
    assert (link.expires_at - link.shared_at).total_seconds() == 60
    assert is_live(link.expires_at)
    assert tokens_match(link.token, link.token_hash)
    assert not tokens_match(link.token + "x", link.token_hash)
