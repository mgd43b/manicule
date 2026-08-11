"""The unauthenticated share route, driven end to end.

``GET /shared/{token}`` is the highest-risk endpoint on this surface: no credential, and it
returns conversation content. The sharing machinery it stands on is already covered by
``tests/generation/test_sharing.py`` and ``tests/test_storage_conversations.py``; what is
tested here is that the **route reaches it and adds nothing**, because the way this feature
goes wrong is a second path to the same rows with one predicate missing.

Every negative here has a positive beside it. A route that returned nothing whatever the token
would satisfy "an unknown token discloses nothing" perfectly and would not be a feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from manicule.core.anchors import HeadingAnchor, PageAnchor
from manicule.generation.answers import Citation, Verification
from manicule.generation.history import Turn
from manicule.generation.sharing import hash_token
from tests.api.support import backend_with_a_document, client_for, envelope
from tests.app.fakes import FakeBackend

FORBIDDEN = 403
NOT_FOUND = 404
CONVERSATION = "conv_1"


def _citation() -> Citation:
    """One citation carrying every field an anonymous viewer must not receive."""
    return Citation(
        slot=1,
        document_id="doc-private",
        chunk_id="chunk-private",
        uri="file:///corpus/runbooks/deploy.md",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=PageAnchor(page=4),
        quote="the client retries twice before failing over",
        verification=Verification.RESOLVED,
    )


def _seeded() -> tuple[FakeBackend, str]:
    """A backend holding one conversation with a citation-bearing answer."""
    backend, _ = backend_with_a_document()
    backend.conversations_.seed(
        CONVERSATION,
        Turn(role="user", content="what does the client do"),
        Turn(role="assistant", content="It retries twice.", citations=(_citation(),)),
    )
    return backend, CONVERSATION


def test_a_minted_link_resolves_and_returns_labels_rather_than_passages() -> None:
    """The positive control, and the whole disclosure contract in one assertion.

    The label and the verification state survive; the passage text, the document id, the chunk
    id, the URI and the anchor do not. Those are not blanked fields — the anonymous payload
    has nowhere to put them.
    """
    backend, conversation = _seeded()
    with client_for(backend) as client:
        minted = envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))
        token = str(minted["data"]["token"])
        body = envelope(client.get(f"/shared/{token}"))
    turns = body["data"]["turns"]
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    label = turns[1]["citations"][0]
    assert label["title"] == "Deploy runbook"
    assert label["heading_path"] == ["Operations", "Rollback"]
    assert label["location"] == "page 4"
    assert label["verification"] == "resolved"
    assert set(label) == {"slot", "title", "heading_path", "location", "verification"}


def test_the_shared_payload_carries_no_identifier_of_any_kind() -> None:
    """No document id, no chunk id, no URI, no quote — and no conversation id either.

    The last one matters: an id plus a store handle bound to the owning workspace turns into
    full citations, so handing one to an anonymous reader rebuilds the two-step the
    single-statement resolution exists to replace.
    """
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        response = client.get(f"/shared/{token}")
    for forbidden in (
        "doc-private",
        "chunk-private",
        "file:///corpus/runbooks/deploy.md",
        "retries twice before failing over",
        conversation,
    ):
        assert forbidden not in response.text, f"{forbidden!r} reached an anonymous viewer"


def test_an_unknown_token_discloses_nothing_and_looks_like_every_other_failure() -> None:
    """One empty answer for unknown, expired, revoked, deleted and switched-off.

    Distinguishing them for an unauthenticated caller tells them which of their guesses was
    closest.
    """
    backend, _ = _seeded()
    with client_for(backend) as client:
        body = envelope(client.get("/shared/definitely-not-a-token"))
    assert body["ok"] is True
    assert body["data"]["turns"] == []


def test_revoking_a_link_stops_it_resolving() -> None:
    """Revocation clears the stored hash, so the token stops resolving rather than looking
    revoked beside one that still works."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        assert envelope(client.get(f"/shared/{token}"))["data"]["turns"]
        client.delete(f"/api/v1/conversations/{conversation}/share")
        after = envelope(client.get(f"/shared/{token}"))
    assert after["data"]["turns"] == []


def test_deleting_a_conversation_also_revokes_its_link() -> None:
    """A soft delete that leaves a public link resolving is a delete that did not delete."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        client.delete(f"/api/v1/conversations/{conversation}")
        after = envelope(client.get(f"/shared/{token}"))
    assert after["data"]["turns"] == []


def test_an_expired_link_stops_resolving() -> None:
    """A capability with no expiry accumulates forever; one past its expiry is not a capability."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        stored, _, boundary = backend.conversations_.shares[conversation]
        backend.conversations_.shares[conversation] = (
            stored,
            datetime.now(UTC) - timedelta(seconds=1),
            boundary,
        )
        after = envelope(client.get(f"/shared/{token}"))
    assert after["data"]["turns"] == []


def test_only_a_digest_of_the_token_is_stored() -> None:
    """The stored form is a hash, which is what stops a backup being a set of live links."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
    stored, _, _ = backend.conversations_.shares[conversation]
    assert stored != token
    assert stored == hash_token(token)


def test_a_listing_of_conversations_never_carries_the_token() -> None:
    """A bearer capability in a listing is one in every log and cache in front of the route."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        listing = client.get("/api/v1/conversations")
    assert token not in listing.text
    stored, _, _ = backend.conversations_.shares[conversation]
    assert stored not in listing.text


def test_sharing_switched_off_refuses_to_mint_and_stops_resolving_what_was_minted() -> None:
    """Both halves, because an operator switching sharing off has decided the disclosure
    already made is the problem — so the *read* path checks it too, not only minting."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        assert envelope(client.get(f"/shared/{token}"))["data"]["turns"]

    disabled, _ = backend_with_a_document(security={"sharing": {"enabled": False}})
    disabled.conversations_ = backend.conversations_
    with client_for(disabled) as client:
        refused = client.post(f"/api/v1/conversations/{conversation}/share", json={})
        read = envelope(client.get(f"/shared/{token}"))
    assert refused.status_code == FORBIDDEN
    assert envelope(refused)["error"]["type"] == "PolicyError"
    assert read["data"]["turns"] == []


def test_a_requested_lifetime_is_clamped_to_the_configured_ceiling() -> None:
    """A route that surfaces the choice to a user cannot mint a capability outliving policy."""
    backend, conversation = _seeded()
    ceiling = backend.settings.security.sharing.link_ttl_s
    with client_for(backend) as client:
        minted = envelope(
            client.post(
                f"/api/v1/conversations/{conversation}/share", json={"ttl_s": ceiling * 100}
            )
        )
    expires = datetime.fromisoformat(str(minted["data"]["expires_at"]))
    assert expires <= datetime.now(UTC) + timedelta(seconds=ceiling + 5)


def test_re_sharing_invalidates_the_previous_token() -> None:
    """One live link per conversation, so re-sharing is an explicit new act with a new snapshot."""
    backend, conversation = _seeded()
    with client_for(backend) as client:
        first = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        second = str(
            envelope(client.post(f"/api/v1/conversations/{conversation}/share", json={}))["data"][
                "token"
            ]
        )
        assert first != second
        assert envelope(client.get(f"/shared/{first}"))["data"]["turns"] == []
        assert envelope(client.get(f"/shared/{second}"))["data"]["turns"]


def test_sharing_a_conversation_that_does_not_exist_is_a_refusal() -> None:
    """Not a link to nothing. A minted token for an absent conversation is a live credential
    that would start working the moment an id was reused."""
    backend, _ = _seeded()
    with client_for(backend) as client:
        response = client.post("/api/v1/conversations/conv_nope/share", json={})
    assert response.status_code == NOT_FOUND


@pytest.mark.parametrize(
    ("kind", "path", "disclosed"),
    [
        ("prose", ("Operations", "Rollback"), True),
        ("code", ("PaymentGateway", "charge_customer_card"), False),
    ],
)
def test_a_breadcrumb_is_disclosed_by_block_kind_and_not_by_anchor(
    kind: str, path: tuple[str, ...], *, disclosed: bool
) -> None:
    """A source file's ``heading_path`` is its symbol chain, not a section title.

    The same field reads ``Operations > Rollback`` for a runbook and
    ``PaymentGateway > charge_customer_card`` for a private repository, so the discriminator
    is the block kind. Driven through the route rather than through the redactor, because the
    property under test is that the route reaches the redactor at all.
    """
    from manicule.core.content import BlockKind  # noqa: PLC0415 - only this test names one

    backend, _ = backend_with_a_document()
    citation = Citation(
        slot=1,
        document_id="doc-private",
        chunk_id="chunk-private",
        uri="file:///corpus/gateway.py",
        title="gateway.py",
        heading_path=path,
        kind=BlockKind(kind),
        anchor=HeadingAnchor(path=path),
        quote="def charge_customer_card(...)",
        verification=Verification.RESOLVED,
    )
    backend.conversations_.seed(
        CONVERSATION,
        Turn(role="assistant", content="It charges the card.", citations=(citation,)),
    )
    with client_for(backend) as client:
        token = str(
            envelope(client.post(f"/api/v1/conversations/{CONVERSATION}/share", json={}))["data"][
                "token"
            ]
        )
        body = envelope(client.get(f"/shared/{token}"))
    trail = body["data"]["turns"][0]["citations"][0]["heading_path"]
    assert (trail == list(path)) is disclosed
