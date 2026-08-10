"""Share links: a bearer capability for an unauthenticated URL, treated like one."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from manicule.core.anchors import PageAnchor
from manicule.core.errors import PolicyError
from manicule.generation.answers import Citation, Verification
from manicule.generation.sharing import (
    hash_token,
    is_live,
    new_share,
    redact_for_anonymous,
    require_sharing_enabled,
    tokens_match,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def test_a_minted_link_carries_the_token_once_and_stores_only_its_hash() -> None:
    """The database is backed up, exported and imported, so a plaintext token travels into
    artefacts that leave the access boundary that created it."""
    link = new_share("conv-1", ttl_s=30 * 24 * 3600, now=NOW)

    assert len(link.token) >= 40
    assert link.token not in link.token_hash
    assert link.token_hash == hash_token(link.token)
    assert tokens_match(link.token, link.token_hash)
    assert not tokens_match("some other token", link.token_hash)


def test_a_link_expires_and_a_missing_expiry_is_treated_as_expired() -> None:
    """Fails closed: a row without an expiry predates this feature or was written by
    something that skipped it, and "no expiry" reading as "never expires" is how a
    permanently-public link comes about."""
    link = new_share("conv-1", ttl_s=3600, now=NOW)

    assert is_live(link.expires_at, now=NOW + timedelta(minutes=59))
    assert not is_live(link.expires_at, now=NOW + timedelta(hours=2))
    assert not is_live(None, now=NOW)


def test_a_link_with_no_lifetime_is_refused_rather_than_minted_dead() -> None:
    with pytest.raises(ValueError, match="positive lifetime"):
        new_share("conv-1", ttl_s=0)


def test_sharing_can_be_switched_off_entirely() -> None:
    """A document *title* can itself be sensitive and an anonymous viewer sees titles, so
    this is one switch rather than a per-field disclosure policy nobody configures right."""
    require_sharing_enabled(True)

    with pytest.raises(PolicyError, match=r"security\.sharing\.enabled"):
        require_sharing_enabled(False)


def test_an_anonymous_viewer_gets_the_label_and_the_verification_state_but_no_passage() -> None:
    """The same message renders differently by audience, and the difference is **content
    only** — never the existence of a citation, never its label, never whether it verified."""
    citation = Citation(
        slot=1,
        document_id="doc-1",
        uri="https://intranet.invalid/runbook",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=PageAnchor(page=4),
        chunk_id="c1",
        quote="Roll back with `deploy --rollback`.",
        verification=Verification.RESOLVED,
    )

    shared = redact_for_anonymous(citation)

    assert shared.title == "Deploy runbook"
    assert shared.heading_path == ("Operations", "Rollback")
    assert shared.location == "page 4"
    assert shared.verification is Verification.RESOLVED

    # A *different type*, not a blanked-out Citation. The corpus's internal identifiers have
    # nowhere to go: `LineAnchor.symbol` is a private repository's function name and
    # `CellAnchor` names a spreadsheet and a cell, so keeping the anchor "because it is not
    # text" would disclose exactly the things a workspace boundary exists to hold.
    fields = set(type(shared).model_fields)
    assert not fields & {"quote", "uri", "document_id", "chunk_id", "anchor"}
