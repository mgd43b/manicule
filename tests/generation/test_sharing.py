"""Share links: a bearer capability for an unauthenticated URL, treated like one."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from manicule.core.anchors import Anchor, CellAnchor, HeadingAnchor, LineAnchor, PageAnchor
from manicule.core.content import BlockKind
from manicule.core.errors import PolicyError
from manicule.generation.answers import Citation, Verification
from manicule.generation.sharing import (
    CitationLabel,
    hash_token,
    is_live,
    new_share,
    redact_for_anonymous,
    require_sharing_enabled,
    tokens_match,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
CEILING_S = 30 * 24 * 3600


def test_a_minted_link_carries_the_token_once_and_stores_only_its_hash() -> None:
    """The database is backed up, exported and imported, so a plaintext token travels into
    artifacts that leave the access boundary that created it."""
    link = new_share("conv-1", ttl_s=30 * 24 * 3600, maximum_ttl_s=CEILING_S, now=NOW)

    assert len(link.token) >= 40
    assert link.token not in link.token_hash
    assert link.token_hash == hash_token(link.token)
    assert tokens_match(link.token, link.token_hash)
    assert not tokens_match("some other token", link.token_hash)


def test_a_link_expires_and_a_missing_expiry_is_treated_as_expired() -> None:
    """Fails closed: a row without an expiry predates this feature or was written by
    something that skipped it, and "no expiry" reading as "never expires" is how a
    permanently-public link comes about."""
    link = new_share("conv-1", ttl_s=3600, maximum_ttl_s=CEILING_S, now=NOW)

    assert is_live(link.expires_at, now=NOW + timedelta(minutes=59))
    assert not is_live(link.expires_at, now=NOW + timedelta(hours=2))
    assert not is_live(None, now=NOW)


def test_a_link_with_no_lifetime_is_refused_rather_than_minted_dead() -> None:
    with pytest.raises(ValueError, match="positive lifetime"):
        new_share("conv-1", ttl_s=0, maximum_ttl_s=CEILING_S)


def test_sharing_can_be_switched_off_entirely() -> None:
    """A document *title* can itself be sensitive and an anonymous viewer sees titles, so
    this is one switch rather than a per-field disclosure policy nobody configures right."""
    require_sharing_enabled(True)

    with pytest.raises(PolicyError, match=r"security\.sharing\.enabled"):
        require_sharing_enabled(False)


def citation(anchor: Anchor | None = None, kind: BlockKind = BlockKind.PROSE) -> Citation:
    return Citation(
        slot=1,
        document_id="doc-1",
        uri="https://intranet.invalid/runbook",
        title="Deploy runbook",
        heading_path=("Operations", "Rollback"),
        anchor=anchor or PageAnchor(page=4),
        kind=kind,
        chunk_id="c1",
        quote="Roll back with `deploy --rollback`.",
        verification=Verification.RESOLVED,
    )


@pytest.mark.parametrize(
    ("anchor", "kind", "expected_location", "expected_trail"),
    [
        pytest.param(
            PageAnchor(page=4), BlockKind.PROSE, "page 4", ("Operations", "Rollback"), id="page"
        ),
        pytest.param(
            LineAnchor(start=41, end=58, symbol="charge_customer_card"),
            BlockKind.CODE,
            "",
            (),
            id="code-symbol-chain-suppressed",
        ),
        pytest.param(
            CellAnchor(sheet="Q3 Layoffs", ref="B4:D12"),
            BlockKind.TABLE,
            "",
            (),
            id="sheet-name-suppressed-in-both-fields",
        ),
        pytest.param(
            LineAnchor(start=1, end=4),
            BlockKind.PROSE,
            "",
            ("Operations", "Rollback"),
            id="prose-with-a-line-anchor-keeps-its-breadcrumb",
        ),
        pytest.param(
            HeadingAnchor(path=("Operations", "Rollback")),
            BlockKind.HEADING,
            "",
            ("Operations", "Rollback"),
            id="heading",
        ),
    ],
)
def test_an_anonymous_viewer_gets_a_label_and_never_the_anchors_contents(
    anchor: Anchor,
    kind: BlockKind,
    expected_location: str,
    expected_trail: tuple[str, ...],
) -> None:
    """The same message renders differently by audience, and the difference is **content
    only** — never the existence of a citation, never whether it verified.

    **Parametrized over the anchors that actually disclose something**, which the original
    test was not: it used a ``PageAnchor`` throughout and asserted on field *names*, so it
    passed while the anchor's contents flowed through a differently-named field. Rendering the
    location with the prompt's own helper put ``lines 41-58 of charge_customer_card`` and
    ``Q3 Layoffs!B4:D12`` back in — a private repository's function name and a spreadsheet
    nobody outside the workspace should learn the title of.
    """
    shared = redact_for_anonymous(citation(anchor=anchor, kind=kind))

    assert shared.title == "Deploy runbook"
    assert shared.verification is Verification.RESOLVED
    assert shared.location == expected_location
    assert shared.heading_path == expected_trail

    # Values, not field names. A field-name check is what let the leak through.
    rendered = shared.model_dump_json()
    for secret in ("charge_customer_card", "Q3 Layoffs", "B4:D12", "deploy --rollback", "intranet"):
        assert secret not in rendered, f"{secret!r} reached an anonymous viewer"


def test_the_anonymous_projection_has_nowhere_to_put_the_corpus() -> None:
    """Structural, not remembered: a helper that blanks fields is one a route can skip."""
    fields = set(CitationLabel.model_fields)

    assert not fields & {"quote", "uri", "document_id", "chunk_id", "anchor"}
