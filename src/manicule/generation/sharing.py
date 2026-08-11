"""Shared conversation links: a bearer capability, treated like one.

The obvious implementation of this feature is an exfiltration primitive, so the starting
point is what it must *not* be: an unauthenticated route where the token is the entire
authorization decision, resolving a workspace from the row it just read, with no expiry, no
revocation, no soft-delete predicate, and passage text in the payload. That is not a
transcript link; it is a public read endpoint over the corpus.

What is built instead:

- **256 bits**, compared in constant time, **stored hashed** and shown to its creator once.
  The argument for hashing is not that the token protects the row from somebody holding the
  database — that person has the conversation anyway. It is that a share token is a live
  credential for an unauthenticated URL, and the database is backed up, exported and
  imported, so plaintext tokens travel into artefacts that leave the access boundary that
  created them.
- **Expiry, by default.** A capability with no expiry accumulates forever and the set of live
  ones becomes unknowable.
- **Revocation that clears the hash**, rather than flipping a boolean beside a still-valid
  token.
- **A snapshot, not a live view.** Somebody shares a conversation after turn 2, keeps using
  it, and turn 7 is public the moment it is written; nobody re-reads a link they already
  sent. Re-sharing after further turns is an explicit new act.
- **Labels, never passage text.** §11.3, and :func:`redact_for_anonymous` is where it is
  enforced.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.anchors import Anchor, PageAnchor
from manicule.core.content import BlockKind
from manicule.core.errors import PolicyError
from manicule.generation.answers import Citation, Verification

TOKEN_BYTES = 32
"""256 bits. Guessing is not the threat model a shorter token would fail; leaking is."""


def mint_token() -> str:
    """A fresh share token. Shown to its creator exactly once and never stored."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """The stored form. SHA-256, like the API-key store.

    No salt and no work factor, deliberately: this is a 256-bit random value, not a password,
    so there is no dictionary to defend against and a work factor would defend against nothing
    while slowing every read of a shared link.

    **There is no rate limiter in front of this**, and the entropy is what carries the
    property rather than a limit on attempts — 256 bits is not guessable at any request rate.
    Rate limiting belongs to Operations (#14) and is worth having for the resource cost of an
    unauthenticated route; it is not what makes the token safe, and saying it was would be a
    guarantee resting on something that does not exist.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(candidate: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented token against a stored hash."""
    return hmac.compare_digest(hash_token(candidate), stored_hash)


@dataclass(frozen=True, slots=True)
class ShareLink:
    """A minted link. :attr:`token` exists only here and in the response that returns it."""

    conversation_id: str
    token: str
    token_hash: str
    shared_at: datetime
    expires_at: datetime

    @property
    def path(self) -> str:
        return f"/shared/{self.token}"

    def model_copy(self, **changes: object) -> ShareLink:
        """A copy with fields replaced. Named for consistency with the pydantic types."""
        import dataclasses  # noqa: PLC0415 - only this helper needs it

        return dataclasses.replace(self, **changes)


def new_share(
    conversation_id: str,
    *,
    ttl_s: int,
    maximum_ttl_s: int,
    now: datetime | None = None,
) -> ShareLink:
    """Mint a link for a conversation.

    ``maximum_ttl_s`` is ``security.sharing.link_ttl_s`` and is a **ceiling**, not a default:
    a requested lifetime is clamped to it rather than refused, so a route that surfaces the
    choice to a user cannot mint a capability that outlives the policy.

    It is a **required** argument. As an optional one it was a ceiling nobody passed — the
    only production caller omitted it — so the clamp existed and never ran, and a hundred-year
    link minted cleanly. A policy that has to be opted into is not a policy.

    Raises:
        ValueError: ``ttl_s`` is not positive. A link that expires at or before the moment it
            is created is not a shorter-lived capability, it is a broken feature.
    """
    if ttl_s <= 0:
        msg = f"a share link needs a positive lifetime; got ttl_s={ttl_s}"
        raise ValueError(msg)
    if maximum_ttl_s <= 0:
        msg = f"the share-link ceiling must be positive; got maximum_ttl_s={maximum_ttl_s}"
        raise ValueError(msg)
    ttl_s = min(ttl_s, maximum_ttl_s)
    moment = now or datetime.now(UTC)
    token = mint_token()
    return ShareLink(
        conversation_id=conversation_id,
        token=token,
        token_hash=hash_token(token),
        shared_at=moment,
        expires_at=moment + timedelta(seconds=ttl_s),
    )


def require_sharing_enabled(enabled: bool) -> None:
    """Refuse to mint a link where sharing is switched off.

    One switch rather than a per-field disclosure policy nobody configures correctly. A
    document *title* can itself be sensitive and an anonymous viewer sees titles, so a
    deployment that cannot disclose those turns the feature off entirely.

    Raises:
        PolicyError: Sharing is disabled.
    """
    if not enabled:
        msg = (
            "sharing is disabled by security.sharing.enabled. A share link is an "
            "unauthenticated URL that discloses document titles and heading paths; enable it "
            "deliberately, or export the conversation instead."
        )
        raise PolicyError(msg)


class CitationLabel(BaseModel):
    """What an anonymous viewer of a shared conversation is given about one citation.

    A **different type**, not a blanked-out :class:`~manicule.generation.answers.Citation`.
    The projection has to be structural: a helper that clears two fields is one a route can
    forget to call, and the fields it left behind were the corpus's internal identifiers —
    ``document_id``, ``chunk_id``, and the anchor itself, whose ``symbol`` is a private
    repository's function name and whose ``sheet``/``ref`` name a spreadsheet and a cell.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: int = Field(ge=1)
    title: str
    heading_path: tuple[str, ...] = ()
    location: str = Field(default="", description="A human location: 'page 4'. Never an id.")
    verification: Verification


def redact_for_anonymous(citation: Citation) -> CitationLabel:
    """The same citation as an anonymous viewer of a shared link receives it.

    **The label and the verification state survive; the passage text does not.** Two of this
    design's commitments pull against each other here — citations are the product and should
    be checkable, and passage text is corpus content a person with no workspace membership
    must not receive — and this is the resolution: the same message renders differently by
    audience, and *the difference is content only*. Never the existence of a citation, never
    its label, never whether it verified.

    So an anonymous viewer is told "this claim was verified against 'Deploy runbook' §
    Rollback at generation time" and cannot read the runbook. That is a weaker guarantee than
    checking it themselves, and it is honestly an attestation rather than a link they could
    follow — which is why :attr:`Citation.uri` goes too. An authenticated viewer with access
    to the workspace opens the same conversation and sees the passages, because they could
    have retrieved them anyway.
    """
    return CitationLabel(
        slot=citation.slot,
        title=citation.title,
        heading_path=anonymous_trail(citation.kind, citation.heading_path),
        location=anonymous_location(citation.anchor),
        verification=citation.verification,
    )


def anonymous_location(anchor: Anchor) -> str:
    """Where a citation points, for a reader outside the workspace.

    **Deliberately not** :func:`~manicule.generation.prompt.describe_location`. That renderer
    exists for the *model*, which is already holding the passage, so it spends the anchor
    freely: a :class:`~manicule.core.anchors.LineAnchor` becomes ``"lines 41-58 of
    charge_customer_card"`` and a :class:`~manicule.core.anchors.CellAnchor` becomes
    ``"Q3 Layoffs!B4"``.

    Those are the two examples this module's own docstring gives for why the anchor is
    dropped from :class:`CitationLabel` — so reusing that function put the field's contents
    back through a differently-named field, and a test asserting on field *names* passed the
    whole time. Dropping a field and then re-encoding it as a string is not a structural
    guarantee; it is the same disclosure with an extra step.

    So: a page number, which is meaningless without the document, and nothing else.
    """
    return f"page {anchor.page}" if isinstance(anchor, PageAnchor) else ""


DISCLOSABLE_TRAIL: frozenset[BlockKind] = frozenset(
    {BlockKind.PROSE, BlockKind.HEADING, BlockKind.LIST, BlockKind.PANEL, BlockKind.MEDIA}
)
"""Kinds whose ``heading_path`` is a section title rather than corpus structure.

``TABLE`` is excluded because the spreadsheet parser puts the **sheet name** in the heading
path, and ``CODE`` because the source parser puts the **symbol chain** there. An allowlist
rather than a denylist: a kind added later is not disclosed until somebody says it may be.
"""


def anonymous_trail(kind: BlockKind, heading_path: tuple[str, ...]) -> tuple[str, ...]:
    """The breadcrumb an anonymous viewer may see.

    §11.3 discloses the heading path deliberately, and for a wiki page or a document that is
    exactly the "'Deploy runbook' § Rollback" attestation the feature is for. For **source
    code** it is something else: the code parser sets ``heading_path`` to the symbol chain, so
    the same field that reads ``Operations > Rollback`` for a runbook reads
    ``PaymentGateway > charge_customer_card`` for a private repository.

    **The discriminator is the block kind, not the anchor type**, and using the anchor was
    wrong in both directions. It missed the spreadsheet entirely — ``anonymous_location``
    refuses to render a ``CellAnchor`` precisely because the sheet name discloses, and then
    the sheet name went out in the next field, because that is where the parser puts it. And
    it over-suppressed: Markdown and plaintext emit a ``LineAnchor`` for ordinary prose, so
    every one of those citations lost the breadcrumb the attestation is *for*.

    The title still goes — a file name is title-class disclosure and the operator's one
    switch covers it — but a sheet name and a symbol chain do not.
    """
    return heading_path if kind in DISCLOSABLE_TRAIL else ()


def is_live(expires_at: datetime | None, *, now: datetime | None = None) -> bool:
    """Whether a link has not expired. A missing expiry is treated as expired.

    Fails closed on purpose: a row without an expiry predates this feature or was written by
    something that skipped it, and treating "no expiry" as "never expires" is how the
    permanently-public link came about in the first place.
    """
    if expires_at is None:
        return False
    moment = now or datetime.now(UTC)
    return expires_at > moment


__all__ = [
    "DISCLOSABLE_TRAIL",
    "TOKEN_BYTES",
    "CitationLabel",
    "ShareLink",
    "anonymous_location",
    "anonymous_trail",
    "hash_token",
    "is_live",
    "mint_token",
    "new_share",
    "redact_for_anonymous",
    "require_sharing_enabled",
    "tokens_match",
]
