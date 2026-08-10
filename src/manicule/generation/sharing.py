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

from manicule.core.errors import PolicyError
from manicule.generation.answers import Citation

TOKEN_BYTES = 32
"""256 bits. Guessing is not the threat model a shorter token would fail; leaking is."""


def mint_token() -> str:
    """A fresh share token. Shown to its creator exactly once and never stored."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """The stored form. SHA-256, like the API-key store.

    No salt and no work factor, deliberately: this is a 256-bit random value, not a password,
    so there is no dictionary to defend against and a slow hash would only slow the read path
    that already has a rate limiter in front of it.
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


def new_share(conversation_id: str, *, ttl_s: int, now: datetime | None = None) -> ShareLink:
    """Mint a link for a conversation.

    Raises:
        ValueError: ``ttl_s`` is not positive. A link that expires at or before the moment it
            is created is not a shorter-lived capability, it is a broken feature.
    """
    if ttl_s <= 0:
        msg = f"a share link needs a positive lifetime; got ttl_s={ttl_s}"
        raise ValueError(msg)
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


def redact_for_anonymous(citation: Citation) -> Citation:
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
    return citation.model_copy(update={"quote": "", "uri": ""})


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
    "TOKEN_BYTES",
    "ShareLink",
    "hash_token",
    "is_live",
    "mint_token",
    "new_share",
    "redact_for_anonymous",
    "require_sharing_enabled",
    "tokens_match",
]
