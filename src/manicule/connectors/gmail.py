"""The Gmail connector: history-based discovery, raw fetch and complete reconciliation.

``docs/connectors/gmail.md`` §2-§7 is the design; :mod:`manicule.connectors.gmail_client` is the
HTTP half and :mod:`manicule.connectors.config` the settings. This module is the part that
decides what is in scope, what changed, and what is still in the mailbox.

**Scope is label ids, and nothing else.** The full walk and reconciliation list each excluded
label into an in-memory set of message ids, held for that one walk, then page the mailbox and
skip those ids. Incremental discovery cannot do that — a history record names a message, not
its labels — so it reads each candidate's *current* label ids and tests them against the excluded
*label* ids. Both rest on the same resolved set, so the three paths agree on what is in scope
(``gmail.md`` §3).

**A message is fetched once.** Its bytes never change, so every message carries one constant
version token and a stored message is never fetched again by id. What a changed representation
costs is paid through the scope fingerprint instead: the representation version is part of it,
so bumping it discards the history position and the next sync is a complete walk (§4.1).

**Nothing here may name a message, an address or a label.** Errors and log records are built from
counts and operation names; a record about one message belongs to that message's document
(§10).
"""

from __future__ import annotations

import email.parser
import email.policy
import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Final
from urllib.parse import quote

from manicule.connectors import config as gmail_settings
from manicule.connectors.config import (
    ALWAYS_EXCLUDED_LABELS,
    GMAIL_CONNECTOR_NAME,
    GmailConfig,
    resolve_label_ids,
)
from manicule.connectors.credentials import mailbox_mismatch
from manicule.connectors.errors import ConnectorError, CursorExpiredError, NotFoundError
from manicule.connectors.gmail_client import GmailClient, HistoryPage, Profile
from manicule.core.content import Metadata, RawDocument
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = [
    "MESSAGE_MEDIA_TYPE",
    "GmailConnector",
]

_log = logging.getLogger("manicule.connectors.gmail")

MESSAGE_MEDIA_TYPE: Final = "message/rfc822"
"""What a fetched message is handed to the pipeline as; the existing mail parser reads it."""

_VIEW: Final = "https://mail.google.com/mail/"

_FINGERPRINT_KEY: Final = "scope_fingerprint"
_IDENTITY_KEY: Final = "pinned_identity"
_VERSION_KEY: Final = "version_token"

_CANDIDATES: Final = frozenset({"messageAdded", "labelRemoved"})
"""History records that make a message a candidate (§4.2). ``labelAdded`` and ``messageDeleted``
can only take a message out of scope, and that is reconciliation's to act on."""


class GmailConnector:
    """Discovers, fetches and reconciles the messages of one Gmail mailbox."""

    def __init__(
        self,
        config: GmailConfig,
        client: GmailClient,
        *,
        name: str = GMAIL_CONNECTOR_NAME,
        renewal: str = "",
    ) -> None:
        """Args:
        config: The validated Gmail settings.
        client: The REST client; this connector sets it up and tears it down.
        name: The configured instance's name, which files its documents (see
            ``plugin._source_name``).
        renewal: What a person does about a token for the wrong mailbox, appended to that
            refusal. The credential knows the sentence; the connector only repeats it.
        """
        self.name = name
        self._config = config
        self._client = client
        self._renewal = renewal
        self._user_excluded: tuple[str, ...] = ()
        """Label ids of ``exclude_labels``, resolved at setup and again at the start of a run."""
        self._resolved = not config.exclude_labels
        self._next: str | None = None
        self._enumerated = False
        self._fetched = 0

    # --- identity and scope --------------------------------------------------------------

    @property
    def pinned_identity(self) -> str:
        """The digest of the mailbox this instance indexes; a synced source is held to it."""
        return self._config.pinned_identity

    @property
    def source_scope(self) -> str:
        """A non-secret name for this scope: the mailbox's digest, never its address or labels."""
        return f"gmail:{self._config.pinned_identity[:16]}"

    @property
    def scope_fingerprint(self) -> str:
        """Cursor identity: the mailbox, the resolved excluded label ids and the representation.

        A synchronous property over what the connector resolved at setup or at the start of the
        last run, because the pipeline reads it before discovery begins. A label that is deleted
        and recreated under the same name gets a new id after that read; the watermark carries
        the fingerprint it was recorded under and :meth:`_resume_position` compares it with a
        freshly resolved one, so the stale read costs nothing but a complete walk.
        """
        if not self._resolved:
            msg = "the Gmail connector's setup() was not awaited before its scope was read"
            raise ConnectorError(msg)
        return self._config.scope_fingerprint(self.source_scope, self._user_excluded)

    @property
    def _excluded_label_ids(self) -> frozenset[str]:
        """Every label id that takes a message out of scope. System labels' ids are their names."""
        return frozenset((*ALWAYS_EXCLUDED_LABELS, *self._user_excluded))

    # --- lifecycle -----------------------------------------------------------------------

    async def setup(self) -> None:
        """Build the client, then refuse a token for another mailbox before anything is read."""
        await self._client.setup()
        try:
            await self._begin()
        except BaseException:
            await self._client.teardown()
            raise

    async def teardown(self) -> None:
        await self._client.teardown()

    async def health(self) -> HealthReport:
        """One ``getProfile``, which is also the mailbox check."""
        try:
            await self._check_mailbox()
        except ConnectorError as exc:
            return HealthReport.failing(
                f"Gmail did not accept the credential: {exc}",
                remedy="Run `manicule doctor` and see the gmail-credential check.",
            )
        return HealthReport.healthy("Gmail answered and the credential reads the mailbox")

    def metrics(self) -> tuple[Metric, ...]:
        return (
            Metric(name="gmail_requests", value=float(self._client.requests)),
            Metric(name="gmail_throttled", value=float(self._client.throttled)),
            Metric(name="gmail_messages_fetched", value=float(self._fetched)),
        )

    async def _check_mailbox(self) -> Profile:
        """Read the profile and refuse a token that belongs to another mailbox."""
        profile = await self._client.get_profile()
        mismatch = mailbox_mismatch(profile.email_address, self._config.account, self._renewal)
        if mismatch is not None:
            raise mismatch
        return profile

    async def _begin(self) -> Profile:
        """Start of a run: the mailbox pin, then the excluded label names as ids, afresh.

        The profile's ``historyId`` is returned because a complete walk needs it captured
        *before* it lists anything (§4.1), and this is before.

        Raises:
            MailboxMismatchError: The token reads another mailbox. Nothing has been read.
            ConfigError: An ``exclude_labels`` name is not a label of this mailbox.
        """
        profile = await self._check_mailbox()
        if self._config.exclude_labels:
            labels = await self._client.list_labels()
            self._user_excluded = resolve_label_ids(
                self._config.exclude_labels, {label.name: label.id for label in labels}
            )
        self._resolved = True
        return profile

    # --- watermark -----------------------------------------------------------------------

    @property
    def watermark(self) -> Watermark | None:
        """The history position to resume from, once a walk has completed; otherwise ``None``.

        Carries the scope fingerprint and the pinned identity it was recorded under, so a
        position is never resumed under a different scope or mailbox even by a caller that
        does not compare them itself.
        """
        if not self._enumerated or self._next is None:
            return None
        return Watermark(
            value=self._next,
            observed_at=datetime.now(tz=UTC),
            metadata={
                _FINGERPRINT_KEY: self.scope_fingerprint,
                _IDENTITY_KEY: self.pinned_identity,
            },
        )

    def _resume_position(self, watermark: Watermark | None) -> str | None:
        """The stored history position, if it was recorded under this scope and mailbox."""
        if watermark is None:
            return None
        recorded = watermark.metadata
        if recorded.get(_IDENTITY_KEY) != self.pinned_identity:
            return None
        if recorded.get(_FINGERPRINT_KEY) != self.scope_fingerprint:
            return None
        return watermark.value

    # --- discovery -----------------------------------------------------------------------

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        async for batch in self.discover_batches(watermark):
            for document in batch:
                yield document

    async def discover_batches(
        self, watermark: Watermark | None
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """One batch per ``messages.list`` or ``history.list`` page (§4.4)."""
        self._enumerated = False
        self._next = None
        profile = await self._begin()
        position = self._resume_position(watermark)
        if position is not None:
            try:
                async for batch in self._incremental(position):
                    yield batch
            except CursorExpiredError:
                # Google no longer holds history that far back (§4.3). Nothing that was yielded is
                # wrong, and the complete walk that follows yields everything again at the
                # same constant token, so only messages not already stored are fetched.
                _log.warning(
                    "source %r: Gmail no longer holds history from the stored position; "
                    "running a complete walk",
                    self.name,
                )
                self._next = None
                self._enumerated = False
            else:
                return
        async for batch in self._full(profile.history_id):
            yield batch

    async def _full(self, history_id: str) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """§4.1: ``history_id`` was read before the first listing and is the next watermark."""
        async for ids in self._inventory():
            yield [self._discovered(message_id) for message_id in ids]
        self._next = history_id
        self._enumerated = True

    async def _incremental(self, position: str) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """§4.2: candidates from the history log, each judged by its current label ids."""
        excluded = self._excluded_label_ids
        judged: set[str] = set()
        """Candidates already looked up this walk; ids of messages the log named, not the
        mailbox's inventory, and gone when the walk is."""
        following: str | None = None
        async for page in self._client.history(position):
            if following is None:
                following = page.history_id
            batch = await self._judge(page, judged, excluded)
            if batch:
                yield batch
        # ``history`` yields at least one page, but a position is only worth storing if it
        # was read: the alternative is to pretend the walk has no end.
        if following is None:  # pragma: no cover - the client always yields a page
            msg = "Gmail users.history.list returned no page"
            raise ConnectorError(msg)
        self._next = following
        self._enumerated = True

    async def _judge(
        self, page: HistoryPage, judged: set[str], excluded: frozenset[str]
    ) -> list[DiscoveredDoc]:
        """The candidates of one page whose current labels are all in scope.

        A history record is not the message's label set — ``labelRemoved`` names what came off,
        not what is left — so each candidate costs one ``format=minimal`` get, however many
        records name it. A message that no longer exists is not a candidate.
        """
        documents: list[DiscoveredDoc] = []
        for change in page.changes:
            if change.kind not in _CANDIDATES or change.message_id in judged:
                continue
            judged.add(change.message_id)
            try:
                current = await self._client.get_minimal(change.message_id)
            except NotFoundError:
                continue
            if excluded.isdisjoint(current.label_ids):
                documents.append(self._discovered(change.message_id))
        return documents

    def _discovered(self, message_id: str) -> DiscoveredDoc:
        return DiscoveredDoc(
            ref=DocRef(source_id=message_id, uri=self._fallback_uri(message_id)),
            version_token=gmail_settings.REPRESENTATION_VERSION,
            media_type=MESSAGE_MEDIA_TYPE,
        )

    # --- the inventory shared by the full walk and reconciliation --------------------------

    async def _inventory(self) -> AsyncIterator[tuple[str, ...]]:
        """Every in-scope message id, a page at a time (§3).

        Lists the excluded labels into a set first, then pages the mailbox and skips them. A
        failure anywhere raises, so what a caller has seen is never mistaken for the whole.
        """
        excluded = await self._excluded_message_ids()
        async for ids in self._client.list_messages():
            kept = tuple(message_id for message_id in ids if message_id not in excluded)
            if kept:
                yield kept

    async def _excluded_message_ids(self) -> set[str]:
        """The ids of messages carrying a label that is excluded and that listing does not skip.

        SPAM and TRASH are left out of ``messages.list`` by the client, so only DRAFT and the
        configured labels need listing. Held for the length of one walk and not stored: it is
        the excluded messages, a bounded exception to never holding the inventory (§3).
        """
        found: set[str] = set()
        for label in ("DRAFT", *self._user_excluded):
            async for ids in self._client.list_messages([label]):
                found.update(ids)
        return found

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """§5: the whole RFC 5322 message, attachments inline, as ``message/rfc822``.

        Raises:
            NotFoundError: The message was deleted between discovery and fetch.
        """
        message = await self._client.get_raw(ref.source_id)
        self._fetched += 1
        metadata: Metadata = {_VERSION_KEY: gmail_settings.REPRESENTATION_VERSION}
        if message.thread_id is not None:
            metadata["gmail_thread_id"] = message.thread_id
        if message.internal_date_ms is not None:
            metadata["internal_date"] = _iso(message.internal_date_ms)
        return RawDocument(
            source_id=ref.source_id,
            uri=self._uri(ref.source_id, _message_id(message.raw)),
            media_type=MESSAGE_MEDIA_TYPE,
            content=message.raw,
            metadata=metadata,
        )

    # --- reconciliation ------------------------------------------------------------------

    async def reconcile(self) -> AsyncIterator[SourceId]:
        async for batch in self.reconcile_batches():
            for source_id in batch:
                yield source_id

    async def reconcile_batches(self) -> AsyncIterator[Sequence[SourceId]]:
        """§7: the complete in-scope inventory, built exactly as the full walk builds it.

        Starts with the mailbox pin, so a token for another mailbox is refused before an id is
        listed. Raises, rather than ending early, on any failure: only exhaustion of this
        iterator is a complete inventory.
        """
        await self._begin()
        async for ids in self._inventory():
            yield ids

    # --- links ---------------------------------------------------------------------------

    def _fallback_uri(self, message_id: str) -> str:
        """Open the message by its Gmail id; used until its ``Message-ID`` is known."""
        account = quote(self._config.account, safe="")
        return f"{_VIEW}?authuser={account}#all/{quote(message_id, safe='')}"

    def _uri(self, message_id: str, rfc822_id: str | None) -> str:
        """The ``rfc822msgid`` search link (§2); both values percent-encoded."""
        if rfc822_id is None:
            return self._fallback_uri(message_id)
        return (
            f"{_VIEW}?authuser={quote(self._config.account, safe='')}"
            f"#search/rfc822msgid:{quote(rfc822_id, safe='')}"
        )


def _message_id(raw: bytes) -> str | None:
    """The ``Message-ID`` header without its angle brackets, or ``None`` if there is none."""
    try:
        headers = email.parser.BytesHeaderParser(policy=email.policy.default).parsebytes(raw)
        value = headers.get("Message-ID")
    except (ValueError, TypeError, IndexError):
        return None
    if value is None:
        return None
    text = str(value).strip().removeprefix("<").removesuffix(">").strip()
    return text or None


def _iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()
