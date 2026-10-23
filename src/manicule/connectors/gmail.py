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

import email.errors
import email.parser
import email.policy
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from urllib.parse import quote

from pydantic import JsonValue

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
from manicule.core.acquisition import MAX_DISCOVERY_BATCH
from manicule.core.content import Metadata, RawDocument
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = [
    "MESSAGE_MEDIA_TYPE",
    "GmailConnector",
]

_log = logging.getLogger("manicule.connectors.gmail")


def _fitted(page: list[DiscoveredDoc]) -> list[list[DiscoveredDoc]]:
    """One listing page as durable discovery batches, none larger than the journal admits.

    A ``messages.list`` page holds up to 500 ids and a ``history.list`` page can name more
    messages than it has records, while one batch is at most :data:`MAX_DISCOVERY_BATCH`. The
    batches of a page are all yielded, and so all committed, before the next page is requested,
    so the cursor is followed only once its whole page is durable — the property one batch per
    page exists for. An empty page yields nothing.
    """
    return [
        page[start : start + MAX_DISCOVERY_BATCH]
        for start in range(0, len(page), MAX_DISCOVERY_BATCH)
    ]


MESSAGE_MEDIA_TYPE: Final = "message/rfc822"
"""What a fetched message is handed to the pipeline as; the existing mail parser reads it."""

_VIEW: Final = "https://mail.google.com/mail/"

_FINGERPRINT_KEY: Final = "scope_fingerprint"
_IDENTITY_KEY: Final = "pinned_identity"
_VERSION_KEY: Final = "version_token"

_RECHECK_SECONDS: Final = 60.0
"""How long a confirmation that the credential reads the configured mailbox stays fresh."""

_CANDIDATES: Final = frozenset({"messageAdded", "labelRemoved"})
"""History records that make a message a candidate (§4.2). ``labelAdded`` and ``messageDeleted``
can only take a message out of scope, and that is reconciliation's to act on."""


@dataclass(frozen=True, slots=True)
class _Run:
    """What one walk was started with, held by the walk and not by the connector.

    The connector is one object for the life of the process and a discovery and a reconciliation
    can overlap, so the label ids a walk skips and the position it captured are its own: a
    second run resolving the labels afresh must not change what a walk already under way tests.
    """

    history_id: str
    """The mailbox's position, read before anything is listed (§4.1)."""
    excluded: tuple[str, ...]
    """The resolved ids of ``exclude_labels``."""

    @property
    def excluded_labels(self) -> frozenset[str]:
        """Every label id that takes a message out of scope; system labels' ids are their names."""
        return frozenset((*ALWAYS_EXCLUDED_LABELS, *self.excluded))


class GmailConnector:
    """Discovers, fetches and reconciles the messages of one Gmail mailbox."""

    def __init__(
        self,
        config: GmailConfig,
        client: GmailClient,
        *,
        name: str = GMAIL_CONNECTOR_NAME,
        renewal: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args:
        config: The validated Gmail settings.
        client: The REST client; this connector sets it up and tears it down.
        name: The configured instance's name, which files its documents (see
            ``plugin._source_name``).
        renewal: What a person does about a token for the wrong mailbox, appended to that
            refusal. The credential knows the sentence; the connector only repeats it.
        clock: A monotonic clock, for how long a mailbox check stays fresh (see :meth:`fetch`).
        """
        self.name = name
        self._config = config
        self._client = client
        self._renewal = renewal
        self._user_excluded: tuple[str, ...] = ()
        """Label ids of ``exclude_labels``, resolved at setup and again at the start of a run."""
        self._resolved = not config.exclude_labels
        self._clock = clock
        self._checked_at: float | None = None
        self._next: str | None = None
        self._enumerated = False
        self._walked_full = False
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
    def reconciliation_scope(self) -> str:
        """Fences a completed inventory, and a held deletion proposal, to this exact scope.

        Without it a proposal made while labels were excluded could be confirmed after the
        exclusion was removed, deleting messages that are in scope again, and an inventory
        awaiting application could be applied under a scope it was not built for.
        """
        return f"gmail:{self.scope_fingerprint}"

    @property
    def walked_full_inventory(self) -> bool:
        """Whether the walk that last completed listed the whole scope, however it began.

        An incremental sync resumes from a history position and can find it gone, or recorded
        under another scope, and then walks everything; the pipeline asked before discovery and
        needs this answer after it to know a reconciliation should follow (``gmail.md`` §7).
        """
        return self._enumerated and self._walked_full

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
            self._checked_at = None
            raise mismatch
        self._checked_at = self._clock()
        return profile

    async def _recheck_mailbox(self) -> None:
        """Re-confirm the mailbox before a fetch if the last confirmation is not recent.

        A run that resumes at acquisition (a fetch failed after enumeration finished) never goes
        through discovery, so neither ``setup()`` nor the check at the start of a run has run
        for it, and the connector is cached for the life of the process. The check is one
        ``getProfile`` (1 quota unit) at most once a minute, so it costs nothing against 60 for
        a raw fetch, and a refusal comes before the message is read.
        """
        if self._checked_at is None or self._clock() - self._checked_at >= _RECHECK_SECONDS:
            await self._check_mailbox()

    async def _begin(self) -> _Run:
        """Start of a run: the mailbox pin, then the excluded label names as ids, afresh.

        The profile's ``historyId`` is returned because a complete walk needs it captured
        *before* it lists anything (§4.1), and this is before.

        Raises:
            MailboxMismatchError: The token reads another mailbox. Nothing has been read.
            ConfigError: An ``exclude_labels`` name is not a label of this mailbox.
        """
        profile = await self._check_mailbox()
        excluded: tuple[str, ...] = ()
        if self._config.exclude_labels:
            labels = await self._client.list_labels()
            excluded = resolve_label_ids(
                self._config.exclude_labels, {label.name: label.id for label in labels}
            )
        self._user_excluded = excluded
        self._resolved = True
        return _Run(history_id=profile.history_id, excluded=excluded)

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
        """One batch per ``messages.list`` or ``history.list`` page, split to fit (§4.4)."""
        self._enumerated = False
        self._walked_full = False
        self._next = None
        run = await self._begin()
        position = self._resume_position(watermark)
        if position is not None:
            try:
                async for batch in self._incremental(position, run):
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
                self._walked_full = False
            else:
                return
        async for batch in self._full(run):
            yield batch

    async def _full(self, run: _Run) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """§4.1: the run's ``historyId`` was read before the first listing and is the watermark."""
        async for ids in self._inventory(run):
            for batch in _fitted([self._discovered(message_id) for message_id in ids]):
                yield batch
        self._next = run.history_id
        self._walked_full = True
        self._enumerated = True

    async def _incremental(
        self, position: str, run: _Run
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """§4.2: candidates from the history log, each judged by its current label ids."""
        excluded = run.excluded_labels
        judged: set[str] = set()
        """Candidates already looked up this walk; ids of messages the log named, not the
        mailbox's inventory, and gone when the walk is."""
        following: str | None = None
        async for page in self._client.history(position):
            if following is None:
                following = page.history_id
            for batch in _fitted(await self._judge(page, judged, excluded)):
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

    async def _inventory(self, run: _Run) -> AsyncIterator[tuple[str, ...]]:
        """Every in-scope message id, a page at a time (§3).

        Lists the excluded labels into a set first, then pages the mailbox and skips them. A
        failure anywhere raises, so what a caller has seen is never mistaken for the whole.
        """
        excluded = await self._excluded_message_ids(run)
        async for ids in self._client.list_messages():
            kept = tuple(message_id for message_id in ids if message_id not in excluded)
            if kept:
                yield kept

    async def _excluded_message_ids(self, run: _Run) -> set[str]:
        """The ids of messages carrying a label that is excluded and that listing does not skip.

        SPAM and TRASH are left out of ``messages.list`` by the client, so only DRAFT and the
        configured labels need listing. Held for the length of one walk and not stored: it is
        the excluded messages, a bounded exception to never holding the inventory (§3).
        """
        found: set[str] = set()
        for label in ("DRAFT", *run.excluded):
            async for ids in self._client.list_messages([label]):
                found.update(ids)
        return found

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """§5: the whole RFC 5322 message, attachments inline, as ``message/rfc822``.

        Raises:
            NotFoundError: The message was deleted between discovery and fetch.
        """
        await self._recheck_mailbox()
        message = await self._client.get_raw(ref.source_id)
        self._fetched += 1
        metadata: Metadata = {_VERSION_KEY: gmail_settings.REPRESENTATION_VERSION}
        if message.thread_id is not None:
            metadata["gmail_thread_id"] = message.thread_id
        arrived = None
        if message.internal_date_ms is not None:
            arrived = _arrival(message.internal_date_ms)
            metadata["internal_date"] = arrived.isoformat()
        subject, rfc822_id = _headers(message.raw)
        link = self._link(ref.source_id, rfc822_id)
        metadata[PROVENANCE_KEY] = self._provenance(ref.source_id, subject, link, arrived)
        return RawDocument(
            source_id=ref.source_id,
            # The address discovery gave, not the search link: a retained fetch is reusable only
            # while its envelope names the same address as the record asking for it, and the
            # link is not known until the headers are read. The pipeline titles and addresses
            # the document from the provenance record below.
            uri=ref.uri,
            media_type=MESSAGE_MEDIA_TYPE,
            content=message.raw,
            metadata=metadata,
        )

    @staticmethod
    def _provenance(
        message_id: str, subject: str, link: str, arrived: datetime | None
    ) -> JsonValue:
        """What a citation says about the message: its subject, its link, when it arrived.

        A subject Gmail delivers with a control character cannot be a title. The message is
        still cited by its link, and the record keeps that link; only if the link too is refused
        does it say, in words that do not quote the subject, that it could not be described.
        """
        for title in (subject, ""):
            try:
                source = SourceMetadata(
                    title=title,
                    canonical_uri=link,
                    source_id=message_id,
                    version=gmail_settings.REPRESENTATION_VERSION,
                    created_at=arrived,
                    content_type=MESSAGE_MEDIA_TYPE,
                )
            except ValueError:
                continue
            return Provenance(source=source).as_metadata_value()
        return Provenance(
            unavailable_reason="the message could not be described by a title and a link"
        ).as_metadata_value()

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
        run = await self._begin()
        async for ids in self._inventory(run):
            yield ids

    # --- links ---------------------------------------------------------------------------

    def _fallback_uri(self, message_id: str) -> str:
        """Open the message by its Gmail id; used until its ``Message-ID`` is known."""
        account = quote(self._config.account, safe="")
        return f"{_VIEW}?authuser={account}#all/{quote(message_id, safe='')}"

    def _link(self, message_id: str, rfc822_id: str | None) -> str:
        """The link a person opens (§2): the ``rfc822msgid`` search, both values percent-encoded."""
        if rfc822_id is None:
            return self._fallback_uri(message_id)
        return (
            f"{_VIEW}?authuser={quote(self._config.account, safe='')}"
            f"#search/rfc822msgid:{quote(rfc822_id, safe='')}"
        )


_HEADER_END: Final = re.compile(rb"\r?\n\r?\n")
_HEADER_LIMIT: Final = 1 << 20
"""Headers are read from the front of the message and never past this, so a 25 MB message with a
huge attachment costs a header's worth of parsing, not a second copy of the bytes."""


def _headers(raw: bytes) -> tuple[str, str | None]:
    """The ``Subject`` and the ``Message-ID`` (without its angle brackets) of a message.

    Both are read from the headers only and neither is ever logged. An unreadable or absent
    header is an empty subject or no id, never an error: the message is still indexed.
    """
    end = _HEADER_END.search(raw, 0, _HEADER_LIMIT)
    head = raw[: end.start()] if end is not None else raw[:_HEADER_LIMIT]
    try:
        headers = email.parser.BytesHeaderParser(policy=email.policy.default).parsebytes(head)
        subject = headers.get("Subject")
        message_id = headers.get("Message-ID")
        subject_text = " ".join(str(subject).split()) if subject is not None else ""
        cleaned = (
            str(message_id).strip().removeprefix("<").removesuffix(">").strip()
            if message_id is not None
            else ""
        )
    except (ValueError, TypeError, IndexError, email.errors.MessageError):
        return "", None
    return subject_text, cleaned or None


def _arrival(milliseconds: int) -> datetime:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
