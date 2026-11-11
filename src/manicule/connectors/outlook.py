"""The Outlook connector: per-folder delta discovery, MIME fetch, calendar listing, reconciliation.

``docs/connectors/outlook.md`` §2-§8 is the design; :mod:`manicule.connectors.outlook_client` is the
HTTP half and :mod:`manicule.connectors.outlook_config` the settings. This module decides what is
in scope, what changed, and what is still in the mailbox.

**Mail scope is folders, and nothing else.** Graph's message delta is a per-folder operation, so
the connector walks the folder tree, drops the always-excluded folders (Junk Email, Deleted Items,
Drafts, Outbox, Sync Issues) and the configured ones with their subfolders, and keeps one delta
link per folder it walked. The watermark is that map. A folder with no link (new, or whose link
Graph no longer honors) is walked from scratch; the others resume. The full walk, the incremental
walk and reconciliation rest on the same resolved set, so the three agree on what is in scope.

**A message is fetched once.** Its bytes do not change after delivery and ids are immutable, so
every message carries one constant version token and a stored message is never fetched again by
id, however often a flag change or a move names it again. What a changed representation costs is
paid through the scope fingerprint instead (:data:`MAIL_REPRESENTATION_VERSION`).

**The calendar is listed afresh on every sync and keeps no position.** ``calendarView/delta`` fixes
its date window when a round begins and cannot select properties, so a rolling window would mean
re-baselining it constantly; a plain listing of the window is one small request per hundred events
and is always correct. An event's version token is its ``changeKey``, so an unchanged event is not
fetched again. Events leaving the window are removed by reconciliation, which is the intended
behavior of a rolling window (``outlook.md`` §7.1).

**Nothing here may name a message, an address or a folder.** Errors and log records are built from
counts and operation names; a record about one message belongs to that message's document.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, cast
from urllib.parse import quote

from pydantic import JsonValue

from manicule.connectors.errors import (
    ConnectorError,
    CursorExpiredError,
    MailboxMismatchError,
    NotFoundError,
)
from manicule.connectors.mail_headers import message_provenance, read_headers
from manicule.connectors.outlook_auth import identity_mismatch
from manicule.connectors.outlook_client import (
    EventStub,
    Folder,
    MessageStub,
    OutlookClient,
    Profile,
)
from manicule.connectors.outlook_config import (
    ALWAYS_EXCLUDED_FOLDERS,
    EVENT_REPRESENTATION_VERSION,
    MAIL_REPRESENTATION_VERSION,
    OUTLOOK_CONNECTOR_NAME,
    OutlookConfig,
    resolve_folder_ids,
)
from manicule.connectors.outlook_ics import render_event
from manicule.core.content import Metadata, RawDocument
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.mail_ids import MAIL_MESSAGE_ID_KEY, MAIL_THREAD_ID_KEY, normalize_message_id
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = [
    "EVENT_MEDIA_TYPE",
    "EVENT_PREFIX",
    "MESSAGE_MEDIA_TYPE",
    "MESSAGE_PREFIX",
    "OutlookConnector",
]

_log = logging.getLogger("manicule.connectors.outlook")

MESSAGE_MEDIA_TYPE: Final = "message/rfc822"
"""What a fetched message is handed to the pipeline as; the existing mail parser reads it."""

EVENT_MEDIA_TYPE: Final = "text/calendar"
"""What a rendered event is handed to the pipeline as; the existing calendar parser reads it."""

MESSAGE_PREFIX: Final = "msg:"
EVENT_PREFIX: Final = "evt:"
"""Source ids are Graph's immutable ids behind a kind, so a message and an event can never be the
same document. ``!/`` is reserved by the pipeline for attachments and cannot occur in either."""

_FINGERPRINT_KEY: Final = "scope_fingerprint"
_IDENTITY_KEY: Final = "pinned_identity"
_VERSION_KEY: Final = "version_token"

_TREE_FRESH_SECONDS: Final = 60.0
"""How long the folder tree read at setup stays good for the first run that follows it."""

_OBJECT_KEY: Final = "object_identity"

_RECHECK_SECONDS: Final = 60.0
"""How long a confirmation that the credential reads the configured mailbox stays fresh."""

_OWA_ITEM: Final = (
    "https://outlook.office365.com/owa/?ItemID={id}&exvsurl=1&viewmodel=ReadMessageItem"
)
_OWA_EVENT: Final = "https://outlook.office365.com/owa/?itemid={id}&exvsurl=1&path=/calendar/item"
"""Where a person opens an item when Graph gave no ``webLink``. Used only as a fallback."""


@dataclass(frozen=True, slots=True)
class _Run:
    """What one walk was started with, held by the walk and not by the connector.

    The connector is one object for the life of the process and a discovery and a reconciliation
    can overlap, so the folders a walk covers are its own: a second run resolving the tree afresh
    must not change what a walk already under way tests.
    """

    folders: tuple[Folder, ...]
    """The in-scope mail folders, parents before children. Empty when mail is off."""
    excluded_user_ids: tuple[str, ...]
    """The resolved ids of ``exclude_folders``; part of the scope fingerprint."""


class OutlookConnector:
    """Discovers, fetches and reconciles the messages and events of one Microsoft 365 mailbox."""

    def __init__(
        self,
        config: OutlookConfig,
        client: OutlookClient,
        *,
        name: str = OUTLOOK_CONNECTOR_NAME,
        renewal: str = "",
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        max_message_bytes: int | None = None,
        unstored_rotations: Callable[[], int] | None = None,
    ) -> None:
        """Args:
        config: The validated Outlook settings.
        client: The Graph client; this connector sets it up and tears it down.
        name: The configured instance's name, which files its documents.
        renewal: What a person does about a token for the wrong mailbox, appended to that refusal.
        clock: A monotonic clock, for how long a mailbox check stays fresh.
        now: The wall clock, for the calendar window.
        max_message_bytes: ``ingest.max_fetch_bytes``. A message is downloaded as a stream and
            abandoned once it passes this, so one that the pipeline would refuse is never held
            whole in memory first.
        unstored_rotations: How many times the credential's rotated refresh token went nowhere
            durable, reported as a metric (``outlook.md`` §1.2).
        """
        self.name = name
        self._config = config
        self._client = client
        self._renewal = renewal
        self._clock = clock
        self._now = now
        self._max_message_bytes = max_message_bytes
        self._unstored_rotations = unstored_rotations
        self._object: str | None = None
        """Digest of the directory object ``/me`` reported, held for the life of the process."""
        self._tree: tuple[float, _Run] | None = None
        """The folder tree the last ``_begin`` read, and when, so setup's read is not repeated."""
        self._user_excluded: tuple[str, ...] = ()
        self._resolved = not config.exclude_folders
        self._checked_at: float | None = None
        self._links: dict[str, str] | None = None
        self._enumerated = False
        self._walked_full = False
        self._fetched = 0

    # --- identity and scope --------------------------------------------------------------

    @property
    def pinned_identity(self) -> str:
        """The digest of the directory and mailbox this instance indexes."""
        return self._config.pinned_identity

    @property
    def source_scope(self) -> str:
        """A non-secret name for this scope: the identity's digest, never an address or a folder."""
        return f"outlook:{self._config.pinned_identity[:16]}"

    @property
    def reconciliation_scope(self) -> str:
        """Fences a completed inventory, and a held deletion proposal, to this exact scope.

        **It includes the calendar window, which the mail-delta fingerprint deliberately does
        not** (``outlook.md`` §7.1): the calendar keeps no position, so its window is no part of
        what a stored delta link is valid for. But an inventory is the events *in the window*, so
        a deletion proposed under a narrow window must not be confirmable after the window is
        widened, which would delete events that are now in scope. Independent of
        :attr:`scope_fingerprint`, so widening the window re-fences deletion without discarding
        a single delta link.
        """
        window = (
            f"{self._config.calendar_days_back}:{self._config.calendar_days_forward}"
            if self._config.calendar
            else "-"
        )
        return f"outlook:{self.scope_fingerprint}:{window}"

    @property
    def enumerates_full_inventory(self) -> bool:
        """Whether discovery walks the whole scope whatever watermark it is handed.

        True only for a calendar-only instance, which lists its window on every sync. A mail
        instance resumes from delta links and is incremental.
        """
        return not self._config.mail

    @property
    def walked_full_inventory(self) -> bool:
        """Whether the walk that last completed listed the whole scope, however it began.

        An incremental sync can find a folder's delta state gone and walk that folder from
        scratch, but only a walk in which *every* folder started from scratch is a full
        inventory; the pipeline asks before discovery and needs this answer after it to know a
        reconciliation should follow (``outlook.md`` §6).
        """
        return self._enumerated and self._walked_full

    @property
    def scope_fingerprint(self) -> str:
        """Cursor identity: the mailbox, the resolved excluded folder ids and the representation.

        A synchronous property over what the connector resolved at setup or at the start of the
        last run, because the pipeline reads it before discovery begins. The watermark carries
        the fingerprint it was recorded under and :meth:`_resume_links` compares it with a freshly
        resolved one, so a stale read costs nothing but a walk.
        """
        if not self._resolved:
            msg = "the Outlook connector's setup() was not awaited before its scope was read"
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
        """One ``GET /me``, which is also the mailbox check."""
        try:
            await self._check_mailbox()
        except ConnectorError as exc:
            return HealthReport.failing(
                f"Microsoft Graph did not accept the credential: {exc}",
                remedy="Run `manicule doctor` and see the outlook-credential check.",
            )
        return HealthReport.healthy("Graph answered and the credential reads the mailbox")

    def metrics(self) -> tuple[Metric, ...]:
        return (
            Metric(name="outlook_requests", value=float(self._client.requests)),
            Metric(name="outlook_throttled", value=float(self._client.throttled)),
            Metric(name="outlook_items_fetched", value=float(self._fetched)),
            Metric(
                name="outlook_token_rotations_unstored",
                value=float(self._unstored_rotations() if self._unstored_rotations else 0),
            ),
        )

    async def _check_mailbox(self) -> Profile:
        """Read ``/me``; refuse a token for another mailbox, or another account at the address.

        Two checks, one message. The address check is configuration's (``identity_mismatch``). The
        object check is the directory's own: the first object id this process sees for the
        instance is held, and a later different one is refused, as is one that differs from the
        object the stored watermark was recorded under (:meth:`_refuse_other_object`). An address
        can be reassigned to a different account (a departed colleague's, say); the object id
        cannot. Neither value, nor either digest, is in the message.
        """
        profile = await self._client.get_me()
        mismatch = identity_mismatch(profile.addresses, self._config.account, self._renewal)
        if mismatch is None:
            digest = self._object_digest(profile.object_id)
            if self._object is not None and self._object != digest:
                mismatch = _other_object(self._renewal)
            elif self._object is None:
                self._object = digest
        if mismatch is not None:
            self._checked_at = None
            raise mismatch
        self._checked_at = self._clock()
        return profile

    def _object_digest(self, object_id: str) -> str:
        """A digest of the object id and the tenant; the id itself is never stored or printed."""
        return hashlib.blake2b(
            f"outlook-object:{self._config.tenant_id}:{object_id.casefold()}".encode(),
            digest_size=20,
        ).hexdigest()

    def _refuse_other_object(self, watermark: Watermark | None) -> None:
        """Refuse a stored watermark recorded for a different directory object.

        A watermark from before the object was recorded carries none and is accepted: the
        config-only pin (tenant and account) is what holds until the first completed sync.
        """
        if watermark is None or self._object is None:
            return
        recorded = watermark.metadata.get(_OBJECT_KEY)
        if isinstance(recorded, str) and recorded != self._object:
            raise _other_object(self._renewal)

    async def _recheck_mailbox(self) -> None:
        """Re-confirm the mailbox before a fetch if the last confirmation is not recent.

        A run that resumes at acquisition never goes through discovery, so neither ``setup()`` nor
        the check at the start of a run has run for it, and the connector is cached for the life
        of the process. One ``GET /me`` at most once a minute is cheap against a MIME fetch.
        """
        if self._checked_at is None or self._clock() - self._checked_at >= _RECHECK_SECONDS:
            await self._check_mailbox()

    async def _begin(self) -> _Run:
        """Start of a run: the mailbox pin, then the folder tree.

        The tree read at setup is reused by the first run that starts within
        :data:`_TREE_FRESH_SECONDS`, so a sync does not read it twice back to back; a later run
        reads it afresh, because a folder created or deleted since is exactly what scope depends
        on. The mailbox check is never skipped.

        Raises:
            MailboxMismatchError: The token reads another mailbox. Nothing has been read.
            ConfigError: An ``exclude_folders`` path is not a folder of this mailbox.
        """
        await self._check_mailbox()
        if not self._config.mail:
            self._user_excluded = ()
            self._resolved = True
            return _Run(folders=(), excluded_user_ids=())
        held = self._tree
        if held is not None and self._clock() - held[0] < _TREE_FRESH_SECONDS:
            self._tree = None  # consumed: a second run in the window reads its own
            return held[1]
        folders = [folder async for folder in self._client.list_folders()]
        user_ids = resolve_folder_ids(
            self._config.exclude_folders, {folder.path: folder.id for folder in folders}
        )
        excluded = set(user_ids)
        # Concurrently, within the client's own cap of four in flight: five small lookups that
        # would otherwise be five round trips on every run.
        found = await asyncio.gather(
            *(self._client.well_known_folder_id(name) for name in ALWAYS_EXCLUDED_FOLDERS)
        )
        excluded.update(folder_id for folder_id in found if folder_id is not None)
        self._user_excluded = user_ids
        self._resolved = True
        run = _Run(folders=_in_scope(folders, excluded), excluded_user_ids=user_ids)
        self._tree = (self._clock(), run)
        return run

    # --- watermark -----------------------------------------------------------------------

    @property
    def watermark(self) -> Watermark | None:
        """The delta links to resume from, once a walk has completed; otherwise ``None``.

        Carries the scope fingerprint and the pinned identity it was recorded under, so a
        position is never resumed under a different scope or mailbox even by a caller that does
        not compare them itself.
        """
        if not self._enumerated or self._links is None:
            return None
        return Watermark(
            value=json.dumps({"mail": self._links}, sort_keys=True, separators=(",", ":")),
            observed_at=datetime.now(tz=UTC),
            metadata={
                _FINGERPRINT_KEY: self.scope_fingerprint,
                _IDENTITY_KEY: self.pinned_identity,
                **({_OBJECT_KEY: self._object} if self._object is not None else {}),
            },
        )

    def _resume_links(self, watermark: Watermark | None) -> dict[str, str]:
        """The stored per-folder delta links, if recorded under this scope and mailbox."""
        if watermark is None:
            return {}
        recorded = watermark.metadata
        if recorded.get(_IDENTITY_KEY) != self.pinned_identity:
            return {}
        if recorded.get(_FINGERPRINT_KEY) != self.scope_fingerprint:
            return {}
        return _parse_links(watermark.value)

    # --- discovery -----------------------------------------------------------------------

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        async for batch in self.discover_batches(watermark):
            for document in batch:
                yield document

    async def discover_batches(
        self, watermark: Watermark | None
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """One batch per delta page, then one per calendar page (§4.4)."""
        self._enumerated = False
        self._walked_full = False
        self._links = None
        run = await self._begin()
        self._refuse_other_object(watermark)
        stored = self._resume_links(watermark)
        links: dict[str, str] = {}
        from_scratch = True
        for folder in run.folders:
            link = stored.get(folder.id)
            try:
                async for batch in self._walk_folder(folder, link, links):
                    yield batch
            except CursorExpiredError:
                if link is None:
                    # Paging state that expired during a walk from scratch is not a missing
                    # folder, and skipping the rest of it would commit a partial inventory as
                    # complete. Only a first request answered "no such folder" is a vanished
                    # folder (:class:`NotFoundError`, below).
                    raise
                _log.warning(
                    "source %r: Graph no longer holds a folder's delta state; walking it again",
                    self.name,
                )
                # A fallback that completes is a walk from scratch, so it leaves the aggregate
                # flag alone: only a folder that *resumed* makes the run incremental.
                async for batch in self._walk_new(folder, links):
                    yield batch
            except NotFoundError:
                self._skip_vanished()
            else:
                from_scratch = from_scratch and link is None
        if self._config.calendar:
            async for events in self._client.list_events(*self._window()):
                batch = [self._discovered_event(stub) for stub in events]
                if batch:
                    yield batch
        self._links = links
        self._walked_full = from_scratch
        self._enumerated = True

    async def _walk_new(
        self, folder: Folder, links: dict[str, str]
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """A folder walked from scratch after its stored state was lost.

        The stored link's 404 was ambiguous (state gone, or folder gone), so a first request
        that is itself answered "no such folder" is the folder having vanished.
        """
        try:
            async for batch in self._walk_folder(folder, None, links):
                yield batch
        except NotFoundError:
            self._skip_vanished()

    def _skip_vanished(self) -> None:
        """A folder deleted between the tree read and its walk; reconciliation removes its mail."""
        _log.warning("source %r: a folder disappeared during the walk; skipped", self.name)

    async def _walk_folder(
        self, folder: Folder, link: str | None, links: dict[str, str]
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """One folder's delta round; records the link it ends on in ``links``."""
        async for page in self._client.message_delta(folder.id, link):
            if page.messages:
                yield [self._discovered_message(stub) for stub in page.messages]
            if page.delta_link is not None:
                links[folder.id] = page.delta_link

    def _discovered_message(self, stub: MessageStub) -> DiscoveredDoc:
        metadata: dict[str, JsonValue] = {}
        if stub.web_link:
            metadata["web_link"] = stub.web_link
        if stub.conversation_id:
            metadata["conversation_id"] = stub.conversation_id
        if stub.received:
            metadata["received"] = stub.received
        return DiscoveredDoc(
            ref=DocRef(
                source_id=MESSAGE_PREFIX + stub.message_id,
                uri=stub.web_link or _OWA_ITEM.format(id=quote(stub.message_id, safe="")),
                metadata=metadata,
            ),
            version_token=MAIL_REPRESENTATION_VERSION,
            media_type=MESSAGE_MEDIA_TYPE,
        )

    @staticmethod
    def fetched_revision_at_least(discovered: str, fetched: str) -> bool:
        """Whether fetched bytes are not older than the revision discovery listed.

        An event's ``changeKey`` is opaque and cannot be ordered, and the pipeline's default
        (equality) would reject an event **edited between discovery and fetch** as a stale body,
        so the next sync would retry the same enumeration forever (``outlook.md`` §7.1). A fetch
        happens after discovery, and a key only moves when the event is edited, so a different key
        of the same representation is the newer revision and is accepted: its token is what gets
        stored, so the next listing, at that same key, fetches nothing. If the key were somehow
        older (a lagging replica), the next listing differs from the stored one and fetches again.
        A different representation is not accepted, and a message's token is constant.
        """
        if discovered == fetched:
            return True
        _, _, wanted = discovered.partition("|")
        _, _, got = fetched.partition("|")
        return bool(wanted) and wanted == got

    def _discovered_event(self, stub: EventStub) -> DiscoveredDoc:
        return DiscoveredDoc(
            ref=DocRef(
                source_id=EVENT_PREFIX + stub.event_id,
                uri=stub.web_link or _OWA_EVENT.format(id=quote(stub.event_id, safe="")),
            ),
            version_token=f"{stub.change_key}|{EVENT_REPRESENTATION_VERSION}",
            media_type=EVENT_MEDIA_TYPE,
        )

    def _window(self) -> tuple[datetime, datetime]:
        """The calendar window around now, read once per walk (§7.1)."""
        now = self._now()
        return (
            now - timedelta(days=self._config.calendar_days_back),
            now + timedelta(days=self._config.calendar_days_forward),
        )

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """§5, §7.2: a message as its whole MIME, or an event rendered as iCalendar.

        Raises:
            NotFoundError: The item was deleted between discovery and fetch.
        """
        await self._recheck_mailbox()
        if ref.source_id.startswith(MESSAGE_PREFIX):
            return await self._fetch_message(ref)
        if ref.source_id.startswith(EVENT_PREFIX):
            return await self._fetch_event(ref)
        msg = "the Outlook connector was asked for a source id it did not issue"
        raise NotFoundError(msg)

    async def _fetch_message(self, ref: DocRef) -> RawDocument:
        graph_id = ref.source_id.removeprefix(MESSAGE_PREFIX)
        raw = await self._client.get_message_mime(graph_id, max_bytes=self._max_message_bytes)
        self._fetched += 1
        meta = ref.metadata
        web_link = _meta_text(meta, "web_link")
        conversation = _meta_text(meta, "conversation_id")
        received = _meta_text(meta, "received")
        if not (web_link and conversation and received):
            # A discovery that did not carry them (an older one, or a resumed run): ask once.
            stub = await self._client.get_message_stub(graph_id)
            web_link = web_link or stub.web_link or ""
            conversation = conversation or stub.conversation_id or ""
            received = received or stub.received or ""
        arrived = _parse_time(received)
        subject, internet_id = read_headers(raw)
        metadata: Metadata = {_VERSION_KEY: MAIL_REPRESENTATION_VERSION}
        if conversation:
            metadata[MAIL_THREAD_ID_KEY] = conversation
        normalized = normalize_message_id(internet_id) if internet_id else None
        if normalized is not None:
            metadata[MAIL_MESSAGE_ID_KEY] = normalized
        if arrived is not None:
            metadata["internal_date"] = arrived.isoformat()
        metadata[PROVENANCE_KEY] = message_provenance(
            source_id=ref.source_id,
            subject=subject,
            link=web_link or ref.uri,
            version=MAIL_REPRESENTATION_VERSION,
            media_type=MESSAGE_MEDIA_TYPE,
            arrived=arrived,
        )
        return RawDocument(
            source_id=ref.source_id,
            # The address discovery gave, so a retained fetch stays reusable while its envelope
            # names the same address as the record asking for it.
            uri=ref.uri,
            media_type=MESSAGE_MEDIA_TYPE,
            content=raw,
            metadata=metadata,
        )

    async def _fetch_event(self, ref: DocRef) -> RawDocument:
        detail = await self._client.get_event(ref.source_id.removeprefix(EVENT_PREFIX))
        self._fetched += 1
        version = f"{detail.change_key}|{EVENT_REPRESENTATION_VERSION}"
        subject = detail.fields.get("subject")
        link = detail.web_link or ref.uri
        created = detail.fields.get("createdDateTime")
        metadata: Metadata = {_VERSION_KEY: version}
        metadata[PROVENANCE_KEY] = _event_provenance(
            ref.source_id,
            subject if isinstance(subject, str) else "",
            link,
            version,
            _parse_time(created if isinstance(created, str) else ""),
        )
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=EVENT_MEDIA_TYPE,
            content=render_event(detail.fields),
            metadata=metadata,
        )

    # --- reconciliation ------------------------------------------------------------------

    async def reconcile(self) -> AsyncIterator[SourceId]:
        async for batch in self.reconcile_batches():
            for source_id in batch:
                yield source_id

    async def reconcile_batches(self) -> AsyncIterator[Sequence[SourceId]]:
        """§6: the complete in-scope inventory, built from the same folders the walk covers.

        Starts with the mailbox pin, so a token for another mailbox is refused before an id is
        listed. Raises, rather than ending early, on any failure: only exhaustion of this iterator
        is a complete inventory.
        """
        run = await self._begin()
        for folder in run.folders:
            async for ids in self._client.list_message_ids(folder.id):
                if ids:
                    yield [MESSAGE_PREFIX + message_id for message_id in ids]
        if self._config.calendar:
            async for events in self._client.list_events(*self._window()):
                if events:
                    yield [EVENT_PREFIX + stub.event_id for stub in events]


def _other_object(renewal: str) -> MailboxMismatchError:
    """The refusal for an address that now belongs to a different directory object."""
    return MailboxMismatchError(
        (
            "the account at the configured address is not the one this instance synced as: the "
            "directory object behind it changed. Indexing another account is a different "
            "instance under a new name, so the run did not begin. " + renewal
        ).rstrip()
    )


def _parse_links(value: str) -> dict[str, str]:
    """The folder-to-link map a watermark carries; anything that is not one is no links."""
    try:
        parsed: object = json.loads(value)
    except ValueError:
        return {}
    mail: object = (
        cast("dict[str, object]", parsed).get("mail") if isinstance(parsed, dict) else None
    )
    if not isinstance(mail, dict):
        return {}
    return {
        folder: link
        for folder, link in cast("dict[object, object]", mail).items()
        if isinstance(folder, str) and isinstance(link, str) and link
    }


def _in_scope(folders: Sequence[Folder], excluded: set[str]) -> tuple[Folder, ...]:
    """The folders that are neither excluded nor below an excluded one, in tree order."""
    by_id = {folder.id: folder for folder in folders}

    def below_excluded(folder: Folder) -> bool:
        seen: set[str] = set()
        current: Folder | None = folder
        while current is not None and current.id not in seen:
            if current.id in excluded:
                return True
            seen.add(current.id)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        return False

    return tuple(folder for folder in folders if not below_excluded(folder))


def _meta_text(metadata: dict[str, JsonValue], key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def _parse_time(value: str) -> datetime | None:
    """Graph's ISO 8601 instant (``2026-02-03T09:14:00Z``, sometimes with fractions) as UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _event_provenance(
    source_id: str, subject: str, link: str, version: str, created: datetime | None
) -> JsonValue:
    """What a citation says about an event: its subject, its link and its change key."""
    for title in (" ".join(subject.split()), ""):
        try:
            source = SourceMetadata(
                title=title,
                canonical_uri=link,
                source_id=source_id,
                version=version,
                created_at=created,
                content_type=EVENT_MEDIA_TYPE,
            )
        except ValueError:
            continue
        return Provenance(source=source).as_metadata_value()
    return Provenance(
        unavailable_reason="the event could not be described by a title and a link"
    ).as_metadata_value()
