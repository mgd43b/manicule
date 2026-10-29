"""The mbox connector: old mail archives, one ``message/rfc822`` document per message.

``docs/connectors/mbox.md`` is the design; :mod:`manicule.connectors.mbox_reader` is the part that
cuts a file into messages and identifies them. This module walks a root for mbox files, decides
which files to read, and answers the connector contract.

**A file is the unit of change, a message is the unit of identity.** A message's ``source_id`` is
the digest of its bytes (§4), so it never changes and every message carries one constant version
token. What changes is which messages exist, which is a property of files, so the watermark is a
map of the files the last complete walk read, each with the size and modification time recorded
*before* it was opened (§5.1). An unchanged file costs a ``stat``.

**Nothing here may name a message, an address or a subject.** Errors and log records are built from
counts and operation names; the discovery journal carries a digest, a path and two integers, and
whatever can name a person stays in the document it belongs to.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import sqlite3
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email import policy
from email.errors import MessageError
from email.parser import BytesHeaderParser
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue

from manicule.connectors import globs, mbox_reader
from manicule.connectors.config import MBOX_CONNECTOR_NAME, MBOX_REPRESENTATION_VERSION, MboxConfig
from manicule.connectors.errors import ConnectorError, NotFoundError
from manicule.connectors.filesystem import IGNORED_DIRECTORIES
from manicule.core.acquisition import MAX_DISCOVERY_BATCH
from manicule.core.content import Metadata, RawDocument
from manicule.core.mail_ids import MAIL_MESSAGE_ID_KEY, normalize_message_id
from manicule.core.protocols import MailMessageIdIndex
from manicule.core.provenance import PROVENANCE_KEY, LocalSnapshot, Provenance, SourceMetadata
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

if TYPE_CHECKING:
    from manicule.core.protocols import DocStore

__all__ = [
    "MESSAGE_MEDIA_TYPE",
    "MboxConnector",
]

_log = logging.getLogger("manicule.connectors.mbox")

MESSAGE_MEDIA_TYPE: Final = "message/rfc822"
"""What a message is delivered as; the existing mail parser reads it."""

_FILES_KEY: Final = "files"
_FINGERPRINT_KEY: Final = "scope_fingerprint"
_VERSION_KEY: Final = "version_token"
_PATH: Final = "path"
_OFFSET: Final = "offset"
_LENGTH: Final = "length"
_THUMBPRINT_BYTES: Final = 16

SETTLE_SECONDS: Final = 60.0
"""How long a file must be quiet before discovery reads it (``mbox.md`` §5.1)."""

_SIDE_FILES: Final = frozenset({".msf"})
"""Thunderbird's summary files, skipped by name so they are not opened (``mbox.md`` §2)."""

_HEADERS_ONLY: Final = 1 << 20

_LOAD_BATCH: Final = 1_000
"""How many known ids are written to the temporary table at once."""


class _KeySet:
    """A set of byte strings kept in a temporary SQLite file rather than in memory.

    One belongs to one walk and is removed when the walk ends (``mbox.md`` §5.3): the connector
    is a single object for the life of the process, and a discovery and a reconciliation can
    overlap. The working set is a batch, however many messages the archive holds.
    """

    def __init__(self, directory: Path | None) -> None:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed with the set
            dir=directory, prefix="mbox-keys-", suffix=".db"
        )
        self._file = handle
        self._db = sqlite3.connect(handle.name, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=OFF")
        self._db.execute("PRAGMA synchronous=OFF")
        self._db.execute("CREATE TABLE keys (k BLOB PRIMARY KEY) WITHOUT ROWID")
        self._db.execute("BEGIN")

    def add(self, key: bytes) -> bool:
        """Insert ``key``; ``True`` if it was not there."""
        return self._db.execute("INSERT OR IGNORE INTO keys VALUES (?)", (key,)).rowcount == 1

    def add_all(self, keys: Sequence[bytes]) -> None:
        self._db.executemany("INSERT OR IGNORE INTO keys VALUES (?)", [(key,) for key in keys])

    def present(self, keys: Sequence[bytes]) -> set[bytes]:
        found: set[bytes] = set()
        for key in keys:
            if self._db.execute("SELECT 1 FROM keys WHERE k = ?", (key,)).fetchone() is not None:
                found.add(key)
        return found

    def close(self) -> None:
        with suppress(sqlite3.Error):
            self._db.close()
        self._file.close()


def _thumbprint(message_id: str) -> bytes:
    return hashlib.blake2b(message_id.encode(), digest_size=_THUMBPRINT_BYTES).digest()


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A message found by a walk, before the skip rule has judged it."""

    path: Path
    located: mbox_reader.Located


class MboxConnector:
    """Discovers, fetches and reconciles the messages of the mbox files under one root."""

    def __init__(
        self,
        config: MboxConfig,
        *,
        name: str = MBOX_CONNECTOR_NAME,
        store: DocStore | None = None,
        other_sources: Sequence[str] = (),
        cache_dir: Path | None = None,
        max_message_bytes: int | None = None,
        settle_seconds: float = SETTLE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Args:
        config: The validated settings; ``root`` must be set.
        name: The configured instance's name, which files its documents (see
            ``plugin._source_name``).
        store: Where the documents of other mail sources are read, for
            ``skip_known_message_ids``; this source's own are looked up in it too.
        other_sources: The names of the other configured mail sources (type ``gmail`` or
            ``mbox``). Empty means nothing can be known elsewhere.
        cache_dir: Where a walk keeps its temporary key sets.
        max_message_bytes: ``ingest.max_fetch_bytes``. The pipeline checks a fetched message
            against it only after the fetch returns, so ``fetch`` refuses a longer span before
            reading it (§5.2). ``None`` means no limit.
        settle_seconds: How long a file must have been unmodified before discovery reads it
            (§5.1), so a message still being appended is never indexed half-written.
        clock: Wall time, for ``settle_seconds``.
        """
        self.name = name
        self._config = config
        self._root = Path(config.root).expanduser().resolve()
        self._store = store
        self._others = tuple(other_sources)
        self._cache_dir = cache_dir
        self._max_message_bytes = max_message_bytes
        self._settle = settle_seconds
        self._clock = clock
        self._next: str | None = None
        self._enumerated = False
        self._walked_full = False

    # --- identity and scope --------------------------------------------------------------

    @property
    def other_sources(self) -> tuple[str, ...]:
        """The configured mail sources ``skip_known_message_ids`` compares against."""
        return self._others

    @property
    def source_scope(self) -> str:
        """A non-secret name for this scope: a digest of the root, never the path."""
        digest = hashlib.blake2b(str(self._root).encode(), digest_size=8).hexdigest()
        return f"mbox:{digest}"

    @property
    def scope_fingerprint(self) -> str:
        return self._config.scope_fingerprint(self.source_scope)

    @property
    def reconciliation_scope(self) -> str:
        """Fences a completed inventory, and a held deletion proposal, to this exact scope."""
        return f"mbox:{self.scope_fingerprint}"

    @property
    def walked_full_inventory(self) -> bool:
        """Whether the walk that last completed read every file, however it began.

        An incremental sync skips unchanged files, so its manifest names a delta; a sync that
        found no usable watermark read everything and a reconciliation should follow it
        (``mbox.md`` §5.1, ``ingest.md`` §11).
        """
        return self._enumerated and self._walked_full

    # --- watermark -----------------------------------------------------------------------

    @property
    def watermark(self) -> Watermark | None:
        """The files the last **complete** walk read, once one has; otherwise ``None``."""
        if not self._enumerated or self._next is None:
            return None
        return Watermark(
            value=self._next,
            observed_at=datetime.now(tz=UTC),
            metadata={_FINGERPRINT_KEY: self.scope_fingerprint},
        )

    def _resume(self, watermark: Watermark | None) -> dict[str, list[int]] | None:
        """The stored file map, if it was recorded under this scope and is readable."""
        if watermark is None or watermark.metadata.get(_FINGERPRINT_KEY) != self.scope_fingerprint:
            return None
        try:
            recorded = json.loads(watermark.value)[_FILES_KEY]
            return {str(key): [int(value[0]), int(value[1])] for key, value in recorded.items()}
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            return None

    # --- the walk ------------------------------------------------------------------------

    def _require_root(self) -> None:
        if not self._root.exists():
            msg = "the configured root does not exist or cannot be read"
            raise ConnectorError(msg)

    def _candidates(self, *, strict: bool) -> Iterator[Path]:
        """Every regular file that may hold mail, in a stable order, as both passes see it."""
        if self._root.is_file():
            yield self._root
            return
        yield from self._walk(self._root, strict=strict)

    def _walk(self, directory: Path, *, strict: bool) -> Iterator[Path]:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            if strict:
                raise
            _log.warning("mbox %s: a directory could not be listed and was skipped", self.name)
            return
        for entry in entries:
            if not self._config.include_hidden and entry.name.startswith("."):
                continue
            if entry.is_symlink():
                continue
            relative = PurePosixPath(entry.relative_to(self._root))
            if globs.covers(relative, self._config.exclude):
                continue
            if entry.is_dir():
                if entry.name not in IGNORED_DIRECTORIES:
                    yield from self._walk(entry, strict=strict)
            elif entry.is_file() and entry.suffix.lower() not in _SIDE_FILES:
                yield entry

    def _key(self, path: Path) -> str:
        """A file's name in the watermark: relative to the root, POSIX-separated."""
        return path.name if path == self._root else str(PurePosixPath(path.relative_to(self._root)))

    def _messages(
        self,
        seen: _KeySet,
        *,
        previous: dict[str, list[int]] | None,
        current: dict[str, list[int]] | None,
        strict: bool,
        settle: float = 0.0,
    ) -> Generator[_Candidate]:
        """Each message not yet seen in this walk, from every file that needs reading.

        ``previous`` is the stored file map; a file it matches is skipped without being opened
        (and carried into ``current``). Reconciliation passes none and so reads everything.
        """
        unreadable = 0
        for path in self._candidates(strict=strict):
            key = self._key(path)
            try:
                status = path.stat()
                token = [status.st_size, status.st_mtime_ns]
                if previous is not None and previous.get(key) == token:
                    if current is not None:
                        current[key] = token
                    continue
                if not mbox_reader.is_mbox(path):
                    continue
                if settle and 0 <= self._clock() - status.st_mtime < settle:
                    # A future modification time is not "recent": it would never settle.
                    if strict:
                        msg = (
                            f"a file under the root was modified within the last {settle:g} "
                            f"seconds, so this walk cannot yet stand for the whole archive. "
                            f"Try again shortly"
                        )
                        raise ConnectorError(msg)
                    continue
                for located in mbox_reader.scan(path, status.st_size):
                    if seen.add(bytes.fromhex(located.digest)):
                        yield _Candidate(path, located)
            except OSError:
                if strict:
                    raise
                unreadable += 1
                continue
            if current is not None:
                current[key] = token
        if unreadable:
            _log.warning(
                "mbox %s: %d files could not be read and will be tried again", self.name, unreadable
            )

    # --- discovery -----------------------------------------------------------------------

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        async for batch in self.discover_batches(watermark):
            for document in batch:
                yield document

    async def discover_batches(
        self, watermark: Watermark | None
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]:
        """At most 250 records at a time, taken from the scan in a worker thread (§5.2)."""
        self._enumerated = False
        self._walked_full = False
        self._require_root()
        previous = self._resume(watermark)
        self._walked_full = previous is None
        current: dict[str, list[int]] = {}
        seen = _KeySet(self._cache_dir)
        known: _KeySet | None = None
        skipped = 0
        stream = self._messages(
            seen,
            previous=previous,
            current=current,
            strict=previous is None,
            settle=self._settle,
        )
        try:
            while True:
                try:
                    batch = await _pull(stream)
                except OSError:
                    # A full walk is recorded as the whole inventory, so one that could not read
                    # part of the archive must not complete (§5.1). No path in the message.
                    msg = (
                        "a file or directory under the root could not be read, so this walk "
                        "cannot stand for the whole archive. Fix its permissions or exclude it"
                    )
                    raise ConnectorError(msg) from None
                if not batch:
                    break
                if known is None and self._can_skip():
                    known = await self._load_known()
                admitted = await self._judge(batch, known)
                skipped += len(batch) - len(admitted)
                if admitted:
                    yield [self._discovered(candidate) for candidate in admitted]
        finally:
            stream.close()
            seen.close()
            if known is not None:
                known.close()
        if skipped:
            _log.info(
                "mbox %s: %d messages skipped, held by another mail source", self.name, skipped
            )
        self._next = json.dumps({_FILES_KEY: current}, sort_keys=True, separators=(",", ":"))
        self._enumerated = True

    def _can_skip(self) -> bool:
        return (
            self._config.skip_known_message_ids
            and bool(self._others)
            and isinstance(self._store, MailMessageIdIndex)
        )

    async def _load_known(self) -> _KeySet:
        """The ``Message-ID`` thumbprints other mail sources hold, in a temporary table."""
        known = _KeySet(self._cache_dir)
        index = self._store
        if not isinstance(index, MailMessageIdIndex):  # pragma: no cover - _can_skip checked
            return known
        pending: list[bytes] = []
        async with aclosing(index.known_mail_message_ids(self._others)) as ids:
            async for message_id in ids:
                pending.append(_thumbprint(message_id))
                if len(pending) >= _LOAD_BATCH:
                    await asyncio.to_thread(known.add_all, pending)
                    pending = []
        await asyncio.to_thread(known.add_all, pending)
        return known

    async def _judge(self, batch: list[_Candidate], known: _KeySet | None) -> list[_Candidate]:
        """Drop a message another source holds, unless this source already holds it (§6)."""
        if known is None or self._store is None:
            return batch
        keyed = {
            candidate.located.digest: _thumbprint(candidate.located.message_id)
            for candidate in batch
            if candidate.located.message_id is not None
        }
        held_elsewhere = await asyncio.to_thread(known.present, list(keyed.values()))
        admitted: list[_Candidate] = []
        for candidate in batch:
            digest = candidate.located.digest
            if keyed.get(digest) in held_elsewhere and (
                await self._store.find_document(self.name, digest) is None
            ):
                continue
            admitted.append(candidate)
        return admitted

    def _discovered(self, candidate: _Candidate) -> DiscoveredDoc:
        located = candidate.located
        return DiscoveredDoc(
            ref=DocRef(
                source_id=located.digest,
                uri=f"{candidate.path.as_uri()}#sha256={located.digest}",
                metadata={
                    _PATH: str(candidate.path),
                    _OFFSET: located.offset,
                    _LENGTH: located.length,
                },
            ),
            version_token=MBOX_REPRESENTATION_VERSION,
            media_type=MESSAGE_MEDIA_TYPE,
            size_bytes=located.size,
        )

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read the message at its recorded position and verify it is the message named (§5.2).

        Raises:
            NotFoundError: The file is gone, is outside the root, or no longer holds that
                message at that position. The pipeline reads this as the inventory having
                changed and enumerates again, which re-locates the message.
        """
        path = Path(str(ref.metadata.get(_PATH, "")))
        offset, length = ref.metadata.get(_OFFSET), ref.metadata.get(_LENGTH)
        if not isinstance(offset, int) or not isinstance(length, int) or not self._contains(path):
            msg = "the message is not in a file this source serves"
            raise NotFoundError(msg)
        if self._max_message_bytes is not None and length > self._max_message_bytes:
            msg = (
                f"a message of {length} bytes is above the {self._max_message_bytes}-byte fetch "
                f"cap and was not read. Raise ingest.max_fetch_bytes to index it"
            )
            raise ConnectorError(msg)
        try:
            content = await asyncio.to_thread(mbox_reader.read, path, offset, length, ref.source_id)
        except (OSError, mbox_reader.MessageChangedError) as exc:
            msg = "the message is no longer where it was found"
            raise NotFoundError(msg) from exc
        subject, created, message_id = _summarize(content)
        metadata: Metadata = {_VERSION_KEY: MBOX_REPRESENTATION_VERSION}
        if message_id is not None:
            metadata[MAIL_MESSAGE_ID_KEY] = message_id
        metadata[PROVENANCE_KEY] = self._provenance(ref.source_id, path, subject, created)
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=MESSAGE_MEDIA_TYPE,
            content=content,
            metadata=metadata,
        )

    def _contains(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - resolution failure is itself a refusal
            return False
        return resolved == self._root or self._root in resolved.parents

    def _provenance(
        self, digest: str, path: Path, subject: str, created: datetime | None
    ) -> JsonValue:
        """What a citation says about the message: its subject, its stated date and its file.

        A subject with a control character cannot be a title; the message is still described by
        its identity and its file. The snapshot is the path *relative to the root*, so a citation
        reproduces on another machine.
        """
        try:
            snapshot: LocalSnapshot | None = LocalSnapshot(path=self._key(path))
        except ValueError:
            snapshot = None
        for title in (subject, ""):
            try:
                source = SourceMetadata(
                    title=title,
                    source_id=digest,
                    version=MBOX_REPRESENTATION_VERSION,
                    created_at=created,
                    content_type=MESSAGE_MEDIA_TYPE,
                )
            except ValueError:
                continue
            return Provenance(source=source, snapshot=snapshot).as_metadata_value()
        return Provenance(
            unavailable_reason="the message could not be described"
        ).as_metadata_value()

    # --- reconciliation ------------------------------------------------------------------

    async def reconcile(self) -> AsyncIterator[SourceId]:
        async for batch in self.reconcile_batches():
            for source_id in batch:
                yield source_id

    async def reconcile_batches(self) -> AsyncIterator[Sequence[SourceId]]:
        """Every message in the archive now, read strictly (§5.4).

        Raises, rather than ending early, on a directory or file that cannot be read: only
        exhaustion of this iterator is a complete inventory, and absence is deletion to it.
        """
        self._require_root()
        seen = _KeySet(self._cache_dir)
        stream = self._messages(seen, previous=None, current=None, strict=True)
        try:
            while batch := await _pull(stream):
                yield [candidate.located.digest for candidate in batch]
        finally:
            stream.close()
            seen.close()


async def _pull(stream: Iterator[_Candidate]) -> list[_Candidate]:
    """The next batch of a walk, read in a worker thread.

    If this task is canceled the worker is waited for before the cancellation propagates, so the
    caller's cleanup never closes a generator or a database a thread is still using.
    """
    work = asyncio.ensure_future(
        asyncio.to_thread(lambda: list(itertools.islice(stream, MAX_DISCOVERY_BATCH)))
    )
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        await asyncio.wait({work})
        raise


def _summarize(content: bytes) -> tuple[str, datetime | None, str | None]:
    """The subject, the ``Date`` and the normalized ``Message-ID`` from a message's headers.

    Read from the headers only, and never logged. A header that is absent or unreadable is an
    empty subject, no date or no id, never an error: the message is still indexed.
    """
    head = content[:_HEADERS_ONLY]
    end = head.find(b"\n\n")
    if end != -1:
        head = head[:end]
    try:
        headers = BytesHeaderParser(policy=policy.default).parsebytes(head)
        subject = " ".join(str(headers.get("Subject", "")).split())
        raw_id = headers.get("Message-ID")
        message_id = normalize_message_id(str(raw_id)) if raw_id is not None else None
        created: datetime | None = None
        if (raw_date := headers.get("Date")) is not None:
            with suppress(ValueError, TypeError, IndexError):
                parsed = parsedate_to_datetime(str(raw_date))
                created = parsed if parsed.tzinfo is not None else None
    except (ValueError, TypeError, IndexError, MessageError):
        return "", None, None
    return subject, created, message_id
