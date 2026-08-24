"""A website connector over page blobs in one pinned local Git commit."""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from manicule.connectors.config import GitSiteConfig
from manicule.connectors.git_reader import (
    GitBlobTooLargeError,
    GitSourceError,
    GitTreeEntry,
    PinnedGitReader,
)
from manicule.connectors.site_routes import (
    MAX_MANIFEST_BYTES,
    SiteRouteRecord,
    canonical_site_url,
    infer_site_records,
    parse_site_manifest,
    records_from_manifest,
)
from manicule.core.content import RawDocument
from manicule.core.provenance import (
    PROVENANCE_KEY,
    LocalSnapshot,
    Provenance,
    SourceMetadata,
)
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = ["GitSiteConnector"]

_PROFILE = "git-site-v1"
_COMMIT = "git_commit"
_OBJECT_ID = "git_blob"
_PATH = "git_path"
_ROUTE_DIGEST = "route_digest"
_MEDIA_TYPE = "media_type"
_VERSION_TOKEN = "version_token"  # noqa: S105 - a change token, not a credential


@dataclass(frozen=True, slots=True)
class _Page:
    record: SiteRouteRecord
    entry: GitTreeEntry
    uri: str
    media_type: str
    version_token: str


@dataclass(frozen=True, slots=True)
class _Inventory:
    commit: str
    reader: PinnedGitReader
    pages: tuple[_Page, ...]
    by_identity: dict[str, _Page]
    watermark: Watermark


def _media_type(record: SiteRouteRecord) -> str:
    if record.media_type is not None:
        return record.media_type
    return {
        ".md": "text/markdown",
        ".mdx": "text/mdx",
        ".html": "text/html",
    }.get(PurePosixPath(record.source).suffix.lower(), "application/octet-stream")


def _relative(path: str, root: str) -> str:
    if root == ".":
        return path
    prefix = f"{root}/"
    if not path.startswith(prefix):
        raise GitSourceError("Git returned a path outside the configured content root")
    return path[len(prefix) :]


def _matches(path: str, pattern: str) -> bool:
    """Match POSIX globs, including the useful zero-directory meaning of ``**/``."""
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])
    )


def _admitted(path: str, *, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    return any(_matches(path, pattern) for pattern in include) and not any(
        _matches(path, pattern) for pattern in exclude
    )


def _token(entry: GitTreeEntry, record: SiteRouteRecord) -> str:
    material = json.dumps(
        {
            "blob": entry.object_id,
            "profile": _PROFILE,
            "route": record.digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.blake2b(material, digest_size=20).hexdigest()


class GitSiteConnector:
    """Index public pages directly from Git objects without copying their source bytes."""

    full_inventory_authority = "direct_current_content"

    def __init__(self, config: GitSiteConfig, *, name: str = "git-site") -> None:
        self.name = name
        self._config = config
        self._repository = Path(config.repository).expanduser()
        self._lock = asyncio.Lock()
        self._prepared: _Inventory | None = None
        self._current: _Inventory | None = None
        self._completed: _Inventory | None = None
        self._inventories: dict[str, _Inventory] = {}
        self._run_inventory: contextvars.ContextVar[_Inventory | None] = contextvars.ContextVar(
            f"git-site-inventory-{id(self)}", default=None
        )

    @property
    def source_scope(self) -> str:
        try:
            repository = self._repository.resolve(strict=False)
        except OSError:
            repository = self._repository.absolute()
        material = f"{repository}\0{self._config.content_root}".encode()
        return f"git-site:{hashlib.blake2b(material, digest_size=20).hexdigest()}"

    @property
    def scope_fingerprint(self) -> str:
        return self._config.scope_fingerprint(self.source_scope)

    @property
    def reconciliation_scope(self) -> str:
        return f"git-site:{self.scope_fingerprint}"

    @property
    def watermark(self) -> Watermark | None:
        return None if self._completed is None else self._completed.watermark

    async def setup(self) -> None:
        """Validate the repository and prepare the first atomic inventory."""
        async with self._lock:
            if self._prepared is None and self._current is None:
                self._prepared = await self._pin_inventory()

    async def teardown(self) -> None:
        """Reap every lazily-created Git batch process."""
        readers = {
            id(inventory.reader): inventory.reader for inventory in self._inventories.values()
        }
        if self._prepared is not None:
            readers[id(self._prepared.reader)] = self._prepared.reader
        for reader in readers.values():
            await reader.aclose()

    async def _inventory_for_discovery(self) -> _Inventory:
        async with self._lock:
            if self._prepared is not None:
                inventory = self._prepared
                self._prepared = None
            else:
                inventory = await self._pin_inventory()
            self._current = inventory
            self._run_inventory.set(inventory)
            for previous in self._inventories.values():
                if previous.reader is not inventory.reader:
                    # Closing only the process is safe: a later fetch for an already-issued
                    # reference reopens it against the same immutable commit and object id.
                    await previous.reader.aclose()
            return inventory

    async def _pin_inventory(self) -> _Inventory:
        reader = PinnedGitReader(
            self._repository,
            revision=self._config.revision,
            max_blob_bytes=self._config.max_bytes or 256 * 1024 * 1024,
        )
        try:
            entries = await reader.setup(content_root=self._config.content_root)
            return await self._build_inventory(reader, entries)
        except BaseException:
            await asyncio.shield(reader.aclose())
            raise

    async def _build_inventory(
        self, reader: PinnedGitReader, entries: tuple[GitTreeEntry, ...]
    ) -> _Inventory:
        """Validate all route facts atomically before making an inventory visible."""
        ordinary = tuple(entry for entry in entries if entry.ordinary_blob)
        admitted = tuple(
            entry
            for entry in ordinary
            if entry.path != self._config.route_manifest
            and _admitted(
                _relative(entry.path, self._config.content_root),
                include=self._config.include,
                exclude=self._config.exclude,
            )
        )
        if self._config.max_bytes is not None:
            oversized = next(
                (entry for entry in admitted if (entry.size or 0) > self._config.max_bytes), None
            )
            if oversized is not None:
                raise GitBlobTooLargeError(
                    f"Git page {oversized.path!r} exceeds the configured "
                    f"{self._config.max_bytes}-byte limit"
                )

        if self._config.route_manifest is None:
            records = infer_site_records(
                (entry.path for entry in admitted), content_root=self._config.content_root
            )
        else:
            manifest_entry = await reader.lookup(self._config.route_manifest)
            if manifest_entry is None or not manifest_entry.ordinary_blob:
                raise GitSourceError("configured route manifest is not an ordinary pinned Git blob")
            manifest = parse_site_manifest(
                await reader.read_entry(manifest_entry, max_bytes=MAX_MANIFEST_BYTES)
            )
            records = records_from_manifest(
                manifest,
                (entry.path for entry in admitted),
                content_root=self._config.content_root,
            )

        by_path = {entry.path: entry for entry in admitted}
        pages = tuple(
            _Page(
                record=record,
                entry=by_path[record.source],
                uri=canonical_site_url(self._config.base_url, record.route),
                media_type=_media_type(record),
                version_token=_token(by_path[record.source], record),
            )
            for record in records
        )
        inventory = _Inventory(
            commit=reader.commit,
            reader=reader,
            pages=pages,
            by_identity={page.record.identity: page for page in pages},
            watermark=Watermark(
                value=reader.commit,
                observed_at=datetime.now(UTC),
                metadata={
                    "full_inventory_authority": self.full_inventory_authority,
                    "route_policy": _PROFILE,
                },
            ),
        )
        existing = self._inventories.get(inventory.commit)
        if existing is not None:
            await reader.aclose()
            return existing
        self._inventories[inventory.commit] = inventory
        return inventory

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield a body-free inventory from exactly one resolved commit."""
        del watermark  # A local tree is cheap to enumerate; tokens make unchanged bodies free.
        inventory = await self._inventory_for_discovery()
        for page in inventory.pages:
            yield DiscoveredDoc(
                ref=DocRef(
                    source_id=page.record.identity,
                    uri=page.uri,
                    metadata={
                        _COMMIT: inventory.commit,
                        _OBJECT_ID: page.entry.object_id,
                        _PATH: page.entry.path,
                        _ROUTE_DIGEST: page.record.digest,
                        _MEDIA_TYPE: page.media_type,
                    },
                ),
                version_token=page.version_token,
                title=page.record.title or PurePosixPath(page.record.source).stem,
                media_type=page.media_type,
                size_bytes=page.entry.size,
            )
        # This line is reached only when the async generator is fully exhausted.
        self._completed = inventory

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read the exact blob named by a reference from its pinned inventory."""
        commit = ref.metadata.get(_COMMIT)
        if not isinstance(commit, str):
            raise GitSourceError("Git site reference has no pinned commit")
        inventory = self._inventories.get(commit)
        page = inventory.by_identity.get(ref.source_id) if inventory is not None else None
        if inventory is None or page is None:
            raise GitSourceError("Git site reference does not belong to a known inventory")
        expected = {
            _OBJECT_ID: page.entry.object_id,
            _PATH: page.entry.path,
            _ROUTE_DIGEST: page.record.digest,
            _MEDIA_TYPE: page.media_type,
        }
        if (
            any(ref.metadata.get(key) != value for key, value in expected.items())
            or ref.uri != page.uri
        ):
            raise GitSourceError("Git site reference contradicts its pinned inventory")
        body = await inventory.reader.read_entry(page.entry, max_bytes=self._config.max_bytes)
        provenance = Provenance(
            source=SourceMetadata(
                title=page.record.title or PurePosixPath(page.record.source).stem,
                canonical_uri=page.uri,
                source_id=page.record.identity,
                version=page.entry.object_id,
                content_type=page.media_type,
            ),
            snapshot=LocalSnapshot(path=page.entry.path),
        )
        return RawDocument(
            source_id=page.record.identity,
            uri=page.uri,
            media_type=page.media_type,
            content=body,
            metadata={
                PROVENANCE_KEY: provenance.as_metadata_value(),
                _VERSION_TOKEN: page.version_token,
                _COMMIT: inventory.commit,
                _OBJECT_ID: page.entry.object_id,
                _PATH: page.entry.path,
                _ROUTE_DIGEST: page.record.digest,
            },
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield identities from the same pinned inventory as this task's discovery."""
        inventory = self._run_inventory.get() or self._current
        if inventory is None:
            raise RuntimeError("Git site connector has not completed setup or discovery")
        for page in inventory.pages:
            yield page.record.identity
