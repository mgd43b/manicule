"""The documents in a Git repository: Markdown and text at one pinned commit, and nothing else.

A repository is mostly code, and code is not what anyone searching it for an answer wants. This
connector lists the tree of one commit, keeps the paths a source's ``include`` admits and its
``exclude`` does not, and reads only those blobs. A large source tree costs one directory listing.

**Everything is read from one commit.** The revision is resolved once at the start of a sync, and
discovery, every fetch and the reconciliation that follows all read that commit's objects, never a
working tree. A push or a checkout while a sync runs therefore cannot hand it half of two states,
and a body is always the one discovery named. For an ``https`` repository the commit comes from a
managed mirror refreshed first (:mod:`manicule.connectors.git_mirror`); for a local path it comes
from the repository as it stands.

**A commit nobody has pushed to costs a ``rev-parse``.** The watermark is the commit a clean run
last indexed. When the revision still resolves to it, under the same rules, discovery lists
nothing and nothing is fetched, so a source can be checked every few minutes without it being a
cost. A different commit lists the admitted tree, and each blob-id token decides which bodies are
read.

**A document's identity is its path from the repository root.** Not from the content root and not
from a site route: the connector has no site, and the path is the one handle a repository itself
promises. A rename is therefore a deletion and an addition, which is what ``reconcile`` and the
pipeline's content-hash deduplication are for. The change token is the blob's object id, so a
commit that touched other files changes nothing here, and a document whose bytes did not change is
never fetched.

**Paths say what a document is.** Under ``layout = "auto"`` a path inside BMAD's output folders is
recognized (:mod:`manicule.connectors.bmad_layout`) and the document is recorded as a PRD, a
story or an epic's retrospective, with its initiative and epic. That is enrichment: it never
decides what is admitted, and a repository with no BMAD in it is indexed exactly the same way.

**What is skipped is counted, not hidden.** A document over ``max_bytes``, a suffix with no media
type, and a path no citation could carry are left out of both discovery and reconciliation — so one
that grows past the limit leaves the index instead of lingering stale — and their counts travel on
the watermark and in the log.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import quote

from manicule.connectors import bmad_layout, globs
from manicule.connectors.config import GitDocsConfig
from manicule.connectors.git_mirror import GitMirror
from manicule.connectors.git_reader import (
    GitSourceError,
    GitTreeEntry,
    PinnedGitReader,
    path_below,
)
from manicule.connectors.git_remote import is_remote, mirror_directory
from manicule.core.content import Metadata, RawDocument
from manicule.core.errors import ConfigError
from manicule.core.provenance import (
    MAX_FIELD_CHARS,
    MAX_SECTION_DEPTH,
    PROVENANCE_KEY,
    LocalSnapshot,
    Provenance,
    SourceMetadata,
)
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = ["GitDocsConnector"]

_log = logging.getLogger("manicule.connectors.git_docs")

_POLICY = "git-docs-v1"
_COMMIT = "git_commit"
_OBJECT_ID = "git_blob"
_PATH = "git_path"
_MEDIA_TYPE = "media_type"
_VERSION_TOKEN = "version_token"  # noqa: S105 - a change token, not a credential
BMAD_LAYOUT = "bmad_layout"
BMAD_KIND = "bmad_kind"
BMAD_INITIATIVE = "bmad_initiative"
BMAD_EPIC = "bmad_epic"

_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdx": "text/mdx",
    ".txt": "text/plain",
}
"""The suffixes this connector can hand a parser, matched case-insensitively."""

_KEPT_INVENTORIES = 4
"""How many pinned commits stay readable. A reference is fetched during the run that issued it, so
two is the working set; the rest is headroom for a run that overlaps the next one's discovery."""


@dataclass(frozen=True, slots=True)
class _Document:
    entry: GitTreeEntry
    uri: str
    media_type: str
    title: str
    section_path: tuple[str, ...]
    facts: bmad_layout.BmadFacts | None
    version_token: str


@dataclass(frozen=True, slots=True)
class _Inventory:
    commit: str
    reader: PinnedGitReader
    documents: tuple[_Document, ...]
    by_identity: dict[str, _Document]
    watermark: Watermark


def _token(entry: GitTreeEntry, facts: bmad_layout.BmadFacts | None, address: str) -> str:
    """What makes this document different from itself an hour ago.

    The blob, the facts read from its path, the policy, and the **template** its citation address
    is made from, so that changing how any of them is derived re-ingests rather than leaving
    documents described by the old rules or cited at an address nobody now configures. **Not** the
    commit: it moves with every push, and a token that moved with it would make every commit
    re-embed the whole corpus. (The address itself contains the commit, which is why the template
    and not the address is what goes in.)
    """
    material = json.dumps(
        {
            "blob": entry.object_id,
            "policy": _POLICY,
            "address": address,
            "facts": None
            if facts is None
            else [facts.layout, facts.kind, facts.initiative, facts.epic],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.blake2b(material, digest_size=20).hexdigest()


class GitDocsConnector:
    """Index the Markdown and text in a Git repository, read from one commit at a time."""

    # Deliberately **not** ``enumerates_full_inventory``, and so no ``full_inventory_authority``
    # either: both describe a manifest that is the whole source, and a quiet run's is empty. A run
    # that finds the commit it last synced yields nothing at all, and a manifest that is empty
    # because nothing changed must never be read as "the whole source is empty". The connector
    # says which kind of run it just had through :attr:`walked_full_inventory`, after the fact, as
    # the mail connectors do.

    def __init__(
        self,
        config: GitDocsConfig,
        *,
        name: str = "git-docs",
        token: str | None = None,
        mirror_root: Path | None = None,
        mirror: GitMirror | None = None,
    ) -> None:
        """Build the connector. Nothing is read or fetched until a sync asks.

        Args:
            config: Validated configuration.
            name: What this source is called. It is part of every document's identity.
            token: The credential for a private remote, already read from the environment.
            mirror_root: Where mirrors live. Required for an ``https`` repository.
            mirror: A ready-made mirror, which a test passes to fetch from a repository on disk
                instead of the address in ``config``. Configuration never supplies one.

        Raises:
            ConfigError: An ``https`` repository was given neither a ``mirror_root`` nor a
                ``mirror``.
        """
        self.name = name
        self._config = config
        self._remote = is_remote(config.repository)
        self._local = None if self._remote else Path(config.repository).expanduser()
        self._mirror: GitMirror | None = None
        if self._remote:
            if mirror is None and mirror_root is None:
                msg = "an https repository needs a directory to mirror into"
                raise ConfigError(msg)
            self._mirror = mirror or GitMirror(
                config.repository,
                mirror_directory(cast("Path", mirror_root), config.repository),
                token=token,
                username=config.username,
                timeout_s=config.fetch_timeout_s,
            )
        self._lock = asyncio.Lock()
        self._inventories: dict[str, _Inventory] = {}
        self._completed: Watermark | None = None
        self._enumerated = False
        self._walked_full = False
        self._run_inventory: contextvars.ContextVar[_Inventory | None] = contextvars.ContextVar(
            f"git-docs-inventory-{id(self)}", default=None
        )

    @property
    def mirror(self) -> GitMirror | None:
        """The managed mirror of an ``https`` repository, or ``None`` for a local path."""
        return self._mirror

    @property
    def source_scope(self) -> str:
        if self._local is not None:
            try:
                where = str(self._local.resolve(strict=False))
            except OSError:
                where = str(self._local.absolute())
        else:
            where = self._config.repository
        material = f"{where}\0{self._config.content_root}".encode()
        return f"git-docs:{hashlib.blake2b(material, digest_size=20).hexdigest()}"

    @property
    def scope_fingerprint(self) -> str:
        return self._config.scope_fingerprint(self.source_scope)

    @property
    def reconciliation_scope(self) -> str:
        return f"git-docs:{self.scope_fingerprint}"

    @property
    def watermark(self) -> Watermark | None:
        return self._completed

    @property
    def walked_full_inventory(self) -> bool:
        """Whether the run that last completed listed every document, not just found nothing new.

        Asked after discovery, because the answer is made during it: a run that meets the commit it
        last synced lists nothing, and only a run that listed the tree is evidence for a
        reconciliation to follow.
        """
        return self._enumerated and self._walked_full

    async def setup(self) -> None:
        """Check what can be checked without the network.

        A local path must be a directory now, so a typo is a startup refusal naming the setting
        and not a sync that lists nothing. A remote is not contacted here: the container awaits
        this before it hands the connector to anyone, and a network that is down at startup must
        not stop an installation from serving what it has already indexed.
        """
        if self._local is not None and not self._local.is_dir():
            msg = (
                f"the git-docs repository {self._config.repository!r} is not a directory. Give "
                f"the path of a checkout, or an https address to mirror"
            )
            raise ConfigError(msg)

    async def teardown(self) -> None:
        """Reap every lazily created Git batch process."""
        for inventory in tuple(self._inventories.values()):
            await inventory.reader.aclose()

    async def _reader_for(self) -> PinnedGitReader:
        if self._mirror is not None:
            ref = await self._mirror.refresh(self._config.revision)
            return PinnedGitReader(
                self._mirror.directory, revision=ref, max_blob_bytes=self._config.max_bytes
            )
        assert self._local is not None  # noqa: S101 - one of the two is set in __init__
        return PinnedGitReader(
            self._local, revision=self._config.revision, max_blob_bytes=self._config.max_bytes
        )

    async def _pin(self) -> _Inventory:
        """Resolve the revision to one commit and list its documents. Fetches first if remote."""
        return await self._list(await self._reader_for())

    async def _list(self, reader: PinnedGitReader) -> _Inventory:
        """List the tree at the commit ``reader`` is pinned to, or close it and raise."""
        try:
            entries = await reader.setup(content_root=self._config.content_root)
            return self._inventory(reader, entries)
        except BaseException:
            await asyncio.shield(reader.aclose())
            raise

    def _unchanged_since(self, watermark: Watermark | None, commit: str) -> bool:
        """Whether the commit is the one the last clean run already indexed, under the same rules.

        The commit alone is not enough. What a document is *said to be* depends on this
        connector's own rules (``_POLICY``: which suffixes, how a path is read as BMAD, how a token
        is made) and on the scope, and a release that changes a rule must re-ingest at a commit
        nobody has pushed to. Both ride on the watermark and are compared here, so a stale one is
        a full listing, which is always safe.
        """
        return (
            watermark is not None
            and watermark.value == commit
            and watermark.metadata.get("policy") == _POLICY
            and watermark.metadata.get("scope") == self.scope_fingerprint
        )

    def _inventory(self, reader: PinnedGitReader, entries: tuple[GitTreeEntry, ...]) -> _Inventory:
        if self._config.content_root != "." and not entries:
            # An empty listing under a named root is almost always a typo or a moved folder, and
            # an empty inventory is what reconciliation reads as "everything was deleted".
            msg = (
                f"content_root {self._config.content_root!r} has no files at commit "
                f"{reader.commit[:12]}. Check the setting, or the revision"
            )
            raise GitSourceError(msg)
        paths = [entry.path for entry in entries if entry.object_type == "blob"]
        detected = bmad_layout.detect(paths)
        layouts = self._layouts(detected)
        roots: frozenset[str] = (
            bmad_layout.v7_roots(paths) if bmad_layout.V7 in layouts else frozenset()
        )

        documents: list[_Document] = []
        oversized = unsupported = 0
        for entry in entries:
            if not entry.ordinary_blob or not globs.admitted(
                path_below(entry.path, self._config.content_root),
                include=self._config.include,
                exclude=self._config.exclude,
            ):
                continue
            media_type = _MEDIA_TYPES.get(PurePosixPath(entry.path).suffix.lower())
            parts = entry.path.split("/")
            if (
                media_type is None
                or not entry.path.isprintable()
                or len(entry.path) > MAX_FIELD_CHARS
                or len(parts) - 1 > MAX_SECTION_DEPTH
            ):
                unsupported += 1
                continue
            if (entry.size or 0) > self._config.max_bytes:
                oversized += 1
                continue
            facts = bmad_layout.classify(entry.path, layouts=layouts, v7_roots=roots)
            documents.append(
                _Document(
                    entry=entry,
                    uri=self.address_of(reader.commit, entry.path),
                    media_type=media_type,
                    title=PurePosixPath(entry.path).stem,
                    section_path=tuple(parts[:-1]),
                    facts=facts,
                    version_token=_token(entry, facts, self._address_template() or ""),
                )
            )
        if oversized or unsupported:
            _log.warning(
                "git-docs %r skipped %d document(s) over %d bytes and %d with no usable "
                "suffix or path at commit %s",
                self.name,
                oversized,
                self._config.max_bytes,
                unsupported,
                reader.commit[:12],
            )
        _log.info(
            "git-docs %r: %d document(s) at commit %s, BMAD layout %s",
            self.name,
            len(documents),
            reader.commit[:12],
            bmad_layout.describe(detected),
        )
        return _Inventory(
            commit=reader.commit,
            reader=reader,
            documents=tuple(documents),
            by_identity={document.entry.path: document for document in documents},
            watermark=Watermark(
                value=reader.commit,
                observed_at=datetime.now(UTC),
                metadata={
                    "policy": _POLICY,
                    "scope": self.scope_fingerprint,
                    "documents": len(documents),
                    "skipped_oversized": oversized,
                    "skipped_unsupported": unsupported,
                    "bmad_layout": bmad_layout.describe(detected),
                },
            ),
        )

    def _layouts(self, detected: frozenset[str]) -> frozenset[str]:
        setting = self._config.layout
        if setting == bmad_layout.NONE:
            return frozenset()
        if setting == bmad_layout.AUTO:
            return detected
        return frozenset({setting})

    def _address_template(self) -> str | None:
        """The citation address with ``{commit}`` and ``{path}`` still in it, or ``None``."""
        if self._config.blob_url is not None:
            return self._config.blob_url
        if self._remote:
            return f"{self._config.repository.removesuffix('.git')}/blob/{{commit}}/{{path}}"
        return None

    def address_of(self, commit: str, path: str) -> str:
        """Where a reader opens ``path`` as it was at ``commit``."""
        template = self._address_template()
        if template is not None:
            quoted = quote(path, safe="/")
            return template.replace("{commit}", commit).replace("{path}", quoted)
        assert self._local is not None  # noqa: S101 - a remote always has a template
        return (self._local.expanduser().resolve(strict=False) / path).as_uri()

    async def _remember(self, inventory: _Inventory) -> _Inventory:
        """Keep ``inventory`` readable, or return the one already kept for the same commit."""
        existing = self._inventories.get(inventory.commit)
        if existing is not None:
            await inventory.reader.aclose()
            return existing
        self._inventories[inventory.commit] = inventory
        while len(self._inventories) > _KEPT_INVENTORIES:
            oldest = next(iter(self._inventories))
            await self._inventories.pop(oldest).reader.aclose()
        return inventory

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield what a commit holds, or nothing when it is the commit already indexed.

        The stored watermark is the last commit a clean run indexed. The revision is resolved
        first (for a remote, after a fetch, which is itself a no-op when nothing was pushed), and
        if it is that commit, under the same rules, then **no document is listed and none is
        fetched**: a periodic check of a quiet repository costs a ``rev-parse``. Otherwise the
        whole admitted tree is listed and each document's blob-id token decides, as it always did,
        whether its bytes are read. A commit's diff against the previous one would list less, and
        was not used: a mirror keeps one commit, so the previous one is usually not there.
        """
        self._enumerated = False
        async with self._lock:
            reader = await self._reader_for()
            try:
                commit = await reader.resolve()
            except BaseException:
                await asyncio.shield(reader.aclose())
                raise
            if self._unchanged_since(watermark, commit):
                await reader.aclose()
                inventory = None
                # Cleared, so a reconciliation later in this task cannot reuse an inventory an
                # earlier run in the same task pinned at some other commit.
                self._run_inventory.set(None)
            else:
                inventory = await self._remember(await self._list(reader))
                self._run_inventory.set(inventory)
        if inventory is None:
            assert watermark is not None  # noqa: S101 - _unchanged_since required one
            _log.info("git-docs %r: commit %s unchanged, nothing to do", self.name, commit[:12])
            self._complete(
                watermark.model_copy(update={"observed_at": datetime.now(UTC)}), full=False
            )
            return
        for document in inventory.documents:
            yield DiscoveredDoc(
                ref=DocRef(
                    source_id=document.entry.path,
                    uri=document.uri,
                    metadata={
                        _COMMIT: inventory.commit,
                        _OBJECT_ID: document.entry.object_id,
                        _PATH: document.entry.path,
                        _MEDIA_TYPE: document.media_type,
                    },
                ),
                version_token=document.version_token,
                title=document.title,
                media_type=document.media_type,
                size_bytes=document.entry.size,
            )
        # Reached only when the generator is exhausted: a consumer that stops early leaves the
        # previous watermark in place, which is the promise the protocol makes about it.
        self._complete(inventory.watermark, full=True)

    def _complete(self, watermark: Watermark, *, full: bool) -> None:
        self._completed = watermark
        self._enumerated = True
        self._walked_full = full

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read the exact blob a reference names from the commit that issued it."""
        commit = ref.metadata.get(_COMMIT)
        if not isinstance(commit, str):
            raise GitSourceError("git-docs reference has no pinned commit")
        inventory = self._inventories.get(commit)
        document = inventory.by_identity.get(ref.source_id) if inventory is not None else None
        if inventory is None or document is None:
            raise GitSourceError("git-docs reference does not belong to a known inventory")
        expected = {
            _OBJECT_ID: document.entry.object_id,
            _PATH: document.entry.path,
            _MEDIA_TYPE: document.media_type,
        }
        if (
            any(ref.metadata.get(key) != value for key, value in expected.items())
            or ref.uri != document.uri
        ):
            # The inventory, not the reference, is the authority. A reference that disagrees was
            # forged or stale, and reading what it names could read any object in the repository.
            raise GitSourceError("git-docs reference contradicts its pinned inventory")
        body = await inventory.reader.read_entry(document.entry, max_bytes=self._config.max_bytes)
        web = document.uri if document.uri.startswith(("http://", "https://")) else ""
        provenance = Provenance(
            source=SourceMetadata(
                title=document.title,
                canonical_uri=web,
                source_id=document.entry.path,
                version=document.entry.object_id,
                content_type=document.media_type,
                section_path=document.section_path,
            ),
            snapshot=LocalSnapshot(path=document.entry.path),
        )
        metadata: Metadata = {
            PROVENANCE_KEY: provenance.as_metadata_value(),
            _VERSION_TOKEN: document.version_token,
            _COMMIT: inventory.commit,
            _OBJECT_ID: document.entry.object_id,
            _PATH: document.entry.path,
        }
        if document.facts is not None:
            metadata[BMAD_LAYOUT] = document.facts.layout
            metadata[BMAD_KIND] = document.facts.kind
            if document.facts.initiative is not None:
                metadata[BMAD_INITIATIVE] = document.facts.initiative
            if document.facts.epic is not None:
                metadata[BMAD_EPIC] = document.facts.epic
        return RawDocument(
            source_id=document.entry.path,
            uri=document.uri,
            media_type=document.media_type,
            content=body,
            metadata=metadata,
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield the path of every document the revision holds, resolved now.

        Inside a run that has already discovered, this task's own pinned inventory is used, so a
        reconciliation that follows a sync in one task agrees with what that sync saw. Anywhere
        else — a scheduled pass, a manual command, a pass after a restart — it pins a fresh one,
        which for a remote **fetches first**. Reusing the last sync's would be wrong in both
        directions: after a restart there is none, and after a sync it names the commit that sync
        pinned, so a document the revision has since dropped would still be reported present and
        never leave the index.

        A fetch that fails raises, and a pass that raises deletes nothing
        (``docs/ingest.md`` §11.1): the paths seen so far are a prefix, not the truth.
        """
        inventory = self._run_inventory.get()
        if inventory is not None:
            for document in inventory.documents:
                yield document.entry.path
            return
        async with self._lock:
            fresh = await self._pin()
        try:
            for document in fresh.documents:
                yield document.entry.path
        finally:
            await asyncio.shield(fresh.reader.aclose())
