"""What the application service needs, stated as protocols rather than as imports.

The service is the layer both surfaces call, and it is the only layer with any behaviour in
it. Writing it against protocols buys the two properties that make that worth doing:

**The service imports no database, no model runtime and no web framework.** Those arrive
through :class:`Backend`, which :mod:`manicule.app.runtime` implements over the container.

**Every guard can be checked against a component that breaks its half of the bargain.** A
store that ignores its workspace scope, a retriever that returns another tenant's chunk, an
ingest run that reports success having indexed nothing — each is a dozen lines here and
impossible against a migrated database.

The shapes deliberately mirror the components they stand for, so the production
implementations satisfy them structurally with no adapter in between.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Imported for real rather than under TYPE_CHECKING, because it is re-exported: `answering()`
# already declares the one method the answer path needs, and a second protocol of the same
# shape here would be a copy to keep in step. The module loads no provider library.
from manicule.generation.answering import SupportsAnswer as Answering

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from manicule.app.results import ApiKeySummary, Check
    from manicule.config.settings import Settings
    from manicule.core.content import Chunk, Document, DocumentStatus
    from manicule.core.embedding import IndexFingerprints
    from manicule.core.retrieval import Filter, Query
    from manicule.ingest.pipeline import RunReport
    from manicule.ingest.reindex import ReindexReport
    from manicule.plugins.registry import Discovery
    from manicule.retrieval.retriever import RetrievalResult


@runtime_checkable
class DocumentSurface(Protocol):
    """The reads and writes the service performs against the relational store.

    A subset of what :class:`~manicule.storage.docstore.SqliteDocStore` offers, chosen so the
    service cannot reach past what a surface legitimately does — it lists, reads, deletes and
    counts, and it cannot write a chunk or advance a watermark.
    """

    @property
    def workspace_id(self) -> str:
        """Which tenant this handle serves. Never a parameter, so it cannot be forgotten."""
        ...

    async def get_document(self, document_id: str) -> Document | None: ...

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol it narrows
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]: ...

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
    ) -> int: ...

    async def count_chunks(self, document_id: str | None = None) -> int: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def soft_delete_document(self, document_id: str) -> None: ...

    async def document_statistics(self) -> Mapping[str, Mapping[str, int]]:
        """Document counts grouped by ``source``, ``media_type`` and ``status``.

        Three aggregates rather than a listing the surface counts itself: a page of documents
        answers a different question, and summing one would report the page rather than the
        corpus while looking exactly like a total.
        """
        ...

    async def index_fingerprints(self) -> IndexFingerprints: ...

    async def connector_metadata(self, connector: str) -> Mapping[str, object]: ...


@runtime_checkable
class Retrieving(Protocol):
    """One workspace's retrieval, end to end.

    :class:`~manicule.retrieval.retriever.Retriever` satisfies this. The service holds the
    narrower shape so a test can drive ``search`` and ``ask`` without assembling a pipeline.
    """

    async def retrieve(self, query: Query) -> RetrievalResult: ...


@runtime_checkable
class Ingesting(Protocol):
    """The ingest operations a surface can start."""

    async def index_path(
        self, path: Path, *, name: str, limit: int | None = None, force: bool = False
    ) -> RunReport:
        """Ingest a file or a directory as the source called ``name``."""
        ...

    async def sync(self, connector: str, *, limit: int | None = None) -> RunReport:
        """Run one configured connector."""
        ...

    async def reindex(self, document_id: str) -> ReindexReport:
        """Re-parse one document from its retained bytes. No network."""
        ...

    async def import_archive(self, path: Path, *, force: bool = False) -> RunReport:
        """Ingest an exported corpus archive.

        The archive carries retained source bytes and document metadata, never chunks or
        vectors, so the importing installation produces both with **its own** fingerprints.
        Copying chunks across machines would move an index built by another chunker and
        another embedder into a store whose fingerprint says otherwise.
        """
        ...


@runtime_checkable
class Maintenance(Protocol):
    """Whole-installation operations: the ones that touch files rather than rows."""

    async def schema_revision(self) -> str | None: ...

    async def backup(self, target: Path) -> Mapping[str, object]: ...

    async def restore(self, source: Path, *, force: bool = False) -> Mapping[str, object]: ...

    async def reset_index(self) -> tuple[int, int, bool]:
        """Empty the index. Returns documents removed, chunks removed, vectors removed."""
        ...

    async def export_corpus(self, target: Path) -> tuple[int, int]:
        """Write a portable archive of this workspace's retained bytes and metadata.

        Returns the number of documents written and the number of bytes they occupy.
        """
        ...

    async def workspaces(self) -> Sequence[tuple[str, str, str]]:
        """Every workspace this installation knows about: ``(id, name, mode)``.

        Names and modes only. A cross-workspace *listing* is configuration; a cross-workspace
        *read* is a tenancy breach, and this handle cannot perform one.
        """
        ...


@runtime_checkable
class Keys(Protocol):
    """API keys for one workspace.

    Workspace-scoped like everything else, and for the same reason: a key is an identity, and
    an identity that can be minted in one tenant and used in another is not isolation.
    """

    async def issue(
        self, name: str, *, role: str, expires_days: int | None = None
    ) -> tuple[ApiKeySummary, str]:
        """Mint a key. Returns its record and the secret, which exists only here.

        The secret is returned once and never stored — only a digest is. A lost key is
        reissued rather than recovered, which is the property that makes a leaked backup not
        also a leaked credential.
        """
        ...

    async def list_keys(self) -> Sequence[ApiKeySummary]: ...

    async def revoke(self, name_or_id: str) -> ApiKeySummary:
        """Revoke a key by name or id. Immediate."""
        ...


class Backend(Protocol):
    """Everything the service is given, built lazily and never by the service itself.

    Each accessor is async and may be expensive: ``retriever`` constructs an embedder,
    ``answerer`` loads a provider library. ``doctor`` and ``stats`` must not pay for either,
    which is why they are separate calls rather than fields — a field would be built by
    whoever assembled the backend, for every command.
    """

    @property
    def settings(self) -> Settings: ...

    @property
    def workspace(self) -> str: ...

    @property
    def discovery(self) -> Discovery | None:
        """What plugin discovery found, or ``None`` if it has not run."""
        ...

    async def documents(self) -> DocumentSurface: ...

    async def retriever(self) -> Retrieving: ...

    async def answerer(self) -> Answering: ...

    async def ingestion(self) -> Ingesting: ...

    async def maintenance(self) -> Maintenance: ...

    async def keys(self) -> Keys: ...

    async def component_checks(self) -> Sequence[Check]:
        """Health of whatever is already constructed, without constructing anything else."""
        ...


__all__ = [
    "Answering",
    "Backend",
    "DocumentSurface",
    "Ingesting",
    "Keys",
    "Maintenance",
    "Retrieving",
]
