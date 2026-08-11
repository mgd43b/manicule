"""Components that break their half of the bargain, so the guards can be seen firing.

Every fake here is written to be *wrong* in one specific, named way. That is the point: a
guard checked only against a component that behaves correctly is a guard nobody has watched
work, and this project has been bitten by exactly that.

The one to read carefully is :class:`LeakyStore`, which ignores its workspace scope entirely.
It exists so that ``tests/app/test_tenancy.py`` cannot be satisfied by a service that happens
not to be asked a cross-tenant question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, override

from manicule.app.ports import (
    Answering,
    DocumentSurface,
    Ingesting,
    Keys,
    Maintenance,
    Retrieving,
)
from manicule.app.results import ApiKeySummary, Check
from manicule.app.tenancy import belongs_to
from manicule.config.settings import Settings
from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus
from manicule.core.embedding import IndexFingerprints
from manicule.core.errors import UnknownEntityError
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.retrieval import Candidate, Confidence, ConfidenceBand, Context, Query
from manicule.generation.answers import AnswerEnvelope, AnswerEvent, EventKind
from manicule.ingest.pipeline import RunReport
from manicule.ingest.reindex import ReindexReport
from manicule.retrieval.retriever import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Mapping, Sequence

    from manicule.core.retrieval import Filter
    from manicule.generation.answering import AnswerRequest, AnswerResult
    from manicule.plugins.registry import Discovery


def make_document(
    workspace: str,
    *,
    source: str = "local",
    source_id: str = "notes.md",
    title: str = "Notes",
    status: DocumentStatus = DocumentStatus.INDEXED,
) -> Document:
    """A document whose id is derived the way the real one is.

    Derived rather than written down, so a test cannot be made to pass by editing a literal
    id until it matches. The identity under test *is* this function's second line.
    """
    return Document(
        id=document_id(workspace, source, source_id),
        source=source,
        source_id=source_id,
        uri=f"file:///corpus/{source_id}",
        title=title,
        content_hash=content_hash(f"{workspace}/{source_id}"),
        media_type="text/markdown",
        status=status,
    )


def make_chunk(document: Document, *, text: str = "the client retries twice") -> Chunk:
    """One chunk of a document, with a resolvable anchor."""
    return Chunk(
        id=chunk_id(document.id, 0, text),
        document_id=document.id,
        text=text,
        embed_text=text,
        anchor=HeadingAnchor(path=("Retry policy",)),
        heading_path=("Retry policy",),
        kind=BlockKind.PROSE,
        position=0,
        token_count=6,
    )


@dataclass
class FakeStore:
    """A well-behaved document store, scoped to one workspace.

    The control against which :class:`LeakyStore` is the experiment: everything a test asserts
    about a refusal has to be shown *not* to happen here, or the test is only proving that the
    service refuses everything.
    """

    workspace_id: str = "default"
    documents: dict[str, Document] = field(default_factory=dict[str, Document])
    chunks: dict[str, list[Chunk]] = field(default_factory=dict[str, list[Chunk]])
    deleted: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def add(self, document: Document, *chunks: Chunk) -> Document:
        self.documents[document.id] = document
        self.chunks[document.id] = list(chunks)
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        """A page, filtered the way a correct store filters it.

        Honouring ``document_ids`` and the workspace scope matters here rather than being
        pedantry about a fake: a correct store is the control the leaky one is measured
        against, and a control that also leaked would make the experiment meaningless.
        """
        wanted = list(self.documents.values())
        if filter is not None:
            wanted = [
                document
                for document in wanted
                if belongs_to(self.workspace_id, document)
                and (not filter.document_ids or document.id in filter.document_ids)
                and (not filter.sources or document.source in filter.sources)
                and (not filter.media_types or document.media_type in filter.media_types)
            ]
        return wanted[offset : offset + limit]

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        return self.chunks.get(document_id, [])

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
    ) -> int:
        chosen = [
            document
            for document in self.documents.values()
            if (source is None or document.source == source)
            and (statuses is None or document.status in statuses)
        ]
        return len(chosen)

    async def count_chunks(self, document_id: str | None = None) -> int:
        if document_id is not None:
            return len(self.chunks.get(document_id, []))
        return sum(len(chunks) for chunks in self.chunks.values())

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append((document_id, "hard"))
        self.documents.pop(document_id, None)

    async def soft_delete_document(self, document_id: str) -> None:
        self.deleted.append((document_id, "soft"))

    async def document_statistics(self) -> Mapping[str, Mapping[str, int]]:
        by_source: dict[str, int] = {}
        by_media: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for document in self.documents.values():
            by_source[document.source] = by_source.get(document.source, 0) + 1
            by_media[document.media_type] = by_media.get(document.media_type, 0) + 1
            by_status[document.status.value] = by_status.get(document.status.value, 0) + 1
        return {"by_source": by_source, "by_media_type": by_media, "by_status": by_status}

    async def index_fingerprints(self) -> IndexFingerprints:
        return IndexFingerprints()

    async def connector_metadata(self, connector: str) -> Mapping[str, object]:
        del connector
        return {}


class LeakyStore(FakeStore):
    """A store that ignores its workspace scope. **Deliberately broken.**

    Two ways at once, because a surface guard has to survive both:

    * ``get_document`` returns any document it holds, whatever workspace minted its id —
      inherited, because :class:`FakeStore` does not scope that lookup either.
    * ``list_documents`` returns **every** document it holds, ignoring the filter, the limit
      and the offset.

    Ignoring the *limit* as well as the filter is what makes this useful. A leaky store that
    still truncated would let the surface's "some of what I asked for came back missing" check
    catch a foreign document by accident, and the identity check — the one that would still
    fire if every ``WHERE`` clause in storage were deleted — would never be exercised.

    This is what a store written without the ``WHERE`` clause looks like, and it is the only
    way to see the surface's own check fire. A test that used a correct store would pass
    whether or not the check existed.
    """

    @override
    async def list_documents(
        self,
        filter: Filter | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        del filter, limit, offset
        return list(self.documents.values())


@dataclass
class FakeRetriever:
    """A retriever that returns exactly the candidates it was given."""

    candidates: list[Candidate] = field(default_factory=list[Candidate])
    seen: list[Query] = field(default_factory=list[Query])

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.seen.append(query)
        return RetrievalResult(
            context=Context(query=query, passages=tuple(self.candidates)),
            candidates=list(self.candidates),
            confidence=Confidence(
                score=0.5, band=ConfidenceBand.MEDIUM, reason="a fake, so this is a constant"
            ),
        )


@dataclass
class FakeAnswerer:
    """An answerer that emits one delta and a final envelope."""

    text: str = "The client retries twice."
    calls: list[object] = field(default_factory=list[object])

    def answer(
        self, request: AnswerRequest, result: AnswerResult | None = None
    ) -> AsyncIterator[AnswerEvent]:
        """Record the request, then hand back the stream.

        A plain function returning a generator rather than an async generator function,
        because the two have different *declared* return types and the protocol asks for an
        ``AsyncIterator``. The recording therefore happens when ``answer`` is called rather
        than when the stream is first iterated — which is what
        ``tests/app/test_tenancy.py`` relies on to prove the model was never reached.
        """
        del result
        self.calls.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[AnswerEvent]:
        yield AnswerEvent(kind=EventKind.DELTA, text=self.text)
        yield AnswerEvent(
            kind=EventKind.FINAL,
            envelope=AnswerEnvelope(text=self.text, corpus_consulted=True, confidence=0.5),
        )


@dataclass
class FakeIngestion:
    """An ingest surface that records what it was asked to do."""

    report: RunReport = field(default_factory=lambda: RunReport(connector="local", discovered=1))
    paths: list[Path] = field(default_factory=list[Path])
    synced: list[str] = field(default_factory=list[str])
    reindexed: list[str] = field(default_factory=list[str])
    imported: list[Path] = field(default_factory=list[Path])

    async def index_path(
        self, path: Path, *, name: str, limit: int | None = None, force: bool = False
    ) -> RunReport:
        del name, limit, force
        self.paths.append(path)
        return self.report

    async def sync(self, connector: str, *, limit: int | None = None) -> RunReport:
        del limit
        self.synced.append(connector)
        return self.report

    async def reindex(self, document_id: str) -> ReindexReport:
        self.reindexed.append(document_id)
        return ReindexReport(documents=1, chunks=3)

    async def import_archive(self, path: Path, *, force: bool = False) -> RunReport:
        del force
        self.imported.append(path)
        return self.report


@dataclass
class FakeMaintenance:
    """Whole-installation operations, recorded rather than performed."""

    revision: str | None = "abc123"
    workspace_rows: list[tuple[str, str, str]] = field(
        default_factory=lambda: [("default", "default", "personal")]
    )
    reset: tuple[int, int, bool] = (0, 0, False)

    async def schema_revision(self) -> str | None:
        return self.revision

    async def backup(self, target: Path) -> Mapping[str, object]:
        return {
            "created_at": "2026-01-01T00:00:00Z",
            "files": [],
            "counts": {},
            "path": str(target),
        }

    async def restore(self, source: Path, *, force: bool = False) -> Mapping[str, object]:
        del force
        return {"files": [], "path": str(source)}

    async def reset_index(self) -> tuple[int, int, bool]:
        return self.reset

    async def export_corpus(self, target: Path) -> tuple[int, int]:
        del target
        return 0, 0

    async def workspaces(self) -> Sequence[tuple[str, str, str]]:
        return list(self.workspace_rows)


@dataclass
class FakeKeys:
    """API keys, held in memory."""

    issued: list[ApiKeySummary] = field(default_factory=list[ApiKeySummary])
    workspace: str = "default"

    async def issue(
        self, name: str, *, role: str, expires_days: int | None = None
    ) -> tuple[ApiKeySummary, str]:
        del expires_days
        summary = ApiKeySummary(
            id=f"key-{len(self.issued)}",
            name=name,
            prefix="mnk_abc",
            role=role,
            workspace=self.workspace,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.issued.append(summary)
        return summary, "mnk_abcdefghijklmnop"

    async def list_keys(self) -> Sequence[ApiKeySummary]:
        return list(self.issued)

    async def revoke(self, name_or_id: str) -> ApiKeySummary:
        for summary in self.issued:
            if name_or_id in {summary.id, summary.name}:
                return summary
        msg = f"no API key named {name_or_id!r} in workspace {self.workspace!r}"
        raise UnknownEntityError(msg)


@dataclass
class FakeBackend:
    """Everything the service is given, assembled from the fakes above."""

    settings: Settings = field(default_factory=Settings)
    store: FakeStore = field(default_factory=FakeStore)
    retriever_: FakeRetriever = field(default_factory=FakeRetriever)
    answerer_: FakeAnswerer = field(default_factory=FakeAnswerer)
    ingestion_: FakeIngestion = field(default_factory=FakeIngestion)
    maintenance_: FakeMaintenance = field(default_factory=FakeMaintenance)
    keys_: FakeKeys = field(default_factory=FakeKeys)
    discovery: Discovery | None = None
    checks: list[Check] = field(default_factory=list[Check])

    @property
    def workspace(self) -> str:
        return self.settings.workspace

    async def documents(self) -> DocumentSurface:
        return self.store

    async def retriever(self) -> Retrieving:
        return self.retriever_

    async def answerer(self) -> Answering:
        return self.answerer_

    async def ingestion(self) -> Ingesting:
        return self.ingestion_

    async def maintenance(self) -> Maintenance:
        return self.maintenance_

    async def keys(self) -> Keys:
        return self.keys_

    async def component_checks(self) -> Sequence[Check]:
        return list(self.checks)


__all__ = [
    "FakeAnswerer",
    "FakeBackend",
    "FakeIngestion",
    "FakeKeys",
    "FakeMaintenance",
    "FakeRetriever",
    "FakeStore",
    "LeakyStore",
    "make_chunk",
    "make_document",
]
