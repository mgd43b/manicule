"""In-memory doubles for the pipeline, and the ones that misbehave on purpose.

Every guard in :mod:`manicule.ingest` has a fake here that breaks the rule the guard exists
for. That is the repo's pattern and it is not decoration: a pipeline whose whole purpose is
surviving failure is certified by nothing if only its happy path is exercised. Each of these
was checked by disabling the guard and watching the suite go red.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Collection, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import override

from manicule.connectors import CursorExpiredError
from manicule.core.acquisition import AcquiredSource
from manicule.core.anchors import LineAnchor
from manicule.core.content import (
    BlockKind,
    Chunk,
    Commit,
    Document,
    DocumentRevision,
    DocumentStatus,
    Metadata,
    ParsedBlock,
    RawDocument,
    Retention,
)
from manicule.core.embedding import (
    UNRECORDED_IDENTITY,
    EmbedFingerprint,
    IndexFingerprints,
    StoredVector,
    Vector,
    VectorState,
    choose_stored_vector,
    classify_stored_vector,
    embedding_input_identity,
)
from manicule.core.errors import ParseError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.glossary import GlossaryEntry
from manicule.core.ids import chunk_id, content_hash
from manicule.core.retrieval import Candidate, Filter
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark
from manicule.ingest.ports import GlossaryWriter
from manicule.ingest.workers import AttemptResult, InProcessRunner
from manicule.parsers.chain import Attempt, Outcome
from manicule.parsers.expansion import ExpandedMember, MemberFailure, MemberOutcome
from tests.fakes import MEDIA_TYPE, HashEmbedder

CONTAINER_MEDIA_TYPE = "application/x-fake-archive"


# --- gates ------------------------------------------------------------------------------------


class Gate:
    """A place work stops until a test lets it through, and a count of who was inside.

    **The reason every concurrency fake here is built on one of these rather than on a sleep.**
    A test that sleeps and then asserts "two fetches overlapped" passes when the machine is idle
    and fails on a loaded runner, and — much worse — passes against a sequential implementation
    often enough to look green. A gate makes the assertion about arrivals rather than about
    timing: :meth:`wait_for` returns only when the stated number of callers are all inside at
    once, and if they never are, it raises instead of guessing.

    :attr:`peak` is what a bound is asserted against, because a bound is a statement about a
    maximum and sampling ``inside`` is a statement about when somebody looked.
    """

    def __init__(self, *, capacity: int | None = None, opened: bool = False) -> None:
        self.arrivals = asyncio.Semaphore(0)
        self.inside = 0
        self.peak = 0
        self.entries = 0
        self.capacity = capacity
        """When set, arriving with this many already inside raises rather than being counted.

        A ceiling asserted at the moment it is crossed rather than afterwards from a peak. The
        two differ in what they say when the code is wrong: a peak reports a number, and this
        reports the document that was the straw.
        """

        self._open = asyncio.Event()
        if opened:
            self._open.set()

    @asynccontextmanager
    async def holding(self) -> AsyncGenerator[None]:
        """Count one caller for the length of the block, parking until the gate opens.

        An opened gate does not park and does not yield to the loop, so a caller is counted for
        exactly as long as the work inside the block takes — which is what makes the same object
        serve both "hold everything until I say" and "just tell me how many overlapped".
        """
        self.inside += 1
        self.peak = max(self.peak, self.inside)
        self.entries += 1
        try:
            if self.capacity is not None and self.inside > self.capacity:
                msg = f"{self.inside} callers inside a gate with room for {self.capacity}"
                raise AssertionError(msg)
            self.arrivals.release()
            await self._open.wait()
            yield
        finally:
            self.inside -= 1

    async def pass_through(self) -> None:
        """Arrive, wait for the gate to open, and leave."""
        async with self.holding():
            pass

    async def wait_for(self, callers: int, *, patience_s: float = 10.0) -> None:
        """Block until ``callers`` have arrived, or fail saying how many ever did.

        ``patience_s`` is how long the *test* is willing to be wrong for, not a timeout on
        anything under test — which is why the expiry is an assertion failure naming the peak
        rather than a ``TimeoutError`` a caller would have to interpret. A caller wrapping this
        in its own ``asyncio.timeout`` would get the bare error and none of the diagnosis.

        Raises:
            AssertionError: They did not arrive. That is what a sequential implementation looks
                like from here, and it is the failure this whole file exists to produce.
        """
        try:
            async with asyncio.timeout(patience_s):
                for _ in range(callers):
                    await self.arrivals.acquire()
        except TimeoutError:
            msg = (
                f"waited for {callers} callers to be inside the gate at once; the most that ever "
                f"were is {self.peak}, from {self.entries} arrival(s)"
            )
            raise AssertionError(msg) from None

    def open(self) -> None:
        """Let everyone waiting through, and everyone who arrives later."""
        self._open.set()


# --- stores --------------------------------------------------------------------------------


class MemoryIngestStore:
    """Everything :class:`~manicule.ingest.ports.IngestStore` promises, in dictionaries.

    Real enough to be worth testing against: it enforces the one invariant the pipeline
    depends on — that deleting chunks writes tombstones — because a fake that skipped it would
    make the sweep's tests pass while proving nothing.
    """

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.tombstones: list[str] = []
        self.metadata: dict[str, Metadata] = {}
        self.lineage: dict[str, tuple[str | None, str | None]] = {}
        self.parse_lineage: dict[str, str] = {}
        """``documents.parse_fp``, kept beside the documents rather than in them.

        The real store never writes lineage from a domain object — the pipeline builds a fresh
        one per ingest and cannot know a parse fingerprint before the chain has picked a
        parser — so a fake that let ``upsert_document`` carry it would clear the lineage at the
        start of every run and make change detection pass for the wrong reason.
        """
        self.glossary_lineage_by_id: dict[str, str] = {}
        """``documents.glossary_fp``, beside the documents for the same reason ``parse_fp`` is.

        The domain :class:`~manicule.core.content.Document` deliberately does not carry it —
        change detection must not consult it, because a detector change is not a reason to
        re-parse a document — so a fake that let ``upsert_document`` carry it would let a test
        pass by a route the real store does not have.
        """

        self.originals: dict[str, tuple[str | None, str | None]] = {}
        self.seen: dict[str, int] = {}
        self.deleted_at: dict[str, datetime] = {}
        self.updated_at: dict[str, datetime] = {}
        self.watermarks: dict[str, Watermark] = {}
        self.connector_meta: dict[str, Metadata] = {}
        self.state = IndexFingerprints()
        self.staged_publications: list[tuple[str, tuple[str, ...]]] = []

    # documents

    def _with_lineage(self, document: Document) -> Document:
        """A document as a reader sees it: with the lineage the store holds, not the caller's."""
        return document.model_copy(update={"parse_fp": self.parse_lineage.get(document.id)})

    async def get_document(self, document_id: str) -> Document | None:
        stored = self.documents.get(document_id)
        return None if stored is None else self._with_lineage(stored)

    async def find_document(self, source: str, source_id: SourceId) -> Document | None:
        for document in self.documents.values():
            if (
                document.source == source
                and document.source_id == source_id
                and document.id not in self.deleted_at
            ):
                return self._with_lineage(document)
        return None

    async def upsert_document(self, document: Document) -> Document:
        self.documents[document.id] = document
        self.deleted_at.pop(document.id, None)
        return self._with_lineage(document)

    async def commit_document(self, document: Document, *, expected: DocumentRevision) -> Commit:
        """The same write, refused when the stored document is no longer ``expected``.

        Atomic here for a reason the real store has to work for: nothing between the comparison
        and the write awaits, so no other task can be scheduled in between. ``upsert_document``
        below never suspends either, which is what makes calling it rather than repeating its
        body safe — and what would stop being true if it grew an ``await`` that did.
        """
        stored = self.documents.get(document.id)
        current = (
            None if stored is None or document.id in self.deleted_at else self._with_lineage(stored)
        )
        if current is None or current.revision != expected:
            return Commit(committed=False, stored=current)
        return Commit(committed=True, stored=await self.upsert_document(document))

    async def stage_vectors(self, publication_id: str, chunks: Sequence[Chunk]) -> None:
        self.staged_publications.append((publication_id, tuple(chunk.id for chunk in chunks)))

    async def publish_failure(
        self,
        document: Document,
        *,
        expected: DocumentRevision | None,
        original_omitted_reason: str | None,
    ) -> Commit:
        current = await self.get_document(document.id)
        if expected is not None and (current is None or current.revision != expected):
            return Commit(committed=False, stored=current)
        stored = await self.upsert_document(document)
        self.originals[document.id] = (document.original_ref, original_omitted_reason)
        return Commit(committed=True, stored=stored)

    async def publish_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        *,
        expected: DocumentRevision | None,
        chunk_fp: str | None,
        embed_fp: str | None,
        parse_fp: str | None,
        glossary_entries: Sequence[GlossaryEntry] | None,
        glossary_fp: str | None,
        original_omitted_reason: str | None,
    ) -> Commit:
        current = await self.get_document(document.id)
        if expected is not None and (current is None or current.revision != expected):
            return Commit(committed=False, stored=current)
        await self.upsert_document(document)
        await self.replace_chunks(document.id, chunks)
        self.lineage[document.id] = (chunk_fp, embed_fp)
        if parse_fp is None:
            self.parse_lineage.pop(document.id, None)
            self.documents[document.id] = self.documents[document.id].model_copy(
                update={"parse_fp": None}
            )
        else:
            self.parse_lineage[document.id] = parse_fp
            self.documents[document.id] = self.documents[document.id].model_copy(
                update={"parse_fp": parse_fp}
            )
        if glossary_entries is not None and isinstance(self, GlossaryWriter):
            await self.replace_glossary_entries(
                document.id, glossary_entries, fingerprint=glossary_fp or ""
            )
        if glossary_fp is not None:
            self.glossary_lineage_by_id[document.id] = glossary_fp
        self.originals[document.id] = (document.original_ref, original_omitted_reason)
        return Commit(committed=True, stored=self._with_lineage(self.documents[document.id]))

    async def set_status(self, document_id: str, status: DocumentStatus, detail: str = "") -> None:
        existing = self.documents.get(document_id)
        if existing is None:
            return
        self.documents[document_id] = existing.model_copy(
            update={
                "status": status,
                "status_detail": detail or None,
                "failed_stage": existing.failed_stage if status is DocumentStatus.FAILED else None,
            }
        )

    async def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        await self.replace_chunks(document_id, [])

    async def soft_delete_document(self, document_id: str) -> None:
        self.deleted_at[document_id] = UTC_ZERO

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        del filter
        return list(self.documents.values())[offset : offset + limit]

    # chunks

    async def replace_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        for stale in self.chunks.get(document_id, []):
            self.tombstones.append(stale.id)
        self.chunks[document_id] = list(chunks)
        for chunk in chunks:
            if chunk.id in self.tombstones:
                self.tombstones.remove(chunk.id)

    async def get_chunks(self, chunk_ids: Sequence[str]) -> Sequence[Chunk]:
        wanted = set(chunk_ids)
        return [c for chunks in self.chunks.values() for c in chunks if c.id in wanted]

    async def count_chunks(self, document_id: str | None = None) -> int:
        if document_id is not None:
            return len(self.chunks.get(document_id, []))
        return sum(len(chunks) for chunks in self.chunks.values())

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        return list(self.chunks.get(document_id, []))

    # bookkeeping

    async def record_seen(self, document_id: str, *, version_token: str | None = None) -> None:
        self.seen[document_id] = self.seen.get(document_id, 0) + 1
        if version_token is not None and document_id in self.documents:
            self.documents[document_id] = self.documents[document_id].model_copy(
                update={"version_token": version_token}
            )

    async def annotate(self, document_id: str, updates: Metadata) -> None:
        """Merge keys into a document's metadata, as :class:`SqliteDocStore` does.

        **This used to write to a side dictionary that nothing ever read**, so an annotation
        vanished the moment it was made and any test asserting on one through this fake was
        asserting nothing. The real store merges into ``documents.metadata`` and
        ``rows.to_document`` reads it straight back, so a double that kept annotations somewhere
        else diverged from the thing it stands in for on the one operation it exists to model.
        Found while asserting that a metadata-precedence change did not erase accumulated state —
        the assertion failed against the fake and passed against the real store.
        """
        merged: Metadata = dict(self.metadata.get(document_id, {}))
        merged.update(updates)
        self.metadata[document_id] = merged
        stored = self.documents.get(document_id)
        if stored is not None:
            combined: Metadata = dict(stored.metadata)
            combined.update(updates)
            self.documents[document_id] = stored.model_copy(update={"metadata": combined})

    async def set_lineage(
        self,
        document_id: str,
        *,
        chunk_fp: str | None,
        embed_fp: str | None,
        parse_fp: str | None = None,
        glossary_fp: str | None = None,
    ) -> None:
        current = self.lineage.get(document_id, (None, None))
        self.lineage[document_id] = (
            chunk_fp if chunk_fp is not None else current[0],
            embed_fp if embed_fp is not None else current[1],
        )
        if parse_fp is not None:
            self.parse_lineage[document_id] = parse_fp
        if glossary_fp is not None:
            self.glossary_lineage_by_id[document_id] = glossary_fp

    async def set_original(
        self, document_id: str, *, ref: str | None, omitted_reason: str | None
    ) -> None:
        self.originals[document_id] = (ref, omitted_reason)

    async def requeue_stale(
        self,
        statuses: Collection[DocumentStatus],
        older_than: datetime,
        *,
        detail: str = "",
    ) -> int:
        wanted = set(statuses)
        requeued = 0
        for document_id, document in list(self.documents.items()):
            stamp = self.updated_at.get(document_id)
            if document.status in wanted and stamp is not None and stamp < older_than:
                self.documents[document_id] = document.model_copy(
                    update={
                        "status": DocumentStatus.PENDING,
                        "status_detail": detail or None,
                        "failed_stage": None,
                    }
                )
                requeued += 1
        return requeued

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        glossary_fp_other_than: str | None = None,
        glossary_fp_unrecorded: bool = False,
    ) -> int:
        found = await self.select_documents(
            source=source, statuses=statuses, glossary_fp_other_than=glossary_fp_other_than
        )
        if glossary_fp_unrecorded:
            found = [d for d in found if d.id not in self.glossary_lineage_by_id]
        return len(found)

    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        parse_fp_current: Collection[str] | None = None,
        glossary_fp_other_than: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Document]:
        chosen = [
            self._with_lineage(d) for d in self.documents.values() if d.id not in self.deleted_at
        ]
        if source is not None:
            chosen = [d for d in chosen if d.source == source]
        if statuses is not None:
            wanted = set(statuses)
            chosen = [d for d in chosen if d.status in wanted]
        if media_types is not None:
            allowed = set(media_types)
            chosen = [d for d in chosen if d.media_type in allowed]
        if chunk_fp_other_than is not None:
            chosen = [
                d for d in chosen if self.lineage.get(d.id, (None, None))[0] != chunk_fp_other_than
            ]
        if parse_fp_current is not None:
            current = set(parse_fp_current)
            chosen = [d for d in chosen if d.parse_fp is None or d.parse_fp not in current]
        if glossary_fp_other_than is not None:
            # Unrecorded counts as different, which is the half of the predicate a fake is most
            # likely to get wrong: `.get(id) != x` happens to be right here only because the
            # default is `None`, and the real store has to say `IS NULL OR <> x` because SQL
            # would otherwise drop those rows on three-valued logic.
            chosen = [
                d for d in chosen if self.glossary_lineage_by_id.get(d.id) != glossary_fp_other_than
            ]
        # Sliced after every predicate, in that order, because the store applies `OFFSET` to the
        # filtered result and a fake that skipped first would page through a different set.
        chosen = chosen[offset:]
        return chosen[:limit] if limit is not None else chosen

    # sync state

    async def get_watermark(self, connector: str) -> Watermark | None:
        return self.watermarks.get(connector)

    async def set_watermark(self, connector: str, watermark: Watermark) -> None:
        self.watermarks[connector] = watermark

    async def known_source_ids(self, connector: str) -> AsyncIterator[SourceId]:
        for document in list(self.documents.values()):
            if document.source == connector and document.id not in self.deleted_at:
                yield document.source_id

    async def connector_metadata(self, connector: str) -> Metadata:
        return dict(self.connector_meta.get(connector, {}))

    async def record_connector_metadata(self, connector: str, updates: Metadata) -> None:
        merged = dict(self.connector_meta.get(connector, {}))
        for key, value in updates.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        self.connector_meta[connector] = merged

    # index state

    async def index_fingerprints(self) -> IndexFingerprints:
        return self.state

    async def record_index_fingerprints(self, state: IndexFingerprints) -> None:
        self.state = state

    # the sweep

    async def take_tombstones(self, limit: int) -> Sequence[str]:
        return self.tombstones[:limit]

    async def clear_tombstones(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id_ in chunk_ids:
            if chunk_id_ in self.tombstones:
                self.tombstones.remove(chunk_id_)

    async def soft_deleted_before(self, cutoff: datetime, *, limit: int = 1000) -> Sequence[str]:
        return [
            document_id for document_id, stamp in list(self.deleted_at.items()) if stamp < cutoff
        ][:limit]


class MemoryGlossaryStore(MemoryIngestStore):
    """An ingest store that can also hold the definitions a document states.

    Kept apart from :class:`MemoryIngestStore` for the reason
    :class:`~manicule.ingest.ports.GlossaryWriter` is kept apart from ``IngestStore``: the
    pipeline decides whether it has somewhere to put definitions by a structural check on the
    store it was given, so a fake that folded the method in would switch detection on for every
    pipeline test, including the ones that are about something else entirely.
    """

    def __init__(self) -> None:
        super().__init__()
        self.glossary: dict[str, list[GlossaryEntry]] = {}

    async def replace_glossary_entries(
        self, document_id: str, entries: Sequence[GlossaryEntry], *, fingerprint: str
    ) -> None:
        """Both writes together, because the real store makes them one transaction.

        A fake that stored the rows and left the fingerprint to a second call would let a
        pipeline that forgot the second call pass — and a document holding entries with no
        recorded detector is the state that versioning them exists to make unreachable.
        """
        self.glossary[document_id] = list(entries)
        self.glossary_lineage_by_id[document_id] = fingerprint

    async def glossary_entries(self, document_id: str) -> Sequence[GlossaryEntry]:
        return list(self.glossary.get(document_id, []))

    async def glossary_lineage(self, document_id: str) -> str | None:
        return self.glossary_lineage_by_id.get(document_id)


UTC_ZERO = datetime.fromtimestamp(0, tz=UTC)
"""A fixed instant, so a soft delete in a fake is always outside any grace period."""


@dataclass
class VectorRow:
    """One stored row, holding what a Lance row holds.

    The chunk travels with the vector because a real row carries ``chunk_json``, and the
    embedding-input identity travels with both because a real row carries ``embed_identity``.
    A fake that kept only the vector could not answer whether a stored vector still describes
    a chunk, which is the question the reuse path is about.
    """

    document_id: str
    vector: Vector
    embed_text: str
    identity: str


class MemoryVectors:
    """A vector store that records what it was asked to remove."""

    def __init__(self) -> None:
        self.rows: dict[str, VectorRow] = {}
        self.deleted_documents: list[str] = []
        self._fingerprint: EmbedFingerprint | None = None
        self._middleware: tuple[str, ...] = ()

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        self._fingerprint = fingerprint
        self._middleware = tuple(embed_text_middleware)

    async def fingerprint(self) -> EmbedFingerprint | None:
        return self._fingerprint

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        del publication_id
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.rows[chunk.id] = VectorRow(
                document_id=chunk.document_id,
                # Stored as a tuple whatever the caller handed over. A real row is a typed
                # column and reads back the same shape however it was written, so a fake that
                # kept the caller's container would make a reused vector — which arrives as a
                # tuple — compare unequal to the identical vector it replaced.
                vector=tuple(vector),
                embed_text=chunk.embed_text,
                identity=self._identity_of(chunk),
            )

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        verdicts: dict[str, StoredVector] = {}
        for chunk in chunks:
            if self._fingerprint is None:
                verdicts[chunk.id] = StoredVector(state=VectorState.ABSENT)
                continue
            verdicts[chunk.id] = choose_stored_vector(
                self._classify(chunk, self.rows.get(chunk.id)),
                self._classify(chunk, self._row_by_identity(self._identity_of(chunk))),
            )
        return verdicts

    def _classify(self, chunk: Chunk, row: VectorRow | None) -> StoredVector:
        if row is None or self._fingerprint is None:
            return StoredVector(state=VectorState.ABSENT)
        return classify_stored_vector(
            chunk,
            recorded_identity=row.identity,
            stored_embed_text=row.embed_text,
            stored_vector=list(row.vector),
            embed=self._fingerprint,
            middleware=self._middleware,
        )

    def _row_by_identity(self, identity: str) -> VectorRow | None:
        """Any row recorded against ``identity``, whichever chunk id it is filed under.

        A scan, because this is a fake and the set is small; the real store has a column to
        query. What matters is that both answer the same question — a chunk id carries its
        position, so a document with a paragraph inserted at the top renames every chunk below
        it without moving one embedding input.
        """
        if identity == UNRECORDED_IDENTITY:
            return None
        return next((row for row in self.rows.values() if row.identity == identity), None)

    def _identity_of(self, chunk: Chunk) -> str:
        if self._fingerprint is None:
            return UNRECORDED_IDENTITY
        return embedding_input_identity(
            chunk.embed_text,
            document_id=chunk.document_id,
            embed=self._fingerprint,
            middleware=self._middleware,
        )

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002
    ) -> list[Candidate]:
        del vector, k, filter
        return []

    async def delete_document(self, document_id: str) -> None:
        self.deleted_documents.append(document_id)
        for key, row in list(self.rows.items()):
            if row.document_id == document_id:
                del self.rows[key]

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id_ in chunk_ids:
            self.rows.pop(chunk_id_, None)

    async def count(self) -> int:
        return len(self.rows)


class RefusingVectors(MemoryVectors):
    """Fails every upsert, so the crash window between chunks and vectors is reachable."""

    @override
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        del chunks, vectors, publication_id
        msg = "the vector store is unavailable"
        raise RuntimeError(msg)


# --- parsers -------------------------------------------------------------------------------


class LineParser:
    """One block per line."""

    media_types = frozenset({MEDIA_TYPE})

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for number, line in enumerate(raw.as_text().splitlines(), start=1):
            if line.strip():
                yield ParsedBlock(
                    kind=BlockKind.PROSE, text=line, anchor=LineAnchor(start=number, end=number)
                )

    async def resolve(self, anchor: object, raw: RawDocument) -> str | None:
        del anchor, raw
        return None


class ExplodingParser(LineParser):
    """Raises something that is not a decline. The chain must advance and record ``failed``."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        msg = "the parser library segfaulted, metaphorically"
        raise RuntimeError(msg)
        yield  # pragma: no cover - unreachable, present to make this an async generator


class DecliningParser(LineParser):
    """Inspects the input and reports that it is not its kind."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        raise ParseError("not my kind of document")
        yield  # pragma: no cover - unreachable, present to make this an async generator


class EmptyParser(LineParser):
    """Succeeds and produces nothing, the way a scanned PDF does."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        return
        yield  # pragma: no cover - unreachable, present to make this an async generator


class FakeArchive:
    """A container whose members are named in its own body, one per line."""

    media_types = frozenset({CONTAINER_MEDIA_TYPE})

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        return
        yield  # pragma: no cover - unreachable, present to make this an async generator

    async def resolve(self, anchor: object, raw: RawDocument) -> str | None:
        del anchor, raw
        return None

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        if raw.media_type != CONTAINER_MEDIA_TYPE:
            raise ParseError("not an archive")
        for line in raw.as_text().splitlines():
            name, _, body = line.partition("=")
            if not name:
                continue
            if body == "!encrypted":
                yield MemberFailure(
                    source_id=f"{raw.source_id}!/{name}",
                    uri=f"fake:{raw.uri}!/{name}",
                    status=DocumentStatus.FAILED,
                    reason="member is encrypted",
                    depth=1,
                )
                continue
            yield ExpandedMember(
                source_id=f"{raw.source_id}!/{name}",
                uri=f"fake:{raw.uri}!/{name}",
                raw=RawDocument(
                    source_id=f"{raw.source_id}!/{name}",
                    uri=f"fake:{raw.uri}!/{name}",
                    media_type=MEDIA_TYPE,
                    content=body,
                ),
                depth=1,
            )


# --- chunker and embedder -------------------------------------------------------------------


class BlockChunker:
    """One chunk per block, with a breadcrumb on ``embed_text``."""

    fingerprint = ChunkFingerprint(
        chunker="block",
        version="1",
        max_tokens=64,
        overlap_tokens=0,
        tokenizer_id="whitespace",
    )

    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        return [
            Chunk(
                id=chunk_id(document.id, position, block.text),
                document_id=document.id,
                text=block.text,
                embed_text=f"{document.title} > {block.text}",
                anchor=block.anchor,
                heading_path=block.heading_path,
                kind=block.kind,
                position=position,
                token_count=max(1, len(block.text.split())),
            )
            for position, block in enumerate(blocks)
        ]


class RefusingEmbedder(HashEmbedder):
    """Fails every batch, so the embed stage's failure path is reachable."""

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        del texts
        msg = "the model runtime is out of memory"
        raise RuntimeError(msg)


class CountingEmbedder(HashEmbedder):
    """Records the size of every batch it was given."""

    def __init__(self, dimension: int = 5) -> None:
        super().__init__(dimension=dimension)
        self.batches: list[int] = []

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.batches.append(len(texts))
        return await super().embed(texts)


class ExclusiveEmbedder(CountingEmbedder):
    """Refuses to be called while it is already inside a call, and says which document did it.

    **The strongest available statement of "the model is serialized"**, and it is stronger than
    a peak counter for one reason: it fails at the moment of overlap, in the task that caused
    it, rather than reporting a number afterwards that somebody has to interpret. With one
    accelerator and one unified-memory pool, a second concurrent batch is contention rather than
    throughput (``docs/ingest.md`` §6.6), so overlap is a defect and not a slow path.

    ``inside`` is written and read with no ``await`` between, which is what makes the check
    sound on an event loop: two coroutines cannot interleave a read-modify-write.
    """

    def __init__(self, dimension: int = 5) -> None:
        super().__init__(dimension=dimension)
        self.inside = 0
        self.overlaps = 0

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.inside:
            self.overlaps += 1
            msg = f"a second batch of {len(texts)} reached the embedder while one was running"
            raise AssertionError(msg)
        self.inside += 1
        try:
            return await super().embed(texts)
        finally:
            self.inside -= 1


class GatedEmbedder(ExclusiveEmbedder):
    """Serialized, and parked inside the model until a test lets it out.

    What makes the embedder the bottleneck on purpose, which is the only way to observe what
    the stages in front of it do when the stage behind them stops.
    """

    def __init__(self, dimension: int = 5) -> None:
        super().__init__(dimension=dimension)
        self.gate = Gate()

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.inside:
            self.overlaps += 1
            msg = f"a second batch of {len(texts)} reached the embedder while one was running"
            raise AssertionError(msg)
        self.inside += 1
        try:
            await self.gate.pass_through()
            return await HashEmbedder.embed(self, texts)
        finally:
            self.inside -= 1


# --- parse runners ------------------------------------------------------------------------------


class GatedRunner:
    """A parse runner that parks every attempt, so parse concurrency is observable.

    Stands in for :class:`~manicule.ingest.workers.WorkerPool` rather than for a parser: what is
    being measured is how many attempts one connector sync has in the pool at once, and a real
    pool would make that a fact about subprocess scheduling instead of about the pipeline.
    """

    def __init__(self, parsers: Mapping[str, object], *, capacity: int | None = None) -> None:
        self._inner = InProcessRunner(parsers)
        self.gate = Gate(capacity=capacity)

    async def run_attempt(self, name: str, raw: RawDocument) -> AttemptResult:
        await self.gate.pass_through()
        return await self._inner.run_attempt(name, raw)


class BrokenRunner:
    """A runner that reports one named document's attempt as a killed worker.

    The shape :class:`~manicule.ingest.workers.WorkerPool` produces when a parser overruns its
    deadline or its memory limit: a *hard failure* naming the limit, never a decline. Reproduced
    here rather than by killing a real worker because what is under test is that one document's
    dead worker does not disturb the documents beside it.
    """

    def __init__(self, parsers: Mapping[str, object], *, kill: str) -> None:
        self._inner = InProcessRunner(parsers)
        self._kill = kill
        self.killed: list[str] = []

    async def run_attempt(self, name: str, raw: RawDocument) -> AttemptResult:
        if raw.source_id == self._kill:
            self.killed.append(raw.source_id)
            return AttemptResult(
                [], Attempt(parser=name, outcome=Outcome.FAILED, reason="worker killed: timeout")
            )
        return await self._inner.run_attempt(name, raw)


# --- connectors ------------------------------------------------------------------------------


class DictConnector:
    """A connector over a dictionary, with a settable version token per document."""

    def __init__(self, documents: Mapping[str, str], *, name: str = "memory") -> None:
        self.name = name
        self.documents = dict(documents)
        self.media_types: dict[str, str] = {}
        self.fetches: list[str] = []
        self.fail_fetch: set[str] = set()
        self.reconcile_fails_after: int | None = None
        self.hidden: set[str] = set()
        self.metadata: dict[str, Metadata] = {}
        """What this connector attaches to fetched bytes, per source id.

        A dictionary is about as far from a filesystem as a source gets, which is why the
        source-metadata tests drive the pipeline through *this* connector rather than through the
        one that reads sidecar manifests. If a record only worked when it came from a real
        directory, the interface would be coupled to the connector that happens to populate it
        today, and the next connector to supply one would need the pipeline changed.
        """
        self.tokens: dict[str, str] = {}
        """Overrides the content-derived version token, per source id.

        Real change signals move independently of the bytes: a page's version number moves when
        its title is corrected. Without an override, every test is confined to the case where a
        token and a content hash always agree — which is exactly the case that hides a document
        whose metadata went stale while its bytes stood still.
        """

    @property
    def watermark(self) -> Watermark | None:
        """Nothing: a dictionary has no change signal to report a position from.

        The honest answer for a source with no native watermark, and the one that keeps the
        pipeline re-enumerating rather than believing an invented position.
        """
        return None

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        for source_id, text in sorted(self.documents.items()):
            yield DiscoveredDoc(
                ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
                version_token=self.tokens.get(source_id, content_hash(text)),
                media_type=self.media_types.get(source_id, MEDIA_TYPE),
            )

    async def fetch(self, ref: DocRef) -> RawDocument:
        self.fetches.append(ref.source_id)
        if ref.source_id in self.fail_fetch:
            msg = f"503 fetching {ref.source_id}"
            raise RuntimeError(msg)
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=self.media_types.get(ref.source_id, MEDIA_TYPE),
            content=self.documents[ref.source_id],
            metadata={
                **self.metadata.get(ref.source_id, {}),
                "version_token": self.tokens.get(
                    ref.source_id, content_hash(self.documents[ref.source_id])
                ),
            },
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        emitted = 0
        for source_id in sorted(self.documents):
            if source_id in self.hidden:
                continue
            if self.reconcile_fails_after is not None and emitted >= self.reconcile_fails_after:
                msg = "429 while enumerating"
                raise RuntimeError(msg)
            emitted += 1
            yield source_id


class ObservedConnector(DictConnector):
    """A connector that counts what it yielded and can park inside every fetch.

    Two observations, and they answer the two halves of "is the fetch stage really concurrent":
    :attr:`fetching` says how many fetches overlapped, and :attr:`yields` says how far ahead of
    durable progress the source was paged. The second is the one an unbounded design gets wrong
    while looking fine — it is the count that decides whether a pagination cursor lives long
    enough to be used (``docs/connectors/confluence.md`` §2).
    """

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        name: str = "memory",
        fetch_capacity: int | None = None,
        park_fetches: bool = False,
    ) -> None:
        super().__init__(documents, name=name)
        self.fetching = Gate(capacity=fetch_capacity, opened=not park_fetches)
        self.yields = 0
        self.yielded = asyncio.Semaphore(0)
        """Released as each document leaves ``discover``, before the pipeline has taken it.

        A permit therefore means "the source produced this one", which is the quantity a
        backpressure claim is about. Counting acceptances instead would be counting the
        pipeline's own bookkeeping back to itself.
        """

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        async for found in super().discover(watermark):
            self.yields += 1
            self.yielded.release()
            yield found

    @override
    async def fetch(self, ref: DocRef) -> RawDocument:
        async with self.fetching.holding():
            return await super().fetch(ref)


class ManualClock:
    """A monotonic clock advanced by a test, never by wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward by exactly ``seconds`` without sleeping."""
        self.now += seconds


class ManualLeaseClock:
    """A timezone-aware acquisition clock advanced without sleeping."""

    def __init__(self) -> None:
        self.now = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class ClockedGatedEmbedder(GatedEmbedder):
    """A parked embedder that advances virtual time once per completed document batch."""

    def __init__(self, clock: ManualClock, *, seconds_per_document: float = 0.05) -> None:
        super().__init__()
        self.clock = clock
        self.seconds_per_document = seconds_per_document

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        embedded = await super().embed(texts)
        self.clock.advance(self.seconds_per_document)
        return embedded


class ExpiringCursorConnector(ObservedConnector):
    """A paginated source whose next cursor expires while its consumer is suspended.

    This is deliberately a source-level fixture rather than a Confluence fake. It isolates the
    contract at the pipeline boundary: a page response issues its next cursor, yielding the
    page suspends inside ``discover``, and asking for the following page validates how long the
    consumer held that cursor. The manual clock and :attr:`cursor_issued` event make every
    transition test-controlled.
    """

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        clock: ManualClock,
        page_size: int,
        cursor_lifetime_seconds: float,
        response_seconds: float = 0.019,
    ) -> None:
        super().__init__(documents, name="synthetic-cursor-source")
        self.clock = clock
        self.page_size = page_size
        self.cursor_lifetime_seconds = cursor_lifetime_seconds
        self.response_seconds = response_seconds
        self.cursor_issued = asyncio.Event()
        self.enumeration_completed = asyncio.Event()
        self.cursors_issued = 0
        self.pages_requested = 0

    @property
    @override
    def watermark(self) -> Watermark:
        return Watermark(
            value="synthetic-position-1",
            observed_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        )

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        source_ids = sorted(self.documents)
        for start in range(0, len(source_ids), self.page_size):
            self.pages_requested += 1
            self.clock.advance(self.response_seconds)
            page = source_ids[start : start + self.page_size]
            has_next = start + self.page_size < len(source_ids)
            received_at = self.clock()
            if has_next:
                self.cursors_issued += 1
                self.cursor_issued.set()

            for source_id in page:
                self.yields += 1
                self.yielded.release()
                yield DiscoveredDoc(
                    ref=DocRef(
                        source_id=source_id,
                        uri=f"https://source.example.test/documents/{source_id}",
                    ),
                    version_token=self.tokens.get(
                        source_id, content_hash(self.documents[source_id])
                    ),
                    media_type=self.media_types.get(source_id, MEDIA_TYPE),
                )

            if has_next:
                held = self.clock() - received_at
                if held > self.cursor_lifetime_seconds:
                    msg = (
                        f"a synthetic search cursor was held for {held:g}s, longer than its "
                        f"{self.cursor_lifetime_seconds:g}s lifetime"
                    )
                    raise CursorExpiredError(msg)
        self.enumeration_completed.set()


class PausedEnumerationConnector(ObservedConnector):
    """A source that parks before requesting the record after a durable prefix."""

    def __init__(self, documents: Mapping[str, str], *, pause_after: int) -> None:
        super().__init__(documents, name="paused-synthetic-source")
        self.pause_after = pause_after
        self.paused = asyncio.Event()
        self.release = asyncio.Event()

    @property
    @override
    def watermark(self) -> Watermark:
        return Watermark(
            value="paused-position-1",
            observed_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        )

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        emitted = 0
        async for discovered in super().discover(watermark):
            yield discovered.model_copy(
                update={
                    "ref": discovered.ref.model_copy(
                        update={
                            "uri": (f"https://source.example.test/documents/{discovered.source_id}")
                        }
                    )
                }
            )
            emitted += 1
            if emitted == self.pause_after:
                self.paused.set()
                await self.release.wait()


# --- middleware --------------------------------------------------------------------------------


class PassThrough:
    """Touches nothing. The baseline every check must accept."""

    name = "pass-through"
    mutates_embedded_text = False

    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        return raw

    async def after_parse(self, document: Document, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        del document
        return blocks

    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return chunks

    async def after_store(self, document: Document) -> None:
        del document


class DiscardingHook(PassThrough):
    """Returns ``None`` from ``after_chunk``, where a value was required."""

    name = "discarding"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document, chunks
        return None  # pyright: ignore[reportReturnType] - the defect under test


class WrongTypeHook(PassThrough):
    """Returns something that is not a ``RawDocument`` from ``before_parse``."""

    name = "wrong-type"

    @override
    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        del raw
        return "just some text"  # pyright: ignore[reportReturnType] - the defect under test


class TextRewriter(PassThrough):
    """Rewrites ``Chunk.text``, which no middleware may ever do."""

    name = "text-rewriter"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return [c.model_copy(update={"text": c.text.upper()}) for c in chunks]


class BlockRewriter(PassThrough):
    """Rewrites ``ParsedBlock.text``: the same corruption, one hook earlier."""

    name = "block-rewriter"

    @override
    async def after_parse(self, document: Document, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        del document
        return [b.model_copy(update={"text": b.text.upper()}) for b in blocks]


class UndeclaredRewriter(PassThrough):
    """Rewrites ``embed_text`` without declaring it, so no fingerprint describes the corpus."""

    name = "undeclared"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return [c.model_copy(update={"embed_text": c.embed_text + " extra"}) for c in chunks]


class DeclaredRewriter(UndeclaredRewriter):
    """The same rewrite, declared. The legitimate case."""

    name = "declared"
    mutates_embedded_text = True


class Skipper(PassThrough):
    """Excludes every document before parsing."""

    name = "skipper"

    @override
    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        del raw
        return None


class Exploder(PassThrough):
    """Raises. One document fails; the batch does not."""

    name = "exploder"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document, chunks
        msg = "the hook could not reach its service"
        raise RuntimeError(msg)


# --- blobs --------------------------------------------------------------------------------------


class MemoryBlobs:
    """Retains bytes in a dictionary, and refuses anything over a cap."""

    def __init__(self, max_bytes: int = 1 << 30) -> None:
        self.data: dict[str, bytes] = {}
        self.staged: dict[str, tuple[Retention, AcquiredSource]] = {}
        self._max = max_bytes

    async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
        del media_type
        if len(data) > self._max:
            return Retention(
                omitted_reason=f"original bytes not retained: {len(data)} exceeds the cap"
            )
        digest = content_hash(data)
        self.data[digest] = data
        return Retention(ref=digest)

    async def get(self, digest: str) -> bytes | None:
        return self.data.get(digest)

    async def retain_acquisition(
        self, key: str, raw: RawDocument
    ) -> tuple[Retention, AcquiredSource]:
        acquired = AcquiredSource.from_raw(raw)
        retained = await self.retain(raw.as_bytes(), raw.media_type)
        if retained.ref is not None:
            self.staged[key] = (retained, acquired)
        return retained, acquired

    async def resume_acquisition(self, key: str) -> tuple[Retention, AcquiredSource] | None:
        return self.staged.get(key)

    async def complete_acquisition(self, key: str) -> None:
        self.staged.pop(key, None)
