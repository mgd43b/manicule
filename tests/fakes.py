"""In-memory implementations of every protocol.

They exist for two reasons: to prove the protocols are implementable without dragging in a
model or a database, and to give the conformance suites something real to be run against —
a suite that has never passed is not evidence of anything.

Deliberately kept honest. Where an implementation would be wrong, there is a matching
``Broken*`` class, so that the suites are shown to fail as well as to pass.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime
from typing import override

from manicule.core.anchors import Anchor, LineAnchor, Unlocated
from manicule.core.content import (
    BlockKind,
    Chunk,
    Document,
    DocumentStatus,
    ParsedBlock,
    RawDocument,
)
from manicule.core.embedding import EmbedFingerprint, Pooling, Vector
from manicule.core.errors import FingerprintMismatchError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

MEDIA_TYPE = "text/x-fake"


def make_document(text: str = "alpha\nbeta\ngamma") -> Document:
    return Document(
        id=document_id("default", "fake", "doc-1"),
        source="fake",
        source_id="doc-1",
        uri="fake://doc-1",
        title="Doc 1",
        content_hash=content_hash(text),
        media_type=MEDIA_TYPE,
        status=DocumentStatus.PARSED,
    )


def make_raw(text: str = "alpha\nbeta\ngamma") -> RawDocument:
    return RawDocument(source_id="doc-1", uri="fake://doc-1", media_type=MEDIA_TYPE, content=text)


def make_chunks(document: Document, count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=chunk_id(document.id, position, f"chunk {position}"),
            document_id=document.id,
            text=f"chunk {position}",
            embed_text=f"Doc 1 > chunk {position}",
            anchor=LineAnchor(start=position + 1, end=position + 1),
            position=position,
            token_count=3,
        )
        for position in range(count)
    ]


class LineParser:
    """One block per line, with anchors that resolve."""

    media_types = frozenset({MEDIA_TYPE})

    def __init__(self) -> None:
        self.setup_calls = 0
        self.teardown_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1

    async def teardown(self) -> None:
        self.teardown_calls += 1

    async def health(self) -> HealthReport:
        return HealthReport.healthy()

    def metrics(self) -> tuple[Metric, ...]:
        return (Metric(name="setup_calls", value=float(self.setup_calls)),)

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for number, line in enumerate(raw.as_text().splitlines(), start=1):
            if line.strip():
                yield ParsedBlock(
                    kind=BlockKind.PROSE, text=line, anchor=LineAnchor(start=number, end=number)
                )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        if not isinstance(anchor, LineAnchor):
            return None
        lines = raw.as_text().splitlines()
        if anchor.end > len(lines):
            return None
        return "\n".join(lines[anchor.start - 1 : anchor.end])


class LyingParser(LineParser):
    """Emits anchors pointing one line further down than the text it claims.

    The exact defect the round-trip check exists to catch: nothing raises, the citations
    look well-formed, and every one of them points at the wrong line.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for number, line in enumerate(raw.as_text().splitlines(), start=1):
            if line.strip():
                yield ParsedBlock(
                    kind=BlockKind.PROSE,
                    text=line,
                    anchor=LineAnchor(start=number + 1, end=number + 1),
                )


class SilentParser(LineParser):
    """Claims it cannot locate anything, without saying why it resolved text anyway."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for line in raw.as_text().splitlines():
            if line.strip():
                yield ParsedBlock(
                    kind=BlockKind.PROSE, text=line, anchor=Unlocated(reason="not implemented")
                )


class BlockChunker:
    """One chunk per block, with the title as the breadcrumb."""

    fingerprint = ChunkFingerprint(
        chunker="block",
        version="1",
        max_tokens=64,
        overlap_tokens=0,
        tokenizer_id="whitespace",
    )

    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for position, block in enumerate(blocks):
            chunks.append(
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
            )
        return chunks


class HashEmbedder:
    """Deterministic vectors at whatever dimension its fingerprint declares.

    The dimension is a constructor argument, so a test can ask for an unusual one and see
    whether anything downstream had assumed a common one.
    """

    def __init__(self, dimension: int = 5, model_id: str = "fake/embedder") -> None:
        self.fingerprint = EmbedFingerprint(
            model_id=model_id,
            dimension=dimension,
            pooling=Pooling.MEAN,
            normalized=True,
            tokenizer_id="whitespace",
            max_sequence_length=128,
            backend="fake",
        )

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        dimension = self.fingerprint.dimension
        return [[((hash(text) >> (i * 3)) % 100) / 100 for i in range(dimension)] for text in texts]


class TruncatingEmbedder(HashEmbedder):
    """Returns vectors one dimension short of what it advertises."""

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = await super().embed(texts)
        return [vector[:-1] for vector in vectors]


class MemoryVectorStore:
    """A vector store with no assumptions about dimension."""

    def __init__(self) -> None:
        self._fingerprint: EmbedFingerprint | None = None
        self._rows: dict[str, tuple[Chunk, Vector]] = {}

    async def ensure_ready(self, fingerprint: EmbedFingerprint) -> None:
        if self._fingerprint is None:
            self._fingerprint = fingerprint
            return
        self._fingerprint.require_match(fingerprint)

    async def fingerprint(self) -> EmbedFingerprint | None:
        return self._fingerprint

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        expected = self._fingerprint.dimension if self._fingerprint else None
        for chunk, vector in zip(chunks, vectors, strict=True):
            if expected is not None and len(vector) != expected:
                msg = f"vector for {chunk.id} has {len(vector)} dimensions, expected {expected}"
                raise ValueError(msg)
            self._rows[chunk.id] = (chunk, vector)

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002
    ) -> list[Candidate]:
        del filter
        scored = [
            Candidate(chunk=chunk, score=-_distance(vector, stored), scores={"dense": 1.0})
            for chunk, stored in self._rows.values()
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[:k]

    async def delete_document(self, document_id: str) -> None:
        stale = [key for key, (chunk, _) in self._rows.items() if chunk.document_id == document_id]
        for key in stale:
            del self._rows[key]

    async def count(self) -> int:
        return len(self._rows)


class FixedDimensionVectorStore(MemoryVectorStore):
    """A store that assumes 768 dimensions, the way a schema written once does."""

    _ASSUMED = 768

    @override
    async def ensure_ready(self, fingerprint: EmbedFingerprint) -> None:
        if fingerprint.dimension != self._ASSUMED:
            msg = f"table is {self._ASSUMED}-dimensional"
            raise ValueError(msg)
        await super().ensure_ready(fingerprint)


class ForgetfulVectorStore(MemoryVectorStore):
    """A store that checks the dimension but not the model."""

    @override
    async def ensure_ready(self, fingerprint: EmbedFingerprint) -> None:
        if self._fingerprint and self._fingerprint.dimension != fingerprint.dimension:
            raise FingerprintMismatchError("dimension changed")
        self._fingerprint = fingerprint


def _distance(left: Vector, right: Vector) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=False)) ** 0.5


class TopKStage:
    """Keeps the best ``k`` candidates."""

    name = "top_k"

    def __init__(self, k: int = 2) -> None:
        self._k = k

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        ordered = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        return [candidate.scored_by(self.name, candidate.score) for candidate in ordered[: self._k]]


class MutatingStage:
    """Sorts the list it was handed, in place, then returns a copy of it.

    Returning a new list hides the damage: the caller's list has been reordered, so an
    earlier stage's record of what it produced is no longer what it produced.
    """

    name = "mutating"

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return list(candidates)


class AliasingStage:
    """Hands back the very list it was given."""

    name = "aliasing"

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        return candidates


class MemoryConnector:
    """A connector over a dictionary."""

    def __init__(self, documents: dict[SourceId, str] | None = None) -> None:
        self.name = "memory"
        self._documents = documents or {"a": "alpha", "b": "beta"}

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        for source_id, text in sorted(self._documents.items()):
            yield DiscoveredDoc(
                ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
                version_token=content_hash(text),
                media_type=MEDIA_TYPE,
            )

    async def fetch(self, ref: DocRef) -> RawDocument:
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=MEDIA_TYPE,
            content=self._documents[ref.source_id],
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        for source_id in sorted(self._documents):
            yield source_id


class ForgetfulConnector(MemoryConnector):
    """Never reports what exists, which would delete the whole index."""

    @override
    async def reconcile(self) -> AsyncIterator[SourceId]:
        return
        yield  # pragma: no cover - unreachable, present to make this an async generator


def now() -> datetime:
    return datetime.now(tz=UTC)


class PassThroughMiddleware:
    """A middleware that touches nothing. The baseline the contract must accept."""

    name = "pass-through"
    mutates_embedded_text = False

    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        return raw

    async def after_parse(self, document: Document, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        return blocks

    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        return chunks

    async def after_store(self, document: Document) -> None:
        return


class RedactingMiddleware(PassThroughMiddleware):
    """Rewrites ``embed_text`` and says so. The legitimate case."""

    name = "redacting"
    mutates_embedded_text = True

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embed_text": chunk.embed_text.replace("chunk", "[REDACTED]")})
            for chunk in chunks
        ]


class TextRewritingMiddleware(PassThroughMiddleware):
    """Rewrites ``Chunk.text``, which no middleware may do.

    This is the defect ``assert_middleware_contract`` exists to catch: every parser test
    still passes, and the corpus acquires citations quoting text no source contains.
    """

    name = "text-rewriting"
    mutates_embedded_text = False

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"text": chunk.text.replace("chunk", "[REDACTED]")})
            for chunk in chunks
        ]


class UndeclaredEmbedMiddleware(PassThroughMiddleware):
    """Rewrites ``embed_text`` without declaring it, so no fingerprint describes the corpus."""

    name = "undeclared-embed"
    mutates_embedded_text = False

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embed_text": chunk.embed_text + " extra context"})
            for chunk in chunks
        ]


class BlockRewritingMiddleware(PassThroughMiddleware):
    """Rewrites ``ParsedBlock.text`` in ``after_parse``.

    The same corruption as rewriting ``Chunk.text``, one hook earlier: block text becomes
    chunk text, and the block's anchor still points at source text that no longer matches.
    """

    name = "block-rewriting"
    mutates_embedded_text = False

    @override
    async def after_parse(self, document: Document, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        return [block.model_copy(update={"text": block.text.upper()}) for block in blocks]
