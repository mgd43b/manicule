"""In-memory implementations of every protocol.

They exist for two reasons: to prove the protocols are implementable without dragging in a
model or a database, and to give the conformance suites something real to be run against —
a suite that has never passed is not evidence of anything.

Deliberately kept honest. Where an implementation would be wrong, there is a matching
``Broken*`` class, so that the suites are shown to fail as well as to pass.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import override

from manicule.config.settings import Settings
from manicule.core.anchors import Anchor, LineAnchor, Unlocated
from manicule.core.content import (
    BlockKind,
    Chunk,
    Document,
    DocumentStatus,
    ParsedBlock,
    RawDocument,
)
from manicule.core.embedding import (
    UNRECORDED_CHECKSUM,
    UNRECORDED_IDENTITY,
    VECTOR_CHECKSUM_VERSION,
    EmbedFingerprint,
    Pooling,
    StoredVector,
    Vector,
    VectorState,
    canonical_stored_vector,
    choose_stored_vector,
    classify_stored_vector,
    embedding_input_identity,
    vector_checksum,
    verify_stored_checksum,
)
from manicule.core.errors import FingerprintMismatchError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.protocols import DocStore
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


@dataclass(frozen=True, slots=True)
class _Row:
    """One stored row, named so a test reading this file can see what a Lance row holds."""

    chunk: Chunk
    vector: tuple[float, ...]
    identity: str
    checksum: str


class MemoryVectorStore:
    """A vector store with no assumptions about dimension.

    Holds the same four things per row a Lance row holds — the chunk, the vector, the
    embedding-input identity and the vector checksum — and classifies them with the same shared
    rule, so a pipeline test run against this store measures the reuse behavior the real one has
    rather than a convenient approximation of it. In particular it stores the *canonical* vector
    and hashes that, because a fake that hashed what it was handed would accept a row the real
    store rejects and hide the one bug this pair exists to catch.
    """

    def __init__(self) -> None:
        self._fingerprint: EmbedFingerprint | None = None
        self._middleware: tuple[str, ...] = ()
        self._rows: dict[str, _Row] = {}

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        self._middleware = tuple(embed_text_middleware)
        if self._fingerprint is None:
            self._fingerprint = fingerprint
            return
        self._fingerprint.require_match(fingerprint)

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
        expected = self._fingerprint.dimension if self._fingerprint else None
        for chunk, vector in zip(chunks, vectors, strict=True):
            if expected is not None and len(vector) != expected:
                msg = f"vector for {chunk.id} has {len(vector)} dimensions, expected {expected}"
                raise ValueError(msg)
            # A tuple whatever the caller handed over; see `MemoryVectors` for why the
            # container type must not survive a write.
            stored = canonical_stored_vector(vector)
            self._rows[chunk.id] = _Row(
                chunk, stored, self._identity_of(chunk), vector_checksum(stored)
            )

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        verdicts: dict[str, StoredVector] = {}
        for chunk in chunks:
            wanted = self._identity_of(chunk)
            elsewhere = next(
                (
                    row
                    for row in self._rows.values()
                    if row.identity == wanted != UNRECORDED_IDENTITY
                ),
                None,
            )
            verdicts[chunk.id] = choose_stored_vector(
                self._classify(chunk, self._rows.get(chunk.id)),
                self._classify(chunk, elsewhere),
            )
        return verdicts

    def _classify(self, chunk: Chunk, row: _Row | None) -> StoredVector:
        if row is None or self._fingerprint is None:
            return StoredVector(state=VectorState.ABSENT)
        return classify_stored_vector(
            chunk,
            recorded_identity=row.identity,
            stored_embed_text=row.chunk.embed_text,
            stored_vector=list(row.vector),
            embed=self._fingerprint,
            middleware=self._middleware,
            recorded_checksum=row.checksum,
            recorded_checksum_version=(
                VECTOR_CHECKSUM_VERSION if row.checksum else UNRECORDED_CHECKSUM
            ),
        )

    def _identity_of(self, chunk: Chunk) -> str:
        """What this store records beside a vector, or nothing when it has no fingerprint yet.

        A store that has never been through ``ensure_ready`` cannot name the vector space it is
        writing into, so it records :data:`~manicule.core.embedding.UNRECORDED_IDENTITY` rather
        than inventing one — the same thing a row predating the identity column holds.
        """
        if self._fingerprint is None:
            return UNRECORDED_IDENTITY
        return embedding_input_identity(
            chunk.embed_text,
            document_id=chunk.document_id,
            embed=self._fingerprint,
            middleware=self._middleware,
        )

    def corrupt(
        self,
        chunk_id: str,
        *,
        vector: Vector | None = None,
        identity: str | None = None,
        checksum: str | None = None,
    ) -> None:
        """Damage one row the way a half-written directory or an edited table would.

        ``vector`` replaces the stored vector **without touching the checksum**, which is what
        a bit flip in the numbers looks like — pass a wrong-length one for a row that cannot be
        read at the index's dimension. ``identity`` replaces the recorded identity, for a row
        whose metadata claims something the chunk beside it contradicts. ``checksum`` replaces
        the checksum without touching the vector, which is the same corruption arriving from
        the other side.
        """
        row = self._rows[chunk_id]
        self._rows[chunk_id] = _Row(
            row.chunk,
            row.vector if vector is None else tuple(vector),
            row.identity if identity is None else identity,
            row.checksum if checksum is None else checksum,
        )

    def forget_vector(self, chunk_id: str) -> None:
        """Delete one vector row, leaving the chunk wherever the document store has it."""
        del self._rows[chunk_id]

    def vector_of(self, chunk_id: str) -> Vector | None:
        """The stored vector, for a test that has to compare one against itself later."""
        row = self._rows.get(chunk_id)
        return None if row is None else row.vector

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002
    ) -> list[Candidate]:
        del filter
        scored = [
            Candidate(chunk=row.chunk, score=-_distance(vector, row.vector), scores={"dense": 1.0})
            for row in self._rows.values()
            if verify_stored_checksum(
                row.vector,
                recorded=row.checksum,
                version=VECTOR_CHECKSUM_VERSION if row.checksum else UNRECORDED_CHECKSUM,
                required=False,
            ).accepts
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[:k]

    async def delete_document(self, document_id: str) -> None:
        stale = [key for key, row in self._rows.items() if row.chunk.document_id == document_id]
        for key in stale:
            del self._rows[key]

    async def count(self) -> int:
        return len(self._rows)


class FixedDimensionVectorStore(MemoryVectorStore):
    """A store that assumes 768 dimensions, the way a schema written once does."""

    _ASSUMED = 768

    @override
    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        if fingerprint.dimension != self._ASSUMED:
            msg = f"table is {self._ASSUMED}-dimensional"
            raise ValueError(msg)
        await super().ensure_ready(fingerprint, embed_text_middleware=embed_text_middleware)


class IdKeyedVectorStore(MemoryVectorStore):
    """A store that answers reuse on the chunk id, which is the plausible wrong answer.

    Every write succeeds, every search returns, and every other conformance check passes. What
    it gets wrong is a chunk whose ``embed_text`` moved while its ``text`` did not: the id is
    content-derived from ``text``, so it survives, and this store hands back a vector for a
    string the corpus no longer contains — silently, for as long as the index lives.
    """

    @override
    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        return {
            chunk.id: (
                StoredVector(state=VectorState.READABLE, vector=tuple(row.vector))
                if (row := self._rows.get(chunk.id)) is not None
                else StoredVector(state=VectorState.ABSENT)
            )
            for chunk in chunks
        }


class PrehashingVectorStore(MemoryVectorStore):
    """A store that checksums the vector it was handed instead of the one it stores.

    One line shorter than the right thing and indistinguishable from it for any vector that is
    already unit length — which is every vector in these fixtures and no vector a real embedder
    returns. What it produces is a digest no readback can match, so the whole corpus reads as
    corrupt on the first search after a deploy.
    """

    @override
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        await super().upsert(chunks, vectors, publication_id=publication_id)
        for chunk, vector in zip(chunks, vectors, strict=True):
            row = self._rows[chunk.id]
            self._rows[chunk.id] = _Row(
                row.chunk, row.vector, row.identity, vector_checksum(vector)
            )


class ForgetfulVectorStore(MemoryVectorStore):
    """A store that checks the dimension but not the model."""

    @override
    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        if self._fingerprint and self._fingerprint.dimension != fingerprint.dimension:
            raise FingerprintMismatchError("dimension changed")
        self._fingerprint = fingerprint
        self._middleware = tuple(embed_text_middleware)


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


class RawVectorStage:
    """Returns every candidate it was built with, exactly as a vector index would.

    The dense leg's raw half: Lance holds ``id``, ``vector``, ``document_id``, ``kind``,
    ``lang`` and ``position`` and nothing about tenancy or liveness, so a search against it
    returns rows of which an unknown number are invisible. A stage that stops here has
    performed a scoped query as an unscoped one and cannot tell.
    """

    name = "raw_vector"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        found = [
            Candidate(chunk=chunk, score=1.0, scores={self.name: 1.0}) for chunk in self._chunks
        ]
        return [*candidates, *found]


class HydratingStage(RawVectorStage):
    """The same search, joined back through the document store before it returns.

    Workspace, soft-deletion and status live on ``documents`` in the authoritative store, so
    this is the only place they can be applied — which is why the join is inside the stage
    rather than beside it.
    """

    name = "hydrated_vector"

    def __init__(self, chunks: Sequence[Chunk], docstore: DocStore) -> None:
        super().__init__(chunks)
        self._docstore = docstore

    @override
    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        found = await super().run(query, candidates)
        live: list[Candidate] = []
        for candidate in found:
            document = await self._docstore.get_document(candidate.chunk.document_id)
            if document is not None and document.status is DocumentStatus.INDEXED:
                live.append(candidate)
        return live


class MemoryConnector:
    """A connector over a dictionary."""

    def __init__(
        self, documents: dict[SourceId, str] | None = None, *, name: str = "memory"
    ) -> None:
        # Named by the caller, defaulting to the type, exactly as a real connector is. The
        # name becomes the `source` half of every document's identity, so a fake that could
        # only ever be called "memory" cannot exercise two instances of one type at all.
        self.name = name
        self._documents = documents or {"a": "alpha", "b": "beta"}
        self._watermark: Watermark | None = None

    @property
    def watermark(self) -> Watermark | None:
        return self._watermark

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        seen: list[str] = []
        for source_id, text in sorted(self._documents.items()):
            yield DiscoveredDoc(
                ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
                version_token=content_hash(text),
                media_type=MEDIA_TYPE,
            )
            seen.append(source_id)
        # After the loop, so an abandoned enumeration leaves the watermark where it was.
        self._watermark = Watermark(value=seen[-1], observed_at=now()) if seen else None

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


class EagerWatermarkConnector(MemoryConnector):
    """Advances its watermark as it yields, rather than when the walk finishes.

    Nothing about this looks wrong: every document is correct, the position is real, and an
    uninterrupted run behaves identically to a correct connector. It goes wrong only when a
    run is interrupted — and then the stored position is *past* documents nobody received, so
    the next sync starts after them and they are never enumerated again. No error, nothing to
    notice, and no later sync fixes it.
    """

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        for source_id, text in sorted(self._documents.items()):
            self._watermark = Watermark(value=source_id, observed_at=now())
            yield DiscoveredDoc(
                ref=DocRef(source_id=source_id, uri=f"memory://{source_id}"),
                version_token=content_hash(text),
                media_type=MEDIA_TYPE,
            )


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


# --- configurations ------------------------------------------------------------------------
#
# The thing under test here is a configuration rather than a component: a local-only data
# policy is a promise about where content goes, and it is kept or broken by what the endpoints
# resolve to. The two ``Unenforced``/``Banning`` subclasses are the ``Broken*`` convention
# applied to a policy gate — a check that has only ever passed proves nothing.


class UnenforcedLocalOnly(Settings):
    """Settings whose local-only policy is typed, documented, and enforced nowhere.

    Exactly the failure ``docs/contracts.md`` §5 names as worse than an absent guarantee. The
    endpoints still say where they point; the gate simply does not act on it, which is what a
    later edit weakening ``policy_problems`` would look like.
    """

    @override
    def policy_problems(self) -> list[str]:
        return [problem for problem in super().policy_problems() if "cloud_allowed" not in problem]


class BanningLocalOnly(Settings):
    """Settings that refuse every endpoint, including the ones on this machine.

    The other direction, and not a hypothetical: classifying by provider name refused an
    OpenAI-compatible server on ``127.0.0.1``, so the safe configuration was the one that
    failed. A policy that rejects the deployment it exists to permit gets switched off.
    """

    @override
    def policy_problems(self) -> list[str]:
        if self.security.data_policy.cloud_allowed:
            return super().policy_problems()
        return [
            *super().policy_problems(),
            *(
                f"security.data_policy.cloud_allowed is false, but the {endpoint.describe()} "
                f"is not on this machine."
                for endpoint in self.selected_endpoints
            ),
        ]


def local_only(
    llm: dict[str, str],
    model: type[Settings] = Settings,
    **rest: object,
) -> Settings:
    """Settings forbidding cloud processing, with ``llm`` as the generation provider."""
    return model(
        llm=llm,  # pyright: ignore[reportArgumentType] - a settings section, validated on the way in
        security={"data_policy": {"cloud_allowed": False}},  # pyright: ignore[reportArgumentType]
        **rest,  # pyright: ignore[reportArgumentType] - further sections, validated the same way
    )


LOOPBACK_OLLAMA = {"provider": "ollama", "base_url": "http://127.0.0.1:11434"}
LAN_OLLAMA = {"provider": "ollama", "base_url": "http://gpu-box.lan:11434"}


def loopback_ollama(model: type[Settings] = Settings) -> Settings:
    """An Ollama on this machine under a local-only policy. The legitimate case.

    It must start. A predicate that refuses this is not enforcing a policy, it is banning a
    supported deployment, and the way round it is to turn the policy off.
    """
    return local_only(LOOPBACK_OLLAMA, model)


def lan_ollama(model: type[Settings] = Settings) -> Settings:
    """An Ollama on another machine under a local-only policy. The defect.

    The provider name is on the local list, so the classification that read the name said
    "nothing leaves" whatever ``base_url`` was set to: this configuration started cleanly
    while every prompt, every retrieved passage and every question crossed the network to
    another host — under the one setting that exists to prevent exactly that.
    """
    return local_only(LAN_OLLAMA, model)
