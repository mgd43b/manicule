"""Doubles for the evaluation suites, including the ones that must be refused.

Two embedders carry the load here and they are opposites by construction:

- :class:`BagOfWordsEmbedder` has semantic content. Texts sharing words land near each other,
  so searching for a document's title returns that document. It is crude — lexical overlap
  projected into a fixed number of buckets — and crude is the point: the probe is a liveness
  check, and anything above the floor should clear it.
- :class:`MeaninglessEmbedder` has none. Its vector is a hash of the whole string, so two texts
  sharing every word but one are as far apart as two unrelated ones. It produces well-shaped,
  normalised, deterministic vectors and raises nothing, which is exactly why an evaluation
  harness has to be able to catch it rather than trusting that somebody would notice.

Both are deterministic across processes. ``hash()`` on a string is salted per interpreter, so
an embedder built on it would give a different index on every run — which would make a suite
that depends on ranking flaky for a reason unrelated to anything it is testing.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, override

from manicule.core.embedding import (
    EmbedFingerprint,
    Pooling,
    StoredVector,
    VectorState,
    classify_stored_vector,
    embedding_input_identity,
)
from manicule.core.retrieval import Candidate
from manicule.evaluation.corpus import CorpusVersion
from manicule.evaluation.systems import ResultItem, SystemResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import Vector
    from manicule.core.retrieval import Filter

DIMENSION = 256
"""Wide enough that hashing tokens into buckets rarely collides on a small corpus."""


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return values if norm == 0.0 else [value / norm for value in values]


class _FakeEmbedder:
    """Shared fingerprint plumbing. Neither subclass loads anything."""

    def __init__(self, model_id: str) -> None:
        self.fingerprint = EmbedFingerprint(
            model_id=model_id,
            dimension=DIMENSION,
            pooling=Pooling.MEAN,
            normalized=True,
            tokenizer_id="whitespace",
            max_sequence_length=512,
            backend="fake",
        )

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        raise NotImplementedError


class BagOfWordsEmbedder(_FakeEmbedder):
    """A vector with real semantic content: which words the text contains.

    Every token is hashed into a bucket and counted, then the vector is normalised. Two texts
    about the same thing share tokens and therefore share buckets, which is all a probe asks
    for.
    """

    def __init__(self, model_id: str = "fake/bag-of-words") -> None:
        super().__init__(model_id)

    @override
    def _vector(self, text: str) -> Vector:
        buckets = [0.0] * DIMENSION
        for token in text.casefold().split():
            cleaned = token.strip(".,;:!?—-()[]\"'")
            if not cleaned:
                continue
            buckets[int.from_bytes(_digest(cleaned)[:4], "big") % DIMENSION] += 1.0
        return _normalise(buckets)


class MeaninglessEmbedder(_FakeEmbedder):
    """A vector with no semantic content, and nothing about it says so.

    The whole string is hashed and the digest is stretched into a unit vector. It is
    deterministic, correctly shaped, correctly normalised, and its similarities are unrelated
    to meaning — so a system built on it ranks documents in an order that ignores the query.
    Every guard in :mod:`manicule.evaluation.probe` exists because of this class.
    """

    def __init__(self, model_id: str = "fake/meaningless") -> None:
        super().__init__(model_id)

    @override
    def _vector(self, text: str) -> Vector:
        raw = b""
        counter = 0
        while len(raw) < DIMENSION:
            raw += _digest(f"{counter}\x00{text}")
            counter += 1
        # Centred on zero, so the vectors point in genuinely arbitrary directions. Bytes taken
        # as-is are all positive, which would make every pair of vectors similar and hide the
        # very absence of structure this class exists to have.
        return _normalise([byte / 255.0 - 0.5 for byte in raw[:DIMENSION]])


class CosineVectorStore:
    """A vector store that really ranks by cosine similarity.

    Needed because the fixed-list doubles elsewhere in the suite ignore the query vector
    entirely, and the whole question here is what a vector says about a query.
    """

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, Vector]] = {}
        self._fingerprint: EmbedFingerprint | None = None
        self._middleware: tuple[str, ...] = ()
        self.searches = 0

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        self._fingerprint = fingerprint
        self._middleware = tuple(embed_text_middleware)

    async def fingerprint(self) -> EmbedFingerprint | None:
        return self._fingerprint

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, list(vector))

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        verdicts: dict[str, StoredVector] = {}
        for chunk in chunks:
            row = self._rows.get(chunk.id)
            if row is None or self._fingerprint is None:
                verdicts[chunk.id] = StoredVector(state=VectorState.ABSENT)
                continue
            stored_chunk, vector = row
            verdicts[chunk.id] = classify_stored_vector(
                chunk,
                recorded_identity=embedding_input_identity(
                    stored_chunk.embed_text,
                    document_id=stored_chunk.document_id,
                    embed=self._fingerprint,
                    middleware=self._middleware,
                ),
                stored_embed_text=stored_chunk.embed_text,
                stored_vector=vector,
                embed=self._fingerprint,
                middleware=self._middleware,
            )
        return verdicts

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
    ) -> list[Candidate]:
        self.searches += 1
        admitted = [
            (chunk, stored)
            for chunk, stored in self._rows.values()
            if filter is None or not filter.document_ids or chunk.document_id in filter.document_ids
        ]
        scored = [
            Candidate(chunk=chunk, score=sum(a * b for a, b in zip(vector, stored, strict=True)))
            for chunk, stored in admitted
        ]
        # Ties broken by chunk id so two runs of the same query agree. Without it a run's
        # ranking would depend on dictionary order, and a comparison between two runs would be
        # measuring that.
        scored.sort(key=lambda candidate: (-candidate.score, candidate.chunk.id))
        return scored[:k]

    async def delete_document(self, document_id: str) -> None:
        for key in [k for k, (chunk, _) in self._rows.items() if chunk.document_id == document_id]:
            del self._rows[key]

    async def count(self) -> int:
        return len(self._rows)


class FixedSystem:
    """A system under comparison that answers every query with the same list.

    The crudest possible failure, and it must be caught: it retrieves, in the sense that it
    returns documents, and it retrieves nothing in the sense that matters.
    """

    def __init__(
        self,
        items: Sequence[ResultItem],
        *,
        config_label: str = "fixed",
        corpus_version: CorpusVersion | None = None,
        configuration: Mapping[str, str] | None = None,
    ) -> None:
        self._items = tuple(items)
        self._config_label = config_label
        self._corpus_version = corpus_version or CorpusVersion(
            label="fixture", digest="sha256:fixture", document_count=500
        )
        self._configuration = dict(configuration or {"kind": "fixed"})
        self.asked: list[str] = []

    @property
    def config_label(self) -> str:
        return self._config_label

    @property
    def corpus_version(self) -> CorpusVersion:
        return self._corpus_version

    async def search(self, text: str, *, limit: int) -> SystemResult:
        self.asked.append(text)
        return SystemResult(
            config_label=self._config_label,
            configuration=dict(self._configuration),
            corpus_version=self._corpus_version,
            items=self._items[:limit],
        )


class LookupSystem(FixedSystem):
    """A system that answers correctly for the queries it knows and returns nothing otherwise.

    The counterpart to :class:`FixedSystem`: something that genuinely discriminates, so the
    probe can be shown to *admit* as well as to refuse. A guard that refuses everything is as
    uninformative as one that refuses nothing.
    """

    def __init__(
        self,
        answers: Mapping[str, Sequence[ResultItem]],
        *,
        config_label: str = "lookup",
        corpus_version: CorpusVersion | None = None,
        configuration: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            (),
            config_label=config_label,
            corpus_version=corpus_version,
            configuration=configuration or {"kind": "lookup"},
        )
        self._answers = {text: tuple(items) for text, items in answers.items()}

    @override
    async def search(self, text: str, *, limit: int) -> SystemResult:
        self.asked.append(text)
        return SystemResult(
            config_label=self.config_label,
            configuration=dict(self._configuration),
            corpus_version=self.corpus_version,
            items=self._answers.get(text, ())[:limit],
        )


def an_item(document_id: str, *, text: str = "passage text", title: str = "") -> ResultItem:
    return ResultItem(document_id=document_id, text=text, title=title or document_id)


__all__ = [
    "DIMENSION",
    "BagOfWordsEmbedder",
    "CosineVectorStore",
    "FixedSystem",
    "LookupSystem",
    "MeaninglessEmbedder",
    "an_item",
]
