"""Doubles for the retrieval suites, and the broken ones that prove the guards fire.

Every ``Broken*`` here reproduces a defect that raises nothing on its own: a leg that skips
its scoping join, a fusion that reintroduces the magnitudes RRF exists to discard, a reranker
that swallows its own failure. Each is what the matching guard is aimed at, and a guard nobody
has watched fail is not evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from manicule.core.embedding import StoredVector, VectorState
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.profile import Profiles

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import Vector

WORKSPACE = "default"
SCOPE = frozenset({WORKSPACE})


def a_query(text: str = "authentication", *, limit: int = 3, **kwargs: object) -> Query:
    """A query scoped to the default workspace."""
    return Query(text=text, limit=limit, filter=Filter(workspace_ids=SCOPE), **kwargs)  # pyright: ignore[reportArgumentType]


def profiles(**overrides: object) -> Profiles:
    return Profiles(overrides)


class ListVectorStore:
    """A vector store that returns a fixed list, best first, honoring only ``k``.

    Stands in for LanceDB where the point of the test is what the *stage* does with rows that
    an index has already ranked — the dense leg's whole difficulty is that those rows say
    nothing about tenancy or liveness, and this reproduces exactly that.
    """

    def __init__(self, chunks: Sequence[Chunk], *, scores: Sequence[float] | None = None) -> None:
        self.chunks = list(chunks)
        self.scores = (
            list(scores) if scores is not None else [1.0 - i * 0.01 for i in range(len(chunks))]
        )
        self.requested: list[int] = []
        self.filters: list[Filter | None] = []

    async def ensure_ready(
        self, fingerprint: object, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        del fingerprint, embed_text_middleware

    async def fingerprint(self) -> None:
        return None

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        del chunks, vectors

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> dict[str, StoredVector]:
        # A store that never writes holds nothing to reuse. Answered rather than omitted,
        # because the protocol's answer is total: an entry per chunk, so the caller is never
        # left deciding what a missing key meant.
        return {chunk.id: StoredVector(state=VectorState.ABSENT) for chunk in chunks}

    async def search(
        self,
        vector: Vector,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
    ) -> list[Candidate]:
        del vector
        self.requested.append(k)
        self.filters.append(filter)
        admitted = [
            (chunk, score)
            for chunk, score in zip(self.chunks, self.scores, strict=True)
            if filter is None or not filter.document_ids or chunk.document_id in filter.document_ids
        ]
        return [Candidate(chunk=chunk, score=score) for chunk, score in admitted[:k]]

    async def delete_document(self, document_id: str) -> None:
        del document_id

    async def count(self) -> int:
        return len(self.chunks)


class FixedScorer:
    """A cross-encoder that returns logits from a table, or a default."""

    def __init__(self, scores: dict[str, float] | None = None, *, default: float = 0.0) -> None:
        self.model_id = "fake/cross-encoder"
        self._scores = scores or {}
        self._default = default
        self.calls: list[list[tuple[str, str]]] = []

    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self._scores.get(passage, self._default) for _, passage in pairs]


class ExplodingScorer(FixedScorer):
    """A cross-encoder whose forward pass fails.

    The reranker must let this out. Returning the input instead would mean a profile that says
    it reranks produced an unreranked list, which nothing downstream can detect.
    """

    @override
    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        del pairs
        msg = "the forward pass failed"
        raise RuntimeError(msg)


class ShortScorer(FixedScorer):
    """A cross-encoder that returns fewer scores than it was given pairs.

    Positional data with a length mismatch means some passage is ranked by another passage's
    score, which produces a plausible ordering computed from the wrong numbers.
    """

    @override
    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = await super().score(pairs)
        return scores[:-1]


class RecordingStage:
    """A stage that returns a fixed list and remembers what it was handed."""

    def __init__(self, name: str, produced: Sequence[Candidate], *, delay: float = 0.0) -> None:
        self.name = name
        self._produced = list(produced)
        self._delay = delay
        self.seen: list[list[Candidate]] = []

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        self.seen.append(list(candidates))
        if self._delay:
            import asyncio  # noqa: PLC0415 - only the delaying variant needs it

            await asyncio.sleep(self._delay)
        return [*candidates, *self._produced]


class FrameReadingStage:
    """A stage whose output depends on whether anyone is recording.

    The one thing the trace frame's contract forbids. It is here so the conformance run with no
    frame installed can be shown to catch it, rather than merely asserted to.
    """

    name = "frame_reading"

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        from manicule.retrieval.trace import current_frame  # noqa: PLC0415

        if current_frame() is None:
            return []
        return list(candidates)


__all__ = [
    "SCOPE",
    "WORKSPACE",
    "ExplodingScorer",
    "FixedScorer",
    "FrameReadingStage",
    "ListVectorStore",
    "RecordingStage",
    "ShortScorer",
    "a_query",
    "profiles",
]
