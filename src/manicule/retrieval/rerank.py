"""The cross-encoder: one model, one relevance logit per ``(query, passage)`` pair.

A cross-encoder encodes the pair *jointly* and emits a scalar from a model trained for exactly
that. It is deterministic, cheap relative to generation, and on a fixed scale for a fixed
model. That is a different thing from prompting a generative model for a number, which is
non-deterministic, costs a generation call per candidate, and has to parse an integer out of
prose — where a failure to parse is indistinguishable from a genuine "irrelevant".

Three rules follow from what a reranker is for, and each closes a way for a ranking to be
wrong while everything still looks fine:

**The reranker's failure is the query's failure.** It raises; it never returns its input. A
profile that says it reranks and produced an unreranked list has misreported which pipeline
ran — and an evaluation harness cannot see the difference. There is deliberately no
``try``/``except`` anywhere in this module.

**It truncates to what it scored.** The head is rescored and returned; the tail is dropped
rather than concatenated. A concatenated tail still carries fusion's scores, which are on the
order of 0.016, beside logits on a completely different scale — one list, two scales, and every
comparison between them meaningless.

**It says which model produced the ranking.** A recorded result that cannot name its reranker
cannot be reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.retrieval.trace import RerankReport, record

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Metadata
    from manicule.core.retrieval import Candidate, Query
    from manicule.retrieval.profile import Profiles


@runtime_checkable
class PairScorer(Protocol):
    """Scores ``(query, passage)`` pairs jointly.

    The seam between the stage and a model runtime. It exists so that the ordering, truncation
    and failure rules above are exercised by the suite without a two-gigabyte download, and so
    that a different cross-encoder implementation is a substitution rather than a fork.
    """

    model_id: str

    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One relevance logit per pair, in order. Unbounded, and model-specific."""
        ...


class CrossEncoderReranker:
    """Rescore the head of a fused list with a dedicated model.

    Cost is linear in the number of pairs and dominates the rest of the pipeline by a wide
    margin: the two legs are one forward pass over a short query plus two indexed lookups,
    while this is one forward pass over a full-length passage per candidate. That is what the
    profiles are actually buying — not three settings, but "no second model", "a second model
    over 20 passages" and "a second model over 50".
    """

    def __init__(self, *, scorer: PairScorer, profiles: Profiles, name: str = "rerank") -> None:
        self.name = name
        self.model_id = scorer.model_id
        """Which model ranked. Recorded on every run, because a result must name it."""

        self._scorer = scorer
        self._profiles = profiles

    async def setup(self) -> None:
        """Prepare the scorer, if it has anything to prepare.

        Delegated rather than skipped: the container sets up the component it constructed, and
        the weights live one level down. A reranker whose model is loaded lazily on the first
        query would stall that query for as long as the download takes, under a profile that
        chose to pay for a second model precisely so it would be ready.
        """
        prepare = getattr(self._scorer, "setup", None)
        if prepare is not None:
            await prepare()

    async def teardown(self) -> None:
        """Release the scorer, if it holds anything."""
        release = getattr(self._scorer, "teardown", None)
        if release is not None:
            await release()

    def describe(self) -> Metadata:
        """The model that ranked. A result that cannot name it cannot be reproduced."""
        return {"model_id": self.model_id}

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        """Score the head jointly with the query and return it, reordered.

        The passage scored is ``chunk.text`` — what will be quoted — rather than
        ``embed_text``, which carries a heading breadcrumb that exists to make a passage
        *findable* and would otherwise be judged as though it were part of the answer.
        """
        profile = self._profiles.for_query(query)
        head_size = max(profile.candidates, query.limit)
        head = candidates[:head_size]
        if not head:
            record(RerankReport(model_id=self.model_id, pairs=0, truncated_from=len(candidates)))
            return []

        # No try/except, on purpose: a reranker that swallowed its own failure would return an
        # unreranked list under a profile that reports it reranked.
        scores = await self._scorer.score(
            [(query.text, candidate.chunk.text) for candidate in head]
        )
        if len(scores) != len(head):
            msg = (
                f"reranker {self.model_id!r} scored {len(scores)} of {len(head)} pairs. They "
                f"are positional, so a mismatch means some passage would be ranked by another "
                f"passage's score."
            )
            raise ValueError(msg)

        record(
            RerankReport(model_id=self.model_id, pairs=len(head), truncated_from=len(candidates))
        )
        rescored = [
            candidate.scored_by(self.name, score)
            for candidate, score in zip(head, scores, strict=True)
        ]
        rescored.sort(key=lambda candidate: (-candidate.scores[self.name], candidate.chunk.id))
        return rescored


__all__ = ["CrossEncoderReranker", "PairScorer"]
