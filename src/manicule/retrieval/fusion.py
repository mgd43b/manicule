"""Reciprocal rank fusion: ranks in, one ordering out.

    ``rrf(d) = Σ over legs where d appears:  1 / (K + rank_leg(d))``    K = 60, ranks 1-based

**The entire reason to use RRF is that it does not need the legs' scores to be comparable**,
and they are not. Cosine similarity is a bounded, absolute, model-defined quantity; BM25 is an
unbounded, corpus-relative one whose sign is negative and whose better values are more
negative. No scaling makes them commensurable across corpora. Discarding the magnitudes and
keeping the order is not an approximation — it is the point.

So there is no score weighting, no per-leg weighting and no normalisation step, and adding one
is not a tuning knob. Multiplying each rank term by the item's own score is the shape that
looks harmless and fights itself: with a lexical score derived from ``bm25()`` by taking its
absolute value and inverting, the best lexical hit gets the *smallest* weight, the rows arrive
in the right order and are then reweighted worst-first, and the output is still a plausible
ranked list with nothing raised.

What ``K`` is doing is worth understanding before anyone tunes it. **RRF is a consensus
operator, not a ranking operator.** At ``K = 60`` with 20 candidates a leg, the last rung
scores 76% of the first — within-leg ordering is compressed almost flat on purpose, so that
appearing in *both* legs dominates appearing high in one. Lowering ``K`` sharpens within-leg
ranking and weakens the consensus effect, which is the opposite of why RRF was chosen. The
consequence is that the fused list is a good candidate set and a mediocre final ordering, which
is exactly the job description for a cross-encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.retrieval.config import FusionConfig
from manicule.retrieval.trace import FusionReport, not_comparable, record

if TYPE_CHECKING:
    from manicule.core.content import Metadata
    from manicule.core.retrieval import Candidate, Query

DEGRADED_REASON = "a configured fusion leg contributed no candidates"


class RRFStage:
    """Fuse the configured legs' rank ladders.

    The legs are named in configuration rather than hardcoded, which is what makes "is a
    learned-sparse leg better than BM25" and "are three legs better than two" configuration
    edits and therefore measurements, rather than rewrites of this file.
    """

    def __init__(self, *, config: FusionConfig | None = None, name: str = "rrf") -> None:
        self.name = name
        self._config = config or FusionConfig()

    @property
    def legs(self) -> tuple[str, ...]:
        """The stage names this fusion reads. Checked against the pipeline at startup."""
        return self._config.legs

    @property
    def k(self) -> int:
        """The RRF constant, recorded on every run that uses it."""
        return self._config.k

    def describe(self) -> Metadata:
        """Which legs, and with what constant. Both change every number downstream."""
        return dict(self._config.model_dump(mode="json"))

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        """Recover each leg's ladder from the flat list, then fuse.

        Fusion receives one list, not a list per leg, and the ladders are still in it: a
        candidate carries a leg's score if and only if that leg found it. Recovering them is
        exact rather than approximate, because each leg's score is monotone within that leg —
        cosine descending is the dense order, and the store already negates ``bm25()`` so that
        higher is better there too.
        """
        del query  # fusion is a function of the ladders alone
        fused: dict[str, float] = {}
        per_leg: dict[str, int] = {}
        seen: dict[str, set[str]] = {}

        for leg in self._config.legs:
            ladder = [candidate for candidate in candidates if leg in candidate.scores]
            # Descending by that leg's own score, then by chunk id, so two candidates a leg
            # scored identically fuse in the same order on every run. An unstable tie-break
            # would make two runs of one pipeline incomparable for no reason at all.
            ladder.sort(key=lambda candidate: (-candidate.scores[leg], candidate.chunk.id))
            per_leg[leg] = len(ladder)
            for rank, candidate in enumerate(ladder, start=1):
                chunk_id = candidate.chunk.id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self._config.k + rank)
                seen.setdefault(chunk_id, set()).add(leg)

        wanted = set(self._config.legs)
        overlap = sum(1 for legs in seen.values() if legs == wanted)
        degraded = any(count == 0 for count in per_leg.values())
        record(
            FusionReport(
                legs=self._config.legs,
                k=self._config.k,
                per_leg=per_leg,
                overlap=overlap,
                degraded=degraded,
            )
        )
        if degraded:
            not_comparable(DEGRADED_REASON)

        # A candidate no configured leg scored contributes nothing and sorts last rather than
        # being dropped: fusion orders candidates, it does not filter them, and a stage that
        # quietly removed what it did not understand would make the pipeline order-dependent.
        ordered = sorted(
            candidates,
            key=lambda candidate: (-fused.get(candidate.chunk.id, 0.0), candidate.chunk.id),
        )
        return [
            candidate.scored_by(self.name, fused.get(candidate.chunk.id, 0.0))
            for candidate in ordered
        ]


__all__ = ["DEGRADED_REASON", "RRFStage"]
