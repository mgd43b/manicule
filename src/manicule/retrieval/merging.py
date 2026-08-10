"""Folding a leg's results into the list it was handed.

Every leg after the first receives whatever the previous stages produced and must return the
union: a chunk both legs found carries both scores, which is precisely what fusion reads and
what confidence's cross-leg agreement term counts.

It lives on its own because it is the one operation in the pipeline where mutating the input
would be natural, ordinary-looking, and undetectable downstream — the caller's list would be
reordered or re-scored, an earlier stage's record of what it produced would no longer be what
it produced, and the pipeline could not be replayed stage by stage. The stage contract forbids
returning the list you were given for this reason, and this function is why no leg has to
think about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.retrieval import Candidate


def union_scored(
    existing: Sequence[Candidate], found: Sequence[Candidate], stage: str
) -> list[Candidate]:
    """A new list holding ``existing`` then whatever of ``found`` is new, scored by ``stage``.

    A candidate present in both keeps every score it already carried and gains this stage's.
    Order follows first appearance, which keeps the fold deterministic without implying that
    the concatenation is a ranking — the next stage supplies that.

    ``scored_by`` sets the stage's score as the candidate's *effective* score, so after a merge
    the effective score is this leg's opinion. That is only ever read by a stage that has been
    told which key to read, and fusion reads the names it was configured with rather than
    whichever key a store happened to write.
    """
    merged: dict[str, Candidate] = {}
    order: list[str] = []
    for candidate in existing:
        merged[candidate.chunk.id] = candidate
        order.append(candidate.chunk.id)
    for candidate in found:
        chunk_id = candidate.chunk.id
        previous = merged.get(chunk_id)
        if previous is None:
            order.append(chunk_id)
            merged[chunk_id] = candidate.scored_by(stage, candidate.score)
            continue
        combined = {**previous.scores, **candidate.scores}
        merged[chunk_id] = previous.model_copy(update={"scores": combined}).scored_by(
            stage, candidate.score
        )
    return [merged[chunk_id] for chunk_id in order]


__all__ = ["union_scored"]
