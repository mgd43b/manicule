"""The evidence ledger: every passage a run has seen, once, in a stable order.

Three rules, and each of them is a defect that was reasoned about rather than a preference.

**Deduplicated by ``chunk.id``.** Nothing downstream enforces uniqueness in a
:class:`~manicule.core.retrieval.Context`, and the citation binder deduplicates by *slot*
rather than by chunk — so the same passage arriving from two sub-questions would be numbered
twice, cited twice, and counted twice in every figure built on
:class:`~manicule.generation.answers.CitationAccounting`.

**Scores are merged by taking the maximum per stage, never overwritten.** That is the rule
:func:`manicule.retrieval.expansion.merge_rankings` already settled for the two-query glossary
path: it is the only combination that cannot make a second search *lower* a confidence the
first search had earned.

**Order is by best score, and it is fixed once :meth:`ranked` is called.** Slot numbers are
positional into the assembled context, so the ordering here is the citation numbering. A
ledger that re-sorted after the prompt was rendered would produce citations that pass every
level of verification and name the wrong passage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from manicule.core.retrieval import Candidate


class EvidenceLedger:
    """Accumulates candidates across a run's retrievals, de-duplicated by chunk id.

    Insertion order is remembered and is the tie-break, so a run over the same corpus with the
    same plan produces the same numbering — which is what makes two research runs comparable at
    all.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[str, Candidate] = {}
        self._first_seen: dict[str, int] = {}
        self._support: dict[str, int] = {}
        self._added = 0

    def __len__(self) -> int:
        return len(self._by_chunk)

    @property
    def support(self) -> dict[str, int]:
        """How many sub-questions retrieved each chunk, by chunk id.

        Corroboration across independently-asked sub-questions, recorded because it is the one
        signal a multi-step run has that a single retrieval does not. It is **not** folded into
        any score: ``retrieval.confidence`` is computed per run from that run's own pipeline
        identity, and a number this module invented has no business in it.
        """
        return dict(self._support)

    def add(self, candidates: Iterable[Candidate]) -> int:
        """Merge one retrieval's candidates in, and report how many were new.

        Returns:
            The number of chunks this call added that the ledger had not already seen. The
            count a caller records as ``fresh``: a cycle that returns nothing new is the
            signal that another cycle is not worth its latency.
        """
        fresh = 0
        for candidate in candidates:
            chunk_id = candidate.chunk.id
            self._support[chunk_id] = self._support.get(chunk_id, 0) + 1
            existing = self._by_chunk.get(chunk_id)
            if existing is None:
                self._by_chunk[chunk_id] = candidate
                self._first_seen[chunk_id] = self._added
                self._added += 1
                fresh += 1
                continue
            self._by_chunk[chunk_id] = _merged(existing, candidate)
        return fresh

    def ranked(self) -> tuple[Candidate, ...]:
        """Every passage, best first, ties broken by when it was first seen.

        Deterministic in full: the same retrievals in the same order produce the same tuple,
        which is what lets a run be replayed and compared rather than merely repeated.
        """
        return tuple(
            sorted(
                self._by_chunk.values(),
                key=lambda candidate: (-candidate.score, self._first_seen[candidate.chunk.id]),
            )
        )


def _merged(existing: Candidate, arriving: Candidate) -> Candidate:
    """One chunk seen twice, keeping the strongest evidence for it.

    The effective score is the larger of the two and each stage's score is merged by maximum,
    so a passage that ranked poorly for one sub-question and well for another is held at its
    best — the same rule the glossary's two-query merge uses, and for the same reason.

    ``publication_id`` is deliberately taken from whichever candidate is kept rather than
    reconciled: two candidates for one chunk id from different publications cannot both be
    live, and inventing a rule for a state the store does not produce would be a rule nobody
    could test.
    """
    if arriving.score > existing.score:
        winner, loser = arriving, existing
    else:
        winner, loser = existing, arriving
    scores = dict(winner.scores)
    for stage, score in loser.scores.items():
        current = scores.get(stage)
        if current is None or score > current:
            scores[stage] = score
    return winner.model_copy(update={"scores": scores})


def corroborated(passages: Sequence[Candidate], support: dict[str, int]) -> int:
    """How many of ``passages`` more than one sub-question found.

    Reported on its own rather than blended into confidence, because it measures a different
    thing: confidence describes one retrieval's ranking, and this describes agreement between
    several. Two numbers that answer two questions stay two numbers.
    """
    return sum(1 for candidate in passages if support.get(candidate.chunk.id, 0) > 1)


__all__ = ["EvidenceLedger", "corroborated"]
