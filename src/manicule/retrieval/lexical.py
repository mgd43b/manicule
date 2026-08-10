"""The lexical leg: BM25 over the authoritative store.

The statement itself is settled and built in storage — one joined query over the FTS index,
``chunks`` and ``documents``, with the filter applied inline and ``LIMIT`` last. This stage
adds three things and changes none of it.

**It re-keys the score.** The store returns candidates carrying a ``bm25`` score — a key
naming the *algorithm*. ``Candidate.scores`` is keyed by *stage*, so the stage writes its own
name over the top and both keys survive. That indirection is what lets this leg be swapped for
a learned-sparse one without touching the fusion stage: fusion reads the leg names it was
configured with, never a key some store happened to write.

**It merges rather than replaces.** A chunk both legs found carries both scores, which is what
fusion and the cross-leg agreement term are computed from.

**Zero results is an event, not a warning.** An empty match is legitimate — an all-stopword
query, a query that tokenizes to nothing — and it is also what a failure of the lexical index
looks like. Either way the pipeline continues on one leg and produces a well-formed ranking,
so the run records that it was single-leg and becomes inadmissible as a measurement. Logging
it instead would leave an evaluation harness unable to tell a hard corpus from a broken index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.retrieval.merging import union_scored
from manicule.retrieval.trace import LexicalReport, not_comparable, record

if TYPE_CHECKING:
    from manicule.core.content import Metadata
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Candidate, Query
    from manicule.retrieval.profile import Profiles

DEGRADED_REASON = "the lexical leg returned nothing, so this run was fused from one ladder"
"""Why a single-leg run may not be compared with a two-leg one.

The harness must refuse the comparison rather than averaging over it. This is the difference
between a metric that moved because the corpus is hard and one that moved because the lexical
index threw — both are legitimate outcomes of a query and only one is a legitimate input to a
measurement.
"""


class LexicalStage:
    """BM25 search, merged into whatever the pipeline has so far."""

    def __init__(self, *, docstore: DocStore, profiles: Profiles, name: str = "lexical") -> None:
        self.name = name
        self._docstore = docstore
        self._profiles = profiles

    def describe(self) -> Metadata:
        """Nothing to declare: the statement and its whole filter belong to the store."""
        return {}

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        """Search, merge, and record whether this leg had an opinion at all.

        The whole filter goes to the store, unsplit. This leg needs none of the dense leg's
        pre-filter machinery: it is one statement against the store that owns every column the
        filter names, so the restriction is applied before ``LIMIT`` rather than around it.
        """
        profile = self._profiles.for_query(query)
        k = max(profile.candidates, query.limit)

        found = await self._docstore.search_lexical(query.text, k, query.filter)
        degraded = not found
        record(
            LexicalReport(query_text=query.text, requested=k, matched=len(found), degraded=degraded)
        )
        if degraded:
            not_comparable(DEGRADED_REASON)
        return union_scored(candidates, found, self.name)


__all__ = ["DEGRADED_REASON", "LexicalStage"]
