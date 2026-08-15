"""The L1 query-result cache: it caches *decisions*, never content.

The cached value is the ranked list of chunk ids with their per-stage scores — the decision the
pipeline reached — and never the chunk text. On a hit, the ids are re-hydrated through the same
join the dense leg uses.

This is not a memory optimization. It is what makes the cache incapable of the failure a
content-caching version invites: **a hit cannot serve a soft-deleted, unindexed or
foreign-workspace chunk, because the entry holds no chunks.** The boundary is re-enforced on
every hit rather than snapshotted at the moment of the miss. Re-hydration costs one indexed
lookup against a full pipeline that includes an embedding forward pass and possibly a
cross-encoder.

**If hydration drops anything, the entry is stale: evict it and run the pipeline.** Returning a
shortened list would be correct and misleading — the ranking was computed over a candidate set
that no longer exists, and the replacement for the dropped candidate was never considered.

Invalidation is a generation counter, which the store bumps and the key includes, so one bump
invalidates everything at once with no eviction pass and no per-entry bookkeeping. An
in-process counter is sufficient because exactly one instance holds a data directory: the
writer and the reader are the same process, so there is no cross-process invalidation problem
to solve. A TTL sits underneath as a bound on staleness from anything the counter was not
taught about — belt-and-braces, never the mechanism.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from manicule.core.retrieval import Candidate
from manicule.retrieval.hydration import visible_documents

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Filter, PipelineIdentity, Query


@dataclass(frozen=True, slots=True)
class CachedRanking:
    """A decision the pipeline reached, without any of the content it reached it about."""

    chunk_ids: tuple[str, ...]
    scores: tuple[tuple[tuple[str, float], ...], ...]
    """Per candidate, that candidate's ``scores`` mapping as sorted pairs.

    Pairs rather than a dict so the entry is hashable and obviously immutable: a cached ranking
    that a caller could edit in place is a cache that answers a different question on the second
    hit.
    """

    identity: PipelineIdentity
    incomparable: tuple[str, ...] = ()
    """Why the run that populated this entry was inadmissible as a measurement.

    Carried because a hit is not a fresh run: it reports the identity *and the defects* of the
    run behind it. Dropping them would make a degraded run look clean the second time it was
    asked for.
    """

    exhausted_budget: bool = False
    """Whether a leg of the populating run stopped at its own caps. Caps confidence."""

    stored_at: float = field(default_factory=time.monotonic)


def cache_key(
    query: Query,
    *,
    generation: int,
    identity: PipelineIdentity,
    expanded: str = "",
) -> str:
    """A stable digest of everything that could change this query's ranking.

    Five of the inputs are worth saying out loud:

    * **``Query.limit``**, which looks like a presentation concern and is not. Retrieval depth
      is the larger of the limit and the profile's head, so a larger limit is a *deeper run*,
      and serving a cached ten-result ranking to a request for fifty returns a short list that
      looks like a corpus with nothing more in it.
    * **The whole filter**, not just the workspace. Two filters produce two rankings, and a key
      that omits one is a cache that answers a different question.
    * **The pipeline identity**, because comparing two pipelines is the evaluation harness's
      entire method and a cache that cannot tell them apart would serve one's ranking as the
      other's.
    * **The expanded query form**, when glossary lookup produced one. Two runs of one pipeline
      over one text return different rankings when a glossary term was defined between them, or
      when expansion was switched off, or when a second definition arrived and turned the term
      into a conflict — and every one of those changes this string. The generation counter
      catches the first because the definition is a row; it catches neither of the others,
      because they are configuration and scope rather than content.
    * **Not the conversation history.** Retrieval runs on the query text; nothing in this
      pipeline reads history. Including it would guarantee a miss on every turn of a
      conversation — the one place a user actually repeats themselves. If a history-conditioned
      query rewrite ever ships, history joins the key in the same commit.
    """
    payload = {
        "generation": generation,
        "filter": _canonical_filter(query.filter),
        "profile": query.profile.value,
        "overrides": _sorted(identity.overrides),
        "limit": query.limit,
        "pipeline": list(identity.stages),
        "reranker": identity.reranker_model_id,
        "rrf_k": identity.rrf_k,
        "embed_fingerprint": identity.embed_fingerprint,
        "text": query.text.strip(),
        "expanded": expanded.strip(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_filter(value: Filter) -> dict[str, object]:
    dumped = value.model_dump(mode="json")
    return {name: _sorted(dumped[name]) for name in sorted(dumped)}


def _sorted(value: object) -> object:
    if isinstance(value, list | set | frozenset | tuple):
        return sorted(str(item) for item in value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    if isinstance(value, dict):
        return {str(name): _sorted(item) for name, item in sorted(value.items())}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return value


class L1QueryCache:
    """Ranked chunk ids, keyed by everything that could change them.

    Bounded and least-recently-used. Disabled by construction when ``entries`` is zero, which
    is what an evaluation run sets: a hit is not a retrieval run, its latency is the cache's,
    and a quality metric computed from one is a metric computed twice from the same sample.
    """

    def __init__(self, *, entries: int = 512, ttl_s: float = 300.0) -> None:
        self._entries = max(0, entries)
        self._ttl_s = ttl_s
        self._store: OrderedDict[str, CachedRanking] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evicted_stale = 0

    @property
    def enabled(self) -> bool:
        return self._entries > 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: str) -> CachedRanking | None:
        """The ranking stored under ``key``, if it is there and has not aged out."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.monotonic() - entry.stored_at > self._ttl_s:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, key: str, ranking: CachedRanking) -> None:
        """Remember a ranking, evicting the least recently used if full."""
        if not self.enabled:
            return
        self._store[key] = ranking
        self._store.move_to_end(key)
        while len(self._store) > self._entries:
            self._store.popitem(last=False)

    def evict(self, key: str) -> None:
        """Forget one entry. Called when hydration proves it stale."""
        if self._store.pop(key, None) is not None:
            self.evicted_stale += 1

    def clear(self) -> None:
        self._store.clear()

    @staticmethod
    def record(
        candidates: Sequence[Candidate],
        identity: PipelineIdentity,
        *,
        incomparable: Sequence[str] = (),
        exhausted_budget: bool = False,
    ) -> CachedRanking:
        """Reduce a ranking to the decision it represents."""
        return CachedRanking(
            chunk_ids=tuple(candidate.chunk.id for candidate in candidates),
            scores=tuple(tuple(sorted(candidate.scores.items())) for candidate in candidates),
            identity=identity,
            incomparable=tuple(incomparable),
            exhausted_budget=exhausted_budget,
        )


async def rehydrate(
    entry: CachedRanking, docstore: DocStore, join: Filter
) -> list[Candidate] | None:
    """Rebuild a cached ranking's candidates, or ``None`` if the entry is stale.

    Stale means *anything* dropped: a chunk that no longer exists, a document that has been
    soft-deleted, one whose status has moved off ``indexed``, or one this workspace cannot see.
    The ranking was computed over a candidate set that no longer exists, and the candidate that
    would have replaced the dropped one was never considered — so a shortened list would be a
    correct answer to a question nobody asked.
    """
    if not entry.chunk_ids:
        return []
    chunks = list(await docstore.get_chunks(list(entry.chunk_ids)))
    if len(chunks) != len(entry.chunk_ids):
        return None
    by_id = {chunk.id: chunk for chunk in chunks}
    visible = await visible_documents(docstore, join, [chunk.document_id for chunk in chunks])
    if any(chunk.document_id not in visible for chunk in chunks):
        return None
    last = entry.identity.stages[-1] if entry.identity.stages else ""
    rebuilt: list[Candidate] = []
    for chunk_id, pairs in zip(entry.chunk_ids, entry.scores, strict=True):
        scores = dict(pairs)
        # The effective score is the last stage's, exactly as it was when the ranking was
        # computed. Recomputing it from whatever key looks largest would silently reorder a
        # ranking on its way out of the cache.
        effective = scores.get(last, 0.0)
        chunk = by_id[chunk_id]
        rebuilt.append(
            Candidate(
                chunk=chunk,
                publication_id=visible[chunk.document_id],
                score=effective,
                scores=scores,
            )
        )
    return rebuilt


__all__ = ["CachedRanking", "L1QueryCache", "cache_key", "rehydrate"]
