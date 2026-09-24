"""Cross-workspace search: one scoped leg per workspace, merged, and never one unscoped query.

``docs/retrieval.md`` §3.2 settles the rule this module implements, and the rule is about
*merging*: every chunk lives in exactly one workspace, so reciprocal rank fusion degenerates
into "whichever workspace returned the shorter list wins", and BM25 is not comparable across
workspaces because its IDF is computed over the whole lexical index while relevance is being
judged per workspace. What *is* comparable is the dense leg's cosine — every vector came from
one model in one space, which the application runtime checks per workspace before a leg is
opened — and, when a reranker ran, the reranker's score over the merged pool.

Three properties follow, and each is a way a merged search could be wrong while looking fine.

**Each leg is the ordinary dense leg, bound to one workspace's handles.** The hydrating join
inside it (§4.2) is what makes a vector search a scoped search, and a leg that skipped it would
turn N scoped queries back into one unscoped one. Binding the configured stage to another
workspace's stores, rather than writing a second search, is what keeps that join the one join.

**Each leg over-fetches on its own.** The top-k trap the dense leg exists to avoid — match
first, filter afterwards, return nothing — has a cross-workspace form: fetch the top ``k`` of
the union and let the per-workspace join drop what it drops. A workspace whose best rows were
all soft-deleted would then contribute nothing even though its next-best beat everything else.
So every leg runs its own over-fetch against its own live fraction, and the merged list is
re-trimmed *after* every leg has filtered, never before.

**Attribution is kept, not reconstructed.** Every merged candidate records which workspaces
returned it. A correct store returns a chunk from exactly one — a chunk id is a digest of its
document id, which is a digest of its workspace — so a chunk offered by two legs is a store that
ignored its scope, and the application service's per-workspace identity check refuses it. The
merge does not guess which leg to believe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.retrieval.cache import CachedRanking, rehydrate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.protocols import DocStore, VectorStore
    from manicule.core.retrieval import Candidate, Filter, Query


@dataclass(frozen=True, slots=True)
class WorkspaceLeg:
    """One workspace's read handles, for the scoped dense leg a cross-workspace search runs.

    Both handles are scoped to :attr:`workspace` by construction — the document store carries
    it on every statement, and the vector store is that workspace's directory or collection —
    so nothing a leg does can name another tenant's rows, and the join inside the leg would
    drop them if it did.
    """

    workspace: str
    docstore: DocStore
    vectors: VectorStore


@dataclass(frozen=True, slots=True)
class Merged:
    """Every leg's candidates in one ranking, with where each one came from."""

    candidates: list[Candidate]
    origins: dict[str, tuple[str, ...]]
    """Chunk id to the workspaces whose leg returned it, in leg order. One entry each, unless a
    store ignored its scope — see the module docstring for why that is not resolved here."""


def leg_query(query: Query, workspace: str) -> Query:
    """The query one leg runs: the same request, scoped to exactly one workspace.

    Every other field is carried unchanged, the collection ids included. They are the union of
    every named workspace's resolutions, and that is safe to hand each leg: membership resolves
    through the leg's own scoped store, which answers an id from another workspace with no
    members — so each leg ends up restricted to its own workspace's collections, and a leg none
    of whose collections are its own matches nothing rather than everything.
    """
    scoped = query.filter.model_copy(update={"workspace_ids": frozenset({workspace})})
    return query.model_copy(update={"filter": scoped})


def require_one_leg_per_workspace(query: Query, legs: Sequence[WorkspaceLeg]) -> None:
    """Refuse legs that do not cover the query's workspaces exactly once each.

    A workspace named with no leg is a partial answer reported as a whole one: the merged
    ranking would simply lack that workspace's passages and nothing would say so. A leg for a
    workspace nobody named is a widening. Both are refused rather than repaired.

    Raises:
        ValueError: A workspace has no leg, two legs, or a leg nobody asked for.
    """
    offered = [leg.workspace for leg in legs]
    doubled = sorted({name for name in offered if offered.count(name) > 1})
    named = query.filter.workspace_ids
    missing = sorted(named - set(offered))
    extra = sorted(set(offered) - named)
    if doubled or missing or extra:
        problems = [
            f"{label}: {', '.join(repr(name) for name in names)}"
            for label, names in (
                ("offered twice", doubled),
                ("named with no leg", missing),
                ("offered and not named", extra),
            )
            if names
        ]
        msg = (
            f"a cross-workspace search needs exactly one leg per named workspace "
            f"({'; '.join(problems)}). A missing leg is a partial answer that reports itself as "
            f"a whole one, and an extra one searches a workspace nobody asked for."
        )
        raise ValueError(msg)


def merge_on_similarity(ranked: Sequence[tuple[str, Sequence[Candidate]]], *, stage: str) -> Merged:
    """Merge per-workspace dense rankings into one, ordered by that leg's cosine.

    Descending by ``stage``'s score, then by chunk id, so two candidates scored identically
    merge in the same order on every run — an unstable tie-break would make two runs of one
    search incomparable for no reason. A chunk offered by more than one leg is kept once and
    keeps every origin.

    Args:
        ranked: ``(workspace, candidates)`` per leg, in the order the legs ran.
        stage: The dense leg's name, which is the score key every candidate carries.

    Raises:
        ValueError: A candidate carries no ``stage`` score. It cannot be placed on the one
            scale this merge is allowed to use, and sinking it to the bottom would rank it by
            omission.
    """
    kept: dict[str, Candidate] = {}
    origins: dict[str, list[str]] = {}
    for workspace, candidates in ranked:
        for candidate in candidates:
            if stage not in candidate.scores:
                msg = (
                    f"a candidate from workspace {workspace!r} carries no {stage!r} score, so it "
                    f"has no cosine to merge on. Scores recorded: "
                    f"{', '.join(sorted(candidate.scores)) or 'none'}."
                )
                raise ValueError(msg)
            chunk_id = candidate.chunk.id
            kept.setdefault(chunk_id, candidate)
            seen = origins.setdefault(chunk_id, [])
            if workspace not in seen:
                seen.append(workspace)
    ordered = sorted(
        kept.values(), key=lambda candidate: (-candidate.scores[stage], candidate.chunk.id)
    )
    return Merged(
        candidates=ordered,
        origins={chunk_id: tuple(workspaces) for chunk_id, workspaces in origins.items()},
    )


async def rehydrate_across(
    entry: CachedRanking, joins: Mapping[str, tuple[DocStore, Filter]]
) -> tuple[list[Candidate], dict[str, tuple[str, ...]]] | None:
    """Rebuild a cached cross-workspace ranking through each workspace's own join, or ``None``.

    The single-workspace rule, applied per workspace: each chunk is re-read and re-joined
    through the store of every workspace that returned it, under that workspace's resolved
    filter, and **anything dropped makes the whole entry stale**. A cached ranking cannot serve
    a soft-deleted or foreign chunk for the reason it cannot on one workspace — it holds ids and
    scores, never content — and a shortened list would be a correct answer to a different
    question.

    Args:
        entry: A ranking recorded with :attr:`~manicule.retrieval.cache.CachedRanking.origins`.
        joins: Per workspace, its store and the join filter its leg ran under.
    """
    rebuilt: dict[int, Candidate] = {}
    for workspace, (docstore, join) in joins.items():
        positions = [
            position for position, origins in enumerate(entry.origins) if workspace in origins
        ]
        if not positions:
            continue
        subset = CachedRanking(
            chunk_ids=tuple(entry.chunk_ids[position] for position in positions),
            scores=tuple(entry.scores[position] for position in positions),
            identity=entry.identity,
        )
        candidates = await rehydrate(subset, docstore, join)
        if candidates is None:
            return None
        for position, candidate in zip(positions, candidates, strict=True):
            rebuilt.setdefault(position, candidate)
    if len(rebuilt) != len(entry.chunk_ids):
        # A chunk no leg of this search can vouch for: its origin is a workspace this query did
        # not name — which a key that includes the workspace set should make unreachable — or it
        # has none, because the entry is a single workspace's ranking and was never one of these.
        return None
    ordered = [rebuilt[position] for position in range(len(entry.chunk_ids))]
    origins = dict(zip(entry.chunk_ids, entry.origins, strict=True))
    return ordered, origins


__all__ = [
    "Merged",
    "WorkspaceLeg",
    "leg_query",
    "merge_on_similarity",
    "rehydrate_across",
    "require_one_leg_per_workspace",
]
