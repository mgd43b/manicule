"""The dense leg: embed, over-fetch, hydrate, floor.

Three operations that are deliberately **one stage**, because the middle two cannot be
separated without turning a security boundary into a configuration option.

The trap this stage exists to avoid was found and measured on the other leg. With ``k = 5``,
three matching live chunks, five in a soft-deleted document and five in another workspace,
matching first and filtering afterwards returned **zero** live in-workspace results — a total
loss of recall, silently, with a well-formed empty result set. The lexical leg fixed it by
filtering inside the statement, before ``LIMIT``.

**The dense leg cannot do that, and the reason is structural rather than an oversight.** The
vector table holds ``id``, ``vector``, ``document_id``, ``kind``, ``lang``, ``position`` and
the chunk itself. It holds no ``deleted_at``, no ``status`` and no ``workspace_id``, and it
holds none of them *deliberately*: liveness and tenancy live on ``documents`` in the
authoritative store, and copying them into a derived one creates a value that can disagree. So
a search returns ``k`` rows of which an unknown number are invisible, and the join that removes
them necessarily runs afterwards. The answer is to over-fetch by a factor the system measures
about itself, and to say out loud when even that was not enough.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from manicule.core.retrieval import Candidate, SupportsGeneration
from manicule.retrieval import prefilter
from manicule.retrieval.config import DenseConfig
from manicule.retrieval.hydration import visible_documents
from manicule.retrieval.merging import union_scored
from manicule.retrieval.ports import SupportsLiveChunkCount
from manicule.retrieval.profile import retrieval_depth
from manicule.retrieval.trace import DenseReport, Regime, Shortfall, record

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Metadata
    from manicule.core.protocols import DocStore, Embedder, VectorStore
    from manicule.core.retrieval import Filter, Query
    from manicule.retrieval.profile import Profiles

UNMEASURED_LIVE_FRACTION = 1.0
"""What the fraction is when no store will report it.

Chosen so the derivation lands exactly on ``overfetch_min``: a store that cannot say how
dilute its index is gets the floor, and the retry loop supplies whatever the floor missed. It
costs round trips on a dilute index and changes nothing about which candidates come back.
"""


def derive_over_fetch(k: int, live_fraction: float, config: DenseConfig) -> int:
    """How many rows to ask the vector store for, to end up with ``k`` live ones.

    A constant multiplier is wrong in both directions — 2x is too little for a fifty-workspace
    deployment and 20x is wasteful for a personal one, and neither knows which it is in. So the
    factor comes from a quantity the system can measure about itself, and the cap firing is not
    only a limit: **it is the detector for which regime this deployment is in.** A workspace
    holding 2% of the corpus computes a 50x factor, hits the cap, and the cap appearing in the
    trace is the signal that the pre-filter plan is the one it should be running.
    """
    fraction = min(1.0, max(0.05, live_fraction))
    wanted = math.ceil(k / fraction)
    wanted = max(wanted, config.overfetch_min * k)
    return min(wanted, config.overfetch_max * k, config.absolute_row_cap)


class DenseStage:
    """Nearest-neighbor search, scoped by a join the configuration cannot remove.

    The join is inside this stage rather than beside it because a pipeline that could be
    configured without it is a pipeline that can be configured into a cross-tenant leak, and
    configuration is not where a security boundary should live. Folding it in also buys the
    invariant the whole design rests on: *every* stage's output is already live, in-workspace
    and visible, so the assertion holds at any point rather than at one privileged one.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: VectorStore,
        docstore: DocStore,
        profiles: Profiles,
        config: DenseConfig | None = None,
        name: str = "dense",
        workspace: str | None = None,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._vectors = vectors
        self._docstore = docstore
        self._profiles = profiles
        self._config = config or DenseConfig()
        self._workspace = workspace
        self._fraction_cache: dict[tuple[int, str], float] = {}

    def describe(self) -> Metadata:
        """The settings this leg ran under, for the record.

        A leg bound to one workspace of a cross-workspace search also names that workspace,
        because every leg of one is called ``dense`` — its score key has to be the same in each,
        or the merge would have no one scale to read — and a trace of four spans all called
        ``dense`` is otherwise a trace nobody can attribute.
        """
        described = dict(self._config.model_dump(mode="json"))
        if self._workspace is not None:
            described["workspace"] = self._workspace
        return described

    def with_vectors(self, vectors: VectorStore) -> DenseStage:
        """Return this configured leg bound to the vector handle retrieval must use.

        Runtime storage may decorate the configured plugin store with a live-generation
        resolver.  Retrieval stages are constructed by the plugin container, so the
        composition root explicitly rebinds the built-in dense leg to that decorated handle
        rather than letting it retain the undecorated store it captured during construction.
        """
        return DenseStage(
            embedder=self._embedder,
            vectors=vectors,
            docstore=self._docstore,
            profiles=self._profiles,
            config=self._config,
            name=self.name,
            workspace=self._workspace,
        )

    def for_workspace(
        self, workspace: str, *, vectors: VectorStore, docstore: DocStore
    ) -> DenseStage:
        """Return this configured leg bound to another workspace's two stores.

        What a cross-workspace search runs once per workspace (``docs/retrieval.md`` §3.2). The
        embedder, the profiles and the over-fetch configuration are this leg's, so every
        workspace is searched by one model under one set of rules — which is what makes their
        cosines one scale. The hydrating join is this leg's too, run against ``docstore``: the
        bound leg is as scoped as the one it was made from, and could not be made less so by a
        caller that forgot something, because there is nothing to pass.
        """
        return DenseStage(
            embedder=self._embedder,
            vectors=vectors,
            docstore=docstore,
            profiles=self._profiles,
            config=self._config,
            name=self.name,
            workspace=workspace,
        )

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        """Search, scope, floor, and merge into ``candidates`` without touching it."""
        profile = self._profiles.for_query(query)
        k = retrieval_depth(profile, query)
        config = self._config

        split = await prefilter.resolve(
            query.filter, self._docstore, prefilter_id_limit=config.prefilter_id_limit
        )
        if split.matches_nothing:
            record(
                DenseReport(
                    requested=k,
                    fetched=0,
                    survived=0,
                    over_fetch=0,
                    live_fraction=0.0,
                    live_fraction_measured=False,
                    regime=Regime.EMPTY,
                    outcome=Shortfall.SATISFIED,
                )
            )
            return list(candidates)

        # **The query half of the prefix scheme, and it has to be here.** Several retrieval
        # models are trained asymmetrically — `nomic-embed-text` with
        # `search_query:`/`search_document:`, Qwen3-Embedding with a query-side instruction —
        # so the same string embedded as a query and as a document is meant to produce
        # different vectors. The embedder cannot make that distinction: `Embedder.embed` is
        # its only entry point and `ingest/embedding.py` calls it identically.
        #
        # Doing only the document half would be worse than doing neither, because a corpus
        # embedded with `search_document:` and searched with bare queries retrieves worse than
        # one embedded with neither and nothing raises. That is why the scheme is one value in
        # the embedding fingerprint rather than a middleware on the ingest side: a query has no
        # stored row for `embedding_input_identity` to key on, so middleware could express the
        # document half and could never express this one (`docs/embeddings.md` §9.1).
        scheme = self._embedder.fingerprint.prefix_scheme
        vector = (await self._embedder.embed([scheme.query(query.text)]))[0]
        fraction, measured = await self._live_fraction(query)
        over_fetch = derive_over_fetch(k, fraction, config)

        kept: list[Candidate] = []
        fetched = 0
        by_join = 0
        by_floor = 0
        expansions = 0
        outcome = Shortfall.SATISFIED
        while True:
            rows = await self._vectors.search(vector, over_fetch, split.pushdown)
            fetched = len(rows)
            live = await self._hydrate(rows, split.join)
            by_join = fetched - len(live)
            kept = [c for c in live if max(c.score, 0.0) >= profile.min_score]
            by_floor = len(live) - len(kept)

            if len(kept) >= k:
                outcome = Shortfall.SATISFIED
                break
            if fetched < over_fetch:
                # The store returned fewer rows than it was asked for, so the predicate admits
                # no more. Every candidate it can offer has been examined; expanding would ask
                # the same question again.
                outcome = Shortfall.EXHAUSTED_CORPUS
                break
            if expansions >= config.max_expansions or over_fetch >= config.absolute_row_cap:
                outcome = Shortfall.EXHAUSTED_BUDGET
                break
            over_fetch = min(over_fetch * config.expansion_factor, config.absolute_row_cap)
            expansions += 1

        record(
            DenseReport(
                requested=k,
                fetched=fetched,
                survived=len(kept),
                over_fetch=over_fetch,
                live_fraction=fraction,
                live_fraction_measured=measured,
                dropped_by_join=by_join,
                dropped_by_min_score=by_floor,
                expansions=expansions,
                outcome=outcome,
                regime=split.regime,
                resolved_id_count=split.resolved_id_count,
                resolved_id_count_exact=split.count_is_exact,
            )
        )
        return union_scored(candidates, kept[:k], self.name)

    async def _hydrate(self, rows: Sequence[Candidate], join: Filter) -> list[Candidate]:
        """The join that makes a vector search a scoped search.

        One statement does three jobs, and they are the same statement on purpose. It applies
        the workspace, soft-delete and status boundary the vector table has no columns for; it
        applies the post-filter for whichever join-requiring fields were too numerous to push
        down; and it re-reads the chunk from SQLite, because SQLite is authoritative and a
        divergence between the two copies of a chunk's text must resolve toward the truth
        rather than toward whichever one the query happened to read.
        """
        if not rows:
            return []

        documents = await visible_documents(
            self._docstore, join, [candidate.chunk.document_id for candidate in rows]
        )
        if not documents:
            return []

        ordered = [
            row for row in rows if documents.get(row.chunk.document_id) == row.publication_id
        ]
        stored = {
            chunk.id: chunk
            for chunk in await self._docstore.get_chunks([row.chunk.id for row in ordered])
        }
        return [
            row.model_copy(update={"chunk": stored[row.chunk.id]})
            for row in ordered
            if row.chunk.id in stored
        ]

    async def _live_fraction(self, query: Query) -> tuple[float, bool]:
        """The share of vector-table rows a search in this workspace could return.

        Workspace-scoped and stopping there, which is what keeps it cacheable. Workspace and
        liveness are the two exclusions this leg *cannot* push down and must therefore absorb
        by over-fetching; everything else either has a column or took the pre-filter path, so
        it is already the store's problem. Folding the whole filter in would make this a
        per-query aggregate — two counts on the hot path of every search, to refine a number
        that is then clamped and rounded to a multiple anyway.

        Returns:
            The fraction, and whether it was measured. An unmeasured fraction is not an
            approximation of a measured one: it is the floor, with the retry loop behind it.
        """
        # Only the numerator is optional: every vector store reports its row count, because the
        # protocol requires it. A document store that cannot say how much of its workspace is
        # live is an ordinary store, and this leg simply starts at its floor.
        if not isinstance(self._docstore, SupportsLiveChunkCount):
            return UNMEASURED_LIVE_FRACTION, False

        key: tuple[int, str] | None = None
        if isinstance(self._docstore, SupportsGeneration):
            key = (self._docstore.generation, ",".join(sorted(query.filter.workspace_ids)))
            cached = self._fraction_cache.get(key)
            if cached is not None:
                return cached, True

        rows = await self._vectors.count()
        live = await self._docstore.live_chunk_count()
        # The denominator is the vector table's row count, not the chunk count: unswept
        # tombstones are still rows and still consume top-k slots, so counting SQLite's chunks
        # would call an index clean while it was full of pending deletions.
        fraction = 1.0 if rows <= 0 else min(1.0, live / rows)
        if key is not None:
            self._fraction_cache = {key: fraction}
        return fraction, True


__all__ = ["UNMEASURED_LIVE_FRACTION", "DenseStage", "derive_over_fetch"]
