"""Deciding what the vector store is asked and what is resolved before it (§3.3).

The vector table has a column for three of a filter's fields and no columns for the rest, so
every dense search has to split the restriction in two. The split has two regimes — push a
resolved document-id list down, or over-fetch and post-filter — plus one early exit that has
to be written down explicitly because collapsing it is a filter bypass.

The threshold between the regimes is genuinely unknown and this module does not pretend
otherwise. What it settles is the *decision procedure*, and the fact that every query records
the two inputs that would set the number from a real corpus rather than from an argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from manicule.core.retrieval import Filter
from manicule.retrieval.trace import Regime

if TYPE_CHECKING:
    from manicule.core.protocols import DocStore

PUSHED_DOWN_FIELDS: Final = frozenset({"document_ids", "kinds", "langs"})
""":class:`~manicule.core.retrieval.Filter` fields the vector table has a promoted column for.

Stated here rather than imported from :mod:`manicule.storage.vectors`, because importing that
module drags LanceDB and pyarrow into a package that otherwise needs neither, and this list
has to be readable by a stage that has not chosen a vector store yet. The copy is held to the
original by ``tests/retrieval/test_prefilter.py``, which imports both and fails if they
diverge — a duplicated constant nothing compares is a constant that drifts.
"""

WORKSPACE_FIELD: Final = "workspace_ids"
"""Resolved by neither store: the hydrating join enforces it (``docs/retrieval.md`` §4.2)."""

JOIN_REQUIRING_FIELDS: Final = frozenset(
    {"sources", "media_types", "collection_ids", "tag_ids", "updated_after", "updated_before"}
)
"""Fields that need a join the vector table has no columns for.

Each is a property of a *document*, so each is resolved in the document store and reaches the
vector store — if it reaches it at all — as ``document_ids``.
"""


class Resolution:
    """How one query's filter was split, and what the trace should say about it.

    Immutable by convention; it is built once per dense search and read from there.
    """

    __slots__ = ("count_is_exact", "join", "pushdown", "regime", "resolved_id_count")

    def __init__(
        self,
        *,
        pushdown: Filter,
        join: Filter,
        regime: Regime,
        resolved_id_count: int = 0,
        count_is_exact: bool = True,
    ) -> None:
        self.pushdown = pushdown
        """What the vector store is asked. Only fields it has a column for, plus the workspace
        it is exempt from honoring."""

        self.join = join
        """What the hydrating join applies. The scope, and every join-requiring field —
        so the join is both the tenancy boundary and the post-filter, in one statement."""

        self.regime = regime
        self.resolved_id_count = resolved_id_count
        self.count_is_exact = count_is_exact

    @property
    def matches_nothing(self) -> bool:
        """Whether the filter resolved to no document at all.

        **Not the same as "nothing was resolved".** Both produce an empty id set and they are
        opposite instructions: one means *do not constrain*, the other means *constrain to
        nothing*. A query filtered to a collection that happens to be empty must return
        nothing — collapsing the two would return the whole workspace, ranked and plausible.
        """
        return self.regime is Regime.EMPTY


def _restrict(source: Filter, fields: frozenset[str]) -> Filter:
    """A filter carrying ``source``'s workspace and only the named fields."""
    kept = {name: getattr(source, name) for name in fields}
    return Filter(workspace_ids=source.workspace_ids, **kept)


def join_filter(source: Filter) -> Filter:
    """The document-level half of ``source``: what a hydrating join can apply.

    Everything a *document* is restricted by, and nothing a *chunk* is. ``kinds`` and ``langs``
    are chunk properties with no meaning in a query over ``documents``, so a store asked to
    apply them refuses — correctly, because silently dropping a restriction returns rows the
    filter was written to exclude.

    They are not dropped here either, and the distinction matters. On the dense leg they go to
    the vector store, which has a column for each. On a cache hit they were **already applied**
    when the ranking was computed, and a chunk id is derived from its content — so the chunk
    behind a cached id is the same chunk, of the same kind, in the same language. What can have
    changed since is exactly the document-level half: a soft delete, a status, a workspace, a
    source, an ``updated_at``. That is what this filter re-applies, and it is why re-applying
    only this half is complete rather than partial.
    """
    return _restrict(source, frozenset({"document_ids"}) | JOIN_REQUIRING_FIELDS)


async def resolve(
    query_filter: Filter, docstore: DocStore, *, prefilter_id_limit: int
) -> Resolution:
    """Split ``query_filter`` between the vector store and the hydrating join.

    Args:
        query_filter: The restriction the query carries, whole.
        docstore: The store for the workspace the query names.
        prefilter_id_limit: Above this many resolved documents, stop resolving and
            post-filter instead. Resolution stops one past the limit, so a query against a
            large corpus never materializes its whole document list to answer a question the
            first thousand rows already answered.

    Returns:
        A :class:`Resolution`. When :attr:`Resolution.matches_nothing` is set the caller must
        return no candidates rather than falling through to an unconstrained search.

    Raises:
        ValueError: The store cannot honor a field the filter sets — collection or tag
            membership, until the store that owns those relations can resolve them. Refused
            rather than dropped: a silently ignored restriction returns rows the filter was
            written to exclude, and the search still looks like it worked.
    """
    join = join_filter(query_filter)
    requested = frozenset(query_filter.restricting_fields)

    if not (requested & JOIN_REQUIRING_FIELDS):
        # Step 1: nothing to resolve. Tested on the *fields*, never on the result, because an
        # empty result means the opposite thing three lines below.
        return Resolution(
            pushdown=_restrict(query_filter, PUSHED_DOWN_FIELDS),
            join=join,
            regime=Regime.UNRESOLVED,
        )

    documents = await docstore.list_documents(join, limit=prefilter_id_limit + 1)
    resolved = [document.id for document in documents]

    if not resolved:
        # Step 3. The one that has to be written down.
        return Resolution(
            pushdown=_restrict(query_filter, PUSHED_DOWN_FIELDS),
            join=join,
            regime=Regime.EMPTY,
        )

    if len(resolved) <= prefilter_id_limit:
        # Step 4: small enough to push down. Selectivity is now the store's problem.
        pushdown = _restrict(query_filter, PUSHED_DOWN_FIELDS - {"document_ids"}).model_copy(
            update={"document_ids": frozenset(resolved)}
        )
        return Resolution(
            pushdown=pushdown,
            join=join,
            regime=Regime.PREFILTER,
            resolved_id_count=len(resolved),
        )

    # Step 5: too many to push down. The join applies them instead, over a batch bounded by
    # the over-fetch cap — which is why the join and the post-filter are the same statement.
    return Resolution(
        pushdown=_restrict(query_filter, PUSHED_DOWN_FIELDS),
        join=join,
        regime=Regime.POSTFILTER,
        resolved_id_count=len(resolved),
        count_is_exact=False,
    )


__all__ = [
    "JOIN_REQUIRING_FIELDS",
    "PUSHED_DOWN_FIELDS",
    "WORKSPACE_FIELD",
    "Resolution",
    "join_filter",
    "resolve",
]
