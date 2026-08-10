"""Splitting a filter between the vector store and the join, and the case that must not merge."""

from __future__ import annotations

import pytest

from manicule.core.retrieval import Filter
from manicule.retrieval import prefilter
from manicule.retrieval.trace import Regime
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import EXEMPT_FILTER_FIELDS, PUSHED_DOWN_FILTER_FIELDS
from tests.retrieval.fakes import SCOPE
from tests.storage_helpers import make_document


def test_the_pushdown_list_matches_the_one_the_vector_store_enforces() -> None:
    """A duplicated constant nothing compares is a constant that drifts.

    Retrieval cannot import the vector store's copy — that would drag LanceDB and pyarrow into
    a package which has not chosen a vector store yet — so the copy is held to the original
    here, where both are importable.
    """
    assert prefilter.PUSHED_DOWN_FIELDS == PUSHED_DOWN_FILTER_FIELDS
    assert prefilter.WORKSPACE_FIELD in EXEMPT_FILTER_FIELDS


def test_the_three_sets_partition_the_filter() -> None:
    """Every field is resolved somewhere, and no field is resolved twice.

    A field in neither set is one both stores would drop, which is a restriction that silently
    does not apply — the exact failure the stores' refusals exist to prevent, arriving through
    a gap between them instead.
    """
    everything = set(Filter.model_fields)
    accounted = (
        prefilter.PUSHED_DOWN_FIELDS | prefilter.JOIN_REQUIRING_FIELDS | {prefilter.WORKSPACE_FIELD}
    )
    assert accounted == everything
    assert not (prefilter.PUSHED_DOWN_FIELDS & prefilter.JOIN_REQUIRING_FIELDS)


async def test_no_join_requiring_field_means_do_not_constrain(store: SqliteDocStore) -> None:
    """Nothing to resolve, so nothing is resolved and the store is asked what it can answer."""
    split = await prefilter.resolve(
        Filter(workspace_ids=SCOPE, langs=frozenset({"en"})), store, prefilter_id_limit=1000
    )

    assert split.regime is Regime.UNRESOLVED
    assert not split.matches_nothing
    assert split.pushdown.langs == frozenset({"en"})


async def test_a_join_requiring_field_that_matches_nothing_constrains_to_nothing(
    store: SqliteDocStore,
) -> None:
    """The opposite instruction, from an identical empty id set.

    This is the case that has to be written down. A query filtered to a collection that happens
    to be empty must return nothing; treating it as "no restriction" returns the whole
    workspace, ranked and plausible, with every result violating the filter that was asked for.
    """
    split = await prefilter.resolve(
        Filter(workspace_ids=SCOPE, sources=frozenset({"nowhere"})), store, prefilter_id_limit=1000
    )

    assert split.regime is Regime.EMPTY
    assert split.matches_nothing


async def test_a_small_resolved_set_is_pushed_down(store: SqliteDocStore) -> None:
    """Below the limit, selectivity becomes the vector store's problem rather than ours."""
    document = make_document(source="fs", source_id="a")
    await store.upsert_document(document)

    split = await prefilter.resolve(
        Filter(workspace_ids=SCOPE, sources=frozenset({"fs"})), store, prefilter_id_limit=1000
    )

    assert split.regime is Regime.PREFILTER
    assert split.pushdown.document_ids == frozenset({document.id})
    assert split.resolved_id_count == 1
    assert split.count_is_exact


async def test_too_many_ids_invert_the_plan_without_materialising_them_all(
    store: SqliteDocStore,
) -> None:
    """Resolution stops one past the limit rather than listing a corpus to answer a threshold.

    The count then is a lower bound, and it says so: a figure recorded as exact when it is not
    would skew the very distribution the threshold is meant to be set from.
    """
    for index in range(5):
        await store.upsert_document(make_document(source="fs", source_id=f"doc-{index}"))

    split = await prefilter.resolve(
        Filter(workspace_ids=SCOPE, sources=frozenset({"fs"})), store, prefilter_id_limit=2
    )

    assert split.regime is Regime.POSTFILTER
    assert split.pushdown.document_ids == frozenset()
    assert not split.count_is_exact
    assert split.resolved_id_count == 3  # the limit, plus the one that proved there were more


async def test_the_join_carries_every_join_requiring_field(store: SqliteDocStore) -> None:
    """The join is the tenancy boundary *and* the post-filter, in one statement.

    Writing the post-filter separately would be a second predicate to keep in step with the
    first, and the one that falls behind returns rows the filter excluded.
    """
    await store.upsert_document(make_document(source="fs", source_id="a"))

    split = await prefilter.resolve(
        Filter(workspace_ids=SCOPE, sources=frozenset({"fs"})), store, prefilter_id_limit=1000
    )

    assert split.join.sources == frozenset({"fs"})
    assert split.join.workspace_ids == SCOPE


async def test_a_field_no_store_can_resolve_is_refused_rather_than_dropped(
    store: SqliteDocStore,
) -> None:
    """Collection and tag membership need join tables no store here has.

    Refused rather than ignored, and by the store rather than by a hardcoded list in retrieval
    — so a store that later grows the capability simply works, and one that has not says which
    field it cannot honour.
    """
    with pytest.raises(ValueError, match="collection_ids"):
        await prefilter.resolve(
            Filter(workspace_ids=SCOPE, collection_ids=frozenset({"c1"})),
            store,
            prefilter_id_limit=1000,
        )
