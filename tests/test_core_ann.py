"""The ANN lifecycle rule, checked without a vector store under it.

``manicule.core.ann`` is the one place ``docs/storage.md`` §6.2 is written down as code, and
everything in it is a pure function of numbers the caller already has. So it is tested here
rather than through LanceDB: a threshold comparison that only ever runs behind a real index
build is a rule nobody can exercise at the boundaries, and the boundaries are where a
lifecycle gets its transitions wrong.

``tests/test_storage_vectors.py`` covers the other half — that a real LanceDB table actually
reaches these states and answers correctly in each of them.
"""

from __future__ import annotations

import pytest

from manicule.core.ann import (
    MINIMUM_ANN_INDEX_THRESHOLD,
    PQ_CODE_BITS,
    PQ_CODEBOOK_ROWS,
    AnnIndex,
    AnnIndexState,
    AnnLifecycle,
    ann_index_name,
    classify,
    parse_ann_index_name,
    partitions_for,
    sub_vectors_for,
)


def index(
    *, indexed: int = 1_000, unindexed: int = 0, name: str = "manicule_ivfpq_g1_p31"
) -> AnnIndex:
    """An index as the store would report one, with its name already read."""
    read = parse_ann_index_name(name)
    return AnnIndex(
        name=name,
        index_type="IVF_PQ",
        distance_type="cosine",
        indexed_rows=indexed,
        unindexed_rows=unindexed,
        num_sub_vectors=4,
        build_generation=None if read is None else read[0],
        num_partitions=None if read is None else read[1],
    )


# --- the transition ------------------------------------------------------------------------


def test_a_corpus_below_the_threshold_is_exhaustive_rather_than_lacking_an_index() -> None:
    """The distinction the whole module exists for.

    Exhaustive search is exact. Reporting a small corpus as "no index" invites somebody to
    build one, and an IVF-PQ index over ten thousand vectors trades recall — permanently — for
    latency nobody was waiting on.
    """
    assert classify(threshold=100_000, rows=99_999, index=None) is AnnLifecycle.EXHAUSTIVE


def test_crossing_the_threshold_makes_a_build_due() -> None:
    """The transition #261 found nothing executing: one more row, and the state changes."""
    below = classify(threshold=100_000, rows=99_999, index=None)
    at = classify(threshold=100_000, rows=100_000, index=None)

    assert below is AnnLifecycle.EXHAUSTIVE
    assert at is AnnLifecycle.PENDING
    assert AnnIndexState(lifecycle=at, threshold=100_000, rows=100_000).due


def test_a_pending_index_is_still_exact() -> None:
    """``pending`` is slow, not wrong, and a status surface that implies otherwise misleads.

    Nothing about crossing the threshold degrades a result. Every query is still compared
    against every vector; what changed is that it now costs enough to be worth indexing.
    """
    state = AnnIndexState(lifecycle=AnnLifecycle.PENDING, threshold=256, rows=300)

    assert state.exact
    assert state.due


def test_a_threshold_of_zero_is_a_choice_rather_than_a_corpus_that_is_small() -> None:
    """``disabled`` and ``exhaustive`` are the same behavior reached two different ways.

    Collapsing them would tell an operator who switched indexing off that their corpus is
    below a threshold they had deliberately removed.
    """
    assert classify(threshold=0, rows=10_000_000, index=None) is AnnLifecycle.DISABLED
    assert not AnnIndexState(lifecycle=AnnLifecycle.DISABLED, threshold=0, rows=10**7).due


def test_switching_the_threshold_off_does_not_report_an_existing_index_as_gone() -> None:
    """Configuration stops new builds; it does not destroy what is already built.

    The index is still the search path, so it is still what status has to describe.
    """
    assert classify(threshold=0, rows=200_000, index=index(unindexed=150_000)) is AnnLifecycle.READY


# --- the refresh policy --------------------------------------------------------------------


def test_an_index_goes_stale_when_its_uncovered_tail_would_itself_deserve_one() -> None:
    """The stated refresh policy, and it is deliberately the same comparison as the first build.

    One dial rather than two that interact: whatever row count is worth indexing is the row
    count worth re-indexing for.
    """
    fresh = classify(threshold=1_000, rows=100_999, index=index(indexed=100_000, unindexed=999))
    tail = classify(threshold=1_000, rows=101_000, index=index(indexed=100_000, unindexed=1_000))

    assert fresh is AnnLifecycle.READY
    assert tail is AnnLifecycle.STALE


def test_uncovered_rows_are_reported_as_coverage_rather_than_as_absence() -> None:
    """They are searched — LanceDB scans them and merges — so this is latency, not loss."""
    partial = index(indexed=750, unindexed=250)

    assert partial.coverage == pytest.approx(0.75)
    assert partial.covered_rows == 1_000


def test_an_empty_table_is_fully_covered_rather_than_not_covered_at_all() -> None:
    """``0/0`` is a coverage gap nobody can close, and reporting one is a permanent false alarm."""
    assert index(indexed=0, unindexed=0).coverage == 1.0


# --- what a build can actually do ----------------------------------------------------------


def test_a_corpus_below_the_codebook_size_is_not_buildable() -> None:
    """An 8-bit product quantizer needs 256 vectors, and LanceDB refuses below that.

    Checked here so the storage layer's refusal is the same number as the configuration
    validator's, rather than two constants that agree until one moves.
    """
    assert not AnnIndexState(
        lifecycle=AnnLifecycle.PENDING, threshold=256, rows=PQ_CODEBOOK_ROWS - 1
    ).buildable
    assert AnnIndexState(
        lifecycle=AnnLifecycle.PENDING, threshold=256, rows=PQ_CODEBOOK_ROWS
    ).buildable


def test_the_configuration_floor_is_the_codebook_size() -> None:
    """Stated as one identity rather than two literals that happen to match today."""
    assert MINIMUM_ANN_INDEX_THRESHOLD == PQ_CODEBOOK_ROWS


# --- the parameters ------------------------------------------------------------------------


def test_partitions_follow_the_square_root_the_document_specifies() -> None:
    """``num_partitions ≈ sqrt(n)``, which is the whole of ``docs/storage.md`` §6.2's rule."""
    assert partitions_for(100_000) == 316
    assert partitions_for(1_000_000) == 1_000


def test_partitions_never_exceed_the_vectors_available_to_place_them() -> None:
    """IVF trains one centroid per partition; more partitions than rows cannot be trained."""
    assert partitions_for(1) == 1
    assert partitions_for(0) == 1


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [(1024, 64), (768, 48), (384, 24), (100, 5), (4, 1)],
)
def test_sub_vectors_divide_the_dimension_exactly(dimension: int, expected: int) -> None:
    """PQ splits the vector, so a count that does not divide it is not a configuration at all.

    The common widths land on 16 dimensions per sub-vector exactly. The awkward ones give up
    width rather than correctness: 100 becomes 5 sub-vectors of 20, never 6 of 16.667.
    """
    assert sub_vectors_for(dimension) == expected
    assert dimension % sub_vectors_for(dimension) == 0


# --- the self-describing name ----------------------------------------------------------------


def test_the_name_carries_the_two_facts_lancedb_does_not_record() -> None:
    """Partition count and build generation survive a restart because the name holds them.

    LanceDB reports the index type, its metric and its row coverage, and nothing about how many
    partitions it was trained with. Keeping that in a table beside the index would create a
    second record to go stale; keeping it in the name cannot.
    """
    assert parse_ann_index_name(ann_index_name(build_generation=3, num_partitions=316)) == (3, 316)


def test_an_index_this_project_did_not_build_is_reported_rather_than_adopted() -> None:
    """An operator's index is somebody's deliberate act, and its parameters are unknown.

    Guessing them from the row count would produce a number that is right until the partition
    rule changes and then confidently wrong.
    """
    foreign = index(name="somebodys_own_index")

    assert parse_ann_index_name("somebodys_own_index") is None
    assert not foreign.recognized
    assert foreign.num_partitions is None
    assert foreign.build_generation is None


def test_the_row_floor_is_derived_from_the_code_width_rather_than_written_down() -> None:
    """Two numbers that must agree, made into one.

    The floor is ``2 ** num_bits`` because a PQ codebook holds one centroid per code. Writing
    ``256`` beside a build that took the library's default width would be two facts that agree
    until the default moves — and the failures that produces are a threshold no build can ever
    satisfy, or a threshold that lets a doomed build through.
    """
    assert PQ_CODEBOOK_ROWS == 2**PQ_CODE_BITS
