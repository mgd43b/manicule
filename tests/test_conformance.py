"""The conformance suites, run against implementations that pass and ones that do not.

A suite that has only ever passed is not evidence. Each check here is shown to catch the
defect it was written for, which is what makes it worth the later tickets' time.
"""

from __future__ import annotations

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, ParsedBlock
from manicule.core.ids import chunk_id
from manicule.core.retrieval import Candidate, Query
from manicule.testing import (
    assert_chunker_contract,
    assert_connector_contract,
    assert_embedder_contract,
    assert_parser_contract,
    assert_retrieval_stage_contract,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_rejects_foreign_vectors,
)
from tests.fakes import (
    AliasingStage,
    BlockChunker,
    FixedDimensionVectorStore,
    ForgetfulConnector,
    ForgetfulVectorStore,
    HashEmbedder,
    LineParser,
    LyingParser,
    MemoryConnector,
    MemoryVectorStore,
    MutatingStage,
    SilentParser,
    TopKStage,
    TruncatingEmbedder,
    make_chunks,
    make_document,
    make_raw,
)

# --- parsers ---------------------------------------------------------------------------


async def test_a_parser_whose_anchors_resolve_passes() -> None:
    blocks = await assert_parser_contract(LineParser(), make_raw())
    assert len(blocks) == 3


async def test_the_round_trip_check_catches_an_off_by_one_anchor() -> None:
    """Nothing raises in the parser. Every citation is one line out. This is the whole point."""
    with pytest.raises(AssertionError, match="does not claim"):
        await assert_parser_contract(LyingParser(), make_raw())


async def test_a_parser_that_gives_up_honestly_still_passes() -> None:
    """Unlocated with a reason is a legitimate answer, and better than a guess."""
    await assert_parser_contract(SilentParser(), make_raw())


# --- chunkers --------------------------------------------------------------------------


def _blocks(text: str = "alpha beta gamma") -> list[ParsedBlock]:
    return [
        ParsedBlock(
            kind=BlockKind.PROSE, text=word, anchor=LineAnchor(start=index + 1, end=index + 1)
        )
        for index, word in enumerate(text.split())
    ]


def test_a_chunker_that_keeps_order_and_breadcrumbs_passes() -> None:
    chunks = assert_chunker_contract(
        BlockChunker(), make_document(), _blocks(), embedder=HashEmbedder()
    )
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


def test_a_chunk_budget_above_the_embedder_limit_is_caught() -> None:
    """Past the limit the input is truncated with no error, and the chunk over-claims."""
    chunker = BlockChunker()
    chunker.fingerprint = chunker.fingerprint.model_copy(update={"max_tokens": 4096})
    with pytest.raises(AssertionError, match="never saw"):
        assert_chunker_contract(chunker, make_document(), [], embedder=HashEmbedder())


def test_a_chunker_counting_with_the_wrong_tokenizer_is_caught() -> None:
    chunker = BlockChunker()
    chunker.fingerprint = chunker.fingerprint.model_copy(update={"tokenizer_id": "other"})
    with pytest.raises(AssertionError, match="not a budget"):
        assert_chunker_contract(chunker, make_document(), [], embedder=HashEmbedder())


# --- embedders -------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [3, 5, 17, 1024])
async def test_an_embedder_passes_at_any_dimension_it_declares(dimension: int) -> None:
    """The check reads the dimension from the embedder, so it can never encode one itself."""
    await assert_embedder_contract(HashEmbedder(dimension=dimension))


async def test_vectors_that_disagree_with_the_fingerprint_are_caught() -> None:
    with pytest.raises(AssertionError, match="the fingerprint says"):
        await assert_embedder_contract(TruncatingEmbedder(dimension=8))


# --- vector stores ---------------------------------------------------------------------


async def test_a_dimension_agnostic_store_passes() -> None:
    chunks = make_chunks(make_document())
    await assert_vector_store_is_dimension_agnostic(MemoryVectorStore, chunks)


async def test_a_store_with_a_baked_in_dimension_is_caught() -> None:
    """This is what makes "never hardcode the dimension" a build failure rather than advice."""
    chunks = make_chunks(make_document())
    with pytest.raises((AssertionError, ValueError)):
        await assert_vector_store_is_dimension_agnostic(FixedDimensionVectorStore, chunks)


async def test_a_store_that_guards_its_fingerprint_passes() -> None:
    await assert_vector_store_rejects_foreign_vectors(MemoryVectorStore)


async def test_a_store_that_only_checks_the_dimension_is_caught() -> None:
    """Two 1024-dimension models write cleanly into one index and ruin it quietly."""
    with pytest.raises(AssertionError, match="same dimension"):
        await assert_vector_store_rejects_foreign_vectors(ForgetfulVectorStore)


# --- retrieval stages ------------------------------------------------------------------


async def test_a_well_behaved_stage_passes() -> None:
    query = Query(text="anything")
    candidates = _candidates()
    kept = await assert_retrieval_stage_contract(TopKStage(k=2), query, candidates)
    assert len(kept) == 2
    assert all("top_k" in candidate.scores for candidate in kept)


async def test_a_stage_that_mutates_its_input_is_caught() -> None:
    """An order-dependent pipeline cannot be compared with another one."""
    with pytest.raises(AssertionError, match="mutated"):
        await assert_retrieval_stage_contract(MutatingStage(), Query(text="q"), _candidates())


async def test_a_stage_that_hands_back_the_same_list_is_caught() -> None:
    """Aliasing means a later stage's mutation rewrites an earlier stage's record."""
    with pytest.raises(AssertionError, match="the very list"):
        await assert_retrieval_stage_contract(AliasingStage(), Query(text="q"), _candidates())


def _candidates() -> list[Candidate]:
    document = make_document()
    return [
        Candidate(chunk=chunk, score=float(index), scores={"dense": float(index)})
        for index, chunk in enumerate(make_chunks(document))
    ]


# --- connectors ------------------------------------------------------------------------


async def test_a_connector_that_reconciles_passes() -> None:
    await assert_connector_contract(MemoryConnector())


async def test_a_connector_that_skips_reconciliation_is_caught() -> None:
    """Reconciliation drives deletion, so returning nothing would empty the index."""
    with pytest.raises(AssertionError, match="reconcile"):
        await assert_connector_contract(ForgetfulConnector())


def test_chunk_ids_are_derived_not_generated() -> None:
    """Re-ingesting an unchanged document must replace rows, not accumulate them."""
    document = make_document()
    assert chunk_id(document.id, 0, "x") == chunk_id(document.id, 0, "x")
    assert chunk_id(document.id, 0, "x") != chunk_id(document.id, 1, "x")
    assert chunk_id(document.id, 0, "x") != chunk_id(document.id, 0, "y")
