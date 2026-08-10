"""The conformance suites, run against implementations that pass and ones that do not.

A suite that has only ever passed is not evidence. Each check here is shown to catch the
defect it was written for, which is what makes it worth the later tickets' time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, Chunk, DocumentStatus, ParsedBlock
from manicule.core.embedding import Vector, require_within_context
from manicule.core.errors import ContextOverflowError
from manicule.core.ids import chunk_id
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore
from manicule.testing import (
    assert_chunker_contract,
    assert_connector_contract,
    assert_embedder_contract,
    assert_local_only_policy_is_enforced,
    assert_middleware_contract,
    assert_parser_contract,
    assert_pipeline_enforces_scope,
    assert_refuses_oversized_chunks,
    assert_retrieval_stage_contract,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_rejects_foreign_vectors,
    closing,
)
from tests.fakes import (
    AliasingStage,
    BanningLocalOnly,
    BlockChunker,
    BlockRewritingMiddleware,
    EagerWatermarkConnector,
    FixedDimensionVectorStore,
    ForgetfulConnector,
    ForgetfulVectorStore,
    HashEmbedder,
    HydratingStage,
    LineParser,
    LyingParser,
    MemoryConnector,
    MemoryVectorStore,
    MutatingStage,
    PassThroughMiddleware,
    RawVectorStage,
    RedactingMiddleware,
    SilentParser,
    TextRewritingMiddleware,
    TopKStage,
    TruncatingEmbedder,
    UndeclaredEmbedMiddleware,
    UnenforcedLocalOnly,
    lan_ollama,
    local_only,
    loopback_ollama,
    make_chunks,
    make_document,
    make_raw,
)
from tests.storage_helpers import make_chunk as make_stored_chunk
from tests.storage_helpers import make_document as make_stored_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

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


# --- the re-embed path -----------------------------------------------------------------


async def test_a_re_embed_path_that_checks_its_batch_passes() -> None:
    embedder = HashEmbedder(dimension=4)

    async def embed_batch(chunks: Sequence[Chunk]) -> list[Vector]:
        require_within_context(chunks, embedder.fingerprint)
        return await embedder.embed([chunk.embed_text for chunk in chunks])

    await assert_refuses_oversized_chunks(embed_batch, embedder)


async def test_a_re_embed_path_that_trusts_the_fingerprint_is_caught() -> None:
    """The hole the chunker's own guard cannot cover.

    Re-embedding does not re-chunk, so a model reconfigured to a shorter sequence length
    truncates every oversized chunk with an unchanged fingerprint and no error anywhere.
    """
    embedder = HashEmbedder(dimension=4)

    async def embed_batch(chunks: Sequence[Chunk]) -> list[Vector]:
        return await embedder.embed([chunk.embed_text for chunk in chunks])

    with pytest.raises(AssertionError, match="require_within_context"):
        await assert_refuses_oversized_chunks(embed_batch, embedder)


def test_the_guard_names_the_chunks_that_do_not_fit() -> None:
    document = make_document()
    chunks = [
        chunk.model_copy(update={"token_count": count})
        for chunk, count in zip(make_chunks(document, count=3), (10, 5000, 9000), strict=True)
    ]
    fingerprint = HashEmbedder().fingerprint

    with pytest.raises(ContextOverflowError) as caught:
        require_within_context(chunks, fingerprint)

    message = str(caught.value)
    assert "2 chunk(s)" in message
    assert "9000 tokens" in message
    assert str(fingerprint.max_sequence_length) in message


def test_the_guard_passes_a_batch_that_fits() -> None:
    document = make_document()
    require_within_context(make_chunks(document), HashEmbedder().fingerprint)


def test_token_counts_from_the_wrong_tokenizer_are_refused() -> None:
    """A count taken under a different vocabulary is not a measurement of anything relevant."""
    document = make_document()
    chunker = BlockChunker()
    foreign = chunker.fingerprint.model_copy(update={"tokenizer_id": "some/other-tokenizer"})

    with pytest.raises(ContextOverflowError, match="different vocabulary"):
        require_within_context(make_chunks(document), HashEmbedder().fingerprint, foreign)


def test_a_raised_sequence_limit_does_not_invalidate_anything() -> None:
    """Why the limit stays out of identity: growing it harms nothing already stored."""
    document = make_document()
    generous = HashEmbedder().fingerprint.model_copy(update={"max_sequence_length": 100_000})
    require_within_context(make_chunks(document), generous)
    assert generous.matches(HashEmbedder().fingerprint)


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
    query = _query()
    candidates = _candidates()
    kept = await assert_retrieval_stage_contract(TopKStage(k=2), query, candidates)
    assert len(kept) == 2
    assert all("top_k" in candidate.scores for candidate in kept)


async def test_a_stage_that_mutates_its_input_is_caught() -> None:
    """An order-dependent pipeline cannot be compared with another one."""
    with pytest.raises(AssertionError, match="mutated"):
        await assert_retrieval_stage_contract(MutatingStage(), _query(), _candidates())


async def test_a_stage_that_hands_back_the_same_list_is_caught() -> None:
    """Aliasing means a later stage's mutation rewrites an earlier stage's record."""
    with pytest.raises(AssertionError, match="the very list"):
        await assert_retrieval_stage_contract(AliasingStage(), _query(), _candidates())


def _query(text: str = "anything") -> Query:
    return Query(text=text, filter=Filter(workspace_ids=frozenset({DEFAULT_WORKSPACE})))


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


async def test_a_connector_whose_watermark_advances_as_it_yields_is_caught() -> None:
    """An uninterrupted run of this connector is indistinguishable from a correct one.

    Which is the whole problem: the defect only shows when a run is interrupted, and then it
    shows as documents that are in the source, were enumerated once, and are in no index —
    permanently, because the next sync starts past them.
    """
    with pytest.raises(AssertionError, match="abandoned"):
        await assert_connector_contract(EagerWatermarkConnector())


async def test_a_watermark_is_offered_only_after_a_complete_enumeration() -> None:
    """The positive half: draining the walk does produce one."""
    connector = MemoryConnector()
    assert connector.watermark is None

    async with closing(connector.discover(None)) as stream:
        async for _ in stream:
            pass

    assert connector.watermark is not None


def test_chunk_ids_are_derived_not_generated() -> None:
    """Re-ingesting an unchanged document must replace rows, not accumulate them."""
    document = make_document()
    assert chunk_id(document.id, 0, "x") == chunk_id(document.id, 0, "x")
    assert chunk_id(document.id, 0, "x") != chunk_id(document.id, 1, "x")
    assert chunk_id(document.id, 0, "x") != chunk_id(document.id, 0, "y")


# --- middleware ------------------------------------------------------------------------


async def test_middleware_contract_accepts_a_middleware_that_touches_nothing() -> None:
    document = make_document()
    chunks = make_chunks(document)

    returned = await assert_middleware_contract(PassThroughMiddleware(), document, chunks)

    assert [chunk.text for chunk in returned] == [chunk.text for chunk in chunks]


async def test_middleware_contract_accepts_declared_embed_text_rewriting() -> None:
    """Rewriting embed_text is the reason after_chunk exists. It must not be rejected."""
    document = make_document()
    chunks = make_chunks(document)

    returned = await assert_middleware_contract(RedactingMiddleware(), document, chunks)

    assert any("[REDACTED]" in chunk.embed_text for chunk in returned)
    assert all("[REDACTED]" not in chunk.text for chunk in returned)


async def test_middleware_contract_catches_a_rewritten_text() -> None:
    """The defect no parser suite can catch, and the reason this check exists."""
    document = make_document()
    chunks = make_chunks(document)

    with pytest.raises(AssertionError, match=r"rewrote Chunk\.text"):
        await assert_middleware_contract(TextRewritingMiddleware(), document, chunks)


async def test_middleware_contract_catches_undeclared_embed_text_mutation() -> None:
    """Not a corrupted citation, but a corpus no fingerprint describes."""
    document = make_document()
    chunks = make_chunks(document)

    with pytest.raises(AssertionError, match="without declaring mutates_embedded_text"):
        await assert_middleware_contract(UndeclaredEmbedMiddleware(), document, chunks)


async def test_middleware_contract_needs_a_chunk_to_have_an_opinion() -> None:
    document = make_document()

    with pytest.raises(AssertionError, match="at least one chunk"):
        await assert_middleware_contract(PassThroughMiddleware(), document, [])


async def test_middleware_contract_catches_a_rewritten_block() -> None:
    """after_parse can corrupt a citation exactly as after_chunk can, one hook earlier."""
    document = make_document()
    chunks = make_chunks(document)

    with pytest.raises(AssertionError, match=r"rewrote ParsedBlock\.text"):
        await assert_middleware_contract(
            BlockRewritingMiddleware(), document, chunks, blocks=_blocks()
        )


async def test_middleware_contract_accepts_untouched_blocks() -> None:
    document = make_document()
    chunks = make_chunks(document)

    await assert_middleware_contract(PassThroughMiddleware(), document, chunks, blocks=_blocks())


# --- the workspace boundary --------------------------------------------------------------


async def _scoped_fixture(store: SqliteDocStore, engine: AsyncEngine) -> tuple[list[Chunk], Chunk]:
    """A corpus holding one live chunk and three a search must never return.

    Soft-deleted, ``pending``, and another workspace's — the three things the Lance table
    cannot distinguish, because none of them is a column it has.
    """
    visible: list[Chunk] = []
    for source_id, status in (
        ("live", DocumentStatus.INDEXED),
        ("removed", DocumentStatus.INDEXED),
        ("waiting", DocumentStatus.PENDING),
    ):
        document = make_stored_document(source_id=source_id, status=status)
        await store.upsert_document(document)
        chunk = make_stored_chunk(document, 0, f"authentication {source_id}")
        await store.replace_chunks(document.id, [chunk])
        if source_id == "removed":
            await store.soft_delete_document(document.id)
        visible.append(chunk)

    other = SqliteDocStore(engine, workspace_id="beta")
    await other.ensure_workspace()
    foreign_document = make_stored_document(source_id="theirs", workspace_id="beta")
    await other.upsert_document(foreign_document)
    foreign = make_stored_chunk(foreign_document, 0, "authentication theirs")
    await other.replace_chunks(foreign_document.id, [foreign])

    return [*visible, foreign], visible[0]


@pytest.mark.contract
async def test_a_pipeline_that_hydrates_through_the_document_store_passes(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The check must admit the stage the design actually calls for."""
    everything, live = await _scoped_fixture(store, engine)

    kept = await assert_pipeline_enforces_scope(
        [HydratingStage(everything, store)], store, _query()
    )

    assert [candidate.chunk.id for candidate in kept] == [live.id]


@pytest.mark.contract
async def test_a_dense_stage_that_skipped_the_hydrating_join_is_caught(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The defect the exemption in ``predicate_for`` would otherwise make invisible.

    Every returned row is well-formed, ranked and plausible; three of the four belong to
    documents this query must not see, and nothing downstream can tell.
    """
    everything, _ = await _scoped_fixture(store, engine)

    with pytest.raises(AssertionError, match="another workspace or has been soft-deleted"):
        await assert_pipeline_enforces_scope([RawVectorStage(everything)], store, _query())


@pytest.mark.contract
async def test_a_pipeline_returning_a_pending_document_is_caught(
    store: SqliteDocStore,
) -> None:
    """A document mid-ingest has chunks whose vectors and text need not agree yet."""
    document = make_stored_document(source_id="waiting", status=DocumentStatus.PENDING)
    await store.upsert_document(document)
    chunk = make_stored_chunk(document, 0, "authentication waiting")
    await store.replace_chunks(document.id, [chunk])

    with pytest.raises(AssertionError, match="rather than 'indexed'"):
        await assert_pipeline_enforces_scope([RawVectorStage([chunk])], store, _query())


@pytest.mark.contract
async def test_a_pipeline_that_returns_nothing_is_not_evidence(store: SqliteDocStore) -> None:
    """A check that has never seen a candidate has not checked anything.

    Off by default is wrong here and right for the runtime assertion, where an empty result
    is an ordinary outcome rather than a fixture that proves nothing.
    """
    with pytest.raises(AssertionError, match="without seeing one"):
        await assert_pipeline_enforces_scope([RawVectorStage([])], store, _query())

    await assert_pipeline_enforces_scope(
        [RawVectorStage([])], store, _query(), expect_results=False
    )


# --- the local-only data policy ------------------------------------------------------------


@pytest.mark.contract
def test_a_loopback_ollama_is_admitted_under_a_local_only_policy() -> None:
    """A check that rejects the legitimate case is a ban, not a policy."""
    assert_local_only_policy_is_enforced(loopback_ollama())


@pytest.mark.contract
def test_a_lan_ollama_is_refused_under_a_local_only_policy() -> None:
    """The provider is spelled the same; the endpoint is on another machine."""
    settings = lan_ollama()

    assert settings.cloud_providers_in_use == frozenset({"ollama"})
    assert_local_only_policy_is_enforced(settings)


@pytest.mark.contract
def test_a_problem_that_is_not_about_egress_does_not_read_as_a_refusal() -> None:
    """A configuration has problems for several reasons at once, and some name an endpoint.

    A stray ``base_url`` on an in-process provider is reported whatever the data policy says.
    Blaming the endpoint for it would report a local endpoint as refused when nothing refused
    it — a false alarm on the branch that exists to catch over-strictness.
    """
    settings = local_only(
        {"provider": "ollama", "base_url": "http://127.0.0.1:11434"},
        embedding={"provider": "mlx"},
        providers={"mlx": {"base_url": "http://gpu-box.lan:11434"}},
    )

    assert any("dials nothing" in problem for problem in settings.policy_problems())
    assert_local_only_policy_is_enforced(settings)


@pytest.mark.contract
def test_a_policy_that_admits_the_lan_case_is_caught() -> None:
    """The defect itself: the endpoint is off the machine and the gate says nothing.

    Without this, the check would be one that has only ever passed against code that happens
    to be right, which is not evidence about the next edit.
    """
    with pytest.raises(AssertionError, match="admitted anyway"):
        assert_local_only_policy_is_enforced(lan_ollama(UnenforcedLocalOnly))


@pytest.mark.contract
def test_a_policy_that_refuses_the_loopback_case_is_caught_too() -> None:
    """Both directions, because a check that only refuses is a ban with a policy's name."""
    with pytest.raises(AssertionError, match="not a policy, it is a ban"):
        assert_local_only_policy_is_enforced(loopback_ollama(BanningLocalOnly))
