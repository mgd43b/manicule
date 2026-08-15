"""The four checks that run before a single document is discovered.

Each one guards a failure that raises nothing and makes every answer quietly wrong, which is
why they are refusals rather than warnings: there is nothing downstream that can notice.
"""

from __future__ import annotations

import pytest

from manicule.chunking import TokenCounter
from manicule.core.content import Chunk
from manicule.core.embedding import EmbedFingerprint, IndexFingerprints, Pooling
from manicule.core.errors import FingerprintMismatchError, PolicyError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.ingest.refusals import (
    check_before_run,
    require_coherent,
    require_measured,
)
from manicule.storage.vectors import table_name
from tests.ingest import fakes


def embed(model: str = "fake/embedder", *, max_sequence_length: int = 512) -> EmbedFingerprint:
    return EmbedFingerprint(
        model_id=model,
        dimension=8,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="whitespace",
        max_sequence_length=max_sequence_length,
    )


def chunk(*, max_tokens: int = 256, middleware: tuple[str, ...] = ()) -> ChunkFingerprint:
    return ChunkFingerprint(
        chunker="block",
        version="1",
        max_tokens=max_tokens,
        overlap_tokens=0,
        tokenizer_id="whitespace",
        embed_text_middleware=middleware,
    )


class PublishedMemoryVectors(fakes.MemoryVectors):
    @property
    def publication_pointer(self) -> str:
        return "reembed-synthetic-generation"


def provisional_chunk() -> ChunkFingerprint:
    """A chunker that counted with a stand-in vocabulary, as the chunker itself would build it."""
    return ChunkFingerprint(
        chunker="block",
        version="1",
        max_tokens=256,
        overlap_tokens=0,
        tokenizer_id=TokenCounter(
            "whitespace", lambda text: len(text.split()), provisional=True
        ).tokenizer_id,
    )


async def test_an_estimated_corpus_is_refused_before_anything_is_compared() -> None:
    """``docs/parsing.md`` §1.2, as a refusal rather than as a docstring.

    Nothing the other checks discover could make these chunks admissible: the count came from
    a vocabulary the model does not use, inflated by a factor chosen without measuring
    anything, so it is neither the model's number nor reproducible from it. This ran first for
    that reason — it reads nothing and it cannot be resolved by anything read later.
    """
    store = fakes.MemoryIngestStore()

    with pytest.raises(PolicyError, match="stand-in"):
        await check_before_run(embed=embed(), chunk=provisional_chunk(), store=store)

    assert store.state.is_empty, "a refused run must not commit the index to anything"


def test_the_estimated_refusal_reads_the_identity_it_refuses_on() -> None:
    """One mechanism, not two guards that can drift apart.

    The flag is derived from ``tokenizer_id``, so a fingerprint cannot be provisional in one
    place and measured in another.
    """
    require_measured(chunk())

    with pytest.raises(PolicyError, match=r"provisional:"):
        require_measured(provisional_chunk())


async def test_a_fresh_index_accepts_whatever_the_first_run_brings() -> None:
    """The one moment at which no comparison is possible, and none is needed."""
    store = fakes.MemoryIngestStore()

    committed = await check_before_run(embed=embed(), chunk=chunk(), store=store)

    assert committed.embed == embed()
    assert store.state.chunk == chunk()


async def test_a_different_model_at_the_same_dimension_is_refused() -> None:
    """Dimensional equality is not vector-space equality.

    Two unrelated models of the same size pass any size check, and every stored vector becomes
    noise relative to every new query — degrading toward random with no error raised anywhere.
    """
    store = fakes.MemoryIngestStore()
    await check_before_run(embed=embed("first/model"), chunk=chunk(), store=store)

    with pytest.raises(FingerprintMismatchError):
        await check_before_run(embed=embed("second/model"), chunk=chunk(), store=store)


async def test_the_refusal_prices_the_repair() -> None:
    """``--re-embed`` is a priced decision rather than an open-ended one."""
    store = fakes.MemoryIngestStore()
    document = _stored(store)
    await store.replace_chunks(document, _chunks())
    await check_before_run(embed=embed("first/model"), chunk=chunk(), store=store)

    with pytest.raises(FingerprintMismatchError) as caught:
        await check_before_run(embed=embed("second/model"), chunk=chunk(), store=store)

    assert any("re-embed" in note for note in caught.value.__notes__)
    assert any("2 stored chunk" in note for note in caught.value.__notes__)


async def test_a_vector_directory_from_another_instance_is_detected() -> None:
    """Two places cannot see this; three can.

    Swapping ``vectors/`` for another instance's, or restoring half a backup, leaves SQLite
    and the configuration agreeing with each other and disagreeing with what is on disk.
    """
    store = fakes.MemoryIngestStore()
    store.state = IndexFingerprints(embed=embed("agreed/model"), chunk=chunk())
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embed("someone/elses"))

    with pytest.raises(FingerprintMismatchError):
        await check_before_run(
            embed=embed("agreed/model"),
            chunk=chunk(),
            store=store,
            vectors=vectors,
        )


async def test_adding_a_text_mutating_middleware_is_as_loud_as_changing_the_budget() -> None:
    """Otherwise it changes every vector while both refusals pass.

    Neither fingerprint knows middleware exists, so without the fold two instances with
    identical configuration and different middleware produce different vectors from identical
    source bytes and nothing notices.
    """
    store = fakes.MemoryIngestStore()
    await check_before_run(embed=embed(), chunk=chunk(), store=store)

    with pytest.raises(FingerprintMismatchError):
        await check_before_run(
            embed=embed(), chunk=chunk(middleware=("redactor@1.0",)), store=store
        )


async def test_a_middleware_that_declares_nothing_does_not_invalidate_a_corpus() -> None:
    """A re-index nobody needed is a real cost, and the check must not manufacture one."""
    store = fakes.MemoryIngestStore()
    await check_before_run(embed=embed(), chunk=chunk(), store=store)

    await check_before_run(embed=embed(), chunk=chunk(), store=store)


def test_a_budget_longer_than_the_model_reads_is_refused_at_startup() -> None:
    """Each setting is valid alone, and the pair is not.

    Past the limit the input is dropped with no error raised, so the chunk is indexed as its
    opening tokens while still claiming all of its text. This is the single reason the check
    is a startup refusal rather than a per-document guard: the embedder runs in-process and
    gets none of the protection the parse stage gets, so the failure mode is removed rather
    than caught.
    """
    with pytest.raises(PolicyError, match="chunk budget"):
        require_coherent(embed=embed(max_sequence_length=256), chunk=chunk(max_tokens=512))


def test_a_budget_within_the_model_is_accepted() -> None:
    require_coherent(embed=embed(max_sequence_length=512), chunk=chunk(max_tokens=512))


def _stored(store: fakes.MemoryIngestStore) -> str:
    from tests.fakes import make_document  # noqa: PLC0415 - local to this helper

    document = make_document()
    store.documents[document.id] = document
    return document.id


def _chunks() -> list[Chunk]:
    from tests.fakes import make_chunks, make_document  # noqa: PLC0415

    return make_chunks(make_document(), count=2)


# --- the pointer `doctor` prints ---------------------------------------------------------------


async def test_a_first_ingest_records_which_table_its_vectors_are_in() -> None:
    """``docs/storage.md`` §6.5 promised this and nothing did it.

    The column shipped, the backup manifest carried it, the retrieval trace reported it, and
    ``doctor`` printed a healthy index as "N document(s) in no vector table" — because the
    only assignment carried the stored value forward, and the stored value starts ``NULL``.
    """
    store = fakes.MemoryIngestStore()

    committed = await check_before_run(
        embed=embed(), chunk=chunk(), store=store, vectors=fakes.MemoryVectors()
    )

    assert committed.vector_table == table_name(embed())
    assert store.state.vector_table == table_name(embed())


async def test_a_published_generation_pointer_is_not_replaced_by_its_inner_table_name() -> None:
    store = fakes.MemoryIngestStore()
    vectors = PublishedMemoryVectors()

    committed = await check_before_run(embed=embed(), chunk=chunk(), store=store, vectors=vectors)

    assert committed.vector_table == "reembed-synthetic-generation"
    assert store.state.vector_table == "reembed-synthetic-generation"


async def test_an_index_with_no_vector_store_records_no_table() -> None:
    """There is no table, and naming one would describe something that does not exist."""
    store = fakes.MemoryIngestStore()

    committed = await check_before_run(embed=embed(), chunk=chunk(), store=store)

    assert committed.vector_table is None
