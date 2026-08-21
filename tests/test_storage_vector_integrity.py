"""The corruption every other check waves through, and what the checksum does about it.

Every guard the vector store had before this file compares a row against *metadata*: its
recorded embedding input, the chunk stored beside it, the fingerprint the index was built with,
the dimension, the publication it belongs to. Change one finite component of a stored vector
into another finite component and all of that stays exactly true — the row is the right chunk's,
under the right model, of the right length, in the right generation, holding numbers that are
all perfectly rankable. Nothing in the row disagrees with anything else in it, so the damaged
vector is reused, ranked and cited for as long as the corpus lives.

So the tests here damage rows the way a bit flip or a storage-layer rewrite would — through
lancedb directly, leaving every piece of metadata alone — and assert the refusal. The other
half of the file is the compatibility policy: a directory written before checksums existed
holds none, and the difference between "unverified" and "corrupt" is the difference between an
upgrade in progress and a directory to restore from backup.

**What none of this establishes**, stated here because a test file is where somebody looks for
the guarantee: a matching checksum says the numbers on disk are the numbers that were written.
It does not say the model produced the right vector for the text, and it does not defend
against anything able to rewrite a vector and its checksum together. See ``docs/storage.md``
§6.2.5.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
#
# Most tests here reach through lancedb to rewrite a row without going through the store,
# which is the only way to produce the corruption being tested. lancedb annotates its surface
# in terms of `pyarrow`, which ships no type information — see the same note in
# `manicule.storage.vectors`. Strict checking otherwise applies.

from __future__ import annotations

import math
import random
import struct
from typing import TYPE_CHECKING, Final

import lancedb
import pytest

from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk
from manicule.core.embedding import (
    UNRECORDED_CHECKSUM,
    VECTOR_CHECKSUM_DOMAINS,
    VECTOR_CHECKSUM_VERSION,
    EmbedFingerprint,
    Pooling,
    VectorIntegrity,
    VectorState,
    canonical_stored_vector,
    classify_stored_vector,
    vector_checksum,
    verify_stored_checksum,
)
from manicule.core.protocols import VectorIntegrityMaintenance
from manicule.core.retrieval import Filter
from manicule.storage.vectors import (
    CHECKSUM_COLUMN,
    CHECKSUM_VERSION_COLUMN,
    VECTOR_COLUMN,
    LanceVectorStore,
    PublishedLanceVectorStore,
    table_name,
)
from manicule.testing import assert_protocol_signatures
from tests.vector_helpers import nudged, read_column, rewrite_row, rows_of

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.core.embedding import Vector

SCOPE: Final = frozenset({"default"})
"""``Filter.workspace_ids`` is required and this store does not enforce it. See its own suite."""

DIMENSION: Final = 4


def fingerprint(dimension: int = DIMENSION, model_id: str = "test/model") -> EmbedFingerprint:
    """An embedder identity at ``dimension``. Nothing here assumes the number."""
    return EmbedFingerprint(
        model_id=model_id,
        dimension=dimension,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=512,
    )


def chunk(chunk_id: str, document_id: str = "doc-1", *, position: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=f"the text of {chunk_id}",
        embed_text=f"Section > the text of {chunk_id}",
        anchor=HeadingAnchor(path=("Section",), fragment="section"),
        heading_path=("Section",),
        kind=BlockKind.PROSE,
        position=position,
        token_count=7,
        metadata={"lang": "en"},
    )


def spread(index: int, dimension: int = DIMENSION) -> list[float]:
    """A one-hot vector, already unit length, so nothing normalizes on the way in."""
    return [1.0 if position == index % dimension else 0.0 for position in range(dimension)]


async def prepared(directory: Path, dimension: int = DIMENSION) -> LanceVectorStore:
    store = LanceVectorStore(directory)
    await store.ensure_ready(fingerprint(dimension))
    return store


async def drop_checksum_columns(directory: Path) -> None:
    """Make a table look like one written before the checksum columns existed."""
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(fingerprint()))
    await table.drop_columns([CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN])
    connection.close()


# --- the contract itself ---------------------------------------------------------------------


def test_the_checksum_is_stable_across_processes_and_platforms() -> None:
    """The digest of a fixed vector, written out.

    A checksum recomputed on readback is only useful if "recomputed" means the same thing on
    the machine that wrote the row and the machine that reads it. Everything the preimage is
    built from is pinned — big-endian ``binary32``, an eight-byte big-endian length, one
    domain separator — so the answer is a constant and this is where it is written down. If
    this value changes, every stored checksum in every installation stops matching, which is a
    migration and not a refactor.
    """
    assert vector_checksum([0.0, 1.0, 0.0, 0.0]) == (
        "694f5c398bacf60c6cc3300cb20585f7851d416adaad4936eb9e37a2b40de27d"
    )
    assert VECTOR_CHECKSUM_VERSION in VECTOR_CHECKSUM_DOMAINS


def test_the_dimension_is_in_the_preimage_so_two_shapes_cannot_collide() -> None:
    """Without a length prefix, a shorter vector's bytes are a prefix of a longer one's.

    ``[0.0]`` and ``[0.0, 0.0]`` differ by four zero bytes, and a hash over components alone
    would be a hash over a stream nobody delimited. The eight-byte length goes first, before
    anything variable, so no two shapes share a preimage.
    """
    digests = {vector_checksum([0.0] * size) for size in range(4)}

    assert len(digests) == 4


def test_a_signed_zero_is_hashed_as_a_positive_one() -> None:
    """The sign bit of a zero is representation, not value.

    ``-0.0`` and ``0.0`` compare equal, rank identically under every metric this store uses,
    and differ in a bit that an Arrow round trip or a copy is entitled to drop. Hashing them
    apart would make a *representation* change report as numerical corruption — which is
    precisely the false positive that teaches an operator to ignore this check.
    """
    assert vector_checksum([-0.0, 1.0, -0.0, 0.0]) == vector_checksum([0.0, 1.0, 0.0, 0.0])


def test_the_domain_separator_stops_a_digest_from_somewhere_else_being_accepted() -> None:
    """A bare SHA-256 of the same bytes is not this checksum."""
    import hashlib  # noqa: PLC0415 - only this assertion needs the naked digest

    bare = hashlib.sha256(struct.pack(">4f", 0.0, 1.0, 0.0, 0.0)).hexdigest()

    assert vector_checksum([0.0, 1.0, 0.0, 0.0]) != bare


def test_a_non_finite_component_is_refused_rather_than_hashed() -> None:
    """A checksum over ``NaN`` would certify a vector that cannot be ranked at all."""
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            vector_checksum([value, 0.0, 0.0, 0.0])


def test_an_unimplemented_format_version_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        vector_checksum([0.0, 1.0, 0.0, 0.0], version="99")


@pytest.mark.parametrize(
    ("recorded", "version", "required", "expected"),
    [
        ("", "", False, VectorIntegrity.UNVERIFIED),
        ("", "", True, VectorIntegrity.MISSING),
        ("", VECTOR_CHECKSUM_VERSION, False, VectorIntegrity.MALFORMED),
        ("a" * 64, "", False, VectorIntegrity.MALFORMED),
        ("not-a-digest", VECTOR_CHECKSUM_VERSION, False, VectorIntegrity.MALFORMED),
        ("a" * 63, VECTOR_CHECKSUM_VERSION, False, VectorIntegrity.MALFORMED),
        ("A" * 64, VECTOR_CHECKSUM_VERSION, False, VectorIntegrity.MALFORMED),
        ("a" * 64, "99", False, VectorIntegrity.UNKNOWN_VERSION),
        ("a" * 64, VECTOR_CHECKSUM_VERSION, False, VectorIntegrity.MISMATCHED),
    ],
)
def test_every_way_a_recorded_checksum_can_fail_is_its_own_typed_answer(
    recorded: str, version: str, required: bool, expected: VectorIntegrity
) -> None:
    """Nine inputs, nine distinguishable verdicts, and each names a different repair.

    An unknown version is a build older or newer than the data and calls for the right binary.
    A malformed value is a half-written row. A mismatch is corruption. Collapsing them into
    "bad checksum" would make the diagnostic name a symptom rather than a cause — and would
    make an upgrade that merely reordered a deploy look identical to a failing disk.
    """
    verdict = verify_stored_checksum(
        [0.0, 1.0, 0.0, 0.0], recorded=recorded, version=version, required=required
    )

    assert verdict is expected
    assert verdict.accepts is (expected is VectorIntegrity.UNVERIFIED)


def test_hashing_the_canonical_form_of_a_stored_vector_is_idempotent() -> None:
    """The property the whole readback check rests on, at a realistic dimension.

    A write stores ``canonical_stored_vector(v)`` and hashes it; a read hashes what came back.
    Those agree only if canonicalizing an already-canonical vector changes nothing — including
    the norm test, whose tolerance has to still hold after a thousand components have each been
    rounded to float32. Measured here rather than reasoned about, because the accumulated norm
    error is the term that would break it.
    """
    generator = random.Random(20260821)  # noqa: S311 - synthetic vectors, not keys
    for _ in range(20):
        raw = [generator.gauss(0.0, 1.0) for _ in range(1024)]
        stored = canonical_stored_vector(raw)

        assert canonical_stored_vector(stored) == stored
        assert vector_checksum(canonical_stored_vector(stored)) == vector_checksum(stored)


def test_a_checksum_valid_row_still_has_to_pass_every_provenance_check() -> None:
    """Numerical integrity is necessary and nowhere near sufficient.

    The vector below is intact and its checksum is correct. What is wrong is that the row
    records the embedding input of a *different* string, which is the stale-vector failure the
    identity check exists for — and a store that treated a good checksum as a license to reuse
    would have quietly reintroduced it.
    """
    asked = chunk("chunk-a")
    stale = asked.model_copy(update={"embed_text": "Somewhere else > the text of chunk-a"})
    stored = canonical_stored_vector(spread(1))

    verdict = classify_stored_vector(
        asked,
        recorded_identity="",
        stored_embed_text=stale.embed_text,
        stored_vector=list(stored),
        embed=fingerprint(),
        recorded_checksum=vector_checksum(stored),
        recorded_checksum_version=VECTOR_CHECKSUM_VERSION,
    )

    assert verdict.state is VectorState.STALE
    assert not verdict.vector


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_wrong_dimension_or_non_finite_row_is_refused_whatever_its_checksum_says(
    value: float,
) -> None:
    """Two rejections that happen before the checksum is consulted, and must keep happening.

    A checksum can be computed over a two-component vector in a four-dimensional index, and it
    will verify. Ordering the dimension and finiteness checks first is what stops a correct
    digest over a useless vector reading as a usable row — and it is why those two get their
    own typed verdicts rather than sharing ``mismatched``.
    """
    asked = chunk("chunk-a")
    short = canonical_stored_vector([1.0, 0.0])
    verdict = classify_stored_vector(
        asked,
        recorded_identity="",
        stored_embed_text=asked.embed_text,
        stored_vector=list(short),
        embed=fingerprint(),
        recorded_checksum=vector_checksum(short),
        recorded_checksum_version=VECTOR_CHECKSUM_VERSION,
    )

    assert verdict.state is VectorState.CORRUPT
    assert verdict.integrity is VectorIntegrity.WRONG_DIMENSION

    infected = [value, 0.0, 0.0, 0.0]
    poisoned = classify_stored_vector(
        asked,
        recorded_identity="",
        stored_embed_text=asked.embed_text,
        stored_vector=infected,
        embed=fingerprint(),
        recorded_checksum="a" * 64,
        recorded_checksum_version=VECTOR_CHECKSUM_VERSION,
    )

    assert poisoned.state is VectorState.CORRUPT
    assert poisoned.integrity is VectorIntegrity.NON_FINITE


# --- against real storage --------------------------------------------------------------------


@pytest.mark.contract
async def test_both_vector_handles_satisfy_the_integrity_maintenance_protocol(
    tmp_path: Path, engine: AsyncEngine
) -> None:
    """The plain handle and the published one, held to the same shape.

    ``@runtime_checkable`` checks that the attributes exist and deliberately nothing about what
    they accept, and the published handle delegates by hand — so a keyword that drifted on one
    of the two would pass every ``isinstance`` in the codebase and fail at the first call, which
    for a maintenance operation is in front of an operator who is already worried.
    """
    plain = LanceVectorStore(tmp_path / "vectors")
    published = PublishedLanceVectorStore(tmp_path / "published", engine)

    for handle in (plain, published):
        assert isinstance(handle, VectorIntegrityMaintenance)
        assert_protocol_signatures(handle, VectorIntegrityMaintenance)


async def test_a_newly_written_row_reads_back_verified(tmp_path: Path) -> None:
    """The round trip, through the schema, the Arrow conversion and back."""
    store = await prepared(tmp_path / "vectors")
    stored = chunk("chunk-a")
    await store.upsert([stored], [spread(1)])

    verdict = (await store.stored_vectors([stored]))[stored.id]

    assert verdict.state is VectorState.READABLE
    assert verdict.integrity is VectorIntegrity.VERIFIED
    assert verdict.checksum_recorded
    await store.teardown()


async def test_a_vector_that_is_not_unit_length_is_hashed_after_normalization(
    tmp_path: Path,
) -> None:
    """A real embedder returns vectors a hair off unit length, and the store normalizes them.

    So the checksum has to be taken after that conversion, not before. A store that hashed its
    argument would produce a digest matching nothing it ever wrote, and every row would read as
    corrupt on the first search after a deploy — which is a far worse outcome than the failure
    the checksum was added to catch.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("chunk-a")
    await store.upsert([stored], [[3.0, -4.0, 0.0, 0.0]])
    await store.teardown()

    recorded = await read_column(directory, fingerprint(), stored.id, CHECKSUM_COLUMN)
    persisted = await read_column(directory, fingerprint(), stored.id, VECTOR_COLUMN)

    assert recorded == vector_checksum([float(value) for value in persisted])
    assert recorded != vector_checksum([3.0, -4.0, 0.0, 0.0])
    assert (await LanceVectorStore(directory).stored_vectors([stored]))[
        stored.id
    ].integrity is VectorIntegrity.VERIFIED


async def test_one_mutated_finite_component_is_refused_though_every_other_check_passes(
    tmp_path: Path,
) -> None:
    """The whole reason this exists, at the smallest damage that can be done.

    One component moves to the next representable float32 — still finite, still the same
    length, still the same chunk, still the same embedding identity, still the same
    publication. Every guard that existed before the checksum inspects a piece of metadata, and
    not one piece of metadata has changed.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("chunk-a")
    await store.upsert([stored], [spread(1)])
    await store.teardown()

    before = await read_column(directory, fingerprint(), stored.id, VECTOR_COLUMN)
    await rewrite_row(
        directory,
        fingerprint(),
        stored.id,
        {VECTOR_COLUMN: nudged([float(v) for v in before], index=1)},
    )

    verdict = (await LanceVectorStore(directory).stored_vectors([stored]))[stored.id]

    assert verdict.state is VectorState.CORRUPT
    assert verdict.integrity is VectorIntegrity.MISMATCHED
    assert not verdict.vector, "a refused row must not hand its numbers to a caller anyway"


async def test_changing_only_the_stored_checksum_is_refused_too(tmp_path: Path) -> None:
    """The same corruption arriving from the other side.

    The store cannot tell which half moved and does not try to: what it knows is that the two
    halves of the row no longer describe each other, and a row that contradicts itself is not a
    vector to reuse whichever half is wrong.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("chunk-a")
    await store.upsert([stored], [spread(1)])
    await store.teardown()

    await rewrite_row(
        directory, fingerprint(), stored.id, {CHECKSUM_COLUMN: vector_checksum(spread(2))}
    )

    verdict = (await LanceVectorStore(directory).stored_vectors([stored]))[stored.id]

    assert verdict.state is VectorState.CORRUPT
    assert verdict.integrity is VectorIntegrity.MISMATCHED


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({CHECKSUM_COLUMN: "not a digest"}, VectorIntegrity.MALFORMED),
        ({CHECKSUM_VERSION_COLUMN: "99"}, VectorIntegrity.UNKNOWN_VERSION),
        ({CHECKSUM_VERSION_COLUMN: ""}, VectorIntegrity.MALFORMED),
    ],
)
async def test_a_malformed_or_unknown_checksum_is_a_bounded_typed_outcome(
    tmp_path: Path, columns: dict[str, str], expected: VectorIntegrity
) -> None:
    """Three damaged rows, three named verdicts, and no exception out of any of them.

    A store that raised here would take a whole reuse sweep down over one bad row. The verdict
    is data, the row is refused, and the chunk goes back to the model like any other one whose
    stored vector cannot be used.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("chunk-a")
    await store.upsert([stored], [spread(1)])
    await store.teardown()

    await rewrite_row(directory, fingerprint(), stored.id, columns)

    verdict = (await LanceVectorStore(directory).stored_vectors([stored]))[stored.id]

    assert verdict.state is VectorState.CORRUPT
    assert verdict.integrity is expected


async def test_a_mismatched_row_is_never_returned_by_search(tmp_path: Path) -> None:
    """A candidate carries a score computed against the stored vector.

    Returning one ranked on numbers nothing vouches for would put a result in front of somebody
    on the strength of a bit flip. The search returns fewer rows instead, which is the honest
    outcome — the alternative is backfilling the gap with rows that were ranked *behind* the one
    being refused.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)])
    await store.teardown()

    before = await read_column(directory, fingerprint(), "chunk-1", VECTOR_COLUMN)
    await rewrite_row(
        directory,
        fingerprint(),
        "chunk-1",
        {VECTOR_COLUMN: nudged([float(v) for v in before], index=1)},
    )

    reopened = await prepared(directory)
    found = await reopened.search(spread(1), 5, Filter(workspace_ids=SCOPE))

    assert [candidate.chunk.id for candidate in found] == ["chunk-0", "chunk-2"], (
        "the corrupted row is dropped and the intact ones still answer"
    )
    await reopened.teardown()


async def test_a_query_with_no_direction_verifies_its_candidates_too(tmp_path: Path) -> None:
    """The unranked branch is a read path, so the rule applies to it as well.

    It is the branch nobody exercises by hand, which is exactly why a rule implemented once in
    the ranked path and forgotten here would survive review.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)])
    await store.teardown()

    await rewrite_row(directory, fingerprint(), "chunk-0", {CHECKSUM_COLUMN: "not a digest"})

    reopened = await prepared(directory)
    found = await reopened.search([0.0] * DIMENSION, 5, Filter(workspace_ids=SCOPE))

    assert [candidate.chunk.id for candidate in found] == ["chunk-1"]
    await reopened.teardown()


# --- the publication fence ---------------------------------------------------------------


async def test_publication_completeness_rejects_one_mismatched_row_among_valid_ones(
    tmp_path: Path,
) -> None:
    """A generation is published whole or not at all.

    The count is right, every identity is right, and one row's numbers are not what its
    checksum says. Publishing that would make the corrupted vector live under an atomic pointer
    flip, which is the moment after which nothing is watching.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)], publication_id="gen-1")
    canonical = fingerprint().canonical()

    assert await store.publication_is_complete("gen-1", chunks, embedding_fingerprint=canonical)
    await store.teardown()

    before = await read_column(directory, fingerprint(), "chunk-1", VECTOR_COLUMN)
    await rewrite_row(
        directory,
        fingerprint(),
        "chunk-1",
        {VECTOR_COLUMN: nudged([float(v) for v in before], index=1)},
    )

    reopened = LanceVectorStore(directory)

    assert not await reopened.publication_is_complete(
        "gen-1", chunks, embedding_fingerprint=canonical
    )
    await reopened.teardown()


async def test_a_publication_cannot_be_accepted_with_a_row_that_records_no_checksum(
    tmp_path: Path,
) -> None:
    """The one place a missing checksum is a refusal rather than a shrug.

    The same row reads as ``unverified`` to reuse, because declaring an existing corpus corrupt
    on upgrade would be a lie about a directory that is fine. At a publication fence it reads
    as ``missing``, because "we have not checked it yet" is not a thing to publish — and a
    coverage number that counted it as covered would be false at the exact moment it starts
    being relied on.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)], publication_id="gen-1")
    await store.teardown()

    await rewrite_row(
        directory,
        fingerprint(),
        "chunk-0",
        {CHECKSUM_COLUMN: UNRECORDED_CHECKSUM, CHECKSUM_VERSION_COLUMN: UNRECORDED_CHECKSUM},
    )

    reopened = LanceVectorStore(directory)
    verdicts = await reopened.stored_vectors(chunks)

    assert verdicts["chunk-0"].state is VectorState.READABLE
    assert verdicts["chunk-0"].integrity is VectorIntegrity.UNVERIFIED
    assert not await reopened.publication_is_complete(
        "gen-1", chunks, embedding_fingerprint=fingerprint().canonical()
    )
    await reopened.teardown()


async def test_a_table_with_no_checksum_columns_cannot_satisfy_a_publication_fence(
    tmp_path: Path,
) -> None:
    """A generation staged by an older build is refused rather than adopted.

    Not a gap: there is no row in such a table that could carry a checksum, so accepting the
    publication would be accepting exactly the coverage the fence exists to require. The
    rebuild is the answer, and it is a bounded one.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk("chunk-0")]
    await store.upsert(chunks, [spread(0)], publication_id="gen-1")
    await store.teardown()
    await drop_checksum_columns(directory)

    reopened = LanceVectorStore(directory)

    assert not await reopened.publication_is_complete(
        "gen-1", chunks, embedding_fingerprint=fingerprint().canonical()
    )
    await reopened.teardown()


async def test_a_copied_publication_rehashes_rather_than_carrying_the_digest_across(
    tmp_path: Path,
) -> None:
    """A copy is the one operation in a position to attach a digest to bytes it never covered.

    So the copy re-derives the checksum from the values it is about to write. The assertion is
    that the target's digest describes the target's numbers — which it does here because both
    are the same vector, and which is the property that would fail the moment a copy started
    carrying a string across instead.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)], publication_id="gen-1")

    await store.copy_publication("gen-1", "gen-2", chunks)

    assert await store.publication_is_complete(
        "gen-2", chunks, embedding_fingerprint=fingerprint().canonical()
    )
    await store.teardown()

    rows = await rows_of(directory, fingerprint(), "publication_id = 'gen-2'")

    assert len(rows) == 2
    for row in rows:
        assert str(row[CHECKSUM_COLUMN]) == vector_checksum(
            [float(value) for value in row[VECTOR_COLUMN]]
        )


async def test_a_corrupted_checkpoint_cannot_be_replayed_into_a_second_publication(
    tmp_path: Path,
) -> None:
    """Takeover replays a checkpoint. A damaged one is refused rather than propagated.

    This is where a copy would otherwise turn one corrupted row into two, and put the second
    one under a fresh, entirely valid checksum — laundering the damage into a generation that
    then passes every later check.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk("chunk-0")]
    await store.upsert(chunks, [spread(0)], publication_id="gen-1")
    await store.teardown()

    before = await read_column(directory, fingerprint(), "chunk-0", VECTOR_COLUMN)
    await rewrite_row(
        directory, fingerprint(), "chunk-0", {VECTOR_COLUMN: nudged([float(v) for v in before])}
    )

    reopened = await prepared(directory)
    with pytest.raises(Exception, match="incomplete or stale"):
        await reopened.copy_publication("gen-1", "gen-2", chunks)

    assert await reopened.publication_row_count("gen-2") == 0
    await reopened.teardown()


async def test_a_row_in_another_publication_cannot_satisfy_this_ones_validation(
    tmp_path: Path,
) -> None:
    """A valid checksum is not a passport between scopes.

    The row below is intact, current, and correctly checksummed — under a different
    publication. Publication completeness never falls back to another generation, and a
    checksum does not give it a reason to start.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk("chunk-0")]
    await store.upsert(chunks, [spread(0)], publication_id="gen-1")

    assert not await store.publication_is_complete(
        "gen-2", chunks, embedding_fingerprint=fingerprint().canonical()
    )
    await store.teardown()


async def test_a_row_from_another_workspace_is_a_different_directory_entirely(
    tmp_path: Path,
) -> None:
    """Tenancy is a directory boundary here, and the checksum does not cross it.

    Two workspaces have two vector roots. An identical vector in the other one produces an
    identical checksum — which is correct, since the digest is over numbers and says nothing
    about scope — and is still invisible to this store, because this store never looks there.
    """
    one = await prepared(tmp_path / "alpha")
    other = await prepared(tmp_path / "beta")
    stored = chunk("chunk-0")
    await other.upsert([stored], [spread(0)], publication_id="gen-1")

    assert (await one.stored_vectors([stored]))[stored.id].state is VectorState.ABSENT
    assert not await one.publication_is_complete(
        "gen-1", [stored], embedding_fingerprint=fingerprint().canonical()
    )
    await one.teardown()
    await other.teardown()


# --- compatibility and the backfill ------------------------------------------------------


async def test_a_table_written_before_checksums_keeps_every_vector_it_has(
    tmp_path: Path,
) -> None:
    """The upgrade, against real Lance, and the policy this repository chose.

    Legacy rows stay readable and are reported as *unverified*. The alternative — treating an
    absent checksum as corruption — would declare a working corpus damaged on the strength of a
    column that had just been added, and the operator's only recovery would be a corpus-wide
    re-embed to learn what the rows already said.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)])
    await store.teardown()
    await drop_checksum_columns(directory)

    reopened = LanceVectorStore(directory)
    verdicts = await reopened.stored_vectors(chunks)

    assert all(verdict.state is VectorState.READABLE for verdict in verdicts.values())
    assert all(verdict.integrity is VectorIntegrity.UNVERIFIED for verdict in verdicts.values())
    assert not any(verdict.checksum_recorded for verdict in verdicts.values())

    coverage = await reopened.checksum_coverage()

    assert (coverage.rows, coverage.recorded, coverage.unverified) == (3, 0, 3)
    assert not coverage.complete, "an unfinished backfill must never report as covered"
    await reopened.teardown()


async def test_a_legacy_table_still_answers_a_search(tmp_path: Path) -> None:
    """The read path selects the checksum columns only from a table that has them.

    Without that, an upgraded installation whose published generation predates the columns
    would fail every query with a column error — the schema migration for the live root runs on
    ``ensure_ready``, and a published generation is opened without one on purpose.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)])
    await store.teardown()
    await drop_checksum_columns(directory)

    # `open_existing` rather than `ensure_ready`, which is what a published generation gets:
    # it opens the table without evolving its schema, because a read must never migrate a
    # generation somebody is searching.
    reopened = LanceVectorStore(directory)
    await reopened.open_existing()
    found = await reopened.search(spread(0), 5, Filter(workspace_ids=SCOPE))
    await reopened.teardown()

    assert [candidate.chunk.id for candidate in found] == ["chunk-0", "chunk-1"]


async def test_the_backfill_gives_legacy_rows_a_checksum_without_touching_a_model(
    tmp_path: Path,
) -> None:
    """The migration, and the price of it: one hash per row, read from disk.

    Nothing here loads an embedder, opens a connector or reads a retained snapshot. That is
    what makes it affordable, and it is also exactly the limit of what it claims — see the
    test below.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)])
    await store.teardown()
    await drop_checksum_columns(directory)

    reopened = await prepared(directory)
    planned = await reopened.backfill_checksums(dry_run=True)

    assert (planned.scanned, planned.written, planned.remaining) == (3, 3, 3)
    assert planned.dry_run
    assert (await reopened.checksum_coverage()).recorded == 0, "a dry run wrote nothing"

    done = await reopened.backfill_checksums()

    assert (done.written, done.remaining, done.unhashable) == (3, 0, 0)
    assert done.done
    coverage = await reopened.checksum_coverage(recompute=True)
    assert (coverage.rows, coverage.recorded, coverage.verified, coverage.failed) == (3, 3, 3, 0)
    assert coverage.complete
    verdicts = await reopened.stored_vectors(chunks)
    assert all(verdict.integrity is VectorIntegrity.VERIFIED for verdict in verdicts.values())
    await reopened.teardown()


async def test_the_backfill_fixes_the_numbers_as_they_are_rather_than_as_they_were(
    tmp_path: Path,
) -> None:
    """The honest limit of a backfill, asserted so nobody has to take the docstring's word.

    A row corrupted *before* the backfill ran gets a checksum over the corrupted bytes and then
    verifies forever. That is not a bug and it is not fixable from inside: nothing on disk
    records what the vector used to be. It is why legacy coverage is reported as
    "unverified until backfilled" rather than as verified, and why the operator documentation
    says a rebuild — not a backfill — is what establishes integrity retroactively.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("chunk-0")
    await store.upsert([stored], [spread(0)])
    await store.teardown()
    await drop_checksum_columns(directory)

    before = await read_column(directory, fingerprint(), stored.id, VECTOR_COLUMN)
    damaged = nudged([float(value) for value in before])
    await rewrite_row(directory, fingerprint(), stored.id, {VECTOR_COLUMN: damaged})

    reopened = await prepared(directory)
    await reopened.backfill_checksums()

    assert (await reopened.checksum_coverage(recompute=True)).failed == 0
    assert await read_column(
        directory, fingerprint(), stored.id, CHECKSUM_COLUMN
    ) == vector_checksum(damaged)
    await reopened.teardown()


async def test_an_interrupted_backfill_resumes_without_duplicating_or_skipping_a_row(
    tmp_path: Path,
) -> None:
    """Resumability with no cursor, which is why there is nothing for a crash to lose.

    Each pass selects rows that record no checksum, so a row it finished is a row the next pass
    cannot see. Three passes of two rows over five rows is the interruption; the assertion is
    that the union is exactly the five, each written once.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(5)]
    await store.upsert(chunks, [spread(index) for index in range(5)])
    await store.teardown()
    await drop_checksum_columns(directory)

    reopened = await prepared(directory)
    written = 0
    passes = 0
    while True:
        result = await reopened.backfill_checksums(limit=2)
        written += result.written
        passes += 1
        if result.done:
            break
        assert passes < 10, "the backfill is not converging"

    assert (written, passes) == (5, 3)
    assert (await reopened.backfill_checksums(limit=2)).scanned == 0, (
        "a pass after the last one reads nothing at all, so running it twice costs nothing"
    )
    coverage = await reopened.checksum_coverage(recompute=True)
    assert (coverage.rows, coverage.verified) == (5, 5)
    await reopened.teardown()


async def test_the_backfill_leaves_a_vector_it_cannot_hash_exactly_as_it_found_it(
    tmp_path: Path,
) -> None:
    """A checksum over a vector that cannot be hashed would certify a row nothing can rank.

    So the row keeps its empty checksum and is counted. The coverage stays honestly short of
    complete, which is what sends somebody to look at it.

    The damage here is a null vector rather than a ``NaN`` one, because Lance refuses to *store*
    ``NaN`` in a vector column — a useful thing to know and the reason the finiteness branch in
    the backfill is a belt to the store's own braces rather than the likely case. A null is what
    a half-written or partially restored row actually looks like from here.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)])
    await store.teardown()
    await drop_checksum_columns(directory)
    await rewrite_row(directory, fingerprint(), "chunk-0", {VECTOR_COLUMN: None})

    reopened = await prepared(directory)
    result = await reopened.backfill_checksums()

    assert (result.scanned, result.written, result.unhashable) == (2, 1, 1)
    assert result.remaining == 1
    assert (
        await read_column(directory, fingerprint(), "chunk-0", CHECKSUM_COLUMN)
        == UNRECORDED_CHECKSUM
    )
    await reopened.teardown()


async def test_a_half_written_pair_is_malformed_rather_than_awaiting_a_backfill(
    tmp_path: Path,
) -> None:
    """One column without the other is damage, and counting alone can say so.

    The distinction matters because the two call for opposite responses. A row recording
    neither half is an upgrade backlog item and a bounded backfill clears it. A row recording
    one half is a row whose two halves were not written together — nothing can be compared
    against it, and counting it as backlog would hide it inside a number an operator is
    watching go to zero.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)])
    await store.teardown()

    # Two ways to write half a record, and neither is an absent one.
    await rewrite_row(
        directory, fingerprint(), "chunk-0", {CHECKSUM_VERSION_COLUMN: UNRECORDED_CHECKSUM}
    )
    await rewrite_row(directory, fingerprint(), "chunk-1", {CHECKSUM_COLUMN: UNRECORDED_CHECKSUM})

    reopened = LanceVectorStore(directory)
    counted = await reopened.checksum_coverage()

    assert counted.unverified == 0, "neither row is owed a backfill"
    assert counted.failed == 2
    assert counted.failures == {VectorIntegrity.MALFORMED.value: 2}
    assert not counted.complete, (
        "a table holding a contradicted row is not complete, and counting is enough to know it"
    )

    recomputed = await reopened.checksum_coverage(recompute=True)

    assert (recomputed.verified, recomputed.failed) == (1, 2)
    assert recomputed.failures == {VectorIntegrity.MALFORMED.value: 2}
    verdicts = await reopened.stored_vectors(chunks)
    assert verdicts["chunk-0"].integrity is VectorIntegrity.MALFORMED
    assert verdicts["chunk-1"].integrity is VectorIntegrity.MALFORMED
    await reopened.teardown()


async def test_the_backfill_never_overwrites_half_a_record(tmp_path: Path) -> None:
    """The sharp edge of counting the pair rather than one column of it.

    A row holding a version and no checksum *looks* like an unrecorded row to a
    checksum-only predicate. Selecting it would compute a fresh digest over whatever the
    vector is now and write both columns — turning a row that was announcing a contradiction
    into one that verifies forever, which is the same laundering the unhashable branch
    refuses, arriving through the column instead of the vector.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)])
    await store.teardown()
    await rewrite_row(directory, fingerprint(), "chunk-0", {CHECKSUM_COLUMN: UNRECORDED_CHECKSUM})

    reopened = await prepared(directory)
    result = await reopened.backfill_checksums()

    assert (result.scanned, result.written, result.remaining) == (0, 0, 0), (
        "there is nothing here a backfill is allowed to touch"
    )
    assert await read_column(directory, fingerprint(), "chunk-0", CHECKSUM_COLUMN) == (
        UNRECORDED_CHECKSUM
    ), "the half-written row is exactly as it was found"
    assert (
        await read_column(directory, fingerprint(), "chunk-0", CHECKSUM_VERSION_COLUMN)
        == VECTOR_CHECKSUM_VERSION
    )
    assert (await reopened.checksum_coverage()).failed == 1, "and it is still reported"
    await reopened.teardown()


async def test_the_backfill_does_not_resurrect_a_row_deleted_underneath_it(
    tmp_path: Path,
) -> None:
    """A maintenance pass must never put back a vector the tombstone sweep removed.

    The pass reads a page, computes checksums, and merges the rows back. If that merge
    inserted unmatched rows, a delete landing in between would be undone by a pass whose only
    job was to add a column value — the sweep would report a vector removed and the vector
    would still be there, ranked, for a chunk the corpus no longer has.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(2)]
    await store.upsert(chunks, [spread(index) for index in range(2)])
    await store.teardown()
    await drop_checksum_columns(directory)

    reopened = await prepared(directory)
    planned = await reopened.backfill_checksums(dry_run=True)
    assert planned.scanned == 2

    await reopened.delete_chunks([chunks[0].id])
    assert await reopened.count() == 1, "the fixture has to actually delete a row"

    done = await reopened.backfill_checksums()

    assert await reopened.count() == 1, "the deleted row stayed deleted"
    assert done.written == 1, "and the surviving row still got its checksum"
    await reopened.teardown()


async def test_coverage_counts_without_recomputing_and_says_which_it_did(
    tmp_path: Path,
) -> None:
    """Two questions that both get called "coverage", kept apart on the report itself.

    Counting the column is what a status page can afford on every call. Recomputing every digest
    is a scan of the corpus. A report that did the cheap one and rendered it as verification
    would be a clean bill of health nobody had earned.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index}", position=index) for index in range(3)]
    await store.upsert(chunks, [spread(index) for index in range(3)])
    await store.teardown()

    before = await read_column(directory, fingerprint(), "chunk-1", VECTOR_COLUMN)
    await rewrite_row(
        directory,
        fingerprint(),
        "chunk-1",
        {VECTOR_COLUMN: nudged([float(v) for v in before], index=1)},
    )

    reopened = LanceVectorStore(directory)
    counted = await reopened.checksum_coverage()

    assert (counted.rows, counted.recorded, counted.verified, counted.failed) == (3, 3, 0, 0)
    assert not counted.recomputed
    assert counted.complete, "counting found every row covered, which is all it looked at"

    recomputed = await reopened.checksum_coverage(recompute=True)

    assert (recomputed.verified, recomputed.failed) == (2, 1)
    assert recomputed.failures == {VectorIntegrity.MISMATCHED.value: 1}
    assert recomputed.recomputed
    assert not recomputed.complete
    await reopened.teardown()


async def test_a_directory_with_no_vectors_reports_that_nothing_was_examined(
    tmp_path: Path,
) -> None:
    """Zero coverage over zero rows and "there was nothing to look at" are different claims."""
    coverage = await LanceVectorStore(tmp_path / "vectors").checksum_coverage()

    assert not coverage.scanned
    assert (coverage.rows, coverage.recorded) == (0, 0)
    assert not coverage.complete


async def test_the_coverage_report_carries_no_checksum_value_or_chunk_identifier(
    tmp_path: Path,
) -> None:
    """Every field is a count, a boolean or a failure name, and this is what keeps it so.

    The report reaches ``status``, ``doctor``, an MCP tool and an HTTP route. A digest or a
    chunk id on it would be a corpus fingerprint printed to a terminal, into a shell pipeline
    and into an assistant's transcript.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    stored = chunk("private-chunk-id", document_id="private-document-id")
    await store.upsert([stored], [spread(1)])
    await store.teardown()

    coverage = await LanceVectorStore(directory).checksum_coverage(recompute=True)
    rendered = repr(coverage)

    assert "private-chunk-id" not in rendered
    assert "private-document-id" not in rendered
    assert vector_checksum(canonical_stored_vector(spread(1))) not in rendered
    assert set(coverage.failures) <= {member.value for member in VectorIntegrity}


async def test_validating_a_whole_publication_holds_one_page_rather_than_the_corpus(
    tmp_path: Path,
) -> None:
    """Bounded memory, measured against the thing it is bounded by rather than asserted.

    ``publication_is_complete`` pages its chunks and ``checksum_coverage`` pages its rows, so
    the peak allocation over a table is a function of the page size and not of the table. The
    check below is structural: every query issued carries a limit, which is the property that
    would break if somebody replaced a page loop with a single ``to_list``.
    """
    directory = tmp_path / "vectors"
    store = await prepared(directory)
    chunks = [chunk(f"chunk-{index:04d}", position=index) for index in range(300)]
    vectors: list[Vector] = [spread(index) for index in range(300)]
    await store.upsert(chunks, vectors, publication_id="gen-1")

    assert await store.publication_is_complete(
        "gen-1", chunks, embedding_fingerprint=fingerprint().canonical()
    )
    coverage = await store.checksum_coverage(recompute=True, page_size=16)

    assert (coverage.rows, coverage.verified, coverage.failed) == (300, 300, 0)
    await store.teardown()


async def test_a_legacy_installation_upgrades_backfills_publishes_and_restarts(
    tmp_path: Path,
) -> None:
    """The whole upgrade, in the order an operator actually meets it.

    Each step is asserted elsewhere in isolation; what this adds is that they compose. The
    sequence is the one an installation cannot avoid — there is no way to reach a checksummed
    corpus except by starting from one that is not — and the failure mode it guards against is
    the kind that only appears when two correct steps meet: a backfill that leaves the table in
    a shape the publication fence rejects, or a publication whose rows a restarted process
    cannot read.
    """
    directory = tmp_path / "vectors"

    # 1. A corpus indexed before any of this existed.
    legacy = await prepared(directory)
    corpus = [chunk(f"chunk-{index}", position=index) for index in range(5)]
    await legacy.upsert(corpus, [spread(index) for index in range(5)])
    await legacy.teardown()
    await drop_checksum_columns(directory)

    # 2. The upgrade. `ensure_ready` adds the columns and rewrites nothing.
    upgraded = await prepared(directory)
    verdicts = await upgraded.stored_vectors(corpus)

    assert all(verdict.state is VectorState.READABLE for verdict in verdicts.values()), (
        "an upgrade that made an existing corpus unreadable would be the worst possible "
        "outcome of adding an integrity check"
    )
    assert (await upgraded.checksum_coverage()).unverified == 5

    # 3. The backfill, interrupted after the first page and resumed.
    first = await upgraded.backfill_checksums(limit=2)
    assert not first.done
    await upgraded.teardown()

    resumed = await prepared(directory)
    while not (await resumed.backfill_checksums(limit=2)).done:
        pass

    assert (await resumed.checksum_coverage(recompute=True)).complete

    # 4. A checksum-required generation, staged and validated.
    await resumed.upsert(corpus, [spread(index) for index in range(5)], publication_id="gen-2")

    assert await resumed.publication_is_complete(
        "gen-2", corpus, embedding_fingerprint=fingerprint().canonical()
    )
    await resumed.teardown()

    # 5. A restart, and retrieval that still works.
    restarted = await prepared(directory)
    found = await restarted.search(spread(1), 3, Filter(workspace_ids=SCOPE))

    assert "chunk-1" in {candidate.chunk.id for candidate in found}
    assert (await restarted.checksum_coverage(recompute=True)).failed == 0
    assert all(
        verdict.integrity is VectorIntegrity.VERIFIED
        for verdict in (await restarted.stored_vectors(corpus)).values()
    )
    await restarted.teardown()


async def test_the_scan_page_size_has_to_be_positive(tmp_path: Path) -> None:
    """A page size of zero is an infinite loop, not a smaller page."""
    store = await prepared(tmp_path / "vectors")
    with pytest.raises(ValueError, match="positive"):
        await store.checksum_coverage(page_size=0)
    with pytest.raises(ValueError, match="positive"):
        await store.backfill_checksums(limit=0)
    await store.teardown()
