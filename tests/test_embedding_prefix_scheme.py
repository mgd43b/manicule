"""The asymmetric query/document prefixes, and the two sides that have to move together.

Several retrieval models are trained asymmetrically — ``nomic-embed-text`` with
``search_query:``/``search_document:``, Qwen3-Embedding with a query-side instruction — so the
same string embedded as a document and as a query is meant to produce different vectors.
Applying one side, or neither, is not visible from any output: the vectors are well formed,
normalized, and rank against each other happily. Only retrieval quality moves, and nothing
reports it.

So these tests assert on **what reached the model**, which is the only seam where the two
halves are observable at all. ``docs/embeddings.md`` §9.1 has the design, including why this
cannot live in a backend and why ``ChunkFingerprint.embed_text_middleware`` — which can express
the document half and never the query half — is the wrong home.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from manicule.config.settings import EmbeddingSettings, Settings
from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk
from manicule.core.embedding import PrefixScheme, embedding_input_identity
from manicule.core.errors import ContextOverflowError
from manicule.core.protocols import Embedder
from manicule.embedding.cards import load_tokenizer, read_card
from manicule.embedding.config import EmbedderConfig
from manicule.embedding.plugin import read_embedder_card
from manicule.ingest.embedding import embed_chunks
from manicule.plugins import BuildContext
from manicule.testing import write_model
from tests.embedding_fakes import StubEmbedder
from tests.fakes import HashEmbedder
from tests.storage_helpers import make_chunk, make_document

SAME_TEXT = "how does authentication work"
"""One string, embedded on both sides, because that is the case the asymmetry is about."""


def _sized_chunk(text: str, token_count: int) -> Chunk:
    """A chunk whose ``embed_text`` is exactly ``text``, with no breadcrumb in front of it.

    ``make_chunk`` adds a heading breadcrumb, which is right for the tests about *what* gets
    prefixed and wrong for the ones about *how long* the result is: the breadcrumb would make
    the measured length something other than the budget under test.
    """
    return Chunk(
        id="sized",
        document_id="doc",
        text=text,
        embed_text=text,
        anchor=Unlocated(reason="synthetic"),
        position=0,
        token_count=token_count,
    )


def _chunks(*texts: str) -> list[Chunk]:
    document = make_document(source_id="prefixes")
    return [make_chunk(document, position, text) for position, text in enumerate(texts)]


async def _embedded(embedder: HashEmbedder, chunks: list[Chunk]) -> list[str]:
    """What the model was handed for ``chunks``, through the one path ingest uses."""
    await embed_chunks(cast("Embedder", embedder), chunks)
    return embedder.seen


async def test_a_chunk_reaches_the_model_behind_the_document_prefix() -> None:
    """Without this the corpus is embedded bare while queries carry ``search_query:``.

    That is the half-applied scheme §9.1 calls worse than applying none: the two sides land in
    different regions and every ranking degrades, with no error anywhere.
    """
    embedder = HashEmbedder(prefix_scheme=PrefixScheme.NOMIC)
    chunks = _chunks(SAME_TEXT)

    seen = await _embedded(embedder, chunks)

    assert seen == [f"search_document: {chunks[0].embed_text}"]


async def test_the_prefix_goes_in_front_of_the_breadcrumb_not_the_bare_text() -> None:
    """``embed_text`` is what the model reads, and it is not the chunk's text.

    A prefix applied to ``text`` would embed a string the corpus does not store and the reuse
    identity does not describe, so nothing downstream would agree about what was embedded.
    """
    embedder = HashEmbedder(prefix_scheme=PrefixScheme.NOMIC)
    chunks = _chunks(SAME_TEXT)

    seen = await _embedded(embedder, chunks)

    assert chunks[0].embed_text != chunks[0].text
    assert seen[0].removeprefix("search_document: ") == chunks[0].embed_text


async def test_the_scheme_prefixes_every_chunk_in_a_batch_and_not_only_the_first() -> None:
    """Chunks are batched and reordered by length before the model sees them.

    A prefix applied to the batch rather than to each member would survive a one-chunk test and
    lose every chunk after the first in a real ingest.
    """
    embedder = HashEmbedder(prefix_scheme=PrefixScheme.NOMIC)

    seen = await _embedded(embedder, _chunks("alpha", "beta gamma delta", "epsilon"))

    assert len(seen) == 3
    assert all(text.startswith("search_document: ") for text in seen)


async def test_the_default_scheme_hands_the_model_the_chunk_unchanged() -> None:
    """``BAAI/bge-m3`` is symmetric, and a prefix it was never trained with is a regression."""
    embedder = HashEmbedder()
    chunks = _chunks(SAME_TEXT)

    seen = await _embedded(embedder, chunks)

    assert seen == [chunks[0].embed_text]


async def test_a_query_side_only_scheme_leaves_the_document_side_bare() -> None:
    """Qwen3 instructs the query and not the document; prefixing both would be a third space."""
    embedder = HashEmbedder(prefix_scheme=PrefixScheme.QWEN3)
    chunks = _chunks(SAME_TEXT)

    seen = await _embedded(embedder, chunks)

    assert seen == [chunks[0].embed_text]


@pytest.mark.parametrize("scheme", [PrefixScheme.NOMIC, PrefixScheme.QWEN3])
async def test_the_same_text_reaches_the_model_differently_on_the_two_sides(
    scheme: PrefixScheme,
) -> None:
    """The property the whole mechanism exists for, stated once.

    A document and a query that are the same string must not be handed to the model as the same
    string under an asymmetric scheme — and because the embedding cache is keyed on exactly what
    is handed to ``embed``, this is also what stops one side's cached vector being served to the
    other without anyone writing a rule about it.
    """
    embedder = HashEmbedder(prefix_scheme=scheme)
    chunks = _chunks(SAME_TEXT)

    document_side = (await _embedded(embedder, chunks))[-1]
    await embedder.embed([scheme.query(chunks[0].embed_text)])
    query_side = embedder.seen[-1]

    assert document_side != query_side
    assert document_side.endswith(chunks[0].embed_text)
    assert query_side.endswith(chunks[0].embed_text)


# --- the budget the document side has to be paid out of -------------------------------------


def test_the_document_prefix_comes_out_of_the_usable_sequence_length(tmp_path: Path) -> None:
    """``max_sequence_length`` means what is left for a chunk's own text, prefix included.

    The chunker reads this number and refuses to start when its budget exceeds it, so a limit
    that did not know about the prefix would let a corpus be chunked to exactly the length that
    overflows once every chunk is prefixed — and past the limit, input is dropped with nothing
    raised on the backends that truncate.
    """
    write_model(tmp_path / "model")

    bare = read_card(str(tmp_path / "model"))
    prefixed = read_card(str(tmp_path / "model"), prefix_scheme=PrefixScheme.NOMIC)

    cost = len(load_tokenizer(bare).content_ids(PrefixScheme.NOMIC.document_prefix))
    assert cost > 0
    assert prefixed.max_sequence_length == bare.max_sequence_length - cost


def test_a_query_only_scheme_leaves_the_chunk_budget_where_it_was(tmp_path: Path) -> None:
    """Qwen3's instruction is never prepended to a chunk, so charging it would waste context."""
    write_model(tmp_path / "model")

    bare = read_card(str(tmp_path / "model"))
    prefixed = read_card(str(tmp_path / "model"), prefix_scheme=PrefixScheme.QWEN3)

    assert prefixed.max_sequence_length == bare.max_sequence_length


def test_a_configured_sequence_length_is_charged_the_prefix_too(tmp_path: Path) -> None:
    """The override is the model's limit less *its* special tokens, not less our prefix.

    An operator can read the first number off their model's declaration. Expecting them to also
    subtract a prefix manicule chose would make the setting mean something different depending
    on another setting, and the error would be silent truncation.
    """
    write_model(tmp_path / "model")

    prefixed = read_card(
        str(tmp_path / "model"),
        max_sequence_length_override=20,
        prefix_scheme=PrefixScheme.NOMIC,
    )

    cost = len(load_tokenizer(prefixed).content_ids(PrefixScheme.NOMIC.document_prefix))
    assert prefixed.max_sequence_length == 20 - cost


def test_the_card_carries_the_scheme_its_budget_was_computed_for(tmp_path: Path) -> None:
    """Two values that could disagree would be a fingerprint describing a budget it never paid."""
    write_model(tmp_path / "model")

    card = read_card(str(tmp_path / "model"), prefix_scheme=PrefixScheme.NOMIC)

    assert card.prefix_scheme is PrefixScheme.NOMIC
    assert card.fingerprint(backend="fake").prefix_scheme is PrefixScheme.NOMIC


def test_the_configured_scheme_reaches_the_card_every_backend_builds_from(tmp_path: Path) -> None:
    """`read_embedder_card` is the shared route `onnx` and `manicule-mlx` both take.

    The setting is core's and the application is core's, but the *recording* happens wherever a
    fingerprint is built — so a factory that read every other setting and dropped this one would
    produce an embedder that records `none` and is then correctly never prefixed. The setting
    would appear to be in force and would not be, which is the failure mode configuration
    validation exists to prevent and the one a type checker cannot see.
    """
    write_model(tmp_path / "model")
    settings = Settings(
        embedding=EmbeddingSettings(model=str(tmp_path / "model"), prefix_scheme=PrefixScheme.NOMIC)
    )
    context = BuildContext(
        settings=settings,
        config=EmbedderConfig(),
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        components=None,  # pyright: ignore[reportArgumentType] - unused by this helper
    )

    card, _ = read_embedder_card(context)

    assert card.prefix_scheme is PrefixScheme.NOMIC
    assert card.fingerprint(backend="onnx").prefix_scheme is PrefixScheme.NOMIC


def test_a_stored_vector_is_not_reused_when_the_scheme_moves() -> None:
    """Reuse is keyed on the embedding *input*, and the input here changed invisibly.

    `chunks.embed_text` is stored unprefixed and re-prefixed on the way to the model, so the
    text a reuse rule sees is byte-identical before and after a scheme is adopted. What makes
    the stored vector inadmissible is the fingerprint folded into the same identity — and
    without that, `reindex --repair` would keep every unprefixed vector under a configuration
    that now prefixes, which is the half-applied scheme §9.1 calls worse than none.
    """
    bare = HashEmbedder().fingerprint
    prefixed = HashEmbedder(prefix_scheme=PrefixScheme.NOMIC).fingerprint
    same_text = "the chunk text is not what changed"

    assert embedding_input_identity(
        same_text, document_id="doc-1", embed=bare
    ) != embedding_input_identity(same_text, document_id="doc-1", embed=prefixed)


# --- the prefix is charged once, not twice ---------------------------------------------------


async def test_a_chunk_sized_to_the_budget_survives_its_own_prefix(tmp_path: Path) -> None:
    """The budget the chunker is handed has to be one the backend will actually accept.

    Two numbers are in play and collapsing them charges the prefix twice.
    `max_sequence_length` is what is left for a chunk's own text once the document prefix has
    been netted out; `ModelCard.input_capacity` is what the model reads, prefix included. The
    backend's raw-input guard sees the *prefixed* string, so measuring it against the reduced
    number refuses a chunk sized to exactly the budget that produced it — and the corpus this
    matters for is the ordinary one, where most chunks sit near the limit.
    """
    write_model(tmp_path / "model")
    card = read_card(str(tmp_path / "model"), prefix_scheme=PrefixScheme.NOMIC)
    embedder = StubEmbedder(card)
    await embedder.setup()
    try:
        budget = card.max_sequence_length
        text = " ".join(["alpha"] * budget)
        assert embedder.count_tokens(text) == budget
        assert card.document_prefix_tokens > 0
        assert card.input_capacity == budget + card.document_prefix_tokens

        vectors = await embed_chunks(embedder, [_sized_chunk(text, budget)])

        assert len(vectors) == 1
    finally:
        await embedder.teardown()


async def test_a_chunk_past_the_budget_is_still_refused(tmp_path: Path) -> None:
    """The guard that was double-charging still has to fire when it should.

    Widening a limit to fix a false refusal is the obvious way to turn one bug into a worse
    one: past the model's real capacity the input is truncated with nothing raised, and the
    stored vector describes an opening fragment while its chunk claims all of its text.
    """
    write_model(tmp_path / "model")
    card = read_card(str(tmp_path / "model"), prefix_scheme=PrefixScheme.NOMIC)
    embedder = StubEmbedder(card)
    await embedder.setup()
    try:
        over = card.input_capacity + 1
        text = " ".join(["alpha"] * over)

        with pytest.raises(ContextOverflowError):
            await embedder.embed([PrefixScheme.NOMIC.document(text)])
    finally:
        await embedder.teardown()
