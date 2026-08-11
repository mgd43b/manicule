"""The chunker: budgets, structure, breadcrumbs, and the refusals that protect an index."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import pytest

from manicule.chunking import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    PROVISIONAL_SAFETY_FACTOR,
    StructuralChunker,
    TokenCounter,
)
from manicule.chunking.breadcrumb import render
from manicule.chunking.sentences import sentences
from manicule.chunking.tokens import tiktoken_tokenizer_id
from manicule.core.anchors import CellAnchor, LineAnchor, PageAnchor
from manicule.core.content import BlockKind, Document, DocumentStatus, Metadata, ParsedBlock
from manicule.core.embedding import EmbedFingerprint, Pooling, Vector
from manicule.core.errors import ConfigError, ContextOverflowError
from manicule.core.fingerprints import PROVISIONAL_TOKENIZER_PREFIX
from manicule.testing import assert_chunker_contract

TOKENIZER_ID = "test/whitespace"


def counter(provisional: bool = False) -> TokenCounter:
    return TokenCounter(TOKENIZER_ID, lambda text: len(text.split()), provisional=provisional)


def make_chunker(**kwargs: object) -> StructuralChunker:
    return StructuralChunker(counter(), **kwargs)  # pyright: ignore[reportArgumentType] - test-only pass-through


def document(title: str = "Token Refresh", **metadata: object) -> Document:
    carried: Metadata = {}
    for key, value in metadata.items():
        carried[key] = value  # pyright: ignore[reportArgumentType] - test values are JSON-shaped
    return Document(
        id="doc-1",
        source="fixtures",
        source_id="s1",
        uri="doc",
        title=title,
        content_hash="h",
        media_type="text/plain",
        status=DocumentStatus.PARSED,
        metadata=carried,
    )


def prose(text: str, start: int = 1, end: int = 1, path: Sequence[str] = ()) -> ParsedBlock:
    return ParsedBlock(
        kind=BlockKind.PROSE,
        text=text,
        anchor=LineAnchor(start=start, end=end),
        heading_path=tuple(path),
    )


class _Embedder:
    """The minimum an embedder has to offer for the chunker to check its own budget."""

    def __init__(self, limit: int) -> None:
        self.fingerprint = EmbedFingerprint(
            model_id="test/model",
            dimension=8,
            pooling=Pooling.MEAN,
            normalized=True,
            tokenizer_id=TOKENIZER_ID,
            max_sequence_length=limit,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """A vector per input, so this double satisfies the protocol rather than half of it.

        The values are arbitrary and nothing here reads them; what matters is that the
        conformance suite is handed a real ``Embedder`` and not something that happens to
        carry the two attributes the budget check reads.
        """
        return [[float(len(text))] * self.fingerprint.dimension for text in texts]


# --- the budget refusal --------------------------------------------------------------------


async def test_a_budget_wider_than_the_model_refuses_to_start() -> None:
    """Past a model's sequence length the input is dropped with no error raised.

    The chunk is then indexed as its opening tokens while still claiming all of its text — a
    citation quoting words the index never saw. There is nothing downstream that can detect
    it, so the only place to catch it is before a corpus exists.
    """
    chunker = StructuralChunker(counter(), embedder=_Embedder(256), max_tokens=512)
    with pytest.raises(ContextOverflowError, match="attends to 256"):
        await chunker.setup()


async def test_a_budget_inside_the_model_limit_starts() -> None:
    chunker = StructuralChunker(counter(), embedder=_Embedder(512), max_tokens=512)
    await chunker.setup()


async def test_a_chunker_with_no_embedder_bound_starts_and_marks_its_chunks() -> None:
    """Parsing without embedding is legitimate — a dry run, a fixture suite.

    What is not legitimate is those chunks reaching an index: a count taken with a stand-in
    vocabulary can undercount by an unknown margin, and undercounting is the direction that
    truncates. So they are inflated and marked, and ingest refuses them.
    """
    chunker = StructuralChunker(counter(provisional=True))
    await chunker.setup()
    assert chunker.provisional
    chunks = chunker.chunk(document(), [prose("alpha beta gamma")])
    assert chunks[0].metadata["provisional"] is True


def test_a_chunk_records_the_language_its_blocks_agree_on() -> None:
    """A promoted column nothing populates is a filter field that can never match.

    Both stores read ``metadata["lang"]`` into a column a ``langs`` filter resolves against,
    and nothing was putting it there: ``ParsedBlock.lang`` stopped at the chunker.
    """
    chunker = make_chunker()
    blocks = [
        prose("alpha beta gamma").model_copy(update={"lang": "fr"}),
        prose("delta epsilon", start=2, end=2).model_copy(update={"lang": "fr"}),
    ]

    chunks = chunker.chunk(document(), blocks)

    assert [chunk.metadata["lang"] for chunk in chunks] == ["fr"]


def test_a_chunk_whose_blocks_disagree_about_language_claims_none() -> None:
    """``ParsedBlock.lang`` means a code language on one block and a natural one on the next.

    Naming either as the chunk's language would make a ``langs`` filter return the other.
    ``None`` is what undetermined already means everywhere else it appears.
    """
    chunker = make_chunker()
    blocks = [
        prose("alpha beta gamma").model_copy(update={"lang": "en"}),
        ParsedBlock(
            kind=BlockKind.CODE,
            text="def f():\n    return 1",
            anchor=LineAnchor(start=2, end=3),
            lang="python",
        ),
    ]

    chunks = chunker.chunk(document(), blocks)

    assert len(chunks) == 1
    assert "lang" not in chunks[0].metadata


def test_a_provisional_count_is_inflated_rather_than_trusted() -> None:
    inflated = counter(provisional=True)("one two three four")
    assert inflated > int(4 * PROVISIONAL_SAFETY_FACTOR) - 1


def test_two_stand_in_counters_that_disagree_do_not_share_a_fingerprint() -> None:
    """The defect, in one line: an estimator returning 1 and one returning 999.

    Both chunk a corpus differently and both used to record ``tokenizer_id`` as the literal
    string ``"provisional"``, so the two fingerprints were byte-identical — an index built
    with either was accepted as an index built with the other. Naming the counter is now
    required, which makes the collision unrepresentable rather than merely unlikely.
    """
    one = TokenCounter.provisionally(lambda _: 1, tokenizer_id="always-one")
    other = TokenCounter.provisionally(lambda _: 999, tokenizer_id="always-999")

    assert one.tokenizer_id != other.tokenizer_id
    assert StructuralChunker(one).fingerprint != StructuralChunker(other).fingerprint


def test_a_stand_in_counter_that_cannot_name_itself_is_refused() -> None:
    """``provisionally(lambda t: 1)`` is the call that produced two corpora with one identity.

    A callable carries no version anyone can read — every lambda's ``__qualname__`` is
    ``<lambda>`` and its ``id()`` differs between two runs of the same program — so deriving
    an identity here would be either indistinguishing or irreproducible. The caller is the
    only party that knows.
    """
    with pytest.raises(ConfigError, match="must name itself"):
        TokenCounter.provisionally(lambda _: 1)

    with pytest.raises(ConfigError, match="without a counter"):
        TokenCounter.provisionally(tokenizer_id="pretending-to-be-tiktoken")


def test_the_default_stand_in_records_its_vocabulary_and_its_version() -> None:
    """``cl100k_base`` has meant different boundaries across releases; the name alone lies."""
    identity = tiktoken_tokenizer_id()

    assert identity.startswith("tiktoken/cl100k_base@")
    assert identity != "tiktoken/cl100k_base@"


def test_every_construction_path_stamps_the_provisional_identity() -> None:
    """The public constructor inflates too, so it is where the stamp goes.

    A caller reaching ``TokenCounter(...)`` directly — which the fixtures in this suite do —
    would otherwise multiply every count by the safety factor while recording an id saying it
    had not.
    """
    stamped = TokenCounter(TOKENIZER_ID, lambda text: len(text.split()), provisional=True)
    measured = TokenCounter(TOKENIZER_ID, lambda text: len(text.split()), provisional=False)

    assert stamped.tokenizer_id.startswith(PROVISIONAL_TOKENIZER_PREFIX)
    assert str(PROVISIONAL_SAFETY_FACTOR) in stamped.tokenizer_id
    assert TOKENIZER_ID in stamped.tokenizer_id
    assert measured.tokenizer_id == TOKENIZER_ID
    assert StructuralChunker(stamped).fingerprint.provisional
    assert not StructuralChunker(measured).fingerprint.provisional


def test_a_counter_that_names_nothing_cannot_be_built() -> None:
    """An unnamed counter produces boundaries no fingerprint can describe."""
    with pytest.raises(ConfigError, match="must name what counted"):
        TokenCounter("", lambda _: 1, provisional=False)


def test_the_fingerprint_records_everything_that_moves_a_boundary() -> None:
    """A change to any of these re-chunks the corpus, so each is compared before ingest.

    ``tokenizer_id`` in particular: the same budget measured with a different vocabulary
    produces different boundaries, and a model swap that keeps the dimension but changes the
    vocabulary would otherwise pass the embedder check and quietly re-chunk everything.
    """
    chunker = make_chunker(grammars={"python": "0.23.0"}, version_components={"html_text": "2"})
    fingerprint = chunker.fingerprint
    assert fingerprint.max_tokens == MAX_TOKENS
    assert fingerprint.overlap_tokens == OVERLAP_TOKENS
    assert fingerprint.tokenizer_id == TOKENIZER_ID
    assert fingerprint.grammars == {"python": "0.23.0"}
    assert "html_text=2" in fingerprint.version


def test_a_grammar_bump_changes_only_the_grammar_field() -> None:
    """Recorded per language so a Python grammar upgrade invalidates Python documents and
    leaves the rest of the corpus alone."""
    before = make_chunker(grammars={"python": "0.23.0", "rust": "0.21.0"}).fingerprint
    after = make_chunker(grammars={"python": "0.24.0", "rust": "0.21.0"}).fingerprint
    assert before.changed_fields(after) == frozenset({"grammars"})


# --- structure -----------------------------------------------------------------------------


def test_a_table_is_kept_whole_rather_than_severed_at_a_token_count() -> None:
    """A table cut in half is a grid of numbers with no header and no way to read it."""
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text="Region | Q1\nEMEA | 12\nAPAC | 15",
        anchor=CellAnchor(sheet="Regional", ref="A1:B3"),
        metadata={"rows": ["Region | Q1", "EMEA | 12", "APAC | 15"], "header_rows": 1},
    )
    chunks = make_chunker().chunk(document(), [table])
    assert len(chunks) == 1
    assert chunks[0].text == table.text


def test_an_oversized_table_splits_by_rows_and_repeats_the_header() -> None:
    """Each part must be independently meaningful and independently retrievable.

    Header rows come from the parser, never from the first row looking bold — a guess would
    silently promote a data row on every table that has no header.
    """
    rows = ["Region | Value", *[f"Region-{index} | {index}" for index in range(200)]]
    refs = [f"A{index + 1}:B{index + 1}" for index in range(len(rows))]
    carried: Metadata = {"rows": [*rows], "header_rows": 1, "row_refs": [*refs]}
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text="\n".join(rows),
        anchor=CellAnchor(sheet="Regional", ref="A1:B201"),
        metadata=carried,
    )
    chunks = make_chunker().chunk(document(), [table])
    assert len(chunks) > 1
    assert all(chunk.text.startswith("Region | Value") for chunk in chunks)
    assert all(isinstance(chunk.anchor, CellAnchor) for chunk in chunks)
    later = chunks[-1].anchor
    assert isinstance(later, CellAnchor)
    assert later.ref.startswith("A1:B1,"), (
        "a part whose header is repeated into its text must address the header rows too, or "
        "the citation resolves to fewer rows than it quotes"
    )


def test_a_chunk_never_spans_a_page_boundary() -> None:
    """A chunk naming one page while half its text is on the next reads correctly and is
    wrong. Where two blocks' anchors cannot combine, the chunk closes."""
    blocks = [
        ParsedBlock(kind=BlockKind.PROSE, text="alpha on one", anchor=PageAnchor(page=1)),
        ParsedBlock(kind=BlockKind.PROSE, text="beta on two", anchor=PageAnchor(page=2)),
    ]
    chunks = make_chunker().chunk(document(), blocks)
    assert [chunk.anchor for chunk in chunks] == [PageAnchor(page=1), PageAnchor(page=2)]


def test_a_heading_starts_a_chunk_and_never_becomes_one() -> None:
    """A heading is a boundary and a breadcrumb component, not content."""
    blocks = [
        ParsedBlock(
            kind=BlockKind.HEADING, text="Configuration", anchor=LineAnchor(start=1, end=1)
        ),
        prose("Rotation runs hourly.", 2, 2, ("Configuration",)),
        ParsedBlock(kind=BlockKind.HEADING, text="Rollback", anchor=LineAnchor(start=4, end=4)),
        prose("Restore the previous release.", 5, 5, ("Rollback",)),
    ]
    chunks = make_chunker().chunk(document(), blocks)
    assert len(chunks) == 2
    assert chunks[0].text == "Rotation runs hourly."
    assert chunks[0].heading_path == ("Configuration",)


def test_a_document_of_only_headings_is_indexed_rather_than_reported_as_empty() -> None:
    """A stub page or a table of contents would otherwise produce zero chunks and look
    exactly like a document with no extractable text — which would put it in the bucket that
    triggers the scanned-corpus warning it has nothing to do with."""
    blocks = [
        ParsedBlock(kind=BlockKind.HEADING, text="Overview", anchor=LineAnchor(start=1, end=1)),
        ParsedBlock(kind=BlockKind.HEADING, text="Details", anchor=LineAnchor(start=2, end=2)),
    ]
    chunks = make_chunker().chunk(document(), blocks)
    assert len(chunks) == 1
    assert chunks[0].kind is BlockKind.PROSE
    assert "Overview" in chunks[0].text


def test_an_oversized_paragraph_splits_at_sentences_before_it_splits_at_tokens() -> None:
    """A chunk that begins mid-sentence retrieves poorly and reads worse when cited."""
    body = " ".join(f"Sentence number {index} explains the rollout." for index in range(200))
    chunks = make_chunker().chunk(document(), [prose(body)])
    assert len(chunks) > 1
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)
    assert not any(chunk.metadata.get("hard_split") for chunk in chunks)


def test_a_single_sentence_longer_than_the_budget_is_recorded_as_a_hard_split() -> None:
    """A minified line or a pasted base64 blob has no sentence to cut at.

    It still must not be truncated — that is data loss — so it is cut at a token boundary and
    the fact is recorded, because a document full of these is a document that will retrieve
    badly and somebody should be able to find out.
    """
    blob = "x" + " ".join("q" for _ in range(2000))
    chunks = make_chunker().chunk(document(), [prose(blob)])
    assert len(chunks) > 1
    assert any(chunk.metadata.get("hard_split") for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == blob.replace(" ", "")


# --- overlap ---------------------------------------------------------------------------------


def test_overlap_repeats_whole_sentences_between_adjacent_prose_chunks() -> None:
    """The sentence that answers a question can straddle a boundary, and a chunk holding half
    of it retrieves poorly and reads worse when cited."""
    body = " ".join(f"Sentence {index} covers the rollout in detail." for index in range(200))
    chunks = make_chunker().chunk(document(), [prose(body)])
    assert len(chunks) > 1
    shared = _shared_prefix_tokens(chunks[0].text, chunks[1].text)
    assert shared > 0, "adjacent prose chunks carried no overlap at all"
    assert shared <= OVERLAP_TOKENS
    repeated = " ".join(chunks[1].text.split()[:shared])
    assert repeated in sentences_joined(chunks[0].text), (
        "the overlap window must be whole sentences: a chunk that begins on a fragment is "
        "exactly what the window exists to prevent"
    )
    assert repeated.endswith(".")


def sentences_joined(text: str) -> str:
    return " ".join(sentences(text))


def test_code_is_never_overlapped() -> None:
    """An overlapping code fragment emits lines that another chunk already claims, which is
    indistinguishable from an anchor that is simply wrong."""
    source = "\n".join(f"line_{index} = compute({index})" for index in range(900))
    block = ParsedBlock(
        kind=BlockKind.CODE, text=source, anchor=LineAnchor(start=1, end=900), lang="python"
    )
    chunks = make_chunker().chunk(document(), [block])
    assert len(chunks) > 1
    first_lines = set(chunks[0].text.splitlines())
    second_lines = set(chunks[1].text.splitlines())
    assert not (first_lines & second_lines)


def test_the_overlap_window_is_capped_at_half_the_preceding_chunk() -> None:
    """The window and the minimum chunk size are the same number.

    Uncapped, a minimum-sized chunk's successor would open with an exact copy of the whole
    thing — two chunks, one entirely contained in the other, both matching the same query and
    both consuming a slot in the ranking.
    """
    chunker = StructuralChunker(counter(), max_tokens=40, overlap_tokens=30, breadcrumb_tokens=4)
    body = " ".join(f"Short sentence {index} here." for index in range(40))
    chunks = chunker.chunk(document(), [prose(body)])
    assert len(chunks) > 1
    for earlier, later in pairwise(chunks):
        shared = _shared_prefix_tokens(earlier.text, later.text)
        assert shared <= len(earlier.text.split()) / 2 + 1


def _shared_prefix_tokens(previous: str, current: str) -> int:
    previous_words, current_words = previous.split(), current.split()
    for size in range(min(len(previous_words), len(current_words)), 0, -1):
        if previous_words[-size:] == current_words[:size]:
            return size
    return 0


# --- text versus embed_text --------------------------------------------------------------


def test_the_breadcrumb_reaches_the_embedder_and_never_the_quotation() -> None:
    """``text`` is what a user is shown as the quote, and a quote prefixed with navigation is
    not a quote."""
    chunks = make_chunker().chunk(
        document(title="Auth Service"), [prose("Rotation runs hourly.", path=("Configuration",))]
    )
    chunk = chunks[0]
    assert chunk.text == "Rotation runs hourly."
    assert chunk.embed_text.startswith("Auth Service > Configuration")
    assert chunk.text in chunk.embed_text


def test_a_breadcrumb_that_would_stutter_says_it_once() -> None:
    """A page titled "Auth Service" under a parent of the same name yields one element.

    Without this, roughly a third of real breadcrumbs repeat themselves, and the repetition
    reaches the embedder as emphasis nobody intended.
    """
    chunks = make_chunker().chunk(
        document(title="Auth Service", ancestors=["Platform", "Auth Service"]),
        [prose("Rotation runs hourly.", path=("Auth Service", "Configuration"))],
    )
    assert chunks[0].embed_text.startswith("Platform > Auth Service > Configuration")


def test_a_document_with_no_hierarchy_gets_no_breadcrumb_rather_than_an_invented_one() -> None:
    """A fabricated breadcrumb is a fabricated signal in the vector."""
    chunks = make_chunker().chunk(document(title=""), [prose("Just some text.")])
    assert chunks[0].embed_text == chunks[0].text


def test_an_over_long_breadcrumb_is_elided_from_the_middle() -> None:
    """The two ends carry the most information: the outermost says which corpus and product
    area, the innermost says what this section is. Truncating the tail throws away the element
    that disambiguates "Configuration"."""
    parts = ["ENG", "Platform", "Auth Service", "Token Refresh", "Rotation", "Configuration"]
    rendered = render(parts, lambda text: len(text.split()), 6)
    assert rendered.startswith("ENG")
    assert rendered.endswith("Configuration")
    assert "…" in rendered


# --- minimum size ----------------------------------------------------------------------------


def test_a_tiny_trailing_chunk_is_merged_backwards_rather_than_left_or_dropped() -> None:
    """A short text produces a vector dominated by its few tokens and wins queries it should
    lose. It is never dropped — dropping is data loss."""
    chunker = StructuralChunker(counter(), max_tokens=40, min_tokens=10, breadcrumb_tokens=4)
    blocks = [prose(" ".join(f"word{index}" for index in range(20)), 1, 1), prose("tail", 2, 2)]
    chunks = chunker.chunk(document(), blocks)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("tail")


# --- the shipped conformance suite ---------------------------------------------------------


def test_the_chunker_passes_the_shipped_contract() -> None:
    blocks = [
        prose("Rotation runs hourly and is not configurable.", 1, 1, ("Configuration",)),
        prose("Restore the previous release.", 3, 3, ("Configuration",)),
    ]
    chunks = assert_chunker_contract(make_chunker(), document(), blocks, _Embedder(512))
    assert chunks
