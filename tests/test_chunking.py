"""The chunker: budgets, structure, breadcrumbs, and the refusals that protect an index."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import cast, override

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
from manicule.core.errors import ChunkingError, ConfigError, ContextOverflowError
from manicule.core.fingerprints import PROVISIONAL_TOKENIZER_PREFIX
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
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


@pytest.mark.parametrize("with_header", [True, False])
def test_one_oversized_table_row_is_lossless_and_every_fragment_is_bounded(
    with_header: bool,
) -> None:
    header = "Header | Value"
    row = f"Item | {'x' * 350}"
    rows = [header, row] if with_header else [row]
    refs = ["A1:B1", "A2:B2"] if with_header else ["A1:B1"]
    chunker = StructuralChunker(
        byte_counter(), max_tokens=128, overlap_tokens=0, breadcrumb_tokens=16, min_tokens=1
    )
    metadata = cast(
        "Metadata",
        {"rows": rows, "header_rows": 1 if with_header else 0, "row_refs": refs},
    )
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text="\n".join(rows),
        anchor=CellAnchor(sheet="Synthetic", ref="A1:B2" if with_header else "A1:B1"),
        metadata=metadata,
    )

    chunks = chunker.chunk(document(title="Synthetic table"), [table])

    assert len(chunks) > 1
    assert all(chunk.token_count <= 128 for chunk in chunks)
    assert all(chunk.metadata.get("hard_split_at") == "row" for chunk in chunks)
    if with_header:
        prefix = f"{header}\n"
        assert all(chunk.text.startswith(prefix) for chunk in chunks)
        assert "".join(chunk.text.removeprefix(prefix) for chunk in chunks) == row
    else:
        assert "".join(chunk.text for chunk in chunks) == row


def test_an_oversized_header_and_row_fall_back_to_a_lossless_bounded_split() -> None:
    header = "H" * 150
    row = "R" * 200
    chunker = StructuralChunker(
        byte_counter(), max_tokens=128, overlap_tokens=0, breadcrumb_tokens=16, min_tokens=1
    )
    metadata: Metadata = {
        "rows": [header, row],
        "header_rows": 1,
        "row_refs": ["A1", "A2"],
    }
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text=f"{header}\n{row}",
        anchor=CellAnchor(sheet="Synthetic", ref="A1:A2"),
        metadata=metadata,
    )

    chunks = chunker.chunk(document(title="Synthetic table"), [table])

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == table.text
    assert all(chunk.token_count <= 128 for chunk in chunks)
    assert all(chunk.metadata.get("hard_split_at") == "row" for chunk in chunks)


# A header row wide enough that the table alone exceeds the 448-token text budget, which is
# what makes `_split_table` run at all: under the budget a table is placed whole and the
# splitting path is never reached, so a small header-only table was never at risk.
_WIDE_HEADER = " | ".join(" ".join(f"{name}{index}" for index in range(120)) for name in "abcde")


@pytest.mark.parametrize(
    ("label", "metadata"),
    [
        ("every row is a header row", {"rows": [_WIDE_HEADER], "header_rows": 1}),
        ("header_rows exceeds the row count", {"rows": [_WIDE_HEADER], "header_rows": 9}),
        ("rows is empty beside real text", {"rows": [], "header_rows": 0}),
    ],
)
def test_an_oversized_table_with_no_row_to_split_at_is_still_indexed(
    label: str, metadata: Metadata
) -> None:
    """A table the chunker cannot split by row must not vanish.

    ``_split_table`` builds its parts from the rows *after* the header, so each of these
    produced no parts and returned nothing — and a block that yields no unit reaches no chunk,
    no vector and no citation. It is the quietest possible content loss: the table is simply
    absent, and a document that was only this table looks exactly like one with no extractable
    text. Every parser that emits ``rows`` can produce the shape, so the guard is here rather
    than in seven parsers that must each remember it.

    Prose splitting is the fallback because it is the answer this function already gives when
    ``rows`` is absent: there is no row boundary to split at, so keep the text whole.
    """
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text=_WIDE_HEADER,
        anchor=CellAnchor(sheet="Regional", ref="A1:E1"),
        metadata=metadata,
    )
    chunks = make_chunker().chunk(document(), [table])
    assert chunks, f"{label}: the table produced no chunks at all"
    indexed = " ".join(chunk.text for chunk in chunks).split()
    assert set(_WIDE_HEADER.split()) <= set(indexed), (
        f"{label}: words from the table reached no chunk, so they are in no vector and "
        f"quotable in no citation"
    )


def test_a_table_of_only_header_rows_splits_at_its_rows_rather_than_as_prose() -> None:
    """Knowing the boundaries and not using them is the defect ``rows`` exists to prevent.

    An all-header table has no data row to repeat a header into, but its row boundaries are
    just as known as any other table's. Falling back to prose here would discard them and cut
    the table mid-row and mid-cell — measured on this fixture, three of its sixty rows ended up
    in no chunk intact — which is exactly what emitting ``rows`` was added to stop. It would
    also leave every part claiming the whole table's ``CellAnchor`` while quoting a fifth of it.
    """
    rows = [" | ".join(f"r{index}c{column}" for column in range(12)) for index in range(60)]
    refs = [f"A{index + 1}:L{index + 1}" for index in range(len(rows))]
    table = ParsedBlock(
        kind=BlockKind.TABLE,
        text="\n".join(rows),
        anchor=CellAnchor(sheet="Regional", ref="A1:L60"),
        metadata={"rows": [*rows], "header_rows": len(rows), "row_refs": [*refs]},
    )
    chunks = make_chunker().chunk(document(), [table])
    assert len(chunks) > 1, "the fixture is meant to exceed the budget and be split"

    intact = {line for chunk in chunks for line in chunk.text.split("\n")}
    assert set(rows) <= intact, "a row was cut across two chunks, so it is quotable in neither"
    assert not any(chunk.metadata.get("hard_split") for chunk in chunks), (
        "a table split at boundaries the parser supplied is not a hard split, and counting it "
        "as one would tell `doctor` the corpus retrieves worse than it does"
    )
    # Asserted against the block's own anchor rather than against a literal ref, because what
    # matters is that each part was narrowed — not how `_collapse_areas` happens to spell the
    # result. A literal would also pin today's spelling: `_adjacent` compares the left area's
    # end column against the right area's start column, so multi-column rows do not collapse
    # and the ref reads `A1:L1,A2:L2,…`. That is a defect in its own right and not this one's.
    anchors = [chunk.anchor for chunk in chunks]
    assert all(isinstance(anchor, CellAnchor) for anchor in anchors)
    assert table.anchor not in anchors, (
        "every part claimed the whole table, so each citation resolves to sixty rows while "
        "quoting a fraction of them — the tightness a split is supposed to buy"
    )
    assert len(set(anchors)) == len(anchors), "two parts addressed the same rows"


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


def test_an_inline_break_stays_inside_the_chunk_that_holds_it() -> None:
    """A block under the budget reaches a chunk with its line breaks intact.

    ``text`` is what is cited and shown, so a break the page drew has to survive packing —
    otherwise the parser's fidelity is undone one step later and nothing says so.
    """
    chunks = make_chunker().chunk(document(), [prose("primary endpoint\nsecondary endpoint")])

    assert len(chunks) == 1
    assert chunks[0].text == "primary endpoint\nsecondary endpoint"


def test_only_a_blank_line_splits_an_oversized_block_and_a_lone_break_never_does() -> None:
    """§4.5's rule, asserted from the chunker's side of it.

    Two paragraphs, each with an inline break in it, and each too long to sit with the other.
    The break must not become a boundary — a chunk beginning at one would start mid-paragraph
    — and the blank line must, which is the entire distinction the parsers owe.
    """
    first = " ".join(f"Alpha sentence {index} explains the rollout." for index in range(120))
    second = " ".join(f"Beta sentence {index} explains the rollback." for index in range(120))
    body = f"{first}\ninline continuation of the first.\n\n{second}\ninline continuation."

    chunks = make_chunker().chunk(document(), [prose(body)])

    assert len(chunks) > 1
    assert not any(chunk.text.startswith("inline continuation") for chunk in chunks), (
        "a lone newline is not a boundary, so no chunk may begin at one"
    )
    assert any("rollout.\ninline continuation of the first." in chunk.text for chunk in chunks), (
        "and the break survives inside the paragraph it belongs to"
    )


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


def test_code_ending_a_mostly_prose_chunk_is_not_overlapped() -> None:
    """The overlap guards ask the group's *dominant* kind, which is not each unit's kind.

    ``test_code_is_never_overlapped`` above uses a document that is entirely code, so
    ``_dominant_kind`` is CODE and the guard fires before the walk begins. The case it cannot
    reach is the ordinary one: a section of prose that ends with a code sample. There the
    dominant kind is PROSE, both guards pass, and the backwards walk took whatever unit sat at
    the end of the previous chunk — the code block.

    That is the defect the pure-code test was written to prevent, arriving through the door it
    does not cover: the sample is emitted in two chunks, indexed twice, and can be cited from
    the chunk that is not where it lives. Overlap exists so a *sentence* split across a boundary
    stays searchable from both sides; a duplicated code block is not that.
    """
    sentences_before = " ".join(f"Sentence {index} covers the rollout." for index in range(120))
    sample = "\n".join(f"call_{index}(argument)" for index in range(12))
    blocks = [
        prose(sentences_before, start=1, end=120),
        ParsedBlock(
            kind=BlockKind.CODE,
            text=sample,
            anchor=LineAnchor(start=121, end=132),
            lang="python",
        ),
        prose(
            " ".join(f"Sentence {index} covers the rollback." for index in range(120)),
            start=133,
            end=252,
        ),
    ]

    chunks = make_chunker().chunk(document(), blocks)

    assert len(chunks) > 1
    lines = [line for line in sample.splitlines() if line]
    for line in lines:
        carrying = [index for index, chunk in enumerate(chunks) if line in chunk.text]
        assert len(carrying) <= 1, (
            f"{line!r} appears in chunks {carrying}: a code line was copied into an adjacent "
            f"chunk as overlap"
        )


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


# --- hierarchy from a source record ----------------------------------------------------------


def _with_record(section_path: tuple[str, ...], **extra: object) -> Document:
    """A document carrying a validated source record declaring ``section_path``."""
    record = Provenance(
        source=SourceMetadata(
            title="Retry policy",
            canonical_uri="https://docs.example.test/pages/123456",
            section_path=section_path,
        )
    )
    return document(title="Retry policy", **{PROVENANCE_KEY: record.as_metadata_value()}, **extra)


def test_a_source_records_hierarchy_reaches_the_breadcrumb() -> None:
    """Chunk-level section citations: the document's place in its source, plus the chunk's own.

    This is the whole of what propagates from a document's source record into a chunk, and it
    propagates as *text the embedder reads* rather than as a copy on the chunk row. A section
    titled "Configuration" is unretrievable without knowing what it configures, and for a
    mirrored page the answer to "what" lives in the manifest rather than anywhere in the bytes.

    The two halves stay distinguishable: ``Engineering > Runbooks`` is where the *document* sits
    at its source, and ``Retry policy > Timeouts`` is where this *passage* sits inside the
    document. Concatenated for the embedder, kept apart in the record and on the chunk.
    """
    chunks = make_chunker().chunk(
        _with_record(("Engineering", "Runbooks")),
        [prose("Twice, with backoff.", path=("Retry policy", "Timeouts"))],
    )

    assert chunks[0].embed_text.startswith("Engineering > Runbooks > Retry policy > Timeouts")
    assert chunks[0].heading_path == ("Retry policy", "Timeouts"), (
        "the chunk keeps its own position and does not absorb the document's"
    )
    assert PROVENANCE_KEY not in chunks[0].metadata, (
        "the record is resolved through document_id at citation time, never copied per chunk — "
        "for a document of two hundred chunks that is one copy rather than two hundred"
    )


def test_a_validated_hierarchy_wins_over_the_untyped_ancestors_key() -> None:
    """Where both spellings are present, the checked one is used.

    ``section_path`` has been through depth, length and control-character validation and
    ``ancestors`` has been through none, so preferring ``ancestors`` would embed the weaker of a
    connector's own two answers into every vector — and a breadcrumb is not something anybody
    reads afterwards to check.
    """
    chunks = make_chunker().chunk(
        _with_record(("Engineering", "Runbooks"), ancestors=["Unvalidated", "Legacy"]),
        [prose("Twice, with backoff.", path=("Timeouts",))],
    )

    assert chunks[0].embed_text.startswith("Engineering > Runbooks")
    assert "Unvalidated" not in chunks[0].embed_text


def test_a_document_with_only_the_ancestors_key_keeps_working() -> None:
    """The older convention is untouched, so a connector that has not adopted the record is fine.

    Without this, adding the record would silently empty the breadcrumb of every document from
    every connector that fills in ``ancestors`` — and an empty breadcrumb is not a visible
    failure, it is a section nobody can retrieve.
    """
    chunks = make_chunker().chunk(
        document(title="Retry policy", ancestors=["Engineering", "Runbooks"]),
        [prose("Twice, with backoff.", path=("Timeouts",))],
    )

    assert chunks[0].embed_text.startswith("Engineering > Runbooks > Retry policy > Timeouts")


def test_a_record_with_no_hierarchy_falls_back_to_the_ancestors_key() -> None:
    """An empty ``section_path`` is not an instruction to discard a hierarchy that is there.

    A manifest may know a page's title and URL and nothing about where it sits. Treating the
    record's presence as authoritative about a field it never filled in would throw away the
    connector's own answer for no reason.
    """
    chunks = make_chunker().chunk(
        _with_record((), ancestors=["Engineering", "Runbooks"]),
        [prose("Twice, with backoff.", path=("Timeouts",))],
    )

    assert chunks[0].embed_text.startswith("Engineering > Runbooks")


def test_an_over_long_breadcrumb_is_elided_from_the_middle() -> None:
    """The two ends carry the most information: the outermost says which corpus and product
    area, the innermost says what this section is. Truncating the tail throws away the element
    that disambiguates "Configuration"."""
    parts = ["ENG", "Platform", "Auth Service", "Token Refresh", "Rotation", "Configuration"]
    rendered = render(parts, lambda text: len(text.split()), 6)
    assert rendered.startswith("ENG")
    assert rendered.endswith("Configuration")
    assert "…" in rendered


def test_a_single_identifier_breadcrumb_is_still_bounded() -> None:
    rendered = render(["synthetic_identifier_" * 40], len, 32)
    assert len(rendered) <= 32
    assert rendered.endswith("…")


# --- final rendered budget ---------------------------------------------------------------


def byte_counter() -> TokenCounter:
    """Deterministic exact synthetic tokenizer; every UTF-8 byte is one content token."""
    return TokenCounter("test/utf8-bytes-v1", lambda text: len(text.encode()), provisional=False)


@pytest.mark.parametrize(("maximum", "overlap"), [(512, 64), (768, 96)])
def test_every_supported_block_kind_obeys_the_final_exact_budget(
    maximum: int, overlap: int
) -> None:
    chunker = StructuralChunker(
        byte_counter(), max_tokens=maximum, overlap_tokens=overlap, breadcrumb_tokens=64
    )
    blocks = [
        ParsedBlock(
            kind=kind,
            text=("synthetic café 数据 payload. " * 80),
            anchor=LineAnchor(start=index + 1, end=index + 1),
            metadata=(
                {"rows": ["synthetic café 数据 payload. " * 80], "header_rows": 0}
                if kind is BlockKind.TABLE
                else {}
            ),
        )
        for index, kind in enumerate(
            (BlockKind.PROSE, BlockKind.LIST, BlockKind.TABLE, BlockKind.PANEL, BlockKind.MEDIA)
        )
    ]
    blocks.append(
        ParsedBlock(
            kind=BlockKind.CODE,
            text="synthetic_identifier_" * 180,
            anchor=LineAnchor(start=20, end=20),
        )
    )

    chunks = chunker.chunk(document(title="Synthetic budget"), blocks)

    assert chunks
    assert all(chunk.token_count == len(chunk.embed_text.encode()) for chunk in chunks)
    assert all(chunk.token_count <= maximum for chunk in chunks)
    assert any(chunk.metadata.get("hard_split") for chunk in chunks)


def test_overlap_is_shrunk_against_the_complete_rendered_input() -> None:
    chunker = StructuralChunker(
        byte_counter(), max_tokens=128, overlap_tokens=32, breadcrumb_tokens=24, min_tokens=1
    )
    shared = LineAnchor(start=1, end=1)
    blocks = [
        ParsedBlock(
            kind=BlockKind.PROSE,
            text=" ".join(f"Prior {index:02d}." for index in range(9)),
            anchor=shared,
            heading_path=("Config",),
        ),
        ParsedBlock(
            kind=BlockKind.PROSE,
            text=" ".join(f"Current {index:02d}." for index in range(8)),
            anchor=shared,
            heading_path=("Config",),
        ),
    ]

    chunks = chunker.chunk(document(title="Synthetic service"), blocks)

    assert len(chunks) == 2
    assert max(chunk.token_count for chunk in chunks) <= 128
    assert all(chunk.token_count == len(chunk.embed_text.encode()) for chunk in chunks)
    assert chunks[1].text.startswith("Prior 08.\n\nCurrent 00.")
    crumb = chunks[1].embed_text.split("\n\n", maxsplit=1)[0]
    next_overlap = f"{crumb}\n\nPrior 07. {chunks[1].text}"
    assert len(next_overlap.encode()) > 128


def test_one_oversized_code_line_is_losslessly_split_with_narrow_provenance() -> None:
    source = "const syntheticIdentifier = '数据-café-';" * 120
    chunker = StructuralChunker(
        byte_counter(), max_tokens=512, overlap_tokens=64, breadcrumb_tokens=64
    )
    block = ParsedBlock(
        kind=BlockKind.CODE,
        text=source,
        anchor=LineAnchor(start=37, end=37, symbol="syntheticIdentifier"),
        lang="typescript",
    )

    chunks = chunker.chunk(document(title="Synthetic module"), [block])

    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks) == source
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.metadata.get("hard_split_at") == "line" for chunk in chunks)
    assert all(
        chunk.anchor == LineAnchor(start=37, end=37, symbol="syntheticIdentifier")
        for chunk in chunks
    )


def test_mixed_code_lines_reconstruct_and_each_anchor_covers_only_its_payload() -> None:
    source = "short line\n" + ("x" * 260) + "\ntail line"
    chunker = StructuralChunker(
        byte_counter(), max_tokens=96, overlap_tokens=0, breadcrumb_tokens=16, min_tokens=1
    )
    block = ParsedBlock(
        kind=BlockKind.CODE,
        text=source,
        anchor=LineAnchor(start=10, end=12, symbol="synthetic"),
        lang="text",
    )

    chunks = chunker.chunk(document(title="Synthetic module"), [block])

    assert "".join(chunk.text for chunk in chunks) == source
    for chunk in chunks:
        represented: set[int] = set()
        if "short line" in chunk.text:
            represented.add(10)
        if "x" in chunk.text:
            represented.add(11)
        if "tail line" in chunk.text:
            represented.add(12)
        assert represented
        assert chunk.anchor == LineAnchor(
            start=min(represented), end=max(represented), symbol="synthetic"
        )


def test_post_middleware_finalization_recounts_and_refuses_growth() -> None:
    chunker = StructuralChunker(
        byte_counter(), max_tokens=96, overlap_tokens=0, breadcrumb_tokens=16
    )
    [chunk] = chunker.chunk(document(title="Synthetic"), [prose("bounded payload")])
    grown = chunk.model_copy(
        update={"embed_text": chunk.embed_text + " middleware", "token_count": 1}
    )

    [final] = chunker.finalize([grown])
    assert final.token_count == len(final.embed_text.encode())

    oversized = grown.model_copy(update={"embed_text": grown.embed_text + ("x" * 200)})
    with pytest.raises(ChunkingError, match="above the configured 96-token chunk budget"):
        chunker.finalize([oversized])


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


# --- table metadata and cell anchors ------------------------------------------------------


def test_adjacent_row_ranges_collapse_for_a_multi_column_table() -> None:
    """`_adjacent` compared one edge of each area, so it could only ever be True at one column.

    For `A1:C1` and `A2:C2` it took `left`'s *end* cell against `right`'s *start* cell — `C1`
    against `A2`, whose columns are `C` and `A` — and answered False. Row ranges therefore never
    collapsed for any table wider than one column: every row stayed its own area, and the
    anchor `_collapse_areas` exists to compact grew one comma-separated range per row.

    The negatives are the point of the parametrization: a gap in the rows, a shift in the
    columns, and a widened right-hand area must all still refuse.
    """
    from manicule.chunking.chunker import (  # noqa: PLC0415
        _adjacent,  # pyright: ignore[reportPrivateUsage] - the predicate under test
        _collapse_areas,  # pyright: ignore[reportPrivateUsage] - its only caller
    )

    assert _adjacent("A1:C1", "A2:C2"), "the case that could never be True"
    assert _adjacent("B3:D3", "B4:D4")
    assert _adjacent("A1", "A2"), "the single-column case, which already worked"

    assert not _adjacent("A1:C1", "A3:C3"), "a skipped row is not adjacent"
    assert not _adjacent("A1:C1", "B2:D2"), "shifted columns are a different area"
    assert not _adjacent("A1:C1", "A2:D2"), "a wider row is a different area"
    assert not _adjacent("x", "y"), "an unreadable reference collapses nothing"

    assert _collapse_areas(["A1:C1", "A2:C2", "A3:C3"]) == ["A1:C3"]


def test_an_unsplit_table_does_not_carry_the_parsers_row_list_onto_the_chunk() -> None:
    """`rows` on a chunk is the `[first, last]` pair, never the table's text.

    A table small enough not to be split copied its block metadata straight onto the unit, and
    `_group_metadata` then carried `rows` onto the chunk — so the chunk held the whole table a
    second time, beside its own text. It also made `rows` mean two different things depending
    on a size threshold: the pair that `_split_table` writes on a split part, and the list of
    row strings on an unsplit one. One key, two types, decided by whether the table fit.
    """
    rows = [f"| r{index} | value {index} |" for index in range(6)]
    block = ParsedBlock(
        kind=BlockKind.TABLE,
        text="\n".join(rows),
        anchor=LineAnchor(start=1, end=6),
        metadata=cast(
            "Metadata",
            {"rows": rows, "header_rows": 1, "row_refs": [f"A{i}:B{i}" for i in range(6)]},
        ),
    )

    chunks = make_chunker().chunk(document(), [block])

    assert len(chunks) == 1, "the fixture is meant to fit in one chunk"
    assert "rows" not in chunks[0].metadata, "the parser's row list must not ride onto the chunk"

    # And the contrast that makes `rows` mean one thing: split the same table and the key comes
    # back — as the `[first, last]` pair, which is what a chunk's `rows` is for.
    narrow = StructuralChunker(counter(), max_tokens=24, overlap_tokens=0, breadcrumb_tokens=4)
    parts = narrow.chunk(document(), [block])
    assert len(parts) > 1, "the fixture is meant to split at this budget"
    pairs = [part.metadata["rows"] for part in parts if "rows" in part.metadata]
    assert pairs, "a split part records which rows it holds"
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in pairs), (
        f"`rows` on a chunk is a [first, last] pair, got {pairs}"
    )


def test_memoizing_the_token_counter_changes_no_chunk() -> None:
    """The counter remembers answers, and that must be invisible in the output.

    `TokenCounter` re-ran the tokenizer over the whole string every time it was asked, and the
    chunker asks the same question many times: packing a group, checking whether a candidate
    fits, computing an overlap cap and hard-splitting an oversized unit all measure overlapping
    strings. Instrumented over one 40-block page — 478 calls for 81 distinct strings, 623,382
    characters counted against 139,241 distinct — so 78% of the work was a question already
    answered. With the real `cl100k_base` tokenizer a 60-block page chunks 1.9x faster.

    **The speedup is worth nothing if a boundary moves**, because `CHUNKER_VERSION` would then
    have to move with it and bill the corpus for a re-chunk and re-embed. So the property under
    test is equality, over randomized documents across every block kind rather than one fixture:
    a memo is only sound because a token count is a pure function of the text and the
    vocabulary, and this is what holds that claim to the output.
    """
    import random  # noqa: PLC0415 - only this derivation needs it

    from manicule.chunking.tokens import TokenCounter  # noqa: PLC0415

    kinds = [BlockKind.PROSE, BlockKind.LIST, BlockKind.CODE, BlockKind.TABLE, BlockKind.HEADING]

    def blocks(seed: int) -> list[ParsedBlock]:
        rng = random.Random(seed)  # noqa: S311 - a seeded fixture generator, not a key
        return [
            ParsedBlock(
                kind=rng.choice(kinds),
                text=" ".join(f"w{rng.randrange(999)}" for _ in range(rng.randrange(1, 400))),
                anchor=LineAnchor(start=index * 3 + 1, end=index * 3 + 3),
            )
            for index in range(rng.randrange(3, 30))
        ]

    class Unmemoized(TokenCounter):
        """The counter with its memo defeated, for the comparison to mean anything.

        Overriding `_remember` rather than poking `_memo_chars`: the budget check resets the
        counter after it clears, so a doctored value disables one store and nothing more. The
        first version of this test did exactly that and compared the memo against itself —
        both arms measured 180 calls where the real difference is 658 against 180.
        """

        @override
        def _remember(self, text: str, raw: int) -> None:
            return

    def chunked(seed: int, *, memo: bool) -> list[tuple[str, str, int, int]]:
        counter = (TokenCounter if memo else Unmemoized)(
            "whitespace", lambda text: max(1, len(text.split())), provisional=False
        )
        chunker = StructuralChunker(counter, max_tokens=128, overlap_tokens=16, breadcrumb_tokens=8)
        return [
            (chunk.id, chunk.text, chunk.position, chunk.token_count)
            for chunk in chunker.chunk(document(), blocks(seed))
        ]

    for seed in range(200):
        assert chunked(seed, memo=True) == chunked(seed, memo=False), (
            f"document {seed} chunks differently with the counter's memo; the memo must be "
            f"invisible in the output or CHUNKER_VERSION has to move with it"
        )


def test_packing_lines_does_not_grow_quadratically_with_the_budget() -> None:
    """A code block must not hand the tokenizer work that squares with the chunk budget.

    `_split_lines` rebuilt the whole accumulated prefix and counted it again for every line, so
    the cost was quadratic in the number of lines that fit *one chunk* — which is set by
    `max_tokens`. The block size only multiplies it.

    That is the axis, and getting it wrong is why the first version of this test passed against
    the defect. Measured over a 2,000-line file, tokenized characters as a multiple of the
    block:

        max_tokens   128     256     512    1024
        before      24.0x   44.5x   87.1x  169.9x     <- doubles with the budget
        after        8.7x    9.3x    9.6x   10.1x     <- flat

    At the shipped budget of 512 that is nine times the tokenizer work, on the stage every
    document of every ingest passes through.

    Asserted as a ratio that does not grow, never as a time or an absolute count: a wall-clock
    bound is not stable on a loaded runner, and an absolute would need editing whenever the
    packing changed for a good reason. What must never come back is the shape.
    """
    from manicule.chunking.tokens import TokenCounter  # noqa: PLC0415

    def tokenized_per_character(budget: int) -> float:
        counted: list[int] = []

        def count(text: str) -> int:
            counted.append(len(text))
            return max(1, len(text.split()))

        block = ParsedBlock(
            kind=BlockKind.CODE,
            text="\n".join(f"line_{index} = compute({index})" for index in range(2000)),
            anchor=LineAnchor(start=1, end=2000),
            lang="python",
        )
        StructuralChunker(
            TokenCounter("whitespace", count, provisional=False),
            max_tokens=budget,
            overlap_tokens=0,
            breadcrumb_tokens=8,
        ).chunk(document(), [block])
        return sum(counted) / len(block.text)

    narrow, wide = tokenized_per_character(128), tokenized_per_character(1024)

    # Eight times the budget. Quadratic multiplies the ratio by about eight — it did, 24x to
    # 170x. Bisection leaves it flat. Two is generous room for the search's own overhead.
    assert wide < narrow * 2, (
        f"tokenized {narrow:.1f}x the block at max_tokens=128 and {wide:.1f}x at 1024: the work "
        f"per character is growing with the budget, which is the quadratic pack coming back"
    )
