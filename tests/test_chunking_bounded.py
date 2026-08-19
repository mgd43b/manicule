"""What the chunker is allowed to hand a tokenizer, however large the block is.

A parser can emit one block of several megabytes — a page whose whole body is a single
element, a generated file with no interior structure — and the chunker's job is to cut it
into 512-token pieces. Cutting it is cheap; *measuring* it is not, and the two searches that
find each cut used to bisect from ``len(text)``, asking the tokenizer about half a megabyte
to locate a boundary two thousand characters in. Their callers peel one budget at a time, so
the search reran over the whole shrinking tail: O(n^2) tokenizer work in one block, which is
enough to hold a connector's ordered settlement on a single document for a quarter of an
hour while the process looks healthy and busy.

**These assert sizes, not seconds.** An elapsed-time threshold for this would be a test that
fails on a loaded CI runner and passes on a fast one that reintroduced the defect, so the
counter here records the length of every string it is given and the evidence is structural.
The opt-in benchmark in ``tests/benchmarks`` is where wall-clock belongs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from manicule.chunking import MAX_TOKENS, StructuralChunker, TokenCounter
from manicule.chunking import chunker as chunker_module
from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus, ParsedBlock

MIB = 1024 * 1024

BUDGET = MAX_TOKENS - chunker_module.BREADCRUMB_TOKENS


def characters_per_token(text: str) -> int:
    """A stand-in vocabulary at four characters per token, which is roughly BGE-M3 on prose.

    **Not whitespace words, and the difference is the whole test.** The rest of this suite
    counts words, which is exact, free and reproduces real boundaries — but it reports *one
    token* for a megabyte with no space in it, so under it every pathological shape below
    fits a 512-token budget in one chunk and the searches this file exists to bound are
    never entered. A rate per character says what a real tokenizer says about a megabyte of
    identifier: that it is hundreds of thousands of tokens and has to be cut.
    """
    return max(1, len(text) // 4)


class Recorder:
    """A token counter that remembers how much text it was asked about.

    A stand-in rather than a real vocabulary: the question here is what reaches the
    tokenizer, and a counter that is exact and free answers it without a model download.
    """

    def __init__(self, rate: Callable[[str], int] | None = None) -> None:
        self.sizes: list[int] = []
        self._rate = rate or characters_per_token

    def __call__(self, text: str) -> int:
        self.sizes.append(len(text))
        return self._rate(text)

    @property
    def largest(self) -> int:
        return max(self.sizes)

    @property
    def total(self) -> int:
        return sum(self.sizes)


def chunker_recording(recorder: Recorder, **overrides: object) -> StructuralChunker:
    return StructuralChunker(
        TokenCounter("test/recording", recorder, provisional=False),
        **overrides,  # pyright: ignore[reportArgumentType] - test-only pass-through
    )


def document() -> Document:
    return Document(
        id="doc-1",
        source="fixtures",
        source_id="s1",
        uri="doc",
        title="Synthetic Large Block",
        content_hash="h",
        media_type="text/plain",
        status=DocumentStatus.PARSED,
    )


SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. "


def sentences_of(size: int) -> str:
    """One paragraph, many ordinary sentences, no blank line anywhere."""
    return (SENTENCE * (size // len(SENTENCE) + 1))[:size]


def newline_free_of(size: int) -> str:
    """An identifier-like run with no sentence, word or line boundary to cut at."""
    return ("abcdefghijklmnopqrstuvwxyz0123456789" * (size // 36 + 1))[:size]


def paragraphs_of(size: int) -> str:
    body = sentences_of(size)
    return "\n\n".join(body[index : index + 4000] for index in range(0, len(body), 4000))


def multibyte_of(size: int) -> str:
    """Text whose every character is multiple bytes, and which no sentence rule splits.

    The boundary rule wants a terminator followed by something that starts a sentence in the
    Latin sense, so this reaches the character-level fallback — which is the path that has to
    cut without ever landing between a scalar's bytes.
    """
    unit = "日本語のテキストです。これはテストのための文章です。"
    return (unit * (size // len(unit) + 1))[:size]


def code_of(size: int) -> str:
    line = "    result = compute(alpha, beta, gamma, delta) + offset - correction\n"
    return (line * (size // len(line) + 1))[:size]


SHAPES: dict[str, tuple[Callable[[int], str], BlockKind]] = {
    "one enormous paragraph": (sentences_of, BlockKind.PROSE),
    "newline-free identifier": (newline_free_of, BlockKind.PROSE),
    "many ordinary sentences in one block": (paragraphs_of, BlockKind.PROSE),
    "multibyte": (multibyte_of, BlockKind.PROSE),
    "one enormous code block": (code_of, BlockKind.CODE),
}


def block_of(shape: str, size: int) -> ParsedBlock:
    build, kind = SHAPES[shape]
    text = build(size)
    end = len(text.splitlines()) or 1
    return ParsedBlock(kind=kind, text=text, anchor=LineAnchor(start=1, end=end))


def chunk_shape(shape: str, size: int, **overrides: object) -> tuple[list[Chunk], Recorder]:
    recorder = Recorder()
    chunks = chunker_recording(recorder, **overrides).chunk(document(), [block_of(shape, size)])
    return chunks, recorder


def without_whitespace(text: str) -> str:
    """Every character that carries content, in order, with the layout discarded.

    The comparison a lossless-split assertion can actually make across all five shapes.
    Paragraph and sentence splitting is a boundary finder that strips its pieces and rejoins
    them with a single space — normalization that predates this change and is what
    ``docs/parsing.md`` §4.2 asks for — so an assertion on exact bytes would be testing that
    rule rather than this one. Dropping whitespace from both sides leaves the question this
    file is asking: did a character of the block fail to reach a chunk?
    """
    return "".join(text.split())


@pytest.mark.parametrize("shape", list(SHAPES))
def test_no_tokenizer_call_receives_the_whole_multi_megabyte_block(shape: str) -> None:
    """The reproduction, as a size assertion.

    Before the bounded searches this failed on every shape: ``_to_units`` counted the block
    to find out whether it fit, ``_split_prose`` counted it again to build a template, and
    the prefix searches bisected from its full length. A 2 MiB block produced 2 MiB
    tokenizer calls and 1.28 GB of tokenizer input in total.
    """
    _, recorder = chunk_shape(shape, 2 * MIB)

    probe = MAX_TOKENS * chunker_module.PROBE_CHARS_PER_TOKEN
    assert recorder.largest <= 2 * probe, (
        f"the largest string handed to the tokenizer was {recorder.largest:,} characters "
        f"from a 2 MiB block. A search that starts at {probe:,} characters and doubles until "
        f"the budget is exceeded cannot need more than one doubling past its first probe, so "
        f"a larger call means something is measuring the block itself rather than a prefix "
        f"of it"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_largest_tokenizer_input_does_not_grow_with_the_block(shape: str) -> None:
    """The bound is on the *cut*, not on a fraction of the input.

    This is the assertion that a bisection from ``len(text)`` cannot pass however generous
    the constant above is made: there, the largest call is half the block and quadruples with
    it; here the block grows eightfold and the largest call does not move at all.
    """
    _, small = chunk_shape(shape, 256 * 1024)
    _, large = chunk_shape(shape, 2 * MIB)

    assert small.largest == large.largest, (
        f"the largest tokenizer input was {small.largest:,} characters for a 256 KiB block "
        f"and {large.largest:,} for a 2 MiB one. The longest prefix that fits a fixed token "
        f"budget is a fixed number of characters, so growing the block must not grow the "
        f"question asked about it"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
def test_total_tokenizer_work_grows_linearly_with_the_block(shape: str) -> None:
    """Bounding each call is not enough on its own.

    Every call could be small and the total still quadratic, which is exactly what the
    enclosing peel-one-budget-at-a-time loops did: each search was O(n) and there were O(n)
    of them. Doubling the block must roughly double the work, not quadruple it — measured at
    3.85x per doubling before this change, and 2.00x after.
    """
    _, small = chunk_shape(shape, 512 * 1024)
    _, large = chunk_shape(shape, 1024 * 1024)

    growth = large.total / small.total
    assert growth <= 2.4, (
        f"doubling the block multiplied tokenizer input by {growth:.2f}x "
        f"({small.total:,} -> {large.total:,} characters). Anything approaching 4x is the "
        f"quadratic search returning: a bounded call size hides it, because the cost is in "
        f"how many times the tail is re-measured"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_bounded_split_loses_no_text(shape: str) -> None:
    """Nothing is truncated, sampled or dropped on the way through the bounded path.

    Overlap is off, so the concatenation is the non-overlap payload and every character is
    expected exactly once. The character-level test below admits no normalization at all;
    this one covers the shapes whose splitting legitimately rewrites whitespace.
    """
    size = 512 * 1024
    chunks, _ = chunk_shape(shape, size, overlap_tokens=0)

    rebuilt = without_whitespace("".join(chunk.text for chunk in chunks))
    assert rebuilt == without_whitespace(block_of(shape, size).text)


@pytest.mark.parametrize("shape", ["newline-free identifier", "multibyte"])
def test_the_character_level_split_reconstructs_the_block_exactly(shape: str) -> None:
    """Byte-for-byte, for the shapes that reach the character-level cut.

    Neither shape offers a paragraph, sentence or line boundary, so every cut is made by the
    prefix search itself and no normalization stands between the input and the output. A
    multibyte block is here because a search that cut on bytes rather than on characters
    would round-trip Latin text perfectly and corrupt this.
    """
    size = 512 * 1024
    original = block_of(shape, size).text
    chunks, _ = chunk_shape(shape, size, overlap_tokens=0)

    assert "".join(chunk.text for chunk in chunks) == original


@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_chunk_still_satisfies_the_exact_budget(shape: str) -> None:
    """The bound on tokenizer input buys nothing if it costs the invariant it protects."""
    chunks, _ = chunk_shape(shape, 512 * 1024)

    exact = TokenCounter("test/recording", characters_per_token, provisional=False)
    oversized = [chunk.id for chunk in chunks if exact(chunk.embed_text) > MAX_TOKENS]
    assert not oversized, (
        f"{len(oversized)} chunk(s) exceed the {MAX_TOKENS}-token budget they claim. Past the "
        f"embedder's sequence length the text is dropped with no error raised, so the chunk "
        f"would be indexed as its opening tokens while still claiming all of its text"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("probe", [1, 3, 8, 64, 4096])
def test_the_probe_size_cannot_move_a_boundary(shape: str, probe: int) -> None:
    """Why :data:`PROBE_CHARS_PER_TOKEN` is not in the chunk fingerprint.

    A tuning constant that changed a boundary would have to be recorded in
    :class:`~manicule.core.fingerprints.ChunkFingerprint`, and every retune would then be a
    corpus-wide re-chunk and re-embed. It changes none, because the doubling continues until
    the predicate genuinely fails: the probe decides how many questions are asked and never
    which prefix is the answer. Run from a probe of one character to one of two megabytes.
    """
    size = 128 * 1024
    reference, _ = chunk_shape(shape, size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chunker_module, "PROBE_CHARS_PER_TOKEN", probe)
        tuned, _ = chunk_shape(shape, size)

    assert [chunk.model_dump() for chunk in tuned] == [chunk.model_dump() for chunk in reference], (
        f"a probe of {probe} characters per token produced different chunks from the default "
        f"of {chunker_module.PROBE_CHARS_PER_TOKEN}. The search is only allowed to be faster "
        f"or slower for a bad guess — if it can also be *different*, the constant belongs in "
        f"the fingerprint and this is a version bump"
    )


def rows_block(rows: Iterable[str], header_rows: int) -> ParsedBlock:
    listed = list(rows)
    return ParsedBlock(
        kind=BlockKind.TABLE,
        text="\n".join(listed),
        anchor=LineAnchor(start=1, end=len(listed)),
        metadata={"rows": listed, "header_rows": header_rows},  # pyright: ignore[reportArgumentType] - test values are JSON-shaped
    )


def test_one_enormous_table_row_stays_bounded_and_keeps_its_header() -> None:
    """A single row wider than the budget is the table shape of the same defect.

    It reaches the repeated-prefix search, which bisected from the row's full length too, so
    a spreadsheet cell holding a pasted document cost the same quadratic measurement.
    """
    recorder = Recorder()
    header = "id | description"
    wide = "1 | " + ("lorem ipsum dolor sit amet " * 40000)
    chunks = chunker_recording(recorder).chunk(document(), [rows_block([header, wide], 1)])

    probe = MAX_TOKENS * chunker_module.PROBE_CHARS_PER_TOKEN
    assert recorder.largest <= 2 * probe
    assert len(chunks) > 1
    assert all(header in chunk.text for chunk in chunks), (
        "every fragment of a split row keeps the parser-declared header, or the fragments "
        "that lose it are a grid of values with nothing to say what they are"
    )


def test_one_enormous_code_line_stays_bounded() -> None:
    """A minified bundle or a base64 blob: one line, no interior boundary, megabytes long."""
    recorder = Recorder()
    text = "import x\n" + ("A" * (2 * MIB)) + "\nprint(x)\n"
    block = ParsedBlock(
        kind=BlockKind.CODE,
        text=text,
        anchor=LineAnchor(start=1, end=3),
        lang="python",
    )
    chunks = chunker_recording(recorder).chunk(document(), [block])

    probe = MAX_TOKENS * chunker_module.PROBE_CHARS_PER_TOKEN
    assert recorder.largest <= 2 * probe
    assert "".join(chunk.text for chunk in chunks).count("A") == 2 * MIB, (
        "the oversized line is cut, never sampled: every character of it reaches a chunk"
    )


def whitespace_words(text: str) -> int:
    return len(text.split())


def one_token_per_character(text: str) -> int:
    return max(1, len(text))


@pytest.mark.parametrize(
    ("text", "rate"),
    [
        ("word " * 10000, whitespace_words),
        ("x" * 40000, characters_per_token),
        ("日本語" * 10000, one_token_per_character),
    ],
)
def test_the_bounded_search_finds_the_same_prefix_the_full_range_search_did(
    text: str, rate: Callable[[str], int]
) -> None:
    """The equivalence the no-bump decision rests on, checked directly rather than argued.

    Bisecting to a doubled ceiling and bisecting to ``len(text)`` return the same prefix
    whenever the predicate is monotone — which bisection already required to be correct at
    all. This exercises the primitive against a full-range bisection over counters with very
    different characters-per-token ratios.
    """
    recorder = Recorder(rate)
    chunker = chunker_recording(recorder)
    fits = chunker._fits_budget  # pyright: ignore[reportPrivateUsage] - the primitive under test

    probe_chars = chunker._probe_chars  # pyright: ignore[reportPrivateUsage] - as above
    bounded = chunker_module._search_ceiling(text, fits, probe_chars)  # pyright: ignore[reportPrivateUsage] - as above

    low, high, exhaustive = 1, len(text), 1
    while low <= high:
        middle = (low + high) // 2
        if fits(text[:middle]):
            exhaustive = middle
            low = middle + 1
        else:
            high = middle - 1

    assert exhaustive <= bounded, (
        f"the doubling search stopped at {bounded:,} characters but a prefix of "
        f"{exhaustive:,} still fits the budget, so the ceiling excluded the real answer"
    )
    assert chunker._longest_prefix_within_budget(text) == exhaustive  # pyright: ignore[reportPrivateUsage] - as above
