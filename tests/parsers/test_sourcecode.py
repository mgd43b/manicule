"""Code parsing: line anchors from the parse tree, and symbols that cannot move them.

Two failures shape this suite, and neither of them raises anything.

The first is an anchor one line out. It produces a citation that resolves, reads perfectly
and quotes the wrong code, so it is caught by the round trip rather than by review —
:func:`test_an_anchor_one_line_out_fails_the_round_trip` runs that exact defect past the same
harness and shows it failing, which is what makes the harness load-bearing here.

The second is a grammar that is missing on one machine and present on another. Falling back
to line-splitting when it is missing would give one repository two chunkings, and the two
would differ silently forever after. The guard is
:func:`test_a_missing_grammar_refuses_rather_than_line_splitting`, and beside it a check that
the refusal is not a ``ParseError`` — because a ``ParseError`` declines to the next parser in
the chain, which for source code is the plain-text parser, which line-splits.

**No test here needs network access.** The parse tests use the grammars this environment
actually has and skip per language, naming the language and the remedy, when it has none —
never a blanket skip, which would let the suite pass while testing nothing. The refusal tests
point the cache at an empty directory and the manifest at an address that refuses
connections, so a test that started downloading fails instead of passing slowly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast, override

import pytest
from tree_sitter import Node, Tree
from tree_sitter import Parser as TSParser

from manicule.chunking import StructuralChunker
from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import aclose, parsing, read_blocks
from manicule.parsers import grammars
from manicule.parsers.sourcecode import SourceCodeConfig, SourceCodeParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, document_for, raw_from, raw_of

MEDIA_TYPES_BY_SUFFIX = {
    ".py": "text/x-python",
    ".rs": "text/x-rust",
    ".sh": "text/x-shellscript",
    ".sql": "application/sql",
    ".ts": "text/x-typescript",
}
"""How this suite routes its own fixtures. Deliberately not derived from the filename by the
parser: routing is the pipeline's job, and a parser that sniffed extensions would be claiming
documents nobody sent it."""

FIXTURE_LANGUAGES = ("bash", "python", "rust", "sql", "typescript")
"""Every language the fixture corpus is written in."""

UNREACHABLE_MANIFEST = "http://127.0.0.1:9/manicule-tests-must-not-download.json"
"""Discard port on the loopback interface: a fetch through it refuses at once rather than
hanging, so a test that starts downloading fails rather than stalling CI."""


@pytest.fixture(scope="session", autouse=True)
def seed_grammars() -> None:
    """Pre-seed the declared grammars once, exactly as ``manicule init`` does.

    Best effort: a machine with no route to the manifest still runs this suite, and the parse
    tests skip per language rather than pretending to have run.
    """
    with suppress(grammars.GrammarFetchError):
        grammars.prefetch(FIXTURE_LANGUAGES)


@pytest.fixture(autouse=True)
def default_grammar_configuration() -> Iterator[None]:
    """Start and finish every test on the pack's default cache and manifest.

    The pack keeps one registry per process, so a test that points it somewhere else would
    otherwise decide what every test after it can see.
    """
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)
    yield
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


def require_grammar(*languages: str) -> None:
    """Skip precisely, naming what did not run and how to make it run.

    Precisely, because a blanket skip on "tree-sitter is unavailable" would let this whole
    suite pass green on a machine where nothing was parsed at all.
    """
    absent = grammars.missing_grammars(languages)
    if absent:
        pytest.skip(
            f"no grammar cached for {', '.join(absent)} in {grammars.cache_directory()}, so "
            f"the assertions for {', '.join(absent)} did not run — fetch them with "
            f"`manicule doctor --fix`, or point the cache at a pre-seeded directory"
        )


def parser(**overrides: object) -> SourceCodeParser:
    """A parser on the declared set, with configuration overridden per test."""
    return SourceCodeParser(SourceCodeConfig(**overrides))  # pyright: ignore[reportArgumentType] - test-only keyword pass-through


def fixtures(corpus: Path) -> list[Path]:
    return sorted((corpus / "sourcecode").iterdir())


def fixture(corpus: Path, name: str) -> RawDocument:
    path = corpus / "sourcecode" / name
    return raw_from(path, MEDIA_TYPES_BY_SUFFIX[path.suffix])


def _no_tags_query(language: str) -> None:
    """Stand in for a language the pack has no tags query for."""
    del language


async def anchors(raw: RawDocument, **overrides: object) -> list[LineAnchor]:
    """Every block's anchor, which is all most of these tests need."""
    blocks = await read_blocks(parser(**overrides), raw)
    return [block.anchor for block in blocks if isinstance(block.anchor, LineAnchor)]


# --- the corpus ----------------------------------------------------------------------------


async def test_every_fixture_round_trips_and_the_corpus_stays_within_its_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The whole obligation, over every fixture, blocks and chunks alike.

    The corpus floor is the part worth explaining: without it, a suite whose fixtures stopped
    being generated would divide zero by zero and pass every ratio it declares.
    """
    require_grammar(*FIXTURE_LANGUAGES)
    raws = [
        raw_from(path, MEDIA_TYPES_BY_SUFFIX[path.suffix])
        for path in fixtures(corpus)
        if path.name != "hostile-malformed-utf8.py"
    ]

    await check_corpus(parser(), raws, chunker=chunker, min_blocks=300)


async def test_blocks_never_claim_a_line_another_block_already_claimed(corpus: Path) -> None:
    """Overlapping blocks quote each other.

    A construct that closes on the line the next one opens on is the ordinary way this
    happens — a ``;`` closing a statement is a sibling of what follows it — and the result is
    two citations each containing a fragment of the other's text.
    """
    require_grammar(*FIXTURE_LANGUAGES)
    for path in fixtures(corpus):
        if path.name == "hostile-malformed-utf8.py":
            continue
        found = await anchors(raw_from(path, MEDIA_TYPES_BY_SUFFIX[path.suffix]))
        for earlier, later in pairwise(found):
            assert earlier.end < later.start, f"{path.name}: {earlier} overlaps {later}"


# --- the refusals --------------------------------------------------------------------------


async def test_a_missing_grammar_refuses_rather_than_line_splitting(
    tmp_path: Path, corpus: Path
) -> None:
    """The corpus-consistency guard, stated as the thing that must not happen.

    A fallback splitter here would give a machine with no grammar a set of chunks, a machine
    with one a different set, and no signal anywhere that the two corpora disagree. So the
    document stops: it is stored, visible, and re-indexable the moment the grammar arrives.
    """
    refusing = parser(grammar_cache_dir=tmp_path, grammar_manifest_url=UNREACHABLE_MANIFEST)

    with pytest.raises(grammars.GrammarUnavailableError) as raised:
        await read_blocks(refusing, fixture(corpus, "typical-tokens.py"))

    assert raised.value.reason == "grammar unavailable: python — run manicule doctor --fix"


async def test_the_grammar_refusal_is_not_a_decline_to_the_next_parser(
    tmp_path: Path, corpus: Path
) -> None:
    """A decline and a refusal need different answers.

    ``ParseError`` is a decline: the fallback chain gives the document to the next parser,
    and the tail of that chain is the plain-text parser, which line-splits. So the refusal
    being a ``ParseError`` would reintroduce the fallback through the routing layer while
    every direct test of the parser still passed.
    """
    refusing = parser(grammar_cache_dir=tmp_path, grammar_manifest_url=UNREACHABLE_MANIFEST)

    with pytest.raises(grammars.GrammarUnavailableError) as raised:
        await read_blocks(refusing, fixture(corpus, "typical-store.rs"))

    assert not isinstance(raised.value, ParseError)


async def test_undecodable_bytes_are_declined_so_the_next_parser_gets_a_turn(
    corpus: Path,
) -> None:
    """Bytes that are not text are somebody else's document, and declining says so.

    Indexing them anyway would store a page of replacement characters, which matches queries
    by accident and cites nothing.
    """
    require_grammar("python")

    with pytest.raises(ParseError):
        await read_blocks(parser(), fixture(corpus, "hostile-malformed-utf8.py"))


async def test_a_media_type_the_parser_does_not_declare_is_declined() -> None:
    """A parser that accepted anything would make the fallback chain unreachable."""
    require_grammar("python")

    with pytest.raises(ParseError) as raised:
        await read_blocks(parser(), raw_of("# hello\n", "text/markdown"))

    assert "text/markdown" in str(raised.value)


def test_the_declared_media_types_follow_the_declared_languages() -> None:
    """Routing reads the declaration, so the two cannot be allowed to drift.

    A parser declaring more media types than languages claims documents it will refuse; one
    declaring fewer leaves a configured language unreachable.
    """
    narrow = parser(languages=("python", "rust"))

    assert narrow.media_types == {"text/x-python", "text/x-rust"}
    assert narrow.languages == ("python", "rust")


# --- the round trip, and the proof it catches something --------------------------------------


class _OffByOneParser(SourceCodeParser):
    """Emits every anchor one line further down than the text it claims.

    The exact defect the round trip exists to catch: nothing raises, every citation is
    well-formed, and every one of them quotes the line below the one it means.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            anchor = block.anchor
            if not isinstance(anchor, LineAnchor):
                yield block
                continue
            yield block.model_copy(
                update={"anchor": anchor.model_copy(update={"start": anchor.start + 1})}
            )


async def test_an_anchor_one_line_out_fails_the_round_trip(corpus: Path) -> None:
    """Disabling the guard has to turn the suite red, or the guard is decoration.

    Everything else in this file asserts that the parser is right. This asserts that being
    wrong in the least visible way available is detected.
    """
    require_grammar("python")
    raw = fixture(corpus, "typical-tokens.py")

    with pytest.raises(AssertionError):
        await assert_round_trip(_OffByOneParser(SourceCodeConfig()), raw, fixture=raw.uri)


async def test_resolve_reads_only_the_bytes_it_is_handed(corpus: Path) -> None:
    """An anchor read back from storage months later must resolve as it did when written.

    A ``resolve`` that depended on state left by ``parse`` would work in every test that
    parsed first and fail in the citation path, which is the only place it matters.
    """
    require_grammar("python")
    raw = fixture(corpus, "typical-tokens.py")
    found = await anchors(raw)

    fresh = parser()
    resolved = await fresh.resolve(found[1], raw)

    assert resolved is not None
    assert resolved.splitlines()[0].startswith("class Token")


async def test_an_anchor_that_does_not_fit_these_bytes_resolves_to_nothing() -> None:
    """``None`` rather than a clamped range.

    An anchor naming lines the document does not have has diverged from it, and returning
    the last few lines instead would hide that behind a plausible quotation.
    """
    require_grammar("python")
    raw = raw_of("x = 1\n", "text/x-python")

    assert await parser().resolve(LineAnchor(start=40, end=50), raw) is None


# --- symbols -------------------------------------------------------------------------------


async def test_a_symbol_never_moves_a_chunk_boundary(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundaries come from the parse tree; the tags query only supplies a name.

    This is what makes a later pack release safe. A language that gains a tags query gets
    better symbols and must not get different chunks, because different chunks mean stored
    embeddings that no longer describe the text they claim.
    """
    require_grammar("python")
    raw = fixture(corpus, "hard-nested-classes.py")
    with_symbols = await anchors(raw, max_block_chars=240)

    monkeypatch.setattr(grammars, "tags_query", _no_tags_query)
    monkeypatch.setattr(grammars, "NODE_TYPE_DEFINITIONS", {})
    without_symbols = await anchors(raw, max_block_chars=240)

    assert [(anchor.start, anchor.end) for anchor in with_symbols] == [
        (anchor.start, anchor.end) for anchor in without_symbols
    ]
    assert {anchor.symbol for anchor in without_symbols} == {None}
    assert any(anchor.symbol for anchor in with_symbols)


async def test_the_symbol_names_the_innermost_definition_enclosing_the_block(
    corpus: Path,
) -> None:
    """``symbol`` is what reaches the embedder through the breadcrumb.

    A chunk of a method has to embed under its own name and its class's, or a query for
    "token refresh" finds the README instead of the method.
    """
    require_grammar("python")
    found = await anchors(fixture(corpus, "hard-nested-classes.py"), max_block_chars=240)

    assert "Outer.Middle.Inner.deepest" in {anchor.symbol for anchor in found}


async def test_a_rust_symbol_is_written_with_rusts_own_scope_separator(corpus: Path) -> None:
    """``Anchor.render`` for Rust reads as a mistake to anyone who writes Rust.

    The symbol is read by people who know the language, so it is joined with the separator
    that language uses.
    """
    require_grammar("rust")
    found = await anchors(fixture(corpus, "hard-scoped-modules.rs"), max_block_chars=48)

    assert "ledger::Anchor::render" in {anchor.symbol for anchor in found}


async def test_a_block_spanning_several_definitions_is_named_after_none_of_them(
    corpus: Path,
) -> None:
    """A run of imports belongs to no definition, and saying it belongs to the last one
    would put a wrong name in the breadcrumb, which reaches the embedder."""
    require_grammar("python")
    found = await anchors(fixture(corpus, "typical-tokens.py"))

    assert found[0].symbol is None


async def test_a_language_named_only_by_the_in_repo_table_still_gets_its_symbol(
    corpus: Path,
) -> None:
    """Bash has no tags query in the pack, and its functions are still worth naming."""
    require_grammar("bash")
    found = await anchors(fixture(corpus, "table-symbols.sh"))

    assert "prepare_workspace" in {anchor.symbol for anchor in found}


async def test_a_language_with_no_symbol_source_still_anchors_exactly(corpus: Path) -> None:
    """``symbol=None`` with exact lines is honest and cites correctly.

    SQL has neither a tags query nor an entry in the in-repo table. The degradation has to be
    the symbol alone — a language that lost its line anchors too would be uncitable, and a
    language that invented a symbol would be worse.
    """
    require_grammar("sql")
    raw = fixture(corpus, "absent-symbols.sql")

    report = await check_fixture(parser(), raw)
    found = await anchors(raw)

    assert report.blocks > 0
    assert {anchor.symbol for anchor in found} == {None}


async def test_a_decorated_definition_is_named_by_the_definition_not_the_decorator(
    corpus: Path,
) -> None:
    """The block covers a node that is not itself a definition.

    A decorator run and the class it decorates are one node, and that node has no name; the
    naive answer is no symbol at all for every decorated definition in the corpus.
    """
    require_grammar("python")
    found = await anchors(fixture(corpus, "hard-decorated-handlers.py"))

    assert {"archive_handler", "RestoreHandler"} <= {anchor.symbol for anchor in found}


async def test_an_exported_typescript_definition_is_named_by_what_it_exports(
    corpus: Path,
) -> None:
    """``export`` wraps the declaration, and TypeScript's own tags query names neither.

    The pack exposes only the TypeScript half of the upstream query pair — signatures,
    interfaces, abstract classes — so an ordinary exported class is unnamed unless this
    repository names it.
    """
    require_grammar("typescript")
    found = await anchors(fixture(corpus, "typical-sessions.ts"))

    assert {"SessionCache", "openSession"} <= {anchor.symbol for anchor in found}


async def test_the_heading_path_carries_the_enclosing_symbol_chain(corpus: Path) -> None:
    """The breadcrumb is built from ``heading_path``, so the chain has to be in it.

    ``src/auth/token.py > TokenStore > refresh`` is what makes a chunk of a method findable;
    a flattened symbol string in ``anchor`` alone never reaches the embedder.
    """
    require_grammar("python")
    blocks = await read_blocks(
        parser(max_block_chars=240), fixture(corpus, "hard-nested-classes.py")
    )

    assert ("Outer", "Middle", "Inner", "deepest") in {block.heading_path for block in blocks}


# --- splitting -----------------------------------------------------------------------------


async def test_a_definition_too_large_for_one_block_splits_at_node_boundaries(
    corpus: Path,
) -> None:
    """Splitting code costs nothing in provenance and gains something.

    Each part gets its own real line anchor naming the definition it covers, so a long
    function becomes chunks that each cite the code they contain rather than one chunk that
    cites all of it.
    """
    require_grammar("python")
    raw = fixture(corpus, "hard-oversized-function.py")
    found = await anchors(raw)

    parts = [anchor for anchor in found if anchor.symbol == "accumulate_readings"]
    assert len(parts) > 1
    for earlier, later in pairwise(parts):
        assert later.start == earlier.end + 1, "the parts must tile the definition, not sample it"


async def test_a_long_string_literal_is_never_cut_in_half(corpus: Path) -> None:
    """The one splitting rule stated as a prohibition rather than a preference.

    Python's grammar gives a string children that begin on lines of their own, so taking the
    next node boundary down inside one looks entirely reasonable — and produces two blocks
    that are each syntactically nothing, one of which quotes text with no opening delimiter.
    """
    require_grammar("python")
    blocks = await read_blocks(parser(max_block_chars=200), fixture(corpus, "hard-long-string.py"))

    for block in blocks:
        assert block.text.count('"""') % 2 == 0, f"a triple-quoted string was cut: {block.anchor}"


async def test_a_block_over_budget_is_emitted_whole_rather_than_cut_unsafely(
    corpus: Path,
) -> None:
    """There is a floor, and it is the token.

    A construct with no sibling boundary anywhere inside it — a minified line, one enormous
    literal — is emitted over budget. The chunker measures it again with the real tokenizer;
    what it must never receive is a block cut through the middle of a token.
    """
    require_grammar("python")
    blocks = await read_blocks(parser(max_block_chars=200), fixture(corpus, "hard-long-string.py"))

    assert any(len(block.text) > 200 for block in blocks)


async def test_chunking_a_split_definition_keeps_every_part_citable(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """A chunk's anchor is a merge of its blocks', and a merge is where a location goes
    wrong with nothing raising."""
    require_grammar("python")
    raw = fixture(corpus, "hard-oversized-function.py")
    built = parser()
    blocks = await read_blocks(built, raw)

    chunks = chunker.chunk(document_for(raw), blocks)

    assert chunks
    for chunk in chunks:
        assert isinstance(chunk.anchor, LineAnchor)
        assert await built.resolve(chunk.anchor, raw) is not None


# --- the awkward files -----------------------------------------------------------------------


async def test_code_between_definitions_is_emitted_rather_than_dropped(corpus: Path) -> None:
    """Imports and module-level statements are content.

    A parser that emitted only definitions would silently drop the part of a file that says
    what it depends on and how it is configured — data loss that looks like tidiness.
    """
    require_grammar("python")
    blocks = await read_blocks(parser(), fixture(corpus, "typical-tokens.py"))

    assert "import json" in blocks[0].text
    assert all(block.kind is BlockKind.CODE for block in blocks)


async def test_a_file_of_nothing_but_comments_still_produces_a_block(corpus: Path) -> None:
    """Commentary is prose worth citing, and a file of it must not vanish."""
    require_grammar("python")
    blocks = await read_blocks(parser(), fixture(corpus, "degenerate-comments-only.py"))

    assert len(blocks) == 1
    assert blocks[0].text.startswith("# This file holds nothing but commentary.")


async def test_an_empty_file_produces_no_blocks_rather_than_an_empty_one(
    corpus: Path,
) -> None:
    """A placeholder block would be retrievable and would cite nothing."""
    require_grammar("python")

    assert await read_blocks(parser(), fixture(corpus, "degenerate-empty.py")) == []


async def test_a_file_with_no_trailing_newline_anchors_its_last_line(corpus: Path) -> None:
    """The last line of a file with no terminator is where off-by-one lives."""
    require_grammar("python")
    raw = fixture(corpus, "degenerate-no-trailing-newline.py")
    built = parser()
    blocks = await read_blocks(built, raw)

    last = blocks[-1]
    assert isinstance(last.anchor, LineAnchor)
    assert last.text.rstrip().endswith("return FIRST_CONSTANT")
    assert await built.resolve(last.anchor, raw) == last.text


async def test_a_syntactically_broken_file_still_produces_anchored_blocks(
    corpus: Path,
) -> None:
    """Every repository contains one, and refusing them would make the parser useless.

    The grammar produces ERROR nodes rather than failing, and an ERROR node still reports
    where it is — so the blocks are fewer and coarser, and every one of them still cites
    exactly the lines it holds.
    """
    require_grammar("python")
    raw = fixture(corpus, "hostile-broken-syntax.py")

    report = await check_fixture(parser(), raw)

    assert report.blocks > 0
    assert report.unlocated == 0


async def test_astral_plane_text_does_not_shift_a_line_number(corpus: Path) -> None:
    """Four bytes, one character, and the place a byte count masquerading as a character
    count comes apart — in an identifier and in a string literal alike."""
    require_grammar("python")
    raw = fixture(corpus, "hostile-astral.py")
    built = parser()
    blocks = await read_blocks(built, raw)

    for block in blocks:
        assert await built.resolve(block.anchor, raw) == block.text
    assert any(
        "\U00020022" in (block.anchor.symbol or "")
        for block in blocks
        if isinstance(block.anchor, LineAnchor)
    )


async def test_a_file_past_the_size_cap_parses_without_special_handling(corpus: Path) -> None:
    """The streaming path, and the only fixture allowed past the corpus size cap.

    Worth its own test because everything that goes wrong at this size goes wrong quietly:
    quadratic measurement turns a parse into a hang, and the extension's own object handling
    stops being forgiving.
    """
    require_grammar("python")
    raw = fixture(corpus, "module-large.py")

    found = await anchors(raw)

    assert len(found) > 100
    assert found[-1].symbol == "stage_109"


# --- what the chunk fingerprint records ------------------------------------------------------


def test_the_parser_reports_a_grammar_version_for_every_declared_language() -> None:
    """``ChunkFingerprint.grammars`` is per language so a grammar bump invalidates the
    documents in that language and leaves the rest of the corpus alone."""
    built = parser(languages=("python", "rust"))

    assert built.grammar_versions() == {
        "python": grammars.pack_version(),
        "rust": grammars.pack_version(),
    }


def test_the_declared_language_set_is_the_same_value_however_it_was_written() -> None:
    """It feeds the chunk fingerprint, so a reordered configuration file must not read as a
    different chunking process."""
    written_one_way = parser(languages=("rust", "python"))
    written_another = parser(languages=("python", "rust"))

    assert written_one_way.grammar_versions() == written_another.grammar_versions()
    assert written_one_way.languages == written_another.languages


def test_a_typo_in_the_declared_set_fails_when_the_parser_is_built() -> None:
    """Not at the first document that needs it, by which point a corpus has been indexed
    differently here than on another machine."""
    with pytest.raises(Exception, match="csharp"):
        parser(languages=("python", "c_sharp"))


# --- the block stream's lifetime ---------------------------------------------------------


@dataclass
class TreeLedger:
    """Counts parse trees made and released, so a stranded one is a number not a hunch."""

    made: int = 0
    released: int = 0

    @property
    def held(self) -> int:
        return self.made - self.released


class _TrackedTree:
    """A parse tree that reports its own release.

    A ``tree_sitter.Tree`` cannot be weak-referenced, so its lifetime is observed by wrapping
    it. Only ``root_node`` is forwarded because that is the only member the parser reaches
    for — if that stops being true this fails loudly rather than under-reporting.
    """

    def __init__(self, tree: Tree, ledger: TreeLedger) -> None:
        self._tree = tree
        self._ledger = ledger
        ledger.made += 1

    @property
    def root_node(self) -> Node:
        return self._tree.root_node

    def __del__(self) -> None:
        self._ledger.released += 1


class _TrackedGrammarParser:
    """Stands in for the grammar's parser, wrapping each tree it produces."""

    def __init__(self, parser: TSParser, ledger: TreeLedger) -> None:
        self._parser = parser
        self._ledger = ledger

    def parse(self, data: bytes) -> Tree:
        # Cast because the spy is a Tree only as far as this parser is concerned: it forwards
        # the one member the parser uses, which is what makes the lifetime observable at all.
        return cast("Tree", _TrackedTree(self._parser.parse(data), self._ledger))


def _track_trees(monkeypatch: pytest.MonkeyPatch) -> TreeLedger:
    """Make every parse tree this test creates report when it is released."""
    ledger = TreeLedger()
    real = grammars.load_parser

    def tracked(language: str) -> TSParser:
        return cast("TSParser", _TrackedGrammarParser(real(language), ledger))

    monkeypatch.setattr(grammars, "load_parser", tracked)
    return ledger


async def test_the_parse_tree_is_gone_before_the_first_block_is_yielded(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suspended generator must be holding nothing native.

    ``parse`` is an async generator, and one abandoned part-way stays suspended until CPython
    finalises it through the loop that created it — running a native destructor against a
    torn-down runtime if that loop has closed, which surfaces as a crash inside the
    interpreter naming no library anybody here wrote. Releasing in a ``finally`` answers the
    closed case; keeping the tree out of the frame answers the never-closed case too, so that
    is what is asserted, at the moment the generator is suspended.
    """
    require_grammar("python")
    ledger = _track_trees(monkeypatch)
    stream = parser().parse(fixture(corpus, "typical-tokens.py"))
    try:
        await anext(stream)

        assert ledger.made == 1
        assert ledger.held == 0, "the parse tree is still alive while the stream is suspended"
    finally:
        await aclose(stream)


class _HoldsItsTreeParser(SourceCodeParser):
    """Keeps its parse tree in the generator frame and releases it after the last block.

    Correct on a full drain, which is every ordinary run, and wrong on exactly the path that
    is hard to notice. Here to prove the test above is load-bearing.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        tree = grammars.load_parser("python").parse(raw.as_bytes())
        async for block in super().parse(raw):
            yield block
        del tree


async def test_a_parser_holding_its_tree_across_a_yield_is_caught_by_that_test(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof the check above detects the thing it describes rather than passing by luck."""
    require_grammar("python")
    ledger = _track_trees(monkeypatch)
    stream = _HoldsItsTreeParser(SourceCodeConfig()).parse(fixture(corpus, "typical-tokens.py"))
    try:
        await anext(stream)

        assert ledger.held == 1
    finally:
        await aclose(stream)


async def test_a_stream_stopped_after_one_block_leaves_the_parser_usable(
    corpus: Path,
) -> None:
    """The ordinary early exit: a consumer that stops, or an assertion that fails between
    blocks. Re-parsing afterwards proves nothing was left in a state the next read trips
    over."""
    require_grammar("python")
    raw = fixture(corpus, "typical-tokens.py")
    built = parser()

    async with parsing(built, raw) as blocks:
        async for _ in blocks:
            break

    assert len(await read_blocks(built, raw)) > 0
