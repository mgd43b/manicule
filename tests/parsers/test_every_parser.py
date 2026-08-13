"""Every registered parser, held to the shipped contract over its own fixture corpus.

The suites beside this one go deep on one parser each. This one goes wide, and it is driven
by :data:`manicule.parsers.plugin.PARSERS` rather than by a list written here — so a parser
added to the registration table without fixtures fails this file, and a parser whose anchors
stop resolving fails it whether or not anybody wrote a suite for that format.

**The shipped contract runs over every fixture, with no exceptions.**
:func:`~manicule.testing.assert_parser_contract` is the promise a third-party parser is held
to — resolving each block's anchor returns the text that block claims — and there is no
fixture in this repository it is allowed to skip.

**The six assertions run over a declared subset**, because one of them cannot be run over
everything. Assertion 3 asks that no location's resolved text *contains* another block's
text, which is what catches an anchor confused with its neighbour. One fixture in the corpus
repeats wording on purpose — two DOCX sections with the same heading text — and the assertion
cannot tell that apart from a wrong anchor. It is named in :attr:`Corpus.ambiguous` **with the
reason and with the test that covers it instead**, and a fixture may not be listed there
merely because it fails.

**The location budget is not asserted here**, and
:func:`test_a_parsers_corpus_round_trips_within_its_location_budget` says why: it is a ratio
over a corpus, and this corpus is deliberately weighted towards hostile input. Each parser's
own suite runs it over the fixtures that parser declares representative.

**A parser that declines a fixture is not a failure**, and the sweep records it rather than
tripping over it: the hostile fixtures exist precisely to be declined, and which of them a
given parser declines is that parser's own decision. What is a failure is a parser that
declines *everything* — a corpus of nothing has no ratio to exceed — so each parser declares
the fixtures it must accept, and the sweep fails if one of those is refused.

Nothing here is proof on its own that the assertions bite; ``tests/test_conformance.py`` and
``tests/test_roundtrip.py`` run parsers that break each rule and require every check to catch
them. This file is the coverage, those are the calibration.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.errors import ManiculeError, ParseError
from manicule.core.protocols import Parser
from manicule.parsers import config as parser_config
from manicule.parsers.plugin import PARSERS
from manicule.testing import (
    ParserProfile,
    RoundTripReport,
    assert_parser_contract,
    assert_round_trip,
)
from tests.parsers.support import document_for, raw_from


@dataclass(frozen=True, slots=True)
class Corpus:
    """Where one parser's fixtures are, and what routes them.

    Args:
        directory: The generator's directory under the built corpus.
        media_types: Suffix to media type. A parser's own routing table, restated here so a
            fixture named with the wrong extension is a visible mistake rather than a
            document that quietly never reaches the parser.
        required: Fixtures the parser must accept. Everything else may be declined; these may
            not, so a parser cannot pass by refusing its whole corpus.
        min_blocks: A floor on the corpus size, so a generator that silently stopped emitting
            fixtures cannot pass every ratio by dividing zero by zero.
        encoding: Declared for the fixtures whose bytes are deliberately not UTF-8.
    """

    directory: str
    media_types: dict[str, str]
    required: tuple[str, ...]
    min_blocks: int
    encoding: str = "utf-8"
    ambiguous: dict[str, str] = field(default_factory=dict[str, str])
    """Fixtures the six assertions cannot run over, and why, one entry per fixture.

    Only ever for repeated wording that assertion 3's containment check cannot distinguish
    from a wrong anchor. The shipped contract still runs over every one of them, and the value
    names the test that covers what the fixture is actually for.
    """


_XLSX = parser_config.XLSX_MEDIA_TYPE
_DOCX = parser_config.WORD_MEDIA_TYPE
_PPTX = parser_config.SLIDES_MEDIA_TYPE

CORPORA: dict[str, Corpus] = {
    "pdf": Corpus(
        directory="pdf",
        media_types={".pdf": "application/pdf"},
        required=("typical.pdf", "multicolumn.pdf", "upright.pdf"),
        min_blocks=40,
    ),
    "markdown": Corpus(
        directory="markdown",
        media_types={".md": "text/markdown", ".mdx": "text/mdx"},
        required=("typical.md", "structure.md", "components.mdx"),
        min_blocks=60,
    ),
    "html": Corpus(
        directory="web",
        media_types={".html": "text/html", ".htm": "text/html"},
        required=("typical.html", "structure.html"),
        min_blocks=40,
    ),
    "adf": Corpus(
        directory="adf",
        media_types={".json": parser_config.ADF_MEDIA_TYPE},
        required=("typical.json", "structure.json"),
        min_blocks=40,
    ),
    "confluence": Corpus(
        directory="confluence",
        media_types={".storage": parser_config.CONFLUENCE_MEDIA_TYPE},
        required=("typical.storage", "structure.storage", "hostile.storage"),
        min_blocks=60,
        ambiguous={
            "structure.storage": (
                "two sections share the heading path 'Capacity review > Region > "
                "Configuration', so the anchored one's resolved span contains the other's "
                "whole heading block, and assertion 3 cannot tell that apart from a misplaced "
                "anchor. The repeat is the point of the fixture. Covered by test_confluence.py"
                "::test_a_repeated_heading_path_is_unlocated_unless_an_anchor_macro_addresses_it"
            )
        },
    ),
    "docx": Corpus(
        directory="word",
        media_types={".docx": _DOCX},
        required=("word_typical.docx", "word_structurally_hard.docx"),
        min_blocks=40,
        ambiguous={
            "word_repeated_heading_path.docx": (
                "two sections share their heading text, so one section's resolved span "
                "contains the other's whole heading block, and assertion 3 cannot tell that "
                "apart from a misplaced anchor. Covered by test_word.py::"
                "test_two_sections_with_one_heading_path_and_no_fragment_are_unlocated"
            )
        },
    ),
    "pptx": Corpus(
        directory="slides",
        media_types={".pptx": _PPTX},
        required=("slides_typical.pptx", "slides_structurally_hard.pptx"),
        min_blocks=20,
    ),
    "spreadsheet": Corpus(
        directory="spreadsheet",
        media_types={".xlsx": _XLSX, ".csv": parser_config.CSV_MEDIA_TYPE},
        required=("spreadsheet_typical.xlsx", "spreadsheet_typical.csv"),
        min_blocks=10,
    ),
    "notebook": Corpus(
        directory="notebook",
        media_types={".ipynb": parser_config.NOTEBOOK_MEDIA_TYPE},
        required=("notebook_typical.ipynb", "notebook_structurally_hard.ipynb"),
        min_blocks=20,
    ),
    "email": Corpus(
        directory="mail",
        media_types={".eml": "message/rfc822"},
        required=("typical.eml", "multipart.eml", "html-only.eml"),
        min_blocks=10,
    ),
    "plaintext": Corpus(
        directory="plaintext",
        # .bin is mapped deliberately: not-text.bin is a fixture that must be *declined*
        # when something routes it here as text/plain, and a fixture the sweep never
        # offers the parser is a refusal nobody is checking.
        media_types={".txt": "text/plain", ".bin": "text/plain"},
        required=("typical.txt", "hard-wrapped.txt"),
        min_blocks=10,
    ),
    "structured": Corpus(
        directory="structured",
        media_types={
            ".json": "application/json",
            ".toml": "application/toml",
            ".yaml": "application/yaml",
        },
        required=("typical.yaml", "typical.json", "typical.toml"),
        min_blocks=20,
    ),
    "sourcecode": Corpus(
        directory="sourcecode",
        media_types={
            ".py": "text/x-python",
            ".rs": "text/x-rust",
            ".ts": "text/x-typescript",
            ".sh": "text/x-shellscript",
            ".sql": "application/sql",
        },
        required=("typical-tokens.py", "typical-store.rs", "typical-sessions.ts"),
        min_blocks=30,
    ),
}
"""One entry per parser that reads documents. ``archive`` is absent on purpose — it produces
no blocks at all, so there is no anchor to round-trip; ``tests/parsers/test_archive.py``
holds it to the obligations that do apply to it."""

BLOCKLESS = frozenset({"archive"})
"""Registered parsers that emit no blocks, and so have nothing for this sweep to check."""


def test_every_registered_parser_is_covered_by_this_sweep() -> None:
    """A parser added to the registration table without fixtures fails here.

    The list of parsers is read from the plugin rather than written out below, so this is not
    a reminder to update two places — it is the thing that makes the sweep exhaustive. Without
    it, ``assert_parser_contract`` would cover whichever parsers somebody remembered.
    """
    registered = {registration.name for registration in PARSERS}
    covered = set(CORPORA) | BLOCKLESS

    assert registered == covered, (
        f"parsers registered but not swept: {sorted(registered - covered)}; "
        f"swept but not registered: {sorted(covered - registered)}. Add fixtures under "
        f"tests/corpus/ and an entry in CORPORA, or declare the parser blockless"
    )


@pytest.mark.parametrize("name", sorted(CORPORA))
async def test_every_fixture_of_every_parser_passes_the_shipped_contract(
    name: str, corpus: Path
) -> None:
    """``assert_parser_contract`` over one parser's whole corpus, ambiguous fixtures included.

    This is the obligation ``CONTRIBUTING.md`` states for every parser, and the reason it is
    stated separately from the rest: a citation pointing at a page that does not exist looks
    exactly like one that does, so nothing downstream can catch it. There is no exclusion list
    here, because there is nothing about a fixture that could justify one.
    """
    entry = CORPORA[name]
    parser = _build(name)
    accepted: list[str] = []
    declined: list[str] = []

    for path in _fixtures(corpus / entry.directory, entry):
        try:
            await assert_parser_contract(parser, _raw(path, entry))
        except ParseError:
            declined.append(path.name)
        except ManiculeError as refusal:
            # A refusal that is not a decline — a missing grammar, a limit hit. Distinct from
            # ParseError on purpose, and not something a fixture sweep may swallow.
            pytest.fail(f"{name}: {path.name} was refused rather than declined: {refusal}")
        else:
            accepted.append(path.name)

    _require_accepted(name, entry, declined)
    assert accepted, f"{name}: declined every fixture, so the contract checked nothing"


@pytest.mark.parametrize("name", sorted(CORPORA))
async def test_a_parsers_corpus_round_trips_within_its_location_budget(
    name: str, corpus: Path, chunker: StructuralChunker
) -> None:
    """The six assertions over one parser's unambiguous fixtures.

    The chunker is passed so the harness checks chunk anchors as well as block anchors:
    chunking is where a block's anchor is copied onto text that is a fraction of it, and a
    chunk whose anchor addresses less than its own text is a citation nobody can check by
    reading it.

    **The location budget is deliberately not asserted here.** It is a ratio over a corpus,
    and this corpus is not a realistic one — it is weighted towards hostile and degenerate
    input, because that is what a fixture corpus is for. A notebook corpus that is one-sixth
    "two sections a cell id cannot tell apart" would fail a ceiling written for real
    notebooks, and the repair would be to raise the ceiling, which would weaken it everywhere.
    Each parser's own suite runs :func:`~manicule.testing.assert_location_budget` over the
    fixtures it declares representative; :func:`test_every_parser_declares_a_location_budget`
    below checks that a budget exists to be run.
    """
    entry = CORPORA[name]
    parser = _build(name)
    reports: list[RoundTripReport] = []
    declined: list[str] = []

    for path in _fixtures(corpus / entry.directory, entry):
        if path.name in entry.ambiguous:
            continue
        try:
            # ``assert_round_trip`` rather than ``check_fixture``: the shipped contract is
            # established over this same corpus by the test above, and running it twice
            # re-parses every large fixture for no signal.
            raw = _raw(path, entry)
            reports.append(
                await assert_round_trip(
                    parser,
                    raw,
                    fixture=raw.uri,
                    chunker=chunker,
                    document=document_for(raw),
                )
            )
        except ParseError:
            declined.append(path.name)
        except ManiculeError as refusal:
            pytest.fail(f"{name}: {path.name} was refused rather than declined: {refusal}")

    _require_accepted(name, entry, declined)
    total = sum(report.blocks for report in reports)
    assert total >= entry.min_blocks, (
        f"{name}: the fixture corpus produced {total} blocks, fewer than the "
        f"{entry.min_blocks} this sweep claims to cover. A corpus that quietly stopped "
        f"producing fixtures passes every check that divides by it"
    )


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_no_fixture_is_excused_from_the_six_assertions_without_a_reason(
    name: str, corpus: Path
) -> None:
    """An exclusion has to name a real fixture, say why, and name what covers it instead.

    Without this the exclusion list is a place to put anything that fails, which would turn
    the sweep from a check into a record of what somebody could not be bothered to fix.
    """
    entry = CORPORA[name]
    directory = corpus / entry.directory
    present = {path.name for path in directory.iterdir()}

    for fixture, reason in entry.ambiguous.items():
        assert fixture in present, f"{name}: excused {fixture!r}, which is not in the corpus"
        assert "Covered by" in reason, (
            f"{name}: {fixture!r} is excused from the six assertions without naming the test "
            f"that covers what it is for"
        )


def _require_accepted(name: str, entry: Corpus, declined: Sequence[str]) -> None:
    """Fail if the parser declined a fixture it is declared to have to accept."""
    refused = sorted(set(entry.required) & set(declined))
    assert not refused, (
        f"{name}: declined {refused}, which this suite lists as fixtures the parser must "
        f"accept. A parser that declines its whole corpus satisfies every ratio"
    )


@pytest.mark.parametrize("name", sorted(BLOCKLESS | set(CORPORA)))
def test_every_parser_declares_a_location_budget(name: str) -> None:
    """Both ratios are declared by the parser and enforced by its own suite.

    Without a declared profile there is nothing for ``assert_location_budget`` to enforce, and
    "a citation carries a correct location, or none" is satisfied by never carrying one. The
    ceilings are checked here rather than assumed because a parser added without a profile
    would otherwise be exempt from the budget and from nothing else, which is invisible.

    Read with ``getattr`` because ``profile`` is not on the ``Parser`` protocol: it is what
    the test harness asks of a parser, not what the pipeline asks, and putting it on the
    protocol would make it a runtime requirement for something only tests read.
    """
    profile = getattr(_build(name), "profile", None)

    assert isinstance(profile, ParserProfile), f"{name} declares no ParserProfile"
    assert profile.name, f"{name}: the profile has no name, so a failure would not say whose"
    assert 0.0 <= profile.max_unlocated_ratio <= 1.0
    assert profile.max_pagelevel_ratio is None or 0.0 <= profile.max_pagelevel_ratio <= 1.0


@pytest.mark.parametrize("name", sorted(CORPORA))
async def test_a_parser_accepts_the_media_types_it_claims(name: str) -> None:
    """A parser's own declaration and the registration's must agree.

    Routing reads the *registration*, so a parser that quietly narrows its media types would
    be handed documents it then declines — and the chain would fall through to the next parser
    with no indication that the one meant to handle it was installed all along.
    """
    registration = next(r for r in PARSERS if r.name == name)
    parser = _build(name)

    assert parser.media_types, f"{name} declares no media types, so nothing routes to it"
    assert set(parser.media_types) == set(registration.media_types), (
        f"{name}: the parser claims {sorted(parser.media_types)} but registration declares "
        f"{sorted(registration.media_types)}. Routing reads the registration"
    )


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_a_parsers_fixtures_all_route_to_it(name: str, corpus: Path) -> None:
    """Every fixture's extension resolves to a media type the parser claims.

    A fixture with an extension nothing maps would be silently skipped by the sweep above,
    and a whole generator could stop being exercised without a single test turning red.
    """
    entry = CORPORA[name]
    parser = _build(name)
    directory = corpus / entry.directory
    unmapped = sorted(
        path.name for path in directory.iterdir() if path.suffix not in entry.media_types
    )

    assert not unmapped, f"{name}: no media type declared for {unmapped}"
    assert set(entry.media_types.values()) <= set(parser.media_types), (
        f"{name}: fixtures declare media types the parser does not claim: "
        f"{sorted(set(entry.media_types.values()) - set(parser.media_types))}"
    )


def _fixtures(directory: Path, entry: Corpus) -> Iterator[Path]:
    """Every fixture the sweep should offer this parser, in a stable order."""
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in entry.media_types:
            yield path


def _raw(path: Path, entry: Corpus):  # noqa: ANN202 - RawDocument, inferred from the helper
    return raw_from(path, entry.media_types[path.suffix], encoding=entry.encoding)


def _build(name: str) -> Parser:
    """Construct one registered parser from its declared module, factory and config model.

    Through the registration rather than by importing the class, so the sweep exercises the
    same description the container builds from. A registration naming a class that does not
    exist fails here rather than at the first document of that type.
    """
    from importlib import import_module  # noqa: PLC0415 - one import per parser, on demand

    registration = next(r for r in PARSERS if r.name == name)
    module = import_module(registration.module)
    factory = getattr(module, registration.factory)
    built: Parser = factory(registration.config_model())
    return built
