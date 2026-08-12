"""What each parser was built out of, so a bump can name the documents it changed.

``ChunkFingerprint.grammars`` records a tree-sitter version per language and calls the result
selective invalidation. Nine other libraries decide what a document is reduced to —
``pypdfium2``, ``selectolax``, ``lxml``, ``python-docx``, ``python-pptx``,
``python-calamine``, ``nbformat``, ``markdown-it-py``, ``ruamel.yaml`` — and until this module
existed none of them was recorded anywhere. A bump changed the text the same bytes produce,
change detection keys on those bytes and so never re-read what was stored, and the corpus
quietly held two generations of extracted text.

This is the table that closes it, and three properties of it are load-bearing.

**It answers without importing a parser.** Every value is distribution metadata, which is a
fact about what is installed rather than about what has been initialised — the same reasoning
:func:`~manicule.parsers.grammars.pack_version` records for the grammar pack. Change
detection asks this question once per document, and a table that imported ``pypdfium2`` to
answer it would load a native extension on a corpus that contains no PDFs.

**Names are PEP 503 canonical, and that is checked rather than assumed.** ``ruamel.yaml`` is
the name in ``pyproject.toml`` and ``ruamel-yaml`` is the name it installs under; recording
one while looking up the other is how a version silently stops moving. :data:`PARSERS` is
written in canonical form and ``tests/parsers/test_versions.py`` asserts both that
canonicalising each name is a no-op and that each one resolves to a real version — because a
lookup that quietly returned a default would produce a fingerprint that never changes, which
is worse than no fingerprint at all.

**A missing distribution raises.** :class:`importlib.metadata.PackageNotFoundError` reaches
the caller. The alternative is a placeholder, and a placeholder in an identity field is the
defect this module exists to remove, one level along.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final

from manicule.core.fingerprints import ParseFingerprint

__all__ = [
    "PARSERS",
    "ParserVersions",
    "current_parse_fingerprints",
    "distributions_recorded",
    "parse_fingerprint",
]


@dataclass(frozen=True, slots=True)
class ParserVersions:
    """How one parser's identity is assembled."""

    rules: str
    """The version of manicule's own extraction rules for this parser.

    Bumped by hand when this repository changes which blocks a parser emits or what its
    anchors address. It exists because a dependency map alone would leave manicule's own
    changes — the ones a maintainer makes deliberately — as the only thing nothing records.
    ``docs/parsing.md`` §3.2 already does this for one case under the name ``web-blocks/1``.
    """

    distributions: tuple[str, ...] = ()
    """PEP 503 canonical names of the libraries whose behaviour decides text or anchors.

    Empty where a parser is built on the standard library alone. That is a real answer, not a
    gap: ``adf`` reads JSON, ``archive`` reads zip and tar, ``plaintext`` splits lines, and
    none of them can be moved by a dependency bump.
    """


PARSERS: Final[dict[str, ParserVersions]] = {
    "adf": ParserVersions(rules="1"),
    "archive": ParserVersions(rules="1"),
    "docx": ParserVersions(rules="1", distributions=("python-docx", "lxml")),
    "email": ParserVersions(rules="1", distributions=("selectolax",)),
    # 1 -> 2: CDATA sections are recovered as text rather than deleted by the HTML parser's
    # bogus-comment reparse. Every document containing one produces different text now, and
    # the bump is what re-parses them from retained bytes instead of leaving a corpus that is
    # wrong behind a fingerprint claiming it is current.
    "html": ParserVersions(rules="2", distributions=("selectolax",)),
    "markdown": ParserVersions(rules="1", distributions=("markdown-it-py",)),
    "notebook": ParserVersions(rules="1", distributions=("nbformat",)),
    "pdf": ParserVersions(rules="1", distributions=("pypdfium2",)),
    "plaintext": ParserVersions(rules="1"),
    "pptx": ParserVersions(rules="1", distributions=("python-pptx", "lxml")),
    "sourcecode": ParserVersions(rules="1", distributions=("tree-sitter",)),
    "spreadsheet": ParserVersions(rules="1", distributions=("python-calamine",)),
    "structured": ParserVersions(rules="1", distributions=("ruamel-yaml",)),
}
"""Every parser manicule ships, by registered name.

Four entries need their reasoning on the page, because the obvious table is wrong in four
places:

``docx`` and ``pptx`` name ``lxml``
    Neither library parses XML itself; both hand ``word/document.xml`` and
    ``ppt/presentation.xml`` to ``lxml``, so the XML parse *is* the extraction step and
    neither wrapper would notice it changing. ``lxml`` is a transitive dependency rather than
    a declared one, which is exactly why it is easy to leave out and exactly why leaving it
    out would be silent.

``email`` names ``selectolax``
    A mail with an HTML-only body is reduced through the web parser, and its
    :class:`~manicule.core.anchors.LineAnchor` addresses lines of the *converted* text. A
    converter upgrade therefore shifts every anchor in every HTML email — round-tripping
    today, pointing at the wrong paragraph after a bump.

``sourcecode`` names ``tree-sitter`` and not the grammar pack
    The pack release is already recorded, per language, in
    :attr:`~manicule.core.fingerprints.ChunkFingerprint.grammars`, and recording it twice
    would make one bump look like two independent events. The tree-sitter runtime underneath
    it is recorded here because nothing else records it, and it is what turns a grammar into
    a parse tree.

``spreadsheet`` names only ``python-calamine``
    CSV goes through the standard library's ``csv`` module, which moves with the interpreter
    rather than with a distribution. There is no version to record and no honest way to
    invent one.
"""


@cache
def parse_fingerprint(parser: str) -> ParseFingerprint | None:
    """What ``parser`` would produce today, or ``None`` if manicule does not ship it.

    ``None`` rather than a placeholder fingerprint. A third-party parser's version is
    something manicule cannot read, and a fingerprint saying ``unknown`` would be a constant
    standing in for a thing that varies — which is precisely the defect in a stand-in
    tokenizer id. What ``None`` means downstream is stated where it is acted on: change
    detection treats "no recorded lineage, and none obtainable" as unchanged, so a plugin
    corpus is not re-parsed on every sync, while repair selection treats a null ``parse_fp``
    as eligible, so it is reachable by ``reindex --re-parse`` on demand. Both defaults point
    away from claiming currency that cannot be proved.

    Cached for the life of the process. The answer is a fact about the installed environment,
    which does not change under a running process, and change detection asks it once per
    document.

    Args:
        parser: A registered parser name, as recorded in ``parser_used``.

    Raises:
        PackageNotFoundError: A declared distribution is not installed. That means the parser
            could not have run, so answering with anything would be a fiction.
    """
    from importlib.metadata import version  # noqa: PLC0415 - see the module docstring

    declared = PARSERS.get(parser)
    if declared is None:
        return None
    return ParseFingerprint(
        parser=parser,
        version=declared.rules,
        libraries={name: version(name) for name in declared.distributions},
    )


def current_parse_fingerprints() -> tuple[ParseFingerprint, ...]:
    """What every parser manicule ships would produce today.

    The argument :func:`~manicule.ingest.reindex.select` takes to find the documents a library
    bump left behind: its complement over ``documents.parse_fp`` is exactly the stale set.
    Sorted by parser name so that the value is a set of facts rather than a dictionary order.

    Raises:
        PackageNotFoundError: A declared distribution is not installed, so one of these
            parsers could not run. Answering with a partial set would make every document that
            parser produced look stale, which is a repair that cannot succeed.
    """
    return tuple(
        fingerprint
        for name in sorted(PARSERS)
        if (fingerprint := parse_fingerprint(name)) is not None
    )


def distributions_recorded() -> frozenset[str]:
    """Every distribution any parser's identity depends on.

    What ``.github/dependabot.yml`` has to keep grouped under ``index-affecting-extraction``,
    and what the test comparing the two reads. A distribution that reaches a fingerprint
    without reaching that group is a bump that lands in a pull request titled "the runtime
    group" and re-parses a corpus.
    """
    return frozenset(name for entry in PARSERS.values() for name in entry.distributions)
