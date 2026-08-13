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

    Bumped by hand when this repository changes which blocks a parser emits, what their
    metadata says about how they are to be chunked, or what their anchors address. It exists
    because a dependency map alone would leave manicule's own changes — the ones a maintainer
    makes deliberately — as the only thing nothing records. ``docs/parsing.md`` §3.2 already
    does this for one case under the name ``web-blocks/1``.

    **The middle clause is not padding; it was added because leaving it out let a bump through
    uncounted.** #109 taught four parsers to emit ``rows``, and a table block's ``rows`` is what
    :meth:`~manicule.chunking.chunker.StructuralChunker._split_table` divides an oversized table
    at. Measured across the built corpus, that commit moved no block's ``text``, no block's
    anchor and no block's ``heading_path`` — so on the two clauses this sentence used to have,
    nothing had changed and no bump was warranted. What it moved was how the same text is *cut*:
    on a 300-row table, four to five rows had been severed mid-cell at a chunk boundary and no
    longer were.

    Neither fingerprint caught that. ``ChunkFingerprint`` did not move because the chunker's own
    code was untouched — only its input was — and ``ParseFingerprint`` did not move because
    nobody bumped it. The corpus was left holding two generations of *chunks* behind lineage
    claiming to be current, which is this module's opening paragraph one stage along. A
    ``rules`` bump is the only instrument that reaches it: blocks are not persisted, so no
    chunk-level repair can recover a row boundary that was never recorded, and only ``re_parse``
    reads the retained bytes again.
    """

    distributions: tuple[str, ...] = ()
    """PEP 503 canonical names of the libraries whose behaviour decides text or anchors.

    Empty where a parser is built on the standard library alone. That is a real answer, not a
    gap: ``adf`` reads JSON, ``archive`` reads zip and tar, ``plaintext`` splits lines, and
    none of them can be moved by a dependency bump.
    """


PARSERS: Final[dict[str, ParserVersions]] = {
    # 1 -> 2: table blocks carry `rows`, so an oversized table is split at its row boundaries
    # with the header repeated into every part instead of wherever the token budget landed. The
    # text is byte-identical — this parser already joined the same rendered rows with `\n` — and
    # `_table`'s docstring had promised the header-repeating split since before it emitted
    # anything `_split_table` could use. See the shared note under `markdown` below for the
    # measurement and for why `email` does not move with these four.
    "adf": ParserVersions(rules="2"),
    "archive": ParserVersions(rules="1"),
    # Started at 1 rather than inheriting `html`'s 2. A version is a statement about one
    # parser's own output, and these are different parsers: the documents this reads were
    # previously read by `html`, and what re-parses them is `Change.ROUTING` noticing that the
    # media type they arrive under has changed. A bump here could not have done it — the stored
    # documents name `html` as the parser they used, so this entry is not the one their lineage
    # is compared against. That remains true of every document ingested before this parser
    # existed; the bump below is about the ones ingested since.
    #
    # 1 -> 2: consecutive paragraphs inside a rich-text macro body are joined with a blank line
    # rather than a single newline. A blank line is what `chunking.sentences.paragraphs` reads as
    # a paragraph boundary, so under version 1 a macro body was one paragraph however many `<p>`
    # elements it held — and past the chunk budget it was split into sentences and repacked with
    # spaces, losing every boundary the page had. Every panel and every unsupported macro with
    # more than one prose block now produces different text and different chunk boundaries, so
    # the bump is what re-parses them from retained bytes instead of leaving a corpus that
    # disagrees with the parser behind a fingerprint claiming it is current.
    #
    # 2 -> 3: an inline `<br>` contributes a newline. Under version 2 it contributed no
    # character at all, so `a<br/>b` read `ab` — two fragments glued into a word the page does
    # not contain. Every storage document with a break in prose gains a newline, and every one
    # with a break in a heading, a table cell, a list item or a task body gains the space that
    # was missing between the fragments either side of it.
    #
    # 3 -> 4: table blocks carry `rows`. `_table_text` became `_table_rows` with the
    # `"\n".join(...)` moved to its caller, so the text every storage document produces is
    # byte-identical and only the metadata beside it is new — but that metadata is what
    # `_split_table` divides an oversized table at, so a stored table stops being cut mid-row.
    # See the shared note under `markdown` below.
    "confluence": ParserVersions(rules="4", distributions=("selectolax",)),
    "docx": ParserVersions(rules="1", distributions=("python-docx", "lxml")),
    # 1 -> 2: an HTML-only mail body's line numbers address the text
    # `mail._html_to_text` builds from the web parser's blocks, and the web parser now
    # recovers CDATA sections instead of deleting them — so a recovered body becomes a block
    # and every line after it moves. Bumped even though this parser's own rules are
    # unchanged, because what it extracts changed and `parse_fp` is what re-parses it.
    #
    # 2 -> 3: the same reasoning, one change along. A `<br>` in an HTML-only body now puts a
    # newline *inside* a block, and `_html_to_text` hands the result to `lines_of` — so the
    # break becomes a line of the canonical text and every `LineAnchor` after it moves by one.
    # This parser's own rules are again unchanged; what it extracts is not.
    #
    # **Deliberately not moved to 4 when `html` did**, which breaks the run of two above and is
    # therefore the entry most at risk of being tidied into symmetry later. Both bumps above
    # moved the *text* an HTML-only body's `LineAnchor`s address. #109 did not: it added `rows`
    # to what the web parser says *about* a table block, and `_html_to_text` joins the blocks'
    # `text` and never reads their metadata — a body carrying a 300-row table parses to
    # byte-identical blocks and byte-identical anchors across it, run rather than inferred.
    # `test_an_html_body_is_built_from_block_text_alone` in `tests/parsers/test_mail.py` is what
    # keeps that true; a bump here would be symmetry rather than a fact, and it would re-parse
    # and re-embed every email in the corpus to produce identical text.
    "email": ParserVersions(rules="3", distributions=("selectolax",)),
    # 1 -> 2: CDATA sections are recovered as text rather than deleted by the HTML parser's
    # bogus-comment reparse. Every document containing one produces different text now, and
    # the bump is what re-parses them from retained bytes instead of leaving a corpus that is
    # wrong behind a fingerprint claiming it is current.
    #
    # 2 -> 3: an inline `<br>` contributes a newline, where it contributed no character at all
    # before and `a<br/>b` read `ab`. Every HTML document with a break in prose gains a newline;
    # every one with a break in a heading, a table cell or a list item gains the space that was
    # missing between the fragments either side of it. `html_text_version` is deliberately not
    # bumped with it: `web-blocks/1` names the *rule* email applies to these blocks — join them
    # with a blank line — and that rule is unchanged. It sits in `ChunkFingerprint.version`,
    # where a bump refuses ingest against the whole corpus; a change to what one parser extracts
    # is what `parse_fp` is for, which is the division the 1 -> 2 pair above already drew.
    #
    # 3 -> 4: table blocks carry `rows`, by the same edit and for the same reason as
    # `confluence`'s 3 -> 4 above — `_table_text` became `_table_rows` and the join moved to the
    # caller. `email` is deliberately **not** bumped with it, which is the reverse of the 1 -> 2
    # and 2 -> 3 pairs above: those moved the *text* an HTML-only body's `LineAnchor`s address,
    # and this does not. `mail._html_to_text` joins the blocks' `text` and never reads their
    # metadata, and an HTML-only body carrying a 300-row table parses to byte-identical blocks
    # and byte-identical anchors across this change — run, not inferred.
    "html": ParserVersions(rules="4", distributions=("selectolax",)),
    # 1 -> 2: two changes from #109, and the second is the one that is easy to miss.
    #
    # Table blocks carry `rows`, as for the three parsers above. **And `header_rows` changed
    # meaning**: it counts source lines rather than `tr_open` tokens, so the `| --- | --- |`
    # delimiter travels with the header it delimits and an ordinary single-header table reports
    # 2 where it used to report 1. That is a changed value under an unchanged key, which no
    # reader of a diff of `PARSERS` would infer and no test outside the markdown suite would
    # catch.
    #
    # **What the four bumps cost is the shape `docs/parsing.md` §4.5 prices**; what is specific
    # to them is what moved and how it was established. Every one of the four was measured by
    # parsing the built corpus at 41682bb^ and at 41682bb and diffing block by block: 647 blocks
    # over 49 documents, **0 whose text moved, 0 whose anchor moved, 0 whose heading_path
    # moved**, and 20 metadata fields that did — `rows` on 16 blocks and `header_rows` on 4
    # markdown blocks. So the case for these bumps is not that extraction changed; it is that
    # chunking did, and chunking is downstream of metadata nothing else records. On a 300-row
    # glossary table the rows severed across a chunk boundary go 4 -> 0 for adf, confluence and
    # html and 5 -> 0 for markdown, the fragments being exactly the shape #109 described:
    # `'T063 | Term 63 '` ending one chunk and `'Expansion Words Here'` beginning the next.
    #
    # The corpus itself shows no difference at all — it holds no table over the chunk budget, so
    # `_split_table` is never reached in it. That is why this was invisible, and it is not
    # evidence that nothing changed.
    "markdown": ParserVersions(rules="2", distributions=("markdown-it-py",)),
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
    as eligible, so it is reachable by ``document reindex --stale`` on demand. Both defaults
    point
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
