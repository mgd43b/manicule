"""The one normaliser both sides of every round-trip comparison pass through.

Exact string equality is the wrong assertion. PDF extraction reintroduces ligatures and
hyphenation, HTML collapses whitespace differently from its source, and DOCX splits a
sentence across runs. Comparing raw strings would fail parsers that are behaving correctly,
and the usual repair — loosening the comparison per parser until the suite passes — leaves
no assertion at all.

So there is one normaliser, defined here, used by every check.

**Order is load-bearing.** De-hyphenation needs the line breaks that whitespace collapsing
destroys. Running the steps the other way round silently disables it, and nothing fails to
say so — the hyphenated words simply stop being joined and every affected comparison starts
relying on the substring check being lenient.

**Stored text is never normalised.** ``Chunk.text`` is what a user is shown, and showing a
whitespace-flattened, ligature-substituted rendering of a quotation is a change to the
quotation. This module exists for comparisons and for nothing else.
"""

from __future__ import annotations

import re
import unicodedata

NORMALISER_VERSION = "1"
"""Bumped when the steps below change.

Recorded in test output rather than in
:class:`~manicule.core.fingerprints.ChunkFingerprint`. The normaliser cannot move a chunk
boundary — it never runs during ingest — so putting it in the fingerprint would force a
re-chunk and re-embed of an entire corpus in exchange for a change that cannot alter one
stored byte.
"""

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}
"""``U+FB00``-``U+FB04``, spelled by codepoint because they are indistinguishable from their
letter sequences on the page. A normalisation rule an implementer cannot see is not a
specification."""

_SOFT_HYPHEN = "­"
_ZERO_WIDTH_SPACE = "\u200b"
_NO_BREAK_SPACE = "\u00a0"
"""Spelled by escape rather than typed. A no-break space is indistinguishable from a space
in a source file, and a normalisation rule nobody can see in the code is not a
specification."""

_LIGATURE_TABLE = str.maketrans({**_LIGATURES, _SOFT_HYPHEN: ""})

_HYPHEN_LINE_BREAK = re.compile(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)")
"""A word split across a line break. Only between word characters: a line ending in a dash
used as punctuation, or one followed by a bullet, is not a split word."""

_WHITESPACE_RUN = re.compile(rf"[\s{_ZERO_WIDTH_SPACE}{_NO_BREAK_SPACE}]+")
"""Every whitespace run. ``\\s`` covers ``U+00A0`` under Unicode matching but not
``U+200B``, which is formally a format character and would otherwise survive into a
comparison and break it invisibly."""


def normalise(text: str) -> str:
    """Reduce ``text`` to the form both sides of a round-trip comparison are checked in.

    The five steps of ``docs/parsing.md`` §3.2, in the order that document fixes:

    1. Unicode NFC — **not NFKC**, which would fold the ligatures for free but also rewrite
       ``½``, superscripts and full-width forms. Those are content a citation is supposed to
       reproduce verbatim, so the five ligatures are folded explicitly and nothing else is.
    2. Ligatures to letter sequences; soft hyphens removed.
    3. **While line breaks still exist**, join words split by a hyphen at a line end.
    4. Every whitespace run to a single space.
    5. Strip.
    """
    folded = unicodedata.normalize("NFC", text).translate(_LIGATURE_TABLE)
    joined = _HYPHEN_LINE_BREAK.sub("", folded)
    return _WHITESPACE_RUN.sub(" ", joined).strip()


def contains_claimed_text(resolved: str | None, claimed: str) -> bool:
    """Whether the text at a location contains what a chunk or block claims came from there.

    **The containment predicate, and there is exactly one of it.** The parser conformance
    suite asks this question of every fixture at test time
    (:func:`manicule.testing.assert_parser_contract`); citation verification asks the same
    question of a single anchor at answer time (``docs/generation.md`` §3.3, level 2). Two
    notions of "this anchor resolves" would drift, and the drift would surface as citations
    that pass CI and fail in production, or the reverse — so the runtime check and the
    test-suite check are the same call, not a copy of it.

    That matters more than it looks. The usual repair for a comparison that fails — loosening
    it until the suite passes — leaves no assertion at all, and a second, runtime-only
    comparison is that repair arriving by another route: nothing would fail, and the
    verification the whole design rests on would quietly be checking less than the tests do.

    ``resolved`` is ``None`` when the anchor addresses no text at all, which is what
    :meth:`~manicule.core.protocols.Parser.resolve` returns for
    :class:`~manicule.core.anchors.Unlocated` and for a located anchor that no longer fits
    these bytes. Both are failures of containment, not empty successes.

    Args:
        resolved: What the anchor resolves to over the source bytes, or ``None``.
        claimed: The text the chunk or block says is at that location.

    Returns:
        Whether ``claimed`` is present in ``resolved``, both under :func:`normalise`.
    """
    if resolved is None:
        return False
    claim = normalise(claimed)
    if not claim:
        # An empty claim is contained in everything, so without this a whitespace-only chunk
        # reaches the strongest verification level whether or not its anchor points anywhere.
        # Absent evidence is a failure of containment, exactly as ``resolved is None`` is.
        return False
    return claim in normalise(resolved)


__all__ = ["NORMALISER_VERSION", "contains_claimed_text", "normalise"]
