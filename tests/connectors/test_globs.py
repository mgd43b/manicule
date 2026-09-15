"""Path patterns: what configuration may say, and what a pattern then matches.

Two connectors admit and refuse paths with these functions, so the rules are worth pinning
directly rather than inferring them from a walk or a Git tree. Until the filesystem connector
grew an ``exclude`` the matching lived inside ``git_site`` and was exercised only through a
fixture that happened to contain a draft — which is coverage of the caller, not of the rule.
"""

from __future__ import annotations

import pytest

from manicule.connectors import globs


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("notes/retry.md", "notes/*.md"),
        ("notes/retry.md", "**/*.md"),
        ("retry.md", "**/*.md"),
        ("a/b/c/retry.md", "**/*.md"),
        ("archive/old.md", "archive/**"),
        ("deep/archive/old.md", "**/archive/**"),
        ("archive/old.md", "**/archive/**"),
        ("README.md", "README.md"),
    ],
)
def test_a_pattern_matches_the_paths_it_names(path: str, pattern: str) -> None:
    """Including the two cases ``fnmatch`` alone gets wrong.

    ``**/*.md`` and ``**/archive/**`` both read, to ``fnmatch``, as requiring at least one
    directory before the rest — so a top-level ``retry.md`` or ``archive/`` would survive a
    pattern written to catch them anywhere. Anybody writing the pattern means both depths.
    """
    assert globs.matches(path, pattern)


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("docs/README.md", "README.md"),
        ("notes/archive.md", "**/archive/**"),
        ("notes/retry.txt", "**/*.md"),
        ("archive", "archive/**"),
    ],
)
def test_a_pattern_does_not_match_a_path_it_merely_resembles(path: str, pattern: str) -> None:
    """A pattern anchored at the root stays anchored, and a subtree pattern names the subtree.

    ``README.md`` naming every README at every depth would make the root-file case unsayable,
    and it is the one an operator reaches for first.
    """
    assert not globs.matches(path, pattern)


@pytest.mark.parametrize(
    ("pattern", "here", "deeper", "recursive"),
    [
        ("*.md", "retry.md", "notes/retry.md", "**/*.md"),
        ("notes/*.md", "notes/retry.md", "notes/deep/retry.md", "notes/**/*.md"),
        ("archive/*", "archive/old.md", "archive/deep/old.md", "archive/**"),
    ],
)
def test_a_single_star_stops_at_a_separator(
    pattern: str, here: str, deeper: str, recursive: str
) -> None:
    """``*`` is one path segment, which is the whole difference between a level and a subtree.

    ``fnmatch`` translates ``*`` to ``.*``, which spans ``/`` — so ``archive/*`` silently meant
    the entire subtree and ``*.md`` meant every Markdown file in the corpus. An operator who
    writes ``archive/*`` and gets the subtree has been given something they did not ask for, and
    on an ``exclude`` that is content disappearing from the index without being named.

    ``**`` is how the subtree is asked for, and it still crosses separators — the pair is the
    point, and a matcher with only one of them cannot express both.
    """
    assert globs.matches(here, pattern)
    assert not globs.matches(deeper, pattern)
    assert globs.matches(deeper, recursive), "and `**` is how the subtree is asked for"


def test_no_patterns_match_nothing() -> None:
    """The default for a connector that ships none, so it is the case that runs most often."""
    assert not globs.matches_any("anything/at/all.md", ())


def test_an_exclusion_beats_an_inclusion() -> None:
    """The only ordering that lets a broad ``include`` be corrected in place.

    The other way round, every exclusion would have to be restated as a narrower include, and
    the pattern list stops describing intent the moment it has to describe arithmetic.
    """
    assert globs.admitted("docs/guide.md", include=("**/*.md",), exclude=())
    assert not globs.admitted(
        "docs/drafts/guide.md", include=("**/*.md",), exclude=("**/drafts/**",)
    )
    assert not globs.admitted("docs/guide.txt", include=("**/*.md",), exclude=())


@pytest.mark.parametrize(
    "pattern",
    ["**/*.md", "archive/**", "a/b/c.md", "README.md", "x" * 1_024],
)
def test_a_usable_pattern_is_returned_unchanged(pattern: str) -> None:
    """Validation normalizes nothing. A pattern that survives is the one that was written."""
    assert globs.path_glob(pattern) == pattern


@pytest.mark.parametrize(
    ("pattern", "because"),
    [
        ("", "empty"),
        (" notes/**", "padded"),
        ("notes/** ", "padded"),
        ("x" * 1_025, "over the length bound"),
        ("/absolute/**", "absolute"),
        ("notes\\deep\\**", "Windows-separated"),
        ("notes/\0/**", "holds a NUL"),
        ("../outside/**", "traverses"),
        ("notes/../outside", "traverses"),
        ("./notes/**", "a dot segment"),
    ],
)
def test_a_pattern_that_could_not_mean_what_it_says_is_refused(pattern: str, because: str) -> None:
    """Refused where an operator is still looking at the configuration file.

    Every one of these either matches nothing for ever or reaches for something outside the
    boundary patterns are relative to. Accepted, they are indistinguishable from a pattern that
    is simply not being hit yet — which is the shape of a corpus quietly indexing what somebody
    wrote a pattern to keep out.
    """
    with pytest.raises(ValueError, match="path patterns"):
        globs.path_glob(pattern)
    assert because  # named in the parametrization so a failure says which rule lapsed
