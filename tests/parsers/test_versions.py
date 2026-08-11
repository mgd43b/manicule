"""The parser version table, and the three ways it could be quietly wrong.

A fingerprint that never changes is worse than no fingerprint at all: it asserts that two
generations of extracted text are the same generation. Every check here is about a way this
table could stop moving without anybody noticing.
"""

from __future__ import annotations

import fnmatch
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any, cast

import pytest
from packaging.utils import canonicalize_name

from manicule.core.fingerprints import ParseFingerprint
from manicule.parsers.plugin import PARSERS as REGISTERED
from manicule.parsers.versions import (
    PARSERS,
    current_parse_fingerprints,
    distributions_recorded,
    parse_fingerprint,
)

DEPENDABOT = Path(__file__).resolve().parents[2] / ".github" / "dependabot.yml"


def dependency_groups() -> dict[str, dict[str, list[str]]]:
    """Dependabot's groups for the Python ecosystem, read from the file CI reads.

    ``ruamel.yaml`` ships no type information, which is why ``pyproject.toml`` relaxes
    ``reportUnknownMemberType`` under ``src/manicule/parsers``. That relaxation is deliberately
    scoped to the directory that touches these libraries, so the loss is taken here on one
    line rather than by widening it to a whole test package — and
    ``reportUnnecessaryTypeIgnoreComment`` is an error, so the suppression disappears the day
    upstream ships stubs.
    """
    from ruamel.yaml import YAML  # noqa: PLC0415 - test-only, and only the parsers extra has it

    loaded = cast(
        "dict[str, list[dict[str, Any]]]",
        YAML(typ="safe").load(DEPENDABOT.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )
    return next(
        cast("dict[str, dict[str, list[str]]]", block["groups"])
        for block in loaded["updates"]
        if block["package-ecosystem"] == "uv"
    )


def test_every_registered_parser_has_a_recorded_version() -> None:
    """A parser with no entry records no lineage, so its documents never re-parse.

    The failure is silent by construction: the parser works, the documents index, and the day
    its library changes what they say, nothing selects them for repair.
    """
    registered = {registration.name for registration in REGISTERED}

    assert set(PARSERS) == registered, (
        "manicule.parsers.versions.PARSERS and the registered parser set have diverged"
    )


def test_no_parser_records_a_version_for_a_library_it_does_not_use() -> None:
    """The other direction: a spurious entry invalidates documents for nothing.

    Stated here independently of :data:`PARSERS` rather than derived from it, so that the two
    have to be edited together. Deriving it from the parsers' import statements was the
    obvious alternative and does not work: ``lxml`` is imported by neither ``word.py`` nor
    ``slides.py``: it is what ``python-docx`` and ``python-pptx`` parse OOXML *with*, which is
    exactly why it is easy to leave out and exactly why leaving it out would be silent.
    """
    imported_by = {
        "lxml": ("docx", "pptx"),
        "markdown-it-py": ("markdown",),
        "nbformat": ("notebook",),
        "python-calamine": ("spreadsheet",),
        "python-docx": ("docx",),
        "python-pptx": ("pptx",),
        "pypdfium2": ("pdf",),
        "ruamel-yaml": ("structured",),
        "selectolax": ("html", "email"),
        "tree-sitter": ("sourcecode",),
    }

    for distribution, parsers in imported_by.items():
        recorded = {name for name, entry in PARSERS.items() if distribution in entry.distributions}
        assert recorded == set(parsers), (
            f"{distribution} is recorded for {sorted(recorded)} and used by {sorted(parsers)}"
        )
    assert set(imported_by) == distributions_recorded()


@pytest.mark.parametrize("distribution", sorted(distributions_recorded()))
def test_every_recorded_distribution_name_is_canonical_and_resolves(distribution: str) -> None:
    """The PEP 503 trap, in both directions.

    ``ruamel.yaml`` is the name in ``pyproject.toml`` and ``ruamel-yaml`` is the name it
    installs under. A table written in the first form and looked up in the second is a lookup
    that raises; a table written in either and *defaulted* on failure is a version that never
    moves. So the name has to be canonical, and it has to resolve to something real — checked
    rather than assumed, because "it worked on my machine" and "it silently returned a
    placeholder" look identical from the outside.
    """
    assert canonicalize_name(distribution) == distribution, (
        f"{distribution!r} is not its own PEP 503 canonical form "
        f"({canonicalize_name(distribution)!r}); the recorded key and the lookup would differ"
    )
    resolved = installed_version(distribution)
    assert resolved, f"{distribution} resolved to an empty version"


def test_a_fingerprint_carries_the_installed_versions_and_nothing_invented() -> None:
    """Values come from the environment, not from a constant beside them."""
    pdf = parse_fingerprint("pdf")

    assert pdf is not None
    assert pdf.parser == "pdf"
    assert pdf.libraries == {"pypdfium2": installed_version("pypdfium2")}
    assert pdf.version


def test_a_parser_with_no_parsing_libraries_records_an_empty_map() -> None:
    """A real answer rather than a gap: nothing about ``adf`` can move under a bump."""
    adf = parse_fingerprint("adf")

    assert adf is not None
    assert adf.libraries == {}
    assert "adf" in adf.describe()


def test_a_parser_manicule_does_not_ship_records_nothing() -> None:
    """``None`` rather than a placeholder.

    A third-party parser's version is not something this repository can read, and a
    fingerprint saying ``unknown`` would be a constant standing in for a thing that varies —
    which is the defect ``tokenizer_id`` used to have.
    """
    assert parse_fingerprint("some-plugin-parser") is None


def test_a_missing_distribution_raises_rather_than_defaulting() -> None:
    """A version that falls back to a constant is the constant this table replaced."""
    with pytest.raises(PackageNotFoundError):
        installed_version("a-distribution-that-is-not-installed")


def test_the_current_set_covers_every_shipped_parser() -> None:
    """What ``reindex --re-parse`` compares against. A partial set is a repair that cannot end."""
    current = current_parse_fingerprints()

    assert {fingerprint.parser for fingerprint in current} == set(PARSERS)
    assert len({fingerprint.canonical() for fingerprint in current}) == len(current), (
        "two parsers sharing one canonical form would make a bump in either invalidate both"
    )


def test_a_library_bump_moves_only_the_parsers_that_use_it() -> None:
    """Selective invalidation, at the level the table is responsible for."""
    before = parse_fingerprint("pdf")
    after = ParseFingerprint(parser="pdf", version="1", libraries={"pypdfium2": "0.0.0"})
    markdown = parse_fingerprint("markdown")

    assert before is not None
    assert markdown is not None
    assert not before.matches(after)
    assert before.changed_fields(after) == {"libraries"}
    assert markdown.canonical() != before.canonical()
    assert markdown.matches(parse_fingerprint("markdown"))  # pyright: ignore[reportArgumentType]


def test_every_recorded_distribution_is_grouped_as_index_affecting() -> None:
    """The two halves of the same fact, kept in step.

    A distribution that reaches a parse fingerprint decides stored text. If Dependabot does
    not group it under ``index-affecting-extraction``, its bump arrives in a pull request
    titled "the runtime group" and re-parses a corpus on the strength of a routine-looking
    review. The patterns are matched the way Dependabot matches them — ``ruamel*`` covers
    ``ruamel-yaml`` — because a pattern that matches nothing is exactly how this went wrong
    before.
    """
    groups = dependency_groups()
    patterns = [
        *groups["index-affecting-extraction"]["patterns"],
        *groups["index-affecting-chunking"]["patterns"],
    ]

    ungrouped = sorted(
        name
        for name in distributions_recorded()
        if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    )
    assert ungrouped == [], (
        f"{ungrouped} decide stored text and no index-affecting pattern matches them"
    )


def test_the_runtime_group_excludes_everything_a_parse_fingerprint_records() -> None:
    """Group order alone is not the guard; the wildcard group repeats the exclusions.

    Relying on ordering means one edit away from a library that decides stored text arriving
    under a name that says it does not.
    """
    excluded = dependency_groups()["runtime"]["exclude-patterns"]

    leaked = sorted(
        name
        for name in distributions_recorded()
        if not any(fnmatch.fnmatch(name, pattern) for pattern in excluded)
    )
    assert leaked == [], f"{leaked} would be swept into the runtime group by its wildcard"
