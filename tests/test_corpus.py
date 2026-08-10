"""The fixture corpus itself, which every parser suite is measured against.

A corpus that differs between two builds cannot be used to assert that a parser does not
differ between two builds. Every check below is about the corpus rather than about a parser,
and each one guards a way the corpus could stop meaning anything while every parser suite
still reported green.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from tests.corpus import MAX_FIXTURE_BYTES, build_all, generators

MIN_FIXTURES = 150
"""A floor on the corpus size.

Every ratio and every "for each fixture" assertion in the parser suites divides by this
corpus. A generator that silently stopped writing files would make all of them pass, so the
size is asserted rather than assumed. Raise it when the corpus grows; it is a floor, not a
count.
"""


_A_YEAR_LATER = 1_800_000_000.0
"""An instant a long way from now, used to build the corpus a second time.

Far enough that no granularity hides it: a zip member's timestamp resolves to two seconds and
a PDF's ``/CreationDate`` to one, so two builds a few hundred milliseconds apart agree even
when the generator is reading the clock.
"""


def _move_the_clock_forward(patch: pytest.MonkeyPatch) -> None:
    """Make every clock a generator can read report a different instant.

    ``time.time`` is what reportlab stamps into a PDF's dates, and ``time.localtime`` is what
    :mod:`zipfile` fills a member's ``date_time`` from. Patching both covers every mechanism
    that has actually gone wrong here, and any generator that ignores the clock — which is all
    of them, once they are correct — is unaffected.
    """

    def later(seconds: float | None = None) -> time.struct_time:
        return time.gmtime(_A_YEAR_LATER if seconds is None else seconds)

    patch.setattr(time, "time", lambda: _A_YEAR_LATER)
    patch.setattr(time, "localtime", later)


def _digest(root: Path) -> dict[str, str]:
    """Every file under ``root``, keyed by relative path, hashed."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_whole_corpus_is_byte_identical_on_every_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property everything else in the suite rests on, checked over every generator.

    Not a hypothetical: four separate sources of drift were live here and each was invisible
    until this ran. ``reportlab`` stamps the wall clock into a PDF's ``/CreationDate`` and
    derives an encrypted file's key from ``os.urandom``; pdfium writes a fresh trailer ``/ID``
    on every save; and :mod:`zipfile` fills a member's timestamp from the local clock whenever
    it is handed a name rather than a :class:`zipfile.ZipInfo`, which is what python-docx and
    python-pptx do — so the Office fixtures differed between two runs *and* between two
    timezones. None of it made a test fail. It made the corpus unusable as a baseline, which
    is a slower and quieter kind of wrong.

    **The clock is moved between the two builds, and that is what makes this a check.**
    Building twice back to back compares two instants a few hundred milliseconds apart, and a
    zip timestamp has two-second granularity — so a generator that stamps the wall clock
    agrees with itself most of the time and this test passes for the wrong reason. It did:
    with the fixed timestamps removed from one generator it still went green here, and the
    drift surfaced on CI instead. Moving the clock a year makes the comparison a property of
    the generators rather than of how fast the machine is.
    """
    first, second = tmp_path / "first", tmp_path / "second"
    build_all(first)
    with monkeypatch.context() as later:
        _move_the_clock_forward(later)
        build_all(second)

    left, right = _digest(first), _digest(second)

    assert sorted(left) == sorted(right), "the two builds wrote different files"
    differing = sorted(name for name in left if left[name] != right[name])
    assert not differing, (
        f"{len(differing)} fixture(s) differ between two builds of the same corpus: "
        f"{differing[:8]}. A corpus that churns cannot be used to assert that a parser "
        f"does not"
    )


def test_the_corpus_is_not_quietly_shrinking(tmp_path: Path) -> None:
    """Every per-parser ratio divides by this corpus, so its size is an assertion."""
    built = _digest(build_all(tmp_path / "corpus"))

    assert len(built) >= MIN_FIXTURES, (
        f"the corpus built {len(built)} fixtures, fewer than the {MIN_FIXTURES} the suites "
        f"claim to cover. Every 'for each fixture' assertion passes over an empty directory"
    )


def test_every_generator_writes_something(tmp_path: Path) -> None:
    """A generator that stops producing files takes its parser's coverage with it.

    ``build_all`` discovers generators rather than listing them, which is what keeps adding a
    parser to one module — and is also what would let a generator that quietly writes nothing
    go unnoticed, because nothing enumerates what it was supposed to write.
    """
    root = build_all(tmp_path / "corpus")

    empty = sorted(name for name, _ in generators() if not any((root / name).iterdir()))

    assert not empty, f"generator(s) wrote no fixtures: {empty}"


def test_no_fixture_exceeds_the_size_cap_without_saying_so(tmp_path: Path) -> None:
    """The cap keeps the repository small and the suite fast.

    ``build_all`` enforces it, and this is the check that the enforcement is real rather than
    a function nobody calls: one deliberate large fixture per parser is allowed and is named
    ``*-large.*`` so the intent is visible in the filename.
    """
    root = build_all(tmp_path / "corpus")

    oversized = sorted(
        path.name
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size > MAX_FIXTURE_BYTES
    )

    assert all("-large" in name for name in oversized), (
        f"fixture(s) over the {MAX_FIXTURE_BYTES}-byte cap without the '-large' marker: "
        f"{[name for name in oversized if '-large' not in name]}"
    )
