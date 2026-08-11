"""How the offline grammar suite gets a real bundle, and when a missing one is a failure.

A bundle is built from compiled grammar libraries, so a suite that asserts anything about one
needs those libraries on the machine. There are two honest ways to handle their absence and
only one of them is right per environment:

- **A developer's machine with an empty grammar cache** should skip. Nothing is wrong; the
  grammars have simply never been fetched, and failing would make a first checkout red for a
  reason that has nothing to do with the change under test.
- **CI, which pre-seeds grammars as an explicit step**, must fail. A skipped conformance suite
  certifies nothing, and this suite exists precisely because the code parser's guarantees were
  once green everywhere while being checked nowhere.

:data:`REQUIRE_BUNDLE_ENV` is what tells the two apart, and it is **deliberately outside
manicule's ``MANICULE_`` namespace**: ``manicule_environment`` deletes every variable with that
prefix before each test, so a switch named that way is scrubbed before it is ever read and the
job goes green having skipped everything. That mistake has already been made once in this
repository, in the equivalent switch for embedding models; the name here is chosen so it
cannot be made again.

Bundles are built with the shipped builder rather than hand-assembled. A fixture that wrote its
own ``grammars.json`` would test the reader against the fixture's idea of the format, which is
how a format ends up with two definitions and no failing test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest

from manicule.parsers import grammar_bundle, grammars

REQUIRE_BUNDLE_ENV: Final = "REQUIRE_GRAMMAR_BUNDLE"
"""Set to any non-empty value to turn this suite's skips into failures. CI sets it."""

BUNDLE_REQUIRED: Final = bool(os.environ.get(REQUIRE_BUNDLE_ENV, "").strip())
"""Read at import, before any fixture has had a chance to touch the environment.

Belt and braces against the scrubbing bug the name already avoids: reading it here means a
future fixture that deletes the variable cannot silently disarm the switch.
"""

BUNDLE_LANGUAGES: Final[tuple[str, ...]] = ("python", "rust")
"""What the suite bundles. Two languages, about 2 MB, because every property under test —
resolution, verification, seeding, refusal — is a property of the bundle rather than of its
size, and copying the full declared set into a temporary directory per test would be 84 MB of
I/O to learn the same thing.

Two rather than one: a bundle with a single entry cannot show that seeding a subset leaves the
rest alone, and cannot distinguish "the bundle was used" from "the one language happened to
work"."""

NAME_TRAP_LANGUAGE: Final = "csharp"
"""The language whose library file is *not* ``libtree_sitter_csharp``.

It is ``libtree_sitter_c_sharp``, so a builder that constructs the file name from the language
key silently omits C# from every bundle it writes. Kept out of :data:`BUNDLE_LANGUAGES` because
its library is 6 MB and only one test needs it."""

UNREACHABLE_MANIFEST: Final = "http://127.0.0.1:9/manicule-tests-must-not-download.json"
"""The discard port on loopback. A fetch through this refuses immediately, so a test that
begins downloading fails rather than passing slowly against the real release."""


def source_cache() -> Path:
    """This machine's populated grammar cache, which is what a bundle is built from.

    The pack is put back on its default configuration first. A bundle is built from the real
    machine's grammars, and a test that has just pointed the pack at an empty directory would
    otherwise build an empty bundle and then assert against it — a fixture agreeing with itself,
    which is the failure mode this repository has hit more often than any code bug.
    """
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)
    return grammars.cache_directory()


def require_source_grammars(languages: tuple[str, ...] = BUNDLE_LANGUAGES) -> None:
    """Skip unless ``languages`` are on this machine — or fail, under the CI switch."""
    grammars.configure_pack(grammars.DECLARED_LANGUAGES)
    absent = grammars.missing_grammars(languages)
    if not absent:
        return
    detail = f"{list(absent)} are not in the grammar cache at {source_cache()}"
    if BUNDLE_REQUIRED:
        pytest.fail(
            f"{detail}, and {REQUIRE_BUNDLE_ENV} is set. Pre-seed the declared set before "
            f"running this suite; a skipped offline-bundle suite reports green while proving "
            f"nothing about an air-gapped install."
        )
    pytest.skip(f"{detail}. Pre-seed them to enable the offline grammar bundle suite.")


def build_bundle(
    destination: Path, languages: tuple[str, ...] = BUNDLE_LANGUAGES
) -> grammar_bundle.GrammarBundle:
    """A real bundle at ``destination``, built by the shipped builder from this machine."""
    require_source_grammars(languages)
    return grammar_bundle.build(languages, destination, source=source_cache())


__all__ = [
    "BUNDLE_LANGUAGES",
    "BUNDLE_REQUIRED",
    "NAME_TRAP_LANGUAGE",
    "REQUIRE_BUNDLE_ENV",
    "UNREACHABLE_MANIFEST",
    "build_bundle",
    "require_source_grammars",
    "source_cache",
]
