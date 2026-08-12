"""How the offline vocabulary suite gets a real bundle, and when a missing one is a failure.

A bundle is built from vocabulary files that are already in this machine's ``tiktoken`` cache,
so a suite that asserts anything about one needs those files present. There are two honest
ways to handle their absence and only one of them is right per environment:

- **A developer's machine with a cold cache** should skip. Nothing is wrong; the vocabularies
  have simply never been fetched, and failing would make a first checkout red for a reason
  that has nothing to do with the change under test.
- **CI, which pre-seeds vocabularies as an explicit step**, must fail. A skipped conformance
  suite certifies nothing, and this suite exists precisely because an air-gapped install's
  ability to answer a question was green everywhere while being checked nowhere.

:data:`REQUIRE_BUNDLE_ENV` is what tells the two apart, and it is **deliberately outside
manicule's ``MANICULE_`` namespace**: ``manicule_environment`` deletes every variable with that
prefix before each test, so a switch named that way is scrubbed before it is ever read and the
job goes green having skipped everything. That mistake has already been made twice in this
repository — in the embedding-model switch and in the grammar one — and the name here is
chosen so it cannot be made a third time.

Bundles are built with the shipped builder rather than hand-assembled. A fixture that wrote
its own ``vocabularies.json`` would test the reader against the fixture's idea of the format,
which is how a format ends up with two definitions and no failing test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest

from manicule import vocabularies
from manicule.vocabularies import bundle as bundles

REQUIRE_BUNDLE_ENV: Final = "REQUIRE_VOCABULARY_BUNDLE"
"""Set to any non-empty value to turn this suite's skips into failures. CI sets it."""

BUNDLE_REQUIRED: Final = bool(os.environ.get(REQUIRE_BUNDLE_ENV, "").strip())
"""Read at import, before any fixture has had a chance to touch the environment.

Belt and braces against the scrubbing bug the name already avoids: reading it here means a
future fixture that deletes the variable cannot silently disarm the switch.
"""

BUNDLE_ENCODINGS: Final[tuple[str, ...]] = ("cl100k_base", "o200k_base")
"""What the suite bundles: both encodings manicule actually asks for, 5.3 MB between them.

Two rather than one, for the same reason the grammar suite bundles two languages: a bundle
with a single entry cannot show that seeding a subset leaves the rest alone, and cannot
distinguish "the bundle was used" from "the one file happened to be there". These two are also
the pair that made the original defect confusing — ``cl100k_base`` is the chunker's, so an
install could index perfectly and then fail on ``o200k_base`` at the first question.
"""


def require_source_vocabularies(encodings: tuple[str, ...] = BUNDLE_ENCODINGS) -> None:
    """Skip unless ``encodings`` are in this machine's cache — or fail, under the CI switch."""
    absent = vocabularies.missing_vocabularies(encodings)
    if not absent:
        return
    detail = f"{list(absent)} are not in the tiktoken cache at {vocabularies.cache_directory()}"
    if BUNDLE_REQUIRED:
        pytest.fail(
            f"{detail}, and {REQUIRE_BUNDLE_ENV} is set. Pre-seed them before running this "
            f"suite; a skipped offline-vocabulary suite reports green while proving nothing "
            f"about an air-gapped install."
        )
    pytest.skip(f"{detail}. Pre-seed them to enable the offline vocabulary bundle suite.")


def build_bundle(
    destination: Path, encodings: tuple[str, ...] = BUNDLE_ENCODINGS
) -> bundles.VocabularyBundle:
    """A real bundle at ``destination``, built by the shipped builder from this machine."""
    require_source_vocabularies(encodings)
    return bundles.build(encodings, destination)


__all__ = [
    "BUNDLE_ENCODINGS",
    "BUNDLE_REQUIRED",
    "REQUIRE_BUNDLE_ENV",
    "build_bundle",
    "require_source_vocabularies",
]
