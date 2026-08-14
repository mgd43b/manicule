"""Test configuration.

The environment isolation and the ``settings`` fixture are shipped in
:mod:`manicule.testing.fixtures` rather than written here, so that a plugin author gets the
same setup with one import and manicule's own suite proves that import works.

The storage fixtures are local: they build a real migrated database in a temporary directory,
which is a manicule-internal concern rather than something a plugin author needs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from manicule.testing.fixtures import manicule_environment, settings
from tests.corpus import build_all
from tests.storage_helpers import data_dir, engine, store

__all__ = [
    "color_environment",
    "corpus",
    "data_dir",
    "engine",
    "grammar_cache",
    "grammar_fetch_never_sleeps",
    "manicule_environment",
    "model_cache",
    "settings",
    "store",
    "vocabulary_cache",
]


@pytest.fixture(scope="session", autouse=True)
def model_cache() -> None:
    """Pin the Hugging Face cache to this machine's real one, for the whole session.

    The same hazard as ``grammar_cache`` below, arriving through a different library.
    ``manicule_environment`` redirects ``XDG_CACHE_HOME`` at every test, and recent
    ``huggingface_hub`` resolves its cache through that variable — so a model sitting on disk
    becomes invisible the moment a test imports the hub lazily, and every embedding suite skips
    on a machine where the weights are right there. Worse in CI, where the pre-seed step would
    download several hundred megabytes into a directory nothing later reads, and the suite would
    report green having checked nothing.

    Session-scoped and autouse so it runs while the environment is still the real one. The
    resolved path is written back to ``HF_HUB_CACHE``, which takes precedence over both
    ``HF_HOME`` and the XDG variable, so a later redirection cannot move it.
    """
    try:
        import huggingface_hub.constants as hub  # noqa: PLC0415 - an embeddings extra
    except ImportError:
        return
    os.environ["HF_HUB_CACHE"] = str(hub.HF_HUB_CACHE)


@pytest.fixture(scope="session", autouse=True)
def grammar_cache() -> None:
    """Pin the tree-sitter grammar cache to this machine's real one, for the whole session.

    ``manicule_environment`` redirects ``XDG_CACHE_HOME`` at every test, which is right for
    everything manicule writes and wrong for this one thing. Grammars are not per-test state:
    they are a machine resource, pre-seeded once by ``manicule doctor --fix`` or by CI, and a
    per-test cache directory is empty by construction — so every assertion about the code
    parser would skip or refuse, on a machine where the grammars are sitting right there.

    It was invisible for the worst possible reason: the pack resolves its cache through
    ``platformdirs``, which honors ``XDG_CACHE_HOME`` on Linux and uses
    ``~/Library/Caches`` on macOS. The suite therefore passed on a developer's machine and
    failed on CI, which is exactly the "one corpus, two behaviors" split that
    ``manicule.parsers.grammars`` exists to prevent — arriving through the test harness rather
    than through the code.

    Session-scoped and autouse so it runs before any function-scoped fixture has redirected
    anything: at this point the environment is the real one, so the path captured here is the
    real cache, and every later ``configure_pack`` call restores that same path explicitly.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


@pytest.fixture(scope="session", autouse=True)
def grammar_fetch_never_sleeps() -> Iterator[None]:
    """Take the backoff out of :func:`~manicule.parsers.grammars.prefetch` for the suite.

    Several tests point the manifest at the discard port precisely so that a fetch fails at
    once, and they would each now wait out the real retry policy instead — five seconds apiece
    to re-observe a message they already have. The waiting is the only part being removed: the
    retry still runs, it simply runs its attempts back to back.

    The tests that are *about* the retry set their own policy over this one, which is the point
    of it being read at call time rather than captured. Nothing here can make a fetch succeed
    that would otherwise fail, so a suite arranged this way cannot report a working download
    that a real one would not perform.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(grammars, "FETCH_RETRY_DELAYS", (0.0, 0.0))
        yield


@pytest.fixture(scope="session", autouse=True)
def vocabulary_cache() -> None:
    """Pin the tiktoken vocabulary cache to this machine's real one, for the whole session.

    The third instance of the hazard ``model_cache`` and ``grammar_cache`` above describe, and
    it only became one when the vocabulary cache stopped living in the system temporary
    directory. While it did, ``manicule_environment``'s redirected ``XDG_CACHE_HOME`` could
    not hide it — the cache was somewhere no test fixture had any reason to move. Now that the
    default is durable, and therefore *under* the directory every test redirects, a per-test
    cache is empty by construction and the whole offline-bundle suite skips on a machine where
    the vocabularies are sitting right there.

    Session-scoped and autouse, so it runs while the environment is still the real one and the
    path captured is the real cache. Written into ``TIKTOKEN_CACHE_DIR`` rather than restored
    per test, because that variable is what both ``tiktoken`` and
    ``manicule.vocabularies.cache_directory`` consult first, so one assignment makes every
    later resolution agree.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    os.environ[vocabularies.CACHE_DIR_ENV] = str(vocabularies.cache_directory())


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every generator's fixtures, built once per session.

    Built rather than committed: it keeps the repository small, makes each fixture's
    structure reviewable as code, and lets the hostile cases exist without being stored.
    """
    return build_all(tmp_path_factory.mktemp("corpus"))


CLEARED_TERMINAL_VARIABLES: frozenset[str] = frozenset(
    {"NO_COLOR", "FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE", "COLORTERM"}
)
"""Variables that decide whether output is a terminal, and whether it is colored.

Every one is cleared, so the suite states its own answer instead of inheriting the caller's.

The list is **not** trusted to be complete on its own —
``tests/app/test_cli.py::test_the_color_isolation_accounts_for_every_variable_rich_reads``
checks it against what Rich's source actually consults. That check exists because a
hand-written list here has already been wrong twice: ``TERM`` was missing, which is the bug
this fixture was written for, and ``TTY_COMPATIBLE`` was missing from the first version of the
fix — Rich reads it *before* ``FORCE_COLOR``, so ``TTY_COMPATIBLE=0`` reintroduced the original
failure exactly.
"""

BASELINE_TERM = "xterm-256color"
"""The terminal the suite says it is running on, whatever the caller's shell says.

Set rather than cleared, because an absent ``TERM`` is not a neutral answer — it is a terminal
that declares no capability. ``TERM`` is not a color switch, which is why it was missed: it is
a *capability*, and ``dumb`` tells Rich the stream cannot render ANSI at all, which beats
``FORCE_COLOR`` saying the stream is a terminal.
"""


@pytest.fixture(autouse=True)
def color_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test the same color environment, rather than the caller's.

    Whether manicule colors its output is decided entirely by environment variables, so
    without this the suite's result depends on the shell that started it. It did: a positive
    control asserting that human output *is* colored passed in an ordinary terminal and failed
    under ``TERM=dumb``, which is what a shell buffer that cannot render ANSI reports.

    That failure mattered more than a flaky test. The control exists so that the neighboring
    assertion — that ``--json`` emits no ANSI — is not vacuous, and a control that inverts with
    the environment is the vacuum wearing a disguise: in a ``TERM=dumb`` shell it fails, the
    obvious repair is to delete it, and the assertion it was protecting quietly stops meaning
    anything.

    The baseline is **no opinion, on a capable terminal**: the switches are cleared, so nothing
    is forced either way, and ``TERM`` names a terminal that can render ANSI. A test wanting
    color sets ``FORCE_COLOR`` itself and now gets it deterministically; a test wanting none
    needs nothing, because the runner's stdout is not a terminal.

    Size is deliberately left alone. ``COLUMNS`` and ``LINES`` are inherited, and the tests that
    care pin them per case — a width this fixture chose would silently become the width every
    layout assertion was written against.
    """
    for name in CLEARED_TERMINAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", BASELINE_TERM)
