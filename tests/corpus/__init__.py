"""The fixture corpus, built by script rather than committed.

Generating fixtures keeps the repository small, makes every fixture's structure inspectable
as code instead of as an opaque binary, and lets the hostile cases — a zip bomb, a very large
generated file — exist without ever being committed. Only what cannot be generated faithfully
is checked in, under ``tests/corpus/committed/`` with its provenance recorded alongside it: a
fixture corpus is published with the project, so every committed file is public-domain, CC0,
or authored for this repository.

Run this module to write the corpus somewhere and look at it::

    python -m tests.corpus /tmp/manicule-corpus
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

COMMITTED = Path(__file__).parent / "committed"
"""Fixtures that cannot be generated faithfully, with ``PROVENANCE.md`` beside them."""

MAX_FIXTURE_BYTES = 256 * 1024
"""Size cap, with one deliberate exception per parser for a generated large file that
exercises the streaming path. Named ``*-large.*`` so the check below can see the intent."""


def generators() -> Iterator[tuple[str, Callable[[Path], None]]]:
    """Every ``build(dest)`` in this package, in a stable order.

    Discovered rather than listed, so adding a parser's fixtures is adding one module. A
    module without a ``build`` is a mistake worth failing on rather than skipping, because a
    silently skipped generator is a parser whose suite quietly stops having a corpus.
    """
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda entry: entry.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        build = getattr(module, "build", None)
        if not callable(build):
            msg = f"tests/corpus/{info.name}.py has no build(dest: Path) function"
            raise TypeError(msg)
        yield info.name, build  # pyright: ignore[reportUnknownVariableType] - checked callable above


def build_all(dest: Path) -> Path:
    """Write every generator's fixtures into ``dest/<name>/`` and return ``dest``."""
    for name, build in generators():
        target = dest / name
        target.mkdir(parents=True, exist_ok=True)
        build(target)
    _check_sizes(dest)
    return dest


def _check_sizes(dest: Path) -> None:
    oversized = [
        path
        for path in sorted(dest.rglob("*"))
        if path.is_file() and path.stat().st_size > MAX_FIXTURE_BYTES and "-large" not in path.stem
    ]
    if oversized:
        listed = ", ".join(f"{path.name} ({path.stat().st_size} bytes)" for path in oversized)
        msg = (
            f"fixture(s) over the {MAX_FIXTURE_BYTES}-byte cap: {listed}. Name the one "
            f"deliberate large fixture per parser '*-large.*' so the intent is visible."
        )
        raise AssertionError(msg)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: python -m tests.corpus <destination>\n")
        return 2
    destination = build_all(Path(argv[1]).expanduser().resolve())
    sys.stdout.write(f"corpus written to {destination}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script entry point
    raise SystemExit(main(sys.argv))
