"""The MIT claim, checked against installed metadata rather than asserted in prose.

manicule was GPL-3.0-or-later because one dependency was, and it is MIT again because that
dependency moved behind a package boundary rather than because anything about it changed. That
makes "manicule is MIT" a claim about a **dependency closure**, and a claim about a closure is
exactly the kind that stops being true in a routine-looking version bump.

``CONTRIBUTING.md`` says the license of a new dependency is checked by a person at selection
time, and that stays true — this file cannot tell you whether a library is *worth* adding. What
it can do is fail the moment one arrives whose terms contradict the license manicule publishes,
which is the half a person forgets rather than the half they deliberate over.

Two things are deliberately **not** checked here:

* **LGPL.** ``selectolax`` declares MIT and its wheel bundles an LGPL-2.1 engine beside a
  permissive one; manicule imports only the permissive one. That is invisible to distribution
  metadata, so a metadata check that claimed to cover it would be lying. ``docs/parsing.md``
  §12 carries it, and it is a redistribution condition on an image rather than a constraint on
  this project's license.
* **Whether a dependency should exist at all.** Adding one is a decision; this is a floor.
"""

from __future__ import annotations

import importlib.metadata as md
import tomllib
from pathlib import Path
from typing import Final

import pytest
from packaging.requirements import Requirement

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

COPYLEFT: Final[tuple[str, ...]] = (
    "GPL",
    "EUPL",
    "CDDL",
    "SSPL",
    "BUSL",
    "OSL",
    "CC-BY-NC",
    "CC-BY-SA",
)
"""Substrings that mean a license reaches manicule's own terms if it is in the closure.

``LGPL`` matches ``GPL`` and is filtered separately — dynamic use of an LGPL library is what
LGPL permits, and none of the three in question is a source obligation on this project.
"""


def _declared_license(name: str) -> str:
    """Every field a distribution might state its terms in, flattened into one string."""
    meta = md.metadata(name)
    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License")]
    parts = [
        meta.get("License-Expression") or "",
        (meta.get("License") or "")[:200],
        *classifiers,
    ]
    return " ".join(part for part in parts if part)


def _is_copyleft(blob: str) -> bool:
    without_lgpl = blob.replace("LGPL", "").replace("Lesser General Public", "")
    return any(token in without_lgpl for token in COPYLEFT)


def _runtime_closure(root: str, extras: frozenset[str]) -> tuple[set[str], set[str]]:
    """Every distribution ``root`` pulls in, and the ones that are not installed here.

    Markers are evaluated for *this* interpreter and platform, which is the point: on Apple
    silicon ``mlx-embeddings`` resolves and on the Linux runner it does not, and both are
    correct answers about the machine the check is running on.
    """
    seen: set[str] = set()
    unresolved: set[str] = set()
    stack: list[tuple[str, frozenset[str]]] = [(root, extras)]
    while stack:
        name, wanted = stack.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            requires = md.metadata(key).get_all("Requires-Dist") or []
        except md.PackageNotFoundError:
            seen.discard(key)
            unresolved.add(key)
            continue
        for raw in requires:
            requirement = Requirement(raw)
            environments = [{"extra": extra} for extra in wanted] or [{"extra": ""}]
            if requirement.marker is not None and not any(
                requirement.marker.evaluate(env) for env in environments
            ):
                continue
            stack.append((requirement.name, frozenset(requirement.extras)))
    return seen, unresolved


def _extras_of(pyproject: Path) -> frozenset[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return frozenset(data["project"].get("optional-dependencies", {}))


def test_manicule_declares_mit() -> None:
    """The published license, read from the built distribution rather than from `LICENSE`."""
    assert "MIT" in _declared_license("manicule")


UNCHECKABLE: Final[frozenset[str]] = frozenset({"sentence-transformers"})
"""Declared dependencies whose metadata is not on this machine, and therefore not checked.

Named rather than skipped over, because "no copyleft found" and "nothing was looked at" produce
the same green. There is exactly one, and it is deliberate: the `rerank` extra is torch, kept
out of the dev group so CI does not download two gigabytes for a model whose *rules* are tested
through a seam (see `pyproject.toml`). Its own closure is unchecked with it.

A new name appearing here is the thing to notice: it means a dependency was added that this
floor cannot see, and it wants either an install or a sentence saying why not.
"""


def test_the_closure_check_says_what_it_could_not_check() -> None:
    """A floor that silently shrinks is worse than no floor, because it still reports green."""
    _, unresolved = _runtime_closure("manicule", _extras_of(REPO_ROOT / "pyproject.toml"))

    assert unresolved <= UNCHECKABLE, (
        f"{sorted(unresolved - UNCHECKABLE)} are declared dependencies of manicule whose "
        f"metadata is not installed, so their terms were not checked and neither were their "
        f"own dependencies'. Install them, or add them to UNCHECKABLE with the reason."
    )


def test_nothing_in_manicules_closure_is_copyleft() -> None:
    """The whole point of the package split, in one assertion.

    A copyleft dependency here does not merely add an obligation — it contradicts the license
    manicule publishes, and it would do so silently. This is the check that turns "we moved the
    MLX backend out" from a thing that was done once into a thing that stays done.
    """
    installed, _ = _runtime_closure("manicule", _extras_of(REPO_ROOT / "pyproject.toml"))
    assert len(installed) > 50, (
        f"only {len(installed)} distributions resolved, which is too few to be manicule's real "
        f"closure — the walk is broken and this check is passing by looking at nothing"
    )
    offenders = {
        name: _declared_license(name)
        for name in sorted(installed)
        if _is_copyleft(_declared_license(name))
    }

    assert offenders == {}, (
        f"copyleft in manicule's own dependency closure: {offenders}. manicule is MIT "
        f"(`LICENSE`), and a copyleft dependency contradicts that rather than merely adding to "
        f"it. If the library is worth having, the shape of the answer is a separate "
        f"distribution — see packages/manicule-mlx — not a relicense."
    )


def test_the_mlx_backend_is_not_reachable_from_manicule() -> None:
    """The boundary itself: manicule must not depend on the package that carries the GPL.

    Distinct from the check above, because it fails for a different reason and with a different
    remedy. That one catches a copyleft library arriving in core; this one catches core growing
    an edge to `manicule-mlx` — which would import the GPL transitively while every individual
    dependency still looked permissive.
    """
    installed, _ = _runtime_closure("manicule", _extras_of(REPO_ROOT / "pyproject.toml"))

    assert "manicule-mlx" not in installed, (
        "manicule depends on manicule-mlx, which is GPL-3.0-or-later. The dependency runs the "
        "other way: manicule-mlx depends on manicule, and claims the `embedder.mlx` slot "
        "through the `manicule.plugins` entry-point group. Nothing under src/manicule may name "
        "it — an extra pointing at it would make `manicule is MIT` need a footnote."
    )


def test_the_mlx_package_declares_the_license_it_carries() -> None:
    """It is GPL-3.0-or-later, and that is not an accident to be tidied away.

    Asserted because the failure mode is somebody "fixing" the inconsistency in the direction
    that makes the repository look uniform: relabeling this package MIT while it still links
    `mlx-embeddings` would be the only genuinely wrong answer available.
    """
    blob = _declared_license("manicule-mlx")

    assert "GPL-3.0-or-later" in blob, (
        f"manicule-mlx declares {blob!r}. It links mlx-embeddings, which is GPL-3.0, so this "
        f"package carries the copyleft obligation and has to say so."
    )


@pytest.mark.parametrize("package", ["manicule-plugin-example", "manicule-plugin-hostile"])
def test_the_example_plugins_are_mit(package: str) -> None:
    """They exist to be copied from, so their terms are advice about a reader's own plugin."""
    assert "MIT" in _declared_license(package)
