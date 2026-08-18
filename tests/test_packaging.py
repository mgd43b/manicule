"""What a release ships, held against what this repository says it ships.

Three facts decide whether `uv tool install "manicule[all]"` produces a working program, and
none of them is checked by running the test suite — every other test here runs from the source
tree, with the dev group installed and `src/` on the path, which is not what anybody installs.

* The `all` extra and the Dockerfile's `EXTRAS` are the same set. They are two spellings of one
  decision — "what an installation of manicule contains" — written in two files that cannot see
  each other.
* The console script survives an installation without the `serve` extra. `manicule.entry` exists
  for that and would be silently pointless if the entry point were ever pointed back at
  `manicule.cli.main:main`, which is the obvious-looking simplification.
* The built-in plugins admit the version that is running. Every one of them declares
  `core_version=">=0.1,<0.2"`, and release-please bumping `pyproject.toml` to 0.2.0 would ship a
  manicule whose own parsers refuse to load — an entirely mechanical failure that no other test
  in this repository would notice, on the one commit nobody rehearses.

**What this does not do.** It does not build a wheel; that costs seconds and needs network, and
the `dist` job in ci.yml does it on every pull request against the artifact itself. This is the
part that can be checked from the tree, so it is checked where a developer sees it fail.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from manicule.core.version import CORE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"
SRC = REPO_ROOT / "src" / "manicule"

# The two extras `all` deliberately omits, and the reason is in pyproject.toml beside them: on
# x86_64 Linux `rerank` resolves torch and 2.72 GB of CUDA wheels, and `browser-auth` resolves
# playwright and then a browser download. Named here so that adding a third heavyweight extra
# has to be a decision recorded in this list rather than an omission that looks like this one.
DELIBERATELY_OMITTED = frozenset({"rerank", "browser-auth"})


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _project(pyproject: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], pyproject["project"])


def _extras(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    return cast(dict[str, list[str]], _project(pyproject)["optional-dependencies"])


def test_all_extra_is_every_extra_but_the_heavyweights(pyproject: dict[str, Any]) -> None:
    """`all` covers every extra except the two that download gigabytes."""
    extras = _extras(pyproject)
    declared = set(extras) - {"all"}

    # `all` is one self-referencing requirement — `manicule[a,b,c]` — rather than a copy of the
    # dependency lists, so that a dependency added to `serve` needs no edit here at all.
    (requirement,) = (Requirement(entry) for entry in extras["all"])
    assert requirement.name == "manicule"

    assert requirement.extras == declared - DELIBERATELY_OMITTED, (
        "the `all` extra and the extras it aggregates have drifted. An extra added to "
        "pyproject.toml is not in the documented install until it is named in `all` — or "
        "listed in DELIBERATELY_OMITTED here, with its reason written beside it."
    )


def test_the_image_installs_what_the_documented_install_installs() -> None:
    """The Dockerfile's `EXTRAS` and the `all` extra are the same set.

    Not a tidiness check. These are the two ways to obtain manicule, and a component present in
    one and absent from the other is a command that works for half the people who have it —
    reported, eventually, as a bug in the command.
    """
    match = re.search(r'^ARG EXTRAS="(?P<value>[^"]*)"', DOCKERFILE.read_text(), re.MULTILINE)
    assert match is not None, "no `ARG EXTRAS=` in the Dockerfile; this test is reading for it"

    image_extras = set(re.findall(r"--extra\s+([a-z0-9-]+)", match.group("value")))
    with PYPROJECT.open("rb") as handle:
        (requirement,) = (Requirement(e) for e in _extras(tomllib.load(handle))["all"])

    assert image_extras == requirement.extras, (
        "the container and `manicule[all]` install different extras.\n"
        f"  only in the image:   {sorted(image_extras - requirement.extras)}\n"
        f"  only in `all`:       {sorted(requirement.extras - image_extras)}"
    )


def test_the_console_script_is_guarded(pyproject: dict[str, Any]) -> None:
    """The entry point is `manicule.entry`, which imports nothing a bare install lacks.

    Pointing it back at `manicule.cli.main:main` reads like removing an indirection and is the
    defect: importing `manicule.cli` imports Typer, so on an installation without the `serve`
    extra the interpreter fails inside the package before any guard could run, and the person
    gets a traceback naming a library rather than a command to type.
    """
    scripts = cast(dict[str, str], _project(pyproject)["scripts"])
    assert scripts["manicule"] == "manicule.entry:main"

    # And the module it names holds to its own contract: nothing at module scope that a bare
    # install would not have. `sys` is the standard library; the CLI import is inside `main`.
    source = (SRC / "entry.py").read_text()
    module_scope_imports = re.findall(r"^(?:from|import)\s+(\S+)", source, re.MULTILINE)
    assert set(module_scope_imports) <= {"__future__", "sys"}, (
        f"manicule/entry.py imports {module_scope_imports} at module scope. Anything beyond the "
        "standard library defeats the guard: the import fails before it can be reported."
    )


def test_the_builtin_plugins_admit_the_running_version() -> None:
    """Every built-in plugin's declared `core_version` range contains the version installed.

    This is the test that fails on release-please's version-bump pull request, which is exactly
    where it should fail: the bump and the pins move together, in one reviewed commit, rather
    than the pins being discovered a release later by somebody whose parsers all vanished.
    """
    running = Version(CORE_VERSION)
    if running == Version("0.0.0.dev0"):  # pragma: no cover - only in an uninstalled tree
        pytest.skip("manicule is not installed; CORE_VERSION has no distribution to read")

    declarations = {
        path.relative_to(REPO_ROOT): match.group("range")
        for path in sorted(SRC.rglob("plugin.py"))
        if (match := re.search(r'core_version="(?P<range>[^"]+)"', path.read_text()))
    }
    assert declarations, "no built-in plugin declares a core_version; this test is reading wrong"

    refused = {
        path: declared
        for path, declared in declarations.items()
        # `prereleases=True` because a release candidate is still the core that is running, and
        # a plugin refusing to load under one would make every pre-release untestable.
        if not SpecifierSet(declared, prereleases=True).contains(running)
    }
    assert not refused, (
        f"manicule {running} is running, and these built-in plugins refuse it: {refused}.\n"
        "The version bump moved past the range they declare. Widen the pins in the same commit "
        "as the bump — a release that ships without them loads no parsers, no storage and no "
        "embedder, and reports each one as an incompatible plugin."
    )
