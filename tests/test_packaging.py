"""What a release ships, held against what this repository says it ships.

What decides whether `uv tool install "manicule[all]"` produces a working program is checked
nowhere else: every other test here runs from the source tree, with the dev group installed and
`src/` on the path, which is not what anybody installs.

* The `all` extra and the Dockerfile's `EXTRAS` are the same set. They are two spellings of one
  decision — "what an installation of manicule contains" — written in two files that cannot see
  each other.
* The console script survives an installation without the `serve` extra. `manicule.entry` exists
  for that and would be silently pointless if the entry point were ever pointed back at
  `manicule.cli.main:main`, which is the obvious-looking simplification.
* Every plugin admits the version that is running. All of them declare
  `core_version=">=0.1,<0.2"`, and release-please bumping to 0.2.0 would ship a manicule whose
  own parsers refuse to load — an entirely mechanical failure that no other test in this
  repository would notice, on the one commit nobody rehearses.
* The release workflow builds and publishes both distributions, and no others. `manicule` is
  MIT and `manicule-mlx` is GPL-3.0-or-later; the README tells an Apple silicon reader to
  install the second, and a workflow that quietly stopped shipping it would make that
  instruction false without failing anything.

**What this does not do.** It does not build a wheel; that costs seconds and needs network, and
the `dist` job in ci.yml does it on every pull request against the artifact itself. This is the
part that can be checked from the tree, so it is checked where a developer sees it fail.
"""

from __future__ import annotations

import ast
import builtins
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from manicule import entry
from manicule.core.version import CORE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"
SRC = REPO_ROOT / "src" / "manicule"
PACKAGES = REPO_ROOT / "packages"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

# The workspace members that go to PyPI. `manicule` is MIT; `manicule-mlx` is
# GPL-3.0-or-later because it links `mlx-embeddings`, which is the entire reason it is a
# separate distribution rather than an extra. The other two members are test fixtures — a
# reference plugin and a deliberately hostile one — and publishing either would put a parser
# that hangs on purpose on the index.
#
# Written down here rather than inferred, because neither answer is a safe default for a
# workspace member nobody classified: a new package silently published is a mistake that
# cannot be taken back, and one silently withheld is a release that quietly does nothing.
PUBLISHED = ("manicule", "manicule-mlx")

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
    #
    # Parsed rather than grepped. A regular expression over the source reads the docstring too,
    # and this module's docstring is *about* imports — the line "a manicule whose own modules
    # fail to / import is a broken installation" made `^import\s+(\S+)` report a module named
    # `is`. `ast.parse(...).body` is module scope by construction, so the deferred import inside
    # `main` is excluded because of where it is rather than because of how it is spelled.
    tree = ast.parse((SRC / "entry.py").read_text())
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "sys"}, (
        f"manicule/entry.py imports {sorted(imported)} at module scope. Anything beyond the "
        "standard library defeats the guard: the import fails before it can be reported."
    )


def test_every_plugin_admits_the_running_version() -> None:
    """Every plugin here declares a `core_version` range that contains the running version.

    This is the test that fails on release-please's version-bump pull request, which is exactly
    where it should fail: the bump and the pins move together, in one reviewed commit, rather
    than the pins being discovered a release later by somebody whose parsers all vanished.
    """
    running = Version(CORE_VERSION)
    if running == Version("0.0.0.dev0"):  # pragma: no cover - only in an uninstalled tree
        pytest.skip("manicule is not installed; CORE_VERSION has no distribution to read")

    # Both trees. `src/manicule/*/plugin.py` is the six built-ins; `packages/*/src/**` is
    # `manicule-mlx` and the two fixture plugins, which declare the same range and break on the
    # same bump — the fixtures included, because the dev group installs them and their suites
    # are how plugin discovery is tested at all.
    candidates = [*SRC.rglob("plugin.py"), *PACKAGES.glob("*/src/*/__init__.py")]
    declarations = {
        path.relative_to(REPO_ROOT): match.group("range")
        for path in sorted(candidates)
        if (match := re.search(r'core_version="(?P<range>[^"]+)"', path.read_text()))
    }
    assert declarations, "no plugin declares a core_version; this test is reading the wrong paths"

    refused = {
        path: declared
        for path, declared in declarations.items()
        # `prereleases=True` because a release candidate is still the core that is running, and
        # a plugin refusing to load under one would make every pre-release untestable.
        if not SpecifierSet(declared, prereleases=True).contains(running)
    }
    assert not refused, (
        f"manicule {running} is running, and these plugins refuse it: {refused}.\n"
        "The version bump moved past the range they declare. Widen the pins in the same commit "
        "as the bump — a release that ships without them loads no parsers, no storage and no "
        "embedder, and reports each one as an incompatible plugin."
    )


def test_the_release_workflow_builds_every_published_distribution() -> None:
    """`release.yml` builds exactly the workspace members that are meant to reach PyPI.

    The failure this exists for is silent in the worst direction. `manicule-mlx` is what an
    Apple silicon reader is told to install, and a release workflow that does not build it
    publishes a README instructing people to install a package that is not there — green run,
    green release, and the instruction is simply false.

    The other direction is worse and also covered: `packages/` holds parsers that hang and
    allocate without bound on purpose, and a `uv build` that stopped naming its package would
    put them on the index, where nothing can be taken back.
    """
    workflow = RELEASE_WORKFLOW.read_text()
    built = set(re.findall(r"uv build --package\s+(\S+)", workflow))
    assert built == set(PUBLISHED), (
        "release.yml and PUBLISHED disagree about what ships.\n"
        f"  built by the workflow: {sorted(built)}\n"
        f"  expected:              {sorted(PUBLISHED)}"
    )

    # Every published distribution is also uploaded. Building one and forgetting to publish it
    # is the same false instruction with an extra step in between.
    published = set(re.findall(r"packages-dir:\s*dist/(\S+)", workflow))
    assert published == set(PUBLISHED), (
        f"built but not published: {sorted(built - published)}; "
        f"published but not built: {sorted(published - built)}"
    )


def test_every_workspace_member_is_classified() -> None:
    """No workspace member is left neither published nor deliberately withheld.

    `PUBLISHED` is a list, and a list goes stale the moment somebody adds a package without
    reading it. This is what makes that impossible to do quietly: a new member fails here,
    naming itself, and whoever added it decides which side it is on rather than inheriting an
    answer from whichever default the tooling happened to have.
    """
    members = {
        cast(dict[str, Any], tomllib.loads((path / "pyproject.toml").read_text())["project"])[
            "name"
        ]
        for path in sorted(PACKAGES.iterdir())
        if (path / "pyproject.toml").is_file()
    }
    withheld = {"manicule-plugin-example", "manicule-plugin-hostile"}

    assert members == (set(PUBLISHED) - {"manicule"}) | withheld, (
        f"packages/ holds {sorted(members)}, which is neither the published set nor the "
        "withheld one. Add it to PUBLISHED in this file, or to `withheld` here with the "
        "reason it must never reach PyPI."
    )


def test_only_an_absent_dependency_gets_the_install_hint() -> None:
    """A missing module is translated; an installed-but-incompatible one is not.

    `from typer import Removed` against a Typer that no longer has `Removed` raises a plain
    `ImportError` whose `.name` is still `'typer'`. Catching `ImportError` rather than
    `ModuleNotFoundError` therefore answers a version conflict with "install `manicule[all]`" —
    advice that cannot help, printed over the incompatibility it has just hidden.

    `main` narrows to `ModuleNotFoundError`, which is what makes that impossible.
    """
    hint = entry.install_hint(ModuleNotFoundError("No module named 'typer'", name="typer"))
    assert hint is not None
    assert "manicule[all]" in hint

    # A module nothing here provides gets no hint, so `main` re-raises it untouched: a manicule
    # whose own modules fail to import is broken, not incomplete.
    assert entry.install_hint(ModuleNotFoundError("no module named 'nacl'", name="nacl")) is None

    # The premise of the narrowing, asserted against the exception the interpreter really
    # constructs rather than assumed: a "cannot import name" failure carries the *module* in
    # `.name` — so it would match `_PROVIDED_BY` — and is not a `ModuleNotFoundError`, which is
    # the only reason the guard never sees it.
    # Compiled at run time rather than written as an import statement. The symbol is absent on
    # purpose — that absence *is* the fixture — and a static checker is right to reject the
    # literal form, so writing it literally would trade a real demonstration for a suppression
    # comment. What is under test is CPython's own behavior, which only a genuine failed import
    # exhibits.
    premise = compile("from json import ThisSymbolDoesNotExist", "<premise>", "exec")
    with pytest.raises(ImportError) as mismatch:
        exec(premise, {})  # noqa: S102 - the compiled statement above is the fixture

    assert mismatch.value.name == "json"
    assert not isinstance(mismatch.value, ModuleNotFoundError), (
        "a `cannot import name` error is now a ModuleNotFoundError, so narrowing to it no longer "
        "separates an absent dependency from an incompatible one. manicule/entry.py needs a "
        "different discriminator."
    )


def _import_raising(exc: ImportError) -> Callable[..., Any]:
    """An `__import__` that fails on the CLI module alone, with `exc`, and is otherwise real."""
    real = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "manicule.cli.main":
            raise exc
        return real(name, *args, **kwargs)

    return fake


def test_main_translates_an_absent_dependency_and_propagates_everything_else(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` acts on the exception *class*, not just on `.name`.

    The distinction it draws is invisible to `install_hint`, which never sees the class — so
    testing that function alone leaves `except ImportError` and `except ModuleNotFoundError`
    indistinguishable, and a revert to the broad one passes. Verified by mutation: with only the
    `install_hint` assertions above, reverting the narrowing failed nothing.

    This drives `main` itself, with the import made to fail on demand, which is the only place
    the `except` clause is observable.
    """
    absent = ModuleNotFoundError("No module named 'typer'", name="typer")
    monkeypatch.setattr(builtins, "__import__", _import_raising(absent))
    with pytest.raises(SystemExit) as exited:
        entry.main()

    assert exited.value.code == 1
    stderr = capsys.readouterr().err
    assert "manicule[all]" in stderr
    assert "Traceback" not in stderr

    # The same module name, carried by the exception a version conflict raises. It must reach the
    # caller as itself: the person needs to read "cannot import name", not an install hint for a
    # package they already have.
    incompatible = ImportError("cannot import name 'Removed' from 'typer'", name="typer")
    monkeypatch.setattr(builtins, "__import__", _import_raising(incompatible))
    with pytest.raises(ImportError) as propagated:
        entry.main()

    assert propagated.value is incompatible
