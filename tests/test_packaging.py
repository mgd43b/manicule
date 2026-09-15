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
* Every plugin admits the version that is running. This is not hypothetical: they all declared
  `core_version=">=0.1,<0.2"`, release-please bumped to 0.2.0, and `v0.2.0` was tagged with a
  manicule whose own parsers, storage and embedder all refused to load. This test failed on that
  release pull request, exactly as designed, and was merged past — so the check is sound and the
  thing to protect is reading it. An entirely mechanical failure that no other test in this
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
COMPOSE = REPO_ROOT / "compose.yaml"
SRC = REPO_ROOT / "src" / "manicule"
PACKAGES = REPO_ROOT / "packages"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

QDRANT_PIN = re.compile(r"^qdrant/qdrant:(v\d+\.\d+\.\d+)$")
"""A Qdrant image pinned to an exact release.

Anchored, and to all three components. A tag is what reproducibility rests on here: `latest`
floats by definition, and a `vMAJOR.MINOR` tag floats to the newest patch on its own — the
Dockerfile's own note about `python:3.14-slim-bookworm` says so. Either would leave the suite
certifying the vector store against whatever was published that morning.
"""

# The workspace members that go to PyPI. `manicule` is MIT; `manicule-mlx` is
# GPL-3.0-or-later because it links `mlx-embeddings`, which is the entire reason it is a
# separate distribution rather than an extra. Three members are withheld: two are test
# fixtures — a reference plugin and a deliberately hostile one, and publishing either would put
# a parser that hangs on purpose on the index — and the third is `manicule-plugin-wikilinks`,
# which is neither.
#
# `manicule-plugin-wikilinks` is withheld because **publishing is a release decision, not an
# authoring one**: it needs a version line release-please can move, a build and an upload step in
# `release.yml`, and an answer to what its `core_version` range means once the two distributions
# can move independently. None of that is settled by writing the plugin, and a package added to
# `PUBLISHED` without it would be built by a workflow that does not know about it. It is
# installable from the workspace today, which is what its tests and this repository need.
#
# Written down here rather than inferred, because neither answer is a safe default for a
# workspace member nobody classified: a new package silently published is a mistake that
# cannot be taken back, and one silently withheld is a release that quietly does nothing.
PUBLISHED = ("manicule", "manicule-mlx", "manicule-ollama")

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


def test_the_image_does_not_pin_the_embedding_provider_in_the_environment() -> None:
    """If it did, `[embedding] provider` in `/data/config.toml` would silently stop choosing the
    backend, leaving the `ollama` extra this image ships for exactly that purpose unreachable.
    """
    # Comment lines excluded: the Dockerfile's own explanation of why this is unset names the
    # variable, and a bare substring search would find that prose and never see the real ENV
    # block at all.
    code_lines = [
        line for line in DOCKERFILE.read_text().splitlines() if not line.lstrip().startswith("#")
    ]
    assert not any("MANICULE_EMBEDDING__PROVIDER" in line for line in code_lines), (
        "the Dockerfile sets MANICULE_EMBEDDING__PROVIDER. The environment outranks "
        "/data/config.toml in manicule's settings sources, so this pins the backend and makes "
        "`[embedding] provider` unsettable from the config file — silently, since `onnx` is a "
        "registered provider and resolves cleanly. The field already defaults to `onnx`; remove "
        "the variable instead of restoring it."
    )


def _qdrant_service_image(path: Path, job: str | None) -> str:
    """The image `services.qdrant.image` names in a parsed document, not in its prose.

    Read from the service definition rather than matched in the file's text, because the text
    also contains sentences *about* the image — this repository's workflows and manifests
    explain themselves at length, and the comment block right above the CI service says
    `qdrant/qdrant` in prose. A regular expression over the raw bytes reads those sentences as
    pins, so a service that was removed or renamed would go on matching its own explanation.
    `test_ci_and_release_sync_the_same_way` parses for the same reason.

    ``job`` names the workflow job the service hangs off, or ``None`` for a compose file, which
    declares its services at the top level.
    """
    import yaml  # noqa: PLC0415 - a test-only dependency, kept out of this module's import cost

    document = cast(dict[str, Any], yaml.safe_load(path.read_text()))
    if job is not None:
        jobs = cast(dict[str, Any], document["jobs"])
        assert job in jobs, f"no `{job}` job in {path.name}; this test is reading for one"
        document = cast(dict[str, Any], jobs[job])
    services = cast(dict[str, Any], document.get("services") or {})
    assert "qdrant" in services, (
        f"no `qdrant` service in {path.name}"
        + (f" job `{job}`" if job else "")
        + "; this test is reading for one. If the service moved or was removed, this test is "
        "what needs updating."
    )
    return cast(str, cast(dict[str, Any], services["qdrant"])["image"])


def test_the_qdrant_the_suite_tests_against_is_the_one_compose_runs() -> None:
    """The two Qdrant services name one exact release, and Dependabot only sees one of them.

    Not a tidiness check, and the asymmetry is the whole reason for it. Dependabot's `docker`
    ecosystem reads `image:` in a YAML manifest as well as a Dockerfile, so the pin in
    `compose.yaml` is tracked and gets a pull request when a release lands. The one in
    `ci.yml` is not: that ecosystem does not scan `.github/workflows/`, and `github-actions`
    updates `uses:` references rather than a job's `services.*.image`.

    So the tracked pin moves and the untracked one stays, silently — and the untracked one is
    the server the vector-store conformance suite actually runs against, which makes it the
    half that matters. Holding them equal turns the bump Dependabot *does* raise into a failing
    build until both move together, which is the same trick
    `test_the_image_installs_what_the_documented_install_installs` plays on the Dockerfile's
    extras, for the same reason: two copies of one decision, and only one of them maintained.

    **Equality alone is not enough**, which is the second assertion. Two references that both
    said `latest` would be equal and would agree about nothing: the suite would certify the
    backend against whatever was published that morning, and a wire-behavior regression would
    arrive as a test failure on an unrelated pull request. The tag has to be an exact release
    before it is worth comparing.
    """
    pins = {
        COMPOSE.name: _qdrant_service_image(COMPOSE, None),
        CI_WORKFLOW.name: _qdrant_service_image(CI_WORKFLOW, "qdrant"),
    }

    floating = {name: image for name, image in pins.items() if not QDRANT_PIN.match(image)}
    assert not floating, (
        "a Qdrant service is not pinned to an exact release.\n"
        + "".join(f"  {name}: {image}\n" for name, image in sorted(floating.items()))
        + "`latest` floats by definition and a `vMAJOR.MINOR` tag floats to the newest patch, "
        "so either would leave the conformance suite testing against whatever was published "
        "most recently. Pin `vMAJOR.MINOR.PATCH`."
    )

    versions = {cast(re.Match[str], QDRANT_PIN.match(image)).group(1) for image in pins.values()}
    assert len(versions) == 1, (
        "the Qdrant the test suite runs against and the one `docker compose` starts have "
        "drifted.\n"
        + "".join(f"  {name}: {image}\n" for name, image in sorted(pins.items()))
        + "Dependabot tracks the compose pin and not the workflow one, so this is what a "
        "merged bump looks like. Move the other to match."
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


def test_every_package_depends_on_a_manicule_that_exists() -> None:
    """Every sibling package pins `manicule`, and the pin admits the core it ships beside.

    **The other half of the same bump, and the half that had no test.**
    :func:`test_every_plugin_admits_the_running_version` reads `core_version=` out of Python
    files; a distribution also states a range in its `pyproject.toml`, and those went past 0.2
    unguarded. `manicule-ollama` required `manicule[embeddings]>=0.1,<0.2` and
    `manicule-plugin-wikilinks` required `manicule>=0.1,<0.2`, so at 0.2.0 both were
    *uninstallable* alongside the core they are released with.

    That failure looks nothing like the other one, which is why it needs its own check. A stale
    `core_version` produces a running manicule that refuses its plugins — loud, and every suite
    sees it. A stale dependency pin produces a resolver error for somebody who typed
    `pip install manicule-ollama`, on a machine nobody here is sitting at, and no test in this
    repository runs a resolver.

    Extras are kept on the requirement rather than stripped: `manicule[embeddings]` and
    `manicule` are the same distribution and the same range applies, and parsing it back through
    ``Requirement`` is what makes that true rather than assumed.
    """
    running = Version(CORE_VERSION)
    if running == Version("0.0.0.dev0"):  # pragma: no cover - only in an uninstalled tree
        pytest.skip("manicule is not installed; CORE_VERSION has no distribution to read")

    # **Every occurrence, kept whole.** This check has now been wrong four times in one way: it
    # collapsed many requirements into one answer and then asserted over the survivor. A filter
    # that only kept what it approved of could not report what it skipped; a dict keyed by
    # manifest kept only the last entry, so an incompatible bound was masked by a compatible one
    # later in the same file. Both are the same mistake. So nothing is collapsed here — every
    # `manicule` requirement in every section is collected with the file it came from, and both
    # assertions below run over the whole list.
    found: list[tuple[Path, Requirement]] = []
    for manifest in sorted(PACKAGES.glob("*/pyproject.toml")):
        with manifest.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        optional = cast("dict[str, list[str]]", project.get("optional-dependencies", {}))
        wanted = [*project.get("dependencies", []), *(e for g in optional.values() for e in g)]
        where = manifest.relative_to(REPO_ROOT)
        found.extend(
            (where, requirement)
            for stated in wanted
            if (requirement := Requirement(stated)).name == "manicule"
        )

    assert found, "no sibling package requires manicule; this test is reading the wrong paths"

    # **An unbounded requirement, even beside a bounded one.** `manicule-mlx` depended on a bare
    # `"manicule"` while its plugin manifest said `<0.3`, so a resolver could install a core the
    # plugin then refuses — and the failure arrives as an incompatible plugin rather than as the
    # dependency conflict it is. An optional group is not installed by default, so a bare entry
    # in `dependencies` beside a pinned one in an extra *is* the default install.
    bare = sorted(
        f"{path}: {requirement}" for path, requirement in found if not requirement.specifier
    )
    assert not bare, (
        f"these requirements name manicule with no version bound: {bare}.\n"
        "Their plugin manifests declare a `core_version` range, so an unbounded requirement lets "
        "a resolver install a core the plugin then refuses. State the same range in both."
    )

    # `prereleases=True` for the reason the sibling test gives: a release candidate is still the
    # core being released beside these.
    refused = sorted(
        f"{path}: {requirement}"
        for path, requirement in found
        if requirement.specifier
        and not SpecifierSet(str(requirement.specifier), prereleases=True).contains(running)
    )
    assert not refused, (
        f"manicule {running} is being released, and these requirements exclude it: {refused}.\n"
        "Installing them would resolve to an older core or fail outright. Widen the pins in the "
        "same commit as the bump, alongside the plugin `core_version` ranges."
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


def test_the_lockfile_cannot_drift_from_the_version_being_released() -> None:
    """Two guards keep `uv.lock` on the version being released, and both are read out of CI.

    `uv.lock` records a `version` for both workspace members, because both are
    `source = {editable = ...}`. release-please bumps `pyproject.toml` and, through
    `extra-files`, `manicule-mlx`'s — and has never touched the lockfile. So every release left
    it a version behind: `main` shipped 0.1.10, 0.1.11 and 0.1.12 with a lockfile still pinned
    at 0.1.9.

    **What made it invisible is the reason this is a workflow test and not a content one.**
    `uv run` and `uv sync` *repair* a stale lockfile in place, silently, before doing anything
    else. A test that read `uv.lock` and compared it to `pyproject.toml` would therefore pass
    unconditionally under `uv run pytest` — uv rewrites the file on the way to starting pytest,
    so the assertion never sees the state it exists to catch. The drift surfaced only as a
    working tree that went dirty on a contributor's first command, in a file they had not
    touched, which then rode into whatever pull request was open.

    So the guards have to sit where uv has not already been:

    * `uv lock --check` in ci.yml, which resolves and *refuses* instead of rewriting — and must
      run before the job's first `uv sync`, or it checks a file that was just repaired.
    * the re-lock step in release.yml, which puts the new lockfile in the release pull request
      itself, so the bump and the lock move in one reviewed commit.

    Losing either is silent, which is what makes them worth pinning here.
    """
    import yaml  # noqa: PLC0415 - a test-only dependency, kept out of this module's import cost

    # Parsed, not grepped. The prose in these workflows discusses `uv sync` by name — this
    # comment does too — and a regular expression over the raw file reads those sentences as
    # commands, so the ordering assertion below failed on its own explanation. `run:` strings
    # from the parsed job graph are the commands and nothing else.
    ci = cast(dict[str, Any], yaml.safe_load(CI_WORKFLOW.read_text()))
    commands = [
        cast(str, step["run"])
        for job in cast(dict[str, dict[str, Any]], ci["jobs"]).values()
        for step in cast(list[dict[str, Any]], job.get("steps") or [])
        if isinstance(step.get("run"), str)
    ]

    guards = [index for index, run in enumerate(commands) if "uv lock --check" in run]
    assert guards, (
        "no job in ci.yml runs `uv lock --check`. Without it a stale lockfile is repaired by "
        "the next `uv sync` and never reported, which is how three releases shipped with "
        "uv.lock pinned a version behind."
    )

    # Ordering is the whole of the check's value, so it is asserted rather than assumed. Steps
    # are compared within the job that holds the guard: jobs run on their own runners with their
    # own checkouts, so a sync in some other job cannot repair the file this one reads.
    guard_job = next(
        name
        for name, job in cast(dict[str, dict[str, Any]], ci["jobs"]).items()
        if any(
            "uv lock --check" in cast(str, step["run"])
            for step in cast(list[dict[str, Any]], job.get("steps") or [])
            if isinstance(step.get("run"), str)
        )
    )
    within = [
        cast(str, step["run"])
        for step in cast(list[dict[str, Any]], ci["jobs"][guard_job].get("steps") or [])
        if isinstance(step.get("run"), str)
    ]
    before = within[: next(i for i, run in enumerate(within) if "uv lock --check" in run)]
    repairs = [run for run in before if re.search(r"\buv (?:sync|run)\b", run)]
    assert not repairs, (
        f"job {guard_job!r} in ci.yml runs {repairs} before `uv lock --check`. uv repairs a "
        "stale lockfile in place, so the check would resolve a file that had just been "
        "rewritten and pass unconditionally. Move the check above the first sync."
    )

    release = cast(dict[str, Any], yaml.safe_load(RELEASE_WORKFLOW.read_text()))
    steps = [
        step
        for job in cast(dict[str, dict[str, Any]], release["jobs"]).values()
        for step in cast(list[dict[str, Any]], job.get("steps") or [])
    ]
    relocks = [
        step
        for step in steps
        if isinstance(step.get("run"), str)
        # `uv lock` and not `uv lock --check`: this step must *write* the lockfile. The check is
        # ci.yml's job, and a `--check` here would fail the release rather than fix it.
        and re.search(r"\buv lock\b(?! --check)", cast(str, step["run"]))
    ]
    assert relocks, (
        "no step in release.yml runs `uv lock`. release-please bumps two pyproject.toml files "
        "and knows nothing about uv.lock, so without this the release pull request ships a "
        "lockfile naming the previous version."
    )

    # And it re-locks the release branch rather than whatever happened to be checked out. The
    # branch comes from the action's `pr` output; re-locking anywhere else puts the lockfile in
    # a commit the release does not contain.
    guarded = [
        step
        for step in relocks
        if "headBranchName" in str(step.get("env", "")) or "headBranchName" in str(step)
    ]
    assert guarded, (
        "release.yml re-locks, but the step does not read `headBranchName` from the "
        "release-please `pr` output. Re-locking off the release branch commits the lockfile "
        "somewhere the release will not contain."
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
    withheld = {
        "manicule-plugin-example",
        "manicule-plugin-hostile",
        "manicule-plugin-wikilinks",
    }

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


def test_each_published_distribution_publishes_from_its_own_environment() -> None:
    """No two publish jobs share a GitHub environment, and every published package has one.

    This is a PyPI constraint wearing a GitHub Actions costume. A *pending* trusted publisher
    must have a unique claim set, and the environment name is part of it — the OIDC `sub` reads
    `repo:owner/repo:environment:<name>`. Two packages publishing from one environment therefore
    cannot both be registered before their first release: PyPI refuses the second with "a pending
    trusted publisher matching this configuration has already been registered for a different
    project name", and the only ways out are ordering the first release by hand or coming back
    here.

    Found the hard way, on the first real release. Pinned so the next package added to
    `PUBLISHED` cannot rediscover it.
    """
    import yaml  # noqa: PLC0415 - a test-only dependency, kept out of this module's import cost

    workflow = cast(dict[str, Any], yaml.safe_load(RELEASE_WORKFLOW.read_text()))
    jobs = cast(dict[str, dict[str, Any]], workflow["jobs"])

    environments: dict[str, str] = {}
    for name, job in jobs.items():
        steps = cast(list[dict[str, Any]], job.get("steps") or [])
        if not any("gh-action-pypi-publish" in str(step.get("uses", "")) for step in steps):
            continue
        environment = job.get("environment")
        assert isinstance(environment, dict), (
            f"job {name!r} uploads to PyPI without an `environment:`. The environment is half of "
            "what the trusted publisher matches on; without it the OIDC claims cannot identify "
            "which project is being published."
        )
        environments[name] = cast(str, environment["name"])

    assert len(environments) == len(PUBLISHED), (
        f"{len(PUBLISHED)} distributions are published but {len(environments)} jobs upload to "
        f"PyPI: {environments}"
    )
    assert len(set(environments.values())) == len(environments), (
        f"two publish jobs share an environment: {environments}. Each needs its own, or their "
        "pending trusted publishers collide on PyPI and the second cannot be registered."
    )


# Anything that is already a destination rather than a repository path. `#` alone is an anchor
# within the rendered page, which resolves on PyPI as well as on GitHub.
_ABSOLUTE = ("http://", "https://", "#", "mailto:")


def _markdown_links(text: str) -> list[str]:
    return [target for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)]


def test_the_pypi_long_description_has_no_relative_links(pyproject: dict[str, Any]) -> None:
    """Applying the configured rewrites to README.md leaves no relative link behind.

    README.md *is* the PyPI long description, and PyPI resolves a relative link against nothing —
    so `docs/surfaces.md` and the other fifteen rendered as 404s on the project page from the
    moment 0.1.0 was published. `hatch-fancy-pypi-readme` rewrites them at build time, which
    keeps the in-repo links relative so they follow the branch a contributor is reading.

    The rewrite is a pair of regular expressions, and the failure mode is a link written in a
    form they do not match: nothing errors, the build succeeds, and one more dead link appears on
    a page nobody looks at until someone clicks it. So the patterns are read from pyproject.toml
    and applied here to the real file — an unmatched link fails the build instead.
    """
    hooks = cast(dict[str, Any], pyproject["tool"]["hatch"]["metadata"]["hooks"])
    config = cast(dict[str, Any], hooks["fancy-pypi-readme"])

    fragments = cast(list[dict[str, str]], config["fragments"])
    rendered = "".join((REPO_ROOT / fragment["path"]).read_text() for fragment in fragments)

    for substitution in cast(list[dict[str, str]], config["substitutions"]):
        rendered = re.sub(substitution["pattern"], substitution["replacement"], rendered)

    survivors = [t for t in _markdown_links(rendered) if not t.startswith(_ABSOLUTE)]
    assert not survivors, (
        f"these links would reach PyPI unrewritten and 404 there: {survivors}. Either write them "
        "in a form the substitutions in pyproject.toml match, or add a substitution for the form "
        "you need."
    )

    # And the rewrite is doing real work rather than passing because the file has no relative
    # links left. Without this, hard-coding absolute URLs in README.md would silently turn the
    # substitutions into dead configuration and this test would still pass.
    source_relative = [
        t
        for t in _markdown_links((REPO_ROOT / "README.md").read_text())
        if not t.startswith(_ABSOLUTE)
    ]
    assert source_relative, (
        "README.md has no relative links, so the fancy-pypi-readme substitutions rewrite "
        "nothing. Either they are dead configuration and should be removed, or a link that "
        "should be relative has been hard-coded to an absolute URL."
    )


def test_images_are_rewritten_to_raw_urls(pyproject: dict[str, Any]) -> None:
    """A screenshot must become a `raw.` URL, not a `blob.` one.

    `blob` serves GitHub's HTML page *for* the file. As a link that is right; as an `![image]`
    source it is a page where an image should be, so the screenshot renders broken rather than
    missing — which reads as a bug in the page rather than a bad link.
    """
    hooks = cast(dict[str, Any], pyproject["tool"]["hatch"]["metadata"]["hooks"])
    config = cast(dict[str, Any], hooks["fancy-pypi-readme"])
    rendered = (REPO_ROOT / "README.md").read_text()
    for substitution in cast(list[dict[str, str]], config["substitutions"]):
        rendered = re.sub(substitution["pattern"], substitution["replacement"], rendered)

    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", rendered)

    # Stated as "no image is a blob URL" rather than "every image is a raw URL", because the CI
    # badge is a `github.com/.../badge.svg` endpoint that really does serve an SVG. What is
    # always wrong is `/blob/`, which is the HTML page for a file.
    blobs = [url for url in images if "/blob/" in url]
    assert not blobs, (
        f"{blobs} are images pointing at GitHub's HTML page for a file rather than at raw "
        "content; they render broken on PyPI. The substitution ordering in pyproject.toml puts "
        "the image rule first for this reason."
    )

    raw = [url for url in images if url.startswith("https://raw.githubusercontent.com/")]
    assert raw, (
        "no image was rewritten to a raw URL, so the image substitution matched nothing. Either "
        "README.md no longer embeds a repository image, or the pattern has stopped matching it."
    )


def test_every_all_extra_install_resolves_from_the_wheels_being_tested() -> None:
    """`manicule[all]` must not reach an index for a distribution this tree also builds.

    The failure this pins is one CI found rather than review: `all` gained `ollama`, and three
    separate steps install `manicule[all]` — two in `ci.yml` and one in `release.yml`. Two were
    given `--find-links` and the third was not, so it went looking for `manicule-ollama` on PyPI,
    where the first release including it had not happened yet. It failed loudly, which was luck:
    once the package *is* published, the same omission resolves the **previous** release's
    backend against this tree's core and passes, and the step goes on reporting that the
    documented install works while checking a pair that was never built together.

    So every `[all]` install is held to resolving the workspace's own distributions from the
    directory they were just built into. A fourth published backend added the same way fails
    here rather than six months later.
    """
    installs: list[str] = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        # The whole `uv pip install` invocation, continuations included, so the flags and the
        # `[all]` they belong to are read together rather than as separate lines.
        for match in re.finditer(r"uv pip install(?:[^\n]*\\\n)*[^\n]*", text):
            command = match.group(0)
            if "[all]" in command and "--find-links" not in command:
                installs.append(f"{workflow.name}: {' '.join(command.split())}")

    assert installs == [], (
        "these install `manicule[all]` without pointing at the locally built distributions, so "
        "they resolve a workspace member from an index instead of from this tree:\n  "
        + "\n  ".join(installs)
    )


def _publish_container_steps() -> list[dict[str, Any]]:
    """The steps of the job that builds and pushes the published image."""
    import yaml  # noqa: PLC0415 - a test-only dependency, kept out of this module's import cost

    workflow = cast(dict[str, Any], yaml.safe_load(RELEASE_WORKFLOW.read_text()))
    return cast(list[dict[str, Any]], workflow["jobs"]["publish-container"]["steps"])


def _publish_container_build_step() -> dict[str, Any]:
    """The `release.yml` step that builds the image that gets published."""
    for step in _publish_container_steps():
        if "offline smoke test runs inside it" in str(step.get("name", "")):
            return step
    msg = "release.yml has no image-build step in publish-container"
    raise AssertionError(msg)


def _image_build_command() -> str:
    """The build step's shell, **with its comment lines removed**.

    Asserting against the raw `run` block would be the shape of test this repository has been
    bitten by before: the step explains itself in comments, those comments name the very
    options under test, and a check that searched the whole block would stay green while the
    option it was written to protect was deleted from the command.
    """
    run = str(_publish_container_build_step()["run"])
    return "\n".join(line for line in run.splitlines() if not line.lstrip().startswith("#"))


def _build_output_argument() -> str:
    """The value of the build command's `--output`, which is where the exporter is configured."""
    command = _image_build_command()
    match = re.search(r"--output\s+[\"']?([^\"'\\\n]+)", command)
    assert match, f"the image build passes no --output:\n{command}"
    return match.group(1)


SOURCE_DATE_EPOCH = "1700000000"
"""The epoch the published image is built against. **This value is never to change.**

It is what every layer's timestamps are rewritten to, so moving it moves every layer digest
and every node re-pulls the whole 1.74 GB image once — which is the cost this whole mechanism
exists to avoid. There is no reason to prefer one instant over another, and every reason to
keep the one already published.
"""


def test_the_published_image_is_built_with_reproducible_layer_timestamps() -> None:
    """Without this the 1.36 GB model layer is a new blob every release, and every node re-pulls it.

    The layer's content is byte-identical between releases — same pinned revisions, same
    content-addressed filenames — but BuildKit takes each file's mtime from the filesystem, and
    `prefetch_embedding_models.py` downloads 2.3 GB at a different instant on every run. That
    lands in the tar headers, so the blob digest moves and a node pulling the next version
    fetches the whole thing again. Measured across 0.1.17-0.1.21: five releases, five digests,
    all 1361.8 MB, with every mtime equal to its build time.
    """
    command = _image_build_command()

    assert "docker buildx build" in command, (
        "the published image must be built with `docker buildx build`: the daemon's built-in "
        "builder has no `rewrite-timestamp`, so the layer timestamps stay as-downloaded"
    )
    assert "rewrite-timestamp=true" in _build_output_argument(), (
        "the build's `--output` must set `rewrite-timestamp=true`, which is what normalizes "
        "the mtimes that otherwise make a byte-identical model layer a new blob every release"
    )


def test_the_release_builds_on_a_builder_that_can_rewrite_timestamps() -> None:
    """`rewrite-timestamp` is an exporter option the daemon's own builder does not have.

    Asserting the command alone would leave this half unguarded: drop the `docker-container`
    driver and `docker buildx build` quietly falls back to the daemon builder, where the
    option is not supported and the layers go back to carrying their download times.
    """
    drivers = [
        str(cast(dict[str, Any], step.get("with", {})).get("driver", ""))
        for step in _publish_container_steps()
        if "setup-buildx-action" in str(step.get("uses", ""))
    ]

    assert "docker-container" in drivers, (
        f"publish-container must set up a `docker-container` builder before building; "
        f"found drivers {drivers}"
    )


def test_the_image_builds_against_a_source_date_epoch_that_never_moves() -> None:
    """A per-release epoch would defeat the whole thing while looking like best practice.

    The reproducible-builds convention is to set `SOURCE_DATE_EPOCH` to the commit timestamp.
    Here that is exactly wrong: every layer's timestamps — and so every layer's digest — would
    move with it, and no release would share anything with the one before. Measured: identical
    content under two different epochs produces two different digests.

    So the value is asserted exactly rather than merely checked for being a number. A different
    literal is the same failure as an expression, only quieter: it would pass a shape check and
    reset every published digest once.
    """
    step = _publish_container_build_step()
    epoch = str(cast(dict[str, Any], step.get("env", {})).get("SOURCE_DATE_EPOCH", ""))

    assert epoch == SOURCE_DATE_EPOCH, (
        f"SOURCE_DATE_EPOCH is {epoch!r} and must be {SOURCE_DATE_EPOCH!r}. It is read from the "
        f"environment by buildx and rewritten into every layer, so any other value — an "
        f"expression computed per run, or simply a different instant — moves every layer "
        f"digest and costs every node one full 1.74 GB re-pull."
    )


def _dockerfile_instructions() -> list[str]:
    """The Dockerfile's instructions: comments dropped, line continuations joined.

    Read out of the code rather than matched in the file's text, for the reason
    `test_the_image_does_not_pin_the_embedding_provider_in_the_environment` gives: this file
    explains itself at length, and its prose names the very paths and variables these tests
    read for. A bare substring search finds the explanation and never reaches the instruction.
    """
    code = [
        line for line in DOCKERFILE.read_text().splitlines() if not line.lstrip().startswith("#")
    ]
    # Continuations joined: one instruction spans several physical lines, with the keyword on
    # the first and much of what these tests read for on the last.
    instructions: list[str] = []
    for line in code:
        if instructions and instructions[-1].endswith("\\"):
            instructions[-1] = instructions[-1][:-1] + " " + line.strip()
        else:
            instructions.append(line)
    return instructions


def _stage_instructions(stage: str) -> list[str]:
    """The instructions belonging to one build stage, `FROM <stage>` to the next `FROM`.

    Stage-scoped because several variables this file cares about are legitimately set in one
    stage and meaningless in another. `PYTHONDONTWRITEBYTECODE` is the example that caught a
    test out: the runtime stage has always set it, so searching the whole Dockerfile for the
    name passes whether or not the *build* stage — the only one that compiles anything — sets
    it at all.
    """
    instructions = _dockerfile_instructions()
    starts = [
        index
        for index, line in enumerate(instructions)
        if line.startswith("FROM ") and line.rstrip().endswith(f" AS {stage}")
    ]
    assert len(starts) == 1, f"expected exactly one `FROM ... AS {stage}`; found {len(starts)}"

    begin = starts[0]
    following = [
        index
        for index, line in enumerate(instructions)
        if index > begin and line.startswith("FROM ")
    ]
    return instructions[begin : following[0] if following else len(instructions)]


def _model_cache_copy() -> str:
    """The Dockerfile instruction that puts the fetched weights into the image."""
    matches = [line for line in _dockerfile_instructions() if "prefetch_embedding_models" in line]
    assert len(matches) == 1, (
        f"expected exactly one Dockerfile instruction calling prefetch_embedding_models.py; "
        f"found {len(matches)}"
    )
    return matches[0]


def test_the_image_ships_only_the_hub_cache_out_of_the_download() -> None:
    """Anything else in an `HF_HOME` is scratch, and one piece of it is a per-build timestamp.

    `huggingface-hub` depends on `hf_xet`, and a Xet download writes `$HF_HOME/xet/logs/` —
    a trace named `xet_<wall clock>_<pid>.log` holding ~135 KB of microsecond-stamped JSON
    lines. Copying the whole `HF_HOME` carried that into the 1.36 GB model layer, which gave
    the layer a new digest on every release and cost every node a full re-pull for a version
    bump. It is the one thing #356 could not fix: `rewrite-timestamp` rewrites tar headers, and
    this timestamp is in a file's name and body.

    Asserted as an allowlist — `hub` is the only source this instruction may copy from — rather
    than against the one spelling that caused it. Naming `cp -a /tmp/hf/.` alone would leave
    `cp -a /tmp/hf/hub … && cp -a /tmp/hf/xet …` passing, which puts the same bytes back by
    another route; the property worth holding is that nothing but `hub` is copied at all.
    """
    copy = _model_cache_copy()

    # Bare path arguments only: `target=/tmp/hf` and `HF_HOME=/tmp/hf` are the cache mount and
    # the download's own environment, neither of which puts anything in the image, and both of
    # which are a token that starts with its own key rather than with the path.
    download = "/tmp/hf"  # noqa: S108 - read out of the Dockerfile, never opened by this process
    sources = sorted(
        {
            argument
            for token in copy.split()
            if (argument := token.strip("\"'")).startswith(download)
        }
    )

    assert sources == [f"{download}/hub"], (
        f"the model step must copy `/tmp/hf/hub` out of the download and nothing else. Every "
        f"other path in an `HF_HOME` is scratch, and `xet/logs/` in it is named and filled with "
        f"the wall clock, which resets the 1.36 GB layer's digest every release and costs every "
        f"node a full re-pull. `cp -a /tmp/hf/.` is how it was written for five releases; a "
        f"second source beside `hub` is the same regression by another route. Found {sources} "
        f"in:\n{copy}"
    )


def test_the_image_compiles_only_deterministic_bytecode() -> None:
    """A timestamp `.pyc` is 147.6 MB of churn; no `.pyc` at all is a tax on every command.

    CPython's default invalidation mode is `TIMESTAMP`: the header holds the mtime of the
    source it was compiled from, and uv stamps installed files with the moment of the install.
    Measured on the published layers, that made 10,130 of the 10,140 files differing between
    0.1.22 and 0.1.23 bytecode — re-pulled by every node for the eight `.py` files that
    actually changed. `rewrite-timestamp` cannot touch it, for the same reason it could not
    touch `xet/logs/`: the timestamp is inside the file rather than in its tar header.

    Shipping none was measured and rejected — the offline smoke test went from 24.0s to 43.1s,
    because `PYTHONDONTWRITEBYTECODE` in the runtime stage means nothing is cached on the way
    past and every invocation recompiles. So the bytecode is made deterministic instead, with
    PEP 552 hash-based `.pyc` files that record the source's hash in place of its mtime.

    `unchecked` is asserted rather than merely `hash-based`: `checked-hash` is equally
    reproducible and would pass a looser test, but it re-hashes every source file on every
    import, which is the run-time half of the cost this arrangement exists to avoid.
    """
    instructions = _dockerfile_instructions()

    compiled = [line for line in instructions if "UV_COMPILE_BYTECODE" in line]
    assert not compiled, (
        f"the image sets UV_COMPILE_BYTECODE, and uv writes timestamp-invalidated `.pyc` "
        f"files: a fresh digest on every build, and ~147 MB re-pulled per release. Let "
        f"`compileall --invalidation-mode unchecked-hash` do it instead. Found: {compiled}"
    )

    passes = [line for line in instructions if "compileall" in line]
    assert passes, (
        "the image must compile its bytecode with `compileall`, once in `deps` for the "
        "dependency tree and once in `build` for manicule's own code. Without it the venv "
        "ships no bytecode at all and every container start and CLI invocation recompiles — "
        "measured at 24.0s to 43.1s on the smoke test."
    )
    unchecked = [line for line in passes if "unchecked-hash" in line]
    assert unchecked == passes, (
        f"every `compileall` must pass `--invalidation-mode unchecked-hash`. Without the flag "
        f"CPython writes timestamp `.pyc` files and the churn returns; with `checked-hash` the "
        f"layers are reproducible but every import re-hashes its source. Found {len(passes)} "
        f"pass(es), {len(unchecked)} with the flag."
    )

    # Scoped to `deps`, and deliberately not to the whole file: the runtime stage has always
    # set this name, so an unscoped search passes even with the build stage letting its own
    # `python -c` steps cache timestamp bytecode into the venv after `compileall` ran.
    dont_write = [line for line in _stage_instructions("deps") if "PYTHONDONTWRITEBYTECODE" in line]
    assert dont_write, (
        "the `deps` stage must set PYTHONDONTWRITEBYTECODE. `compileall` writes through it, "
        "but the grammar seed, the vocabulary pre-seed and the weight prefetch all import "
        "manicule out of this venv, and an ordinary import caches what it touches — scattering "
        "freshly-stamped timestamp `.pyc` files over the deterministic ones. The runtime stage "
        "setting it is a different statement and does not cover this."
    )


def test_the_runtime_copies_the_dependency_tree_before_the_release_tree() -> None:
    """One `COPY` of the venv is ~327 MB re-pulled to deliver a few megabytes of Python.

    The dependency tree is ~900 MB and moves only when `uv.lock` does; manicule's own code
    moves every release. Copied as one directory they share a layer and a digest, so a version
    bump re-pulls all of it. Copied from `deps` and then from `build`, BuildKit compares
    against what the first copy already wrote and the second holds only the files that differ.

    The order is the assertion. Reversed, the stale tree lands on top of the fresh one and the
    image ships a manicule that is missing whatever the release changed.
    """
    venv_copies = [
        line
        for line in _dockerfile_instructions()
        if line.startswith("COPY") and line.rstrip().endswith("/opt/manicule/venv")
    ]

    assert len(venv_copies) == 2, (
        f"expected the venv to be copied twice — `--from=deps` for the dependency tree and "
        f"`--from=build` over the top for manicule's own — so that a release moves a layer the "
        f"size of what changed. Found {len(venv_copies)}: {venv_copies}"
    )
    order = [
        "deps" if "--from=deps" in line else "build" if "--from=build" in line else "?"
        for line in venv_copies
    ]
    assert order == ["deps", "build"], (
        f"the venv must be copied from `deps` first and `build` second. In the other order the "
        f"dependency tree overwrites manicule's own code and the image ships a stale one. "
        f"Found {order} in:\n{venv_copies[0]}\n{venv_copies[1]}"
    )
