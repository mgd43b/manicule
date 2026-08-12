"""The staged grammar pack: one directory named in three files that cannot see each other.

#80: the image build downloaded the tree-sitter release on every run, so a dropped transfer at
a third-party host blocked every merge in the repository. It is staged now — CI restores the
pack from a cache and hands it to the build — and that arrangement is spread across three files
with nothing but convention holding them together:

- ``.github/workflows/ci.yml`` copies the libraries into a directory,
- ``.dockerignore`` decides whether that directory is in the build context at all,
- ``Dockerfile`` looks for it and downloads the release when it is not there.

**Every way this drifts is silent, and one way is silent and green.** Rename the directory on
one side and the build stops finding it — and *keeps working*, because not finding it means
fetching the release, which is exactly the behaviour the change was made to stop relying on. CI
would go green while the cache it reports having restored is used by nothing. That is the
failure this module exists to catch, and it is the same shape as the one
``tests/test_ci_switches.py`` was written for: a mechanism that is correct, configured, and
pointed at nothing.

Read out of the three files rather than from a constant kept here. A fourth copy of the path
would drift from the other three and the drift would be invisible, which is the defect itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

STAGED_COPY = re.compile(r"^COPY\s+(\S+)\s+(\S*grammar-cache)\s*$", re.MULTILINE)
"""The line that carries the staged pack into the build, and the two paths it names."""


def instructions() -> str:
    """The Dockerfile with its commentary removed, which is the only honest thing to search.

    Written after two of the assertions below were caught passing on prose. This file explains
    itself at length — ``--network=none`` and ``--prefetch`` are both *discussed* in comments
    several lines from where they are *used* — so ``"--prefetch" in dockerfile`` stayed true
    with the flag deleted from the instruction that runs it. Both guards reported a build step
    that no longer existed, which is precisely the failure they were written to catch, one level
    up. Shell comments inside the heredocs go too: they are commentary in the same way.
    """
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def staged_directory() -> str:
    """The context directory the image build reads its grammars from, per the Dockerfile."""
    found = STAGED_COPY.search(DOCKERFILE.read_text(encoding="utf-8"))
    assert found is not None, (
        "no `COPY <dir> <...grammar-cache>` in the Dockerfile. The image build takes its "
        "grammars from a staged directory; if that is no longer how it works, this module and "
        "the container job in .github/workflows/ci.yml are both describing something gone"
    )
    return found.group(1)


def container_commands() -> list[str]:
    """Every shell command the container job runs, in order."""
    from ruamel.yaml import YAML  # noqa: PLC0415 - test-only, see tests/test_ci_switches.py

    workflow = cast(
        "dict[str, Any]",
        YAML(typ="safe").load(WORKFLOW.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )
    steps = cast("list[dict[str, Any]]", workflow["jobs"]["container"]["steps"])
    return [str(step.get("run", "")) for step in steps]


def test_the_directory_ci_stages_into_is_the_one_the_image_build_reads() -> None:
    """The rename that leaves a green job quietly downloading the release on every run.

    Nothing fails when these two disagree. The build finds no staged pack, falls back to
    fetching it, and passes — which is the pre-#80 behaviour wearing the cache's clothes, and
    the log line that would tell you so is one line in a build that prints thousands.
    """
    staged = staged_directory()
    commands = container_commands()

    assert any(staged in command for command in commands), (
        f"the Dockerfile builds its grammars from {staged!r} in the build context and no step "
        f"of the container job writes there. The build will not fail — it will download the "
        f"release itself, on every run, which is what the staging exists to stop"
    )


def test_both_sides_name_the_staged_directory_after_the_pack_release() -> None:
    """What keeps a stale cache from being packaged under a release it is not.

    The libraries are compiled against one ``tree-sitter-language-pack`` release, and the bundle
    the image ships *records* a release. Staging them under a directory named for the version,
    and looking for the version the image installs, is what makes a mismatch a miss — it falls
    through to the download — instead of a bundle whose manifest names a release its libraries
    did not come from. Drop the version from either side and that becomes possible, with nothing
    to see: the bundle builds, the image ships, and its manifest is wrong.
    """
    staging = [command for command in container_commands() if staged_directory() in command]

    assert any("pack_version()" in command for command in staging), (
        "the container job stages grammars into a directory that is not named for the pack "
        "release. A cache left over from another release would then be staged as if it were "
        "this one"
    )
    assert re.search(r"grammar-cache/\$\{installed\}", instructions()), (
        "the Dockerfile no longer looks for the staged directory belonging to the pack release "
        "it installs, so libraries built for another release could be packaged under this one"
    )


def test_the_build_context_admits_the_staged_directory() -> None:
    """``.dockerignore`` excludes everything by default, so this needs saying explicitly.

    Checked as text, and the limit is worth stating: this catches the negation being deleted,
    which is the way it actually goes wrong. It does not reimplement Docker's matching, so it
    cannot prove the directory arrives — the container job proves that, by failing at the
    ``COPY`` if it does not.
    """
    staged = staged_directory()
    ignore = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    admitted = {
        line.strip().removeprefix("!").rstrip("/") for line in ignore if line.startswith("!")
    }
    prefixes = {staged, *(str(parent) for parent in Path(staged).parents if str(parent) != ".")}

    assert admitted & prefixes, (
        f"nothing in .dockerignore re-includes {staged!r}, and the first line of that file "
        f"excludes everything. The image build would fail at the COPY that reads it"
    )


def test_a_clean_clone_can_still_build_the_image() -> None:
    """Somebody who has never run CI must be able to type ``docker build .`` and get an image.

    Two things make that true and both are easy to remove while CI stays green, because CI
    always has the other path. The placeholder is what makes the ``COPY`` resolve when nothing
    has been staged — an absent directory is a build that fails on its first grammar
    instruction — and the download is what fills a bundle that the empty directory cannot.
    """
    staged = REPO_ROOT / staged_directory()

    assert staged.is_dir(), f"{staged_directory()} is not in the repository, so `COPY` fails"
    assert any(path.name == ".gitkeep" for path in staged.iterdir()), (
        f"{staged_directory()} carries no committed placeholder, so git will not clone the "
        f"directory and the Dockerfile's COPY of it fails on a fresh checkout"
    )
    assert "--prefetch" in instructions(), (
        "the Dockerfile no longer has a branch that fetches the grammar release, so an image "
        "can only be built by someone who has already staged one"
    )


def test_the_image_is_still_proved_offline() -> None:
    """#22's guarantee, which the staging must not have quietly relaxed.

    The smoke test is the assertion that the image needs no network; staging changes where the
    *build* gets its grammars and must leave that untouched. ``docs/deployment.md`` and
    ``docs/web.md`` both promise this flag by name, and until now nothing checked it was there.
    """
    assert "--network=none" in instructions(), (
        "the image's smoke test no longer runs with the network switched off, so a build "
        "passing it no longer says the image needs no network — which two documents promise"
    )
