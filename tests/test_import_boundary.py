"""Core carries no implementation dependencies, and this is what says so.

A plugin author depends on manicule's contracts. If importing those contracts also installed
a vector database, a model runtime and a web framework, nobody would, and the extension
mechanism would exist without being usable. The boundary is only real while something fails
when it is crossed.

Checked in a subprocess, because by the time this test module is imported the suite has
already imported a good deal, and a check run in-process would be measuring the wrong thing.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

IMPLEMENTATION_MODULES = (
    # storage (#2)
    "lancedb",
    "sqlalchemy",
    "alembic",
    "aiosqlite",
    "pyarrow",
    # embeddings and reranking (#3, #6)
    "mlx",
    "mlx_embeddings",
    "onnxruntime",
    "torch",
    "transformers",
    "sentence_transformers",
    "numpy",
    "tiktoken",
    # parsing (#4)
    "pypdfium2",
    "tree_sitter",
    "selectolax",
    "docx",
    "pptx",
    "nbformat",
    "markdown_it",
    # generation (#7)
    "litellm",
    "openai",
    "anthropic",
    # interfaces (#8, #11, #12)
    "fastapi",
    "uvicorn",
    "starlette",
    "typer",
    "rich",
    "click",
    "mcp",
    "fastmcp",
    "jinja2",
    # networking and scheduling (#9, #14)
    "httpx",
    "aiohttp",
    "requests",
    "apscheduler",
    "authlib",
)
"""What core must not pull in. Every entry is a library this project will eventually use."""

MANICULE_PACKAGES = (
    "manicule",
    "manicule.core",
    "manicule.config",
    "manicule.plugins",
    "manicule.container",
    "manicule.testing",
)


PARSING_LIBRARIES = (
    "pypdfium2",
    "tree_sitter",
    "tree_sitter_language_pack",
    "selectolax",
    "docx",
    "pptx",
    "nbformat",
    "markdown_it",
    "python_calamine",
    "ruamel",
    "tiktoken",
)
"""The libraries the built-in parsers are built on.

Listed separately from :data:`IMPLEMENTATION_MODULES` because the claim about them is
stronger: these *are* installed in this environment and importable, so the check below is not
"a missing package stays missing" but "an installed package is not loaded until something
needs it".
"""


def _modules_added_by_discovery() -> set[str]:
    """Run plugin discovery in a fresh interpreter and report what it loaded.

    In a subprocess for the same reason as the checks below, and one step further: by the time
    the suite reaches this module every parser has already been imported by some other test,
    so an in-process check would report nothing at all.
    """
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "from manicule.plugins import discover\n"
        "discover()\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    added: list[str] = json.loads(completed.stdout)
    return set(added)


def _modules_added_by_importing(*packages: str) -> set[str]:
    """Import ``packages`` in a fresh interpreter and report what *they* loaded.

    The difference across the import, not the contents of ``sys.modules`` afterwards. An
    interpreter arrives with things already loaded — coverage hooks, ``sitecustomize``,
    whatever the environment injects — and none of it is manicule's doing. Measuring the
    total would make this test a report on the runner rather than on the code.
    """
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        f"for name in {list(packages)!r}:\n"
        "    __import__(name)\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    added: list[str] = json.loads(completed.stdout)
    return set(added)


@pytest.mark.contract
def test_importing_manicule_pulls_in_no_implementation() -> None:
    loaded = _modules_added_by_importing(*MANICULE_PACKAGES)
    leaked = sorted(name for name in IMPLEMENTATION_MODULES if name in loaded)
    assert leaked == [], (
        f"importing manicule loaded {', '.join(leaked)}. Core defines contracts; "
        f"implementations arrive as plugins, and their imports belong inside the factories "
        f"that build them"
    )


@pytest.mark.contract
def test_core_alone_needs_nothing_but_pydantic() -> None:
    """The narrower claim: the protocols and types on their own."""
    added = _modules_added_by_importing("manicule.core")
    third_party = {
        name.split(".")[0]
        for name in added
        if not name.startswith(("_", "manicule"))
        and name.split(".")[0] not in sys.stdlib_module_names
    }
    # Pydantic and what pydantic itself needs. Nothing else has any business being here.
    assert third_party <= {
        "annotated_types",
        "pydantic",
        "pydantic_core",
        "typing_extensions",
        "typing_inspection",
    }, f"manicule.core imported {sorted(third_party)}"


@pytest.mark.contract
def test_registering_the_built_in_parsers_loads_no_parsing_library() -> None:
    """Discovery runs before configuration is read, in every process that starts.

    Registration needs two things about a parser: the media types it claims, so a document can
    be routed without building every installed parser, and its configuration model, so settings
    written for it are validated rather than ignored. Both live in ``manicule.parsers.config``,
    which imports nothing heavier than pydantic. Without that separation, an installation whose
    corpus is entirely Markdown would load pdfium, tree-sitter, python-docx, python-pptx,
    selectolax, nbformat, calamine and ruamel.yaml at startup — including for ``manicule
    doctor``, which is not going to parse anything at all.
    """
    loaded = _modules_added_by_discovery()
    leaked = sorted(
        name
        for name in PARSING_LIBRARIES
        if any(module == name or module.startswith(f"{name}.") for module in loaded)
    )
    assert leaked == [], (
        f"plugin discovery loaded {', '.join(leaked)}. A parser's library belongs inside the "
        f"factory that builds the parser; what registration needs eagerly — media types and a "
        f"config model — belongs in manicule.parsers.config"
    )


@pytest.mark.contract
def test_the_parsing_plugin_registers_through_the_public_entry_point() -> None:
    """The built-in parsers take the same route a third-party plugin takes.

    If they had a shorter internal route, the extension mechanism could stop working while
    every built-in parser still ran, and nobody would find out until somebody wrote a plugin.
    """
    from manicule.plugins import ENTRY_POINT_GROUP, installed_entry_points  # noqa: PLC0415

    found = {point.name: point.value for point in installed_entry_points(ENTRY_POINT_GROUP)}

    assert found.get("parsing") == "manicule.parsers.plugin:PLUGIN"


@pytest.mark.contract
def test_the_installed_distribution_declares_no_implementation_dependency() -> None:
    """The runtime check above cannot see a dependency that is installed but not imported."""
    from importlib.metadata import requires  # noqa: PLC0415

    declared = requires("manicule") or []
    names = {
        line.split(";")[0].split("[")[0].strip().split(" ")[0].split(">")[0].split("=")[0].lower()
        for line in declared
        if "extra ==" not in line
    }
    forbidden = {"lancedb", "sqlalchemy", "litellm", "fastapi", "typer", "httpx", "numpy"}
    assert not names & forbidden, f"manicule requires {sorted(names & forbidden)}"
