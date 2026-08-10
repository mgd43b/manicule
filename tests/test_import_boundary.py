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


def _modules_after_importing(*packages: str) -> set[str]:
    """Import ``packages`` in a clean interpreter and report what ended up loaded."""
    script = (
        "import json, sys\n"
        f"for name in {list(packages)!r}:\n"
        "    __import__(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded: list[str] = json.loads(completed.stdout)
    return set(loaded)


@pytest.mark.contract
def test_importing_manicule_pulls_in_no_implementation() -> None:
    loaded = _modules_after_importing(*MANICULE_PACKAGES)
    leaked = sorted(name for name in IMPLEMENTATION_MODULES if name in loaded)
    assert leaked == [], (
        f"importing manicule loaded {', '.join(leaked)}. Core defines contracts; "
        f"implementations arrive as plugins, and their imports belong inside the factories "
        f"that build them"
    )


@pytest.mark.contract
def test_core_alone_needs_nothing_but_pydantic() -> None:
    """The narrower claim: the protocols and types on their own."""
    loaded = _modules_after_importing("manicule.core")
    third_party = {
        name.split(".")[0]
        for name in loaded
        if not name.startswith(("_", "manicule"))
        and name.split(".")[0] not in sys.stdlib_module_names
    }
    assert third_party <= {
        "pydantic",
        "pydantic_core",
        "annotated_types",
        "typing_extensions",
        "typing_inspection",
    }, f"manicule.core imported {sorted(third_party)}"


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
