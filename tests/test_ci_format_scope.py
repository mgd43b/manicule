"""The formatter's scope is decided once, so the obvious command and CI's check cannot disagree.

``ruff format .`` is the command a developer runs. CI used to check ``src tests packages``. The
two sets were not the same: ``.`` also rewrites ``tools/`` and the Python inside markdown
documents, and **nothing verified those**. So the natural command introduced changes CI could
never fail on, they drifted, and the exclusion ended up written down twice — the second time
silently undoing the first, in one session.

The fix is not a longer path list. It is that there is **no** path list: the scope lives in
``[tool.ruff.format] exclude`` in ``pyproject.toml``, which every invocation of ruff reads, so
CI running ``ruff format --check .`` is running the developer's command over the developer's
files. Two scopes cannot drift apart when there is one of them.

That is the property here, and it is asserted out of the workflow file CI actually runs rather
than a copy kept beside it — a second copy would drift, which is the defect under test.

**What this does not catch, said plainly.** It does not check that the exclusions are the *right*
ones; that is a judgement recorded in the comment beside them. It does not run ruff, so a scope
that is single-sourced and wrong still passes. And a *new* format step added to some other
workflow with its own path list is invisible here — this reads the lint job of ``ci.yml``, which
is where the check lives today.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

RUFF_FORMAT = re.compile(r"^\s*run:\s*uv run ruff format (?P<args>.+)$", re.MULTILINE)
RUFF_CHECK = re.compile(r"^\s*run:\s*uv run ruff check (?P<args>.+)$", re.MULTILINE)


def _workflow() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ruff" in text, f"{WORKFLOW} does not mention ruff; this file is reading the wrong one"
    return text


def test_ci_formats_the_same_thing_the_developer_command_does() -> None:
    """The format step is ``ruff format --check .`` — no path list, so no second scope.

    A path list here is the defect itself, not a style preference: it is a set of files that can
    differ from the set ``ruff format .`` rewrites, and every file in the difference is one a
    developer reformats and CI never checks.
    """
    found = RUFF_FORMAT.findall(_workflow())
    assert found, "no `uv run ruff format` step in the workflow; the format check has gone"
    for args in found:
        assert args.split() == ["--check", "."], (
            f"the format step runs `ruff format {args}`. It must be `ruff format --check .`, "
            f"with the scope in `[tool.ruff.format] exclude` rather than in this argument list. "
            f"A path list is a second scope, and the two drift: `src tests packages` used to "
            f"leave `tools/` and markdown code blocks formatted by the obvious command and "
            f"verified by nothing."
        )


def test_ci_lints_the_same_way() -> None:
    """The same property for ``ruff check``, which already held and is worth keeping."""
    found = RUFF_CHECK.findall(_workflow())
    assert found, "no `uv run ruff check` step in the workflow; the lint check has gone"
    for args in found:
        assert args.split() == ["."], (
            f"the lint step runs `ruff check {args}`; it must be `ruff check .` for the same "
            f"reason the format step must be `.`."
        )


def test_the_format_exclusions_are_still_linted() -> None:
    """Excluded from *formatting* only, which is what makes excluding them safe.

    ``tools/`` is generator scripts and the design documents are prose, so neither is the
    formatter's business — but ``tools/`` is still real Python that ships in the repository, and
    a top-level ``exclude`` would drop it out of ``ruff check`` too, silently. The narrow key is
    the whole point, so this asserts the broad one was not used instead.
    """
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ruff = config["tool"]["ruff"]
    excluded = ruff.get("format", {}).get("exclude", [])
    assert excluded, (
        "`[tool.ruff.format] exclude` is empty. The workflow passes no path list, so this is now "
        "the only place the formatter's scope is written down; emptying it silently widens what "
        "the formatter rewrites."
    )
    assert "exclude" not in ruff, (
        f"a top-level `[tool.ruff] exclude` was added: {ruff.get('exclude')}. That hides files "
        f"from `ruff check` as well as from the formatter, and `tools/` is meant to stay linted "
        f"— `[tool.ruff.lint.per-file-ignores]` carries entries for it that would stop applying "
        f"to anything."
    )
