"""The console script's entry point, and the only manicule module that runs before the
program's dependencies are known to be there.

`manicule.cli` imports Typer at package scope, and Typer is in the `serve` extra — deliberately,
because `tests/test_import_boundary.py` refuses it in core and says why. The consequence is that
`pip install manicule` on its own produces a `manicule` command whose first act is a nine-line
traceback ending in `ModuleNotFoundError: No module named 'typer'`. That is a true statement
about the process and a useless one to the person who typed `manicule --version`: it names a
library they have never heard of and no action they can take.

This module is what turns it into a sentence. It lives at `manicule.entry` rather than
`manicule.cli.entry` because Python imports a package before its submodules — `manicule.cli.entry`
would execute `manicule/cli/__init__.py`, and therefore `import typer`, before any guard here
could run. `manicule/__init__.py` reads distribution metadata and nothing else, so importing this
module is safe on a bare install.

Only a missing *extra* is translated, and "missing" means `ModuleNotFoundError` specifically
rather than `ImportError` broadly. The distinction is load-bearing: `from typer import Removed`
against an incompatible Typer raises a plain `ImportError` whose `.name` is still `'typer'`, so
catching the base class would answer a version conflict with "install `manicule[all]`" — advice
that cannot help, over a real incompatibility it has just hidden. An installed-but-wrong
dependency and an absent one need different sentences, and only the second one is this module's.

Anything else propagates untouched, for the same reason: a manicule whose own modules fail to
import is a broken installation rather than an incomplete one.
"""

from __future__ import annotations

import sys

#: Third-party module name -> the extra that provides it. Only the modules `manicule.cli` and
#: its imports reach at *module* scope belong here; a dependency imported inside a command is
#: that command's failure to report, with the context of what was being attempted.
_PROVIDED_BY = {
    "typer": "serve",
    "click": "serve",
    "rich": "serve",
    "fastmcp": "serve",
    "mcp": "serve",
    "pydantic_settings": "serve",
}

_HINT = """\
manicule is installed without the dependencies its command line needs: no module named {module!r},
which comes from the {extra!r} extra.

Install the program rather than the library:

    uv tool install "manicule[all]"

or, into the environment you are already in:

    pip install "manicule[all]"

`manicule[all]` is every extra except `rerank` and `browser-auth`, which download gigabytes and
are opted into by name — `manicule[all,rerank]`. See https://github.com/mgd43b/manicule#install
"""


def install_hint(exc: ModuleNotFoundError) -> str | None:
    """What to tell someone about `exc`, or `None` if this module has nothing to say about it.

    Separate from :func:`main` so that the decision can be tested against a constructed
    exception. Exercising it through the import statement instead would need the import to fail
    on demand, and a test that reaches for that is testing its own machinery.
    """
    extra = _PROVIDED_BY.get(exc.name or "")
    return None if extra is None else _HINT.format(module=exc.name, extra=extra)


def main() -> None:
    """The console-script entry point, past the guard."""
    try:
        # Deferred on purpose, and the deferral *is* this module. At the top of the file this
        # import runs at interpreter start, before the `except` below exists to catch it, and
        # the guard becomes the traceback it was written to replace.
        from manicule.cli.main import main as _main  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        hint = install_hint(exc)
        if hint is None:
            raise
        # stderr, because stdout is the `--json` channel and a caller piping this into `jq`
        # should read an empty stream rather than prose. Same rule the CLI itself holds to.
        sys.stderr.write(hint)
        raise SystemExit(1) from exc

    _main()


__all__ = ["install_hint", "main"]
