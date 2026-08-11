"""The command line: nineteen commands over the application service, and nothing else.

Importing this package imports Typer and Rich. That belongs to a surface and would not belong
in :mod:`manicule.core`, which carries no implementation dependencies at all.

Only :data:`app` is re-exported. The console script's entry point is
``manicule.cli.main:main``, and re-exporting that function here would shadow the *module* of
the same name — so ``from manicule.cli import main`` would hand back a function to anybody
who wanted the module, which is a trap worth not setting.
"""

from __future__ import annotations

from manicule.cli.main import app

__all__ = ["app"]
