"""The command line: nineteen commands over the application service, and nothing else.

Importing this package imports Typer and Rich. That belongs to a surface and would not belong
in :mod:`manicule.core`, which carries no implementation dependencies at all.

Only :data:`app` is re-exported. :func:`manicule.cli.main.main` is what actually runs the
command line, and re-exporting it here would shadow the *module* of the same name — so
``from manicule.cli import main`` would hand back a function to anybody who wanted the module,
which is a trap worth not setting.

The console script does **not** point here. Its entry point is :mod:`manicule.entry`, because
the import on the line below is the one that fails on an installation without the ``serve``
extra: a guard inside this package would be unreachable, having already raised on the way in.
"""

from __future__ import annotations

from manicule.cli.main import app

__all__ = ["app"]
