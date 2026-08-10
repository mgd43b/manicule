"""Every built-in parser, behind the ``Parser`` protocol.

**Importing this package is cheap and stays cheap.** Each parser lives in its own module and
imports its library at that module's top level, and nothing here imports those modules. The
registration in :mod:`manicule.parsers.plugin` names them as strings and constructs them in a
factory, so an installation whose corpus is all Markdown never loads pdfium, tree-sitter or
the Office readers at all.

What every parser here has in common is stated once, in :mod:`manicule.parsers.base`: an
anchor is built only from a location the library reports, ordinals are converted to 1-based
inclusive at the parser boundary, and a parser that cannot handle its input declines rather
than guessing.
"""

from __future__ import annotations

__all__: list[str] = []
