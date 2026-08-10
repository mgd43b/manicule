"""manicule — self-hosted document search and answers, with citations that resolve.

Importing this package gives you the contracts and the wiring, and nothing heavier: no
vector store, no model runtime, no web framework. Implementations arrive as plugins, found
through the ``manicule.plugins`` entry-point group.
"""

from __future__ import annotations

from manicule.core.version import CORE_VERSION

__version__ = CORE_VERSION

__all__ = ["__version__"]
