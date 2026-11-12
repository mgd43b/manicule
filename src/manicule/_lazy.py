"""Package re-exports that import their module the first time a name is used.

A package ``__init__`` that re-exports its submodules' names imports every one of those
submodules whenever *anything* below it is imported, because Python runs the package before
the submodule. ``manicule.core`` re-exports its whole vocabulary, so reading
``manicule.core.version`` used to import acquisition, anchors, content, embedding, protocols and
the rest, and the command line paid for all of it before ``--version`` could answer.

:func:`lazy_exports` keeps every public import path working — ``from manicule.core import
Chunk`` still resolves — and defers the import to the first use of the name. The package keeps
its imports under ``TYPE_CHECKING`` so that the checker and readers still see where each name
comes from; ``tests/test_import_boundary.py`` holds the two lists to each other.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence


def lazy_exports(
    namespace: MutableMapping[str, Any], exports: Mapping[str, Sequence[str]]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """A module ``__getattr__`` and ``__dir__`` serving ``exports`` from ``namespace``.

    ``exports`` maps a module to the names the package re-exports from it. A resolved name is
    stored in ``namespace``, so each import happens once and later lookups are ordinary
    attribute reads that never reach ``__getattr__`` again.
    """
    origin = {name: module for module, names in exports.items() for name in names}
    package = str(namespace["__name__"])

    def __getattr__(name: str) -> Any:  # noqa: N807, ANN401 - the module protocol, any value
        module = origin.get(name)
        if module is None:
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        value = getattr(importlib.import_module(module), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:  # noqa: N807 - the module protocol's name
        return sorted({*namespace, *origin})

    return __getattr__, __dir__


__all__ = ["lazy_exports"]
