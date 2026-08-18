"""The single place MLX is imported outside the backend's own methods.

MLX installs on Apple Silicon and nowhere else, so on Linux — where CI type-checks — ``import
mlx.core`` cannot resolve at all. Confining the import here is what lets the root
``pyproject.toml`` switch ``reportMissingImports`` off for this package alone, rather than
following MLX into every caller. The narrow loss — a misspelled import here goes unreported by
the checker — is covered by the thing that cannot be fooled, which is running the code.

This module moved out of ``manicule.embedding.runtimes`` when the backend did. It was always
MLX-specific; it stayed in manicule only because the backend did.
"""

from __future__ import annotations

from types import ModuleType


def mlx_core() -> ModuleType:
    """The ``mlx.core`` module."""
    import mlx.core as mx  # noqa: PLC0415 - Apple-only, and the reason this function exists

    return mx


def mlx_usable() -> bool:
    """Whether MLX is installed *and* can actually evaluate on this machine.

    Both halves matter. The package can be present in an environment whose accelerator is not,
    and the failure that produces is not an ``ImportError`` — so importability alone is not the
    question worth asking.
    """
    try:
        mx = mlx_core()
        mx.eval(mx.array([1.0]) + 1)
    except Exception:  # noqa: BLE001 - any failure here means the same thing: no usable MLX
        return False
    return True


__all__ = ["mlx_core", "mlx_usable"]
