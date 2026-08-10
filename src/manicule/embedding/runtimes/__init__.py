"""Everything that touches a library shipping no type information.

Four dependencies of the embedding stack are untyped or partly typed — ``mlx-embeddings``,
``onnxruntime``, ``tokenizers`` and ``huggingface-hub``. Under pyright's strict mode every call
into them yields an ``Unknown`` that spreads to each expression downstream, thousands of errors
that say nothing about this code. ``pyproject.toml`` relaxes the four rules that report *that*,
and only for this directory.

The point of the directory is where the boundary falls. Everything deciding what a vector
**is** — :mod:`manicule.embedding.pooling`, :mod:`manicule.embedding.cards`,
:mod:`manicule.embedding.artifacts`, :mod:`manicule.embedding.base` — stays fully checked, and
reaches these libraries only through the small typed functions here. What is permitted is a
value a third party declined to type, at the one seam where it arrives.

Hand-written stubs were the alternative, and a stub that drifts from its library is worse than
none: it type-checks confidently against an API that no longer exists. That risk is not
theoretical here — ``mlx-embeddings`` is version 0.1.0 and already binds one attribute to
different meanings on different architectures.

Two of those dependencies are also **platform-conditional**: MLX installs on Apple Silicon and
nowhere else, so on Linux — where CI type-checks — ``import mlx.core`` cannot resolve at all.
``reportMissingImports`` is therefore off for this directory, which is a real if narrow loss: a
misspelled import here would go unreported by the checker. It is covered instead by the thing
that cannot be fooled, which is running the code — the plugin factory imports these modules and
``tests/test_embedding_backends.py`` builds and runs both backends. The suppression is confined
to this directory precisely so that everything importable everywhere keeps being checked, which
is why :func:`mlx_core` exists rather than each caller importing MLX for itself.
"""

from __future__ import annotations

from types import ModuleType


def mlx_core() -> ModuleType:
    """The ``mlx.core`` module.

    The single place MLX is imported outside a backend's own methods. Callers that need raw MLX
    — currently only the test asserting what the convenience field does — come through here, so
    that ``import mlx.core`` never appears in a file a type checker is expected to resolve on a
    platform where MLX does not exist.
    """
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
