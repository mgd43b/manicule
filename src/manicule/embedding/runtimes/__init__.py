"""Everything that touches a library shipping no type information.

Three dependencies of the embedding stack are untyped or partly typed — ``onnxruntime``,
``tokenizers`` and ``huggingface-hub``. Under pyright's strict mode every call into them yields
an ``Unknown`` that spreads to each expression downstream, thousands of errors that say nothing
about this code. ``pyproject.toml`` relaxes the four rules that report *that*, and only for this
directory.

The point of the directory is where the boundary falls. Everything deciding what a vector
**is** — :mod:`manicule.embedding.pooling`, :mod:`manicule.embedding.cards`,
:mod:`manicule.embedding.artifacts`, :mod:`manicule.embedding.base` — stays fully checked, and
reaches these libraries only through the small typed functions here. What is permitted is a
value a third party declined to type, at the one seam where it arrives.

Hand-written stubs were the alternative, and a stub that drifts from its library is worse than
none: it type-checks confidently against an API that no longer exists.

**MLX is no longer among them.** ``mlx-embeddings`` is GPL-3.0, so the backend that links it
lives in the ``manicule-mlx`` distribution, and the ``mlx_core``/``mlx_usable`` helpers went
with it to :mod:`manicule_mlx.runtime`. That package declares its own execution environment in
the root ``pyproject.toml``, for the same two reasons this directory has one: an untyped
library, and a platform-conditional import that cannot resolve on the Linux runner where CI
type-checks.
"""

from __future__ import annotations

__all__: list[str] = []
