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
"""

from __future__ import annotations

__all__: list[str] = []
