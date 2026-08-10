"""Embedding backends, and the pooling manicule does for itself.

Two runtimes, one vector. ``docs/embeddings.md`` is the design; the short version is that a
backend contributes a forward pass and nothing else, because the field a backend calls its
embedding is not reliably the reduction the model was trained with.

**This module re-exports nothing on purpose.** Plugin discovery runs in every process that
starts, including ``manicule doctor``, and it imports this package to reach
:mod:`manicule.embedding.plugin`. A convenience re-export here would make that import pull in
numpy, tokenizers and huggingface-hub for an installation that has not selected an embedder at
all — the same mistake the parsers avoid by keeping registration in
:mod:`manicule.parsers.config`. ``tests/test_import_boundary.py`` fails the build if it
returns.

Import the module you want:

- :mod:`manicule.embedding.pooling` — the reduction, and the rank check in front of it
- :mod:`manicule.embedding.cards` — what a model declares about itself
- :mod:`manicule.embedding.artifacts` — which bytes a backend runs, and which it refuses
- :mod:`manicule.embedding.cache` — the fingerprint-keyed memo
- :mod:`manicule.embedding.base` — the shared path both backends sit behind
- :mod:`manicule.embedding.runtimes` — the two backends, and the seams over untyped
  libraries they sit behind
"""

from __future__ import annotations

__all__: list[str] = []
