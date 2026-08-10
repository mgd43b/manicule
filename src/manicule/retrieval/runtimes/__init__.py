"""Model runtimes for retrieval. One directory, so one suppression covers them.

Everything that imports a machine-learning framework lives here and nowhere else. The stages
themselves reach a runtime through :class:`~manicule.retrieval.rerank.PairScorer`, which is
pure Python, so the module that decides what a *ranking* is stays fully type-checked while the
untyped third-party surface is confined to this directory.
"""

from __future__ import annotations

__all__: list[str] = []
