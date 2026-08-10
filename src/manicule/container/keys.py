"""Typed keys for every extension point.

``container.get(keys.EMBEDDER)`` returns an :class:`~manicule.core.protocols.Embedder`, and
a type checker knows it. Resolution by string returns something a type checker has to be
told about, which is where the telling stops matching the truth.
"""

from __future__ import annotations

from manicule.core.protocols import (
    Chunker,
    Connector,
    DocStore,
    Embedder,
    Generator,
    Middleware,
    Parser,
    Reranker,
    RetrievalStage,
    VectorStore,
)
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import ComponentKey

PARSER: ComponentKey[Parser] = ComponentKey(ComponentKind.PARSER)
CHUNKER: ComponentKey[Chunker] = ComponentKey(ComponentKind.CHUNKER)
EMBEDDER: ComponentKey[Embedder] = ComponentKey(ComponentKind.EMBEDDER)
VECTOR_STORE: ComponentKey[VectorStore] = ComponentKey(ComponentKind.VECTOR_STORE)
DOC_STORE: ComponentKey[DocStore] = ComponentKey(ComponentKind.DOC_STORE)
RETRIEVAL_STAGE: ComponentKey[RetrievalStage] = ComponentKey(ComponentKind.RETRIEVAL_STAGE)
RERANKER: ComponentKey[Reranker] = ComponentKey(ComponentKind.RERANKER)
GENERATOR: ComponentKey[Generator] = ComponentKey(ComponentKind.GENERATOR)
CONNECTOR: ComponentKey[Connector] = ComponentKey(ComponentKind.CONNECTOR)
MIDDLEWARE: ComponentKey[Middleware] = ComponentKey(ComponentKind.MIDDLEWARE)

__all__ = [
    "CHUNKER",
    "CONNECTOR",
    "DOC_STORE",
    "EMBEDDER",
    "GENERATOR",
    "MIDDLEWARE",
    "PARSER",
    "RERANKER",
    "RETRIEVAL_STAGE",
    "VECTOR_STORE",
]
