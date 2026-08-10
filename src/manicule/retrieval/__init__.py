"""Hybrid retrieval: two legs, a fusion, a rerank, and a context inside a token budget.

    dense (vectors) + BM25 (lexical) -> RRF -> cross-encoder -> context assembly

The design in five sentences, from ``docs/retrieval.md``:

**A pipeline is a declared list of uniform stages. Every stage's output is already live,
in-workspace and visible. Fusion sees ranks and never scores. Confidence describes the
retrieval, not the answer. Nothing new ships without a measured improvement.**

The second of those is why the workspace and soft-delete boundary is enforced inside the dense
leg rather than once at the end. Enforcing at the end is correct and useless: the excluded rows
have already consumed the top-``k`` slots, and the query returns a well-formed empty list.
Enforcing at every stage boundary makes it an invariant that can be asserted at any point,
which is what turns "we filtered" into something a test can fail on.

Nothing in this package is imported by ``import manicule``. The tokenizer and the cross-encoder
runtime are imported inside the factories that need them.
"""

from __future__ import annotations

from manicule.retrieval.assembly import ContextAssembler, ContextBudgetError
from manicule.retrieval.cache import CachedRanking, L1QueryCache, cache_key, rehydrate
from manicule.retrieval.confidence import band_for, score_confidence
from manicule.retrieval.config import (
    DEFAULT_RERANKER_MODEL,
    DenseConfig,
    FusionConfig,
    LexicalConfig,
    RerankConfig,
)
from manicule.retrieval.dense import DenseStage, derive_over_fetch
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.hydration import visible_documents
from manicule.retrieval.lexical import LexicalStage
from manicule.retrieval.merging import union_scored
from manicule.retrieval.ports import (
    SupportsDocumentCount,
    SupportsIndexState,
    SupportsLiveChunkCount,
    SupportsVectorCount,
)
from manicule.retrieval.prefilter import Resolution, resolve
from manicule.retrieval.profile import Profiles, retrieval_depth
from manicule.retrieval.rerank import CrossEncoderReranker, PairScorer
from manicule.retrieval.retriever import RetrievalResult, Retriever
from manicule.retrieval.router import QueryRouter, Routing, UtilityKind
from manicule.retrieval.runner import PipelineRun, PipelineRunner
from manicule.retrieval.tokens import ContextTokenCounter, ContextTokenDriftError
from manicule.retrieval.trace import Regime, RetrievalTrace, Route, Shortfall, StageSpan
from manicule.retrieval.utility import UtilityAnswer, handlers_for

__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "CachedRanking",
    "ContextAssembler",
    "ContextBudgetError",
    "ContextTokenCounter",
    "ContextTokenDriftError",
    "CrossEncoderReranker",
    "DenseConfig",
    "DenseStage",
    "FusionConfig",
    "L1QueryCache",
    "LexicalConfig",
    "LexicalStage",
    "PairScorer",
    "PipelineRun",
    "PipelineRunner",
    "Profiles",
    "QueryRouter",
    "RRFStage",
    "Regime",
    "RerankConfig",
    "Resolution",
    "RetrievalResult",
    "RetrievalTrace",
    "Retriever",
    "Route",
    "Routing",
    "Shortfall",
    "StageSpan",
    "SupportsDocumentCount",
    "SupportsIndexState",
    "SupportsLiveChunkCount",
    "SupportsVectorCount",
    "UtilityAnswer",
    "UtilityKind",
    "band_for",
    "cache_key",
    "derive_over_fetch",
    "handlers_for",
    "rehydrate",
    "resolve",
    "retrieval_depth",
    "score_confidence",
    "union_scored",
    "visible_documents",
]
