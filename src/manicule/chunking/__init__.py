"""Structure-aware chunking: blocks in, retrievable chunks out.

Importing this package costs nothing beyond the standard library and pydantic. The token
counter it needs arrives from the bound embedder, and ``tiktoken`` — the stand-in used when
no embedder is bound — is imported on first use, inside the function that needs it.
"""

from __future__ import annotations

from manicule.chunking.chunker import (
    BREADCRUMB_TOKENS,
    CHUNKER_NAME,
    CHUNKER_VERSION,
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_TOKENS,
    StructuralChunker,
    finalize_chunks,
)
from manicule.chunking.tokens import PROVISIONAL_SAFETY_FACTOR, SupportsTokenCount, TokenCounter

__all__ = [
    "BREADCRUMB_TOKENS",
    "CHUNKER_NAME",
    "CHUNKER_VERSION",
    "MAX_TOKENS",
    "MIN_TOKENS",
    "OVERLAP_TOKENS",
    "PROVISIONAL_SAFETY_FACTOR",
    "StructuralChunker",
    "finalize_chunks",
    "SupportsTokenCount",
    "TokenCounter",
]
