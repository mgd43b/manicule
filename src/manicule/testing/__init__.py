"""Conformance suites every implementation of a manicule protocol should pass."""

from __future__ import annotations

from manicule.testing.contracts import (
    assert_chunker_contract,
    assert_connector_contract,
    assert_embedder_contract,
    assert_middleware_contract,
    assert_parser_contract,
    assert_refuses_oversized_chunks,
    assert_retrieval_stage_contract,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_rejects_foreign_vectors,
)

__all__ = [
    "assert_chunker_contract",
    "assert_connector_contract",
    "assert_embedder_contract",
    "assert_middleware_contract",
    "assert_parser_contract",
    "assert_refuses_oversized_chunks",
    "assert_retrieval_stage_contract",
    "assert_vector_store_is_dimension_agnostic",
    "assert_vector_store_rejects_foreign_vectors",
]
