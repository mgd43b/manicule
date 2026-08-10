"""Conformance suites every implementation of a manicule protocol should pass."""

from __future__ import annotations

from manicule.testing.contracts import (
    assert_chunk_relation_store_contract,
    assert_chunker_contract,
    assert_collection_store_contract,
    assert_connector_contract,
    assert_embedder_contract,
    assert_local_only_policy_is_enforced,
    assert_middleware_contract,
    assert_parser_contract,
    assert_pipeline_enforces_scope,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
    assert_retrieval_stage_contract,
    assert_tag_store_contract,
    assert_trash_store_contract,
    assert_vector_store_is_dimension_agnostic,
    assert_vector_store_rejects_foreign_vectors,
    assert_version_store_contract,
    closing,
)
from manicule.testing.normalise import NORMALISER_VERSION, normalise
from manicule.testing.roundtrip import (
    ParserProfile,
    RoundTripReport,
    assert_location_budget,
    assert_round_trip,
)

__all__ = [
    "NORMALISER_VERSION",
    "ParserProfile",
    "RoundTripReport",
    "assert_chunk_relation_store_contract",
    "assert_chunker_contract",
    "assert_collection_store_contract",
    "assert_connector_contract",
    "assert_embedder_contract",
    "assert_local_only_policy_is_enforced",
    "assert_location_budget",
    "assert_middleware_contract",
    "assert_parser_contract",
    "assert_pipeline_enforces_scope",
    "assert_protocol_signatures",
    "assert_refuses_oversized_chunks",
    "assert_retrieval_stage_contract",
    "assert_round_trip",
    "assert_tag_store_contract",
    "assert_trash_store_contract",
    "assert_vector_store_is_dimension_agnostic",
    "assert_vector_store_rejects_foreign_vectors",
    "assert_version_store_contract",
    "closing",
    "normalise",
]
