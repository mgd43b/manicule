"""What a client can establish about a corpus, and what it costs to establish it.

The recipe under test is :mod:`tests.mcp.qualification`, and these are the claims it has to
support before anybody relies on it: that a collection resolves to one thing, that its size is
a computed number rather than a remembered one, that a scope reaches every search and is
reported back, that an unknown scope stops the search rather than widening it, that a passage
carries enough to cite, that a topic the corpus never mentions produces a refusal rather than a
plausible sentence — and that all of it fits in a budget written down in advance.

**No model runs here, and none is mocked either.** Everything below is retrieval and
arithmetic, so a failure is a failure of the recipe rather than of a sampler. What a hosted
model does with :class:`~tests.mcp.qualification.Qualification` is documented as an optional,
manually recorded exercise in ``docs/surfaces.md`` §4.2; it is not something a suite should be
made to depend on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import Client

from manicule.mcp.server import build_server
from tests.mcp.qualification import (
    ABSENT_TOPIC,
    ABSENT_TOPIC_TERMS,
    CHUNK_COUNT,
    COLLECTION,
    DOCUMENT_COUNT,
    EVIDENCE_QUERIES,
    Budget,
    build_fixture,
    deduplicate,
    qualify,
)

if TYPE_CHECKING:
    from tests.mcp.qualification import Qualification


async def _qualification() -> Qualification:
    service, _ = await build_fixture()
    return await qualify(service)


# --- identity and counts ----------------------------------------------------------------------


async def test_the_collection_resolves_to_exactly_one_target() -> None:
    """One name in, one collection out, and an id to carry forward.

    "Exactly one" is the part worth asserting: a client that resolved a name to two collections
    and picked the first would produce a scoped, confident, wrong answer.
    """
    qualified = await _qualification()
    assert qualified.collections_seen == 1
    assert qualified.collection_name == COLLECTION
    assert qualified.collection_id


async def test_the_counts_are_the_fixture_counted_rather_than_a_number_carried_along() -> None:
    """The totals match the corpus, and they came from ``collection_counts``."""
    qualified = await _qualification()
    assert (qualified.documents, qualified.chunks) == (DOCUMENT_COUNT, CHUNK_COUNT)


async def test_a_collection_listing_reports_no_total_of_its_own() -> None:
    """The invariant #03 asks to preserve, asserted as an absence.

    Copying a remembered total into ``collection_list`` would shorten this recipe by one call
    and make every number it reports as old as the last write. There is a place that counts;
    this is not it.
    """
    service, _ = await build_fixture()
    async with Client(build_server(service)) as client:
        listed = (await client.call_tool("collection_list", {})).structured_content or {}
    row: dict[str, Any] = listed["data"]["collections"][0]
    assert "documents" not in row
    assert "chunks" not in row


async def test_the_count_is_computed_from_membership_every_time_it_is_asked() -> None:
    """Membership changes and the next count changes with it, with nothing recomputed by hand.

    This is what "computed rather than stored" means operationally, and it is the assertion a
    cache would break. A stored total would still be right immediately after the write that set
    it — so the test moves membership *without* touching any total, and asks again.
    """
    service, backend = await build_fixture()
    collection_id = next(iter(backend.organization_.collections))
    async with Client(build_server(service)) as client:
        before = (
            await client.call_tool("collection_counts", {"collection_id": collection_id})
        ).structured_content or {}
        removed = next(iter(backend.organization_.members[collection_id]))
        await backend.organization_.remove_from_collection(collection_id, [removed])
        after = (
            await client.call_tool("collection_counts", {"collection_id": collection_id})
        ).structured_content or {}

    assert before["data"]["documents"] == DOCUMENT_COUNT
    assert after["data"]["documents"] == DOCUMENT_COUNT - 1
    assert after["data"]["chunks"] < before["data"]["chunks"]


# --- scope ------------------------------------------------------------------------------------


async def test_every_search_reports_the_scope_it_ran_under() -> None:
    """Supported searches and the control alike: the envelope repeats the collection.

    Repeated rather than remembered, because there is no session between calls — so a scope
    that was dropped and a scope that was applied are indistinguishable without this field.
    """
    qualified = await _qualification()
    for finding in (*qualified.supported, qualified.control):
        assert finding.scope == (COLLECTION,), finding.query


async def test_an_unknown_collection_refuses_and_never_reaches_a_retriever() -> None:
    """The security property, watched at the retriever rather than inferred from the message.

    A refusal that arrived *after* an unscoped search had run would carry the same error type
    and read identically in a log. What separates the two is whether a query reached the
    retriever at all, so that is what is asserted — the fixture's retriever records every query
    it is handed, and the assertion is that it was handed none.
    """
    service, backend = await build_fixture()
    async with Client(build_server(service)) as client:
        envelope = (
            await client.call_tool(
                "search", {"query": "admission control", "collections": ["Engineering Archives"]}
            )
        ).structured_content or {}

    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "UnknownEntityError"
    assert envelope["data"] is None
    assert backend.retriever_.seen == [], (
        "the refusal happened after a search ran, which is the failure this checks for: the "
        "caller gets an error and the workspace has already been searched unscoped"
    )


async def test_a_scoped_search_sees_only_the_scoped_collection() -> None:
    """The control for the test above: scoping is a restriction, not a no-op.

    A retriever that ignored ``collection_ids`` would satisfy every assertion about scope being
    reported while the scope did nothing at all, so the filter is checked where it lands.
    """
    service, backend = await build_fixture()
    collection_id = next(iter(backend.organization_.collections))
    async with Client(build_server(service)) as client:
        await client.call_tool(
            "search", {"query": "admission control", "collections": [COLLECTION], "limit": 3}
        )
    assert [query.filter.collection_ids for query in backend.retriever_.seen] == [
        frozenset({collection_id})
    ]


# --- evidence and provenance -------------------------------------------------------------------


async def test_every_supported_passage_can_be_cited_down_to_the_heading() -> None:
    """Title, canonical URI, heading path and the structured record, on every one.

    Claim-level provenance is the difference between "the handbook says so" and a link a reader
    can open at the paragraph. The fixture gives every document a source record, so anything
    short of complete here is the surface losing it rather than the corpus lacking it.
    """
    qualified = await _qualification()
    assert qualified.evidence
    assert qualified.citable == qualified.evidence
    for passage in qualified.evidence:
        assert passage.canonical_uri is not None
        assert passage.canonical_uri.startswith("https://docs.example.test/architecture/")
        assert len(passage.heading_path) == 2


async def test_the_supported_question_is_answered_from_more_than_one_document() -> None:
    """Three angles, and they agree — which is what makes it evidence rather than one sentence."""
    qualified = await _qualification()
    documents = {passage.document_id for passage in qualified.evidence}
    assert len(documents) == DOCUMENT_COUNT
    for finding in qualified.supported:
        assert finding.passages, finding.query


async def test_repeated_passages_are_paid_for_once() -> None:
    """Deduplication happens before synthesis, and it actually removes something here.

    Asserted with a strict inequality on purpose: three overlapping searches that produced no
    duplicate at all would make the deduplication step untested, and this test would still pass
    against a version of :func:`~tests.mcp.qualification.deduplicate` that returned its input.
    """
    qualified = await _qualification()
    returned = sum(len(finding.passages) for finding in qualified.supported)
    assert len(qualified.evidence) < returned
    assert len(deduplicate(qualified.supported)) == len(qualified.evidence)


# --- the control ------------------------------------------------------------------------------


async def test_the_absent_topic_produces_a_refusal_rather_than_a_claim() -> None:
    """The control search comes back holding something, and it supports nothing.

    That is the whole design of the control: the corpus uses the word "ownership", so the
    nearest lexical match is returned and is plausible-looking. What makes it not evidence is
    that it mentions nothing the topic is about, which is a fact the recipe records rather than
    a judgment it makes.
    """
    qualified = await _qualification()
    assert qualified.control.passages, "the control returned nothing, so it tests nothing"
    assert qualified.control_support == ()
    joined = " ".join(passage.text.lower() for passage in qualified.control.passages)
    for term in ABSENT_TOPIC_TERMS:
        assert term not in joined


async def test_the_control_is_weakly_ranked_and_that_is_not_treated_as_evidence() -> None:
    """Low confidence beside a returned passage, and neither one is support.

    Both halves matter. A control that scored `none` would be refused by any client; the case
    that gets answered wrongly is a low-confidence hit that shares a word with the question,
    which is exactly what this returns.
    """
    qualified = await _qualification()
    assert qualified.control.confidence_band in {"low", "none"}
    assert all(finding.confidence_band in {"high", "medium"} for finding in qualified.supported), (
        "the supported searches have to outrank the control, or the comparison says nothing"
    )


async def test_the_refusal_says_what_was_searched_and_how_much_was_not_read() -> None:
    """The sentence a client can send, assembled from facts and from no model.

    It has to carry the scope, the size of the corpus and the sample it took, because "we found
    nothing" without those three reads as "there is nothing" — which is the claim the corpus
    cannot support and the reason this control exists.
    """
    qualified = await _qualification()
    refusal = qualified.refusal()
    assert COLLECTION in refusal
    assert ABSENT_TOPIC in refusal
    assert str(DOCUMENT_COUNT) in refusal
    assert str(CHUNK_COUNT) in refusal
    assert "proof of absence" in refusal


async def test_the_searches_saw_less_than_the_collection_holds() -> None:
    """Top-`limit` retrieval is a sample, and the numbers say by how much.

    Without this the recipe could report "nothing found" from a search that happened to have
    covered the whole corpus, and the caveat about non-exhaustiveness would be true in general
    and vacuous here.
    """
    qualified = await _qualification()
    assert qualified.accounting.deduplicated_passages < qualified.chunks


# --- the budget --------------------------------------------------------------------------------


async def test_the_recipe_stays_inside_the_budget_it_declares() -> None:
    """Searches, passages, bytes and tokens, each against a ceiling written down beforehand."""
    budget = Budget()
    qualified = await _qualification()
    accounting = qualified.accounting

    assert accounting.searches == len(EVIDENCE_QUERIES) + 1
    assert accounting.searches <= budget.searches
    assert accounting.requested_passages == accounting.searches * budget.limit
    assert accounting.returned_passages <= accounting.requested_passages
    assert accounting.deduplicated_passages <= accounting.returned_passages
    assert accounting.payload_bytes <= budget.payload_bytes
    assert accounting.within(budget)


async def test_the_token_estimate_is_reported_or_its_absence_is() -> None:
    """A number when this machine can count, ``None`` when it cannot — never a quiet zero.

    ``ContextTokenCounter`` refuses rather than downloading a BPE vocabulary, so a host without
    one is a host where this cannot be measured. Reporting that as absent keeps an unmeasured
    run visibly unmeasured; reporting it as zero would put an unmeasured run inside every
    budget.
    """
    qualified = await _qualification()
    tokens = qualified.accounting.generator_tokens
    if tokens is None:
        return
    assert tokens > 0
    assert tokens <= Budget().generator_tokens


async def test_a_smaller_limit_costs_fewer_bytes() -> None:
    """The budget is a dial rather than a description, so it has to move something.

    A ``limit`` the recipe accepted and did not pass through would leave every assertion above
    passing while the bound did nothing — the same class of defect as an annotation that is
    computed and never published.
    """
    service, _ = await build_fixture()
    wide = await qualify(service, budget=Budget(limit=5))
    service, _ = await build_fixture()
    narrow = await qualify(service, budget=Budget(limit=2))

    assert narrow.accounting.requested_passages < wide.accounting.requested_passages
    assert narrow.accounting.payload_bytes < wide.accounting.payload_bytes


async def test_the_recipe_asks_no_model_anything() -> None:
    """No generator is reached, so nothing here can be a summarizer in disguise.

    The fake answerer records every request it is given. An empty log is the assertion: the
    recipe assembles evidence and hands it on, and the decision about what the evidence means is
    made outside it, where it can be seen.
    """
    service, backend = await build_fixture()
    await qualify(service)
    assert backend.answerer_.calls == []
