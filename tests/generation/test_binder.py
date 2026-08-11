"""What happens to a model's citations before a reader sees them.

**A suite in which every citation is valid certifies nothing.** It passes identically against
a verifier that returns ``True`` unconditionally. So the cases that matter here are the ones
where the model gets it wrong: a slot that names no passage, a marker the model mangled, an
anchor that no longer resolves — and the one nothing catches, stated as a test so that the
guarantee stays the narrow one this system can actually keep.
"""

from __future__ import annotations

import re
import time

import pytest

from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import Document
from manicule.core.retrieval import Candidate
from manicule.generation import verification
from manicule.generation.answers import DropReason, EventKind, Verification
from manicule.generation.binder import CitationBinder
from manicule.generation.markers import ATTEMPT_PREFIX
from manicule.generation.verification import (
    AnchorResolver,
    CitationVerifier,
    UnverifiableSource,
)
from tests.generation.fakes import (
    BrokenResolver,
    FakeParser,
    SlowResolver,
    candidate,
    context,
    document,
    resolver,
)

ROLLBACK = "Roll back with `deploy --rollback`."
CANARY = "Canary deploys ship to one host first."


def two_passages() -> tuple[Candidate, ...]:
    return (
        candidate(
            chunk_id="c1", document_id="doc-1", text=ROLLBACK, heading_path=("Ops", "Rollback")
        ),
        candidate(chunk_id="c2", document_id="doc-2", text=CANARY, heading_path=("Ops", "Canary")),
    )


def two_documents() -> dict[str, Document]:
    return {"doc-1": document(document_id="doc-1"), "doc-2": document(document_id="doc-2")}


def honest_parser() -> FakeParser:
    """A parser whose anchors resolve to exactly what the chunks claim."""
    return FakeParser(
        resolutions={"Rollback": f"## Rollback\n{ROLLBACK}", "Canary": f"## Canary\n{CANARY}"}
    )


async def bind(
    text: str,
    *,
    parser: FakeParser | None = None,
    resolver_override: AnchorResolver | None = None,
    timeout_s: float = 5.0,
    passages: tuple[Candidate, ...] | None = None,
    documents: dict[str, Document] | None = None,
    chunks: int = 1,
) -> CitationBinder:
    """Run one model output through a binder over the two-passage fixture."""
    verifier = CitationVerifier(
        resolver_override or resolver(parser or honest_parser()), timeout_s=timeout_s
    )
    assembled = context(passages if passages is not None else two_passages())
    run = verifier.start(assembled, documents or two_documents(), started_at=time.monotonic())
    binder = CitationBinder(run=run)
    for piece in _split(text, chunks):
        await binder.feed(piece)
    await binder.finish()
    await run.aclose()
    return binder


def _split(text: str, chunks: int) -> list[str]:
    """Cut ``text`` into ``chunks`` roughly equal pieces.

    Feeding in one piece is the case that never happens in production: a provider splits its
    output wherever the network did, and a marker straddling that boundary is ordinary.
    """
    if chunks <= 1:
        return [text]
    size = max(len(text) // chunks, 1)
    return [text[at : at + size] for at in range(0, len(text), size)] or [text]


# --- the guarantee, and the three ways a model breaks it ---------------------------------


async def test_a_good_citation_survives_and_is_built_entirely_from_the_context() -> None:
    """Not one field of a citation comes from the model; it contributes a small integer."""
    binder = await bind(f"Use the runbook.{ATTEMPT_PREFIX}:1]]")

    assert len(binder.citations) == 1
    citation = binder.citations[0]
    assert citation.slot == 1
    assert citation.quote == ROLLBACK
    assert citation.title == "Deploy runbook"
    assert citation.uri == "https://example.invalid/doc-1"
    assert citation.chunk_id == "c1"
    assert citation.verification is Verification.RESOLVED
    assert binder.text == f"Use the runbook.{ATTEMPT_PREFIX}:1]]"


async def test_a_citation_to_a_passage_that_was_never_retrieved_is_dropped() -> None:
    """Invention. Two passages were offered and the model named a ninth.

    Nothing about slot 9 exists to verify, so it fails at level 0 — free, and the reason the
    model cannot invent a page number is that it never writes one.
    """
    binder = await bind(f"Deploys are safe.{ATTEMPT_PREFIX}:9]] Really.")

    assert binder.citations == ()
    assert [drop.reason for drop in binder.drops] == [DropReason.OUT_OF_RANGE]
    assert binder.drops[0].slot == 9
    assert binder.text == "Deploys are safe. Really.", "the sentence must stand exactly as written"


async def test_a_mangled_marker_is_deleted_and_counted_as_an_attempt() -> None:
    """The ``cite:`` prefix exists so a malformed attempt is not mistaken for prose."""
    binder = await bind(f"Roll back first.{ATTEMPT_PREFIX}:one]] Then verify.")

    assert binder.citations == ()
    assert [drop.reason for drop in binder.drops] == [DropReason.MALFORMED_MARKER]
    assert binder.drops[0].slot is None
    assert binder.accounting.markers_seen == 1, "an attempt is counted even when it is malformed"
    assert binder.text == "Roll back first. Then verify."


async def test_verification_does_not_catch_misattribution_and_this_is_the_honest_limit() -> None:
    """The model cites a real, resolvable passage for a claim it does not support.

    Every level of the ladder passes: slot 2 is in range, its anchor is located, and it
    resolves to exactly the text the chunk claims. The citation is *resolvable* and *wrong*.

    This is not a gap to close with a cleverer check — detecting it means deciding whether a
    passage entails a sentence — and pinning it here is what keeps the guarantee narrow:
    every citation resolves to a real location whose text is what the model was given. It is
    **not** a claim that the passage supports the sentence.
    """
    binder = await bind(f"Roll back with the canary flag.{ATTEMPT_PREFIX}:2]]")

    assert len(binder.citations) == 1
    assert binder.citations[0].quote == CANARY
    assert binder.citations[0].verification is Verification.RESOLVED
    assert binder.drops == ()


# --- the ladder --------------------------------------------------------------------------


async def test_an_anchor_that_has_drifted_from_the_document_is_dropped() -> None:
    """Level 2. The chunk claims text the retained bytes no longer carry at that anchor."""
    drifted = FakeParser(
        resolutions={"Rollback": "## Rollback\nThis section was rewritten.", "Canary": CANARY}
    )

    binder = await bind(f"Roll back.{ATTEMPT_PREFIX}:1]]", parser=drifted)

    assert binder.citations == ()
    assert binder.drops[0].reason is DropReason.UNRESOLVABLE
    assert binder.drops[0].reached is Verification.LOCATED


async def test_a_passage_the_parser_could_not_place_is_dropped_at_level_one() -> None:
    """A citation pointing nowhere by the parser's own admission is not shown."""
    passages = (
        candidate(
            chunk_id="c1", text=ROLLBACK, anchor=Unlocated(reason="scanned page, no text layer")
        ),
    )

    binder = await bind(
        f"Roll back.{ATTEMPT_PREFIX}:1]]", passages=passages, documents={"doc-1": document()}
    )

    assert binder.citations == ()
    assert binder.drops[0].reason is DropReason.UNLOCATED
    assert "scanned page" in binder.drops[0].detail


async def test_missing_source_bytes_drop_the_citation_rather_than_downgrade_it() -> None:
    """A citation into a document whose bytes are gone cannot be shown with a highlight.

    The reader is not told otherwise: the level is a property of the *configuration*, so a
    single lost blob is a failure rather than a quiet degradation to a weaker claim.
    """
    passages = (candidate(chunk_id="c1", text=ROLLBACK),)
    documents = {"doc-1": document(original_ref=None)}

    binder = await bind(f"Roll back.{ATTEMPT_PREFIX}:1]]", passages=passages, documents=documents)

    assert binder.citations == ()
    assert binder.drops[0].reason is DropReason.UNRESOLVABLE
    assert "retained bytes" in binder.drops[0].detail


async def test_without_retained_bytes_the_ceiling_drops_and_citations_still_work() -> None:
    """Level 2 becomes *impossible* rather than failing, so level 1 is the strongest available.

    The answer names the level it reached rather than reporting the same word for two
    different amounts of checking.
    """
    verifier = CitationVerifier(UnverifiableSource("retention is off"))
    assembled = context(two_passages())
    run = verifier.start(assembled, two_documents())
    binder = CitationBinder(run=run)
    await binder.feed(f"Roll back.{ATTEMPT_PREFIX}:1]]")
    await run.aclose()

    assert verifier.ceiling is Verification.LOCATED
    assert len(binder.citations) == 1
    assert binder.citations[0].verification is Verification.LOCATED


async def test_a_citation_into_a_document_that_left_the_index_is_dropped() -> None:
    """A citation that cannot name the document it points into is not a citation."""
    binder = await bind(
        f"Roll back.{ATTEMPT_PREFIX}:1]]", documents={"doc-2": document(document_id="doc-2")}
    )

    assert binder.citations == ()
    assert "no longer in the index" in binder.drops[0].detail


async def test_a_parser_that_raises_drops_citations_and_never_the_answer() -> None:
    """One broken parser must not end an answer, in the same way one bad document never ends
    a batch."""
    binder = await bind(
        f"Roll back.{ATTEMPT_PREFIX}:1]] Then verify.", resolver_override=BrokenResolver()
    )

    assert binder.citations == ()
    assert binder.drops[0].reason is DropReason.UNRESOLVABLE
    assert "RuntimeError" in binder.drops[0].detail
    assert binder.text == "Roll back. Then verify."


async def test_verification_that_does_not_finish_in_time_is_a_drop_with_its_own_reason() -> None:
    """A slow disk and a wrong citation want different remedies, so they get different reasons.

    Dropping a possibly-good citation is the uncomfortable half of this. Sending an unverified
    one, under a design whose entire claim is verification, is the unacceptable half.
    """
    binder = await bind(
        f"Roll back.{ATTEMPT_PREFIX}:1]]", resolver_override=SlowResolver(), timeout_s=0.05
    )

    assert binder.citations == ()
    assert binder.drops[0].reason is DropReason.VERIFICATION_TIMEOUT


# --- the binder's edits, and the ones it refuses to make ---------------------------------


async def test_a_partly_verifying_marker_keeps_only_the_slots_that_survived() -> None:
    """``[[cite:1,9]]`` where only 1 verifies becomes ``[[cite:1]]``. Deletion, never repair."""
    binder = await bind(f"Both apply.{ATTEMPT_PREFIX}:1,9]]")

    assert [citation.slot for citation in binder.citations] == [1]
    assert binder.text == f"Both apply.{ATTEMPT_PREFIX}:1]]"


async def test_the_answer_is_never_edited_except_by_deleting_markers() -> None:
    """The one property the whole design rests on, checked as a property.

    Strip every marker from what the model wrote and from what the reader gets, and the two
    must be identical. Any trimming, rewriting or reflowing shows up here.
    """
    written = (
        f"Roll back first.{ATTEMPT_PREFIX}:1]]\n\n"
        f"- Canary deploys ship to one host.{ATTEMPT_PREFIX}:2]]\n"
        f"- This claim is unsupported.{ATTEMPT_PREFIX}:9]]\n\n"
        "```python\nitems[0] = argv[1]\n```\n"
    )
    binder = await bind(written)

    marker = re.compile(re.escape(ATTEMPT_PREFIX) + r"[^\]]*\]\]")
    assert marker.sub("", binder.text) == marker.sub("", written)
    assert "```python\nitems[0] = argv[1]\n```" in binder.text, "code must survive untouched"


async def test_all_citations_dropped_with_a_non_empty_context_is_flagged_ungrounded() -> None:
    """A single failed marker is a model slip. All of them failing is a different event."""
    binder = await bind(f"One.{ATTEMPT_PREFIX}:8]] Two.{ATTEMPT_PREFIX}:9]]")

    assert binder.ungrounded is True


async def test_an_answer_with_no_markers_is_recorded_and_not_judged() -> None:
    """ "The sources do not cover this" is a correct answer, and nothing can distinguish it
    from a model that forgot."""
    binder = await bind("The indexed documents do not cover this.")

    assert binder.ungrounded is False
    assert binder.accounting.offered_no_citations is True


async def test_one_slot_cited_many_times_is_one_citation_and_one_drop() -> None:
    """A verdict is a property of the slot, so repeating a marker does not multiply it."""
    good = await bind(f"A{ATTEMPT_PREFIX}:1]] B{ATTEMPT_PREFIX}:1]] C{ATTEMPT_PREFIX}:1]]")
    bad = await bind(f"A{ATTEMPT_PREFIX}:9]] B{ATTEMPT_PREFIX}:9]]")

    assert len(good.citations) == 1
    assert good.accounting.markers_seen == 3
    assert len(bad.drops) == 1


async def test_events_arrive_in_stream_order_so_a_citation_lands_where_its_marker_was() -> None:
    verifier = CitationVerifier(resolver(honest_parser()))
    assembled = context(two_passages())
    run = verifier.start(assembled, two_documents())
    binder = CitationBinder(run=run)

    kinds = [event.kind for event in await binder.feed(f"First.{ATTEMPT_PREFIX}:1]] Second.")]
    kinds.extend(event.kind for event in await binder.finish())
    await run.aclose()

    assert kinds == [EventKind.DELTA, EventKind.CITATION, EventKind.DELTA, EventKind.DELTA]


# --- the guard is load-bearing -----------------------------------------------------------


async def test_disabling_the_containment_predicate_would_let_a_drifted_anchor_through() -> None:
    """Proof that level 2 is doing work rather than passing everything.

    The drift case above is only meaningful if the predicate is what rejects it. Replacing it
    with one that always agrees must make that citation survive — and if this test ever
    passes *without* the patch, the predicate has stopped being consulted.
    """

    drifted = FakeParser(resolutions={"Rollback": "## Rollback\nThis section was rewritten."})

    def always_agrees(resolved: str | None, claimed: str) -> bool:
        del resolved, claimed
        return True

    original = verification.contains_claimed_text
    verification.contains_claimed_text = always_agrees
    try:
        binder = await bind(f"Roll back.{ATTEMPT_PREFIX}:1]]", parser=drifted)
    finally:
        verification.contains_claimed_text = original

    assert len(binder.citations) == 1, (
        "with the predicate disabled the drifted anchor survives, which is what makes the "
        "test above an assertion rather than a description"
    )


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (HeadingAnchor(path=("Ops", "Rollback")), Verification.RESOLVED),
        (Unlocated(reason="no text layer"), Verification.BOUND),
    ],
)
async def test_the_level_reached_travels_with_each_verdict(
    anchor: HeadingAnchor | Unlocated, expected: Verification
) -> None:
    verifier = CitationVerifier(resolver(honest_parser()))
    passages = (candidate(chunk_id="c1", text=ROLLBACK, anchor=anchor),)
    assembled = context(passages)
    run = verifier.start(assembled, {"doc-1": document()})
    verdict = await run.verdict(1)
    await run.aclose()

    assert verdict.reached is expected
