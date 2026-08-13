"""What a full ingest records about the detector that read the document.

The pipeline is where the fingerprint is *stamped*, and three of its cases are easy to write
and wrong: a document that produces no entries, a document ingested while detection is switched
off, and a document whose detector raised. All three would work if lineage were recorded beside
the rows, and all three are the states in which there are no rows to record it beside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from manicule.core.content import DocumentStatus
from manicule.core.fingerprints import DETECTION_DISABLED, GlossaryFingerprint
from manicule.ingest.glossary_lineage import glossary_fingerprint
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

if TYPE_CHECKING:
    from manicule.core.content import Document
    from manicule.ingest.pipeline import IngestPipeline, RunReport

EXPANSION = "Network Operations Workspace"
GLOSSARY = f"NOW - {EXPANSION}"
PROSE = "The scheduler restarts nightly, which is fine."


async def ingest(
    pipeline: IngestPipeline, store: fakes.MemoryGlossaryStore, source_id: str, text: str
) -> tuple[Document, RunReport]:
    """Run one document through, and hand back the row and the report it produced."""
    report = await pipeline.run(fakes.DictConnector({source_id: text}))
    document = await store.find_document("memory", source_id)
    assert document is not None
    return document, report


async def test_a_successful_ingest_records_the_entries_and_what_produced_them() -> None:
    """Requirement 6, at the one place a full ingest can satisfy it."""
    store = fakes.MemoryGlossaryStore()
    pipeline, _, _ = build(store=store)

    document, _ = await ingest(pipeline, store, "glossary", GLOSSARY)

    assert document.status is DocumentStatus.INDEXED
    assert [entry.acronym for entry in store.glossary[document.id]] == ["NOW"]
    assert store.glossary_lineage_by_id[document.id] == glossary_fingerprint().canonical()


async def test_a_document_stating_nothing_still_records_the_detector_that_read_it() -> None:
    """Requirement 2 at ingest, which is where the empty case first arises.

    Ordinary prose is the overwhelming majority of any corpus, so an implementation that only
    stamped documents with entries would leave almost everything permanently unstamped — and
    every sweep would then select almost everything, for ever, having nothing to change.
    """
    store = fakes.MemoryGlossaryStore()
    pipeline, _, _ = build(store=store)

    document, _ = await ingest(pipeline, store, "runbook", PROSE)

    assert store.glossary[document.id] == []
    assert store.glossary_lineage_by_id[document.id] == glossary_fingerprint().canonical()


async def test_detection_switched_off_records_the_disabled_state_and_keeps_the_rows() -> None:
    """Requirement 8 at ingest, and both halves of it matter.

    ``rag.glossary.detect_on_ingest`` is documented as the switch an operator throws while
    investigating a detector that is producing rubbish: existing entries stay queryable and no
    new ones are written. So the rows are left exactly as they are — clearing them would be the
    silent erasure requirement 7 forbids, arrived at through a setting rather than a crash.

    What is recorded is that no detector ran. Turning detection back on changes the installed
    fingerprint, so every document stamped this way is selected by the next survey.
    """
    store = fakes.MemoryGlossaryStore()
    pipeline, _, _ = build(store=store)
    document, _ = await ingest(pipeline, store, "glossary", GLOSSARY)
    survivor = list(store.glossary[document.id])
    assert survivor, "the fixture must have something to leave alone"

    off, _, _ = build(store=store, detect_glossary=False)
    again, _ = await ingest(off, store, "glossary", GLOSSARY + "\nand a second line")

    assert store.glossary[again.id] == survivor, (
        "turning detection off must not empty a working glossary"
    )
    recorded = GlossaryFingerprint.model_validate_json(store.glossary_lineage_by_id[again.id])
    assert recorded.detector == DETECTION_DISABLED
    assert not recorded.detects


async def test_a_detector_failure_costs_the_glossary_and_nothing_else() -> None:
    """Requirement 7 at ingest: fail closed, keep the index, and say so.

    ``detect_entries`` is regular expressions over lines with no model to be unavailable, so it
    raising means a bug in this repository. Three things follow and none of them is automatic:
    the document is still indexed with its chunks and its vectors, because a glossary bug must
    not cost a working index; its previous entries and its stale lineage are untouched, so it is
    still selected by the repair and still reported by ``doctor``; and the run says which
    document it was, because the alternative is a detector that has stopped working and a screen
    of green counters.
    """
    store = fakes.MemoryGlossaryStore()
    pipeline, _, vectors = build(store=store)
    first, _ = await ingest(pipeline, store, "glossary", GLOSSARY)
    survivor = list(store.glossary[first.id])
    stale = store.glossary_lineage_by_id[first.id]

    def explode(*_: object, **__: object) -> list[object]:
        msg = "catastrophic backtracking in a definition pattern"
        raise RuntimeError(msg)

    with pytest.MonkeyPatch.context() as patched:
        # Patched where the pipeline *bound* the name, not where it is defined: the module-level
        # `from ... import detect_entries` took a reference at import time, and patching the
        # detector module would leave the pipeline calling the original — a test that passed by
        # never reproducing the failure it is named for.
        from manicule.ingest import pipeline as under_test  # noqa: PLC0415

        patched.setattr(under_test, "detect_entries", explode)
        document, report = await ingest(
            pipeline, store, "glossary", GLOSSARY + "\nHDR - Hot Draining Router"
        )

    assert document.status is DocumentStatus.INDEXED, "a detector bug must not fail a document"
    assert store.chunks[document.id], "its chunks were written"
    assert vectors.rows, "its vectors were written"
    assert document.status_detail is None, (
        "a detector failure must not read as a parse failure: re_parse treats an indexed "
        "outcome carrying a detail as a document that was not rebuilt"
    )
    assert report.glossary_failures, "the run has to say which document lost its glossary"
    assert "glossary detection failed" in report.glossary_failures[0]
    assert store.glossary[document.id] == survivor, "a working glossary was erased"
    assert store.glossary_lineage_by_id[document.id] == stale, (
        "the fingerprint advanced past a detector run that did not happen"
    )


async def test_a_detector_failure_reaches_the_run_report() -> None:
    """The other half of "not silently". A stale column found next week is not a report."""
    from manicule.ingest.pipeline import DocumentOutcome, RunReport  # noqa: PLC0415

    report = RunReport(connector="local")
    report.record(
        DocumentOutcome(
            source_id="glossary",
            status=DocumentStatus.INDEXED,
            document_id="d1",
            glossary_detail="glossary detection failed: RuntimeError: boom",
        )
    )

    assert report.glossary_failures == ["d1: glossary detection failed: RuntimeError: boom"]
    recorded = cast("dict[str, Any]", report.as_metadata()["last_run"])
    assert recorded["glossary_failures"] == report.glossary_failures


async def test_a_document_with_no_extractable_text_records_lineage_too() -> None:
    """The other empty path, and it is a derived empty rather than an absent one.

    A document whose blocks produced no chunk states no definitions, and that determination is
    the output of a detector reading nothing. Left unstamped it would be selected by every
    sweep for ever, having nothing to do.
    """
    store = fakes.MemoryGlossaryStore()
    pipeline, _, _ = build(store=store, parsers={"lines": fakes.EmptyParser()})

    document, _ = await ingest(pipeline, store, "empty", "nothing survives this parser")

    assert document.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert store.glossary[document.id] == []
    assert store.glossary_lineage_by_id[document.id] == glossary_fingerprint().canonical()


async def test_a_detector_change_does_not_make_change_detection_re_parse() -> None:
    """The coupling the specification forbids, checked in the direction nobody thinks about.

    A detector fix must reach an existing corpus — that is what the repair is for — and it must
    *not* do so by making every document look like it needs re-parsing. Re-parsing is a rung
    that reads retained bytes and re-embeds whatever moves, and charging a corpus for it because
    a regular expression changed is the coupling the parse-lineage half of this was built to
    avoid.

    Held structurally: ``Document`` deliberately does not carry ``glossary_fp``, so change
    detection has no way to consult it even by accident.
    """
    from manicule.core.content import Document  # noqa: PLC0415

    assert "glossary_fp" not in Document.model_fields, (
        "putting glossary lineage on the domain document is how a detector change becomes a "
        "corpus-wide re-parse; it belongs on the row, like chunk_fp and embed_fp"
    )
