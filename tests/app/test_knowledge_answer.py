"""An answer's citation carries the evidence of the memory it cites.

``tests/app/test_service.py`` holds the same test for source metadata, and its docstring says why
this one exists: **an untested field on a contract is one the next refactor removes for free.**
``search`` is covered over every surface in ``test_surface_parity.py``, but the answer path builds
its citations in a different place from the hit path, from a different object, and the fake
answerer emits an envelope with **no citations at all** unless a test hands it one — so
``AnswerCitation.knowledge`` is never constructed by any test that does not do exactly this.

``ask`` is also the output a reader is most likely to act on, which is why the evidence has to be
on it: an answer that says "run the runbook" and cites a memory three years stale is the case this
whole format exists for.
"""

from __future__ import annotations

from manicule.app import service as service_module
from manicule.app.service import ApplicationService
from manicule.core.anchors import HeadingAnchor
from manicule.core.retrieval import Candidate
from manicule.generation.answers import AnswerEnvelope, Citation, Verification
from manicule.ingest.knowledge import read
from tests.app.fakes import FakeBackend, make_chunk, make_document
from tests.knowledge_samples import EVIDENCE, LOCATOR, SECOND_SOURCE, memory


def _backend_citing(text: str) -> tuple[FakeBackend, ApplicationService]:
    """A backend whose answerer cites one passage of one memory, read as ingest reads it."""
    backend = FakeBackend()
    document = make_document(
        backend.workspace,
        source_id="reference_vpn_gateway_failover.md",
        title="VPN gateway failover is manual",
        knowledge=read(text),
    )
    chunk = make_chunk(document, text="Run the failover runbook.")
    backend.store.add(document)
    backend.store.chunks[document.id] = [chunk]
    backend.retriever_.candidates = [Candidate(chunk=chunk, score=0.9)]
    backend.answerer_.envelope = AnswerEnvelope(
        text="Run the failover runbook.",
        corpus_consulted=True,
        confidence=0.5,
        citations=(
            Citation(
                slot=1,
                document_id=document.id,
                chunk_id=chunk.id,
                uri=document.uri,
                title=document.title,
                heading_path=chunk.heading_path,
                anchor=HeadingAnchor(path=chunk.heading_path),
                quote=chunk.text,
                verification=Verification.RESOLVED,
            ),
        ),
    )
    return backend, ApplicationService(backend)


async def test_an_answer_citation_carries_the_memorys_evidence() -> None:
    _, service = _backend_citing(memory(evidence=EVIDENCE + SECOND_SOURCE))

    answered = await service.ask("how do I fail over the gateway?")

    assert len(answered.citations) == 1, "the fixture must produce a citation to assert about"
    cited = answered.citations[0]
    assert cited.knowledge is not None, (
        "an answer citation reports no evidence, so a memory cited in the one output a reader "
        "is most likely to act on looks like folklore"
    )
    assert cited.knowledge.knowledge_type == "constraint"
    assert cited.knowledge.evidence_status == "corroborated"
    assert cited.knowledge.evidence[0].locator == LOCATOR
    assert cited.knowledge.newest_said_at == "2026-04-01T09:00:00+02:00"
    assert cited.provenance is None, "evidence is not provenance, on this path as on a hit"


async def test_a_refused_record_reaches_the_citation_rather_than_vanishing() -> None:
    _, service = _backend_citing(memory(knowledge_type="recipe"))

    answered = await service.ask("how do I fail over the gateway?")

    cited = answered.citations[0]
    assert cited.knowledge is not None
    assert cited.knowledge.evidence == ()
    assert "knowledge_type" in cited.knowledge.unavailable_reason


async def test_a_cited_document_that_is_not_a_memory_reports_no_knowledge() -> None:
    _, service = _backend_citing("# Just a page\n\nNo front matter.\n")

    answered = await service.ask("how do I fail over the gateway?")

    assert answered.citations[0].knowledge is None


def test_a_stored_citation_replays_without_live_knowledge() -> None:
    """A stored citation records what was shown; this block reports what is true now.

    Attaching a live record to a historical one would let a replayed conversation claim it had
    cited evidence that was added after the answer was given, which is the reason ``provenance``
    is ``None`` on a replay too. Driven through the real replay function, which builds a citation
    from storage and has no document to read a record from.
    """
    stored = Citation(
        slot=1,
        document_id="d",
        chunk_id="c",
        uri="file:///x",
        title="x",
        heading_path=("Title",),
        anchor=HeadingAnchor(path=("Title",)),
        quote="q",
        verification=Verification.RESOLVED,
    )

    replayed = service_module._citation(stored)

    assert replayed.knowledge is None
    assert replayed.provenance is None
