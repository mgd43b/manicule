"""What a memory's evidence looks like in a terminal.

Rendering only: the payload is built by hand here because the service's part — which documents get
one, and what is in it — is asserted over every surface in ``test_surface_parity.py``. What this
file adds is what the command line does with it, which is where two of the failures live: a quote
is somebody else's message, so brackets in it must not be read as markup, and the word for how well
the sources hold up must not be the word a citation's quotation check already uses.
"""

from __future__ import annotations

import pytest

from manicule.app import results as r
from manicule.cli import render

LOCATOR = "https://example.slack.com/archives/C0123/p1700000000000100"


def _knowledge(*, quote: str = "failover is manual", **overrides: object) -> r.KnowledgeReference:
    fields: dict[str, object] = {
        "knowledge_type": "constraint",
        "evidence_status": "single_source",
        "evidence": (
            r.EvidenceReference(
                kind="slack",
                locator=LOCATOR,
                said_at="2026-03-02T14:05:00+00:00",
                quote=quote,
                author="Jane Doe",
            ),
        ),
        "oldest_said_at": "2026-03-02T14:05:00+00:00",
        "newest_said_at": "2026-03-02T14:05:00+00:00",
        "relationships": (r.RelationshipReference(relation="supersedes", target="reference_old"),),
    }
    return r.KnowledgeReference(**{**fields, **overrides})  # type: ignore[arg-type]


def _search(capsys: pytest.CaptureFixture[str], knowledge: r.KnowledgeReference | None) -> str:
    render.render_search(
        render.console(),
        r.SearchResult(
            query="failover",
            profile="balanced",
            count=1,
            hits=(
                r.SearchHit(
                    document_id="d",
                    chunk_id="c",
                    uri="file:///corpus/m.md",
                    title="VPN gateway failover",
                    score=0.5,
                    text="Run the failover runbook.",
                    knowledge=knowledge,
                ),
            ),
            confidence=0.5,
            confidence_band="medium",
            confidence_reason="",
        ),
    )
    return " ".join(capsys.readouterr().out.split())


def test_a_memorys_evidence_is_printed_under_the_passage_that_matched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _search(capsys, _knowledge())

    assert "constraint · single_source · 1 source(s) attested, newest said 2026-03-02" in out
    assert "slack: Jane Doe, 2026-03-02 “failover is manual”" in out
    assert LOCATOR in out
    assert "supersedes reference_old" in out


def test_the_word_is_attested_and_never_verified(capsys: pytest.CaptureFixture[str]) -> None:
    """``verified`` is what an answer's citation column says of a quotation that resolved.

    Manicule cannot open a Slack permalink; borrowing the word for sources it did not check would
    let a reader take one for the other.
    """
    out = _search(capsys, _knowledge())

    assert "attested" in out
    assert "verified" not in out


def test_brackets_in_a_quote_are_text_not_markup(capsys: pytest.CaptureFixture[str]) -> None:
    """A message that happens to contain ``[red]`` is a message, not a style."""
    out = _search(capsys, _knowledge(quote="use [red]flags[/red] and [bold] carefully"))

    assert "use [red]flags[/red] and [bold] carefully" in out


def test_a_refused_record_is_printed_not_swallowed(capsys: pytest.CaptureFixture[str]) -> None:
    out = _search(capsys, r.KnowledgeReference(unavailable_reason="evidence[0].said_at: bad"))

    assert "memory record refused: evidence[0].said_at: bad" in out
    assert "attested" not in out


def test_an_ordinary_hit_prints_nothing_extra(capsys: pytest.CaptureFixture[str]) -> None:
    out = _search(capsys, None)

    assert "Run the failover runbook." in out
    assert "attested" not in out
    assert "memory record" not in out


def test_an_answers_cited_memory_shows_its_evidence_beside_the_citation_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    render.render_answer(
        render.console(),
        r.AnswerResultPayload(
            question="how do I fail over?",
            text="Run the runbook [1].",
            citations=(
                r.AnswerCitation(
                    slot=1,
                    document_id="d",
                    chunk_id="c",
                    uri="file:///corpus/m.md",
                    title="VPN gateway failover",
                    verification="resolved",
                    knowledge=_knowledge(),
                ),
            ),
        ),
    )
    out = " ".join(capsys.readouterr().out.split())

    assert "resolved" in out, "the citation's own verification is still the quotation check"
    assert "[1]" in out
    assert "1 source(s) attested" in out
