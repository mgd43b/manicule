"""A structured memory: what it claims, what it rests on, and what is refused.

Most of what is asserted here is about what is **not** accepted, because the format's value is that
a memory with a citation can be trusted to have one. A record that silently dropped a malformed
source would be worse than no record: it would present as a memory somebody cited.

Two layers, tested together because they are one rule seen from two doors. The model
(:mod:`manicule.core.knowledge`) decides what a record is; the reader
(:mod:`manicule.ingest.knowledge`) finds it in a file, never raising at ingest and always raising
at the authoring door.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from manicule.core.knowledge import (
    KNOWLEDGE_KEY,
    MAX_QUOTE_CHARS,
    EvidenceStatus,
    Knowledge,
    KnowledgeError,
    KnowledgeType,
    RelationKind,
)
from manicule.ingest.knowledge import check, read
from tests.knowledge_samples import (
    BODY,
    EVIDENCE,
    LOCATOR,
    SECOND_SOURCE,
    SLUG,
    memory,
)

WITHOUT_QUOTE = EVIDENCE.rsplit("    quote:", 1)[0]
"""The sample's evidence with its last field removed, for a test that supplies its own."""


def _refusal(text: str) -> str:
    """The reason ingest recorded for ``text``, failing if it was accepted."""
    found = read(text)
    assert found is not None, "the document does not declare a structured memory at all"
    assert found.record is None, "this was meant to be refused and was read"
    return found.unavailable_reason


def _record_of(text: str) -> object:
    found = read(text)
    assert found is not None
    assert found.record is not None, found.unavailable_reason
    return found.record


# --- the record ---------------------------------------------------------------------------------


def test_a_declared_memory_is_read_with_its_evidence() -> None:
    found = read(memory())

    assert found is not None
    assert found.record is not None
    record = found.record
    assert record.knowledge_type is KnowledgeType.CONSTRAINT
    (source,) = record.evidence
    assert source.kind == "slack"
    assert source.locator == LOCATOR
    assert source.author == "Jane Doe"
    assert source.said_at == datetime(2026, 3, 2, 14, 5, tzinfo=UTC)
    assert "cert rotation: nothing" in source.quote, "a colon inside a quoted message survives"


def test_the_ages_are_the_evidences_in_whatever_offset_each_source_gave() -> None:
    """The Jira source is 09:00 at +02:00, which is 07:00 UTC on the first of April.

    Comparing wall-clock strings instead of instants would order sources written in different
    offsets by the hour on their clock rather than by when they were said.
    """
    found = read(memory(evidence=EVIDENCE + SECOND_SOURCE))
    assert found is not None
    assert found.record is not None

    assert found.record.oldest_said_at == datetime(2026, 3, 2, 14, 5, tzinfo=UTC)
    assert found.record.newest_said_at == datetime(2026, 4, 1, 7, 0, tzinfo=UTC)


def test_the_evidence_status_is_derived_from_how_many_places_the_evidence_points_at() -> None:
    one = read(memory())
    two = read(memory(evidence=EVIDENCE + SECOND_SOURCE))
    same_twice = read(memory(evidence=EVIDENCE + EVIDENCE.split("evidence:\n")[1]))

    assert one is not None
    assert one.record is not None
    assert one.record.evidence_status is EvidenceStatus.SINGLE_SOURCE
    assert two is not None
    assert two.record is not None
    assert two.record.evidence_status is EvidenceStatus.CORROBORATED
    assert same_twice is not None
    assert same_twice.record is not None
    assert same_twice.record.evidence_status is EvidenceStatus.SINGLE_SOURCE, (
        "the same message cited twice is one source said once"
    )


def test_relationships_are_typed_and_declared() -> None:
    found = read(
        memory(
            front="relationships:\n  - relation: supersedes\n    target: reference_old_gateway\n"
        )
    )

    assert found is not None
    assert found.record is not None
    (link,) = found.record.relationships
    assert link.relation is RelationKind.SUPERSEDES
    assert link.target == "reference_old_gateway"


def test_a_multi_line_quote_keeps_its_newline_and_its_tab() -> None:
    block = WITHOUT_QUOTE + '    quote: "first line\\n\\tsecond line"\n'

    found = read(memory(evidence=block))

    assert found is not None
    assert found.record is not None
    assert found.record.evidence[0].quote == "first line\n\tsecond line"


# --- what is refused, and the sentence that says how to fix it ----------------------------------


@pytest.mark.parametrize(
    ("was", "becomes", "says"),
    [
        ("2026-03-02T14:05:00Z", "2026-03-02", "full timestamp"),
        ("2026-03-02T14:05:00Z", "1700000000", "full timestamp"),
        ("2026-03-02T14:05:00Z", "2026-03-02T14:05:00", "UTC offset"),
        (LOCATOR, "javascript:alert(1)", "'javascript' scheme"),
        (LOCATOR, "file:///etc/passwd", "'file' scheme"),
        (LOCATOR, "https://", "names no host"),
        ("kind: slack", "kind: Slack", "one lowercase token"),
        ("kind: slack", 'kind: "team chat"', "one lowercase token"),
        ("    quote:", "    note:", "evidence[0].note"),
        ("author: Jane Doe", 'author: "Jane\\e[2J"', "U+001B"),
    ],
)
def test_a_source_that_cannot_be_cited_is_refused_with_what_to_change(
    was: str, becomes: str, says: str
) -> None:
    assert was in EVIDENCE, "the sample no longer contains what this case breaks"

    assert says in _refusal(memory(evidence=EVIDENCE.replace(was, becomes)))


def test_a_quote_is_required_because_a_permalink_is_all_that_dies() -> None:
    assert "evidence[0].quote" in _refusal(memory(evidence=WITHOUT_QUOTE))


def test_a_blank_quote_is_refused() -> None:
    assert "quote is empty" in _refusal(memory(evidence=WITHOUT_QUOTE + '    quote: "   "\n'))


def test_a_quote_over_the_limit_is_refused_rather_than_truncated() -> None:
    block = WITHOUT_QUOTE + f'    quote: "{"x" * (MAX_QUOTE_CHARS + 1)}"\n'

    assert f"over the {MAX_QUOTE_CHARS}-character limit" in _refusal(memory(evidence=block))


def test_a_quote_cannot_carry_a_terminal_escape() -> None:
    """Every field is printed by ``manicule search``, so a quote is a route to the screen.

    A newline and a tab are allowed because a pasted message has them; an escape sequence is not,
    because ``\\x1b`` opens one and the next bytes can clear or rewrite what a reader is looking at.
    """
    reason = _refusal(memory(evidence=WITHOUT_QUOTE + '    quote: "ok\\e[2Jgone"\n'))

    assert "U+001B" in reason


def test_a_memory_with_no_evidence_is_refused() -> None:
    assert "evidence" in _refusal(memory(evidence=""))
    assert "at least 1" in _refusal(memory(evidence="evidence: []\n"))


@pytest.mark.parametrize(
    ("front", "evidence", "says"),
    [
        ("evidence_status: corroborated\n", EVIDENCE, "points at one place"),
        ("evidence_status: single_source\n", EVIDENCE + SECOND_SOURCE, "points at 2 places"),
        ("evidence_status: superseded\n", EVIDENCE, "superseded_by"),
        ("evidence_status: verified\n", EVIDENCE, "single_source"),
        (
            "relationships:\n  - relation: relates_to\n    target: a\n"
            "  - relation: relates_to\n    target: a\n",
            EVIDENCE,
            "more than once",
        ),
        ("relationships:\n  - relation: relates_to\n    target: a/b\n", EVIDENCE, "not a slug"),
        ("relationships:\n  - relation: owns\n    target: a\n", EVIDENCE, "relation"),
    ],
)
def test_a_record_that_contradicts_itself_is_refused(front: str, evidence: str, says: str) -> None:
    assert says in _refusal(memory(front=front, evidence=evidence))


def test_an_unknown_type_names_the_ones_there_are() -> None:
    reason = _refusal(memory(knowledge_type="recipe"))

    assert "procedure" in reason
    assert "known_issue" in reason


def test_an_unknown_evidence_field_names_the_fields_there_are() -> None:
    reason = _refusal(memory(evidence=EVIDENCE.replace("author:", "by:")))

    assert "evidence[0].by" in reason
    assert "kind, locator, said_at, quote, author" in reason


def test_there_is_no_word_verified_in_the_vocabulary() -> None:
    """Manicule cannot open a permalink, so no status may claim it did."""
    assert "verified" not in {status.value for status in EvidenceStatus}


# --- storing it ---------------------------------------------------------------------------------


def test_a_stored_record_reads_back_equal() -> None:
    found = read(memory(evidence=EVIDENCE + SECOND_SOURCE))
    assert found is not None

    assert Knowledge.from_metadata({KNOWLEDGE_KEY: found.as_metadata_value()}) == found


def test_a_refusal_is_stored_with_its_reason_and_reads_back() -> None:
    found = read(memory(knowledge_type="recipe"))
    assert found is not None

    back = Knowledge.from_metadata({KNOWLEDGE_KEY: found.as_metadata_value()})

    assert back is not None
    assert back.record is None
    assert "recipe" in back.unavailable_reason


@pytest.mark.parametrize(
    "stored",
    [None, "text", 3, [], {}, {"record": {"knowledge_type": "fact"}}, {"unavailable_reason": ""}],
)
def test_a_stored_value_that_no_longer_validates_reads_as_nothing(stored: object) -> None:
    """Fail closed: a corrupt row must not break every listing that touches it."""
    assert Knowledge.from_metadata({KNOWLEDGE_KEY: stored}) is None


def test_a_stored_record_is_revalidated_so_a_poisoned_row_is_not_trusted() -> None:
    found = read(memory())
    assert found is not None
    value = found.as_metadata_value()
    record = value["record"]
    assert isinstance(record, dict)
    evidence = record["evidence"]
    assert isinstance(evidence, list)
    first = evidence[0]
    assert isinstance(first, dict)
    first["locator"] = "javascript:alert(1)"

    assert Knowledge.from_metadata({KNOWLEDGE_KEY: value}) is None


# --- finding it in a file -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "# Just a page\n\nNo front matter at all.\n",
        "---\ntitle: A page\nstatus: draft\nsource: https://example.test\n---\n\n# A page\n",
        "---\nsources:\n  - https://example.test\nevidence: none\n---\n\n# Not ours\n",
        "---\n- just\n- a list\n---\n\n# Odd\n",
    ],
)
def test_an_ordinary_document_is_never_read_as_a_memory(text: str) -> None:
    """The marker is one key, because ``status:`` and ``source:`` mean something everywhere."""
    assert read(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "---",
        "---\n",
        "---\nknowledge_type\n---\n",
        "---\nknowledge_type: [unclosed\n---\n",
        "---\nknowledge_type: fact\n",
        "---\nknowledge_type: fact\nevidence: &a [*a]\n---\n",
        "\x00\x01---\nknowledge_type: fact\n---\n",
        "---\nknowledge_type: fact\nevidence: 3\n---\n",
        "---\nknowledge_type: fact\nevidence:\n  - said_at: 2026-13-45T99:99:99Z\n---\n",
        "---\nknowledge_type: fact\nevidence: !!python/object:os.system {}\n---\n",
    ],
)
def test_reading_never_raises_whatever_the_file_holds(text: str) -> None:
    """A file arrives by a sync; one that cannot be read is still a document.

    The two that look odd are the point: an impossible date is a constructor error inside the YAML
    library rather than a syntax error, and a tag naming a Python object is what the safe loader
    exists to refuse. Either escaping would fail the document's whole ingest.
    """
    found = read(text)

    assert found is None or found.record is not None or found.unavailable_reason


def test_a_yaml_error_is_a_stored_reason_that_names_the_usual_cause() -> None:
    unquoted = WITHOUT_QUOTE + "    quote: failover: manual\n"

    reason = _refusal(memory(evidence=unquoted))

    assert "not valid YAML" in reason
    assert "wrapped in quotes" in reason


def test_front_matter_over_the_size_bound_is_not_read() -> None:
    """The bound is a refusal, not a truncation: half a block would be read as all of it."""
    padding = "".join(f"k{i}: {'x' * 80}\n" for i in range(1200))

    assert read(memory(front=padding)) is None, "the closing fence is past what is scanned"


# --- the authoring door -------------------------------------------------------------------------


def test_a_well_formed_memory_passes_the_door() -> None:
    check(memory(evidence=EVIDENCE + SECOND_SOURCE), slug=SLUG)


@pytest.mark.parametrize(
    "text",
    [
        "# A note\n\nJust text.\n",
        "---\nname: a\ndescription: b\nstatus: draft\nsource: x\n---\n\nJust text.\n",
    ],
)
def test_the_door_is_silent_on_a_document_that_is_not_a_memory(text: str) -> None:
    """Nothing here makes authoring stricter for a collection that holds anything else."""
    check(text, slug="anything")


def test_evidence_without_a_type_is_refused_because_it_would_be_ignored() -> None:
    with pytest.raises(KnowledgeError, match=r"evidence.*no knowledge_type"):
        check(memory(knowledge_type=None), slug=SLUG)


@pytest.mark.parametrize("misspelled", ["sources", "citations", "citation"])
def test_evidence_under_a_name_nothing_reads_is_refused(misspelled: str) -> None:
    text = memory(front=f"{misspelled}:\n  - {LOCATOR}\n")

    with pytest.raises(KnowledgeError, match=rf"{misspelled}, which nothing reads"):
        check(text, slug=SLUG)


def test_a_broken_record_raises_at_the_door_with_the_reason_ingest_would_store() -> None:
    text = memory(knowledge_type="recipe")

    with pytest.raises(KnowledgeError) as raised:
        check(text, slug=SLUG)

    assert str(raised.value) == _refusal(text)


def test_a_memory_may_not_relate_to_itself() -> None:
    text = memory(front=f"relationships:\n  - relation: relates_to\n    target: {SLUG}\n")

    with pytest.raises(KnowledgeError, match="its own target"):
        check(text, slug=SLUG)


EMPTY_APPLICABILITY = BODY.replace(
    "Production only, since March 2026. The staging pair still fails over on its own.\n", ""
)


@pytest.mark.parametrize(
    ("body", "missing"),
    [
        (BODY.replace("# VPN gateway failover is manual\n", ""), "`# Title`"),
        (BODY.replace("# VPN", "## VPN"), "`# Title`"),
        (BODY.replace("## Problem", "## Issue"), "`## Problem`"),
        (BODY.replace("## Answer", "## Solution"), "`## Answer`"),
        (BODY.replace("## Applicability", "## Where"), "`## Applicability`"),
        (EMPTY_APPLICABILITY, "`## Applicability`"),
    ],
)
def test_a_body_without_its_title_and_sections_is_refused(body: str, missing: str) -> None:
    """The body is what is indexed, so it is what a reader finds a memory by."""
    with pytest.raises(KnowledgeError, match="the body is missing") as raised:
        check(memory(body=body), slug=SLUG)

    assert missing in str(raised.value)


def test_a_section_heading_inside_a_code_fence_is_not_a_section() -> None:
    fenced = (
        "# Title\n\n## Problem\n\nText.\n\n```\n## Answer\nnot a heading\n```\n\n"
        "## Applicability\n\nHere.\n"
    )

    with pytest.raises(KnowledgeError, match="`## Answer`"):
        check(memory(body=fenced), slug=SLUG)


@pytest.mark.parametrize("heading", ["## answer", "## Answer:", "##   Answer  "])
def test_section_names_are_matched_the_way_a_person_writes_them(heading: str) -> None:
    check(memory(body=BODY.replace("## Answer", heading)), slug=SLUG)


def test_a_long_memory_is_checked_to_its_last_section() -> None:
    """The ingest-time scan is bounded and the door's is not, or this would be refused.

    The last section sits past the front-matter bound, which is the failure a shared bounded
    slice would produce: a memory refused for lacking a section it has.
    """
    long_answer = BODY.replace("## Answer\n", "## Answer\n\n" + ("Detail. " * 20000) + "\n")

    check(memory(body=long_answer), slug=SLUG)


def test_the_clock_is_not_an_input_to_what_is_a_record() -> None:
    """A ``said_at`` in the future is accepted, because validity cannot depend on today.

    A check that read the clock would make a stored record valid on Monday and refused on Tuesday
    over the same bytes, which is the one thing a re-read of the database must not do.
    """
    future = (datetime.now(UTC) + timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")

    assert _record_of(memory(evidence=EVIDENCE.replace("2026-03-02T14:05:00Z", future)))


@pytest.mark.parametrize(
    ("was", "becomes", "says"),
    [
        ("kind: slack", 'kind: "slack\\n"', "one lowercase token"),
        ("target: reference_vpn_cert_rotation", 'target: "a\\n"', "not a slug"),
    ],
)
def test_a_trailing_newline_is_not_a_token(was: str, becomes: str, says: str) -> None:
    """``$`` matches before a final newline, so an anchored ``match`` accepted ``"slack\\n"``."""
    front = "relationships:\n  - relation: relates_to\n    target: reference_vpn_cert_rotation\n"
    text = memory(front=front, evidence=EVIDENCE)
    assert was in text + EVIDENCE or was.startswith("target")

    assert says in _refusal(text.replace(was, becomes))
