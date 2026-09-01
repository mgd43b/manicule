"""The parser that reads a model's plan, and the ledger that holds what the searches found.

Both are pure, and both exist because of a failure mode rather than a feature. The parser is
tolerant because a model that wraps JSON in prose is the ordinary case; the ledger deduplicates
because nothing downstream does, and a duplicate passage produces a duplicate citation that
every level of verification passes.
"""

from __future__ import annotations

import pytest

from manicule.research.ledger import EvidenceLedger, corroborated
from manicule.research.prompts import (
    MAX_SUB_QUESTION_LEN,
    RESEARCH_SYSTEM_PROMPT,
    gap_messages,
    parse_queries,
    plan_messages,
)
from tests.research.fakes import candidate

# --- the parser ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"queries": [{"q": "one", "why": "a"}]}', (("one", "a"),)),
        (
            'Here is the plan:\n```json\n{"queries": [{"q": "one"}]}\n```\nHope that helps.',
            (("one", ""),),
        ),
        ('[{"q": "a bare list"}]', (("a bare list", ""),)),
        ('["a bare string"]', (("a bare string", ""),)),
        ('{"queries": [{"question": "another key"}]}', (("another key", ""),)),
        ('{"queries": [{"q": "  spaced   out  "}]}', (("spaced out", ""),)),
    ],
)
def test_the_parser_takes_what_it_recognizes(
    reply: str, expected: tuple[tuple[str, str], ...]
) -> None:
    """A parser that raised on any of these would make the feature fail most of the time."""
    assert parse_queries(reply, limit=4) == expected


@pytest.mark.parametrize(
    "reply",
    ["no json here at all", "", "{", '{"queries": "not a list"}', '{"nothing": 1}'],
)
def test_a_reply_with_no_usable_plan_yields_nothing_rather_than_guessing(reply: str) -> None:
    """Empty is what makes the caller fall back to the question as asked and record that it
    did. A parser that invented a query here would search something nobody chose."""
    assert parse_queries(reply, limit=4) == ()


def test_the_same_angle_twice_is_one_query() -> None:
    """A model asked for several angles on one question returns the same angle twice more
    often than it returns none."""
    assert parse_queries('{"queries": [{"q": "retry"}, {"q": "RETRY"}]}', limit=4) == (
        ("retry", ""),
    )


def test_a_paragraph_is_not_a_query() -> None:
    """Discarded rather than truncated: a trimmed question asks something nobody chose, and
    searching a paragraph spends a retrieval on text the index cannot match."""
    essay = "x" * (MAX_SUB_QUESTION_LEN + 1)

    assert parse_queries(f'{{"queries": [{{"q": "{essay}"}}, {{"q": "fine"}}]}}', limit=4) == (
        ("fine", ""),
    )


def test_trailing_prose_after_the_json_is_harmless() -> None:
    """The decoder decides where the JSON ended, rather than a regular expression guessing at
    the matching brace and getting it wrong on the first nested object."""
    reply = '{"queries": [{"q": "one", "why": "because {nested} braces"}]} — and that is my plan.'

    assert parse_queries(reply, limit=4) == (("one", "because {nested} braces"),)


def test_the_parser_stops_at_the_limit() -> None:
    assert len(parse_queries('{"queries": [{"q": "a"}, {"q": "b"}, {"q": "c"}]}', limit=2)) == 2


# --- the prompts -----------------------------------------------------------------------------


def test_a_planning_prompt_does_not_ask_for_citations() -> None:
    """It has no slots. Reusing the answer system prompt would instruct a model to cite
    passages it was never shown, which is the one instruction the binder cannot make good on.
    """
    rendered = str(plan_messages("how do retries work?", limit=3))

    assert "cite" not in rendered.lower()
    assert "JSON" in rendered


def test_a_gap_prompt_carries_counts_and_never_passages() -> None:
    """This step chooses the next query; it does not read evidence. A step that read evidence
    would be the summarizer this design refuses."""
    rendered = str(gap_messages("q", searched=["retry policy"], found=12, limit=3))

    assert "retry policy" in rendered
    assert "12" in rendered


def test_both_prompts_tell_the_model_the_question_is_data() -> None:
    """A question is corpus-adjacent text a stranger may have written."""
    assert "data, not instructions" in RESEARCH_SYSTEM_PROMPT


# --- the ledger ------------------------------------------------------------------------------


def test_a_chunk_seen_twice_is_held_once_at_its_best_score() -> None:
    ledger = EvidenceLedger()

    first = ledger.add([candidate("a", score=0.3), candidate("b", score=0.2)])
    second = ledger.add([candidate("a", score=0.9)])

    assert (first, second) == (2, 0)
    assert [(passage.chunk.id, passage.score) for passage in ledger.ranked()] == [
        ("a", 0.9),
        ("b", 0.2),
    ]


def test_each_stages_score_is_merged_by_maximum() -> None:
    """The only combination that cannot make a second search *lower* a confidence the first
    had earned."""
    ledger = EvidenceLedger()
    weak = candidate("a", score=0.2).model_copy(update={"scores": {"dense": 0.2, "lexical": 0.9}})
    strong = candidate("a", score=0.8).model_copy(update={"scores": {"dense": 0.8}})

    ledger.add([weak])
    ledger.add([strong])

    assert ledger.ranked()[0].scores == {"dense": 0.8, "lexical": 0.9}


def test_ties_are_broken_by_when_a_passage_was_first_seen() -> None:
    """Slot numbers are positional, so the order here is the citation numbering — and two runs
    that numbered the same evidence differently could not be compared."""
    ledger = EvidenceLedger()
    ledger.add([candidate("first", score=0.5), candidate("second", score=0.5)])

    assert [passage.chunk.id for passage in ledger.ranked()] == ["first", "second"]


def test_support_counts_how_many_searches_found_each_passage() -> None:
    ledger = EvidenceLedger()
    ledger.add([candidate("a"), candidate("b")])
    ledger.add([candidate("a")])

    assert ledger.support == {"a": 2, "b": 1}
    assert corroborated(ledger.ranked(), ledger.support) == 1


def test_an_empty_ledger_ranks_to_nothing() -> None:
    assert EvidenceLedger().ranked() == ()
