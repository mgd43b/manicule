"""The thread duplication measurement is executed by the suite, at a size that costs nothing.

``docs/connectors/gmail.md`` §8 records a decision on the strength of this program, so the suite
keeps it working. What it asserts is the shape of the instrument rather than a magnitude: that
the generator quotes the way its docstring says, that the ground truth can place every passage
it put in a message, that the scoring arithmetic gives known answers on a hand-built page, and
that the whole program still runs end to end through the shipped runtime.

The end-to-end run uses the suite's stand-in embedder, whose vectors carry no meaning, so its
numbers are not a measurement of anything and nothing here reads them as one. The recorded
figures come from a real model, run by hand.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from manicule.container import keys
from manicule.core.anchors import LineAnchor
from manicule.plugins.registry import discover
from tests.benchmarks.thread_corpus import ASPECTS, STYLES, Corpus, build_corpus, quote
from tests.benchmarks.thread_duplication import (
    Asked,
    ChunkTruth,
    Hit,
    canonical_lines,
    carried,
    locate,
    measure,
    rows,
    score,
)
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    from pathlib import Path

    from tests.benchmarks.thread_corpus import Message


def small(styles: tuple[str, ...] = STYLES) -> Corpus:
    """Two threads per style, of two and five messages, and two distractors."""
    return build_corpus(lengths=(2, 5), styles=styles, copies=1, distractors=2)


class CountingEmbedder(HashEmbedder):
    """The stand-in, counting its own tokens so the structural chunker binds to it.

    Without a count the chunker measures with a provisional tokenizer, and ingest refuses to
    index provisional chunks — correctly, and not what this test is about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fingerprint = self.fingerprint.model_copy(update={"max_sequence_length": 8192})

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_every_keyword_lookup_is_made_of_words_its_statement_contains() -> None:
    """The lexical leg matches only a chunk holding every term, so this is what reaches it."""
    for statement, _, _, keywords in ASPECTS:
        words = set(re.findall(r"\w+", statement.lower()))
        assert set(re.findall(r"\w+", keywords.lower())) <= words, keywords


def test_quoting_nests_one_marker_per_level() -> None:
    assert quote("a\n\n> b\n>> c") == "> a\n>\n>> b\n>>> c"


@pytest.mark.parametrize(
    ("style", "carried_by_last"),
    [("top", 5), ("inline", 3), ("html", 5)],
)
async def test_each_style_repeats_what_its_docstring_says(style: str, carried_by_last: int) -> None:
    corpus = small((style,))
    [_, thread] = corpus.threads
    last = thread.messages[-1]
    assert len(last.carries) == carried_by_last
    assert last.carries[0] is last.own
    text = "\n".join((await canonical_lines(last.raw)).values())
    quoted_lines = [line for line in text.splitlines() if line.startswith(">")]
    if style == "html":
        assert not quoted_lines, "the converter drops a blockquote, so no marker survives"
    else:
        assert quoted_lines


async def test_the_ground_truth_places_every_passage_the_generator_wrote() -> None:
    corpus = small()
    for message in corpus.messages:
        located = locate(await canonical_lines(message.raw), message.carries)
        assert set(located) == {passage.id for passage in message.carries}


async def test_a_chunk_carries_a_passage_only_when_it_holds_most_of_it() -> None:
    message = small(("top",)).threads[0].messages[-1]
    lines = await canonical_lines(message.raw)
    located = locate(lines, message.carries)
    own = located[message.own.id]
    first, last = min(own), max(own)

    whole = carried(LineAnchor(start=1, end=max(lines)), located)
    assert whole == {passage.id for passage in message.carries}
    assert message.own.id in carried(LineAnchor(start=first, end=last), located)
    assert message.own.id not in carried(LineAnchor(start=1, end=first - 1), located)


def _asked(message: Message, companion: Message) -> Asked:
    from manicule.evaluation.queries import EvalQuery  # noqa: PLC0415

    return Asked(
        query=EvalQuery(id="q", text="q"),
        passage=message.own,
        origin=message,
        companion=companion,
        style="top",
        length=5,
        position=0,
        form="question",
    )


def test_scoring_a_hand_built_page() -> None:
    """A quote first, then the original, a repeat, and the companion: every figure is known."""
    corpus = small(("top",))
    thread = corpus.threads[1]
    origin, reply, later = thread.messages[0], thread.messages[1], thread.messages[2]
    target = origin.own.id
    companion = corpus.companions[target]
    page = [
        Hit(1, 0.9, ChunkTruth(reply, frozenset({reply.own.id, target}))),
        Hit(2, 0.8, ChunkTruth(origin, frozenset({target}))),
        Hit(3, 0.7, ChunkTruth(later, frozenset({target, reply.own.id}))),
        Hit(4, 0.6, ChunkTruth(companion, frozenset())),
    ]

    outcome = score(_asked(origin, companion), page, pipeline="hybrid", k=4)
    assert outcome.origin == "below"
    assert outcome.repeats == 2
    assert outcome.redundant == 2, "the original and the later quote add nothing new"
    assert outcome.companion
    assert outcome.page == 4

    tied = [Hit(1, 0.8, page[0].truth), Hit(2, 0.8, page[1].truth)]
    assert score(_asked(origin, companion), tied, pipeline="hybrid", k=2).origin == "tied"
    assert score(_asked(origin, companion), page[:1], pipeline="hybrid", k=1).origin == (
        "quote only"
    )
    assert score(_asked(origin, companion), page[3:], pipeline="hybrid", k=1).origin == (
        "not found"
    )
    assert score(_asked(origin, companion), page[1:], pipeline="hybrid", k=1).origin == "first"


async def test_the_program_runs_end_to_end_through_the_runtime(
    manicule_environment: Path,
) -> None:
    """Indexed through the Gmail connector and the shipped pipeline, queried under each pipeline."""
    found = discover()
    found.registry.bind("test").add(keys.EMBEDDER.named("local"), lambda _: CountingEmbedder())
    overrides = {"embedding": {"provider": "local"}}
    corpus = small()

    result = await measure(
        corpus,
        manicule_environment / "index",
        overrides=overrides,
        discovery=found,
        pipelines=("hybrid", "hybrid+collapse"),
    )

    assert result.indexed.documents == len(corpus.messages)
    assert result.runs["hybrid+collapse"].configuration["stages"][-1] == "collapse"
    assert result.runs["hybrid+collapse"].stage_ms["collapse"]
    expected = len(result.query_set.queries) * 2  # two values of k
    assert all(len(run.outcomes) == expected for run in result.runs.values())
    assert rows(result.outcomes, pipeline="hybrid", k=10)[-1].queries == len(
        result.query_set.queries
    )
