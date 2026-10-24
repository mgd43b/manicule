"""Round 6's instrument, executed by the suite at a size that costs nothing.

``docs/retrieval.md`` §5.4 records a decision on the strength of
:mod:`tests.benchmarks.query_shapes`. What is asserted is the shape of the instrument: the gate
selects what its docstring says, the substitute fusion changes a gated query's lexical weight and
nothing else, and the rule chooses what it says it chooses. Nothing here reads a measured number.
"""

from __future__ import annotations

import pytest

from manicule.core.retrieval import Candidate, Filter, Query
from manicule.retrieval.config import FusionConfig
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.trace import installed
from tests.benchmarks import thread_duplication
from tests.benchmarks.lexical_questions import DOCS, Asked, Outcome
from tests.benchmarks.query_shapes import (
    CANDIDATES,
    PLAIN,
    SHORT,
    Gate,
    bare_identifiers,
    choose,
    exactly_the_counted,
    gated_weighting,
    plain_lookup,
    verdicts,
)
from tests.storage_helpers import make_chunk, make_document


@pytest.mark.parametrize(
    ("text", "short", "plain"),
    [
        ("retrieval", True, True),
        ("memory-authoring", True, True),
        ("Route manifest", True, True),
        ("russet survey tool downtime notices", False, True),
        ("escape_match_query", False, False),
        ("SEARCH_SQL", False, False),
        ("RRFStage", False, False),
        ("docs/retrieval.md", False, False),
        ("bge-m3", False, False),
        ("what is retrieval", False, False),
        ("how does reciprocal rank fusion combine the two lists", False, False),
        ("", False, False),
    ],
)
def test_the_gate_selects_plain_words_and_never_code(text: str, short: bool, plain: bool) -> None:
    assert plain_lookup(text, SHORT) is short
    assert plain_lookup(text, PLAIN) is plain


def test_every_bare_identifier_falls_outside_every_gate() -> None:
    asked = bare_identifiers(DOCS)
    assert asked
    assert {item.group for item in asked} == {"bare identifiers"}
    assert not any(gate.gated(item.text) for item in asked for gate in CANDIDATES.values())


async def test_a_gated_query_has_its_lexical_ladder_weighted_and_nothing_else_does() -> None:
    document = make_document(source_id="d")
    chunks = [make_chunk(document, i, f"passage {i}") for i in range(3)]
    candidates = [
        Candidate(chunk=chunks[0], score=0.0, scores={"dense": 0.9}),
        Candidate(chunk=chunks[1], score=0.0, scores={"dense": 0.5, "lexical": 2.0}),
        Candidate(chunk=chunks[2], score=0.0, scores={"lexical": 3.0}),
    ]
    stage = RRFStage(config=FusionConfig(k=60))
    scope = Filter(workspace_ids=frozenset({"w"}))
    short = Query(text="retrieval", filter=scope)
    question = Query(text="what is retrieval for", filter=scope)
    shipped = {c.chunk.id: c.scores["rrf"] for c in await stage.run(short, candidates)}

    gated: set[str] = set()
    with installed(), gated_weighting(Gate(most=SHORT, lexical_weight=0.0), gated):
        dense_only = await stage.run(short, candidates)
        untouched = await stage.run(question, candidates)
    assert gated == {"retrieval"}, "the substitute records what it gated, and only that"
    assert [c.chunk.id for c in dense_only] == [chunks[0].id, chunks[1].id, chunks[2].id]
    assert {c.chunk.id: c.scores["rrf"] for c in dense_only}[chunks[2].id] == 0.0
    assert {c.chunk.id: c.scores["rrf"] for c in untouched} == pytest.approx(shipped)
    assert {c.chunk.id: c.scores["rrf"] for c in await stage.run(short, candidates)} == (
        pytest.approx(shipped)
    ), "the shipped arithmetic is restored"

    with installed() as frame, gated_weighting(Gate(most=SHORT, lexical_weight=0.0), set()):
        await stage.run(short, candidates)
    assert not frame.incomparable, "a weight of 0 is not a leg that returned nothing"


def test_the_choice_needs_every_model_and_breaks_ties_by_order() -> None:
    both = {"short-dense": (True, 0.0), "short-half": (True, 0.004), "plain-half": (True, 0.1)}
    assert choose([both, both]) == "plain-half"
    near = {"short-dense": (True, 0.0), "short-half": (True, 0.004)}
    assert choose([near, near]) == "short-dense", "within 0.005, the earlier candidate"
    assert choose([near, {"short-dense": (False, 0.0), "short-half": (True, 0.0)}]) == (
        "short-half"
    )
    assert choose([{"short-dense": (True, 0.0)}, {"short-dense": (False, 0.0)}]) is None
    assert next(iter(CANDIDATES)) == "shipped"
    assert not CANDIDATES["shipped"].gated("retrieval")


def test_a_run_that_gated_other_queries_than_were_counted_is_refused() -> None:
    """A glossary rewording changes the text fusion sees; the counted subsets would be wrong."""
    gate = CANDIDATES["short-dense"]
    asked = ["retrieval", "what is retrieval for", "ADF"]
    exactly_the_counted(gate, asked, {"retrieval", "ADF"})
    with pytest.raises(RuntimeError, match="missed"):
        exactly_the_counted(gate, asked, {"retrieval"})
    with pytest.raises(RuntimeError, match="should not have"):
        exactly_the_counted(gate, asked, {"retrieval", "ADF", "Atlassian Document Format"})
    assert Gate().most is None


def _outcome(asked: Asked, rank: int | None) -> Outcome:
    return Outcome(asked, "or-first-16", "hybrid", rank, int(rank is not None), None, None, None)


def test_a_loss_confined_to_the_gated_headings_fails_however_the_family_averages() -> None:
    """The gated subset is scored on its own, so the whole family's average cannot hide it.

    Two gated headings fall from first to tenth while 38 ungated ones rise: the family's lower
    bound stays positive, the gated subset's does not, and the candidate fails. A gate that
    selects nothing scores no gated family at all, rather than one of zero questions.
    """
    title = Asked("title-retrieval", "retrieval", "titles", documents=frozenset({"d"}))
    gated = [
        Asked(f"g{i}", text, "headings", (("a.md", str(i)),))
        for i, text in enumerate(("Auth", "Scope"))
    ]
    ungated = [
        Asked(f"u{i}", f"The rule that heading {i} states", "headings", (("a.md", f"9.{i}"),))
        for i in range(38)
    ]
    shipped = [
        _outcome(title, 2),
        *(_outcome(a, 1) for a in gated),
        *(_outcome(a, 3) for a in ungated),
    ]
    candidate = [
        _outcome(title, 1),
        *(_outcome(a, 10) for a in gated),
        *(_outcome(a, 1) for a in ungated),
    ]
    dense = [_outcome(title, 1)]
    docs = dict.fromkeys(CANDIDATES, shipped) | {"short-dense": candidate}
    threads: dict[str, list[thread_duplication.Outcome]] = {name: [] for name in CANDIDATES}
    selected = dict.fromkeys(CANDIDATES, frozenset({"Auth", "Scope", "retrieval"}))

    found = verdicts(dense, docs, threads, selected)["short-dense"]
    assert found.families["headings"].interval[0] > 0
    assert found.families["headings, gated"].count == 2
    assert found.families["headings, gated"].interval[1] < -0.02
    assert found.closes_half
    assert not found.fixes, "the gated loss alone fails it"

    nothing = verdicts(dense, docs, threads, dict.fromkeys(CANDIDATES, frozenset()))
    assert "headings, gated" not in nothing["short-dense"].families
    assert "keyword, gated" not in nothing["short-dense"].families
    assert nothing["short-dense"].fixes, "with no gated family, the whole family is the check"
