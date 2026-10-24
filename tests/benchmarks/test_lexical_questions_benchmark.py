"""The lexical-leg measurement is executed by the suite, at a size that costs nothing.

``docs/retrieval.md`` §4.1 records a decision on the strength of this program, so the suite keeps
it working. What it asserts is the shape of the instrument: that every label names a section the
documents actually have, that the scoring gives known answers, that each way of building a match
is the expression it claims to be, and that the whole program runs end to end through the
shipped runtime. The end-to-end run uses the suite's stand-in embedder, so its numbers measure
nothing and nothing here reads them as one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from manicule.container import keys
from manicule.core.retrieval import Candidate, Filter, Query
from manicule.evaluation.queries import load_query_set
from manicule.plugins.registry import discover
from manicule.retrieval.config import FusionConfig
from manicule.retrieval.fusion import DEGRADED_REASON, RRFStage
from manicule.retrieval.trace import installed
from manicule.storage.fts import FUNCTION_WORDS, escape_match_query, match_words
from tests.benchmarks.lexical_questions import (
    AUTHORED,
    CAP,
    CAPPED,
    DOCS,
    HELD_OUT_COPIES,
    HELD_OUT_DISTRACTORS,
    HELD_OUT_SEED,
    MULTI_HOP_TARGETS,
    PASTE_STRIDE,
    PASTED,
    UNANSWERABLE,
    VARIANTS,
    Asked,
    FamilyChange,
    Fusion,
    Place,
    RoundFour,
    RoundThree,
    RoundThreeB,
    _lost,  # pyright: ignore[reportPrivateUsage]
    _rank,  # pyright: ignore[reportPrivateUsage]
    bootstrap_interval,
    fusion_overrides,
    identifier_questions,
    index_docs,
    paste_questions,
    round_four_winner,
    run_docs,
    run_threads,
    summarize,
    time_at_scale,
    two_word_questions,
    weighting,
)
from tests.benchmarks.test_thread_duplication_benchmark import CountingEmbedder, small
from tests.benchmarks.thread_corpus import build_corpus
from tests.benchmarks.thread_duplication import queries
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path


def _sections(path: str) -> set[str]:
    text = (DOCS / path).read_text(encoding="utf-8")
    return {match.group(1).rstrip(".") for match in re.finditer(r"^#+ ([0-9][0-9.]*) ", text, re.M)}


def test_every_label_names_a_section_the_documents_have() -> None:
    """A label pointing nowhere scores every system zero on that query, identically and silently."""
    targets = [target for _, _, target in AUTHORED]
    targets.extend(t for group in MULTI_HOP_TARGETS.values() for t in group)
    for path, number in targets:
        assert number in _sections(path), f"{path} has no section {number}"


def test_a_section_number_names_its_subsections_and_nothing_else() -> None:
    place = Place("retrieval.md", ("Retrieval", "6. Rerank", "6.2 The model"), "d")
    assert place.within(("retrieval.md", "6"))
    assert place.within(("retrieval.md", "6.2"))
    assert not place.within(("retrieval.md", "6.20"))
    assert not place.within(("storage.md", "6"))
    assert not Place("retrieval.md", ("Retrieval", "16. Later"), "d").within(("retrieval.md", "6"))


def test_each_variant_is_the_expression_it_claims_to_be() -> None:
    question = "how do I rotate the signing token"
    assert VARIANTS["and"](question) == '"how" "do" "I" "rotate" "the" "signing" "token"'
    assert VARIANTS["or"](question).count(" OR ") == 6
    assert VARIANTS["or-content"](question) == '"rotate" OR "signing" OR "token"'
    assert VARIANTS["and-content"](question) == '"rotate" "signing" "token"'
    assert VARIANTS["all-but-one"](question) == (
        '("signing" "token") OR ("rotate" "token") OR ("rotate" "signing")'
    )


def test_a_question_of_function_words_alone_keeps_them() -> None:
    """Dropping every word would turn a question into no search at all."""
    assert VARIANTS["or-content"]("what is it") == '"what" OR "is" OR "it"'


def test_the_winning_variant_is_the_shipped_rule_and_the_baseline_is_the_old_one() -> None:
    """A re-run measures what ships, and compares it against the rules it replaced."""
    assert VARIANTS["or-first-16"] is escape_match_query
    question = "how do I rotate the signing token"
    assert VARIANTS["and"](question) == '"how" "do" "I" "rotate" "the" "signing" "token"'
    assert VARIANTS["and"]("- token token") == '"-" "token" "token"', "verbatim, defects included"


def test_scoring_reads_the_rank_of_the_first_labeled_chunk() -> None:
    asked = Asked("q", "?", "authored", (("retrieval.md", "6"),))
    places = [
        Place("storage.md", ("6. Derived",), "a"),
        Place("retrieval.md", ("6. Rerank", "6.1 A cross-encoder"), "b"),
    ]
    assert _rank(asked, places) == (2, 1)
    assert _rank(asked, places[:1]) == (None, 0)
    assert _rank(Asked("t", "?", "titles", documents=frozenset({"a"})), places) == (1, 1)


def test_the_statement_is_timed_for_every_variant_and_every_set() -> None:
    """Per set, because a mean over all of them is a mean over whichever set is largest."""
    texts = [("authentication and token rotation", "Auth"), ("vector index builds", "Index")]
    sets = {"prose": ["how is a token rotated", "zzz"], "identifiers": ["vector"]}
    timings = time_at_scale(texts, sets, size=200)
    assert set(timings) == {(variant, group) for variant in VARIANTS for group in sets}
    assert timings["and", "prose"][2] == 0, "no chunk holds every word of either question"
    assert timings["or-content", "prose"][2] > 0


async def test_the_program_runs_end_to_end_through_the_runtime(
    manicule_environment: Path, tmp_path: Path
) -> None:
    """Indexed through the filesystem connector and the shipped pipeline, queried per variant."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "retrieval.md").write_text(
        "# Retrieval\n\n## 5. Fusion\n\n### 5.1 RRF takes ranks\n\n"
        "Reciprocal rank fusion combines the dense and lexical lists by rank.\n",
        encoding="utf-8",
    )
    (root / "storage.md").write_text(
        "# Storage\n\n## 6. Derived indexes\n\nThe full text index is kept in step by triggers.\n",
        encoding="utf-8",
    )
    found = discover()
    found.registry.bind("test").add(keys.EMBEDDER.named("local"), lambda _: CountingEmbedder())
    overrides = {"embedding": {"provider": "local"}}

    index = await index_docs(
        manicule_environment / "index", overrides=overrides, discovery=found, root=root
    )
    assert index.documents == 2
    asked = [
        Asked("rrf", "how does reciprocal rank fusion combine lists", "authored",
              (("retrieval.md", "5.1"),)),
    ]  # fmt: skip
    for variant in ("and", "or-content"):
        outcomes = await run_docs(
            index, asked, variant=variant, pipeline="lexical", overrides=overrides, discovery=found
        )
        row = summarize("authored", outcomes)
        if variant == "and":
            assert row.empty == 1, "no chunk holds 'how' and 'does'"
        else:
            assert row.empty == 0
            assert row.hits[5] == 1


async def test_the_mail_half_can_run_twice_in_one_work_directory(
    manicule_environment: Path,
) -> None:
    """``--work`` keeps the docs/ index; the mail index is rebuilt, so a rerun cannot fail on it."""
    found = discover()
    found.registry.bind("test").add(keys.EMBEDDER.named("local"), lambda _: CountingEmbedder())
    overrides = {"embedding": {"provider": "local"}}
    for _ in range(2):
        runs = await run_threads(
            manicule_environment,
            small(),
            variants=("and", "and-content"),
            overrides=overrides,
            discovery=found,
        )
        assert set(runs) == {
            ("and", "dense"),
            ("and", "hybrid"),
            ("and-content", "hybrid"),
            ("and", "lexical"),
            ("and-content", "lexical"),
        }
        keywords = [o for o in runs["and-content", "lexical"] if o.asked.form == "keywords"]
        assert all(not o.single_leg for o in keywords), "every keyword lookup reaches the leg"


def test_a_narrowed_run_times_only_the_variants_it_names() -> None:
    timings = time_at_scale(
        [("token rotation", "Auth")], {"prose": ["token"]}, size=20, variants=["and"]
    )
    assert set(timings) == {("and", "prose")}


def test_every_two_word_question_has_two_content_words_and_round_twos_target() -> None:
    """The two rules build different matches only where there is more than one content word."""
    two = two_word_questions()
    one = identifier_questions()
    assert [target for _, target in two] == [target for _, target in one]
    for question, _ in two:
        words = match_words(question)
        assert len([w for w in words if w.lower() not in FUNCTION_WORDS]) == 2, question
        assert escape_match_query(question) != VARIANTS["and-content"](question)


def test_the_bootstrap_interval_is_reproducible_and_brackets_the_mean() -> None:
    differences = [0.5, -0.25, 0.0, 1.0, -0.5, 0.25] * 10
    low, high = bootstrap_interval(differences, resamples=2000)
    assert (low, high) == bootstrap_interval(differences, resamples=2000)
    assert low < sum(differences) / len(differences) < high


def _round(**changes: object) -> RoundThree:
    figures: dict[str, object] = {
        "gain": 0.15,
        "gain_split": (120, 10),
        "gain_p": 1e-6,
        "natural": -0.01,
        "natural_interval": (-0.04, 0.02),
        "natural_count": 190,
        "lost_titles": 0,
        "lost_keywords": 0,
        "unanswerable_out_of_none": 0,
        "latency": {"authored": (30.0, 60.0), "identifiers": (10.0, 20.0)},
        "by_match": {True: (100, 0.02), False: (90, -0.04)},
    }
    figures.update(changes)
    return RoundThree(**figures)  # pyright: ignore[reportArgumentType]


def test_a_lost_lookup_is_not_paid_for_by_a_gained_one() -> None:
    """Condition 3 counts the incumbent's hits the challenger misses, paired by query."""
    assert _lost([True, True, False, False], [False, False, True, True]) == 2
    assert _lost([True, False], [True, True]) == 0


def test_round_three_switches_only_when_every_condition_holds() -> None:
    assert _round().switch, "a gain of 0.15 tolerates a natural-question loss down to -0.05"
    assert not _round(gain=0.09).switch, "condition 1: a gain under 0.10 does not count"
    assert not _round(gain_p=0.02).switch, "condition 1: nor does one a sign test cannot see"
    assert not _round(natural_interval=(-0.051, 0.0)).switch, "condition 2: the -0.05 floor"
    assert not _round(gain=0.12, natural_interval=(-0.041, 0.0)).switch, "a third of 0.12"
    assert not _round(lost_titles=2).switch
    assert not _round(unanswerable_out_of_none=1).switch
    assert not _round(latency={"authored": (49.0, 101.0)}).switch
    assert not _round(latency={"authored": (49.0, 80.0), "example": (30.0, 120.0)}).switch, (
        "the worst p95 need not belong to the slowest median"
    )
    assert "keep and-content" in _round(lost_keywords=2).describe()


def _content_words(text: str) -> int:
    words = match_words(text)
    return len([w for w in words if w.lower() not in FUNCTION_WORDS] or words)


def test_round_three_questions_fit_the_cap() -> None:
    """Round 3b's cap changes nothing round 3 measured: every question it asked fits under it."""
    texts = [text for text, _, _ in AUTHORED]
    texts.extend(UNANSWERABLE)
    for name in ("multi-hop-queries.json", "example-queries.json"):
        texts.extend(query.text for query in load_query_set(DOCS / "evaluation" / name).queries)
    texts.extend(question for question, _ in identifier_questions())
    texts.extend(question for question, _ in two_word_questions())
    for seed, copies, distractors in ((HELD_OUT_SEED, HELD_OUT_COPIES, HELD_OUT_DISTRACTORS),):
        _, asked = queries(build_corpus(seed=seed, copies=copies, distractors=distractors))
        texts.extend(item.query.text for item in asked)
    assert max(map(_content_words, texts)) <= CAP


def test_a_paste_is_its_window_long_and_targets_its_own_section() -> None:
    place = Place("retrieval.md", ("Retrieval", "6. Rerank", "6.2 The model"), "d")
    long_text = " ".join(f"word{i}" for i in range(130))
    passages = [(place, long_text)] + [(place, "short")] * (PASTE_STRIDE - 1)
    passages.append((Place("intro.md", ("Introduction",), "e"), long_text))
    pasted = paste_questions(passages)
    assert [item.group for item in pasted] == ["pasted, 40 words", "pasted, 120 words"]
    assert all(item.targets == (("retrieval.md", "6.2"),) for item in pasted)
    assert len(pasted[1].text.split()) == 120
    assert escape_match_query(pasted[1].text).count(" OR ") == CAP - 1
    assert VARIANTS["or-content"](pasted[1].text).count(" OR ") == 119


def test_round_three_b_hands_a_trade_off_to_the_owner() -> None:
    fast = dict.fromkeys(PASTED, (20.0, 40.0))
    assert RoundThreeB(-0.01, (-0.02, 0.0), 100, fast).verdict == f"ship {CAPPED}"
    assert "owner" in RoundThreeB(-0.05, (-0.08, -0.02), 100, fast).verdict
    slow = {**fast, PASTED[1]: (60.0, 90.0)}
    assert "owner" in RoundThreeB(0.0, (-0.01, 0.01), 100, slow).verdict


def test_a_fusion_sets_the_constant_through_the_stage_configuration() -> None:
    base = {"plugins": {"config": {"chunker.structural": {"max_tokens": 510}}}}
    merged = fusion_overrides(base, Fusion(k=10))
    assert merged["plugins"]["config"] == {
        "chunker.structural": {"max_tokens": 510},
        "retrieval_stage.rrf": {"k": 10},
    }
    assert base["plugins"]["config"] == {"chunker.structural": {"max_tokens": 510}}


async def test_weighting_scales_the_lexical_ladder_and_nothing_else() -> None:
    document = make_document(source_id="d")
    chunks = [make_chunk(document, i, f"passage {i}") for i in range(3)]
    candidates = [
        Candidate(chunk=chunks[0], score=0.0, scores={"dense": 0.9}),
        Candidate(chunk=chunks[1], score=0.0, scores={"dense": 0.5, "lexical": 2.0}),
        Candidate(chunk=chunks[2], score=0.0, scores={"lexical": 3.0}),
    ]
    stage = RRFStage(config=FusionConfig(k=60))
    query = Query(text="q", filter=Filter(workspace_ids=frozenset({"w"})))
    equal = await stage.run(query, candidates)
    with weighting(0.5):
        halved = await stage.run(query, candidates)
    with weighting(1.0):
        assert [c.chunk.id for c in await stage.run(query, candidates)] == [
            c.chunk.id for c in equal
        ]
    fused = {c.chunk.id: c.scores["rrf"] for c in halved}
    assert fused[chunks[0].id] == pytest.approx(1 / 61), "the dense leg's first, untouched"
    assert fused[chunks[1].id] == pytest.approx(1 / 62 + 0.5 / 62)
    assert fused[chunks[2].id] == pytest.approx(0.5 / 61), "the lexical leg's first, halved"
    assert {c.chunk.id: c.scores["rrf"] for c in equal}[chunks[2].id] == pytest.approx(1 / 61)

    dense_only = [candidate for candidate in candidates if "lexical" not in candidate.scores]
    with installed() as frame, weighting(0.5):
        await stage.run(query, dense_only)
    assert DEGRADED_REASON in frame.incomparable, "a leg that gave nothing is still recorded"
    with installed() as shipped_frame:
        await stage.run(query, dense_only)
    weighted_report = frame.take_diagnostics()
    assert weighted_report["per_leg"] == {"dense": 2, "lexical": 0}
    assert weighted_report == shipped_frame.take_diagnostics(), (
        "the weighted run records the same report the shipped stage does"
    )


def test_round_four_picks_the_best_qualifier_and_breaks_ties_by_order() -> None:
    def verdict(change: float, low: float = -0.01) -> RoundFour:
        family = FamilyChange(100, change, (low, change + 0.01), (60, 10))
        return RoundFour({"prose": family, "keyword": family, "identifiers": family}, 0)

    assert round_four_winner({"k30": verdict(0.0), "k10": verdict(0.0)}) is None
    assert round_four_winner({"k30": verdict(0.03), "k10": verdict(0.034)}) == "k30"
    assert round_four_winner({"k30": verdict(0.03), "k10": verdict(0.04)}) == "k10"
    assert round_four_winner({"k30": verdict(0.05, low=-0.03)}) is None, "inadmissible"
    assert round_four_winner({"k10": verdict(0.034), "k30": verdict(0.03)}) == "k30", (
        "the tie-break follows FUSIONS, not the order the verdicts arrive in"
    )
