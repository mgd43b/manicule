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

from manicule.container import keys
from manicule.plugins.registry import discover
from manicule.storage.fts import escape_match_query
from tests.benchmarks.lexical_questions import (
    AUTHORED,
    DOCS,
    MULTI_HOP_TARGETS,
    VARIANTS,
    Asked,
    Place,
    _rank,  # pyright: ignore[reportPrivateUsage]
    index_docs,
    run_docs,
    summarize,
    time_at_scale,
)
from tests.benchmarks.test_thread_duplication_benchmark import CountingEmbedder

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


def test_the_shipped_rule_is_one_of_the_measured_ones() -> None:
    """The report compares candidates; it has to contain the one that shipped."""
    for question in ("how do I rotate the signing token", "NOT", "what is it", "chunks_fts"):
        assert escape_match_query(question) in {variant(question) for variant in VARIANTS.values()}


def test_scoring_reads_the_rank_of_the_first_labeled_chunk() -> None:
    asked = Asked("q", "?", "authored", (("retrieval.md", "6"),))
    places = [
        Place("storage.md", ("6. Derived",), "a"),
        Place("retrieval.md", ("6. Rerank", "6.1 A cross-encoder"), "b"),
    ]
    assert _rank(asked, places) == (2, 1)
    assert _rank(asked, places[:1]) == (None, 0)
    assert _rank(Asked("t", "?", "titles", documents=frozenset({"a"})), places) == (1, 1)


def test_the_statement_is_timed_for_every_variant() -> None:
    texts = [("authentication and token rotation", "Auth"), ("vector index builds", "Index")]
    timings = time_at_scale(texts, ["how is a token rotated", "zzz"], size=200)
    assert set(timings) == set(VARIANTS)
    assert timings["and"][2] == 0, "no chunk holds every word of either question"
    assert timings["or-content"][2] > 0


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
