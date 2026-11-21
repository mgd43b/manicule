"""The lexical plan benchmark's shape, at a size that runs in seconds.

What the benchmark measures is in its docstring; what these pin is that it measures it: that
the corpus has the documents each shape names, that every statement it times returns the same
ranking, and that the table it prints says what the chooser would do.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from manicule.storage.fts import LexicalPlan
from tests.benchmarks.lexical_plans import (
    FIRST,
    QUERIES,
    RANKED,
    TODAY,
    VARIANTS,
    Corpus,
    Shape,
    build_corpus,
    child,
    connect,
    equivalent,
    plan_of,
    render,
    scopes,
    shapes,
    sorts,
    statement_for,
    warm,
)

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    return build_corpus(tmp_path_factory.mktemp("lexical-plans"), 10_000)


def _shape(corpus: Corpus, query: str, scope: str) -> Shape:
    workspace, restriction = scopes(corpus.bulk_documents)[scope]
    return Shape(f"{query} @ {scope}", QUERIES[query], restriction, workspace)


def test_the_corpus_holds_what_the_shapes_ask_for(corpus: Corpus) -> None:
    db = sqlite3.connect(corpus.path)
    try:

        def count(sql: str) -> int:
            return int(db.execute(sql).fetchone()[0])

        assert count("SELECT count(*) FROM chunks") == corpus.chunks > 9_000
        assert count("SELECT count(*) FROM documents WHERE id LIKE 'fam8-%'") == 200
        assert count("SELECT count(*) FROM documents WHERE id LIKE 'fam5-%'") == 16
        assert count("SELECT count(*) FROM documents WHERE id LIKE 'fam9-%'") == 64
        assert count("SELECT count(DISTINCT workspace_id) FROM documents") == 3
        assert count("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'zr001'") >= 2
        # The dilution the joined statement exists to survive: documents a search must skip.
        assert count("SELECT count(*) FROM documents WHERE deleted_at IS NOT NULL") > 0
        assert count("SELECT count(*) FROM documents WHERE status <> 'indexed'") > 0
        # Every source and type a selective shape asks for exists, at any size.
        for column, wanted in (("source", 5), ("media_type", 4)):
            assert count(f"SELECT count(DISTINCT {column}) FROM documents") == wanted  # noqa: S608
        assert count("SELECT count(*) FROM source_reader_grants") == 4
    finally:
        db.close()


def test_every_scope_names_as_many_documents_as_it_claims(corpus: Corpus) -> None:
    named = scopes(corpus.bulk_documents)
    sizes = {
        scope: len(restriction.document_ids) if restriction is not None else 0
        for scope, (_, restriction) in named.items()
    }

    assert sizes["ids 4x5=20"] == 4
    assert sizes["ids 64x8=512"] == 64
    assert sizes["ids 65x8=520"] == 65
    assert sizes["ids 200x8"] == 200
    assert sizes["no filter"] == 0
    assert len({shape.name for shape in shapes(corpus.bulk_documents)}) == len(
        shapes(corpus.bulk_documents)
    ), "two shapes share a name, so one row of the table would hide the other"


@pytest.mark.parametrize(
    ("query", "scope"),
    [
        ("common-3", "no filter"),
        ("common-3", "source 5%"),
        ("common-3", "readers 5%"),
        ("common-3", "ids 16x8=128"),
        ("common-3", "ids 64x9=576"),
        ("mid-3", "time 5%"),
        ("rare-2", "ids 4x5=20"),
        ("common+rare", "small workspace"),
        ("no-match", "kind 5%"),
        ("single-common", "source 5%"),
    ],
)
def test_every_statement_timed_returns_the_same_ranking(
    corpus: Corpus, query: str, scope: str
) -> None:
    """The table's ``agree`` column, asserted: a faster statement that ranked differently would
    be a different search, and a benchmark that timed it would be recommending one."""
    db = connect(corpus.path)
    try:
        measured = warm(db, _shape(corpus, query, scope), VARIANTS, reps=1, cap=60)
    finally:
        db.close()

    assert measured.agree
    assert all(milliseconds is not None for milliseconds in measured.times.values())


@pytest.mark.parametrize(
    ("query", "scope", "expected"),
    [
        ("common-3", "no filter", LexicalPlan.RANKED),
        ("common-3", "source 5%", LexicalPlan.DIRECT),
        ("single-common", "no filter", LexicalPlan.DIRECT),
        ("paste-16", "no filter", LexicalPlan.DIRECT),
        ("prose-6", "readers 5%", LexicalPlan.RANKED),
        ("common-3", "ids 4x5=20", LexicalPlan.DOCUMENT_FIRST),
        ("common-3", "ids 200x8", LexicalPlan.DIRECT),
    ],
)
def test_the_table_reports_the_statement_the_store_would_choose(
    corpus: Corpus, query: str, scope: str, expected: LexicalPlan
) -> None:
    db = connect(corpus.path)
    try:
        measured = warm(db, _shape(corpus, query, scope), VARIANTS[:2], reps=1, cap=60)
    finally:
        db.close()

    assert measured.chosen.plan is expected


def test_the_plans_it_prints_are_the_ones_that_sort_and_the_one_that_does_not(
    corpus: Corpus,
) -> None:
    shape = _shape(corpus, "common-3", "source 5%")
    db = connect(corpus.path)
    try:
        assert not sorts(plan_of(db, RANKED, shape))
        assert sorts(plan_of(db, TODAY, shape))
        assert sorts(plan_of(db, FIRST, _shape(corpus, "common-3", "ids 4x5=20")))
    finally:
        db.close()


def test_today_is_the_statement_as_released_and_the_others_are_not(corpus: Corpus) -> None:
    shape = _shape(corpus, "common-3", "ids 16x8=128")

    today, _ = statement_for(TODAY, shape)
    anchored, _ = statement_for(VARIANTS[1], shape)
    first, _ = statement_for(FIRST, shape)

    assert "AND d.id IN" in today
    assert "AND +d.id IN" in anchored
    assert "AND d.id IN" in first
    assert "ORDER BY chunks_fts.rank" in statement_for(RANKED, shape)[0]


def test_a_cold_run_prints_one_number_for_a_fresh_connection(
    corpus: Corpus, capsys: pytest.CaptureFixture[str]
) -> None:
    import json  # noqa: PLC0415 - only this test builds the spec by hand

    from tests.benchmarks.lexical_plans import encode_shape  # noqa: PLC0415

    shape = _shape(corpus, "common-3", "source 5%")
    spec = json.dumps({"path": str(corpus.path), "shape": encode_shape(shape), "variant": "ranked"})

    child(spec, 60)

    assert float(capsys.readouterr().out.strip()) > 0


def test_ties_are_compared_as_sets_and_scores_exactly() -> None:
    base = [("a", -3.0), ("b", -2.0), ("c", -2.0), ("d", -1.0)]

    assert equivalent(base, [("a", -3.0), ("c", -2.0), ("b", -2.0), ("e", -1.0)]), (
        "the last score's members may be any of the tied, and equal scores may swap places"
    )
    assert not equivalent(base, [("a", -3.0), ("b", -2.0), ("x", -2.0), ("d", -1.0)])
    assert not equivalent(base, [("a", -3.0), ("b", -2.0), ("c", -2.0), ("d", -1.0000001)])
    assert not equivalent(base, base[:3])
    assert equivalent([], [])


def test_the_report_has_a_column_per_statement_and_flags_a_disagreement(
    corpus: Corpus,
) -> None:
    db = connect(corpus.path)
    try:
        measured = warm(db, _shape(corpus, "common-3", "no filter"), VARIANTS[:3], reps=1, cap=60)
    finally:
        db.close()

    table = render([measured], cold_times={})

    assert table.splitlines()[0].startswith(
        "| shape | rows | today | direct | ranked | doc-first |"
    )
    assert "| ranked | yes |" in table
    measured.agree = False
    assert "| NO |" in render([measured], cold_times={})
