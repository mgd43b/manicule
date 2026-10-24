"""Round 5's instrument, executed by the suite at a size that costs nothing.

``docs/retrieval.md`` §5.4 records a decision on the strength of
:mod:`tests.benchmarks.title_lookups`, so the suite keeps it honest: the new question set asks
what it claims to, each candidate changes the one thing it names and restores it afterward, the
rewrite reaches the lexical index, and the rule picks what its docstring says it picks. Nothing
here reads a measured number.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING

import pytest

from manicule.container import keys
from manicule.plugins.registry import discover
from manicule.storage import docstore as docstore_module
from manicule.storage.fts import SEARCH_SQL
from tests.benchmarks import thread_duplication
from tests.benchmarks.lexical_questions import Asked, index_docs, run_docs
from tests.benchmarks.test_thread_duplication_benchmark import CountingEmbedder
from tests.benchmarks.title_lookups import (
    CANDIDATES,
    Candidate,
    Change,
    Verdict,
    candidate_overrides,
    choose,
    heading_questions,
    heading_weight,
    revision,
    rewrite_heading_text,
    unchanged,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_heading_is_asked_once_and_only_when_its_text_is_unique(tmp_path: Path) -> None:
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "a.md").write_text(
        "# A\n\n## 1. Tests\n\n## 2. The cache key\n\n### 2.1. Why a counter\n", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text(
        "# B\n\n## 3. Tests\n\n## Unnumbered\n\n### 1. First step\n\n### 1. First again\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluation" / "c.md").write_text("## 9. Excluded\n", encoding="utf-8")
    asked = heading_questions(tmp_path)
    assert [(a.text, a.targets) for a in asked] == [
        ("The cache key", (("a.md", "2"),)),
        ("Why a counter", (("a.md", "2.1"),)),
    ], "a repeated heading or number has no one answer, and evaluation/ is never indexed"
    assert {a.group for a in asked} == {"headings"}


def test_a_heading_weight_changes_the_statement_and_restores_it() -> None:
    with heading_weight(1.0):
        assert docstore_module.SEARCH_SQL.count("bm25(chunks_fts, 1.0, 1.0)") == 2
    assert docstore_module.SEARCH_SQL == SEARCH_SQL


def test_only_the_rerank_candidate_names_a_reranker() -> None:
    base = {"embedding": {"provider": "local"}, "rag": {"profile": "balanced"}}
    merged = candidate_overrides(base, CANDIDATES["rerank"])
    assert merged["rag"] == {"profile": "balanced", "reranker": "cross_encoder"}
    assert base["rag"] == {"profile": "balanced"}
    assert candidate_overrides(base, CANDIDATES["breadcrumb"]) == base


def test_a_rag_override_is_merged_into_the_pipeline_not_over_it(tmp_path: Path) -> None:
    settings = thread_duplication.settings_for(
        tmp_path, {"rag": {"reranker": "cross_encoder"}}, ("dense", "lexical", "rrf")
    )
    assert settings.rag.reranker == "cross_encoder"
    assert settings.rag.pipeline == ("dense", "lexical", "rrf")
    assert not settings.rag.cache.enabled


async def test_the_rewrite_puts_a_title_in_reach_of_the_lexical_leg(
    manicule_environment: Path, tmp_path: Path
) -> None:
    """The defect and its repair in miniature: a title no chunk's text or headings carry."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "glossary.md").write_text(
        "# Terms\n\nA list of words and meanings.\n", encoding="utf-8"
    )
    (root / "other.md").write_text("# Other\n\nNothing about word lists.\n", encoding="utf-8")
    found = discover()
    found.registry.bind("test").add(keys.EMBEDDER.named("local"), lambda _: CountingEmbedder())
    overrides = {"embedding": {"provider": "local"}}
    index = await index_docs(
        manicule_environment / "index", overrides=overrides, discovery=found, root=root
    )
    glossary = next(
        place.document for place in index.places.values() if place.path == "glossary.md"
    )
    asked = [Asked("title-glossary", "glossary", "titles", documents=frozenset({glossary}))]

    def lexical() -> list[int | None]:
        return [o.rank for o in outcomes]

    outcomes = await run_docs(
        index,
        asked,
        variant="or-first-16",
        pipeline="lexical",
        overrides=overrides,
        discovery=found,
    )
    assert lexical() == [None], "the title is in no indexed column"
    assert await rewrite_heading_text(index.data_dir, overrides, found) == 2
    outcomes = await run_docs(
        index,
        asked,
        variant="or-first-16",
        pipeline="lexical",
        overrides=overrides,
        discovery=found,
    )
    assert lexical() == [1]


def _verdict(*, titles: float = 0.93, worse: int = 0, low: float = -0.01) -> Verdict:
    family = Change(100, 0.0, (low, 0.01), (5, 5))
    return Verdict(
        titles=(0.95, 0.85, titles),
        titles_worse=worse,
        families={"titles": Change(19, 0.08, (0.0, 0.1), (4, 0)), "headings": family},
        unanswerable_out_of_none=0,
    )


def test_a_candidate_fixes_the_defect_only_when_every_condition_holds() -> None:
    assert _verdict().fixes
    assert _verdict(titles=0.90).fixes, "exactly half the gap is enough"
    assert not _verdict(titles=0.89).fixes, "under half the gap"
    assert not _verdict(worse=1).fixes, "a title ranked worse"
    assert not _verdict(low=-0.03).fixes, "another family's interval reaches below -0.02"
    assert _verdict(low=-0.03).families["titles"].interval[0] == 0.0


def test_the_choice_needs_every_model_and_never_picks_the_reranker() -> None:
    both = {"breadcrumb": (True, 0.0), "heading-1.0": (True, 0.004), "rerank": (True, 0.2)}
    assert choose([both, both]) == "breadcrumb", "within 0.005, the earlier candidate"
    assert choose([{"rerank": (True, 0.2)}, {"rerank": (True, 0.2)}]) is None
    one = {"breadcrumb": (False, 0.1), "heading-1.0": (True, 0.0)}
    assert choose([both, one]) == "heading-1.0", "a qualifier must fix the defect under both"
    assert choose([{"breadcrumb": (True, 0.0)}, {"breadcrumb": (False, 0.0)}]) is None
    assert next(iter(CANDIDATES)) == "shipped"
    assert CANDIDATES["shipped"] == Candidate()


def test_a_run_refuses_an_index_that_changed_schema_under_it(tmp_path: Path) -> None:
    """The first Qwen3 run of round 5 was upgraded mid-run by a migration added to the checkout."""
    with contextlib.closing(sqlite3.connect(tmp_path / "manicule.db")) as db, db:
        db.execute("CREATE TABLE alembic_version (version_num TEXT)")
        db.execute("INSERT INTO alembic_version VALUES ('before')")
    revisions = {tmp_path: revision(tmp_path)}
    unchanged(revisions)
    with contextlib.closing(sqlite3.connect(tmp_path / "manicule.db")) as db, db:
        db.execute("UPDATE alembic_version SET version_num = 'after'")
    with pytest.raises(RuntimeError, match="changed schema revision"):
        unchanged(revisions)
