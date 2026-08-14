"""Running a comparison end to end, and the four things that stop one being recorded.

The end-to-end path is exercised against real pipelines rather than doubles, because the
question "does a comparison of two manicule configurations work" is not answerable by two
objects that agree to return lists.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, override

import pytest

from manicule.evaluation.corpus import CorpusVersion, corpus_version_of
from manicule.evaluation.errors import (
    AtChanceError,
    ConfigurationDriftError,
    CorpusMismatchError,
)
from manicule.evaluation.harness import PreferenceHarness
from manicule.evaluation.judging import ScriptedJudge, SlotJudge, StreamJudge
from manicule.evaluation.preference import Preference, PreferenceStore, Slot
from manicule.evaluation.probe import DiscriminationProbe, ProbeItem, probe_from_titles
from manicule.evaluation.queries import EvalQuery, Intent, Provenance, QuerySet
from manicule.evaluation.report import build_report
from manicule.evaluation.systems import RetrieverSystem, SystemResult
from tests.evaluation.fakes import (
    BagOfWordsEmbedder,
    FixedSystem,
    LookupSystem,
    MeaninglessEmbedder,
    an_item,
)
from tests.evaluation.pipeline import SCOPE, build_corpus, dense_only_retriever

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.evaluation.systems import SystemUnderComparison
    from manicule.storage.docstore import SqliteDocStore

QUERIES = QuerySet(
    name="fixture set",
    provenance=Provenance.AUTHORED,
    queries=(
        EvalQuery(id="q1", text="aurora ledger configuration", intent=Intent.LOOKUP),
        EvalQuery(id="q2", text="basalt gateway failure handling", intent=Intent.HOW_DOES_X_WORK),
        EvalQuery(id="q3", text="cinder scheduler capacity limits", intent=Intent.LOOKUP),
        EvalQuery(id="q4", text="delta warehouse configuration", intent=Intent.EXACT_IDENTIFIER),
    ),
)


def working_systems(count: int = 40) -> tuple[SystemUnderComparison, SystemUnderComparison]:
    """Two systems that both clear the probe, differing only in their labels."""
    answers = {f"question {i}": [an_item(f"d{i}")] for i in range(count)}
    version = CorpusVersion(label="fixture", digest="sha256:aaa", document_count=400)
    return (
        LookupSystem(answers, config_label="alpha", corpus_version=version),
        LookupSystem(answers, config_label="beta", corpus_version=version),
    )


def a_probe(count: int = 40) -> DiscriminationProbe:
    return DiscriminationProbe(
        [ProbeItem(text=f"question {i}", document_ids=frozenset({f"d{i}"})) for i in range(count)],
        k=3,
    )


async def test_two_manicule_configurations_are_compared_end_to_end(
    store: SqliteDocStore, tmp_path: Path
) -> None:
    """The thing this package is for, through the shipped retriever and a real store."""
    chunks = await build_corpus(store)
    version = await corpus_version_of(store, label="fixture", workspace_ids=SCOPE)
    sides = [
        RetrieverSystem(
            await dense_only_retriever(store, BagOfWordsEmbedder(), chunks),
            config_label=label,
            corpus_version=version,
            workspace_ids=SCOPE,
        )
        for label in ("candidate", "baseline")
    ]
    harness = PreferenceHarness(
        left=sides[0],
        right=sides[1],
        probe=DiscriminationProbe(
            await probe_from_titles(store, workspace_ids=SCOPE, limit=24), k=3
        ),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )
    judge = ScriptedJudge({"q1": Preference.LEFT, "q2": Preference.RIGHT, "q3": Preference.TIE})

    records = await harness.compare(QUERIES, judge)

    assert [record.query_id for record in records] == ["q1", "q2", "q3"]
    assert judge.seen == ["q1", "q2", "q3", "q4"], "q4 was offered and skipped, not withheld"
    assert all(record.admissible for record in records)
    assert all(record.left.items for record in records)
    assert records[0].left_probe.discriminates
    # Each record is self-contained: the configuration that produced each side travels with it.
    assert records[0].left.configuration["stages"] == ["dense"]

    report = build_report(records)
    assert report.is_evidence
    assert {row.label for row in report.by_intent} == {"lookup", "how_does_x_work"}


async def test_a_side_that_retrieves_at_chance_stops_the_run_before_anything_is_recorded(
    store: SqliteDocStore, tmp_path: Path
) -> None:
    """The first of the three places the rule is enforced, and the one that matters most.

    Nothing is judged, nothing is written, and the file does not exist. A harness that recorded
    first and filtered later would leave judgments about noise on disk, indistinguishable from
    real ones to anything that read the file afterwards.
    """
    chunks = await build_corpus(store)
    version = await corpus_version_of(store, label="fixture", workspace_ids=SCOPE)
    path = tmp_path / "preferences.jsonl"
    harness = PreferenceHarness(
        left=RetrieverSystem(
            await dense_only_retriever(store, BagOfWordsEmbedder(), chunks),
            config_label="candidate",
            corpus_version=version,
            workspace_ids=SCOPE,
        ),
        right=RetrieverSystem(
            await dense_only_retriever(store, MeaninglessEmbedder(), chunks),
            config_label="no-semantics",
            corpus_version=version,
            workspace_ids=SCOPE,
        ),
        probe=DiscriminationProbe(
            await probe_from_titles(store, workspace_ids=SCOPE, limit=24), k=3
        ),
        store=PreferenceStore(path),
    )
    judge = ScriptedJudge({"q1": Preference.LEFT})

    with pytest.raises(AtChanceError, match="no-semantics"):
        await harness.compare(QUERIES, judge)

    assert judge.seen == [], "no query should have been shown to a judge"
    assert not path.exists(), "nothing may be written for a run that was refused"


async def test_two_sides_pointed_at_different_corpora_are_refused(tmp_path: Path) -> None:
    """Comparing across different content is a measurement of the content."""
    left, _ = working_systems()
    right = LookupSystem(
        {f"question {i}": [an_item(f"d{i}")] for i in range(40)},
        config_label="beta",
        corpus_version=CorpusVersion(label="a-different-corpus", document_count=400),
    )
    harness = PreferenceHarness(
        left=left,
        right=right,
        probe=a_probe(),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )

    with pytest.raises(CorpusMismatchError, match="corpus labels differ"):
        await harness.certify()


async def test_a_side_that_changes_configuration_mid_run_stops_the_run(
    tmp_path: Path,
) -> None:
    """Half the records would name a pipeline that was not running when they were made."""

    class DriftingSystem(LookupSystem):
        """Reports one configuration, then another. Nothing else about it changes."""

        @override
        async def search(self, text: str, *, limit: int) -> SystemResult:
            result = await super().search(text, limit=limit)
            self._configuration = {"kind": f"changed-{len(self.asked)}"}
            return result

    left, _ = working_systems()
    right = DriftingSystem(
        {query.text: [an_item("d0")] for query in QUERIES.queries}
        | {f"question {i}": [an_item(f"d{i}")] for i in range(40)},
        config_label="beta",
        corpus_version=CorpusVersion(label="fixture", digest="sha256:aaa", document_count=400),
    )
    harness = PreferenceHarness(
        left=left,
        right=right,
        probe=a_probe(),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )

    with pytest.raises(ConfigurationDriftError, match="changed configuration mid-run"):
        await harness.compare(
            QUERIES, ScriptedJudge({q.id: Preference.LEFT for q in QUERIES.queries})
        )


async def test_both_sides_may_not_share_a_label(tmp_path: Path) -> None:
    """A file in which the winner cannot be identified is a file of nothing."""
    left, _ = working_systems()
    same = LookupSystem(
        {f"question {i}": [an_item(f"d{i}")] for i in range(40)},
        config_label="alpha",
        corpus_version=CorpusVersion(label="fixture", digest="sha256:aaa", document_count=400),
    )

    with pytest.raises(ValueError, match="labeled"):
        PreferenceHarness(
            left=left,
            right=same,
            probe=a_probe(),
            store=PreferenceStore(tmp_path / "preferences.jsonl"),
        )


async def test_a_judge_with_a_pure_position_bias_cannot_produce_a_win(tmp_path: Path) -> None:
    """What blinding buys, demonstrated rather than asserted.

    Slot assignment is a keyed hash of the query id, so a judge who always picks the first list
    splits its choices between the two systems. Without blinding this judge would report 100%
    for whichever side was passed as ``left`` — a clean, significant, entirely false result,
    and a number a harness must never be able to produce.
    """
    left, right = working_systems()
    queries = QuerySet(
        name="many",
        provenance=Provenance.AUTHORED,
        queries=tuple(EvalQuery(id=f"p{i}", text=f"question {i}") for i in range(40)),
    )
    harness = PreferenceHarness(
        left=left,
        right=right,
        probe=a_probe(),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )

    records = await harness.compare(queries, SlotJudge(Slot.A))

    report = build_report(records)
    assert report.overall.decided == 40
    low, high = report.overall.interval
    assert low < 0.5 < high, "a position-biased judge must not separate the two systems"
    assert not report.overall.separates


async def test_a_person_judges_through_two_text_streams(tmp_path: Path) -> None:
    """The interactive path, driven by a string so it is exercised rather than assumed."""
    left, right = working_systems()
    queries = QuerySet(
        name="four",
        provenance=Provenance.AUTHORED,
        queries=tuple(EvalQuery(id=f"p{i}", text=f"question {i}") for i in range(4)),
    )
    output = io.StringIO()
    harness = PreferenceHarness(
        left=left,
        right=right,
        probe=a_probe(),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )

    records = await harness.compare(
        queries,
        StreamJudge(output=output, input_stream=io.StringIO("a\nz\nt\ns\nq\n")),
    )

    rendered = output.getvalue()
    assert "unrecognized choice 'z'" in rendered
    assert "alpha" not in rendered, "a judge must not see either configuration label"
    assert "beta" not in rendered, "a judge must not see either configuration label"
    # a, then a retry, then a tie, then a skip, then quit: two judgments recorded.
    assert [record.query_id for record in records] == ["p0", "p1"]
    assert records[1].preference is Preference.TIE


async def test_a_system_that_returns_nothing_useful_is_refused_before_judging(
    tmp_path: Path,
) -> None:
    """The crudest failure, through the whole harness rather than through the probe alone."""
    left, _ = working_systems()
    harness = PreferenceHarness(
        left=left,
        right=FixedSystem(
            [an_item("always-this-one")],
            config_label="fixed",
            corpus_version=CorpusVersion(label="fixture", digest="sha256:aaa", document_count=400),
        ),
        probe=a_probe(),
        store=PreferenceStore(tmp_path / "preferences.jsonl"),
    )

    with pytest.raises(AtChanceError, match="fixed"):
        await harness.certify()
