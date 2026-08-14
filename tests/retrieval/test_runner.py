"""The runner: it times, it counts, it records — and it never changes what a stage returns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.errors import ConfigError
from manicule.core.retrieval import Candidate
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.runner import PipelineRunner, require_unique_names
from manicule.retrieval.trace import current_frame, installed

if TYPE_CHECKING:
    from collections.abc import Callable

    from manicule.core.protocols import RetrievalStage
from manicule.storage.docstore import SqliteDocStore
from tests.retrieval.fakes import (
    FrameReadingStage,
    ListVectorStore,
    RecordingStage,
    a_query,
    profiles,
)
from tests.storage_helpers import make_chunk, make_document

DOCUMENT = make_document()


def candidate(position: int, **scores: float) -> Candidate:
    return Candidate(
        chunk=make_chunk(DOCUMENT, position, f"passage {position}"),
        score=1.0,
        scores=dict(scores),
    )


async def test_the_runner_times_every_stage_including_ones_that_never_thought_about_it() -> None:
    """A self-reported number is unverifiable and optional.

    A third-party stage whose author never instrumented it gets timed identically, a stage
    cannot under-report by measuring only the part it considers its own work, and there is
    nothing to check a self-reported figure against.
    """
    slow = RecordingStage("slow", [candidate(0)], delay=0.01)
    runner = PipelineRunner([slow])

    run = await runner.run(a_query())

    assert [span.name for span in run.spans] == ["slow"]
    assert run.spans[0].wall_ms >= 5.0
    assert run.wall_ms >= run.spans[0].wall_ms


async def test_the_runner_counts_what_went_in_and_what_came_out() -> None:
    """Attribution is only possible if you can see which stage's output changed."""
    first = RecordingStage("first", [candidate(0), candidate(1)])
    second = RecordingStage("second", [candidate(2)])

    run = await PipelineRunner([first, second]).run(a_query())

    assert [(span.candidates_in, span.candidates_out) for span in run.spans] == [(0, 2), (2, 3)]


async def test_the_runner_records_each_stage_s_declared_configuration(
    store: SqliteDocStore,
) -> None:
    """A recorded result has to name the thing it measured."""
    from manicule.retrieval.dense import DenseStage  # noqa: PLC0415
    from tests.fakes import HashEmbedder  # noqa: PLC0415

    stage = DenseStage(
        embedder=HashEmbedder(),
        vectors=ListVectorStore([]),
        docstore=store,
        profiles=profiles(),
    )

    run = await PipelineRunner([stage]).run(a_query())

    assert run.spans[0].config["overfetch_min"] == 3


async def test_two_stages_sharing_a_name_are_refused() -> None:
    """Scores are keyed by stage name, so the second silently overwrites the first's record.

    Fusion is then computed from a ladder missing half its rungs, which produces a plausible
    ordering and raises nothing at all.
    """
    twin = [RecordingStage("dense", []), RecordingStage("dense", [])]
    with pytest.raises(ConfigError, match="more than once"):
        require_unique_names(twin)
    with pytest.raises(ConfigError, match="more than once"):
        PipelineRunner(twin)


async def test_a_stage_that_reads_the_frame_to_decide_what_to_do_is_visible() -> None:
    """The defect the rule below is aimed at, shown to be detectable before it is ruled out.

    Nothing in the pipeline's behavior may depend on whether anyone is recording. A stage that
    returns different candidates when observed is not a stage anybody can measure, and a check
    that has never seen one fail is not evidence that none can.
    """
    reading = FrameReadingStage()
    given = [candidate(0)]

    assert current_frame() is None
    unobserved = await reading.run(a_query(), given)
    with installed():
        observed = await reading.run(a_query(), given)

    assert unobserved != observed


@pytest.mark.parametrize("build", [lambda: RecordingStage("s", [candidate(0)]), RRFStage])
async def test_the_shipped_stages_are_indifferent_to_being_recorded(
    build: Callable[[], RetrievalStage],
) -> None:
    """The rule that pays down the trace frame's implicit coupling.

    Each stage is run twice over identical input, once with a frame installed and once without,
    and must produce the same candidates. A stage whose diagnostics changed its output would
    make every recorded measurement a measurement of the recording.
    """
    stage = build()
    given = [candidate(0, dense=0.9), candidate(1, lexical=2.0)]

    unobserved = await stage.run(a_query(), list(given))
    with installed():
        observed = await stage.run(a_query(), list(given))

    assert unobserved == observed


async def test_the_runner_carries_a_stage_s_diagnostics_without_widening_its_signature() -> None:
    """The whole reason the frame exists.

    Widening the return to ``(candidates, report)`` would make every recorded result
    unreplayable; per-run state on a stage would be a race, since stages are singletons shared
    across concurrent queries.
    """

    class Chatty:
        name = "chatty"

        async def run(self, query: object, candidates: list[Candidate]) -> list[Candidate]:
            del query
            from manicule.retrieval.trace import record  # noqa: PLC0415

            record(fetched=41, note="over-fetched once")
            return list(candidates)

    run = await PipelineRunner([Chatty()]).run(a_query())

    assert run.diagnostics("chatty") == {"fetched": 41, "note": "over-fetched once"}


async def test_concurrent_runs_do_not_swap_diagnostics() -> None:
    """Two queries' numbers attached to the wrong runs is worse than no numbers at all."""
    import asyncio  # noqa: PLC0415

    class Tagging:
        def __init__(self, name: str, tag: int) -> None:
            self.name = name
            self._tag = tag

        async def run(self, query: object, candidates: list[Candidate]) -> list[Candidate]:
            del query
            from manicule.retrieval.trace import record  # noqa: PLC0415

            record(tag=self._tag)
            await asyncio.sleep(0.01)
            record(tag_again=self._tag)
            return list(candidates)

    first, second = await asyncio.gather(
        PipelineRunner([Tagging("s", 1)]).run(a_query()),
        PipelineRunner([Tagging("s", 2)]).run(a_query()),
    )

    assert first.diagnostics("s") == {"tag": 1, "tag_again": 1}
    assert second.diagnostics("s") == {"tag": 2, "tag_again": 2}


async def test_the_runtime_scope_assertion_checks_what_was_actually_served(
    store: SqliteDocStore,
) -> None:
    """The opt-in form of the check that makes the vector store's exemption safe.

    It replays each stage's recorded output rather than re-running the pipeline, so what gets
    checked is the ranking the caller is about to receive, not a second run that may differ.
    ``expect_results=False``: at runtime an empty result is an ordinary outcome — the corpus
    genuinely had nothing — where in a fixture it means the check has seen nothing at all.
    """
    document = make_document(source_id="live")
    await store.upsert_document(document)
    chunk = make_chunk(document, 0, "authentication live")
    await store.replace_chunks(document.id, [chunk])
    good = RecordingStage("dense", [Candidate(chunk=chunk, score=1.0, scores={"dense": 1.0})])

    run = await PipelineRunner([good], docstore=store, assert_scope=True).run(a_query())

    assert len(run.candidates) == 1


async def test_the_runtime_scope_assertion_fails_the_query_on_a_leak(
    store: SqliteDocStore,
) -> None:
    """A scoped search that returned another tenant's chunk has already failed.

    Returning it with a warning attached is precisely the failure the check exists to prevent,
    so it is fatal to the query rather than advisory.
    """
    foreign_document = make_document(source_id="theirs", workspace_id="beta")
    leaked = make_chunk(foreign_document, 0, "authentication theirs")
    leaky = RecordingStage("dense", [Candidate(chunk=leaked, score=1.0, scores={"dense": 1.0})])

    with pytest.raises(AssertionError, match="another workspace"):
        await PipelineRunner([leaky], docstore=store, assert_scope=True).run(a_query())


async def test_an_empty_result_is_not_a_runtime_failure(store: SqliteDocStore) -> None:
    """The parameter that separates the runtime use from the fixture one."""
    run = await PipelineRunner(
        [RecordingStage("dense", [])], docstore=store, assert_scope=True
    ).run(a_query())
    assert run.candidates == []


def test_the_scope_assertion_needs_a_store_to_ask() -> None:
    """Visibility is decided by the store, because that is what retrieval consults."""
    with pytest.raises(ConfigError, match="no document store"):
        PipelineRunner([RecordingStage("dense", [])], assert_scope=True)


async def test_the_scope_assertion_is_off_by_default(store: SqliteDocStore) -> None:
    """It costs a document lookup per candidate per stage, so it is opt-in."""
    foreign_document = make_document(source_id="theirs", workspace_id="beta")
    leaked = make_chunk(foreign_document, 0, "authentication theirs")
    leaky = RecordingStage("dense", [Candidate(chunk=leaked, score=1.0, scores={"dense": 1.0})])

    run = await PipelineRunner([leaky], docstore=store).run(a_query())

    assert len(run.candidates) == 1
