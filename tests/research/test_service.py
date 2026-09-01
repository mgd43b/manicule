"""What ``research`` reports, through the application service the surfaces call.

The loop's own suite drives it directly; this one is about the composition — that the evidence
becomes an ordinary answer, that the tenancy check runs over the whole accumulated ledger rather
than one retrieval's context, and that the numbers on the payload are the run's own rather than
plausible constants.
"""

from __future__ import annotations

import pytest

from manicule.app.service import ApplicationService
from manicule.app.tenancy import CrossWorkspaceError
from manicule.core.errors import ConfigError
from manicule.core.retrieval import Candidate
from tests.api.support import backend_with_a_document
from tests.app.fakes import FakeBackend, make_chunk

PLAN_TWO = '{"queries": [{"q": "retry policy", "why": "a"}, {"q": "backoff", "why": "b"}]}'
NO_MORE = '{"queries": []}'


def service(*, replies: list[str] | None = None) -> tuple[ApplicationService, FakeBackend]:
    """The service over a backend holding one real, workspace-owned document.

    Derived ids rather than literals, so a tenancy assertion cannot be made to pass by editing
    a string until it matches.
    """
    backend, _ = backend_with_a_document()
    backend.generator_.replies = list(replies or [PLAN_TWO, NO_MORE])
    return ApplicationService(backend), backend


def owned(backend: FakeBackend, count: int) -> list[Candidate]:
    """``count`` candidates over the document this backend actually holds."""
    document = next(iter(backend.organization_.documents.values()))
    return [
        Candidate(chunk=make_chunk(document, text=f"passage {index}"), score=0.9 - index / 100)
        for index in range(count)
    ]


async def test_the_report_names_the_searches_it_actually_ran() -> None:
    """A reader deciding whether to trust a multi-search report needs to see which angles it
    took. A report that missed the obvious question is only visible if the questions it did ask
    are on the page."""
    api, _ = service()

    report = await api.research("how do retries work?")

    assert [step.question for step in report.sub_questions] == ["retry policy", "backoff"]
    assert report.planned == 2
    assert report.model_planned is True


async def test_a_run_that_could_not_plan_says_so_on_the_payload() -> None:
    api, _ = service(replies=["not json", NO_MORE])

    report = await api.research("how do retries work?")

    assert report.model_planned is False


async def test_the_bounds_are_reported_alongside_what_was_spent() -> None:
    """A caller cannot tell a run that finished from one that hit a ceiling unless both numbers
    are on the payload."""
    api, _ = service()

    report = await api.research("q")

    assert report.cycles_allowed == api.settings.research.max_cycles
    assert report.cycles_run >= 1
    # Exact, not `>= 1`: the field is documented as the loop's own planning calls, counted so
    # that it and the answer never double-count one model call — and `>= 1` holds for any
    # non-zero constant, including one that silently included the report's own generation.
    assert report.model_calls == 2
    assert "no further searches" in report.stopped_early


async def test_a_passage_two_searches_found_is_counted_as_corroborated() -> None:
    """Reported on its own rather than blended into confidence: it measures agreement between
    searches, and confidence measures one search's ranking."""
    api, backend = service()
    backend.retriever_.candidates = owned(backend, 2)

    report = await api.research("q")

    assert report.passages_found == 2
    assert report.corroborated == 2


async def test_the_corroboration_count_is_the_runs_own_number() -> None:
    """It was briefly a dict of zeros, which is the failure this asserts against: a count that
    is always zero and a corpus where nothing corroborates look identical on the payload."""
    api, backend = service(replies=['{"queries": [{"q": "only one"}]}', NO_MORE])
    backend.retriever_.candidates = owned(backend, 1)

    report = await api.research("q")

    assert report.passages_found == 1
    assert report.corroborated == 0


async def test_a_passage_from_another_workspace_stops_the_report_before_the_model() -> None:
    """Several searches is several chances to surface another tenant's row, and the check runs
    over the accumulated ledger rather than each retrieval separately — the union is what
    reaches the model."""
    api, backend = service()
    stranger = make_chunk(next(iter(backend.organization_.documents.values())))
    backend.retriever_.candidates = [
        Candidate(chunk=stranger.model_copy(update={"document_id": "not-ours"}), score=0.9)
    ]

    with pytest.raises(CrossWorkspaceError):
        await api.research("q")


async def test_a_generator_without_a_prompt_channel_is_refused_by_the_service() -> None:
    """The refusal reaches the surface as a configuration error rather than as a run that
    silently searched once."""

    class Promptless:
        model_id = "protocol-only/model"
        context_window = 32768

        def generate(self, query: object, context: object) -> object:  # pragma: no cover
            raise AssertionError

    api, backend = service()
    backend.generator_ = Promptless()  # pyright: ignore[reportAttributeAccessIssue] - a fake

    with pytest.raises(ConfigError, match="prepared prompt"):
        await api.research("q")


async def test_a_report_budget_wider_than_the_window_is_refused_before_any_search() -> None:
    api, backend = service()
    backend.generator_.context_window = 2048

    with pytest.raises(ConfigError, match=r"research\.report_tokens"):
        await api.research("q")

    assert backend.retriever_.seen == [], "a refused configuration still spent a retrieval"


async def test_the_view_hook_cannot_change_the_report() -> None:
    """The same contract ``ask`` makes: the payload is identical whether or not one is passed."""
    api, _ = service()
    seen: list[object] = []

    with_hook = await api.research("q", on_event=seen.append)
    api2, _ = service()
    without = await api2.research("q")

    assert seen
    assert with_hook.text == without.text
    assert with_hook.citations == without.citations


async def test_the_report_widens_the_profile_the_query_actually_runs_under() -> None:
    """Not the configured default, which is a different profile the moment a caller passes one.

    ``Profiles.for_query`` resolves the report's overrides against ``query.profile``, so
    computing them from ``settings.rag.profile`` disagrees with what they are applied to — and
    it disagreed downward. The widening is a ``max`` whose job is to never *reduce* what the
    profile already allows, and computed from the wrong base it reduces exactly that: a
    ``precise`` run, whose ``final_top_k`` is 10, was widened against ``balanced``'s 5 and read
    five passages instead of ten.

    Visible only where the research budget sits below the profile's own, which is why it is
    configured that way here rather than left to the defaults, where the two maxima coincide
    and the defect is invisible.
    """
    backend, document = backend_with_a_document(research={"report_passages": 3})
    backend.generator_.replies = ['{"queries": [{"q": "one"}]}', NO_MORE]
    backend.retriever_.candidates = [
        Candidate(chunk=make_chunk(document, text=f"passage {index}"), score=0.9 - index / 100)
        for index in range(12)
    ]
    api = ApplicationService(backend)

    report = await api.research("q", profile="precise")

    assert report.passages_cited == 10, (
        "the report read the passages a `balanced` run would, not the `precise` one asked for"
    )


async def test_a_run_whose_searches_found_nothing_still_consulted_the_corpus() -> None:
    """'We did not look' and 'we looked and there is nothing' are different claims, and `ask`
    keeps them apart. Reporting the first when the second happened tells a reader the report
    could not have had citations, when in fact the corpus simply does not cover the question."""
    api, backend = service()
    backend.retriever_.candidates = []

    report = await api.research("q")

    assert report.passages_found == 0
    assert report.corpus_consulted is True


async def test_a_report_does_not_claim_the_glossary_matched_nothing() -> None:
    """The payload used to extend `Glossed` and populate none of it, so every report positively
    asserted that no term fired — including runs whose sub-questions were expanded. A run has
    several retrievals and therefore several glossary stories; publishing one empty triple is a
    claim it cannot support."""
    api, _ = service()

    report = await api.research("q")

    assert not hasattr(report, "expansions")
    assert not hasattr(report, "explicit_definition")
