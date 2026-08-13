"""What the operations mean, checked once, where they are implemented.

Both surfaces call this class, so everything asserted here is asserted about both of them.
That is the whole reason the layer exists, and it is why these tests do not go through Typer
or FastMCP: driving a rule through an adapter tests the adapter.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import huggingface_hub
import pytest

from manicule import vocabularies
from manicule.app import results as r
from manicule.app.results import CheckState
from manicule.app.service import ApplicationService, hardware
from manicule.config.settings import Settings, config_file
from manicule.core.content import Document, DocumentStatus
from manicule.core.errors import ConfigError, UnknownEntityError
from manicule.core.ids import document_id
from manicule.core.provenance import Provenance
from manicule.core.retrieval import Candidate, RetrievalProfile
from manicule.embedding.runtimes import hub
from manicule.embedding.runtimes.hub import OFFLINE_ENV
from manicule.ingest.pipeline import RunReport
from manicule.plugins.registry import discover
from tests.app.fakes import FakeBackend, make_chunk, make_document


@pytest.fixture
def backend() -> FakeBackend:
    made = FakeBackend()
    document = made.store.add(make_document(made.workspace))
    made.store.chunks[document.id] = [make_chunk(document)]
    return made


@pytest.fixture
def service(backend: FakeBackend) -> ApplicationService:
    return ApplicationService(backend)


@pytest.fixture
def config_home(manicule_environment: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config file location inside the test's own directory."""
    path = manicule_environment / "config.toml"
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))
    return path


# --- search and ask ------------------------------------------------------------------------


async def test_a_search_carries_the_workspace_into_the_filter(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """The scope is on the query before a stage ever runs.

    ``Filter`` has no default workspace and cannot be constructed without one, so this is
    checking that the service passes the right one rather than that it passes any.
    """
    await service.search("retry")
    assert backend.retriever_.seen[0].filter.workspace_ids == frozenset({backend.workspace})


async def test_a_search_hit_carries_its_document_s_uri_and_title(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """A hit with no location is a hit nobody can follow."""
    document = next(iter(backend.store.documents.values()))
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(document), score=0.9)]
    found = await service.search("retry")
    assert found.hits[0].uri == document.uri
    assert found.hits[0].title == document.title
    assert found.hits[0].anchor["kind"] == "heading"


async def test_an_unknown_profile_is_refused_by_name(service: ApplicationService) -> None:
    with pytest.raises(ConfigError) as caught:
        await service.search("retry", profile="turbo")
    assert "fast" in str(caught.value)


async def test_the_configured_profile_is_used_when_none_is_given(
    backend: FakeBackend,
) -> None:
    backend.settings = Settings(rag={"profile": "precise"})  # pyright: ignore[reportArgumentType]
    service = ApplicationService(backend)
    await service.search("retry")
    assert backend.retriever_.seen[0].profile is RetrievalProfile.PRECISE


async def test_an_empty_query_is_refused_rather_than_searched_for(
    service: ApplicationService,
) -> None:
    with pytest.raises(ConfigError):
        await service.search("   ")


async def test_an_answer_reports_the_confidence_band_as_well_as_the_score(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """Two different claims, and the payload carries both.

    The envelope the generation layer produces carries the score and not the band, so this is
    the surface joining what retrieval knew to what generation returned.
    """
    document = next(iter(backend.store.documents.values()))
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(document), score=0.9)]
    answered = await service.ask("what is the retry policy")
    assert answered.confidence == 0.5
    assert answered.confidence_band == "medium"
    assert answered.text


async def test_an_answer_citation_carries_the_documents_source_metadata(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """A citation on an answer reports the record, not only a search hit does.

    **This test exists because its absence was found by disabling a guard.** The answer path
    hydrates its documents in ``_require_scoped_context`` and threads them into
    ``_answer_payload``; blanking that dictionary left the whole suite green — because the fake
    answerer emitted an envelope with **no citations at all**, so ``AnswerCitation.provenance``
    had never been constructed by any test on any surface, while the pull request describing it
    claimed it was populated. An untested field on a contract is one the next refactor removes for
    free, and `ask` is the output a reader is most likely to act on.

    The envelope therefore carries a real citation here, pointing at the seeded document's own
    chunk, so the payload is assembled the way a real answer assembles it.
    """
    from manicule.core.anchors import HeadingAnchor  # noqa: PLC0415 - only this test builds one
    from manicule.core.provenance import LocalSnapshot, Provenance, SourceMetadata  # noqa: PLC0415
    from manicule.generation.answers import (  # noqa: PLC0415 - keeps generation out of the rest
        AnswerEnvelope,
        Citation,
        Verification,
    )

    canonical = "https://docs.example.test/pages/123456/retry-policy"
    document = make_document(
        backend.workspace,
        source_id="123456.html",
        title="Retry policy",
        provenance=Provenance(
            source=SourceMetadata(
                title="Retry policy",
                canonical_uri=canonical,
                source_id="123456",
                version="7",
                section_path=("Engineering", "Runbooks"),
            ),
            snapshot=LocalSnapshot(path="mirror/123456.html"),
        ),
    )
    chunk = make_chunk(document)
    backend.store.add(document)
    backend.store.chunks[document.id] = [chunk]
    backend.retriever_.candidates = [Candidate(chunk=chunk, score=0.9)]
    backend.answerer_.envelope = AnswerEnvelope(
        text="The client retries twice.",
        corpus_consulted=True,
        confidence=0.5,
        citations=(
            Citation(
                slot=1,
                document_id=document.id,
                chunk_id=chunk.id,
                uri=document.uri,
                title=document.title,
                heading_path=chunk.heading_path,
                anchor=HeadingAnchor(path=chunk.heading_path),
                quote=chunk.text,
                verification=Verification.RESOLVED,
            ),
        ),
    )

    answered = await service.ask("what is the retry policy")

    assert len(answered.citations) == 1, "the fixture must produce a citation to assert about"
    cited = answered.citations[0]
    assert cited.provenance is not None, (
        "an answer citation reports no source metadata, so a mirrored page is cited by its local "
        "filename in the one output a reader is most likely to act on"
    )
    assert cited.provenance.canonical_uri == canonical
    assert cited.provenance.source_id == "123456"
    assert cited.provenance.version == "7"
    assert cited.provenance.section_path == ("Engineering", "Runbooks")
    assert cited.provenance.snapshot_path == "mirror/123456.html", (
        "and the local snapshot is retained beside it, for audit"
    )


async def test_an_answer_carries_the_reason_for_its_confidence_as_a_search_does(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """The same judgement, from the same object, and it reached only one of the two commands.

    ``search`` has printed the reason since retrieval learned to admit ignorance; ``ask`` had
    the band and not the sentence that makes a small number legible, so an answer over
    passages the corpus does not really have showed a bare score above prose that reads like
    an answer.
    """
    document = next(iter(backend.store.documents.values()))
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(document), score=0.9)]

    answered = await service.ask("what is the retry policy")
    found = await service.search("what is the retry policy")

    assert answered.confidence_reason == found.confidence_reason
    assert answered.confidence_reason


async def test_streaming_and_collecting_produce_the_same_answer(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """``on_event`` is a view hook. It cannot change what the answer is.

    A surface that rendered progress *and* produced a different result would be a second
    answer path, which is exactly what this layer exists to prevent.
    """
    document = next(iter(backend.store.documents.values()))
    backend.retriever_.candidates = [Candidate(chunk=make_chunk(document), score=0.9)]
    seen: list[str] = []
    quiet = await service.ask("q")
    noisy = await service.ask("q", on_event=lambda event: seen.append(event.text))
    assert quiet.text == noisy.text
    assert "".join(seen).strip() == quiet.text


# --- documents -----------------------------------------------------------------------------


async def test_document_get_includes_chunks_only_when_asked(
    service: ApplicationService,
) -> None:
    """Chunk text is the bulk of a document, and most callers of ``get`` do not want it."""
    document_id = next(iter((await service.document_list()).documents)).id
    assert (await service.document_get(document_id)).chunks == ()
    assert (await service.document_get(document_id, chunks=True)).chunks


async def test_a_delete_defaults_to_the_trash(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """Recoverable by default, and the payload says which kind of delete happened."""
    document_id = next(iter(backend.store.documents))
    outcome = await service.document_delete(document_id)
    assert outcome.mode == "soft"
    assert backend.store.deleted == [(document_id, "soft")]


async def test_a_hard_delete_is_asked_for_explicitly(
    service: ApplicationService, backend: FakeBackend
) -> None:
    document_id = next(iter(backend.store.documents))
    outcome = await service.document_delete(document_id, hard=True)
    assert outcome.mode == "hard"
    assert backend.store.deleted == [(document_id, "hard")]


async def test_reindexing_reports_what_could_not_be_repaired(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """An unrepairable document is named rather than counted as a success."""
    from manicule.ingest.reindex import ReindexReport  # noqa: PLC0415

    unrepairable = ReindexReport()
    unrepairable.unrepairable.append("no retained bytes")

    async def reindex(document_id: str) -> ReindexReport:
        del document_id
        return unrepairable

    backend.ingestion_.reindex = reindex
    document_id = next(iter(backend.store.documents))
    outcome = await service.document_reindex(document_id)
    assert outcome.status == "failed"
    assert "no retained bytes" in outcome.detail


# --- ingest --------------------------------------------------------------------------------


async def test_indexing_a_path_that_does_not_exist_says_so(
    service: ApplicationService, tmp_path: Path
) -> None:
    with pytest.raises(UnknownEntityError):
        await service.index_path(tmp_path / "nowhere")


async def test_an_ingest_report_keeps_outcomes_that_are_neither_success_nor_failure(
    service: ApplicationService, backend: FakeBackend, tmp_path: Path
) -> None:
    """A PDF with no extractable text is not an ingest and not a failure.

    Folding it into either would hide the one outcome that needs looking at, so ``by_status``
    carries the run's own counters rather than a summary of them.
    """
    report = RunReport(connector="local", discovered=2)
    report.by_status = {
        DocumentStatus.INDEXED.value: 1,
        DocumentStatus.NO_EXTRACTABLE_TEXT.value: 1,
    }
    backend.ingestion_.report = report
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    outcome = await service.index_path(tmp_path)
    assert outcome.ingested == 1
    assert outcome.failed == 0
    assert outcome.by_status[DocumentStatus.NO_EXTRACTABLE_TEXT.value] == 1


async def test_syncing_a_connector_that_is_not_configured_lists_what_is(
    service: ApplicationService,
) -> None:
    with pytest.raises(UnknownEntityError) as caught:
        await service.connector_sync("notion")
    assert "none configured" in str(caught.value)


# --- state ---------------------------------------------------------------------------------


async def test_stats_group_three_ways(service: ApplicationService) -> None:
    stats = await service.stats()
    assert stats.documents == 1
    assert stats.by_source == {"local": 1}
    assert stats.by_media_type == {"text/markdown": 1}


async def test_doctor_checks_configuration_transport_plugins_storage_and_the_index(
    service: ApplicationService,
) -> None:
    diagnosis = await service.doctor()
    assert {check.name for check in diagnosis.checks} >= {
        "configuration",
        "transport",
        "plugins",
        "storage",
        "permissions",
        "index",
    }
    assert diagnosis.state in {"ok", "degraded", "failing", "unknown"}


async def test_doctor_reports_the_worst_state_it_found(backend: FakeBackend) -> None:
    """A rollup that reported the best state would be a diagnostic nobody could act on."""
    backend.settings = Settings(
        security={"transport": {"bind_host": "192.0.2.10"}}  # pyright: ignore[reportArgumentType]
    )
    diagnosis = await ApplicationService(backend).doctor()
    states: set[CheckState] = {check.state for check in diagnosis.checks}
    assert "failing" in states
    assert diagnosis.state == "failing"


async def test_doctor_names_an_unauthenticated_public_bind_as_failing(
    backend: FakeBackend,
) -> None:
    """The one configuration this project refuses to ship, reported in the words to fix it."""
    backend.settings = Settings(
        security={"transport": {"bind_host": "0.0.0.0"}}  # pyright: ignore[reportArgumentType]  # noqa: S104 - the subject of the check
    )
    diagnosis = await ApplicationService(backend).doctor()
    transport = next(check for check in diagnosis.checks if check.name == "transport")
    assert transport.state == "failing"
    assert "security.auth.mode" in transport.detail


async def test_doctor_fails_on_a_data_directory_other_accounts_can_read(
    backend: FakeBackend, tmp_path: Path
) -> None:
    """Failing, not degraded. The data directory holds the corpus, so this is an exposure.

    ``chmod`` after ``mkdir`` rather than ``mkdir(mode=...)``, because ``mkdir``'s mode is
    masked by the process ``umask`` — under ``umask 077`` the directory would come out
    ``0700`` and this test would pass having created nothing to object to.
    """
    data_dir = tmp_path / "exposed"
    data_dir.mkdir()
    data_dir.chmod(0o755)
    backend.settings = Settings(data_dir=data_dir)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")
    assert check.state == "failing"
    assert str(data_dir) in check.detail
    assert "chmod 0700" in check.detail
    assert diagnosis.state == "failing"


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o701, 0o770])
async def test_doctor_objects_to_any_group_or_other_bit(
    backend: FakeBackend, tmp_path: Path, mode: int
) -> None:
    """Not only ``0755``. Execute alone on a directory is enough to read a named path through it."""
    data_dir = tmp_path / f"mode-{mode:o}"
    data_dir.mkdir()
    data_dir.chmod(mode)
    backend.settings = Settings(data_dir=data_dir)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")
    assert check.state == "failing"


async def test_doctor_accepts_the_mode_the_storage_layer_writes(
    backend: FakeBackend, tmp_path: Path
) -> None:
    """``0700`` is what ``prepare_data_dir`` creates, so a stock install must pass.

    The check earns nothing if it fails on a correct installation: an operator who sees
    ``doctor`` fail on a healthy machine learns to ignore it.
    """
    data_dir = tmp_path / "private"
    data_dir.mkdir()
    data_dir.chmod(0o700)
    (data_dir / "manicule.db").write_bytes(b"")
    backend.settings = Settings(data_dir=data_dir)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")
    assert check.state == "ok"


async def test_doctor_says_it_does_not_know_when_the_data_directory_is_not_there(
    backend: FakeBackend, tmp_path: Path
) -> None:
    """Absent is not exposed, and reporting it as one sends an operator to chmod nothing."""
    backend.settings = Settings(data_dir=tmp_path / "never-created")
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")
    assert check.state == "unknown"


@pytest.fixture
def grammar_seeds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record what the pre-seed is asked for instead of fetching it.

    Every test here that reaches ``init`` or ``doctor --fix`` goes through this. Without it a
    suite run on a machine with an empty grammar cache would download the declared set — 80-odd
    megabytes, from a unit test, on the first checkout — and one run on a machine that has them
    would prove nothing about the call ever happening. That the seeding *works* is proved
    against a real bundle in ``tests/parsers/test_grammar_bundle.py``; what is asserted here is
    which operations ask for it and what they do with the answer.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    asked: list[tuple[str, ...]] = []

    def spy(languages: Sequence[str], **_: object) -> tuple[str, ...]:
        asked.append(tuple(languages))
        return tuple(languages)

    monkeypatch.setattr(grammars, "prefetch", spy)
    return asked


def _empty_grammar_cache(cache: Path) -> Settings:
    """Settings whose code parser reads a cache directory with nothing in it.

    Configured rather than pointed at the pack directly: ``doctor`` applies the configured cache
    before it asks what is present, and a test that redirected the pack would be answered about
    a directory the service never reads.
    """
    return Settings(
        plugins={  # pyright: ignore[reportArgumentType] - validated on the way in
            "config": {"parser.sourcecode": {"grammar_cache_dir": str(cache)}}
        }
    )


async def test_doctor_reports_grammars_it_cannot_find_without_fetching_them(
    backend: FakeBackend, tmp_path: Path, grammar_seeds: list[tuple[str, ...]]
) -> None:
    """A report is a report. ``doctor`` on its own must not start an 80 MB download."""
    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "degraded"
    assert grammar_seeds == []


async def test_doctor_fix_asks_the_pre_seed_for_the_whole_declared_set(
    backend: FakeBackend, tmp_path: Path, grammar_seeds: list[tuple[str, ...]]
) -> None:
    """The repair, and the only thing in this command that writes to the machine.

    The declared set, not the missing subset: what a machine is asked to hold is a decision made
    in configuration, and the pre-seed is the one that works out what is already there. Asking
    for the difference would put that arithmetic in two places.

    What is asserted here is the wiring. That a seed actually populates a cache — and that the
    check then reports ``ok`` and says what it seeded — is proved against a real bundle with no
    network in ``tests/parsers/test_grammar_bundle.py``, because a double that claims to have
    seeded and writes nothing is exactly the state ``prefetch`` exists to catch.
    """
    from manicule.parsers.grammars import DECLARED_LANGUAGES  # noqa: PLC0415 - a parsing extra

    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    await ApplicationService(backend).doctor(fix=True)

    assert grammar_seeds == [DECLARED_LANGUAGES]


async def test_a_repair_is_reported_from_the_cache_rather_than_from_what_the_seed_returned(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ok`` means the grammars are there, not that something said it put them there.

    The pre-seed has its own version of this assertion for a reason that is not hypothetical:
    the grammar pack's own ``prefetch`` returns successfully for a language already in its
    process-global registry **whatever the configured cache holds**, which is how a container
    build ships an image with no grammars in it and is told it succeeded. This is the same
    check one layer up, and it costs a directory listing. The double here is that failure with
    the honesty removed: it reports success and writes nothing.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    def claim_success(languages: Sequence[str], **_: object) -> tuple[str, ...]:
        return tuple(languages)

    monkeypatch.setattr(grammars, "prefetch", claim_success)
    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    diagnosis = await ApplicationService(backend).doctor(fix=True)
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "degraded", "doctor believed a pre-seed that wrote nothing"


async def test_a_repair_that_could_not_complete_is_failing_rather_than_degraded(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked for and not done. Without ``--fix`` the same state is degraded, and rightly so:
    nothing was attempted, and an installation with no code in its corpus is fine as it is."""
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    def refuse(languages: Sequence[str], **_: object) -> tuple[str, ...]:
        raise grammars.GrammarFetchError(f"no route to the grammar release for {list(languages)}")

    monkeypatch.setattr(grammars, "prefetch", refuse)
    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    diagnosis = await ApplicationService(backend).doctor(fix=True)
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "failing"
    assert "no route to the grammar release" in check.detail
    assert diagnosis.state == "failing"


async def test_doctor_says_it_does_not_know_when_the_parsing_extra_is_not_installed(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core installs without the parsing extra, and ``doctor`` is not the command that
    discovers that by failing to run. There are no grammars to check and nothing is wrong."""
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    def absent(languages: Sequence[str]) -> tuple[str, ...]:
        del languages
        raise ImportError("No module named 'tree_sitter_language_pack'")

    monkeypatch.setattr(grammars, "missing_grammars", absent)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "unknown"
    assert "manicule[parsers]" in check.detail


async def test_a_code_parser_configuration_that_does_not_validate_is_reported(
    backend: FakeBackend,
) -> None:
    """The declared language set decides what this installation parses at all, and a
    configuration nothing can read is a diagnosis rather than a traceback out of ``doctor``."""
    backend.settings = Settings(
        plugins={  # pyright: ignore[reportArgumentType] - deliberately wrong, one layer down
            "config": {"parser.sourcecode": {"languages": "python"}}
        }
    )

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "grammars")

    assert check.state == "failing"
    assert "parser.sourcecode" in check.detail


@pytest.fixture
def vocabulary_seeds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record what the vocabulary pre-seed is asked for instead of fetching it.

    The counterpart of :func:`grammar_seeds`, for the same two reasons: a suite run on a
    machine with an empty ``tiktoken`` cache would otherwise download 5 MB from a unit test,
    and one run on a machine that already has it would prove nothing about the call happening
    at all. That the seeding *works* is proved against a real bundle with the network cut in
    ``tests/vocabularies/test_vocabulary_bundle.py``; what is asserted here is which
    operations ask for it and what they do with the answer.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    asked: list[tuple[str, ...]] = []

    def spy(encodings: Sequence[str], **_: object) -> tuple[str, ...]:
        asked.append(tuple(encodings))
        return tuple(encodings)

    monkeypatch.setattr(vocabularies, "prefetch", spy)
    return asked


@pytest.fixture
def empty_vocabulary_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``tiktoken`` at a cache directory with nothing in it.

    Through the environment because that is the only hook ``tiktoken`` reads, and it is what a
    deployment sets — so this is the same mechanism a container uses rather than a test-only
    door into the code under test.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    cache = tmp_path / "tiktoken"
    cache.mkdir()
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(cache))
    return cache


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_doctor_reports_a_missing_vocabulary_without_fetching_one(
    backend: FakeBackend, vocabulary_seeds: list[tuple[str, ...]]
) -> None:
    """A report is a report: ``doctor`` on its own must not start a download."""
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "failing"
    assert "manicule doctor --fix" in check.detail
    assert vocabulary_seeds == []


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_a_missing_vocabulary_is_failing_where_a_missing_grammar_is_degraded(
    backend: FakeBackend, tmp_path: Path, vocabulary_seeds: list[tuple[str, ...]]
) -> None:
    """The severity decision, asserted as a decision rather than left to a reader.

    Grammars are degraded because there are installations for which their absence costs
    nothing — a corpus of Markdown and PDFs works perfectly without one — and a red check on a
    healthy machine teaches an operator to ignore ``doctor``. No such installation exists for
    a vocabulary: every context is measured with it whatever the corpus is made of, so a
    machine without one cannot answer a question at all. Marking both the same way would make
    one of the two answers a lie, and which one depends on which way they were made the same.
    """
    del vocabulary_seeds
    backend.settings = _empty_grammar_cache(tmp_path / "grammars")

    diagnosis = await ApplicationService(backend).doctor()
    states = {check.name: check.state for check in diagnosis.checks}

    assert states["grammars"] == "degraded"
    assert states["vocabularies"] == "failing"
    assert diagnosis.state == "failing"


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_doctor_fix_asks_the_pre_seed_for_every_encoding_this_install_uses(
    backend: FakeBackend, vocabulary_seeds: list[tuple[str, ...]]
) -> None:
    """Every encoding, not the missing subset, and read from this installation's settings.

    Two encodings, and the pair is the point: ``cl100k_base`` is the chunker's stand-in and
    ``o200k_base`` is what a context is measured with. Seeding only the first is an install
    that indexes and cannot answer, which is the defect the whole area exists to close.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    await ApplicationService(backend).doctor(fix=True)

    wanted = vocabularies.required_encodings(backend.settings.rag.context.encoding)
    assert vocabulary_seeds == [wanted]
    assert len(wanted) > 1


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_a_vocabulary_repair_is_reported_from_the_cache_and_not_from_the_seed(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ok`` means the vocabulary is there, not that something said it put it there.

    ``tiktoken`` writes its cache on a best-effort basis — it swallows every write error when
    the directory was not named by an environment variable — so "the fetch returned" and "the
    file is on disk" are genuinely different facts. The double here is that failure with the
    honesty removed: it reports success and writes nothing.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    def claim_success(encodings: Sequence[str], **_: object) -> tuple[str, ...]:
        return tuple(encodings)

    monkeypatch.setattr(vocabularies, "prefetch", claim_success)

    diagnosis = await ApplicationService(backend).doctor(fix=True)
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "failing", "doctor believed a pre-seed that wrote nothing"


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_a_vocabulary_repair_that_could_not_complete_says_what_stopped_it(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message an air-gapped operator reads, carried through rather than replaced."""
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    def refuse(encodings: Sequence[str], **_: object) -> tuple[str, ...]:
        message = f"no route to the blob store for {list(encodings)}"
        raise vocabularies.VocabularyFetchError(message)

    monkeypatch.setattr(vocabularies, "prefetch", refuse)

    diagnosis = await ApplicationService(backend).doctor(fix=True)
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "failing"
    assert "no route to the blob store" in check.detail


async def test_doctor_says_it_does_not_know_when_the_retrieval_extra_is_not_installed(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core installs without the retrieval extra, and ``doctor`` is not the command that
    discovers that by failing to run. There is no token counter to give a vocabulary to."""
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    def absent(encodings: Sequence[str]) -> tuple[str, ...]:
        del encodings
        raise ImportError("No module named 'tiktoken'")

    monkeypatch.setattr(vocabularies, "missing_vocabularies", absent)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "unknown"
    assert "manicule[retrieval]" in check.detail


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_an_encoding_no_installed_tiktoken_defines_is_a_configuration_failure(
    backend: FakeBackend,
) -> None:
    """``rag.context.encoding`` holding a model name is the specific mistake to catch.

    ``docs/retrieval.md`` §7.2 asks the fitter to name an encoding and never a model, because
    a model name makes an estimate look authoritative about a model that is not being used.
    Reaching this far, it must be a diagnosis rather than a refusal at the first question.
    """
    backend.settings = Settings(
        rag={"context": {"encoding": "gpt-4o"}}  # pyright: ignore[reportArgumentType] - validated on the way in
    )

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "failing"
    assert "never a model name" in check.detail


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_init_pre_seeds_the_vocabularies_and_reports_what_it_did(
    backend: FakeBackend, config_home: Path, vocabulary_seeds: list[tuple[str, ...]]
) -> None:
    """Install time is when a vocabulary arrives, because it must never arrive during a query.

    Through the same call ``doctor --fix`` makes, so an install and a repair cannot come to
    mean different things — which is exactly what would have happened had ``init`` seeded
    grammars and left this to be discovered at the first search.
    """
    del config_home
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    report = await ApplicationService(backend).initialise()

    wanted = vocabularies.required_encodings(backend.settings.rag.context.encoding)
    assert vocabulary_seeds == [wanted]
    assert any("vocabularies" in note for note in report.notes)


@pytest.mark.usefixtures("empty_vocabulary_cache")
async def test_a_vocabulary_pre_seed_that_fails_leaves_init_finished_and_says_so(
    backend: FakeBackend, config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note, not an exception, exactly as the grammar pre-seed does it.

    The configuration file is on disk by then, so raising would leave a written config beside
    a command that reported failure and a retry that needs ``--force``. An air-gapped host
    with no bundle must still be able to finish installing.
    """
    from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra, not core

    def refuse(encodings: Sequence[str], **_: object) -> tuple[str, ...]:
        del encodings
        message = "no route to the blob store"
        raise vocabularies.VocabularyFetchError(message)

    monkeypatch.setattr(vocabularies, "prefetch", refuse)

    report = await ApplicationService(backend).initialise()

    assert await asyncio.to_thread(config_home.exists)
    assert any("no route to the blob store" in note for note in report.notes)


async def test_doctor_builds_nothing_expensive(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """A diagnostic that needs the system working is not a diagnostic.

    Asserted through the backend's own call log: ``doctor`` must not reach for the retriever
    or the answerer, because building either loads a model runtime — on a machine whose
    problem may be that the model runtime does not load.
    """
    await service.doctor()
    assert backend.retriever_.seen == []
    assert backend.answerer_.calls == []


# --- configuration ---------------------------------------------------------------------------


async def test_configuration_is_returned_with_secrets_masked(
    backend: FakeBackend,
) -> None:
    """This is what an MCP tool hands to an assistant, so a live key here is a leaked key."""
    backend.settings = Settings(
        llm={"provider": "openai"},  # pyright: ignore[reportArgumentType]
        providers={"openai": {"api_key": "sk-super-secret"}},  # pyright: ignore[reportArgumentType]
    )
    value = await ApplicationService(backend).config_get()
    assert "sk-super-secret" not in json.dumps(value.value)


async def test_reading_a_setting_that_does_not_exist_says_so(
    service: ApplicationService,
) -> None:
    with pytest.raises(UnknownEntityError):
        await service.config_get("rag.nope")


async def test_writing_a_credential_is_refused(
    service: ApplicationService, config_home: Path
) -> None:
    """Credentials belong in the environment, which is not copied into backups and exports."""
    del config_home
    with pytest.raises(ConfigError) as caught:
        await service.config_set("providers.openai.api_key", "sk-nope")
    assert "environment" in str(caught.value)


async def test_a_setting_is_written_and_reads_back(
    service: ApplicationService, config_home: Path
) -> None:
    change = await service.config_set("rag.profile", "precise")
    assert change.value == "precise"
    written = await asyncio.to_thread(config_home.read_text, "utf-8")
    assert written.count("precise") == 1


async def test_a_value_that_parses_as_json_is_stored_as_json(
    service: ApplicationService, config_home: Path
) -> None:
    """``false`` is a boolean; ``qwen2.5:14b`` is a string. Neither needs quoting at a terminal."""
    del config_home
    assert (await service.config_set("rag.cache.enabled", "false")).value is False
    assert (await service.config_set("llm.model", "qwen2.5:14b")).value == "qwen2.5:14b"


async def test_a_setting_that_would_not_validate_is_refused_before_the_file_is_written(
    service: ApplicationService, config_home: Path
) -> None:
    """A config file that fails at the next start is a manicule that does not start."""
    with pytest.raises(ConfigError):
        await service.config_set("rag.profile", "turbo")
    assert not await asyncio.to_thread(config_home.exists)


async def test_the_whole_tree_is_validated_not_just_the_key_being_written(
    service: ApplicationService, config_home: Path
) -> None:
    """Two settings that are each valid and jointly wrong is what ``policy_problems`` catches.

    A routable bind host is a valid string, and ``security.auth.mode = "none"`` is a valid
    mode. Together they are an unauthenticated document index on the network, and the refusal
    has to happen when the second one is written rather than at the next start — because by
    then the file says something the process will not run.
    """
    with pytest.raises(ConfigError):
        await service.config_set("security.transport.bind_host", "192.0.2.10")
    assert not await asyncio.to_thread(config_home.exists), (
        "the file was written before validation refused it"
    )


# --- workspaces --------------------------------------------------------------------------------


async def test_workspace_list_counts_only_the_active_workspace(
    backend: FakeBackend,
) -> None:
    """Counting another tenant's rows through this handle would be the breach it prevents."""
    backend.maintenance_.workspace_rows = [
        ("default", "default", "personal"),
        ("other", "other", "personal"),
    ]
    listed = await ApplicationService(backend).workspace_list()
    by_id = {workspace.id: workspace for workspace in listed.workspaces}
    assert by_id["default"].active
    assert by_id["default"].documents == 1
    assert by_id["other"].documents is None


async def test_switching_to_an_unknown_workspace_is_refused_unless_asked_for(
    service: ApplicationService, config_home: Path
) -> None:
    del config_home
    with pytest.raises(UnknownEntityError):
        await service.workspace_switch("nope")
    switched = await service.workspace_switch("nope", create=True)
    assert switched.active == "nope"
    assert switched.previous == "default"


# --- plugins -----------------------------------------------------------------------------------


async def test_plugin_list_reports_the_components_each_plugin_registers(
    backend: FakeBackend,
) -> None:
    backend.discovery = discover()
    listed = await ApplicationService(backend).plugin_list()
    assert listed.count == len(backend.discovery.manifests)
    assert any(
        component.kind == "parser" for plugin in listed.plugins for component in plugin.components
    )


async def test_the_registry_is_not_consulted_unless_installation_is_allowed(
    backend: FakeBackend,
) -> None:
    """A plugin runs with this process's full authority, so browsing a catalogue is opt-in."""
    backend.discovery = discover()
    listed = await ApplicationService(backend).plugin_list(registry=True)
    assert listed.available == ()
    assert "allow_install" in listed.registry_error


async def test_adding_a_plugin_that_is_not_installed_never_installs_it(
    backend: FakeBackend,
) -> None:
    """manicule does not run a package manager. The refusal names what would."""
    backend.discovery = discover()
    with pytest.raises(UnknownEntityError) as caught:
        await ApplicationService(backend).plugin_add("nonexistent-plugin")
    assert "Install the distribution" in str(caught.value)


async def test_disabling_a_plugin_writes_configuration_and_touches_no_package(
    backend: FakeBackend, config_home: Path
) -> None:
    backend.discovery = discover()
    changed = await ApplicationService(backend).plugin_remove("parsing")
    assert changed.enabled is False
    assert changed.installed is True
    assert "parsing" in await asyncio.to_thread(config_home.read_text, "utf-8")


async def test_disabling_and_re_enabling_leaves_the_configuration_where_it_started(
    backend: FakeBackend, config_home: Path
) -> None:
    """A round trip that accumulated state would make the second one behave differently."""
    backend.discovery = discover()
    service = ApplicationService(backend)
    await service.plugin_remove("parsing")
    await service.plugin_add("parsing")
    written = await asyncio.to_thread(config_home.read_text, "utf-8")
    assert "disabled = []" in written.replace("disabled = [\n]", "disabled = []")


# --- init and upgrade ----------------------------------------------------------------------------


def test_the_hardware_probe_says_what_it_actually_measured() -> None:
    """``init`` picks the embedding backend from this, so a wrong reading indexes a corpus."""
    probe = hardware()
    assert probe["system"]
    assert isinstance(probe["apple_silicon"], bool)


async def test_init_writes_a_loopback_bind_explicitly(
    service: ApplicationService, config_home: Path, grammar_seeds: list[tuple[str, ...]]
) -> None:
    """Written out rather than left to the default, so an operator can see it is loopback."""
    del grammar_seeds
    report = await service.initialise()
    assert report.path == str(config_file())
    assert "127.0.0.1" in await asyncio.to_thread(config_home.read_text, "utf-8")
    assert any("127.0.0.1" in note for note in report.notes)


async def test_init_refuses_to_overwrite_an_existing_configuration(
    service: ApplicationService, config_home: Path, grammar_seeds: list[tuple[str, ...]]
) -> None:
    del grammar_seeds
    await service.initialise()
    with pytest.raises(ConfigError):
        await service.initialise()
    assert (await service.initialise(force=True)).path == str(config_home)


async def test_init_pre_seeds_the_declared_grammars_and_reports_what_it_did(
    backend: FakeBackend, config_home: Path, tmp_path: Path, grammar_seeds: list[tuple[str, ...]]
) -> None:
    """Install time is when grammars arrive, because they must never arrive during ingest.

    The pack ships none in its wheel and fetches them per language on first use, which would
    make a file chunk one way on a machine that reached the network and another way on a
    machine that did not (``docs/parsing.md`` §8.1). Pre-seeding here is what makes an install
    with grammars missing something a person is told about rather than something they discover
    when every ``.py`` file in their corpus comes back unsupported.
    """
    del config_home
    from manicule.parsers.grammars import DECLARED_LANGUAGES  # noqa: PLC0415 - a parsing extra

    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    report = await ApplicationService(backend).initialise()

    assert grammar_seeds == [DECLARED_LANGUAGES]
    assert any("grammars" in note for note in report.notes)


async def test_a_pre_seed_that_fails_leaves_init_finished_and_says_so(
    backend: FakeBackend, config_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note, not an exception, and the configuration file is on disk either way.

    Raising here would leave a written config beside a command that reported failure — so the
    retry would need ``--force`` — and would make an air-gapped host with no bundle unable to
    finish installing software it is perfectly able to run over a corpus of Markdown. The
    failure is not hidden: it is in the report, ``doctor`` repeats it, and every source
    document refused names the command that fixes it.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    def refuse(languages: Sequence[str], **_: object) -> tuple[str, ...]:
        del languages
        raise grammars.GrammarFetchError("no route to the grammar release")

    monkeypatch.setattr(grammars, "prefetch", refuse)
    backend.settings = _empty_grammar_cache(tmp_path / "cache")

    report = await ApplicationService(backend).initialise()

    assert await asyncio.to_thread(config_home.exists)
    assert any("no route to the grammar release" in note for note in report.notes)


async def test_upgrade_backs_up_and_then_refuses_to_run_a_package_manager(
    service: ApplicationService,
) -> None:
    """The dangerous part is done; the part that fetches and runs code is a person's decision."""
    report = await service.upgrade()
    assert report.performed is False
    assert "uv tool install" in report.detail
    assert report.backup is not None


async def test_upgrade_can_skip_the_backup_when_asked(service: ApplicationService) -> None:
    assert (await service.upgrade(skip_backup=True)).backup is None


async def test_upgrade_names_a_destination_without_creating_one(tmp_path: Path) -> None:
    """The service decides *where*; the backend is what touches a disk.

    Caught in review of the change that moved this destination out of the data directory: the
    service was creating the parent itself, so an ``upgrade`` against a backend that writes
    nothing still left a directory behind — in a real data directory's sibling, from a unit
    test that had asked for no filesystem at all.

    Its own backend and its own ``data_dir``, rather than the shared fixture's, because the
    assertion is about a path on disk and the shared one resolves to wherever this machine
    keeps its data directory.
    """
    data_dir = tmp_path / "data"
    service = ApplicationService(FakeBackend(settings=Settings(data_dir=data_dir)))
    sibling = tmp_path / "data-backups"

    report = await service.upgrade()

    assert report.backup is not None, "the destination is still named"
    assert report.backup.startswith(str(sibling)), "and it is the sibling, not a subdirectory"
    assert not sibling.exists(), "a service call that reached no storage created a directory"


# --- the weights, which nothing pre-seeds ----------------------------------------------------


@pytest.fixture
def weights_on_disk(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Decide what the model cache holds, without one being on the machine running the test.

    The probe is a real cache lookup, so on a developer's laptop it answers "present" and on a
    fresh CI runner it answers "absent" — which would make every assertion below true or false
    depending on who ran it. Patched at the hub seam, which is also the only place the cache is
    consulted.
    """

    def decide(present: bool) -> None:
        def cached(*_args: object, **_kwargs: object) -> bool:
            return present

        monkeypatch.setattr(hub, "is_cached", cached)

    return decide


async def test_doctor_says_a_first_index_has_a_download_in_front_of_it(
    backend: FakeBackend, weights_on_disk: Callable[[bool], None]
) -> None:
    """The pause this exists to explain is minutes long and produces no manicule output at all.

    ``ok`` rather than ``degraded``: a machine that has not downloaded a model yet is not a
    broken machine, and a red check on a healthy install is how an operator learns to stop
    reading ``doctor``. What it owes the reader is the sentence, not the alarm.
    """
    weights_on_disk(False)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "models")

    assert check.state == "ok"
    assert "is not on this machine yet" in check.detail
    assert "1.1 GB" in check.detail
    # The rollup is not asserted equal to "ok" — other checks in this fixture report
    # "unknown" — but a pending download must not be what makes a machine look unwell.
    assert diagnosis.state not in {"degraded", "failing"}


async def test_doctor_names_the_narrow_pre_seed_rather_than_the_suite_s_own(
    backend: FakeBackend, weights_on_disk: Callable[[bool], None]
) -> None:
    """``--full --mlx`` fetches 3.6 GB to seed a backend that loads 1.17 GB of it.

    Advice that costs the reader three times what saying nothing would is worse than silence,
    so the command named is the one that fetches exactly the configured backend's artefact.
    """
    weights_on_disk(False)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "models")

    assert "--backend mlx" in check.detail
    assert "--full" not in check.detail


async def test_doctor_is_quiet_about_weights_that_are_already_here(
    backend: FakeBackend, weights_on_disk: Callable[[bool], None]
) -> None:
    weights_on_disk(True)
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "models")

    assert check.state == "ok"
    assert "no download is pending" in check.detail


async def test_weights_that_cannot_be_fetched_and_are_not_here_is_a_failure(
    backend: FakeBackend, weights_on_disk: Callable[[bool], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one genuinely broken state, and the reason this check can fail at all.

    An install that has told the hub it may not look and has no weights cannot answer a
    question, and will not say so until somebody asks one. Everything else here is a machine
    part-way through a normal first run.
    """
    weights_on_disk(False)
    monkeypatch.setenv(OFFLINE_ENV, "1")
    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "models")

    assert check.state == "failing"
    assert OFFLINE_ENV in check.detail
    assert diagnosis.state == "failing"


async def test_doctor_never_downloads_a_model_to_report_on_one(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic that fetched a gigabyte to report on a gigabyte would be absurd.

    Asserted at ``snapshot_download`` itself, with ``local_files_only`` as the thing being
    checked — not at manicule's ``snapshot`` wrapper, which ``is_cached`` does not call. A test
    that patched the wrapper would pass whatever the probe did, which is a check whose name is
    wider than its assertion.
    """
    calls: list[bool] = []

    def probe(*_args: object, local_files_only: bool = False, **_kwargs: object) -> str:
        calls.append(local_files_only)
        if not local_files_only:
            message = "doctor reached the network to answer whether it would need to"
            raise AssertionError(message)
        return "/nowhere"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", probe)
    diagnosis = await ApplicationService(backend).doctor()

    assert any(check.name == "models" for check in diagnosis.checks)
    assert calls, "the models check answered without consulting the cache at all"
    assert all(calls), "the cache probe was allowed to reach the hub"


# --- a cache the operating system will take back ---------------------------------------------


async def test_doctor_will_not_call_an_impermanent_vocabulary_cache_healthy(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything is present, everything works, and it will stop working with no warning.

    ``ok`` here would be a check reporting health about a machine one temp sweep away from
    refusing every question — a check whose name outruns what it verifies. It says so instead,
    and names the variable that fixes it.
    """
    swept = Path(tempfile.gettempdir()) / "swept-away"
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(swept))

    def nothing_missing(_encodings: Sequence[str]) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(vocabularies, "missing_vocabularies", nothing_missing)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "degraded"
    assert vocabularies.CACHE_DIR_ENV in check.detail
    assert "reclaimed" in check.detail


async def test_doctor_is_quiet_about_a_vocabulary_cache_that_will_survive(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The steady state, which must stay ``ok`` or the warning above teaches nothing."""
    durable = tmp_path.anchor + "durable-vocabularies"
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, durable)

    def nothing_missing(_encodings: Sequence[str]) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(vocabularies, "missing_vocabularies", nothing_missing)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "ok"
    assert "reclaimed" not in check.detail


# --- the diagnosis as a machine reads it -------------------------------------------------------


async def test_a_diagnosis_carries_the_four_things_a_reader_needs_before_the_checks(
    service: ApplicationService,
) -> None:
    """Version, schema version and a timestamp, beside the overall state.

    A diagnosis is pasted into issues, stored by monitors and compared against yesterday's.
    Without a time on it there is no telling a live reading from a stale one somebody kept, and
    without the two versions there is no telling a check that changed meaning from a machine
    that changed state.
    """
    from datetime import datetime  # noqa: PLC0415 - only this assertion parses the stamp

    from manicule.app.results import DOCTOR_SCHEMA_VERSION  # noqa: PLC0415
    from manicule.core.version import CORE_VERSION  # noqa: PLC0415

    diagnosis = await service.doctor()

    assert diagnosis.schema_version == DOCTOR_SCHEMA_VERSION
    assert diagnosis.manicule_version == CORE_VERSION
    assert diagnosis.state in {"ok", "degraded", "failing", "unknown"}
    # Parsed rather than pattern-matched: a string that looks like a timestamp and does not
    # parse is worse than no timestamp, because it is trusted.
    stamp = datetime.fromisoformat(diagnosis.checked_at)
    assert stamp.tzinfo is not None, "a timestamp with no zone cannot be compared to anything"


async def test_every_check_carries_an_identifier_that_is_selected_on_rather_than_read(
    service: ApplicationService,
) -> None:
    """The names are the contract. A monitor matches on them; the prose is free to change.

    Asserted as a property of every check rather than against a list of the ones that exist
    today, so a check added tomorrow is held to the same rule without anybody editing this.
    """
    diagnosis = await service.doctor()

    assert diagnosis.checks, "a diagnosis with no checks proves nothing about their identifiers"
    for check in diagnosis.checks:
        assert check.name, "a check with no identifier cannot be selected on"
        assert check.name == check.name.strip()
        assert " " not in check.name, f"{check.name!r} has a space in it, so it is prose"
        assert check.name.islower(), f"{check.name!r} is not stable against a change of case"


async def test_the_json_a_diagnosis_dumps_to_is_the_shape_it_declares(
    service: ApplicationService,
) -> None:
    """Dumped and re-read, because the contract is the wire format rather than the class."""
    diagnosis = await service.doctor()

    dumped = json.loads(json.dumps(diagnosis.model_dump(mode="json")))

    assert set(dumped) == {"state", "schema_version", "manicule_version", "checked_at", "checks"}
    assert set(dumped["checks"][0]) == {"name", "state", "detail", "facts", "remedy"}


async def test_a_failing_required_check_is_reported_without_the_command_failing(
    backend: FakeBackend,
) -> None:
    """The exit-status decision, asserted where it is made rather than only written down.

    ``doctor`` succeeded: it was asked for a diagnosis and it produced one. The envelope's
    ``ok`` and the command's exit status track *that*, uniformly across every operation, and a
    script gates on the payload — ``docs/deployment.md`` §2 carries the recipe. Reporting a
    broken machine as a failed operation would make ``ok: true`` and exit 0 disagree for this
    one command.
    """
    backend.settings = Settings(
        security={"transport": {"bind_host": "0.0.0.0"}}  # pyright: ignore[reportArgumentType]  # noqa: S104 - the subject of the check
    )
    diagnosis = await ApplicationService(backend).doctor()

    assert diagnosis.state == "failing"
    failing = [check for check in diagnosis.checks if check.state == "failing"]
    assert failing, "the fixture was supposed to produce a failing check"
    assert all(check.remedy for check in failing if check.name == "transport")


async def test_a_warning_only_installation_is_degraded_rather_than_failing(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The middle state has to exist and be reachable, or operators read every run as binary.

    The claim is about the *severity of the finding*, not about the rollup. This fixture's
    plugin discovery has not run, which is honestly ``unknown``, and ``unknown`` outranks
    ``degraded`` — so asserting the overall state here would be asserting something about the
    fixture rather than about the warning.
    """
    data_dir = tmp_path / "private"
    data_dir.mkdir()
    data_dir.chmod(0o700)
    backend.settings = Settings(data_dir=data_dir)
    monkeypatch.setenv(vocabularies.CACHE_DIR_ENV, str(tempfile.gettempdir()))

    def nothing_missing(_encodings: Sequence[str]) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(vocabularies, "missing_vocabularies", nothing_missing)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "vocabularies")

    assert check.state == "degraded"
    assert check.remedy, "a warning nobody can act on is a warning that teaches nothing"
    failing = [check.name for check in diagnosis.checks if check.state == "failing"]
    assert failing == [], f"this installation was supposed to warn, not fail: {failing}"


async def test_a_check_that_can_be_repaired_says_so_in_a_field_rather_than_only_in_prose(
    backend: FakeBackend, tmp_path: Path
) -> None:
    """The remediation bug6 asks for, as something a script can act on.

    The advice was always in ``detail``, which means a consumer wanting it had to parse an
    English sentence that is free to be reworded. ``remedy`` is the same advice in a field.
    """
    data_dir = tmp_path / "exposed"
    data_dir.mkdir()
    data_dir.chmod(0o755)
    backend.settings = Settings(data_dir=data_dir)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")

    assert check.state == "failing"
    assert check.remedy.startswith("chmod 0700 ")
    assert check.facts["exposed"] is True
    # The group and other bits alone, which is what `exposure` measures. `0755` exposes `055`.
    assert check.facts["exposed_bits"] == "055"


async def test_a_healthy_check_offers_no_remedy(backend: FakeBackend, tmp_path: Path) -> None:
    """A remedy on a healthy check is advice to fix what is not broken."""
    data_dir = tmp_path / "private"
    data_dir.mkdir()
    data_dir.chmod(0o700)
    backend.settings = Settings(data_dir=data_dir)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")

    assert check.state == "ok"
    assert check.remedy == ""


async def test_a_diagnosis_names_the_home_directory_rather_than_the_account(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The redaction bug6 asks for, and the reason it is ``~`` rather than a black box.

    ``doctor``'s output is the thing an operator pastes into an issue. The paths in it run
    through ``$HOME``, whose name is the account name. What is kept is everything below it, so
    the reader can still ``cd`` to what it names and paste the ``chmod`` back.
    """
    home = tmp_path / "home" / "someone"
    data_dir = home / "manicule"
    data_dir.mkdir(parents=True)
    data_dir.chmod(0o755)
    monkeypatch.setattr(Path, "home", lambda: home)
    backend.settings = Settings(data_dir=data_dir)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")

    assert str(home) not in check.detail, "the account's home directory reached the output"
    assert str(home) not in json.dumps(check.model_dump(mode="json"))
    assert "~/manicule" in check.detail, "redaction ate the part that made it diagnosable"
    assert check.facts["path"] == "~/manicule"
    assert check.remedy == "chmod 0700 ~/manicule"


async def test_a_path_outside_the_home_directory_is_reported_whole(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/srv/manicule`` names no account, and hiding it would cost the reader the location."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    data_dir = tmp_path / "srv"
    data_dir.mkdir()
    data_dir.chmod(0o755)
    backend.settings = Settings(data_dir=data_dir)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "permissions")

    assert str(data_dir) in check.detail
    assert check.facts["path"] == str(data_dir)


async def test_a_diagnosis_never_prints_the_value_of_an_environment_variable(
    backend: FakeBackend,
    weights_on_disk: Callable[[bool], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming the switch is the diagnosis; its contents are somebody's environment.

    ``models`` reported ``HF_HUB_OFFLINE=<value>`` verbatim. The value carried the whole
    diagnosis nowhere — that the variable is *set* is the fact that matters — and printing an
    environment variable's contents into an output made for pasting is the one thing bug6 names
    outright.
    """
    private_value = "1-and-something-nobody-meant-to-share"
    weights_on_disk(False)
    monkeypatch.setenv(OFFLINE_ENV, private_value)

    diagnosis = await ApplicationService(backend).doctor()
    check = next(check for check in diagnosis.checks if check.name == "models")

    assert check.state == "failing", "the fixture was supposed to produce the offline refusal"
    dumped = json.dumps(check.model_dump(mode="json"))
    assert private_value not in dumped, "an environment variable's value reached the diagnosis"
    assert OFFLINE_ENV in check.detail, "the switch that caused this has to be named"
    assert check.facts["offline_env_set"] is True


# --- collections ------------------------------------------------------------------------------


async def test_the_orphan_sweep_reports_without_deleting_unless_asked(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """Reporting is the default, and the default is the whole safety argument.

    In a corpus where collections are optional, "in no collection" describes most of it. A
    verb that deleted on sight would be one keystroke from emptying the workspace, so the run
    that happens by default names what *would* go and moves nothing.
    """
    document = next(iter(backend.store.documents.values()))
    backend.organisation_.documents[document.id] = document

    reported = await service.collection_orphans()

    assert reported.count == 1
    assert reported.deleted is False
    assert reported.document_ids == (document.id,)
    assert backend.store.deleted == [], "the report deleted something"


async def test_the_orphan_sweep_trashes_rather_than_destroys(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """A soft delete, so an explicit cleanup is recoverable and an irreversible one is not here.

    Emptying the trash is a separate verb that already exists, and it stays separate: two
    decisions, taken at two moments, rather than one flag that means both.
    """
    document = next(iter(backend.store.documents.values()))
    backend.organisation_.documents[document.id] = document

    swept = await service.collection_orphans(delete=True)

    assert swept.deleted is True
    assert swept.document_ids == (document.id,)
    assert backend.store.deleted == [(document.id, "soft")], (
        "the sweep hard-deleted, so what it removed cannot be restored"
    )


async def test_a_document_in_a_collection_is_not_an_orphan(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """The positive control. A sweep that reported everything would pass the tests above."""
    document = next(iter(backend.store.documents.values()))
    backend.organisation_.documents[document.id] = document
    collection = await backend.organisation_.create_collection("alpha")
    await backend.organisation_.add_to_collection(collection.id, [document.id])

    reported = await service.collection_orphans(delete=True)

    assert reported.count == 0
    assert backend.store.deleted == []


async def test_renaming_reports_the_new_name_and_keeps_the_id(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """Rename is an update, not a replace: the id a caller already holds keeps working."""
    made = await service.collection_create("alpha")

    renamed = await service.collection_rename(made.id, "alpha runbooks")

    assert renamed.id == made.id
    assert renamed.name == "alpha runbooks"
    assert (await backend.organisation_.get_collection(made.id)) is not None


async def test_a_search_names_the_collections_it_was_scoped_to(
    service: ApplicationService, backend: FakeBackend
) -> None:
    """Diagnostics carry the scope, so a scoped search is distinguishable from a wide one.

    Two searches returning the same passages — one restricted to a collection that happens to
    contain everything, one unrestricted — are otherwise identical in a log.
    """
    await service.collection_create("alpha")

    scoped = await service.search("anything", collections=["alpha"])
    wide = await service.search("anything")

    assert scoped.collections == ("alpha",)
    assert wide.collections == ()
    del backend


async def test_updating_a_description_cannot_erase_one_by_omission(
    service: ApplicationService,
) -> None:
    """A set, not a merge — so the field is required and omission is a call that fails.

    This is a defect this branch shipped and this test is what found it, so it is worth being
    exact about. ``describe_collection`` writes whatever it is handed, and ``description``
    defaulted to ``None``; every surface could therefore reach the verb without mentioning the
    field, and ``collection update <id>`` with no arguments silently erased the description it
    is named for changing. Nothing raised and nothing rendered differently — the value was
    simply gone.

    Required, so a surface that forgets is a ``TypeError`` at the call rather than a caller
    who has to have been careful. Clearing is still possible and now has to be asked for.
    """
    import inspect  # noqa: PLC0415 - only this assertion reads a signature

    made = await service.collection_create("alpha", description="worked examples")
    parameter = inspect.signature(service.collection_update).parameters["description"]
    assert parameter.default is inspect.Parameter.empty, (
        "description has a default again, so `collection update <id>` with no arguments "
        "erases the description instead of failing to call"
    )

    kept = await service.collection_update(made.id, description="still worked examples")
    assert kept.description == "still worked examples"

    cleared = await service.collection_update(made.id, description="")
    assert cleared.description is None, (
        "an empty string neither cleared the description nor was rejected; 'no description' "
        "now has two spellings on the wire"
    )


# --- the connector-identity check, and the two settings around it --------------------------------


def _with_connectors(backend: FakeBackend, **connectors: object) -> ApplicationService:
    """A service whose settings carry the given sources."""
    from manicule.config.settings import ConnectorSettings  # noqa: PLC0415

    settings = backend.settings.model_copy(
        update={
            "connectors": {
                name: ConnectorSettings.model_validate(spec) for name, spec in connectors.items()
            }
        }
    )
    backend.settings = settings
    return ApplicationService(backend)


def _check(diagnosis: r.Diagnosis, name: str) -> r.Check:
    """The named check, or a failure saying which names were there.

    ``next`` with no default raises ``StopIteration``, which pytest reports as an error with no
    indication that the check simply was not emitted — so the names are listed instead.
    """
    for check in diagnosis.checks:
        if check.name == name:
            return check
    offered = ", ".join(sorted(c.name for c in diagnosis.checks))
    message = f"no check named {name!r}; the diagnosis carried: {offered}"
    raise AssertionError(message)


async def test_doctor_names_the_sources_whose_document_identity_is_about_to_change(
    backend: FakeBackend,
) -> None:
    """The whole point of the check: which values, what happens, what to run.

    A warning that says "some documents are affected" sends an operator to read every row in
    their corpus, so the affected ``source`` values are named the way the permissions check names
    the offending path.
    """
    backend.store.add(make_document(backend.workspace, source="confluence-snapshot"))
    service = _with_connectors(
        backend, **{"team-handbook": {"type": "confluence-snapshot", "options": {"root": "/x"}}}
    )

    check = _check(await service.doctor(), "connectors")

    assert check.state == "degraded"
    # The enumeration specifically, with its count -- not merely the string appearing somewhere.
    # A bare `"confluence-snapshot" in detail` passed on a message that named no source at all,
    # because the remedy command below happens to contain the same word. That assertion was
    # weaker than the name of this test, which is the defect it exists to catch elsewhere.
    assert "'confluence-snapshot' (1 document(s))" in check.detail
    assert "'team-handbook'" in check.detail
    assert check.facts["affected"] == ["confluence-snapshot"]
    assert check.facts["documents"] == 1
    assert check.facts["instances"] == ["team-handbook"]


async def test_the_check_states_the_consequence_rather_than_only_the_condition(
    backend: FakeBackend,
) -> None:
    """Somebody who syncs without being told will conclude the corpus doubled."""
    backend.store.add(make_document(backend.workspace, source="confluence-snapshot"))
    service = _with_connectors(
        backend, **{"team-handbook": {"type": "confluence-snapshot", "options": {"root": "/x"}}}
    )

    detail = _check(await service.doctor(), "connectors").detail

    assert "new ids" in detail
    assert "doubled" in detail


async def test_the_check_names_only_remedies_that_exist(backend: FakeBackend) -> None:
    """It must not point at a regime that cannot be invoked.

    ``manicule.ingest.reconcile.reconcile`` exists and is tested, and **nothing in the product
    calls it** — there is no command, no service method and no sweep that runs it. Naming it here
    would read as though there were a way to bulk-reconcile and the operator had missed it. The
    two commands named instead were run against a real index before this was written.
    """
    backend.store.add(make_document(backend.workspace, source="confluence-snapshot"))
    service = _with_connectors(
        backend, **{"team-handbook": {"type": "confluence-snapshot", "options": {"root": "/x"}}}
    )

    detail = _check(await service.doctor(), "connectors").detail

    assert "manicule document list --source confluence-snapshot" in detail
    assert "manicule document delete" in detail
    assert "reconcile" not in detail.replace("bulk reconciliation", "")


async def test_an_instance_named_after_its_own_type_is_not_reported(
    backend: FakeBackend,
) -> None:
    """Its documents keep their ids, so warning about it would be a warning about nothing."""
    backend.store.add(make_document(backend.workspace, source="filesystem"))
    service = _with_connectors(backend, filesystem={"type": "filesystem"})

    check = _check(await service.doctor(), "connectors")

    assert check.state == "ok"
    assert check.facts["affected"] == []


async def test_a_renamed_instance_with_no_documents_under_its_type_is_not_reported(
    backend: FakeBackend,
) -> None:
    """A fresh install configured the new way has nothing to migrate and should hear nothing."""
    service = _with_connectors(
        backend, **{"team-handbook": {"type": "confluence-snapshot", "options": {"root": "/x"}}}
    )

    assert _check(await service.doctor(), "connectors").state == "ok"


async def test_syncing_a_disabled_connector_is_refused(backend: FakeBackend) -> None:
    """The switch is what the operator trusted.

    ``enabled`` was reported by ``connector list`` and read by nothing, so somebody who turned a
    source off and checked was told it was off by the same program that would then sync it.
    """
    from manicule.core.errors import PolicyError  # noqa: PLC0415

    service = _with_connectors(backend, docs={"type": "filesystem", "enabled": False})

    with pytest.raises(PolicyError, match="enabled = false"):
        await service.connector_sync("docs")


async def test_an_enabled_connector_is_not_refused(backend: FakeBackend) -> None:
    """The guard above must reject the disabled case and nothing else."""
    service = _with_connectors(backend, docs={"type": "filesystem", "enabled": True})

    report = await service.connector_sync("docs")

    assert report.connector


async def test_schedule_s_is_refused_rather_than_silently_doing_nothing() -> None:
    """There is no scheduler. A setting that does nothing is a promise the software breaks.

    Loud rejection is the point: a config carrying it was already being ignored, and an operator
    who believed it was polling needs to find out from manicule rather than from a stale index.
    """
    from pydantic import ValidationError  # noqa: PLC0415

    from manicule.config.settings import ConnectorSettings  # noqa: PLC0415

    with pytest.raises(ValidationError, match="schedule_s"):
        ConnectorSettings.model_validate({"type": "filesystem", "schedule_s": 300})


async def test_the_check_offers_its_command_as_a_remedy_not_only_inside_a_sentence(
    backend: FakeBackend,
) -> None:
    """``remedy`` is the structured half of ``detail``, exactly as ``facts`` is.

    A surface that wants to offer the action should not have to find it inside prose — that is
    the same reasoning ``Check`` gives for carrying ``facts`` beside ``detail``, and the
    permissions check is the precedent for setting both. Leaving it empty on the one check whose
    entire purpose is naming a command is the omission this pins.
    """
    backend.store.add(make_document(backend.workspace, source="confluence-snapshot"))
    service = _with_connectors(
        backend, **{"team-handbook": {"type": "confluence-snapshot", "options": {"root": "/x"}}}
    )

    check = _check(await service.doctor(), "connectors")

    assert check.remedy == "manicule document list --source confluence-snapshot"


async def test_a_healthy_connector_check_offers_no_remedy(backend: FakeBackend) -> None:
    """Empty on a healthy check, per ``Check.remedy``'s own contract."""
    service = _with_connectors(backend, filesystem={"type": "filesystem"})

    assert _check(await service.doctor(), "connectors").remedy == ""


async def test_the_policy_hint_is_true_of_a_single_forbidding_setting(
    backend: FakeBackend,
) -> None:
    """``PolicyError`` is raised for two shapes and the hint is printed under both.

    ``policy_problems`` reports settings that are individually valid and jointly wrong. But a
    connector configured ``enabled = false`` — and ``require_sharing_enabled`` before it — raise
    the same type with a single setting as the whole cause. A hint saying only "two settings
    disagree with each other; the message lists every one of them" sends that reader looking for
    a second setting that does not exist.
    """
    from manicule.app.dispatch import run_op  # noqa: PLC0415

    service = _with_connectors(backend, docs={"type": "filesystem", "enabled": False})

    envelope = await run_op(
        "connector_sync", backend.workspace, lambda: service.connector_sync("docs")
    )

    assert envelope.error is not None
    # It must not *lead* with the two-settings claim, which is what sent the reader hunting.
    # The conditional mention of that case later is fine and is why this checks the opening.
    assert envelope.error.hint.startswith("Configuration forbids this")


# --- the document-identity dry run ---------------------------------------------------------------


def _declaring(backend: FakeBackend, declared: str, *, source_id: str) -> Document:
    """A document keyed on its path while its record declares an identity of its own."""
    from manicule.core.provenance import LocalSnapshot, SourceMetadata  # noqa: PLC0415

    return make_document(
        backend.workspace,
        source="handbook",
        source_id=source_id,
        provenance=Provenance(
            source=SourceMetadata(
                title="Retry Runbook",
                canonical_uri="https://docs.example.test/pages/1002",
                source_id=declared,
                version="7",
            ),
            snapshot=LocalSnapshot(path="pages/1002.html"),
        ),
    )


async def test_doctor_reports_the_old_and_new_identity_of_every_document_about_to_move(
    backend: FakeBackend,
) -> None:
    """The dry run §10 asks for: both identities, reuse, citations, and a command.

    Each field is asserted rather than the presence of the check, because the check existing is
    not the requirement — a report that named the affected count and left an operator unable to
    tell which row becomes which would be the same warning #98 gave, one release later.
    """
    backend.store.add(_declaring(backend, "1002", source_id="/corpus/pages/1002.html"))

    check = _check(await ApplicationService(backend).doctor(), "document-identity")

    assert check.state == "degraded"
    assert check.facts["affected"] == 1
    assert check.facts["chunks_reusable"] is False
    assert check.facts["vectors_reusable"] is False
    assert check.facts["citations_change"] is True
    moving = check.facts["documents"]
    assert isinstance(moving, list)
    row = moving[0]
    assert isinstance(row, dict)
    assert row["old_source_id"] == "/corpus/pages/1002.html"
    assert row["new_source_id"] == "1002"
    assert row["old_document_id"] == document_id(
        backend.workspace, "handbook", "/corpus/pages/1002.html"
    )
    assert row["new_document_id"] == document_id(backend.workspace, "handbook", "1002")
    assert row["old_document_id"] != row["new_document_id"]
    assert check.remedy == "manicule document list --source handbook"


async def test_the_identity_check_says_why_chunks_cannot_be_carried_across(
    backend: FakeBackend,
) -> None:
    """Two reasons, and naming only the first would imply re-keying was an option.

    The ids move *and* the text changes, because the documents whose identity moves are the ones
    whose body now reaches the storage parser. An operator told only about the ids would
    reasonably ask why a migration cannot simply recompute them.
    """
    backend.store.add(_declaring(backend, "1002", source_id="/corpus/pages/1002.html"))

    detail = _check(await ApplicationService(backend).doctor(), "document-identity").detail

    assert "chunk ids derive from the document id" in detail
    assert "storage parser" in detail
    assert "Collection membership and tags do not survive" in detail
    assert "doubled" in detail


async def test_the_identity_check_clears_once_the_documents_are_keyed_on_what_they_declare(
    backend: FakeBackend,
) -> None:
    """It has to stop firing on its own, or it is a warning nobody can ever satisfy."""
    backend.store.add(_declaring(backend, "1002", source_id="1002"))

    check = _check(await ApplicationService(backend).doctor(), "document-identity")

    assert check.state == "ok"
    assert check.facts["affected"] == 0
    assert check.remedy == ""


async def test_the_identity_check_ignores_documents_with_nothing_to_declare(
    backend: FakeBackend,
) -> None:
    """An ordinary local file is keyed on its path because that is all it has said."""
    backend.store.add(make_document(backend.workspace, source="handbook", source_id="notes.md"))

    assert _check(await ApplicationService(backend).doctor(), "document-identity").state == "ok"
