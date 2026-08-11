"""What the operations mean, checked once, where they are implemented.

Both surfaces call this class, so everything asserted here is asserted about both of them.
That is the whole reason the layer exists, and it is why these tests do not go through Typer
or FastMCP: driving a rule through an adapter tests the adapter.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from manicule.app.results import CheckState
from manicule.app.service import ApplicationService, hardware
from manicule.config.settings import Settings, config_file
from manicule.core.content import DocumentStatus
from manicule.core.errors import ConfigError, UnknownEntityError
from manicule.core.retrieval import Candidate, RetrievalProfile
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
    service: ApplicationService, config_home: Path
) -> None:
    """Written out rather than left to the default, so an operator can see it is loopback."""
    report = await service.initialise()
    assert report.path == str(config_file())
    assert "127.0.0.1" in await asyncio.to_thread(config_home.read_text, "utf-8")
    assert any("127.0.0.1" in note for note in report.notes)


async def test_init_refuses_to_overwrite_an_existing_configuration(
    service: ApplicationService, config_home: Path
) -> None:
    await service.initialise()
    with pytest.raises(ConfigError):
        await service.initialise()
    assert (await service.initialise(force=True)).path == str(config_home)


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
