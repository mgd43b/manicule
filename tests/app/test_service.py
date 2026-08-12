"""What the operations mean, checked once, where they are implemented.

Both surfaces call this class, so everything asserted here is asserted about both of them.
That is the whole reason the layer exists, and it is why these tests do not go through Typer
or FastMCP: driving a rule through an adapter tests the adapter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
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
