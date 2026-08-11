"""The composition root, against a real database rather than a fake of one.

The property that matters most here is the one nothing else in the suite could have noticed:
**a default configuration starts.** Every other test in this project registers the components
it needs, so a manicule that could not assemble itself from its own defaults would have passed
all of them.

No embedding model is loaded. Every operation exercised here — diagnosis, listing, counting,
resetting, backing up — is one an operator runs on an installation that is not working, and
none of them has any business loading a multi-gigabyte model to answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.container.container import build_container, check_wiring
from manicule.generation.config import GENERATOR_NAME
from manicule.plugins import ENTRY_POINT_GROUP, installed_entry_points
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import discover
from manicule.storage.config import DOC_STORE_NAME, VECTOR_STORE_NAME

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

STALE_INSTALL = (
    "an entry point declared in pyproject.toml is missing from the installed distribution. "
    "Run `uv sync --reinstall-package manicule` and try again."
)


@pytest.fixture
async def runtime(manicule_environment: Path) -> AsyncIterator[Runtime]:
    """A whole manicule over a temporary data directory."""
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        yield opened


# --- a default installation assembles ---------------------------------------------------------


def test_a_default_configuration_names_only_components_that_are_installed() -> None:
    """The check that had never been run: manicule against its own defaults.

    Until the storage plugin existed, ``storage.db = "sqlite"`` named a component nothing
    provided, so no installation could get past ``check_wiring``. Until ``llm.generator`` was
    separated from ``llm.provider``, the same was true of the generator: ``provider`` names a
    *vendor* and the registered component is one implementation that reaches all of them.
    """
    problems = check_wiring(Settings(), discover().registry)
    assert problems == []


def test_the_storage_plugin_registers_through_the_public_entry_point() -> None:
    """The same route a third-party plugin takes. There is no shorter internal one."""
    found = {point.name: point.value for point in installed_entry_points(ENTRY_POINT_GROUP)}
    assert found.get("storage") == "manicule.storage.plugin:PLUGIN", STALE_INSTALL

    registry = discover().registry
    assert registry.has(ComponentKind.DOC_STORE, DOC_STORE_NAME)
    assert registry.has(ComponentKind.VECTOR_STORE, VECTOR_STORE_NAME)


def test_the_generator_setting_names_the_component_the_plugin_registers() -> None:
    """``llm.generator``'s default and the registered name are two strings that must agree.

    They live in different packages — configuration cannot import the generation plugin
    without inverting the dependency — so this is the guard against them drifting.
    """
    assert Settings().llm.generator == GENERATOR_NAME


def test_a_container_can_be_built_from_nothing_but_defaults(manicule_environment: Path) -> None:
    """No configuration file, no environment, no arguments."""
    container = build_container(Settings(data_dir=manicule_environment / "data"))
    assert container.registry.names(ComponentKind.DOC_STORE) == [DOC_STORE_NAME]


# --- the runtime, end to end -------------------------------------------------------------------


async def test_the_database_is_migrated_before_it_is_read(runtime: Runtime) -> None:
    """A query against an un-migrated database fails at whichever statement happens to run."""
    maintenance = await runtime.maintenance()
    assert await maintenance.schema_revision() is not None


async def test_doctor_is_healthy_on_a_fresh_installation(runtime: Runtime) -> None:
    """The first thing anybody runs, on the installation they have just made."""
    diagnosis = await ApplicationService(runtime).doctor()
    failing = [check for check in diagnosis.checks if check.state == "failing"]
    assert failing == [], failing
    assert diagnosis.state == "ok"


async def test_an_empty_index_reports_itself_as_empty_rather_than_broken(
    runtime: Runtime,
) -> None:
    service = ApplicationService(runtime)
    status = await service.index_status()
    assert status.documents == 0
    assert status.embed_fingerprint is None
    assert (await service.stats()).documents == 0


async def test_the_workspace_row_exists_after_the_store_is_opened(runtime: Runtime) -> None:
    """Everything relational hangs off it, so it is created when the handle is."""
    maintenance = await runtime.maintenance()
    assert [row[0] for row in await maintenance.workspaces()] == ["default"]


async def test_resetting_an_empty_index_is_not_an_error(runtime: Runtime) -> None:
    """A reset has to be safe to run on an installation whose state nobody is sure of."""
    reset = await ApplicationService(runtime).reset_index()
    assert reset.documents == 0
    assert reset.vectors_removed is False


async def test_a_backup_is_taken_and_names_what_it_contains(
    runtime: Runtime, manicule_environment: Path
) -> None:
    report = await ApplicationService(runtime).backup(manicule_environment / "backup")
    assert report.schema_revision is not None
    assert (manicule_environment / "backup").is_dir()


async def test_the_vector_store_is_prepared_before_anything_writes_a_vector(
    manicule_environment: Path,
) -> None:
    """``ensure_ready`` is called by the runtime, and by nothing else.

    Without it the store never learns which vector space it holds: the first upsert raises,
    the document is recorded ``failed`` at the ``store`` stage, and nothing in that message
    says the index was never prepared. Both paths that touch vectors for real — ingest and
    retrieval — go through :meth:`Runtime.prepared_vectors`.

    A fake embedder, because this is about the *wiring*: the real one loads a model, and the
    question here is whether anybody calls the method at all.
    """
    from tests.fakes import HashEmbedder  # noqa: PLC0415 - a fake, local to this assertion

    found = discover()
    # Named ``local`` rather than something invented: provider names are also what the
    # credential policy reads, and anything outside the keyless set is required to carry an
    # API key. A stand-in that tripped that check would be testing the wrong thing.
    found.registry.bind("test").add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    settings = Settings(
        data_dir=manicule_environment / "data",
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
    )
    async with Runtime(settings, discovery=found) as opened:
        unprepared = await opened.vectors()
        assert await unprepared.fingerprint() is None, "the store was ready before anyone asked"

        prepared = await opened.prepared_vectors()
        held = await prepared.fingerprint()
        assert held is not None
        assert held.dimension == HashEmbedder().fingerprint.dimension


async def test_the_runtime_disposes_its_engine_on_the_way_out(
    manicule_environment: Path,
) -> None:
    """A pool left open holds file handles on the database somebody is about to restore over."""
    opened = Runtime.open(data_dir=manicule_environment / "data")
    async with opened:
        await opened.documents()
        assert opened.require_engine() is not None
    with pytest.raises(Exception, match="engine has not been opened"):
        opened.require_engine()
