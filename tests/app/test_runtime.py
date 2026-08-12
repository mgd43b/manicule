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

import os
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
from manicule.storage.backup import BackupError
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are what is being checked")
async def test_a_backup_into_a_pre_existing_exposed_directory_is_refused_end_to_end(
    runtime: Runtime, manicule_environment: Path
) -> None:
    """The whole stack, not a fake of it: service, runtime, storage, one real ``stat``.

    ``docs/storage.md`` §7.1 promised this refusal and nothing implemented it (#60), which is
    the kind of gap only a test that goes all the way down can close. The directory exists
    *before* the call, because that is the case ``mkdir(mode=0o700, exist_ok=True)`` never
    reaches.
    """
    target = manicule_environment / "shared"
    target.mkdir()
    target.chmod(0o755)
    service = ApplicationService(runtime)

    with pytest.raises(BackupError, match="group or other permissions") as refusal:
        await service.backup(target)
    assert str(target) in str(refusal.value)
    assert not any(target.iterdir()), "a refused backup leaves the corpus where it was"

    report = await service.backup(target, allow_insecure_target=True)
    assert report.files, "the escape hatch is an escape hatch, not a second refusal"


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


# --- the implementations the HTTP surface added, against a real database ----------------------


async def test_the_runtime_satisfies_every_port_the_service_asks_for(runtime: Runtime) -> None:
    """Structurally, at run time, against the real objects.

    The service is written against protocols, and every suite that drives it drives fakes of
    them. That proves the *service*; it proves nothing about the objects a running manicule
    actually hands it. A missing method here would be an ``AttributeError`` from inside a
    route on somebody's installation, and green everywhere in this repository.
    """
    from manicule.app.ports import (  # noqa: PLC0415 - only this assertion needs them
        Conversing,
        DocumentSurface,
        Keys,
        Organising,
        Telemetry,
    )

    assert isinstance(await runtime.documents(), DocumentSurface)
    assert isinstance(await runtime.organisation(), Organising)
    assert isinstance(await runtime.conversations(), Conversing)
    assert isinstance(await runtime.telemetry(), Telemetry)
    assert isinstance(await runtime.keys(), Keys)


async def test_the_conversation_store_is_one_handle_rather_than_two(runtime: Runtime) -> None:
    """The answer path and the surfaces share it.

    Two stores over one engine would be two places a share link can be minted from and two
    opinions about what has been deleted.
    """
    assert await runtime.conversations() is await runtime.conversations()


async def test_a_conversation_round_trips_through_the_real_store(runtime: Runtime) -> None:
    """Create, list, read, rename, delete — the SQL the surface depends on, executed.

    ``list_conversations`` and ``get_conversation`` were written for this surface, and their
    correlated subquery has never run anywhere else in the suite.
    """
    store = await runtime.conversations()
    identifier = await store.create_conversation(title="First")
    listed = await store.list_conversations()
    assert [record.id for record in listed] == [identifier]
    assert listed[0].messages == 0

    assert await store.rename_conversation(identifier, "Renamed")
    record = await store.get_conversation(identifier)
    assert record is not None
    assert record.title == "Renamed"

    assert await store.soft_delete_conversation(identifier)
    assert await store.get_conversation(identifier) is None
    assert await store.list_conversations() == []
    assert not await store.rename_conversation(identifier, "Again"), (
        "a deleted conversation was renamed, so the soft-delete predicate is missing"
    )


async def test_query_telemetry_round_trips_through_the_real_store(runtime: Runtime) -> None:
    """The SQL behind ``GET /admin/query-logs``, executed rather than faked."""
    telemetry = await runtime.telemetry()
    await telemetry.record_query(
        "retry policy", profile="balanced", chunk_ids=["a", "b"], confidence=0.5, elapsed_ms=12
    )
    rows, total = await telemetry.query_logs(limit=10)
    assert total == 1
    assert rows[0]["query"] == "retry policy"
    assert rows[0]["chunks"] == 2, "the chunk ids were stored but not counted back"
    assert rows[0]["confidence"] == 0.5


async def test_the_audit_trail_round_trips_and_filters(runtime: Runtime) -> None:
    """Including the ``event_type`` filter, which is a second statement nothing else runs."""
    telemetry = await runtime.telemetry()
    await telemetry.record_audit("conversation.shared", details={"conversation_id": "c1"})
    await telemetry.record_audit("api_key.created", details={"name": "widget"})

    rows, total = await telemetry.audit_logs(limit=10)
    assert total == 2
    assert [row["event_type"] for row in rows] == ["api_key.created", "conversation.shared"], (
        "the audit trail is not newest-first"
    )
    filtered, count = await telemetry.audit_logs(event_type="api_key.created")
    assert count == 1
    assert filtered[0]["details"] == {"name": "widget"}


async def test_an_issued_key_verifies_and_a_revoked_one_stops(runtime: Runtime) -> None:
    """``verify`` is what every authenticated request runs, and it is four predicates in SQL.

    Nothing else in the suite executes it. A wrong column name would be an authentication path
    that always refuses — or, worse, one whose workspace predicate silently does nothing.
    """
    store = await runtime.keys()
    summary, secret = await store.issue("widget", role="member")

    verified = await store.verify(secret)
    assert verified is not None
    assert verified.id == summary.id
    assert verified.role == "member"
    assert verified.workspace == runtime.workspace

    assert await store.verify("mnk_not_a_key") is None
    assert await store.verify("") is None

    await store.revoke("widget")
    assert await store.verify(secret) is None, "a revoked key still authenticates"


async def test_a_key_minted_in_one_workspace_does_not_authenticate_in_another(
    manicule_environment: Path,
) -> None:
    """The predicate that makes a key an identity **in a tenant** rather than in an install.

    Two runtimes over the **same data directory**, differing only in workspace — which is what
    a second workspace on one machine actually is. Without this, the workspace clause in
    ``verify`` could be deleted and every other test in this repository would still pass: the
    fixtures only ever hold one tenant, so there is no foreign key to present.

    A key that resolved across the boundary would let anybody holding one workspace's
    credential read another's corpus, which is the whole thing the design is for.
    """
    data_dir = manicule_environment / "data"
    async with Runtime.open(data_dir=data_dir, workspace="alpha") as alpha:
        _, secret = await (await alpha.keys()).issue("shared-name", role="admin")
        assert await (await alpha.keys()).verify(secret) is not None, "the control failed"

    async with Runtime.open(data_dir=data_dir, workspace="beta") as beta:
        assert await (await beta.keys()).verify(secret) is None, (
            "a key minted in 'alpha' authenticated in 'beta'"
        )
        # And the listing does not show it either, so the two are separate in both directions.
        assert await (await beta.keys()).list_keys() == []


async def test_an_expired_key_stops_verifying(runtime: Runtime) -> None:
    """The expiry predicate, driven by writing a past expiry the issuing path cannot produce.

    ``issue`` only ever writes a future one, so without reaching into the row this predicate
    would be present and never evaluated — which is indistinguishable from absent.
    """
    from datetime import timedelta  # noqa: PLC0415 - only this test needs them

    from sqlalchemy import update  # noqa: PLC0415

    from manicule.storage import models  # noqa: PLC0415
    from manicule.storage.engine import session_factory  # noqa: PLC0415
    from manicule.storage.types import utcnow  # noqa: PLC0415

    store = await runtime.keys()
    summary, secret = await store.issue("temporary", role="viewer", expires_days=30)
    assert await store.verify(secret) is not None

    sessions = session_factory(runtime.require_engine())
    async with sessions.begin() as session:
        await session.execute(
            update(models.ApiKey)
            .where(models.ApiKey.id == summary.id)
            .values(expires_at=utcnow() - timedelta(seconds=1))
        )
    assert await store.verify(secret) is None, "an expired key still authenticates"


async def test_collections_and_tags_reach_the_real_store(runtime: Runtime) -> None:
    """The organisation handle is the document store, and the surface needs both halves."""
    store = await runtime.organisation()
    collection = await store.create_collection("Runbooks", description="Operational")
    assert [item.id for item in await store.list_collections()] == [collection.id]

    tag = await store.ensure_tag("urgent")
    assert (await store.ensure_tag("urgent")).id == tag.id, "ensure_tag is not idempotent"
    assert [item.id for item in await store.list_tags()] == [tag.id]

    await store.delete_collection(collection.id)
    assert await store.list_collections() == []
