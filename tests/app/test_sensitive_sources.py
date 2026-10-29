"""An installation holding mail is never served without authentication — not even on loopback.

``docs/connectors/gmail.md`` §10, mgd43b/manicule#462. The index is not permission-aware, so
what these tests defend is that **no sequence of configuration edits, flags or deletions short
of deleting the source's documents produces an unauthenticated listener over a sensitive
source**: not a loopback bind, not ``--no-authentication``, not disabling the connector, not
removing it from configuration, not soft-deleting or purging its documents.

The data-directory half runs against a real migrated SQLite database, because the refusal reads
that file directly. The connector is a synthetic ``mailbox`` type registered as sensitive: the
real ``gmail`` connector cannot sync yet, and what is under test is the mechanism rather than
Google.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.api.app import build_app
from manicule.api.serve import address_for as http_address_for
from manicule.app.bind import (
    require_sensitive_source_authentication,
    resolve_bind,
)
from manicule.app.sensitive import held_sources, mail_sources, sensitive_instances
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.container import keys
from manicule.core.errors import PolicyError
from manicule.mcp.serve import address_for as mcp_address_for
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import ComponentRegistry
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import database_path
from tests.app.fakes import FakeBackend
from tests.app.fakes import make_chunk as fake_chunk
from tests.app.fakes import make_document as fake_document
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

SENSITIVE = frozenset({"mailbox"})
MAIL = "work-mail"
"""A connector *instance* name, the identity documents carry as ``source``."""

EVERYWHERE = "0.0.0.0"  # noqa: S104 - the address these tests exist to refuse


def _settings(
    data_dir: Path, *, enabled: bool = True, configured: bool = True, authenticated: bool = False
) -> Settings:
    connectors = {MAIL: ConnectorSettings(type="mailbox", enabled=enabled)} if configured else {}
    security = {"auth": {"mode": "api_key"}} if authenticated else {}
    return Settings(data_dir=data_dir, connectors=connectors, security=security)  # pyright: ignore[reportArgumentType]


def _refused_everywhere(settings: Settings) -> None:
    """Every way of asking to serve, with every waiver there is, is refused."""
    for kwargs in (
        {},
        {"host": EVERYWHERE, "allow_public": True, "allow_unauthenticated": True},
        {"allow_unauthenticated": True},
    ):
        with pytest.raises(PolicyError, match=repr(MAIL)):
            resolve_bind(settings, sensitive_types=SENSITIVE, **kwargs)  # pyright: ignore[reportArgumentType]


# --- the registry says which types are sensitive ---------------------------------------------


def test_only_the_mail_connectors_are_registered_as_sensitive() -> None:
    from manicule.plugins import discover  # noqa: PLC0415

    registry = discover().registry
    mail = {"gmail", "mbox"}
    assert registry.sensitive_names(ComponentKind.CONNECTOR) == mail
    assert not any(record.sensitive for record in registry.records() if record.name not in mail)


def test_a_component_is_not_sensitive_unless_it_says_so() -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.CONNECTOR.named("plain"), lambda _context: object())
    registry.add(keys.CONNECTOR.named("mailbox"), lambda _context: object(), sensitive=True)
    assert registry.sensitive_names(ComponentKind.CONNECTOR) == {"mailbox"}


# --- criterion 1: an enabled sensitive instance refuses, on loopback and wide ----------------


def test_an_enabled_sensitive_instance_is_refused_on_loopback_and_wide(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _refused_everywhere(settings)
    with pytest.raises(PolicyError, match=repr(MAIL)) as caught:
        resolve_bind(settings, sensitive_types=SENSITIVE)
    assert "security.auth.mode" in str(caught.value)
    assert "disable or remove the connector" in str(caught.value)
    assert "delete-source" not in str(caught.value)  # nothing is indexed yet to delete


def test_authentication_lifts_the_configuration_half(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data", authenticated=True)
    assert resolve_bind(settings, sensitive_types=SENSITIVE).loopback
    assert not resolve_bind(
        settings, sensitive_types=SENSITIVE, host=EVERYWHERE, allow_public=True
    ).loopback


def test_a_type_nobody_registered_as_sensitive_is_served_as_before(tmp_path: Path) -> None:
    assert resolve_bind(_settings(tmp_path / "data"), sensitive_types=frozenset()).loopback


def test_a_disabled_instance_with_nothing_indexed_does_not_refuse(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data", enabled=False)
    assert resolve_bind(settings, sensitive_types=SENSITIVE).loopback


# --- criterion 4: no flag overrides it --------------------------------------------------------


def test_no_authentication_does_not_override_the_refusal(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with pytest.raises(PolicyError, match="--no-authentication was passed and does not apply"):
        resolve_bind(
            settings,
            sensitive_types=SENSITIVE,
            host=EVERYWHERE,
            allow_public=True,
            allow_unauthenticated=True,
        )
    with pytest.raises(PolicyError, match="does not apply"):
        require_sensitive_source_authentication(settings, SENSITIVE, allow_unauthenticated=True)


def test_the_refusal_has_no_parameter_that_overrides_it() -> None:
    """The only argument naming a waiver is the one the refusal reports as ignored."""
    import inspect  # noqa: PLC0415

    names = set(inspect.signature(require_sensitive_source_authentication).parameters)
    assert names == {"settings", "sensitive_types", "allow_unauthenticated"}
    # `sensitive_types` has no default: omitting it must be a type error, not "nothing sensitive".
    assert (
        inspect.signature(resolve_bind).parameters["sensitive_types"].default
        is inspect.Parameter.empty
    )


# --- criterion 2: disabling or removing the connector does not lift it; deleting does --------


async def _synced_mail(engine: AsyncEngine, *documents: str) -> SqliteDocStore:
    """A store in which the mail source has been marked and holds ``documents``."""
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.mark_source_sensitive(MAIL)
    for source_id in documents:
        await store.upsert_document(make_document(MAIL, source_id, title="Subject"))
    return store


async def test_a_marked_source_with_documents_refuses_after_the_connector_is_removed(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    # Not configured at all: the configuration half sees nothing.
    settings = _settings(data_dir, configured=False)
    assert held_sources(data_dir) == (MAIL,)
    _refused_everywhere(settings)
    with pytest.raises(PolicyError, match=repr(MAIL)) as caught:
        resolve_bind(settings, sensitive_types=SENSITIVE)
    assert "manicule document delete-source <name> --yes" in str(caught.value)


async def test_a_disabled_connector_still_refuses_while_its_documents_remain(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    _refused_everywhere(_settings(data_dir, enabled=False))


async def test_a_soft_deleted_document_still_counts(engine: AsyncEngine, data_dir: Path) -> None:
    store = await _synced_mail(engine, "m1")
    await store.soft_delete_document(make_document(MAIL, "m1").id)
    assert held_sources(data_dir) == (MAIL,)
    _refused_everywhere(_settings(data_dir, configured=False))


async def test_a_purged_row_still_counts(engine: AsyncEngine, data_dir: Path) -> None:
    store = await _synced_mail(engine, "m1")
    document_id = make_document(MAIL, "m1").id
    await store.soft_delete_document(document_id)
    assert await store.purge_document(document_id, reason="purged")
    assert held_sources(data_dir) == (MAIL,)
    _refused_everywhere(_settings(data_dir, configured=False))


async def test_deleting_every_document_from_the_source_lifts_the_refusal(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = await _synced_mail(engine, "m1", "m2", "m3")
    await store.soft_delete_document(make_document(MAIL, "m1").id)
    settings = _settings(data_dir, configured=False)
    _refused_everywhere(settings)

    assert await store.delete_source_documents(MAIL) == 3

    assert held_sources(data_dir) == ()
    assert resolve_bind(settings, sensitive_types=SENSITIVE).loopback


async def test_deleting_one_by_one_lifts_it_only_with_the_last(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = await _synced_mail(engine, "m1", "m2")
    await store.delete_document(make_document(MAIL, "m1").id)
    assert held_sources(data_dir) == (MAIL,)
    await store.delete_document(make_document(MAIL, "m2").id)
    assert held_sources(data_dir) == ()


async def test_a_source_no_sensitive_connector_synced_is_not_held(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Documents alone are not the trigger; the mark is. Wikis and files serve as before."""
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.upsert_document(make_document("wiki", "p1"))
    assert held_sources(data_dir) == ()
    assert resolve_bind(_settings(data_dir, configured=False), sensitive_types=SENSITIVE).loopback


async def test_a_mark_with_no_documents_is_not_held(engine: AsyncEngine, data_dir: Path) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.mark_source_sensitive(MAIL)
    assert held_sources(data_dir) == ()


async def test_the_mark_is_stamped_once_and_a_second_sync_leaves_it_alone(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = await _synced_mail(engine)
    first = await _stamp(engine)
    await store.mark_source_sensitive(MAIL)
    assert await _stamp(engine) == first


async def _stamp(engine: AsyncEngine) -> object:
    from sqlalchemy import select  # noqa: PLC0415

    from manicule.storage import models  # noqa: PLC0415
    from manicule.storage.engine import session_factory  # noqa: PLC0415

    async with session_factory(engine)() as session:
        return (
            await session.execute(
                select(models.Connector.sensitive_synced_at).where(models.Connector.name == MAIL)
            )
        ).scalar_one()


# --- an unreadable data directory ------------------------------------------------------------


def test_no_database_yet_holds_nothing(tmp_path: Path) -> None:
    assert held_sources(tmp_path / "never-created") == ()
    assert resolve_bind(
        _settings(tmp_path / "never-created", configured=False), sensitive_types=SENSITIVE
    ).loopback


def test_a_database_from_before_the_mark_holds_nothing(tmp_path: Path) -> None:
    """The columns the query reads do not exist, so there is nothing it could find."""
    tmp_path.mkdir(exist_ok=True)
    connection = sqlite3.connect(database_path(tmp_path))
    connection.execute("CREATE TABLE connectors (id TEXT, workspace_id TEXT, name TEXT)")
    connection.commit()
    connection.close()
    assert held_sources(tmp_path) == ()


def test_a_database_that_cannot_be_read_refuses_rather_than_passes(tmp_path: Path) -> None:
    path = database_path(tmp_path)
    path.write_bytes(b"this is not a sqlite database" * 100)
    settings = _settings(tmp_path, configured=False)
    with pytest.raises(PolicyError, match="could not be read"):
        resolve_bind(settings, sensitive_types=SENSITIVE)


def test_an_unreadable_database_does_not_matter_when_authentication_is_on(tmp_path: Path) -> None:
    database_path(tmp_path).write_bytes(b"not a database" * 100)
    settings = _settings(tmp_path, configured=False, authenticated=True)
    assert resolve_bind(settings, sensitive_types=SENSITIVE).loopback


# --- the application and the other servers refuse for the same reason -------------------------


async def test_build_app_refuses_over_documents_a_removed_connector_left(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    service = ApplicationService(FakeBackend(settings=_settings(data_dir, configured=False)))
    with pytest.raises(PolicyError, match=repr(MAIL)):
        build_app(service)
    with pytest.raises(PolicyError, match="does not apply"):
        build_app(service, allow_unauthenticated=True)


async def test_both_socket_servers_refuse_over_mail_with_every_flag_passed(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    service = ApplicationService(FakeBackend(settings=_settings(data_dir, configured=False)))
    with pytest.raises(PolicyError, match=repr(MAIL)):
        http_address_for(service, allow_public=True, allow_unauthenticated=True)
    with pytest.raises(PolicyError, match=repr(MAIL)):
        mcp_address_for(service, transport="http", allow_public=True, allow_unauthenticated=True)


async def test_stdio_opens_no_socket_so_it_is_not_refused(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    service = ApplicationService(FakeBackend(settings=_settings(data_dir, configured=False)))
    assert mcp_address_for(service, transport="stdio").transport == "stdio"


# --- criterion 3: doctor fails the check, and the commands that remedy it still run -----------


async def test_doctor_fails_transport_without_serving(engine: AsyncEngine, data_dir: Path) -> None:
    await _synced_mail(engine, "m1")
    service = ApplicationService(FakeBackend(settings=_settings(data_dir, configured=False)))
    diagnosis = await service.doctor()
    transport = next(c for c in diagnosis.checks if c.name == "transport")
    assert transport.state == "failing"
    assert transport.facts["sensitive_sources"] == [MAIL]
    assert "loopback" in transport.detail
    assert "--no-authentication does not change that" in transport.detail


async def test_doctor_is_quiet_about_it_with_authentication_on_or_nothing_held(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _synced_mail(engine, "m1")
    authenticated = ApplicationService(
        FakeBackend(settings=_settings(data_dir, configured=False, authenticated=True))
    )
    check = next(c for c in (await authenticated.doctor()).checks if c.name == "transport")
    assert check.state != "failing"
    assert "sensitive_sources" not in check.facts

    empty = ApplicationService(
        FakeBackend(settings=_settings(data_dir / "elsewhere", configured=False))
    )
    check = next(c for c in (await empty.doctor()).checks if c.name == "transport")
    assert check.state == "ok"
    assert check.facts["sensitive_sources"] == []


async def test_commands_that_open_no_socket_still_run(engine: AsyncEngine, data_dir: Path) -> None:
    """The remedy stays reachable: none of these goes through the bind rule."""
    store = await _synced_mail(engine, "m1")
    backend = FakeBackend(settings=_settings(data_dir, configured=False))
    service = ApplicationService(backend)
    document = fake_document(backend.workspace, source=MAIL)
    backend.store.add(document, fake_chunk(document))

    outcome = await service.document_delete(document.id, hard=True)
    assert outcome.deleted
    assert (await service.document_delete_source(MAIL)).source == MAIL
    # And the real store the rule reads is what the same call empties.
    assert await store.delete_source_documents(MAIL) == 1
    assert held_sources(data_dir) == ()


# --- which instances the runtime marks ---------------------------------------------------------


def test_every_configured_instance_of_a_sensitive_type_is_marked_enabled_or_not(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        connectors={
            "a": ConnectorSettings(type="mailbox"),
            "b": ConnectorSettings(type="mailbox", enabled=False),
            "c": ConnectorSettings(type="filesystem"),
        },
    )
    assert sensitive_instances(settings, SENSITIVE) == {"a", "b"}


@pytest.mark.parametrize("directory", ["with#hash", "with%percent", "with space"])
async def test_a_data_directory_whose_path_has_uri_metacharacters_is_still_read(
    tmp_path: Path, migrated_template: Path, directory: str
) -> None:
    """``#`` and ``?`` end a SQLite URI's path, so an unescaped one opens a different database.

    That database is empty, its "no such table" reads as "holds nothing", and the mailbox in the
    real one would be served. Each of these must find the mail.
    """
    import shutil  # noqa: PLC0415

    from manicule.storage.engine import create_engine, prepare_data_dir  # noqa: PLC0415

    root = tmp_path / directory
    prepare_data_dir(root)
    shutil.copy2(migrated_template, database_path(root))
    built = create_engine(root)
    try:
        await _synced_mail(built, "m1")
    finally:
        await built.dispose()

    assert held_sources(root) == (MAIL,)


def test_the_runtime_names_the_gmail_instances_it_hands_the_pipeline(tmp_path: Path) -> None:
    """The pipeline marks only what the runtime names, so this is where a gap would fail open."""
    from manicule.app.runtime import Runtime  # noqa: PLC0415

    settings = Settings(
        data_dir=tmp_path,
        connectors={
            "mail": ConnectorSettings(type="gmail"),
            "notes": ConnectorSettings(type="filesystem"),
        },
    )
    assert Runtime(settings).sensitive_instances() == {"mail"}


async def test_doctor_redacts_the_home_directory_in_an_unreadable_database_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`doctor` output is pasted into issues, so the account name stays out of it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database_path(data_dir).write_bytes(b"not a database" * 100)
    service = ApplicationService(FakeBackend(settings=_settings(data_dir, configured=False)))

    check = next(c for c in (await service.doctor()).checks if c.name == "transport")

    assert check.state == "failing"
    assert str(tmp_path) not in check.detail
    assert "~/data/manicule.db" in check.detail


def test_the_database_name_agrees_with_the_storage_engines() -> None:
    from manicule.app.sensitive import DATABASE_FILENAME  # noqa: PLC0415

    assert database_path(Path("data")).name == DATABASE_FILENAME


@pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX modes and a user the mode binds",
)
def test_a_database_that_cannot_be_examined_refuses_rather_than_reading_as_absent(
    tmp_path: Path,
) -> None:
    """``Path.exists`` is False for a permission error; absent and unreadable are different."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sqlite3.connect(database_path(data_dir)).close()
    data_dir.chmod(0o000)
    try:
        with pytest.raises(PolicyError, match="could not be read"):
            resolve_bind(_settings(data_dir, configured=False), sensitive_types=SENSITIVE)
    finally:
        data_dir.chmod(0o700)


# --- mail is sensitive by what it is, whichever connector indexed it (mbox.md §7) -----------------


async def _indexed(
    engine: AsyncEngine, source: str, source_id: str, media_type: str
) -> SqliteDocStore:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.upsert_document(make_document(source, source_id, media_type=media_type))
    return store


@pytest.mark.parametrize("media_type", ["message/rfc822", "application/vnd.ms-outlook"])
async def test_a_source_holding_a_mail_document_refuses_whichever_connector_wrote_it(
    engine: AsyncEngine, data_dir: Path, media_type: str
) -> None:
    await _indexed(engine, "files", "letter", media_type)
    settings = _settings(data_dir, configured=False)

    assert held_sources(data_dir) == (), "no sensitive connector marked it"
    assert mail_sources(data_dir) == ("files",)
    with pytest.raises(PolicyError, match="holds mail") as caught:
        resolve_bind(settings, sensitive_types=SENSITIVE)
    assert "'files'" in str(caught.value)
    assert "delete-source" in str(caught.value)
    _refused_everywhere_with(settings, "files")


def _refused_everywhere_with(settings: Settings, source: str) -> None:
    with pytest.raises(PolicyError, match=repr(source)):
        resolve_bind(
            settings,
            sensitive_types=SENSITIVE,
            host=EVERYWHERE,
            allow_public=True,
            allow_unauthenticated=True,
        )


async def test_a_member_of_an_archive_with_a_mail_type_counts(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """An ``.eml`` inside a ``.zip`` is a row with its own media type, and is found."""
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    container = make_document("files", "bundle.zip", media_type="application/zip")
    await store.upsert_document(container)
    await store.upsert_document(
        make_document(
            "files",
            "zip:bundle.zip!/letter.eml",
            media_type="message/rfc822",
            container_id=container.id,
            container_depth=1,
        )
    )
    assert mail_sources(data_dir) == ("files",)


async def test_deleting_the_source_lifts_it_and_authentication_overrides_it(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = await _indexed(engine, "files", "letter", "message/rfc822")
    assert mail_sources(data_dir) == ("files",)
    assert resolve_bind(
        _settings(data_dir, configured=False, authenticated=True), sensitive_types=SENSITIVE
    ).loopback

    assert await store.delete_source_documents("files") == 1

    assert mail_sources(data_dir) == ()
    assert resolve_bind(_settings(data_dir, configured=False), sensitive_types=SENSITIVE).loopback


async def test_a_trashed_mail_document_still_counts(engine: AsyncEngine, data_dir: Path) -> None:
    store = await _indexed(engine, "files", "letter", "message/rfc822")
    await store.soft_delete_document(make_document("files", "letter").id)
    assert mail_sources(data_dir) == ("files",)


async def test_documents_that_are_not_mail_do_not_refuse(
    engine: AsyncEngine, data_dir: Path
) -> None:
    await _indexed(engine, "files", "page", "text/markdown")
    assert mail_sources(data_dir) == ()
    assert resolve_bind(_settings(data_dir, configured=False), sensitive_types=SENSITIVE).loopback


def test_doctor_reports_the_source_that_holds_mail() -> None:
    from manicule.app.sensitive import SensitiveExposure  # noqa: PLC0415

    exposure = SensitiveExposure(mail=("files",))
    assert exposure
    assert exposure.sources == ("files",)
