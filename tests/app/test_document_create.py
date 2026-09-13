"""Authoring a document: identity, bounds, and what happens when the index declines.

``document_create`` is the one operation that puts content **into** a corpus, so most of what is
asserted here is about what it will not do. The path it writes to is derived from a collection
and a slug and never supplied; the collection has to be one configuration names; the file is
kept when indexing fails; an existing slug is refused rather than replaced.

**The ingest surface is faked and the filesystem is real.** A fake connector would make the
containment assertions true by construction — the thing under test is where bytes land on a
disk — so the connector is built through the real factory over a real ``tmp_path`` root, and
what is faked is the pipeline behind it, which would otherwise need an embedder to answer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app import commands
from manicule.app.commands import Command
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import (
    AuthMode,
    AuthoringSettings,
    AuthSettings,
    ConnectorSettings,
    SecuritySettings,
    Settings,
)
from manicule.connectors.plugin import ConnectorsPlugin
from manicule.container import Container
from manicule.core.content import DocumentStatus
from manicule.core.errors import ConfigError, PolicyError, UnknownEntityError
from manicule.core.ids import document_id
from manicule.ingest.pipeline import RunReport
from manicule.mcp.server import build_server
from manicule.plugins.registry import ComponentRegistry
from tests.app.fakes import FakeBackend, FakeIngestion, FakeStore, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.app.ports import Watching
    from manicule.app.results import Envelope

WORKSPACE = "default"
SOURCE = "memories"
COLLECTION = "memory"

BODY = """\
---
name: project_retry_policy
description: The client retries twice, then gives up.
metadata:
  node_type: memory
  type: project
---

The client retries twice. See [[project_backoff]].
"""


class IndexingIngestion(FakeIngestion):
    """A fake ingest that also **stores** what it was asked to index.

    ``FakeIngestion`` records the path and reports a run; nothing appears in the document store,
    which is right for every caller that only wants to know an ingest was started. Authoring is
    the caller that reads the store afterwards to decide whether the document is published, so
    against the plain fake it would report ``indexed: false`` for every successful write — a
    test passing for the reason the feature fails.

    ``fails`` is the other half: an ingest that runs and stores nothing, which is the partial
    failure §4.5 of the design is about.
    """

    def __init__(self, store: FakeStore, *, fails: bool = False) -> None:
        super().__init__()
        self.store = store
        self.fails = fails

    @override
    async def index_path(
        self,
        path: Path,
        *,
        name: str,
        limit: int | None = None,
        force: bool = False,
        watching: Watching | None = None,
    ) -> RunReport:
        del limit, force, watching
        self.paths.append(path)
        if self.fails:
            return RunReport(connector=name, discovered=1, error="the embedder declined")
        document = make_document(
            WORKSPACE,
            source=name,
            source_id=str(path),
            title=path.stem,
            status=DocumentStatus.INDEXED,
        )
        self.store.add(document, make_chunk(document))
        return RunReport(connector=name, discovered=1, by_status={DocumentStatus.INDEXED.value: 1})


def settings_for(root: Path, *, collections: tuple[str, ...] = (COLLECTION,)) -> Settings:
    """Configuration declaring one filesystem source and authoring into it.

    Authentication is on, because ``manicule.app.bind.require_authoring_authentication`` refuses
    to build an application that would serve configured authoring without it — every bind,
    loopback included. Carried by the shared builder rather than by the one column that needs it,
    so this fixture is a configuration somebody could actually run.
    """
    return Settings(
        connectors={SOURCE: ConnectorSettings(type="filesystem", options={"root": str(root)})},
        authoring=AuthoringSettings(source=SOURCE, collections=collections),
        security=SecuritySettings(auth=AuthSettings(mode=AuthMode.API_KEY)),
    )


async def backend_for(
    settings: Settings, *, fails: bool = False, existing: Sequence[str] | None = None
) -> FakeBackend:
    """A backend whose connector comes out of the **real** container.

    The real registry, so ``FilesystemConfig`` validation and ``build_filesystem`` are the ones
    the product runs and the root this writes beneath is the root a sync would read.
    """
    backend = FakeBackend(settings=settings)
    backend.ingestion_ = IndexingIngestion(backend.store, fails=fails)
    registry = ComponentRegistry().bind("connectors")
    ConnectorsPlugin().register(registry)
    container = Container(settings, registry)
    for name, configured in settings.connectors.items():
        if configured.type == "filesystem":
            backend.ingestion_.connectors[name] = await container.connector(name)
    # The workspace's collections, which default to the configured ones and are separable from
    # them on purpose: configuration naming a collection nobody created is a real state, and it
    # is the one this fixture would otherwise make unreachable.
    for name in settings.authoring.collections if existing is None else existing:
        await backend.organization_.create_collection(name)
    return backend


async def service_for(
    settings: Settings, *, fails: bool = False, existing: Sequence[str] | None = None
) -> ApplicationService:
    return ApplicationService(await backend_for(settings, fails=fails, existing=existing))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The corpus directory, a level below ``tmp_path``.

    Its own directory because ``manicule_environment`` builds ``home/`` and ``work/`` directly
    under ``tmp_path`` for every test, and "the refusal wrote nothing" is only a statement worth
    making about a directory nothing else writes to.
    """
    made = tmp_path / "corpus"
    made.mkdir()
    return made


def written_under(root: Path) -> list[str]:
    """Every path under ``root``, relative and sorted.

    A list of names rather than a bare emptiness check, so a refusal that left something behind
    fails saying *what* it left — which is the whole question when the thing being asserted is
    that a traversal wrote nothing.
    """
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


# --- the ordinary case -------------------------------------------------------------------------


async def test_a_created_document_is_on_disk_indexed_and_in_its_collection(root: Path) -> None:
    """All three, because they are three writes and any of them can be the one that did not run.

    The file is the record; the index is what makes it findable; the membership is what makes a
    scoped search find it. A result reporting success while the document sat outside every
    collection would be exactly the unscoped pile taking a collection on this call prevents.
    """
    service = await service_for(settings_for(root))

    created = await service.document_create(collection=COLLECTION, slug="retry_policy", body=BODY)

    written = root / COLLECTION / "retry_policy.md"
    assert written.read_text(encoding="utf-8") == BODY
    assert created.path == str(written)
    assert created.indexed is True
    assert created.member is True
    assert created.overwritten is False
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    assert backend.organization_.members["col-0"] == [created.document_id]


async def test_the_identity_is_the_slug_so_a_new_title_does_not_move_it(root: Path) -> None:
    """The reason identity is the slug rather than the title, asserted rather than asserted about.

    A retitled document keeps every citation, every inbound link and every collection membership
    that names it. Derived here the way the product derives it, so the test cannot be made to
    pass by editing a literal until it matches.
    """
    service = await service_for(settings_for(root))
    first = await service.document_create(collection=COLLECTION, slug="retry_policy", body=BODY)

    retitled = BODY.replace("name: project_retry_policy", "name: project_retry_policy_v2").replace(
        "The client retries twice, then gives up.", "Something else entirely."
    )
    second = await service.document_create(
        collection=COLLECTION, slug="retry_policy", body=retitled, overwrite=True
    )

    assert second.document_id == first.document_id
    assert second.document_id == document_id(
        WORKSPACE, SOURCE, str(root / COLLECTION / "retry_policy.md")
    )
    assert second.overwritten is True
    assert (root / COLLECTION / "retry_policy.md").read_text(encoding="utf-8") == retitled


async def test_a_body_with_no_trailing_newline_gets_one_and_nothing_else(root: Path) -> None:
    """The one edit made to what a caller sent, and it is a property of the file not the content.

    Asserted because "manicule does not author front matter" is a claim about *content*, and a
    reader could reasonably wonder how far it goes. This far: a final newline, which git and
    every other reader of the corpus treat as a difference, and no other byte.
    """
    service = await service_for(settings_for(root))
    await service.document_create(collection=COLLECTION, slug="terse", body="# Terse\n\nOne line.")
    assert (root / COLLECTION / "terse.md").read_text(encoding="utf-8") == (
        "# Terse\n\nOne line.\n"
    )


# --- the bounds --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "../escape",
        "nested/slug",
        "/absolute",
        "..",
        ".",
        "",
        "trailing/",
        "back\\slash",
    ],
)
async def test_a_slug_that_is_not_one_path_component_is_refused(root: Path, slug: str) -> None:
    """Refused, and **nothing is written** — including nothing outside the root.

    The second half is what makes this more than an argument check. A refusal that happened
    after the directory had been created, or after a file had been opened, would leave the
    traversal's footprint on disk while reporting that it had not happened.
    """
    service = await service_for(settings_for(root))
    with pytest.raises(ValueError, match="slug"):
        await service.document_create(collection=COLLECTION, slug=slug, body=BODY)
    assert written_under(root) == []


async def test_authoring_is_refused_until_it_is_configured(root: Path) -> None:
    """The default, and it is off. An installation that configured nothing has no authoring.

    Named as the reason no surface needs a switch of its own: the tool is published over stdio
    and over a socket, and on an installation that never set these two values it writes nothing
    anywhere.
    """
    settings = Settings(
        connectors={SOURCE: ConnectorSettings(type="filesystem", options={"root": str(root)})}
    )
    service = await service_for(settings)
    with pytest.raises(ConfigError, match=r"authoring\.source"):
        await service.document_create(collection=COLLECTION, slug="anything", body=BODY)
    assert written_under(root) == []


async def test_a_collection_outside_the_configured_set_is_refused_even_when_it_exists(
    root: Path,
) -> None:
    """Creating a collection must not also be the act of granting write access to it.

    The collection here is real — the workspace holds it — and authoring still refuses, because
    the scope is the configured list rather than the workspace's contents. A check that resolved
    the name first and asked about configuration second would pass this test for a corpus and
    fail it for the one document somebody cared about.
    """
    service = await service_for(settings_for(root))
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    await backend.organization_.create_collection("secrets")

    with pytest.raises(PolicyError, match=r"authoring\.collections"):
        await service.document_create(collection="secrets", slug="leak", body=BODY)
    assert written_under(root) == []


async def test_a_configured_collection_the_workspace_does_not_have_refuses_before_writing(
    root: Path,
) -> None:
    """Configuration naming a collection nobody created is an error, not a directory to make.

    Writing the file and failing to file it would leave content on disk in a collection that
    does not exist — findable by a later sync, in nothing.
    """
    service = await service_for(
        settings_for(root, collections=(COLLECTION, "absent")), existing=[COLLECTION]
    )
    with pytest.raises(UnknownEntityError, match="absent"):
        await service.document_create(collection="absent", slug="orphan", body=BODY)
    assert written_under(root) == []


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("", "an empty document has nothing to index or cite"),
        ("   \n\n  ", "whitespace is empty"),
        ("---\nname: unterminated\n\nProse.\n", "the front-matter fence never closes"),
    ],
)
async def test_a_body_that_would_not_parse_as_intended_is_refused(
    root: Path, body: str, why: str
) -> None:
    """Validated, never authored. Nothing here invents front matter; it refuses what cannot work.

    The unterminated fence is the case worth having: left alone it parses, silently, into a
    document whose first heading is a line of YAML and whose every heading path hangs beneath it.
    """
    service = await service_for(settings_for(root))
    with pytest.raises(ValueError, match=r"empty|fence"):
        await service.document_create(collection=COLLECTION, slug="broken", body=body)
    assert written_under(root) == [], why


# --- overwriting -------------------------------------------------------------------------------


async def test_an_existing_slug_is_refused_and_the_refusal_names_what_holds_it(root: Path) -> None:
    """Named, so the caller can read the existing fact and revise it rather than guess.

    The content is asserted unchanged as well as the refusal, because "refused" and "refused
    after writing" read identically in an exception.
    """
    service = await service_for(settings_for(root))
    first = await service.document_create(collection=COLLECTION, slug="retry_policy", body=BODY)

    with pytest.raises(PolicyError) as refusal:
        await service.document_create(
            collection=COLLECTION, slug="retry_policy", body="# Replaced\n"
        )

    assert first.document_id in str(refusal.value)
    assert "overwrite" in str(refusal.value)
    assert (root / COLLECTION / "retry_policy.md").read_text(encoding="utf-8") == BODY


async def test_a_file_nothing_has_indexed_still_counts_as_taking_the_slug(root: Path) -> None:
    """The half a document lookup alone would miss, and it is the dangerous half.

    A file put there by hand, by git, or by a write whose ingest failed is content manicule has
    never seen — so an overwrite guard that only asked the index would destroy precisely the
    content nothing else knows about.
    """
    service = await service_for(settings_for(root))
    (root / COLLECTION).mkdir()
    (root / COLLECTION / "byhand.md").write_text("# Written by hand\n", encoding="utf-8")

    with pytest.raises(PolicyError, match="not indexed"):
        await service.document_create(collection=COLLECTION, slug="byhand", body=BODY)
    assert (root / COLLECTION / "byhand.md").read_text(encoding="utf-8") == ("# Written by hand\n")


# --- partial failure ---------------------------------------------------------------------------


async def test_an_ingest_that_fails_keeps_the_file_and_says_so(root: Path) -> None:
    """The contract: the file is the record, so the record survives the index not taking it.

    Deleting a caller's content because embedding hiccuped is the worse of the two failures, and
    it would destroy the durable half to keep the derived half tidy.
    """
    service = await service_for(settings_for(root), fails=True)

    created = await service.document_create(collection=COLLECTION, slug="retry_policy", body=BODY)

    assert created.indexed is False
    assert created.member is False
    assert created.detail == "the embedder declined"
    written = root / COLLECTION / "retry_policy.md"
    assert written.read_text(encoding="utf-8") == BODY
    assert created.path == str(written)
    assert created.document_id == document_id(WORKSPACE, SOURCE, str(written)), (
        "the id a later sync will index the kept file under has to be reportable now"
    )


async def test_a_kept_file_reaches_a_caller_as_a_failure_with_the_path_still_on_it(
    root: Path,
) -> None:
    """``ok: false`` **and** ``data``. Either alone is useless to whoever sent the content.

    Through ``run_op`` rather than by inspecting the payload, because the envelope is what every
    surface returns and the claim is about what a caller receives.
    """
    service = await service_for(settings_for(root), fails=True)
    envelope = await run_op(
        "document_create",
        service.workspace,
        lambda: service.document_create(collection=COLLECTION, slug="retry_policy", body=BODY),
    )
    body = envelope.as_json()

    assert body["ok"] is False
    assert body["error"]["type"] == "DocumentNotIndexedError"
    assert str(root / COLLECTION / "retry_policy.md") in body["error"]["message"]
    assert body["data"]["path"] == str(root / COLLECTION / "retry_policy.md")
    assert body["data"]["indexed"] is False


# --- parity ------------------------------------------------------------------------------------


async def _from_tool(service: ApplicationService, arguments: dict[str, Any]) -> Any:
    server = build_server(service)
    result = await server.call_tool("document_create", arguments)
    return result.structured_content


def _from_cli(
    monkeypatch: pytest.MonkeyPatch, service: ApplicationService, argv: Sequence[str], body: str
) -> Any:
    """Run the command with the service already built, and parse its ``--json`` output.

    ``stdin`` is monkeypatched rather than a file passed, because standard input is the path a
    person and a script both take and the one that would otherwise go unexercised.
    """

    async def dispatch(command: Command) -> Envelope:
        return await run_op(
            command.op, service.workspace, lambda: commands.run(service, command, commands.silent)
        )

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    result = CliRunner().invoke(cli.app, ["--json", *argv], input=body)
    assert result.exit_code in {0, 1}, result.output
    return json.loads(result.stdout)


def _from_http(service: ApplicationService, payload: dict[str, Any]) -> Any:
    """Post through the **real** application, authenticated as a member.

    A key rather than an unauthenticated client, because this configuration has authoring
    switched on and the application refuses to be built without authentication — so an
    unauthenticated column here would be testing a deployment that cannot exist.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415 - only this column needs it

    from manicule.api.app import build_app  # noqa: PLC0415 - keeps FastAPI out of the CLI path

    secret = asyncio.run(service.api_key_create("parity", role="member")).secret
    with TestClient(build_app(service), client=("127.0.0.1", 41234)) as client:
        body: Any = client.post(
            "/api/v1/documents", json=payload, headers={"X-API-Key": secret}
        ).json()
        return body


def _comparable(envelope: dict[str, Any]) -> dict[str, Any]:
    """One envelope with the clock removed, so two runs of one operation can be equal.

    ``elapsed_ms`` is excluded **by name** rather than by tolerance, on
    ``tests/app/test_surface_parity.py``'s reasoning: a comparison that ignored whatever happened
    to differ would pass on a surface that had quietly stopped reporting a field.
    """
    data = envelope.get("data")
    if not isinstance(data, dict):
        return envelope
    typed = cast("dict[str, Any]", data)
    return {**envelope, "data": {k: v for k, v in typed.items() if k != "elapsed_ms"}}


def test_the_three_envelope_surfaces_author_identically(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The same call, three ways round, compared as serialized JSON.

    **Each column gets a fresh service over the same root**, with the file removed in between.
    A write is not idempotent in the way a read is — the second call would meet the overwrite
    guard — so running three columns against one service would compare one success against two
    refusals and call it parity.

    The browser surface is the fourth adapter and is deliberately not here. ``POST /ui/documents``
    is asserted **absent** by ``tests/web/test_boundaries.py``: that surface renders envelopes and
    does not author, and adding a form from this change would undo that decision from a different
    package. The page that would show an authored document is the documents listing, which that
    file already holds to rendering what the tool reports.
    """
    arguments = {"collection": COLLECTION, "slug": "retry_policy", "body": BODY}

    def fresh() -> ApplicationService:
        written = root / COLLECTION / "retry_policy.md"
        if written.exists():
            written.unlink()
        return asyncio.run(service_for(settings_for(root)))

    from_tool = asyncio.run(_from_tool(fresh(), arguments))
    from_cli = _from_cli(
        monkeypatch, fresh(), ["document", "create", COLLECTION, "retry_policy"], BODY
    )
    from_http = _from_http(fresh(), arguments)

    assert _comparable(from_tool) == _comparable(from_cli)
    assert _comparable(from_tool) == _comparable(from_http)
    assert from_tool["ok"] is True
    assert from_tool["op"] == "document_create"


def test_the_terminal_is_told_why_a_kept_file_is_not_indexed(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Human output, not ``--json``, and it is the only place the reason reaches a person.

    ``print_envelope`` renders a retained payload *instead of* the error whenever one is
    attached, so a renderer that showed the path and dropped ``detail`` would leave somebody
    looking at a file on disk with nothing saying what went wrong — the exact moment the
    sentence is worth having.
    """
    service = asyncio.run(service_for(settings_for(root), fails=True))

    async def dispatch(command: Command) -> Envelope:
        return await run_op(
            command.op, service.workspace, lambda: commands.run(service, command, commands.silent)
        )

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    result = CliRunner().invoke(
        cli.app, ["document", "create", COLLECTION, "retry_policy"], input=BODY
    )

    assert result.exit_code == 1
    # The filename rather than the whole path: a terminal wraps an absolute one, and what is
    # being asserted is that the renderer reached for it at all.
    assert "retry_policy.md" in result.output
    assert "is not indexed" in result.output
    assert "the embedder declined" in result.output
