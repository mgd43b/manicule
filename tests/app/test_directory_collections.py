"""Authoring with ``authoring.collections_from_directories``: a collection is its first write.

In this mode the authorable collections are the authoring source root's top-level directories,
and ``document_create`` into a name that is not one yet creates the directory, the collection
and the rule joining them. Most of what is asserted here is what that does **not** widen to:
hidden directories, a name taken by a file or a link, a case variant, and a collection the
workspace already holds for some other purpose. Every refusal asserts that nothing was written
*and* that no collection was created, because a refusal that left a collection behind would be
the half-made state this mode exists to remove.

The filesystem is real and the ingest is faked, for the reason ``test_document_create`` gives.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from manicule.api.app import build_app
from manicule.app.service import (
    ApplicationService,
    _write_authored,  # pyright: ignore[reportPrivateUsage]
)
from manicule.config.settings import (
    AuthMode,
    AuthoringSettings,
    AuthSettings,
    ConnectorSettings,
    SecuritySettings,
    Settings,
)
from manicule.core.errors import PolicyError, UnknownEntityError
from manicule.core.organization import CollectionRule, directory_prefix
from manicule.mcp.server import build_server
from tests.app.fakes import FakeBackend, make_chunk, make_document
from tests.app.test_document_create import (
    BODY,
    SOURCE,
    IndexingIngestion,
    _authoring_check,  # pyright: ignore[reportPrivateUsage]
    backend_for,
    written_under,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def directory_settings(root: Path, *, exclude: Sequence[str] = ()) -> Settings:
    """One filesystem source, authored into by directory rather than by a list."""
    options: dict[str, JsonValue] = {"root": str(root)}
    if exclude:
        options["exclude"] = list[JsonValue](exclude)
    return Settings(
        connectors={SOURCE: ConnectorSettings(type="filesystem", options=options)},
        authoring=AuthoringSettings(source=SOURCE, collections_from_directories=True),
        security=SecuritySettings(auth=AuthSettings(mode=AuthMode.API_KEY)),
    )


async def directory_service(
    root: Path, *, existing: Sequence[str] = (), exclude: Sequence[str] = ()
) -> tuple[ApplicationService, FakeBackend]:
    backend = await backend_for(directory_settings(root, exclude=exclude), existing=existing)
    return ApplicationService(backend), backend


def collection_names(backend: FakeBackend) -> list[str]:
    return sorted(item.name for item in backend.organization_.collections.values())


@pytest.fixture
def root(tmp_path: Path) -> Path:
    made = tmp_path / "corpus"
    made.mkdir()
    return made


# --- the setting -------------------------------------------------------------------------------


def test_the_list_and_directory_mode_cannot_both_be_set() -> None:
    """Two ways of naming one set; both at once is refused rather than one silently winning."""
    with pytest.raises(ValidationError, match="collections_from_directories"):
        AuthoringSettings(source=SOURCE, collections=("memory",), collections_from_directories=True)


def test_directory_mode_alone_configures_authoring() -> None:
    assert AuthoringSettings(source=SOURCE, collections_from_directories=True).configured
    assert not AuthoringSettings(collections_from_directories=True).configured


def test_directory_mode_is_off_by_default() -> None:
    assert AuthoringSettings().collections_from_directories is False


# --- the ordinary case -------------------------------------------------------------------------


async def test_a_new_name_creates_the_directory_the_collection_and_its_rule(root: Path) -> None:
    """The whole feature in one call: no config edit, no `collection create`, no `rule set`."""
    service, backend = await directory_service(root)

    created = await service.document_create(collection="journal", slug="first", body=BODY)

    assert (root / "journal" / "first.md").read_text(encoding="utf-8") == BODY
    assert created.collection_created is True
    assert created.indexed is True
    assert created.member is True
    made = backend.organization_.collections[created.collection_id]
    assert made.name == "journal"
    assert made.rule == CollectionRule(
        uri_prefixes=frozenset({directory_prefix(str(root / "journal"))})
    )
    assert backend.organization_.members[made.id] == [created.document_id]


async def test_the_second_write_reuses_the_collection_the_first_made(root: Path) -> None:
    service, backend = await directory_service(root)
    first = await service.document_create(collection="journal", slug="first", body=BODY)

    second = await service.document_create(collection="journal", slug="second", body=BODY)

    assert second.collection_created is False
    assert second.collection_id == first.collection_id
    assert collection_names(backend) == ["journal"]


async def test_a_directory_that_arrived_by_git_is_adopted(root: Path) -> None:
    """A directory with documents and no collection yet gets its collection on the first write."""
    (root / "notes").mkdir()
    (root / "notes" / "old.md").write_text("# Old\n", encoding="utf-8")
    service, backend = await directory_service(root)

    created = await service.document_create(collection="notes", slug="new", body=BODY)

    assert created.collection_created is True
    assert collection_names(backend) == ["notes"]


async def test_an_existing_collection_that_is_its_directory_is_written_into(root: Path) -> None:
    """A collection made by hand the documented way — with its prefix — is the directory."""
    service, backend = await directory_service(root, existing=())
    await backend.organization_.create_collection(
        "memory", rule=CollectionRule(uri_prefixes=frozenset({str(root / "memory")}))
    )

    created = await service.document_create(collection="memory", slug="fact", body=BODY)

    assert created.collection_created is False
    assert created.collection_id == "col-0"


async def test_a_listed_collection_still_refuses_a_name_off_the_list(root: Path) -> None:
    """The default is unchanged: list mode never creates a collection."""
    settings = Settings(
        connectors={SOURCE: ConnectorSettings(type="filesystem", options={"root": str(root)})},
        authoring=AuthoringSettings(source=SOURCE, collections=("memory",)),
        security=SecuritySettings(auth=AuthSettings(mode=AuthMode.API_KEY)),
    )
    backend = await backend_for(settings)

    with pytest.raises(PolicyError, match=r"authoring\.collections"):
        await ApplicationService(backend).document_create(collection="journal", slug="x", body=BODY)
    assert collection_names(backend) == ["memory"]
    assert written_under(root) == []


# --- the bounds --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("..", "collection name"),
        (".", "collection name"),
        ("../escape", "separator"),
        ("a/b", "separator"),
        ("/absolute", "separator"),
        ("back\\slash", "separator"),
        ("", "collection name"),
        (" padded", "normalization"),
        (".git", "hidden"),
        (".trash", "hidden"),
        ("\uff4d\uff45\uff4d\uff4f", "different spelling"),
        ("two  spaces", "different spelling"),
    ],
)
async def test_a_name_that_cannot_be_a_directory_collection_is_refused(
    root: Path, name: str, match: str
) -> None:
    service, backend = await directory_service(root)

    with pytest.raises(ValueError, match=match):
        await service.document_create(collection=name, slug="x", body=BODY)

    assert written_under(root) == []
    assert collection_names(backend) == []


async def test_a_name_taken_by_a_file_is_refused(root: Path) -> None:
    (root / "journal").write_text("not a directory\n", encoding="utf-8")
    service, backend = await directory_service(root)

    with pytest.raises(PolicyError, match="not a directory"):
        await service.document_create(collection="journal", slug="x", body=BODY)

    assert written_under(root) == ["journal"]
    assert collection_names(backend) == []


async def test_a_name_taken_by_a_symbolic_link_is_refused(root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "journal").symlink_to(elsewhere, target_is_directory=True)
    service, backend = await directory_service(root)

    with pytest.raises(PolicyError, match="symbolic link"):
        await service.document_create(collection="journal", slug="x", body=BODY)

    assert list(elsewhere.iterdir()) == []
    assert collection_names(backend) == []


async def test_a_case_variant_of_an_existing_directory_is_refused(root: Path) -> None:
    """``Notes`` and ``notes`` are one directory on a case-insensitive filesystem."""
    (root / "Notes").mkdir()
    service, backend = await directory_service(root)

    with pytest.raises(PolicyError, match="only by case"):
        await service.document_create(collection="notes", slug="x", body=BODY)

    assert written_under(root) == ["Notes"]
    assert collection_names(backend) == []


async def test_a_collection_made_for_something_else_is_not_taken_over(root: Path) -> None:
    """The directory-mode form of "creating a collection is not granting write access to it".

    The workspace holds ``secrets`` — a Confluence space, a hand-curated list — with no directory
    beneath the root and no rule selecting one. Accepting it by name would make this mode "write
    into any collection the workspace holds".
    """
    service, backend = await directory_service(root, existing=("secrets",))

    with pytest.raises(PolicyError, match="does not take over"):
        await service.document_create(collection="secrets", slug="leak", body=BODY)

    assert written_under(root) == []
    assert backend.organization_.collections["col-0"].rule is None


async def test_an_excluded_directory_is_refused_before_a_collection_is_made(root: Path) -> None:
    service, backend = await directory_service(root, exclude=["private/**"])

    with pytest.raises(PolicyError, match="exclude"):
        await service.document_create(collection="private", slug="x", body=BODY)

    assert written_under(root) == []
    assert collection_names(backend) == []


async def test_a_taken_slug_in_a_new_directory_makes_no_collection(root: Path) -> None:
    """The conflict check refuses before the collection is created, not after."""
    (root / "journal").mkdir()
    (root / "journal" / "x.md").write_text("# Somebody else's\n", encoding="utf-8")
    service, backend = await directory_service(root)

    with pytest.raises(PolicyError, match="already exists"):
        await service.document_create(collection="journal", slug="x", body=BODY)

    assert collection_names(backend) == []


async def test_a_write_that_fails_takes_back_what_it_made(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory and collection are both undone when the file itself never landed."""
    service, backend = await directory_service(root)

    def refuses(target: Path, content: str, *, replace: bool) -> None:
        del content, replace
        target.parent.mkdir(parents=True, exist_ok=True)
        raise PermissionError(13, "read-only file system", str(target))

    monkeypatch.setattr("manicule.app.service._write_authored", refuses)

    with pytest.raises(PermissionError):
        await service.document_create(collection="journal", slug="x", body=BODY)

    assert written_under(root) == []
    assert collection_names(backend) == []


async def test_a_collection_created_by_another_writer_first_is_used(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cross-process race: the store refuses the duplicate and the write uses the winner."""
    service, backend = await directory_service(root)
    store = backend.organization_
    original = store.create_collection
    rule = CollectionRule(uri_prefixes=frozenset({directory_prefix(str(root / "journal"))}))

    async def lost_the_race(name: str, **kwargs: Any) -> Any:
        await original(name, rule=rule)  # the other writer, a moment earlier
        return await original(name, **kwargs)

    monkeypatch.setattr(store, "create_collection", lost_the_race)

    created = await service.document_create(collection="journal", slug="x", body=BODY)

    assert created.collection_created is False
    assert collection_names(backend) == ["journal"]
    assert (root / "journal" / "x.md").exists()


# --- over MCP ----------------------------------------------------------------------------------


async def test_the_tool_creates_a_collection_and_says_so(root: Path) -> None:
    service, backend = await directory_service(root)

    result = await build_server(service).call_tool(
        "document_create", {"collection": "journal", "slug": "x", "body": BODY}
    )

    envelope = dict(result.structured_content or {})
    assert envelope["ok"] is True, envelope
    assert envelope["data"]["collection_created"] is True
    assert collection_names(backend) == ["journal"]


# --- doctor ------------------------------------------------------------------------------------


async def test_doctor_lists_the_directories_and_is_quiet_when_each_is_a_collection(
    root: Path,
) -> None:
    service, _ = await directory_service(root)
    await service.document_create(collection="journal", slug="x", body=BODY)
    (root / ".git").mkdir()
    (root / "README.md").write_text("# Corpus\n", encoding="utf-8")

    check = _authoring_check(await service.doctor())

    assert check.state == "ok"
    assert check.facts["mode"] == "directories"
    assert check.facts["collections"] == ["journal"], "hidden entries and files are not collections"


async def test_doctor_reports_a_directory_with_no_collection_as_degraded(root: Path) -> None:
    """Not ``failing``: the next write into it creates the collection."""
    (root / "notes").mkdir()
    service, _ = await directory_service(root)

    check = _authoring_check(await service.doctor())

    assert check.state == "degraded"
    assert check.facts["missing"] == ["notes"]
    assert check.remedy == f"manicule collection create notes --uri-prefix {root / 'notes'}"


async def test_doctor_reports_a_directory_collection_without_its_rule(root: Path) -> None:
    (root / "notes").mkdir()
    service, _ = await directory_service(root, existing=("notes",))

    check = _authoring_check(await service.doctor())

    assert check.state == "degraded"
    assert check.facts["uncovered"] == ["notes"]


async def test_doctor_reports_directories_that_cannot_be_collections(root: Path) -> None:
    """A directory whose name the workspace would spell differently, or a pair differing by case.

    Their documents are indexed and in no collection, and authoring into them refuses.
    """
    fullwidth = "\uff4d\uff45\uff4d\uff4f"
    (root / fullwidth).mkdir()
    (root / "Notes").mkdir()
    (root / "notes").mkdir(exist_ok=True)
    # Three directories, or two where the filesystem folds case.
    case_sensitive = len(list(root.iterdir())) == 3  # noqa: ASYNC240
    service, _ = await directory_service(root, existing=("Notes", "notes"))

    check = _authoring_check(await service.doctor())

    assert check.state == "degraded"
    expected = [fullwidth, "Notes", "notes"] if case_sensitive else [fullwidth]
    assert sorted(check.facts["malformed"]) == sorted(expected)


async def test_doctor_leaves_out_an_excluded_directory(root: Path) -> None:
    (root / "private").mkdir()
    service, _ = await directory_service(root, exclude=["private"])

    check = _authoring_check(await service.doctor())

    assert check.state == "ok"
    assert check.facts["collections"] == []
    assert check.facts["excluded"] == []


# --- removing and moving an authored document --------------------------------------------------


async def test_remove_deletes_the_file_and_trashes_the_index_entry(root: Path) -> None:
    """Both halves, so the next sync has nothing to bring back."""
    service, backend = await directory_service(root)
    created = await service.document_create(collection="journal", slug="x", body=BODY)

    removed = await service.document_remove(collection="journal", slug="x")

    assert not (root / "journal" / "x.md").exists()
    assert removed.document_id == created.document_id
    assert removed.file_removed is True
    assert removed.index_removed is True
    assert backend.store.deleted == [(created.document_id, "soft")], (
        "soft, as reconciliation would have done it"
    )


async def test_remove_of_a_slug_nothing_holds_is_refused(root: Path) -> None:
    service, _ = await directory_service(root)

    with pytest.raises(UnknownEntityError, match="no authored document"):
        await service.document_remove(collection="journal", slug="nothing")


async def test_remove_takes_a_file_the_index_never_had(root: Path) -> None:
    """A write whose ingest failed, or a file committed by hand: the file still goes."""
    (root / "journal").mkdir()
    (root / "journal" / "x.md").write_text("# Stray\n", encoding="utf-8")
    service, _ = await directory_service(root)

    removed = await service.document_remove(collection="journal", slug="x")

    assert (removed.file_removed, removed.index_removed) == (True, False)
    assert written_under(root) == ["journal"]


@pytest.mark.parametrize("slug", ["../escape", "a/b", "..", ""])
async def test_remove_refuses_a_slug_that_is_not_one_path_component(
    root: Path, tmp_path: Path, slug: str
) -> None:
    (tmp_path / "escape.md").write_text("# Outside\n", encoding="utf-8")
    service, _ = await directory_service(root)

    with pytest.raises(ValueError, match="slug"):
        await service.document_remove(collection="journal", slug=slug)
    assert (tmp_path / "escape.md").exists()


async def test_remove_refuses_a_collection_off_the_configured_list(root: Path) -> None:
    """In list mode the same scope as creating: an unlisted collection is not authoring's."""
    settings = Settings(
        connectors={SOURCE: ConnectorSettings(type="filesystem", options={"root": str(root)})},
        authoring=AuthoringSettings(source=SOURCE, collections=("memory",)),
        security=SecuritySettings(auth=AuthSettings(mode=AuthMode.API_KEY)),
    )
    (root / "secrets").mkdir()
    (root / "secrets" / "x.md").write_text("# Not authoring's\n", encoding="utf-8")
    service = ApplicationService(await backend_for(settings))

    with pytest.raises(PolicyError, match=r"authoring\.collections"):
        await service.document_remove(collection="secrets", slug="x")
    assert (root / "secrets" / "x.md").exists()


async def test_move_writes_the_destination_and_removes_the_origin(root: Path) -> None:
    service, backend = await directory_service(root)
    original = await service.document_create(collection="inbox", slug="fact", body=BODY)

    moved = await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    assert (root / "kept" / "fact.md").read_text(encoding="utf-8") == BODY
    assert not (root / "inbox" / "fact.md").exists()
    assert moved.moved.collection_created is True
    assert moved.moved.document_id != original.document_id, "identity follows the path"
    assert moved.removed is not None
    assert moved.removed.document_id == original.document_id
    assert backend.store.deleted == [(original.document_id, "soft")]
    kept = backend.organization_.collections[moved.moved.collection_id]
    assert backend.organization_.members[kept.id] == [moved.moved.document_id]


async def test_move_copies_the_bytes_exactly(root: Path) -> None:
    """A file written by hand is moved as it is, not re-rendered or re-validated."""
    (root / "inbox").mkdir()
    exact = "no trailing newline, no front matter"
    (root / "inbox" / "fact.md").write_text(exact, encoding="utf-8")
    service, _ = await directory_service(root)

    await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    assert (root / "kept" / "fact.md").read_text(encoding="utf-8") == exact


async def test_move_onto_a_taken_slug_leaves_the_original_alone(root: Path) -> None:
    service, _ = await directory_service(root)
    await service.document_create(collection="inbox", slug="fact", body=BODY)
    await service.document_create(collection="kept", slug="fact", body="# Other\n")

    with pytest.raises(PolicyError, match="already holds"):
        await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    assert (root / "inbox" / "fact.md").read_text(encoding="utf-8") == BODY
    assert (root / "kept" / "fact.md").read_text(encoding="utf-8") == "# Other\n"


async def test_move_of_a_missing_document_is_refused_and_creates_nothing(root: Path) -> None:
    service, backend = await directory_service(root)

    with pytest.raises(UnknownEntityError, match="does not exist"):
        await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    assert written_under(root) == []
    assert collection_names(backend) == []


async def test_move_into_the_same_collection_is_refused(root: Path) -> None:
    service, _ = await directory_service(root)
    with pytest.raises(ValueError, match="nowhere to move"):
        await service.document_move(slug="fact", from_collection="inbox", to_collection="inbox")


async def test_move_into_a_hidden_name_is_refused_before_anything_moves(root: Path) -> None:
    service, _ = await directory_service(root)
    await service.document_create(collection="inbox", slug="fact", body=BODY)

    with pytest.raises(ValueError, match="hidden"):
        await service.document_move(slug="fact", from_collection="inbox", to_collection=".git")

    assert (root / "inbox" / "fact.md").exists()


async def test_remove_and_move_are_tools_over_mcp(root: Path) -> None:
    service, _ = await directory_service(root)
    server = build_server(service)
    await service.document_create(collection="inbox", slug="a", body=BODY)
    await service.document_create(collection="inbox", slug="b", body=BODY)

    moved = await server.call_tool(
        "document_move", {"slug": "a", "from_collection": "inbox", "to_collection": "kept"}
    )
    removed = await server.call_tool("document_remove", {"collection": "inbox", "slug": "b"})

    assert dict(moved.structured_content or {})["ok"] is True, moved.structured_content
    assert dict(removed.structured_content or {})["ok"] is True, removed.structured_content
    assert written_under(root) == ["inbox", "kept", "kept/a.md"]


async def test_remove_and_move_are_routes_over_http(root: Path) -> None:
    """Through the real application with a member's key, as ``POST /documents`` is."""
    service, _ = await directory_service(root)
    await service.document_create(collection="inbox", slug="a", body=BODY)
    await service.document_create(collection="inbox", slug="b", body=BODY)
    secret = (await service.api_key_create("member", role="member")).secret
    headers = {"X-API-Key": secret}

    with TestClient(build_app(service), client=("127.0.0.1", 41234)) as client:
        moved = client.post(
            "/api/v1/documents/move",
            json={"slug": "a", "from_collection": "inbox", "to_collection": "kept"},
            headers=headers,
        )
        removed = client.post(
            "/api/v1/documents/remove",
            json={"collection": "inbox", "slug": "b"},
            headers=headers,
        )

    assert moved.status_code == HTTPStatus.OK, moved.text
    assert removed.status_code == HTTPStatus.OK, removed.text
    assert written_under(root) == ["inbox", "kept", "kept/a.md"]


async def test_move_keeps_crlf_line_endings(root: Path) -> None:
    """Byte for byte, line endings included, so git sees a rename rather than a rewrite."""
    (root / "inbox").mkdir()
    (root / "inbox" / "fact.md").write_bytes(b"# Fact\r\n\r\nWritten on Windows.\r\n")
    service, _ = await directory_service(root)

    await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    assert (root / "kept" / "fact.md").read_bytes() == b"# Fact\r\n\r\nWritten on Windows.\r\n"


async def test_a_move_that_cannot_remove_the_original_says_where_both_are(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ok: false`` with the destination on the payload, not an error with no data."""
    service, _ = await directory_service(root)
    await service.document_create(collection="inbox", slug="fact", body=BODY)

    async def stuck(self: ApplicationService, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        raise PermissionError(13, "read-only file system")

    monkeypatch.setattr(ApplicationService, "_unauthor", stuck)

    result = await build_server(service).call_tool(
        "document_move", {"slug": "fact", "from_collection": "inbox", "to_collection": "kept"}
    )

    envelope = dict(result.structured_content or {})
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "DocumentNotRemovedError"
    assert envelope["data"]["moved"]["path"] == str(root / "kept" / "fact.md")
    assert envelope["data"]["removed"] is None
    assert (root / "inbox" / "fact.md").exists()
    assert (root / "kept" / "fact.md").exists()


async def test_losing_the_slug_race_keeps_the_collection_it_made(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file that won is a document in that directory, so its collection stays."""
    service, backend = await directory_service(root)

    def taken(target: Path, content: str, *, replace: bool) -> None:
        del content, replace
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# The winner\n", encoding="utf-8")
        raise FileExistsError(17, "exists", str(target))

    monkeypatch.setattr("manicule.app.service._write_authored", taken)

    with pytest.raises(PolicyError, match="created by something else"):
        await service.document_create(collection="journal", slug="x", body=BODY)

    assert collection_names(backend) == ["journal"]


# --- review: removal and move are held to creation's checks --------------------------------------


async def test_remove_through_a_symlinked_directory_is_refused(root: Path, tmp_path: Path) -> None:
    """A link inside the root must not let a remove reach what creation would refuse."""
    private = root / "private"
    private.mkdir()
    (private / "x.md").write_text("# Excluded\n", encoding="utf-8")
    (root / "journal").symlink_to(private, target_is_directory=True)
    service, _ = await directory_service(root, exclude=["private/**"])

    with pytest.raises(PolicyError, match="symbolic link"):
        await service.document_remove(collection="journal", slug="x")
    assert (private / "x.md").exists()


async def test_remove_in_an_excluded_directory_is_refused(root: Path) -> None:
    (root / "private").mkdir()
    (root / "private" / "x.md").write_text("# Excluded\n", encoding="utf-8")
    service, _ = await directory_service(root, exclude=["private/**"])

    with pytest.raises(PolicyError, match="exclude"):
        await service.document_remove(collection="private", slug="x")
    assert (root / "private" / "x.md").exists()


async def test_move_out_of_an_excluded_directory_is_refused(root: Path) -> None:
    (root / "private").mkdir()
    (root / "private" / "x.md").write_text("# Excluded\n", encoding="utf-8")
    service, backend = await directory_service(root, exclude=["private/**"])

    with pytest.raises(PolicyError, match="exclude"):
        await service.document_move(slug="x", from_collection="private", to_collection="kept")
    assert written_under(root) == ["private", "private/x.md"]
    assert collection_names(backend) == []


async def test_remove_retires_the_entry_a_sidecar_identity_was_stored_under(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingestion keys a file whose manifest declares an identity on that identity, not the path."""
    (root / "journal").mkdir()
    (root / "journal" / "x.md").write_text("# Mirrored\n", encoding="utf-8")
    service, backend = await directory_service(root)
    mirrored = make_document("default", source=SOURCE, source_id="page-1", title="x")
    backend.store.add(mirrored, make_chunk(mirrored))

    def declares(target: Path) -> str:
        del target
        return "page-1"

    monkeypatch.setattr("manicule.app.service._declared_identity", declares)

    removed = await service.document_remove(collection="journal", slug="x")

    assert removed.index_removed is True
    assert removed.document_id == mirrored.id
    assert backend.store.deleted == [(mirrored.id, "soft")]


async def test_move_reindexes_the_destination_once_the_original_is_retired(root: Path) -> None:
    """So a slug-resolving stage that kept the live original now resolves to the destination."""
    service, backend = await directory_service(root)
    await service.document_create(collection="inbox", slug="fact", body=BODY)
    ingestion = backend.ingestion_
    assert isinstance(ingestion, IndexingIngestion)
    before = len(ingestion.paths)

    await service.document_move(slug="fact", from_collection="inbox", to_collection="kept")

    destination = root / "kept" / "fact.md"
    assert ingestion.paths[before:] == [destination, destination]


@pytest.mark.parametrize("name", ["venv", "node_modules", "__pycache__"])
async def test_a_directory_the_walk_ignores_is_refused_and_not_reported(
    root: Path, name: str
) -> None:
    (root / name).mkdir()
    service, backend = await directory_service(root)

    with pytest.raises(ValueError, match="never walks"):
        await service.document_create(collection=name, slug="x", body=BODY)
    assert collection_names(backend) == []
    check = _authoring_check(await service.doctor())
    assert check.facts["collections"] == []


async def test_a_rule_narrowed_to_another_source_is_not_the_directory(root: Path) -> None:
    """A covering prefix ANDed with another source selects nothing in this directory."""
    service, backend = await directory_service(root, existing=())
    await backend.organization_.create_collection(
        "secrets",
        rule=CollectionRule(
            uri_prefixes=frozenset({str(root / "secrets")}), sources=frozenset({"confluence"})
        ),
    )

    with pytest.raises(PolicyError, match="does not take over"):
        await service.document_create(collection="secrets", slug="x", body=BODY)
    assert written_under(root) == []


async def test_a_failed_audit_takes_back_the_collection_it_made(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, backend = await directory_service(root)

    async def unauditable(self: ApplicationService, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise SQLAlchemyError("the audit log is locked")

    monkeypatch.setattr(ApplicationService, "_audit", unauditable)

    with pytest.raises(SQLAlchemyError):
        await service.document_create(collection="journal", slug="x", body=BODY)
    assert collection_names(backend) == []
    assert written_under(root) == []


def test_a_failed_exclusive_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk mid-write must not leave a truncated document behind."""
    target = tmp_path / "journal" / "x.md"

    def full(fd: int, mode: str, **kwargs: object) -> object:
        del mode, kwargs
        os.close(fd)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("manicule.app.service.os.fdopen", full)

    with pytest.raises(OSError, match="No space"):
        _write_authored(target, "# x\n", replace=False)
    assert not target.exists()
