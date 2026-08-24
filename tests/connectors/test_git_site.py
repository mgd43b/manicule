"""The Git website connector pins identity, routes and bytes to one commit."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from manicule.connectors.config import GitSiteConfig
from manicule.connectors.git_reader import GitSourceError
from manicule.connectors.git_site import GitSiteConnector
from manicule.connectors.site_routes import SiteRouteError
from manicule.core.content import RawDocument
from manicule.core.protocols import Connector
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.sources import DiscoveredDoc, DocRef
from manicule.testing import assert_connector_contract, assert_protocol_signatures

_GIT = shutil.which("git") or "git"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603 - test-owned fixed executable and arguments
        [_GIT, "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Manicule Tests",
        "-c",
        "user.email=tests@example.test",
        "commit",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / "docs").mkdir()
    (root / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "docs" / "drafts").mkdir()
    (root / "docs" / "drafts" / "secret.md").write_text("secret", encoding="utf-8")
    _commit(root, "initial")
    return root


def _config(repository: Path, **changes: object) -> GitSiteConfig:
    values: dict[str, object] = {
        "repository": str(repository),
        "content_root": "docs",
        "base_url": "https://docs.example.test/manual",
    }
    values.update(changes)
    return GitSiteConfig.model_validate(values)


async def _discovered(connector: GitSiteConnector) -> list[DiscoveredDoc]:
    return [document async for document in connector.discover(None)]


async def test_inferred_routes_fetch_exact_blobs_and_publish_provenance(repository: Path) -> None:
    connector = GitSiteConnector(_config(repository), name="handbook")
    await connector.setup()
    assert connector.watermark is None
    try:
        discovered = await _discovered(connector)
        assert [(item.source_id, item.ref.uri) for item in discovered] == [
            ("/guide/", "https://docs.example.test/manual/guide/"),
            ("/", "https://docs.example.test/manual/"),
        ]
        assert all(item.size_bytes is not None for item in discovered)
        assert connector.watermark is not None
        assert connector.watermark.value == _git(repository, "rev-parse", "HEAD")

        raw = await connector.fetch(discovered[0].ref)
        assert raw.content == b"# Guide\n"
        assert raw.uri == "https://docs.example.test/manual/guide/"
        assert raw.metadata["git_commit"] == connector.watermark.value
        provenance = Provenance.model_validate(raw.metadata[PROVENANCE_KEY])
        assert provenance.source is not None
        assert provenance.source.canonical_uri == raw.uri
        assert provenance.snapshot is not None
        assert provenance.snapshot.path == "docs/guide.md"
        assert [identity async for identity in connector.reconcile()] == ["/guide/", "/"]
    finally:
        await connector.teardown()


async def test_a_sync_keeps_reading_its_commit_after_head_moves(repository: Path) -> None:
    connector = GitSiteConnector(_config(repository))
    await connector.setup()
    try:
        first = await _discovered(connector)
        old_commit = connector.watermark.value if connector.watermark is not None else ""
        guide = next(item for item in first if item.source_id == "/guide/")

        (repository / "docs" / "guide.md").write_text("# Revised\n", encoding="utf-8")
        new_commit = _commit(repository, "revise")

        assert (await connector.fetch(guide.ref)).content == b"# Guide\n"
        second = await _discovered(connector)
        revised = next(item for item in second if item.source_id == "/guide/")
        assert revised.version_token != guide.version_token
        assert (await connector.fetch(revised.ref)).content == b"# Revised\n"
        assert connector.watermark is not None
        assert old_commit != new_commit == connector.watermark.value
    finally:
        await connector.teardown()


async def test_manifest_ids_survive_source_moves_and_route_metadata_changes(
    repository: Path,
) -> None:
    manifest = repository / "routes.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "pages": [
                    {"source": "docs/index.md", "id": "home", "route": "/"},
                    {
                        "source": "docs/guide.md",
                        "id": "guide",
                        "route": "/start/",
                        "title": "Start here",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _commit(repository, "routes")
    connector = GitSiteConnector(_config(repository, route_manifest="routes.json"))
    await connector.setup()
    try:
        first = await _discovered(connector)
        old = next(item for item in first if item.source_id == "guide")
        assert old.ref.uri.endswith("/manual/start/")

        (repository / "docs" / "guide.md").rename(repository / "docs" / "tutorial.md")
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        loaded["pages"][1].update(
            {"source": "docs/tutorial.md", "route": "/tutorial/", "title": "Tutorial"}
        )
        manifest.write_text(json.dumps(loaded), encoding="utf-8")
        _commit(repository, "move guide")

        second = await _discovered(connector)
        moved = next(item for item in second if item.source_id == "guide")
        assert moved.version_token != old.version_token
        assert moved.ref.uri.endswith("/manual/tutorial/")
        assert (await connector.fetch(moved.ref)).content == b"# Guide\n"
    finally:
        await connector.teardown()


async def test_manifest_mismatch_fails_before_any_document_is_yielded(repository: Path) -> None:
    (repository / "routes.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pages": [{"source": "docs/index.md", "route": "/"}],
            }
        ),
        encoding="utf-8",
    )
    _commit(repository, "incomplete routes")
    connector = GitSiteConnector(_config(repository, route_manifest="routes.json"))
    with pytest.raises(SiteRouteError, match="omitted admitted sources"):
        await connector.setup()
    await connector.teardown()


async def test_forged_reference_is_refused_without_reading_an_arbitrary_blob(
    repository: Path,
) -> None:
    connector = GitSiteConnector(_config(repository))
    await connector.setup()
    try:
        page = (await _discovered(connector))[0]
        forged = DocRef(
            source_id=page.source_id,
            uri=page.ref.uri,
            metadata={**page.ref.metadata, "git_path": "docs/index.md"},
        )
        with pytest.raises(GitSourceError, match="contradicts its pinned inventory"):
            await connector.fetch(forged)
    finally:
        await connector.teardown()


async def test_a_second_pipeline_sync_skips_unchanged_git_bodies(repository: Path) -> None:
    """Tree and route metadata are enough to decide that an immutable blob is unchanged."""
    from tests.ingest.test_pipeline import build  # noqa: PLC0415 - reuse the in-memory harness

    connector = GitSiteConnector(_config(repository))
    await connector.setup()
    fetched = 0
    original_fetch = connector.fetch

    async def counted_fetch(ref: DocRef) -> RawDocument:
        nonlocal fetched
        fetched += 1
        return await original_fetch(ref)

    connector.fetch = counted_fetch
    pipeline, store, _ = build()
    try:
        first = await pipeline.run(connector, retain_source_bytes=False)
        fetched_after_first = fetched
        second = await pipeline.run(connector, retain_source_bytes=False)

        assert first.indexed == 2
        assert fetched_after_first == 2
        assert second.skipped_version == 2
        assert fetched == fetched_after_first
        document = await store.find_document("git-site", "/guide/")
        assert document is not None
        assert document.original_ref is None
    finally:
        await connector.teardown()


@pytest.mark.contract
async def test_git_site_satisfies_the_connector_contract(repository: Path) -> None:
    connector = GitSiteConnector(_config(repository))
    await connector.setup()
    try:
        await assert_connector_contract(connector)
        assert_protocol_signatures(connector, Connector)
    finally:
        await connector.teardown()


def test_git_site_configuration_is_closed_and_canonical(repository: Path) -> None:
    config = _config(repository, base_url="HTTPS://DOCS.EXAMPLE.TEST:443/manual/")
    assert config.base_url == "https://docs.example.test/manual/"
    with pytest.raises(ValidationError):
        _config(repository, repository_extra="ignored")
    with pytest.raises(ValidationError):
        _config(repository, content_root="../private")
    with pytest.raises(ValidationError):
        _config(repository, include=("/absolute/*.md",))
