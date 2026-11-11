"""The Git documents connector: what it reads, what it leaves alone, and what it says about each.

Driven against real repositories on disk, because the properties that matter — one commit read
from start to finish, an unchanged blob never fetched, a deleted file leaving the index — are
properties of Git's object model that a fake would only restate.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from manicule.connectors.bmad_layout import V6, V7
from manicule.connectors.config import GitDocsConfig
from manicule.connectors.git_docs import (
    BMAD_EPIC,
    BMAD_INITIATIVE,
    BMAD_KIND,
    BMAD_LAYOUT,
    GitDocsConnector,
)
from manicule.connectors.git_mirror import GitFetchError, GitMirror
from manicule.connectors.git_reader import GitSourceError
from manicule.core.content import RawDocument
from manicule.core.errors import ConfigError
from manicule.core.protocols import Connector
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.sources import DiscoveredDoc, DocRef
from manicule.testing import assert_connector_contract, assert_protocol_signatures
from tests.connectors.test_git_mirror import commit, git

BIG = "x" * 300


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A project: some documents, a lot of code, and the things that look like documents and are
    not (vendored packages, an agent framework's prompts, a generated dump)."""
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "--quiet")
    files = {
        "README.md": "# Project\n",
        "docs/guide.md": "# Guide\n",
        "docs/notes.txt": "plain notes\n",
        "docs/diagram.png": "not text",
        "docs/LICENSE": "no suffix",
        "src/app.py": "print('hello')\n",
        "src/lib/README.md": "# Library\n",
        "node_modules/pkg/README.md": "# Vendored\n",
        "_bmad/core/agent.md": "# An agent prompt\n",
        ".claude/skills/review/SKILL.md": "# A skill\n",
        "docs/dump.txt": BIG,
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    commit(root, "initial")
    return root


def config_for(repository: Path | str, **changes: object) -> GitDocsConfig:
    values: dict[str, object] = {"repository": str(repository), "max_bytes": 200}
    values.update(changes)
    return GitDocsConfig.model_validate(values)


async def discovered(connector: GitDocsConnector) -> list[DiscoveredDoc]:
    return [doc async for doc in connector.discover(None)]


async def identities(connector: GitDocsConnector) -> list[str]:
    return [doc.source_id for doc in await discovered(connector)]


async def test_only_documents_are_listed_and_code_is_never_read(repository: Path) -> None:
    """The reason this connector exists: a repository is mostly code, and none of it is wanted."""
    connector = GitDocsConnector(config_for(repository), name="project")
    await connector.setup()
    try:
        assert await identities(connector) == [
            "README.md",
            "docs/guide.md",
            "docs/notes.txt",
            "src/lib/README.md",
        ]
    finally:
        await connector.teardown()


async def test_what_an_agent_framework_installs_is_not_a_document_about_the_project(
    repository: Path,
) -> None:
    """BMAD alone writes hundreds of Markdown files under ``_bmad/``; every one is a prompt."""
    connector = GitDocsConnector(config_for(repository))
    try:
        found = await identities(connector)
        assert not [path for path in found if path.startswith(("_bmad/", ".claude/"))]
        assert not [path for path in found if "node_modules" in path]
    finally:
        await connector.teardown()


async def test_a_narrower_exclude_replaces_the_default_rather_than_extending_it(
    repository: Path,
) -> None:
    connector = GitDocsConnector(config_for(repository, exclude=["src/**"]))
    try:
        found = await identities(connector)
        assert "_bmad/core/agent.md" in found
        assert "src/lib/README.md" not in found
    finally:
        await connector.teardown()


async def test_content_root_bounds_the_listing_but_not_the_identity(repository: Path) -> None:
    """Cited by its path from the repository root, because that is the one a reader can open."""
    connector = GitDocsConnector(config_for(repository, content_root="docs"))
    try:
        assert await identities(connector) == ["docs/guide.md", "docs/notes.txt"]
    finally:
        await connector.teardown()


async def test_a_content_root_with_nothing_in_it_is_an_error_not_an_empty_corpus(
    repository: Path,
) -> None:
    """An empty inventory is what reconciliation reads as 'everything was deleted'."""
    connector = GitDocsConnector(config_for(repository, content_root="no-such-folder"))
    try:
        with pytest.raises(GitSourceError, match="no-such-folder"):
            await identities(connector)
    finally:
        await connector.teardown()


async def test_an_oversized_document_is_skipped_counted_and_kept_out_of_reconciliation(
    repository: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Skipped, not fatal: one stray dump must not make every document in the repository stale.
    And absent from ``reconcile`` too, so a document that grows past the limit leaves the index
    instead of lingering at its last small version."""
    connector = GitDocsConnector(config_for(repository))
    try:
        with caplog.at_level(logging.WARNING, logger="manicule.connectors.git_docs"):
            found = await identities(connector)
        assert "docs/dump.txt" not in found
        assert connector.watermark is not None
        assert connector.watermark.metadata["skipped_oversized"] == 1
        assert connector.watermark.metadata["skipped_unsupported"] == 0
        assert "skipped 1 document(s) over 200 bytes" in caplog.text
        assert "docs/dump.txt" not in [item async for item in connector.reconcile()]
    finally:
        await connector.teardown()


async def test_a_suffix_with_no_media_type_is_skipped_when_an_include_admits_it(
    repository: Path,
) -> None:
    """``**/*`` would otherwise hand a PNG to a text parser."""
    connector = GitDocsConnector(config_for(repository, include=["docs/**"]))
    try:
        assert await identities(connector) == ["docs/guide.md", "docs/notes.txt"]
        assert connector.watermark is not None
        assert connector.watermark.metadata["skipped_unsupported"] == 2  # the PNG and LICENSE
    finally:
        await connector.teardown()


async def test_a_document_carries_its_path_its_blob_and_where_it_sits(repository: Path) -> None:
    connector = GitDocsConnector(config_for(repository), name="project")
    try:
        by_id = {doc.source_id: doc for doc in await discovered(connector)}
        guide = by_id["docs/guide.md"]
        raw = await connector.fetch(guide.ref)

        assert guide.media_type == raw.media_type == "text/markdown"
        assert by_id["docs/notes.txt"].media_type == "text/plain"
        assert raw.content == b"# Guide\n"
        provenance = Provenance.model_validate(raw.metadata[PROVENANCE_KEY])
        assert provenance.source is not None
        assert provenance.source.source_id == "docs/guide.md"
        assert provenance.source.section_path == ("docs",)
        assert provenance.source.canonical_uri == "", "a local path has no address to cite"
        assert provenance.snapshot is not None
        assert provenance.snapshot.path == "docs/guide.md"
        assert raw.uri.startswith("file://")
    finally:
        await connector.teardown()


async def test_a_forge_address_makes_every_citation_open_the_document_at_that_commit(
    repository: Path,
) -> None:
    connector = GitDocsConnector(
        config_for(
            repository, blob_url="https://forge.example.org/team/project/blob/{commit}/{path}"
        )
    )
    try:
        guide = next(doc for doc in await discovered(connector) if doc.source_id == "docs/guide.md")
        commit_id = git(repository, "rev-parse", "HEAD")
        raw = await connector.fetch(guide.ref)

        expected = f"https://forge.example.org/team/project/blob/{commit_id}/docs/guide.md"
        assert raw.uri == guide.ref.uri == expected
        provenance = Provenance.model_validate(raw.metadata[PROVENANCE_KEY])
        assert provenance.source is not None
        assert provenance.source.canonical_uri == expected
    finally:
        await connector.teardown()


def test_an_https_repository_gets_a_blob_address_without_being_told_one(tmp_path: Path) -> None:
    connector = GitDocsConnector(
        GitDocsConfig.model_validate({"repository": "https://git.example.org/team/docs.git"}),
        mirror_root=tmp_path,
    )

    assert connector.address_of("abc123", "docs/a b.md") == (
        "https://git.example.org/team/docs/blob/abc123/docs/a%20b.md"
    )


async def test_the_whole_sync_reads_one_commit_even_if_head_moves_underneath_it(
    repository: Path,
) -> None:
    """Discovery named a blob; a push in between must not change which bytes that name means."""
    connector = GitDocsConnector(config_for(repository))
    try:
        first = await discovered(connector)
        guide = next(doc for doc in first if doc.source_id == "docs/guide.md")

        (repository / "docs" / "guide.md").write_text("# Revised\n", encoding="utf-8")
        new_commit = commit(repository, "revise")

        assert (await connector.fetch(guide.ref)).content == b"# Guide\n"
        second = await discovered(connector)
        revised = next(doc for doc in second if doc.source_id == "docs/guide.md")
        assert revised.version_token != guide.version_token
        assert (await connector.fetch(revised.ref)).content == b"# Revised\n"
        assert connector.watermark is not None
        assert connector.watermark.value == new_commit
    finally:
        await connector.teardown()


async def test_a_commit_that_touched_no_document_changes_no_token(repository: Path) -> None:
    """The token is the blob, not the commit. One that moved with every push would re-embed the
    whole corpus each time anybody touched a line of code."""
    connector = GitDocsConnector(config_for(repository))
    try:
        before = {doc.source_id: doc.version_token for doc in await discovered(connector)}
        (repository / "src" / "app.py").write_text("print('changed')\n", encoding="utf-8")
        commit(repository, "code only")
        after = {doc.source_id: doc.version_token for doc in await discovered(connector)}

        assert before == after
    finally:
        await connector.teardown()


async def test_a_second_pipeline_sync_fetches_no_unchanged_body(repository: Path) -> None:
    from tests.ingest.test_pipeline import build  # noqa: PLC0415 - reuse the in-memory harness

    connector = GitDocsConnector(config_for(repository), name="project")
    fetched = 0
    original = connector.fetch

    async def counted(ref: DocRef) -> RawDocument:
        nonlocal fetched
        fetched += 1
        return await original(ref)

    connector.fetch = counted
    pipeline, store, _ = build()
    try:
        first = await pipeline.run(connector, retain_source_bytes=False)
        after_first = fetched
        (repository / "src" / "app.py").write_text("print('changed')\n", encoding="utf-8")
        commit(repository, "code only")
        second = await pipeline.run(connector, retain_source_bytes=False)

        assert first.indexed == 4
        assert after_first == 4
        assert second.skipped_version == 4
        assert fetched == after_first
        assert await store.find_document("project", "docs/guide.md") is not None
    finally:
        await connector.teardown()


async def test_a_commit_that_was_already_indexed_lists_nothing(repository: Path) -> None:
    """The stored SHA is the whole point of a periodic check: a repository nobody has pushed to
    costs a ``rev-parse``, not a tree walk and a comparison of every token."""
    connector = GitDocsConnector(config_for(repository))
    try:
        first = await discovered(connector)
        assert connector.walked_full_inventory
        stored = connector.watermark
        assert stored is not None

        again = [doc async for doc in connector.discover(stored)]

        assert first
        assert again == []
        assert not connector.walked_full_inventory, "an empty manifest is not a full inventory"
        assert connector.watermark is not None
        assert connector.watermark.value == stored.value
        assert connector.watermark.metadata == stored.metadata
    finally:
        await connector.teardown()


async def test_a_new_commit_lists_the_tree_again(repository: Path) -> None:
    connector = GitDocsConnector(config_for(repository))
    try:
        await discovered(connector)
        stored = connector.watermark
        (repository / "docs" / "new.md").write_text("# New\n", encoding="utf-8")
        commit(repository, "add a page")

        found = [doc.source_id async for doc in connector.discover(stored)]

        assert "docs/new.md" in found
        assert len(found) == 5
        assert connector.walked_full_inventory
    finally:
        await connector.teardown()


@pytest.mark.parametrize("stale", ["policy", "scope"])
async def test_a_watermark_recorded_under_other_rules_is_not_trusted(
    repository: Path, stale: str
) -> None:
    """A release that changes how a document is described must re-ingest at a commit nobody has
    pushed to, so the rules the watermark was recorded under are compared as well as the commit."""
    connector = GitDocsConnector(config_for(repository))
    try:
        await discovered(connector)
        stored = connector.watermark
        assert stored is not None
        other = stored.model_copy(update={"metadata": {**stored.metadata, stale: "something-else"}})

        found = [doc async for doc in connector.discover(other)]

        assert len(found) == 4
        assert connector.walked_full_inventory
    finally:
        await connector.teardown()


async def test_a_quiet_second_sync_tells_the_pipeline_it_was_not_a_full_inventory(
    repository: Path,
) -> None:
    """The hazard in skipping: a manifest that is empty because nothing changed must never be read
    as 'the source is empty'. The pipeline decides that from two answers the connector gives, so
    both are asserted through the pipeline's own readers: never declared as always-full, and
    reported as full only for the run that actually listed the tree."""
    from manicule.ingest.pipeline import (  # noqa: PLC0415 - import cost
        snapshot_enumerates_full_inventory,
        snapshot_walked_full_inventory,
    )
    from tests.ingest.test_pipeline import build  # noqa: PLC0415 - reuse the in-memory harness

    connector = GitDocsConnector(config_for(repository), name="project")
    fetched = 0
    original = connector.fetch

    async def counted(ref: DocRef) -> RawDocument:
        nonlocal fetched
        fetched += 1
        return await original(ref)

    connector.fetch = counted
    pipeline, store, _ = build()
    try:
        first = await pipeline.run(connector, retain_source_bytes=False)
        assert snapshot_walked_full_inventory(connector)
        second = await pipeline.run(connector, retain_source_bytes=False)

        assert not snapshot_enumerates_full_inventory(connector)
        assert not snapshot_walked_full_inventory(connector)
        assert first.indexed == 4
        assert (second.indexed, second.skipped_version) == (0, 0), "nothing was listed"
        assert fetched == 4
        assert await store.find_document("project", "docs/guide.md") is not None
    finally:
        await connector.teardown()


async def test_a_deleted_document_leaves_the_index_through_reconciliation(repository: Path) -> None:
    """After a sync, the next pass must not reuse the commit that sync pinned.

    Discovery runs in a task of its own so its pinned inventory is not this task's, which is how a
    scheduled pass relates to the sync before it.
    """
    connector = GitDocsConnector(config_for(repository))
    try:
        await asyncio.create_task(identities(connector))
        (repository / "docs" / "guide.md").unlink()
        commit(repository, "drop the guide")

        remaining = [item async for item in connector.reconcile()]

        assert "docs/guide.md" not in remaining
        assert "README.md" in remaining
    finally:
        await connector.teardown()


async def test_a_pass_after_a_restart_enumerates_the_current_revision(repository: Path) -> None:
    connector = GitDocsConnector(config_for(repository))
    try:
        assert "docs/guide.md" in [item async for item in connector.reconcile()]
    finally:
        await connector.teardown()


async def test_a_reference_that_disagrees_with_its_inventory_is_refused(repository: Path) -> None:
    """The inventory, not the reference, is the authority. Reading what a forged one names could
    read any object in the repository."""
    connector = GitDocsConnector(config_for(repository))
    try:
        guide = next(doc for doc in await discovered(connector) if doc.source_id == "docs/guide.md")
        forged = guide.ref.model_copy(
            update={"metadata": {**guide.ref.metadata, "git_blob": "0" * 40}}
        )
        with pytest.raises(GitSourceError, match="contradicts"):
            await connector.fetch(forged)
        elsewhere = guide.ref.model_copy(update={"source_id": "src/app.py"})
        with pytest.raises(GitSourceError):
            await connector.fetch(elsewhere)
    finally:
        await connector.teardown()


async def test_a_local_path_that_is_not_a_directory_is_a_startup_refusal(tmp_path: Path) -> None:
    connector = GitDocsConnector(config_for(tmp_path / "missing"))

    with pytest.raises(ConfigError, match="not a directory"):
        await connector.setup()


@pytest.mark.contract
async def test_git_docs_satisfies_the_connector_contract(repository: Path) -> None:
    connector = GitDocsConnector(config_for(repository))
    await connector.setup()
    try:
        await assert_connector_contract(connector)
        assert_protocol_signatures(connector, Connector)
    finally:
        await connector.teardown()


# -- BMAD ---------------------------------------------------------------------------------------


def bmad_repository(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "bmad"
    root.mkdir()
    git(root, "init", "--quiet")
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    commit(root, "bmad")
    return root


V7_FILES = {
    "_bmad-output/initiative-checkout/initiative-checkout.md": "# Checkout\n",
    "_bmad-output/initiative-checkout/tickets/tickets.toml": "[[epics]]\n",
    "_bmad-output/initiative-checkout/tickets/epic-cart/epic-cart.md": "# Cart\n",
    "_bmad-output/initiative-checkout/tickets/epic-cart/tickets/story-ui-plan.md": "# Plan\n",
    "_bmad/core/agent.md": "# not a document\n",
}


async def test_a_v7_document_is_cited_as_what_it_is(tmp_path: Path) -> None:
    root = bmad_repository(tmp_path, V7_FILES)
    connector = GitDocsConnector(config_for(root))
    try:
        docs = {doc.source_id: doc for doc in await discovered(connector)}
        plan = await connector.fetch(
            docs["_bmad-output/initiative-checkout/tickets/epic-cart/tickets/story-ui-plan.md"].ref
        )

        assert plan.metadata[BMAD_LAYOUT] == V7
        assert plan.metadata[BMAD_KIND] == "story-plan"
        assert plan.metadata[BMAD_INITIATIVE] == "checkout"
        assert plan.metadata[BMAD_EPIC] == "cart"
        assert connector.watermark is not None
        assert connector.watermark.metadata["bmad_layout"] == V7
        assert not [path for path in docs if path.startswith("_bmad/")]
    finally:
        await connector.teardown()


async def test_a_v6_document_is_cited_as_what_it_is(tmp_path: Path) -> None:
    root = bmad_repository(
        tmp_path,
        {
            "_bmad-output/planning-artifacts/prd.md": "# PRD\n",
            "_bmad-output/implementation-artifacts/2-1-login.md": "# Story\n",
            "_bmad-output/implementation-artifacts/sprint-status.yaml": "epics: []\n",
        },
    )
    connector = GitDocsConnector(config_for(root))
    try:
        docs = {doc.source_id: doc for doc in await discovered(connector)}
        prd = await connector.fetch(docs["_bmad-output/planning-artifacts/prd.md"].ref)
        story = await connector.fetch(
            docs["_bmad-output/implementation-artifacts/2-1-login.md"].ref
        )

        assert (prd.metadata[BMAD_LAYOUT], prd.metadata[BMAD_KIND]) == (V6, "prd")
        assert (story.metadata[BMAD_KIND], story.metadata[BMAD_EPIC]) == ("story", "2")
        assert BMAD_INITIATIVE not in story.metadata
    finally:
        await connector.teardown()


async def test_a_repository_with_no_bmad_in_it_is_indexed_the_same_way_without_facts(
    repository: Path,
) -> None:
    connector = GitDocsConnector(config_for(repository))
    try:
        guide = next(doc for doc in await discovered(connector) if doc.source_id == "docs/guide.md")
        raw = await connector.fetch(guide.ref)

        assert not [key for key in raw.metadata if key.startswith("bmad_")]
        assert connector.watermark is not None
        assert connector.watermark.metadata["bmad_layout"] == "none"
    finally:
        await connector.teardown()


async def test_layout_none_reads_nothing_into_a_path_and_changes_what_is_admitted_by_nothing(
    tmp_path: Path,
) -> None:
    root = bmad_repository(tmp_path, V7_FILES)
    auto = GitDocsConnector(config_for(root))
    plain = GitDocsConnector(config_for(root, layout="none"))
    try:
        enriched = await discovered(auto)
        bare = await discovered(plain)
        epic = "_bmad-output/initiative-checkout/tickets/epic-cart/epic-cart.md"

        assert [doc.source_id for doc in enriched] == [doc.source_id for doc in bare]
        assert (
            BMAD_KIND
            not in (await plain.fetch(next(d for d in bare if d.source_id == epic).ref)).metadata
        )
        assert (
            BMAD_KIND
            in (await auto.fetch(next(d for d in enriched if d.source_id == epic).ref)).metadata
        )
    finally:
        await auto.teardown()
        await plain.teardown()


async def test_changing_how_facts_are_derived_re_ingests_the_document(tmp_path: Path) -> None:
    """The token covers the facts, so a document described by the old rules is fetched again
    rather than being left with a description nobody would now produce."""
    root = bmad_repository(tmp_path, V7_FILES)
    auto = GitDocsConnector(config_for(root))
    plain = GitDocsConnector(config_for(root, layout="none"))
    try:
        epic = "_bmad-output/initiative-checkout/tickets/epic-cart/epic-cart.md"
        with_facts = next(d for d in await discovered(auto) if d.source_id == epic)
        without = next(d for d in await discovered(plain) if d.source_id == epic)

        assert with_facts.version_token != without.version_token
    finally:
        await auto.teardown()
        await plain.teardown()


# -- a remote -----------------------------------------------------------------------------------


def remote_connector(tmp_path: Path, origin: Path) -> GitDocsConnector:
    """An ``https`` source whose mirror fetches from a repository on disk."""
    mirror = GitMirror(
        origin.as_uri(), tmp_path / "mirror", allowed_protocols="file", timeout_s=60.0
    )
    return GitDocsConnector(
        GitDocsConfig.model_validate(
            {"repository": "https://git.example.org/team/project.git", "max_bytes": 200}
        ),
        name="remote",
        mirror=mirror,
    )


async def test_a_push_to_the_remote_is_seen_at_the_next_sync_with_no_checkout(
    tmp_path: Path, repository: Path
) -> None:
    connector = remote_connector(tmp_path, repository)
    try:
        assert "docs/guide.md" in await identities(connector)

        (repository / "docs" / "new.md").write_text("# New\n", encoding="utf-8")
        pushed = commit(repository, "add a page")
        found = await discovered(connector)
        new = next(doc for doc in found if doc.source_id == "docs/new.md")

        assert connector.watermark is not None
        assert connector.watermark.value == pushed
        assert (await connector.fetch(new.ref)).content == b"# New\n"
        assert new.ref.uri == f"https://git.example.org/team/project/blob/{pushed}/docs/new.md"
    finally:
        await connector.teardown()


async def test_a_remote_nobody_pushed_to_transfers_nothing_and_lists_nothing(
    tmp_path: Path, repository: Path
) -> None:
    connector = remote_connector(tmp_path, repository)
    try:
        await discovered(connector)
        stored = connector.watermark
        assert stored is not None

        assert [doc async for doc in connector.discover(stored)] == []

        (repository / "docs" / "new.md").write_text("# New\n", encoding="utf-8")
        commit(repository, "push")
        assert "docs/new.md" in [doc.source_id async for doc in connector.discover(stored)]
    finally:
        await connector.teardown()


async def test_a_remote_that_cannot_be_reached_fails_the_sync_and_deletes_nothing(
    tmp_path: Path,
) -> None:
    """Both discovery and reconciliation raise, and a pass that raises produces no deletions:
    the paths seen so far would be a prefix, not the truth (``docs/ingest.md`` §11.1)."""
    unreachable = remote_connector(tmp_path, tmp_path / "no-such-origin")
    try:
        with pytest.raises(GitFetchError):
            await identities(unreachable)
        with pytest.raises(GitFetchError):
            _ = [item async for item in unreachable.reconcile()]
        assert unreachable.watermark is None
    finally:
        await unreachable.teardown()


async def test_a_remote_is_not_contacted_when_the_connector_is_set_up(tmp_path: Path) -> None:
    """A network that is down at startup must not stop an installation serving what it holds."""
    connector = remote_connector(tmp_path, tmp_path / "no-such-origin")

    await connector.setup()


def test_an_https_repository_with_nowhere_to_mirror_is_refused() -> None:
    with pytest.raises(ConfigError, match="mirror"):
        GitDocsConnector(
            GitDocsConfig.model_validate({"repository": "https://git.example.org/team/docs"})
        )


# -- scope --------------------------------------------------------------------------------------


def test_the_scope_changes_with_what_decides_the_documents_and_not_with_how_they_are_reached(
    repository: Path,
) -> None:
    base = GitDocsConnector(config_for(repository))
    same = GitDocsConnector(config_for(repository))
    narrower = GitDocsConnector(config_for(repository, include=["docs/**"]))
    other_root = GitDocsConnector(config_for(repository, content_root="docs"))

    assert base.scope_fingerprint == same.scope_fingerprint
    assert base.scope_fingerprint != narrower.scope_fingerprint
    assert base.source_scope != other_root.source_scope


def test_moving_the_token_to_another_variable_is_not_a_new_source() -> None:
    first = GitDocsConfig.model_validate(
        {"repository": "https://git.example.org/team/docs", "token_env": "A_TOKEN"}
    )
    second = GitDocsConfig.model_validate(
        {
            "repository": "https://git.example.org/team/docs",
            "token_env": "B_TOKEN",
            "username": "bot",
            "fetch_timeout_s": 10,
        }
    )

    assert first.scope_fingerprint("s") == second.scope_fingerprint("s")


# -- configuration ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"repository": "ssh://git@git.example.org/team/docs"},
        {"repository": "https://user:secret@git.example.org/team/docs"},
        {"repository": "https://git.example.org/team/docs", "revision": "main:elsewhere"},
        {"repository": "/srv/docs", "token_env": "TOKEN"},
        {"repository": "https://git.example.org/x", "token_env": "ghp_realtoken-123"},
        {"repository": "https://git.example.org/x", "username": "a:b"},
        {"repository": "/srv/docs", "blob_url": "https://forge.example.org/blob/{commit}"},
        {"repository": "/srv/docs", "blob_url": "https://forge.example.org/{path}/{other}"},
        {"repository": "/srv/docs", "blob_url": "ftp://forge.example.org/{path}"},
        {"repository": "/srv/docs", "content_root": "../outside"},
        {"repository": "/srv/docs", "include": ["/absolute/*.md"]},
        {"repository": "/srv/docs", "exclude": ["../outside/**"]},
        {"repository": "/srv/docs", "include": []},
        {"repository": "/srv/docs", "layout": "bmad-v8"},
        {"repository": "/srv/docs", "max_bytes": 0},
        {"repository": "/srv/docs", "no_such_setting": True},
    ],
)
def test_configuration_that_cannot_work_is_refused_when_it_is_loaded(
    changes: dict[str, object],
) -> None:
    """A setting that appears to be in force and silently is not is worse than one that fails."""
    with pytest.raises(ValidationError):
        GitDocsConfig.model_validate(changes)


def test_a_refused_address_never_echoes_the_secret_it_was_given() -> None:
    with pytest.raises(ValidationError) as caught:
        GitDocsConfig.model_validate({"repository": "https://user:hunter2@git.example.org/x"})

    assert "hunter2" not in str(caught.value)
