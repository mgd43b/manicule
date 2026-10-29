"""The mbox connector through the real pipeline runner and the real stores.

A parser's own suite once passed while every ``.eml`` was indexed with zero chunks through the
runner, so every property that depends on what the pipeline does with a message is driven here
through :class:`~manicule.ingest.pipeline.IngestPipeline` with durable acquisition on, over the
real document store, blob store and vector store. ``docs/connectors/mbox.md`` §4-§6, §8.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest
import pytest_asyncio

from manicule.connectors.config import MboxConfig
from manicule.connectors.errors import NotFoundError
from manicule.connectors.mbox import MboxConnector
from manicule.core.acquisition import MAX_DISCOVERY_BATCH
from manicule.core.content import DocumentStatus, RawDocument
from manicule.core.mail_ids import MAIL_MESSAGE_ID_KEY
from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.sources import DocRef
from manicule.ingest.pipeline import IngestPipeline, RunReport
from manicule.ingest.reconcile import confirm_proposed_deletion, connector_scope, reconcile
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import LanceVectorStore
from tests.connectors import mbox_support as archives
from tests.connectors.gmail_support import mail_pipeline
from tests.connectors.mbox_support import Filing, message, messages, with_headers, write_mbox
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

NAME = "oldmail"


@dataclass
class Harness:
    store: SqliteDocStore
    engine: AsyncEngine
    data_dir: Path
    vectors: LanceVectorStore
    root: Path

    def pipeline(self, **options: object) -> IngestPipeline:
        return mail_pipeline(self.store, self.engine, self.data_dir, self.vectors, **options)

    def connector(
        self,
        name: str = NAME,
        *,
        root: Path | None = None,
        others: tuple[str, ...] = (),
        **options: object,
    ) -> MboxConnector:
        config = MboxConfig.model_validate({"root": str(root or self.root), **options})
        return MboxConnector(
            config,
            name=name,
            store=self.store,
            other_sources=others,
            cache_dir=self.data_dir / "cache",
        )

    async def sync(
        self,
        connector: MboxConnector | None = None,
        *,
        retain: bool | None = None,
        **options: object,
    ) -> RunReport:
        report = await self.pipeline(**options).run(
            connector or self.connector(), retain_source_bytes=retain
        )
        assert report.error == ""
        return report

    async def live(self, source: str = NAME) -> dict[str, DocumentStatus]:
        """Source ids of the live message documents of ``source``, with their status."""
        documents = await self.store.list_documents(limit=10_000)
        return {
            d.source_id: d.status
            for d in documents
            if d.source == source and d.container_id is None
        }


@pytest_asyncio.fixture
async def vectors(data_dir: Path) -> AsyncIterator[LanceVectorStore]:
    store = LanceVectorStore(data_dir / "vectors")
    await store.ensure_ready(HashEmbedder().fingerprint)
    yield store
    await store.teardown()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    directory = tmp_path / "archive"
    directory.mkdir()
    return directory


@pytest.fixture
def harness(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    vectors: LanceVectorStore,
    root: Path,
) -> Harness:
    return Harness(store, engine, data_dir, vectors, root)


def identities(path: Path) -> list[str]:
    return archives.reader_digests(path)


# --- a message through the runner (§4, §8) ------------------------------------------------------


@pytest.mark.parametrize("retain", [True, False])
async def test_a_message_is_indexed_through_the_runner(harness: Harness, retain: bool) -> None:
    """Headers, body and attachment produce chunks, a title and a citation address."""
    path = write_mbox(harness.root / "Inbox", [message(1, attachment=True)])
    (digest,) = identities(path)

    await harness.sync(retain=retain)

    document = await harness.store.find_document(NAME, digest)
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    chunks = await harness.store.document_chunks(document.id)
    text = "\n".join(chunk.text for chunk in chunks)
    assert chunks, "a message that parses must produce chunks"
    assert "Body of archived message 1" in text
    assert "Archived message 1" in text, "the headers"
    assert document.title == "Archived message 1"
    assert document.uri == f"{path.as_uri()}#sha256={digest}"
    assert document.metadata[MAIL_MESSAGE_ID_KEY] == "archive.1@example.org"
    provenance = document.metadata[PROVENANCE_KEY]
    assert isinstance(provenance, dict)
    assert provenance["snapshot"] == {"path": "Inbox", "retrieved_at": None}
    attachment = await harness.store.find_document(NAME, f"mail:{digest}!/notes.txt")
    assert attachment is not None
    assert attachment.container_id == document.id
    assert (document.original_ref is not None) is retain


async def test_every_dialect_and_a_thunderbird_profile_are_indexed_together(
    harness: Harness,
) -> None:
    archives.thunderbird(harness.root)
    archives.apple_export(harness.root, [message(5)])
    write_mbox(harness.root / "legacy" / "Old", messages(2, start=10), Filing(dialect="mboxo"))
    write_mbox(harness.root / "sun" / "Sent", messages(2, start=20), Filing(dialect="mboxcl"))
    archives.not_mbox(harness.root)

    await harness.sync()

    subjects = {
        d.title for d in await harness.store.list_documents(limit=1_000) if d.container_id is None
    }
    assert subjects == {f"Archived message {n}" for n in (1, 3, 4, 5, 10, 11, 20, 21)}, (
        "an expunged message, a folder's placeholder, .msf caches, a Maildir and .emlx are not mail"
    )


async def test_an_archive_larger_than_the_fetch_limit_is_indexed(harness: Harness) -> None:
    """§1: the cap is on a message, not on the archive."""
    path = write_mbox(harness.root / "Big", [archives.padded(n, 3_000) for n in range(40)])
    limit = 20_000
    assert path.stat().st_size > 5 * limit

    report = await harness.sync(max_fetch_bytes=limit)

    assert report.error == ""
    assert len(await harness.live()) == 40


async def test_a_message_over_the_limit_fails_alone(harness: Harness) -> None:
    write_mbox(harness.root / "Mixed", [message(1), archives.padded(2, 60_000), message(3)])
    await harness.pipeline(max_fetch_bytes=20_000).run(harness.connector())
    assert len(await harness.live()) == 2


async def test_more_than_one_discovery_batch_is_indexed(harness: Harness) -> None:
    count = MAX_DISCOVERY_BATCH + 50
    write_mbox(harness.root / "Many", messages(count))
    await harness.sync()
    assert len(await harness.live()) == count


async def test_discovery_batches_never_exceed_the_journal_limit() -> None:
    """3,000 messages in one file are yielded 250 at a time (§5.2); no pipeline needed to see it."""
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        write_mbox(Path(directory) / "Huge", messages(3_000))
        connector = MboxConnector(MboxConfig(root=directory), name=NAME)
        sizes = [len(batch) async for batch in connector.discover_batches(None)]
    assert sum(sizes) == 3_000
    assert max(sizes) == MAX_DISCOVERY_BATCH
    assert len(sizes) == 12


# --- identity (§4) ------------------------------------------------------------------------------


async def test_one_message_in_several_files_is_one_document(harness: Harness) -> None:
    shared = message(1)
    write_mbox(harness.root / "a" / "Inbox", [shared, message(2)])
    write_mbox(
        harness.root / "b" / "All Mail", [shared, message(3)], Filing(dialect="mboxo", eol=b"\r\n")
    )
    write_mbox(
        harness.root / "c" / "Read",
        [with_headers(shared, "X-Mozilla-Status: 0001", "Status: RO")],
    )

    await harness.sync()

    live = await harness.live()
    assert len(live) == 3
    first = await harness.store.find_document(NAME, identities(harness.root / "a" / "Inbox")[0])
    assert first is not None
    assert "/a/Inbox#sha256=" in first.uri, "the first file in walk order"


async def test_messages_sharing_a_message_id_are_two_documents(harness: Harness) -> None:
    """§4: ``Message-ID`` is not the identity, because the second would overwrite the first."""
    write_mbox(
        harness.root / "Inbox",
        [
            message(1, message_id="<same@example.org>"),
            message(2, message_id="<same@example.org>"),
            message(3, with_id=False),
            message(4, with_id=False),
        ],
    )
    await harness.sync()
    assert len(await harness.live()) == 4


async def test_reordering_and_moving_a_message_between_files_keeps_its_document(
    harness: Harness,
) -> None:
    path = write_mbox(harness.root / "Inbox", messages(4))
    await harness.sync()
    before = await harness.live()

    write_mbox(path, list(reversed(messages(4))))
    write_mbox(harness.root / "Archive", [message(0)])
    path.rename(harness.root / "Renamed")
    await harness.sync(harness.connector())

    assert set(before) <= set(await harness.live())
    uris = {d.uri.split("#")[0] for d in await harness.store.list_documents(limit=100)}
    assert uris <= {(harness.root / name).as_uri() for name in ("Renamed", "Archive")}, (
        "an unchanged record's address is refreshed to where the message is now"
    )


# --- change detection (§5.1) --------------------------------------------------------------------


class Counting(MboxConnector):
    """Counts fetches, so a sync that fetches nothing can be told from one that fetched and kept."""

    fetched = 0

    @override
    async def fetch(self, ref: DocRef) -> RawDocument:
        type(self).fetched += 1
        return await super().fetch(ref)


def counting(harness: Harness, **options: object) -> Counting:
    config = MboxConfig.model_validate({"root": str(harness.root), **options})
    Counting.fetched = 0
    return Counting(config, name=NAME, store=harness.store, cache_dir=harness.data_dir / "cache")


async def test_an_unchanged_archive_costs_no_fetches_and_an_append_costs_only_the_new_messages(
    harness: Harness,
) -> None:
    path = write_mbox(harness.root / "Inbox", messages(5))
    write_mbox(harness.root / "Quiet", messages(3, start=50))
    await harness.sync(counting(harness))
    assert Counting.fetched == 8

    unchanged = await harness.sync(counting(harness))
    assert Counting.fetched == 0
    assert unchanged.error == ""

    archives.append_mbox(path, messages(2, start=5))
    await harness.sync(counting(harness))
    assert Counting.fetched == 2
    assert len(await harness.live()) == 10


async def test_changing_an_option_reads_every_file_again(harness: Harness) -> None:
    write_mbox(harness.root / "Inbox", messages(4))
    await harness.sync(counting(harness))
    await harness.sync(counting(harness, include_hidden=True))
    # Nothing is fetched twice, but the walk was complete: the stored map was discarded.
    assert Counting.fetched == 0
    connector = harness.connector(include_hidden=True)
    batches = [batch async for batch in connector.discover_batches(None)]
    assert sum(len(batch) for batch in batches) == 4


async def test_a_file_that_cannot_be_read_is_tried_again_and_does_not_stop_the_run(
    harness: Harness,
) -> None:
    write_mbox(harness.root / "Good", messages(2))
    bad = write_mbox(harness.root / "Bad", messages(2, start=10))
    bad.chmod(0)
    try:
        report = await harness.sync()
        assert report.error == ""
        assert len(await harness.live()) == 2
    finally:
        bad.chmod(0o644)
    await harness.sync(harness.connector())
    assert len(await harness.live()) == 4


async def test_a_missing_root_is_an_error_and_not_an_empty_archive(harness: Harness) -> None:
    write_mbox(harness.root / "Inbox", messages(2))
    await harness.sync()
    connector = harness.connector(root=harness.root / "gone")
    report = await harness.pipeline().run(connector)
    assert report.error
    assert len(await harness.live()) == 2


# --- deletions (§5.4) ---------------------------------------------------------------------------


async def test_a_message_that_left_every_file_leaves_the_index_at_reconciliation(
    harness: Harness,
) -> None:
    path = write_mbox(harness.root / "Inbox", messages(10))
    other = write_mbox(harness.root / "Also", [message(0)])
    await harness.sync()
    assert len(await harness.live()) == 10

    write_mbox(path, [m for n, m in enumerate(messages(10)) if n not in {2, 3}])
    connector = harness.connector()
    refused = await reconcile(connector, harness.store, max_delete_fraction=1.0)

    assert refused.applied_count == 2
    live = await harness.live()
    assert len(live) == 8
    assert identities(other)[0] in live, "still in another file"


async def test_a_large_removal_waits_for_confirmation(harness: Harness) -> None:
    path = write_mbox(harness.root / "Inbox", messages(10))
    await harness.sync()
    write_mbox(path, messages(4))
    connector = harness.connector()

    proposed = await reconcile(connector, harness.store)
    assert proposed.refused
    assert len(await harness.live()) == 10
    applied = await confirm_proposed_deletion(
        connector.name, harness.store, scope=connector_scope(connector)
    )
    assert applied.applied_count == 6
    assert len(await harness.live()) == 4


async def test_removing_a_file_removes_only_the_messages_no_other_file_holds(
    harness: Harness,
) -> None:
    keep = write_mbox(harness.root / "Keep", messages(4))
    drop = write_mbox(harness.root / "Drop", [message(0), message(1), message(100)])
    await harness.sync()
    drop.unlink()
    result = await reconcile(harness.connector(), harness.store, max_delete_fraction=1.0)
    assert result.applied_count == 1
    assert set(identities(keep)) <= set(await harness.live())


async def test_reconciliation_refuses_an_archive_it_cannot_list(harness: Harness) -> None:
    write_mbox(harness.root / "Inbox", messages(3))
    locked = harness.root / "locked"
    write_mbox(locked / "Folder", messages(2, start=10))
    await harness.sync()
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            _ = [source_id async for source_id in harness.connector().reconcile()]
        result = await reconcile(harness.connector(), harness.store, max_delete_fraction=1.0)
        assert result.applied_count == 0
    finally:
        locked.chmod(0o755)
    assert len(await harness.live()) == 5


# --- fetch (§5.2) -------------------------------------------------------------------------------


async def test_a_message_that_moved_since_discovery_is_not_indexed_from_other_bytes(
    harness: Harness,
) -> None:
    path = write_mbox(harness.root / "Inbox", messages(3))
    connector = harness.connector()
    discovered = [d async for batch in connector.discover_batches(None) for d in batch]
    write_mbox(path, [message(9), *messages(3)])  # every offset has shifted

    with pytest.raises(NotFoundError):
        await connector.fetch(discovered[0].ref)
    with pytest.raises(NotFoundError):
        await connector.fetch(discovered[0].ref.model_copy(update={"metadata": {}}))


async def test_a_path_outside_the_root_is_not_served(harness: Harness, tmp_path: Path) -> None:
    outside = write_mbox(tmp_path / "elsewhere" / "Box", messages(1))
    (located,) = list(archives.scan_all(outside))
    ref = DocRef(
        source_id=located.digest,
        uri="file:///x",
        metadata={"path": str(outside), "offset": located.offset, "length": located.length},
    )
    with pytest.raises(NotFoundError):
        await harness.connector().fetch(ref)


# --- mail another source holds (§6) -------------------------------------------------------------


async def test_a_message_another_mbox_source_holds_is_skipped_by_default(
    harness: Harness, tmp_path: Path
) -> None:
    first_root = tmp_path / "first"
    write_mbox(first_root / "Inbox", [message(1), message(2), message(3, with_id=False)])
    await harness.sync(harness.connector("first", root=first_root, skip_known_message_ids=False))

    # The same three, filed differently, and one that is new.
    write_mbox(
        harness.root / "Copy",
        [
            with_headers(message(1), "X-Note: a second copy"),
            message(2),
            message(3, with_id=False),
            message(4),
        ],
    )
    await harness.sync(harness.connector(others=("first",)))

    titles = {
        d.title
        for d in await harness.store.list_documents(limit=1_000)
        if d.source == NAME and d.container_id is None
    }
    assert titles == {"Archived message 3", "Archived message 4"}, (
        "1 and 2 share a Message-ID with the other source; 3 has none and is never skipped"
    )


async def test_the_skip_can_be_turned_off(harness: Harness, tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    write_mbox(first_root / "Inbox", messages(3))
    await harness.sync(harness.connector("first", root=first_root, skip_known_message_ids=False))
    write_mbox(harness.root / "Copy", [with_headers(m, "X-Note: copy") for m in messages(3)])
    await harness.sync(harness.connector(others=("first",), skip_known_message_ids=False))
    assert len(await harness.live()) == 3


async def test_a_message_this_source_already_holds_is_never_skipped(
    harness: Harness, tmp_path: Path
) -> None:
    """§5.4: a skip must not make a document this source holds vanish from a full walk."""
    write_mbox(harness.root / "Inbox", messages(3))
    await harness.sync()  # indexed first, nothing else exists
    other = tmp_path / "other"
    write_mbox(other / "Inbox", messages(3))
    await harness.sync(harness.connector("other", root=other, skip_known_message_ids=False))

    connector = harness.connector(others=("other",))
    walked = [d.source_id async for batch in connector.discover_batches(None) for d in batch]
    assert sorted(walked) == sorted(await harness.live()), "all three are held, so all are yielded"


async def test_reconciliation_does_not_apply_the_skip(harness: Harness, tmp_path: Path) -> None:
    other = tmp_path / "other"
    write_mbox(other / "Inbox", messages(3))
    await harness.sync(harness.connector("other", root=other, skip_known_message_ids=False))
    write_mbox(harness.root / "Inbox", messages(3))
    await harness.sync(harness.connector(others=("other",), skip_known_message_ids=False))

    kept = harness.connector(others=("other",))
    result = await reconcile(kept, harness.store, max_delete_fraction=1.0)
    assert result.applied_count == 0
    assert len(await harness.live()) == 3


async def test_a_gmail_document_is_known_by_the_message_id_in_its_link(
    harness: Harness,
) -> None:
    """Gmail records no ``mail_message_id``; its link carries the id (``gmail.md`` §2)."""
    from tests.storage_helpers import make_document  # noqa: PLC0415

    await harness.store.ensure_workspace()
    gmail = make_document("mail", "gid-1", title="Archived message 1").model_copy(
        update={
            "metadata": {
                PROVENANCE_KEY: {
                    "source": {
                        "title": "Archived message 1",
                        "canonical_uri": "https://mail.google.com/mail/?authuser=a%40example.org"
                        "#search/rfc822msgid:archive.1%40example.org",
                    }
                }
            }
        }
    )
    await harness.store.upsert_document(gmail)
    await harness.store.upsert_document(
        make_document("mail", "gid-2", title="No id").model_copy(
            update={
                "metadata": {
                    PROVENANCE_KEY: {
                        "source": {"canonical_uri": "https://mail.google.com/mail/#all/gid-2"}
                    }
                }
            }
        )
    )
    write_mbox(harness.root / "Inbox", [message(1), message(2)])

    await harness.sync(harness.connector(others=("mail",)))

    assert {d.title for d in await harness.store.list_documents(limit=100) if d.source == NAME} == {
        "Archived message 2"
    }


async def test_diagnostics_name_no_message(harness: Harness) -> None:
    write_mbox(harness.root / "Inbox", messages(3))
    report = await harness.sync()
    dumped = json.dumps(report.as_metadata(), default=str)
    assert "Archived message" not in dumped
    assert "example.org" not in dumped
