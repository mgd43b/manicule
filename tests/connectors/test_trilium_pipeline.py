"""The Trilium connector through the real pipeline runner and the real HTML parser.

A parser's own suite once passed while every ``.eml`` was indexed with zero chunks through the
runner, so what depends on what the pipeline does with a note (that its HTML reaches the web
parser, that its title is the heading, that an unchanged note is not read again, that a deleted
one leaves) is driven here through :class:`~manicule.ingest.pipeline.IngestPipeline`.
``docs/connectors/trilium.md`` §3, §4, §6.
"""

from __future__ import annotations

from pydantic import SecretStr

from manicule.connectors.config import TriliumConfig
from manicule.connectors.trilium import TriliumConnector
from manicule.connectors.trilium_client import TriliumClient
from manicule.core.content import DocumentStatus, RawDocument
from manicule.core.protocols import Connector
from manicule.core.sources import DocRef
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.reconcile import reconcile
from manicule.parsers.config import WebConfig
from manicule.parsers.web import WebParser
from manicule.testing import assert_protocol_signatures
from tests.connectors.fake_trilium import BASE_URL, TOKEN, FakeNote, FakeTrilium, Waits, vault
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

NAME = "notes"


def connector_for(server: FakeTrilium) -> TriliumConnector:
    settings = TriliumConfig(base_url=server.base_url, token=SecretStr(TOKEN))
    client = TriliumClient(settings, transport=server.transport(), sleep=Waits())
    return TriliumConnector(settings, client, name=NAME)


def counting(connector: TriliumConnector) -> list[str]:
    """Wrap ``fetch`` so a test can see which notes a sync actually read."""
    fetched: list[str] = []
    original = connector.fetch

    async def counted(ref: DocRef) -> RawDocument:
        fetched.append(ref.source_id)
        return await original(ref)

    connector.fetch = counted
    return fetched


def pipeline() -> tuple[IngestPipeline, fakes.MemoryIngestStore]:
    parsers = {"web": WebParser(WebConfig()), "lines": fakes.LineParser()}
    routes = {"text/html": ["web"], "text/plain": ["lines"], "text/markdown": ["lines"]}
    built, store, _ = build(parsers=parsers, routes=routes)
    return built, store


async def test_a_note_is_indexed_through_the_runner_under_its_own_title() -> None:
    server = vault()
    server.add(
        FakeNote(
            "md", "Readme", parents=("work",), type="code", mime="text/x-markdown", content="# Hi"
        )
    )
    built, store = pipeline()
    connector = connector_for(server)

    report = await built.run(connector, retain_source_bytes=False)

    assert report.error == ""
    assert report.indexed == 3
    document = await store.find_document(NAME, "plan")
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert document.uri == f"{BASE_URL}/#root/plan"
    text = "\n".join(chunk.text for chunk in store.chunks[document.id])
    assert "quarterly plan" in text
    assert "Plan" in text, "the title is the document's first heading"


async def test_an_unchanged_note_is_not_read_again_and_an_edited_one_is() -> None:
    server = vault()
    built, _ = pipeline()
    connector = connector_for(server)
    fetched = counting(connector)

    first = await built.run(connector, retain_source_bytes=False)
    assert sorted(fetched) == ["garden", "plan"]
    fetched.clear()
    second = await built.run(connector, retain_source_bytes=False)

    assert first.indexed == 2
    assert second.skipped_version == 2
    assert fetched == []

    server.edit("plan", "<p>revised plan</p>")
    await built.run(connector, retain_source_bytes=False)
    assert fetched == ["plan"]


async def test_renaming_a_folder_re_reads_the_notes_inside_it_and_no_others() -> None:
    server = vault()
    built, _ = pipeline()
    connector = connector_for(server)
    fetched = counting(connector)
    await built.run(connector, retain_source_bytes=False)
    fetched.clear()

    server.rename("work", "Job")
    await built.run(connector, retain_source_bytes=False)

    assert fetched == ["plan"]


async def test_a_deleted_note_leaves_the_index_at_reconciliation() -> None:
    server = vault()
    built, store = pipeline()
    connector = connector_for(server)
    await built.run(connector, retain_source_bytes=False)
    server.delete("garden")

    result = await reconcile(connector, store, max_delete_fraction=1.0)

    assert result.applied_count == 1
    gone = await store.find_document(NAME, "garden")
    assert gone is None or gone.status is DocumentStatus.DELETED
    kept = await store.find_document(NAME, "plan")
    assert kept is not None
    assert kept.status is DocumentStatus.INDEXED


async def test_a_truncated_sweep_deletes_nothing() -> None:
    server = vault()
    built, store = pipeline()
    await built.run(connector_for(server), retain_source_bytes=False)
    server.cap = 2
    short = TriliumConfig(base_url=BASE_URL, token=SecretStr(TOKEN), max_notes=2)
    client = TriliumClient(short, transport=server.transport(), sleep=Waits())
    connector = TriliumConnector(short, client, name=NAME)

    result = await reconcile(connector, store, max_delete_fraction=1.0)

    assert not result.clean
    assert "max_notes" in result.error
    assert result.applied_count == 0

    for note_id in ("plan", "garden"):
        document = await store.find_document(NAME, note_id)
        assert document is not None
        assert document.status is DocumentStatus.INDEXED


def test_the_connector_has_the_protocol_s_signatures() -> None:
    assert_protocol_signatures(connector_for(vault()), Connector)
