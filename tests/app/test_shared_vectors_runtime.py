"""Cross-workspace vector reuse through the production runtime, on the embedded backend.

``docs/storage.md`` §6.2.6. The ingest-level tests hold the reuse rule to its counts; these hold
the assembly to it: the runtime finds the other workspace, opens its published LanceDB handle
for reading, and the second workspace's sync of a source the first already holds reaches the
model for nothing. What it stores is checked through the same integrity path a freshly written
vector is, and the operator's count is checked to be the operator's alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from manicule.app.caller import Caller, acting_as
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Role, Settings
from manicule.container import keys
from manicule.core.embedding import VectorState
from manicule.plugins.registry import discover
from tests.app.test_reembed_runtime import CountingEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.core.content import Chunk
    from manicule.core.protocols import VectorIntegrityMaintenance
    from manicule.storage.docstore import SqliteDocStore

SOURCE = "handbook"

PAGES = {
    f"page-{number}": "\n\n".join(
        f"Paragraph {line} of invented page {number}, long enough to be a chunk of its own."
        for line in range(3)
    )
    for number in range(4)
}


def _runtime(data_dir: Path, workspace: str, embedder: CountingEmbedder) -> Runtime:
    found = discover()
    found.registry.bind(f"shared-vectors-{workspace}").add(
        keys.EMBEDDER.named("local"),
        lambda _: embedder,
        metadata_factory=lambda _: embedder.fingerprint,
    )
    return Runtime(
        Settings(
            data_dir=data_dir,
            workspace=workspace,
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
            plugins={"config": {"chunker.structural": {"max_tokens": 128}}},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


async def _sync(runtime: Runtime) -> dict[str, int]:
    connector = fakes.DictConnector(PAGES, name=SOURCE)
    connector.media_types = dict.fromkeys(PAGES, "text/markdown")
    report = await (await runtime.pipeline()).run(connector)
    return dict(report.by_status)


async def _chunks(runtime: Runtime) -> list[Chunk]:
    store = cast("SqliteDocStore", await runtime.documents())
    chunks: list[Chunk] = []
    for document in await store.list_documents():
        chunks.extend(await store.document_chunks(document.id))
    return chunks


async def _stored(runtime: Runtime) -> dict[str, tuple[float, ...]]:
    """Each chunk's stored vector, by the embedding input it was made from."""
    chunks = await _chunks(runtime)
    verdicts = await (await runtime.prepared_vectors()).stored_vectors(chunks)
    assert {verdict.state for verdict in verdicts.values()} == {VectorState.READABLE}
    return {chunk.embed_text: verdicts[chunk.id].vector for chunk in chunks}


async def test_a_second_workspace_reaches_the_model_for_nothing_it_already_holds(
    tmp_path: Path,
) -> None:
    first_model = CountingEmbedder()
    async with _runtime(tmp_path, "alpha", first_model) as first:
        assert await _sync(first) == {"indexed": len(PAGES)}
        held = await _stored(first)
        theirs = {chunk.document_id for chunk in await _chunks(first)}
    assert first_model.texts == len(held) > 0

    second_model = CountingEmbedder()
    async with _runtime(tmp_path, "beta", second_model) as second:
        assert await _sync(second) == {"indexed": len(PAGES)}
        assert second_model.texts == 0, "every embedding input was already held by alpha"

        copied = await _stored(second)
        assert copied == held, "each copy is the stored vector exactly"
        assert {chunk.document_id for chunk in await _chunks(second)}.isdisjoint(theirs)

        # The integrity path a freshly written vector goes through, recomputed from the bytes.
        vectors = cast("VectorIntegrityMaintenance", await second.prepared_vectors())
        coverage = await vectors.checksum_coverage(recompute=True)
        assert (coverage.rows, coverage.verified, coverage.failed) == (len(held), len(held), 0)

        service = ApplicationService(second)
        assert (await service.stats()).vectors_from_other_workspaces == len(held)
        with acting_as(Caller(role=Role.ADMIN, user_id="a-network-admin")):
            assert (await service.stats()).vectors_from_other_workspaces is None

    async with _runtime(tmp_path, "alpha", CountingEmbedder()) as first:
        assert (await ApplicationService(first).stats()).vectors_from_other_workspaces == 0


async def test_only_a_workspace_holding_the_document_is_asked(tmp_path: Path) -> None:
    """The cheap check before any vector store is read: does the other workspace hold it at all.

    A workspace indexed under the same model but holding other documents is not a donor for
    this one, so a first sync of an unrelated corpus costs no read of its vectors.
    """
    async with _runtime(tmp_path, "alpha", CountingEmbedder()) as first:
        await _sync(first)
    async with _runtime(tmp_path, "gamma", CountingEmbedder()) as unrelated:
        connector = fakes.DictConnector({"elsewhere": "Text nobody else holds."}, name=SOURCE)
        connector.media_types = {"elsewhere": "text/markdown"}
        await (await unrelated.pipeline()).run(connector)

    model = CountingEmbedder()
    async with _runtime(tmp_path, "beta", model) as second:
        await second.pipeline()
        registry = second._workspaces  # pyright: ignore[reportPrivateUsage] - the unit under test
        held = await registry.donors(model.fingerprint, (), source=SOURCE, source_id="page-0")
        nowhere = await registry.donors(model.fingerprint, (), source=SOURCE, source_id="absent")

    assert [donor.workspace for donor in held] == ["alpha"]
    assert nowhere == []


async def test_a_superseded_donor_handle_is_not_closed_under_a_reader(tmp_path: Path) -> None:
    """Several documents lend at once, so a handle another document may still be reading stays
    open until the registry closes; and a donor that cannot be opened is tried once per state."""
    from sqlalchemy import update  # noqa: PLC0415

    from manicule.storage import models  # noqa: PLC0415

    async with _runtime(tmp_path, "alpha", CountingEmbedder()) as first:
        await _sync(first)

    model = CountingEmbedder()
    closed: list[str] = []
    async with _runtime(tmp_path, "beta", model) as second:
        await second.pipeline()
        registry = second._workspaces  # pyright: ignore[reportPrivateUsage] - the unit under test
        (before,) = await registry.donors(model.fingerprint, (), source=SOURCE, source_id="page-0")
        original = cast("Any", before.vectors).teardown

        async def recording_teardown() -> None:
            closed.append("old")
            await original()

        before.vectors.teardown = recording_teardown  # pyright: ignore[reportAttributeAccessIssue]
        async with second.require_engine().begin() as connection:
            await connection.execute(
                update(models.Workspace)
                .where(models.Workspace.id == "alpha")
                .values(derived_reset_epoch=models.Workspace.derived_reset_epoch + 1)
            )
        await registry.donors(model.fingerprint, (), source=SOURCE, source_id="page-0")
        assert closed == [], "a superseded handle is retired, not closed under its readers"

        attempts: list[str] = []

        async def failing(*args: object, **kwargs: object) -> object:
            del kwargs
            attempts.append(str(args[0]))
            msg = "the vector directory is gone"
            raise RuntimeError(msg)

        registry._donor_vectors.clear()  # pyright: ignore[reportPrivateUsage] - force a reopen
        registry._vectors_for = failing  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        for _ in range(3):
            assert (
                await registry.donors(model.fingerprint, (), source=SOURCE, source_id="page-0")
                == []
            )
        assert attempts == ["alpha"], "one failed open per index state, not one per document"
    assert closed == ["old"], "the retired handle is closed with the registry"
