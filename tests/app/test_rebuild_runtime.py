"""Production assembly of connector-free offline replacement generations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from manicule.app.runtime import Runtime
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.rebuild import RebuildState
from manicule.core.source_lifecycle import LifecycleRefusalError
from manicule.ingest.reembed import ReembedState
from manicule.plugins.registry import discover
from manicule.storage.docstore import SqliteDocStore
from tests.app.test_reembed_runtime import CountingEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from pathlib import Path


def _runtime(data_dir: Path, embedder: CountingEmbedder) -> Runtime:
    found = discover()
    found.registry.bind("offline-rebuild-runtime-test").add(
        keys.EMBEDDER.named("local"), lambda _: embedder
    )
    return Runtime(
        Settings(
            data_dir=data_dir,
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )


async def test_runtime_rebuilds_a_promoted_snapshot_without_a_connector_capability(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    embedder = CountingEmbedder()
    embedder.fingerprint = embedder.fingerprint.model_copy(update={"max_sequence_length": 1024})
    connector = fakes.DictConnector(
        {
            "one": " ".join(["alpha"] * 100),
            "two": " ".join(["beta"] * 100),
            "three": " ".join(["gamma"] * 100),
        },
        name="runtime-snapshot",
    )
    connector.media_types.update({"one": "text/plain", "two": "text/plain", "three": "text/plain"})
    async with _runtime(data_dir, embedder) as runtime:
        acquired = await (await runtime.pipeline()).run(connector, acquire_only=True)
        source_calls = tuple(connector.fetches)
        documents = cast("SqliteDocStore", await runtime.documents())
        snapshot = await documents.latest_unsettled_acquisition_run(connector.name)

        assert acquired.derivation_deferred
        assert snapshot is not None
        snapshot_id = snapshot.id
        plan = await (await runtime.ingestion()).rebuild_plan(snapshot.id)
        assert plan.runnable
        assert plan.documents == 3

        checkpoint = await (await runtime.ingestion()).rebuild_run(snapshot.id, "runtime-owner")

        assert checkpoint.state is RebuildState.PUBLISHED
        assert checkpoint.documents_built == 3
        assert checkpoint.chunks_built == 3
        assert tuple(connector.fetches) == source_calls
        assert await documents.count_documents() == 3

    second = CountingEmbedder()
    second.fingerprint = second.fingerprint.model_copy(
        update={"model_id": "second/model", "max_sequence_length": 1024}
    )
    async with _runtime(data_dir, second) as restarted:
        planned = await (await restarted.ingestion()).reembed_start(
            "second-model", "planning-owner"
        )
        assert planned.chunks_completed == 0
        assert second.texts == 0

    resumed_embedder = CountingEmbedder()
    resumed_embedder.fingerprint = resumed_embedder.fingerprint.model_copy(
        update={"model_id": "second/model", "max_sequence_length": 1024}
    )
    async with _runtime(data_dir, resumed_embedder) as resumed:
        published = await (await resumed.ingestion()).reembed_resume(
            "second-model", "resume-owner"
        )

        assert published.state is ReembedState.PUBLISHED
        assert published.chunks_completed == 3
        assert resumed_embedder.texts == 3
        assert tuple(connector.fetches) == source_calls

        reset = await (await resumed.maintenance()).reset_derived()
        rebuilt_again = await (await resumed.ingestion()).rebuild_run(
            snapshot_id, "post-reset-owner"
        )
        with pytest.raises(LifecycleRefusalError, match="resumable work"):
            await (await resumed.maintenance()).plan_snapshot_deletion(snapshot_id)

        assert reset.snapshot_items == 3
        assert rebuilt_again.state is RebuildState.PUBLISHED
        assert rebuilt_again.documents_built == 3
        assert tuple(connector.fetches) == source_calls
