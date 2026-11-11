"""``authoring.local_evidence`` reaches the pipelines the runtime builds, not just the unit.

``IngestPipeline`` defaults to no local scheme, so a runtime that forgot to pass the setting would
refuse every local locator that arrives by a sync while the authoring door, which reads settings
directly, kept accepting them. Only the shipped construction path can show the difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from manicule.app.runtime import Runtime
from manicule.config.settings import AuthoringSettings, Settings
from manicule.container import keys
from manicule.plugins.registry import discover
from tests.fakes import HashEmbedder
from tests.knowledge_samples import EVIDENCE, LOCATOR, memory

if TYPE_CHECKING:
    from pathlib import Path

SESSION = "claude-session:0f1e2d3c-4b5a-4968-8776-655443322110#a1b2c3d4-e5f6-4789-8abc-def012345678"
ID = "reference_vpn_gateway_failover.md"


def _runtime(environment: Path, *, local_evidence: tuple[str, ...]) -> Runtime:
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    settings = Settings(
        data_dir=environment / "data",
        workspace="alpha",
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
        authoring=AuthoringSettings(local_evidence=local_evidence),
    )
    return Runtime(settings, discovery=found)


async def _stored_reason(environment: Path, *, local_evidence: tuple[str, ...]) -> tuple[bool, str]:
    from tests.ingest.fakes import DictConnector  # noqa: PLC0415 - fakes, local to this harness

    source = DictConnector({ID: memory(evidence=EVIDENCE.replace(LOCATOR, SESSION))}, name="notes")
    source.media_types[ID] = "text/markdown"
    async with _runtime(environment, local_evidence=local_evidence) as opened:
        await (await opened.pipeline()).run(source)
        store = cast("Any", await opened.documents())
        document = await store.find_document("notes", ID)
    assert document is not None
    assert document.knowledge is not None
    return document.knowledge.record is not None, document.knowledge.unavailable_reason


@pytest.mark.parametrize(
    ("enabled", "accepted"), [((), False), (("claude-session",), True)], ids=["off", "on"]
)
async def test_the_runtime_pipeline_honors_the_setting(
    manicule_environment: Path, enabled: tuple[str, ...], accepted: bool
) -> None:
    has_record, reason = await _stored_reason(manicule_environment, local_evidence=enabled)

    assert has_record is accepted, reason
    if not accepted:
        assert "authoring.local_evidence" in reason
