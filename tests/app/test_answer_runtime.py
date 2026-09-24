"""What an answer leaves behind, through the service and the shipped runtime.

The generation tests assemble the answer path over fakes, and a fake blob reader holds no
database connection. The runtime's does: citation verification reads retained bytes through
the blob store, on the engine everything else uses, in tasks of its own that the answer closes
when it finishes first. So this runs the real ingest, the real blob store and the real engine
behind :class:`~manicule.app.service.ApplicationService`, with only the embedder, the chunker
and the model replaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.plugins.registry import discover
from tests.fakes import HashEmbedder
from tests.generation.fakes import ScriptedGenerator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

WORKSPACE = "alpha"
QUESTION = "orchard irrigation"
ANSWER = "The schedule runs at dawn."
PASSAGES = {
    "irrigation": "The orchard irrigation schedule runs at dawn.",
    "pruning": "Orchard pruning happens in late winter.",
}


def _runtime(environment: Path) -> Runtime:
    """A runtime that retains source bytes, so citations verify against the blob store."""
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    bound.add(keys.EMBEDDER.named("local"), lambda _: HashEmbedder())
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    bound.add(keys.GENERATOR.named("scripted"), lambda _: ScriptedGenerator(script=(ANSWER,)))
    settings = Settings(
        data_dir=environment / "data",
        workspace=WORKSPACE,
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        llm={"generator": "scripted"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block"},  # pyright: ignore[reportArgumentType]
        storage={"retain_source_bytes": True},  # pyright: ignore[reportArgumentType]
    )
    return Runtime(settings, discovery=found)


@pytest.fixture
async def indexed(manicule_environment: Path) -> Path:
    from tests.ingest.fakes import DictConnector  # noqa: PLC0415 - fakes, local to this harness

    async with _runtime(manicule_environment) as opened:
        source = DictConnector(dict(PASSAGES), name="notes")
        for source_id in PASSAGES:
            source.media_types[source_id] = "text/plain"
        report = await (await opened.pipeline()).run(source)
        assert report.indexed == len(PASSAGES)
    return manicule_environment


async def test_an_answer_that_outruns_its_verification_leaves_no_connection_open(
    indexed: Path,
    monkeypatch: pytest.MonkeyPatch,
    unclosed_connections: Callable[[], list[str]],
) -> None:
    """A one-sentence answer ends while its citations are still being verified.

    Closing the answer cancels the verification still in flight, and a blob read canceled
    while the pool was opening its connection stranded that connection: aiosqlite dropped the
    handle its thread had just opened, and SQLAlchemy never received it, so neither the session
    nor the engine's disposal could close it. The test suite's session-end check caught it, and
    ``test_query_cache_runtime`` had switched retention off to keep out of its way.
    """
    reads: list[str] = []
    async with _runtime(indexed) as opened:
        blobs = await opened.blobs()
        read = blobs.get

        async def counted(digest: str) -> bytes | None:
            reads.append(digest)
            return await read(digest)

        monkeypatch.setattr(blobs, "get", counted)
        answered = await ApplicationService(opened).ask(QUESTION, limit=5)

    assert answered.text == ANSWER
    assert reads, "nothing read the retained bytes, so nothing here was verified against them"
    assert unclosed_connections() == [], "the answer left a database connection open"
