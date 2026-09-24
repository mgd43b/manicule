"""What an answer leaves behind, through the service and the shipped runtime.

The generation tests assemble the answer path over fakes, and a fake blob reader holds no
database connection. The runtime's does: citation verification reads retained bytes through
the blob store, on the engine everything else uses, in tasks of its own that the answer closes
when it finishes first. So this runs the real ingest, the real blob store and the real engine
behind :class:`~manicule.app.service.ApplicationService`, with only the embedder, the chunker
and the model replaced.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.generation.verification import VerificationRun
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


async def test_shutdown_waits_for_a_citation_check_the_answer_stopped_waiting_for(
    indexed: Path,
    monkeypatch: pytest.MonkeyPatch,
    unclosed_connections: Callable[[], list[str]],
) -> None:
    """An answer's close waits for its citation checks only until its deadline, and a check
    can outlast that: canceled, it may still be finishing a read. The runtime is what disposes
    the engine that read goes through, so shutting down waits for it first. Disposing under it
    would leave the read's connection outside any pool anything will ever dispose.
    """
    close = VerificationRun.aclose

    async def promptly(run: VerificationRun, deadline_s: float = 0.05) -> None:
        del deadline_s  # the shipped deadline is five seconds, and this needs it to pass
        await close(run, deadline_s=0.05)

    monkeypatch.setattr(VerificationRun, "aclose", promptly)
    runtime = _runtime(indexed)
    runtime.acquire()
    blobs = await runtime.blobs()
    read = blobs.get
    reading = asyncio.Event()
    release = asyncio.Event()

    async def straggling(digest: str) -> bytes | None:
        """A read that finishes before it honors a cancellation, however many arrive — the way
        the engine finishes opening a connection — and reads through the engine to do it."""
        reading.set()
        canceled: asyncio.CancelledError | None = None
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                canceled = error
        data = await read(digest)
        if canceled is not None:
            raise canceled
        return data

    monkeypatch.setattr(blobs, "get", straggling)
    asking = asyncio.create_task(ApplicationService(runtime).ask(QUESTION, limit=5))
    try:
        answered_in_time, _ = await asyncio.wait({asking}, timeout=5)
        if not answered_in_time:
            release.set()  # the close is waiting on the check, so only this ends the answer
        answered = await asking
    finally:
        closing = asyncio.create_task(runtime.aclose())
        early, _ = await asyncio.wait({closing}, timeout=0.2)
        release.set()
        await closing

    assert answered_in_time, "the answer's close waited past its deadline for a check"
    assert answered.text == ANSWER
    assert reading.is_set(), "no citation check reached the blob store, so none was left running"
    assert not early, "shutdown disposed the engine while a citation check was still reading it"
    assert unclosed_connections() == [], "the check left running stranded a connection"
