"""Parse workers: the process boundary a deadline can actually be enforced across.

``asyncio.wait_for`` cancels the *await*, not the work. A parser sitting inside a C
extension — pypdfium2, lxml, tree-sitter, python-calamine, all of which manicule uses — holds
the GIL or blocks in native code and observes no cancellation until it returns on its own.
Running it in a thread does not help, because Python cannot kill a thread. So a parser that
hangs is not a caught exception; it is the run stopping. **A timeout is only enforceable
across a process boundary**, and that — not isolation for its own sake — is why parse runs
here.

Three properties this module exists to hold:

**One attempt per dispatch.** ``docs/parsing.md`` §6.3 makes the limits per *parser*, not per
document: a chain of three parsers on a thirty-second limit may legitimately take ninety
seconds. Dispatching a whole chain would make the last parser fail for the first parser's
reasons.

**A killed parser is a hard failure, not a decline.** A parser that declined inspected the
input and reported that it is not its kind, which is information. A parser the pipeline
killed reported nothing at all. Collapsing them lets a chain of timeouts end at
``unsupported_media_type``, which reads as "manicule does not handle this format" when the
truth is "every parser that handles this format ran out of time" — and sends whoever reads it
to write a parser that already exists. :class:`~manicule.parsers.chain.Outcome.FAILED` is
what advances the chain *and* classifies as ``failed`` at stage ``parse``.

**Workers hold no store handles.** They receive bytes and return blocks. Everything
transactional happens in the parent, which is what keeps the write ordering in
``docs/storage.md`` §8.2 true regardless of how many workers died.

``spawn``, never ``fork``: forking a process that has loaded a model runtime and opened SQLite
copies both into a child that must not touch either. ``spawn`` costs interpreter startup once
per worker, amortised over the run.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.ingest.limits import (
    ADDRESS_SPACE_HEADROOM,
    kill,
    limit_address_space,
    resident_bytes,
)
from manicule.parsers.chain import Attempt, Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manicule.core.content import ParsedBlock, RawDocument

MEGABYTE = 1024 * 1024

type ParserBuilder = Callable[[str], Awaitable[object]]
"""Name to parser, inside a worker.

``object`` rather than :class:`~manicule.core.protocols.Parser` because the value is handed
straight to :func:`attempt_one`, which asks what the parser *is* — a container as well as a
parser, or neither. Narrowing here would only mean widening again there."""


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Everything one parser's turn produced.

    ``members`` is what makes a container work at all. An archive parser's ``parse`` yields
    zero blocks *by design* — the container's content is the documents inside it — so a
    pipeline that only ever asked for blocks would see an empty result, advance the chain, and
    end at ``no_extractable_text``: a zip full of PDFs reported as though it were a scan. The
    decision is made in the worker because that is the only process holding the parser object,
    and asking the parent to know which parsers expand would mean constructing them twice.
    """

    blocks: list[ParsedBlock]
    attempt: Attempt
    members: tuple[object, ...] = ()
    """:class:`~manicule.parsers.expansion.MemberOutcome` values. Typed loosely here so that
    this module does not import the expansion vocabulary it only ever passes through."""


@runtime_checkable
class ParseRunner(Protocol):
    """Something that can give one parser one turn at one document.

    :class:`InProcessRunner` satisfies this without a subprocess; :class:`WorkerPool` satisfies
    it across a process boundary. The pipeline is written against the protocol so that the
    same chain loop, the same outcome vocabulary and the same classification serve both — a
    test that ran an easier code path than production would certify nothing about the one that
    matters.
    """

    async def run_attempt(self, name: str, raw: RawDocument) -> AttemptResult: ...


async def attempt_one(parser: object, name: str, raw: RawDocument) -> AttemptResult:
    """Give one already-constructed parser its turn, expanding it if it is a container.

    Shared by the in-process runner and by the worker, so that the two cannot diverge on the
    question they most easily would: whether this document is a container. Expansion is tried
    **first** for a parser that supports it, because such a parser's ``parse`` succeeds while
    producing nothing, and running it first would spend the attempt discovering that.
    """
    from manicule.parsers.chain import ParserChain  # noqa: PLC0415 - avoids an import cycle
    from manicule.parsers.expansion import SupportsExpansion, read_members  # noqa: PLC0415

    if isinstance(parser, SupportsExpansion):
        from manicule.core.errors import ParseError  # noqa: PLC0415

        try:
            members = await read_members(parser, raw)
        except ParseError as exc:
            return AttemptResult(
                [], Attempt(parser=name, outcome=Outcome.DECLINED, reason=str(exc))
            )
        except Exception as exc:  # noqa: BLE001 - a container's own bug fails one document
            reason = f"{type(exc).__name__}: {exc}"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        return AttemptResult(
            [], Attempt(parser=name, outcome=Outcome.PARSED), members=tuple(members)
        )

    chain = ParserChain(parsers={name: parser}, chains={})  # pyright: ignore[reportArgumentType]
    blocks, attempt = await chain.attempt(name, raw)
    return AttemptResult(blocks, attempt)


class InProcessRunner:
    """Runs attempts in this process, against parsers already built.

    Legitimate for a single document from a command that has no pool, and indispensable in
    tests — but **not** what a batch uses. Nothing here can enforce a deadline: a parser inside
    a native extension observes no cancellation, so a hang in this runner is the run stopping.
    """

    def __init__(self, parsers: Mapping[str, object]) -> None:
        self._parsers = dict(parsers)

    async def run_attempt(self, name: str, raw: RawDocument) -> AttemptResult:
        parser = self._parsers.get(name)
        if parser is None:
            reason = f"no parser named {name!r} is available to this runner"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        return await attempt_one(parser, name, raw)


class WorkerConfig(BaseModel):
    """What a worker needs to build parsers, and nothing else.

    Deliberately not the whole :class:`~manicule.config.settings.Settings` object. A parse
    worker has no use for a provider credential, and a subprocess that never receives one
    cannot leak it — so the narrow shape is a security property, not only a smaller pickle.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace: str = "default"
    data_dir: Path
    cache_dir: Path
    parser_fallbacks: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    plugins_enabled: tuple[str, ...] | None = None
    plugins_disabled: tuple[str, ...] = ()
    plugin_config: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    memory_limit_bytes: int = Field(default=1024 * MEGABYTE, ge=MEGABYTE)


@dataclass(frozen=True, slots=True)
class _Request:
    """One attempt, sent to a worker."""

    parser: str
    raw: RawDocument


@dataclass(frozen=True, slots=True)
class _Reply:
    """What a worker produced."""

    result: AttemptResult


@dataclass(frozen=True, slots=True)
class _Ready:
    """A worker's first message: it has imported everything and taken what limits it can."""

    address_space_limited: bool


# --- the child -----------------------------------------------------------------------------


def _worker_main(connection: Connection, config: WorkerConfig) -> None:  # pragma: no cover
    """A worker's whole life. Runs in a spawned subprocess, covered by its own process.

    Not measured by the parent's coverage run, and exercised by
    ``tests/ingest/test_workers.py`` through the pool that owns it — which is the only way it
    can be exercised, since a worker that ran in this process would not be a worker.
    """
    limited = limit_address_space(config.memory_limit_bytes * ADDRESS_SPACE_HEADROOM)
    connection.send(_Ready(address_space_limited=limited))
    build = _parser_builder(config)
    try:
        while True:
            request = connection.recv()
            if not isinstance(request, _Request):
                return
            connection.send(_attempt_in_child(build, request))
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        connection.close()


def _parser_builder(config: WorkerConfig) -> ParserBuilder:
    """A function from parser name to parser, built once per worker.

    Resolution goes through the container rather than by importing a parser module, because
    the container is what applies per-component configuration. A worker that constructed
    parsers any other way would parse the same document differently from the parent, which is
    the one thing a worker must never do.
    """
    from manicule.config.settings import Settings  # noqa: PLC0415 - child-only imports
    from manicule.container import keys  # noqa: PLC0415
    from manicule.container.container import Container  # noqa: PLC0415
    from manicule.plugins.registry import discover  # noqa: PLC0415

    settings = Settings(
        workspace=config.workspace,
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        parser_fallbacks=dict(config.parser_fallbacks),
        plugins={  # pyright: ignore[reportArgumentType] - validated by the model it builds
            "enabled": config.plugins_enabled,
            "disabled": config.plugins_disabled,
            "config": config.plugin_config,
        },
    )
    enabled = settings.plugins.enabled
    found = discover(
        enabled=None if enabled is None else frozenset(enabled),
        disabled=frozenset(settings.plugins.disabled),
    )
    container = Container(settings, found.registry, discovery=found)

    async def build(name: str) -> object:
        return await container.aget(keys.PARSER.named(name))

    return build


def _attempt_in_child(build: ParserBuilder, request: _Request) -> _Reply:
    """Run one parser and classify what it produced, or why it could not run."""

    async def run() -> _Reply:
        try:
            parser = await build(request.parser)
        except Exception as exc:  # noqa: BLE001 - a build failure is this attempt's failure
            reason = f"{type(exc).__name__}: {exc}"
            failed = Attempt(parser=request.parser, outcome=Outcome.FAILED, reason=reason)
            return _Reply(AttemptResult([], failed))
        return _Reply(await attempt_one(parser, request.parser, request.raw))

    return asyncio.run(run())


# --- the parent ----------------------------------------------------------------------------


@dataclass
class _Worker:
    """One live subprocess and the pipe to it."""

    process: SpawnProcess
    connection: Connection
    address_space_limited: bool = False
    documents: int = 0

    @property
    def pid(self) -> int:
        return self.process.pid or 0

    def terminate(self) -> None:
        """Stop the worker, hard, and reap it so no zombie survives the run."""
        if self.process.is_alive():
            kill(self.pid)
        self.process.join(timeout=5)
        self.connection.close()
        self.process.close()


@dataclass(frozen=True, slots=True)
class _Killed:
    """A worker stopped by the parent rather than by its own code."""

    reason: str


class WorkerPool:
    """A fixed pool of parse workers, each given one attempt at a time.

    The pool size is what a run depends on, not the identity of any worker: a killed worker is
    replaced immediately, and workers are recycled after ``max_documents`` to bound leaks in
    native parser libraries — a category of bug no amount of care in manicule prevents.
    """

    def __init__(
        self,
        config: WorkerConfig,
        *,
        workers: int = 0,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.25,
        max_documents: int = 500,
    ) -> None:
        self._config = config
        self._size = workers if workers > 0 else default_worker_count()
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._max_documents = max_documents
        self._context = multiprocessing.get_context("spawn")
        self._idle: asyncio.Queue[_Worker] = asyncio.Queue()
        self._live: list[_Worker] = []
        self._kills: dict[str, int] = {}
        self._started = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def kills(self) -> Mapping[str, int]:
        """How many workers were killed, by reason.

        The number ``doctor`` reports per media type: a parser that hangs or leaks on one
        format is visible here long before anybody notices a slow sync.
        """
        return dict(self._kills)

    async def setup(self) -> None:
        """Start every worker. Idempotent, so the container may call it more than once."""
        if self._started:
            return
        self._started = True
        for _ in range(self._size):
            await self._idle.put(await self._spawn())

    async def teardown(self) -> None:
        """Stop every worker, including ones still holding a document."""
        self._started = False
        while not self._idle.empty():
            self._idle.get_nowait()
        for worker in self._live:
            await asyncio.to_thread(worker.terminate)
        self._live.clear()

    async def __aenter__(self) -> Self:
        await self.setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.teardown()

    async def run_attempt(self, name: str, raw: RawDocument) -> AttemptResult:
        """Give one parser one turn, under this pool's time and memory limits.

        Returns the same :class:`AttemptResult` an in-process attempt returns, so the chain
        loop cannot tell the two apart — which is the point. A worker that was killed comes
        back as a hard failure naming the limit it hit, and a hard failure is what advances
        the chain *and* classifies as ``failed`` at stage ``parse``. It is emphatically not a
        decline: a chain of three timeouts must not end at ``unsupported_media_type``.
        """
        await self.setup()
        worker = await self._idle.get()
        try:
            outcome = await self._dispatch(worker, _Request(parser=name, raw=raw))
        except BaseException:
            # A cancelled await leaves a worker whose state nobody knows. Replacing it is
            # cheaper than reasoning about what it was in the middle of.
            await self._retire(worker)
            await self._idle.put(await self._spawn())
            raise
        if isinstance(outcome, _Killed):
            self._kills[outcome.reason] = self._kills.get(outcome.reason, 0) + 1
            await self._retire(worker)
            await self._idle.put(await self._spawn())
            return AttemptResult(
                [],
                Attempt(
                    parser=name,
                    outcome=Outcome.FAILED,
                    reason=f"worker killed: {outcome.reason}",
                ),
            )

        worker.documents += 1
        if worker.documents >= self._max_documents:
            await self._retire(worker)
            worker = await self._spawn()
        await self._idle.put(worker)
        return outcome.result

    # --- internals -------------------------------------------------------------------------

    async def _dispatch(self, worker: _Worker, request: _Request) -> _Reply | _Killed:
        worker.connection.send(request)
        limit = self._config.memory_limit_bytes
        return await asyncio.to_thread(
            _await_reply,
            worker.connection,
            worker.pid,
            self._timeout_s,
            self._poll_interval_s,
            limit,
        )

    async def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main, args=(child, self._config), daemon=True
        )
        process.start()
        child.close()
        worker = _Worker(process=process, connection=parent)
        self._live.append(worker)
        ready = await asyncio.to_thread(_await_ready, parent)
        worker.address_space_limited = ready.address_space_limited if ready else False
        return worker

    async def _retire(self, worker: _Worker) -> None:
        if worker in self._live:
            self._live.remove(worker)
        await asyncio.to_thread(worker.terminate)


def default_worker_count() -> int:
    """``min(4, cpu_count - 1)``, and never zero.

    One core is left for the parent, which is doing the embedding and every write. Capped at
    four because past that the bottleneck moves to the single embedder the parallelism exists
    to keep fed.
    """
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def _await_ready(connection: Connection, timeout: float = 60.0) -> _Ready | None:
    """Wait for a worker's first message. Runs in a thread, never on the event loop."""
    if not connection.poll(timeout):
        return None
    try:
        message = connection.recv()
    except (EOFError, OSError):  # pragma: no cover - a worker that died before saying hello
        return None
    return message if isinstance(message, _Ready) else None


def _await_reply(
    connection: Connection,
    pid: int,
    timeout_s: float,
    poll_interval_s: float,
    memory_limit_bytes: int,
) -> _Reply | _Killed:
    """Wait for one reply, killing the worker if it runs out of time or memory.

    Runs in a worker thread so that a blocking wait never occupies the event loop. The wait is
    a poll loop rather than a single blocking read because the memory check has to happen
    *while* the parser is running: by the time a runaway returns, it has already allocated.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kill(pid)
            return _Killed("timeout")
        if connection.poll(min(poll_interval_s, remaining)):
            try:
                message = connection.recv()
            except (EOFError, OSError):
                return _Killed("worker exited without replying")
            except Exception as exc:  # noqa: BLE001 - a corrupt reply is this attempt's failure
                return _Killed(f"unreadable reply: {type(exc).__name__}")
            if isinstance(message, _Reply):
                return message
            return _Killed(f"unexpected reply: {type(message).__name__}")
        resident = resident_bytes(pid)
        if resident is not None and resident > memory_limit_bytes:
            kill(pid)
            return _Killed("memory limit")


def worker_config(settings: object) -> WorkerConfig:
    """Build a worker's configuration from the application's settings.

    Takes the settings object structurally so that this module — and therefore the pipeline
    that imports it — does not force a configuration import on anything that only wants the
    types.
    """
    from manicule.config.settings import Settings  # noqa: PLC0415 - narrow, and only here

    if not isinstance(settings, Settings):  # pragma: no cover - a wiring mistake, not a path
        msg = f"expected Settings, got {type(settings).__name__}"
        raise TypeError(msg)
    return WorkerConfig(
        workspace=settings.workspace,
        data_dir=settings.data_dir,
        cache_dir=settings.cache_dir,
        parser_fallbacks=dict(settings.parser_fallbacks),
        plugins_enabled=settings.plugins.enabled,
        plugins_disabled=settings.plugins.disabled,
        plugin_config=settings.plugins.config,
        memory_limit_bytes=settings.ingest.parse_memory_limit_mb * MEGABYTE,
    )


__all__ = [
    "MEGABYTE",
    "AttemptResult",
    "InProcessRunner",
    "ParseRunner",
    "WorkerConfig",
    "WorkerPool",
    "attempt_one",
    "default_worker_count",
    "worker_config",
]
