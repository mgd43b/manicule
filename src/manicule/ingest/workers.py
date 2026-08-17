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
per parse worker, amortized over the run. Offline relational stages use disposable spawn
children instead: middleware and chunker allocations must disappear with the document and
must never be deserialized in the serving parent before their target-derived bounds pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import pickle
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from manicule.chunking import finalize_chunks
from manicule.ingest.limits import (
    ADDRESS_SPACE_HEADROOM,
    kill,
    limit_address_space,
    resident_bytes,
)
from manicule.parsers.chain import Attempt, Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manicule.container.container import Container
    from manicule.core.content import Document, ParsedBlock, RawDocument
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Chunker
    from manicule.ingest.middleware import MiddlewareRunner

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


def _attempt_output_bytes(result: AttemptResult) -> int:
    """Exact bytes the complete parse reply would put on the process pipe."""
    return len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


def retained_size(value: object, seen: set[int] | None = None) -> int:
    """Count live Python objects without constructing a serialized duplicate."""
    visited: set[int] = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    held = sys.getsizeof(value)
    if isinstance(value, Mapping):
        mapped = cast("Mapping[object, object]", value)
        return held + sum(
            retained_size(key, visited) + retained_size(item, visited)
            for key, item in mapped.items()
        )
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return held
    if isinstance(value, Sequence):
        sequence = cast("Sequence[object]", value)
        return held + sum(retained_size(item, visited) for item in sequence)
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        held += retained_size(cast("dict[object, object]", fields), visited)
    slots = getattr(type(value), "__slots__", ())
    names = cast("tuple[object, ...]", slots if isinstance(slots, tuple) else (slots,))
    for name in names:
        if isinstance(name, str) and hasattr(value, name):
            held += retained_size(getattr(value, name), visited)
    return held


def _bounded_stage_result(value: object | None, limit: int) -> StageResult:
    """Measure the actual pipe payload before it can reach the serving process."""
    candidate = StageResult(value, 0, 0)
    live_bytes = retained_size(candidate)
    actual = 0
    while True:
        candidate = StageResult(value, actual, live_bytes)
        measured = len(pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL))
        if measured == actual:
            break
        actual = measured
    if actual > limit or live_bytes > limit:
        return StageResult(None, 0, 0, StageFailure.MEMORY_BOUND)
    return StageResult(value, actual, live_bytes)


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

    def __init__(
        self,
        parsers: Mapping[str, object],
        *,
        middleware: object | None = None,
        chunker: object | None = None,
    ) -> None:
        self._parsers = dict(parsers)
        self._middleware = cast("MiddlewareRunner | None", middleware)
        self._chunker = cast("Chunker | None", chunker)

    async def open_stage_session(self, *, memory_limit_bytes: int) -> Self:
        del memory_limit_bytes
        return self

    async def aclose(self) -> None:
        return

    async def run_attempt(
        self,
        name: str,
        raw: RawDocument,
        *,
        max_output_bytes: int | None = None,
        memory_limit_bytes: int | None = None,
    ) -> AttemptResult:
        del memory_limit_bytes
        parser = self._parsers.get(name)
        if parser is None:
            reason = f"no parser named {name!r} is available to this runner"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        result = await attempt_one(parser, name, raw)
        produced = _attempt_output_bytes(result)
        if max_output_bytes is None or produced <= max_output_bytes:
            return result
        return AttemptResult(
            [], Attempt(parser=name, outcome=Outcome.FAILED, reason="memory_bound")
        )

    async def run_before_parse(
        self, raw: RawDocument, *, max_output_bytes: int, memory_limit_bytes: int
    ) -> StageResult:
        del memory_limit_bytes
        if self._middleware is None:
            raise RuntimeError("the in-process runner has no middleware stage")
        value = await self._middleware.before_parse(raw)
        return _bounded_stage_result(value, max_output_bytes)

    async def run_after_parse_and_chunk(
        self,
        document: Document,
        blocks: list[ParsedBlock],
        *,
        max_output_bytes: int,
        memory_limit_bytes: int,
        title: str,
        media_type: str,
        detect_glossary: bool,
    ) -> StageResult:
        del memory_limit_bytes
        if self._middleware is None or self._chunker is None:
            raise RuntimeError("the in-process runner has no middleware/chunker stage")
        transformed = await self._middleware.after_parse(document, blocks)
        chunks = tuple(self._chunker.chunk(document, transformed))
        chunks = tuple(await self._middleware.after_chunk(document, chunks)) if chunks else ()
        chunks = tuple(finalize_chunks(self._chunker, chunks))
        if detect_glossary:
            from manicule.ingest.glossary import detect_entries  # noqa: PLC0415

            entries = tuple(detect_entries(chunks, title=title, media_type=media_type))
        else:
            entries = ()
        return _bounded_stage_result((chunks, entries), max_output_bytes)


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
    chunker: str = "structural"
    middleware: tuple[str, ...] = ()
    plugins_enabled: tuple[str, ...] | None = None
    plugins_disabled: tuple[str, ...] = ()
    plugin_config: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    memory_limit_bytes: int = Field(default=1024 * MEGABYTE, ge=MEGABYTE)
    stage_tokenizer_file: Path | None = None
    stage_tokenizer_id: str | None = None
    stage_max_tokens: int | None = Field(default=None, gt=0)
    stage_overlap_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _complete_stage_tokenizer(self) -> Self:
        """Refuse a partial local tokenizer binding rather than silently using a stand-in."""
        values = (
            self.stage_tokenizer_file,
            self.stage_tokenizer_id,
            self.stage_max_tokens,
            self.stage_overlap_tokens,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("the isolated stage tokenizer binding must be complete")
        return self

    def resolved_stage_tokenizer(self) -> tuple[Path, str, int, int] | None:
        """Return the complete local tokenizer binding, if the serving runtime supplied it."""
        if self.stage_tokenizer_file is None:
            return None
        return (
            self.stage_tokenizer_file,
            cast("str", self.stage_tokenizer_id),
            cast("int", self.stage_max_tokens),
            cast("int", self.stage_overlap_tokens),
        )


@dataclass(frozen=True, slots=True)
class _Request:
    """One attempt, sent to a worker."""

    parser: str
    raw: RawDocument
    max_output_bytes: int | None = None
    memory_limit_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class _Reply:
    """What a worker produced."""

    result: AttemptResult


@dataclass(frozen=True, slots=True)
class _Ready:
    """A worker's first message: it has built its parsers and taken what limits it can."""

    address_space_limited: bool
    error: str = ""
    """Why this worker cannot serve, when it cannot. A worker that failed to build its parsers
    says so here rather than dying silently and leaving every later document blame the pipe."""


@dataclass(frozen=True, slots=True)
class StageResult:
    """Bounded result from isolated middleware/chunker execution."""

    value: object | None
    serialized_bytes: int
    retained_bytes: int
    reason: StageFailure | None = None


class StageFailure(StrEnum):
    """Public, bounded classification of an isolated relational-stage failure."""

    MEMORY_BOUND = "memory_bound"
    TIMEOUT = "timeout"
    WORKER_DIED = "worker_died"
    CONFIGURATION = "configuration"
    INVALID_REPLY = "invalid_reply"
    STAGE_FAILED = "stage_failed"


@dataclass(frozen=True, slots=True)
class _StageRequest:
    kind: Literal["before_parse", "after_parse_and_chunk"]
    raw: RawDocument | None = None
    document: Document | None = None
    blocks: list[ParsedBlock] | None = None
    max_output_bytes: int = 0
    memory_limit_bytes: int = 0
    title: str = ""
    media_type: str = "application/octet-stream"
    detect_glossary: bool = True


@dataclass(frozen=True, slots=True)
class _StageReady:
    reason: StageFailure | None = None


@dataclass(frozen=True, slots=True)
class _StageClose:
    pass


# --- the child -----------------------------------------------------------------------------


def _worker_main(connection: Connection, config: WorkerConfig) -> None:  # pragma: no cover
    """A worker's whole life. Runs in a spawned subprocess, covered by its own process.

    Not measured by the parent's coverage run, and exercised by
    ``tests/ingest/test_workers.py`` through the pool that owns it — which is the only way it
    can be exercised, since a worker that ran in this process would not be a worker.
    """
    limited = limit_address_space(config.memory_limit_bytes * ADDRESS_SPACE_HEADROOM)
    # **Plugins are discovered before the parent is told this worker is ready**, and the order
    # is load-bearing twice. The parent starts the per-attempt clock when it hears back, so
    # saying "ready" first would spend a cold worker's whole plugin discovery against the first
    # document's deadline — killing it for a delay it did not cause, and then paying the
    # identical cost again in its replacement. And a configuration error in discovery is
    # reported here, as a worker that never becomes ready, rather than as every subsequent
    # document failing with "worker exited without replying" while the real cause is invisible.
    try:
        build = _parser_builder(config)
    except BaseException as exc:  # noqa: BLE001 - reported across the pipe rather than lost
        connection.send(_Ready(address_space_limited=limited, error=f"{type(exc).__name__}: {exc}"))
        connection.close()
        return
    connection.send(_Ready(address_space_limited=limited))
    try:
        while True:
            request = connection.recv()
            if not isinstance(request, _Request):
                return
            connection.send(_attempt_in_child(build, request))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
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


async def _stage_components(config: WorkerConfig) -> tuple[object, object, object]:
    """Construct one document-scoped production middleware/chunker session."""
    from manicule.chunking import StructuralChunker, TokenCounter  # noqa: PLC0415
    from manicule.config.settings import Settings  # noqa: PLC0415 - child-only imports
    from manicule.container.container import Container  # noqa: PLC0415
    from manicule.embedding.runtimes.tokenization import FastTokenizer  # noqa: PLC0415
    from manicule.ingest.middleware import MiddlewareRunner  # noqa: PLC0415
    from manicule.plugins.registry import discover  # noqa: PLC0415

    settings = Settings(
        workspace=config.workspace,
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        parser_fallbacks=dict(config.parser_fallbacks),
        rag={"chunker": config.chunker},  # pyright: ignore[reportArgumentType]
        plugins={  # pyright: ignore[reportArgumentType]
            "enabled": config.plugins_enabled,
            "disabled": config.plugins_disabled,
            "config": config.plugin_config,
            "middleware": config.middleware,
        },
    )
    enabled = settings.plugins.enabled
    found = discover(
        enabled=None if enabled is None else frozenset(enabled),
        disabled=frozenset(settings.plugins.disabled),
    )
    container = Container(settings, found.registry, discovery=found)
    try:
        middleware = MiddlewareRunner(await container.middleware())
        _require_structural_stage_chunker(config.chunker)
        binding = config.resolved_stage_tokenizer()
        if binding is not None:
            tokenizer_file, tokenizer_id, max_tokens, overlap_tokens = binding
            tokenizer = FastTokenizer(tokenizer_file)
            counter = TokenCounter(
                tokenizer_id,
                lambda text: len(tokenizer.content_ids(text)),
                provisional=False,
            )
            chunker = StructuralChunker(
                counter,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        else:
            # Parse-only/test pools have no serving embedder to bind. The vocabulary loader is
            # deliberately cache-only; unlike constructing the embedder component, it cannot
            # discover a model card or contact Hugging Face.
            chunker = StructuralChunker(TokenCounter.provisionally())
    except BaseException:
        with contextlib.suppress(BaseException):
            await container.aclose()
        raise
    else:
        return middleware, chunker, container


def _require_structural_stage_chunker(name: str) -> None:
    """Refuse a component that has no serializable, network-free stage construction yet."""
    if name != "structural":
        raise ValueError("offline stage isolation currently requires the structural chunker")


async def _stage_in_child(
    middleware: object, chunker: object, request: _StageRequest
) -> StageResult:
    """Run one request against the document's persistent isolated component instances."""
    runner = cast("MiddlewareRunner", middleware)
    configured_chunker = cast("Chunker", chunker)
    if request.kind == "before_parse":
        if request.raw is None:
            return StageResult(None, 0, 0, StageFailure.INVALID_REPLY)
        return _bounded_stage_result(
            await runner.before_parse(request.raw), request.max_output_bytes
        )
    if request.document is None or request.blocks is None:
        return StageResult(None, 0, 0, StageFailure.INVALID_REPLY)
    blocks = await runner.after_parse(request.document, request.blocks)
    chunks = tuple(configured_chunker.chunk(request.document, blocks))
    chunks = tuple(await runner.after_chunk(request.document, chunks)) if chunks else ()
    chunks = tuple(finalize_chunks(configured_chunker, chunks))
    if request.detect_glossary:
        from manicule.ingest.glossary import detect_entries  # noqa: PLC0415

        entries = tuple(detect_entries(chunks, title=request.title, media_type=request.media_type))
    else:
        entries = ()
    return _bounded_stage_result((chunks, entries), request.max_output_bytes)


def _stage_worker_main(
    connection: Connection, config: WorkerConfig, memory_limit_bytes: int
) -> None:  # pragma: no cover - exercised through the spawned process
    async def run() -> None:
        effective_limit = min(config.memory_limit_bytes, memory_limit_bytes)
        limit_address_space(effective_limit * ADDRESS_SPACE_HEADROOM)
        container: object | None = None
        try:
            middleware, chunker, container = await _stage_components(config)
        except BaseException:  # noqa: BLE001 - bounded classification, no private detail
            connection.send(_StageReady(StageFailure.CONFIGURATION))
            return
        try:
            connection.send(_StageReady())
            while True:
                request = connection.recv()
                if isinstance(request, _StageClose):
                    return
                if not isinstance(request, _StageRequest):
                    connection.send(StageResult(None, 0, 0, StageFailure.INVALID_REPLY))
                    continue
                try:
                    connection.send(await _stage_in_child(middleware, chunker, request))
                except BaseException:  # noqa: BLE001 - bounded document failure
                    connection.send(StageResult(None, 0, 0, StageFailure.STAGE_FAILED))
        finally:
            if container is not None:
                with contextlib.suppress(BaseException):
                    await cast("Container", container).aclose()

    try:
        asyncio.run(run())
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        return
    finally:
        connection.close()


def _attempt_in_child(build: ParserBuilder, request: _Request) -> _Reply:
    """Run one parser and classify what it produced, or why it could not run."""

    async def run() -> _Reply:
        if request.memory_limit_bytes is not None:
            limit_address_space(request.memory_limit_bytes * ADDRESS_SPACE_HEADROOM)
        try:
            parser = await build(request.parser)
        except Exception as exc:  # noqa: BLE001 - a build failure is this attempt's failure
            reason = f"{type(exc).__name__}: {exc}"
            failed = Attempt(parser=request.parser, outcome=Outcome.FAILED, reason=reason)
            return _Reply(AttemptResult([], failed))
        result = await attempt_one(parser, request.parser, request.raw)
        if request.max_output_bytes is not None:
            produced = _attempt_output_bytes(result)
            if produced > request.max_output_bytes:
                refused = Attempt(
                    parser=request.parser,
                    outcome=Outcome.FAILED,
                    reason="memory_bound",
                )
                return _Reply(AttemptResult([], refused))
        return _Reply(result)

    return asyncio.run(run())


# --- the parent ----------------------------------------------------------------------------


@dataclass
class _Worker:
    """One live subprocess and the pipe to it."""

    process: SpawnProcess
    connection: Connection
    address_space_limited: bool = False
    documents: int = 0
    retired: bool = False

    @property
    def pid(self) -> int:
        return self.process.pid or 0

    def alive(self) -> bool:
        """Whether this worker's process is still running, without ever raising.

        Asked from the reply thread as well as the loop, and a ``Process`` that has been closed
        raises from ``is_alive``. The honest answer for a closed handle is "no".
        """
        try:
            return self.process.is_alive()
        except (ValueError, OSError):
            return False

    def terminate(self) -> None:
        """Stop the worker, hard, reap it and close its handles. Idempotent."""
        if self.retired:
            return
        self.retired = True
        if self.alive():
            kill(self.pid)
        with contextlib.suppress(ValueError, OSError, AssertionError):
            self.process.join(timeout=5)
        with contextlib.suppress(OSError):
            self.connection.close()
        with contextlib.suppress(ValueError, OSError):
            self.process.close()


class _IsolatedStageSession:
    """One document's stateful middleware/chunker process and its cancellation ownership."""

    def __init__(
        self,
        context: SpawnContext,
        config: WorkerConfig,
        *,
        timeout_s: float,
        poll_interval_s: float,
        memory_limit_bytes: int,
    ) -> None:
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._child = child
        self._process = context.Process(
            target=_stage_worker_main,
            args=(child, config, memory_limit_bytes),
            daemon=True,
        )
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._memory_limit_bytes = min(config.memory_limit_bytes, memory_limit_bytes)
        self._closed = False
        self._startup_failure: StageFailure | None = None

    async def start(self) -> Self:
        try:
            self._process.start()
        except BaseException:  # noqa: BLE001 - bounded worker startup classification
            self._startup_failure = StageFailure.WORKER_DIED
            with contextlib.suppress(BaseException):
                self._child.close()
            with contextlib.suppress(BaseException):
                self._connection.close()
            self._closed = True
            return self
        self._child.close()
        reply = await self._exchange(None)
        if isinstance(reply, _StageReady) or (
            isinstance(reply, StageResult) and reply.reason is not None
        ):
            self._startup_failure = reply.reason
        else:
            self._startup_failure = StageFailure.INVALID_REPLY
        if self._startup_failure is not None:
            await self.aclose()
        return self

    async def run_before_parse(
        self, raw: RawDocument, *, max_output_bytes: int, memory_limit_bytes: int
    ) -> StageResult:
        del memory_limit_bytes
        if self._startup_failure is not None:
            return StageResult(None, 0, 0, self._startup_failure)
        return await self._request(
            _StageRequest(kind="before_parse", raw=raw, max_output_bytes=max_output_bytes)
        )

    async def run_after_parse_and_chunk(
        self,
        document: Document,
        blocks: list[ParsedBlock],
        *,
        max_output_bytes: int,
        memory_limit_bytes: int,
        title: str,
        media_type: str,
        detect_glossary: bool,
    ) -> StageResult:
        del memory_limit_bytes
        if self._startup_failure is not None:
            return StageResult(None, 0, 0, self._startup_failure)
        return await self._request(
            _StageRequest(
                kind="after_parse_and_chunk",
                document=document,
                blocks=blocks,
                max_output_bytes=max_output_bytes,
                title=title,
                media_type=media_type,
                detect_glossary=detect_glossary,
            )
        )

    async def _request(self, request: _StageRequest) -> StageResult:
        reply = await self._exchange(request)
        return (
            reply
            if isinstance(reply, StageResult)
            else StageResult(None, 0, 0, StageFailure.INVALID_REPLY)
        )

    async def _exchange(self, message: object | None) -> object:
        task = asyncio.create_task(
            asyncio.to_thread(
                _exchange_stage_message,
                self._process,
                self._connection,
                message,
                timeout_s=self._timeout_s,
                poll_interval_s=self._poll_interval_s,
                memory_limit_bytes=self._memory_limit_bytes,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            _kill_stage_process(self._process)
            with contextlib.suppress(BaseException):
                await _join_despite_cancellation(task)
            cleanup = asyncio.create_task(asyncio.to_thread(self._close_sync, False))
            await _join_despite_cancellation(cleanup)
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        cleanup = asyncio.create_task(asyncio.to_thread(self._close_sync, True))
        interrupted = await _join_despite_cancellation(cleanup)
        if interrupted:
            raise asyncio.CancelledError

    def _close_sync(self, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if graceful and _stage_process_alive(self._process):
            with contextlib.suppress(BaseException):
                self._connection.send(_StageClose())
        if _stage_process_alive(self._process):
            self._process.kill()
        with contextlib.suppress(BaseException):
            self._process.join(timeout=5)
        with contextlib.suppress(BaseException):
            self._connection.close()
        with contextlib.suppress(BaseException):
            self._process.close()


async def _join_despite_cancellation(task: asyncio.Task[object]) -> bool:
    """Join owned cleanup work even when the caller repeats cancellation."""
    interrupted = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        else:
            return interrupted


async def _settle_despite_cancellation(
    task: asyncio.Task[object],
) -> tuple[bool, BaseException | None]:
    """Reach an owned lifecycle endpoint and report both cancellation and failure.

    Lifecycle callers need both facts: cancellation has precedence once requested, while an
    ordinary setup/teardown failure must still be observable when nobody canceled it.
    """
    interrupted = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        except BaseException as exc:  # noqa: BLE001 - lifecycle exception precedence
            return interrupted, exc
        else:
            return interrupted, None


@dataclass(frozen=True, slots=True)
class _Killed:
    """A worker stopped by the parent rather than by its own code."""

    reason: str


@dataclass(frozen=True, slots=True)
class _PoolStopped:
    """The pool generation that owned an attempt was closed by teardown."""


_NO_PERMIT = object()


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
        self._idle: asyncio.Queue[_Worker | None] = asyncio.Queue()
        self._live: list[_Worker] = []
        self._kills: dict[str, int] = {}
        self._started = False
        self._lifecycle = asyncio.Lock()
        self._generation = 0
        self._closed = asyncio.Event()
        self._closed.set()

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
        """Start the pool. Idempotent, so the container may call it more than once.

        A slot that could not be filled becomes an empty permit rather than a missing one, and
        the pool fills it on first use. The failure being avoided is a pool that silently
        shrinks: with the obvious shape — spawn eagerly, propagate — a transient ``OSError``
        would leave ``_started`` true and the queue short, and the *next* attempt would block
        on an empty queue with no timeout, no error and no log. At one worker, which is what
        ``default_worker_count`` returns on a two-core machine, that is a run that hangs
        forever.
        """
        async with self._lifecycle:
            if self._started:
                return
            spawning = asyncio.create_task(self._spawn_permits())
            interrupted, failure = await _settle_despite_cancellation(spawning)
            if interrupted or failure is not None:
                rollback = asyncio.create_task(self._terminate_snapshot(list(self._live)))
                rollback_interrupted, rollback_failure = await _settle_despite_cancellation(
                    rollback
                )
                self._live.clear()
                self._drain_idle()
                self._started = False
                if interrupted or rollback_interrupted:
                    raise asyncio.CancelledError
                if failure is not None:
                    raise failure
                if rollback_failure is not None:
                    raise rollback_failure
                raise RuntimeError("parse worker setup rollback failed without a cause")

            permits = spawning.result()
            for permit in permits:
                self._idle.put_nowait(permit)
            self._generation += 1
            self._closed = asyncio.Event()
            self._started = True

    async def teardown(self) -> None:
        """Stop every worker, including ones still holding a document."""
        async with self._lifecycle:
            self._started = False
            self._generation += 1
            self._closed.set()
            self._drain_idle()
            # The cleanup task owns this snapshot until every worker reaches terminate's known
            # endpoint.  Clearing `_live` first is safe only because cancellation cannot detach
            # the task; it also prevents replacements from making an already-retired worker
            # visible as live again.
            live, self._live = self._live, []
            stopping = asyncio.create_task(self._terminate_snapshot(live))
            interrupted, failure = await _settle_despite_cancellation(stopping)
            if interrupted:
                raise asyncio.CancelledError
            if failure is not None:
                raise failure

    async def __aenter__(self) -> Self:
        await self.setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.teardown()

    async def run_attempt(
        self,
        name: str,
        raw: RawDocument,
        *,
        max_output_bytes: int | None = None,
        memory_limit_bytes: int | None = None,
    ) -> AttemptResult:
        """Give one parser one turn, under this pool's time and memory limits.

        Returns the same :class:`AttemptResult` an in-process attempt returns, so the chain
        loop cannot tell the two apart — which is the point. A worker that was killed comes
        back as a hard failure naming the limit it hit, and a hard failure is what advances
        the chain *and* classifies as ``failed`` at stage ``parse``. It is emphatically not a
        decline: a chain of three timeouts must not end at ``unsupported_media_type``.
        """
        await self.setup()
        generation = self._generation
        worker = await self._acquire(generation)
        if isinstance(worker, _PoolStopped):
            reason = "parse worker pool stopped during this attempt"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        if worker is None:
            self._kills["spawn failed"] = self._kills.get("spawn failed", 0) + 1
            reason = "no parse worker could be started for this attempt"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        try:
            outcome = await self._dispatch(
                worker,
                _Request(
                    parser=name,
                    raw=raw,
                    max_output_bytes=max_output_bytes,
                    memory_limit_bytes=memory_limit_bytes,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - see below; the guarantee needs the breadth
            # **A broken pipe is one document's failure, not the run's.** A worker can die
            # between being handed back as idle and being dispatched to — recycled by the OOM
            # killer, or crashed on the previous document in a way that surfaced late — and
            # `send` then raises `BrokenPipeError` here. Letting that propagate would put a
            # process-level accident on the ingest loop's exception path and end the batch,
            # which is exactly the guarantee this whole module exists to hold. So it is
            # recorded against the document, the worker is replaced, and the run continues.
            interrupted = await self._complete_replacement(worker, generation)
            if interrupted:
                raise asyncio.CancelledError from None
            self._kills["dispatch failed"] = self._kills.get("dispatch failed", 0) + 1
            reason = f"worker unreachable: {type(exc).__name__}: {exc}"
            return AttemptResult([], Attempt(parser=name, outcome=Outcome.FAILED, reason=reason))
        except BaseException:
            # Cancellation and interpreter shutdown are *not* one document's problem, and
            # swallowing them would make Ctrl-C wait for a corpus. The worker still has to go:
            # a canceled await leaves one whose state nobody knows, and replacing it is
            # cheaper than reasoning about what it was in the middle of.
            await self._complete_replacement(worker, generation)
            raise
        if isinstance(outcome, _Killed):
            self._kills[outcome.reason] = self._kills.get(outcome.reason, 0) + 1
            interrupted = await self._complete_replacement(worker, generation)
            if interrupted:
                raise asyncio.CancelledError
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
            interrupted = await self._complete_replacement(worker, generation)
            if interrupted:
                raise asyncio.CancelledError
        else:
            interrupted = await self._complete_return(worker, generation)
            if interrupted:
                raise asyncio.CancelledError
        return outcome.result

    async def run_before_parse(
        self, raw: RawDocument, *, max_output_bytes: int, memory_limit_bytes: int
    ) -> StageResult:
        session = await self.open_stage_session(memory_limit_bytes=memory_limit_bytes)
        try:
            return await session.run_before_parse(
                raw,
                max_output_bytes=max_output_bytes,
                memory_limit_bytes=memory_limit_bytes,
            )
        finally:
            await session.aclose()

    async def run_after_parse_and_chunk(
        self,
        document: Document,
        blocks: list[ParsedBlock],
        *,
        max_output_bytes: int,
        memory_limit_bytes: int,
        title: str,
        media_type: str,
        detect_glossary: bool,
    ) -> StageResult:
        session = await self.open_stage_session(memory_limit_bytes=memory_limit_bytes)
        try:
            return await session.run_after_parse_and_chunk(
                document,
                blocks,
                max_output_bytes=max_output_bytes,
                memory_limit_bytes=memory_limit_bytes,
                title=title,
                media_type=media_type,
                detect_glossary=detect_glossary,
            )
        finally:
            await session.aclose()

    async def open_stage_session(self, *, memory_limit_bytes: int) -> _IsolatedStageSession:
        return await _IsolatedStageSession(
            self._context,
            self._config,
            timeout_s=self._timeout_s,
            poll_interval_s=self._poll_interval_s,
            memory_limit_bytes=memory_limit_bytes,
        ).start()

    # --- internals -------------------------------------------------------------------------

    async def _spawn_permits(self) -> list[_Worker | None]:
        return [await self._try_spawn() for _ in range(self._size)]

    def _drain_idle(self) -> None:
        while not self._idle.empty():
            self._idle.get_nowait()

    async def _terminate_snapshot(self, workers: list[_Worker]) -> None:
        """Terminate all workers even when one termination itself reports an error."""
        outcomes = await asyncio.gather(
            *(asyncio.to_thread(worker.terminate) for worker in workers),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    async def _acquire(self, generation: int) -> _Worker | _PoolStopped | None:
        """Own checkout through selection, close cleanup and any lazy spawn."""
        checkout = asyncio.create_task(self._acquire_owned(generation))
        try:
            return await self._deliver_checkout(checkout)
        except asyncio.CancelledError:
            checkout.cancel()
            while not checkout.done():
                try:
                    await asyncio.shield(checkout)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                except BaseException:  # noqa: BLE001 - caller cancellation has precedence
                    break
            if checkout.done() and not checkout.cancelled():
                with contextlib.suppress(BaseException):
                    checked_out = checkout.result()
                    if isinstance(checked_out, _Worker):
                        restoring = asyncio.create_task(
                            self._restore_permit(checked_out, generation)
                        )
                        await _join_despite_cancellation(restoring)
                        restoring.result()
            raise

    async def _deliver_checkout(
        self, checkout: asyncio.Task[_Worker | _PoolStopped | None]
    ) -> _Worker | _PoolStopped | None:
        """Transfer a completed checkout to its caller without canceling its owner task."""
        return await asyncio.shield(checkout)

    async def _acquire_owned(self, generation: int) -> _Worker | _PoolStopped | None:
        """Take a permit from the queue, and a worker with it.

        The queue holds *permits*, not workers: an entry may be ``None`` where a spawn has not
        succeeded yet. That is what keeps the permit count equal to the pool size no matter how
        many spawns have failed — the invariant whose loss turns a transient ``OSError`` into a
        run that blocks forever on an empty queue.
        """
        async with self._lifecycle:
            if not self._started or generation != self._generation:
                return _PoolStopped()
            closed = self._closed

        permit_task: asyncio.Task[_Worker | None] = asyncio.create_task(self._idle.get())
        closed_task = asyncio.create_task(closed.wait())
        permit: _Worker | object | None = _NO_PERMIT
        try:
            done, _ = await asyncio.wait(
                (permit_task, closed_task), return_when=asyncio.FIRST_COMPLETED
            )
            if closed_task in done:
                closed_task.result()
                if permit_task in done:
                    permit = permit_task.result()
                else:
                    permit = _NO_PERMIT
                    permit_task.cancel()
                closed_task.cancel()
                await asyncio.gather(permit_task, closed_task, return_exceptions=True)
                if permit is not _NO_PERMIT:
                    await self._restore_permit(cast("_Worker | None", permit), generation)
                return _PoolStopped()

            # Record ownership before the first cleanup await. Cancellation during the close
            # task's join must restore this exact permit rather than silently shrinking the
            # pool.
            permit = permit_task.result()
            await self._finish_permit_selection(closed_task)
            if permit is not None:
                return permit
            async with self._lifecycle:
                if not self._started or generation != self._generation:
                    return _PoolStopped()
                # Holding the lifecycle lock across readiness means teardown either precedes
                # this spawn or owns the resulting child; cancellation is handled by `_spawn`,
                # which reaps an appended child before it propagates.
                worker = await self._try_spawn()
                if worker is None:
                    # The permit goes back so the count is unchanged, and the next attempt
                    # tries again.
                    self._idle.put_nowait(None)
                    permit = _NO_PERMIT
                return worker
        except BaseException:
            if not permit_task.done():
                permit_task.cancel()
            closed_task.cancel()
            await asyncio.gather(permit_task, closed_task, return_exceptions=True)
            if permit is _NO_PERMIT and permit_task.done() and not permit_task.cancelled():
                permit = permit_task.result()
            if permit is not _NO_PERMIT:
                await self._restore_permit(cast("_Worker | None", permit), generation)
            raise

    async def _finish_permit_selection(self, closed_task: asyncio.Task[bool]) -> None:
        """Join the losing close waiter before checkout transfers its worker."""
        closed_task.cancel()
        await asyncio.gather(closed_task, return_exceptions=True)

    async def _restore_permit(self, permit: _Worker | None, generation: int) -> None:
        """Return a permit consumed at the same instant its checkout was canceled."""
        async with self._lifecycle:
            if self._started and generation == self._generation:
                self._idle.put_nowait(permit)
                return
        if permit is not None:
            await self._retire(permit)

    async def _dispatch(self, worker: _Worker, request: _Request) -> _Reply | _Killed:
        """Send one request and wait for its reply, entirely off the event loop.

        The *write* runs in the worker thread too, not only the wait. ``Connection.send``
        pickles the whole document and blocks until the child drains it, and a document may be
        up to ``ingest.max_fetch_bytes`` against a pipe buffer measured in kilobytes — so
        sending on the loop stalls every other connector's fetches for the length of the
        transfer, on exactly the documents that are already slow.
        """
        exchange = asyncio.create_task(
            asyncio.to_thread(
                _exchange,
                worker,
                request,
                self._timeout_s,
                self._poll_interval_s,
                min(
                    self._config.memory_limit_bytes,
                    request.memory_limit_bytes or self._config.memory_limit_bytes,
                ),
            )
        )
        try:
            return await asyncio.shield(exchange)
        except asyncio.CancelledError:
            _stop(worker)
            with contextlib.suppress(BaseException):
                await _join_despite_cancellation(exchange)
            raise

    async def _try_spawn(self) -> _Worker | None:
        """Start one worker, or return ``None`` if it could not be started.

        Returning rather than raising because the caller's response is never to abort: the pool
        keeps its permit, the document is failed on its own, and the next attempt tries again.
        A machine briefly out of file descriptors should cost a document, not a corpus.
        """
        try:
            return await self._spawn()
        except Exception:  # noqa: BLE001 - a spawn failure is not this run's ending
            return None

    async def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe(duplex=True)
        try:
            process = self._context.Process(
                target=_worker_main, args=(child, self._config), daemon=True
            )
            process.start()
        except BaseException:
            # Neither end is owned by a `_Worker` yet, so nothing else will ever close them.
            with contextlib.suppress(OSError):
                parent.close()
            with contextlib.suppress(OSError):
                child.close()
            raise
        child.close()
        worker = _Worker(process=process, connection=parent)
        self._live.append(worker)
        readiness = asyncio.create_task(asyncio.to_thread(_await_ready, parent))
        try:
            ready = await asyncio.shield(readiness)
        except BaseException:
            # Once appended, this coroutine owns the child until it returns a `_Worker`.
            # Stop it, join the pipe-reading thread, and reach retire's endpoint before
            # cancellation can propagate; otherwise both a child and a thread lose an owner.
            _stop(worker)
            with contextlib.suppress(BaseException):
                await _join_despite_cancellation(readiness)
            retiring = asyncio.create_task(self._retire(worker))
            await _join_despite_cancellation(retiring)
            with contextlib.suppress(BaseException):
                retiring.result()
            raise
        if ready is None or ready.error:
            # A worker that never said hello has an unknown pipe state: its greeting may still
            # be in flight, and the next reply read would return it instead of a result — a
            # blameless document recorded as killed. One that said hello *and* reported an
            # error cannot serve at all. Either way it is not a worker.
            retiring = asyncio.create_task(self._retire(worker))
            interrupted = await _join_despite_cancellation(retiring)
            retiring.result()
            if interrupted:
                raise asyncio.CancelledError
            detail = ready.error if ready is not None else "it never reported itself ready"
            msg = f"a parse worker could not start: {detail}"
            raise RuntimeError(msg)
        worker.address_space_limited = ready.address_space_limited
        return worker

    async def _retire(self, worker: _Worker) -> None:
        if worker in self._live:
            self._live.remove(worker)
        await asyncio.to_thread(worker.terminate)

    async def _replace(self, worker: _Worker, generation: int) -> None:
        """Stop a worker and return its permit, with a fresh worker on it if one can be had.

        One method rather than the pair repeated at four call sites, because the pair is only
        correct together — and because the permit must go back *whatever* happens to the spawn.
        Retiring without returning the permit shrinks the pool silently, and a pool that loses
        a permit per failure ends a long run blocked on an empty queue with nothing said.
        """
        async with self._lifecycle:
            # Removal from `_live` and process termination are one lifecycle operation.
            # Otherwise teardown can snapshot the list in between them, return with the old
            # child still being reaped, and let setup overlap two physical generations.
            await self._retire(worker)
            if not self._started or generation != self._generation:
                return
            # Teardown cannot snapshot between append-to-live and permit publication.
            self._idle.put_nowait(await self._try_spawn())

    async def _complete_replacement(self, worker: _Worker, generation: int) -> bool:
        """Reach retire/spawn/permit restoration despite repeated caller cancellation."""
        replacement = asyncio.create_task(self._replace(worker, generation))
        interrupted = await _join_despite_cancellation(replacement)
        replacement.result()
        return interrupted

    async def _return_or_retire(self, worker: _Worker, generation: int) -> None:
        async with self._lifecycle:
            if self._started and generation == self._generation:
                self._idle.put_nowait(worker)
                return
        await self._retire(worker)

    async def _complete_return(self, worker: _Worker, generation: int) -> bool:
        """Publish a release only to its owning generation, joined through cancellation."""
        returning = asyncio.create_task(self._return_or_retire(worker, generation))
        interrupted = await _join_despite_cancellation(returning)
        returning.result()
        return interrupted


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


def _exchange(
    worker: _Worker,
    request: _Request,
    timeout_s: float,
    poll_interval_s: float,
    memory_limit_bytes: int,
) -> _Reply | _Killed:
    """Send one request and wait for its reply, killing the worker if it overruns.

    Runs in a worker thread, so neither the blocking write nor the wait ever occupies the event
    loop. The wait is a poll loop rather than a single blocking read because the memory check
    has to happen *while* the parser is running: by the time a runaway returns, it has already
    allocated.

    **Every kill is gated on the worker still being ours.** The thread cannot be canceled, so
    when the awaiting task is canceled the pool retires this worker and reaps it — and a
    reaped pid is released to the operating system for reuse. Signaling a bare integer after
    that point sends ``SIGKILL`` to whatever process now holds it, and the likeliest victim is
    one of this pool's own replacements. Asking the ``Process`` object rather than trusting the
    number is what makes that impossible.
    """
    worker.connection.send(request)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # One last look before killing. A child that replied inside the final poll interval
            # has its answer sitting in the pipe, and killing it now would record a document
            # that parsed perfectly well as "worker killed: timeout".
            if worker.connection.poll(0):
                return _read_reply(worker.connection)
            _stop(worker)
            return _Killed("timeout")
        if worker.connection.poll(min(poll_interval_s, remaining)):
            return _read_reply(worker.connection)
        resident = resident_bytes(worker.pid) if worker.alive() else None
        if resident is not None and resident > memory_limit_bytes:
            _stop(worker)
            return _Killed("memory limit")


def _exchange_stage_message(
    process: SpawnProcess,
    connection: Connection,
    message: object | None,
    *,
    timeout_s: float,
    poll_interval_s: float,
    memory_limit_bytes: int,
) -> object:
    """Exchange one message without surrendering ownership of the persistent child."""
    if message is not None:
        try:
            connection.send(message)
        except (BrokenPipeError, EOFError, OSError):
            return StageResult(None, 0, 0, StageFailure.WORKER_DIED)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_stage_process(process)
            return StageResult(None, 0, 0, StageFailure.TIMEOUT)
        if connection.poll(min(poll_interval_s, remaining)):
            try:
                return connection.recv()
            except (EOFError, OSError):
                return StageResult(None, 0, 0, StageFailure.WORKER_DIED)
        try:
            alive = process.is_alive()
        except (ValueError, OSError):
            alive = False
        resident = resident_bytes(process.pid) if process.pid is not None and alive else None
        if resident is not None and resident > memory_limit_bytes:
            _kill_stage_process(process)
            return StageResult(None, 0, 0, StageFailure.MEMORY_BOUND)
        if not alive:
            return StageResult(None, 0, 0, StageFailure.WORKER_DIED)


def _kill_stage_process(process: SpawnProcess) -> None:
    if _stage_process_alive(process):
        with contextlib.suppress(ValueError, OSError):
            process.kill()


def _stage_process_alive(process: SpawnProcess) -> bool:
    try:
        return process.is_alive()
    except (AssertionError, ValueError, OSError):
        return False


def _stop(worker: _Worker) -> None:
    """Kill a worker, but only while it is still this pool's to kill."""
    if worker.retired or not worker.alive():
        return
    kill(worker.pid)


def _read_reply(connection: Connection) -> _Reply | _Killed:
    """Read one message, turning every way that can fail into this attempt's failure."""
    try:
        message = connection.recv()
    except (EOFError, OSError):
        return _Killed("worker exited without replying")
    except Exception as exc:  # noqa: BLE001 - a corrupt reply is this attempt's failure
        return _Killed(f"unreadable reply: {type(exc).__name__}")
    if isinstance(message, _Reply):
        return message
    return _Killed(f"unexpected reply: {type(message).__name__}")


def worker_config(
    settings: object, *, chunker: object | None = None, embedder: object | None = None
) -> WorkerConfig:
    """Build a worker's configuration from the application's settings.

    Takes the settings object structurally so that this module — and therefore the pipeline
    that imports it — does not force a configuration import on anything that only wants the
    types.
    """
    from manicule.config.settings import Settings  # noqa: PLC0415 - narrow, and only here

    if not isinstance(settings, Settings):  # pragma: no cover - a wiring mistake, not a path
        msg = f"expected Settings, got {type(settings).__name__}"
        raise TypeError(msg)
    stage_tokenizer_file: Path | None = None
    stage_tokenizer_id: str | None = None
    stage_max_tokens: int | None = None
    stage_overlap_tokens: int | None = None
    fingerprint = getattr(chunker, "fingerprint", None)
    card = getattr(embedder, "card", None)
    if (
        getattr(fingerprint, "chunker", None) == "structural"
        and card is not None
        and getattr(card, "path", None) is not None
    ):
        tokenizer_file = Path(card.path) / "tokenizer.json"
        if not tokenizer_file.is_file():
            msg = f"resolved embedding tokenizer is missing: {tokenizer_file}"
            raise ValueError(msg)
        resolved_fingerprint = cast("ChunkFingerprint", fingerprint)
        stage_tokenizer_file = tokenizer_file
        stage_tokenizer_id = resolved_fingerprint.tokenizer_id
        stage_max_tokens = resolved_fingerprint.max_tokens
        stage_overlap_tokens = resolved_fingerprint.overlap_tokens
    return WorkerConfig(
        workspace=settings.workspace,
        data_dir=settings.data_dir,
        cache_dir=settings.cache_dir,
        parser_fallbacks=dict(settings.parser_fallbacks),
        chunker=settings.rag.chunker,
        middleware=settings.plugins.middleware,
        plugins_enabled=settings.plugins.enabled,
        plugins_disabled=settings.plugins.disabled,
        plugin_config=settings.plugins.config,
        memory_limit_bytes=settings.ingest.parse_memory_limit_mb * MEGABYTE,
        stage_tokenizer_file=stage_tokenizer_file,
        stage_tokenizer_id=stage_tokenizer_id,
        stage_max_tokens=stage_max_tokens,
        stage_overlap_tokens=stage_overlap_tokens,
    )


__all__ = [
    "MEGABYTE",
    "AttemptResult",
    "InProcessRunner",
    "ParseRunner",
    "StageResult",
    "WorkerConfig",
    "WorkerPool",
    "attempt_one",
    "default_worker_count",
    "retained_size",
    "worker_config",
]
