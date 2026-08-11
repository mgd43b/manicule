"""The answer path: policy, prompt, provider, binder, persistence.

This is the thing that sits **above** :class:`~manicule.core.protocols.Generator` and outside
any protocol. Everything that makes a citation trustworthy lives here, because a boundary a
plugin can omit is not a boundary — and a third-party generator is exactly the component that
would implement ``generate`` and forget to verify anything.

Order matters in two places and both are load-bearing:

**Verification starts before the model call**, since it depends only on the context. By the
time a marker arrives the answer for that passage is usually already in hand.

**Persistence lives in this module's ``finally``**, not in the consumer, because on a client
disconnect there is no consumer left to run it. Whatever text was produced becomes a message
that exists on the server, has an id, can be shared, and can be rated.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field

from manicule.config.profiles import profile_config
from manicule.config.settings import Settings
from manicule.core.content import Document
from manicule.core.errors import GenerationError
from manicule.core.generation import FinishReason, Token
from manicule.core.protocols import Generator, aclose, generating
from manicule.core.retrieval import Context, Query
from manicule.generation.answers import (
    AnswerEnvelope,
    AnswerEvent,
    CitationAccounting,
    GenerationTrace,
    PolicyDrop,
)
from manicule.generation.binder import CitationBinder
from manicule.generation.budget import (
    GENERATION_ENCODING,
    TokenEstimator,
    drift_problem,
    usable_prompt_tokens,
)
from manicule.generation.history import HistoryPlan, Turn, fit_history
from manicule.generation.policy import EgressPolicy, filter_context
from manicule.generation.ports import ConversationStore, StoredMessage
from manicule.generation.prompt import ChatMessage, build_messages, messages_text, system_message
from manicule.generation.redaction import Redactor
from manicule.generation.verification import CitationVerifier, DocumentLookup, load_documents

PERSIST_DEADLINE_S = 10.0
"""How long the shielded write of a partial answer may take.

A cancelled request that can outlive its own shutdown indefinitely is a worse failure than a
lost partial answer.
"""


def accepted_extras(generator: Generator) -> frozenset[str]:
    """Which optional keywords the bound generator declares beyond the protocol.

    Inspected once, so a generator that cannot take conversation history is a recorded fact
    rather than a conversation that quietly vanishes. See
    :func:`~manicule.core.protocols.generating` for why the extras travel this way at all.
    """
    try:
        signature = inspect.signature(generator.generate)
    except (TypeError, ValueError):  # pragma: no cover - a callable with no introspectable shape
        return frozenset()
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )


@dataclass(slots=True)
class AnswerRequest:
    """One question, with everything that is known before the model is asked."""

    query: Query
    context: Context
    conversation_id: str | None = None
    confidence: float | None = None
    corpus_consulted: bool = True
    """False for an answer that never consulted the corpus.

    Such an answer carries **no citations**, states that the corpus was not consulted, and has
    confidence *absent* — not 1.0 and not 0.0. With an empty context there are no slots, so
    every marker the model emits fails at level 0 and is deleted, and the answer is **not**
    flagged ungrounded: that flag means "the context was non-empty and nothing survived", and
    this context was empty by design.
    """

    query_log_id: str | None = None
    history: Sequence[Turn] = ()


@dataclass(frozen=True, slots=True)
class Prepared:
    """The prompt as it will actually be sent, and what it cost to make.

    ``sent_query`` and ``sent_context`` are **redacted copies**. They are what the generator
    receives; the originals stay with the binder and the verifier, which is the whole reason
    a redacted passage still produces a citation that resolves.
    """

    plan: HistoryPlan
    sent_query: Query
    sent_context: Context
    sent_history: tuple[ChatMessage, ...]
    sent_documents: Mapping[str, Document]
    messages: tuple[ChatMessage, ...]
    redaction_counts: Mapping[str, int]


@dataclass(slots=True)
class AnswerResult:
    """What one answer produced, once its stream is finished."""

    envelope: AnswerEnvelope = field(default_factory=AnswerEnvelope)
    trace: GenerationTrace = field(default_factory=GenerationTrace)
    message_id: str | None = None
    drift: str = ""


class Answerer:
    """Turns a question and an assembled context into a stream of answer events."""

    def __init__(
        self,
        *,
        generator: Generator,
        verifier: CitationVerifier,
        documents: DocumentLookup,
        settings: Settings,
        policy: EgressPolicy,
        redactor: Redactor,
        estimator: TokenEstimator | None = None,
        conversations: ConversationStore | None = None,
        history_tokens: int | None = None,
    ) -> None:
        self._generator = generator
        self._verifier = verifier
        self._documents = documents
        self._settings = settings
        self._policy = policy
        self._redactor = redactor
        self._estimator = estimator or TokenEstimator(
            safety_factor=settings.llm.token_safety_factor
        )
        self._conversations = conversations
        # Derived from the profile rather than defaulted, because the startup window
        # cross-check reserves `profile.history_tokens` and refuses a configuration that does
        # not fit. A separate default here could spend twice what that check approved, which
        # is the prompt overflow the refusal exists to prevent, arriving from inside.
        self._history_tokens = (
            history_tokens
            if history_tokens is not None
            else profile_config(settings.rag.profile, settings.rag.overrides).history_tokens
        )
        self._extras = accepted_extras(generator)
        self._detached: set[asyncio.Future[str]] = set()

    @property
    def history_budget(self) -> int:
        """Tokens conversation history may spend, as the profile reserved them."""
        return self._history_tokens

    @property
    def history_supported(self) -> bool:
        """Whether the bound generator can be given conversation history at all.

        Recorded rather than assumed. A generator written strictly to the protocol has no
        channel for it, and a conversation that silently stops reaching the model looks
        exactly like a model with a short memory.
        """
        return "history" in self._extras

    @property
    def system_prompt_tokens(self) -> int:
        """What the system prompt costs, including any operator additions.

        Counted rather than assumed, because operator text is appended to it and a long custom
        prompt that displaced passages instead of being refused is precisely the failure the
        startup cross-check exists to prevent.
        """
        return self._estimator.count(
            system_message(self._settings.llm.system_prompt_extra)["content"]
        )

    async def answer(
        self, request: AnswerRequest, result: AnswerResult | None = None
    ) -> AsyncIterator[AnswerEvent]:
        """Stream one answer. Always ends with a ``final`` event carrying the envelope.

        Every failure is reported **in band**. A truncated answer that simply ends is
        indistinguishable from a complete one, and the reader has no way to know which they
        got.

        ``result`` is filled in as the answer proceeds and is where the
        :class:`~manicule.generation.answers.GenerationTrace`, the persisted message id and
        any token drift end up. Passing one in is how a caller reads them: an async generator
        cannot also return a value, and putting the trace on the final event would make every
        consumer carry a record it mostly does not want. It sits **beside** the retrieval
        trace and deliberately not in ``query_logs``, which is whole-query product telemetry.
        """
        started = time.monotonic()
        result = result or AnswerResult()
        binder: CitationBinder | None = None
        run = None
        prepared: Prepared | None = None
        policy_drops: tuple[PolicyDrop, ...] = ()
        estimate = 0
        try:
            try:
                documents = await load_documents(self._documents, request.context)
                context, policy_drops = filter_context(request.context, documents, self._policy)
                run = self._verifier.start(context, documents, started_at=started)
                binder = CitationBinder(run=run)

                prepared = await self._prepare(request, context, documents)
                estimate = self._estimator.count(messages_text(prepared.messages))

                # Closed explicitly. `_stream` owns the `generating()` block, and an async
                # generator abandoned by an unwinding `for` loop runs its `finally` only when
                # the interpreter's finalizer gets to it — a later tick, or never on a loop
                # that closes first. That would put the provider connection outside
                # `answering()`'s promise entirely.
                async with aclosing(self._stream(prepared, documents, binder, result)) as events:
                    async for event in events:
                        yield event

            except Exception as exc:  # noqa: BLE001 - the last thing standing between a failure and silence
                # Everything before the first token can fail too — the document store, the
                # redactor, the estimator's first use of its vocabulary. A caller written to
                # the documented contract reads `events[-1].envelope`, so a failure that
                # emits nothing at all is the one shape it cannot survive.
                result.envelope = self._failed(binder, result, f"{type(exc).__name__}: {exc}")

            # **Outside the guard**, deliberately. All of this runs after the answer has been
            # streamed in full, so a bug in the bookkeeping was turning a delivered, complete
            # answer into `finish_reason=error` — telling the reader that the text in front of
            # them had failed, and persisting it that way. A failure here loses a trace, which
            # is a diagnostic; it must not lose the answer, which is the product.
            if binder is not None and prepared is not None:
                result.envelope = self._envelope(request, binder, policy_drops, result, prepared)
                result.trace = self._trace(
                    binder,
                    prepared,
                    result,
                    policy_drops=policy_drops,
                    estimate=estimate,
                    started=started,
                )
                result.drift = drift_problem(
                    estimate=estimate,
                    measured=usable_prompt_tokens(result.envelope.usage, estimate),
                    tolerance=self._settings.llm.token_drift_tolerance,
                    model=self._generator.model_id,
                )
            yield AnswerEvent.final(result.envelope)
        finally:
            # Two independent cleanups, so a failure in the first cannot skip the second.
            # Persistence is the one that must not be lost: the answer that exists nowhere is
            # the one that most needed to be ratable.
            try:
                if run is not None:
                    await run.aclose()
            finally:
                self._settle_envelope(binder, result)
                await self._persist(request, binder, result)

    # --- preparation ---------------------------------------------------------------

    async def _prepare(
        self, request: AnswerRequest, context: Context, documents: Mapping[str, Document]
    ) -> Prepared:
        """Fit history, redact what is leaving, and build the prompt.

        Redaction is applied to **a copy on its way out**. The verification chain —
        ``Chunk.text`` ↔ ``Anchor`` ↔ retained bytes — never sees a redacted string, which is
        why a redacted passage still produces a citation that resolves.
        """
        plan = fit_history(request.history, budget=self._history_tokens, estimator=self._estimator)
        # Built from the same decision `_stream` makes about what the generator will accept.
        # Estimating a conversation that never leaves the machine inflates the count, and that
        # count feeds the drift comparison — so a generator with no history channel would
        # report tokenizer drift forever, and could silently degrade a real measurement that
        # happened to equal the inflated figure.
        offered = plan.messages if self.history_supported else ()
        if not self._policy.should_redact:
            return Prepared(
                plan=plan,
                sent_query=request.query,
                sent_context=context,
                sent_history=offered,
                sent_documents=_referenced(context, documents),
                messages=tuple(
                    build_messages(
                        query_text=request.query.text,
                        context=context,
                        documents=documents,
                        history=offered,
                        system_extra=self._settings.llm.system_prompt_extra,
                    )
                ),
                redaction_counts={},
            )

        passages = [candidate.chunk.text for candidate in context.passages]
        turns = [message["content"] for message in offered]
        # **Every text channel that leaves, not only the passage bodies.** A slot label is
        # rendered from the document's title and the chunk's heading path, so redacting the
        # body alone ships `"Q3 comp review — someone@example.invalid.docx"` above a redacted
        # passage — three of the five channels protected, two not, under a claim that covers
        # all five. The extra strings are a handful of short ones.
        titles = [document.title for document in documents.values()]
        uris = [document.uri for document in documents.values()]
        trails = [part for candidate in context.passages for part in candidate.chunk.heading_path]
        redacted, counts = await self._redactor.redact_all(
            [request.query.text, *passages, *turns, *titles, *uris, *trails]
        )
        cursor = 1
        query_text = redacted[0] or "[REDACTED]"
        passage_texts, cursor = redacted[cursor : cursor + len(passages)], cursor + len(passages)
        turn_texts, cursor = redacted[cursor : cursor + len(turns)], cursor + len(turns)
        title_texts, cursor = redacted[cursor : cursor + len(titles)], cursor + len(titles)
        uri_texts, cursor = redacted[cursor : cursor + len(uris)], cursor + len(uris)
        trail_texts = redacted[cursor:]

        sent_documents = _referenced(
            context,
            {
                document_id: document.model_copy(update={"title": title, "uri": uri})
                for (document_id, document), title, uri in zip(
                    documents.items(), title_texts, uri_texts, strict=True
                )
            },
        )
        trail = iter(trail_texts)
        sent_context = context.model_copy(
            update={
                "passages": tuple(
                    candidate.model_copy(
                        update={
                            "chunk": candidate.chunk.model_copy(
                                update={
                                    "text": text,
                                    "heading_path": tuple(
                                        next(trail) for _ in candidate.chunk.heading_path
                                    ),
                                }
                            )
                        }
                    )
                    for candidate, text in zip(context.passages, passage_texts, strict=True)
                )
            }
        )
        sent_history: list[ChatMessage] = [
            {"role": message["role"], "content": text}
            for message, text in zip(offered, turn_texts, strict=True)
        ]
        return Prepared(
            plan=plan,
            sent_query=request.query.model_copy(update={"text": query_text}),
            sent_context=sent_context,
            sent_history=tuple(sent_history),
            sent_documents=sent_documents,
            messages=tuple(
                build_messages(
                    query_text=query_text,
                    context=sent_context,
                    documents=sent_documents,
                    history=sent_history,
                    system_extra=self._settings.llm.system_prompt_extra,
                )
            ),
            redaction_counts=counts,
        )

    # --- streaming -----------------------------------------------------------------

    async def _stream(
        self,
        prepared: Prepared,
        documents: Mapping[str, Document],
        binder: CitationBinder,
        result: AnswerResult,
    ) -> AsyncGenerator[AnswerEvent]:
        """Drive the provider and bind what comes back.

        A failure from *any* generator becomes an in-band error token here, including one a
        plugin raised from the middle of its own iterator. The guarantee that a dying stream
        says so belongs to this layer, not to the adapter, because a plugin cannot be relied
        on to keep it.
        """
        extra: dict[str, object] = {}
        # The prompt itself, when the generator can take it. Everything below is the fallback
        # for a generator that cannot: the *parts*, from which it builds its own — and note
        # that `sent_documents` rather than `documents` is what carries the redacted titles
        # and URIs, so passing the raw map here would send what redaction had just removed.
        if "messages" in self._extras:
            extra["messages"] = prepared.messages
        if self.history_supported:
            extra["history"] = prepared.sent_history
        if "documents" in self._extras:
            extra["documents"] = prepared.sent_documents
        first_token_at: float | None = None
        began = time.monotonic()
        try:
            async with generating(
                self._generator, prepared.sent_query, prepared.sent_context, extra=extra
            ) as tokens:
                async for token in tokens:
                    if token.text and first_token_at is None:
                        first_token_at = time.monotonic()
                    for event in await self._on_token(token, binder, result):
                        yield event
        except GenerationError as exc:
            for event in await self._fail(binder, result, str(exc)):
                yield event
        except Exception as exc:  # noqa: BLE001 - a provider's own bug must not end mid-stream silently
            for event in await self._fail(binder, result, f"{type(exc).__name__}: {exc}"):
                yield event
        finally:
            result.trace = result.trace.model_copy(
                update={
                    "first_token_ms": _millis(began, first_token_at),
                    "total_ms": _millis(began, time.monotonic()),
                }
            )

    async def _on_token(
        self, token: Token, binder: CitationBinder, result: AnswerResult
    ) -> list[AnswerEvent]:
        """What one provider token produced, as a list.

        A list rather than a generator for the same reason the binder returns one: this is
        consumed from inside another async generator, and a nested one abandoned by an
        unwinding loop keeps its cleanup until something finalises it.
        """
        produced: list[AnswerEvent] = []
        if token.text:
            produced.extend(await binder.feed(token.text))
        if token.finish_reason is not None:
            produced.extend(await binder.finish())
            result.envelope = result.envelope.model_copy(
                update={
                    "finish_reason": token.finish_reason,
                    "usage": token.usage,
                    "error": token.error,
                }
            )
        return produced

    async def _fail(
        self, binder: CitationBinder, result: AnswerResult, detail: str
    ) -> list[AnswerEvent]:
        """Report a mid-stream failure in band, keeping what was already delivered.

        **Citations already emitted stand.** They were verified when they were emitted, and
        the provider failing afterwards says nothing about them — the answer is marked
        truncated, not uncited.
        """
        produced = await binder.finish()
        result.envelope = result.envelope.model_copy(
            update={"finish_reason": FinishReason.ERROR, "error": detail}
        )
        return produced

    # --- results -------------------------------------------------------------------

    def _envelope(
        self,
        request: AnswerRequest,
        binder: CitationBinder,
        policy_drops: tuple[PolicyDrop, ...],
        result: AnswerResult,
        prepared: Prepared,
    ) -> AnswerEnvelope:
        return result.envelope.model_copy(
            update={
                "text": binder.text,
                "confidence": request.confidence if request.corpus_consulted else None,
                "citations": binder.citations,
                "dropped": binder.drops,
                "policy_dropped": policy_drops,
                "accounting": binder.accounting,
                "verification_level": self._verifier.ceiling,
                "ungrounded": binder.ungrounded and request.corpus_consulted,
                "corpus_consulted": request.corpus_consulted,
                "context_truncated": request.context.truncated,
                "egress": self._policy.egress,
                "redacted": bool(prepared.redaction_counts),
            }
        )

    def _trace(
        self,
        binder: CitationBinder,
        prepared: Prepared,
        result: AnswerResult,
        *,
        policy_drops: tuple[PolicyDrop, ...],
        estimate: int,
        started: float,
    ) -> GenerationTrace:
        accounting = binder.accounting
        plan = prepared.plan
        usage = result.envelope.usage
        return result.trace.model_copy(
            update={
                "model": self._generator.model_id,
                "endpoint": self._policy.endpoint.base_url,
                "egress": self._policy.egress,
                "context_window": self._generator.context_window,
                "num_ctx_sent": getattr(self._generator, "num_ctx", None),
                "estimated_prompt_tokens": estimate,
                "true_prompt_tokens": usable_prompt_tokens(usage, estimate),
                "completion_tokens": usage.completion_tokens if usage else None,
                "encoding_name": GENERATION_ENCODING,
                "safety_factor": self._settings.llm.token_safety_factor,
                "total_ms": _millis(started, time.monotonic()),
                "finish_reason": result.envelope.finish_reason,
                "slots_offered": accounting.slots_offered,
                "markers_seen": accounting.markers_seen,
                "citations_verified": accounting.verified,
                "drops": binder.drops,
                "verification_level": self._verifier.ceiling,
                "verification_cache_hits": binder.run.cache_hits,
                "redaction_scope": self._policy.redaction_scope.value,
                # Names and counts. **Never what a detector matched** — that would turn the
                # trace into the leak the detector existed to prevent.
                "detectors_fired": dict(prepared.redaction_counts),
                "policy_dropped": policy_drops,
                "turns_offered": plan.turns_offered,
                "turns_sent": plan.turns_sent if self.history_supported else 0,
                "history_tokens": plan.tokens if self.history_supported else 0,
                "history_conditioned_retrieval": False,
            }
        )

    def _failed(
        self, binder: CitationBinder | None, result: AnswerResult, detail: str
    ) -> AnswerEnvelope:
        """The envelope for an answer that never reached the model, or died before its end."""
        return result.envelope.model_copy(
            update={
                "text": binder.text if binder else "",
                "citations": binder.citations if binder else (),
                "dropped": binder.drops if binder else (),
                "accounting": binder.accounting if binder else CitationAccounting(),
                "finish_reason": FinishReason.ERROR,
                "error": detail,
                "egress": self._policy.egress,
            }
        )

    def _settle_envelope(self, binder: CitationBinder | None, result: AnswerResult) -> None:
        """Give an abandoned answer a finish reason before it is written down.

        Reaching here with none means the stream was closed part-way — a client that
        disconnected. Persisting it unmarked would store a partial answer that reads exactly
        like a complete one, which is the distinction the whole in-band reporting rule exists
        to preserve.
        """
        if result.envelope.finish_reason is not None:
            return
        result.envelope = result.envelope.model_copy(
            update={
                "text": binder.text if binder else "",
                "citations": binder.citations if binder else (),
                "dropped": binder.drops if binder else (),
                "accounting": binder.accounting if binder else CitationAccounting(),
                "finish_reason": FinishReason.ERROR,
                "error": result.envelope.error or "the answer was abandoned before it finished",
                "egress": self._policy.egress,
            }
        )

    # --- persistence ---------------------------------------------------------------

    async def _persist(
        self, request: AnswerRequest, binder: CitationBinder | None, result: AnswerResult
    ) -> None:
        """Write the answer, whatever became of it, under a shield and a deadline.

        **The folklore about this is wrong and the design tests it rather than assuming it.**
        :exc:`asyncio.CancelledError` is delivered *once*, so an ``await`` inside a ``finally``
        reached by cancellation completes normally — a write here is not silently skipped and
        needs no shield to run at all.

        The real exposure is narrower and is a **second** cancellation: a shutdown escalation
        arriving while the write is in flight. That is what the shield is for. It carries a
        deadline for the same reason a cancelled request must not outlive its own shutdown,
        and it never restarts the request or issues another provider call — it writes what is
        already in hand and returns.
        """
        if self._conversations is None or binder is None or request.conversation_id is None:
            return
        message = StoredMessage(
            conversation_id=request.conversation_id,
            role="assistant",
            content=binder.text,
            citations=binder.citations,
            envelope=result.envelope,
            finish_reason=result.envelope.finish_reason,
            profile_used=request.query.profile.value,
            confidence_score=result.envelope.confidence,
            query_log_id=request.query_log_id,
            response_time_ms=result.trace.total_ms,
        )
        write = asyncio.ensure_future(self._conversations.append(message))
        try:
            result.message_id = await asyncio.wait_for(asyncio.shield(write), PERSIST_DEADLINE_S)
        except (TimeoutError, asyncio.CancelledError) as exc:
            # The shielded write keeps going; only the wait for it was abandoned. A reference
            # is held so the task is not collected mid-write — which is how "shielded" turns
            # into "silently dropped" — and its outcome is consumed so a failed write does not
            # surface later as an unhandled-task traceback.
            self._detached.add(write)
            write.add_done_callback(self._detached.discard)
            write.add_done_callback(_swallow)
            result.message_id = None
            # A **second** cancellation propagates: this is a `finally` unwinding a shutdown,
            # and swallowing it would strand the caller. A timeout does not — it is a slow
            # conversation store, and turning a complete, delivered answer into a raised
            # `TimeoutError` that reads like a provider timeout is a worse failure than a
            # lost message id.
            if isinstance(exc, asyncio.CancelledError):
                raise
        except Exception:  # noqa: BLE001 - a storage failure must not replace the answer's own error
            result.message_id = None


@asynccontextmanager
async def answering(
    answerer: Answerer, request: AnswerRequest, result: AnswerResult | None = None
) -> AsyncGenerator[AsyncIterator[AnswerEvent]]:
    """Consume an answer, closing it on **every** exit path.

    :func:`~manicule.core.protocols.generating`'s sibling one layer up, and it exists because
    of the same fact about async generators: an abandoned one stays suspended at its ``yield``
    and its ``finally`` does not run until something finalises it. That ``finally`` is where
    the partial answer is written down, so **a client disconnect that abandons the stream is
    exactly the case where persistence would silently not happen** — the answer that most
    needs to be ratable is the one that would exist nowhere.

    Closing it is what turns a cancellation into a ``GeneratorExit`` at the suspension point,
    which runs the cleanup. The deadline is wider than
    :data:`PERSIST_DEADLINE_S` on purpose: a close that expired before the shielded write
    could finish would defeat the thing it is here to guarantee::

        async with answering(answerer, request, result) as events:
            async for event in events:
                ...
    """
    stream = answerer.answer(request, result)
    try:
        yield stream
    finally:
        await aclose(stream, timeout=PERSIST_DEADLINE_S * 2)


def _referenced(context: Context, documents: Mapping[str, Document]) -> dict[str, Document]:
    """Only the documents the **surviving** passages point into.

    Policy filtering removes a passage from the context and left its document in this map,
    which then went to the generator through the ``documents`` extra. The built-in ignores it
    when it is handed a prompt — so nothing showed — but a plugin does not, and receives the
    title, URI and metadata of a source an operator marked ``local_only`` precisely so it
    would not leave. A leak of exactly what the policy exists to stop, through the seam the
    policy sits above.
    """
    live = {candidate.chunk.document_id for candidate in context.passages}
    return {
        document_id: document for document_id, document in documents.items() if document_id in live
    }


def _swallow(finished: asyncio.Future[str]) -> None:
    """Consume a detached write's outcome, so a failure is not reported at collection time."""
    if not finished.cancelled():
        finished.exception()


def _millis(start: float, end: float | None) -> int | None:
    return None if end is None else max(int((end - start) * 1000), 0)


__all__ = [
    "PERSIST_DEADLINE_S",
    "AnswerRequest",
    "AnswerResult",
    "Answerer",
    "accepted_extras",
    "answering",
]
