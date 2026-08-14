"""The provider adapter: one call, one dependency, and the traps under it.

Local and hosted models differ by a ``base_url`` and a model prefix. There is no per-vendor
client, no per-vendor type, and no per-vendor branch to keep one alive.

**The library's types stop here.** Its response, delta and exception classes are converted
into :class:`~manicule.core.generation.Token`, :class:`~manicule.core.generation.Usage`,
:class:`~manicule.core.generation.FinishReason` and :mod:`manicule.core.errors` types.
Nothing above this module imports the provider library, and ``tests/test_import_boundary.py``
keeps it out of ``import manicule`` entirely.

Four things in here are traps rather than plumbing, and each is documented where it is
handled: the ``ollama_chat`` prefix (§4.2), the three timeouts (§4.5), the retry boundary
(§4.6) and the usage fallbacks (§4.10).
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import Any, Protocol, cast

from manicule.config.profiles import ProfileConfig
from manicule.config.providers import Egress
from manicule.config.settings import LlmSettings
from manicule.core.content import Document
from manicule.core.errors import (
    ConfigError,
    ContentFilteredError,
    ContextWindowError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from manicule.core.generation import FinishReason, Token, Usage
from manicule.core.lifecycle import HealthReport
from manicule.core.protocols import CLOSE_DEADLINE_S, bounded
from manicule.core.retrieval import Context, Query
from manicule.generation.prompt import ChatMessage, build_messages

OLLAMA_PROVIDER = "ollama"
OLLAMA_PREFIX = "ollama_chat"
"""**The Ollama provider prefix is ``ollama_chat``, not ``ollama``.**

They are different endpoints: ``ollama/`` routes to ``/api/generate`` and ``ollama_chat/`` to
``/api/chat``. The failure under ``ollama/`` is **double-templating**, not a missing template:
the library first flattens the ``messages`` array into a single string using *its own* prompt
template, and Ollama then applies the *model's* real template on top of that already-formatted
string. The model receives its own turn markers wrapped around a foreign approximation of
them.

It still answers — plausibly, in fluent prose, with nothing raised anywhere — and it follows
instructions worse, which for this design means it follows the citation protocol worse. The
only symptom is more dropped markers, which reads as a weak model rather than as a wiring
mistake. So configuration keeps the name an operator expects and the wire gets the endpoint
that works.
"""

FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "error": FinishReason.ERROR,
}


class CompletionCall(Protocol):
    """The one provider call, as this module uses it.

    A protocol rather than a direct import so that the whole adapter — timeouts, retries,
    error mapping, usage handling — is exercised by tests without a network, while the
    default remains the real library.
    """

    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...  # noqa: ANN401 - a provider SDK's own kwargs


def compose_model(provider: str, model: str) -> str:
    """``<provider>/<model>``, with the one correction that is not cosmetic."""
    name = provider.strip().lower()
    prefix = OLLAMA_PREFIX if name == OLLAMA_PROVIDER else name
    return f"{prefix}/{model.strip()}"


LOCAL_COST_MAP_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"
"""The provider library's switch for using the model table it ships with.

Set by :func:`_litellm` when the operator has not, and the reason is the same one behind
:mod:`manicule.vocabularies`: **importing the library performs an HTTPS GET to a third-party
host** — the model pricing and context-window table, from raw.githubusercontent.com, on a
five-second timeout — and it does it on the path that answers a question. It is not a call to
the configured model server, and it is not optional at import.

Its failure is silent: the library falls back to the copy in its own wheel. So an air-gapped
host pays five seconds per process and gets the local table anyway, and a networked host gets
whichever revision that repository is serving today. Setting this makes both hosts use the
same table — the one pinned in the lockfile — which is the difference between an answer that
depends on the machine and one that does not.

Only set when unset, so an operator who wants the live table, or a mirror through
``LITELLM_MODEL_COST_MAP_URL``, keeps it.
"""


def _litellm() -> Any:  # noqa: ANN401 - the library's own surface, deliberately untyped here
    """Import the provider library. Deferred, so registration never pays for it."""
    os.environ.setdefault(LOCAL_COST_MAP_ENV, "True")
    import litellm  # noqa: PLC0415 - the whole point is that this is not a module import

    return litellm


def error_table() -> tuple[tuple[type[Exception], type[GenerationError]], ...]:
    """Provider exceptions to manicule errors, **most specific first**.

    Table-driven rather than a chain of ``except`` blocks, and the order is asserted by a
    test. ``ContextWindowExceededError`` and ``ContentPolicyViolationError`` both subclass
    the library's ``BadRequestError``, so a generic bad-request arm placed above them
    swallows the two cases that have specific, actionable remedies and reports them as a
    generic bad request. This is the one place in the adapter where a correct-looking
    refactor silently deletes a diagnosis.
    """
    litellm = _litellm()
    exceptions = litellm.exceptions
    return (
        (exceptions.ContextWindowExceededError, ContextWindowError),
        (exceptions.ContentPolicyViolationError, ContentFilteredError),
        (exceptions.AuthenticationError, ProviderAuthError),
        (exceptions.PermissionDeniedError, ProviderAuthError),
        (exceptions.RateLimitError, ProviderRateLimitError),
        (exceptions.Timeout, ProviderTimeoutError),
        (exceptions.APIConnectionError, ProviderConnectionError),
        (exceptions.ServiceUnavailableError, ProviderConnectionError),
        (exceptions.InternalServerError, ProviderConnectionError),
        (exceptions.BadRequestError, ProviderRequestError),
        (exceptions.APIError, ProviderRequestError),
    )


MAX_LEADING_CHUNKS = 8
"""How many text-free opening chunks may be held while the retry window is still open.

Enough for the role-only chunk every OpenAI-compatible provider sends, plus slack for a
provider that sends a few more; small enough that a keep-alive stream cannot be buffered.
"""

TIMEOUT_SETTINGS: Mapping[str, str] = {
    "first token": "llm.first_token_timeout_s",
    "gap between tokens": "llm.stream_idle_timeout_s",
    "total generation time": "llm.timeout_s",
}
"""Which setting governs each interval, so a message names the knob that would have helped."""

RETRYABLE: frozenset[type[GenerationError]] = frozenset(
    {ProviderConnectionError, ProviderRateLimitError, ProviderTimeoutError}
)
"""Which mapped failures get another attempt, **and only before the first token**.

Not authentication, not a context-window error, not a content filter: none of those gets
better by being asked again, and retrying them turns one clear error into three and a longer
wait.
"""


class LitellmGenerator:
    """One :class:`~manicule.core.protocols.Generator`, for every provider.

    ``history`` is an **optional** keyword on :meth:`generate`, which
    :func:`manicule.testing.assert_protocol_signatures` explicitly permits: a caller working
    from the protocol never passes it, and the answer path passes it when the bound generator
    accepts it. The protocol has no channel for conversation history and widening it is a
    seam change; a plugin generator that wants multi-turn adds the same keyword, and one that
    does not is recorded as not having it rather than quietly losing the conversation.
    """

    def __init__(
        self,
        settings: LlmSettings,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        egress: Egress = Egress.LOOPBACK,
        completion: CompletionCall | None = None,
        profile: ProfileConfig | None = None,
        profile_name: str = "",
        system_prompt_tokens: int = 0,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._base_url = base_url
        self._egress = egress
        self._completion = completion
        self._profile = profile
        self._profile_name = profile_name
        self._system_prompt_tokens = system_prompt_tokens
        self._extra_params = dict(extra_params or {})
        # Derived from the profile arithmetic, never configured, so the window manicule
        # demands cannot disagree with the budget it exists to satisfy — and so the served
        # window stops varying with the host's available memory.
        self._num_ctx = (
            None
            if profile is None
            else profile.context_tokens
            + profile.history_tokens
            + system_prompt_tokens
            + settings.max_tokens
        )
        self.model_id = compose_model(settings.provider, settings.model)
        self.context_window = settings.context_window or 0

    # --- identity ------------------------------------------------------------------

    @property
    def is_ollama(self) -> bool:
        return self._settings.provider.strip().lower() == OLLAMA_PROVIDER

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def num_ctx(self) -> int | None:
        """The window demanded of a served local runtime, or ``None``.

        **Derived, never configured**, so it cannot disagree with the budget it exists to
        satisfy — and so the served window stops varying with the host's available memory.
        """
        return self._num_ctx if self.is_ollama else None

    # --- lifecycle -----------------------------------------------------------------

    async def setup(self) -> None:
        """Refuse a configuration that cannot answer, before the first question.

        Deliberately **not** a live probe of a hosted provider: that costs money and latency
        on every start, and the two things checkable without it — the credential resolves,
        the library knows the model — catch the mistakes people actually make. A diagnostic
        command performs the live probe on request, which is where a check with a cost
        belongs.
        """
        self._require_known_model()
        self._require_credential()
        if self.is_ollama:
            await self._require_ollama_ready()
        if self.context_window <= 0:
            self.context_window = await self.resolve_context_window()
        self._require_budget_fits()

    def _require_known_model(self) -> None:
        litellm = _litellm()
        try:
            litellm.get_llm_provider(model=self.model_id)
        except Exception as exc:
            prefixes = ", ".join(sorted(str(name) for name in litellm.provider_list)[:12])
            msg = (
                f"llm.provider {self._settings.provider!r} with llm.model "
                f"{self._settings.model!r} composes to {self.model_id!r}, which the provider "
                f"library does not recognize: {exc}. Provider prefixes include {prefixes}. "
                f"For an OpenAI-compatible server, use provider 'openai' with a base_url."
            )
            raise ConfigError(msg) from exc

    def _require_budget_fits(self) -> None:
        """The startup window cross-check, at the one place that knows both numbers.

        **The predicate is retrieval's**, not a second copy of the rule. Retrieval owns the
        requirement — ``context_tokens`` and ``history_tokens`` are its budgets — and assigns
        the enforcement point here, because this is where the served window finally becomes
        known. Two statements of one rule are two rules that will disagree, and the one that
        disagrees silently is the one that lets a prompt overflow.
        """
        if self._profile is None:
            return
        # Deferred: retrieval's assembly module reaches for the tokenizer machinery, and
        # registration must stay free of it.
        from manicule.retrieval.assembly import window_problem  # noqa: PLC0415

        problem = window_problem(
            self._profile,
            context_window=self.context_window,
            model_id=self.model_id,
            system_prompt_tokens=self._system_prompt_tokens,
            generation_reserve=self._settings.max_tokens,
        )
        if problem:
            # The shared predicate says "choose a lighter profile" without knowing which one is
            # in force, and that is unactionable on its own: an operator cannot pick a lighter
            # profile without being told what they are running. Naming it is this layer's to
            # add, because configuration is what this layer holds.
            named = (
                f" The configured profile is {self._profile_name!r}." if self._profile_name else ""
            )
            raise ConfigError(f"{problem}{named}")

    def _require_credential(self) -> None:
        from manicule.config.providers import env_var_names, needs_credential  # noqa: PLC0415

        provider = self._settings.provider.strip().lower()
        if not needs_credential(provider) or self._api_key:
            return
        expected = " or ".join(env_var_names(provider))
        msg = (
            f"provider {provider!r} needs an API key and none resolved. Set {expected}, or "
            f"providers.{provider}.api_key."
        )
        raise ConfigError(msg)

    async def _require_ollama_ready(self) -> None:
        """Check that the endpoint answers and the named model is pulled.

        A configuration whose endpoint is absent fails here, with the two commands that fix
        it — not at the first question a user asks.
        """
        tags = await self._ollama_get("/api/tags")
        if tags is None:
            msg = (
                f"no Ollama server answered at {self._ollama_base()}. Start one with `ollama "
                f"serve`, point llm.base_url at the right host, or select a hosted provider."
            )
            raise ConfigError(msg)
        listed = cast("list[object]", tags.get("models") or [])
        models = {
            str(cast("dict[str, object]", entry).get("name", ""))
            for entry in listed
            if isinstance(entry, dict)
        }
        if self._settings.model not in models:
            available = ", ".join(sorted(models)) or "none"
            msg = (
                f"llm.model {self._settings.model!r} is not pulled on the Ollama server at "
                f"{self._ollama_base()}. Pulled: {available}. Run `ollama pull "
                f"{self._settings.model}`."
            )
            raise ConfigError(msg)

    def _ollama_base(self) -> str:
        return (self._base_url or "http://localhost:11434").rstrip("/")

    async def _ollama_get(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, object] | None:
        """One JSON call to the Ollama admin API, or ``None`` if it did not answer.

        The provider library speaks HTTP and brings its own client, which is what keeps
        Ollama a *runtime* dependency of one configuration rather than an install dependency
        of manicule. There is no Ollama client library here and there will not be one.
        """
        import httpx  # noqa: PLC0415 - transitively present via the provider library

        url = f"{self._ollama_base()}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._settings.first_token_timeout_s) as client:
                response = (
                    await client.post(url, json=payload)
                    if payload is not None
                    else await client.get(url)
                )
                response.raise_for_status()
                body: object = response.json()
                return cast("dict[str, object]", body) if isinstance(body, dict) else None
        except Exception:  # noqa: BLE001 - every failure here means the same thing: no answer
            return None

    async def resolve_context_window(self) -> int:
        """The window that will actually be **served**, not the model's advertised maximum.

        For Ollama it is read from ``/api/show``: the trained length is the ceiling on what
        manicule may demand through ``num_ctx``, and ``num_ctx`` is what is actually served.
        Reporting the demanded window here instead would make the startup cross-check
        circular — it would be comparing the budget against a number derived from the budget.

        For a hosted provider it is the library's own model metadata. When neither can
        describe the model — an OpenAI-compatible server with a private model name —
        ``llm.context_window`` is the escape hatch, and without it this is a refusal rather
        than a guess, because a limit that can only be discovered by exceeding it gets
        discovered in production.
        """
        if self._settings.context_window:
            return self._settings.context_window
        window = await self._ollama_window() if self.is_ollama else self._hosted_window()
        if window > 0:
            return window
        msg = (
            f"could not determine the context window {self.model_id!r} will serve, so the "
            f"profile budget cannot be checked against it. Set llm.context_window to the "
            f"window this endpoint actually serves."
        )
        raise ConfigError(msg)

    async def _ollama_window(self) -> int:
        shown = await self._ollama_get("/api/show", {"model": self._settings.model})
        described = cast("dict[str, object]", (shown or {}).get("model_info") or {})
        for key, value in described.items():
            # The key is architecture-prefixed — `qwen2.context_length`, `llama.context_length`
            # — so the suffix is the only stable part, and hardcoding an architecture would
            # silently stop finding the window the day somebody pulls a different model.
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return 0

    def _hosted_window(self) -> int:
        litellm = _litellm()
        try:
            info: Any = litellm.get_model_info(model=self.model_id)
        except Exception:  # noqa: BLE001 - an unknown model is a normal, handled outcome
            return 0
        described = cast("dict[str, object]", info) if isinstance(info, dict) else {}
        window = described.get("max_input_tokens")
        return window if isinstance(window, int) else 0

    async def health(self) -> HealthReport:
        if not self.is_ollama:
            return HealthReport.healthy(f"{self.model_id} at {self._base_url or 'the default'}")
        if await self._ollama_get("/api/tags") is None:
            return HealthReport.failing(
                f"no Ollama server answered at {self._ollama_base()}",
                remedy="run `ollama serve`, or point llm.base_url at the right host",
            )
        return HealthReport.healthy(f"{self.model_id} at {self._ollama_base()}")

    # --- generation ----------------------------------------------------------------

    def generate(
        self,
        query: Query,
        context: Context,
        *,
        history: Sequence[ChatMessage] = (),
        documents: Mapping[str, Document] | None = None,
        messages: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[Token]:
        """Stream the answer. Iterate through
        :func:`~manicule.core.protocols.generating`, never bare.

        **``messages`` wins when it is given, and the answer path always gives it.** Building
        the prompt down here from ``query`` and ``context`` was wrong in a way that was
        invisible: the answer path had already built a *redacted* prompt — titles, URIs and
        heading paths substituted — and used it for the token estimate, and then this method
        rebuilt an unredacted one from the raw arguments and sent that. The trace reported
        redaction that had been computed and discarded.

        It is also what makes the slot numbering enforceable. The correspondence between "slot
        3" and ``context.passages[2]`` is the whole basis of the citation guarantee, and while
        the prompt was built inside the pluggable component that correspondence was an
        unenforced convention: a plugin that reordered passages, or numbered from zero,
        produced citations naming a passage the model never saw at that number — mechanically
        wrong, passing all three verification levels, and indistinguishable from the
        misattribution the design honestly excludes.

        The fallback remains for a caller working from the protocol alone, which cannot pass
        keywords the protocol does not declare.
        """
        return self.stream(
            messages
            if messages is not None
            else build_messages(
                query_text=query.text,
                context=context,
                documents=documents or {},
                history=history,
                system_extra=self._settings.system_prompt_extra,
            )
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[Token]:
        """The provider call, as an async generator with cleanup around the ``yield``.

        The ``finally`` is the contract: ``aclose()`` throws :exc:`GeneratorExit` in at the
        suspension point and a canceled client's :exc:`asyncio.CancelledError` arrives at
        whatever ``await`` is live inside the provider read. One ``try``/``finally`` covers
        both, it awaits nothing unbounded, and it never yields — the final token carrying a
        finish reason is emitted on the normal path only, because a stream nobody is reading
        has nobody to tell.
        """
        started = time.monotonic()
        total_deadline = started + self._settings.timeout_s
        stream: Any = None
        try:
            stream, leading = await self._open(messages, total_deadline)
            finish: FinishReason | None = None
            usage: Usage | None = None
            pending: list[Any] = list(leading)
            while pending:
                chunk = pending.pop(0)
                finish = _finish_reason(chunk) or finish
                usage = _usage(chunk) or usage
                text = _delta_text(chunk)
                if text:
                    yield Token(text=text)
                if pending:
                    continue
                try:
                    following = await self._next(stream, total_deadline)
                except GenerationError as exc:
                    # After the first token there is no correct retry: restarting means the
                    # reader watches the answer rewind, and continuing splices two
                    # independently-sampled answers into text no single generation produced.
                    # The usage seen so far travels with the error: truncation is the path
                    # where cost accounting matters most.
                    yield Token(finish_reason=FinishReason.ERROR, error=str(exc), usage=usage)
                    return
                if following is not None:
                    pending.append(following)
            yield Token(finish_reason=finish or FinishReason.STOP, usage=usage)
        finally:
            await _close(stream)

    async def _open(
        self, messages: Sequence[ChatMessage], total_deadline: float
    ) -> tuple[Any, list[Any]]:
        """Open the stream and read up to the first chunk **bearing text**, retrying until then.

        Up to the first *token*, not the first chunk. Every OpenAI-compatible provider opens
        with a role-only chunk carrying no content, so stopping at the first chunk closes the
        retry window before anything has been delivered — and a connection dropping a
        millisecond later, which is the commonest real failure shape, becomes terminal for a
        reader who has seen nothing. Re-consuming a couple of empty leading chunks on a fresh
        stream costs nothing and is invisible.

        The retry loop is manicule's, and the library's own ``num_retries`` is deliberately
        unused. Under ``stream=True`` it is wrong in three separate ways, each invisible: it
        wraps the call that *returns* the stream wrapper, so a failure raised mid-iteration
        is never retried; its gate matches every exception the library defines, so a bad API
        key becomes three slow attempts at a bad API key; and it counts total *attempts*, so
        ``num_retries=1`` performs no retry at all.
        """
        attempt = 0
        while True:
            remaining = max(total_deadline - time.monotonic(), 0.0)
            if remaining <= 0:
                raise ProviderTimeoutError(self._timeout_message("total generation time", 0.0))
            budget = min(self._settings.first_token_timeout_s, remaining)
            stream: Any = None
            try:
                stream = await self._call(messages, budget)
                leading: list[Any] = []
                while True:
                    chunk = await self._next(stream, total_deadline, budget, phase="first token")
                    if chunk is None:
                        return stream, leading
                    leading.append(chunk)
                    # Bounded. A provider that streams keep-alive frames with no content would
                    # otherwise have every one of them held until the first token, rebuilt from
                    # scratch on each retry — hundreds of thousands of objects on a long
                    # first-token budget. Past the bound the retry window simply closes, which
                    # is the pre-existing behavior and is safe: nothing has been delivered.
                    if _delta_text(chunk) or len(leading) >= MAX_LEADING_CHUNKS:
                        return stream, leading
            except GenerationError as exc:
                # An attempt that opened a stream and then failed still holds a connection to
                # a model that is working. Retrying without closing it would leave one live
                # provider call per attempt, which is the leak this whole design is careful
                # about, arriving through the recovery path.
                await _close(stream)
                left = max(total_deadline - time.monotonic(), 0.0)
                if type(exc) not in RETRYABLE or attempt >= self._settings.max_retries or not left:
                    raise
                attempt += 1
                # Clamped to the wall clock. An unclamped backoff sleeps seconds past a
                # deadline that has already expired, on a budget that cannot improve.
                await asyncio.sleep(min(_backoff(attempt), left))
            except BaseException:
                # Cancellation lands here, and the caller's `stream` is still unassigned — so
                # this frame is the only thing holding the connection. Losing it means an open
                # response to a model that keeps generating, which is exactly what the cleanup
                # contract exists to prevent.
                await _close(stream)
                raise

    async def _call(self, messages: Sequence[ChatMessage], budget: float) -> Any:  # noqa: ANN401
        completion = self._completion
        if completion is None:
            completion = _litellm().acompletion
        # Provider extras go in first so that everything manicule sets below overrides them.
        # The other order would let a config file quietly switch off streaming, the usage
        # flag or a timeout — each of which is a guarantee this module makes.
        kwargs: dict[str, Any] = dict(self._extra_params)
        kwargs |= {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "stream": True,
            # Under streaming, a usage-bearing final chunk is gated on this flag; without it
            # usage is reachable only through a private attribute, which is not something to
            # build a correctness guarantee on. It is an OpenAI-compatible parameter, so for
            # `ollama_chat` the library drops it before the request and honors it
            # client-side — which is fine, and is not the same as end-to-end support.
            "stream_options": {"include_usage": True},
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "timeout": budget,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self.is_ollama:
            kwargs["keep_alive"] = self._settings.keep_alive
            if self._num_ctx:
                kwargs["num_ctx"] = self._num_ctx
        if budget <= 0:
            raise ProviderTimeoutError(self._timeout_message("total generation time", 0.0))
        try:
            return await asyncio.wait_for(completion(**kwargs), budget)
        except TimeoutError as exc:
            # Which budget bound this call is decided by which was smaller, not by where in
            # the stream we are: a total shorter than the first-token allowance is the
            # constraint that expired, and naming the other one is a wrong remedy.
            expired = (
                "first token"
                if budget >= self._settings.first_token_timeout_s
                else "total generation time"
            )
            raise ProviderTimeoutError(self._timeout_message(expired, budget)) from exc
        except Exception as exc:
            raise map_error(exc, self.model_id, self._api_key_source()) from exc

    async def _next(
        self,
        stream: Any,  # noqa: ANN401 - a provider SDK's own stream wrapper
        total_deadline: float,
        budget: float | None = None,
        *,
        phase: str = "gap between tokens",
    ) -> Any:  # noqa: ANN401 - and its own chunk type
        """The next chunk, under whichever of the three deadlines expires first.

        ``phase`` names which one this call is under rather than leaving it to be inferred:
        the same code serves the first-token wait and the inter-token gap, and a message that
        guesses sends an operator to raise a setting with no effect on the interval that
        actually expired.
        """
        remaining_total = max(total_deadline - time.monotonic(), 0.0)
        gap = self._settings.stream_idle_timeout_s if budget is None else budget
        allowed = min(gap, remaining_total)
        try:
            return await asyncio.wait_for(stream.__anext__(), allowed)
        except StopAsyncIteration:
            return None
        except TimeoutError as exc:
            expired = "total generation time" if remaining_total <= gap else phase
            raise ProviderTimeoutError(self._timeout_message(expired, allowed)) from exc
        except Exception as exc:
            raise map_error(exc, self.model_id, self._api_key_source()) from exc

    def _timeout_message(self, which: str, budget: float) -> str:
        knob = TIMEOUT_SETTINGS.get(which, "llm.timeout_s")
        return (
            f"{self.model_id} exceeded the {which} budget of {budget:.1f}s. Raise {knob}. The "
            f"three cover different intervals: llm.first_token_timeout_s "
            f"({self._settings.first_token_timeout_s}s) covers connect, queue, prompt "
            f"evaluation and model load; llm.stream_idle_timeout_s "
            f"({self._settings.stream_idle_timeout_s}s) covers the gap between two tokens; "
            f"llm.timeout_s ({self._settings.timeout_s}s) is the total wall clock."
        )

    def _api_key_source(self) -> str:
        from manicule.config.providers import env_var_names, needs_credential  # noqa: PLC0415

        provider = self._settings.provider.strip().lower()
        if not needs_credential(provider):
            return ""
        return " or ".join(env_var_names(provider))


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Jittered because an un-jittered backoff synchronizes every client of a rate-limited
    provider onto the same retry instant, which is the shape that turns one bad minute into
    a sustained one.
    """
    return random.uniform(0.0, min(2.0**attempt * 0.25, 8.0))  # noqa: S311 - backoff, not crypto


def map_error(exc: Exception, model: str, key_source: str = "") -> GenerationError:
    """Convert a provider exception, preserving its message text.

    Discarding the provider's own words is how ``"OpenAI error: 429"`` becomes an operator's
    whole afternoon, so the original text is always kept and manicule's context is added in
    front of it.
    """
    if isinstance(exc, GenerationError):
        return exc
    for provider_type, mapped in error_table():
        if isinstance(exc, provider_type):
            return mapped(_message(mapped, exc, model, key_source))
    return ProviderRequestError(f"{model}: {type(exc).__name__}: {exc}")


def _message(mapped: type[GenerationError], exc: Exception, model: str, key_source: str) -> str:
    if mapped is ProviderAuthError and key_source:
        return (
            f"{model} rejected the credential read from {key_source}: {exc}. Check the key, "
            f"or set providers.<name>.api_key."
        )
    if mapped is ContextWindowError:
        return (
            f"{model} says the prompt did not fit its window: {exc}. This is a defect in the "
            f"startup window cross-check rather than a runtime condition — the estimate and "
            f"the server disagreed by more than llm.token_safety_factor allows."
        )
    return f"{model}: {exc}"


def _delta_text(chunk: Any) -> str:  # noqa: ANN401 - a provider SDK's own chunk type
    choices: Any = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta: Any = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None
    return content if isinstance(content, str) else ""


def _finish_reason(chunk: Any) -> FinishReason | None:  # noqa: ANN401
    choices: Any = getattr(chunk, "choices", None) or []
    if not choices:
        return None
    reason: object = getattr(choices[0], "finish_reason", None)
    return FINISH_REASONS.get(str(reason)) if reason else None


def _usage(chunk: Any) -> Usage | None:  # noqa: ANN401
    reported: Any = getattr(chunk, "usage", None)
    if reported is None:
        return None
    prompt: object = getattr(reported, "prompt_tokens", None)
    completion: object = getattr(reported, "completion_tokens", None)
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return Usage(prompt_tokens=max(prompt, 0), completion_tokens=max(completion, 0))


async def _close(stream: Any) -> None:  # noqa: ANN401
    """Release the provider connection, under its own hard deadline.

    What an abandoned generation stream holds is an open HTTP response to a model that is
    still working: on a hosted provider, tokens billed for an answer nobody will read; on a
    local one, the only model on the machine occupied until it finishes, so a user who closed
    a tab and asked again is queued behind their own abandoned answer.

    Past the deadline the connection is abandoned to the pool's own teardown, because a
    shutdown path that can block indefinitely on a misbehaving remote server is a worse
    failure than a leaked socket.
    """
    if stream is None:
        return
    closer: Any = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result: object = closer()
    except Exception:  # noqa: BLE001 - cleanup never raises over the failure it is unwinding
        return
    if asyncio.iscoroutine(result):
        # The same bound, from the same helper, for the same reason: `wait_for` would cancel
        # the close and then wait for that cancellation, so a provider client that catches
        # `CancelledError` during teardown holds the shutdown open regardless.
        await bounded(result, CLOSE_DEADLINE_S)


__all__ = [
    "FINISH_REASONS",
    "OLLAMA_PREFIX",
    "OLLAMA_PROVIDER",
    "RETRYABLE",
    "CompletionCall",
    "LitellmGenerator",
    "compose_model",
    "error_table",
    "map_error",
]
