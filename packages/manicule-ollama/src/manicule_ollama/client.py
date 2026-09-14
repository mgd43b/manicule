"""The only place this package speaks HTTP, and the only place it knows Ollama's API shape.

Confined to one module for the reason :mod:`manicule.embedding.runtimes.hub` gives about the
Hugging Face hub: the rest of the backend takes measured values and has no opinion about where
they came from, and the failures a *network* runtime has — an unreachable host, a model nobody
pulled, a request refused for length — get said once, in the vocabulary of the person reading
them, instead of arriving from inside a library they did not choose.

Three responses carry everything this backend knows about the model, and each is a
*measurement* rather than a declaration this package supplies:

``/api/tags``
    the digest of the blob the server will actually run, which is what makes a re-pull
    invalidate the vectors the previous pull produced.

``/api/show``
    ``model_info``: the architecture's context length, embedding width, pooling type and
    tokenizer flags. GGUF metadata written by whoever converted the model, read here rather
    than inferred from the model's name.

``/api/embed``
    the vectors, and ``prompt_eval_count`` — the server's own count of the tokens it read.
    That number is the whole reason this backend can hold itself to a token budget at all:
    it is how a locally configured tokenizer stops being a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import httpx

from manicule.core.embedding import Vector
from manicule.core.errors import ConfigError, ManiculeError

BACKEND: Final = "ollama"

_CONTEXT_REFUSAL: Final = "exceeds the context length"
"""What Ollama says when ``truncate=false`` stops it dropping the tail of an input.

Matched as a substring of the server's own message rather than by status code alone: a 400
also covers a malformed request, and answering that with a context-overflow error would send
somebody to re-chunk a corpus over a typo.
"""


class OllamaUnavailableError(ManiculeError):
    """The Ollama server could not be reached, or answered something unusable.

    Raised here rather than left as ``httpx``'s own exception for the reason
    :class:`~manicule.embedding.runtimes.hub.ModelUnavailableError` exists: an embedder is
    constructed on the path that answers a query, so an unreachable server is a *search* that
    did not answer, and the message a person needs names the address that was tried.
    """


class OllamaContextOverflowError(ManiculeError):
    """The server refused an input as longer than the context it is serving.

    Not :class:`~manicule.core.errors.ContextOverflowError` at this layer, and translated into
    one by the backend. This module reports what the server said; deciding that it means "a
    chunk claimed text the model never saw" is the embedder's job, and it is the embedder that
    knows which text was in the batch.
    """


@dataclass(frozen=True, slots=True)
class ServedModelInfo:
    """What ``/api/show`` and ``/api/tags`` say about one model on one server."""

    model: str
    """The name as configured, e.g. ``qwen3-embedding:0.6b``."""

    digest: str
    """The manifest digest, without the ``sha256:`` prefix. Moves when the model is re-pulled."""

    architecture: str
    """``general.architecture``: ``qwen3``, ``nomic-bert``. Names which ``<arch>.*`` keys exist."""

    context_length: int
    """``<arch>.context_length``: the longest sequence the architecture can address."""

    embedding_length: int
    """``<arch>.embedding_length``. Recorded for the cross-check; the dimension that reaches a
    fingerprint is measured from a real vector, never from this."""

    pooling_type: int
    """``<arch>.pooling_type``, in llama.cpp's numbering. See :data:`POOLING_TYPES`."""

    capabilities: tuple[str, ...]
    """What the server says this model can do. ``embedding`` has to be among them."""


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """One ``/api/embed`` response."""

    vectors: list[Vector]
    prompt_eval_count: int
    """Total tokens the server read, across every input in the request.

    A total rather than a per-input breakdown, which is why the tokenizer check in
    :mod:`manicule_ollama.served` sends one text at a time: the useful comparison is against a
    single string's count, and a sum would pass while two inputs were wrong in opposite
    directions.
    """


class OllamaClient:
    """A typed seam over the three Ollama endpoints this backend uses.

    Synchronous *and* asynchronous on purpose. The declaration is read during construction —
    the chunker takes the embedder as a construction dependency and refuses to start when its
    budget exceeds this model's sequence limit, which has to happen before ingest rather than
    after — and a constructor cannot await. Embedding is asynchronous because an ingest run
    embeds while an HTTP request is in flight.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        base_url: The server's address.
        timeout_s: Total wall clock for one request.
        connect_timeout_s: Establishing the connection only.
        transport: Synchronous transport, for a suite driving a synthetic server.
        async_transport: Asynchronous transport, likewise. Supplied rather than patched so
            that the request building and the error mapping under test are the real ones.
        """
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self._transport = transport
        self._async_transport = async_transport
        self._async: httpx.AsyncClient | None = None

    # --- reading the declaration, synchronously ------------------------------------------

    def describe(self, model: str) -> ServedModelInfo:
        """Everything the server declares about ``model``, in one value.

        Raises:
            ConfigError: The server does not hold this model, or holds it without the
                ``embedding`` capability, or its metadata omits something with no honest
                substitute.
            OllamaUnavailableError: The server could not be reached.
        """
        name, digest = self._digest(model)
        shown = self._post_sync("/api/show", {"model": model})
        info = _mapping(shown.get("model_info"), "model_info", model)
        capabilities = tuple(
            str(item) for item in cast("Sequence[object]", shown.get("capabilities") or ())
        )
        if "embedding" not in capabilities:
            listed = ", ".join(capabilities) or "none"
            msg = (
                f"{model!r} on {self.base_url} does not declare the 'embedding' capability "
                f"(it declares: {listed}). Ollama will answer /api/embed for a generative "
                f"model by pooling its hidden states, which produces well-shaped vectors "
                f"from a model never trained to make them. Pull an embedding model."
            )
            raise ConfigError(msg)

        architecture = str(info.get("general.architecture") or "")
        if not architecture:
            msg = (
                f"{model!r} on {self.base_url} declares no general.architecture, so the "
                f"context length, width and pooling keys cannot be located — every one of "
                f"them is named after it. This is a GGUF without the metadata manicule reads."
            )
            raise ConfigError(msg)
        return ServedModelInfo(
            model=name,
            digest=digest,
            architecture=architecture,
            context_length=_positive_int(info, f"{architecture}.context_length", model, self),
            embedding_length=_positive_int(info, f"{architecture}.embedding_length", model, self),
            pooling_type=_pooling_type(info, architecture, model, self),
            capabilities=capabilities,
        )

    def _digest(self, model: str) -> tuple[str, str]:
        """The name the server uses for ``model`` and the digest behind it, read synchronously."""
        return self._match_digest(self._post_sync("/api/tags", None, method="GET"), model)

    async def digest(self, model: str) -> str:
        """The same question, asked from the event loop.

        Separate from :meth:`_digest` rather than shared through a thread, because the callers
        differ in kind: the synchronous one runs during construction, where nothing is awaiting,
        and this one runs inside :meth:`~manicule_ollama.backend.OllamaEmbedder.health` — which
        an operator scrapes while an ingest is in flight, and where a blocking HTTP call would
        stall every other coroutine in the process for as long as the server took to answer.
        """
        return self._match_digest(await self._get("/api/tags"), model)[1]

    def _match_digest(self, listed: Mapping[str, object], model: str) -> tuple[str, str]:
        """Pick ``model``'s name and digest out of an ``/api/tags`` listing.

        From ``/api/tags`` rather than ``/api/show``, which does not report it. Matched against
        the name Ollama itself uses, including the ``:latest`` it appends to a bare name — so a
        configuration saying ``nomic-embed-text`` finds ``nomic-embed-text:latest`` rather than
        being told a model it can see in ``ollama list`` does not exist.

        **The server's spelling is what comes back, not the configuration's**, and that is an
        identity decision rather than a tidiness one. ``nomic-embed-text`` and
        ``nomic-embed-text:latest`` are one blob with one digest, so they have to be one
        embedding identity — otherwise an operator who rewrites their configuration to the tag
        ``ollama list`` prints gets a fingerprint mismatch and a full re-embed for a change that
        moved nothing about the vectors.
        """
        models = cast("Sequence[object]", listed.get("models") or ())
        wanted = model if ":" in model else f"{model}:latest"
        names: list[str] = []
        for entry in models:
            if not isinstance(entry, Mapping):
                continue
            record = cast("Mapping[str, object]", entry)
            name = str(record.get("name") or record.get("model") or "")
            names.append(name)
            if name == wanted:
                digest = str(record.get("digest") or "")
                if not digest:
                    msg = (
                        f"{wanted!r} on {self.base_url} is listed without a digest. The "
                        f"digest is what makes a re-pull invalidate the vectors the previous "
                        f"pull produced, so a model without one cannot be given an identity."
                    )
                    raise ConfigError(msg)
                return name, digest.removeprefix("sha256:")
        available = ", ".join(sorted(names)) or "nothing"
        msg = (
            f"{self.base_url} is not serving {wanted!r}. It holds: {available}. Run "
            f"`ollama pull {model}` on that host, or set `embedding.model` to one of them."
        )
        raise ConfigError(msg)

    def embed_sync(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        num_ctx: int,
        keep_alive: str,
        truncate: bool = False,
    ) -> EmbedResult:
        """One embedding request, synchronously. Used by the construction-time measurements."""
        payload = _embed_payload(
            model, inputs, num_ctx=num_ctx, keep_alive=keep_alive, truncate=truncate
        )
        return _embed_result(self._post_sync("/api/embed", payload), model, self.base_url)

    # --- embedding, asynchronously -------------------------------------------------------

    async def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        num_ctx: int,
        keep_alive: str,
        truncate: bool = False,
    ) -> EmbedResult:
        """One embedding request.

        ``truncate`` defaults to **false**, which is the whole point and is the opposite of
        Ollama's own default. With it true — the server's default — an input past the served
        context is silently shortened and answered with a well-formed vector describing its
        opening. With it false the server returns a 400, which reaches a caller as
        :class:`OllamaContextOverflowError` and the backend turns into a
        :class:`~manicule.core.errors.ContextOverflowError` naming the text.
        """
        payload = _embed_payload(
            model, inputs, num_ctx=num_ctx, keep_alive=keep_alive, truncate=truncate
        )
        return _embed_result(await self._post("/api/embed", payload), model, self.base_url)

    async def aclose(self) -> None:
        """Release the connection pool. Safe twice."""
        client, self._async = self._async, None
        if client is not None:
            await client.aclose()

    # --- transport -----------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._async is None:
            self._async = httpx.AsyncClient(
                base_url=self.base_url, timeout=self._timeout, transport=self._async_transport
            )
        return self._async

    async def _get(self, path: str) -> Mapping[str, object]:
        try:
            response = await self._client().get(path)
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(self._unreachable(path, exc)) from exc
        return self._decode(response, path)

    async def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        try:
            response = await self._client().post(path, json=dict(payload))
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(self._unreachable(path, exc)) from exc
        return self._decode(response, path)

    def _post_sync(
        self, path: str, payload: Mapping[str, object] | None, *, method: str = "POST"
    ) -> Mapping[str, object]:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=self._timeout, transport=self._transport
            ) as client:
                response = client.request(
                    method, path, json=None if payload is None else dict(payload)
                )
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(self._unreachable(path, exc)) from exc
        return self._decode(response, path)

    def _decode(self, response: httpx.Response, path: str) -> Mapping[str, object]:
        if response.status_code >= httpx.codes.BAD_REQUEST:
            detail = _error_text(response)
            if _CONTEXT_REFUSAL in detail:
                raise OllamaContextOverflowError(detail)
            raise OllamaUnavailableError(
                f"{self.base_url}{path} answered {response.status_code}: {detail}"
            )
        try:
            parsed: object = response.json()
        except ValueError as exc:
            # `ValueError` rather than `json.JSONDecodeError`, which is one of its subclasses:
            # a response whose bytes are not decodable text raises `UnicodeDecodeError`, also a
            # `ValueError`, and letting that escape would take a diagnostic down with it.
            raise OllamaUnavailableError(
                f"{self.base_url}{path} answered {response.status_code} with a body that is "
                f"not JSON. This is usually a proxy or a captive portal between manicule and "
                f"the server rather than the server itself."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise OllamaUnavailableError(
                f"{self.base_url}{path} answered with {type(parsed).__name__}, not an object"
            )
        return cast("Mapping[str, object]", parsed)

    def _unreachable(self, path: str, exc: Exception) -> str:
        return (
            f"the Ollama server at {self.base_url} could not be reached for {path}: {exc}. "
            f"manicule holds no weights for this backend — the model runs on that host — so "
            f"an unreachable server is not something a retry or a pre-seed fixes. Check "
            f'`[plugins.config."embedder.ollama"] base_url`, and that the server answers '
            f"`curl {self.base_url}/api/tags`."
        )


POOLING_TYPES: Final[Mapping[int, str]] = {
    0: "none",
    1: "mean",
    2: "cls",
    3: "last",
    4: "rank",
}
"""llama.cpp's ``LLAMA_POOLING_TYPE_*`` enumeration, which is what a GGUF records.

Translated to manicule's :class:`~manicule.core.embedding.Pooling` in
:mod:`manicule_ollama.served`, where an unmapped value is refused rather than defaulted — the
reason ``cards.py`` refuses an unsupported Sentence-Transformers flag rather than taking the
next one that happens to be set.
"""


def _embed_payload(
    model: str,
    inputs: Sequence[str],
    *,
    num_ctx: int,
    keep_alive: str,
    truncate: bool,
) -> dict[str, object]:
    """The request body, with ``num_ctx`` always present.

    **Always**, because Ollama's default is not the model's context length. Measured against a
    server holding ``qwen3-embedding:0.6b``, whose GGUF declares 32768: an /api/embed with no
    options was served at 4096 and answered a longer input with a vector built from its first
    4095 tokens. Sending the number makes the served limit the one this backend derived its
    ``max_sequence_length`` from, rather than one the server chose and never mentioned.
    """
    return {
        "model": model,
        "input": list(inputs),
        "truncate": truncate,
        "keep_alive": keep_alive,
        "options": {"num_ctx": num_ctx},
    }


def _embed_result(body: Mapping[str, object], model: str, base_url: str) -> EmbedResult:
    raw = body.get("embeddings")
    if not isinstance(raw, list):
        raise OllamaUnavailableError(
            f"{base_url}/api/embed answered without an 'embeddings' array for {model!r}"
        )
    vectors: list[Vector] = []
    for index, row in enumerate(cast("list[object]", raw)):
        if not isinstance(row, list):
            raise OllamaUnavailableError(
                f"{base_url}/api/embed returned {type(row).__name__} where vector {index} "
                f"belongs. A vector is a list of numbers; anything else is a server this "
                f"backend does not understand rather than a value to coerce."
            )
        try:
            vectors.append(
                [_component(value, index, model, base_url) for value in cast("list[object]", row)]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            # A `null`, a string, or a number no float can hold, somewhere inside the array.
            # Everything this module raises is one of its own two errors precisely so the
            # backend's mapping covers it; a bare `TypeError` from a comprehension would
            # escape that and arrive from inside a library the operator did not choose.
            raise OllamaUnavailableError(
                f"{base_url}/api/embed returned a component of vector {index} for {model!r} "
                f"that is not a number ({exc}). A vector is a list of numbers; anything else "
                f"is a server this backend does not understand rather than a value to coerce."
            ) from exc
    count = body.get("prompt_eval_count")
    return EmbedResult(vectors=vectors, prompt_eval_count=count if isinstance(count, int) else -1)


def _component(value: object, index: int, model: str, base_url: str) -> float:
    """One number out of a returned vector, with the two non-numbers that ``float`` accepts.

    ``float("1.5")`` and ``float(True)`` both succeed, so a component arriving as a string or a
    JSON ``true`` would be *coerced* rather than refused — and 1.0 is a perfectly plausible
    component. The type is checked before the conversion so that a server sending something
    other than numbers is a refusal rather than a vector nobody can tell apart from a real one.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = (
            f"{base_url}/api/embed returned {type(value).__name__} as a component of vector "
            f"{index} for {model!r}. A vector is a list of numbers; `float()` would accept a "
            f"string or a boolean and turn it into a plausible component, so this is refused "
            f"rather than coerced."
        )
        raise OllamaUnavailableError(msg)
    return float(value)


def _error_text(response: httpx.Response) -> str:
    """Ollama's own ``{"error": "..."}`` when it sent one, and the raw body otherwise."""
    try:
        parsed: object = response.json()
    except ValueError:
        return response.text[:400]
    if isinstance(parsed, Mapping):
        error = cast("Mapping[str, object]", parsed).get("error")
        if isinstance(error, str):
            return error
    return response.text[:400]


def _mapping(value: object, field: str, model: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = (
            f"/api/show for {model!r} returned no {field}. manicule reads the architecture's "
            f"context length, width and pooling from it rather than assuming any of them."
        )
        raise ConfigError(msg)
    return cast("Mapping[str, object]", value)


def _positive_int(info: Mapping[str, object], key: str, model: str, client: OllamaClient) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = (
            f"{model!r} on {client.base_url} declares no usable {key} in its GGUF metadata "
            f"(got {value!r}). manicule never assumes one: a wrong context length truncates "
            f"silently and a wrong width builds an index that accepts writes."
        )
        raise ConfigError(msg)
    return value


def _pooling_type(
    info: Mapping[str, object], architecture: str, model: str, client: OllamaClient
) -> int:
    key = f"{architecture}.pooling_type"
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = (
            f"{model!r} on {client.base_url} declares no {key}. Pooling decides whether two "
            f"sets of vectors are comparable and cannot be guessed from a model name — CLS "
            f"and mean of the same token states differ by 0.66-0.80 cosine on a retrieval "
            f"model of this class, with nothing raised. Set `pooling` under this embedder's "
            f"configuration to the reduction the model was trained with."
        )
        raise ConfigError(msg)
    return value


__all__ = [
    "BACKEND",
    "POOLING_TYPES",
    "EmbedResult",
    "OllamaClient",
    "OllamaContextOverflowError",
    "OllamaUnavailableError",
    "ServedModelInfo",
]
