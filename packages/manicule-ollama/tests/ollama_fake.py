"""A synthetic Ollama, so the rules this backend enforces are exercised rather than mocked.

Every claim in :mod:`manicule_ollama` is about what a server does, and most of them are about
what it does *wrong*: a context it serves below the one it declares, a ``truncate`` flag it
ignores, vectors it forgets to normalize, a model somebody re-pulled. None of that can be
arranged on a real server on demand, and a suite that only ran against a healthy one would
certify the happy path and nothing else.

So this is a real ``httpx`` transport rather than a patched client. The request building, the
error mapping, the JSON decoding and the retry-free failure paths under test are the shipped
ones; only the socket is replaced. That is the same reason ``docs/connectors/confluence.md``
gives for driving a synthetic Confluence over ``httpx.MockTransport``.

**Its tokenizer is manicule's synthetic one**, from :func:`manicule.testing.write_tokenizer`: a
word-level vocabulary that wraps input in ``<s> … </s>``. So "how many tokens is this text" has
one answer here — whitespace-separated words, plus two — and the fake can be made to disagree
with it *deliberately*, by exactly one token, which is the only way to show the tokenizer check
failing for the reason it exists.

Uniquely named rather than called ``fake`` or ``support``: every ``packages/*/tests`` directory
lands on ``sys.path`` under pytest's default import mode, so a common basename in two of them
would resolve to whichever was collected first.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

import httpx

DIGEST: Final = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
"""A plausible manifest digest. Sixty-four hex characters, like the real thing."""

MODEL: Final = "synthetic-embed:0.1b"
ARCHITECTURE: Final = "synthbert"
SPECIAL_TOKENS: Final = 2
"""What :func:`manicule.testing.write_tokenizer` wraps every input in: ``<s>`` and ``</s>``."""


type Requests = list[tuple[str, Mapping[str, object]]]
"""Path and decoded body of every request a :class:`FakeOllama` has answered."""


def content_tokens(text: str) -> int:
    """The word-level count manicule's synthetic tokenizer produces for ``text``."""
    return len(text.split())


def _no_requests() -> Requests:
    """An empty request log. A named factory rather than ``list``, which loses the element type."""
    return []


@dataclass
class FakeOllama:
    """One server, configured to behave — or misbehave — in one specific way.

    Every field that is not the obvious value corresponds to a real failure this backend has to
    catch. They are settable individually so that a test turns exactly one thing wrong and the
    assertion names it.
    """

    model: str = MODEL
    digest: str = DIGEST
    architecture: str = ARCHITECTURE
    context_length: int = 64
    embedding_length: int = 8
    pooling_type: int = 1
    capabilities: tuple[str, ...] = ("embedding",)

    dimension: int | None = None
    """What the vectors are actually as wide as. ``None`` means :attr:`embedding_length` —
    a disagreement is the case where a server's output contradicts its own metadata."""

    served_ceiling: int | None = None
    """Total tokens this server will actually read. ``None`` means whatever ``num_ctx`` asks
    for. Set it lower to be the server that declares 32768 and serves 4096."""

    honors_truncate: bool = True
    """Whether ``truncate: false`` is obeyed. ``False`` is the older server that silently
    shortens the input and answers with a vector describing its opening."""

    normalize: bool = True
    count_offset: int = 0
    """Added to every ``prompt_eval_count``. One is enough to be a different vocabulary."""

    report_counts: bool = True
    """Whether ``prompt_eval_count`` is reported at all."""

    count: Callable[[str], int] = content_tokens
    """How this server counts the content tokens of one input.

    Defaults to whitespace-separated words, which is enough for the cases that never compare a
    count against anything. A suite exercising the tokenizer check passes the *real* counter —
    manicule's synthetic ``tokenizer.json`` splits punctuation, so `[pool(model(text))]` is
    eight tokens to it and one word to ``str.split``, and a fake that guessed would fail the
    check for a reason that has nothing to do with the thing being tested."""

    short_answer: bool = False
    """Whether to return one vector fewer than a *batch* was asked for.

    Only a batch: a single-input request still gets its vector, so that the dimension probe and
    the setup checks — which send one text at a time — reach the case this is about.

    Vectors are matched to texts positionally, so a short answer cannot be realigned — filling
    the gap or zipping against the first N texts attaches every later vector to the wrong text.
    A switch here rather than a monkeypatch in a test, so the endpoint under test stays the one
    that ships."""

    non_finite: bool = False
    """Whether the first component of every vector is ``NaN``.

    For the guard that has to run *before* the norm comparison: a NaN compares false against
    every tolerance rather than failing one, so a tolerance check alone waves it through."""

    null_component: bool = False
    """Whether a vector arrives with a ``null`` in it, rather than a number."""

    tags_error: str = ""
    """If set, ``/api/tags`` answers 400 with this message instead of a listing.

    For the health check, which is the one caller that must report rather than raise. The
    message worth setting is Ollama's context refusal, because this client maps that phrase to
    a distinct exception — so a health check catching only the two errors it usually sees would
    let it escape and take the diagnostic down with it."""

    requests: Requests = field(default_factory=_no_requests)
    """Every request, in order, for a test that wants to assert what was *sent*."""

    # --- transports -----------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        """A transport serving this configuration, usable synchronously and asynchronously."""
        return httpx.MockTransport(self._handle)

    # --- the endpoints --------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body: Mapping[str, object] = {}
        if request.content:
            parsed: object = json.loads(request.content)
            if isinstance(parsed, Mapping):
                body = cast("Mapping[str, object]", parsed)
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/tags":
            if self.tags_error:
                return httpx.Response(400, json={"error": self.tags_error})
            return httpx.Response(200, json=self._tags())
        if request.url.path == "/api/show":
            return httpx.Response(200, json=self._show())
        if request.url.path == "/api/embed":
            return self._embed(body)
        return httpx.Response(404, json={"error": f"unknown route {request.url.path}"})

    def _tags(self) -> dict[str, object]:
        return {
            "models": [
                {
                    "name": self.model,
                    "model": self.model,
                    "digest": self.digest,
                    "details": {"family": self.architecture, "format": "gguf"},
                    "capabilities": list(self.capabilities),
                }
            ]
        }

    def _show(self) -> dict[str, object]:
        return {
            "capabilities": list(self.capabilities),
            "details": {"family": self.architecture, "format": "gguf"},
            "model_info": {
                "general.architecture": self.architecture,
                f"{self.architecture}.context_length": self.context_length,
                f"{self.architecture}.embedding_length": self.embedding_length,
                f"{self.architecture}.pooling_type": self.pooling_type,
            },
        }

    def _embed(self, body: Mapping[str, object]) -> httpx.Response:
        inputs = [str(item) for item in cast("Sequence[object]", body.get("input") or ())]
        options = body.get("options")
        num_ctx = self.context_length
        if isinstance(options, Mapping):
            asked = cast("Mapping[str, object]", options).get("num_ctx")
            if isinstance(asked, int):
                num_ctx = min(asked, self.context_length)
        ceiling = self.served_ceiling if self.served_ceiling is not None else num_ctx
        truncate = bool(body.get("truncate", True))

        totals = [self.count(text) + SPECIAL_TOKENS for text in inputs]
        if any(total > ceiling for total in totals) and (self.honors_truncate and not truncate):
            return httpx.Response(
                400, json={"error": "the input length exceeds the context length"}
            )

        vectors: list[list[float | None]] = [
            cast("list[float | None]", self._vector(text)) for text in inputs
        ]
        if self.non_finite:
            vectors = [[float("nan"), *row[1:]] for row in vectors]
        if self.null_component:
            vectors = [[None, *row[1:]] for row in vectors]
        payload: dict[str, object] = {
            "model": self.model,
            "embeddings": vectors[:-1] if self.short_answer and len(vectors) > 1 else vectors,
        }
        if self.report_counts:
            payload["prompt_eval_count"] = sum(
                min(total, ceiling) for total in totals
            ) + self.count_offset * len(inputs)
        # Serialized here rather than through `json=`, which refuses `NaN` outright. Python's
        # own encoder emits it as a bare token and its decoder reads it back, which is exactly
        # what a server built on a C JSON library does — and a backend that never sees one
        # cannot be shown to refuse it.
        return httpx.Response(
            200,
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )

    def _vector(self, text: str) -> list[float]:
        """A deterministic vector for ``text``, so determinism and the cache are testable.

        Derived from a digest rather than a random seed: the same text must give the same
        vector across processes, which is the property the embedding cache is keyed on.
        """
        width = self.dimension if self.dimension is not None else self.embedding_length
        digest = hashlib.sha256(text.encode()).digest()
        raw = [
            (digest[index % len(digest)] + 1) / 256.0 * (1 if index % 3 else -1)
            for index in range(width)
        ]
        if not self.normalize:
            return [value * 3.0 for value in raw]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


def written(count: int) -> str:
    """A text the synthetic tokenizer measures at exactly ``count`` content tokens."""
    return " ".join(["alpha"] * count)


def config_payload(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for :class:`manicule_ollama.config.OllamaEmbedderConfig`."""
    payload: dict[str, Any] = {"base_url": "http://ollama.test:11434"}
    payload.update(overrides)
    return payload


__all__ = [
    "ARCHITECTURE",
    "DIGEST",
    "MODEL",
    "SPECIAL_TOKENS",
    "FakeOllama",
    "Requests",
    "config_payload",
    "content_tokens",
    "written",
]
