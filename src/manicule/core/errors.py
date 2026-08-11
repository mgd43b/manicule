"""The error hierarchy.

Every failure manicule raises on purpose descends from :class:`ManiculeError`, so a caller
can distinguish "this system said no" from "something unexpected broke".
"""

from __future__ import annotations


class ManiculeError(Exception):
    """Base class for every error manicule raises deliberately."""


# --- configuration ---------------------------------------------------------------------


class ConfigError(ManiculeError):
    """Configuration could not be loaded, parsed, or validated."""


class PolicyError(ConfigError):
    """A configuration is internally valid but violates a declared policy.

    Raised before any resource is constructed, so that an impossible combination fails at
    startup rather than at the first request that happens to exercise it.
    """


# --- plugins ---------------------------------------------------------------------------


class PluginError(ManiculeError):
    """Base class for plugin discovery, validation and registration failures."""


class PluginLoadError(PluginError):
    """An entry point could not be imported, or did not resolve to a plugin."""


class IncompatiblePluginError(PluginError):
    """A plugin declares a core version range that the running core does not satisfy."""


class PluginDependencyError(PluginError):
    """A plugin's declared requirements are unmet, or a declared conflict is present."""


class DuplicateComponentError(PluginError):
    """Two plugins registered the same component kind under the same name."""


# --- container -------------------------------------------------------------------------


class ContainerError(ManiculeError):
    """Base class for wiring failures."""


class UnknownComponentError(ContainerError):
    """A component was requested that no installed plugin provides."""


class CircularDependencyError(ContainerError):
    """Resolving a component required resolving itself."""


class ContainerStateError(ContainerError):
    """The container was used outside the lifecycle window it supports."""


# --- pipeline --------------------------------------------------------------------------


class ParseError(ManiculeError):
    """A parser could not handle a document it nominally supports.

    Raising this is how a parser declines a document so that the next parser in the
    fallback chain gets a turn. Any other exception aborts the document.
    """


class FingerprintMismatchError(ManiculeError):
    """An embedding fingerprint does not match the one an index was built with.

    Vectors produced by different models are not comparable, so continuing would silently
    destroy retrieval quality. This is always fatal to the operation that raised it.
    """


class ChunkingError(ManiculeError):
    """A chunker could not produce chunks from a document's blocks."""


class NameInUseError(ManiculeError):
    """A workspace already has a collection or a tag under the name that was asked for.

    Raised rather than absorbed into an upsert, because the two operations differ in what they
    do to somebody else's data. Creating a collection that already exists and quietly handing
    back the existing one merges two people's sets under one name; renaming a tag onto an
    existing name moves every document from one label to the other. Both look like success.
    """


class UnknownEntityError(ManiculeError):
    """An id names nothing this workspace-scoped handle can see.

    One error for "there is no such row" and "there is such a row and it belongs to another
    tenant", and the two are deliberately not distinguished in the message. Telling a caller
    which of the two it hit confirms the existence of another workspace's document from
    outside that workspace, which is a membership oracle — small, but exactly the kind of leak
    a tenancy boundary is supposed to close.
    """


class MiddlewareViolationError(ManiculeError):
    """A middleware hook did something the contract forbids.

    Three shapes, all of them the same defect — a hook whose effect the pipeline cannot
    describe: returning the wrong type or ``None`` where a value was required, rewriting
    ``ParsedBlock.text`` or ``Chunk.text``, or rewriting ``embed_text`` after declaring it
    would not.

    Fatal to the document, never to the batch. A hook that fails on one document is usually
    a document problem, and disabling the hook would make the corpus depend on ingest order.
    """


class WorkerKilledError(ManiculeError):
    """A parse worker was killed for exceeding a limit, rather than failing on its own.

    Deliberately **not** a decline. A parser that declined inspected the input and reported
    that it is not its kind, which is information; a parser the pipeline killed reported
    nothing at all. Collapsing the two lets a chain of timeouts end at
    ``unsupported_media_type``, which reads as "manicule does not handle this format" when
    the truth is "every parser that handles this format ran out of time" — and sends whoever
    reads it to write a parser that already exists.
    """


class ReconciliationRefusedError(ManiculeError):
    """A reconciliation pass proposed more deletion than its ceiling allows.

    Not a failure of the source and not a failure of the diff: a genuine bulk deletion is
    rare and worth a human, and a bug that looks like one is not rare at all. The proposal is
    recorded so it can be confirmed rather than recomputed.
    """


class InstanceLockedError(ManiculeError):
    """Another manicule process already holds this data directory.

    The recovery sweep, the tombstone sweep and the blob GC all assume a single writer. WAL
    permits several, so the assumption is enforced here rather than hoped for: a second
    instance that started anyway would requeue the first one's in-flight documents out from
    under it.
    """


class ContextOverflowError(ManiculeError):
    """Text was offered to an embedder that attends to less of it than was sent.

    Beyond a model's sequence length the input is dropped with no error raised, so the
    resulting vector describes the opening of a chunk while the chunk still claims all of
    its text — a citation quoting words the index never saw.

    Always fatal to the operation that raised it. There is no partial-credit answer here:
    an embedding of the first half of a passage is not a worse embedding of the passage, it
    is an embedding of something else.
    """


# --- generation ------------------------------------------------------------------------


class GenerationError(ManiculeError):
    """Base class for failures on the answer path.

    Provider libraries raise their own hierarchies. Those stop at the adapter and become one
    of these, for the reason ``docs/contracts.md`` §5 gives: a provider-shaped type in core
    forces every consumer to learn every provider, and the set of providers only grows.

    Every subclass keeps the provider's own message text. Discarding it is how
    ``"OpenAI error: 429"`` becomes an operator's whole afternoon.
    """


class ProviderAuthError(GenerationError):
    """The provider rejected the credential.

    Names the environment variable that was read, because "unauthorized" without it sends
    somebody to the wrong dashboard. Never retried: a bad key does not improve by being
    presented three times.
    """


class ProviderRateLimitError(GenerationError):
    """The provider is throttling. Retryable before the first token, and only then."""


class ProviderConnectionError(GenerationError):
    """The provider could not be reached at all."""


class ProviderTimeoutError(GenerationError):
    """A generation exceeded one of its three deadlines.

    Three, because one covers the wrong interval: time to first token, the gap between two
    tokens, and total wall clock. A budget that only bounds the connect leaves a provider
    that opens a stream and then stops sending indistinguishable from a slow answer.
    """


class ContextWindowError(GenerationError):
    """The server says the prompt did not fit the window.

    **A defect in the startup cross-check, not a runtime condition to absorb.** Reaching it
    means manicule's estimate and the server disagreed by more than the safety factor, so it
    surfaces with both counts and the model named rather than being retried or trimmed.
    """


class ContentFilteredError(GenerationError):
    """The provider refused to produce or continue the answer."""


class ProviderRequestError(GenerationError):
    """A provider rejection with no more specific mapping.

    The catch-all arm, and it is deliberately last: ``ContextWindowExceededError`` and
    ``ContentPolicyViolationError`` both subclass litellm's ``BadRequestError``, so a generic
    bad-request arm placed above them swallows the two cases with actionable remedies. The
    mapping is table-driven rather than a chain of ``except`` blocks precisely so that a
    correct-looking refactor cannot silently delete a diagnosis.
    """


class RedactionError(GenerationError):
    """Redaction could not be completed, so nothing was sent.

    The fail-safe direction is refuse-to-send. There is no path where a timeout, an exception
    or a mistake results in unredacted text reaching a remote model — which is what makes the
    setting a boundary rather than a best effort.
    """


class TokenStateError(ManiculeError):
    """A backend returned something other than per-token hidden states.

    Raised where a pooling path expected an array of shape ``(batch, sequence, dimension)``
    and got a different rank. That is not a hypothetical: ``mlx-embeddings`` returns genuine
    token states under ``last_hidden_state`` for some architectures and the *already pooled*
    vector under the same name for others, and an ONNX export names its 3-D output whatever
    the exporter felt like. Pooling a 2-D array does not fail — it reduces over the batch
    axis instead of the sequence axis and produces one plausible, normalised, entirely wrong
    vector per input.

    Fatal, never a warning. Nothing downstream can tell a wrongly pooled vector from a right
    one.
    """


__all__ = [
    "ChunkingError",
    "CircularDependencyError",
    "ConfigError",
    "ContainerError",
    "ContainerStateError",
    "ContentFilteredError",
    "ContextOverflowError",
    "ContextWindowError",
    "DuplicateComponentError",
    "FingerprintMismatchError",
    "GenerationError",
    "IncompatiblePluginError",
    "InstanceLockedError",
    "ManiculeError",
    "MiddlewareViolationError",
    "NameInUseError",
    "ParseError",
    "PluginDependencyError",
    "PluginError",
    "PluginLoadError",
    "PolicyError",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderTimeoutError",
    "ReconciliationRefusedError",
    "RedactionError",
    "TokenStateError",
    "UnknownComponentError",
    "UnknownEntityError",
    "WorkerKilledError",
]
