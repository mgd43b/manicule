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
    "ContextOverflowError",
    "DuplicateComponentError",
    "FingerprintMismatchError",
    "IncompatiblePluginError",
    "InstanceLockedError",
    "ManiculeError",
    "MiddlewareViolationError",
    "ParseError",
    "PluginDependencyError",
    "PluginError",
    "PluginLoadError",
    "PolicyError",
    "ReconciliationRefusedError",
    "TokenStateError",
    "UnknownComponentError",
    "WorkerKilledError",
]
