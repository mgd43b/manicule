"""The ingest pipeline: discover → fetch → parse → chunk → embed → store.

The piece that makes everything else connect end to end, and the piece whose entire purpose is
surviving failure. Four things here are load-bearing and none of them is a ``try``/``except``:

* **Parse runs in a worker subprocess**, because a deadline is only enforceable across a
  process boundary — a parser inside a native extension observes no cancellation, and Python
  cannot kill a thread (:mod:`manicule.ingest.workers`).
* **Embed deliberately does not**, because an in-process embedder is the point. Its failure
  modes are removed at startup rather than caught at runtime
  (:mod:`manicule.ingest.refusals`).
* **The recovery sweep uses an allowlist of non-terminal statuses**, so a status added later is
  not swept rather than swept wrongly (:mod:`manicule.ingest.recovery`).
* **A killed parser is a hard failure, not a decline**, so a chain of timeouts never reports
  itself as an unsupported format.

:mod:`manicule.ingest.watch` is not re-exported here. It is imported directly by whoever
watches a directory, so that the optional dependency it needs is paid for only by them.
"""

from __future__ import annotations

from manicule.ingest.capacity import (
    CapacityDiagnostic,
    CapacityRefusedError,
    CapacityResource,
)
from manicule.ingest.embedding import batch_size, embed_chunks
from manicule.ingest.middleware import MiddlewareRunner, declarations, text_digest
from manicule.ingest.pipeline import (
    BlobSink,
    Change,
    DocumentOutcome,
    IngestPipeline,
    NoRetention,
    RunReport,
)
from manicule.ingest.ports import IngestStore
from manicule.ingest.recovery import InstanceLock, requeue_interrupted
from manicule.ingest.refusals import check_before_run, require_coherent, require_measured
from manicule.ingest.sweeps import SweepGate, SweepResult, sweep_vectors
from manicule.ingest.workers import (
    AttemptResult,
    InProcessRunner,
    ParseRunner,
    WorkerConfig,
    WorkerPool,
    worker_config,
)

__all__ = [
    "AttemptResult",
    "BlobSink",
    "CapacityDiagnostic",
    "CapacityRefusedError",
    "CapacityResource",
    "Change",
    "DocumentOutcome",
    "InProcessRunner",
    "IngestPipeline",
    "IngestStore",
    "InstanceLock",
    "MiddlewareRunner",
    "NoRetention",
    "ParseRunner",
    "RunReport",
    "SweepGate",
    "SweepResult",
    "WorkerConfig",
    "WorkerPool",
    "batch_size",
    "check_before_run",
    "declarations",
    "embed_chunks",
    "requeue_interrupted",
    "require_coherent",
    "require_measured",
    "sweep_vectors",
    "text_digest",
    "worker_config",
]
