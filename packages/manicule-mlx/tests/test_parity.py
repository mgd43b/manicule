"""Real weights, both runtimes, and the measurement that keeps ``backend`` out of identity.

**This file is here rather than in manicule, and that placement is the point.**
``EmbedFingerprint`` excludes ``backend`` from identity so that a corpus moves between machines
— and between backends — without a re-embed. That exclusion is a claim about *numbers*, and the
numbers cannot be produced by manicule alone once the second runtime ships separately. So the
plugin asserts parity against the reference: this package depends on manicule, imports both
backends, and runs in the same CI job. manicule keeps the claim; this suite is what licenses it.

If it ever fails, the correction is to move ``backend`` into ``IDENTITY_FIELDS`` — not to widen
the tolerance. A corpus is not portable if this is false.

Also here: the two MLX-specific facts that can only be shown against real weights. The
convenience field ``text_embeds`` is mean-pooled unconditionally although these models declare
CLS, and the allocator gauges are readable from a thread that is not the worker.

Weights are found in the local model cache and never downloaded from here; a missing model
skips, or fails under ``REQUIRE_EMBEDDING_MODELS``, which CI sets.
"""

# The "reach past the abstraction" test below reads a backend's raw output on purpose, to show
# what manicule declines to use. That means touching private attributes and a runtime that ships
# no type information, which under strict mode is an Unknown per expression. Relaxed for this
# file only, and only the rules that report exactly that.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownParameterType=false

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from manicule.core.embedding import Pooling
from manicule.core.lifecycle import Metric
from manicule.core.protocols import Embedder, TokenStateEmbedder
from manicule.embedding.cards import read_card
from manicule.embedding.pooling import l2_normalize, pool
from manicule.testing import (
    FULL_MODEL,
    PARITY_MODEL,
    REQUIRE_MODELS_ENV,
    assert_embedder_contract,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
    is_required,
    require_model,
)

if TYPE_CHECKING:
    from manicule.embedding.base import PooledEmbedder

pytestmark = pytest.mark.anyio

TEXTS: Final[tuple[str, ...]] = (
    "The retention window is ninety days, after which archived pages are purged.",
    "El gato se sienta en la alfombra y mira por la ventana durante horas.",
    "def embed(texts): return [pool(model(text)) for text in texts]",
    " ".join(["paragraph"] * 200),
)
"""Short, non-English, code, and long. Pooling disagreement grows with length, so the long one
is not padding: it is where a wrong reduction is furthest from a right one."""

COSINE_TOLERANCE: Final = 0.9999
"""How close two runtimes must be, per vector.

Measured on ``bge-m3``: cosine 1.000000 to six decimal places between MLX fp16 weights and the
fp32 ONNX export, with a largest component difference of 1.8e-05. The gate sits roughly a
hundred times looser than the measurement, which leaves room for a different ONNX release
without leaving room for a different model — 8-bit quantized ``bge-m3`` scores 0.9998 against
fp16 and fails this.
"""

COMPONENT_TOLERANCE: Final = 1e-3
"""Largest permitted per-component difference. Cosine alone would pass a scaled vector."""

DEFAULT_CACHE_ENTRIES: Final = 10_000

_LOADED: dict[tuple[str, str], PooledEmbedder] = {}


def requires_mlx(model_id: str) -> None:
    """Skip unless MLX works here — or fail, if a model requiring it was named.

    On a runner that is supposed to be measuring backend parity, an absent runtime and absent
    weights are the same outcome: the comparison did not happen.
    """
    from manicule_mlx.runtime import mlx_usable  # noqa: PLC0415 - Apple-only

    if mlx_usable():
        return
    if is_required(model_id):
        pytest.fail(
            f"MLX is not usable on this machine, and {REQUIRE_MODELS_ENV} names {model_id}. "
            f"Backend parity cannot be measured with one backend"
        )
    pytest.skip("MLX is not usable on this machine")


def requires_metal_allocator() -> None:
    """Skip unless the **Metal allocator** is the one MLX is allocating through.

    Narrower than "MLX is executing on the GPU", and deliberately so. :func:`requires_mlx` asks
    whether ``mx.eval`` works, which succeeds on MLX's CPU device too — so a green job proves
    MLX *evaluates*, not that Metal was involved. Neither do the memory counters: with
    ``mx.set_default_device(mx.cpu)`` a 16 MiB allocation still reports
    ``mx.get_active_memory() == 16777216``. Anyone reaching for a memory counter as proof that
    Metal ran gets a confident wrong answer.

    The default device is deliberately *not* checked. MLX allocates through one process-wide
    allocator chosen by whether Metal is present, not by which device an operation runs on —
    measured, the two differ by 0.006% on this repository's workload shape. What decides the
    allocator is what this function checks.
    """
    from manicule_mlx.runtime import mlx_core  # noqa: PLC0415 - Apple-only

    mx = mlx_core()
    if mx.metal.is_available():
        return
    pytest.skip(
        f"MLX has no Metal here and is on {mx.default_device()}, so the Metal allocator these "
        f"tests assert about is not the one in use"
    )


def require(model_id: str, backend: str) -> None:
    """Skip — or fail under CI — unless this model can run on this backend here."""
    if backend == "mlx":
        requires_mlx(model_id)
    require_model(model_id, mlx=backend == "mlx", onnx=backend == "onnx")


@pytest.fixture(scope="module", autouse=True)
def release_backends() -> Iterator[None]:
    """Load each model once per module and release the worker threads at the end.

    Loading ``bge-m3`` is 2.3 GB per runtime; doing it per test would make this file the slowest
    thing in the suite by an order of magnitude and would measure nothing extra.
    """
    yield None
    for embedder in _LOADED.values():
        embedder._worker.shutdown(wait=True)
    _LOADED.clear()


async def embedder_for(
    model_id: str, backend: str, cache_entries: int | None = None
) -> PooledEmbedder:
    """A loaded embedder, built once and shared across the tests in this module."""
    key = (model_id, backend)
    cached = _LOADED.get(key)
    if cached is not None and cache_entries is None:
        return cached

    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415

    card = read_card(model_id, revision=builtin_model_revision(model_id))
    entries = DEFAULT_CACHE_ENTRIES if cache_entries is None else cache_entries
    if backend == "mlx":
        from manicule_mlx.backend import MlxEmbedder  # noqa: PLC0415

        built: PooledEmbedder = MlxEmbedder(card, cache_entries=entries)
    else:
        from manicule.embedding.runtimes.onnx_backend import OnnxEmbedder  # noqa: PLC0415

        built = OnnxEmbedder(card, cache_entries=entries)
    await built.setup()
    if cache_entries is None:
        _LOADED[key] = built
    return built


# --- what only this backend can be asked ---------------------------------------------------


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_mlx_token_states_are_three_dimensional(model_id: str) -> None:
    """The check ticket #3 asks for, against the weights it was written about.

    A test asserting only that a vector came back passes on a backend returning its pooled
    output under the token-state name, and so certifies the exact bug it exists to catch.
    """
    require(model_id, "mlx")
    embedder = await embedder_for(model_id, "mlx")

    encoded = await embedder.encode(list(TEXTS))

    assert len(encoded.states.shape) == 3
    batch, sequence, dimension = encoded.states.shape
    assert (batch, dimension) == (len(TEXTS), embedder.fingerprint.dimension)
    assert sequence > 1


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_the_mlx_backend_meets_the_shipped_conformance_suites(model_id: str) -> None:
    """manicule publishes these; a backend in another distribution passes the same ones."""
    require(model_id, "mlx")
    embedder = await embedder_for(model_id, "mlx")

    assert isinstance(embedder, Embedder)
    assert isinstance(embedder, TokenStateEmbedder)
    assert_protocol_signatures(embedder, TokenStateEmbedder)
    await assert_embedder_contract(embedder, list(TEXTS))
    await assert_refuses_oversized_chunks(embedder.embed_chunks, embedder)


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_the_mlx_convenience_field_is_the_wrong_reduction_for_this_model(
    model_id: str,
) -> None:
    """``text_embeds`` is mean-pooled unconditionally, and these models declare CLS.

    Asserting that our vectors *match* the convenience field would be asserting the bug. So the
    assertion is that they differ — and, to show the difference is the reduction rather than
    noise, that the convenience field reproduces our *mean* pool exactly.
    """
    require(model_id, "mlx")
    embedder = await embedder_for(model_id, "mlx")
    assert embedder.fingerprint.pooling is Pooling.CLS

    from manicule_mlx.runtime import mlx_core  # noqa: PLC0415 - guarded by require() above

    mx = mlx_core()
    encoded = await embedder.encode(list(TEXTS))
    ours = np.asarray(await embedder.embed(list(TEXTS)), dtype=np.float32)

    ids, mask = embedder._tokenize(TEXTS)
    model = embedder._model
    output = model(mx.array(ids), attention_mask=mx.array(mask))
    convenience = np.asarray(output.text_embeds.astype(mx.float32), dtype=np.float32)

    states = np.asarray(encoded.states, dtype=np.float32)
    mean_pooled = l2_normalize(pool(states, np.asarray(mask), Pooling.MEAN))

    against_convenience = [float(a @ b) for a, b in zip(ours, convenience, strict=True)]
    assert max(against_convenience) < 0.95, (
        f"the convenience field agreed with the model's declared pooling ({against_convenience}). "
        f"If mlx-embeddings has started honoring 1_Pooling/config.json this test should be "
        f"rewritten, not deleted — but check before believing it"
    )
    assert np.allclose(mean_pooled, convenience, atol=1e-3), (
        "the convenience field should be exactly our mean pool, which is what makes the "
        "disagreement above a difference of reduction rather than of arithmetic"
    )


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_the_mlx_backend_is_deterministic_and_batch_invariant(model_id: str) -> None:
    """Padding is per batch, so an unmasked reduction fails this and nothing else does."""
    require(model_id, "mlx")
    embedder = await embedder_for(model_id, "mlx", cache_entries=0)

    alone = np.asarray(await embedder.embed([TEXTS[0]]))
    crowded = np.asarray(await embedder.embed(list(TEXTS)))
    again = np.asarray(await embedder.embed([TEXTS[0]]))

    assert np.allclose(alone[0], again[0], atol=1e-6)
    assert np.allclose(alone[0], crowded[0], atol=1e-4)


# --- the parity claim, which is manicule's and is checked here ------------------------------


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_the_two_runtimes_produce_the_same_vectors(model_id: str) -> None:
    """The measurement that lets ``backend`` stay out of fingerprint identity.

    A failure here is not a flake and is not fixed by widening the tolerance: it means one
    runtime is not computing what the other is, and the correct response is to move ``backend``
    into ``EmbedFingerprint.IDENTITY_FIELDS``, which makes a runtime change a loud error with a
    re-embed path.
    """
    require(model_id, "mlx")
    require(model_id, "onnx")

    mlx_vectors = np.asarray(await (await embedder_for(model_id, "mlx")).embed(list(TEXTS)))
    onnx_vectors = np.asarray(await (await embedder_for(model_id, "onnx")).embed(list(TEXTS)))

    cosines = [float(a @ b) for a, b in zip(mlx_vectors, onnx_vectors, strict=True)]
    worst = min(cosines)
    largest = float(np.max(np.abs(mlx_vectors - onnx_vectors)))

    assert worst > COSINE_TOLERANCE, (
        f"MLX and onnxruntime disagree on {model_id}: worst cosine {worst}, per vector "
        f"{cosines}. Move `backend` into EmbedFingerprint.IDENTITY_FIELDS rather than "
        f"loosening this number — a corpus is not portable if this is false"
    )
    assert largest < COMPONENT_TOLERANCE, (
        f"largest component difference {largest} on {model_id}; cosine alone would pass a "
        f"uniformly scaled vector"
    )


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_the_two_runtimes_record_interchangeable_identities(model_id: str) -> None:
    """Byte-equal canonical fingerprints, which is what an index actually compares.

    The runtime and the artifact are both recorded and neither is in identity — so the MLX
    conversion and the ONNX export of one model write against the same index, and a machine
    that installs or removes this package does not re-embed. That last clause is why this test
    matters more after the split than before it.
    """
    require(model_id, "mlx")
    require(model_id, "onnx")

    mlx_print = (await embedder_for(model_id, "mlx")).fingerprint
    onnx_print = (await embedder_for(model_id, "onnx")).fingerprint

    assert mlx_print.canonical() == onnx_print.canonical()
    assert mlx_print.backend != onnx_print.backend
    assert mlx_print.matches(onnx_print)


# --- the allocator ---------------------------------------------------------------------------


async def test_mlx_memory_metrics_are_readable_off_the_worker_thread() -> None:
    """Reading the allocator gauges from another thread must not abort the process.

    This backend already aborts — ``libc++abi: terminating … There is no Stream(gpu, N) in
    current thread``, uncatchable — when a graph built on one thread is evaluated on another,
    which is why the embedder owns exactly one worker. ``metrics()`` is the one MLX call that
    deliberately does *not* run there: it is synchronous, and an operator scrapes it from the
    event loop or from a metrics thread while a forward pass is in flight on the worker.

    That is safe because MLX's **allocator** is process-global while its **streams** are
    thread-local, and these gauges query the allocator. Safe by a different mechanism than the
    rest of the class, so it is asserted rather than assumed — the failure mode is a dead
    server, not a red test.
    """
    require(PARITY_MODEL, "mlx")
    requires_metal_allocator()
    embedder = await embedder_for(PARITY_MODEL, "mlx")
    await embedder.embed(["a short text, so that something has been allocated"])

    from_a_foreign_thread: list[Metric] = []
    thread = threading.Thread(
        target=lambda: from_a_foreign_thread.extend(embedder.metrics()),
    )
    thread.start()
    thread.join()

    published = {metric.name for metric in from_a_foreign_thread}
    assert {"mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes"} <= published
    assert next(m for m in from_a_foreign_thread if m.name == "mlx_active_bytes").value > 0


def test_repeated_embedding_holds_a_bounded_physical_footprint() -> None:
    """The MLX allocator bound, measured on real weights rather than asserted on a fake.

    Runs ``tools/qualify_memory.py``, which embeds one text at a time and measures the child
    process from *outside* with ``footprint``. That indirection is the whole point: MLX
    allocates through Metal, ``ps`` does not report Metal buffers, and before the bound landed
    this workload took physical footprint from 2.45 to 25.0 GiB while RSS fell. A test that
    measured resident memory would have reported success throughout.

    **Gated on the model being named rather than merely cached.** It is minutes of real forward
    passes, so it runs when somebody has asked for ``BAAI/bge-m3`` explicitly — which is also
    what keeps 4.6 GB of weights out of every CI run.
    """
    if not is_required(FULL_MODEL):
        pytest.skip(f"set {REQUIRE_MODELS_ENV} to include {FULL_MODEL} to run the qualification")
    require_model(FULL_MODEL, mlx=True)
    requires_mlx(FULL_MODEL)
    requires_metal_allocator()

    completed = subprocess.run(  # noqa: S603 - our own script, fixed arguments
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "tools" / "qualify_memory.py"),
            "--passes",
            "40",
            "--settle-passes",
            "5",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passes_measured"] >= 40
    assert report["max_content_tokens"] <= report["content_token_ceiling"]
