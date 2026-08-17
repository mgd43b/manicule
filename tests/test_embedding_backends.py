"""Real weights on the backend this distribution ships, and what a declaration cannot decide.

The synthetic suites check everything that follows from a model's declaration. What does not:
**an export\'s convenience output is not reliably the model\'s own pooling**, and the shapes it
arrives in vary per export with nothing in the names to tell them apart. That can only be shown
against real weights, and it is shown here.

**The cross-backend parity measurement is no longer in this file, and not in this
distribution.** ``EmbedFingerprint`` still excludes ``backend`` from identity so that a corpus
moves between machines — and between runtimes — without a re-embed, and that exclusion is still
manicule\'s claim. But the second runtime ships separately now: ``mlx-embeddings`` is GPL-3.0,
so the Metal backend lives in ``manicule-mlx``. The measurement that licenses the exclusion is
``packages/manicule-mlx/tests/test_parity.py``, which depends on manicule, imports both
backends, and runs in the same CI job. Keeping the two packages in one repository is what keeps
that claim checked rather than merely asserted.

Weights are found in the local model cache and never downloaded from here; a missing model
skips, or fails under ``REQUIRE_EMBEDDING_MODELS``, which CI sets.
"""

# The two "reach past the abstraction" tests below read a backend's raw output on purpose, to
# show what manicule declines to use. That means touching one private attribute per backend and
# a runtime that ships no type information, which under strict mode is an Unknown per
# expression. Relaxed for this file only, and only the rules that report exactly that — the
# modules deciding what a vector is are checked strictly, and so is every other test here.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownParameterType=false

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from manicule.core.embedding import Pooling
from manicule.core.errors import ContextOverflowError
from manicule.core.protocols import Embedder, TokenStateEmbedder
from manicule.embedding.cards import read_card
from manicule.embedding.pooling import l2_normalize
from manicule.testing import (
    assert_embedder_contract,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
)
from tests.embedding_support import FULL_MODEL, PARITY_MODEL, require_model

if TYPE_CHECKING:
    from manicule.embedding.base import PooledEmbedder

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

BACKENDS: Final[tuple[str, ...]] = ("onnx",)
"""The backends *this distribution* ships, which is one.

Kept as a tuple rather than inlined because it is the parameter these suites vary, and because
a second in-tree backend would be added here rather than by rewriting every decorator. A
backend in another distribution tests itself: see the module docstring."""

DEFAULT_CACHE_ENTRIES: Final = 10_000
"""What the backends default to. Restated rather than imported so that a test varying the
cache is varying one number, not silently inheriting a different one."""

_LOADED: dict[tuple[str, str], PooledEmbedder] = {}


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
    """A loaded embedder, built once and shared across the tests in this module.

    ``cache_entries`` is the one setting a test varies, and a test that varies it gets an
    embedder of its own rather than the shared one — a cache disabled for one test and left on
    for the next would make the results depend on test order.
    """
    key = (model_id, backend)
    cached = _LOADED.get(key)
    if cached is not None and cache_entries is None:
        return cached

    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415
    from manicule.embedding.runtimes.onnx_backend import OnnxEmbedder  # noqa: PLC0415

    card = read_card(model_id, revision=builtin_model_revision(model_id))
    entries = DEFAULT_CACHE_ENTRIES if cache_entries is None else cache_entries
    built: PooledEmbedder = OnnxEmbedder(card, cache_entries=entries)
    await built.setup()
    if cache_entries is None:
        _LOADED[key] = built
    return built


def require(model_id: str, backend: str) -> None:
    """Skip — or fail under CI — unless this model can run on this backend here."""
    require_model(model_id, onnx=backend == "onnx")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_token_states_from_a_real_model_are_three_dimensional(
    model_id: str, backend: str
) -> None:
    """The check ticket #3 asks for, against the weights it was written about.

    A test asserting only that a vector came back passes on a backend returning its pooled
    output under the token-state name, and so certifies the exact bug it exists to catch.
    """
    require(model_id, backend)
    embedder = await embedder_for(model_id, backend)

    encoded = await embedder.encode(list(TEXTS))

    assert len(encoded.states.shape) == 3
    batch, sequence, dimension = encoded.states.shape
    assert (batch, dimension) == (len(TEXTS), embedder.fingerprint.dimension)
    assert sequence > 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_a_real_embedder_meets_the_shipped_conformance_suites(
    model_id: str, backend: str
) -> None:
    require(model_id, backend)
    embedder = await embedder_for(model_id, backend)

    assert isinstance(embedder, Embedder)
    assert isinstance(embedder, TokenStateEmbedder)
    assert_protocol_signatures(embedder, TokenStateEmbedder)
    await assert_embedder_contract(embedder, list(TEXTS))
    await assert_refuses_oversized_chunks(embedder.embed_chunks, embedder)


@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_an_onnx_export_offers_no_reliable_shortcut_either(model_id: str) -> None:
    """The counterpoint that makes the rule "never trust it" rather than "prefer ours".

    Three exports, three different answers, none of them announced. ``bge-m3``'s calls its
    outputs ``token_embeddings`` and ``sentence_embedding`` — neither of which is
    ``last_hidden_state`` — and its pooled output genuinely *is* the model's declared CLS
    pooling, cosine 1.000 to our own path. ``bge-small-en-v1.5``'s export publishes no pooled
    output at all. So the same shortcut is right on one ONNX export, absent on another, and
    wrong on MLX, with nothing in any of the names to tell them apart.

    What is stable is rank, and that is asserted here: exactly one output is
    ``(batch, sequence, dimension)``, which is why the backend selects by shape.
    """
    require(model_id, "onnx")
    embedder = await embedder_for(model_id, "onnx")

    ours = np.asarray(await embedder.embed(list(TEXTS)), dtype=np.float32)

    ids, mask = embedder._tokenize(TEXTS)
    session = embedder._session
    assert session is not None
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in {item.name for item in session.get_inputs()}:
        feed["token_type_ids"] = np.zeros_like(ids)
    outputs = [np.asarray(item) for item in session.run(None, feed)]
    names = [item.name for item in session.get_outputs()]

    assert sum(item.ndim == 3 for item in outputs) == 1, (
        f"{model_id}'s export has no single rank-3 output ({names}), so selecting the encoder "
        f"output by shape would be ambiguous and the backend must say so rather than guess"
    )
    assert "last_hidden_state" not in names or model_id != FULL_MODEL, (
        "recorded because it is the trap: this export calls its token states "
        "'token_embeddings', so a name-based lookup would miss them entirely"
    )

    pooled = [item for item in outputs if item.ndim == 2]
    if not pooled:
        return
    cosines = [float(a @ b) for a, b in zip(ours, l2_normalize(pooled[0]), strict=True)]
    assert min(cosines) > COSINE_TOLERANCE, (
        f"{model_id}'s pooled ONNX output disagrees with the model's declared pooling "
        f"({cosines}); recorded here because it is the mirror image of the MLX case"
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_id", [PARITY_MODEL, FULL_MODEL])
async def test_a_real_model_is_deterministic_and_batch_invariant(
    model_id: str, backend: str
) -> None:
    """Padding is per batch, so an unmasked reduction fails this and nothing else does.

    The cache is off: with it on, the repeat is a dictionary lookup and proves nothing about the
    model.
    """
    require(model_id, backend)
    embedder = await embedder_for(model_id, backend, cache_entries=0)

    alone = np.asarray(await embedder.embed([TEXTS[0]]))
    crowded = np.asarray(await embedder.embed(list(TEXTS)))
    again = np.asarray(await embedder.embed([TEXTS[0]]))

    assert np.allclose(alone[0], again[0], atol=1e-6)
    assert np.allclose(alone[0], crowded[0], atol=1e-4)


async def test_bge_m3_is_the_model_the_design_settled_on() -> None:
    """1024 dimensions, CLS pooling, 8190 usable tokens, revision pinned.

    Read from the repository rather than from the model card's prose, and asserted here so that
    a change to any of it is a failing test rather than a quiet re-index. 8190 rather than 8192
    because ``<s>`` and ``</s>`` are not content, and rather than 8194 because XLM-RoBERTa's
    position ids start above the padding index.
    """
    require_model(FULL_MODEL)
    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415

    card = read_card(FULL_MODEL, revision=builtin_model_revision(FULL_MODEL))

    assert card.dimension == 1024
    assert card.pooling is Pooling.CLS
    assert card.architecture == "xlm-roberta"
    assert card.max_sequence_length == 8190
    assert card.special_token_count == 2
    assert card.revision is not None, "an unpinned model lets weights change under a corpus"

    print_ = card.fingerprint(backend="mlx", weights_ref="mlx-community/bge-m3-mlx-fp16")
    assert print_.normalized is True
    assert "backend" not in print_.identity()
    assert "weights_ref" not in print_.identity()


async def test_the_chunk_budget_fits_this_model_sixteen_times_over() -> None:
    """The interaction ``parsing.md`` §1.1 refuses on, from the number this ticket supplies."""
    require_model(FULL_MODEL)

    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415

    assert (
        read_card(FULL_MODEL, revision=builtin_model_revision(FULL_MODEL)).max_sequence_length > 512
    )


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_real_model_refuses_text_it_would_truncate(backend: str) -> None:
    """Past the limit the model drops the remainder and raises nothing, so we raise instead."""
    require(PARITY_MODEL, backend)
    embedder = await embedder_for(PARITY_MODEL, backend)
    limit = embedder.fingerprint.max_sequence_length

    with pytest.raises(ContextOverflowError, match=f"{limit}-token limit"):
        await embedder.embed([" ".join(["word"] * (limit + 50))])
