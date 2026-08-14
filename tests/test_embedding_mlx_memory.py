"""The MLX backend's memory lifecycle, checked without Metal and without weights.

The defect these guard against is one that ordinary measurement cannot see. MLX keeps every
buffer a forward pass has finished with in a size-keyed free list rather than returning it, and
bounds that list only by its own limit — which defaults to very nearly the whole machine
(measured: 60.8 GiB of a 64 GiB Mac). Because manicule pads each batch to its own longest
member, a batch of one gives nearly every pass a distinct buffer size, so almost nothing is
reused and the list only grows. Measured over 46 batch-one ``bge-m3`` passes, physical
footprint rose 2.45 -> 25.0 GiB while RSS *fell*.

**What is checked here is the lifecycle, not the number.** Whether the bound actually holds
physical memory down is a claim about Metal, and no fake can answer it — that is
``tools/qualify_mlx_memory.py``, which runs the real weights and measures a real child process
from outside. What a fake can answer, on every platform and in milliseconds, is whether the
calls that establish the bound happen at all and in the right order. Those are the assertions
that fail the moment somebody removes them.

The MLX runtime is replaced wholesale, so these run on Linux CI with no MLX installed, and on
Apple Silicon without loading Metal or fetching a gigabyte of weights.
"""

# The fakes stand in for a library that ships no type information, and two tests read a private
# attribute to show that a teardown after a failed setup touches nothing. Relaxed for this file
# only, and only those rules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from manicule.core.errors import ConfigError
from manicule.embedding.cards import read_card
from manicule.embedding.config import EmbedderConfig, MlxEmbedderConfig
from manicule.embedding.runtimes.mlx_backend import (
    DEFAULT_CACHE_LIMIT_BYTES,
    MEGABYTE,
    MlxEmbedder,
)
from tests.embedding_support import write_model

pytestmark = pytest.mark.anyio


class FakeMlxCore:
    """Just enough of ``mlx.core`` to record what the backend asks the allocator for."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cache_limits: list[int] = []
        self.active = 1_000
        self.cache = 2_000
        self.peak = 3_000

    def set_cache_limit(self, value: int) -> int:
        self.calls.append("set_cache_limit")
        self.cache_limits.append(value)
        return 999

    def clear_cache(self) -> None:
        self.calls.append("clear_cache")
        self.cache = 0

    def get_active_memory(self) -> int:
        return self.active

    def get_cache_memory(self) -> int:
        return self.cache

    def get_peak_memory(self) -> int:
        return self.peak


def install_fake_mlx(monkeypatch: pytest.MonkeyPatch) -> FakeMlxCore:
    """Put a recording MLX in front of the real one, if there is a real one."""
    core = FakeMlxCore()

    mlx = types.ModuleType("mlx")
    mlx.core = core  # pyright: ignore[reportAttributeAccessIssue] - standing in for a module
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    def load_model(path: Path, *, path_to_repo: str) -> object:
        core.calls.append("load_model")
        return object()

    utils = types.ModuleType("mlx_embeddings.utils")
    utils.load_model = load_model  # pyright: ignore[reportAttributeAccessIssue]
    package = types.ModuleType("mlx_embeddings")
    package.utils = utils  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "mlx_embeddings", package)
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", utils)
    return core


def build(tmp_path: Path, **kwargs: Any) -> MlxEmbedder:
    """An embedder pointed at a synthetic repository, so nothing is downloaded."""
    directory = write_model(tmp_path / "model")
    (directory / "model.safetensors").write_bytes(b"not really weights")
    return MlxEmbedder(read_card(str(directory)), weights=str(directory), **kwargs)


async def test_the_allocator_is_bounded_before_the_weights_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is established first, so the load's own buffers are already inside it.

    Order is the assertion. A limit applied after ``load_model`` leaves more than a gigabyte of
    the model's own allocations already in the free list, which is the state this is supposed
    to prevent rather than inherit.
    """
    core = install_fake_mlx(monkeypatch)
    embedder = build(tmp_path)

    await embedder.setup()

    assert core.calls == ["set_cache_limit", "load_model"]


async def test_the_configured_bound_is_the_one_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = install_fake_mlx(monkeypatch)
    embedder = build(tmp_path, cache_limit_bytes=512 * MEGABYTE)

    await embedder.setup()

    assert core.cache_limits == [512 * MEGABYTE]


async def test_the_default_bound_is_applied_when_nothing_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedder built without the argument is bounded, not unbounded.

    The failure this rules out is a default of ``None`` meaning "leave MLX alone", which reads
    as conservative and is the defect.
    """
    core = install_fake_mlx(monkeypatch)
    embedder = build(tmp_path)

    await embedder.setup()

    assert core.cache_limits == [DEFAULT_CACHE_LIMIT_BYTES]
    assert DEFAULT_CACHE_LIMIT_BYTES < 8 * 1024 * MEGABYTE


async def test_teardown_returns_the_cache_rather_than_leaving_it_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the model frees its buffers into the cache, not to the system."""
    core = install_fake_mlx(monkeypatch)
    embedder = build(tmp_path)
    await embedder.setup()

    await embedder.teardown()

    assert core.calls == ["set_cache_limit", "load_model", "clear_cache"]
    assert core.get_cache_memory() == 0


async def test_teardown_after_a_failed_setup_does_not_reach_for_mlx(tmp_path: Path) -> None:
    """A machine with no MLX must still shut down cleanly.

    Deliberately run with **no** fake installed. ``teardown`` is documented safe after a failed
    setup, and it is called on exactly the machines where the runtime may be absent — so an
    unconditional ``import mlx.core`` to clear a cache that cannot exist would turn a clean
    shutdown into an ImportError. The embedder never reached the allocator, so there is nothing
    to return.
    """
    directory = write_model(tmp_path / "model")
    (directory / "model.safetensors").write_bytes(b"not really weights")
    embedder = MlxEmbedder(read_card(str(directory)), weights=str(directory))

    await embedder.teardown()

    assert embedder._allocator_configured is False


async def test_metrics_publish_what_resident_memory_cannot_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator-facing numbers include the cache, which is where the growth hides.

    An operator reading RSS during the run that motivated this saw 1.5 GiB while the process
    held 25. ``mlx_cache_bytes`` is the number that would have said so.
    """
    core = install_fake_mlx(monkeypatch)
    core.active, core.cache, core.peak = 11, 22, 33
    embedder = build(tmp_path, cache_limit_bytes=64 * MEGABYTE)
    await embedder.setup()

    published = {metric.name: metric.value for metric in embedder.metrics()}

    assert published["mlx_active_bytes"] == 11
    assert published["mlx_cache_bytes"] == 22
    assert published["mlx_peak_bytes"] == 33
    assert published["mlx_cache_limit_bytes"] == 64 * MEGABYTE
    assert all(
        metric.unit == "bytes" for metric in embedder.metrics() if metric.name.startswith("mlx_")
    )


async def test_metrics_omit_the_allocator_before_it_has_been_touched(tmp_path: Path) -> None:
    """Silence rather than zero, and no import of a runtime that may not be installed.

    Reporting ``0`` for a gauge nobody has measured is the misleading-low-value failure this
    whole change is about, in miniature.
    """
    directory = write_model(tmp_path / "model")
    (directory / "model.safetensors").write_bytes(b"not really weights")
    embedder = MlxEmbedder(read_card(str(directory)), weights=str(directory))

    names = {metric.name for metric in embedder.metrics()}

    assert not any(name.startswith("mlx_") for name in names)


def test_the_cache_bound_is_configured_where_only_mlx_can_read_it() -> None:
    """``[plugins.config."embedder.onnx"]`` has no such mechanism, so naming it there is refused.

    Accepting it would leave an operator believing they had bounded something.
    """
    assert MlxEmbedderConfig().cache_limit_mb == DEFAULT_CACHE_LIMIT_BYTES // MEGABYTE

    with pytest.raises(ValueError, match="cache_limit_mb"):
        EmbedderConfig(cache_limit_mb=512)  # pyright: ignore[reportCallIssue] - the point


def test_the_mlx_factory_refuses_configuration_that_would_drop_the_bound() -> None:
    """Built outside the container, with the shared model, the bound would silently not apply.

    The registry validates against the model the component registered, so this is unreachable
    through configuration — which is exactly why it is worth a refusal rather than a fallback
    to defaults that look like they came from somewhere.
    """
    from manicule.embedding.plugin import _build_mlx  # noqa: PLC0415 - a private factory

    class Context:
        config = EmbedderConfig()
        settings = None

    with pytest.raises(ConfigError, match="cache bound would not be applied"):
        _build_mlx(Context())  # pyright: ignore[reportArgumentType] - the point


# --- the qualification harness's verdict, on injected measurements -------------------------
#
# The harness's own arithmetic decides whether the defect is caught, so it is worth checking
# against curves whose shape is known rather than only against a real run. Fakes are the right
# instrument here for the same reason they are wrong for the bound itself: this is arithmetic,
# not Metal.


def _namespace(**overrides: object) -> Any:
    from argparse import Namespace  # noqa: PLC0415 - only this section needs it

    defaults: dict[str, object] = {
        "settle_passes": 10,
        "growth_bound_gib": 1.0,
        "peak_ceiling_gib": 6.0,
        "passes": 120,
    }
    return Namespace(**(defaults | overrides))


def _samples(footprints: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "footprint_bytes": value,
            "resident_bytes": 1,
            "mlx_active_bytes": 1,
            "mlx_cache_bytes": 1,
            "mlx_peak_bytes": 1,
            "seconds": 0.0,
        }
        for index, value in enumerate(footprints)
    ]


def _judge(footprints: list[int], **overrides: object) -> dict[str, Any]:
    from tools.qualify_mlx_memory import _report  # noqa: PLC0415 - a CI script, not runtime

    return _report(
        _namespace(**overrides),
        _samples(footprints),
        baseline=1,
        loaded={},
        aborted=None,
        exit_code=0,
    )


GIB = 1024**3


def test_the_harness_passes_a_flat_curve() -> None:
    report = _judge([3 * GIB] * 40)

    assert report["passed"], report["failures"]


def test_the_harness_fails_a_curve_that_keeps_climbing() -> None:
    """The retention signature: a footprint that never settles."""
    report = _judge([(2 * GIB) + index * GIB // 4 for index in range(40)])

    assert not report["passed"]
    assert any("grew" in failure for failure in report["failures"])
    assert any("ceiling" in failure for failure in report["failures"])


def test_one_reclaimed_sample_does_not_decide_the_verdict() -> None:
    """macOS reclaims under pressure from other processes, and it did during this work.

    A single sample far below its neighbors landing on an endpoint would swing an
    endpoint-to-endpoint difference by gigabytes, in whichever direction happened to be
    convenient — reporting a plateau for a climbing run, which is the one wrong answer that
    matters. The comparison is a median over a window so that it cannot.
    """
    climbing = [(2 * GIB) + index * GIB // 4 for index in range(40)]
    climbing[-1] = GIB // 10  # the excursion, on the endpoint that would hide the growth

    report = _judge(climbing)

    assert not report["passed"]
    assert any("grew" in failure for failure in report["failures"])


def test_the_shortest_judgeable_run_is_judged_rather_than_refused() -> None:
    """Exactly two windows after the settle passes is enough, and was once rejected as too few.

    An off-by-one here fails a run that measured everything it was asked to, which reads as a
    memory regression rather than as a harness bug.
    """
    shortest = 10 + 2 * 5

    report = _judge([3 * GIB] * shortest)

    assert report["passed"], report["failures"]
    assert not any("too few" in failure for failure in report["failures"])
    assert _judge([3 * GIB] * (shortest - 1))["failures"] == [
        "only 19 passes were measured, too few to judge growth after 10 settle passes at a "
        "window of 5"
    ]
