#!/usr/bin/env python3
"""Qualify the MLX backend's physical-memory footprint over a long batch-one run.

This exists because the defect it guards against is **invisible to the ordinary measurement**.
MLX allocates through Metal, and Metal buffers are not ordinary resident anonymous pages, so
``ps`` and ``resource.getrusage`` do not report them. Measured on ``bge-m3`` fp16 before the
cache bound landed, physical footprint rose 2.45 -> 25.0 GiB over 46 batch-one forward passes
while RSS *fell* from 1.59 to 1.30 GiB. Anyone who reaches for RSS concludes there is no
problem, which is precisely what happened before the machine met the macOS out-of-memory
dialog.

So this runs the real embedder in a **child process** and measures that child from outside with
``/usr/bin/footprint``, the same ``phys_footprint`` Activity Monitor shows. RSS is recorded
alongside it — not because it is informative here, but because the two disagreeing is the
finding, and a report that omits the misleading number cannot show that.

Usage::

    uv run tools/qualify_mlx_memory.py                       # 120 passes, human-readable
    uv run tools/qualify_mlx_memory.py --json report.json    # and a report to attach
    uv run tools/qualify_mlx_memory.py --passes 200

**It downloads nothing.** The MLX weights for the model must already be cached; without them it
refuses rather than fetching several gigabytes. That is what lets it be wired to the existing
``REQUIRE_EMBEDDING_MODELS`` switch instead of becoming a download in every CI run.

Apple Silicon only: without Metal there is no unified-memory allocator to qualify.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

GIB = 1024**3
MODEL = "BAAI/bge-m3"

MAX_CONTENT_TOKENS = 512
"""The ceiling this harness holds its generated inputs to, and reports the real maximum against.

Not the model's limit — ``bge-m3`` reads 8192 — but the size a chunker actually produces. A
qualification run at the model's limit would measure a workload nobody runs.
"""

WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "sigma",
)
"""Synthetic filler. Deterministic, meaningless, and no corpus anywhere near it."""


# --- measurement, which is the platform-specific half ---------------------------------------


def phys_footprint(pid: int) -> int | None:
    """``phys_footprint`` for a process, in bytes, or ``None`` if it cannot be read.

    ``footprint`` prints about four significant digits and picks its own unit, so a value near
    4 GiB arrives as megabytes and is exact to a megabyte, while one in the tens of gigabytes
    is rounded to the gigabyte. That is the right way round for this harness: the precision is
    ample everywhere the bounds are decided, and coarse only in the range that has already
    failed.

    ``None`` means "not measured", never "zero" — a caller must not read a failure to measure
    as a process using no memory.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path, fixed arguments
            ["/usr/bin/footprint", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("phys_footprint:"):
            continue
        return _bytes_from(stripped.split(":", 1)[1].strip())
    return None


def resident_bytes(pid: int) -> int | None:
    """Resident memory from ``ps``, in bytes. The number that does *not* see this defect."""
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path, fixed arguments
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    reported = completed.stdout.strip()
    return int(reported) * 1024 if reported.isdigit() else None


def _bytes_from(value: str) -> int | None:
    """``"1810 MB"`` -> bytes. Binary units, which is what ``footprint`` means by ``MB``."""
    scales = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    parts = value.split()
    expected = 2
    if len(parts) != expected or parts[1] not in scales:
        return None
    try:
        return int(float(parts[0].replace(",", "")) * scales[parts[1]])
    except ValueError:
        return None


# --- the child: the real embedder, on the real weights --------------------------------------


def generate(count: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Deterministic inputs of varying length, none over :data:`MAX_CONTENT_TOKENS`.

    Varying on purpose. manicule pads a batch to its own longest member, so at batch one every
    distinct length is a distinct buffer size — which is the shape of workload that made the
    free-buffer cache grow, and a run of identically sized inputs would not reproduce it.

    Trimmed against the model's **own** tokenizer rather than a word count, because the
    relationship between the two is the tokenizer's business and this is a claim about tokens.
    """
    texts: list[str] = []
    for index in range(count):
        words = 40 + (index * 13) % 260
        text = f"document {index} " + " ".join(
            WORDS[(index + position) % len(WORDS)] for position in range(words)
        )
        while count_tokens(text) > MAX_CONTENT_TOKENS:
            kept = text.split()
            text = " ".join(kept[: max(1, int(len(kept) * 0.9))])
        texts.append(text)
    return texts


async def run_child(passes: int, cache_limit_mb: int | None) -> None:
    """Load the model, embed one text at a time, and report after each pass on stdout."""
    # Deferred, and it matters: only the child may import MLX. The parent measures a process
    # holding Metal allocations, and it cannot be one of them.
    import mlx.core as mx  # noqa: PLC0415
    from manicule.embedding.runtimes.mlx_backend import (  # noqa: PLC0415
        DEFAULT_CACHE_LIMIT_BYTES,
        MlxEmbedder,
    )

    from manicule.embedding.cards import read_card  # noqa: PLC0415

    limit = DEFAULT_CACHE_LIMIT_BYTES if cache_limit_mb is None else cache_limit_mb * 1024 * 1024
    card = read_card(MODEL)
    embedder = MlxEmbedder(card, batch_size=1, cache_entries=0, cache_limit_bytes=limit)
    await embedder.setup()

    texts = generate(passes, embedder.count_tokens)
    largest = max(embedder.count_tokens(text) for text in texts)
    _emit({"event": "loaded", "cache_limit_bytes": limit, "max_content_tokens": largest})
    _await_parent()

    for index, text in enumerate(texts):
        started = time.perf_counter()
        vectors = await embedder.embed([text])
        _emit(
            {
                "event": "pass",
                "index": index,
                "seconds": round(time.perf_counter() - started, 4),
                "dimension": len(vectors[0]),
                "mlx_active_bytes": mx.get_active_memory(),
                "mlx_cache_bytes": mx.get_cache_memory(),
                "mlx_peak_bytes": mx.get_peak_memory(),
            }
        )
        _await_parent()

    await embedder.teardown()
    _emit({"event": "done", "mlx_cache_bytes_after_teardown": mx.get_cache_memory()})


def _emit(record: dict[str, object]) -> None:
    print(json.dumps(record), flush=True)


def _await_parent() -> None:
    """Block until the parent has finished measuring this pass.

    Without this the child races to completion and exits while the parent is still draining a
    pipe full of buffered records — measuring, for most of the run, a process that is idle,
    tearing down, or already gone. That failure is not obvious in the output: the numbers stay
    plausible for a while and then collapse, which reads exactly like a fix working.

    It also makes each measurement quiet. The child is parked here, between forward passes,
    allocating nothing, so the footprint being read is the footprint that *survived* the pass
    rather than one caught mid-allocation.
    """
    if sys.stdin.readline() == "":
        # The parent closed the pipe or died; there is nobody left to measure this run.
        raise SystemExit(1)


# --- the parent: spawn, measure from outside, judge -----------------------------------------


def qualify(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run the child and measure it, returning the report."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--passes",
        str(arguments.passes),
    ]
    if arguments.cache_limit_mb is not None:
        command += ["--cache-limit-mb", str(arguments.cache_limit_mb)]

    child = subprocess.Popen(  # noqa: S603 - our own path
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    samples: list[dict[str, Any]] = []
    baseline: int | None = None
    loaded: dict[str, Any] = {}
    aborted: str | None = None

    if child.stdout is None or child.stdin is None:  # pragma: no cover - both are PIPE above
        msg = "the child was started without the pipes the handshake runs over"
        raise RuntimeError(msg)

    for line in child.stdout:
        record = json.loads(line)
        if record["event"] == "loaded":
            loaded = record
            baseline = phys_footprint(child.pid)
            _say(arguments, f"baseline after load: {_gib(baseline)}")
            _release(child)
            continue
        if record["event"] == "done":
            loaded["mlx_cache_bytes_after_teardown"] = record["mlx_cache_bytes_after_teardown"]
            continue

        record["footprint_bytes"] = phys_footprint(child.pid)
        record["resident_bytes"] = resident_bytes(child.pid)
        samples.append(record)

        footprint = record["footprint_bytes"]
        if footprint is not None and footprint > arguments.abort_gib * GIB:
            # Stop rather than reproduce the out-of-memory dialog that motivated all of this.
            aborted = f"footprint {_gib(footprint)} passed the {arguments.abort_gib} GiB guard"
            child.kill()
            break
        if record["index"] % 10 == 0:
            _say(
                arguments,
                f"pass {record['index']:>4}  footprint {_gib(footprint):>10}  "
                f"rss {_gib(record['resident_bytes']):>10}  "
                f"mlx cache {_gib(record['mlx_cache_bytes']):>10}",
            )
        _release(child)

    child.wait()
    return _report(
        arguments,
        samples,
        baseline=baseline,
        loaded=loaded,
        aborted=aborted,
        exit_code=child.returncode,
    )


def _report(
    arguments: argparse.Namespace,
    samples: list[dict[str, Any]],
    *,
    baseline: int | None,
    loaded: dict[str, Any],
    aborted: str | None,
    exit_code: int,
) -> dict[str, Any]:
    """Judge the run. Every failure names the number that produced it."""
    measured = [s for s in samples if s["footprint_bytes"] is not None]
    failures: list[str] = []
    if aborted is not None:
        failures.append(aborted)

    settle = arguments.settle_passes
    growth: int | None = None
    peak: int | None = None
    if len(measured) >= settle + 2 * _WINDOW:
        after_settle = measured[settle:]
        # Medians of the first and last windows rather than the two endpoint samples. macOS
        # reclaims under pressure from *other* processes, and a run that shares a machine
        # occasionally shows a single sample far below its neighbors — observed here, and
        # confirmed against `vmmap` as real reclaim rather than a bad reading. One such sample
        # landing on an endpoint would swing an endpoint-to-endpoint difference by gigabytes,
        # in whichever direction happens to be convenient. A median over a window cannot.
        growth = _median_footprint(after_settle[-_WINDOW:]) - _median_footprint(
            after_settle[:_WINDOW]
        )
        if growth > arguments.growth_bound_gib * GIB:
            failures.append(
                f"physical footprint grew {_gib(growth)} across the {len(after_settle)} passes "
                f"after the first {settle}, over the {arguments.growth_bound_gib} GiB bound. "
                f"This is the retention signature the cache bound exists to prevent"
            )
    elif aborted is None:
        failures.append(
            f"only {len(measured)} passes were measured, too few to judge growth after "
            f"{settle} settle passes at a window of {_WINDOW}"
        )

    if measured:
        peak = max(s["footprint_bytes"] for s in measured)
        if peak > arguments.peak_ceiling_gib * GIB:
            failures.append(
                f"peak physical footprint {_gib(peak)} passed the "
                f"{arguments.peak_ceiling_gib} GiB ceiling"
            )
    if exit_code != 0 and aborted is None:
        failures.append(f"the child exited {exit_code}")

    return {
        "model": MODEL,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "batch_size": 1,
        "passes_requested": arguments.passes,
        "passes_measured": len(measured),
        "max_content_tokens": loaded.get("max_content_tokens"),
        "content_token_ceiling": MAX_CONTENT_TOKENS,
        "cache_limit_bytes": loaded.get("cache_limit_bytes"),
        "mlx_cache_bytes_after_teardown": loaded.get("mlx_cache_bytes_after_teardown"),
        "baseline_footprint_bytes": baseline,
        "peak_footprint_bytes": peak,
        "growth_after_settle_bytes": growth,
        "settle_passes": settle,
        "bounds": {
            "growth_gib": arguments.growth_bound_gib,
            "peak_gib": arguments.peak_ceiling_gib,
        },
        "samples": [
            {
                "index": s["index"],
                "footprint_bytes": s["footprint_bytes"],
                "resident_bytes": s["resident_bytes"],
                "mlx_active_bytes": s["mlx_active_bytes"],
                "mlx_cache_bytes": s["mlx_cache_bytes"],
                "mlx_peak_bytes": s["mlx_peak_bytes"],
                "seconds": s["seconds"],
            }
            for s in samples
        ],
        "failures": failures,
        "passed": not failures,
    }


def _release(child: subprocess.Popen[str]) -> None:
    """Let the child run the next pass, now that this one has been measured."""
    if child.stdin is None:  # pragma: no cover - started with a pipe
        return
    try:
        child.stdin.write("\n")
        child.stdin.flush()
    except (BrokenPipeError, ValueError):
        # The child is gone. The loop over its stdout ends on its own.
        pass


_WINDOW = 5
"""How many samples each end of the growth comparison is taken over. See :func:`_report`."""


def _median_footprint(samples: list[dict[str, Any]]) -> int:
    ordered = sorted(s["footprint_bytes"] for s in samples)
    return ordered[len(ordered) // 2]


def _gib(value: int | None) -> str:
    return "unmeasured" if value is None else f"{value / GIB:.3f} GiB"


def _say(arguments: argparse.Namespace, message: str) -> None:
    if not arguments.quiet:
        print(message, file=sys.stderr, flush=True)


def weights_cached() -> bool:
    """Whether the MLX conversion is already on disk. Never fetches."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - an embeddings extra

    from manicule.embedding.artifacts import mlx_repo  # noqa: PLC0415 - an embeddings extra

    try:
        snapshot_download(mlx_repo(MODEL), allow_patterns=["*.safetensors"], local_files_only=True)
    except Exception:  # noqa: BLE001 - hub raises several unrelated types for "not cached"
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, default=120, help="forward passes, at batch one")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--quiet", action="store_true", help="suppress progress on stderr")
    parser.add_argument(
        "--cache-limit-mb",
        type=int,
        default=None,
        help="override the backend's MLX cache bound. Pass a very large value to reproduce the "
        "unbounded growth this harness was written against.",
    )
    parser.add_argument("--settle-passes", type=int, default=10)
    parser.add_argument("--growth-bound-gib", type=float, default=1.0)
    parser.add_argument("--peak-ceiling-gib", type=float, default=6.0)
    parser.add_argument(
        "--abort-gib",
        type=float,
        default=24.0,
        help="kill the child above this, rather than take the machine into the macOS "
        "out-of-memory dialog",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.child:
        asyncio.run(run_child(arguments.passes, arguments.cache_limit_mb))
        return 0

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("this qualification measures Metal allocations; it needs Apple Silicon")
        return 2
    if not weights_cached():
        print(
            f"the MLX weights for {MODEL} are not cached, and this harness does not download. "
            f"Run tools/prefetch_embedding_models.py --backend mlx first."
        )
        return 2

    report = qualify(arguments)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))

    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    if report["passed"]:
        print(
            f"PASS: {report['passes_measured']} passes, peak "
            f"{_gib(report['peak_footprint_bytes'])}, growth after {report['settle_passes']} "
            f"settle passes {_gib(report['growth_after_settle_bytes'])}",
            file=sys.stderr,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
