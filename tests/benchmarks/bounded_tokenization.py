"""Measure what the structural chunker hands a tokenizer as one block grows.

The unit tests in ``tests/test_chunking_bounded.py`` assert the invariant with a recording
counter, which is the right shape for CI: it is exact, free, and cannot fail because a runner
was busy. What it cannot show is the thing an operator actually felt — that a single parsed
block of a few megabytes took a connector's derivation worker off the air for a quarter of an
hour. That needs the real vocabulary, because the cost is inside the Hugging Face unigram
encode path and no stand-in reproduces its constant.

So this program runs the same shapes through BGE-M3's own tokenizer and reports seconds
alongside the structural numbers. It is **opt-in**: the suite runs it at a size that costs
nothing with a stand-in counter, and the real-vocabulary matrix runs only where the assets
are already cached. Nothing here downloads a model.

Every input is generated: synthetic prose, synthetic identifiers, synthetic CJK. No corpus
text, no page identifiers, no private paths.

Run the published matrix::

    .venv/bin/python -m tests.benchmarks.bounded_tokenization
    .venv/bin/python -m tests.benchmarks.bounded_tokenization --tokenizer bge-m3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from manicule.chunking import MAX_TOKENS, StructuralChunker, TokenCounter
from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, Document, DocumentStatus, ParsedBlock

KIB = 1024
MIB = 1024 * KIB

BGE_M3_REPO = "BAAI/bge-m3"
BGE_M3_PATTERNS = ("tokenizer.json",)

SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. "
CJK_SENTENCE = "日本語のテキストです。これはテストのための文章です。"
CODE_LINE = "    result = compute(alpha, beta, gamma, delta) + offset - correction\n"


def prose(size: int) -> str:
    """One block, many ordinary sentences, no blank line to split on."""
    return (SENTENCE * (size // len(SENTENCE) + 1))[:size]


def newline_free(size: int) -> str:
    """The worst case: no sentence, word or line boundary anywhere in the block."""
    return ("abcdefghijklmnopqrstuvwxyz0123456789" * (size // 36 + 1))[:size]


def multibyte(size: int) -> str:
    """Text the Latin sentence rule cannot split, in characters that are several bytes."""
    return (CJK_SENTENCE * (size // len(CJK_SENTENCE) + 1))[:size]


def code(size: int) -> str:
    return (CODE_LINE * (size // len(CODE_LINE) + 1))[:size]


SHAPES: dict[str, tuple[Callable[[int], str], BlockKind]] = {
    "prose": (prose, BlockKind.PROSE),
    "newline-free": (newline_free, BlockKind.PROSE),
    "multibyte": (multibyte, BlockKind.PROSE),
    "code": (code, BlockKind.CODE),
}

DEFAULT_SIZES = (256 * KIB, 512 * KIB, MIB, 2 * MIB)


@dataclass(frozen=True)
class Measurement:
    """Aggregate-safe evidence: sizes and counts, never text."""

    shape: str
    tokenizer: str
    input_chars: int
    chunks: int
    tokenizer_calls: int
    largest_call_chars: int
    total_chars_tokenized: int
    amplification: float
    """Total characters tokenized per character of input. The quadratic term shows up here
    long before it shows up in seconds, because it grows with the block and a timing does
    not say what it grew with."""

    max_final_tokens: int
    seconds: float


class _Recorder:
    def __init__(self, count: Callable[[str], int]) -> None:
        self.sizes: list[int] = []
        self._count = count

    def __call__(self, text: str) -> int:
        self.sizes.append(len(text))
        return self._count(text)


def stand_in_counter(text: str) -> int:
    """Four characters per token, which is roughly BGE-M3 on Latin prose.

    A whitespace counter is the suite's usual stand-in and is wrong for this measurement in
    the one way that matters: it calls a megabyte with no space in it a single token, so the
    pathological shapes never reach the splitting paths at all.
    """
    return max(1, len(text) // 4)


def bge_m3_counter() -> Callable[[str], int]:
    """BGE-M3's own vocabulary, from the local cache, with no network request permitted.

    Raises:
        RuntimeError: The assets are not on this machine. Callers in the suite skip on it;
            the command-line entry point reports it and exits non-zero, because a benchmark
            that silently measured something else would be worse than one that stopped.
    """
    from manicule.embedding.runtimes.hub import cached_snapshot  # noqa: PLC0415 - optional extra
    from manicule.embedding.runtimes.tokenization import FastTokenizer  # noqa: PLC0415 - as above

    snapshot = cached_snapshot(BGE_M3_REPO, BGE_M3_PATTERNS)
    if snapshot is None:
        msg = (
            f"{BGE_M3_REPO} is not in the local model cache. This program never downloads a "
            f"model; pre-seed it, or run the default stand-in matrix instead."
        )
        raise RuntimeError(msg)
    tokenizer = FastTokenizer(Path(snapshot) / "tokenizer.json")
    return lambda text: len(tokenizer.content_ids(text))


def document() -> Document:
    return Document(
        id="doc-1",
        source="synthetic",
        source_id="s1",
        uri="https://wiki.example.test/synthetic/large-block",
        title="Synthetic Large Block",
        content_hash="h",
        media_type="text/plain",
        status=DocumentStatus.PARSED,
    )


def measure(shape: str, size: int, count: Callable[[str], int], *, tokenizer: str) -> Measurement:
    """Chunk one synthetic block and report what the tokenizer was asked."""
    build, kind = SHAPES[shape]
    text = build(size)
    block = ParsedBlock(
        kind=kind, text=text, anchor=LineAnchor(start=1, end=len(text.splitlines()) or 1)
    )
    recorder = _Recorder(count)
    chunker = StructuralChunker(TokenCounter("benchmark", recorder, provisional=False))

    started = time.perf_counter()
    chunks = chunker.chunk(document(), [block])
    elapsed = time.perf_counter() - started

    # Taken after the clock stops and outside the recording counter, so measuring the result
    # does not appear in the measurement of the work.
    max_final = max((count(chunk.embed_text) for chunk in chunks), default=0)
    total = sum(recorder.sizes)
    return Measurement(
        shape=shape,
        tokenizer=tokenizer,
        input_chars=len(text),
        chunks=len(chunks),
        tokenizer_calls=len(recorder.sizes),
        largest_call_chars=max(recorder.sizes, default=0),
        total_chars_tokenized=total,
        amplification=round(total / len(text), 2),
        max_final_tokens=max_final,
        seconds=round(elapsed, 3),
    )


def run(
    sizes: tuple[int, ...] = DEFAULT_SIZES, *, tokenizer: str = "stand-in"
) -> list[Measurement]:
    count = stand_in_counter if tokenizer == "stand-in" else bge_m3_counter()
    return [measure(shape, size, count, tokenizer=tokenizer) for shape in SHAPES for size in sizes]


def render(measurements: list[Measurement]) -> str:
    header = (
        f"{'shape':<14}{'input':>12}{'chunks':>9}{'calls':>10}"
        f"{'max call':>11}{'tokenized':>15}{'x input':>9}{'max tok':>9}{'seconds':>10}"
    )
    lines = [header, "-" * len(header)]
    for item in measurements:
        lines.append(
            f"{item.shape:<14}{item.input_chars:>12,}{item.chunks:>9,}"
            f"{item.tokenizer_calls:>10,}{item.largest_call_chars:>11,}"
            f"{item.total_chars_tokenized:>15,}{item.amplification:>9.1f}"
            f"{item.max_final_tokens:>9,}{item.seconds:>10.3f}"
        )
    lines.append("")
    lines.append(f"chunk budget {MAX_TOKENS} tokens; every input synthetic.")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", choices=("stand-in", "bge-m3"), default="stand-in")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="block sizes in bytes",
    )
    parser.add_argument("--json", action="store_true", help="emit measurements as JSON")
    args = parser.parse_args(argv)

    try:
        measurements = run(tuple(args.sizes), tokenizer=args.tokenizer)
    except RuntimeError as error:
        print(error, file=sys.stderr)  # noqa: T201 - benchmark output
        return 1

    if args.json:
        print(json.dumps([asdict(item) for item in measurements], indent=2))  # noqa: T201 - benchmark output
    else:
        print(render(measurements))  # noqa: T201 - benchmark output
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
