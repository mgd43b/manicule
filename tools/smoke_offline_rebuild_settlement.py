#!/usr/bin/env python3
"""Rerunnable process-level smoke for offline publication settlement.

The default command creates a synthetic filesystem source and executes every lifecycle phase in
a fresh child process.  Its single JSON result is suitable for CI evidence and contains only
aggregate counters and process memory measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, select, text

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.acquisition import AcquisitionRecordState, AcquisitionRunState
from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.plugins.registry import discover
from manicule.storage import models
from manicule.storage.docstore import SqliteDocStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.embedding import Vector

CONNECTOR = "settlement-smoke-files"
DOCUMENTS = 3


class SmokeEmbedder:
    """Deterministic local embedder with an observable forward-pass count."""

    def __init__(self, identity: str) -> None:
        self.fingerprint = EmbedFingerprint(
            model_id=f"synthetic/{identity}",
            dimension=4,
            pooling=Pooling.MEAN,
            normalized=True,
            tokenizer_id="synthetic-whitespace",
            max_sequence_length=1024,
            backend="synthetic",
        )
        self.texts = 0

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.texts += len(texts)
        vectors: list[Vector] = []
        for value in texts:
            digest = hashlib.sha256(value.encode()).digest()
            vectors.append([digest[index] / 255 for index in range(4)])
        return vectors

    def count_tokens(self, value: str) -> int:
        return len(value.split())


def _runtime(
    data_dir: Path,
    source_dir: Path,
    *,
    enabled: bool,
    identity: str,
    detect_glossary: bool,
) -> tuple[Runtime, SmokeEmbedder]:
    embedder = SmokeEmbedder(identity)
    found = discover()
    found.registry.bind(f"settlement-smoke-{identity}").add(
        keys.EMBEDDER.named("local"),
        lambda _: embedder,
        metadata_factory=lambda _: embedder.fingerprint,
    )
    runtime = Runtime(
        Settings(
            data_dir=data_dir,
            connectors={
                CONNECTOR: {
                    "type": "filesystem",
                    "enabled": enabled,
                    "options": {"root": str(source_dir)},
                }
            },  # pyright: ignore[reportArgumentType]
            embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
            storage={"retain_source_bytes": True},  # pyright: ignore[reportArgumentType]
            ingest={"parse_workers": 1},  # pyright: ignore[reportArgumentType]
            rag={"glossary": {"detect_on_ingest": detect_glossary}},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )
    return runtime, embedder


def _rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


async def _counts(runtime: Runtime) -> dict[str, int]:
    async with runtime.require_engine().connect() as connection:
        return {
            "documents": int(await connection.scalar(select(func.count(models.Document.id))) or 0),
            "chunks": int(await connection.scalar(select(func.count(models.Chunk.id))) or 0),
            "blobs": int(await connection.scalar(select(func.count(models.Blob.hash))) or 0),
            "generations": int(
                await connection.scalar(select(func.count(models.DerivedGeneration.id))) or 0
            ),
        }


async def _phase(  # noqa: PLR0912, PLR0915 - explicit process smoke phases
    args: argparse.Namespace,
) -> dict[str, Any]:
    data_dir = cast("Path", args.data_dir)
    source_dir = cast("Path", args.source_dir)
    enabled = args.phase in {"acquire", "unchanged"}
    runtime, embedder = _runtime(
        data_dir,
        source_dir,
        enabled=enabled,
        identity="first",
        detect_glossary=args.phase in {"second", "unchanged"},
    )
    async with runtime:
        service = ApplicationService(runtime)
        documents = await runtime.documents()
        store = documents
        if not isinstance(store, SqliteDocStore):
            raise TypeError("smoke requires the production SQLite document store")

        if args.phase == "acquire":
            report = await service.connector_sync(CONNECTOR, acquire_only=True)
            snapshot = await store.latest_unsettled_acquisition_run(CONNECTOR)
            if snapshot is None:
                raise AssertionError("acquire-only did not leave a rebuildable snapshot")
            if embedder.texts != 0 or report.discovered != DOCUMENTS:
                raise AssertionError("acquire-only constructed derived work or lost inventory")
            payload: dict[str, Any] = {
                "snapshot_id": snapshot.id,
                "discovered": report.discovered,
                "embed_texts": embedder.texts,
            }
        elif args.phase in {"publish", "second"}:
            report = await service.rebuild_run(args.snapshot_id)
            if report.state != "published" or report.expected_items != DOCUMENTS:
                raise AssertionError("offline rebuild did not publish its exact inventory")
            payload = {
                "generation_id": report.generation_id,
                "vectors_embedded": report.vectors_embedded,
                "vectors_reused": report.vectors_reused,
                "embed_texts": embedder.texts,
            }
        elif args.phase == "restart":
            status = await service.snapshot_status(CONNECTOR)
            verified = await service.snapshot_verify(args.snapshot_id)
            run = await store.get_acquisition_run(args.snapshot_id)
            records = await store.list_acquisition_records(args.snapshot_id)
            if run is None:
                raise AssertionError("published acquisition run disappeared")
            async with runtime.require_engine().connect() as connection:
                vector_publication_id = await connection.scalar(
                    select(models.DerivedGeneration.vector_publication_id).where(
                        models.DerivedGeneration.id == args.generation_id
                    )
                )
                foreign_keys = (await connection.execute(text("PRAGMA foreign_key_check"))).all()
                publication_ids = set(
                    (await connection.execute(select(models.Document.publication_id))).scalars()
                )
                retained_refs = int(
                    await connection.scalar(
                        select(func.count(models.AcquisitionRecord.blob_ref)).where(
                            models.AcquisitionRecord.run_id == args.snapshot_id,
                            models.AcquisitionRecord.blob_ref.is_not(None),
                        )
                    )
                    or 0
                )
            if vector_publication_id is None:
                raise AssertionError("published generation disappeared")
            if foreign_keys or publication_ids != {vector_publication_id}:
                raise AssertionError("relational publication pointers or foreign keys disagree")
            if (
                run.state is not AcquisitionRunState.SETTLED
                or run.indexed_count != DOCUMENTS
                or run.acquired_blob_bytes != 0
                or retained_refs != DOCUMENTS
                or {record.state for record in records} != {AcquisitionRecordState.SETTLED}
            ):
                raise AssertionError("settlement counters or retained ownership disagree")
            if (
                not verified.verified
                or status.lifecycle.pending_items
                or status.lifecycle.backlog_items
            ):
                raise AssertionError("restart status did not converge to verified zero backlog")
            payload = {
                "verified": verified.verified,
                "pending_items": status.lifecycle.pending_items,
                "backlog_items": status.lifecycle.backlog_items,
                "retained_refs": retained_refs,
                "foreign_key_violations": len(foreign_keys),
                "embed_texts": embedder.texts,
            }
        elif args.phase == "unchanged":
            before = await _counts(runtime)
            report = await service.connector_sync(CONNECTOR)
            after = await _counts(runtime)
            async with runtime.require_engine().connect() as connection:
                latest_id = await connection.scalar(
                    select(models.AcquisitionRun.id)
                    .where(models.AcquisitionRun.connector_name == CONNECTOR)
                    .order_by(models.AcquisitionRun.created_at.desc())
                    .limit(1)
                )
            latest = None if latest_id is None else await store.get_acquisition_run(latest_id)
            if latest is None:
                raise AssertionError("unchanged sync did not persist an acquisition result")
            records = await store.list_acquisition_records(latest.id)
            if (
                report.ingested != 0
                or embedder.texts != 0
                or report.lifecycle.reused_items != DOCUMENTS
                or before != after
                or latest.unchanged_count != DOCUMENTS
                or {record.state for record in records} != {AcquisitionRecordState.UNCHANGED}
            ):
                raise AssertionError("unchanged sync duplicated source or derived work")
            payload = {
                "embed_texts": embedder.texts,
                "ingested": report.ingested,
                "reused_items": report.lifecycle.reused_items,
                "counts_unchanged": before == after,
            }
        else:  # pragma: no cover - argparse owns the vocabulary
            raise AssertionError(args.phase)
        payload["counts"] = await _counts(runtime)
    payload["max_rss_bytes"] = _rss_bytes()
    return payload


def _child(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603 - this script and fixed synthetic arguments
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    for index in range(DOCUMENTS):
        (source_dir / f"document-{index}.txt").write_text(
            f"Synthetic settlement smoke document {index}.\n", encoding="utf-8"
        )
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(data_dir),
        "--source-dir",
        str(source_dir),
    ]
    acquire = _child([*base, "--phase", "acquire"])
    publish = _child([*base, "--phase", "publish", "--snapshot-id", acquire["snapshot_id"]])
    restart = _child(
        [
            *base,
            "--phase",
            "restart",
            "--snapshot-id",
            acquire["snapshot_id"],
            "--generation-id",
            publish["generation_id"],
        ]
    )
    second = _child([*base, "--phase", "second", "--snapshot-id", acquire["snapshot_id"]])
    if second["generation_id"] == publish["generation_id"]:
        raise AssertionError("second derived identity replayed the first generation")
    unchanged = _child([*base, "--phase", "unchanged"])
    phases = {
        "acquire": acquire,
        "publish": publish,
        "restart": restart,
        "second": second,
        "unchanged": unchanged,
    }
    peak = max(int(phase["max_rss_bytes"]) for phase in phases.values())
    if peak > args.max_rss_bytes:
        raise AssertionError(f"smoke peak RSS {peak} exceeds bound {args.max_rss_bytes}")
    return {
        "ok": True,
        "peak_rss_bytes": peak,
        "rss_bound_bytes": args.max_rss_bytes,
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--max-rss-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--phase", choices=("acquire", "publish", "restart", "second", "unchanged"))
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--generation-id", default="")
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    args.source_dir = Path(args.source_dir).resolve()
    result = asyncio.run(_phase(args)) if args.phase else _orchestrate(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
