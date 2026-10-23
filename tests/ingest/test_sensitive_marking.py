"""A sync of a sensitive source records that fact before it writes anything.

``docs/connectors/gmail.md`` §10. The refusal to serve without authentication outlives the
connector being disabled or removed from configuration because the source carries a durable
mark; this file defends when that mark is written and for whom.
"""

from __future__ import annotations

import contextlib

from tests.ingest import fakes
from tests.ingest.test_pipeline import build

MAIL = "work-mail"


async def test_a_sensitive_source_is_marked_by_its_first_sync() -> None:
    pipeline, store, _ = build(sensitive_sources=frozenset({MAIL}))

    await pipeline.run(fakes.DictConnector({"m1": "hello"}, name=MAIL))

    assert store.sensitive_sources == {MAIL}


async def test_a_source_nobody_named_sensitive_is_not_marked() -> None:
    pipeline, store, _ = build(sensitive_sources=frozenset({MAIL}))

    await pipeline.run(fakes.DictConnector({"p1": "hello"}, name="wiki"))

    assert store.sensitive_sources == set()


async def test_the_default_marks_nothing() -> None:
    pipeline, store, _ = build()

    await pipeline.run(fakes.DictConnector({"m1": "hello"}, name=MAIL))

    assert store.sensitive_sources == set()


async def test_the_mark_is_written_before_enumeration_so_a_failed_first_sync_keeps_it() -> None:
    """A first walk that dies has already put documents in the index; the mark cannot wait."""
    pipeline, store, _ = build(sensitive_sources=frozenset({MAIL}))
    connector = fakes.DictConnector({"m1": "hello"}, name=MAIL)

    async def explode(_watermark: object) -> object:
        raise RuntimeError("the source went away")
        yield  # pragma: no cover - makes this an async generator

    connector.discover = explode  # type: ignore[method-assign,assignment]

    # However the run reports the failure, it began, and the mark was already down.
    with contextlib.suppress(RuntimeError):
        await pipeline.run(connector)

    assert store.sensitive_sources == {MAIL}
