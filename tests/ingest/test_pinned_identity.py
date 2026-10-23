"""A source that has synced as one identity is refused under another.

The scope fingerprint cannot do this: it includes the identity, so a different one makes the
stored watermark invisible rather than wrong, and the run would quietly begin a complete walk of
someone else's data beside the first person's. So a connector declares ``pinned_identity`` and the
pipeline holds the instance to it before anything is read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

import pytest

from manicule.connectors.errors import PinnedIdentityError
from manicule.core.errors import ConfigError
from manicule.core.sources import Watermark
from manicule.ingest.pipeline import (
    PINNED_IDENTITY_KEY,
    WATERMARK_IDENTITY_KEY,
    snapshot_scope,
)
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from manicule.core.sources import DiscoveredDoc

FIRST = "pin-of-the-first-mailbox"
SECOND = "pin-of-the-second-mailbox"


class Pinned(fakes.DictConnector):
    """A connector that declares the identity it was configured with, and counts discoveries."""

    def __init__(self, identity: str, documents: dict[str, str] | None = None) -> None:
        super().__init__(documents if documents is not None else {"a": "alpha"})
        self.pinned_identity = identity
        self.discoveries = 0

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        self.discoveries += 1
        async for found in super().discover(watermark):
            yield found


async def test_the_first_run_records_the_pin_and_a_repeat_under_it_is_unaffected() -> None:
    pipeline, store, _ = build()

    await pipeline.run(Pinned(FIRST))
    await pipeline.run(Pinned(FIRST))

    assert (await store.connector_metadata("memory"))[PINNED_IDENTITY_KEY] == FIRST


async def test_a_source_that_has_synced_is_refused_under_a_different_identity() -> None:
    pipeline, _, _ = build()
    await pipeline.run(Pinned(FIRST))
    other = Pinned(SECOND)

    with pytest.raises(PinnedIdentityError, match="new name"):
        await pipeline.run(other)

    assert other.discoveries == 0, "refused before anything was enumerated"


async def test_the_refusal_is_a_config_error_and_prints_neither_identity() -> None:
    pipeline, _, _ = build()
    await pipeline.run(Pinned(FIRST))

    with pytest.raises(ConfigError) as caught:
        await pipeline.run(Pinned(SECOND))

    assert FIRST not in str(caught.value)
    assert SECOND not in str(caught.value)
    assert "delete the documents" in str(caught.value)


async def test_a_refused_run_leaves_the_recorded_pin_alone() -> None:
    pipeline, store, _ = build()
    await pipeline.run(Pinned(FIRST))

    with pytest.raises(PinnedIdentityError):
        await pipeline.run(Pinned(SECOND))

    assert (await store.connector_metadata("memory"))[PINNED_IDENTITY_KEY] == FIRST


async def test_deleting_the_documents_releases_the_pin() -> None:
    pipeline, store, _ = build()
    await pipeline.run(Pinned(FIRST))
    document = await store.find_document("memory", "a")
    assert document is not None
    await store.soft_delete_document(document.id)

    report = await pipeline.run(Pinned(SECOND))

    assert report.indexed == 1
    assert (await store.connector_metadata("memory"))[PINNED_IDENTITY_KEY] == SECOND


async def test_a_source_that_never_synced_may_change_identity_freely() -> None:
    pipeline, store, _ = build()
    await store.record_connector_metadata("memory", {PINNED_IDENTITY_KEY: FIRST})

    report = await pipeline.run(Pinned(SECOND))

    assert report.indexed == 1
    assert (await store.connector_metadata("memory"))[PINNED_IDENTITY_KEY] == SECOND


async def test_a_run_that_began_and_did_not_settle_counts_as_synced() -> None:
    """Nothing indexed yet, but an unsettled acquisition holds what the first identity fetched."""
    pipeline, store, _ = build()
    await store.record_connector_metadata("memory", {PINNED_IDENTITY_KEY: FIRST})

    class Unsettled:
        async def latest_unsettled_acquisition_run(
            self, connector: str, *, scope_fingerprint: str | None = None
        ) -> object:
            del scope_fingerprint
            return object() if connector == "memory" else None

    pipeline._acquisitions = Unsettled()  # type: ignore[assignment]

    with pytest.raises(PinnedIdentityError):
        await pipeline.run(Pinned(SECOND))


async def test_a_connector_without_a_pin_is_unaffected() -> None:
    pipeline, store, _ = build()

    report = await pipeline.run(fakes.DictConnector({"a": "alpha"}))

    assert report.indexed == 1
    assert PINNED_IDENTITY_KEY not in await store.connector_metadata("memory")


class Cursored(Pinned):
    """A pinned connector with an incremental cursor, recording what it was handed."""

    def __init__(self, identity: str) -> None:
        super().__init__(identity)
        self.handed: list[Watermark | None] = []

    @property
    @override
    def watermark(self) -> Watermark | None:
        return Watermark(value="cursor-1", observed_at=datetime(2026, 10, 1, tzinfo=UTC))

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        self.handed.append(watermark)
        async for found in super().discover(watermark):
            yield found


async def test_the_same_identity_resumes_from_its_cursor() -> None:
    pipeline, _, _ = build()
    await pipeline.run(Cursored(FIRST))
    second = Cursored(FIRST)

    await pipeline.run(second)

    assert second.handed[0] is not None
    assert second.handed[0].value == "cursor-1"


async def test_a_released_pin_does_not_hand_the_old_accounts_cursor_to_the_new_one() -> None:
    """Deleting the documents releases the pin, but the stored cursor is still the old account's."""
    pipeline, store, _ = build()
    await pipeline.run(Cursored(FIRST))
    document = await store.find_document("memory", "a")
    assert document is not None
    await store.soft_delete_document(document.id)
    switched = Cursored(SECOND)

    await pipeline.run(switched)

    assert switched.handed == [None], "a complete walk, not the first account's position"
    assert (await store.connector_metadata("memory"))[WATERMARK_IDENTITY_KEY] == SECOND


async def test_an_interrupted_switch_still_does_not_trust_the_old_cursor() -> None:
    """The pin moves before discovery; the cursor must not become trusted just because it did."""
    pipeline, store, _ = build()
    await pipeline.run(Cursored(FIRST))
    document = await store.find_document("memory", "a")
    assert document is not None
    await store.soft_delete_document(document.id)
    await store.record_connector_metadata("memory", {PINNED_IDENTITY_KEY: SECOND})

    retried = Cursored(SECOND)
    await pipeline.run(retried)

    assert retried.handed == [None]


def with_own_fingerprint(connector: Pinned | fakes.DictConnector) -> object:
    """The connector, supplying a fingerprint of its own that knows nothing of the identity."""
    connector.scope_fingerprint = "the-connectors-own-fingerprint"  # type: ignore[attr-defined]
    return connector


def test_a_declared_identity_is_part_of_the_durable_scope_fingerprint() -> None:
    """Whoever computes the connector's own fingerprint, another account is another scope."""
    first = with_own_fingerprint(Pinned(FIRST))
    second = with_own_fingerprint(Pinned(SECOND))
    again = with_own_fingerprint(Pinned(FIRST))

    assert snapshot_scope(first)[1] != snapshot_scope(second)[1]  # type: ignore[arg-type]
    assert snapshot_scope(first)[1] == snapshot_scope(again)[1]  # type: ignore[arg-type]


def test_a_connector_without_an_identity_keeps_its_fingerprint() -> None:
    connector = with_own_fingerprint(fakes.DictConnector({}))

    assert snapshot_scope(connector)[1] == "the-connectors-own-fingerprint"  # type: ignore[arg-type]
