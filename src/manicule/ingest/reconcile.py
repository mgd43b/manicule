"""Deletion detection: what `Connector.reconcile` is for, and the guards it needs.

Incremental sync cannot detect deletion, because a deleted document simply stops appearing.
Without a reconciliation pass the index serves removed documents forever, and no amount of
syncing fixes it.

**Cadence.** After every full sync, and on a schedule for connectors that only ever sync
incrementally. Deletion detection that runs only when somebody remembers is deletion detection
that does not run, so :func:`due` is a computed answer rather than an operator's habit.

**The failure that makes reconciliation dangerous**, and why it needs a mechanical guard
rather than care: if ``reconcile()`` raises partway through — an expired cursor, a 429, a
network drop — the ids seen so far are a *prefix*, not the truth. Diffing a prefix against the
full stored set marks everything not yet enumerated as deleted. One transient error, and the
corpus is gone.

Three guards, all required, none of them sufficient alone:

1. **The diff applies only on a clean completion.** A pass that raises produces no deletions
   at all. Partial results are discarded, not salvaged.
2. **A deletion ceiling.** More than ``max_delete_fraction`` of a connector's live documents
   and the pipeline refuses, records the proposal, and surfaces it for confirmation. A genuine
   bulk deletion is rare and worth a human; a bug that looks like one is not rare at all.
3. **Soft delete only.** A mistaken reconciliation is recoverable by clearing ``deleted_at`` —
   free, inside the grace period. Guard 3 is what makes guard 2 tunable rather than terrifying.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Metadata
    from manicule.core.protocols import Connector
    from manicule.core.sources import SourceId
    from manicule.ingest.ports import IngestStore

LAST_RECONCILE_KEY = "last_clean_reconcile_at"
"""Set only when a pass completed cleanly. A pass that raised must not advance it."""

PROPOSED_DELETION_KEY = "proposed_deletion"
"""Where a refused proposal waits for a human, rather than being recomputed from scratch."""


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What one pass concluded."""

    connector: str
    seen: int = 0
    deleted: tuple[SourceId, ...] = ()
    refused: str = ""
    error: str = ""
    proposed: tuple[SourceId, ...] = field(default=())

    @property
    def clean(self) -> bool:
        """Whether the enumeration completed and the diff was applied."""
        return not self.error and not self.refused


def due(metadata: Metadata, *, interval_s: float, now: datetime | None = None) -> bool:
    """Whether a connector is overdue for reconciliation.

    Takes the connector's stored metadata rather than a timestamp, so a connector that has
    never reconciled — where the key is simply absent — is due, which is the answer that
    matters most and the one an ``is None`` check at the call site keeps getting wrong.
    """
    moment = now or datetime.now(UTC)
    recorded = metadata.get(LAST_RECONCILE_KEY)
    if not isinstance(recorded, str):
        return True
    with contextlib.suppress(ValueError):
        return (moment - datetime.fromisoformat(recorded)).total_seconds() >= interval_s
    return True


async def reconcile(
    connector: Connector,
    store: IngestStore,
    *,
    max_delete_fraction: float = 0.1,
    now: datetime | None = None,
) -> Reconciliation:
    """Diff what the source still has against what is indexed, and soft-delete the difference.

    Ids only — no bodies, no versions — which is what makes a weekly full enumeration
    affordable over a rate-limited API.

    Args:
        connector: Asked to enumerate every id that still exists.
        store: Holds what is indexed for this connector.
        max_delete_fraction: The ceiling from guard 2.
        now: For deterministic tests; defaults to the current time.

    Returns:
        What happened, including the refusal or the error when there was one. Never raises for
        a source-side failure: one connector failing to enumerate is not a reason to stop
        reconciling the others.
    """
    seen: set[SourceId] = set()
    try:
        async for source_id in connector.reconcile():
            seen.add(source_id)
    except Exception as exc:  # noqa: BLE001 - a partial enumeration is data, not a crash
        # Guard 1. What was seen is a prefix, and a prefix diffed against the stored set marks
        # everything not yet enumerated as deleted. Discarded rather than salvaged.
        return Reconciliation(
            connector=connector.name,
            seen=len(seen),
            error=f"{type(exc).__name__}: {exc}",
        )

    known: list[SourceId] = []
    stream = store.known_source_ids(connector.name)
    try:
        async for source_id in stream:
            known.append(source_id)
    finally:
        closer = getattr(stream, "aclose", None)
        if closer is not None:
            await closer()

    missing = tuple(sorted(set(known) - seen))
    if not missing:
        await _record_clean(store, connector.name, now)
        return Reconciliation(connector=connector.name, seen=len(seen))

    # Guard 2. Against the live count, not against what the source reported: a source that
    # returned almost nothing is exactly the case the ceiling exists for.
    if known and len(missing) / len(known) > max_delete_fraction:
        refusal = (
            f"reconciling {connector.name!r} would soft-delete {len(missing)} of "
            f"{len(known)} live document(s), above the {max_delete_fraction:.0%} ceiling. A "
            f"bulk deletion this large is rare and worth confirming; a bug that looks like one "
            f"is not rare at all. The proposal is recorded — confirm it, or fix the source and "
            f"reconcile again."
        )
        await store.record_connector_metadata(
            connector.name,
            {PROPOSED_DELETION_KEY: {"source_ids": list(missing), "live": len(known)}},
        )
        return Reconciliation(
            connector=connector.name, seen=len(seen), refused=refusal, proposed=missing
        )

    # Guard 3. Soft only. Chunks, vectors and FTS rows all stay and become invisible at the
    # join, so restoring is clearing a timestamp — no re-embed, no re-parse, no re-fetch.
    for source_id in missing:
        document = await store.find_document(connector.name, source_id)
        if document is not None:
            await store.soft_delete_document(document.id)
    await _record_clean(store, connector.name, now)
    return Reconciliation(connector=connector.name, seen=len(seen), deleted=missing)


async def confirm_proposed_deletion(
    connector: str, store: IngestStore, *, now: datetime | None = None
) -> Sequence[SourceId]:
    """Apply a proposal that guard 2 refused, and clear it.

    Separate from :func:`reconcile` rather than a flag on it, because confirming is a decision
    about a *recorded* proposal — the set somebody looked at — and re-enumerating first would
    confirm a different set than the one that was reviewed.
    """
    metadata = await store.connector_metadata(connector)
    proposal = metadata.get(PROPOSED_DELETION_KEY)
    ids = proposal.get("source_ids") if isinstance(proposal, dict) else None
    if not isinstance(ids, list):
        return []
    applied: list[SourceId] = []
    for source_id in ids:
        if not isinstance(source_id, str):  # pragma: no cover - defensive against edited JSON
            continue
        document = await store.find_document(connector, source_id)
        if document is not None:
            await store.soft_delete_document(document.id)
        applied.append(source_id)
    await store.record_connector_metadata(connector, {PROPOSED_DELETION_KEY: None})
    await _record_clean(store, connector, now)
    return applied


async def _record_clean(store: IngestStore, connector: str, now: datetime | None) -> None:
    moment = now or datetime.now(UTC)
    await store.record_connector_metadata(
        connector, {LAST_RECONCILE_KEY: moment.isoformat(), PROPOSED_DELETION_KEY: None}
    )


__all__ = [
    "LAST_RECONCILE_KEY",
    "PROPOSED_DELETION_KEY",
    "Reconciliation",
    "confirm_proposed_deletion",
    "due",
    "reconcile",
]
