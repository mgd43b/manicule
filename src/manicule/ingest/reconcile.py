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

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, overload
from uuid import uuid4

from manicule.core.protocols import BatchedReconciliationConnector
from manicule.ingest.ports import ReconciliationStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Metadata
    from manicule.core.protocols import Connector
    from manicule.core.sources import SourceId
    from manicule.ingest.ports import IngestStore

_INVENTORY_PAGE_SIZE = 1_000

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
    live_count: int = 0
    missing_count: int = 0
    applied_count: int = 0
    dry_run: bool = False

    @property
    def clean(self) -> bool:
        """Whether the enumeration completed and the diff was applied."""
        return not self.error and not self.refused and not self.dry_run


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
    scope: str | None = None,
    dry_run: bool = False,
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
    if isinstance(store, ReconciliationStore):
        return await _reconcile_durable(
            connector,
            store,
            max_delete_fraction=max_delete_fraction,
            now=now,
            scope=scope or _connector_scope(connector),
            dry_run=dry_run,
        )

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
    if dry_run:
        return Reconciliation(
            connector=connector.name,
            seen=len(seen),
            live_count=len(known),
            missing_count=len(missing),
            dry_run=True,
        )
    if not missing:
        await _record_clean(store, connector.name, now)
        return Reconciliation(connector=connector.name, seen=len(seen), live_count=len(known))

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
            connector=connector.name,
            seen=len(seen),
            refused=refusal,
            proposed=missing,
            live_count=len(known),
            missing_count=len(missing),
        )

    # Guard 3. Soft only. Chunks, vectors and FTS rows all stay and become invisible at the
    # join, so restoring is clearing a timestamp — no re-embed, no re-parse, no re-fetch.
    for source_id in missing:
        document = await store.find_document(connector.name, source_id)
        if document is not None:
            await store.soft_delete_document(document.id)
    await _record_clean(store, connector.name, now)
    return Reconciliation(
        connector=connector.name,
        seen=len(seen),
        deleted=missing,
        live_count=len(known),
        missing_count=len(missing),
        applied_count=len(missing),
    )


@overload
async def confirm_proposed_deletion(
    connector: str,
    store: ReconciliationStore,
    *,
    now: datetime | None = None,
    scope: str,
) -> Reconciliation: ...


@overload
async def confirm_proposed_deletion(
    connector: str,
    store: IngestStore,
    *,
    now: datetime | None = None,
    scope: str | None = None,
) -> Sequence[SourceId]: ...


async def confirm_proposed_deletion(
    connector: str,
    store: IngestStore | ReconciliationStore,
    *,
    now: datetime | None = None,
    scope: str | None = None,
) -> Sequence[SourceId] | Reconciliation:
    """Apply a proposal that guard 2 refused, and clear it.

    Separate from :func:`reconcile` rather than a flag on it, because confirming is a decision
    about a *recorded* proposal — the set somebody looked at — and re-enumerating first would
    confirm a different set than the one that was reviewed.
    """
    moment = now or datetime.now(UTC)
    if isinstance(store, ReconciliationStore):
        if not scope:
            msg = "durable deletion confirmation requires the current reconciliation scope"
            raise ValueError(msg)
        assessed = await store.confirm_reconciliation_proposal(connector, scope=scope, now=moment)
        if assessed is None:
            return Reconciliation(
                connector=connector,
                refused="no durable deletion proposal exists for the current scope",
            )
        return Reconciliation(
            connector=assessed.connector,
            seen=assessed.seen_count,
            live_count=assessed.live_count,
            missing_count=assessed.missing_count,
            applied_count=assessed.applied_count,
        )

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


def _connector_scope(connector: Connector) -> str:
    """Resolve an explicit connector scope while retaining a safe whole-source default."""
    declared = getattr(connector, "reconciliation_scope", None)
    if callable(declared):
        declared = declared()
    if isinstance(declared, str) and declared:
        return declared
    return f"whole-connector:{connector.name}"


async def _reconcile_durable(
    connector: Connector,
    store: ReconciliationStore,
    *,
    max_delete_fraction: float,
    now: datetime | None,
    scope: str,
    dry_run: bool,
) -> Reconciliation:
    """Journal a full pass, or resume a completed one after a process crash."""
    moment = now or datetime.now(UTC)
    completed = await store.latest_completed_reconciliation_inventory(connector.name, scope)
    if completed is None:
        run_id = uuid4().hex
        await store.begin_reconciliation_inventory(run_id, connector.name, scope)
        seen = 0
        try:
            if isinstance(connector, BatchedReconciliationConnector):
                async for source_page in connector.reconcile_batches():
                    if not source_page:
                        continue
                    seen += await store.append_reconciliation_inventory_page(
                        run_id, connector.name, scope, source_page
                    )
            else:
                page: list[SourceId] = []
                async for source_id in connector.reconcile():
                    page.append(source_id)
                    if len(page) == _INVENTORY_PAGE_SIZE:
                        seen += await store.append_reconciliation_inventory_page(
                            run_id, connector.name, scope, page
                        )
                        page.clear()
                if page:
                    seen += await store.append_reconciliation_inventory_page(
                        run_id, connector.name, scope, page
                    )
            completed = await store.complete_reconciliation_inventory(
                run_id, connector.name, scope, now=moment
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                store.cancel_reconciliation_inventory(run_id, connector.name, scope)
            )
            raise
        except Exception as exc:  # noqa: BLE001 - incomplete inventory is durable diagnostic state
            await store.cancel_reconciliation_inventory(run_id, connector.name, scope)
            return Reconciliation(
                connector=connector.name,
                seen=seen,
                error=f"{type(exc).__name__}: {exc}",
            )

    assessed = await store.assess_reconciliation_inventory(
        completed,
        max_delete_fraction=max_delete_fraction,
        dry_run=dry_run,
        now=moment,
    )
    return Reconciliation(
        connector=assessed.connector,
        seen=assessed.seen_count,
        refused=assessed.refused,
        live_count=assessed.live_count,
        missing_count=assessed.missing_count,
        applied_count=assessed.applied_count,
        dry_run=assessed.dry_run,
    )


__all__ = [
    "LAST_RECONCILE_KEY",
    "PROPOSED_DELETION_KEY",
    "Reconciliation",
    "confirm_proposed_deletion",
    "due",
    "reconcile",
]
