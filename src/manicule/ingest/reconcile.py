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
from typing import TYPE_CHECKING, cast, overload
from uuid import uuid4

from manicule.core.protocols import BatchedReconciliationConnector
from manicule.core.reconciliation import (
    LAST_DISCARDED_KEY,
    LAST_RECONCILE_KEY,
    PROPOSED_DELETION_KEY,
)
from manicule.ingest.containers import retire_derived
from manicule.ingest.ports import ReconciliationStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.protocols import Connector
    from manicule.core.sources import SourceId
    from manicule.ingest.ports import IngestStore

_INVENTORY_PAGE_SIZE = 1_000

OVERDUE_FACTOR = 2.0
"""How many intervals may pass before reconciliation is *overdue* rather than merely *due*.

``due`` is what a scheduler acts on; overdue is what an operator is told. The gap is the slack a
pass in progress, a restart, or one failed attempt needs, so that ``doctor`` does not flap
amber every time a pass is on its way."""


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What one pass concluded."""

    connector: str
    seen: int = 0
    deleted: tuple[SourceId, ...] = ()
    refused: str = ""
    error: str = ""
    error_type: str = ""
    blocked: str = ""
    """Why the pass did not start, when something else held the data (a backup)."""
    proposed: tuple[SourceId, ...] = field(default=())
    live_count: int = 0
    missing_count: int = 0
    applied_count: int = 0
    dry_run: bool = False

    @property
    def clean(self) -> bool:
        """Whether the enumeration completed and the diff was applied."""
        return not self.error and not self.refused and not self.blocked and not self.dry_run


@dataclass(frozen=True, slots=True)
class ReconciliationStanding:
    """Where one connector stands with reconciliation: counts and moments, never source ids.

    What ``connector list`` and ``doctor`` read. Built from the connector's stored metadata so
    that the scheduler, the status surfaces and the checks all answer "is it due, is it overdue,
    is a proposal waiting" from the same two functions rather than three readings of the same
    dictionary.
    """

    last_clean_at: datetime | None = None
    last_attempt_at: datetime | None = None
    proposal_pending: bool = False
    proposal_missing: int = 0
    proposal_live: int = 0
    proposal_recorded_at: datetime | None = None
    live_documents: int = 0

    def due(self, *, interval_s: float, now: datetime | None = None) -> bool:
        """Whether the schedule should reconcile this connector now."""
        return self._behind(interval_s, now)

    def overdue(self, *, interval_s: float, now: datetime | None = None) -> bool:
        """Whether reconciliation has fallen behind by more than the slack allows."""
        return self._behind(interval_s * OVERDUE_FACTOR, now)

    def _behind(self, threshold_s: float, now: datetime | None) -> bool:
        """The one rule both questions share: only a connector worth asking is ever behind.

        A connector with no documents has nothing to lose, and one with a proposal waiting is
        waiting on a person, not on the clock; neither is due or late however old the record.
        """
        return (
            self.live_documents > 0
            and not self.proposal_pending
            and _elapsed(self.last_attempt_at, now) >= threshold_s
        )


def _elapsed(since: datetime | None, now: datetime | None) -> float:
    """Seconds since ``since``; infinite for a connector that has never reconciled."""
    if since is None:
        return float("inf")
    return ((now or datetime.now(UTC)) - since).total_seconds()


def standing(metadata: Mapping[str, object], *, live_documents: int) -> ReconciliationStanding:
    """Summarize a connector's stored metadata. Reads counts and timestamps only."""
    clean = _moment(metadata.get(LAST_RECONCILE_KEY))
    proposal = metadata.get(PROPOSED_DELETION_KEY)
    missing = live = 0
    recorded: datetime | None = None
    if isinstance(proposal, dict):
        held = cast("dict[str, object]", proposal)
        raw_missing = held.get("missing")
        ids = held.get("source_ids")
        raw_live = held.get("live")
        missing = (
            raw_missing
            if isinstance(raw_missing, int)
            else len(cast("list[object]", ids))
            if isinstance(ids, list)
            else 0
        )
        live = raw_live if isinstance(raw_live, int) else 0
        recorded = _moment(held.get("recorded_at"))
    return ReconciliationStanding(
        last_clean_at=clean,
        last_attempt_at=last_attempt(metadata),
        proposal_pending=isinstance(proposal, dict),
        proposal_missing=missing,
        proposal_live=live,
        proposal_recorded_at=recorded,
        live_documents=live_documents,
    )


def _moment(recorded: object) -> datetime | None:
    if isinstance(recorded, str):
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(recorded)
    return None


def last_attempt(metadata: Mapping[str, object]) -> datetime | None:
    """When reconciliation last concluded for this connector, by a clean pass or an operator.

    A discarded proposal counts: the operator looked and decided the source was wrong, and
    asking the source the same question again within the hour would only re-propose the same
    deletion. It is not recorded as a *clean* pass anywhere an operator reads.
    """
    moments = [
        moment
        for key in (LAST_RECONCILE_KEY, LAST_DISCARDED_KEY)
        if (moment := _moment(metadata.get(key))) is not None
    ]
    return max(moments) if moments else None


def due(metadata: Mapping[str, object], *, interval_s: float, now: datetime | None = None) -> bool:
    """Whether a connector is due for reconciliation.

    Takes the connector's stored metadata rather than a timestamp, so a connector that has
    never reconciled — where the key is simply absent — is due, which is the answer that
    matters most and the one an ``is None`` check at the call site keeps getting wrong.

    A connector with a proposal waiting is *not* due, whatever the clock says. A newer full
    inventory supersedes the proposal an operator is reviewing (``docs/ingest.md`` §11.1), so a
    scheduler that kept asking would take the question away from the person answering it.
    """
    return standing(metadata, live_documents=1).due(interval_s=interval_s, now=now)


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
            scope=scope or connector_scope(connector),
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
            error_type=type(exc).__name__,
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
            # And whatever was expanded out of it. A member's source id was never in this
            # connector's inventory, so the diff above cannot name it and a later pass never
            # will — the query that finds candidates excludes derived documents on purpose.
            # Leaving them behind is leaving documents that are searchable, citable, and
            # unreachable by the only thing that would have removed them.
            await retire_derived(store, document)
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
            # The same cleanup the unproposed path does, for the same reason: a member's source
            # id was never in this connector's inventory, so nothing here will ever name it
            # again. An operator confirming a proposal removes a container and would otherwise
            # leave its members live and permanently unreachable.
            await retire_derived(store, document)
        applied.append(source_id)
    await store.record_connector_metadata(connector, {PROPOSED_DELETION_KEY: None})
    await _record_clean(store, connector, now)
    return applied


async def discard_proposed_deletion(
    connector: str,
    store: IngestStore | ReconciliationStore,
    *,
    now: datetime | None = None,
    scope: str | None = None,
) -> Reconciliation:
    """Throw away a proposal that guard 2 refused, deleting nothing.

    The operator's other answer: the proposal was a symptom of the source and not a real bulk
    deletion. Documents stay live; the proposal is cleared; the interval restarts so the
    schedule does not re-propose the same deletion on its next poll.

    Returns:
        The counts the proposal held, with ``applied_count`` zero. ``refused`` names the reason
        when no proposal was waiting.
    """
    moment = now or datetime.now(UTC)
    if isinstance(store, ReconciliationStore):
        if not scope:
            msg = "durable proposal discard requires the current reconciliation scope"
            raise ValueError(msg)
        assessed = await store.discard_reconciliation_proposal(connector, scope=scope, now=moment)
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
        )

    metadata = await store.connector_metadata(connector)
    proposal = metadata.get(PROPOSED_DELETION_KEY)
    if not isinstance(proposal, dict):
        return Reconciliation(connector=connector, refused="no deletion proposal exists")
    ids = proposal.get("source_ids")
    live = proposal.get("live")
    await store.record_connector_metadata(
        connector, {PROPOSED_DELETION_KEY: None, LAST_DISCARDED_KEY: moment.isoformat()}
    )
    return Reconciliation(
        connector=connector,
        live_count=live if isinstance(live, int) else 0,
        missing_count=len(ids) if isinstance(ids, list) else 0,
    )


async def _record_clean(store: IngestStore, connector: str, now: datetime | None) -> None:
    moment = now or datetime.now(UTC)
    await store.record_connector_metadata(
        connector, {LAST_RECONCILE_KEY: moment.isoformat(), PROPOSED_DELETION_KEY: None}
    )


def connector_scope(connector: Connector) -> str:
    """Resolve an explicit connector scope while retaining a safe whole-source default.

    Public because confirming or discarding a durable proposal must present the *current* scope
    (``docs/ingest.md`` §11.1), and the caller has the connector in hand when it does.
    """
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
                error_type=type(exc).__name__,
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
    "LAST_DISCARDED_KEY",
    "LAST_RECONCILE_KEY",
    "OVERDUE_FACTOR",
    "PROPOSED_DELETION_KEY",
    "Reconciliation",
    "ReconciliationStanding",
    "confirm_proposed_deletion",
    "connector_scope",
    "discard_proposed_deletion",
    "due",
    "last_attempt",
    "reconcile",
    "standing",
]
