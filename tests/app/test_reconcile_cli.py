"""``manicule connector reconcile``: the one command, its three answers, and what it prints.

Driven through the real argument parser and renderer with the service substituted at the same
seam ``tests/app/test_surface_parity.py`` uses, so what is asserted is what an operator types and
reads: which operation each flag combination becomes, that contradictory flags are refused before
anything runs, and that the human output is counts and next steps and never a source id.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner, Result

import manicule.cli.main as cli
from manicule.app import commands
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.ingest.reconcile import Reconciliation
from tests.app.fakes import FakeBackend, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.app.commands import Command
    from manicule.app.results import Envelope

SECRET_IDS = ("secret-id-1", "secret-id-2")


@pytest.fixture
def backend() -> FakeBackend:
    made = FakeBackend()
    made.settings.connectors["wiki"] = ConnectorSettings.model_validate(
        {"type": "filesystem", "options": {"root": "."}}
    )
    return made


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> list[Command]:
    """Substitute the service, and record every command the line submitted."""
    service = ApplicationService(backend)
    submitted: list[Command] = []

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    async def dispatch(command: Command) -> Envelope:
        submitted.append(command)
        return await run_op(
            command.op,
            service.workspace,
            lambda: commands.run(service, command, commands.silent),
        )

    monkeypatch.setattr(cli, "_execute", execute)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    return submitted


def run(argv: Sequence[str], *, human: bool = False) -> Result:
    return CliRunner().invoke(cli.app, [*([] if human else ["--json"]), *argv])


def test_a_bare_reconcile_runs_a_pass(served: list[Command], backend: FakeBackend) -> None:
    result = run(["connector", "reconcile", "wiki"])

    assert result.exit_code == 0, result.output
    assert [(c.op, dict(c.arguments)) for c in served] == [
        ("connector_reconcile", {"name": "wiki", "dry_run": False})
    ]
    assert backend.ingestion_.reconcile_calls == [("wiki", False, 0.1)]
    assert json.loads(result.stdout)["data"]["outcome"] == "applied"


def test_dry_run_previews(served: list[Command], backend: FakeBackend) -> None:
    result = run(["connector", "reconcile", "wiki", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert served[0].arguments == {"name": "wiki", "dry_run": True}
    assert json.loads(result.stdout)["data"]["outcome"] == "previewed"
    assert backend.ingestion_.reconcile_calls == [("wiki", True, 0.1)]


@pytest.mark.parametrize(("flag", "confirm"), [("--confirm", True), ("--discard", False)])
def test_the_two_answers_become_the_resolve_operation(
    served: list[Command], backend: FakeBackend, flag: str, *, confirm: bool
) -> None:
    result = run(["connector", "reconcile", "wiki", flag])

    assert result.exit_code == 0, result.output
    assert [(c.op, dict(c.arguments)) for c in served] == [
        ("connector_reconcile_resolve", {"name": "wiki", "confirm": confirm})
    ]
    assert backend.ingestion_.resolve_calls == [("wiki", confirm)]
    assert backend.ingestion_.reconcile_calls == [], "an answer must not enumerate the source"


@pytest.mark.parametrize(
    "flags",
    [
        ["--confirm", "--discard"],
        ["--dry-run", "--confirm"],
        ["--dry-run", "--discard"],
        ["--dry-run", "--confirm", "--discard"],
    ],
)
def test_contradictory_flags_are_refused_before_anything_runs(
    served: list[Command], backend: FakeBackend, flags: list[str]
) -> None:
    result = run(["connector", "reconcile", "wiki", *flags])

    assert result.exit_code != 0
    assert "cannot be combined" in result.output
    assert served == []
    assert backend.ingestion_.reconcile_calls == []
    assert backend.ingestion_.resolve_calls == []


def test_a_refused_pass_exits_nonzero_and_names_both_answers(
    served: list[Command], backend: FakeBackend
) -> None:
    backend.ingestion_.reconcile_result = Reconciliation(
        connector="wiki", refused="over", proposed=SECRET_IDS, live_count=10, missing_count=6
    )

    machine = run(["connector", "reconcile", "wiki"])
    human = run(["connector", "reconcile", "wiki"], human=True)

    assert machine.exit_code == 1
    assert json.loads(machine.stdout)["error"]["type"] == "ReconciliationRefusedError"
    assert human.exit_code == 1
    assert "--confirm" in human.output
    assert "--discard" in human.output
    assert len(served) == 2


def test_the_human_output_is_counts_and_never_a_source_id(
    served: list[Command], backend: FakeBackend
) -> None:
    result = run(["connector", "reconcile", "wiki"], human=True)

    assert result.exit_code == 0, result.output
    assert "soft-deleted" in result.output
    assert "11" in result.output
    for secret in SECRET_IDS:
        assert secret not in result.output
    assert served


def test_a_dry_run_says_it_cannot_be_applied_later(served: list[Command]) -> None:
    result = run(["connector", "reconcile", "wiki", "--dry-run"], human=True)

    assert result.exit_code == 0, result.output
    assert "cannot be applied later" in result.output
    assert served


def test_the_listing_shows_a_waiting_proposal_and_how_to_answer_it(
    served: list[Command], backend: FakeBackend
) -> None:
    from manicule.ingest.reconcile import PROPOSED_DELETION_KEY  # noqa: PLC0415

    backend.store.connector_state["wiki"] = {PROPOSED_DELETION_KEY: {"live": 10, "missing": 6}}

    result = run(["connector", "list"], human=True)

    assert result.exit_code == 0, result.output
    assert "6 of 10" in result.output
    assert "reconcile wiki --confirm" in " ".join(result.output.split())
    assert served == [], "listing is a read and is not submitted to a server"


def test_a_full_sync_prints_the_pass_that_followed_it(
    served: list[Command], backend: FakeBackend
) -> None:
    from manicule.ingest.pipeline import RunReport  # noqa: PLC0415

    report = RunReport(connector="wiki", discovered=3)
    report.full_enumeration = True
    report.enumeration_completed = True
    backend.ingestion_.report = report
    backend.store.add(make_document(backend.workspace, source="wiki", source_id="page.md"))

    result = run(["connector", "sync", "wiki"], human=True)

    assert result.exit_code == 0, result.output
    assert "deletion detection" in result.output
    assert served
