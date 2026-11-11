"""``doctor``'s ``outlook-credential`` check: the one place an Outlook credential is asked about.

Check names are a contract (monitors select on them), so the name is pinned here along with the
closed set of words the facts may carry. Entra and Graph are a ``MockTransport``; addresses are at
``example.org``.
"""

from __future__ import annotations

import asyncio
import json
from importlib.machinery import ModuleSpec
from importlib.util import find_spec as _find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from manicule.app import results as r
from manicule.app import service as service_module
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import outlook_auth
from manicule.connectors.outlook_auth import OutlookCredential
from tests.app.fakes import FakeBackend
from tests.connectors.fake_graph import ADDRESS, CLIENT_ID, REFRESH_TOKEN, TENANT, FakeGraph

if TYPE_CHECKING:
    from manicule.connectors.outlook_config import OutlookConfig

CHECK = "outlook-credential"
OPTIONS = {"account": ADDRESS, "tenant_id": TENANT, "client_id": CLIENT_ID}


def _without_httpx(name: str, package: str | None = None) -> ModuleSpec | None:
    """``importlib.util.find_spec`` on an installation lacking the HTTP client."""
    return None if name == "httpx" else _find_spec(name, package)


@pytest.fixture(autouse=True)
def _no_remembered_probes() -> None:
    """Probe results are reused for a while; a test starts from none."""
    service_module._outlook_observations.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    """Entra and Graph, reachable only through the fake, with the token in the environment."""
    fake = FakeGraph()
    real = outlook_auth.outlook_credential

    def through_the_fake(config: OutlookConfig, **kwargs: Any) -> OutlookCredential:
        return real(config, transport=fake.transport(), **kwargs)

    monkeypatch.setattr(outlook_auth, "outlook_credential", through_the_fake)
    monkeypatch.setenv("OUTLOOK_REFRESH_TOKEN", REFRESH_TOKEN)
    return fake


def service_with(**connectors: dict[str, Any]) -> ApplicationService:
    backend = FakeBackend()
    backend.settings = Settings().model_copy(
        update={
            "connectors": {
                name: ConnectorSettings.model_validate(spec) for name, spec in connectors.items()
            }
        }
    )
    return ApplicationService(backend)


async def outlook_check(**connectors: dict[str, Any]) -> r.Check:
    diagnosis = await service_with(**connectors).doctor()
    for check in diagnosis.checks:
        if check.name == CHECK:
            return check
    offered = ", ".join(c.name for c in diagnosis.checks)
    message = f"no check named {CHECK!r}; the diagnosis carried: {offered}"
    raise AssertionError(message)


TOKEN_FILE: list[str] = []
"""The writable token file a test's sources keep their rotated token in, set per test."""


@pytest.fixture(autouse=True)
def _a_writable_token_file(tmp_path: Path) -> None:
    folder = tmp_path / "tokens"
    folder.mkdir()
    token = folder / "token"
    token.write_text(REFRESH_TOKEN + "\n")
    TOKEN_FILE[:] = [str(token)]


def work(**options: object) -> dict[str, Any]:
    """A source keeping its rotated token in a writable file."""
    files: dict[str, object] = {"refresh_token_file": TOKEN_FILE[0]} if TOKEN_FILE else {}
    return {"type": "outlook", "options": {**OPTIONS, **files, **options}}


async def test_a_corpus_without_outlook_sees_the_check_pass(graph: FakeGraph) -> None:
    check = await outlook_check(notes={"type": "filesystem", "options": {"root": "/x"}})

    assert check.state == "ok"
    assert check.facts["sources"] == {}
    assert graph.requests == [], "nothing is asked of Microsoft when no source needs it"


async def test_a_disabled_source_is_not_asked_about(graph: FakeGraph) -> None:
    check = await outlook_check(work={**work(), "enabled": False})

    assert check.state == "ok"
    assert graph.requests == []


async def test_a_working_credential_for_the_right_mailbox_is_ok(graph: FakeGraph) -> None:
    check = await outlook_check(work=work())

    assert check.state == "ok"
    assert check.facts["sources"] == {"work": "ok"}
    assert graph.refreshes == 1


async def test_the_probe_never_rewrites_the_stored_token(graph: FakeGraph, tmp_path: Path) -> None:
    """``doctor`` is a read: a rotated token written from here would race the server for it."""
    graph.rotate = True
    token = tmp_path / "token"
    token.write_text(REFRESH_TOKEN + "\n")

    check = await outlook_check(work=work(refresh_token_file=str(token)))

    assert check.state == "ok"
    assert token.read_text().strip() == REFRESH_TOKEN


async def test_a_missing_token_fails_naming_the_source_and_asks_nothing_of_microsoft(
    graph: FakeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OUTLOOK_REFRESH_TOKEN")
    absent = str(Path(TOKEN_FILE[0]).parent / "never-written")

    check = await outlook_check(work=work(refresh_token_file=absent))

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "missing_secret"}
    assert "'work'" in check.detail
    assert "refresh_token_file" in check.remedy
    assert graph.requests == []


async def test_a_revoked_grant_fails_with_the_login_command_as_the_remedy(
    graph: FakeGraph,
) -> None:
    graph.refresh_failure = (400, "invalid_grant")

    check = await outlook_check(work=work())

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "grant_revoked"}
    assert "`manicule connector login work`" in check.remedy


async def test_a_refused_app_registration_fails(graph: FakeGraph) -> None:
    graph.refresh_failure = (401, "invalid_client")

    check = await outlook_check(work=work())

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "credential_rejected"}


async def test_a_token_for_another_mailbox_fails_and_prints_no_address(graph: FakeGraph) -> None:
    graph.upn = graph.mail_address = "someone.else@example.org"

    check = await outlook_check(work=work())

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "different_mailbox"}
    payload = check.model_dump_json()
    assert "example.org" not in payload
    assert "someone.else" not in payload


async def test_microsoft_not_answering_degrades_rather_than_fails(graph: FakeGraph) -> None:
    graph.refresh_failure = (503, "temporarily_unavailable")

    check = await outlook_check(work=work())

    assert check.state == "degraded"
    assert check.facts["sources"] == {"work": "unreachable"}
    assert "reach Microsoft" in check.remedy


async def test_each_source_is_reported_and_the_first_broken_one_names_the_remedy(
    graph: FakeGraph,
) -> None:
    check = await outlook_check(alpha=work(), beta=work(account="other@example.org"))

    assert check.state == "failing"
    assert check.facts["sources"] == {"alpha": "ok", "beta": "different_mailbox"}
    assert "`manicule connector login beta`" in check.remedy


async def test_options_that_do_not_validate_fail_the_check_instead_of_vanishing(
    graph: FakeGraph,
) -> None:
    """Nothing else reports them, and a dropped source reads as a corpus with no Outlook in it."""
    check = await outlook_check(work={"type": "outlook", "options": {"client_id": CLIENT_ID}})

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "invalid_options"}
    assert "account, tenant_id and client_id" in check.remedy
    assert graph.requests == []


async def test_a_rejected_value_is_not_echoed_when_options_do_not_validate(
    graph: FakeGraph,
) -> None:
    check = await outlook_check(work=work(tenant_id="common-sentinel-alias"))

    assert check.facts["sources"] == {"work": "invalid_options"}
    assert "sentinel" not in check.model_dump_json()


async def test_neither_a_token_nor_microsofts_prose_reaches_the_check(graph: FakeGraph) -> None:
    graph.refresh_failure = (400, "invalid_grant")

    check = await outlook_check(work=work())

    payload = json.dumps(check.model_dump())
    for secret in (REFRESH_TOKEN, "access-token", "sentinel-prose", ADDRESS):
        assert secret not in payload


async def test_the_check_follows_the_gmail_check_in_the_order_doctor_emits_them() -> None:
    diagnosis = await service_with().doctor()
    names = [check.name for check in diagnosis.checks]

    assert names.index(CHECK) == names.index("gmail-credential") + 1


async def test_a_probe_that_outlasts_the_deadline_is_unreachable_and_does_not_hold_doctor(
    graph: FakeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(self: object, account: str) -> None:
        del self, account
        await asyncio.sleep(3600)

    monkeypatch.setattr(OutlookCredential, "confirm_mailbox", never)
    monkeypatch.setattr(service_module, "OUTLOOK_PROBE_DEADLINE_SECONDS", 0.05)

    check = await outlook_check(alpha=work(), beta=work(account="other@example.org"))

    assert check.state == "degraded"
    assert check.facts["sources"] == {"alpha": "unreachable", "beta": "unreachable"}


async def test_a_missing_http_client_fails_with_the_extra_to_install(
    graph: FakeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine installed without the connectors extra gets a diagnosis, not a traceback."""
    monkeypatch.setattr("importlib.util.find_spec", _without_httpx)

    check = await outlook_check(work=work())

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "dependency_missing"}
    assert check.remedy == "pip install 'manicule[connectors]'"


async def test_a_result_reused_from_a_moment_ago_says_how_old_it_is(graph: FakeGraph) -> None:
    await outlook_check(work=work())
    second = await outlook_check(work=work())

    assert second.facts["freshly_measured"] is False
    assert graph.refreshes == 1, "reused, not asked again"


# --- a rotated token that would not be kept (outlook.md §1.2) --------------------------------


async def test_a_source_with_no_token_file_is_degraded_because_a_rotation_would_be_lost(
    graph: FakeGraph,
) -> None:
    source = work()
    del source["options"]["refresh_token_file"]

    check = await outlook_check(work=source)

    assert check.state == "degraded"
    assert check.facts["sources"] == {"work": "token_not_persisted"}
    assert "refresh_token_file" in check.remedy
    assert "read-only" in check.remedy, "the Kubernetes Secret case is named"
    assert graph.refreshes == 1, "the credential itself was still probed"


async def test_a_token_file_in_a_directory_that_cannot_be_written_is_degraded(
    graph: FakeGraph, tmp_path: Path
) -> None:
    readable = tmp_path / "readonly"
    readable.mkdir()
    (readable / "token").write_text(REFRESH_TOKEN + "\n")
    readable.chmod(0o500)
    try:
        check = await outlook_check(work=work(refresh_token_file=str(readable / "token")))
    finally:
        readable.chmod(0o700)

    assert check.state == "degraded"
    assert check.facts["sources"] == {"work": "token_file_unwritable"}


async def test_a_token_file_whose_directory_does_not_exist_is_degraded(
    graph: FakeGraph, tmp_path: Path
) -> None:
    check = await outlook_check(
        work=work(refresh_token_file=str(tmp_path / "no-such-dir" / "token")),
    )

    assert check.facts["sources"] in (
        {"work": "token_file_unwritable"},
        {"work": "missing_secret"},
    )


async def test_a_failing_credential_outranks_a_persistence_warning(graph: FakeGraph) -> None:
    graph.refresh_failure = (400, "invalid_grant")
    source = work()
    del source["options"]["refresh_token_file"]

    check = await outlook_check(work=source)

    assert check.state == "failing"
    assert check.facts["sources"] == {"work": "grant_revoked"}


async def test_the_persistence_words_carry_no_path_and_no_token(graph: FakeGraph) -> None:
    source = work()
    del source["options"]["refresh_token_file"]

    check = await outlook_check(work=source)

    payload = check.model_dump_json()
    assert REFRESH_TOKEN not in payload
    assert "/" not in check.detail.replace("docs/", "")
