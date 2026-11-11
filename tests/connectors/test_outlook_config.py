"""The Outlook configuration: what it refuses, what it resolves, what it fingerprints.

``docs/connectors/outlook.md`` §1, §3. Nothing here touches a network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from manicule.connectors.outlook_config import (
    REFUSED_TENANTS,
    OutlookConfig,
    read_token_file,
    resolve_folder_ids,
    resolve_outlook_credentials,
)
from manicule.core.errors import ConfigError
from tests.connectors.fake_graph import ADDRESS, CLIENT_ID, TENANT
from tests.connectors.outlook_support import outlook_config


def valid(**changes: object) -> OutlookConfig:
    return outlook_config(**changes)


@pytest.mark.parametrize("alias", sorted(REFUSED_TENANTS))
def test_the_tenant_aliases_that_let_the_account_choose_the_directory_are_refused(
    alias: str,
) -> None:
    with pytest.raises(ValidationError, match="alias"):
        valid(tenant_id=alias)


@pytest.mark.parametrize("bad", ["contoso.onmicrosoft.com", "../evil", f"{TENANT}/x", "", " "])
def test_a_tenant_that_is_not_a_guid_is_refused_because_it_becomes_part_of_an_address(
    bad: str,
) -> None:
    with pytest.raises(ValidationError):
        valid(tenant_id=bad)


def test_the_tenant_and_client_ids_are_stored_lowercase() -> None:
    config = valid(tenant_id=TENANT.upper(), client_id=CLIENT_ID.upper())

    assert config.tenant_id == TENANT
    assert config.client_id == CLIENT_ID


def test_a_client_id_that_is_not_a_guid_is_refused() -> None:
    with pytest.raises(ValidationError, match="GUID"):
        valid(client_id="not-a-guid")


@pytest.mark.parametrize(
    "bad", ["", "leak-nobody", "leak@b@c", "leak b@example.org", "@leak.example"]
)
def test_an_account_that_is_not_an_address_is_refused_without_echoing_it(bad: str) -> None:
    with pytest.raises(ValidationError) as refused:
        valid(account=bad)

    if bad.strip():
        assert bad not in str(refused.value), "hide_input_in_errors keeps the value out"


def test_the_account_is_case_folded_so_two_spellings_are_one_scope() -> None:
    assert valid(account="Sync.User@Example.ORG").account == ADDRESS


def test_there_is_no_client_secret_setting_and_no_authority_setting() -> None:
    for forbidden in ("client_secret", "authority", "token_url", "graph_url", "query", "user"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            valid(**{forbidden: "x"})


def test_an_instance_that_would_index_nothing_is_refused() -> None:
    with pytest.raises(ValidationError, match="index nothing"):
        valid(mail=False, calendar=False)


def test_an_empty_calendar_window_is_refused() -> None:
    with pytest.raises(ValidationError, match="empty"):
        valid(mail=False, calendar_days_back=0, calendar_days_forward=0)


def test_the_scopes_asked_for_follow_what_is_indexed_and_are_all_read_only() -> None:
    both = set(valid().scopes)
    assert both == {"offline_access", "User.Read", "Mail.Read", "Calendars.Read"}
    assert "Calendars.Read" not in valid(calendar=False).scopes
    assert "Mail.Read" not in valid(mail=False).scopes
    assert not any("Write" in scope or "Send" in scope for scope in both)


@pytest.mark.parametrize("value", [0, 5, -1])
def test_concurrency_is_bounded_by_outlooks_cap_of_four(value: int) -> None:
    if value in {0, 5, -1}:
        with pytest.raises(ValidationError):
            valid(max_concurrent_requests=value)


@pytest.mark.parametrize("bad", ["", "a//b", "/a", "a/", " / "])
def test_an_empty_folder_segment_is_refused_rather_than_excluding_nothing(bad: str) -> None:
    with pytest.raises(ValidationError, match="empty"):
        valid(exclude_folders=[bad])


def test_folder_paths_are_trimmed_and_deduplicated() -> None:
    assert valid(exclude_folders=[" Archive / 2019 ", "Archive/2019"]).exclude_folders == (
        "Archive/2019",
    )


def test_a_blank_token_is_refused_so_it_cannot_read_as_a_set_one() -> None:
    with pytest.raises(ValidationError, match="empty string"):
        OutlookConfig.model_validate(
            {
                "account": ADDRESS,
                "tenant_id": TENANT,
                "client_id": CLIENT_ID,
                "refresh_token": "",
            }
        )


def test_printing_the_configuration_never_shows_the_token() -> None:
    config = valid(refresh_token=SecretStr("sentinel-token-value"))

    assert "sentinel-token-value" not in repr(config)
    assert "sentinel-token-value" not in config.model_dump_json()


# --- identity and fingerprint ---------------------------------------------------------------


def test_the_pinned_identity_is_a_digest_of_the_directory_and_the_account() -> None:
    config = valid()

    assert ADDRESS not in config.pinned_identity
    assert (
        config.pinned_identity
        != valid(tenant_id="bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee").pinned_identity
    )
    assert config.pinned_identity != valid(account="other@example.org").pinned_identity
    assert config.pinned_identity == valid().pinned_identity


def test_the_scope_fingerprint_moves_with_everything_that_changes_what_is_indexed() -> None:
    base = valid().scope_fingerprint("s", ())

    assert base == valid().scope_fingerprint("s", ())
    assert base != valid().scope_fingerprint("s", ("folder-1",))
    assert base != valid(calendar=False).scope_fingerprint("s", ())
    assert base != valid(mail=False).scope_fingerprint("s", ())
    assert base != valid().scope_fingerprint("other", ())
    assert valid().scope_fingerprint("s", ("b", "a")) == valid().scope_fingerprint("s", ("a", "b"))


def test_the_calendar_window_is_not_part_of_the_fingerprint_because_it_keeps_no_position() -> None:
    assert valid().scope_fingerprint("s", ()) == valid(calendar_days_back=7).scope_fingerprint(
        "s", ()
    )


def test_resolving_folder_paths_is_exact_per_segment_and_ignores_case() -> None:
    known = {"Inbox": "i", "Archive": "a", "Archive/2019": "a19"}

    assert resolve_folder_ids(["archive/2019", "INBOX"], known) == ("a19", "i")


def test_an_unknown_folder_path_is_refused_and_the_error_lists_no_folders() -> None:
    with pytest.raises(ConfigError) as refused:
        resolve_folder_ids(["Archive/Nope"], {"Inbox": "i", "Secret Project": "s"})

    assert "Archive/Nope" in str(refused.value)
    assert "Secret Project" not in str(refused.value)


# --- the token ------------------------------------------------------------------------------


def bare(**changes: object) -> OutlookConfig:
    values: dict[str, object] = {
        "account": ADDRESS,
        "tenant_id": TENANT,
        "client_id": CLIENT_ID,
        **changes,
    }
    return OutlookConfig.model_validate(values)


def test_the_token_is_read_from_the_environment_when_nothing_else_supplies_one() -> None:
    resolved = resolve_outlook_credentials(bare(), {"OUTLOOK_REFRESH_TOKEN": "from-env"})

    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "from-env"


def test_a_configured_token_is_never_overwritten_by_the_environment() -> None:
    resolved = resolve_outlook_credentials(
        bare(refresh_token=SecretStr("configured")), {"OUTLOOK_REFRESH_TOKEN": "from-env"}
    )

    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "configured"


def test_the_file_wins_over_the_environment_because_it_holds_the_newer_token(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text("from-file\n")
    resolved = resolve_outlook_credentials(
        bare(refresh_token_file=str(token)), {"OUTLOOK_REFRESH_TOKEN": "from-env"}
    )

    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "from-file"


def test_a_file_that_does_not_exist_yet_falls_through_to_the_environment(tmp_path: Path) -> None:
    resolved = resolve_outlook_credentials(
        bare(refresh_token_file=str(tmp_path / "missing")), {"OUTLOOK_REFRESH_TOKEN": "from-env"}
    )

    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "from-env"


def test_a_blank_variable_counts_as_unset() -> None:
    with pytest.raises(ConfigError):
        resolve_outlook_credentials(bare(), {"OUTLOOK_REFRESH_TOKEN": ""})


def test_a_missing_token_names_every_place_that_was_tried_and_no_value(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refused:
        resolve_outlook_credentials(
            bare(refresh_token_file=str(tmp_path / "t")), {}, instance="work"
        )

    text = str(refused.value)
    assert "[connectors.work.options]" in text
    assert "refresh_token_file" in text
    assert "$OUTLOOK_REFRESH_TOKEN" in text
    assert "connector login work" in text


def test_a_blank_token_file_is_no_token(tmp_path: Path) -> None:
    blank = tmp_path / "blank"
    blank.write_text("  \n")

    assert read_token_file(str(blank)) is None


def test_an_unreadable_token_file_is_a_config_error_that_does_not_quote_its_contents(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(ConfigError, match="could not be read"):
        read_token_file(str(directory))
