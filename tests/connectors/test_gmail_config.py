"""The Gmail configuration model: what is required, what is refused, what the scope contains.

Fixtures are synthetic throughout: addresses at ``example.org`` and invented label ids.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from manicule.connectors.config import (
    ALWAYS_EXCLUDED_LABELS,
    REPRESENTATION_VERSION,
    GmailConfig,
    resolve_gmail_client_secret,
    resolve_gmail_credentials,
    resolve_label_ids,
)
from manicule.core.errors import ConfigError

ACCOUNT = "mail@example.org"
CLIENT_ID = "1234567890-example.apps.googleusercontent.com"


def gmail(**overrides: object) -> GmailConfig:
    options: dict[str, object] = {"account": ACCOUNT, "client_id": CLIENT_ID}
    options.update(overrides)
    return GmailConfig.model_validate(options)


# --- required settings --------------------------------------------------------------------------


def test_a_missing_account_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="account"):
        GmailConfig.model_validate({"client_id": CLIENT_ID})


def test_a_missing_client_id_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="client_id"):
        GmailConfig.model_validate({"account": ACCOUNT})


def test_a_missing_client_secret_names_the_setting_and_the_variable() -> None:
    with pytest.raises(ConfigError, match=r"client_secret.*\$GMAIL_CLIENT_SECRET"):
        resolve_gmail_credentials(gmail(), environ={"GMAIL_REFRESH_TOKEN": "r"})


def test_a_missing_refresh_token_names_the_setting_and_the_variable() -> None:
    with pytest.raises(ConfigError, match=r"refresh_token.*\$GMAIL_REFRESH_TOKEN"):
        resolve_gmail_credentials(gmail(), environ={"GMAIL_CLIENT_SECRET": "s"})


def test_the_message_points_at_the_instances_own_options_table() -> None:
    with pytest.raises(ConfigError, match=r"\[connectors\.mail\.options\]"):
        resolve_gmail_credentials(gmail(), environ={}, instance="mail")


def test_the_variables_can_be_renamed_and_the_message_follows() -> None:
    config = gmail(client_secret_env="WORK_SECRET", refresh_token_env="WORK_TOKEN")  # noqa: S106 - names of variables
    with pytest.raises(ConfigError, match=r"\$WORK_SECRET"):
        resolve_gmail_credentials(config, environ={})
    with pytest.raises(ConfigError, match=r"\$WORK_TOKEN"):
        resolve_gmail_credentials(config, environ={"WORK_SECRET": "s"})


def test_the_environment_fills_what_configuration_leaves_unset() -> None:
    resolved = resolve_gmail_credentials(
        gmail(), environ={"GMAIL_CLIENT_SECRET": "s", "GMAIL_REFRESH_TOKEN": "r"}
    )

    assert resolved.client_secret is not None
    assert resolved.client_secret.get_secret_value() == "s"
    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "r"


def test_an_explicit_value_is_never_overwritten_by_the_environment() -> None:
    config = gmail(client_secret=SecretStr("explicit-s"), refresh_token=SecretStr("explicit-r"))
    resolved = resolve_gmail_credentials(
        config, environ={"GMAIL_CLIENT_SECRET": "env-s", "GMAIL_REFRESH_TOKEN": "env-r"}
    )

    assert resolved.client_secret is not None
    assert resolved.client_secret.get_secret_value() == "explicit-s"
    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "explicit-r"


def test_a_blank_environment_variable_counts_as_absent() -> None:
    with pytest.raises(ConfigError, match="client_secret"):
        resolve_gmail_credentials(
            gmail(), environ={"GMAIL_CLIENT_SECRET": "", "GMAIL_REFRESH_TOKEN": "r"}
        )


def test_an_explicitly_empty_secret_is_refused_rather_than_read_as_unset() -> None:
    with pytest.raises(ValidationError, match="refresh_token is set to an empty string"):
        gmail(refresh_token="")


# --- the account --------------------------------------------------------------------------------


def test_the_account_is_stored_lowercase_so_one_address_is_one_scope() -> None:
    assert gmail(account="  Mail@Example.ORG ").account == ACCOUNT


@pytest.mark.parametrize(
    "bad",
    [
        "plainsentinel",
        "@sentinel.example.org",
        "sentinel@",
        "a@sentinel@example.org",
        "a sentinel@example.org",
    ],
)
def test_an_account_that_is_not_an_address_is_refused_without_echoing_it(bad: str) -> None:
    with pytest.raises(ValidationError) as caught:
        gmail(account=bad)

    assert "account must be a mailbox address" in str(caught.value)
    assert bad.strip() not in str(caught.value), "a rejected value is never reflected"


# --- settings that are not settings -------------------------------------------------------------


@pytest.mark.parametrize(
    "option", ["query", "include_spam_trash", "user", "max_message_bytes", "token_url"]
)
def test_the_settings_the_design_refuses_are_refused_rather_than_ignored(option: str) -> None:
    with pytest.raises(ValidationError, match=option):
        gmail(**{option: "anything"})


def test_the_quota_budget_defaults_to_ninety_percent_of_the_per_user_limit() -> None:
    assert gmail().quota_units_per_minute == 5_400


@pytest.mark.parametrize("units", [0, 59, 6_001])
def test_a_budget_that_could_not_work_is_refused(units: int) -> None:
    """Below one raw fetch nothing is ever admitted; above the limit it is a promise Google
    will not keep."""
    with pytest.raises(ValidationError, match="quota_units_per_minute"):
        gmail(quota_units_per_minute=units)


def test_the_retry_and_timeout_settings_match_the_confluence_clients() -> None:
    config = gmail()
    assert config.max_retry_after_seconds == 120.0
    assert config.max_retries == 5
    assert config.request_timeout_seconds == 30.0
    with pytest.raises(ValidationError, match="max_retries"):
        gmail(max_retries=-1)
    with pytest.raises(ValidationError, match="request_timeout_seconds"):
        gmail(request_timeout_seconds=0)
    with pytest.raises(ValidationError, match="max_retry_after_seconds"):
        gmail(max_retry_after_seconds=0)


# --- exclude_labels -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", [*ALWAYS_EXCLUDED_LABELS, "spam", "Trash"])
def test_listing_a_label_that_is_always_excluded_is_refused(name: str) -> None:
    with pytest.raises(ValidationError, match="always excluded"):
        gmail(exclude_labels=[name])


def test_an_empty_label_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="empty name"):
        gmail(exclude_labels=["Receipts", "  "])


def test_label_names_are_deduplicated_and_keep_their_slashes_and_spaces() -> None:
    config = gmail(exclude_labels=["Clients/Acme Corp", "Receipts", "Receipts"])

    assert config.exclude_labels == ("Clients/Acme Corp", "Receipts")


def test_names_resolve_to_sorted_deduplicated_ids() -> None:
    ids = resolve_label_ids(
        ["Receipts", "Clients/Acme Corp"],
        {"Receipts": "Label_9", "Clients/Acme Corp": "Label_2", "x": "Label_1"},
    )

    assert ids == ("Label_2", "Label_9")


def test_an_unknown_label_is_refused_by_name_and_lists_nothing_the_mailbox_has() -> None:
    mailbox = {"Receipts": "Label_9", "Private/Medical": "Label_3"}
    with pytest.raises(ConfigError) as caught:
        resolve_label_ids(["Recipts"], mailbox)

    assert "'Recipts'" in str(caught.value)
    assert "Receipts" not in str(caught.value).replace("'Recipts'", "")
    assert "Private/Medical" not in str(caught.value)


# --- the scope fingerprint and the pin ----------------------------------------------------------


def test_changing_the_excluded_labels_changes_the_fingerprint() -> None:
    config = gmail()

    assert config.scope_fingerprint("mail", ["Label_1"]) != config.scope_fingerprint("mail", [])
    assert config.scope_fingerprint("mail", ["Label_1"]) != config.scope_fingerprint(
        "mail", ["Label_1", "Label_2"]
    )


def test_the_order_and_repetition_of_label_ids_do_not_change_the_fingerprint() -> None:
    config = gmail()

    assert config.scope_fingerprint("mail", ["b", "a", "a"]) == config.scope_fingerprint(
        "mail", ["a", "b"]
    )


def test_the_account_is_part_of_the_fingerprint() -> None:
    assert gmail().scope_fingerprint("mail", []) != gmail(
        account="other@example.org"
    ).scope_fingerprint("mail", [])


def test_the_representation_version_is_part_of_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = gmail().scope_fingerprint("mail", [])
    monkeypatch.setattr("manicule.connectors.config.REPRESENTATION_VERSION", "rfc822/2")

    assert gmail().scope_fingerprint("mail", []) != before
    assert REPRESENTATION_VERSION == "rfc822/1"


def test_settings_that_are_not_scope_do_not_change_the_fingerprint() -> None:
    assert gmail(quota_units_per_minute=600).scope_fingerprint(
        "mail", []
    ) == gmail().scope_fingerprint("mail", [])


def test_the_pinned_identity_names_the_account_and_never_contains_it() -> None:
    pin = gmail().pinned_identity

    assert pin == gmail(account="MAIL@example.org").pinned_identity
    assert pin != gmail(account="other@example.org").pinned_identity
    assert "example" not in pin
    assert "@" not in pin


# --- secrets ------------------------------------------------------------------------------------


def test_printing_the_configuration_never_shows_a_secret() -> None:
    config = gmail(
        client_secret=SecretStr("client-secret-sentinel"),
        refresh_token="refresh-sentinel",  # noqa: S106
    )

    for rendered in (repr(config), str(config), config.model_dump_json(), str(config.model_dump())):
        assert "client-secret-sentinel" not in rendered
        assert "refresh-sentinel" not in rendered


def test_the_client_secret_resolves_on_its_own_when_there_is_no_refresh_token_yet() -> None:
    """``connector login`` runs before a refresh token exists, so it cannot ask for one."""
    secret = resolve_gmail_client_secret(gmail(), environ={"GMAIL_CLIENT_SECRET": "s"})

    assert secret.get_secret_value() == "s"


def test_a_missing_client_secret_is_the_same_refusal_whichever_resolver_asked() -> None:
    with pytest.raises(ConfigError, match=r"client_secret.*\$GMAIL_CLIENT_SECRET"):
        resolve_gmail_client_secret(gmail(), environ={}, instance="mail")
