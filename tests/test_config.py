"""Configuration: layering, credential resolution, redaction and policy gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import tomli_w
from pydantic import ValidationError

from manicule.config.loader import (
    _strip,  # pyright: ignore[reportPrivateUsage] - the writer's half of the predicate
    load_settings,
    save_settings,
)
from manicule.config.profiles import PROFILES, profile_config
from manicule.config.providers import (
    Egress,
    ProviderSettings,
    egress_for,
    endpoint_egress,
    env_var_names,
    needs_credential,
    resolve_provider_keys,
    runs_in_process,
)
from manicule.config.settings import REDACTED, AuthMode, Mode, Settings, Theme
from manicule.core.errors import ConfigError, PolicyError
from manicule.core.retrieval import RetrievalProfile

# --- credential resolution ---------------------------------------------------------------


def test_the_convention_is_provider_name_plus_api_key() -> None:
    """No table lookup for the common case: a new provider needs no code change."""
    assert env_var_names("openai")[0] == "OPENAI_API_KEY"
    assert env_var_names("some-new-vendor")[0] == "SOME_NEW_VENDOR_API_KEY"


def test_aliases_cover_the_names_that_disagree_with_the_provider() -> None:
    """Tried after the conventional name, never instead of it."""
    assert env_var_names("xai") == ("XAI_API_KEY", "GROK_API_KEY")
    assert env_var_names("google") == ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def test_a_configured_key_is_never_overwritten_by_the_environment() -> None:
    resolved = resolve_provider_keys(
        {"openai": ProviderSettings(api_key="explicit")},  # pyright: ignore[reportArgumentType]
        environ={"OPENAI_API_KEY": "from-env"},
    )
    assert resolved["openai"].api_key is not None
    assert resolved["openai"].api_key.get_secret_value() == "explicit"


def test_a_selected_provider_picks_its_key_up_with_no_config_file_at_all() -> None:
    resolved = resolve_provider_keys(
        {}, required=frozenset({"anthropic"}), environ={"ANTHROPIC_API_KEY": "sk-x"}
    )
    assert resolved["anthropic"].api_key is not None
    assert resolved["anthropic"].api_key.get_secret_value() == "sk-x"


def test_local_providers_are_not_asked_for_a_credential() -> None:
    """Whether a key is needed is a property of the name, wherever the endpoint turns out to be."""
    assert not needs_credential("ollama")
    assert needs_credential("openai")
    resolved = resolve_provider_keys({}, required=frozenset({"ollama"}), environ={})
    assert resolved["ollama"].api_key is None
    assert resolved["ollama"].base_url == "http://localhost:11434"


def test_authenticated_local_clis_are_not_asked_for_a_second_credential() -> None:
    """The command owns login, while its opaque destination remains conservatively remote."""
    for provider in ("codex", "claude"):
        assert needs_credential(provider), "the provider name alone carries no exemption"
        assert egress_for(provider) is Egress.REMOTE
        configured = Settings.model_validate(
            {
                "llm": {
                    "generator": "cli",
                    "provider": provider,
                    "model": "default",
                    "context_window": 32_768,
                },
                "embedding": {"provider": "mlx"},
            }
        )
        assert not any("has no API key" in problem for problem in configured.policy_problems())


def test_cli_auth_exemption_does_not_leak_to_other_component_selections() -> None:
    configured = Settings.model_validate(
        {
            "llm": {"generator": "litellm", "provider": "codex", "model": "default"},
            "embedding": {"provider": "mlx"},
        }
    )
    problems = configured.policy_problems()
    assert any("provider 'codex'" in problem for problem in problems)
    assert any("llm.generator to 'cli'" in problem for problem in problems)

    configured = Settings.model_validate(
        {
            "llm": {
                "generator": "cli",
                "provider": "codex",
                "model": "default",
                "context_window": 32_768,
            },
            "embedding": {"provider": "codex"},
        }
    )
    assert any("provider 'codex'" in problem for problem in configured.policy_problems())


# --- egress ------------------------------------------------------------------------------


def test_an_in_process_backend_has_no_endpoint_to_check() -> None:
    """The one thing a provider's name settles on its own: there is nowhere to send anything."""
    assert runs_in_process("mlx")
    assert egress_for("mlx") is Egress.IN_PROCESS
    assert egress_for("mlx", "http://gpu-box.lan:11434") is Egress.IN_PROCESS


def test_a_served_provider_with_no_endpoint_falls_back_to_what_its_name_implies() -> None:
    """A default, and stated as one — ``DEFAULT_BASE_URLS`` fills the same value in."""
    assert egress_for("ollama") is Egress.LOOPBACK
    assert egress_for("openai") is Egress.REMOTE


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434",
        "http://127.4.5.6:11434",
        "http://[::1]:11434",
        "http://[::ffff:127.0.0.1]:11434",
        "http://localhost:11434",
        "http://LOCALHOST:11434",
        "http://ollama.localhost:11434",
        "http://user:pass@127.0.0.1:11434",
        "unix:///var/run/ollama.sock",
    ],
)
def test_an_endpoint_on_this_machine_is_loopback(base_url: str) -> None:
    """Including the aliases, because a policy that only knows one spelling is a policy people
    route around."""
    assert endpoint_egress(base_url) is Egress.LOOPBACK


@pytest.mark.parametrize(
    "base_url",
    [
        "http://gpu-box.lan:11434",
        "http://192.168.1.10:11434",
        "http://0.0.0.0:11434",
        "http://[::]:11434",
        "http://169.254.169.254/latest/meta-data",
        "http://[fe80::1]:11434",
        "http://localhost.example.com:11434",
        "http://not-localhost:11434",
        "localhost:11434",
        "http://[::1:11434",
    ],
)
def test_an_endpoint_that_is_not_demonstrably_this_machine_is_remote(base_url: str) -> None:
    """Conservative in one direction on purpose.

    ``0.0.0.0`` and ``::`` are addresses you bind rather than dial; a link-local address is
    another machine on the same link; a hostname is a name, and no name is resolved here — see
    ``egress_for`` for why. A scheme-less or unparseable value is refused rather than guessed
    at, and the refusal names the URL it could not vouch for.
    """
    assert egress_for("ollama", base_url) is Egress.REMOTE
    assert endpoint_egress(base_url) is Egress.REMOTE


def test_an_empty_base_url_is_an_absent_one_rather_than_a_bad_one() -> None:
    """Nothing configured falls back to the name; a URL that parses to no host does not."""
    assert egress_for("ollama", "") is Egress.LOOPBACK
    assert egress_for("ollama", "   ") is Egress.LOOPBACK
    assert endpoint_egress("") is Egress.REMOTE


def test_a_key_in_a_dotenv_file_is_found(manicule_environment: Path) -> None:
    """Provider variables are not prefixed, so the settings sources never see them."""
    (manicule_environment / ".env").write_text('OPENAI_API_KEY="sk-from-dotenv"\n')
    settings = Settings(llm={"provider": "openai"})  # pyright: ignore[reportArgumentType]
    key = settings.provider("openai").api_key
    assert key is not None
    assert key.get_secret_value() == "sk-from-dotenv"


def test_a_dotenv_file_may_hold_variables_that_are_not_settings(
    manicule_environment: Path,
) -> None:
    """A .env file is shared ground: project variables live alongside manicule's.

    Treating every unrecognized line as a misspelled setting would make a normal .env file
    unloadable. A misspelled *prefixed* name is still rejected, which is where a typo shows.
    """
    (manicule_environment / ".env").write_text(
        "DATABASE_URL=postgres://elsewhere\nOPENAI_API_KEY=sk-x\nMANICULE_WORKSPACE=docs\n"
    )
    settings = Settings(llm={"provider": "openai"})  # pyright: ignore[reportArgumentType]
    assert settings.workspace == "docs"
    key = settings.provider("openai").api_key
    assert key is not None
    assert key.get_secret_value() == "sk-x"


def test_a_real_environment_variable_beats_a_dotenv_file(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (manicule_environment / ".env").write_text("OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    settings = Settings(llm={"provider": "openai"})  # pyright: ignore[reportArgumentType]
    key = settings.provider("openai").api_key
    assert key is not None
    assert key.get_secret_value() == "from-env"


# --- layering ----------------------------------------------------------------------------


def test_a_config_file_is_read(manicule_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = manicule_environment / "manicule.toml"
    path.write_text(tomli_w.dumps({"workspace": "docs", "ui": {"theme": "dark"}}))
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))

    settings = Settings()
    assert settings.workspace == "docs"
    assert settings.ui.theme is Theme.DARK


def test_the_environment_overrides_one_field_and_leaves_the_rest(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file and an environment are two halves of one configuration, not two rival ones."""
    path = manicule_environment / "manicule.toml"
    path.write_text(tomli_w.dumps({"workspace": "docs", "ui": {"theme": "dark", "locale": "en"}}))
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))
    monkeypatch.setenv("MANICULE_UI__THEME", "light")

    settings = Settings()
    assert settings.ui.theme is Theme.LIGHT
    assert settings.ui.locale == "en"
    assert settings.workspace == "docs"


def test_a_typo_in_the_config_file_is_an_error_not_a_shrug(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setting that appears to be in force and is not is worse than one that fails."""
    path = manicule_environment / "manicule.toml"
    path.write_text(tomli_w.dumps({"worksapce": "typo"}))
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))

    with pytest.raises(ConfigError, match="worksapce"):
        load_settings()


def test_paths_expand_the_home_shorthand() -> None:
    settings = Settings(data_dir=Path("~/somewhere"))
    assert "~" not in str(settings.data_dir)


def test_durable_ingest_backlogs_have_finite_default_bounds() -> None:
    ingest = Settings().ingest

    assert ingest.max_journal_records > 0
    assert ingest.max_journal_metadata_bytes > 0
    assert ingest.max_acquired_blob_backlog_bytes > 0
    assert ingest.min_disk_headroom_bytes > 0


@pytest.mark.parametrize(
    "field",
    [
        "max_journal_records",
        "max_journal_metadata_bytes",
        "max_acquired_blob_backlog_bytes",
        "min_disk_headroom_bytes",
    ],
)
def test_a_durable_ingest_bound_cannot_be_disabled_with_zero(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings(ingest={field: 0})  # pyright: ignore[reportArgumentType]


# --- secrets ------------------------------------------------------------------------------


def test_reading_the_configuration_never_hands_out_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What ``config show`` and the configuration API return."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print-me")
    settings = Settings(llm={"provider": "openai"})  # pyright: ignore[reportArgumentType]

    rendered = repr(settings.redacted())
    assert "sk-do-not-print-me" not in rendered
    assert "**********" in rendered


def test_secrecy_is_decided_by_the_declared_type_and_not_by_the_name() -> None:
    """A name is a bad proxy for a credential, and it was wrong in both directions.

    The old rule matched substrings. Five ordinary settings contain "token" and are floats and
    ints — `llm.first_token_timeout_s`, `llm.token_safety_factor`, `llm.token_drift_tolerance`,
    `rag.context.system_prompt_tokens`, `ingest.target_batch_tokens` — so all five were masked
    in `config show` *and* refused by `config set`, leaving an operator no way to inspect or
    change them. The hand-maintained exception list (`maxtokens`, `overlaptokens`, `tokenizer`)
    is the scar from the same problem, extended one name at a time.

    The other direction is the serious one: `security.data_policy.auto_redact.hash_salt` is a
    real `SecretStr` and matches none of the markers, so the one thing this machinery exists to
    hide was printed in the clear and written to disk.

    Both are now decided from the annotation. The exhaustive walk is the point — it holds every
    field in `Settings` to its declared type, so a new `SecretStr` is covered on the day it is
    added rather than when somebody notices its name does not match.
    """
    import typing  # noqa: PLC0415 - only this derivation reads annotations

    from pydantic import BaseModel, SecretStr  # noqa: PLC0415 - only this derivation needs them

    from manicule.config.settings import secret_setting  # noqa: PLC0415

    seen: set[int] = set()
    disagreements: list[str] = []

    def walk(model: type[BaseModel], path: tuple[str, ...] = ()) -> None:
        if id(model) in seen:
            return
        seen.add(id(model))
        for name, field in model.model_fields.items():
            here = (*path, name)
            annotation = field.annotation
            declared = annotation is SecretStr or SecretStr in typing.get_args(annotation)
            if secret_setting(here) is not declared:
                disagreements.append(f"{'.'.join(here)} declared={declared}")
            for candidate in (annotation, *typing.get_args(annotation)):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    walk(candidate, here)

    walk(Settings)
    assert not disagreements, (
        f"these settings are classified against their declared type: {disagreements}. "
        f"Secrecy is read from the annotation; a name that merely looks like a credential is "
        f"not one, and a credential whose name does not is still one."
    )

    # And the two that motivated it, stated by name so the failure reads as the bug.
    assert secret_setting(("security", "data_policy", "auto_redact", "hash_salt"))
    assert not secret_setting(("llm", "token_safety_factor"))


def test_a_secret_under_an_arbitrary_key_is_still_masked() -> None:
    """`llm.providers` is a table keyed by a name somebody chose, not by declared fields.

    The walk has to spend that segment on the key rather than looking it up as a field, or the
    resolution stops at `providers` and every provider's `api_key` is handed out in the clear.
    """
    settings = Settings.model_validate(
        {
            "providers": {"openai": {"api_key": "sk-live-do-not-print"}},
            "security": {"data_policy": {"auto_redact": {"hash_salt": "salt-do-not-print"}}},
        }
    )

    rendered = repr(settings.redacted())

    assert "sk-live-do-not-print" not in rendered
    assert "salt-do-not-print" not in rendered
    assert rendered.count("**********") >= 2


def test_a_credential_under_an_untyped_subtree_is_still_redacted() -> None:
    """`plugins.config` has no declared type, so the name rule is what must cover it.

    Secrecy is read from the annotation wherever there is one — but `plugins.config` holds
    arbitrary per-component options, validated against each component's own model rather than
    against `Settings`, so the walk leaves the declared model there and `looks_secret` is the
    only thing left.

    The type-driven rewrite recursed into that subtree without judging anything, so a plugin's
    `api_key` came back from `config show` in the clear while `save_settings` — which reaches
    the same conclusion through `secret_setting` — correctly omitted it from the file. Display
    and disk disagreeing about what is a credential is exactly what one predicate exists to
    prevent, and the leak was on the side that hands the value to a caller.

    Both halves are asserted together, because either alone would pass against a fix that made
    them agree by leaking from both.
    """
    settings = Settings.model_validate(
        {
            "plugins": {
                "config": {
                    "connector.confluence": {
                        "api_key": "sk-do-not-print",
                        "batch_size": 50,
                        "nested": {"client_secret": "also-do-not-print", "depth": 3},
                    }
                }
            }
        }
    )

    def component(tree: Any) -> dict[str, Any]:
        plugins = cast("dict[str, Any]", tree["plugins"])
        config = cast("dict[str, Any]", plugins["config"])
        return cast("dict[str, Any]", config["connector.confluence"])

    shown = component(settings.redacted())
    nested = cast("dict[str, Any]", shown["nested"])
    assert shown["api_key"] == REDACTED
    # Nested, because an untyped subtree is arbitrarily deep and one level would look fixed.
    assert nested["client_secret"] == REDACTED
    # And nothing else is swept up: a component option that is not a credential stays readable,
    # or `config show` becomes useless for the thing it is for.
    assert shown["batch_size"] == 50
    assert nested["depth"] == 3

    written = component(_strip(settings.model_dump(mode="json")))
    assert "api_key" not in written, "the writer already withheld it; the two must not disagree"
    assert written["batch_size"] == 50


def test_saving_the_configuration_never_writes_a_credential(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials belong in the environment, not in backups, exports and version control."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-save-me")
    settings = Settings(llm={"provider": "openai"}, workspace="docs")  # pyright: ignore[reportArgumentType]

    written = save_settings(settings, manicule_environment / "out.toml")
    body = written.read_text()

    assert "sk-do-not-save-me" not in body
    assert "docs" in body
    assert written.stat().st_mode & 0o077 == 0


def test_a_saved_configuration_loads_back(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Settings(workspace="docs", mode=Mode.TEAM, ui={"theme": "dark"})  # pyright: ignore[reportArgumentType]
    path = save_settings(original, manicule_environment / "round-trip.toml")
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))

    reloaded = load_settings()
    assert reloaded.workspace == "docs"
    assert reloaded.mode is Mode.TEAM
    assert reloaded.ui.theme is Theme.DARK


# --- policy gates -------------------------------------------------------------------------


def test_a_hosted_model_under_a_local_only_policy_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    settings = Settings(
        llm={"provider": "openai"},  # pyright: ignore[reportArgumentType]
        security={"data_policy": {"cloud_allowed": False}},  # pyright: ignore[reportArgumentType]
    )
    problems = settings.policy_problems()
    assert any("cloud_allowed" in problem for problem in problems)
    with pytest.raises(PolicyError):
        settings.require_valid()


def test_a_local_provider_pointed_at_another_host_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect this whole section exists for.

    ``is_local("ollama")`` was true whatever ``base_url`` said, so this configuration started
    cleanly while every prompt crossed the network under the setting that forbids it.
    """
    del monkeypatch
    settings = Settings(
        llm={"provider": "ollama", "base_url": "http://gpu-box.lan:11434"},  # pyright: ignore[reportArgumentType]
        security={"data_policy": {"cloud_allowed": False}},  # pyright: ignore[reportArgumentType]
    )

    assert settings.cloud_providers_in_use == frozenset({"ollama"})
    problems = settings.policy_problems()
    assert any("gpu-box.lan" in problem for problem in problems), problems
    with pytest.raises(PolicyError):
        settings.require_valid()


def test_a_local_provider_on_loopback_still_starts_under_that_policy() -> None:
    """The safe configuration must not be the one that fails."""
    settings = Settings(
        llm={"provider": "ollama", "base_url": "http://127.0.0.1:11434"},  # pyright: ignore[reportArgumentType]
        security={"data_policy": {"cloud_allowed": False}},  # pyright: ignore[reportArgumentType]
    )

    assert settings.cloud_providers_in_use == frozenset()
    assert settings.policy_problems() == []


def test_a_hosted_provider_served_on_loopback_is_not_treated_as_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error in the other direction: an OpenAI-compatible server on 127.0.0.1.

    It was classified cloud and refused under a local-only policy, which made the safe
    configuration the one that failed.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    settings = Settings(
        llm={"provider": "openai", "base_url": "http://127.0.0.1:8080/v1"},  # pyright: ignore[reportArgumentType]
        security={"data_policy": {"cloud_allowed": False}},  # pyright: ignore[reportArgumentType]
    )

    assert settings.cloud_providers_in_use == frozenset()
    assert settings.policy_problems() == []


def test_the_endpoints_a_configuration_will_open_are_reported_per_role() -> None:
    """One provider name, two endpoints, and only one of them on this machine."""
    settings = Settings(
        llm={"provider": "ollama", "base_url": "http://gpu-box.lan:11434"},  # pyright: ignore[reportArgumentType]
        embedding={"provider": "ollama"},  # pyright: ignore[reportArgumentType]
    )

    egress = {endpoint.role.value: endpoint.egress for endpoint in settings.selected_endpoints}
    assert egress == {"llm": Egress.REMOTE, "embedding": Egress.LOOPBACK}


def test_a_blank_provider_is_refused_at_validation_rather_than_later(
    manicule_environment: Path,
) -> None:
    """``policy_problems`` reports; it must not be the thing that raises.

    Resolving an endpoint per role means the provider name is read while the report is being
    built, so a name that is not a name has to be caught at the boundary — the same
    ``min_length`` the model fields beside it already carry.
    """
    del manicule_environment
    with pytest.raises(ValidationError, match="provider"):
        Settings(llm={"provider": ""})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="provider"):
        Settings(embedding={"provider": ""})  # pyright: ignore[reportArgumentType]

    # And anything that slips past it is still reported rather than raised.
    problems = Settings(llm={"provider": "   "}).policy_problems()  # pyright: ignore[reportArgumentType]
    assert any("no API key" in problem for problem in problems)


def test_a_base_url_on_an_in_process_provider_is_reported_rather_than_ignored() -> None:
    """A setting that appears to be in force and is not is worse than one that fails."""
    settings = Settings(
        embedding={"provider": "mlx"},  # pyright: ignore[reportArgumentType]
        providers={"mlx": {"base_url": "http://gpu-box.lan:11434"}},  # pyright: ignore[reportArgumentType]
    )

    assert any("dials nothing" in problem for problem in settings.policy_problems())


def test_a_hosted_model_with_no_credential_says_which_variable_to_set() -> None:
    settings = Settings(llm={"provider": "anthropic"})  # pyright: ignore[reportArgumentType]
    problems = settings.policy_problems()
    assert any("ANTHROPIC_API_KEY" in problem for problem in problems)


def test_binding_beyond_loopback_without_authentication_is_refused() -> None:
    """An unauthenticated index on a routable address is readable by anyone who reaches it."""
    settings = Settings(
        security={"transport": {"bind_host": "0.0.0.0"}},  # noqa: S104  # pyright: ignore[reportArgumentType]
    )
    assert any("bind_host" in problem for problem in settings.policy_problems())


def test_oauth_without_a_provider_is_refused() -> None:
    settings = Settings(security={"auth": {"mode": "oauth"}})  # pyright: ignore[reportArgumentType]
    assert any("oauth" in problem for problem in settings.policy_problems())


def test_auditing_to_a_webhook_with_no_webhook_is_refused() -> None:
    settings = Settings(
        security={"audit": {"enabled": True, "destination": "webhook"}},  # pyright: ignore[reportArgumentType]
    )
    assert any("webhooks is empty" in problem for problem in settings.policy_problems())


def test_a_saved_configuration_keeps_list_valued_settings(
    manicule_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Settings(
        events={  # pyright: ignore[reportArgumentType]
            "webhooks": [{"url": "https://example/hook", "events": ["indexed"], "secret": "s3cr3t"}]
        }
    )
    path = save_settings(original, manicule_environment / "hooks.toml")
    body = path.read_text()
    assert "https://example/hook" in body
    assert "s3cr3t" not in body

    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))
    reloaded = load_settings()
    assert reloaded.events.webhooks[0].events == ("indexed",)


def test_the_default_configuration_is_runnable_and_local() -> None:
    settings = Settings()
    assert settings.security.transport.is_loopback
    assert settings.security.auth.mode is AuthMode.NONE
    assert settings.cloud_providers_in_use == frozenset()
    assert settings.policy_problems() == []


def test_every_problem_is_reported_at_once() -> None:
    """Fixing one misconfiguration only to be told about the next is a poor afternoon."""
    settings = Settings(
        llm={"provider": "openai"},  # pyright: ignore[reportArgumentType]
        security={  # pyright: ignore[reportArgumentType]
            "transport": {"bind_host": "0.0.0.0"},  # noqa: S104
            "data_policy": {"cloud_allowed": False},
        },
    )
    assert len(settings.policy_problems()) >= 3


# --- profiles ------------------------------------------------------------------------------


def test_profiles_trade_cost_against_depth_monotonically() -> None:
    fast, balanced, precise = (PROFILES[p] for p in RetrievalProfile)
    assert fast.candidates < balanced.candidates < precise.candidates
    assert fast.final_top_k < balanced.final_top_k < precise.final_top_k
    assert not fast.rerank


def test_overriding_one_field_leaves_the_others_alone() -> None:
    """Overrides start from the named profile, so a partial override is not a partial config."""
    base = PROFILES[RetrievalProfile.BALANCED]
    adjusted = profile_config(RetrievalProfile.BALANCED, {"final_top_k": 8})
    assert adjusted.final_top_k == 8
    assert adjusted.context_tokens == base.context_tokens
    assert adjusted.candidates == base.candidates
