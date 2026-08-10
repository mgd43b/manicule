"""Configuration: layering, credential resolution, redaction and policy gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from manicule.config.loader import load_settings, save_settings
from manicule.config.profiles import PROFILES, profile_config
from manicule.config.providers import (
    ProviderSettings,
    env_var_names,
    is_local,
    resolve_provider_keys,
)
from manicule.config.settings import AuthMode, Mode, Settings, Theme
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
    assert is_local("ollama")
    resolved = resolve_provider_keys({}, required=frozenset({"ollama"}), environ={})
    assert resolved["ollama"].api_key is None
    assert resolved["ollama"].base_url == "http://localhost:11434"


def test_a_key_in_a_dotenv_file_is_found(manicule_environment: Path) -> None:
    """Provider variables are not prefixed, so the settings sources never see them."""
    (manicule_environment / ".env").write_text('OPENAI_API_KEY="sk-from-dotenv"\n')
    settings = Settings(llm={"provider": "openai"})  # pyright: ignore[reportArgumentType]
    key = settings.provider("openai").api_key
    assert key is not None
    assert key.get_secret_value() == "sk-from-dotenv"


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
