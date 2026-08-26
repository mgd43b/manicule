"""Loading and writing the config file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import tomli_w
from pydantic import ValidationError

from manicule.config.settings import Settings, config_file, secret_setting
from manicule.core.errors import ConfigError


def load_settings(**overrides: Any) -> Settings:  # noqa: ANN401 - mirrors Settings' own fields
    """Build settings from every source, with ``overrides`` taking precedence.

    Raises:
        ConfigError: The configuration is malformed. The message names every offending
            field, because a config file usually has more than one thing wrong with it.
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        lines = [
            f"  {'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
        ]
        joined = "\n".join(lines)
        msg = f"invalid configuration in {config_file()}:\n{joined}"
        raise ConfigError(msg) from exc


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Write ``settings`` to a config file and return where it went.

    Secrets are omitted. Credentials belong in the environment, where they are not copied
    into backups, config exports or version control by accident; writing them here would put
    them in all three.

    The file is rewritten rather than patched, so hand-written comments do not survive.
    """
    destination = path or config_file()
    dumped: Any = settings.model_dump(mode="json", exclude_defaults=True)
    body = _strip(dumped)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(tomli_w.dumps(body), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def _strip(value: Any, path: tuple[str, ...] = ()) -> Any:  # noqa: ANN401 - recursive over JSON
    """Drop nulls and secrets; TOML has no null, and secrets do not belong on disk.

    Uses the same predicate as the redacted view, so that what is hidden when configuration is
    displayed and what is omitted when it is written can never disagree. That predicate is now
    resolved against the model, so the path has to be carried down rather than judged one
    segment at a time: `secret_setting` answers about `llm.token_safety_factor`, not about
    `token_safety_factor`.
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in cast("dict[str, Any]", value).items():
            if secret_setting((*path, key)):
                continue
            stripped = _strip(item, (*path, key))
            if stripped is not None:
                cleaned[key] = stripped
        return cleaned
    if isinstance(value, list):
        return [_strip(item, path) for item in cast("list[Any]", value) if item is not None]
    return value


__all__ = ["load_settings", "save_settings"]
