"""Durable request records: path selection, rotation and startup refusals."""

from __future__ import annotations

import json
import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from pydantic import ValidationError

from manicule.app import request_logging
from manicule.config.settings import LoggingSettings, Settings
from manicule.core.errors import ConfigError
from tests.api.support import backend_with_a_document, client_for


@pytest.fixture(autouse=True)
def isolated_request_logger() -> Iterator[None]:
    """Give each case a fresh logger and close every file it opens."""
    logger = request_logging.logger
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    _close_handlers()
    logger.handlers = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def _close_handlers() -> None:
    logger = request_logging.logger
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _record(*, status: int = 200) -> None:
    request_logging.record_request(
        surface="http",
        operation="document_list",
        outcome="ok",
        started=perf_counter(),
        method="GET",
        status=status,
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_production_http_request_is_persisted(tmp_path: Path) -> None:
    backend, _ = backend_with_a_document(data_dir=tmp_path)
    request_logging.configure_request_logging(backend.settings)

    with client_for(backend) as client:
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    (event,) = _lines(tmp_path / "logs" / "requests.jsonl")
    assert event["surface"] == "http"
    assert event["operation"] == "document_list"
    assert event["outcome"] == "ok"


def test_restarting_appends_to_the_existing_file(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    path = tmp_path / "logs" / "requests.jsonl"

    request_logging.configure_request_logging(settings)
    _record(status=201)
    _close_handlers()
    request_logging.configure_request_logging(settings)
    _record(status=202)

    assert [event["status"] for event in _lines(path)] == [201, 202]


def test_rollover_is_bounded_and_every_retained_line_is_json(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        logging=LoggingSettings(max_bytes=220, backup_count=2),
    )
    request_logging.configure_request_logging(settings)
    path = tmp_path / "logs" / "requests.jsonl"

    for status in range(200, 212):
        _record(status=status)

    backups = sorted(path.parent.glob("requests.jsonl.*"))
    retained = [event for candidate in [*backups, path] for event in _lines(candidate)]
    assert [candidate.name for candidate in backups] == ["requests.jsonl.1", "requests.jsonl.2"]
    assert len(retained) < 12
    assert _lines(path)[-1]["status"] == 211


def test_disabled_logging_creates_neither_handlers_nor_directories(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        logging=LoggingSettings(requests=False, file=Path("private/requests.jsonl")),
    )

    request_logging.configure_request_logging(settings)

    assert request_logging.logger.handlers == []
    assert not (tmp_path / "private").exists()


def test_relative_and_absolute_custom_paths_are_honored(tmp_path: Path) -> None:
    relative = Settings(
        data_dir=tmp_path / "data",
        logging=LoggingSettings(file=Path("custom/access.jsonl")),
    )
    request_logging.configure_request_logging(relative)
    _record()
    assert (tmp_path / "data" / "custom" / "access.jsonl").is_file()

    _close_handlers()
    absolute_path = tmp_path / "elsewhere" / "access.jsonl"
    absolute = Settings(
        data_dir=tmp_path / "ignored",
        logging=LoggingSettings(file=absolute_path),
    )
    request_logging.configure_request_logging(absolute)
    _record()
    assert absolute_path.is_file()
    assert not (absolute.data_dir / "elsewhere").exists()


def test_the_environment_can_select_the_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "from-environment" / "requests.jsonl"
    monkeypatch.setenv("MANICULE_LOGGING__FILE", str(configured))
    settings = Settings(data_dir=tmp_path / "data")

    request_logging.configure_request_logging(settings)
    _record()

    assert settings.logging.file == configured
    assert configured.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_active_and_rotated_files_are_private(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "requests.jsonl"
    path.parent.mkdir()
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    previous = path.with_name("requests.jsonl.1")
    previous.write_text("", encoding="utf-8")
    previous.chmod(0o644)
    settings = Settings(
        data_dir=tmp_path,
        logging=LoggingSettings(max_bytes=220, backup_count=2),
    )
    request_logging.configure_request_logging(settings)

    assert stat.S_IMODE(previous.stat().st_mode) == 0o600

    for status in range(200, 206):
        _record(status=status)

    files = [path, *path.parent.glob("requests.jsonl.*")]
    assert len(files) == 3
    assert {stat.S_IMODE(candidate.stat().st_mode) for candidate in files} == {0o600}


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"max_bytes": 0}, "max_bytes"),
        ({"backup_count": 0}, "backup_count"),
    ],
)
def test_rotation_limits_must_be_positive(values: dict[str, int], field: str) -> None:
    with pytest.raises(ValidationError) as caught:
        LoggingSettings.model_validate(values)
    assert field in str(caught.value)


def test_an_unwritable_shape_refuses_startup_before_installing_handlers(tmp_path: Path) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path,
        logging=LoggingSettings(file=Path("not-a-directory/requests.jsonl")),
    )

    with pytest.raises(ConfigError, match="Cannot open request log"):
        request_logging.configure_request_logging(settings)

    assert request_logging.logger.handlers == []
    assert parent.read_text(encoding="utf-8") == "occupied"


def test_an_explicit_handler_prevents_builtin_file_setup(tmp_path: Path) -> None:
    handler = logging.NullHandler()
    request_logging.logger.addHandler(handler)
    request_logging.logger.setLevel(logging.WARNING)
    request_logging.logger.propagate = True

    request_logging.configure_request_logging(Settings(data_dir=tmp_path))

    assert request_logging.logger.handlers == [handler]
    assert request_logging.logger.level == logging.WARNING
    assert request_logging.logger.propagate is True
    assert not (tmp_path / "logs").exists()
