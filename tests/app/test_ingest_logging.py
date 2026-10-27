"""Ingest progress lines reach standard error when a server starts, and not before."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from manicule.app import ingest_logging
from manicule.app.ingest_logging import INGEST_LOGGER, configure_ingest_logging
from manicule.ingest import pipeline
from tests.api.support import backend_with_a_document


@pytest.fixture(autouse=True)
def isolated_ingest_logger() -> Iterator[logging.Logger]:
    """Give each case the logger as a fresh process has it: no handler, no level of its own."""
    logger = logging.getLogger(INGEST_LOGGER)
    original = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield logger
    logger.handlers, level, logger.propagate = original[0], original[1], original[2]
    logger.setLevel(level)


def _stream_of(logger: logging.Logger) -> StringIO:
    (handler,) = logger.handlers
    captured = StringIO()
    cast("logging.StreamHandler[StringIO]", handler).setStream(captured)
    return captured


def test_an_info_line_from_the_pipeline_is_dropped_until_a_server_configures_it() -> None:
    """The bug: nothing but ``manicule.requests`` ever had a handler, so these went nowhere."""
    assert not pipeline._LOG.isEnabledFor(logging.INFO)  # pyright: ignore[reportPrivateUsage]


def test_configuring_ingest_logging_writes_the_pipelines_info_lines_once(
    isolated_ingest_logger: logging.Logger,
) -> None:
    configure_ingest_logging()
    captured = _stream_of(isolated_ingest_logger)

    pipeline._LOG.info("ingest stage timings for example over 60s: embed 1.0/2")  # pyright: ignore[reportPrivateUsage]

    written = captured.getvalue()
    assert written.count("ingest stage timings for example") == 1
    assert "INFO manicule.ingest.pipeline:" in written


def test_configuring_twice_keeps_one_handler_and_an_explicit_one_wins(
    isolated_ingest_logger: logging.Logger,
) -> None:
    configure_ingest_logging()
    configure_ingest_logging()
    assert len(isolated_ingest_logger.handlers) == 1

    isolated_ingest_logger.handlers = []
    own = logging.NullHandler()
    isolated_ingest_logger.addHandler(own)
    configure_ingest_logging()

    assert isolated_ingest_logger.handlers == [own], "an embedder's handler is left alone"


def test_starting_the_http_server_configures_ingest_logging(
    isolated_ingest_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve = importlib.import_module("manicule.api.serve")  # the package re-exports a function

    backend, _ = backend_with_a_document()
    address = SimpleNamespace(host="127.0.0.1", port=1)

    def application(*_args: object, **_kwargs: object) -> tuple[object, SimpleNamespace]:
        return object(), address

    monkeypatch.setattr(serve, "application", application)

    serve.server_for(cast("Any", backend))

    assert isolated_ingest_logger.handlers
    assert pipeline._LOG.isEnabledFor(logging.INFO)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("module", ["api/serve.py", "mcp/serve.py"])
def test_each_transport_configures_ingest_logging_before_it_builds_its_server(module: str) -> None:
    """Both transports start the ingest pipeline, so both have to install the handler."""
    source = (Path(pipeline.__file__).parents[1] / module).read_text(encoding="utf-8")

    assert "configure_ingest_logging()" in source
    assert source.index("configure_ingest_logging()") < source.index("configure_request_logging(")


def test_the_module_imports_without_touching_process_logging(
    isolated_ingest_logger: logging.Logger,
) -> None:
    """Importing, or building a service, must not change a host process's logging."""
    assert ingest_logging.INGEST_LOGGER == INGEST_LOGGER
    assert isolated_ingest_logger.handlers == []
