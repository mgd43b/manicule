"""The retry arithmetic the Confluence and Gmail clients share."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from manicule.connectors.retry import BACKOFF_BASE_SECONDS, backoff, retry_after


def test_backoff_doubles_from_its_base() -> None:
    assert [backoff(attempt) for attempt in range(3)] == [
        BACKOFF_BASE_SECONDS,
        BACKOFF_BASE_SECONDS * 2,
        BACKOFF_BASE_SECONDS * 4,
    ]


def test_retry_after_reads_seconds() -> None:
    assert retry_after({"retry-after": " 7 "}) == 7.0


def test_retry_after_reads_an_http_date() -> None:
    when = datetime.now(tz=UTC) + timedelta(seconds=30)
    asked = retry_after({"retry-after": format_datetime(when, usegmt=True)})

    assert asked is not None
    assert 25.0 <= asked <= 30.0


def test_a_date_in_the_past_is_a_zero_second_wait() -> None:
    when = datetime.now(tz=UTC) - timedelta(seconds=30)

    assert retry_after({"retry-after": format_datetime(when, usegmt=True)}) == 0.0
    assert retry_after({"retry-after": "-5"}) == 0.0


@pytest.mark.parametrize("value", ["", "   ", "soon", "nan", "inf", "-inf"])
def test_a_header_that_is_not_a_finite_pause_is_no_instruction(value: str) -> None:
    """``nan`` would otherwise pass for zero seconds and retry straight into the limit."""
    assert retry_after({"retry-after": value}) is None


def test_no_header_is_no_instruction() -> None:
    assert retry_after({}) is None
