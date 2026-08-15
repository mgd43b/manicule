"""The envelope has one enforced shape, including its sole partial-result exception."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manicule.app import results as r

ERROR = r.ErrorInfo(type="CursorExpiredError", message="the search cursor expired")


@pytest.mark.parametrize(
    "arguments",
    [
        {"ok": True, "data": None, "error": None},
        {"ok": True, "data": {"value": 1}, "error": ERROR},
        {"ok": False, "data": None, "error": None},
        {
            "ok": False,
            "data": r.IngestReport(connector="synthetic-wiki").model_dump(mode="json"),
            "error": ERROR,
        },
        {
            "ok": False,
            "data": r.IngestReport(
                connector="synthetic-wiki",
                outcome="incomplete",
                retry_required=False,
                incomplete_reason=ERROR,
            ).model_dump(mode="json"),
            "error": ERROR,
        },
        {
            "ok": False,
            "data": r.IngestReport(
                connector="synthetic-wiki",
                outcome="incomplete",
                retry_required=True,
                incomplete_reason=r.ErrorInfo(type="OSError", message="the store stopped"),
            ).model_dump(mode="json"),
            "error": ERROR,
        },
    ],
)
def test_invalid_envelope_shapes_are_refused(arguments: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        r.Envelope(op="connector_sync", workspace="default", **arguments)  # type: ignore[arg-type]


def test_the_three_supported_envelope_shapes_validate() -> None:
    success = r.succeeded("stats", "default", r.Stats(documents=0, chunks=0))
    refusal = r.failed("stats", "default", r.ErrorInfo(type="OSError", message="offline"))
    incomplete = r.failed(
        "connector_sync",
        "default",
        ERROR,
        payload=r.IngestReport(
            connector="synthetic-wiki",
            outcome="incomplete",
            retry_required=True,
            incomplete_reason=ERROR,
        ),
    )

    assert success.ok
    assert success.data is not None
    assert success.error is None
    assert not refusal.ok
    assert refusal.data is None
    assert refusal.error is not None
    assert not incomplete.ok
    assert incomplete.data is not None
    assert incomplete.error == ERROR


def test_a_non_ingest_operation_cannot_carry_an_incomplete_ingest_payload() -> None:
    partial = r.IngestReport(
        connector="synthetic-wiki",
        outcome="incomplete",
        retry_required=True,
        incomplete_reason=ERROR,
    )
    with pytest.raises(ValidationError):
        r.Envelope(
            op="stats",
            workspace="default",
            ok=False,
            data=partial.model_dump(mode="json"),
            error=ERROR,
        )


def test_failed_cannot_smuggle_an_unrelated_partial_payload_into_the_envelope() -> None:
    with pytest.raises(ValidationError):
        r.failed(
            "stats",
            "default",
            r.ErrorInfo(type="OSError", message="offline"),
            payload=r.Stats(documents=1, chunks=2),  # type: ignore[arg-type]
        )
