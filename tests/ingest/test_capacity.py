"""Bounds and the aggregate-only vocabulary used when one refuses admission."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import override

import pytest

from manicule.ingest.capacity import (
    CapacityDiagnostic,
    CapacityRefusedError,
    CapacityResource,
    translate_storage_capacity_errors,
)


class SecretInt(int):
    """An integer whose display methods try to smuggle source content across the boundary."""

    leaky_text = "private-title-at-https://example.test/?token=fake-secret-cinder"

    @override
    def __repr__(self) -> str:
        return self.leaky_text

    @override
    def __str__(self) -> str:
        return self.leaky_text

    @override
    def __format__(self, format_spec: str) -> str:
        del format_spec
        return self.leaky_text


def assert_secret_absent(error: BaseException) -> None:
    rendered = f"{error!s}\n{error!r}\n{error.args!r}\n{vars(error)!r}"
    assert SecretInt.leaky_text not in rendered
    assert "fake-secret-cinder" not in rendered


def test_a_refusal_reports_only_closed_aggregate_facts() -> None:
    diagnostic = CapacityDiagnostic(
        resource=CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES,
        limit=1_000,
        used=900,
        requested=200,
    )

    error = CapacityRefusedError(diagnostic)

    assert diagnostic.over_by == 100
    assert diagnostic.as_metadata() == {
        "resource": "acquired_blob_backlog_bytes",
        "limit": 1_000,
        "used": 900,
        "requested": 200,
    }
    assert str(error) == (
        "acquired_blob_backlog_bytes capacity refused: "
        "used=900, requested=200, limit=1000, over_by=100"
    )


def test_a_capacity_diagnostic_cannot_grow_source_shaped_fields() -> None:
    """Closed models make the privacy rule an API constraint rather than a call-site habit."""
    private_title = "private roadmap cinder"
    private_url = "https://example.test/page?token=fake-secret-cinder"
    with pytest.raises(TypeError, match="unexpected keyword") as caught:
        CapacityDiagnostic(
            resource=CapacityResource.JOURNAL_RECORDS,
            limit=1,
            used=1,
            requested=1,
            title=private_title,  # type: ignore[call-arg]
            source_url=private_url,  # type: ignore[call-arg]
        )

    # Unlike Pydantic's ValidationError, TypeError has no errors()/json() representation that
    # retains the rejected mapping. Its only structured value is args, which is safe too.
    rendered = f"{caught.value!s}\n{caught.value!r}\n{caught.value.args!r}\n{vars(caught.value)!r}"
    assert private_title not in rendered
    assert private_url not in rendered
    assert "fake-secret-cinder" not in rendered
    assert not hasattr(caught.value, "errors")
    assert not hasattr(caught.value, "json")


def test_an_error_normalizes_a_subclass_to_the_allowlisted_base_payload() -> None:
    """A declared subclass field is not an ``extra`` and must still not cross the boundary."""

    @dataclass(frozen=True, slots=True)
    class SourceShapedDiagnostic(CapacityDiagnostic):
        source_url: str
        title: str

    private_title = "private acquisition cinder"
    private_url = "https://example.test/fetch?credential=fake-secret-cinder"
    supplied = SourceShapedDiagnostic(
        resource=CapacityResource.JOURNAL_METADATA_BYTES,
        limit=100,
        used=90,
        requested=20,
        source_url=private_url,
        title=private_title,
    )

    error = CapacityRefusedError(supplied)
    dumped = error.diagnostic.as_metadata()
    rendered = str(error)

    assert type(error.diagnostic) is CapacityDiagnostic
    assert dumped == {
        "resource": "journal_metadata_bytes",
        "limit": 100,
        "used": 90,
        "requested": 20,
    }
    assert "source_url" not in dumped
    assert "title" not in dumped
    assert private_title not in rendered
    assert private_url not in rendered
    assert "fake-secret-cinder" not in rendered


def test_a_capacity_diagnostic_must_describe_an_actual_refusal() -> None:
    with pytest.raises(ValueError, match=r"used \+ requested"):
        CapacityDiagnostic(
            resource=CapacityResource.JOURNAL_RECORDS,
            limit=10,
            used=8,
            requested=2,
        )


@pytest.mark.parametrize("field", ["limit", "used", "requested"])
def test_integer_subclasses_are_rejected_without_rendering_them(field: str) -> None:
    values: dict[str, int | CapacityResource] = {
        "resource": CapacityResource.JOURNAL_RECORDS,
        "limit": 1,
        "used": 1,
        "requested": 1,
    }
    values[field] = SecretInt(2)

    with pytest.raises(TypeError, match=f"{field} must be an integer") as caught:
        CapacityDiagnostic(**values)  # type: ignore[arg-type]

    assert_secret_absent(caught.value)


def test_metadata_and_error_boundaries_recheck_exact_integer_types() -> None:
    """Even a value object corrupted around ``frozen=True`` cannot render a scalar subclass."""
    diagnostic = CapacityDiagnostic(
        resource=CapacityResource.JOURNAL_RECORDS,
        limit=1,
        used=1,
        requested=1,
    )
    object.__setattr__(diagnostic, "used", SecretInt(1))

    with pytest.raises(TypeError, match="used must be an integer") as metadata_error:
        diagnostic.as_metadata()
    with pytest.raises(TypeError, match="used must be an integer") as refusal_error:
        CapacityRefusedError(diagnostic)

    assert_secret_absent(metadata_error.value)
    assert_secret_absent(refusal_error.value)


async def test_sqlite_full_is_mapped_without_retaining_database_error_detail() -> None:
    private_detail = "database full at /private/customer?token=fake-secret-cinder"
    private_filename = "/private/customer/private-title-cinder.db"

    @translate_storage_capacity_errors
    async def fail() -> None:
        error = sqlite3.OperationalError(private_detail, private_filename)
        error.sqlite_errorcode = sqlite3.SQLITE_FULL
        raise error

    with pytest.raises(CapacityRefusedError) as caught:
        await fail()

    rendered = f"{caught.value!s}\n{caught.value!r}\n{caught.value.args!r}"
    assert private_detail not in rendered
    assert private_filename not in rendered
    assert "fake-secret-cinder" not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.diagnostic.resource is CapacityResource.DISK_HEADROOM_BYTES
