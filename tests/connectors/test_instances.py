"""Named connector instances: their own options, their own object, their own identity.

A configured source is ``[connectors.<name>]``, and the name is the user's. The type it names is
an implementation detail two sources are entitled to share. Everything here is a property that
was false before: options reached no connector, both instances of a type were one object, and a
document recorded the implementation it arrived through rather than the source it came from.

Fixtures are synthetic throughout — ``https://docs.example.test/`` and temporary roots.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from manicule.config.settings import ConnectorSettings, Settings
from manicule.container import Container, keys
from manicule.core.errors import ConfigError
from manicule.core.ids import document_id
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark
from manicule.plugins.registry import BuildContext, ComponentRegistry


class RootConfig(BaseModel):
    """A connector configuration with a root and a second field, so merging is observable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(default="")
    include_hidden: bool = Field(default=False)


class Rooted:
    """A connector that remembers what it was built with."""

    def __init__(self, config: RootConfig, name: str) -> None:
        self.name = name
        self.config = config

    @property
    def watermark(self) -> Watermark | None:
        return None

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        del watermark
        yield DiscoveredDoc(ref=DocRef(source_id="1001", uri="https://docs.example.test/1001"))

    async def fetch(self, ref: DocRef) -> Any:
        raise NotImplementedError

    async def reconcile(self) -> AsyncIterator[SourceId]:
        yield "1001"


def _build(context: BuildContext) -> Rooted:
    assert isinstance(context.config, RootConfig)
    return Rooted(context.config, name=context.instance)


def _container(settings: Settings, **connectors: ConnectorSettings) -> Container:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.CONNECTOR.named("rooted"), _build, config_model=RootConfig)
    return Container(settings.model_copy(update={"connectors": dict(connectors)}), registry)


def _rooted(root: str, **options: Any) -> ConnectorSettings:
    return ConnectorSettings(type="rooted", options={"root": root, **options})


async def _built(container: Container, instance: str) -> Rooted:
    """``instance``'s connector, narrowed to the fake so its build inputs are readable.

    ``Container.connector`` is typed to the protocol, which has neither ``config`` nor ``root``
    — that is the protocol doing its job. The narrowing belongs here rather than in each test.
    """
    connector = await container.connector(instance)
    assert isinstance(connector, Rooted)
    return connector


# --- options reach the instance -----------------------------------------------------------------


async def test_one_named_instance_receives_its_own_options(settings: Settings) -> None:
    """The acceptance criterion the bug was reported against.

    Before the fix this raised, because the only configuration a factory was ever handed came
    from ``plugins.config."connector.<type>"`` and the instance's own ``options`` were read
    nowhere in the construction path.
    """
    container = _container(settings, handbook=_rooted("/corpus/handbook"))

    connector = await _built(container, "handbook")

    assert connector.config.root == "/corpus/handbook"


async def test_no_global_plugin_configuration_is_required(settings: Settings) -> None:
    """Nothing had to be duplicated into ``plugins.config`` to get here."""
    container = _container(settings, handbook=_rooted("/corpus/handbook"))
    assert container.settings.component_config("connector", "rooted") == {}

    assert (await _built(container, "handbook")).config.root == "/corpus/handbook"


async def test_two_instances_of_one_type_receive_different_options(settings: Settings) -> None:
    container = _container(
        settings,
        handbook=_rooted("/corpus/handbook"),
        runbooks=_rooted("/corpus/runbooks"),
    )

    handbook = await _built(container, "handbook")
    runbooks = await _built(container, "runbooks")

    assert handbook.config.root == "/corpus/handbook"
    assert runbooks.config.root == "/corpus/runbooks"


async def test_two_instances_of_one_type_are_two_objects(settings: Settings) -> None:
    """Caching is per instance, not per type.

    One object serving two configured sources cannot hold two roots, so the second instance
    silently inherited the first's — and, because ``Connector.name`` becomes the ``source`` half
    of a document's identity, filed its documents under the first's name.
    """
    container = _container(
        settings,
        handbook=_rooted("/corpus/handbook"),
        runbooks=_rooted("/corpus/runbooks"),
    )

    assert await container.connector("handbook") is not await container.connector("runbooks")


async def test_one_instance_resolves_to_one_object(settings: Settings) -> None:
    """Per-instance, not per-call: asking twice must not build twice.

    A connector holds a watermark across a run, and two objects for one source would each
    advance their own — so the memoisation is load-bearing rather than an optimisation.
    """
    container = _container(settings, handbook=_rooted("/corpus/handbook"))

    assert await container.connector("handbook") is await container.connector("handbook")


# --- the two layers -----------------------------------------------------------------------------


async def test_a_global_component_configuration_still_configures_an_instance(
    settings: Settings,
) -> None:
    """Existing single-instance configurations keep working, untouched.

    Before named instances carried options, ``plugins.config."connector.<type>"`` was the only
    place settings could go. Every corpus configured that way must keep syncing without its
    author editing anything, so the global slot remains a source of defaults rather than being
    replaced by the new one.
    """
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"connector.rooted": {"root": "/corpus/global"}}}
            )
        }
    )
    container = _container(configured, docs=ConnectorSettings(type="rooted"))

    assert (await _built(container, "docs")).config.root == "/corpus/global"


async def test_instance_options_win_over_the_global_default(settings: Settings) -> None:
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"connector.rooted": {"root": "/corpus/global"}}}
            )
        }
    )
    container = _container(configured, handbook=_rooted("/corpus/handbook"))

    assert (await _built(container, "handbook")).config.root == "/corpus/handbook"


async def test_a_global_default_survives_an_instance_that_overrode_something_else(
    settings: Settings,
) -> None:
    """Merged field by field, not replaced wholesale.

    An instance naming only its own root must not silently lose a global setting its author can
    still see in the file — a replacement would turn ``include_hidden`` off without saying so.
    """
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"connector.rooted": {"include_hidden": True}}}
            )
        }
    )
    container = _container(configured, handbook=_rooted("/corpus/handbook"))

    connector = await _built(container, "handbook")
    assert connector.config.root == "/corpus/handbook"
    assert connector.config.include_hidden is True


# --- misconfiguration names the place it was written ---------------------------------------------


async def test_a_misspelled_instance_option_names_the_instance(settings: Settings) -> None:
    """Not the global slot, which the author never wrote in.

    The whole reason unknown keys are refused rather than ignored is that the likely unknown key
    is a misspelling of a real one; an error pointing at the wrong file spends that loudness on
    sending the author somewhere they have nothing to fix.
    """
    container = _container(
        settings, handbook=ConnectorSettings(type="rooted", options={"rooot": "/corpus/x"})
    )

    with pytest.raises(ConfigError) as caught:
        await container.connector("handbook")

    message = str(caught.value)
    assert "connectors['handbook'].options" in message
    assert "rooot" in message
    assert "root" in message


async def test_an_invalid_instance_option_names_the_instance(settings: Settings) -> None:
    container = _container(
        settings,
        handbook=ConnectorSettings(type="rooted", options={"include_hidden": "yes please"}),
    )

    with pytest.raises(ConfigError) as caught:
        await container.connector("handbook")

    assert "connectors['handbook'].options" in str(caught.value)


async def test_an_error_from_the_merged_case_names_both_places(settings: Settings) -> None:
    """The value came from two files, and blaming one would be wrong half the time."""
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"connector.rooted": {"include_hidden": True}}}
            )
        }
    )
    container = _container(
        configured, handbook=ConnectorSettings(type="rooted", options={"rooot": "/corpus/x"})
    )

    with pytest.raises(ConfigError) as caught:
        await container.connector("handbook")

    message = str(caught.value)
    assert "connectors['handbook'].options" in message
    assert "plugins.config['connector.rooted']" in message


# --- identity ------------------------------------------------------------------------------------


async def test_a_connector_is_named_by_its_instance_not_its_type(settings: Settings) -> None:
    """``Connector.name`` is the configured source name.

    It is not a label. ``ingest/pipeline.py`` reads it as ``source`` for every document, as the
    watermark key and as the run-metadata key, so a connector named after its implementation
    files two sources' documents under one identity.
    """
    container = _container(
        settings,
        handbook=_rooted("/corpus/handbook"),
        runbooks=_rooted("/corpus/runbooks"),
    )

    assert (await container.connector("handbook")).name == "handbook"
    assert (await container.connector("runbooks")).name == "runbooks"


async def test_the_built_in_factories_name_their_connector_after_the_instance(
    settings: Settings, tmp_path: Any
) -> None:
    """The fakes above prove the plumbing; this proves the connectors people actually run.

    Separate on purpose. Every other identity test here builds its own factory, which is free
    to do the right thing whatever ``connectors/plugin.py`` does — so passing them all while
    the shipped factories still hardcoded ``SNAPSHOT_CONNECTOR_NAME`` was entirely possible,
    and that gap is precisely how the original defect stayed invisible.
    """
    from manicule.connectors.confluence_snapshot import (  # noqa: PLC0415
        ConfluenceSnapshotConnector,
    )
    from manicule.container import build_container  # noqa: PLC0415

    handbook = tmp_path / "handbook"
    runbooks = tmp_path / "runbooks"
    handbook.mkdir()
    runbooks.mkdir()
    configured = settings.model_copy(
        update={
            "connectors": {
                "team-handbook": ConnectorSettings(
                    type="confluence-snapshot", options={"root": str(handbook)}
                ),
                "team-runbooks": ConnectorSettings(
                    type="confluence-snapshot", options={"root": str(runbooks)}
                ),
            }
        }
    )
    container = build_container(configured)

    first = await container.connector("team-handbook")
    second = await container.connector("team-runbooks")
    assert isinstance(first, ConfluenceSnapshotConnector)
    assert isinstance(second, ConfluenceSnapshotConnector)

    assert first.name == "team-handbook"
    assert second.name == "team-runbooks"
    assert first.root == handbook.resolve()
    assert second.root == runbooks.resolve()


async def test_a_single_instance_named_after_its_type_keeps_the_name_it_had(
    settings: Settings, tmp_path: Any
) -> None:
    """The compatibility case, stated as a test because it bounds the migration.

    An existing ``[connectors.filesystem] type = "filesystem"`` records ``source =
    "filesystem"`` before this change and after it, so its documents keep their ids and its
    watermark keeps its key. Only an instance whose *name differs from its type* moves — which
    is what makes the affected population "configurations that hit this bug", not everyone.
    """
    from manicule.container import build_container  # noqa: PLC0415

    configured = settings.model_copy(
        update={
            "connectors": {
                "filesystem": ConnectorSettings(type="filesystem", options={"root": str(tmp_path)})
            }
        }
    )
    container = build_container(configured)

    assert (await container.connector("filesystem")).name == "filesystem"


async def test_a_connector_with_no_root_names_the_instance_to_fix(
    settings: Settings,
) -> None:
    """The message the bug report quoted, corrected.

    It used to instruct the reader to set ``plugins.config."connector.confluence-snapshot".root``
    — a *global* setting — when what they had written was a per-instance one. Following it made
    sync work and gave every instance of the type the same root, which is how the second half of
    this bug was reached.
    """
    from manicule.container import build_container  # noqa: PLC0415

    configured = settings.model_copy(
        update={"connectors": {"team-handbook": ConnectorSettings(type="confluence-snapshot")}}
    )
    container = build_container(configured)

    with pytest.raises(ConfigError) as caught:
        await container.connector("team-handbook")

    message = str(caught.value)
    assert "connectors.team-handbook.options" in message
    assert "plugins.config" not in message


async def test_two_instances_of_one_type_do_not_collide_on_document_identity(
    settings: Settings,
) -> None:
    """The failure that makes this more than a naming complaint.

    ``source_id`` for a mirrored wiki page is the page id, and ``documents`` is UNIQUE on
    ``(workspace_id, source, source_id)``. Two instances mirroring two different deployments
    both hold a page ``1001``; with ``source`` holding the *type* they are one row, and the
    second sync overwrites the first's document with no error. It is ``document_id``'s own
    ``workspace_id`` argument one level down.
    """
    container = _container(
        settings,
        handbook=_rooted("/corpus/handbook"),
        runbooks=_rooted("/corpus/runbooks"),
    )
    handbook = await _built(container, "handbook")
    runbooks = await _built(container, "runbooks")

    assert document_id("w", handbook.name, "1001") != document_id("w", runbooks.name, "1001")
