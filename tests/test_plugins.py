"""Plugin discovery, compatibility checking and registration."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import EntryPoint

import pytest
from pydantic import ValidationError

from manicule.container import keys
from manicule.core.errors import (
    DuplicateComponentError,
    IncompatiblePluginError,
    PluginDependencyError,
    PluginLoadError,
    UnknownComponentError,
)
from manicule.core.version import CORE_VERSION
from manicule.plugins import (
    ENTRY_POINT_GROUP,
    ComponentKind,
    ComponentRegistry,
    PluginManifest,
    check_core_version,
    describe,
    discover,
    installed_entry_points,
    load_order,
    require_compatible,
)
from manicule.plugins.manifest import Plugin

# --- the real, installed plugin ------------------------------------------------------------


def test_an_installed_distribution_is_found_without_being_configured() -> None:
    """Discovery is discovery. Nothing named this plugin in a config file first."""
    names = [entry_point.name for entry_point in installed_entry_points()]
    assert "example" in names


def test_the_example_plugin_registers_through_the_public_path() -> None:
    """The extension mechanism manicule ships is the one manicule's own tests depend on."""
    found = discover()

    assert "example" in found.names
    assert found.registry.has(ComponentKind.PARSER, "example")
    assert found.registry.has(ComponentKind.MIDDLEWARE, "trim")
    assert found.registry.has(ComponentKind.RETRIEVAL_STAGE, "passthrough")

    record = found.registry.record(keys.PARSER.named("example"))
    assert record.plugin == "example"
    assert record.config_model is not None


def test_a_disabled_plugin_is_not_loaded() -> None:
    found = discover(disabled=frozenset({"example"}))
    assert "example" not in found.names
    assert found.disabled == ("example",)
    assert not found.registry.has(ComponentKind.PARSER, "example")


def test_an_allow_list_filters_what_was_discovered() -> None:
    assert discover(enabled=frozenset({"nothing-by-this-name"})).names == ()
    assert discover(enabled=frozenset({"example"})).names == ("example",)


def test_describe_names_the_plugins_and_their_components() -> None:
    lines = "\n".join(describe(discover()))
    assert "example" in lines
    assert "parser:example" in lines


# --- compatibility ---------------------------------------------------------------------------


def manifest(**overrides: object) -> PluginManifest:
    base: dict[str, object] = {
        "name": "thing",
        "version": "1.0.0",
        "core_version": ">=0.1,<0.2",
    }
    return PluginManifest.model_validate({**base, **overrides})


def test_a_plugin_written_for_another_core_is_refused_loudly() -> None:
    """Not a warning. The alternative is an attribute error somewhere unrelated, much later."""
    with pytest.raises(IncompatiblePluginError, match="requires manicule >=9"):
        require_compatible(manifest(core_version=">=9.0"), CORE_VERSION, frozenset())


def test_the_message_says_what_was_asked_for_and_what_is_running() -> None:
    problem = check_core_version(manifest(core_version=">=2,<3"), "0.1.0")
    assert problem is not None
    assert ">=2,<3" in problem
    assert "0.1.0" in problem


def test_a_malformed_version_range_is_reported_as_such() -> None:
    problem = check_core_version(manifest(core_version="not a specifier"), "0.1.0")
    assert problem is not None
    assert "PEP 440" in problem


def test_the_running_version_comes_from_installed_metadata() -> None:
    """A literal baked into the source drifts the moment a release happens."""
    assert CORE_VERSION != "0.0.0.dev0"
    assert check_core_version(manifest(core_version=f"=={CORE_VERSION}"), CORE_VERSION) is None


def test_a_prerelease_core_can_still_run_plugins_written_for_the_range() -> None:
    """Specifiers exclude prereleases by default, which would strand every release candidate."""
    assert check_core_version(manifest(core_version=">=0.1,<0.3"), "0.2.0rc1") is None


def test_a_missing_requirement_is_an_error() -> None:
    with pytest.raises(PluginDependencyError, match="not installed"):
        require_compatible(manifest(requires=("absent",)), CORE_VERSION, frozenset({"thing"}))


def test_a_conflict_is_an_error() -> None:
    with pytest.raises(PluginDependencyError, match="conflicts"):
        require_compatible(
            manifest(conflicts=("other",)), CORE_VERSION, frozenset({"thing", "other"})
        )


def test_requirements_are_checked_against_everything_installed() -> None:
    """Not only against what has already registered, so declaration order cannot decide it."""
    require_compatible(manifest(requires=("later",)), CORE_VERSION, frozenset({"thing", "later"}))


def test_load_order_puts_requirements_first() -> None:
    order = load_order(
        [
            manifest(name="c", requires=("a", "b")),
            manifest(name="b", requires=("a",)),
            manifest(name="a"),
        ]
    )
    assert order.index("a") < order.index("b") < order.index("c")


def test_load_order_is_the_same_on_every_machine() -> None:
    """A reproducible startup must stay reproducible, so ties break alphabetically."""
    manifests = [manifest(name=name) for name in ("zulu", "alpha", "mike")]
    assert load_order(manifests) == ["alpha", "mike", "zulu"]
    assert load_order(list(reversed(manifests))) == ["alpha", "mike", "zulu"]


def test_a_requirement_cycle_is_reported_as_a_cycle() -> None:
    with pytest.raises(PluginDependencyError, match="cycle"):
        load_order([manifest(name="a", requires=("b",)), manifest(name="b", requires=("a",))])


@pytest.mark.parametrize("field", ["requires", "conflicts"])
def test_a_manifest_cannot_name_the_same_plugin_twice(field: str) -> None:
    """A duplicate is always a mistake, and silently deduplicating hides which one."""
    with pytest.raises(ValidationError, match="duplicate"):
        manifest(**{field: ("same", "same")})


def test_a_requirement_on_something_absent_does_not_derail_ordering() -> None:
    """The missing requirement is reported by the dependency check, with a better message."""
    assert load_order([manifest(name="a", requires=("absent",))]) == ["a"]


def test_there_is_no_permissions_field() -> None:
    """Plugins run in-process with full privileges, and the manifest does not pretend otherwise.

    A declaration describing a boundary that does not exist is worse than none, because it
    gets believed.
    """
    assert "permissions" not in PluginManifest.model_fields


# --- registration -----------------------------------------------------------------------------


def test_two_plugins_cannot_claim_the_same_component() -> None:
    """Silent shadowing would make behaviour depend on installation order."""
    registry = ComponentRegistry()
    registry.bind("first").add(keys.PARSER.named("pdf"), lambda _: _never(), media_types={"a/b"})
    with pytest.raises(DuplicateComponentError, match="first"):
        registry.bind("second").add(
            keys.PARSER.named("pdf"), lambda _: _never(), media_types={"a/b"}
        )


def test_asking_for_something_absent_lists_what_is_present() -> None:
    registry = ComponentRegistry()
    registry.bind("p").add(
        keys.PARSER.named("markdown"), lambda _: _never(), media_types={"text/markdown"}
    )
    with pytest.raises(UnknownComponentError, match="markdown"):
        registry.record(keys.PARSER.named("pdf"))


def test_an_unnamed_component_cannot_be_registered() -> None:
    with pytest.raises(ValueError, match="unnamed"):
        ComponentRegistry().add(keys.PARSER, lambda _: _never(), media_types={"a/b"})


def test_a_parser_must_declare_what_it_handles() -> None:
    """Routing reads the declaration, so a parser without one could never be reached."""
    with pytest.raises(ValueError, match="must declare the media types"):
        ComponentRegistry().add(keys.PARSER.named("pdf"), lambda _: _never())


def _never() -> object:  # pragma: no cover - factories here are never invoked
    raise AssertionError("factory should not run")


# --- loading ------------------------------------------------------------------------------------


class _Plugin:
    def __init__(self, manifest_: PluginManifest) -> None:
        self.manifest = manifest_

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(keys.MIDDLEWARE.named(self.manifest.name), lambda _: object())


def _points(**targets: object) -> Iterable[EntryPoint]:
    """Entry points resolving to objects held in this module, without installing anything."""
    for name, target in targets.items():
        globals()[f"_target_{name}"] = target
        yield EntryPoint(name=name, value=f"{__name__}:_target_{name}", group=ENTRY_POINT_GROUP)


def test_an_entry_point_may_be_a_plugin_or_a_callable_returning_one() -> None:
    direct = _Plugin(manifest(name="direct"))
    found = discover(
        points=list(_points(direct=direct, factory=lambda: _Plugin(manifest(name="factory"))))
    )
    assert sorted(found.names) == ["direct", "factory"]


def test_an_entry_point_may_be_the_plugin_class_itself() -> None:
    """A class has both attributes, so it satisfies the protocol without being a plugin.

    Calling ``register`` on the class rather than an instance passes the registry as
    ``self``, which fails somewhere confusing. The class is instantiated instead.
    """

    class Klass:
        manifest = manifest(name="klass")

        def register(self, registry: ComponentRegistry) -> None:
            registry.add(keys.MIDDLEWARE.named("from-class"), lambda _: object())

    found = discover(points=list(_points(klass=Klass)))
    assert found.names == ("klass",)
    assert found.registry.has(ComponentKind.MIDDLEWARE, "from-class")


def test_a_manifest_can_be_looked_up_by_name() -> None:
    found = discover()
    assert found.manifest("example") is not None
    assert found.manifest("not-installed") is None


def test_an_entry_point_pointing_at_something_else_is_an_error() -> None:
    with pytest.raises(PluginLoadError, match="neither a plugin"):
        discover(points=list(_points(broken=42)))


def test_an_entry_point_that_cannot_be_imported_is_an_error() -> None:
    point = EntryPoint(name="gone", value="no_such_module:PLUGIN", group=ENTRY_POINT_GROUP)
    with pytest.raises(PluginLoadError, match="could not be imported"):
        discover(points=[point])


def test_the_entry_point_name_and_the_manifest_must_agree() -> None:
    """One name for a plugin. Two names for one plugin is a bug waiting to be filed."""
    with pytest.raises(PluginLoadError, match="must agree"):
        discover(points=list(_points(advertised=_Plugin(manifest(name="declared")))))


def test_two_distributions_providing_the_same_plugin_is_an_error() -> None:
    plugin = _Plugin(manifest(name="same"))
    globals()["_target_same"] = plugin
    duplicated = [
        EntryPoint(name="same", value=f"{__name__}:_target_same", group=ENTRY_POINT_GROUP),
        EntryPoint(name="same", value="other.dist:PLUGIN", group=ENTRY_POINT_GROUP),
    ]
    with pytest.raises(PluginLoadError, match="two installed distributions"):
        discover(points=duplicated)


def test_a_plugin_that_raises_while_registering_names_itself() -> None:
    class Exploding:
        manifest = manifest(name="exploding")

        def register(self, registry: ComponentRegistry) -> None:
            del registry
            msg = "no"
            raise RuntimeError(msg)

    with pytest.raises(PluginLoadError, match="exploding"):
        discover(points=list(_points(exploding=Exploding())))


def test_the_plugin_protocol_is_structural() -> None:
    """No base class to inherit, no decorator to remember."""
    assert isinstance(_Plugin(manifest()), Plugin)
    assert not isinstance(object(), Plugin)
