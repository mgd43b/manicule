"""The built-in storage plugin.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is. Until this existed, ``storage.db = "sqlite"`` named a component nothing provided,
so a container built from configuration could resolve every other kind and not the two that
hold the corpus — the failure ``check_wiring`` reports and no installation could get past.

**Nothing here imports a database.** Registration needs only the configuration models, which
live in :mod:`manicule.storage.config`. SQLAlchemy, Alembic, LanceDB and PyArrow are imported
inside the factories, so a process that never opens the index — ``manicule doctor``, a plugin
listing, a completion script — does not pay for them.

The relational store owns its engine, and the engine is reachable through
:attr:`~manicule.storage.scoped.WorkspaceScoped.engine`. That is deliberate: migrations,
backups and the conversation store all need the same engine, and a second one on the same
file is a second connection pool with its own opinion about whether the schema is current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule.storage.config import (
    DOC_STORE_NAME,
    VECTOR_STORE_NAME,
    DocStoreConfig,
    VectorStoreConfig,
)

if TYPE_CHECKING:
    from manicule.core.protocols import DocStore, VectorStore

__all__ = ["PLUGIN", "StoragePlugin", "build_doc_store", "build_vector_store"]


def build_doc_store(context: BuildContext) -> DocStore:
    """Open the relational store for the configured workspace.

    The workspace is bound to the handle here and nowhere else, so every query the store ever
    runs carries it and no call site can forget to pass one.

    Raises:
        ConfigError: The context carries configuration of some other type. Reachable only by a
            caller building this outside the container, which validates against the registered
            model first — substituting defaults there would build a store whose settings appear
            to be in force and are not.
    """
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415 - see module docstring
    from manicule.storage.engine import create_engine  # noqa: PLC0415

    config = context.config
    if not isinstance(config, DocStoreConfig):
        msg = (
            f"the {DOC_STORE_NAME!r} document store was built with {type(config).__name__} "
            f"where it declares {DocStoreConfig.__name__}. Configuration reaching a factory is "
            f"validated against the model the component registered; a factory called outside "
            f"the container has to supply that model itself."
        )
        raise ConfigError(msg)
    engine = create_engine(context.data_dir, echo=config.echo)
    ingest = context.settings.ingest
    return SqliteDocStore(
        engine,
        workspace_id=context.settings.workspace,
        data_dir=context.data_dir,
        max_journal_records=ingest.max_journal_records,
        max_journal_metadata_bytes=ingest.max_journal_metadata_bytes,
        max_acquired_blob_backlog_bytes=ingest.max_acquired_blob_backlog_bytes,
        min_disk_headroom_bytes=ingest.min_disk_headroom_bytes,
    )


def build_vector_store(context: BuildContext) -> VectorStore:
    """Open the vector store under the data directory.

    It is **not** given a dimension. The dimension is a property of the embedder, read from
    its fingerprint when the first ingest calls ``ensure_ready``; a configured one is a value
    that can disagree with the model, and when it does the index is silently wrong.
    """
    from manicule.storage.engine import VECTORS_DIRNAME, prepare_data_dir  # noqa: PLC0415
    from manicule.storage.vectors import LanceVectorStore  # noqa: PLC0415

    config = context.config
    if not isinstance(config, VectorStoreConfig):
        msg = (
            f"the {VECTOR_STORE_NAME!r} vector store was built with {type(config).__name__} "
            f"where it declares {VectorStoreConfig.__name__}."
        )
        raise ConfigError(msg)
    return LanceVectorStore(prepare_data_dir(context.data_dir) / VECTORS_DIRNAME)


class StoragePlugin:
    """The plugin object the ``storage`` entry point resolves to."""

    manifest = PluginManifest(
        name="storage",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="SQLite for the 35 modeled relational tables and FTS5; LanceDB for vectors.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.DOC_STORE.named(DOC_STORE_NAME),
            build_doc_store,
            config_model=DocStoreConfig,
            summary="Documents, chunks, BM25 over FTS5, collections, tags, versions, trash.",
        )
        registry.add(
            keys.VECTOR_STORE.named(VECTOR_STORE_NAME),
            build_vector_store,
            config_model=VectorStoreConfig,
            summary="Embedded vector search, dimension taken from the embedder's fingerprint.",
        )


PLUGIN = StoragePlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN
