"""The built-in storage plugin.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is. Until this existed, ``storage.db = "sqlite"`` named a component nothing provided,
so a container built from configuration could resolve every other kind and not the two that
hold the corpus — the failure ``check_wiring`` reports and no installation could get past.

**Nothing here imports a database.** Registration needs only the configuration models, which
live in :mod:`manicule.storage.config`. SQLAlchemy, Alembic, LanceDB, PyArrow and
``qdrant-client`` are imported inside the factories, so a process that never opens the index —
``manicule doctor``, a plugin listing, a completion script — does not pay for them. An
installation that leaves the default therefore never imports a network client.

A Qdrant installation still has LanceDB **on disk**, and that part is worth stating rather
than implying: the relational store is not optional and the ``storage`` extra carries
SQLAlchemy, Alembic, LanceDB and PyArrow together, so there is no supported way to install
manicule without the embedded backend present. What a Qdrant installation no longer does is
*import* it.

That distinction used to be lost, and the cost was not a slow start. ``manicule.app.runtime``
decided whether a store wanted the publication-following wrapper by importing the Lance
classes and asking ``isinstance``, so every Qdrant process imported LanceDB in order to be
told it had not configured LanceDB — and on a CPU without AVX2 that import is ``SIGILL``
rather than a delay, which left the networked backend unusable on the hardware it exists to
serve. The capability is now asked of the store itself, through
:class:`~manicule.core.protocols.PublicationAwareVectorStore`, and a backend's module is
imported only once its own store is what the container built.
``tests/test_import_boundary.py`` holds both halves of that line.

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
    QDRANT_VECTOR_STORE_NAME,
    VECTOR_STORE_NAME,
    DocStoreConfig,
    QdrantVectorStoreConfig,
    VectorStoreConfig,
)

if TYPE_CHECKING:
    from manicule.core.protocols import DocStore, VectorStore

__all__ = [
    "PLUGIN",
    "StoragePlugin",
    "build_doc_store",
    "build_qdrant_vector_store",
    "build_vector_store",
]


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
    """Open the embedded vector store under the data directory.

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


def build_qdrant_vector_store(context: BuildContext) -> VectorStore:
    """Open a handle on the Qdrant server this installation is configured against.

    Construction opens no socket. The client is configured here and dials on the first
    operation that needs it, so building the container still costs nothing — ``manicule
    doctor`` resolves every component, and a vector store that connected eagerly would make a
    diagnostic command fail on the thing it was invoked to diagnose.

    The store is given the workspace, because Qdrant has one flat namespace and the isolation a
    directory gives the embedded store has to be in the collection's name instead. It is also
    told it owns the client, so the container's shutdown closes the sockets rather than leaving
    them to the garbage collector.

    Raises:
        ConfigError: The context carries configuration of some other type, or no endpoint. The
            endpoint is normally refused earlier, by
            :meth:`~manicule.config.settings.Settings.policy_problems`; this is the same
            refusal for a store built outside that path, because a client dialed at no address
            fails later and less clearly.
    """
    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415 - see module docstring

    from manicule.storage.qdrant import QdrantVectorStore  # noqa: PLC0415

    config = context.config
    if not isinstance(config, QdrantVectorStoreConfig):
        msg = (
            f"the {QDRANT_VECTOR_STORE_NAME!r} vector store was built with "
            f"{type(config).__name__} where it declares {QdrantVectorStoreConfig.__name__}."
        )
        raise ConfigError(msg)
    settings = context.settings
    url = (settings.storage.vector_db_url or "").strip()
    if not url:
        msg = (
            f"the {QDRANT_VECTOR_STORE_NAME!r} vector store needs storage.vector_db_url, and "
            f"it is empty. Set it to the Qdrant HTTP endpoint, e.g. "
            f"https://qdrant.internal:6333."
        )
        raise ConfigError(msg)
    qdrant = settings.storage.qdrant
    client = AsyncQdrantClient(
        url=url,
        api_key=None if qdrant.api_key is None else qdrant.api_key.get_secret_value(),
        prefer_grpc=qdrant.prefer_grpc,
        grpc_port=qdrant.grpc_port,
        timeout=qdrant.timeout_s,
        # The client's version check runs a blocking request on a background thread from
        # inside `__init__`, and warns from that thread when the server does not answer. That
        # makes constructing a client a network operation, which is exactly what this factory
        # promises it is not: `manicule doctor` builds every component, and a diagnostic that
        # stalls and then warns out of a thread is worse than one that reports a failing
        # vector store. `QdrantVectorStore.health` is where reachability is established, on
        # the surface built to report it; the supported client range is the floor in
        # pyproject.toml.
        check_compatibility=False,
    )
    return QdrantVectorStore(
        client,
        workspace_id=settings.workspace,
        collection_prefix=qdrant.collection_prefix,
        owns_client=True,
    )


class StoragePlugin:
    """The plugin object the ``storage`` entry point resolves to."""

    manifest = PluginManifest(
        name="storage",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary=(
            "SQLite for the 40 modeled relational tables and FTS5; LanceDB or Qdrant for vectors."
        ),
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
        registry.add(
            keys.VECTOR_STORE.named(QDRANT_VECTOR_STORE_NAME),
            build_qdrant_vector_store,
            config_model=QdrantVectorStoreConfig,
            summary="Vector search on a Qdrant server, for an index several processes share.",
        )


PLUGIN = StoragePlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN
