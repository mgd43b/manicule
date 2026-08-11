"""Configuration models for the built-in stores.

Separate from :mod:`manicule.storage.plugin` for the reason every other subsystem separates
them: registration runs in **every** process that starts, and it needs exactly one thing about
a component eagerly — the model that validates settings written for it. This module imports
nothing heavier than pydantic, so discovery does not load SQLAlchemy, Alembic, LanceDB and
PyArrow in order to find out that a store named ``sqlite`` exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

DOC_STORE_NAME = "sqlite"
VECTOR_STORE_NAME = "lancedb"


class DocStoreConfig(BaseModel):
    """Settings for the relational store.

    The database's *location* is not here: it is ``storage.db_url`` and the data directory,
    which are properties of the installation rather than of this component. A component-level
    override would be a second place a path can be set, and two of those disagree by default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    echo: bool = Field(
        default=False,
        description="Log every statement the store emits. For debugging a query, and loud "
        "enough that nobody leaves it on: document text appears in the log.",
    )


class VectorStoreConfig(BaseModel):
    """Settings for the vector store.

    Empty, and declared anyway. A component with no model has any configuration written for it
    rejected rather than ignored, which is what this project wants; a component with an empty
    model gets the same rejection and a place to put the first real setting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


__all__ = [
    "DOC_STORE_NAME",
    "VECTOR_STORE_NAME",
    "DocStoreConfig",
    "VectorStoreConfig",
]
