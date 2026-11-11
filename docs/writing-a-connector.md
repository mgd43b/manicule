# Writing a connector of your own

For a source manicule does not ship and that will never be upstreamed: an internal wiki, a
ticket system, a vendor's export format. A connector is a **plugin**, and a plugin is an
installed Python distribution that advertises an entry point. There is no fork to maintain and
no manicule source to patch, and the built-in connectors register through exactly the same route
(`tests/connectors/test_contract.py` asserts it, so it cannot rot unseen).

The complete, working version of everything on this page is
`packages/manicule-plugin-example` — read `DirectoryConnector` in its `__init__.py` alongside
this. CI builds it, installs it and runs the conformance suite against it.

## 1. The shape of it

```
acme-manicule/
├── pyproject.toml
└── src/acme_manicule/__init__.py
```

```toml
[project]
name = "acme-manicule"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = ["manicule>=0.1,<0.3"]

[project.entry-points."manicule.plugins"]
acme = "acme_manicule:PLUGIN"
```

The entry-point name and `manifest.name` must be the same string. A plugin may register any
number of components of any kinds, so a connector that also needs a parser for its native
format is one distribution, not two.

```python
from pydantic import BaseModel, ConfigDict

from manicule.container import keys
from manicule.plugins import BuildContext, ComponentRegistry, PluginManifest


class AcmeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: str


class AcmePlugin:
    manifest = PluginManifest(
        name="acme",
        version="0.1.0",
        core_version=">=0.1,<0.3",   # the range you have tested
        summary="Acme wiki, read-only.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.CONNECTOR.named("acme-wiki"),
            build_acme,                    # a factory: nothing is constructed here
            config_model=AcmeConfig,
            summary="Pages from the Acme wiki.",
        )


def build_acme(context: BuildContext) -> "AcmeConnector":
    return AcmeConnector(context.instance, context.config)


PLUGIN = AcmePlugin()
```

## 2. What the protocol asks of you

`manicule.core.protocols.Connector` is structural: no base class, no decorator. It has a
`name`, a `watermark`, and three methods.

| Member | Obligation |
|---|---|
| `name` | The **instance** name, `context.instance`, not the type. It becomes the `source` half of every document's identity, so two instances of one type must not share it. |
| `discover(watermark)` | Yield `DiscoveredDoc` for what exists or has changed; `None` means everything. Give each a `version_token` computed **without fetching**: an unchanged document is never downloaded. |
| `fetch(ref)` | Return one document's bytes as a `RawDocument` with an honest `media_type`, **and the revision it read in `metadata["version_token"]`** (below). Parsing is not your job; the parser chain for that media type does it. |
| `reconcile()` | Yield every source id that **still exists**. A watermark sync cannot see a deletion, so this is the only way a removed document leaves the index. It is mandatory so nobody can omit it quietly. |
| `watermark` | A read-only property: how far the last **completed** enumeration got, or `None`. |

Five of these are worth a paragraph each, because every one fails silently or late.

**`fetch` must prove which revision it read.** Durable acquisition (`storage.retain_source_bytes`,
on by default) stores the bytes before indexing them, and refuses a body that carries no proof it
is at least as new as what discovery saw. Put the *same* token `discover` produced in
`metadata["version_token"]`, computed from the source's state around the read, and refuse a
document that changed while you were reading it. A connector that omits this passes
`assert_connector_contract` and still indexes nothing: every document sits in retry.
`tests/connectors/test_example_connector_pipeline.py` drives a connector through the real pipeline
for exactly this reason. Build the token in one function that both methods call; two copies agree
until somebody edits one.

**A version token is a claim about *this* source, so name the source in it.** If the same
identifier, size and timestamp can mean different bytes under another base URL or root
directory, a source repointed at the new location keeps the old one's content, because the token
did not move. Fold a digest of whatever locates the data into the token.

**Advance the watermark when the enumeration ends, never as you yield.** If `discover` is
interrupted, a watermark that already moved sits *past* documents nobody received; the next sync
starts there and never sees them again, with no error. Set it after the loop, as
`DirectoryConnector` does.

**Only resume from a position if your source has a real change feed.** A cursor — a modification
time, an "updated since" query — misses anything that appears *behind* it: a file copied in with
its old timestamp, a page restored from backup, an item edited while the walk was running. If you
do use one, bind it to the configuration that produced it (record a scope fingerprint in
`Watermark.metadata` and treat a watermark with a different one as `None`, as the mbox and
Git-site connectors do), and overlap it slightly, because the gap between "yielded" and
"committed" is real and is only made survivable (the Confluence connector reaches five minutes
back). A source without a change feed should enumerate fully and let the version token do the
skipping, which is what `DirectoryConnector` does: the walk costs one `stat` per file and nothing
is read for an unchanged one.

**An inventory is complete or it is an error.** `reconcile` decides deletion, and a listing that
quietly omitted something looks identical to a source whose files were deleted. Raise on an
unreachable root, a directory you cannot list, a permission failure; ignore only a file that
genuinely vanished between being listed and being read.

The lifecycle hooks (`setup`, `teardown`, `health`) are optional. Add `setup` when you need to
`await` something before the first call — opening a session, say. Factories run on a worker
thread and may block; anything that needs `await` belongs in `setup`.

## 3. Configure it

Install your distribution into the environment manicule runs from. Discovery finds it without
any change to manicule's configuration:

```bash
uv tool install --with ./acme-manicule manicule        # a tool install
uv pip install -e ./acme-manicule                       # a development environment
```

Then name a source. The table key is what you call it; `type` is the implementation you
registered:

```toml
[connectors.handbook]
type = "acme-wiki"
schedule_s = 3600
options = { base_url = "https://wiki.example.org" }
```

`options` is validated against your `config_model` at startup. An unknown key is an error naming
the instance, not a setting that appears to be in force and is not. Defaults shared by every
instance of the type can go under `plugins.config."connector.acme-wiki"`; `options` wins field
by field.

`manicule connector sync handbook` runs it; `manicule serve` runs it on `schedule_s`.
`manicule plugin list` and `manicule connector list` show that it loaded and what its last run
recorded.

## 4. Test it

```python
from manicule.testing import assert_connector_contract

async def test_it_honors_the_contract():
    await assert_connector_contract(AcmeConnector("handbook", fake_config))
```

It checks that `discover`, `fetch` and `reconcile` agree with one another, that `reconcile` does
not omit anything `discover` returned, and that an abandoned enumeration leaves the watermark
where it was. It does **not** check what the ingest pipeline asks of a connector on top of the
protocol, such as the revision proof `fetch` must return. Drive the connector through
`IngestPipeline` as well, with durable acquisition on; the repository's
`tests/connectors/test_example_connector_pipeline.py` is the shortest version. Point the
connector at a recorded fixture or a local fake of the source, never the live service.
`tests/connectors/fake_confluence.py` shows how far that can go.

## 5. Things that will bite you

- **Mail, chat and personal notes are sensitive.** Register with
  `registry.add(..., sensitive=True)`. The index is not permission-aware, so an installation
  holding such a source is refused when served without authentication, loopback included.
  Leave it off for a directory of public handbooks.
- **A plugin runs with the full authority of the manicule process**: its network, filesystem and
  environment. There is no sandbox and no `permissions` field, because a boundary nothing
  enforces gets believed. Install only what you would run as yourself. manicule never installs
  one for you, and `plugins.allow_install` is off by default.
- **A version mismatch is a startup error.** `core_version` is checked before `register` runs,
  with both versions named. Pin the range you have tested.
- **A plugin that cannot load is fatal.** An import error, a bad entry point or a duplicate
  component stops startup, rather than leaving a source that quietly returns nothing.
- **Credentials do not go in `options` as plain strings.** Model them as `pydantic.SecretStr`
  with an `_env` or `_file` alternative, as the Gmail and Outlook configs do, so printing a
  config or a validation error never shows one.
- **Filter, do not remove.** `plugins.enabled` and `plugins.disabled` choose among what was
  discovered. To stop using a connector, uninstall its distribution or disable its plugin.
- **Keep module import cheap.** A plugin nobody has configured should cost one import. Put heavy
  imports inside the factory.

## 6. What this does not give you

Discovery is by entry point only. There is no "drop a file in a directory" loader, so even a
throwaway connector is a small installable package. That is deliberate — a second route would
be one the tests did not exercise — and an editable install is the quick version of it.
