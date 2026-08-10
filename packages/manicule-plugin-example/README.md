# manicule-plugin-example

The smallest complete plugin. Copy it to start a new one.

It provides one component of three kinds — a parser, a middleware and a retrieval stage —
and exists for two reasons:

1. **It is the plugin-authoring documentation**, in the form that cannot go out of date,
   because CI builds and loads it.
2. **manicule's own test suite depends on it.** Entry-point discovery is verified against a
   real installed distribution rather than a stand-in, so the path third-party plugins take
   is the path that is exercised on every commit.

## What a plugin is

An installed distribution advertising an entry point in the `manicule.plugins` group:

```toml
[project.entry-points."manicule.plugins"]
example = "manicule_plugin_example:PLUGIN"
```

The entry point resolves to an object with a `manifest` and a `register` method, or to a
zero-argument callable returning one. The entry-point name and `manifest.name` must match.

## Rules worth knowing before you write one

- **Register factories, not instances.** `register` is called at startup for every installed
  plugin. Constructing anything there makes every installation pay for every plugin.
- **Keep the module top level cheap.** Put heavy imports inside the factory. A plugin nobody
  has configured should cost an import of this file and nothing more.
- **Declare a `config_model`.** Configuration written for a component that declares no model
  is rejected rather than ignored, so the model is how your settings become settable.
- **Pin `core_version` to a range you have tested.** A mismatch is refused at startup, with
  the versions named. That is deliberate: the alternative is an attribute error somewhere
  unrelated, much later.

## Privileges

Plugins are imported into the manicule process and run with its full authority — the
network, the filesystem, the environment, everything. There is no sandbox and no
`permissions` declaration, because manicule cannot enforce one and a guarantee nothing
enforces is worse than an absent one: it gets believed.

Install plugins you would be willing to run as yourself, because that is what happens.
