# The filesystem connector: a directory as a source

The connector behind `manicule index <path>`, and the one a configured source uses when the
content is already on disk. It walks a tree in a stable order, reads bytes, and walks it again to
say what still exists. Its design decisions — identity is the resolved path unless a sidecar
manifest declares otherwise, the change token is size and modification time rather than a hash,
and the media type comes from a table written down in the module rather than from the machine's
mime database — are stated in the docstring of `manicule/connectors/filesystem.py`, which is where
they belong because each is a trap the code has to keep avoiding.

This page is the configuration.

## 1. Configuration

```toml
[connectors.corpus]
type = "filesystem"
schedule_s = 600

[connectors.corpus.options]
root = "/srv/corpus"
include_hidden = false
max_bytes = 5_000_000
exclude = ["archive/**", "README.md"]
```

`FilesystemConfig` is frozen and rejects extra fields. Its fields are:

| Field | Default | Meaning |
|---|---|---|
| `root` | required for a configured source | Directory or file to index, resolved to an absolute path. A relative one would change with the working directory, and a document's identity is built from it. |
| `include_hidden` | `false` | Whether to walk dot-files and dot-directories. Off, because the usual dot-file in a repository is tool state rather than a document. |
| `max_bytes` | absent | Refuse a file larger than this at discovery, before it is read. Absent means no ceiling; none is invented. |
| `exclude` | none | Root-relative POSIX globs this source never walks. See §2. |
| `enriched_profiles` | the standalone-storage profile | Enriched-export conventions to recognize inside HTML files, in precedence order. See [`enriched-html.md`](enriched-html.md) §1a. |

`manicule index <path>` builds the same class with its defaults — the path is the argument rather
than a setting, and there are no command-line flags for the rest. Anything in this table is set in
configuration, under a named source.

Version control and tool output are handled by the walk itself and are not configurable. These
**directory** names are never descended into: `.git`, `.hg`, `.svn`, `.venv`, `venv`,
`node_modules`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`, `.idea` and
`.DS_Store`. A repository's `.git` directory is larger than the repository, and nobody is indexing
theirs on purpose. Most of them begin with a dot and so are already covered by `include_hidden`;
the list is what keeps `venv` and `node_modules` out, and what keeps the rest out when
`include_hidden` is on. `.DS_Store` is in the list but is normally a *file*, where the dot-file
rule is what excludes it — turn `include_hidden` on and it is walked like any other dot-file.

## 2. `exclude`: content that stays where it is and is not searchable

Patterns are POSIX globs matched against the path **relative to the root**, so `README.md` names
the one at the top and not every README beneath it. `**/` matches zero directories as well as
more, so `**/drafts/**` catches a top-level `drafts/` too — which is what anybody writing that
pattern meant.

One inherited sharp edge, shared with `git-site` because both read patterns the same way: **`*`
matches across `/`**, not only within a path segment. So `*.md` is not "the Markdown files at the
top" — it is every `.md` at any depth, `notes/deep/a.md` included. There is no way to cap the
depth: `*/*.md` only sets a *minimum* of one directory, and still matches `notes/deep/a.md`.

The two shapes that are exact are the ones worth reaching for. A pattern with no `*` is a literal
path, which is why `README.md` names the one at the root and no other. A directory name, with or
without `/**`, names that subtree and nothing outside it.

A pattern naming a directory excludes everything beneath it, and the two spellings are the same
thing:

```toml
exclude = ["archive"]       # identical to
exclude = ["archive/**"]    # this
```

An excluded directory is **pruned**, not walked and discarded, so a large kept-on-purpose subtree
costs nothing per sync. An excluded ancestor excludes its descendants, which is why an operator
cannot write into one by another route — see §3.

The default is empty, deliberately, where `git-site` ships opinionated defaults. A website has
parts that are structurally not pages; a directory somebody pointed this connector at is the
corpus, and a default exclusion would be manicule deciding some of it does not count.

### 2.1 Why here rather than on a collection rule

A collection rule (`--uri-prefix`) is the other way to keep something out of a result set, and it
answers a different question. A rule says what a group *is*, so a document it does not select is
still in the corpus and still comes back from an unscoped search. `exclude` says the file is not
the corpus at all.

Use a rule to scope. Use `exclude` for material that must remain in the directory — superseded
work kept because it is the record of what was tried — and must never rank beside what replaced
it. The alternative people reach for otherwise is deleting the directory, which is content leaving
the working tree purely to keep it out of an index.

### 2.2 It governs what enters the index, not what is already in it

Adding a pattern stops those paths being discovered. It does **not** remove documents indexed
before it was written: they stay, served and going stale, because nothing in the product runs a
deletion-detection pass today — `manicule.ingest.reconcile.reconcile` exists and is tested, and
nothing calls it.

So after adding an `exclude` over content that has already been indexed, remove those documents
explicitly:

```bash
manicule document delete <document-id>
```

Soft by default, so it is recoverable inside the grace period.

The connector is nonetheless written so that this is a wiring gap rather than a design one: the
exclusion is applied in the walk that `discover` and `reconcile` both read, so the two never
disagree about what this source holds, and a reconciliation pass would remove exactly these
documents on the day one runs.

## 3. Authoring into an excluded directory is refused

Where `[authoring]` names this source, `document_create` writes a file and ingests the path it
wrote. A document authored beneath an excluded directory would therefore succeed at being written
and then never be indexed — worse than the oversized-body case, which at least fails its own
ingest. So it is refused while it is still an argument, the way `max_bytes` is:

> `/srv/corpus/memory/retry_policy.md` is covered by the `exclude` configured on source
> `'corpus'` (`memory/**`). Written, it would never be walked and so never indexed, so it is
> refused before anything reaches the disk.

`manicule doctor` reports the same misconfiguration as `failing` without waiting for somebody to
try it, beside the other authoring findings. See [`memory-authoring.md`](../memory-authoring.md)
§4.11.
