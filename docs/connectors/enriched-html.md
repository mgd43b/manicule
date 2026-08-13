# Enriched standalone HTML

Some offline exporters write one HTML file per page and put the page's real identity inside it:

```html
<!doctype html>
<html>
  <head><title>Retry Runbook</title></head>
  <body>
    <section data-source-metadata>
      <p><strong>Page ID:</strong> 1002</p>
      <p><strong>Space:</strong> ENG</p>
      <p><strong>Version:</strong> 7</p>
      <p><strong>Last modified:</strong> 2026-08-12T09:45:00Z</p>
      <p><strong>Source:</strong> <a href="https://docs.example.test/pages/1002">canonical page</a></p>
    </section>
    <main data-document-representation="storage">…</main>
  </body>
</html>
```

Ordinary filesystem ingestion indexed the text of that file perfectly well and then cited
`1002.html` and a `file://` URI, because the filename was all the connector was given — and it
read the `<main>` as generic HTML, so a code macro's language, a panel's severity, a task's state
and a Graphviz engine were flattened into prose. Everything needed to cite the page properly, and
to read its body as what it is, was in the file the whole time.

## 1. Using it

```console
$ manicule connector sidecar /path/to/pages
$ manicule connector sync docs      # an ordinary filesystem source over the same root
```

The first command writes `1002.html.source.json` beside `1002.html`. The second is unchanged —
the filesystem connector already reads that manifest. **The pages themselves are never
modified.**

That first form converts with the **built-in default profile**. If your exporter spells its
markers differently — the case [§1a](#1a-configuring-another-exporters-markers) exists for — use
the named-source form instead, or the run will report `no_profile` for every page while the
configured sync over the very same directory adapts all of them:

```console
$ manicule connector sidecar --source docs
1 of 1 page(s) under /path/to/export
source docs; profile(s): custom-storage-export
```

`--source` names a configured **instance** — the key in `[connectors.docs]` — and never a
connector type. It takes both the root and the enriched profiles from the connector
`manicule connector sync docs` would run, so the conversion and the sync cannot hold two readings
of one profile. The source must exist, be enabled, be a filesystem source, and declare at least
one profile; each of those is refused by name rather than quietly falling back to the default,
because falling back is what produced the misleading `no_profile` in the first place.

Every run says which profiles it used, whether or not it wrote anything:

```console
$ manicule connector sidecar /path/to/export
0 of 1 page(s) under /path/to/export
no configured source; profile(s): standalone-storage
adapted no pages at all — every considered file is listed below
skipped          why
pages/1002.html  matches no configured enriched-document profile
                 ('standalone-storage'), so it is an ordinary HTML file as far
                 as this is concerned and is left to generic ingestion
```

That is the same corpus as the successful run above. The difference is the profile set, and the
line naming it is what tells the two apart.

### A bounded subdirectory

A positional argument may be given *with* `--source` to convert part of a source's tree:

```console
$ manicule connector sidecar --source docs pages
```

It is resolved relative to the source's root and must stay inside it. A path that resolves
outside — through `..`, through an absolute path elsewhere, or through a symlink — is refused,
and so is a symlink named directly, because the walk beneath does not follow links either. A
source name is not a licence to write manifests beside somebody else's files.

The manifests a narrowed run writes are **byte-identical** to the ones a whole-root run would
write, because no field in a manifest is relative to the conversion's root —
[§4](#4-two-deliberate-omissions) omits `snapshot_path` for exactly this reason.

**Narrowing narrows duplicate detection, and that is the one cost.** Two pages declaring one page
id are refused as a pair, and the pair is only visible to a run that walked both. A run confined
to `pages/` will happily write a manifest for a page whose twin sits in `other/`. Nothing breaks
silently — the twin keeps its path identity, and once the converted page holds the id it declares,
`doctor`'s `document-identity` check reports two rows for one page ([§5d](#5d-converting-a-corpus-that-is-already-indexed)). But the refusal arrives after a sync
rather than during the conversion. **Convert the whole root unless you have a reason not to.**

`--force` replaces manifests that already exist; without it they are left alone, because one
already there was most likely written by hand or by another tool.

Every page that produced nothing is reported with the reason, and `--json` carries a count per
outcome beside it. A run that adapted nothing says so outright rather than printing `0 of 40`.

**The second command does the interesting half, and it does it with or without the first.** At
fetch the connector reads each `.html` file, and where a profile matches it extracts the storage
body and hands *that* to the dedicated storage parser under
`application/xhtml+xml;profile=confluence-storage`. The wrapper — the metadata banner, the
exporter's navigation, any scripts — reaches no chunk.

What the sidecar adds is **identity**. Discovery has to know what a document is called before it
fetches anything, and discovery does not read files: that is what makes a re-sync of an unchanged
corpus cost a `stat` per file rather than a full read. The manifest is the one thing beside a page
small enough to read on every walk, so it is where the page id has to be written down.

**A page without one is adapted anyway, and it says what it is missing.** It is parsed as storage
format and cited by its own title and URL — but it is keyed on where it sits, so
[§3's](#5-identity-moved-off-the-path-and-that-is-a-migration) *moving the snapshot file must not
create a second document* does **not** hold for it: move it and you get two. That is not left to
be discovered. The document carries `enriched_adaptation.outcome = "identity_not_applied"` with a
reason naming the page id it declares, the consequence, and the command:

```json
{
  "outcome": "identity_not_applied",
  "reason": "this page declares source_id '1002' and is indexed under its path, because a
             document's identity has to be known before it is fetched and discovery does not
             read files. … moving or renaming it will create a second document. Run
             `manicule connector sidecar --source docs` to write the manifest that applies the
             declared identity, then sync."
}
```

**The command in that reason is the one that will work for that page's source.** A source with
custom profiles is told `--source <name>`; a one-off `manicule index <path>` is told the root
form, which is right for it because it also uses the default profile. This used to name the root
form unconditionally, which for a custom-profile corpus was an instruction that reports
`no_profile` and writes nothing — a command that had been written and never run.
`tests/connectors/test_sidecar_source.py` now takes the command off the document, parses it, and
executes it, so the text cannot drift away from what works.

Once the manifest is there the outcome is `adapted` and the reason is gone — the notice clears
itself rather than sitting on every page for ever.

## 1a. Configuring another exporter's markers

Nothing is hard-coded to one exporter's attribute spelling. A profile states four things:

```toml
[connectors.docs]
type = "filesystem"

[connectors.docs.options]
root = "/path/to/pages"

[[connectors.docs.options.enriched_profiles]]
name = "acme-export"
metadata_selector = '[data-acme-page]'
body_selector = '[data-acme-format="storage"]'
representation = "application/xhtml+xml;profile=confluence-storage"
labels = { identifier = "source_id", revision = "version" }
```

Listing a profile **replaces** the default rather than adding to it; list
`name = "standalone-storage"` alongside to keep both. An empty list turns adaptation off, which
is a supported configuration — it is how you establish whether an unexpected parse is the
adapter's doing. `connector sidecar --source` refuses a source configured that way rather than
running: with no profile there is nothing to recognise, so every page would report `no_profile`.

**`labels` replaces the defaults too, and that is the easier one to get wrong.** The profile above
understands `Identifier` and `Revision` and *stops* understanding `Page ID`, `Version`,
`Last modified` and the rest. A page whose metadata section reads `<strong>Page ID:</strong>` under
that profile is refused with `invalid_metadata` — "declares no page id" — rather than adapted,
because the label it was found under maps to nothing. The keys are the spellings **in your
documents**, normalised to lower case; the values are manicule's field names. If a field is not in
the mapping it is not extracted, so a profile that omits an address mapping produces manifests with
no `canonical_uri`:

```toml
labels = { identifier = "source_id", revision = "version", source = "canonical_uri" }
```

Map every field you want cited, not only the identifier.

Three things are refused when the settings load, rather than at the first sync:

- **A selector that is not an attribute selector.** `main` would make every `<main>` on every page
  a storage body, which is the guess this whole mechanism exists instead of. An attribute is a
  statement an exporter made on purpose; an element name is a fact about HTML.
- **A representation outside the allowlist.** A profile's representation decides which parser
  untrusted extracted markup reaches, and a misspelling would route to nothing at all while the
  configuration looked right.
- **A label mapped to a field that does not exist.** An alias that silently does nothing is
  indistinguishable from not having written it — the same reason the manifest reader refuses
  unknown keys.

## 1b. Exactly one of each, or a refusal

There are nine combinations of "how many metadata sections" and "how many storage bodies", and
all nine are enumerated rather than left to a chain of conditions:

| sections \ bodies | 0 | 1 | many |
|---|---|---|---|
| **0** | not an enriched page — ordinary HTML ingestion | no identity of its own | ambiguous |
| **1** | no storage body | **adapted** | ambiguous |
| **many** | ambiguous | ambiguous | ambiguous |

A file that engages a profile and is then ambiguous is **refused**, not offered to the next
profile — falling through is how a document carrying two vocabularies would get to choose which
of its bodies was indexed.

A refusal never costs the document. The wrapper is a perfectly good HTML file and is indexed as
one, with the reason recorded on the document under `enriched_adaptation` so `doctor` and the
conversion report can name it.

## 2. Why a sidecar, and not the other two

The feature request offered three shapes. The recommendation is the third, and the reasoning is
worth keeping because it is mostly about what *already exists*.

| | New ingestion code | Corpus rewritten | Output path derived from | `SourceMetadata` change |
|---|---|---|---|---|
| **A dedicated enriched-HTML connector** | a whole connector: walk, identity, watermark, reconcile | no | — | none |
| **Convert to manifest-plus-body** | none | yes — a directory per page | the page's own metadata | none |
| **Sidecar generation plus an adapter** (this) | **none** | **no** | **the file that was walked to** | **none** |

**A dedicated connector** would duplicate what `confluence-snapshot` and `filesystem` already do
— walking a tree in a stable order, deriving identity, reconciling deletions — for a corpus that
differs from an ordinary directory only in what is *inside* each file. It would also add a second
place where provenance is extracted, and those two would drift.

**A conversion command** emitting the manifest-plus-body layout the snapshot connector reads
means writing a new directory per page, which doubles the corpus and then needs the original HTML
kept as well to satisfy "immutable retention of the original". Worse, it has to *decide where the
new files go*, and the page's own metadata is the obvious thing to name them after — at which
point a page id of `../../../etc/cron.d/x` is a write primitive, and the defence is a path
validation somebody has to keep correct forever.

**A sidecar** needs none of that. [`connectors/sidecar.py`](../../src/manicule/connectors/sidecar.py)
already defines the manifest wire format, and
[`connectors/filesystem.py`](../../src/manicule/connectors/filesystem.py) already discovers it,
folds it into the change token so editing either file re-ingests the page, and skips it as a
document of its own. The whole job is turning the page's metadata section into that manifest.

### The interface needed no changes

`SourceMetadata` was designed to carry a publication's own account of itself without knowing what
published it, and that held here without a field being added:

| The page says | Recorded as |
|---|---|
| Page ID | `source_id` |
| Title (or `<title>`) | `title` |
| Space, ancestors | `section_path`, coarsest first |
| Version | `version` |
| Last modified | `modified_at` |
| Source link | `canonical_uri` |
| Retrieved / exported | `LocalSnapshot.retrieved_at` |
| — | `snapshot_checksum`, computed here |

## 3. Security

The input is a file inside a corpus, which makes it a document anyone with write access to the
wiki it came from could have authored. It is treated that way.

**Metadata cannot cause a write anywhere.** The manifest's path is
`sidecar.manifest_path_for(<the file the walk reached>)`. No value read out of a document reaches
it. Traversal is not refused — it is unrepresentable, which is the same distinction
`sidecar.py` draws between validating a path and never dereferencing one.

**Metadata cannot cause a read outside the root.** The walk does not follow symlinks at all, and
every path is checked against the resolved root. Two mechanisms, and each is pinned by its own
test.

**Nothing is fetched.** A canonical URL is read out of an `href` and written to a field. There is
no network client in this path; a test monkeypatches `socket` to raise and asserts conversion
still succeeds.

**Nothing is executed, and scripts stay inert** — because the source file is not written to at
all. A `<script>` or a CDATA-wrapped macro body is exactly as it was afterwards.

**Values a citation would carry are validated by the real model.** The extracted fields construct
a `SourceMetadata`, so a `javascript:`, `data:`, `vbscript:` or `file:` canonical URI, a control
character in a title, and a naive timestamp are refused here by exactly the code that refuses them
at ingest — not by a second copy of those rules that would drift.

**Existing files are not overwritten** without `--force`.

## 4. Two deliberate omissions

**`snapshot_path` is not written.** The manifest format accepts it, and it is cross-checked at
ingest against the path relative to the **ingestion** root — which a conversion cannot know. A
conversion rooted anywhere else would emit a path that disagrees with the one manicule walked to,
and every manifest would be refused for saying something true about a different tree.

**`snapshot_checksum` is written**, and it has no such coupling. It is compared against the digest
manicule computes over the bytes it actually read, so a page edited after conversion — its version
and modification time now stale in the manifest beside it — is refused with a reason rather than
quietly citing metadata that describes an older revision. Re-running the conversion is the fix,
and it is idempotent.

## 5. Identity moved off the path, and that is a migration

A local file's identity used to be its resolved path, always. It is now the `source_id` a manifest
declares, where one does. The reason is that a mirror reorganised from by-space to by-tree has not
created new pages — but a connector keyed on the path reports every document deleted and every
document new, and the curated collections and tags hanging off the old rows go with them.

**Documents ingested before this are re-keyed in place** by revision `b2e6d0c94a17`, which runs
when the database migrates — before the first read, so no command applies it. The re-key is an
`UPDATE`, never an insert-and-delete: `documents.id` is the parent of five `ON DELETE CASCADE`
foreign keys, so replacing the row would fire every one of them and destroy the chunks, versions,
glossary entries, **collection membership and tags** the migration exists to preserve. Those
travel with the document instead. The previous identity is recorded under
`metadata.previous_identity`, which is what makes the downgrade exact rather than a re-derivation.

Two things it deliberately does not do. It **deletes nothing**, and it **touches nothing it
cannot re-key** — a document whose provenance yields no page id is left exactly as it was, and one
whose declared identity is already held by another document is left alone too, because moving onto
an occupied primary key overwrites a row nothing can restore.

`manicule doctor`'s `document-identity` check reports what is left after that: the collisions.
It clears itself once a person resolves them. It reports, per document:

| | |
|---|---|
| old identity | `documents.source_id` — the path |
| new identity | the `source_id` in the stored provenance record |
| old and new `document_id` | derived, so the mapping is exact rather than described |
| chunks and vectors reusable | **no** |
| citations change | title and URL keep, the chunk named changes |
| recommended command | `manicule document list --source <name>`, then `manicule document delete <id>` |

The mapping is a fact already in the database — the old identity in a column, the new one in the
record written by the same fetch — so unlike the connector-instance rename in #94 nothing here has
to be guessed, which is why this ships a migration and that did not.

**Chunks and vectors are not carried across**, for two reasons rather than one: `chunk_id` derives
from `document_id`, so every chunk id moves; and the documents whose identity moves are precisely
the documents whose *text* changes, because their body now reaches the storage parser instead of
the HTML parser. Re-embedding is unavoidable either way, so the migration leaves the old chunks in
place — removing content before its replacement exists is the worse intermediate state.

That has two consequences, both stated here because neither should be discovered.

**The stored text is wrong until you sync, and `doctor` says so.** A re-keyed page has its correct
identity and its *old* chunks: the generic-HTML parse of the wrapper, metadata banner and all. So
between the migration and the next sync the corpus still returns exactly what this change exists to
keep out of it. The migration logs it, and `doctor`'s **`document-content`** check names the
documents and the command:

```
N document(s) were re-keyed onto the identity their source declares and have not been
re-read since. Their identity is correct and their stored text is not …
Run `manicule connector sync <name>` to rebuild it.
```

It clears itself: the migration records the `content_hash` it saw — only for documents whose parse
actually changes, so a mirrored PDF with a manifest is never reported — and the finding disappears
the moment the page is re-ingested with its extracted body.

**Chunk ids stop matching their own derivation, and nothing reads them that way.** A chunk whose
parent moved no longer equals `chunk_id(document_id, position, text)`, and `glossary_entry_id`
digests the chunk id so the same is true one level down. Nothing recomputes either and compares:
`chunk_id` is called in exactly one place (`chunking/chunker.py`) and `glossary_entry_id` in one
(`storage/glossary.py`), both at write time, to *mint* an id rather than check one. The only cost
is that a later re-parse cannot reuse the vector for such a chunk — it replaces it, which is the
re-embedding this change made unavoidable anyway. `tests/ingest/test_storage_integration.py`
migrates, syncs, and asserts no chunk survives with an id that does not derive.

### 5d. Converting a corpus that is already indexed

The migration above re-keys documents that were in the database before the identity change. It is
**not** what happens when you convert a corpus that this build already indexed under its paths,
and the difference matters because the second is the ordinary workflow:

```console
$ manicule connector sidecar --source docs   # a page that was already indexed path-keyed
$ manicule connector sync docs
$ manicule --json document list              # two rows, one page
```

**Generating a manifest does not re-key the existing row.** The next sync discovers the page under
its declared id and creates a *second* document; the path-keyed one is left behind. Nothing
removes it on its own — the connector no longer discovers it under that path, and no command in
manicule runs reconciliation — so **the old copy must be deleted by hand.**

`doctor` names it rather than reporting a healthy corpus:

```console
$ manicule doctor
 degraded  document-identity  1 document(s) are still keyed on their file's path
while the manifest beside each one declares an identity of its own —
'/path/to/export/pages/1003.html' declares '1003'. … Compare them with
`manicule document list --source docs`, remove whichever is the stale copy with
`manicule document delete <id>`, or correct the manifest that declares the wrong
page id and sync again. There is no bulk remedy for this and none is implied.
```

Then, exactly as it says:

```console
$ manicule document list --source docs
3d1436bf632764209fecb2467fc70e7b
  Escalation · docs · application/xhtml+xml;profile=confluence-storage · indexed
d70c35a339cd9893e4df4ba026efa9ef
  Escalation · docs · application/xhtml+xml;profile=confluence-storage · indexed
2 shown, offset 0, page size 50

$ manicule document delete d70c35a339cd9893e4df4ba026efa9ef
removed d70c35a339cd9893e4df4ba026efa9ef into the trash
```

after which `document-identity` is `ok` again. The delete is the ordinary soft delete, so a copy
removed by mistake is restorable.

The reason `doctor` sees this at all is the `claimed` rule in its identity check: a path-keyed
enriched page is normally *excluded* from that check, because it carries its own notice naming the
command that applies its identity. The exclusion lifts the moment another document holds the
identity it declares — which is precisely the state following that notice produces. Without that,
the tool would instruct an action that creates an orphan and then say nothing about it.

**Convert before the first sync where you can**, and none of this arises: the manifest is read at
the first discovery, the page is keyed by its id from the start, and there is no second row.

Two files declaring one page id are **both** returned to their paths, and the conflict is reported.
Not the second only: `documents` is `UNIQUE` on `(workspace_id, source, source_id)`, so honouring
both would mean a silent overwrite — and keeping the *first* would make ownership depend on walk
order, so renaming a directory would move one page's content onto another page's identity.

## 5a. What a `.html` file's media type is, and when

Discovery declares **no** media type for `.html` and `.htm`, because whether a file is an enriched
export is a fact about its contents and discovery does not read contents.
`DiscoveredDoc.media_type` is `None`, which the pipeline already reads as "this connector has not
made a claim" rather than as a change. Every other suffix keeps its declaration and the routing
check that goes with it.

Guessing instead is wrong in both directions: declare `text/html` and every adapted page
re-ingests on every sync because the stored type disagrees with the declared one; declare the
storage type and every ordinary page does.

## 5b. What is recorded, and where

| Fact | Where |
|---|---|
| extracted-body checksum | `documents.content_hash` — the index is built from the body |
| full-snapshot checksum | `enriched_adaptation.snapshot_checksum` |
| local snapshot path | `LocalSnapshot.path`, and repeated in the record |
| adapter profile | `enriched_adaptation.profile` |
| representation | `documents.media_type`, and `SourceMetadata.content_type` |
| adapter version | `enriched_adaptation.adapter_version` |

`enriched_adaptation` is a connector key rather than a field on `LocalSnapshot`, on the rule
`storage.md` §4.2.1 already set for `space_key`: the provenance record carries what *every* source
has. A snapshot checksum distinct from the content hash exists only where a document's bytes are
*derived from* a local file rather than being it, which is true of an enriched export and of
nothing else manicule ingests.

The manifest records the **representation** and deliberately not the adapter version. The first is
a fact about the page and stays true however this code changes; the second is a fact about this
build, and a manifest declaring it would either be believed — recording a version that never ran —
or checked, refusing every existing manifest the day the adapter improved. The version travels in
the change token instead, so bumping it rebuilds the derived body with no manifest rewritten and
nothing re-downloaded.

## 5c. What this still does not do

Nothing is materialised on disk. There is no derived-artifact directory, so there is nothing to
place, name, exclude from discovery, or garbage-collect — the extraction is deterministic and
happens in memory at the moment it is needed. The same walk therefore cannot index its own output,
because there is no output for it to find.

Graphviz is not rendered, macros are not expanded, page references are not followed and
attachments are not fetched. DOT source is preserved character for character and stays inert.

## 6. Why it is command line only

`connector sidecar` is the one operation that writes into the corpus *directory* rather than into
the index. Everything else manicule does to a corpus is read-only, so an unattended surface able
to write into one is a new kind of authority rather than a new operation. It stays where a person
is present — the rule `reset-index`, `backup`, `import` and `collection orphans` are already held
to.

`--source` does not soften that. It narrows what the command may touch rather than widening who
may run it: the root is no longer the caller's to choose, and a subdirectory given alongside it
must resolve inside the configured root. The authority the boundary is about — writing files into
a directory a request named — is unchanged, so the boundary is unchanged.

Three tests hold the line, and they catch different things:

| | |
|---|---|
| `tests/app/test_surface_parity.py` | `connector_sidecar` is not an MCP tool |
| `tests/api/test_routes.py`, `ABSENT` | the HTTP paths somebody would guess are unrouted |
| `tests/api/test_routes.py::test_no_route_generates_sidecar_manifests` | **no** route mentions sidecar, whatever it is called |

The third exists because the second cannot cover a path nobody predicted — the same reason
`plugins/install` has a test of that shape. A route added at an unguessed path passes the `ABSENT`
table and fails the by-name walk; a route added at a guessed path under an unrelated *name* does
the reverse. Both were confirmed by adding such a route and watching each test fail.

There is no unattended scheduler to exclude it from: manicule has none, and `schedule_s` was
removed in #98 precisely because a setting nothing reads is a promise nothing keeps.
