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

Ordinary filesystem ingestion indexes the text of that file perfectly well and then cites
`1002.html` and a `file://` URI, because the filename is all the connector was given. Everything
needed to cite the page properly was in the file the whole time.

## 1. Using it

```console
$ manicule connector sidecar /path/to/pages
$ manicule connector sync docs      # an ordinary filesystem source over the same root
```

The first command writes `1002.html.source.json` beside `1002.html`. The second is unchanged —
the filesystem connector already reads that manifest. **The pages themselves are never
modified.**

`--force` replaces manifests that already exist; without it they are left alone, because one
already there was most likely written by hand or by another tool.

Every page that produced nothing is reported with the reason. A run that found no metadata
anywhere says so, rather than reporting a clean conversion of nothing.

## 2. Why a sidecar, and not the other two

The feature request offered three shapes. The recommendation is the third, and the reasoning is
worth keeping because it is mostly about what *already exists*.

| | New ingestion code | Corpus rewritten | Output path derived from | `SourceMetadata` change |
|---|---|---|---|---|
| **A dedicated enriched-HTML connector** | a whole connector: walk, identity, watermark, reconcile | no | — | none |
| **Convert to manifest-plus-body** | none | yes — a directory per page | the page's own metadata | none |
| **Sidecar generation** (this) | **none** | **no** | **the file that was walked to** | **none** |

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

## 5. What this does not do

It does not interpret the body. `<main data-document-representation="storage">` holds
storage-format XHTML, which the web parser indexes as HTML today — including the macro bodies
[`recover_cdata`](../../src/manicule/parsers/web.py) rescues from being silently deleted. Giving
that markup real semantics is a parser's job and is tracked separately. This command's job is the
page's identity and provenance.

## 6. Why it is command line only

`connector sidecar` is the one operation that writes into the corpus *directory* rather than into
the index. Everything else manicule does to a corpus is read-only, so an unattended surface able
to write into one is a new kind of authority rather than a new operation. It stays where a person
is present — the rule `reset-index`, `backup`, `import` and `collection orphans` are already held
to, and `tests/app/test_surface_parity.py` keeps it there.
