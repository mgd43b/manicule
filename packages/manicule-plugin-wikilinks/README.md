# manicule-plugin-wikilinks

Turns `[[wikilinks]]` in markdown into typed `chunk_relations` edges, so a corpus that already
carries a graph in its prose becomes one that can be queried as a graph — without anybody
editing a document.

```toml
[plugins]
middleware = ["wikilinks"]

[plugins.config."middleware.wikilinks"]
inbound_limit = 200      # how far the forward-reference search looks; 0 switches it off
sources = ["memories"]   # which sources a link may resolve into; empty means all of them
```

`sources` matters as soon as an installation indexes more than the memory corpus. A slug is a
filename stem and stems are not unique across sources, so an unscoped link can resolve into a
wiki page that happens to share a name — a wrong edge rather than a missing one.

## What it writes

Two relation types, because two things are being written and they do not mean the same thing:

| In the document | Type | Read as |
|---|---|---|
| `The client retries twice. See [[project_backoff]].` | `mentions` | these documents are about related things |
| `- [[project_backoff]]`, `relates_to [[x]]`, `Related: [[a]], [[b]]` | `links_to` | the author asserted a relationship |

A link may name a document that does not exist yet — that is a deliberate part of the writing
convention, a marker of something worth writing later — so an unresolved link is normal rather
than an error. It is resolved from the other end: when the target is published, the plugin looks
for stored documents that link to it and writes those edges then.

Targets are matched after normalization, because the corpus is inconsistent and both spellings
are in real use: `[[feedback-commit-everything]]` and `[[feedback_commit_everything]]` are one
target, as is `[[Feedback Commit Everything.md]]`. `[[target|alias]]` and `[[target#heading]]`
address `target`.

## Why it is a plugin

`[[wikilink]]` is a memory-and-Obsidian idiom, not a markdown one. CommonMark has no such
syntax, so a corpus of ordinary documentation that happened to contain double brackets would
acquire a graph nobody asked for. The convention stays out of the core parser, and the extension
path gets exercised on a component that does real work.

## Versioning, and what happens when the rules change

The rules — what counts as a link, how two spellings are matched, where the line runs between a
declared link and a mention — live in `links.py`, and this package digests that file into the
extractor's identity. manicule records that identity per document, so correcting a rule makes
every already-scanned document visibly stale and the repair selector finds them:

```bash
manicule document reindex --stale-relations   # --dry-run first, to see the selection
```

That is the same rung `--stale-glossary` occupies: it reads stored chunks and writes rows, with
no parser, no connector and no embedder in it. On the run that first enables this plugin every
document records `NULL`, so the selection is the whole corpus — which is exactly what a first
enable should select.

A document that contains no links records the fingerprint with **no edges**, which is a
different fact from one nothing has scanned. Collapsing the two would re-select every link-free
document — most of a corpus — on every repair, for ever.
