# Confluence ingestion

Design for the Confluence connector. Ticket [#9](https://github.com/mgd43b/manicule/issues/9).

OpenDocuments fetches `body.storage` and runs `html.replace(/<[^>]+>/g, ' ')` over it —
every table, code block and heading collapses into a run of words. This is the subsystem
with the widest gap between what it should do and what it does.

---

## 1. Auth

| Deployment | Method | Header |
|---|---|---|
| Cloud | email + API token | Basic `base64(email:token)` |
| Cloud (multi-user) | OAuth 2.0 3LO | Bearer |
| Server / Data Center | Personal Access Token | Bearer |

Config: base URL, credentials, and an optional space allowlist. Everything is fetched as
the token's user — see §9.

## 2. Discovery and change detection

**Full sync** — enumerate once:

```
GET /wiki/rest/api/content/search
    ?cql=type=page AND space in (ENG,OPS)
    &expand=version
    &limit=100
```

**Incremental** — a per-space watermark of the last successful sync:

```
?cql=type=page AND space=ENG AND lastmodified > "2026-08-09 14:30"
```

`lastmodified` is a first-class CQL field and is sortable. This is the whole point:
OpenDocuments walks entire spaces on every sync and compares version numbers client-side.

**Two traps, both verified:**

- **Pagination is cursor-based, not offset.** Follow `_links.next`; the `start` parameter
  no longer works reliably for search.
- **Cursors contain `+`, which must be escaped as `%2B`** before being sent back. Naive
  URL handling silently breaks pagination partway through a sync. Cursors also expire, so
  a sync must not pause indefinitely mid-page.

Per page, `version.number` is the change token — cheaper than hashing content.

## 3. Deletions — the part most implementations miss

CQL returns what exists. A page deleted since the last sync simply stops appearing, so a
watermark sync **never learns it is gone** and the index keeps serving it forever.

Two mechanisms, both needed:

- **Reconciliation pass** — periodically (weekly) list all page IDs in scope with
  `expand=` empty, diff against indexed IDs, and soft-delete the difference. Cheap: IDs
  only, no bodies.
- **Trash check** — `?cql=type=page AND space=ENG AND label=...` does not cover this;
  the reconciliation diff is the reliable route.

## 4. Fetching content

**Cloud — ADF.** A typed JSON document tree, not markup:

```
GET /wiki/api/v2/pages/{id}?body-format=atlas_doc_format
```

**Server / DC — storage format.** ADF is Cloud-only, so Server falls back to
`body.storage` XHTML and needs a real parser (lxml), not a regex.

Also expand `ancestors` — the page hierarchy, used for breadcrumbs in §7.

**Known Cloud bugs to guard against:** batch page-version endpoints have returned 500 with
ADF requested (CONFCLOUD-80964), and there are reports of ADF returning stale page
content. Validate that the returned `version.number` matches what discovery reported; if
it does not, refetch or fall back to `storage`.

## 5. ADF → chunks

The reason ADF is worth the trouble: these arrive as typed nodes rather than markup to
recover structure from.

| Node | Handling |
|---|---|
| `heading` (level) | Drives the heading hierarchy and chunk boundaries |
| `paragraph` / `text` | Body text; marks carry `code`, `link`, emphasis |
| `table` / `tableRow` / `tableCell` | **Keep whole.** Never split a table across chunks |
| `codeBlock` (language) | **Keep whole**, tagged with language. Never prose-chunk code |
| `bulletList` / `orderedList` | Preserve nesting |
| `panel` (info/warning/note) | Keep the semantic — a warning panel is not ordinary prose |
| `expand` | Collapsed content is still content. **Include it** |
| `blockCard` / `inlineCard` | Links to other pages — a cross-reference graph worth keeping |
| `mediaSingle` / `media` | Attachment reference, resolved in §6 |
| `extension` / `bodiedExtension` | Macros — see below |
| `status` / `date` / `mention` | Inline, render to text |

**Macros are where content hides.** `include` and `excerpt-include` pull content from
other pages; if they are not resolved, that content is missing from the chunk while
appearing present in the UI. Resolve them, with a depth limit and cycle detection.

## 6. Attachments

```
GET /wiki/rest/api/content/{id}/child/attachment
```

Download and route through the normal parser chain — a PDF attached to a page is a PDF.
Each attachment keeps a link to its parent page so citations resolve to both.

OpenDocuments ignores attachments entirely.

## 7. Chunking and breadcrumbs

Chunk on ADF structure, not character counts: split at headings, keep tables and code
blocks intact.

Prefix each chunk with its breadcrumb, built from `ancestors`:

```
ENG > Platform > Auth Service > Token Refresh
```

This is what makes a chunk retrievable when the page title alone is ambiguous — a section
called "Configuration" is useless without knowing which service it configures.

## 8. Citations

```
{base}/wiki/spaces/{spaceKey}/pages/{pageId}/{slug}#{heading-anchor}
```

Confluence derives heading anchors from heading text, so a citation can deep-link to the
**exact section**, not just the page. This is strictly better than the page-level citation
OpenDocuments produces, and it costs nothing since the heading is already known from §5.

## 9. Two operational realities to document

**Rate limits.** Cloud throttles. Back off on 429 and honour `Retry-After`; a full first
sync of a large space will hit it.

**The index is not permission-aware.** Content is fetched as the sync user, so the index
contains everything that user can see, and anyone with search access to manicule can then
retrieve it. That is a real access-control widening and it must be stated plainly in the
docs rather than discovered.

---

## Summary of what changes

| | OpenDocuments | manicule |
|---|---|---|
| Discovery | full space walk every sync | CQL watermark, cursor pagination |
| Deletions | never detected | reconciliation diff |
| Body format | `body.storage` XHTML | ADF on Cloud, parsed storage on Server |
| Extraction | `replace(/<[^>]+>/g, ' ')` | typed ADF node walk |
| Tables | destroyed | preserved whole |
| Code blocks | destroyed | preserved, language-tagged |
| Macros | ignored | resolved, with cycle detection |
| Attachments | ignored | parser chain |
| Context | page title | full ancestor breadcrumb |
| Citation | page URL | deep link to heading anchor |
