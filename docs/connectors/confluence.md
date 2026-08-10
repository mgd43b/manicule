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

The credential is resolved and checked **before the connector is constructed**: a token
missing from both configuration and `$CONFLUENCE_API_TOKEN`, or a Cloud token with no email
beside it, is a startup refusal naming what to set. Discovering it at the first page of the
first sync produces a run that reports progress and indexes nothing.

## 2. Discovery and change detection

**Full sync** — enumerate once, per space:

```
GET /wiki/rest/api/content/search
    ?cql=type in (page, attachment) AND space = "ENG" AND status = current
         order by lastmodified asc
    &expand=version,ancestors,space,container
    &limit=100
```

**Incremental** — a per-space watermark of the last successful sync:

```
?cql=... AND lastmodified >= "2026/08/09 14:25"
```

`lastmodified` is a first-class CQL field and is sortable, so the source does the filtering
and a sync costs what changed rather than the whole corpus.

Five things about that query are load-bearing.

- **One query covers pages and attachments.** `type in (page, attachment)` makes attachments
  watermark-aware and reconcilable on the same terms as pages, instead of a per-page call to
  the attachment endpoint that no watermark can narrow.
- **`status = current` is explicit.** Reconciliation (§3) depends on a deleted page *not*
  being returned. A query that included trashed content would report every deleted page as
  still present, and deletion detection would run, succeed, and find nothing, forever.
- **Every value is a quoted, escaped CQL literal.** A space key or page title containing a
  quote would otherwise end the literal and continue as query syntax — the same hazard as SQL
  injection, and against a search endpoint the result is not an error but results.
- **Timestamps keep the offset the source reported.** CQL evaluates `lastmodified` in the
  instance's own timezone, which no API states. Rather than guess it, the watermark is taken
  from `version.when` — which arrives as ISO-8601 with the instance's offset — and formatted
  back in that same offset. Nothing is converted, so nothing depends on a zone anybody had to
  know.
- **`>=` with an overlap, not `>`.** CQL compares to the minute, so two pages saved in the
  same minute are indistinguishable to it and an exact resume drops whichever was not reached.
  The query starts `watermark_overlap_minutes` (default 5) early; re-enumerating that costs a
  version comparison the pipeline was going to make anyway, and missing a page costs a
  document that stays wrong until something unrelated touches it.

**The space list is enumerated and checked each run.** With no allowlist, every visible space
is synced, so one created since the last run needs no configuration change. With an allowlist,
each key is checked against what is visible and an unknown one is a refusal — CQL answers a
query for a space that does not exist with an empty result set, so a typo would otherwise be a
sync that runs, succeeds, indexes nothing, and leaves reconciliation proposing the deletion of
everything that space ever contributed.

**Three traps in pagination, all verified:**

- **Pagination is cursor-based, not offset.** Follow `_links.next`; the `start` parameter
  no longer works reliably for search.
- **Cursors contain `+`, which must be escaped as `%2B`** before being sent back. Naive URL
  handling silently breaks pagination partway through a sync — an unrecognised cursor comes
  back as results rather than as an error, so some pages are simply never seen.

  **The exception is exactly one parameter wide.** The rest of the link is form-encoded, so a
  `cql` whose spaces arrived as `+` must decode back to spaces; treating *those* as literal
  sends the source a query nobody wrote, which is the same bug mirrored. So `+` is data in
  `cursor` and a space everywhere else, and the round trip is: decode without form rules for
  the cursor, and let the HTTP client percent-encode it back to `%2B` on the way out.
- **Cursors expire.** A consumer that stalls mid-enumeration — a slow embed, a paused
  pipeline — resumes onto a cursor the server has forgotten, and a forgotten cursor can be
  answered with a fresh first page rather than an error, which enumerates the start of a space
  twice and its end never. A cursor held longer than `cursor_lifetime_seconds` (default 300)
  is refused **before the request is sent**, so the run fails legibly and is re-run against an
  unadvanced watermark. A `next` link addressing a cursor already followed is refused for the
  same reason: a loop over a paginated search reads as a very large space.

**`_links.next` is resolved against `_links.base` by concatenation, and the origin is
checked.** An instance served from a context path (`/confluence`) has that path in `base` and
not in `next`, and RFC 3986 resolution of a root-absolute reference discards it. Following a
`base` that names another host would send the sync account's credentials wherever a response
asked, so a link that resolves off-origin stops the enumeration.

Per page, `version.number` is the change token — cheaper than hashing content.

**The watermark advances only on a complete enumeration.** It is a per-space map carried in
`Watermark.metadata`, and a space's entry moves only after that space's walk finishes; a
consumer that abandoned discovery part-way is offered no watermark at all. A watermark built
from a prefix skips whatever the rest of the walk would have returned, and nothing looks for
it again.

## 3. Deletions — the part most implementations miss

CQL returns what exists. A page deleted since the last sync simply stops appearing, so a
watermark sync **never learns it is gone** and the index keeps serving it forever.

Two mechanisms, both needed:

- **Reconciliation pass** — periodically (weekly) list all page IDs in scope with
  `expand=` empty, diff against indexed IDs, and soft-delete the difference. Cheap: IDs
  only, no bodies.
- **Trash check** — `?cql=type=page AND space=ENG AND label=...` does not cover this;
  the reconciliation diff is the reliable route.

**Attachments are reconciled with pages**, because they are enumerated with pages (§2). A
document the pass omits is a document the pipeline deletes, so "attachments are handled
elsewhere" would be a slow way of emptying them out of the index.

**A failure mid-enumeration raises rather than returning what it has.** The ids seen so far
are a prefix, not the truth, and diffing a prefix against the stored set marks everything not
yet reached as deleted: one transient error, and the corpus is soft-deleted
([`ingest.md`](../ingest.md) §11.1 carries the pipeline's half of this — clean completion
only, a deletion ceiling, and soft delete only). The connector's half is to fail loudly, so
that "everything is gone" and "nothing answered" never look alike.

## 4. Fetching content

**Cloud — ADF.** A typed JSON document tree, not markup:

```
GET /wiki/api/v2/pages/{id}?body-format=atlas_doc_format
```

**Server / DC — storage format.** ADF is Cloud-only, so Server falls back to
`body.storage` XHTML, which is parsed with a real HTML engine rather than by stripping
angle brackets. It is routed as `text/html`, so the parser chain's HTML parser
(`selectolax`, `docs/parsing.md` §7) handles it — a second parser for a dialect of the same
thing would be two implementations of one job.

Ancestors — the page hierarchy used for breadcrumbs in §7 — come from discovery, which
expands them for every page it returns, so the fetch needs no second call. A ref built
somewhere else (a re-fetch, a targeted single-page sync) carries none, and then the Cloud
ancestors endpoint is asked; an ancestor whose title it omits is skipped rather than filled
in with an id, and the document records that its breadcrumb is incomplete.

**Known Cloud bugs to guard against:** batch page-version endpoints have returned 500 with
ADF requested (CONFCLOUD-80964), and there are reports of ADF returning stale page
content. Validate that the returned `version.number` matches what discovery reported; if
it does not, refetch, and then fall back to `storage` — a different code path on the source's
side, which is the whole reason it is worth trying.

**If every attempt is still stale, the body is stored under the version it actually carries.**
This is the part that makes it self-healing, and it is worth being exact about. Recording the
version that was *asked for* against older bytes would make the document permanently stale:
the next sync would compare the two, find them equal, and skip the page forever. Recording
what came back leaves the stored token behind what discovery reports, so the next sync fetches
it again. The disagreement is recorded on the document either way.

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

Six things the implementation settled:

- **Both formats, not just ADF.** Server and Data Center have the same macros and no ADF, so
  storage format is expanded too. Its macro elements are located with `html.parser` — a real
  parser, reporting the exact span of each element — and the raw source is spliced at those
  spans. Everything that is not a macro therefore reaches the parser chain byte-identical to
  what Confluence returned; rebuilding the page from a parse tree would rewrite entities,
  attribute quoting and self-closing tags across the whole document in order to change one
  element.
- **A nested include is still an include.** One inside an `info` panel is expanded like any
  other. A scan that reported only outermost elements would leave it behind.
- **The target is read from whichever parameter carries it** — the macro's default unnamed
  parameter, or `page`, `pageTitle`, `name`, or a content id. Editors and templates disagree,
  and reading only the editor's own spelling leaves templated pages short.
- **The cycle path starts with the page being fetched**, so a page that includes itself is
  caught on its first macro rather than after a round of duplication.
- **`excerpt-include` takes only the excerpt.** Splicing the whole target instead would put
  content on the including page that a reader of it never sees — the same defect as the
  missing content, mirrored.
- **Nothing unresolved is dropped in silence, and nothing invented is put in its place.**
  Every macro left unexpanded — a cycle, the depth limit, a target this account cannot see, a
  target with no excerpt, or expansion turned off in configuration — is recorded on the
  document with the reason. Substituting manicule's own words would be worse than the gap: it
  would put them inside a quotation attributed to the source.

Included content is spliced at the macro's position, which is where Confluence renders it.
That matters for citations as well as for text: a heading arriving through an include is a
heading on the *rendering* page, and Confluence derives its anchor there — so the deep link in
§8 addresses the page a reader would open.

## 6. Attachments

Discovered with pages rather than through `GET /wiki/rest/api/content/{id}/child/attachment`,
because the CQL query in §2 already covers `type = attachment` — one enumeration, watermarked
and reconcilable, instead of one extra call per page that no watermark can narrow.

Download and route through the normal parser chain — a PDF attached to a page is a PDF.
Each attachment keeps a link to its parent page so citations resolve to both.

Two details worth stating:

- **The size ceiling is enforced against the bytes that arrive**, never against the declared
  `Content-Length`. The declared length is the source's claim about the response, and the
  point of a ceiling is to survive a claim that turns out to be wrong.
- **The media type is the source's, then the download's, then the filename's.** Confluence's
  `metadata.mediaType` first, the response's `Content-Type` next, and the filename extension
  last; nothing is guessed ahead of what was actually said.

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

**The connector supplies the breadcrumb; the chunker assembles it.** `ancestors` on the
document is the space key followed by the ancestor titles, and the chunker appends the page
title and the heading path itself (`manicule.chunking.chunker`). Putting the title in
`ancestors` too would reach the embedder as emphasis nobody intended, since adjacent repeats
are collapsed but a duplicated element two places apart is not.

An attachment's breadcrumb ends at its parent page — `ENG > Token Refresh > diagram.pdf` —
rather than repeating the page's own ancestors, which the attachment's search result does not
carry.

## 8. Citations

```
{base}/wiki/spaces/{spaceKey}/pages/{pageId}/{slug}#{heading-anchor}
```

Confluence derives heading anchors from heading text, so a citation can deep-link to the
**exact section**, not just the page. This is strictly better than the page-level citation
OpenDocuments produces, and it costs nothing since the heading is already known from §5.

The page URL is taken from the source's own `_links.webui`, joined to `_links.base`, rather
than assembled from a slug. `storage.md` §4.2 declines to claim the slug is title-derived and
therefore unstable; not constructing it settles the question either way, and identity rests on
the page id regardless.

## 9. Two operational realities to document

**Rate limits.** Cloud throttles. Back off on 429 and honour `Retry-After`; a full first
sync of a large space will hit it.

Both forms of `Retry-After` are honoured — a delay in seconds and an HTTP-date. A client that
reads only the first waits zero seconds for the second and retries straight back into the same
limit, which turns a throttled sync into a throttled sync that is also a hot loop. A pause
longer than `max_retry_after_seconds` (default 120) stops the run instead of being slept
through: a sync asleep for an hour inside one HTTP call is indistinguishable from a hung one,
and stopping costs re-enumeration and nothing else, because the watermark does not advance.
Backoff elsewhere is deterministic and unjittered — jitter keeps a *fleet* of clients out of
lockstep, and one self-hosted index syncing one Confluence is not a fleet.

**The index is not permission-aware.** Content is fetched as the sync user, so the index
contains everything that user can see, and anyone with search access to manicule can then
retrieve it. That is a real access-control widening and it must be stated plainly in the
docs rather than discovered.

Concretely, and in the terms someone deciding this needs:

- **Confluence's space and page restrictions do not travel with the content.** A page
  restricted to three people, indexed by an account that is one of them, is retrievable by
  every manicule user — text, attachments and the answers generated from them.
- **The sync account is the blast radius.** Point the connector at an account with access to
  exactly what the index is meant to hold; an admin token indexes everything an admin can see.
  A 403 during a sync is therefore not necessarily a fault — it is the boundary working — and
  the connector says so rather than reporting a broken credential.
- **It compounds with retained source bytes** (`storage.md` §7.4): the retained originals are
  byte-identical to what was fetched, and the same audience can reach them.
- This is stated in the configuration model's own documentation as well as here, because a
  configuration file is where somebody decides which spaces to point it at.

## 10. Not yet verified against a live instance

The connector is built and tested against recorded behaviour rather than a real Confluence:
there are no credentials in this repository, and the suites drive a synthetic instance that
reproduces the traps above (a cursor containing `+`, a version disagreement, a macro cycle, a
429 with `Retry-After`, a page deleted between syncs). Where a real instance could still
differ, it is here — check these first when one is available:

| Assumption | What to check | If it is wrong |
|---|---|---|
| `status = current` is accepted by CQL on both deployments | A search returns results rather than a 400 | Deletion detection is the thing at risk: without it, trashed pages may still be returned |
| `order by lastmodified asc` is accepted alongside cursor pagination | Same | Drop the ordering; enumeration becomes non-deterministic but stays correct |
| `_links.next` carries the full query and a raw `+` in the cursor | Log one link verbatim | The `%2B` handling is unnecessary but harmless; a *different* encoding would need its own handling |
| Cloud v2 `GET /pages/{id}/ancestors` returns titles, root first | Fetch one page whose ref carries no ancestors | Breadcrumbs from that path are reversed or short; discovery's own expansion is unaffected |
| `metadata.mediaType` is present on attachment search results | One attachment's discovered media type | Falls back to `Content-Type`, then the filename — already the design |
| Search cursors survive at least `cursor_lifetime_seconds` | A slow sync of a large space | Lower the setting; the failure is a refusal, not corruption |
| A page in the trash stops appearing in CQL results | Trash a page, re-run reconciliation | Deletion is never detected, and the index serves the page forever |

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

---

## 11. Where this lives

| Concern | Module |
|---|---|
| Configuration and the credential refusal | `manicule/connectors/config.py` |
| CQL construction, quoting, watermark timestamps | `manicule/connectors/cql.py` |
| `_links.next`, the `%2B` rule, origin checking | `manicule/connectors/pagination.py` |
| Auth headers, 429/5xx retry, downloads, paging | `manicule/connectors/client.py` |
| `include`/`excerpt-include`, both body formats | `manicule/connectors/macros.py` |
| `discover`, `fetch`, `reconcile`, watermarks | `manicule/connectors/confluence.py` |
| Registration through the `manicule.plugins` entry point | `manicule/connectors/plugin.py` |

The ADF node walk is **not** here: it is `manicule/parsers/adf.py`, reached through the parser
chain like any other format, and so are attachments. The connector's job ends at handing over
bytes and saying honestly what they are.

**One seam the `Connector` protocol does not name.** `discover` takes a watermark; nothing in
the protocol returns the next one. `ConfluenceConnector.watermark` is a read-only property
carrying what to persist if the run completed cleanly — an addition on the implementation, not
a widening of the protocol, since a caller working from `Connector` never sees it. The ingest
pipeline reads it after a clean run ([`ingest.md`](../ingest.md) §13.1 records the watermark
before and after a run, and §13.2 makes advancing it conditional on the run being clean).
