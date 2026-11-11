# Trilium Notes: the `trilium` connector

How manicule indexes a [Trilium Notes](https://triliumnotes.org) server (the maintained
`TriliumNext/Trilium` line) through its REST API, ETAPI. Related: the Confluence connector
([`confluence.md`](confluence.md)), whose scoping and deletion rules this page reuses wherever
the problem is the same, and the mbox connector ([`mbox.md`](mbox.md)), whose sensitivity
reasoning it follows.

> **Status: implemented.** §11 names the code for each section. The connector is built and tested
> against a synthetic server, not a live one: §9 lists every statement here that rests on ETAPI's
> published description and not on an observed response.

A person who keeps notes in Trilium has a tree of notes in one SQLite database, reachable only
through the app. The notes are hierarchical, a note can sit under several parents, and every
note has a type that decides what its content is. This page decides how a note becomes a
document, what identifies it, how change and deletion are found when the API has no change feed,
and what an installation holding personal notes must refuse to do.

The connector ends at `RawDocument`. A note's content goes through the existing parser chain;
there is no Trilium-specific parsing beyond choosing a media type (§3).

---

## 1. Shape, and configuration

**`type = "trilium"` reads one Trilium server over ETAPI, read-only, and yields one document per
note.** It never writes to the server: it makes only `GET` requests.

```toml
[connectors.notes]
type = "trilium"
enabled = true

[connectors.notes.options]
base_url = "https://notes.example.org"   # required; may carry a context path (§2)
token_env = "TRILIUM_ETAPI_TOKEN"        # the default; or `token = "..."` (§2)
root_note_ids = []                       # notes whose subtrees are indexed; empty means all (§5)
exclude_labels = ["manicule-skip"]       # a note carrying one, and its subtree, is never read (§5)
include_archived = false                 # notes labeled #archived (§5)
max_notes = 100_000                      # a sweep this large is presumed cut off and refused (§6.1)
```

`TriliumConfig` is frozen and rejects any other key. `request_timeout_seconds`, `max_retries`
and `max_retry_after_seconds` tune the HTTP layer exactly as Confluence's do. `base_url` is
parsed by the same function as Confluence's (`normalize_base_url`): `http` or `https`, a
hostname, an optional context path (Trilium is often served under `/trilium` behind a reverse
proxy), and no userinfo, query or fragment.

There is deliberately **no date filter, no `since`, no note-type list, and no size setting of
its own.** A filter that applies to a full walk but not to a re-walk is the failure
`gmail.md` §3 states for labels; the types worth indexing are a property of the format, not a
preference (§3); and the ceiling on one note is the installation's `ingest.max_fetch_bytes`,
as it is for mbox.

---

## 2. Auth

ETAPI authenticates with a token created in Trilium under *Options, then ETAPI*. The connector
sends it as `Authorization: Bearer <token>`, which servers accept from v0.93.0 (§9). A token
resolved from `token_env` is never written to configuration or logs, an explicit `token` is never
overwritten by the environment, and a blank variable counts as unset, exactly as
`resolve_credentials` does for Confluence.

- **There is no login flow.** ETAPI also has `POST /auth/login`, which exchanges the account
  password (and a TOTP code where enabled) for a token. manicule never asks for a password and
  has nowhere to put one, which is why this connector takes the token a person created and
  nothing else.
- **The credential is checked before the connector is constructed**, as in `confluence.md` §1.2:
  `build_trilium` raises `ConfigError` naming the setting and the variable when there is no
  token, rather than discovering it at the first request.
- **The HTTP layer is its own client** (`connectors/trilium_client.py`), the Gmail client's shape
  rather than Confluence's: one static credential, no browser session, no pagination. It shares
  the retry arithmetic (`retry.py`) and the origin helpers, and nothing else.
- **A redirect is never followed.** ETAPI does not redirect, so a redirect is always something
  standing between the connector and Trilium, most often an identity-aware proxy answering an
  unauthenticated request by sending the client to its sign-in. One to another host raises
  `UntrustedLinkError` and one on the same origin raises `ConnectorError`; neither is followed,
  because the token travels with the request.
- **A sign-in page is not a note.** The JSON endpoints refuse a sign-in page because it does not
  parse. A note's content endpoint has no such backstop, since a proxy's sign-in page is a `200`
  of `text/html` and so is a text note. The client therefore requires the response's
  `Content-Type` to be the `mime` the note was swept with, which refuses the page for every note
  that is not HTML. **For a text note the media type cannot tell them apart**, and the defense is
  the order of events: the sweep in the same run, a JSON request that a sign-in page cannot
  satisfy, has just proved that the token and the path reach ETAPI.
- **The token can read every note on the server it was created on**, including protected-note
  titles, whatever `root_note_ids` says; the scope is enforced by this connector, not by Trilium
  (§5, §7).

---

## 3. What becomes a document

ETAPI's `Note.type` is one of `text`, `code`, `render`, `file`, `image`, `search`, `relationMap`,
`book`, `noteMap`, `mermaid`, `webView`, `shortcut`, `doc`, `contentWidget`, `launcher`.

| `type` | Indexed | Delivered as |
|---|---|---|
| `text` | yes | the note's HTML, `text/html`, under an `<h1>` of its title |
| `code` | yes | the note's text. A `mime` of `text/x-markdown` is `text/markdown`; any other source type is `text/plain` |
| `mermaid` | yes | `text/plain` |
| everything else | no | containers (`book`, `doc`), saved queries (`search`), drawings and maps, images, UI configuration, and `file` notes |

A `type` outside ETAPI's published list is a Trilium newer than the connector, and it is
reported at warning level by name and not left out quietly.

`file` notes (an attached PDF or office document) are left out of this version rather than
indexed behind a setting. They would go through the parser chain as bytes, but they need a size
and type policy that notes do not, and nothing in this design depends on adding them (§8).

**A note's title is its first heading.** The connector wraps a `text` note's HTML in a minimal
document whose first element is `<h1>` of the title, as the mail parser makes the subject the
first heading, so a search for a note by name finds it and a citation has a heading to anchor
to. The wrapper is the connector's; the body is delivered unmodified. A note with no text is still
a document with its heading: whether it is empty is not knowable from the sweep, and the title is
searchable.

**What is deliberately not translated:** `<a class="reference-link">` internal links and
`include-note` sections are left as the HTML has them, relations (`~internalLink` and the like)
are attributes and not followed, the images Trilium embeds as `api/images/…` are not fetched, and
note attachments (`GET /notes/{id}/attachments`) are not indexed.

**Trilium's own notes are not on any path.** Their ids begin with an underscore (`_hidden`, the
share and launcher configuration), and the walk that places notes neither starts at one nor
enters one. A note that exists only beneath them is not a document. A note *also* cloned into
one is placed by its real parents: Trilium clones a bookmarked note under the launch bar, inside
`_hidden`, and judging the system tree on every path would silently drop every bookmarked note.
The `root` note itself is never a document either.

---

## 4. Identity and citations

**`source_id` is Trilium's `noteId`.** It is stable under rename, move, re-parenting and cloning,
which are the ordinary things a person does with a note, and a content hash would turn each of
them, or any edit, into a delete and a create. This is the Confluence choice (the page id) and
not the mbox one: a mail message has no id that is both present and unique, and a Trilium note
always has one.

**`uri` is `<base_url>/#root/<noteId>`**, the address the web app opens. Whether the app resolves
a bare id in that position is checked in §9. The title and the path are never in the `uri`: they
change, and a citation must not.

**A note under several parents is one document.** The connector delivers it once, with its
**breadcrumb taken from the shortest path to the top, ties broken by the smaller `noteId`**, so two
runs agree. The breadcrumb is the record's `section_path`: the titles from the top down to the
note's parent, without `root`, without the note's own title, and without a protected ancestor,
whose ETAPI title is the placeholder `[protected]` and names nothing. Every parent is recorded in
metadata as `trilium_parent_note_ids`.

**The version token is a digest of everything that is part of the document's text:** the
representation version, `blobId` (Trilium's content hash for the note), `utcDateModified`, the
title, `type`, `mime`, the breadcrumb and the labels. `blobId` alone would miss a rename, which
changes the heading and the breadcrumb without changing a byte of content, and
`utcDateModified` rides along so that a server whose `blobId` were stable across an edit would
still be re-read. Discovery does not fetch to decide; it compares tokens, as `DiscoveredDoc`
requires.

**Metadata carries** `trilium_note_id`, `trilium_type`, `trilium_mime`, `trilium_labels` (the
values of each label name, as a list, because Trilium allows several labels with one name), `trilium_parent_note_ids`, and in the provenance record the title, the
canonical address, `blobId` as the version, `utcDateCreated` and `utcDateModified` as the
note's own dates, and the breadcrumb. A title the record cannot hold (a control character) is
cleaned of it, and a record that cannot be built at all leaves the note indexed with an
`unavailable_reason`.

---

## 5. Scope, and what is excluded

Two settings narrow what is read, and a third governs one kind of note.

- **`root_note_ids`** admits a note when **any** of its paths to the top passes through one of
  them. Empty admits everything not excluded below.
- **`exclude_labels`** removes a note when it carries one of the labels, or when **any** of its
  paths passes through a note that does. The label is matched by the connector over each note's
  own attributes; it does not depend on Trilium's attribute inheritance (§9), because a privacy
  boundary that depends on a server feature being configured a certain way is not one.
- **`include_archived`** governs notes labeled `#archived`, handled as an exclude label that
  can be switched off: the note and its subtree. The default leaves them out, because archiving
  is how a person says "not part of what I look at".

**Exclusion wins over inclusion, and it is judged on every path.** A note cloned into an included
subtree and also under an excluded note is excluded. The failure this avoids is a note someone
marked private being indexed because a second parent was not marked. It costs the opposite
error, a note held out because one of its parents was, and that one is visible to the person who
made the clone and fixed by removing the label.

**Protected notes are never read.** Trilium encrypts a protected note and decrypts it only inside
a session the user unlocked with a password. ETAPI does not offer that session, so the content is
unavailable, and the connector does not try: it skips every note with `isProtected: true` before
any request for its content. The title of a protected note is the literal `[protected]` in ETAPI
responses; it is not used. Each sweep logs **how many protected notes were skipped**, with the counts of notes out of
scope, of a type that is not indexed, and not reachable from the top of the tree (a count, as the
mbox connector logs its skips), because "my notes are indexed" is otherwise silently untrue
for exactly the notes a person was most careful with.

Changing any scope setting changes the **scope fingerprint**, as `MboxConfig.scope_fingerprint`
does, so a run under a different scope is not taken as continuing the last one: a note newly
outside scope is retired by the next inventory and not left indexed. The token is not in it, so
rotating one is not a new source.

---

## 6. Discovery, change detection, deletion

**ETAPI has no change feed and no cursor.** `GET /notes` searches and returns every match in one
response, with `limit` as the only bound. There is no "modified since" parameter, and
`GET /notes/history` lists recent changes with an unspecified window. The design does not lean on
either, and every run is a full inventory.

### 6.1 The sweep

**One request returns every note's metadata and no content.** `Note` objects carry `noteId`,
`title`, `type`, `mime`, `isProtected`, `blobId`, `utcDateModified`, `parentNoteIds` and
`attributes`, which is everything §3 to §5 decide on. The sweep is a single `GET /notes` with a
`search` expression that matches every note, with no ordering asked of the server. Content is requested
(`GET /notes/{id}/content`) only for a note the pipeline decided to fetch, because its token
changed.

**A response that might be short is refused.** The request carries `limit = max_notes`. A response
with that many notes may have been cut off, and `reconcile` deletes what it does not see, so the
sweep raises and says to raise `max_notes`. Raising deletes nothing. The sweep is one response with
no cursor, so `request_timeout_seconds` defaults to 120 and not the 30 an ordinary API call gets,
and a very large vault holds that response in memory while it is read. A sweep that cannot be
proved complete is worse than none, which is `mbox.md` §5.4's rule: absence is deletion. The
alternative, splitting the sweep by subtree when it reaches the limit, was rejected for a server
cap nobody has observed (§10).

**Every entry is read strictly.** An entry without a `noteId`, or whose `parentNoteIds`,
`attributes` or `isProtected` are not the shapes ETAPI documents, fails the sweep. Dropping it
would hide a note from the inventory, and hiding a note deletes it.

### 6.2 `discover`

**Every note in scope is yielded, with its version token, on every run.** The watermark is
accepted and deliberately not used to skip notes, the filesystem connector's rule
and for the same reason: the token does the skipping, and the clock cannot.
A rename or move of an ancestor changes every descendant's breadcrumb without touching the
descendant's own `utcDateModified`, which a date filter would miss. The connector declares
`enumerates_full_inventory`, so every manifest it produces is a full inventory with deletion
authority.

`Connector.watermark` is the newest `utcDateModified` of the last **completed** sweep and is
informational; a run that ends early offers none, as the protocol requires.

### 6.3 `fetch`

`fetch` reads `GET /notes/{id}/content` and returns the bytes with the media type §3 chose. It
judges the note against the most recent sweep's tree, **which it takes afresh once that sweep is
ten minutes old** (`INVENTORY_MAX_AGE_SECONDS`), so a label added or a note protected after
discovery reaches a read within a bounded time and not whenever the next discovery runs. Reads
that arrive with no usable sweep share one request. It needs no state in the discovery journal
beyond the note id. **A note the sweep did not
admit is not served even if a reference names it**: deleted, protected, excluded, or moved out of
every root since, it raises `NotFoundError`, and the pipeline reads that as the inventory having
changed. The same exception is raised for a note that was swept and then deleted or protected
before it was read, which is the ordinary race. A note larger than `ingest.max_fetch_bytes` is
refused while it streams and never loaded to be rejected.

### 6.4 `reconcile`

`reconcile` yields the `noteId` of **every note in scope now**, from a fresh sweep with the same
filters. It trusts no partial sweep: a response that may be truncated, an unreachable server, a
401 or a malformed entry raises, and the pipeline's pass records the error and deletes nothing.
When a note has been deleted in Trilium, or archived, labeled with an exclude label, moved out of
every `root_note_ids` subtree, or protected, it is absent and its document is soft-deleted; the
deletion ceiling (`ingest.reconcile_max_delete_fraction`) holds a large removal as a proposal for
`manicule connector reconcile NAME --confirm`, as for every source. Trilium's own soft deletion
(`isDeleted`, recoverable with `POST /notes/{id}/undelete`) is not special: an undeleted note
simply returns to the sweep.

**`/notes/history` is not used for deletion.** It reports `current_isDeleted`, but it is a list of
recent events with no stated horizon, and a deletion that fell outside it would be missed forever.
Absence from a complete sweep is the only signal that cannot lose one.

---

## 7. Sensitive

**`trilium` registers as sensitive**, with `sensitive=True` beside its `config_model` exactly as
`gmail` and `mbox` do. A person's notes are where passwords, health records, drafts and second
thoughts end up, and the index is not permission-aware: whoever can search the workspace reads
every note in it. The consequences are the existing ones and are not repeated (`gmail.md` §10,
`mbox.md` §7): an installation refuses to serve with `security.auth.mode = "none"` while an enabled
`trilium` instance exists, the first sync marks the source before it enumerates anything, and
`manicule document delete-source <name> --yes` lifts it. Notes belong in a workspace of their own,
and models see the text.

**The token's reach is wider than the scope.** ETAPI tokens are all-or-nothing: the connector sees
protected-note titles and every subtree on the server regardless of `root_note_ids`. The scope in
§5 is enforced here and stops a note from being indexed; it does not stop the connector from having
been *offered* it. A person who needs the server itself to refuse should run a separate Trilium
for that material, which is the same advice `confluence.md` §9 gives for content some people may
not see.

---

## 8. What is tested, and what is deferred

**Tests run against a synthetic server**, `tests/connectors/fake_trilium.py`, in the manner of
`fake_confluence.py`, and `assert_connector_contract` runs over it. Every title, label and host is
`example.org` or invented. It reproduces: a note cloned under several parents; an ancestor renamed
(§6.2); a protected note; an archived note; an excluded subtree with a clone escaping it; a
bookmarked note cloned into the system tree; a search silently cut off at a limit; a note deleted
or protected between the sweep and the fetch; a label added after discovery; concurrent reads
with no sweep; retry exhaustion for 429, 5xx and a dropped connection; a note exactly at the fetch
cap; a 429 with `Retry-After`; a redirect to another host
and one on the same origin; a proxy sign-in page served as a note's content; a token rejected; and
a server under a context path. `tests/connectors/test_trilium_pipeline.py` drives the connector
through the real pipeline and HTML parser: a note is indexed under its title, an unchanged one is
not read again, a renamed folder re-reads only the notes inside it, a deleted note leaves at
reconciliation, and a truncated sweep deletes nothing.

**Deferred, each bounded and independent of the above:**

- **Attachments** (`GET /notes/{id}/attachments`, `…/content`), indexed as members of their note,
  the way Confluence's follow their page.
- **`file` notes**, through the parser chain, with a size and type policy.
- **Internal links**, rewriting `reference-link` hrefs to the target's `uri` so a citation can
  follow one.
- **Revision history** (`GET /notes/{id}/revisions`). Not a document a person searches for.
- **An offline reader for a copy of `document.db` or an export**, as `confluence_snapshot` is to
  Confluence. §10 says why it is not the first path.

---

## 9. Not yet verified against a live server

The connector is built and tested against ETAPI's published description (the OpenAPI document in
the Trilium repository and the ETAPI user guide), not against a response from a running server.
Check these first when one is available:

| Assumption | What to check | If it is wrong |
|---|---|---|
| **The expression `note.noteId != ''` matches every note** | Run the sweep against a vault of a few thousand notes | The sweep returns nothing or errors. `SWEEP_EXPRESSION` is the one place to change |
| **`limit` is honored, and a response shorter than `limit` is complete** | Ask for `limit=10` on a vault of 50 | Truncation detection fails open: a short response is trusted. This is the row that matters most, since reconcile deletes on absence |
| **`includeArchivedNotes=true` returns archived notes**, with their `#archived` label in `attributes` | Archive a note and sweep | The connector judges archiving itself from the label. If archived notes never arrive, `include_archived` has no effect and they are simply absent |
| **The hidden subtree and system notes appear in search**, with ids beginning `_` | Run the sweep and list ids starting `_` | If they do not appear, the rule in §3 is harmless. If other system notes appear without the prefix, it needs widening |
| **`Note.attributes` lists a note's own labels and not inherited ones** | Label a parent `#manicule-skip(inheritable)`, read a child | §5 already walks the tree itself and does not depend on this; if inherited labels *are* listed the metadata would record labels the child does not carry |
| **`blobId` changes whenever content changes, and `utcDateModified` advances on an edit** | Edit a note, read both; rename it, read both | Both are in the token, so one holding is enough. If neither moves on a content edit the edit is missed |
| **`GET /notes/{id}/content` returns the note's `mime` as its `Content-Type`**, and the editor's HTML for a text note | Fetch a text note, a code note and a Markdown note | The check in §2 refuses every note whose type it does not recognize, loudly. It would need to compare the media type's base only, or the note's type |
| **A bare `#root/<noteId>` opens the note in the web app** | Open one | The `uri` needs the full path, which the breadcrumb walk already computes |
| **A protected note's content request fails `400 NOTE_IS_PROTECTED`**, and `isProtected` is true in the sweep | Fetch one | §5 skips before asking, so this is harmless unless `isProtected` is *absent*, which fails the sweep loudly rather than putting ciphertext in the index |
| **`Authorization: Bearer` is accepted** by the TriliumNext line from v0.93.0 | Call any endpoint with it | Send the bare token as `Authorization`, which older servers take |
| **Trilium does not redirect an ETAPI request** | Run a full sync | A same-origin redirect is refused (§2). It would be the connector refusing a normal answer |

---

## 10. What was rejected, and why

- **A content hash as the identity.** §4: it turns every edit into a delete and a create, taking
  tags, collections and history with it. `noteId` is stable and unique.
- **`GET /notes/{id}/export`** (a ZIP of a subtree). It is one response for a whole subtree, so a
  note's change is the subtree's change and no per-note version token exists. It also drops
  identity (files are named by title).
- **Reading `document.db` directly.** It couples the connector to a schema the project does not
  promise, contends with a running server's lock, and sees protected notes as ciphertext with no
  clean refusal. It is the right basis for an *offline* reader (§8), and the wrong one for a live
  server.
- **Trilium's sync protocol.** A private protocol between Trilium instances, not an integration
  surface.
- **Crawling published (`share`) pages.** Only a subtree a person has chosen to publish, rendered
  without its structure, and with no deletion signal.
- **`/notes/history` as the change feed or the deletion signal.** §6.4.
- **A `utcDateModified` filter as the incremental query, with a tree digest to patch its blind
  spot.** The first design. ETAPI's published search documentation does not describe a date
  filter, the sweep already fetches every note's metadata so the filter saves no request, and the
  digest existed only to repair the case the filter could not see. Yielding every note and
  letting the version token decide is less code and has no blind spot.
- **Splitting a possibly truncated sweep by subtree.** Complexity for a server cap nobody has
  observed. A sweep that reaches `max_notes` raises, and raising deletes nothing.
- **Reusing the Confluence client.** It is bound to `ConfluenceConfig`, to browser-session
  credentials and to Confluence's messages. §2.
- **A `since`, `types`, `labels`, `include_files` or `max_note_bytes` setting.** §1.
- **Relying on Trilium's attribute inheritance for exclusion.** §5.
- **Indexing protected notes by asking for a password.** manicule has nowhere to put one.

---

## 11. Where this is implemented

| Part | Module |
|---|---|
| Configuration, scope fingerprint, token resolution (§1, §2, §5) | `manicule/connectors/config.py` (`TriliumConfig`, `resolve_trilium_token`) |
| ETAPI requests: sweep, content, retry, redirect and media-type refusal (§2, §6) | `manicule/connectors/trilium_client.py` |
| The connector: judging the sweep, tree, breadcrumbs, tokens, `discover`, `fetch`, `reconcile` (§3-§6) | `manicule/connectors/trilium.py` |
| Registration, as a sensitive type (§7) | `manicule/connectors/plugin.py` (`build_trilium`) |
| Synthetic server (§8) | `tests/connectors/fake_trilium.py` |
| Tests (§8) | `tests/connectors/test_trilium.py`, `test_trilium_pipeline.py`, `test_trilium_factory.py` |
| Capability entry | `CAPABILITIES.md`, beside `gmail` and `mbox` |
