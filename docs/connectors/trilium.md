# Trilium Notes: the `trilium` connector

Design for a connector that indexes a [Trilium Notes](https://triliumnotes.org) server
(the maintained `TriliumNext/Trilium` line) through its REST API, ETAPI. Related: the Confluence
connector ([`confluence.md`](confluence.md)), whose scoping, credential and deletion rules this
page reuses wherever the problem is the same, and the mbox connector ([`mbox.md`](mbox.md)),
whose identity and sensitivity reasoning it follows.

> **Status: design, not implemented.** Nothing below exists in the code yet, and nothing has been
> run against a live Trilium. §9 lists every statement here that rests on the published API
> description and not on an observed response; the build starts by checking those.

A person who keeps notes in Trilium has a tree of notes in one SQLite database, reachable only
through the app. The notes are hierarchical, a note can sit under several parents, and every
note has a type that decides what its content is. The index cannot search any of that today.
This page decides how a note becomes a document, what identifies it, how change and deletion are
found when the API has no change feed, and what an installation holding personal notes must
refuse to do.

The connector ends at `RawDocument`. A note's content goes through the existing parser chain; there
is no Trilium-specific parsing beyond choosing a media type (§3).

---

## 1. Shape, and configuration

**`type = "trilium"` reads one Trilium server over ETAPI, read-only, and yields one document per
note.** It never writes to the server: of ETAPI's endpoints it calls only `GET`.

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
include_files = false                    # file notes: PDFs, office documents (§3)
max_note_bytes = 10_000_000              # a note larger than this is skipped and counted (§4)
```

`TriliumConfig` is frozen and rejects any other key. `base_url` is parsed by the same function as
Confluence's (`normalize_base_url`): `http` or `https`, a hostname, an optional context path
(Trilium is often served under `/trilium` behind a reverse proxy), and no userinfo, query or
fragment.

There is deliberately **no date filter, no `since`, and no note-type setting**. A filter that
applies to a full walk but not to a re-walk is the failure `gmail.md` §3 states for labels, and
the types worth indexing are a property of the format, not a preference (§3).

---

## 2. Auth

ETAPI authenticates with a token created in Trilium under *Options → ETAPI*. The connector sends it
as `Authorization: Bearer <token>`, which the server accepts from v0.93.0, and refuses an older one
(§9). A token resolved from `token_env` is never written to configuration or logs, and an
explicit `token` is never overwritten by the environment, exactly as `resolve_credentials` does
for Confluence.

- **There is no login flow.** ETAPI also has `POST /auth/login`, which exchanges the account
  password (and a TOTP code where enabled) for a token. manicule never asks for a password and
  has nowhere to put one, which is why this connector takes the token a person created and
  nothing else.
- **The credential is checked before the connector is constructed**, as in `confluence.md` §1.2: a
  connector that discovers its missing token at the first request produces a run that reports
  progress and indexes nothing.
- **The HTTP client is the shared one** (`connectors/client.py`): retry on 429 and 5xx, a
  refusal to follow a redirect off the configured origin (a token must not travel to a host the
  configuration did not name), and credentials never reflected into an error message.
- **The first request of a sync is `GET /etapi/app-info`.** It proves the token, and its
  `appVersion` is recorded in run metadata so a behavior change after a server upgrade is
  attributable.

The token can read every note on the server it was created on, including notes outside
`root_note_ids`; the scope is enforced by this connector, not by Trilium (§5, §7).

---

## 3. What becomes a document

ETAPI's `Note.type` is one of `text`, `code`, `render`, `file`, `image`, `search`, `relationMap`,
`book`, `noteMap`, `mermaid`, `webView`, `shortcut`, `doc`, `contentWidget`, `launcher`.

| `type` | Indexed | Delivered as |
|---|---|---|
| `text` | yes | the note's HTML, `text/html` |
| `code` | yes | the note's text. `mime` of `text/x-markdown` is delivered as `text/markdown`; any other `text/*` or `application/*` source type is `text/plain` with the language kept in metadata |
| `mermaid` | yes | `text/plain` |
| `file` | only with `include_files`, and only where the note's `mime` has a parser | the bytes, with the note's `mime`, through the parser chain exactly as a file on disk |
| `image`, `relationMap`, `noteMap` | no | nothing worth searching; an image has no OCR in this connector |
| `book`, `search`, `render`, `webView`, `shortcut`, `doc`, `contentWidget`, `launcher` | no | containers, saved queries and UI configuration; the notes they point at are indexed in their own right |

**An empty note is not a document.** A `text` note whose content has no text (a container someone
used as a folder) is skipped and counted, since the tree already says everything it holds.

**A note's title is its first heading.** The connector wraps a `text` note's HTML in a minimal
document whose first element is `<h1>` of the title, as the mail parser makes the subject the
first heading, so a search for a note by name finds it and a citation has a heading to anchor to.
The wrapper is the connector's; the body is delivered unmodified.

**What is deliberately not translated** in v1: `<a class="reference-link">` internal links and
`include-note` sections are left as the HTML has them, relations (`~internalLink` and the like)
are attributes and not followed, and the images Trilium embeds as `api/images/…` are not fetched.
Note attachments (`GET /notes/{id}/attachments`) are not indexed. Each is a bounded later change
that does not alter anything here (§8).

**The system tree is skipped.** Note ids beginning with `_` (the hidden subtree, share
configuration, launcher bars, templates' scaffolding) and the `root` note itself are never
documents, and neither is anything beneath `_hidden`.

---

## 4. Identity and citations

**`source_id` is Trilium's `noteId`.** It is stable under rename, move, re-parenting and cloning,
which are the ordinary things a person does with a note, and a content hash would turn each of
them, or any edit, into a delete and a create. This is the Confluence choice (the page id) and
not the mbox one: a mail message has no id that is both present and unique, a Trilium note
always has one, and it is created by Trilium and never reused.

**`uri` is `<base_url>/#root/<noteId>`**, the address the web app opens. Whether the app resolves
a bare id in that position is checked in §9. The title and the path are never in the `uri`:
they change, and a citation must not.

**A note under several parents is one document.** Trilium "clones" a note so one note appears in
several places in the tree. The connector delivers it once, with its **breadcrumb taken from the
shortest path to a root, ties broken by the smaller `noteId`**, so two runs agree. Every parent is
recorded in metadata as `trilium_parent_note_ids`.

**The version token is a digest of the content identity and everything that is part of the
document's text:** `blobId` (Trilium's content hash for the note), the title, `type`, `mime`,
the breadcrumb, and the labels that are recorded in metadata. `blobId` alone would miss a rename,
which changes the document's heading and the way it ranks, without changing a byte of content.
Discovery does not fetch to decide; it compares tokens, as `DiscoveredDoc` requires.

**Metadata carries** `trilium_note_id`, `trilium_type`, `trilium_mime`, `trilium_labels` (name and
value of each label that is not an internal one), `trilium_parent_note_ids`, and the dates:
`utcDateCreated` and `utcDateModified` as `created_at` and `modified_at` in the provenance
record. **Discovery carries the title and nothing else that names content**, because the journal
holds `DiscoveredDoc`s, as it does for Confluence and mail.

---

## 5. Scope, and what is excluded

Two settings narrow what is read, and a third governs one kind of note.

- **`root_note_ids`** admits a note when **any** of its paths to the root passes through one of
  them. Empty admits everything not excluded below.
- **`exclude_labels`** removes a note when **any** of its paths passes through a note carrying one of
  the labels, or when it carries one itself. The label is matched by the connector over the note's
  attributes; it does not depend on Trilium's attribute inheritance (§9), because a privacy
  boundary that depends on a server feature being configured a certain way is not one.
- **`include_archived`** governs notes labeled `#archived`. The default is to leave them out,
  because archiving is how a person says "not part of what I look at".

**Exclusion wins over inclusion, and it is judged on every path.** A note cloned into an included
subtree and also under an excluded note is excluded. The failure this avoids is a note someone
marked private being indexed because a second parent was not marked. It costs the opposite
error, a note held out because one of its parents was, and that one is visible to the person
who made the clone and fixed by removing the label.

**Protected notes are never read.** Trilium encrypts a protected note and decrypts it only inside
a session the user unlocked with a password. ETAPI does not offer that session, so the content
is unavailable, and the connector does not try: it skips every note with `isProtected: true`
before any request for its content. The title of a protected note is the literal `[protected]`
in ETAPI responses; it is not used. Each sync's summary reports **how many protected notes were
skipped**, because "my notes are indexed" is otherwise silently untrue for exactly the notes a
person was most careful with.

Changing any scope setting changes the **scope fingerprint**, as `GitSiteConfig.scope_fingerprint`
does, so a run under a different scope is not taken as continuing the last one: a note newly
outside scope is retired by the next full inventory and not left indexed.

---

## 6. Discovery, change detection, deletion

**ETAPI has no change feed and no cursor.** `GET /notes` searches and returns every match
in one response, ordered by `orderBy`, with `limit` as the only bound. There is no "modified
since" parameter, and `GET /notes/history` lists recent changes with an unspecified window. The
design therefore does not lean on either.

### 6.1 The sweep

**One query returns every note's metadata and no content.** `Note` objects carry `noteId`,
`title`, `type`, `mime`, `isProtected`, `blobId`, `utcDateModified`, `parentNoteIds` and
`attributes`, which is everything §3 to §5 decide on. The sweep is a single `GET /notes` whose
`search` expression matches every note, ordered by `utcDateModified` descending. Content is
requested (`GET /notes/{id}/content`) only for a note whose token changed.

**A response is never trusted to be complete.** The request carries an explicit `limit`. A
response with exactly that many notes **is presumed truncated** and is not used: the sweep
splits by subtree instead, one request per child of the root with `ancestorNoteId`, splitting
again for any child whose response reaches the limit, and de-duplicates by `noteId` across
requests because a clone is in more than one. A response that cannot be proved complete raises.
This is the discipline of `mbox.md` §5.4: absence is deletion, so an inventory that might be
short is worse than none.

### 6.2 `discover`

- **No watermark: yield every note that passes §3 to §5**, each with its version token. The
  pipeline fetches those whose token it does not already hold.
- **With a watermark: yield the notes whose `utcDateModified` is at or after the watermark minus
  a five-minute overlap**, which is `confluence.md` §2's answer to the race between yielding a
  document and committing it. Re-yielding a few notes costs a token comparison, and change
  detection absorbs the rest.
- **A structural change yields everything.** A rename or move of a note changes the breadcrumb of
  every descendant without changing a descendant's own `utcDateModified`. The watermark's metadata
  holds a **tree digest**: a hash of `(noteId, title, parentNoteIds)` over every note that has
  children. When the digest differs from the last completed sweep's, `discover` yields every note
  as though there were no watermark. The tokens in §4 include the breadcrumb, so the notes
  whose path changed are fetched and the rest are skipped. A new leaf note does not change the
  digest; adding a first child to a leaf does, by design.

`Connector.watermark` is the maximum `utcDateModified` seen in a **completed** sweep, with the
tree digest in `Watermark.metadata`; a run that ends early offers none, as the protocol requires.

### 6.3 `fetch`

`fetch` reads `GET /notes/{id}/content`, whose `Content-Type` is the note's MIME type, and
returns the bytes with the media type §3 chose. A note over `max_note_bytes` is skipped and
counted, never loaded to be rejected (the response is streamed and abandoned at the cap). A
note that was deleted or protected between discovery and fetch is skipped, not an error: that
is the ordinary race, and the next sweep settles it.

### 6.4 `reconcile`

`reconcile` yields the `noteId` of **every note in scope now**, from a fresh sweep with the same
filters, in batches. It trusts no partial sweep: a truncated response that cannot be split, an
unreachable server, a 401, or a subtree that fails to list raises, and nothing is deleted.
When a note has been deleted in Trilium, or archived, labeled with an exclude label, moved
out of every `root_note_ids` subtree, or protected, it is absent and its document is soft-deleted;
the deletion ceiling (`ingest.reconcile_max_delete_fraction`) holds a large removal as a
proposal for `manicule connector reconcile NAME --confirm`, as for every source. Trilium's own
soft deletion (`isDeleted`, recoverable with `POST /notes/{id}/undelete`) is not special: an
undeleted note simply returns to the sweep.

**`/notes/history` is not used for deletion.** It reports `current_isDeleted`, but it is a list
of recent events with no stated horizon, and a deletion that fell outside it would be missed
forever. Absence from a complete sweep is the only signal that cannot lose one.

---

## 7. Sensitive

**`trilium` registers as sensitive**, with `sensitive=True` beside its `config_model` exactly as
`gmail` and `mbox` do. A person's notes are where passwords, health records, drafts and
second thoughts end up, and the index is not permission-aware: whoever can search the
workspace reads every note in it. The consequences are the existing ones and are not
repeated (`gmail.md` §10, `mbox.md` §7): an installation refuses to serve with
`security.auth.mode = "none"` while an enabled `trilium` instance exists, the first sync marks
the source before it enumerates anything, and `manicule document delete-source <name> --yes` lifts
it. Notes belong in a workspace of their own, and models see the text.

**The token's reach is wider than the scope.** ETAPI tokens are all-or-nothing: the connector
sees protected-note titles and every subtree on the server regardless of `root_note_ids`. The
scope in §5 is enforced here and stops a note from being indexed; it does not stop the
connector from having been *offered* it. A person who needs the server itself to refuse should run
a separate Trilium for that material, which is the same advice `confluence.md` §9 gives for
content some people may not see.

---

## 8. What is tested, and what is deferred

**Tests run against a synthetic server**, `tests/connectors/fake_trilium.py`, in the manner of
`fake_confluence.py`, and `assert_connector_contract` runs over it. Every title, label and host is
`example.org` or invented, and none of it is a real note. It reproduces: a note cloned under two
parents; an ancestor renamed (the tree digest, §6.2); a protected note; an archived note; an
excluded subtree with a clone escaping it; a sweep whose response hits the limit and must be split;
a note deleted between discovery and fetch; a 429 with `Retry-After`; a redirect off origin; and
a token that reaches the server as `Bearer` and nowhere else.

**Deferred, each bounded and independent of the above:**

- **Attachments** (`GET /notes/{id}/attachments`, `…/content`), indexed as members of their
  note, the way Confluence's follow their page.
- **Internal links**, rewriting `reference-link` hrefs to the target's `uri` so a citation can
  follow one.
- **Revision history** (`GET /notes/{id}/revisions`). Not a document a person searches for.
- **An offline reader for a copy of `document.db` or an export**, as `confluence_snapshot` is to
  Confluence. §10 says why it is not the first path.

---

## 9. Not yet verified against a live server

The design rests on ETAPI's published description (the OpenAPI document in the Trilium repository
and the ETAPI user guide), not on a response from a running server, and there are no Trilium
credentials in this repository. Check these first when one is available:

| Assumption | What to check | If it is wrong |
|---|---|---|
| **A single `GET /notes` can return every note**, and the `search` expression that matches all notes (candidate: `note.noteId != ''`) is accepted | Run the sweep against a vault of a few thousand notes | If Trilium requires a narrower expression or caps results silently, the split in §6.1 becomes the primary path rather than the fallback. The truncation rule still makes it correct, at the cost of more requests |
| **`limit` is honored, and a response shorter than `limit` is complete** | Ask for `limit=10` on a vault of 50 | Truncation detection fails open: a short response is trusted. This is the row that matters most, since reconcile deletes on absence |
| **The hidden subtree and system notes appear in search**, with ids beginning `_` | Run the sweep and list ids starting `_` | If they do not appear, the skip in §3 is harmless. If others do, the rule needs widening |
| **`Note.attributes` lists a note's own labels and not inherited ones** | Label a parent `#manicule-skip(inheritable)`, read a child | §5 already walks the tree itself and does not depend on this; the row is here because if inherited labels *are* listed the metadata would record labels the child does not carry |
| **`blobId` changes whenever content changes, and only then** | Edit a note, read its `blobId`; rename it, read again | If `blobId` is stable across an edit the token never changes and edits are missed. `utcDateModified` would join the token, at the price of re-fetching on a rename |
| **`utcDateModified` advances on a content edit** | Same | The watermark in §6.2 under-yields. The tree digest does not rescue it, so this one is a quiet loss |
| **`GET /notes/{id}/content` for a text note returns the editor's HTML**, not a rendering | Fetch one with a table and a code block | The HTML parser sees something other than what is stored; the wrapper in §3 changes |
| **A bare `#root/<noteId>` opens the note in the web app** | Open one | The `uri` needs the full path, which the breadcrumb walk already computes |
| **A protected note's content request fails with `NOTE_IS_PROTECTED`** or similar, and `isProtected` is true in the sweep | Fetch one | §5 skips before asking, so this is a harmless difference unless `isProtected` is *absent*, which would put protected notes' ciphertext in the index |
| **`Authorization: Bearer` is accepted by the TriliumNext line** from v0.93.0 | Call `/app-info` | Fall back to the bare-token header, which older servers take |
| **`GET /notes/history` lists deletions** | Delete a note, call it | Nothing in the design depends on it (§6.4) |

---

## 10. What was rejected, and why

- **A content hash as the identity.** §4: it turns every edit into a delete and a create, taking tags,
  collections and history with it. `noteId` is stable and unique.
- **`GET /notes/{id}/export`** (a ZIP of a subtree). It is one response for a whole subtree, so a
  note's change is the subtree's change and no per-note version token exists. It also drops
  identity (files are named by title) and would have to be unpacked and re-mapped to ids.
- **Reading `document.db` directly.** It couples the connector to a schema the project does not
  promise, contends with a running server's lock, and sees protected notes as ciphertext
  with no clean refusal. It is the right basis for an *offline* reader (§8), and the wrong
  one for a live server.
- **Trilium's sync protocol.** A private protocol between Trilium instances, not an integration surface.
- **Crawling published (`share`) pages.** Only a subtree a person has chosen to publish, rendered
  without its structure, and with no deletion signal.
- **`/notes/history` as the change feed or the deletion signal.** §6: its horizon is unstated, and
  a change or deletion outside it is lost without a trace.
- **A `utcDateModified` search filter as the incremental query.** Not in the published search
  documentation. The sweep needs no filter, and the one setting that would depend on it is not worth
  a request shape the first server upgrade could break.
- **A `since`, `types` or `labels` setting.** §1.
- **Relying on Trilium's attribute inheritance for exclusion.** §5.
- **Indexing protected notes by asking for a password.** manicule has nowhere to put one.

---

## 11. Where this will be implemented

Planned, for the build ticket; none of it exists yet.

| Part | Module |
|---|---|
| Configuration, scope fingerprint (§1, §5) | `manicule/connectors/config.py` (`TriliumConfig`, `resolve_trilium_credentials`) |
| ETAPI requests: sweep, split, content, `app-info` (§2, §6) | `manicule/connectors/trilium_client.py`, on `connectors/client.py` |
| The connector: tree, breadcrumbs, tokens, `discover`, `fetch`, `reconcile` (§3-§6) | `manicule/connectors/trilium.py` |
| Registration, as a sensitive type (§7) | `manicule/connectors/plugin.py` (`build_trilium`) |
| Synthetic server (§8) | `tests/connectors/fake_trilium.py` |
| Tests (§8) | `tests/connectors/test_trilium_*.py`; the contract suite over the fake |
| Capability entry | `CAPABILITIES.md`, beside `gmail` and `mbox` |
