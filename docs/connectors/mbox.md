# Old mail archives: the mbox connector and the Outlook PST recipe

Design for the `mbox` connector. Design ticket
[#521](https://github.com/mgd43b/manicule/issues/521), build ticket
[#522](https://github.com/mgd43b/manicule/issues/522). Related: the Gmail connector
([`gmail.md`](gmail.md), epic [#464](https://github.com/mgd43b/manicule/issues/464)), which this page
reuses wherever the problem is the same.

> **Status: design.** Nothing on this page is implemented yet. The build PR changes this banner to
> a map from section to module, as `gmail.md` does, and edits whatever here the code turned out to
> do differently.

A person with twenty years of mail has it in two places that no connector reads: **unix mbox
files** (Thunderbird folders, Mutt and Pine archives, `fetchmail` spools, Apple Mail exports,
the output of every "export my mail" tool) and **Outlook PST files**. Both are large, both are
read-only in practice, and both are the most sensitive material manicule handles, for the reasons
`gmail.md` §10 gives: the index is not permission-aware, so whoever can search the workspace reads
every message in it.

This page decides how that mail gets in, how it is told apart from mail already indexed by the
Gmail connector, and what an installation that holds it must refuse to do.

The connector ends at `RawDocument`. A message is delivered as RFC 5322 bytes with the media type
`message/rfc822`, which the existing mail parser ([`parsing.md`](../parsing.md) §10,
`manicule/parsers/mail.py`) already anchors to lines of its canonical text and whose attachments
already route through the parser chain (calendar invitations, archives, nested messages). There is
no mbox-specific parsing.

---

## 1. Shape: a connector, not a parser

A parser takes one `RawDocument` and returns the blocks of one document. An mbox file is the wrong
size for that contract in every dimension:

- **It is a container of thousands of documents.** A parser registered for an mbox media type
  would return one document with a hundred thousand messages in it, whose chunks cannot be
  attributed to a message, and whose change detection is the whole file's hash: one new message
  re-indexes the archive.
- **It does not fit through the door.** The pipeline reads a document whole, bounded by
  `ingest.max_fetch_bytes` (256 MB by default), and retains the original bytes only up to
  `storage.blobs.MAX_ORIGINAL_BYTES` (256 MB). A ten-year Thunderbird `Inbox` is several
  gigabytes. Raising either limit to fit it would put an archive's whole size in memory per fetch.

So the unit the pipeline sees is the message, and the thing that turns a file into messages is a
connector. **`type = "mbox"` walks a configured root, finds the mbox files, streams each one
once, and yields one `message/rfc822` document per message.** It never holds a file, or a
message it is not delivering, in memory. The cap on one message is the pipeline's own
`ingest.max_fetch_bytes`, enforced on the message and not on the archive; the connector adds no
second one.

```toml
[connectors.oldmail]
type = "mbox"
enabled = true

[connectors.oldmail.options]
root = "/srv/archive/mail"            # required: a directory, or one mbox file (§2)
include_hidden = false                # walk dot-files and dot-directories (default false)
exclude = ["**/Trash", "**/Spam"]     # root-relative globs this source never walks (§2)
skip_known_message_ids = true         # the default: skip a message another mail source holds (§6)
```

`MboxConfig` is frozen and rejects any other key. `root` is resolved to an absolute path, because
a document's identity is built from where its file is (§4). There is deliberately **no `query`,
`since` or `folder` setting**: a filter evaluated on a full walk but not on a re-walk is the
failure `gmail.md` §3 states for labels, and a path glob already says which files count.

---

## 2. What is walked, and what counts as an mbox file

**The walk is the filesystem connector's** ([`filesystem.md`](filesystem.md) §1–§2): sorted at
every level so two machines ingest in one order, dot-entries skipped unless `include_hidden`,
symbolic links followed nowhere, `.git`, `node_modules` and the other tool directories never
entered, and `exclude` globs relative to the root that prune a directory unvisited. It shares the
glob matcher and the ignore list rather than copying them. A root that is itself a file is read
as one mbox file and `exclude` has nothing to name.

**A file is an mbox file because of its first bytes, not its name.** Thunderbird's folders have no
extension (`Inbox`, `Sent`, `Archives`), Mutt's are whatever the user called them, Apple Mail's
Export Mailbox writes a `.mbox` *directory* with a file named `mbox` inside, and a Maildir
neighbor of any of them is a directory of single messages that must not be mistaken for one. So a
regular file is an mbox file when it begins with a `From ` line (§3) whose next line is a header
field, `Name: value`. That test reads the first few hundred bytes and rejects:

- **`.msf`** (Thunderbird's summary file, a cache of what the mbox says, and a text format that
  starts with `// <!-- <mdb:mork`) and every other Thunderbird side file (`.dat`, `.json`,
  `.sbd` directories are *descended into* as ordinary directories, since they are where
  subfolders live). `.msf` is also skipped by name before it is opened, so an archive of
  ten thousand folders costs ten thousand `stat`s and not ten thousand `open`s of cache files.
- **Anything that merely contains the word "From".** A text file starting `From the desk of…`
  has no header on its next line.
- **An empty file or one under a few bytes.** Not an error; there is nothing in it.

A file that fails the test is skipped silently, which is right for indexing and is the same
decision in `discover` and `reconcile` (they share the one function), so the two never disagree
about what the source holds.

**Apple Mail `.mbox` bundles are in scope in the one form they can be: the Export Mailbox
folder**, which is a directory named `Something.mbox` holding an `mbox` file in the format below.
No special case is needed: the walk descends into the directory and the sniff accepts the file.
**Mail.app's own library, a tree of one `.emlx` file per message, is out of scope.** `.emlx` is
not mbox: each file is a decimal byte count, a line break, the RFC 5322 bytes, and a trailing
XML property list. It is a one-file-one-document format, so a parser (and a row in the
filesystem connector's media-type table) is the right shape for it rather than this connector. It
is not built here, and nothing in this connector claims to read it.

---

## 3. Formats: how a file is cut into messages

"mbox" is four dialects of one idea, and a reader that handles one silently corrupts the others.
The connector reads them with **one streaming reader of its own**, not the standard library's
`mailbox` module.

### 3.1 Where a message begins

A message begins at a **`From ` line** (the "From_ line", a five-byte prefix followed by anything:
`From alice@example.org Thu Jan  1 00:00:00 2026`, Thunderbird's `From - Thu Jan 01 …`, a quoted
name from `readpst`, `MAILER-DAEMON`). Prefix alone is not enough, because a body line `From the
top, …` begins with the same five bytes. A line is a separator only when **all** of these hold:

1. it begins with `From `;
2. it is the first line of the file, or the line before it is empty (mbox writers put a blank line
   at the end of every message precisely so this is checkable); and
3. the line after it is a header field (a token, a colon).

The third condition is what stops an unescaped `From ` line in a body from splitting a message in
two. It costs one line of lookahead. A message is the separator's following lines up to the next
separator or the end of the file, with the one empty line before the next separator dropped (it
belongs to the separator, not the message).

### 3.2 `Content-Length` (mboxcl, mboxcl2)

A file whose writer was a Sun/SVR4 mailer carries `Content-Length: N` on each message, giving the
body's size in bytes. Such a writer does **not** escape `From ` in bodies, because it does not
need to: the length says where the body ends. Two rules follow, and getting either wrong
corrupts text:

- **A `Content-Length` is believed only if it is checkable.** After the headers, the reader
  looks at what follows `N` bytes of body: either the end of the file, or an empty line and then a
  separator (§3.1). If it does, the body is exactly those `N` bytes. If it does not, the header is
  ignored for this message and the scan below is used. A `Content-Length` copied from the
  original SMTP transaction by a relay, wrong, or stale after an editor changed a body is
  therefore ignored rather than trusted, and a message that really is mboxcl is not cut short by
  a quoted `From ` line it contains.
- **A body read by length is not unescaped** (§3.3). Its `>From ` lines were written that way by
  the author.

### 3.3 `From ` quoting (mboxo, mboxrd)

A writer that scans for the separator must escape body lines that look like one. **mboxo** turns
`From ` at the start of a line into `>From `, which cannot be undone, because a body that already
began with `>From ` is indistinguishable. **mboxrd** adds one `>` to any line matching `>*From `,
which can. The reader does not know which dialect a file is in and cannot, in general, find out.
It applies the mboxrd rule to every message cut by the scan: **a line matching `^>+From ` loses
one `>`.**

This is right for mboxrd (every modern writer, and `readpst`), right for the overwhelmingly common
mboxo line (`>From ` was `From `), and wrong, by one level of quoting, for a quoted `>From ` line
in an mboxo file. That is accepted, and recorded here because it is a choice: the alternative,
leaving the escape in, makes the same message in an mboxrd file and an mboxo file two different
documents (§4) and shows a stray `>` before the word every time a message says "From". The delivered
text of an affected line differs from the author's by a quoting character; its anchors are
consistent with what was delivered, so a citation still resolves.

### 3.4 What else is normalized, and what is dropped

- **Line endings become `\n`.** Thunderbird on Windows and several exporters write CRLF; a
  message identical but for line endings must be one message (§4), and the mail parser's anchors
  are line-based. A bare `\r` is left alone. This changes the bytes of a message that has CRLF
  inside a binary body that is not transfer-encoded, which is vanishingly rare in real mail and
  would already be damaged by the transport that put it in a text file.
- **A message flagged deleted is not a message.** Mail clients delete by flagging and compact
  later, so a "deleted" message is still physically in the file and is the one thing in the
  archive its owner decided not to keep. The reader skips a message when
  `X-Mozilla-Status` has the expunged bit (`0x0008`) or `X-Status` contains `D`. It also skips the
  placeholder message that UW IMAP, Pine, Dovecot and Thunderbird write at the top of a folder
  (`Subject: DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA`, or an `X-IMAP:` header).
  These are the same decisions in `discover` and `reconcile` and are not settings.
- **Unterminated and damaged files.** A final message with no trailing blank line is delivered. Bytes
  before the first separator are ignored. A file that stops mid-line is read up to the line.
  Headers are read to a ceiling of 1 MiB; a message whose headers do not end by then is
  delivered with the bytes read so far as its headers rather than failing the file.

### 3.5 Why not `mailbox.mbox`

`mailbox` from the standard library is the obvious tool, and it was considered first. It is not
used because it does not do the five things this connector needs:

| Need | `mailbox.mbox` |
|---|---|
| Stream a file of any size, once, in constant memory | Builds a table of contents of every message's start and stop on first access: one pass, then every message's offsets resident for the object's life |
| Hash a message while reading it, in the same pass | Hands back the whole message after the pass, so a second read |
| Trust a verifiable `Content-Length` and not an escape | Ignores `Content-Length`; splits on the prefix and a preceding blank line only |
| Say which `From ` lines are not separators (§3.1 (3)) | Does not check the next line |
| Fetch one message by byte offset without scanning | Seeks by its own table, which does not exist until the scan has run in this process |

The reader is small (a line loop and a few headers), it is the part that decides what every
message in the archive *is*, and it is tested against the standard library's `mailbox` on every
fixture where the two should agree and against its documented disagreements where they should not.
The `email` package is still what the mail parser uses to read a delivered message.

---

## 4. Identity and citations

**`source_id` is the SHA-256 of the message's canonical bytes, in lowercase hex (64 characters).**

The canonical bytes are the message **as it will be delivered** (separator removed, quoting
undone, line endings `\n`; §3) with the headers that record *what a mail client did to its copy*
removed: `Status`, `X-Status`, `X-Mozilla-Status`, `X-Mozilla-Status2`, `X-Mozilla-Keys`,
`X-UID`, `X-Keywords`, `X-Evolution`, `Content-Length` and `Lines` (a header's continuation lines
go with it). They are removed from the *hash*, not from the delivered message: the mail parser
still sees what the file says.

What this gives, and what each alternative would not:

| Candidate | Survives reordering | Same message in several files | Missing from some mail | Two different messages, one id |
|---|---|---|---|---|
| **Byte offset or ordinal in the file** | No: compaction, an expunge or an added message moves every message after it, and every one would be a delete and a create, taking tags, collections and history with it | No | n/a | No |
| **`Message-ID` header** | Yes | Yes, and also for *copies that are not the same message* | **No.** Absent from some mail (old mail, scripts, bounces), so it cannot be the key alone | **Yes.** Forwarded and resent mail, mailing-list copies and re-deliveries share one; a hostile or buggy sender repeats one; and `documents` is UNIQUE on `(workspace, source, source_id)`, so the second would silently overwrite the first |
| **Hash of canonical bytes** | Yes | Yes, when the copies are byte-identical, which is what a message filed in two folders or present in an overlapping backup is | Yes: every message has bytes | No, short of a SHA-256 collision |

So the hash is the identity, and **`Message-ID` is not**. That has a cost, and it is stated: two
copies of one logical message that differ in a byte that is not in the dropped list — a
`Received:` line one added, a mailing-list footer — are **two documents**. They are different
bytes and the connector will not guess otherwise; folding them is a policy decision about what is
"the same mail" that belongs to the `Message-ID` check in §6, where it is optional and scoped.

Within one source, then, a message that appears in several files is **one document**. Its `uri`
and the locator that fetch uses (below) are those of the first file in walk order that holds it.
Discovery yields each identity once per run (§5.3).

**`uri` is `file://` + the absolute path (percent-encoded) of the first file in walk order that
holds the message + `#sha256=<source_id>`.** A `file:` URI is what a person can resolve; the
fragment says *which message* in a file of ten thousand, and it is the message's identity, so it
**cannot point at a different message than the one indexed**. The byte offset and length that
`fetch` seeks to are deliberately **not** in it. They travel only in `DocRef.metadata` (§5.2),
because an offset in a citation becomes a wrong citation the first time the file is compacted: the
version token is constant, so an unchanged message is never fetched again and its stored address
is never refreshed, and `#offset=N` would then name some other message. The hash fragment stays
true under any rewrite. What can go stale is the *path*, after a rename or move of the file; the
message itself is not lost, and the walk still finds it (§5.1). Whether a token-equal record
refreshes the stored address is a pipeline behavior the build verifies and records here rather than
assumes.

**Citations inside a message are the mail parser's**: line anchors into the canonical header block
and body, and `mail:<uri>!/name` members for attachments. They reach the pipeline
unchanged.

**Title and provenance** are the `Subject` (the parser makes it the first heading element) and, in
the provenance record (`PROVENANCE_KEY`), the title, the `Date` header as `created_at` when it is
a valid timestamp, and the message's local path as the record's snapshot. A header that does not
validate (a control character in a subject) leaves the message indexed and cited by its
locator, as `gmail.md` §2 does. **Discovery carries no title, subject, address or `Message-ID`**:
the journal holds `DiscoveredDoc`s, and everything that can name a person stays in the
document it belongs to. `Message-ID` is recorded in the document's metadata as `mail_message_id`
(angle brackets stripped) for §6, and nowhere else.

---

## 5. Discovery, change detection, deletion

An archive is usually read-only. When it is not, this is what each thing that can happen to a file
does.

### 5.1 The watermark is per file, and a file is the unit of change

The version token of a message is a constant (`mbox/1`): its bytes are its identity, so a
message under a given `source_id` can never have changed. What changes is *which messages exist*,
and that is a property of files. So the watermark is a JSON map, `{relative path: [size,
mtime_ns]}`, of every file the last **complete** walk read, plus a digest of the scope (below).
Discovery stats each file and **skips one whose size and modification time equal the map's**: an
unchanged archive costs one `stat` per file, not a read. A file whose entry differs, or is absent,
is scanned.

The size and mtime recorded for a file are those **taken before it is opened**, and the scan
reads no further than that size. A message appended while the scan runs is therefore either wholly
in this scan or wholly in the next one, never half-read, and a file that grows during the walk
still differs from its entry afterwards. This is the overlap `gmail.md` §4.1 gets by capturing its
position first, with the same reasoning: seeing a message again costs a version comparison, and
missing it costs the message. The watermark is the connector's `watermark` property, so it is
only handed out after an enumeration that completed.

| What happens to the archive | Discovery | Then |
|---|---|---|
| Nothing | Every file skipped by `stat`. No messages yielded | Free |
| A file is **appended to** (a live Inbox) | The file's token differs, so it is scanned from the top. Every message is yielded again, but each unchanged one carries the constant token and the pipeline skips it, so the cost is one sequential read of that file and a journal record per message, and not a re-index | The new messages are indexed. A tail-only re-scan (resume where the last scan ended, once the bytes before it were checked) would make this proportional to what was added; it is not built, because it needs a stored digest of the prefix to be safe and an archive that is appended to is the exception |
| A file is **rewritten** (compaction, an expunge, a restore) | Same: rescanned | A message that is no longer in any file leaves through reconciliation (§5.4) |
| A file is **touched** (restored from a backup, copied without preserving times) | Rescanned; nothing is new | Nothing changes in the index |
| A file is **renamed or moved** | Its entry is absent under the new path, so it is scanned; its messages' identities are unchanged | The documents stay. Their stored path is the old one until the pipeline refreshes an unchanged record's address (§4) |
| A file is **removed** | Nothing at discovery: an absence is invisible to an incremental sync | Reconciliation (§5.4) |
| A file is **unreadable** (permissions, I/O error) | The error is logged by file *index*, not name, the file is not in the watermark, and the walk continues | It is tried again next sync. The run finishes; an enumeration that skipped a file never writes a watermark that claims the file was read |
| The **root vanishes** | Discovery yields nothing and **raises**; a vanished root is not an empty archive | |

**The scope is in the fingerprint.** The pipeline reuses a stored watermark only under the scope
fingerprint it was recorded with (`gmail.md` §4.1 describes the mechanism). The fingerprint hashes
the resolved root, `include_hidden`, `exclude`, `skip_known_message_ids` and a representation
version. Changing any of them discards the stored map, and the next sync reads every file again.
This is what makes turning `skip_known_message_ids` off (§6) bring back the messages it had
skipped, and what lets the representation version force one complete re-fetch if what is
delivered ever changes.

### 5.2 `discover` and `fetch`

`discover(None)` and `discover(watermark)` are the one walk, differing only in whether the map
has entries to skip on. It implements the `BatchedDiscoveryConnector` protocol
(`discover_batches`) so the pipeline can commit a bounded response before asking for the next
one: **a batch is at most 250 records, the pipeline's `MAX_DISCOVERY_BATCH`**, taken from the
scan 250 at a time. The scan runs in a worker thread, because it is blocking file I/O and the
event loop is serving a search while it runs, and holds one batch in memory.
(This is the lesson of the Gmail page size that exceeded the limit: the batch size is the
connector's responsibility and is tested at the boundary, §8.)

Each record carries `source_id` (§4), the `uri` (§4), `version_token = "mbox/1"`,
`media_type = "message/rfc822"`, `size_bytes` (the canonical size) and, in `DocRef.metadata`,
the three things `fetch` needs: the file's absolute path, the offset and the length of the
message's raw bytes in the file.

`fetch(ref)` **seeks to the offset, reads `length` bytes, re-applies the canonicalization of §3 and
verifies that the SHA-256 of the result equals `ref.source_id`.** The offset is an optimization
and the hash is the authority. A file that was rewritten between discovery and fetch fails the
check, and `fetch` raises `NotFoundError`: the message is no longer where discovery said it was.
That is chosen deliberately and its effect is not small. The pipeline maps `NotFoundError` to a
typed `source_deleted`, which retries the document and **invalidates the completed inventory**
([`ingest.md`](../ingest.md) §8.3.1): the run is recorded as `reenumeration_required`, its
watermark is not committed, and the next sync enumerates again from the last committed watermark.
That is the right response here and a plain fetch failure would not be: the journal holds the
stale offset, so retrying the fetch would fail identically for ever, while a fresh enumeration
reads the rewritten file (its size and mtime differ from the watermark) and re-locates the message,
or finds it gone. The cost of a file being rewritten mid-run is therefore one extra enumeration,
of the files that changed. A message is never indexed from bytes that are not the bytes its
identity names.

`fetch` reads only the message, so a 3 GB file costs one message's worth of memory per fetch, and
the `size_bytes` it declared lets the pipeline's `ingest.max_fetch_bytes` refuse a message that is
itself too large without reading it.

### 5.3 Duplicates inside one run

One message in two files would be yielded twice under one `source_id` with two locators, and the
journal rejects a replayed identity whose record differs. So **a run yields each identity once**,
the first live copy in walk order. A copy flagged deleted (§3.4) is skipped *before* it claims the
identity, so a later live copy of the same message is still indexed. The set of identities seen is
**not held in memory**: a quarter-million
messages is tens of megabytes of digests as Python objects, and an archive of years of
mail is more. It is a temporary SQLite table in the cache directory, keyed on the digest, **created by each walk
and removed when that walk ends**, so the working set is a batch and the rest is on disk. It
belongs to the walk and not to the connector, because the connector is one object for the life of
the process and a discovery and a reconciliation can overlap (`gmail.md` §3 makes the same point
about its exclusion set). `discover` and `reconcile` use the same mechanism.

### 5.4 `reconcile`

`reconcile` yields the `source_id` of every message that is **in the archive now**: the same walk,
the same sniff, the same cut, the same skips for flagged-deleted and placeholder messages, and the
same once-per-identity rule, with the difference that **every file is read** (an unchanged
file's messages must be listed to be kept) and nothing is compared with the watermark. It is
`reconcile_batches` too, in batches of at most 250. It is a full sequential read of the archive
and so is expensive in proportion to the archive and not to the change; a source that is a
static archive wants `reconcile_interval_s` lengthened, which is its existing setting
([`ingest.md`](../ingest.md) §11).

**A reconciliation never trusts a partial walk.** A directory it cannot list, a file it cannot
read, or a root that has gone raises; the inventory is canceled and nothing is deleted. Absence
is deletion to a reconciliation, so the strictness is the same as the filesystem connector's.
When a message has genuinely left every file, the pass soft-deletes its document and the
deletion ceiling (`ingest.reconcile_max_delete_fraction`) holds a large removal as a proposal that
`manicule connector reconcile NAME --confirm` applies.

**Reconciliation does not apply `skip_known_message_ids`.** Its inventory is a superset of what
discovery yields, and that is the safe direction: a pass that applied the skip would delete an mbox
copy the moment the Gmail connector indexed the same mail, which is a deletion nobody asked for and
one the ceiling would then have to catch. A message that was indexed before another source held it
stays indexed (§6).

**Discovery agrees with the inventory about everything this source already holds**, which is the
condition that makes "superset" true in the direction that matters. A full walk is a full
inventory: the pipeline records it as one, and a rebuild led by a full run retires the documents it
does not contain ([`ingest.md`](../ingest.md) §8.3, "deletion authority"). So a skip never removes a
document the source holds: discovery skips a message for a known `Message-ID` only if **this**
source does not already hold that identity (§6). Every document the source holds that is still in
a file is therefore in both discovery's manifest and the reconciliation inventory.

---

## 6. Mail that Gmail already indexes

The same message can be in both places: the Gmail account that is still live, and the mbox export
from five years ago that an old laptop still has. Indexed twice, every query returns it twice,
and `ask` quotes it twice as if two sources agreed.

**`skip_known_message_ids` (default `true`)** makes discovery skip a message whose `Message-ID`
is already held by **another mail source**: another configured instance of type `gmail` or
`mbox`, and not itself. **A message this source already holds is never skipped** (`find_document`
by identity, asked only of the few messages whose `Message-ID` is known elsewhere), so turning the
option on, or syncing Gmail later, never makes a document this source indexed disappear from a full
walk (§5.4). Mail the other source holds is, for the purpose of this option, what that
source's live documents say it holds.

How the other source's `Message-ID` values are known:

- **an `mbox` document** records `mail_message_id` in its metadata (§4);
- **a `gmail` document** records the `Message-ID` in its link, as
  `…#search/rfc822msgid:<percent-encoded id>` (`gmail.md` §2), and has no `mail_message_id`. The
  connector reads the link, which every Gmail document already has and which needs no change to
  the Gmail connector and no re-fetch of a mailbox. A Gmail message without a `Message-ID` was
  given the `#all/<gmail id>` link, carries no id, and can never match.

The ids are read from the document store once per walk, the first time a message with a
`Message-ID` needs judging, into an on-disk table beside the one §5.3 uses. The read is a new
**streaming method on the SQLite document store**, shaped like `known_source_ids`
(an async generator over a column selection, closed with `aclosing`, no `Document` loaded), and it
yields the ids of **live documents only** (`deleted_at IS NULL`) of the named sources. It cannot be
`list_documents(offset=…)`: that pages whole documents, is quadratic over a large Gmail source, and
loads every chunk-less row's metadata to read one field. A trashed Gmail message therefore stops
suppressing its archived copy, which is what trash means. A run with nothing to
judge (an archive whose files are all unchanged) does not read the store.

**The default is to skip, because the failure it prevents is the common one.** An archive and a
live mailbox overlap by default and identically; a duplicate hit is noise in every answer.

**The trade-offs, which are why it is a setting and not a rule:**

| Trade-off | Consequence | What to do |
|---|---|---|
| **A message without a `Message-ID` is never skipped** (§4's reason for not keying on it) | It is indexed, and if Gmail has it too, it appears twice | Accept; these are rare in mail from the last twenty years |
| **Two different messages can share a `Message-ID`** (a resend, a forward that kept the header, a buggy sender) | The second is skipped though it is not a copy, **and its content is not searchable** | If the archive is the only record of such mail, set `skip_known_message_ids = false` |
| **Deliberately different copies** (the archive has the message as received, with a `Received:` trail and the tracking footer; Gmail has the version after a list's rewriting) | One is skipped; **which one depends on which source synced first**, and the content that remains is that source's | Same: turn it off, and accept both |
| **The decision is made once, at the walk** | A message skipped because Gmail held it stays unindexed if Gmail later deletes it, because a file that has not changed is not walked again | Any change to the source's options changes the fingerprint and forces the re-walk |
| **Order of arrival** | An mbox source synced before the Gmail source indexes everything, and Gmail then indexes it too: only the later source can skip (§5.4 explains why reconciliation does not undo the earlier one) | Sync Gmail first, or delete the archive's documents that Gmail now holds |
| **A skip is silent per message** | An archive reads as complete and is not | The run's aggregate count of skipped messages is logged and in the sync result (a number, never an id); `manicule connector list` is unchanged |

The decision to skip is made at discovery, from the stored documents as they are then; it is
never revisited for a file whose `stat` has not changed. This is the one place `discover` and
`reconcile` differ, on purpose (§5.4), and it is why a changed setting, which changes the
fingerprint, re-walks.

---

## 7. Sensitive

**Mail is the most sensitive source here, and `mbox` registers as sensitive**, with
`sensitive=True` beside its `config_model` exactly as `gmail` does
(`manicule/connectors/plugin.py`; [`gmail.md`](gmail.md) §10). The consequences are the existing
ones and are not repeated:

- an installation refuses to serve with `security.auth.mode = "none"` while an enabled `mbox`
  instance exists or its data directory holds any document of a source a sensitive connector
  synced, on every address, loopback included, with no override;
- the first sync marks the source before it enumerates anything;
- `manicule document delete-source <name> --yes` is what removes the refusal;
- mail belongs in a workspace of its own, models see the text (so use local ones or decide that),
  and an export of a marked source carries the mark.

**The filesystem connector is not sensitive, and mail indexed through it is invisible to the
guardrail.** `type = "filesystem"`, and `manicule index <path>`, will index a directory of `.eml`
and `.msg` files today with the installation free to serve them to anyone who can reach the port.
The registry marks a connector *type*, and the type cannot know what is in a directory.

**Content decides, then, and the question is asked of the documents that are there.** The refusal
already reads `documents` with plain `sqlite3`, read-only (`manicule/app/sensitive.py`), for the
sources a sensitive connector marked. It gains a second clause: **a source is also held when it has
a document whose `media_type` is `message/rfc822` or `application/vnd.ms-outlook`**, in any state
(deleted and purged rows count, as for the first clause, because they keep their title). Nothing
is marked and nothing is written; the fact is read from the rows.

That is deliberately a read and not a mark made as documents are written, for three reasons that
the mark would have lost:

- **It covers mail already indexed.** An installation that indexed a folder of `.eml` files last
  year is refused on upgrade, which a mark made at write time would not do for rows it never saw.
- **It covers every write path and attachments.** `documents.media_type` is stored for members as
  well as containers, so an `.eml` inside a `.zip` or a `.tar`, a message attached to a message, and
  a document a plugin added are all rows with a mail media type, found whichever connector or
  command wrote them (`manicule index`, a crawler, a forced sync, an import).
- **It needs no new state.** `delete-source` removes the rows, and the refusal lifts with them.

It is the same refusal for the same reason, with the same remedy (`manicule document delete-source
<name> --yes`), on every address, with no override, and `doctor`'s `transport` check reports it
beside the other sensitive sources.

What this does not do, so nobody leans on it:

- **It is a refusal to serve, decided when a server starts or an application is built.** A server
  that is already serving unauthenticated when a mail file appears in an indexed directory keeps
  serving until it restarts, and the document is searchable until then. Mail indexed through
  `type = "mbox"` is covered earlier, because its type is known sensitive before the first message
  is read; use it, or a separate installation, where the guarantee is complete.
- **A corpus that happens to contain one sample `.eml`** (a test-fixtures directory, a mail
  library's repository) now needs an authenticated server, or an `exclude` of the file. That is
  the price of failing closed for the case where it was real mail, and it is the same trade the
  guardrail made for Gmail.
- **It looks at what the pipeline routed as mail.** A message saved as `.txt` is text, and a PST,
  which this project cannot read (§9), is not indexed at all.

---

## 8. Large files, and what is tested

A fixture generator in the tests writes **synthetic** archives, with every variant above, under
the conventions of `gmail.md`'s synthetic mailbox: every address is `@example.org`, every body
is generated, and nothing is real mail or resembles it.

| Fixture | Exercises |
|---|---|
| mboxo, mboxrd, mboxcl, mboxcl2 files, and a mix in one directory | §3.1–§3.3: separators, quoting, `Content-Length` believed and ignored |
| A body with an unescaped `From ` line, with and without a following header | §3.1 (3) |
| A body with `>From `, `>>From ` | §3.3, and the same message in two dialects getting one identity |
| CRLF and LF copies of one message | §3.4, §4 |
| Thunderbird folders with no extension, `.msf` files, `.sbd` subfolders, an expunged message, a folder-internal-data placeholder | §2, §3.4 |
| An Apple Mail export directory | §2 |
| A directory of `.emlx` and a Maildir | Not read, and not mistaken for mbox |
| One message in three files, and the same message with different `X-Mozilla-Status` | §4, §5.3 |
| Messages with no `Message-ID`, two messages sharing one, and a message another source holds | §4, §6 |
| An attachment (text, a calendar invitation, a nested message, a `.zip`) | The parser and attachment routing unchanged |
| A file larger than the pipeline's fetch limit made of small messages | §1: the archive is read, no message is refused |
| 3,000 messages in one file | Batches of 250 at the boundary (§5.2) |
| A filesystem source holding an `.eml`, one holding a `.zip` of them, and an index made before this change | §7: the server refuses to start unauthenticated |
| A file rewritten between discovery and fetch | §5.2: `NotFoundError` and one re-enumeration |

Every one is driven **through the pipeline runner**, over the real stores, with durable
acquisition on, and not through `parse()` alone: the Gmail work shipped a parser whose suite
passed while every message through the runner had zero chunks. The runner tests assert chunks,
citations that resolve, `reconcile` after a message is removed, append, rewrite and rename, a
changed `skip_known_message_ids`, and the sensitive refusal.

---

## 9. Outlook PST files

**manicule has no PST reader, and this design does not add one.** The reasons are not
effort:

- **PST is not OLE.** `.msg` is a compound file, which `olefile` (BSD-2) reads and
  `manicule/parsers/msg.py` uses ([`parsing.md`](../parsing.md) §10). A PST is a paged database
  (NDB), with a B-tree of nodes and blocks, a property-context layer above it and encryption
  options (none, a permutation, a cyclic cipher) that [MS-PST] defines in hundreds of pages. Nothing in the
  `.msg` reader carries over.
- **The libraries that read it are not usable here.** `libpff` is LGPL-3 with a native C build,
  which the project's image and wheels do not carry. `libpst`, whose tool is `readpst`, is
  **GPL-2.0-or-later**. manicule is MIT, and `tests/test_license_boundary.py` fails the moment a
  copyleft dependency enters the closure; [`parsing.md`](../parsing.md) §12 records the same
  decision for `extract-msg` (GPL-3) and PyMuPDF (AGPL), and for the same reason. A GPL program
  *run* by a person on their own machine is not a dependency of manicule; one *shipped* in its
  image or imported by its code would be.

**The recipe is a conversion outside manicule, and then an ordinary index.** On a machine that
has `readpst` (the `pst-utils` package on Debian and Ubuntu, `libpst` elsewhere), convert each PST to
mbox or to one `.eml` per message, then point manicule at the output. The flags below are those of
the `readpst(1)` manual page; the program is not part of manicule or its test suite, so check
`readpst -h` for your version and expect to adjust.

```console
$ mkdir -p /srv/archive/mail/outlook-2019
$ # mbox: a directory per PST folder, each holding a file named "mbox"
$ readpst -r -t e -o /srv/archive/mail/outlook-2019 /path/to/archive.pst
$ # or one .eml per message: a directory per folder, files numbered 1 to n
$ readpst -e -t e -o /srv/archive/mail/outlook-2019 /path/to/archive.pst
```

- `-r` writes the folders named as in the PST and, in each, the mail in a file called `mbox`, which
  the connector's sniff (§2) accepts. Without `-t e` it also writes the PST's calendar, contacts
  and journal items as files named `calendar`, `contacts` and `journal` beside it, and those are
  written in a mail-compatible form the sniff would accept as well. `-t e` asks for mail only, and
  is why the recipe passes it.
- `-e` writes each message as its own `.eml` file. `-o` must name a directory that exists.
- `-D` includes deleted items. Leave it off: the PST's deleted items are the mail its owner
  chose to throw away, the same reasoning as §3.4's flagged-deleted messages.
- `readpst`'s single-file output uses mboxrd quoting, which §3.3 reads exactly. Its per-message
  `.eml` output does no `From ` quoting at all, since there is no `From ` line to protect.

Then, for the mbox form:

```toml
[connectors.outlook]
type = "mbox"
[connectors.outlook.options]
root = "/srv/archive/mail/outlook-2019"
```

and for the `.eml` form, which is not mbox, the filesystem connector reads them:

```toml
[connectors.outlook]
type = "filesystem"
[connectors.outlook.options]
root = "/srv/archive/mail/outlook-2019"
```

**Prefer the mbox form.** A filesystem source of `.eml` files is one document per file with an
identity that is its path, so it has none of §4's properties (a moved file is a delete and a
create; the same message in two folders is two documents; it has no `skip_known_message_ids`), and
it is known sensitive only by content (§7), once the first message is indexed, and not by type.
The mbox connector is the better target for the same data, and the recipe exists so that both
work. This recipe has been checked against the manual page and not run against a PST in CI: the
GPL boundary is the reason there is no `readpst` in the test environment.

Keep the intermediate files on storage you would trust with the mail itself: the conversion
writes every message in the clear, and deleting the PST does not delete them.

---

## 10. What was rejected, and why

- **A PST/`.msg` reader in-product.** §9.
- **`Message-ID` as the identity.** §4.
- **Offset or ordinal as the identity, or an offset in the citation.** §4.
- **Holding a file's table of contents in memory** (the `mailbox` module's approach). §3.5.
- **Re-reading only an appended tail.** §5.1; the digest of the prefix it needs is a feature of its
  own, and an archive is rarely appended to.
- **A parser for the mbox media type.** §1.
- **Marking a source sensitive when a mail document is written** (in `ingest_raw`). It would miss mail already indexed, mail inside an archive and every path that does not pass there; the read in §7 covers all three.
- **Deciding sensitivity from the file extension.** `.eml` content is mail whatever it is called, and
  a name is a thing a person changes; the media type the pipeline routed the bytes by is what the
  parser will treat as mail.
- **A Gmail change to record `mail_message_id`.** It would change what Gmail fetches, and the link
  already carries what is needed (§6).
- **Applying the Gmail skip in reconciliation.** §5.4.
- **`.emlx` in this connector.** §2.

---

## 11. Where this will be implemented

| Part | Module |
|---|---|
| Configuration, scope fingerprint | `manicule/connectors/config.py` (`MboxConfig`) |
| The reader, canonicalization, identity | `manicule/connectors/mbox_reader.py` |
| The connector: walk, watermark, discover, fetch, reconcile | `manicule/connectors/mbox.py` |
| Registration (sensitive) | `manicule/connectors/plugin.py` |
| Sensitive by content | `manicule/app/sensitive.py` |
| The Message-ID read | `manicule/storage/docstore.py` (a streaming method beside `known_source_ids`) |
| Fixture generator | `tests/connectors/mbox_support.py` |
| Tests | `tests/connectors/test_mbox_*.py` |

The build PR replaces this table with the real one and moves the status banner.
