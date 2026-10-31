# Release notes

## Unreleased

### Passwords for local people: a browser signs in with a handle and password

Under `security.auth.mode = "api_key"` with a `security.auth.session_secret`, a person created
here can now sign in with a handle and password at `POST /auth/password`, and `/ui/login` shows
that form beside the key form. The session is bound to no key, so the person has the full role
their membership gives them. A person may have a password, keys, or both.

- **Create with a password.** `manicule auth add-user HANDLE --password` prompts for it (no echo;
  or reads one line from a pipe), `POST /api/v1/auth/users` takes an optional `password`, and so
  does the Add a person form. With a password the person gets no first key and mints their own
  after signing in.
- **Change your own.** Any signed-in person, whatever their role, can change their own password
  from the **Your password** panel on the API keys page or `POST /api/v1/auth/password`. It needs
  the current password, and it must come from a browser a password started, not a presented key and
  not a key session. It ends every other session and issues a fresh cookie.
- **Recovery.** Manicule sends no mail, so an administrator sets a new password:
  `manicule auth set-password USER`, `PUT /api/v1/auth/users/{user}/password`, or Set password on
  `/ui/users`. That ends the person's sessions and leaves their keys alone.
- **Policy.** At least 12 characters, at most 256 bytes, not only whitespace. Normalized to NFKC;
  nothing is stripped. Stored through scrypt (`hashlib`, no new dependency) as a string that names
  its own cost.
- **Failures.** An unknown handle, a person an identity provider vouches for, a person with no
  password, a disabled person and a wrong password all get one refusal (`403`, reason
  `bad_credentials`) and cost the same work. Failures are charged to the same per-address bucket,
  and counted toward the same `brute_force` alert, as a wrong key.
- **A deliberate difference from keys.** An address that has used up its failed-authentication
  allowance is refused for passwords *before* the password is checked (`429` with `Retry-After`),
  because a password a person chose can be guessed. People behind one shared address share that
  budget for mistyped passwords; keys are unaffected. There is no per-person lockout, since it
  would let anyone who knows a handle lock that person out. Guessing spread across many addresses
  is bounded only by the cost of scrypt and by the buckets being per process.
- **Where a password goes.** Only in a request body, read by a bounded hand-written parser, so a
  rejected request never echoes it. Never in a URL, an MCP tool parameter or a command-line
  argument, and the hash is in no result (a person is reported with `has_password`).

Upgrading runs one migration (`d2f5b8e31a64`), which adds a nullable `users.password_hash`.
Existing people have no password until one is set. See `surfaces.md` §9.2.1.

**Decisions that need review:**

1. No forced change after an administrator sets a password; the person is expected to change it
   and nothing enforces it.
2. No emailed reset, because manicule sends no mail.
3. No per-person lockout, for the reason above.
4. No password for identity-provider people.
5. Passwords cannot be cleared on their own. Disabling a person clears their password along with
   their sessions and keys, in one transaction, so enabling them again restores the membership
   and not the password; an administrator sets a new one.
6. Existing hashes are not re-hashed on sign-in when the cost is raised, because re-hashing would
   end the person's other sessions.

### People without an identity provider: local users, and a browser signs in with a key

Under `security.auth.mode = "api_key"` an administrator can now create a person:
`manicule auth add-user HANDLE [--role admin|member|viewer]`, `POST /api/v1/auth/users`, or the
**Add a person** form on `/ui/users`. The person has no email; they get a first
key named `<handle>-key`, shown once. `manicule auth create-key NAME --user HANDLE` (and `user` on
`POST /api/v1/auth/keys`) mints more.

With `security.auth.session_secret` set (at least 32 characters, usually through
`MANICULE_SECURITY__AUTH__SESSION_SECRET`), `/ui/login` shows a key form and a person can sign a
browser in with one of their own keys. The session is bound to that key: it carries the lesser of
the key's role and the person's, honors the key's `allowed_ips` on every request, and ends when
the key is revoked or expires or the person is disabled. A key no person owns cannot sign a
browser in. Without a `session_secret` nothing changes and sign-in is simply not offered; `manicule
doctor` says how to turn it on. Upgrading runs one migration (`c9a4e7d21b30`). To give an existing
operator key a browser identity, mint a personal key with `auth create-key NAME --user HANDLE` and
revoke the old one. See `surfaces.md` §9.2.1.

### Old mail archives: the `mbox` source, and mail is now sensitive wherever it came from

A `type = "mbox"` source indexes the unix mbox files under a root, one document per message,
through the same mail parser as Gmail. A file is recognized by its first bytes, so Thunderbird
folders with no extension, Mutt archives and an Apple Mail "Export Mailbox" folder all work. The
files are read as a stream: a ten-gigabyte `Inbox` is fine, because the fetch limit applies to one
message and not to the archive. A message is identified by a digest of its bytes, so a message
filed in several folders is one document and rearranging a file loses nothing. After the first
sync an unchanged file costs a `stat`.

`skip_known_message_ids` is on by default: a message whose `Message-ID` is already held by a
`gmail` or another `mbox` source is not indexed again. Turn it off to keep both copies, for
instance when the archive is the only record of mail that shares an id. Messages with no
`Message-ID` are never skipped.

Outlook PST files have no reader and will not get one in the product, because the libraries that
read them are LGPL or GPL. [`connectors/mbox.md`](connectors/mbox.md) §9 is the recipe: convert with
`readpst` outside manicule, then index the output.

**An installation holding mail now refuses to be served without authentication, whichever
connector indexed it.** The refusal that already covered Gmail also fires when any document in the
data directory is a mail message (`message/rfc822`) or an Outlook message
(`application/vnd.ms-outlook`), so a directory of `.eml` files indexed through the filesystem
connector is covered too. This applies to mail you indexed before upgrading: the check reads the
documents that are already there. A directory that merely contains a sample `.eml` is covered as
well; exclude the file or serve with authentication. `manicule document delete-source NAME --yes`
lifts it, as before. [`connectors/mbox.md`](connectors/mbox.md) §7.

### A reply chain no longer fills a search with copies of one message

Mail is indexed one document per message, and each message keeps the reply chain it quotes, so a
long thread repeats its early messages in every later one. Measured on synthetic threads through
the shipped pipeline, a question about a long thread got the same paragraph back from several
replies, and a separate message answering the same question was pushed off the page.

Retrieval now has a `collapse` stage, after fusion, in the default `rag.pipeline`. It drops a hit
whose text already appeared higher up the same results from the same conversation, with `>`
quote markers ignored, and changes nothing else: what is kept keeps its rank, and a source that
records no conversation is not affected at all. Gmail records one per thread. Whichever copy
ranked first is the one kept, so when a reply quoting a message outranks it, the passage is
cited to the reply. A reply that restates something with a few words changed is a correction,
not a copy, and is kept. A page can also hold fewer hits than asked for, since what was dropped
is not replaced — about eight of ten on threads of twenty messages or more. A configuration
that sets `rag.pipeline` itself keeps the pipeline it names; add `"collapse"` after `"rrf"` to
use it. Every search's recorded pipeline identity now includes the new stage, so a result
recorded before this release is not comparable with one recorded after it.

[`connectors/gmail.md`](connectors/gmail.md) §8 has the measurement and the program that re-runs
it; [`retrieval.md`](retrieval.md) §5.5 the rule.

### The Gmail source reads a mailbox

A `type = "gmail"` source now syncs one Google Workspace mailbox, read-only. The first sync walks
the mailbox once; every sync after it reads Gmail's own change log from a stored `historyId`, so it
costs what changed rather than the mailbox, and a message already stored is never fetched again.
Each message is fetched raw and indexed through the existing `message/rfc822` parser, attachments
included. Scope is label ids only: SPAM, TRASH and DRAFT are always out, `exclude_labels` takes
label names, and there is no search query. A message deleted, trashed, marked spam or given an
excluded label leaves the index at the next reconciliation, which a served manicule runs on
`reconcile_interval_s` and after every complete walk; set `reconcile_interval_s` on a mail source
to bound how long deleted mail stays searchable. Fetches are paced under Gmail's per-user quota,
about a hundred raw messages a minute, so a first sync of a large mailbox takes hours.

Read [`connectors/gmail.md`](connectors/gmail.md) §9 and §10 before pointing this at a real
mailbox: whoever can search the workspace reads every message, so keep a Gmail source in a
workspace of its own.

Configuration, the credential and a check in `manicule doctor` shipped first:

`manicule doctor` gained a `gmail-credential` check. For each enabled Gmail source it refreshes
a token and asks Google which mailbox it reads, so a revoked grant or a token for the wrong
account shows up before a sync does. It contacts Google, once per enabled Gmail source, and
reports source names and a fixed word for each outcome, never an address. With no Gmail source it
is `ok` and does nothing.

Separately, the ingest pipeline can now hold a source to the account it first synced as. A
connector may declare a pinned identity, and editing it on a source that still holds documents
is refused before anything is read, with the instruction to use a new name for the other
account. Only Gmail declares one, so no existing source is affected.

### `manicule connector login` mints a Gmail refresh token

For a `type = "gmail"` source, `connector login <name>` now runs Google's installed-app flow: it
opens your browser, you sign in as the account the source indexes, and the refresh token comes
back to a port opened on `127.0.0.1` for that one sign-in (PKCE, `state` checked, `gmail.readonly`
only). The token is checked with one refresh and one `getProfile`, and one for any mailbox other
than the source's `account` is refused. It is written to `--token-file` (created readable only by
you) or to standard output when that is not a terminal, and never to the configuration file, a
terminal or a log. No server has to be running. The `gmail-credential` `doctor` remedy now names
the command. Steps and placeholders for the secret-store hand-off are in
[`connectors/gmail.md`](connectors/gmail.md) §1.3.

### A repeated search is served from the query cache

The query cache never hit through the CLI, HTTP, MCP or the web. Its invalidation counter counted
every committed transaction on the database, and every search records itself in the query log
after it runs — so the search's own log row invalidated the ranking it had just cached, and the
next identical search ran the whole pipeline again. An answer's conversation turn and a spanning
search's audit entry did the same.

A commit that wrote only the record of who asked what — the query log, the audit trail and its
alerts, conversations and their turns, and the people, memberships, sessions and keys that
identify a caller — no longer moves the counter. Everything else still does, including any raw
SQL statement whatever table it names, so a change to the corpus still invalidates every cached
ranking at once. `retrieval.md` §10.3 has the rule. Nothing to configure, and no ranking changes:
a hit re-hydrates the same ids through the same join as before.

### A Qdrant collection's memory and search settings are now configuration

Seven settings under `storage.qdrant` shape the collections manicule keeps on a Qdrant server:
`quantization` and `quantization_always_ram`, `on_disk_vectors` and `on_disk_payload`, `hnsw_m`
and `hnsw_ef_construct`, and `indexing_threshold_kb`. Until now a collection was created with a
size and a distance and nothing else, so it got whatever the running server's defaults were and
no setting anywhere could say otherwise. On a small corpus that costs nothing. On a 320,000-chunk
corpus at 1024 dimensions it is about 1.3 GB of `float32` vectors held in RAM, beside a payload
that carries a second copy of the corpus text.

`quantization = "scalar"` with `on_disk_vectors = true` is the combination for a corpus of that
size: search picks its candidates from an int8 copy of about 330 MB kept in RAM and scores only
those against the originals, which stay on disk. Search asks for that rescoring explicitly,
because Qdrant does not rescore scalar quantization by default and a score against the copy is a
score no checksum covers. The stored vectors, and every checksum over them, are
untouched. None of the seven changes which chunks a filter admits, so none is part of a
collection's name and changing one re-embeds nothing; the graph and quantization settings move
recall, which can move which admitted chunks reach the top of a ranking. `deployment.md` §6.5 has
the table and the example.

**Upgrading changes nothing on a stock Qdrant.** Every default is the value Qdrant gives a
collection nobody tuned, which is what every existing collection already is, and a collection
that already matches is never written to. A server whose own `config.yaml` sets other defaults
for these dials is the exception: the first time manicule opens its existing collections after
the upgrade, they are brought to manicule's values, because the collection is now described by
manicule's configuration rather than by the server's.

**A setting reaches the collection you already have**, not only one created after it. Each time the
store is prepared, manicule reads the collection back, sends one update carrying only what differs,
and logs a `reshaped Qdrant collection` line naming each change. If the server accepts the update
and still reports the old value — which is what a Qdrant too old to know the field does — the store
refuses to open and names every such setting, rather than leaving it reading as configured.

**A collection manicule cannot use is now refused when it is opened.** Named vectors, the wrong
size, a distance other than cosine, multivectors, or a datatype other than `float32` — a collection
made by hand, say, or restored from another installation's snapshot — used to fail every write with a
server error that named no cause, rank with scores nothing was calibrated against, or, for a
`float16` or `uint8` datatype, read as entirely corrupt and drop out of search while every
request succeeded. The refusal names `manicule reset-index`, or a `collection_prefix` of the
installation's own.

That last case is also why there is no `datatype` setting. The checksum is taken over the
`float32` values a point stores, and a collection of any other datatype hands back different
numbers. Scalar quantization is the memory saving that keeps the originals. `storage.md` §6.7 is
the design.

### An existing vector index can move to another backend without being re-embedded

`manicule migrate-vectors` carries a workspace's vectors from the embedded LanceDB directory
into the configured store, calling no embedder. Until now an installation that wanted its index
on a server had two options and both of them threw the vectors away: re-index the corpus, or
restore a snapshot the destination happened to have taken for itself. The vectors were sitting
in a directory the whole time.

They can be moved because the two backends already store the same row. A Lance column and a
Qdrant payload field carry the same names, written from the same object under the same rules, so
a migration is a read in one shape and a write in the same shape rather than a conversion. What
that buys is the difference between a read of a directory and a forward pass of the model over
every chunk in the corpus.

The order is: set `storage.vector_db` and `storage.vector_db_url` to the destination, leave the
`vectors/` directory where it is, then run `manicule migrate-vectors` to see what would move and
`--yes` to move it. Configuring first is deliberate — it is what puts the data-policy refusals in
front of the copy, so a corpus configured `local_only` is refused before a chunk of text leaves
the machine. Afterwards, `manicule vector-checksum --verify` reads the destination, and once it
is serving searches the `vectors/` directory is no longer read.

It plans by default and the plan creates nothing at the destination, so running it to decide
whether to migrate at all leaves an installation untouched. A good deal is refused rather than
worked around: a destination that already holds rows, a rebuild or durable re-embed still in
flight, a source store that records no embedding fingerprint at all, a source whose fingerprint
disagrees with the one the corpus recorded, a destination built for a different model, and a
copy that ends with the destination holding less than the source did.

The one worth knowing about is none of those. Any row whose stored numbers no longer match the
checksum written beside it stops the copy, rather than being carried somewhere the original is
no longer there to be compared against — or skipped, which would leave the destination quietly
short of a corpus nothing downstream asks about completeness.

A vector's recorded checksum and embedding identity are carried rather than recomputed, which is
what keeps both of those properties true: a recomputed checksum would describe whatever arrived
and certify a drifted vector as intact, and a recomputed identity would make every migrated row
miss the reuse lookup on an installation whose `embed_text` middleware had changed — re-embedding
the corpus the migration was run to avoid re-embedding.

One measured caveat, recorded because it is the kind of thing that otherwise gets rediscovered
during an incident: a vector store whose distance is cosine may re-normalize on write, and
whether that moves a stored `float32` depends on the implementation rather than on the vector.
`qdrant/qdrant` does not move it — 0 of 500 random unit vectors — so a migrated corpus verifies
against the server an installation actually runs. `qdrant-client`'s in-process mode, which is a
test convenience and not a backend anybody serves from, moves about one vector in eleven.

What ships is LanceDB to whichever adopting destination is configured, which today means
Qdrant. The copy itself names no backend — the source is asked for one capability and the
destination for another, both protocols — so a backend added later becomes a *destination* by
implementing `AdoptingVectorStore` and nothing else. Becoming a *source* needs more than the
capability, because finding the generation to read is the embedded store's own business. `storage.md` §6.8 is the design;
§6.9 is what durable re-embedding on a networked backend would take, which is still refused and
now says why in enough detail to be decided on.

### A corpus that belongs to no collection now says so

`manicule doctor` gained a `collection-membership` check, and `connector sync`, `index` and
`import` gained two numbers: `in collections` and `in no collection`, for the source the run
was over.

Both exist because of a failure that is quiet in the worst way. Collections and their rules live
in the document store, so rebuilding or restoring that store without recreating them leaves a
corpus whose documents are all perfectly indexed and whose organization is gone. Every signal
there is says the installation is healthy: the sync reports `503 indexed, 0 failed, outcome
complete`, `index --stats` counts 503 documents, and an unscoped search answers. Only a
collection-scoped search fails, and it fails by refusing a name — so an operator who does not
happen to scope one may not notice for a long time, while everything they do search is narrower
than they think.

The check reports three numbers — how many documents, how many collections, and how many
documents no collection holds, by hand or by rule. A workspace with documents and **no
collections at all** is `degraded`: there, a scoped `search` refuses the scope,
`collection_counts` refuses the name and `document_create` refuses the write, so a whole surface
of the product answers refusals while nothing says why. That is also what a corpus nobody
organizes by collection looks like, and the two cannot be told apart from inside the check —
which is the reason it reports the state rather than guessing which one it is. Documents outside
collections that *do* exist stay `ok` with the count in the sentence, because collections are
optional and amber on every partly-filed corpus is how a reader learns to skim `doctor`.

The sync numbers answer the same question about one source, in its own output rather than in a
later refusal. They are measured when the run finishes rather than counted during it, because
nothing in the ingest path ever writes a membership row: rule-driven membership is evaluated at
read time, and a document a run skipped as unchanged is in exactly the collections one it
indexed is. A run that placed nothing anywhere now says so on the line under its table.

Neither is a new way to find these documents — `manicule collection orphans` already lists them,
and still reports rather than removes unless asked to. What is new is that nobody has to know to
run it.

### A collection can be a directory, so a synced file joins it

A collection rule now takes `uri_prefixes`, and a document whose location sits beneath one is a
member. `manicule collection rule set <id> --uri-prefix /corpus/journals` is the whole of it,
and `collection create --uri-prefix` sets one at creation. Every surface carries the new
selector: CLI, HTTP API, control socket and the writable MCP server.

This closes a gap that only showed up as missing search results. `authoring.collections` has
always held that a collection name is also the directory its documents land in, but only
`document_create` acted on that, by adding each document it wrote to the collection by hand. A
file that reached the same directory any other way — a `connector sync` over a tree pulled from
git, an editor, a write whose ingest failed and that a later sync picked up — joined nothing.
The corpus reported it by returning fewer results to a collection-scoped search, with nothing
raised and nothing logged. With a prefix the directory *is* the membership, so a document joins
by arriving.

Prefixes are matched against the document's location rather than its identity, which is what
lets a document that moves between directories change collection at read time and without being
re-indexed. Write one as an ordinary absolute path; it is stored as the `file:` URI the
connector records, and always with a trailing separator, so `/corpus/journals` cannot also
select `/corpus/journals-old`. A URL works too, so a mirrored space can be selected by the
address it mirrors. Membership stays evaluated rather than materialized, and manual members are
still unioned with whatever the rule selects.

`manicule doctor` gained an `authoring` check for the state this replaces. It reports `degraded`
for a collection that authoring writes into but whose rule does not select its own directory —
naming the collection and the exact command to fix it — and `failing` for a configured
collection this workspace does not have, which `document_create` refuses on but only once
somebody tries to write.

### A slow deep offset no longer stops a Data Center inventory

The authoritative Server/Data Center `direct_current_content` walk now converges on a large
space whose content endpoint gets slower as the offset grows. When a request times out at an
offset, the connector retries the same offset and the same immutable scope with a smaller
requested page — halving down to `adaptive_min_page_size` — and changes nothing else about the
request. Before this, a fixed page size failed at the same offset on every attempt while the
source was reachable and could answer that offset a smaller page.

Completion is now proved only by a validated explicit empty page. The walk advances by the rows
each response actually returned and no longer follows the native `next` link, whose absence at a
round offset has been observed while later offsets still held rows. A short page, a timeout, a
locally expected count and a search aggregate are all still not ends.

Four new per-source options bound the adaptation — `adaptive_min_page_size`,
`adaptive_max_attempts_per_offset`, `adaptive_max_seconds_per_offset` and
`adaptive_page_size_growth` — and none is part of the scope fingerprint, so retuning them never
forces a re-enumeration. Only a read timeout or an explicitly classified transient gateway
timeout adapts; authentication, authorization, malformed or untrusted responses, cancellation,
lease loss, storage failures and ordinary 4xx responses are unchanged. Exhausting the bounded
policy raises the original typed timeout, keeps the durable prefix, releases the lease, withholds
promotion and the watermark, and does not reconcile deletions.

Ingest results and snapshot status now carry private-safe aggregate enumeration progress —
current offset as a count, effective requested page size, timeout retries, whether the page size
was reduced, whether the walk reached its explicit empty page, and a typed
`enumeration_failure_code` — across CLI, JSON, HTTP, MCP, control and web surfaces. A walk
shrinking its pages to survive source latency is now distinguishable from a hung one.

The direct inventory also asks for less per row: pages expand `version,space` and attachments
`version,space,container`. Ancestry, bodies and full provenance come from the subsequent item
fetch, which is the response whose bytes are retained.

### Rule-driven collection management

Collection rules are now available through the application service, CLI, HTTP API, control
socket, and writable MCP server. A collection can select documents by source, directory, media
type, tag, or update bounds when it is created, and its rule can later be shown, replaced, or
cleared.

Existing indexes can adopt these rules immediately. Membership remains evaluated at read time,
so matching documents already in the workspace and matching documents ingested later appear
without reconciliation. Rule management does not fetch sources, enumerate the corpus, ingest
documents, rebuild chunks, or re-embed content. Manual membership remains unioned with the rule,
and clearing a rule preserves those manual members.
