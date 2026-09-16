# Release notes

## Unreleased

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
