# Release notes

## Unreleased

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
