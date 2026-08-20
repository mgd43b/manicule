# Release notes

## Unreleased

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
socket, and writable MCP server. A collection can select documents by source, media type, tag,
or update bounds when it is created, and its rule can later be shown, replaced, or cleared.

Existing indexes can adopt these rules immediately. Membership remains evaluated at read time,
so matching documents already in the workspace and matching documents ingested later appear
without reconciliation. Rule management does not fetch sources, enumerate the corpus, ingest
documents, rebuild chunks, or re-embed content. Manual membership remains unioned with the rule,
and clearing a rule preserves those manual members.
