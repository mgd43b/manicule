# Website ingestion: a Git-backed site first, a crawler second

The feature is "index this website". It is not one connector with a local mode and a network
mode.

The first source is a Git checkout containing pages we own. The later source is an HTTP crawler.
They share route normalization, page metadata and HTML parsing, but they do not share discovery,
fetching, credentials, failure modes or security boundaries. They are separate connector types for
the same reason `confluence` and `confluence-snapshot` are separate: a configuration in which half
the fields are forbidden depending on one mode flag is two sources hidden inside one model.

The implementation order is therefore:

1. **`git-site`** indexes a pinned revision of a local repository. It is the useful first release
   for content already on disk, and requires no connector-protocol changes.
2. **`web-crawler` with an authoritative URL inventory** indexes the published site from a sitemap
   or another complete manifest.
3. **Link-traversal crawling** follows only after the ingest pipeline has a durable fetched-
   discovery seam. A crawler learns the next URLs by reading the current page; pretending that is
   ordinary discovery either downloads every page twice or keeps corpus-sized response bodies in
   process memory.

Neither connector owns embeddings, chunking, retrieval or MCP. It ends at `RawDocument`, like the
existing sources. The normal pipeline optionally retains source bytes, then parses, chunks, embeds
and publishes; registration makes the source available through the existing CLI, HTTP and MCP
surfaces.

## 1. The first connector: `git-site`

`filesystem` can already index the repository today. `git-site` adds the facts a filesystem does
not know:

- which files are pages;
- the stable identity of each page;
- the canonical URL a citation should show;
- a Git blob id as a deterministic change token;
- a commit-pinned inventory that cannot change halfway through a sync; and
- route changes as changes to the document, even when its body is unchanged.

It reads Git objects, not the mutable working tree. The default `revision = "HEAD"` is resolved to
one commit at the start of each sync and every operation in that sync uses that commit. A later
sync resolves the revision again, so new commits become visible without restarting the process.
`discover` and `reconcile` can therefore agree even if another process checks out or commits while
the sync is running, and `fetch` can retrieve the exact blob discovery named.

Indexing drafts from a dirty working tree is deliberately absent from the first release. It has a
different consistency problem: the bytes can change between discovery and fetch, and untracked
files have no committed inventory. A later `git-worktree` connector or explicit snapshot command
can solve that without weakening the commit-pinned source.

### 1.1 Configuration

```toml
[connectors.product-docs]
type = "git-site"
schedule_s = 300

[connectors.product-docs.options]
repository = "/srv/product-docs"
revision = "HEAD"
content_root = "docs"
base_url = "https://docs.example.com/"
include = ["**/*.md", "**/*.mdx", "**/*.html"]
exclude = ["**/_partials/**", "**/drafts/**"]
route_manifest = ".manicule-site.json"
```

`GitSiteConfig` is frozen and rejects extra fields. Its fields are:

| Field | Default | Meaning |
|---|---|---|
| `repository` | required | Absolute or resolvable local Git worktree. Reads are bounded to this repository. |
| `revision` | `HEAD` | Commit-ish resolved once at setup. It must resolve to a commit. |
| `content_root` | `.` | Repository-relative boundary containing page inputs. It may not escape the repository. |
| `base_url` | required | Absolute HTTP(S) site root, with no credentials, query or fragment. |
| `include` | page suffixes | Git-style path globs admitted as pages. |
| `exclude` | tool/draft paths | Globs applied after `include`; an exclusion always wins. |
| `route_manifest` | absent | Optional repository-relative v1 route manifest. |
| `max_bytes` | absent | Optional per-page ceiling. The Git reader retains its 256 MiB safety ceiling when absent. |

The repository, revision name, content boundary, route policy and base URL are configuration, not
content. Their canonical serialization is the connector's scope fingerprint. The resolved commit
is not part of the scope: it changes on every successful update and belongs in the watermark.
Changing `base_url`, the route manifest path or the admitted roots forces a complete replacement
enumeration rather than applying an old watermark to a different site.

### 1.2 Route manifest

Path conventions are convenient until a site has one permalink, one generated route or one page
whose public address does not match its filename. The optional manifest is the authoritative route
inventory for those sites:

```json
{
  "version": 1,
  "pages": [
    {
      "source": "docs/getting-started.md",
      "id": "getting-started",
      "route": "/getting-started/",
      "title": "Getting started",
      "media_type": "text/markdown"
    }
  ]
}
```

The manifest is read from the pinned commit. It is bounded in bytes and page count, rejects extra
fields, and is validated before any document is yielded. Every `source` must be a normalized path
under `content_root`, exist as a blob in the same commit, match the include/exclude rules and occur
exactly once. Every non-empty `id` and normalized route must be unique.

When a manifest is present it is authoritative: a matched file omitted from it is not silently
indexed under a guessed URL. That makes a stale generator output visible instead of creating a
mixture of deliberate and invented routes.

Without a manifest, routes follow one small declared convention:

- remove `content_root`;
- turn `index.html`, `index.md` and `index.mdx` into their parent route;
- remove one recognized page suffix from other files;
- preserve directory segments and case;
- percent-encode each URL path segment; and
- join the result to `base_url` without allowing it to change origin.

There are no framework probes. Looking for Docusaurus, Hugo, Jekyll or Next.js configuration and
guessing what each release would publish makes route identity depend on whichever framework parser
happened to be installed. A site with different rules supplies the manifest from its own build,
where the routing authority already lives.

### 1.3 Identity

Identity priority is:

1. `id` in the route manifest;
2. the normalized route, relative to the configured site root.

The local path is never the default identity. A repository can move, and a checkout on another
machine must produce the same documents. A route rename is a delete plus an add unless the manifest
keeps the same explicit id; that makes the migration an author decision instead of a fuzzy rename
guess based on content.

`DocRef.metadata` carries the repository-relative source path, Git blob id, canonical URL and route
record digest needed by `fetch`. It never carries the repository's absolute path. Two files claiming
one id or route are both refused before discovery completes; choosing the first would silently lose
a page and choosing the second would silently overwrite it at `(workspace, connector, source_id)`.

### 1.4 Discovery and change detection

At setup:

1. resolve `revision^{commit}`;
2. list the commit tree below `content_root` in stable bytewise path order;
3. read and validate the route manifest, if configured; and
4. build a bounded page inventory containing no body bytes.

Only ordinary Git blobs are pages. Symlinks and submodule entries are refused when a manifest names
them and otherwise excluded from the inferred inventory; following either would make a pinned tree
reach bytes outside the commit whose identity it claims.

`discover` yields the complete inventory on every sync. Git tree enumeration is local and cheap;
the existing pipeline compares per-document tokens before fetching, so an unchanged page never
reads its blob, parses or embeds. An optimization may later use `git diff-tree` when the old commit
is an ancestor, but it must not replace the full reconciliation inventory or make force-pushes an
incremental claim.

The version token is a digest of:

```text
git blob id
route-record digest
connector extraction/profile version
```

The route digest is load-bearing. Changing a title, canonical URL, media type or explicit identity
changes what the document says and how a citation renders even when its body blob is identical.

The watermark is the resolved commit id plus the route-policy version, observed only after the
complete discovery stream reaches its end. Abandoning the stream leaves the previous watermark in
place, as required by the connector contract.

### 1.5 Fetch and provenance

`fetch` uses `git cat-file blob <id>` (or an equivalent batch process) against the pinned
repository. It verifies that the requested blob is the one in the pinned inventory and applies the
configured byte limit before returning it. It never opens a path supplied by `DocRef` and never
runs a Git hook, checkout, build or submodule command.

The returned `RawDocument` has:

- `source_id`: the manifest id or normalized route;
- `uri`: the canonical HTTP(S) URL, not a `file:` URI;
- `media_type`: manifest declaration or the project's deterministic suffix table;
- `content`: the exact Git blob bytes; and
- `metadata.source_provenance`: a validated `Provenance` record.

`SourceMetadata` records the title when declared, canonical URL, stable source id, Git blob id as
the source version and the published content type. `LocalSnapshot.path` is the repository-relative
path. It has no `retrieved_at`: a commit timestamp is when a commit was written, not when a page was
published or this checkout was retrieved.

Raw Markdown and MDX use the existing parsers. HTML uses the existing structural parser, preserving
the page's own heading ids, tables, lists and code. If the public site's fragments differ from the
source Markdown parser's slug convention, the route manifest should point at committed rendered
HTML instead; the connector does not claim that one generator's invented fragments are another
generator's published addresses.

### 1.6 Referencing the repository instead of retaining another copy

For an owned repository that is guaranteed to remain mounted, the website should be able to opt out
without changing retention for every other source:

```toml
[connectors.product-docs]
type = "git-site"
retain_source_bytes = false

[connectors.product-docs.options]
repository = "/srv/product-docs"
revision = "HEAD"
content_root = "docs"
base_url = "https://docs.example.com/"
```

The connector still reads and hashes the exact Git blob during sync, but Manicule leaves
`original_ref` unset and does not write the bytes below `<data_dir>/blobs`. The configured repository
plus the pinned commit and blob id are the external source of truth. This is the smallest useful
mode for the content described here.

`retain_source_bytes` belongs on `ConnectorSettings`, beside `enabled` and `schedule_s`, rather than
inside `GitSiteConfig.options`. It is a policy of the ingest run, not a capability or interpretation
of this connector. Every connector may use it, including two websites with different policies:

```toml
[storage]
retain_source_bytes = true       # installation default

[connectors.local-docs]
type = "git-site"
retain_source_bytes = false      # reference the durable local repository

[connectors.public-site]
type = "web-crawler"
retain_source_bytes = true       # keep network responses for offline rebuild
```

Its model is `bool | None`, defaulting to `None`: `None` inherits
`storage.retain_source_bytes`, while an explicit boolean overrides it for that connector instance.
One effective value is resolved at the start of a sync and remains fixed for the whole run.

The reference must be `(repository, commit id, blob id)`, not merely a working-tree path. A path can
be edited in place and then no longer contains the bytes that produced the indexed revision. A Git
object is immutable; it can reconstruct those bytes as long as the repository and object remain
available.

This is an operational trade rather than a transparent storage optimization:

- `document reindex`, citation verification against source bytes, connector-free rebuild, portable
  export and a self-contained Manicule backup currently read `original_ref` from the blob store;
  with retention off they cannot silently fall back to the repository;
- re-parsing requires another `connector sync`, which resolves the configured revision again;
- moving or deleting the repository makes the source unavailable; and
- a force-push followed by Git garbage collection can remove an old commit and its blobs.

That behavior is already the contract of disabled retention: every re-index is a re-sync. It is
appropriate when the repository itself is backed up and the operator accepts that dependency.

Do not put a filesystem or `git:` URI into `documents.original_ref` in the first release. Today that
field is a foreign key to Manicule's content-addressed blob table, and readers rely on every non-null
reference resolving and hashing to its own name. Supporting external originals while preserving
ordinary reindex and verification would require a real `OriginalResolver` abstraction, for example
`blob:sha256:... | git:<repository-id>:<commit>:<blob>`, plus schema, backup, health and garbage-
collection rules. That is a later feature, not a shortcut in this connector.

The shared pipeline makes the implementation detail load-bearing. Scheduled and requested syncs for
different sources may overlap, so the application passes the effective policy into
`IngestPipeline.run`; `_Sync` carries its own `BlobSink` and optional acquisition store. A retained
run uses the ordinary `BlobStore` and durable acquisition path. A reference-only run uses
`NoRetention` and the direct bounded pipeline. Never replace `IngestPipeline._blobs` for one run:
another connector may be using it concurrently.

`acquire-only` refuses a connector whose effective policy is false, because there are no retained
bytes to acquire. If a retained durable run was already in progress when configuration changes, its
persisted policy wins until that run settles; the new policy applies to the next run rather than
stranding a resumable snapshot halfway through its lifecycle.

Changing `false` to `true` makes a settled document with no `original_ref` ineligible for the token
skip once, so it is fetched and retained even when its source version is unchanged. Changing `true`
to `false` stops new retention; it does not silently delete blobs already referenced by current or
historical document versions. Releasing those bytes remains an explicit lifecycle plan and confirmed
cleanup operation.

### 1.7 Deletions

`reconcile` yields every source id in the same pinned inventory used by discovery. This is a full,
authoritative pass. A removed file, excluded route or removed manifest entry is absent and becomes
a deletion candidate through the existing reconciliation ceiling.

An unreadable repository, invalid manifest, unresolved revision, duplicate identity or interrupted
tree walk is a failed enumeration, never an empty site. No watermark advances and no deletion is
proved. A configuration change gets a new scope fingerprint and a complete replacement run.

### 1.8 Git process boundary

Use one long-lived `git cat-file --batch` subprocess per connector instance for blob sizes and
content. Invoke Git with a fixed argv, `cwd=repository`, `GIT_OPTIONAL_LOCKS=0`, and prompts and
external helpers disabled. Do not invoke a shell. Put user-configured revisions after Git's
end-of-options marker, resolve them once, validate the resulting object id, and use only that object
id in later commands. A fixed argv prevents shell injection; the end-of-options boundary separately
prevents a revision beginning with `-` from becoming a Git option. Stderr is bounded and converted
to typed connector errors without reflecting repository content.

The first implementation may use one fixed-argv subprocess per operation if that is clearer. The
batch process is a performance improvement, not part of the contract.

## 2. The network connector: `web-crawler`

The network source shares `SiteRoute`, URL normalization and provenance construction with
`git-site`. Everything else is separate.

### 2.1 Configuration and scope

```toml
[connectors.public-docs]
type = "web-crawler"

[connectors.public-docs.options]
start_urls = ["https://docs.example.com/sitemap.xml"]
allowed_origins = ["https://docs.example.com"]
allowed_path_prefixes = ["/"]
exclude = ["/search", "/preview/**"]
inventory = "sitemap"
obey_robots = true
max_pages = 50000
max_depth = 8
max_response_bytes = 8388608
concurrency = 8
per_origin_concurrency = 2
request_delay_s = 0.25
```

All URL-bearing configuration is normalized and secret-free. The scope fingerprint covers start
URLs, allowed origins and paths, exclusions, query policy, canonicalization policy and inventory
mode. Operational dials such as timeouts and concurrency do not change membership and do not belong
in the fingerprint.

The first network release supports `inventory = "sitemap"`. The sitemap and sitemap-index walk is
the complete identity inventory, bounded by `max_pages`, compressed-byte and expanded-byte limits,
nesting depth and cycle detection. Sitemap `<lastmod>` is recorded as metadata but is not trusted as
the sole version token: publishers routinely leave it stale.

### 2.2 URL normalization

One function is used for start URLs, sitemap entries, redirects, canonical links, discovered links
and reconciliation:

- allow only `http` and `https`;
- lowercase scheme and host, IDNA-normalize the host and remove default ports;
- remove fragments before identity comparison;
- remove dot segments without decoding encoded `/` or `..` into a new path;
- reject user info;
- apply one explicit trailing-slash policy;
- drop known tracking parameters;
- default to rejecting other query-bearing URLs unless their keys are allowlisted; and
- re-check origin and path scope after every normalization and redirect.

Fragments are section addresses inside a document, never separate documents. Query strings are not
globally discarded because some sites publish real pages through them; admitting them is explicit
because unbounded calendars, searches and faceted navigation are crawler traps.

### 2.3 Security and politeness

Every outbound request is SSRF-sensitive input, including redirects and URLs found in pages the
owner controls today but another author may edit tomorrow.

- Resolve and connect only to allowed origins. Reject loopback, link-local, private, multicast,
  unspecified and reserved addresses by default, for both IPv4 and IPv6. Revalidate redirects and
  DNS answers rather than checking only the original spelling.
- Never forward authorization, cookies or custom secret headers across an origin change. The first
  release has no authenticated crawl; add credentials only with an origin-pinned credential model.
- Verify TLS by default. There is no silent HTTP downgrade.
- Enforce compressed and decompressed response limits while streaming, before building an HTML
  tree. Bound redirect count, sitemap nesting, pages, depth, links per page and total bytes per run.
- Send a declared user agent. Honor `robots.txt` by default, including disallow rules and the
  connector's stricter configured delay. A deliberate future bypass must be explicit, warning-level
  and unavailable for origins outside the operator allowlist.
- Apply global and per-origin concurrency, request delay, timeout and retry budgets. Retry only
  idempotent requests, respect `Retry-After`, and never turn a rate limit into an unbounded sleep.
- Accept page bodies only from an allowlist of textual media types. A PDF or office document linked
  from a page can later be admitted as an attachment policy; it is not HTML because its URL ends in
  no suffix.

A sign-in page, challenge page or generic error template is not content. Status, media type,
redirect history and configured content selectors are checked before a response becomes a page.

The transport foundation is implemented independently of connector registration. `CrawlerUrlPolicy`
canonicalizes every page URL, produces numeric `ConnectionPlan` dial targets only after all DNS
answers pass the public-address policy, and requires the socket adapter to verify the actual peer
against that plan. The adapter must dial those numeric targets directly while retaining the
original hostname for TLS verification; resolving the hostname a second time would reopen DNS
rebinding between policy and connection.

`CrawlerHttpClient` layers mandatory robots checks, per-origin/user-agent cache expiry, crawl delay,
global and per-origin permits, bounded retries and `Retry-After`, redirect revalidation, streamed
wire/decoded/run byte limits, a textual media allowlist and challenge detection over that plan. Its
public run surface has no robots-bypass parameter. Both pieces remain internal foundations until the
authoritative sitemap inventory and `web-crawler` connector are registered; their presence does not
yet make network crawling a configurable source.

### 2.4 Conditional change detection

For an authoritative URL inventory, discovery may issue bounded `HEAD` requests and use a strong
ETag as the version token. A weak ETag, `Last-Modified` or `Content-Length` is only a hint and cannot
prove equality alone. If the server does not offer a trustworthy token, discovery yields `None` and
the page is fetched; downstream content-hash dedup prevents an unchanged body from being republished.

Fetch pins the discovered revision with `If-Match` when a strong ETag exists. A precondition failure
or a different response ETag is `stale_body`, causing a later re-enumeration rather than publishing
bytes under a token they do not match.

The roadmap's "Conditional GET plus content hash" needs one small pipeline extension before it can
avoid both `HEAD` and a full unchanged `GET`: a conditional fetch must be able to report
`NotModified` and let acquisition reuse the already retained, hash-validated source envelope. The
current `Connector.fetch` returns only `RawDocument`; representing a 304 as an empty document would
publish a deletion-shaped lie. Add an optional protocol rather than widening every connector:

```python
class ConditionalFetchConnector(Protocol):
    async def fetch_if_changed(
        self, ref: DocRef, known_version: str | None
    ) -> RawDocument | NotModified: ...
```

The pipeline supplies the previously settled version, accepts `NotModified` only when it can reopen
and validate retained bytes, and records reuse in the acquisition manifest. Existing connectors
continue using `fetch` unchanged.

### 2.5 HTML extraction

Fetched bytes must remain the retained original so parser fixes can rebuild without a re-crawl.
Therefore boilerplate extraction belongs in a parser, not in the connector's HTTP client.

Register `text/html;profile=crawled-page` to a `CrawledWebParser`. It uses a bounded, deterministic
main-content selection step and then hands the selected subtree to the existing structural HTML
reader. Navigation, cookie banners and repeated footer text are removed without flattening tables,
code, lists or published heading ids. Its extraction library, rules and profile version participate
in `parse_fp`; changing them selects the affected documents for offline rebuild from retained HTML.

The general `text/html` parser remains unchanged. An uploaded HTML file and a crawled page are two
different jobs: the first should preserve the document supplied, while the second must remove site
chrome repeated on every page.

Canonical links are evidence, not authority. An in-scope `<link rel="canonical">` may replace the
display URL after normalization and duplicate checking. An out-of-scope canonical is recorded as a
diagnostic and ignored; it cannot move content or send credentials to another origin.

### 2.6 Why link traversal waits for a pipeline seam

Sitemaps separate inventory from bodies, so they fit `discover` then `fetch`. Arbitrary traversal
does not. To discover links from page B, the crawler has already fetched page A. With the current
protocol it must then either:

- discard A and download it again when `fetch(A)` is called;
- retain every response in connector memory until the pipeline asks for it; or
- put an ephemeral cache path in `DocRef`, which cannot survive a process restart and is not a
  durable acquisition record.

None is acceptable. Before `inventory = "links"`, add an optional fetched-discovery protocol whose
unit is a discovered reference plus its fetched source envelope. The acquisition journal commits
that pair and retained bytes together before the crawl frontier advances:

```python
class FetchedDiscoveryConnector(Protocol):
    def discover_fetched(
        self, watermark: Watermark | None
    ) -> AsyncIterator[FetchedDiscoveryPage]: ...
```

`FetchedDiscoveryPage` contains `DiscoveredDoc`, `RawDocument` and a bounded list of normalized
outlinks. The pipeline persists the discovered row and source blob, then acknowledges the page; only
then may its in-scope outlinks enter the durable frontier. The frontier is a SQLite-backed queue
scoped to the acquisition run, with unique normalized URLs, depth and parent evidence. Restarting
claims the same run and resumes without redownloading acknowledged pages.

Only exhaustion of that durable frontier proves a complete inventory and permits reconciliation.
A page limit, depth limit, cancellation, robots refusal, fetch failure or frontier corruption makes
the snapshot partial and cannot prove deletion. `allow_omissions` may publish a visibly partial
snapshot, but it cannot call absent URLs deleted.

## 3. Failure and deletion semantics

The following are source failures, not empty inventories:

- repository/ref/manifest cannot be read;
- sitemap or robots response fails validation;
- configured limits stop enumeration;
- a crawl frontier is not exhausted;
- redirects or canonical URLs escape scope;
- duplicate page identities are found; or
- a body changes between discovery and fetch.

They produce typed, aggregate-safe diagnostics and do not advance a watermark. URLs, repository
paths, page titles, response fragments and exception text do not enter shared status payloads.

Deletion is authorized only by a complete inventory under the same scope fingerprint:

- `git-site`: exhaustion of the pinned commit tree and route manifest;
- sitemap crawler: exhaustion of every sitemap in the bounded sitemap-index graph; and
- traversal crawler: exhaustion of the durable frontier with no limiting condition.

HTTP 404/410 during acquisition invalidates the completed inventory and requests re-enumeration,
using the recovery behavior the ingest pipeline already has for `source_deleted`. A timeout, 429 or
5xx is retryable and proves nothing about membership.

## 4. Where this lives

The first implementation adds:

```text
src/manicule/connectors/git_site.py        pinned tree, manifest, discovery, fetch, reconcile
src/manicule/connectors/site_routes.py     route models and canonical normalization
src/manicule/connectors/config.py          GitSiteConfig
src/manicule/connectors/plugin.py          lazy factory and built-in registration
src/manicule/config/settings.py            per-instance retention override
src/manicule/app/runtime.py                resolve effective sync retention policy
src/manicule/ingest/pipeline.py            run-local blob sink and acquisition choice
tests/connectors/test_git_site.py           Git object, identity, deletion and hostile cases
tests/connectors/test_site_routes.py        manifest and normalization table tests
tests/ingest/test_connector_retention.py    mixed policy, transition and concurrency tests
```

The network implementation later adds:

```text
src/manicule/connectors/web_crawler.py      inventory and connector lifecycle
src/manicule/connectors/http_policy.py      origin, redirect, robots and address checks
src/manicule/parsers/crawled_web.py         deterministic boilerplate extraction
tests/connectors/test_web_crawler.py
tests/connectors/test_http_policy.py
tests/parsers/test_crawled_web.py
```

Registration imports only Pydantic configuration. Git subprocess helpers, `httpx`, robots parsing,
HTML extraction and parser runtimes stay inside factories or implementation modules so plugin
discovery remains cheap.

## 5. Test gates

`git-site` is ready when tests prove:

- the shared connector contract, including watermark abandonment and reconciliation coverage;
- one commit is pinned across discovery, fetch and reconcile while `HEAD` moves;
- a blob id change fetches, and an unchanged blob with unchanged route metadata skips;
- route metadata alone changes the version token;
- explicit ids survive file moves and duplicate ids/routes refuse the run;
- paths, manifest entries and symlinks cannot escape the repository/content root;
- deleted pages are proposed only after complete inventory exhaustion;
- malformed/oversized manifests and blobs fail before partial authority is claimed;
- canonical provenance contains the public URL and only a relative snapshot path; and
- retention-off mode writes no blob and reports that reindex requires a new sync, while the normal
  retained mode still passes an offline rebuild test; and
- two concurrent connector runs with opposite retention policies cannot observe each other's blob
  sink or durable-acquisition choice;
- enabling retention fetches an unchanged document whose `original_ref` is absent, while disabling
  it never implicitly releases a previously retained current or historical blob; and
- registration loads neither Git process code nor an HTTP/HTML runtime.

The sitemap crawler additionally proves:

- robots, DNS, redirect and canonical checks on every hop;
- private/reserved address rejection for IPv4 and IPv6, including DNS rebinding fixtures;
- sitemap indexes, compression bombs, cycles, limits and incomplete-inventory behavior;
- strong ETag pinning and stale-body recovery;
- 404/410 versus timeout/429/5xx deletion semantics;
- URL canonicalization and duplicate collapse from a table of hostile cases;
- bounded concurrency, response bytes, retries and `Retry-After`; and
- round-trip citations against retained HTML after boilerplate extraction.

No live internet test is required for correctness. `httpx.MockTransport`, a local adversarial HTTP
fixture and temporary Git repositories exercise the contracts deterministically; one opt-in live
smoke test may verify behavior against a site the project controls.

## 6. Release slices

1. **Git inventory and routes** — `GitSiteConfig`, manifest schema, pinned tree, discovery,
   reconciliation and contract tests.
2. **Git fetch and provenance** — batch blob reads, version tokens, canonical source records,
   pipeline integration, no-retention behavior and the existing optional retained-copy path.
3. **Sitemap crawler** — URL/SSRF policy, robots, sitemap inventory, `HEAD` tokens and raw HTML
   fetch using the existing parser.
4. **Crawled HTML profile** — retained originals plus deterministic main-content extraction and
   round-trip evaluation against a small site corpus.
5. **Conditional fetch reuse** — optional protocol and acquisition reuse for real 304 handling.
6. **Durable link traversal** — fetched discovery, SQLite frontier, restart and incomplete-
   inventory proofs.

The useful stopping point for the content described here is slice 2. It indexes the owned site from
Git with stable public citations and no network dependency, while leaving the network crawler's
security and durability contracts explicit instead of burying them in a "local mode" that will be
hard to remove later.
