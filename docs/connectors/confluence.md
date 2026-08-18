# Confluence ingestion

Design for the Confluence connector. Ticket [#9](https://github.com/mgd43b/manicule/issues/9).

The structure of a Confluence page is most of what makes it retrievable, and it is thrown away
by the obvious implementation: fetch `body.storage`, strip the angle brackets, index the words.
Every table, code block and heading collapses into one run of prose, and nothing downstream can
tell that it happened. manicule fetches a typed document tree where one exists and parses the
markup with a real parser where one does not, so a table stays a table.

---

## 1. Auth

Two axes, not one. **Which Confluence this is** (`deployment`) decides how a body is read —
Atlassian Document Format on Cloud, storage format on Server and Data Center. **What a request
proves the account with** (`auth`) is a separate question, because a self-hosted instance behind
an identity provider is Server or Data Center in every respect that touches a body while having
no personal access token to offer.

| `deployment` | `auth` | Credential | Sent as |
|---|---|---|---|
| `cloud` | `api_token` (default) | `email` + API token | `Authorization: Basic base64(email:token)` |
| `server` | `personal_access_token` (default) | Personal access token | `Authorization: Bearer <token>` |
| `server` | `browser_session` | Cookies from a signed-in browser | `Cookie: JSESSIONID=…` |

`auth` may be left unset, and then it is the deployment's usual credential — an existing
configuration keeps working without naming it.

Config: base URL, credentials, and an optional space allowlist. Everything is fetched as the
credential's user — see §9.

**There is no OAuth 2.0 3LO.** An earlier version of this table listed a "Cloud (multi-user) ·
OAuth 2.0 3LO · Bearer" row that nothing implemented. 3LO is the multi-user arrangement — a
registered Atlassian app, a client secret, a redirect URI and a token exchange, so that each
person's own grant fetches what they may see — and manicule's index is deliberately not
permission-aware (§9). Building the credential without the index that would justify it would
buy nothing, so the row is gone rather than half-built. Per-user visibility is
[#13](https://github.com/mgd43b/manicule/issues/13).

### 1.1 Browser sessions, for an instance behind an identity provider

Self-hosted Confluence fronted by an identity provider commonly has personal access tokens
disabled by policy, and then the only credential its users can obtain is the session their
browser already holds.

**Sign in with `--browser`.** It opens a browser at your Confluence, you sign in the way you
sign in to everything else — SSO, MFA, a passkey, whatever your provider asks — and it works:

```console
$ pip install 'manicule[browser-auth]'
$ playwright install chromium

$ manicule connector login wiki --browser
$ manicule connector sync wiki
```

That is the whole of it. There is nothing to configure, no cookie to find and no file to
produce.

**manicule never asks for your password and has nowhere to put one.** No password, no one-time
code and no device approval passes through manicule, and that is a property of the code rather
than a promise about it: there is no parameter that could carry one and no branch that would
accept one.

**What the browser path does and does not guarantee.** manicule opens the browser, so it *could*
in principle read the page you type into. It does not, and that is enforced rather than intended:
the browser is watched only through its cookie jar and whether the window is still open, nothing
is typed, clicked or filled for you, and `tests/connectors/test_browser_login.py` fails if an
accessor that reads page content appears in `connectors/browser.py`. Earlier releases refused to
drive a browser at all so that the guarantee could be about capability — manicule *cannot* see it
— rather than about behavior. That is a real difference and it is stated here rather than
smoothed over; what it is not is a reason to prefer the harder path.

**The two fallbacks, for when `--browser` cannot be used:**

```console
$ manicule connector login wiki --browser-state ./storage-state.json
$ manicule connector login wiki                  # paste the Cookie header
```

`--browser-state` imports a Playwright state file, for a machine that cannot open a window — a
remote shell, a container — but can receive a file from one that can ([§1.1a](#11a-importing-a-browser-state-file)).

The paste path needs nothing installed and uses your own browser, which makes it the answer when
a conditional-access policy declines a driven Chromium — a real case, and the reason it is kept
working rather than removed. It asks you to open developer tools and copy a live session cookie,
which is why it is not the one to reach for first.

Without the extra, `--browser` fails immediately naming both installation commands. It does
**not** quietly fall back to the paste prompt: someone who asked for a browser and got asked for
a cookie header would reasonably conclude the feature is broken.

**Only your Confluence's cookies are taken.** Signing in through an identity provider means
visiting it, and it sets cookies of its own — frequently an account you use at several
companies. Those are filtered out before anything is stored, by domain, path, `secure` and
expiry, using the browser's own rules: a host-only cookie must match the host exactly, a domain
cookie matches on a label boundary (`.example.test` covers `wiki.example.test` and not
`notexample.test`), and a path matches on a segment boundary, which is what makes an instance
under a context path such as `/confluence` work without also importing a neighboring
application's cookies.

**A driven Chromium may be refused by conditional access.** It is a new device to a policy that
tracks them, and some tenants will decline it outright. That is a property of the tenant rather
than a bug here; the paste path is unaffected because the browser is yours.

manicule tries to tell that case apart from "sign-in is just slow", by whether the browser ever
received a cookie from your Confluence at all — and it is a heuristic rather than a diagnosis.
An unauthenticated Confluence usually issues a session cookie on the first visit, so a browser
that reached the instance and was then turned away still gets the *wait longer* message. The
*this will not work* message fires when something in front of Confluence intercepts the request
before Confluence answers at all. The heuristic errs toward telling you to wait, which costs a
timeout you were spending anyway rather than talking you out of a setup that works.

**The session lives in the running server's memory, and nowhere else. No server, no sync.**

That sentence is the whole of the credential design and it has a consequence worth stating
before the reasoning: `manicule connector sync` for a Confluence source **requires a running
`manicule serve`**, and re-authenticating after the server restarts is the expected path rather
than a failure. The refusal you get when there is no session says so and names the command.

There were three places a session could live and now there is one. Each of the two that went had
a defect that could not be fixed where it was:

- **The macOS Keychain prompted, repeatedly.** Every write recreated its item and therefore
  discarded the authorization you had granted, so running syncs meant retyping your login
  password for a program that is not allowed to know it. The machinery that made this worse —
  `security` silently truncating a stdin secret at 128 bytes, so a session record had to be
  written in 120-byte pieces across numbered items and read back and compared — is gone with it.
- **There was no store at all on Linux or in a container**, so `--browser` there captured a
  session it had nowhere to put. `$CONFLUENCE_SESSION_COOKIE` stood in for one, which is a live
  corporate credential written into a shell's history and inherited by every process started
  from it. `session_env` is deleted; the configuration model forbids unknown keys, so a
  configuration still setting it is refused loudly rather than quietly ignored.

A session held in a long-lived process's memory has none of those problems. It is never written
anywhere, it prompts for nothing, it behaves identically on every platform, and it is gone when
the process stops. **That last one is the point rather than the price**: `--browser` has made
re-authenticating a few seconds of clicking, so a credential whose lifetime is the server's is
one you re-take deliberately instead of one that persists indefinitely.

Nothing lands under `<data_dir>`, so `docs/storage.md` §7.1 and the `doctor` permissions check
still do not come into it. A `session_cookie` written into `config.toml` is still a startup error
rather than a working setting.

**The hand-off, and why the browser opens where you are.** `connector login` runs in *your*
process, because `--browser` opens a window and a window opened by a background server is one
nobody is sitting at. The syncs run in the server. What joins them is the control socket
(`docs/deployment.md` §6.1): the command verifies the session against your instance exactly as it
always did, then hands it over a `0600` Unix domain socket to the server, which holds it. Your
command-line process keeps no copy.

If no server is running, `connector login` says so **before** it opens a browser — being asked
to complete a second factor and only then told there was nowhere to put the result would be
asking you to do the work twice.

**Capture proves the session before storing it.** A cookie copied short, copied from the wrong
tab, or copied from a session that had already timed out is indistinguishable from a working one
until something uses it — and otherwise the first thing to use it would be the first page of the
next sync. So `connector login` makes one request as the session, reads back who the instance
says that is, and stores nothing at all if the answer is anybody other than a signed-in user.

**A dead session stops the run; it does not pause it.** Renewal is an out-of-band act — a person
goes to a browser — so there is no interval a sync could usefully wait, and the cursor it was
holding would expire first (§2). Stopping leaves the watermark unadvanced, so a re-run after a
fresh sign-in resumes rather than starting over. `session_max_age_hours` (default 12) is
manicule's own ceiling on how long it will keep using one: a cookie carries no expiry a client
can read, and an age manicule measures itself is the only thing that can turn "too old to try"
into a **startup** refusal rather than something the first page of a sync discovers.

### 1.1a Importing a browser state file

`--browser-state` reads a Playwright `storage_state` JSON document — the thing
`context.storage_state(path=...)` writes — and takes the cookies in it that apply to your
configured instance. It is the path for a machine that cannot open a window (a remote shell, a
container) but can receive a file from one that can.

```console
$ manicule connector login wiki --browser-state ./storage-state.json
```

Four rules, each of which refuses rather than guesses:

- **The file is never written to.** It is yours, it may belong to other tooling, and a login
  that rewrote it would be a surprise in somebody else's workflow.
- **It must not be readable by other users.** It holds live session cookies. A group- or
  world-readable file is refused with the `chmod` to run; `--allow-insecure-state` imports it
  anyway, which is a decision made out loud rather than a silent skip. The check does not apply
  on Windows, where the POSIX bits a stat reports are not the access control the platform
  enforces.
- **It is parsed defensively and never quoted back.** One malformed entry does not cost a good
  file; a file with no usable entry is refused. No part of the document reaches an error
  message, because the whole document is secret.
- **`localStorage` is ignored.** Confluence authenticates with cookies; page state is not
  manicule's business.

A state file from a different site is the usual cause of "no cookies for
`https://confluence.example.test/confluence`" — the document records whatever was signed in when
it was written.

### 1.1b Signing out, and signing back in

```console
$ manicule connector login wiki --forget      # remove the stored session
$ manicule connector login wiki --browser     # take a new one
```

`--forget` asks the running server to drop the session it holds for that instance. It does not
sign you out of Confluence itself, which is your browser's business and your identity
provider's. Stopping the server has the same effect on every session it holds, which is the
blunter version of the same thing.

**A failed login never costs you a working session.** Verification happens before the store is
touched, so a timeout, a closed window, a dead cookie or a state file for the wrong site leaves
whatever was stored exactly as it was. There is no delete-then-write window because the write is
the last thing that happens and it happens only on success.

### 1.1c When it does not work

| Symptom | What it means | What to do |
|---|---|---|
| `browser sign-in needs Playwright` | the extra is not installed | run both commands it names — the package and `playwright install chromium` are two steps |
| `the browser would not start` | the package is installed, the browser is not | `playwright install chromium` |
| `sign-in did not finish ... though the browser did reach <url>` | sign-in was under way and ran out of time | `--timeout 600`, or set `browser_timeout_seconds`. A device-approval step is the usual reason |
| `the browser never received a cookie from <url>` | sign-in never reached Confluence at all | **a longer timeout will not help.** Either conditional access declined the browser — use the paste path, which uses your own — or `base_url` names a different host from the one sign-in lands on |
| `the browser was closed before sign-in finished` | the window was shut | re-run and leave it open until Confluence itself has loaded |
| `the browser finished with no cookies for <url>` | the same as above, seen one layer up | check `base_url` names the site root **including any context path** |
| a sign-in page was stored as content | it cannot be — §1.3 | nothing; the refusal is the design |
| `... was captured N hours ago` | the session aged past `session_max_age_hours` | sign in again; nothing is lost, the watermark did not advance |
| the tenant refuses the browser | conditional access does not recognize a driven Chromium | use the paste path, which uses your own browser |

### 1.2 The credential is checked before the connector is constructed

A missing token, a Cloud token with no email beside it, and a browser session that is absent or
older than `session_max_age_hours` are all startup refusals naming what to set or what to do.
Discovering any of them at the first page of the first sync produces a run that reports progress
and indexes nothing.

### 1.3 A sign-in page is never content

This is the failure mode the whole arrangement is built around, and it is not "the credential
was rejected". A rejected credential announces itself with a 401 and stops the run. An instance
behind an identity provider answers a request it will not serve by **redirecting to that
provider**, and a client that follows the redirect gets a sign-in page with status 200: a
successful response, of a plausible content type, carrying several kilobytes of real text.
Indexing it would put one copy of that page in place of every page the sync tried to read —
plausible, retrievable, citable documents that nothing downstream can tell from the corpus, and
a run that looks from every metric like one that worked.

Two independent layers refuse it, and they run for **every** credential, because a reverse proxy
answers a personal access token exactly as it answers a dead session.

- **Redirects are followed by this connector rather than by the HTTP client.** A redirect off
  the configured origin is refused as an untrusted link — nothing is requested from the identity
  provider and no sign-in page is fetched at all — and a redirect to this instance's own
  `/login.action` or SSO servlet is refused as a dead session. An ordinary same-origin redirect
  is followed, up to five hops.
- **Every response is read for the marks of a sign-in.** Confluence's own authentication filter
  reports the outcome in `X-Seraph-LoginReason`; every REST response names the authenticated
  user in `X-AUSERNAME`, and `anonymous` — or somebody other than the account the session was
  captured as — is a session that has stopped being this one whatever the status says; and
  failing both, an HTML body carrying a sign-in form's own field names is refused.

The second layer matters most on **attachment downloads**, which is where a sign-in page has no
JSON decoder to fall foul of: it arrives as perfectly good `text/html` that the parser chain
would parse, chunk, embed and serve.

## 2. Discovery and change detection

**Full sync** — enumerate once, per space. The CQL differs by deployment, and the blocks below
are **executed** by `tests/connectors/test_cql_contract.py`, which builds each query through the
real builders and compares it to what is written here. There is no second copy of these strings
to drift against this one.

<!-- cql:cloud:full -->
```cql
type in (page, attachment) AND space = "ENG" AND status = current order by lastmodified asc
```

<!-- cql:server:full -->
```cql
type in (page, attachment) AND space = "ENG" order by lastmodified asc
```

Sent to `GET /wiki/rest/api/content/search` with
`&expand=version,ancestors,space,container&limit=100`.

**Data Center can use the direct current-content inventory for complete membership.** The
compatibility default remains:

```toml
[connectors.handbook.options]
full_inventory_authority = "search"
```

For a Server or Data Center connector whose scope is one or more whole spaces, setting
`full_inventory_authority = "direct_current_content"` makes full discovery and reconciliation
enumerate `page` and `attachment` separately through `GET /rest/api/content`, explicitly pinned
to the canonical space and `status=current`. Incremental discovery remains the CQL query below.
Cloud and `root_page_ids` scopes remain entirely CQL-backed; the option does not alter their
cursor identity or cause a replacement walk.

This is an explicit source-authority choice, not a fallback after search looks suspicious. Every
direct member must carry its source id, exact type, current status, canonical space, positive
revision, offset-aware modification time, and required page or attachment metadata. Missing or
mismatched evidence fails the aggregate enumeration; it is never converted to an empty result or
filled from the request. Under strict policy that run cannot promote. `allow_omissions` may still
represent typed body-fetch omissions, but it cannot promote an inventory known to be incomplete.

Data Center native `next` links commonly contain only `start` and `limit`. The connector follows
that native coordinate while re-pinning space, type, status, expansion, and configured page size
on every request. An explicit conflict, extra narrowing parameter, malformed coordinate,
cross-origin link, or loop fails closed. A response page is committed before its next link is
requested, and only a direct walk's true end authorizes reconciliation.

**Incremental** — a per-space watermark of the last successful sync, which adds one clause to
whichever of those two the deployment uses:

<!-- cql:cloud:incremental -->
```cql
type in (page, attachment) AND space = "ENG" AND status = current AND lastmodified >= "2026/08/09 14:25" order by lastmodified asc
```

<!-- cql:server:incremental -->
```cql
type in (page, attachment) AND space = "ENG" AND lastmodified >= "2026/08/09 14:25" order by lastmodified asc
```

`lastmodified` is a first-class CQL field and is sortable, so the source does the filtering
and a sync costs what changed rather than the whole corpus.

Five things about that query are load-bearing.

- **One query covers pages and attachments.** `type in (page, attachment)` makes attachments
  watermark-aware and reconcilable on the same terms as pages, instead of a per-page call to
  the attachment endpoint that no watermark can narrow.
- **`status = current` is written on Cloud and omitted on Server and Data Center**, and that
  is a difference in the products rather than a preference. Cloud's search accepts the field,
  and reconciliation (§3) depends on it: a query that included trashed content would report
  every deleted page as still present, and deletion detection would run, succeed, and find
  nothing, forever. The standard Data Center content-search resource **rejects** `status`
  outright — an HTTP 400 naming the field — and returns current content by default.

  **The decision is read from the declared `deployment` and from nothing else.** Not the URL
  shape, not the hostname, not the context path, not the credential kind, and above all not
  from sending the query to find out: a retry that strips the clause after a 400 would double
  every enumeration's cost on one deployment and mask the day the other changed its mind.
  `ConfluenceConfig.current_only` is the one property that answers it, and all eight query
  sites read it — whole-space discovery, incremental discovery, page-tree discovery,
  attachment discovery, reconciliation, subtree membership, attachment reconciliation, and
  the title lookup an include macro resolves through.

  The builders take `current_only` as a **required keyword argument with no default**. A
  default would be one deployment's answer silently applied to the other, which is precisely
  how a builder acquires seven callers and six correct ones.
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

**The space list is checked each run, and the two cases ask the source two different questions.**

- **No allowlist** — every visible space is enumerated through the paginated
  `/rest/api/space`, so a space created since the last run needs no configuration change. An
  account that can see none at all is a refusal.
- **An allowlist** — each configured key is confirmed with one direct
  `GET /rest/api/space/{key}`, and the catalog is never listed. An allowlist is a scope
  boundary, and reading the whole catalog to answer a question about two keys is that boundary
  being ignored for convenience.

Either way an unknown key is a refusal before any content query goes out: CQL answers a query
for a space that does not exist with an empty result set, so a typo would otherwise be a sync
that runs, succeeds, indexes nothing, and leaves reconciliation proposing the deletion of
everything that space ever contributed.

**Three details of the direct lookup are load-bearing.**

- **The key is one URL-encoded path segment.** A key containing `/` or `?` interpolated raw
  would address a different resource, and the connector would report on whatever answered.
- **The spelling that comes back is the one used.** Confluence space keys are case-insensitive
  to look up and have one canonical casing, and that casing goes into every subsequent CQL
  literal — so the response's key is read rather than the configured string echoed.
- **The refusal does not list what *is* visible.** Enumerating unrelated spaces to improve an
  error message is the request this path exists to stop making. A credential or permission
  failure is also left to surface as itself rather than being folded into "unknown space":
  "your token expired" and "that key is wrong" need different repairs.

The cost difference is the point. On an account entitled to 500 spaces and configured for two,
the catalog walk was six requests carrying 500 space records; the direct lookups are two
requests carrying two. It scales with the configuration instead of with the account.

**Three traps in pagination, all verified:**

- **Pagination is cursor-based, not offset.** Follow `_links.next`; the `start` parameter
  no longer works reliably for search.
- **Cursors contain `+`, which must be escaped as `%2B`** before being sent back. Naive URL
  handling silently breaks pagination partway through a sync — an unrecognized cursor comes
  back as results rather than as an error, so some pages are simply never seen.

  **The exception is exactly one parameter wide.** The rest of the link is form-encoded, so a
  `cql` whose spaces arrived as `+` must decode back to spaces; treating *those* as literal
  sends the source a query nobody wrote, which is the same bug mirrored. So `+` is data in
  `cursor` and a space everywhere else, and the round trip is: decode without form rules for
  the cursor, and let the HTTP client percent-encode it back to `%2B` on the way out.
- **Cursors expire.** A consumer that stalls mid-enumeration — source throttling, journal
  admission, or a paused process — resumes onto a cursor the server has forgotten, and a
  forgotten cursor can be answered with a fresh first page rather than an error, which
  enumerates the start of a space
  twice and its end never. A cursor held longer than `cursor_lifetime_seconds` (default 300)
  is refused **before the request is sent**, so the run fails legibly and is re-run against an
  unadvanced watermark. A `next` link addressing a cursor already followed is refused for the
  same reason: a loop over a paginated search reads as a very large space. Followed-request
  fingerprints live in a temporary disk-backed exact-membership ledger with a capped SQLite page
  cache, so exact long-cycle detection does not grow process memory with the number of pages. A
  digest collision refuses the request and therefore fails closed. The cursor-age check runs
  after that disk write, immediately before the next request, so a stalled ledger cannot create
  an unchecked expiry window of its own.

  **The durable pipeline's side is a journal boundary, not downstream backpressure.** When that
  path is wired, each bounded source response commits atomically before the connector follows its
  next cursor, and local
  fetch/parse/embed work starts from the journal after enumeration. A slow embedder therefore
  cannot hold a live cursor at all (`ingest.md` §8.3.1); source and journal delays still can,
  which is why the typed expiry guard remains. The explicitly supported nonjournal fallback
  keeps its bounded direct hand-off during the staged rollout.

**`_links.next` is resolved against `_links.base` by concatenation, and the origin is
checked.** An instance served from a context path (`/confluence`) has that path in `base` and
not in `next`, and RFC 3986 resolution of a root-absolute reference discards it.

The origin check is not only pagination's. **Every** URL the client is given is checked before
the request goes out, because most of them come from responses — the next page's link, an
attachment's `_links.download` — and every request carries the sync account's credential.
Without a check at the point requests are made, a response decides who receives that
credential.

Per page, `version.number` is the change token — cheaper than hashing content.

**The watermark advances only on a complete enumeration.** It is a per-space map carried in
`Watermark.metadata`, and a space's entry moves only after that space's walk finishes; a
consumer that abandoned discovery part-way is offered no watermark at all. A watermark built
from a prefix skips whatever the rest of the walk would have returned, and nothing looks for
it again. The same metadata records the **scope** those positions were reached in — §2.1.

## 2.1 Scoping to a page tree instead of a whole space

A space is usually far more than anybody wants indexed. `root_page_ids` names one or more
pages, and the source becomes those pages and everything currently beneath them:

```toml
[connectors.architecture]
type = "confluence"

[connectors.architecture.options]
base_url = "https://confluence.example.test/confluence"
deployment = "server"
auth = "browser_session"
spaces = ["ENG"]
root_page_ids = ["100100"]
include_root_pages = true
include_attachments = false
```

That indexes page `100100` and its current descendants, and nothing else in `ENG`. Leaving
`root_page_ids` out is the whole-space behavior every existing configuration has, unchanged.

**The page id is the `pageId` in a page's URL** — a decimal number. A title will not do: two
pages in one space can share one, and the id is what survives a rename. A non-numeric value is
refused when the configuration is read, which is also what makes the id safe to put in a query
unquoted, and therefore makes CQL injection through this setting structurally impossible rather
than escaped.

### `spaces` and `root_page_ids` narrow one scope between them

They are not two lists that add up. `spaces` says which spaces this source may read at all;
`root_page_ids` says which trees inside them it actually reads. The effective scope is the
intersection, and the combinations that cannot be honored that way are **refused at startup**
rather than resolved by a rule nobody would guess:

| Configuration | Result |
|---|---|
| `spaces` only | Whole spaces. Unchanged. |
| `root_page_ids` only | Those trees. Each root's space is read from the root itself. |
| Both, every root inside an allowed space, every allowed space holding a root | Those trees. |
| A root in a space `spaces` does not list | **Refused.** Honoring it would mean `spaces` had silently stopped being an allowlist. |
| A listed space holding no configured root | **Refused.** It would enumerate nothing while appearing to sync, and reconciliation would then propose deleting everything it ever contributed. |
| A root in a space this account cannot see (no allowlist) | **Refused**, for the same reason a mistyped space key is. |

### `include_root_pages` defaults to true

`root_page_ids = ["100100"]` names page `100100`, and a corpus containing everything under it
*except it* is not what anybody writes that down to mean. The default is chosen for the
direction the mistake fails in: `false` leaves exactly one page missing — the one that was
named — and nothing about the run says so, while `true` costs one extra page to somebody who
wanted only the children and they notice immediately. Setting it with no `root_page_ids` is
refused rather than ignored.

It is written into the query rather than applied to the results, so what a run is scoped to is
what it asked for.

### How descendants are resolved, and what bounds it

Confluence's own `ancestor` predicate, which matches at any depth. Executed by the same test as
§2's blocks:

<!-- cql:cloud:subtree -->
```cql
type = page AND space = "ENG" AND status = current AND (ancestor = 100100 OR id = 100100) order by lastmodified asc
```

<!-- cql:server:subtree -->
```cql
type = page AND space = "ENG" AND (ancestor = 100100 OR id = 100100) order by lastmodified asc
```

Sent with `&expand=version,ancestors,space,container`.

**There is no client-side walk**, and therefore no queue of page ids, no cycle detection, no
depth ceiling, and no second enumeration a moved page can fall between. A tree forty levels
deep is one query. A cycle — which Confluence does not permit, but which a client walking
`child/page` would still have to survive — cannot arise, because nothing follows a parent link.

**What comes back is checked against what was asked for.** Every page carries its own ancestor
ids in the same response, so membership is re-derived from the page rather than inferred from
the source having returned it. A page that is not in the configured trees **stops the run**. The
check exists for one failure: a deployment that accepted `ancestor` and did not apply it would
answer with the whole space, and a connector that trusted the query would index all of it while
reporting a subtree. Filtering the difference away instead would produce the right documents
while paying for, and claiming, the whole space.

**An empty answer is not an empty subtree.** Reconciliation deletes what it does not see, so a
descendant enumeration that comes back empty is cross-checked: each root is asked, through
`child/page`, whether it has a child. A root that does stops the run. Without that, a
deployment which accepts the predicate and matches nothing reports the whole subtree as
deleted — a successful query, no rows, no error anywhere.

### Attachments follow their page, and that is where a subtree still pays per space

`ancestor` is relied on for pages and **not** for attachments: Confluence exposes an
attachment's container page, and the container is the authoritative answer in any case. So an
attachment is in scope exactly when the page holding it is, which has one consequence worth
stating plainly rather than discovering:

- The **page** query is narrowed at the source.
- The **attachment** query is not. It stays space-wide, and its results are matched against the
  subtree's page ids client-side.

With `include_attachments = false` — the common shape for a first scoped run — no attachment
query is sent at all and the run costs the page tree only. With it on, a scoped run also
resolves the whole subtree's page ids once (ids and ancestor ids, bounded by the subtree rather
than by the space) so that an attachment added to a page that has *not* changed since the
watermark can still be placed. That is an ids-only enumeration of the subtree per run, and it is
what makes the incremental attachment case correct rather than approximately correct. The
membership index is an exact temporary SQLite table with a fixed page cache, deleted when the
enumeration closes; only the current response page is retained in process memory.

### Changing the roots

A watermark is a position **within a scope**, and the two are meaningless apart.

The effective full-inventory authority is part of that position too. Historical and default
search-backed scopes retain their exact existing fingerprint and watermark representation.
Choosing direct current-content authority for a Data Center whole-space scope creates a distinct
cursor identity, so an old CQL watermark can never turn the first direct walk into an incremental
query. The stable corpus scope remains separate: after the replacement inventory reaches each
member, retained bytes may be adopted from the fenced search-backed predecessor only when the
connector, source identity, revision, URI, media type, byte length, hash, blob, and acquired-source
envelope all agree. This avoids redownloading an unchanged corpus without treating the old
watermark as compatible.
`Watermark.metadata` therefore records the scope its positions were reached in, and when the configured
roots or `include_root_pages` change, every stored position is discarded and the run enumerates
the new scope in full. Anything less loses documents: every page in a newly configured tree that
has not changed since the stored instant is already behind it, so an incremental query would
never return it, and nothing would ever return it again.

The change is reported at `WARNING` on `manicule.connectors.confluence`, naming both scopes.
Documents indexed under a root that is no longer configured are **not** touched by that run —
the next reconciliation pass proposes removing them, and the deletion ceiling (§3) still applies
to that proposal.

A watermark stored before this connector had scopes carries none, and is read as whole-space,
which is the only scope it can have been recorded in.

### `connector sync --limit` is not a subtree

`--limit` bounds how much work a run does. It takes an arbitrary prefix of discovery, which for
a space ordered by `lastmodified` is whichever pages happened to be edited longest ago — not a
coherent part of anything, and not stable between runs.

**No watermark is recorded from a limited run, and the reason is worth being exact about,**
because it is not the one the shape of the code suggests. The pipeline stops consuming discovery
and then reports the run as **clean** — there is no error, so `RunReport.clean` is true and the
watermark write is attempted. What prevents it is the connector: its enumeration was abandoned,
so it offers no position at all, and nothing is stored. Delete that guard and a limited run
silently records a position past every page it never received.

Use `--limit` to try a connector out. Use `root_page_ids` to say what a source *is*.

## 2.2 What the page says about itself

Every successfully fetched page and attachment carries a validated
`Provenance(source=SourceMetadata(...))` under `source_provenance`
([`storage.md`](../storage.md) §4.2.1). That is what lets a citation record the page id, the
version actually retrieved, the source's own modification time and the hierarchy as one
claim-level reference, instead of a title and a URI reassembled from ordinary document fields.

**Every field is read out of a source response. None is inferred, and the table says which
response.** A record assembled partly from what manicule worked out would read, at every
surface, as the publisher's own account of the document — which is the one failure worth
designing against here, because nothing downstream can tell the two apart.

| Field | Cloud | Server / Data Center |
|---|---|---|
| `title` | `GET /api/v2/pages/{id}` → `title` | `GET /rest/api/content/{id}` → `title` |
| `canonical_uri` | `_links.webui` resolved against `_links.base` | same |
| `source_id` | the content id | same |
| `version` | `version.number` **of the body that was retained** | same |
| `modified_at` | `version.createdAt` | `version.when` |
| `created_at` | the page's own top-level `createdAt` | **absent** — it is under `history`, which this connector does not expand |
| `content_type` | the media type the body was routed as | same |
| `section_path` | space key + ancestor titles, own title excluded | same |

Two of those rows are the ones a shared parser would get wrong. `modified_at` is spelled
differently by the two deployments, and reading only one spelling produces a record with **no**
modification time on the other — a hole rather than an error, on whichever deployment nobody
happened to try. And Cloud carries `createdAt` at *two* levels: the version's, which is when the
page was last edited, and the page's, which is when it first existed. Collapsing them makes an
old page look freshly revised.

**A timestamp without a UTC offset is discarded rather than read as UTC.** A naive timestamp is
not a moment; read as UTC it is wrong by the instance's offset, and the symptom is a citation
that is quietly wrong by hours in the field used to decide which of two versions is newer.

**Absent stays absent.** No field is ever filled from this run's clock, from the file's
modification time or from `indexed_at`. The three timestamps are three separate questions —
`modified_at` is the source's, `retrieved_at` is a local snapshot's, `indexed_at` is this
installation's — and a network fetch has no local snapshot at all, so this connector writes only
the first.

**A version disagreement never certifies older bytes.** When the stale-body defense (§4) cannot
retrieve a body at least as new as discovery reported, the fetch fails and produces no provenance
record. A page that changes during the sync can return a *newer* body; in that case the record
states the retained body's version and timestamp, and the `version_disagreement` diagnostic
survives beside it.

**An attachment is cited as itself.** It gets its own content id, its own version, its own
address and its own media type. The page it hangs off survives as a *relationship* — in
`parent_page_id` and in the hierarchy — because a reference to the page would send a reader to a
page that has a diagram somewhere on it rather than to the diagram. Its modification time comes
from the search result that discovered it, which is the only response that ever describes an
attachment: the download is bytes.

Its `content_type` is the narrower of two answers, and the difference is the rule above holding
under pressure. §6 reads an attachment's media type as Confluence's `metadata.mediaType`, then
the download's `Content-Type`, then the **filename extension** — and the third is manicule's
inference, sound enough to route bytes by and not something the publisher said. So the document
is still routed by the guess and the record still says nothing: `content_type` is empty when both
source signals were silent.

**Scoping metadata is not provenance.** `root_page_ids` and `ancestor_ids` (§2.1) explain why
manicule selected a document. They stay in the connector's own keys, because inside a source
record they would read as something Confluence said about the page.

**A record the citation interface refuses is stored as a stated reason, not as silence.** A page
whose title carries a control character is still fetched, still indexed and still searchable;
what it loses is the canonical record, and `unavailable_reason` says so. Absent and refused are
different facts, and only the second is something an operator can act on.

### Documents indexed before any of this existed

Adding provenance to newly fetched pages does not reach a page nobody has edited since. Those
documents are detected and counted by `manicule doctor`, under the `wiki-provenance` check.

**Six of the seven fields could be recovered locally. The seventh cannot, and it is the one that
matters.** A document ingested before this change still holds its page id, its canonical URI, its
fetched version, its media type and its hierarchy. It has never held `modified_at` — that was
stored nowhere — and inventing it from a filesystem or database timestamp is exactly what the
rest of this section forbids. A partial record written with the timestamp left null would be
indistinguishable, at every surface, from a page whose source genuinely reported no modification
time. That is worse than the gap, because it is a gap presenting as an answer.

**So these pages have to be fetched again, and a routine incremental sync will not do it.** An
unchanged page is unreachable twice over: the watermark means discovery never enumerates it, and
the version-token comparison (§2) means it would skip without a fetch even if it were. `manicule
reindex` does not help either — it re-parses retained bytes and touches no network.

**The migration is therefore a full resync of the source, performed deliberately.** `doctor`
reports the count and the affected sources and changes nothing; `doctor --fix` does not arm it
either, which is a limit rather than an omission — that flag seeds grammars and vocabularies,
which are local, cheap and idempotent, and re-crawling somebody else's wiki is none of those.
Hiding a recrawl behind a health command is the same defect as hiding one behind an incremental
sync.

**No automated lever is provided, and that is a decision rather than an oversight.** Clearing the
stored version tokens and invalidating the source's watermark would make the next sync do exactly
the right work, and building it means new methods on the ingest store and the application port.
It is not built because the population it would serve is currently **empty**: this connector has
never been run against a live instance (§10), so there are no pre-change documents anywhere to
migrate. Machinery for a corpus that does not exist would be shaped by a guess about what its
migration needs. The check is what tells somebody when that stops being true.

Re-fetching preserves identity: a page is keyed on its content id, so enrichment updates the
existing document rather than creating a second one. It is safe to interrupt and safe to repeat —
whatever was not reached still carries no record, and the next `doctor` counts exactly those.

## 3. Deletions — the part most implementations miss

CQL returns what exists. A page deleted since the last sync simply stops appearing, so a
watermark sync **never learns it is gone** and the index keeps serving it forever.

**One mechanism, and this section used to claim two.** It listed a "trash check" beside the
reconciliation pass, under the heading "both needed", and then said in its own second clause
that the trash check does not cover this. Both halves cannot be true, and the code has only ever
had one: there is no CQL query that reports what *stopped* existing, because CQL returns what
exists. The reconciliation diff is the mechanism.

- **Reconciliation pass** — periodically (weekly) list every content ID in scope, diff against
  indexed IDs, and soft-delete the difference. Cheap: IDs only, no bodies. The request sends
  **no `expand` parameter at all**; an earlier version of this line said `expand=` empty, which
  is a parameter that is not sent and would not be worth sending.

**Attachments are reconciled with pages**, because they are enumerated with pages (§2). A
document the pass omits is a document the pipeline deletes, so "attachments are handled
elsewhere" would be a slow way of emptying them out of the index.

**Reconciliation enumerates exactly the scope discovery does**, and in a subtree-scoped source
that is a single object answering both (§2.1) rather than the same rule implemented twice.
Scoped on one side only is the failure that has no small version: discovery narrow and
reconciliation wide leaves the index never shrinking, and the reverse soft-deletes every page
outside the tree. A scoped pass expands `ancestors`, which a whole-space pass has no use for,
and that is the one place the two differ in cost.

**A failure mid-enumeration raises rather than returning what it has.** The ids seen so far
are a prefix, not the truth, and diffing a prefix against the stored set marks everything not
yet reached as deleted: one transient error, and the corpus is soft-deleted
([`ingest.md`](../ingest.md) §11.1 carries the pipeline's half of this — clean completion
only, a deletion ceiling, and soft delete only). The connector's half is to fail loudly, so
that "everything is gone" and "nothing answered" never look alike.

Reconciliation uses the same response-page hand-off as discovery. The durable inventory commits
each nonempty ids-only source page before another cursor is requested; an empty page after scope
filtering has nothing to write and is acknowledged at the same boundary without inventing a
transaction. It does not infer a boundary from a local item count, so a configured 250-result
page cannot be split by the pipeline's unrelated inventory write size. Subtree reconciliation
builds its bounded membership index as it yields those native search pages; it never drains the
tree and re-slices the completed result locally.

## 4. Fetching content

**Cloud — ADF.** A typed JSON document tree, not markup:

```
GET /wiki/api/v2/pages/{id}?body-format=atlas_doc_format
```

**Server / DC — storage format.** ADF is Cloud-only, so Server falls back to
`body.storage` XHTML. It is routed as `application/xhtml+xml;profile=confluence-storage` and
read by `manicule.parsers.confluence` (`docs/parsing.md` §2.4), which understands `ac:*` and
`ri:*` as Confluence rather than as unknown HTML elements: a code macro's declared language, a
panel's severity, a task's state, a Graphviz macro's engine and its DOT source, page and
external links, user mentions as display references, and a named placeholder wherever a macro
has no reader here.

> **This section twice argued that storage format needed no parser of its own, and both
> arguments were wrong — the second in the direction nobody checked.**
>
> The first ended "a second parser for a dialect of the same thing would be two implementations
> of one job". Storage format wraps the body of every `code`, `noformat` and `graphviz` macro in
> `<![CDATA[…]]>`, which HTML does not have outside foreign content — so the HTML parser reparsed
> each one as a bogus comment and **deleted the body**. Every code block on every Server or DC
> page was missing from the index, with a fragment of it indexed as prose, for as long as this
> connector existed. `manicule.parsers.web` now recovers CDATA sections as escaped text.
>
> The second survived that fix. It conceded `ac:*` was unread but called it "a missing feature
> rather than lost content". It was not merely missing. Read as generic HTML, `<ac:parameter>` is
> an unknown element wrapping a text node, so a macro's **configuration was indexed as document
> text**: a code block's language, a diagram's engine and a Jira macro's **JQL query** each
> became a prose block, went into the vector, and were quotable in a citation as words the page
> had said. A task's status arrived as a one-word block reading `complete`. The index was
> asserting things the document did not contain, which is the opposite of a missing feature.
>
> Neither argument was ever a measurement. Both were checkable in four lines.

Ancestors — the page hierarchy used for breadcrumbs in §7 — come from three places depending on
the deployment, and this paragraph used to name only the first of them.

- **Discovery**, which expands `ancestors` for every page it returns. The ordinary path on both
  deployments, and the reason a fetch needs no second call.
- **The fetch's own expansion, on Server and Data Center.** `body.storage` is requested with
  `expand=body.storage,version,ancestors,space`, so a storage-format fetch has a complete
  breadcrumb including the space key whatever the ref carried — and prefers it. There is no
  second call there either, and there never was.
- **The Cloud ancestors endpoint**, asked only for a ref built somewhere else (a re-fetch, a
  targeted single-page sync) on a deployment whose body endpoint carries no ancestors. An
  ancestor whose title it omits is skipped rather than filled in with an id.

**A breadcrumb starts at the space key, and on that last path there may be no way to learn it.**
The Cloud body endpoint reports a numeric space id rather than the key, so a ref carrying
neither ancestors nor a space key produces a breadcrumb one level short: `Platform > Auth
Service` where the rest of the corpus reads `ENG > Platform > Auth Service`, which retrieves
worse against exactly the queries a space key disambiguates. Inventing the key is not available
and is not wanted. Saying so is, and `breadcrumb_complete` is false whenever the breadcrumb is
short — including for this, which it did not use to cover.

Every page also keeps its ancestors as **ids** alongside the titles. Titles are what a reader
sees; ids are what says which page an ancestor is, and they are what survives a rename.

**A page that comes back with no ADF body at all** — the format declined for a page that
exists, which is not the same as an empty page — is read as storage format instead. Only that
failure falls back: a 429 or a rejected credential answered by trying a second endpoint would
double the load on a source that has just said stop, and report the wrong problem when it
failed again.

**A storage-format body may be explicitly empty, but it may not be absent.** Server and Data
Center can legitimately return `body.storage.value: ""` for an empty page, and that indexes as
an empty document. A successful response with no `body.storage.value`, a null value or a
non-string value says nothing about the page's content; treating any of those as the empty string
would erase an indexed revision while advancing its version token. They therefore fail as
`BodyUnavailableError`, just as a missing Cloud representation does. A first fetch leaves no
document or watermark behind, while a failed refresh preserves the last indexed revision so the
source can be retried without publishing guessed content.

**Known Cloud bugs to guard against:** batch page-version endpoints have returned 500 with
ADF requested (CONFCLOUD-80964), and there are reports of ADF returning stale page
content. Validate that the returned `version.number` matches what discovery reported; if
it does not, refetch, and then fall back to `storage` — a different code path on the source's
side, which is the whole reason it is worth trying.

**If every attempt is still stale, the fetch fails closed.** This boundary is worth being exact
about: ingest persists the version token from discovery, while the connector owns the fetched
bytes. Returning an older body would therefore store those bytes under the newer discovery token;
the next sync would find the tokens equal and skip the page forever. A failed fetch stores neither
the stale bytes nor that token, so a later sync can retry without first trusting content it cannot
date correctly. An already indexed revision remains available under the pipeline's ordinary
failed-reingest policy.

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

In a **subtree-scoped** source (§2.1) that is a second CQL enumeration rather than the same one,
because only pages have a descendant predicate worth relying on. It is still one enumeration
rather than a call per page, it is still watermarked, and each attachment's scope is decided by
the page holding it — never by the attachment's own position, which Confluence does not expose
as reliably.

Download and route through the normal parser chain — a PDF attached to a page is a PDF.
Each attachment keeps a link to its parent page so citations resolve to both.

Two details worth stating:

- **The size ceiling is enforced against the bytes that arrive**, never against the declared
  `Content-Length`. The declared length is the source's claim about the response, and the
  point of a ceiling is to survive a claim that turns out to be wrong.
- **The media type is the source's, then the download's, then the filename's.** Confluence's
  `metadata.mediaType` first, the response's `Content-Type` next, and the filename extension
  last; nothing is guessed ahead of what was actually said.

Attachments are documents. A connector that indexed only page bodies would leave the diagram,
the spreadsheet and the specification PDF unsearchable while reporting that the space was synced.

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
**exact section**, not just the page. It costs nothing, since the heading is already known from
§5, and it is the difference between an answer a reader can check in one click and one that
hands them a page to search.

The page URL is taken from the source's own `_links.webui`, joined to `_links.base`, rather
than assembled from a slug. `storage.md` §4.2 declines to claim the slug is title-derived and
therefore unstable; not constructing it settles the question either way, and identity rests on
the page id regardless.

## 9. Two operational realities to document

**Rate limits.** Cloud throttles. Back off on 429 and honor `Retry-After`; a full first
sync of a large space will hit it.

Both forms of `Retry-After` are honored — a delay in seconds and an HTTP-date. A client that
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

The connector is built and tested against recorded behavior rather than a real Confluence:
there are no credentials in this repository, and the suites drive a synthetic instance that
reproduces the traps above (a cursor containing `+`, a version disagreement, a macro cycle, a
429 with `Retry-After`, a page deleted between syncs). Where a real instance could still
differ, it is here — check these first when one is available:

| Assumption | What to check | If it is wrong |
|---|---|---|
| **Omitting `status = current` on Server/Data Center returns *current* content only** — especially after a page is trashed | Trash a page, re-run discovery and reconciliation, and confirm it stops being returned | Deletion detection is what is at risk. If a Data Center search returns trashed content by default, reconciliation sees a deleted page as still present and the index serves it forever. This row replaces the one that assumed `status` was accepted on both deployments: it is not — the standard Data Center content-search resource rejects it — and the synthetic suite proves this client sends the right query, not what that product does with it |
| `GET /rest/api/space/{key}` exists and answers for a single space on both deployments | Configure an explicit `spaces` allowlist and run a sync | A configured allowlist refuses every key as missing. Loud rather than quiet, but it would make explicit scoping unusable on that deployment |
| **`ancestor` is accepted and matches descendants at any depth** | Scope one source to a page tree three levels deep and check the grandchildren arrive | A 400 is the loud outcome and needs nothing. The two quiet ones are guarded: a predicate that is accepted and ignored refuses on the first page outside the tree, and one that matches nothing refuses on the `child/page` cross-check |
| **`ancestor in (a, b)` and `id in (a, b)` are accepted, and parenthesized `OR` between them is** | Configure two roots in one space | Fall back to one query per root; the scope is unchanged and the request count rises by the number of roots |
| **A bare numeric literal is accepted where `ancestor` and `id` want a content id** | Any scoped run | Quote them. The ids are checked to be digits before they reach a query, so quoting would be a formatting change rather than a safety one |
| **`GET /rest/api/content/{id}/child/page` exists on both deployments** | Scope to a root whose tree is genuinely one page | The empty-subtree cross-check stops working. It answers `False` on a 404 and so fails open — the guard would go quiet rather than misfire, which is the wrong direction and is why this row is here |
| **`GET /rest/api/content/{id}` reports `status` and `space.key` for a page on Cloud** | Configure any root against Cloud | Root validation cannot establish the space, and refuses. Cloud's v2 page endpoint reports a numeric space id, so the v1 route is the one that answers this |
| **An attachment search result carries `container.id` under `expand=container`** | A scoped run with `include_attachments = true` | Every attachment is judged out of scope and none is indexed — quiet under-collection, and the row is here because nothing downstream would notice |
| `order by lastmodified asc` is accepted alongside cursor pagination | Same | Drop the ordering; enumeration becomes non-deterministic but stays correct |
| `_links.next` carries the full query and a raw `+` in the cursor | Log one link verbatim | The `%2B` handling is unnecessary but harmless; a *different* encoding would need its own handling |
| Cloud v2 `GET /pages/{id}/ancestors` returns titles, root first | Fetch one page whose ref carries no ancestors | Breadcrumbs from that path are reversed or short; discovery's own expansion is unaffected |
| `metadata.mediaType` is present on attachment search results | One attachment's discovered media type | Falls back to `Content-Type`, then the filename — already the design |
| Search cursors survive at least `cursor_lifetime_seconds` | A slow sync of a large space | Lower the setting; the failure is a refusal, not corruption |
| A page in the trash stops appearing in CQL results | Trash a page, re-run reconciliation | Deletion is never detected, and the index serves the page forever |
| `/rest/api/user/current` exists and names the user | `manicule connector login` reports an account | Capture refuses rather than storing an unproven session; the probe endpoint is the thing to change |
| `X-AUSERNAME` is on REST responses, and is the account `user/current` reported | A sync runs rather than refusing as "a different account" | The account check misfires; it is one of three sign-in signals, so removing it costs the weakest of them |
| Sign-in redirects go to `/login.action` or a listed SSO servlet | Let a session expire and re-run | Caught by origin or by the body markers instead; add the path |
| Nothing legitimate redirects off the configured origin | A full sync including attachments | An attachment download refuses instead of following. Cloud is the case to watch: if its `_links.download` redirects to a media CDN, that redirect is now refused loudly rather than followed |
| A driven Chromium is accepted by the tenant's conditional-access policy | `connector login --browser` against the real instance | The browser path is unusable there; the paste path still works, because that browser is the person's own |
| Sign-in completes without the person needing a second tab or window | Same | The poll watches one context's jar; a flow that finishes elsewhere would time out. The state-import path is the workaround |
| The session cookies Confluence needs are set on the configured origin rather than on a parent domain only | Same, then `connector sync` | The origin filter keeps too little and login reports no applicable cookies. The domain rule already accepts a parent-domain cookie, so this is about an unusual scope rather than the common case |
| `browser_timeout_seconds` (default 300) is long enough for the provider's slowest path | A sign-in with device approval | Raise it or pass `--timeout`; the failure is a refusal that stores nothing |

---

## Summary

Each row is a property this connector holds, beside the thing that happens without it. The
right-hand column is what a Confluence sync looks like when each decision above is skipped —
and every one of those failures is quiet, which is why they are worth listing.

| | manicule | Without it |
|---|---|---|
| Discovery | CQL watermark, cursor pagination | a full space walk every sync |
| Scope | a page tree, narrowed at the source and checked on arrival | a whole space, or `--limit`'s arbitrary prefix |
| Deletions | reconciliation diff | removed pages served forever |
| Body format | ADF on Cloud, parsed storage on Server | one dialect handled, the other guessed at |
| Extraction | typed node walk | markup stripped to a run of words |
| Tables | preserved whole | split across chunks, or flattened |
| Code blocks | preserved, language-tagged | prose-chunked |
| Macros | resolved, with cycle detection | content visible in the UI and absent from the index |
| Attachments | routed through the parser chain | unsearchable |
| Context | full ancestor breadcrumb | a section called "Configuration", of nothing |
| Citation | deep link to a heading anchor | a page URL to search by hand |
| Credential | token, or a browser session behind SSO | an instance nobody can authenticate to |
| A sign-in page | refused, twice over | indexed once per page the sync tried to read |

---

## 11. Where this lives

| Concern | Module |
|---|---|
| Configuration and the credential refusal | `manicule/connectors/config.py` |
| The credential seam: tokens, sessions, expiry | `manicule/connectors/credentials.py` |
| Capturing a session, and where one is held | `manicule/connectors/sessions.py` |
| Telling an answer from a sign-in page | `manicule/connectors/intercept.py` |
| CQL construction, quoting, watermark timestamps | `manicule/connectors/cql.py` |
| Root-page validation, subtree membership, the empty-subtree guard | `manicule/connectors/subtree.py` |
| `_links.next`, the `%2B` rule, origin checking | `manicule/connectors/pagination.py` |
| Auth headers, redirects, 429/5xx retry, downloads, paging | `manicule/connectors/client.py` |
| `include`/`excerpt-include`, both body formats | `manicule/connectors/macros.py` |
| `discover`, `fetch`, `reconcile`, watermarks | `manicule/connectors/confluence.py` |
| Registration through the `manicule.plugins` entry point | `manicule/connectors/plugin.py` |

The ADF node walk is **not** here: it is `manicule/parsers/adf.py`, reached through the parser
chain like any other format, and so are attachments. The connector's job ends at handing over
bytes and saying honestly what they are.

**`Connector.watermark` was added to the protocol for this connector**, in
[#9](https://github.com/mgd43b/manicule/issues/9). `discover` consumed a watermark and nothing
produced one, so `connectors.watermark` (`storage.md` §4.7) could not be filled by anything
working through the protocol. It is read-only and reflects the last **completed** enumeration:
a consumer that abandoned discovery part-way is offered nothing at all, because a watermark
advanced by a walk that did not finish is a position past documents nobody received, and the
next sync starts there and never sees them again. `assert_connector_contract` checks it, with
a fake that advances on yield to prove the check fires. `contracts.md` §3 carries the full
statement; the caller's half — persist it only once what the run produced is stored — is
[`ingest.md`](../ingest.md) §13.2.

The five-minute overlap in §2 is what makes the remaining race survivable rather than
theoretical: a run interrupted between yielding a document and committing it re-enumerates that
document next time, and change detection skips it if it did land.

---

## 12. Offline snapshots — the same wiki, from a directory

Everything above needs a base URL, a credential and a reachable instance. Often none of those is
available: an air-gapped install, a wiki nobody has API access to, an export taken once and
archived. `manicule/connectors/confluence_snapshot.py` ingests that export, with **no network and
no credentials**, and it is registered separately as `confluence-snapshot`.

A name of its own rather than a mode of `confluence`, because the two share no configuration —
this one has no base URL, no deployment and no auth. Folding them together would produce a config
model where over half the fields are refused depending on another field's value, and a connector
that reaches no network could then be misconfigured into trying.

### 12.1 The input

One directory is one page:

```
<root>/anything/at/any/depth/
    confluence.json     # the manifest
    body.xhtml          # the raw page representation
```

A directory holding `confluence.json` **is** a page snapshot, and the walk does not descend into
one — so attachments or resources stored beside a body cannot be mistaken for pages of their own.

The manifest carries `page_id` (required), and optionally `title`, `space_key`, `canonical_url`,
`version`, `created_at`, `modified_at`, `ancestors`, `ancestor_ids`, `content_status`, `labels`,
`attachments`, `retrieved_at`, `body_file` and `body_checksum`. Only `page_id` is required, and it
is required for a reason no default can supply: it is the document's identity.

### 12.2 It is a wire format, not an extension of the core record

The manifest is **what somebody else's export tool writes**, so its field names are Confluence's
and its spellings are a compatibility surface. It is mapped onto
[`storage.md`](../storage.md) §4.2.1's `SourceMetadata` rather than being it:

| Manifest field | Where it lands | Generic? |
|---|---|---|
| `page_id` | `SourceMetadata.source_id`, and the document's `source_id` | yes |
| `title` | `SourceMetadata.title` | yes |
| `canonical_url` | `SourceMetadata.canonical_uri` | yes |
| `version` | `SourceMetadata.version` | yes |
| `created_at` / `modified_at` | the same, on the record | yes |
| `space_key` + `ancestors` | `SourceMetadata.section_path`, coarsest first | yes |
| `space_key` | also `metadata["space_key"]` | **no** |
| `ancestor_ids` | `metadata["ancestor_ids"]` | **no** |
| `content_status` | `metadata["content_status"]` | **no** |
| `labels` | `metadata["labels"]` | **no** |
| `attachments` | `metadata["attachments"]` | **no** |

**The rule, and this connector is the first real test of it: the record carries what every source
has; anything one product means and others do not stays in the connector's own keys.** A space key
is tempting to add to the record — it is listed beside the canonical URL as a citation requirement
— and it would be wrong, because a filesystem mirror and a documentation export have nothing to
put there. What the record contributes instead is `section_path`, so the breadcrumb works with no
Confluence knowledge anywhere in core. **No field was added to `SourceMetadata` for this
connector.**

### 12.3 Identity is the page id, never the path

The difference from the filesystem connector, where identity *is* the resolved path. A mirroring
tool that renames a directory, or organizes by space this year and by page tree next year, has not
created new pages — but a connector keyed on the path would report every document deleted and every
document new, dangling every citation into the previous corpus. So `DocRef.source_id` is the page
id and the directory travels in `DocRef.metadata`, which is what that field is for.

`parsers/expansion.py`'s `member_source_id` settled the same question for archive members:
identity comes from a stable key, never from a position.

### 12.4 Three disciplines carried over from the sidecar unchanged

They are the same threats, because a manifest is a file in the corpus and anyone who can get a
directory indexed can write one.

- **The manifest never authorizes a read.** The body is found by *looking* — it is the file beside
  the manifest. A declared `body_file` is compared against what was found, never followed, so
  `../../../../etc/passwd` is a name matching nothing in the directory and at no point a path
  anything opens. Exactly one candidate is the body; several with no declaration is a refusal
  rather than a guess.
- **The change token covers the pair.** A manifest corrected to declare a new version changes what
  every citation says while leaving the body byte-identical, so both files' size and modification
  time go into the token. [`ingest.md`](../ingest.md) §4 names the failure this avoids.
- **An unusable manifest never costs the page.** The reason is recorded on the record instead, and
  the body is still indexed with local identity.

### 12.5 Nothing is dropped in silence, including a snapshot that cannot be used

Every directory holding a manifest becomes a document — even one whose manifest is unreadable, or
whose body is missing or ambiguous. Skipping them is the quiet failure: an export of ten thousand
pages would ingest nine thousand and report a clean run.

Where no `page_id` can be read there is no identity to key on, so one is derived from the directory
behind an explicit `unidentified:` prefix. That identity is deliberately **unstable across a
manifest repair**: fixing the manifest makes the page appear under its real id and makes the
placeholder stop appearing in `reconcile`, which the ordinary deletion pass (§3) then soft-deletes.
A stable-looking placeholder would instead leave two live documents for one page.

### 12.6 What this connector cannot yet do, and says so

Storage-format XHTML is routed as `text/html` and read by the generic HTML parser — which is
exactly what the live connector does for Server and Data Center today.

**Both of the losses this paragraph used to describe are now fixed, and the diagnostic that
described them had to be narrowed twice.** It is recorded here rather than deleted, because the
shape of the mistake is more instructive than either bug.

It first said every `<ac:plain-text-body>` reparsed as a bogus comment so its content was *absent
from the document*. That was true, and #90 fixed it — `recover_cdata` recovers those sections as
escaped text. The diagnostic was not narrowed with the fix, so for as long as #90 has been merged
the connector emitted `!body-content-dropped: … content is absent from this document …` for bodies
that reached the index intact. A warning that outlives its bug sends somebody looking for content
that is already there, and teaches them to distrust the ones that are still true.

It then said the macros themselves were uninterpreted, listing every macro on the page. That was
true until the storage-format parser existed. It now reads most of them, so the list is filtered
against what the parser declares it understands — asked of the parser rather than restated here,
because two copies of that answer drift the first time a macro is taught to one and not the other.

What remains is narrow and true:

- `metadata["uninterpreted_macros"]` — the macros with **no reader**, which the parser emits an
  explicit placeholder for. Absent when there are none.
- `metadata["unrecoverable_macro_body"]` — a CDATA section that is **never closed**, so recovery
  leaves it as it found it rather than guessing where it ended and its content really is absent.
  Separate from the first because "we did not understand this" and "we do not have this" are
  different claims, and only the second is data loss.

The general lesson: **a diagnostic is a claim about behavior, and it goes stale exactly when the
behavior improves** — which is the moment nobody is looking at it. `parsing.md` §2.4 carries the
anchor row, with the media type that gives it meaning.
