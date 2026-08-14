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

**The session lives in the macOS Keychain.** Not `config.toml`, even at `0600`: a session cookie
is the sync account's whole identity at that company rather than a scoped grant, and a
configuration file reaches version control eventually. The configuration model forbids unknown
keys, so a `session_cookie` written there is a startup error rather than a working setting.
Nothing lands under `<data_dir>`, so `docs/storage.md` §7.1 and the `doctor` permissions check
do not come into it. On a machine with no Keychain the fallback is `$CONFLUENCE_SESSION_COOKIE`,
a per-run credential manicule never writes down.

Two things about the Keychain are worth recording because neither is guessable. Items are
created with `-T /usr/bin/security`, which is the narrowest grant that still lets an unattended
sync read them without raising a dialog; `-A`, which would let anything read them silently, is
not used. And **`security` truncates a secret read from stdin at 128 bytes, silently, reporting
success** — measured, not assumed. A session record is longer than that and an instance behind
single sign-on issues cookies of its own besides Confluence's, so the record is written in
120-byte pieces across numbered items and read back and compared before the capture is called
done. Passing the secret as a command-line argument would avoid the chunking and put a live
corporate session into a process listing.

Writing a credential in pieces is also what makes *replacing* one dangerous, because there is a
window in which some of the pieces are the old session and some are the new.
[§1.1d](#11d-how-a-session-is-stored-and-what-happens-if-the-machine-stops-mid-replacement) is
how that window is closed and exactly what it is and is not guaranteed to survive.

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

`--forget` removes everything that instance's session occupies: the record that is current, a
record left behind by a replacement that was interrupted, both commit slots, and a record written
by a version of manicule that predates them (§1.1d). It removes nothing belonging to any other
instance. It does not sign you out of Confluence itself, which is your browser's business and
your identity provider's.

**A failed login never costs you a working session.** Verification happens before the store is
touched, so a timeout, a closed window, a dead cookie or a state file for the wrong site leaves
whatever was stored exactly as it was: the write is the last thing that happens and it happens
only on success.

That covers a login that fails. §1.1d covers a *write* that fails, which is a different problem
and used to have a different answer.

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

### 1.1d How a session is stored, and what happens if the machine stops mid-replacement

This section exists because the answer used to be "you lose the session you had". The store
deleted the old record before writing the replacement, so a `security` invocation that failed on
the fourth of twenty-three pieces — or a laptop that slept, or a `^C` — left the instance with no
credential, having started with a working one. Verifying the new cookies first, which manicule
does, protects you from a bad *cookie*; it does nothing about a bad *write*.

**A replacement is now written somewhere nothing is reading, and published by one small write.**
Four kinds of record live under the keychain service `manicule: confluence session`, each named
after your instance's base URL:

| Record | Account name | What it holds |
|---|---|---|
| Session pieces | `<base_url>#<generation>#<n>` | the session itself, in 120-byte pieces |
| Commit slots | `<base_url>#p0`, `<base_url>#p1` | which generation is current, its size and a checksum |
| Journal | `<base_url>#staged` | generations written but not yet cleared up |
| Legacy | `<base_url>#<n>` | a session stored by a version before all of this |

A replacement notes the generation it is about to write, writes it under a fresh random name,
reads it back and compares it byte for byte, and only then writes the commit slot that is *not*
currently in force. **That write is the commit point.** Everything before it is invisible to a
reader; everything after it is tidying up.

**What you get if the machine stops.** Stop it anywhere before the commit — any piece, the
read-back, the commit slot itself — and the session you had is still there, complete, and the
next sync uses it. Stop it after the commit and the new session is there, complete. There is no
third outcome: a reader never sees pieces of two sessions spliced together, never a half-written
one, and never a session that was not read back and checked. A generation left behind by an
interrupted replacement is harmless — nothing points at it — and `--forget` still removes it.

**What this does not claim.** macOS does not document `security`'s update of a single item as
atomic, and manicule does not assume it is. That is what the second commit slot is for: an
interrupted commit can damage at most the slot being written, and the other slot still holds the
commit before it, checksum and all. If the current generation is ever unreadable, manicule falls
back to that previous commit only when its own checksum still matches — a verified older session
or an error, never a guess. If neither can be verified, `connector login` says the stored session
is incomplete rather than reporting that none exists, because those are different problems.

**Upgrading.** A session stored by an earlier version keeps working and is read as it was
written. Your next successful `connector login` writes the new format and removes the old record
*after* the new one is in force, so an upgrade cannot cost you a session either. Clearing up is
the only step allowed to fail quietly: if it does, the session you just captured is in use and a
warning tells you to run `--forget` to clear what was left over.

**Two logins at once** is not something manicule serializes. Both write their own generation, so
neither can corrupt the other; the last one to commit wins and the other is silently superseded.

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
  handling silently breaks pagination partway through a sync — an unrecognized cursor comes
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

Ancestors — the page hierarchy used for breadcrumbs in §7 — come from discovery, which
expands them for every page it returns, so the fetch needs no second call. A ref built
somewhere else (a re-fetch, a targeted single-page sync) carries none, and then the Cloud
ancestors endpoint is asked; an ancestor whose title it omits is skipped rather than filled
in with an id, and the document records that its breadcrumb is incomplete.

**A page that comes back with no ADF body at all** — the format declined for a page that
exists, which is not the same as an empty page — is read as storage format instead. Only that
failure falls back: a 429 or a rejected credential answered by trying a second endpoint would
double the load on a source that has just said stop, and report the wrong problem when it
failed again.

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
| `status = current` is accepted by CQL on both deployments | A search returns results rather than a 400 | Deletion detection is the thing at risk: without it, trashed pages may still be returned |
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
| `security` still truncates a stdin secret at 128 bytes | The Keychain round-trip test | The chunk size is wrong; the read-back comparison turns it into a refusal rather than a truncated credential |
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
| Capturing a session, and the Keychain | `manicule/connectors/sessions.py` |
| Telling an answer from a sign-in page | `manicule/connectors/intercept.py` |
| CQL construction, quoting, watermark timestamps | `manicule/connectors/cql.py` |
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
