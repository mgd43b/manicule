# Gmail ingestion

Design for the Gmail connector. Ticket [#457](https://github.com/mgd43b/manicule/issues/457),
epic [#464](https://github.com/mgd43b/manicule/issues/464).

> **Status: design. Nothing on this page is implemented yet.** Every "does" below is a
> decision for [#458](https://github.com/mgd43b/manicule/issues/458)–[#463](https://github.com/mgd43b/manicule/issues/463)
> to implement, not a description of code that exists. When a section ships, its ticket
> replaces this banner's claim for that section with the module that implements it (§12).

A mailbox is the most sensitive source manicule has been pointed at, and the obvious
implementation gets three things wrong quietly: it re-downloads the whole mailbox to find what
changed, it never notices that a message was deleted, and it serves the result to whoever can
reach the port. This connector reads each message once, keeps up with Gmail's own change log,
removes what left the mailbox, and refuses to run in an installation that serves without
authentication.

The connector ends at `RawDocument`. A message is fetched as its raw RFC 5322 bytes and handed
to the existing `message/rfc822` parser ([`parsing.md`](../parsing.md) §10), which already
anchors citations to lines of the canonical message text and sends attachments through the
parser chain. There is no Gmail-specific parsing.

---

## 1. Auth

**A Google Workspace *Internal* OAuth client, a refresh token, and one scope:
`https://www.googleapis.com/auth/gmail.readonly`.** Nothing else is requested, and the scope
cannot send, modify or delete mail.

| What | Where it comes from | How a request uses it |
|---|---|---|
| Client id | Configuration (not a secret) | Sent to Google's token endpoint |
| Client secret | `client_secret`, else `$client_secret_env` | Sent to Google's token endpoint |
| Refresh token | `refresh_token`, else `$refresh_token_env` | Exchanged for an access token |
| Access token | Minted by manicule, held in memory | `Authorization: Bearer <token>` |

**Internal, because it removes two failures that look like bugs.** An *External* client in the
"Testing" publishing status is issued refresh tokens that expire after seven days, so a sync
that worked on Monday fails the following Monday with nothing changed. Publishing an External
client instead puts a restricted scope through Google's verification. An Internal client is
available only to accounts in the Workspace organization that owns the project, which is
exactly the audience this connector has, and neither rule applies to it.

### 1.1 What was rejected, and why

- **A service account with domain-wide delegation.** It needs no person to sign in, which is
  attractive for a headless sync, but its key can read *every* mailbox in the domain, not one.
  New Google Cloud organizations also block service-account key creation by default, so the
  setup would start by weakening an organization policy. The blast radius is wrong for a
  connector that indexes one mailbox.
- **IMAP with an app password.** It needs no Cloud project, but administrators commonly disable
  app passwords, the password grants full mailbox access rather than read-only, and IMAP's
  change signal (UIDs under `UIDVALIDITY`) is weaker than Gmail's history log.
- **The device-code flow.** It would let a headless pod sign in without a browser on the same
  machine, but Google does not allow Gmail scopes on that flow — its scope list is identity,
  two Drive scopes and YouTube.
- **An External client.** See above: seven-day tokens in Testing, verification in production.

### 1.2 Where the refresh token lives, and why it is not like the browser session

The Confluence browser session deliberately lives only in the running server's memory
([`confluence.md`](confluence.md) §1.1, `connectors/sessions.py`): re-taking it is a few seconds
of clicking, so a credential whose lifetime is the process's is re-taken on purpose rather than
persisted. That reasoning does not carry over. A refresh token is minted once by a person at a
browser ([#461](https://github.com/mgd43b/manicule/issues/461)) and then has to survive every
restart of a server that may run where no browser exists. Holding it only in memory would mean
a person signs in again after every restart, deploy and node failure.

So the refresh token is a long-lived secret supplied by the deployment's own secret store, read
from the environment exactly as the Confluence API token is: an explicit value in configuration
wins, the named environment variable fills it otherwise, and printing the configuration never
shows it. A credential in a configuration file is a credential in version control eventually,
so the documented path is the environment variable.

**It is resolved and checked in the factory, before the connector exists.** A missing client
secret or refresh token is a `ConfigError` naming the setting and the environment variable.
`doctor` reports whether a refresh succeeds, so a revoked grant is visible before a sync rather
than at its first request ([#458](https://github.com/mgd43b/manicule/issues/458)).

**Access tokens are minted from it and cached until shortly before `expires_in`.** A 401 on an
API call refreshes once and retries once; a second 401 is a typed credential error. Google
answering the refresh with `invalid_grant` — the grant was revoked, the account's password
changed under a policy that revokes tokens, or the token aged out — is the same typed error,
and its renewal text names `manicule connector login <name>`.

**Signing in again replaces the token, and the old one is not reliably dead.** Google keeps at
most 100 refresh tokens per account per client and invalidates the oldest *without warning*
when the 101st is issued. Running `connector login` repeatedly is therefore harmless to the
newest token and fatal to the oldest, which matters only if two installations share one client
and one account. Revoking the grant in the Google account kills every token at once; that is
the off switch.

### 1.3 Setting it up

For a Workspace administrator, once per organization:

1. In a Google Cloud project owned by the Workspace organization, enable the Gmail API.
2. Configure the OAuth consent screen with user type **Internal**.
3. Create an OAuth client of type **Desktop app**. The loopback redirect that
   `connector login` uses is what that type is for.
4. If the domain restricts access to Google data for unconfigured third-party apps, mark the
   client **Trusted** under Security → Access and data control → API controls.

Then, as the account whose mail is indexed:

```console
$ manicule connector login mail          # opens a browser; writes the refresh token where told
$ manicule doctor                        # the credential check should pass
$ manicule connector sync mail
```

```toml
[connectors.mail]
type = "gmail"
enabled = true

[connectors.mail.options]
account = "someone@example.org"   # the mailbox this instance indexes, and no other (§1.4)
client_id = "1234567890-example.apps.googleusercontent.com"
client_secret_env = "GMAIL_CLIENT_SECRET"
refresh_token_env = "GMAIL_REFRESH_TOKEN"
exclude_labels = []          # SPAM, TRASH and DRAFT are always excluded (§3)
```

### 1.4 One instance, one mailbox, pinned

**A token can only read the mailbox that granted it, so the token decides which mailbox is
indexed — and a token is replaced routinely.** Renewing a revoked grant, or re-running
`connector login` from a browser signed in to a different account, would otherwise switch an
instance to another mailbox with every configured field unchanged. The pipeline keys stored
sync state on the connector's name and scope fingerprint and reuses a watermark whenever the
fingerprint matches, so the new mailbox would resume from the old one's history position and
unfinished inventory, while the old mailbox's mail stayed indexed beside it.

So the mailbox is configuration, not a side effect of the credential:

- **`account` is required**, and it is the address the instance indexes.
- **It is checked against `users.getProfile`'s `emailAddress`** in the factory, in `doctor`'s
  credential check and at the start of every run. A token for any other mailbox is a typed
  credential error, and the run does not begin. The error says the token belongs to a different
  mailbox without printing either address, since run-level errors are aggregate (§10). Ordinary
  renewal cannot switch mailboxes, because the check fails before anything is read.
- **`account` is part of the scope fingerprint** as well, so stored state from one mailbox is
  never resumed for another even if the check were bypassed.
- **Indexing a different mailbox is a different instance**, under a new connector name and
  therefore a new source identity. The old instance's documents do not leave by themselves —
  reconciliation is not wired (§7) — so they are deleted explicitly before or after the switch.
  Editing `account` on an existing instance is refused with the same instruction rather than
  treated as a scope change, because a scope change keeps the source and re-walks it, leaving
  removal to reconciliation — and what would be left behind here is someone else's mail.

---

## 2. Identity and citations

- **`source_id` is Gmail's message id.** It is assigned when the message enters the mailbox and
  never changes, which is what lets change detection skip a message by id alone (§4). It is not
  the RFC 5322 `Message-ID` header, which the sender chooses, may be absent, and is shared by
  every copy of a message — a message sent to yourself is two Gmail messages with one header.
- **`uri` is a link a person can open.** The search URL for the message's `Message-ID`
  (`https://mail.google.com/mail/?authuser=<account>#search/rfc822msgid:<id>`), which survives
  Gmail interface changes better than an internal view path, and names the account so a
  browser signed in to several opens the right one. **Both values are percent-encoded before
  they are inserted**: a `Message-ID` can contain `+`, `%` and `#`, an address can contain `+`,
  and none is safe raw — a `#` ends the search term and a `+` reads as a space. The angle
  brackets around a `Message-ID` are stripped first, as Gmail's own `rfc822msgid:` examples
  show it without them. A message without a `Message-ID` falls back to `#all/<gmail id>`. §11
  lists the link as unverified against a live account.
- **Title** is `Subject`, which the parser already makes the first element of the heading path.
- **Metadata** carries the thread id and `internalDate`. Neither appears on any status surface:
  status reports aggregate counts only, never an id, address, subject or label.

---

## 3. Scope

**Everything in the mailbox except SPAM, TRASH and DRAFT, minus any labels in
`exclude_labels`.** There is no inclusion list and no free-form query.

**SPAM, TRASH and DRAFT are always excluded, and not configurable.** A mail index that contains
spam is a phishing corpus that answers questions; trash is mail its owner removed. Drafts are
unsent and rewritten continuously while being edited, so an indexed draft is a stale copy within
minutes and a sent message is indexed separately once it is sent.

**`exclude_labels` takes label names**, resolved to label ids once per run through
`users.labels.list`. An unknown name is a configuration error, not an empty exclusion — a typo
that silently excluded nothing would index precisely the mail the operator meant to keep out.

**Why there is no `query` setting.** Gmail's search syntax is the obvious way to narrow a
mailbox, and it cannot be used here without breaking the one rule scope has to keep: discovery,
incremental discovery and reconciliation must agree on which messages are in scope
([`confluence.md`](confluence.md) §3 states the failure each way round). A full walk can pass a
query to `messages.list`, but `history.list` cannot filter by one, and a history record carries
at most a message's label ids. A query that mentioned a sender, a date or a word would be
evaluated on the full walk and impossible to evaluate on every incremental sync after it. Label
membership is the only scope all three paths can test, so it is the only scope there is.

**Each path tests label ids, and no path translates a label into a search query.** A query
built from names would be lossy in exactly the cases that matter: a name with spaces or a
slash, a nested label, a category such as `CATEGORY_PROMOTIONS` that search spells
`category:promotions` rather than `label:`. So:

- **A full walk and reconciliation** list the excluded labels first — one
  `messages.list?labelIds=<id>` walk per excluded label, DRAFT included — into an exclusion
  set, then page `messages.list?includeSpamTrash=false` and skip every id in that set.
  Listing costs 5 units per request, each returning up to 500 ids, so each excluded label costs
  what its size costs to list.

  **The exclusion set is held in memory, and that is a bounded exception, not the inventory.**
  A connector has no storage of its own — the protocol gives it `discover`, `fetch` and
  `reconcile`, and durable ingest journals what they yield — so the set lives in the connector
  for the length of one walk. It holds only the ids of excluded messages: the drafts, for a
  mailbox indexed whole, and the size of any label added to `exclude_labels` beyond that —
  tens of thousands of ids for a busy category, a few megabytes. The complete inventory is never
  held this way, which is the invariant durable acquisition protects. A mailbox whose exclusions
  run to millions of messages is the case that would need a storage seam for this, and it would
  be a core change rather than this connector's.
- **Incremental discovery** needs no exclusion set of message ids. Each `discover(watermark)`
  walk resolves `exclude_labels` to label ids afresh (one `labels.list` call) and tests each
  message's label ids against that set of excluded *label* ids (§4.2). Nothing is carried from
  one walk to the next, and a changed `exclude_labels` never reaches an incremental walk at
  all: it changes the scope fingerprint, which forces a complete walk first.

A label applied or removed between the exclusion listing and the main walk is caught by the
overlap §4.1 describes: the position is captured before either listing, so the next incremental
sync sees that label change and acts on it.

**`exclude_labels` is a scope field.** It enters the scope fingerprint, and changing it discards
the stored history position, so the next sync is a complete walk under the new scope. **A walk
does not remove anything.** Widening the scope is handled by it: newly included messages are
yielded and fetched. Narrowing is not: messages that are now excluded stay indexed until a
reconciliation against the new inventory removes them, and a narrowing large enough to trip the
deletion ceiling waits for confirmation (§7). Reconciliation after every full walk is part of
[#466](https://github.com/mgd43b/manicule/issues/466), so a scope change is deletion-capable only
once that lands.

---

## 4. Discovery and change detection

Gmail keeps a per-mailbox **history log**: every message added, every message deleted and every
label added or removed advances the mailbox's `historyId`, and `users.history.list` returns the
records after a given id. That is the change signal, and the watermark is a `historyId`.

### 4.1 The full walk — `discover(None)`

1. Call `users.getProfile` and keep its `historyId`. **This is captured before the walk begins**,
   and it becomes the watermark if the walk completes.
2. List the excluded labels into the exclusion set (§3).
3. Page `users.messages.list` (500 ids a page, the maximum) with `includeSpamTrash=false`, and
   yield each id not in the exclusion set, with its version token.

**Why the position is captured first.** Google's own sync recipe takes the `historyId` of the
newest message *after* listing. Anything that arrives while a long walk is in progress — and a
first walk of a large mailbox is long (§6) — may or may not appear in the listing depending on
where the walk is. Capturing the position first means the next incremental sync starts from
before the walk began and sees every such message again. Re-seeing a message costs a version
comparison; missing one costs the message. The overlap that Confluence buys with a five-minute
reach-back ([`confluence.md`](confluence.md) §2) is structural here.

**The version token is a constant.** A Gmail message's bytes never change after it is created —
edits create new messages, and labels are not part of the bytes. So every message carries the
same token (`rfc822/1`), and a message already stored is never fetched again by id.

**The `1` is a representation version, and changing it alone would change nothing.** Its purpose
is to force one complete re-fetch when what this connector fetches changes. But incremental
discovery resumes from the stored history position (§4.2), so a message with no new history is
never yielded, never reaches the version comparison, and keeps its old token forever. The
representation version is therefore **also part of the scope fingerprint**. The pipeline reuses
a stored watermark only under the fingerprint it was recorded with, so bumping the version
discards the history position, the next sync is a complete walk (§4.1), and every in-scope
message is yielded with the new token and fetched. An unfinished run recorded under the old
version must not be resumed either; [#460](https://github.com/mgd43b/manicule/issues/460) tests
that a bump mid-run starts a fresh complete walk rather than finishing the old one.

The alternative was a hash of the message's label ids, which would let labels become searchable
metadata later. It is rejected for now because labels move constantly — every read, star and
archive is a label change — and each would re-fetch a message whose bytes did not change, at the
highest quota cost the API has (§6).

### 4.2 Incremental — `discover(watermark)`

`users.history.list` from the stored `historyId`, for `messageAdded`, `labelAdded` and
`labelRemoved` records. The first response's own `historyId` (the mailbox's position at that
moment) becomes the next watermark if the walk completes; the overlap reasoning in §4.1 applies
unchanged.

| Record | Effect |
|---|---|
| `messageAdded` | A candidate. |
| `labelRemoved` | A candidate — it may be a message restored from Trash or released from an excluded label. |
| `labelAdded` | Nothing at discovery. A label can only take a message out of scope, and out-of-scope messages leave through reconciliation (§7). |
| `messageDeleted` | Nothing at discovery, for the same reason. |

**A candidate is yielded only if its *current* label set is in scope**, and the current set is
looked up, never inferred from the record. A `labelRemoved` record names the labels removed, not
the ones that remain: a message taken out of Trash that still carries an excluded label is out of
scope, and only its full label set says so. History records also "typically" carry only the
message and thread ids. So each candidate costs one `messages.get?format=minimal` (20 units) for
its current label ids, once per walk however many records name it; a message that no longer
exists answers 404 and is skipped.

**Discovery cannot report a deletion**, because the connector protocol has no channel for one:
`discover` yields documents that exist, and absence is established only by a complete inventory
([`contracts.md`](../contracts.md) §3). That is why the last two rows do nothing here, and §7 is
where they take effect.

### 4.3 When the history is gone

Google keeps history "typically" for at least a week, and in rare cases only for hours. A
`startHistoryId` that is too old is answered with **404**, typed as `CursorExpiredError`, and the
run falls back to a complete walk from §4.1.

**That fallback is cheap, and the constant version token is the reason.** A complete walk lists
ids at 5 quota units per request of up to 500 ids, and fetches only the ids not already stored,
so a mailbox of 100,000 messages that missed a week of history costs about 200 requests and
1,000 units of listing plus the new
mail — seconds of quota, not a re-download (§6).

### 4.4 Batches

Each `messages.list` page and each `history.list` page is one discovery batch
(`BatchedDiscoveryConnector`), so durable ingest commits a page's records before the next page
token is followed ([`ingest.md`](../ingest.md) §8.3.1).

---

## 5. Fetching

**`users.messages.get?format=raw`, base64url-decoded, returned as
`RawDocument(media_type="message/rfc822")`.** The raw format is the whole RFC 5322 message with
every attachment inline, so there is no separate `attachments.get` call and no per-attachment
cost. Attachments become documents of their own through the parser chain, with `container_id`
pointing at the message, exactly as an `.eml` file's do.

**A 404 at fetch is a message deleted between discovery and fetch**, typed `NotFoundError`, and
proves nothing about anything else in the mailbox.

**Size** is the pipeline's existing limit on a fetched body, `ingest.max_fetch_bytes`, not a
Gmail setting. A message that exceeds it is refused, never truncated: a truncated message would
parse, cite and answer from the half that arrived. The refusal is recorded against that
message's own document, inside the workspace like the message itself, and counted in the run
report; the count is all that reaches a status surface (§10).

---

## 6. Quotas and pacing

Google's quota page (last updated 2026-10-16; checked for this document on 2026-10-23):

| Limit | Quota units |
|---|---|
| Per user per project, per minute | **6,000** |
| Per project, per minute | 1,200,000 |
| `messages.get`, `format=raw` | **60** (20 for `minimal`) |
| `messages.list` | 5 |
| `history.list` | 2 |
| `getProfile` | 1 |
| `labels.list` | 1 |

**The consequence is the most important number on this page: about 100 raw messages a minute
per mailbox.** 6,000 units divided by 60. A first sync is bound by that, not by parsing or
embedding. At the default pacing budget below (90 messages a minute):

| Messages | First-sync fetch time at the default budget |
|---|---|
| 10,000 | about 1 hour 50 minutes |
| 100,000 | about 18.5 hours |
| 500,000 | about 3.9 days |

After the first sync the budget is ample: a day's new mail is a few hundred fetches, history
pages cost 2 units, and a complete inventory of 100,000 ids is about 1,000 units.

**Paced, not throttled into.** The client meters its own spending against a per-minute budget,
`quota_units_per_minute` (default 5,400 — 90% of the per-user limit, leaving room for anything
else using the same project), and waits before a request would exceed it. A 429 is still
handled — honoring `Retry-After` in both its forms, and stopping the run when the requested
pause exceeds `max_retry_after_seconds`, as the Confluence client does — but it is the
exception, not the steering mechanism. A client that discovers its limit by hitting it spends
most of a day's first sync being told no.

**An interrupted first sync resumes**, through durable acquisition
([`ingest.md`](../ingest.md) §13.2): committed messages are not fetched again, so stopping a long
first sync costs the minutes since the last commit, not the hours before it.

**The discovery journal bounds the largest mailbox a first sync can take.**
`ingest.max_journal_records` (default 1,000,000) caps the discovery records held by acquisition
runs that have not settled, and a run settles only when it finishes — so the cap applies to the
whole first walk, however far fetching has got. A walk that reaches it stops safely rather than
growing the journal without bound, and the first sync cannot complete. A mailbox of more than
about a million messages in scope therefore needs that limit raised, with the disk it implies,
or `exclude_labels` used to bring the first sync under it.

**Errors**, as the client types them ([#459](https://github.com/mgd43b/manicule/issues/459)):

| Gmail answers | Typed as | Effect |
|---|---|---|
| 429; 403 `rateLimitExceeded` / `userRateLimitExceeded` | `RateLimitedError` | Retried after `Retry-After` |
| 404 on `history.list` | `CursorExpiredError` | Complete walk (§4.3) |
| 404 on `messages.get` | `NotFoundError` | That message only |
| 401 after one refresh; `invalid_grant` | Credential error | Run stops; renewal names `connector login` |
| Any other 403 | `PermissionDeniedError` | Run stops |
| 5xx, timeout | `RemoteError` / `RequestTimeoutError` | Retried; proves nothing about membership |

---

## 7. Deletions — and the gap this connector exposes

**A message deleted from Gmail, moved to Trash, marked as spam or given an excluded label must
leave the index.** For mail this is not housekeeping. Deleting a message is how its owner says
it should not exist, and an index that keeps serving it has overruled them.

**The mechanism is reconciliation: a complete id inventory diffed against what is stored**
([`ingest.md`](../ingest.md) §11). `reconcile` builds the same exclusion set and pages the same
`messages.list` walk as the full walk — the same scope, by construction — through
`BatchedReconciliationConnector`, and raises rather than returning a prefix if any page fails.
It is cheap: ids only, 5 units per request of up to 500 ids, so a 100,000-message mailbox is
about 200 requests and 1,000 units, a
few seconds of the per-minute budget. A mail installation can afford to reconcile daily or
hourly.

**Nothing in manicule runs reconciliation today, for any connector.**
`manicule.ingest.reconcile.reconcile` exists, is tested, and has no caller in the product; no
command, service method or schedule invokes it, and `reconcile_interval_s` is read nowhere but
its own definition. `connector sync` never removes a document that left its source. Two things
do remove documents, and neither helps here: `manicule index --watch` removes a file on its
delete event, which only a watched directory has; and offline rebuild publication
([`ingest.md`](../ingest.md) §10.4) retires what a replacement does not contain, which needs
retained source bytes and is not a sync. [`ingest.md`](../ingest.md) §11's
"after every full sync, and on a schedule" describes the design, not the code.

For Confluence that means a deleted page lingers. For a mailbox it means deleted mail is
searchable forever. **Wiring reconciliation into sync is therefore a prerequisite of this
epic**, not an improvement to it: the Gmail connector does not ship to a release before
reconciliation runs on the schedule `reconcile_interval_s` describes. That work is
[#466](https://github.com/mgd43b/manicule/issues/466), which also adds a per-connector interval
so a mail source can reconcile daily or hourly.

**Once it runs, two existing behaviors matter more for mail than for anything else:**

- **The deletion ceiling.** A reconciliation that would delete more than
  `reconcile_max_delete_fraction` (default 10%) of the connector's live documents is refused and
  recorded as a proposal for confirmation (§11.1). Emptying a large folder into Trash, or adding
  a busy label to `exclude_labels`, will trip it. That is the guard working, and the proposal is
  confirmed rather than the ceiling raised.
- **Removed from results is not removed from disk, and nothing here erases a message.** A
  reconciled message is soft-deleted. When the grace period ends (`soft_delete_grace_s`, 30 days
  by default, [`storage.md`](../storage.md) §8.2), the sweep removes its chunks and vectors — the
  derived text — and keeps the document row, which still carries the subject as its title, and
  its reference to the retained bytes, so that a citation can explain itself. Blob collection
  treats every document, version and acquisition reference as a root whatever the document's
  status, so **retained raw mail is kept indefinitely** after the grace period. Lowering
  `soft_delete_grace_s` shortens how long deleted mail is *searchable as derived text*; it does
  not erase it.

  Erasing one message today is manual: `manicule document delete --hard` removes the row, its
  versions and its chunks, and the raw bytes go only once no acquisition record still references
  them and blob collection runs. For an installation that must be able to forget, the honest
  configuration is retention off **from the first sync** (§9), which leaves no raw copy to erase —
  at the re-fetch cost §9 states. Turning it off later does not remove copies already kept. Erasure that follows the source on its own is not designed here; it is a requirement
  on the reconciliation work §7 depends on
  ([#466](https://github.com/mgd43b/manicule/issues/466) decides what happens to a reconciled
  document's retained bytes), not something this connector can add.

**A message restored from Trash comes back**, unless it still carries another excluded label.
Its `labelRemoved` record makes it a candidate and its current labels decide (§4.2). Once it is
yielded, the pipeline must restore the soft-deleted document rather than skip it as unchanged. [#460](https://github.com/mgd43b/manicule/issues/460) tests that round trip.

---

## 8. One document per message

Each message is a document. Quoted reply chains are kept, as [`parsing.md`](../parsing.md) §10
decides for every email: the quoted text is frequently the only statement of what was being
replied to, and trimming it is a retrieval decision rather than a parsing one.

The cost is that a long thread repeats its early messages in every later one, and a search can
return the same quoted paragraph several times from several messages. Whether that is worth
changing — collapsing hits from one thread at retrieval, or indexing threads instead of messages
— is measured on synthetic threads before it is decided, in
[#463](https://github.com/mgd43b/manicule/issues/463). Thread-level documents would also change
what a citation points at, which is why it is not assumed.

---

## 9. Retention

Both values of the per-connector `retain_source_bytes` are supported, and for mail the choice is
a real trade with no default that suits everyone:

- **Retained** (the storage default): every raw message, attachments included, is kept as a blob
  in the data directory. Re-parsing after a parser or chunker change, and offline rebuilds, run
  locally without contacting Gmail ([`ingest.md`](../ingest.md) §10). The cost is a
  byte-identical copy of the mailbox at rest, reachable by whoever can read the data directory or
  its backups.
- **Not retained**: only parsed text, chunks and vectors persist. The cost is that every re-parse
  is a re-fetch, at the §6 pace — a parser change that touches every message of a
  100,000-message mailbox is about eighteen hours of quota, and an offline rebuild is not
  available at all.

The connector does not override the installation's storage default; the deployment decides, and
this section is what it decides with.

**The setting applies to fetches, not to what is already stored.** It is read at each sync and
decides whether that sync keeps the bytes it fetches. Nothing purges bytes already retained when
it changes. For this connector that makes the first sync's value close to permanent: a stored
message is never fetched again (§4.1), so switching retention off later leaves every
already-retained message's raw copy in place, removed only by a hard delete of its document
followed by blob collection (§7) — or by whatever [#466](https://github.com/mgd43b/manicule/issues/466)
decides for reconciled documents. Switching it on later retains only messages fetched from then
on. Choose before the first sync, and choose off then if the installation must be able to forget.

---

## 10. Privacy

**The index is not permission-aware** ([`confluence.md`](confluence.md) §9), and for mail that
means: **whoever can search the workspace reads every message in it**, attachments and generated
answers included. Roles decide what a person may do, not what they may read.

- **Mail belongs in a workspace of its own**, holding nothing else, so that searching anything
  else never returns mail. A process serves one workspace at a time today, so in practice that
  is an installation of its own.
- **That installation serves with authentication, for as long as it holds any mail.** The
  `gmail` connector type is registered as *sensitive*. An installation refuses to serve with
  `security.auth.mode = "none"`, on any bind including loopback — another local account or
  process can reach a loopback port — and `--no-authentication` does not override it, when
  either holds ([#462](https://github.com/mgd43b/manicule/issues/462)):
  - an enabled connector instance is sensitive; or
  - **the data directory holds any document, in any state, from a source a sensitive connector
    has synced.** The first sync of a sensitive source records that durably, beside the source's
    other stored sync state. Disabling the connector or removing it from configuration does not
    lift the refusal, because neither removes a message; only deleting that source's documents
    does, and soft-deleted and purged rows count, since they still carry subjects (§7).

  Checking configuration alone was the first proposal and was rejected: it stops protecting the
  mail at the moment somebody disables the connector, and the mail is still there. A
  configuration slip should not be able to publish a mailbox to whoever can route to a port, and
  neither should tidying up a configuration.
- **Models see the text.** Embedding sends every chunk to the configured embedder, and `ask`
  sends retrieved passages to the configured generator. With a remote provider, mail leaves the
  machine in both directions. Use local models for a mail installation, or decide that this is
  acceptable knowing it happens.
- **At rest**: retained bytes (§9) — indefinitely, including for deleted mail (§7) — soft-deleted
  mail's derived text for the grace period, purged rows' subjects, and any backup of the data
  directory hold mail. Exclude the installation from backups that are kept longer or
  more widely than the mail itself would be.
- **Diagnostics are aggregate.** No log, telemetry record, status surface, run-level error or
  fixture carries an address, subject, label name, message id or body. A record about one
  message — its status, a refusal (§5) — belongs to that message's document and is as private
  as the message is; it is never copied into anything wider.

---

## 11. Not yet verified against a live mailbox

The connector will be built and tested against a synthetic Gmail
([#459](https://github.com/mgd43b/manicule/issues/459)). Where a real mailbox could still
differ, it is here — check these first when one is available:

- What label ids history records carry in practice, and whether they are a message's full current
  set. §4.2 pays a `minimal` get per candidate because a `labelRemoved` record is not the full
  set; if `messageAdded` records always carry it, that one case could skip the lookup.
- Whether saving a draft replaces its message id, and whether `in:drafts` in a list query excludes
  every draft revision.
- Whether messages labeled `CHAT` (old Hangouts conversations saved to Gmail) appear in
  `messages.list` and parse as mail. They are in scope unless excluded.
- Whether the `rfc822msgid` search link (§2) opens the right message with several accounts signed
  in; whether the percent-encoded form finds a `Message-ID` containing `+`, `%` or `#`; and
  whether `authuser` accepts a percent-encoded address containing `+`.
- The exact reason strings on 403 rate-limit answers, which the error table types by.
- How long history is actually retained for a busy mailbox.
- Whether the quota figures in §6 still hold. Google has changed them before; the page carries
  its own update date, and §6 records the one this design was written against.

---

## 12. Where this will live

| Concern | Planned module | Ticket |
|---|---|---|
| `GmailConfig`, credential resolution | `manicule/connectors/config.py` | [#458](https://github.com/mgd43b/manicule/issues/458) |
| Token refresh and caching | `manicule/connectors/credentials.py` | [#458](https://github.com/mgd43b/manicule/issues/458) |
| REST calls, pacing, error typing | `manicule/connectors/gmail_client.py` | [#459](https://github.com/mgd43b/manicule/issues/459) |
| `discover`, `fetch`, `reconcile`, watermarks | `manicule/connectors/gmail.py` | [#460](https://github.com/mgd43b/manicule/issues/460) |
| Registration, marked sensitive | `manicule/connectors/plugin.py` | [#460](https://github.com/mgd43b/manicule/issues/460) |
| `connector login` for Gmail | `manicule/connectors/` and the CLI | [#461](https://github.com/mgd43b/manicule/issues/461) |
| Refusing to serve a sensitive source unauthenticated | `manicule/app/bind.py` | [#462](https://github.com/mgd43b/manicule/issues/462) |
| The synthetic Gmail | `tests/connectors/fake_gmail.py` | [#459](https://github.com/mgd43b/manicule/issues/459) |

No Google client library is added. The four REST calls go through the HTTP stack the Confluence
client already uses, imported by the factory rather than at plugin discovery.

---

## Summary

| | manicule | Without it |
|---|---|---|
| Credential | Internal client, read-only scope, refresh token from the environment | a token that expires in a week, or a key that reads the whole domain |
| Discovery | Gmail's history log, position captured before the walk | the mailbox re-listed and re-hashed every sync |
| Change detection | constant token: a stored message is never fetched again | every label change re-downloads a message |
| Expired history | complete walk that lists ids and fetches only new mail | a full re-download, or silently lost mail |
| Scope | labels only, tested identically on every path | a query the incremental path cannot evaluate |
| Pacing | a metered budget under the per-user quota | a first sync spent hitting 429s |
| Deletions | complete-inventory reconciliation, once it runs | deleted mail searchable forever |
| Exposure | refused unless served with authentication | a mailbox open to whoever can reach the port |
