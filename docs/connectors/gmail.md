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
client_id = "1234567890-example.apps.googleusercontent.com"
client_secret_env = "GMAIL_CLIENT_SECRET"
refresh_token_env = "GMAIL_REFRESH_TOKEN"
exclude_labels = []          # SPAM, TRASH and DRAFT are always excluded (§3)
```

**The mailbox is the one the refresh token was granted by.** There is no `user` setting: with
a per-person grant, the only mailbox the token can read is its own, and a setting that could
name another would only ever produce a 403.

---

## 2. Identity and citations

- **`source_id` is Gmail's message id.** It is assigned when the message enters the mailbox and
  never changes, which is what lets change detection skip a message by id alone (§4). It is not
  the RFC 5322 `Message-ID` header, which the sender chooses, may be absent, and is shared by
  every copy of a message — a message sent to yourself is two Gmail messages with one header.
- **`uri` is a link a person can open.** The search URL for the message's `Message-ID`
  (`https://mail.google.com/mail/?authuser=<address>#search/rfc822msgid:<id>`), which survives
  Gmail interface changes better than an internal view path, and names the account so a
  browser signed in to several opens the right one. A message without a `Message-ID` falls back
  to `#all/<gmail id>`. §11 lists this as unverified against a live account.
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

**Each path tests it the same way.** A full walk and reconciliation ask `messages.list` for
`includeSpamTrash=false` with a query *derived mechanically* from the exclusion set
(`-in:drafts -label:<name> …`), never written by the operator. Incremental discovery tests the
same set against each message's label ids (§4.2). Both are tests of one set of label ids, so
they cannot disagree about what a label means.

**`exclude_labels` is a scope field.** It enters the scope fingerprint, and changing it forces a
complete replacement from a fresh inventory, as every membership-affecting field does.

---

## 4. Discovery and change detection

Gmail keeps a per-mailbox **history log**: every message added, every message deleted and every
label added or removed advances the mailbox's `historyId`, and `users.history.list` returns the
records after a given id. That is the change signal, and the watermark is a `historyId`.

### 4.1 The full walk — `discover(None)`

1. Call `users.getProfile` and keep its `historyId`. **This is captured before the walk begins**,
   and it becomes the watermark if the walk completes.
2. Page `users.messages.list` (500 ids a page, the maximum) under the derived scope query.
3. Yield each id with its version token.

**Why the position is captured first.** Google's own sync recipe takes the `historyId` of the
newest message *after* listing. Anything that arrives while a long walk is in progress — and a
first walk of a large mailbox is long (§6) — may or may not appear in the listing depending on
where the walk is. Capturing the position first means the next incremental sync starts from
before the walk began and sees every such message again. Re-seeing a message costs a version
comparison; missing one costs the message. The overlap that Confluence buys with a five-minute
reach-back ([`confluence.md`](confluence.md) §2) is structural here.

**The version token is a constant.** A Gmail message's bytes never change after it is created —
edits create new messages, and labels are not part of the bytes. So every message carries the
same token (`rfc822/1`), and a message already stored is never fetched again by id. The `1` is
there so that a change to what this connector fetches can force one complete re-fetch by
changing it.

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
| `messageAdded` | Yield the message if it is in scope. History records "typically" carry only the id and thread id, so a record without label ids costs one `messages.get?format=minimal` to learn them. |
| `labelRemoved` that brings a message back into scope | Yield it — this is a message restored from Trash or released from an excluded label. |
| `labelAdded` that takes a message out of scope | Nothing at discovery. It leaves the index through reconciliation (§7). |
| `messageDeleted` | Nothing at discovery, for the same reason. |

**Discovery cannot report a deletion**, because the connector protocol has no channel for one:
`discover` yields documents that exist, and absence is established only by a complete inventory
([`contracts.md`](../contracts.md) §3). That is why the last two rows do nothing here, and §7 is
where they take effect.

### 4.3 When the history is gone

Google keeps history "typically" for at least a week, and in rare cases only for hours. A
`startHistoryId` that is too old is answered with **404**, typed as `CursorExpiredError`, and the
run falls back to a complete walk from §4.1.

**That fallback is cheap, and the constant version token is the reason.** A complete walk lists
ids at 5 quota units per 500 and fetches only the ids not already stored, so a mailbox of
100,000 messages that missed a week of history costs about 1,000 units of listing plus the new
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
Gmail setting. A message that exceeds it is refused by name and counted, never truncated: a
truncated message would parse, cite and answer from the half that arrived.

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
embedding:

| Messages | First-sync fetch time at the ceiling |
|---|---|
| 10,000 | about 1 hour 40 minutes |
| 100,000 | about 17 hours |
| 500,000 | about 3.5 days |

After the first sync the budget is ample: a day's new mail is a few hundred fetches, history
pages cost 2 units, and a complete inventory of 100,000 ids is about 1,000 units.

**Paced, not throttled into.** The client meters its own spending against a per-minute budget,
`quota_units_per_minute` (default 5,400 — 90% of the per-user limit, leaving room for anything
else using the same project), and waits before a request would exceed it. A 429 is still
handled — honoring `Retry-After` in both its forms, and stopping the run when the requested
pause exceeds `max_retry_after_seconds`, as the Confluence client does — but it is the
exception, not the steering mechanism. A client that discovers its limit by hitting it spends a
first sync of seventeen hours being told no.

**An interrupted first sync resumes**, through durable acquisition
([`ingest.md`](../ingest.md) §13.2): committed messages are not fetched again, so stopping a long
first sync costs the minutes since the last commit, not the hours before it.

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
([`ingest.md`](../ingest.md) §11). `reconcile` pages `messages.list` under the same derived
scope query as the full walk — the same scope, by construction — through
`BatchedReconciliationConnector`, and raises rather than returning a prefix if any page fails.
It is cheap: ids only, 5 units per 500, so a 100,000-message mailbox is about 1,000 units, a
few seconds of the per-minute budget. A mail installation can afford to reconcile daily or
hourly.

**Nothing in manicule runs reconciliation today, for any connector.**
`manicule.ingest.reconcile.reconcile` exists, is tested, and has no caller in the product; no
command, service method or schedule invokes it, and `reconcile_interval_s` is read nowhere but
its own definition. `connector sync` never removes a document that left its source. The only
bulk retirement in the product is offline rebuild publication ([`ingest.md`](../ingest.md)
§10.4), which needs retained source bytes and is not a sync. [`ingest.md`](../ingest.md) §11's
"after every full sync, and on a schedule" describes the design, not the code.

For Confluence that means a deleted page lingers. For a mailbox it means deleted mail is
searchable forever. **Wiring reconciliation into sync is therefore a prerequisite of this
epic**, not an improvement to it: the Gmail connector does not ship to a release before
reconciliation runs on the schedule `reconcile_interval_s` describes.

**Once it runs, two existing behaviors matter more for mail than for anything else:**

- **The deletion ceiling.** A reconciliation that would delete more than
  `reconcile_max_delete_fraction` (default 10%) of the connector's live documents is refused and
  recorded as a proposal for confirmation (§11.1). Emptying a large folder into Trash, or adding
  a busy label to `exclude_labels`, will trip it. That is the guard working, and the proposal is
  confirmed rather than the ceiling raised.
- **Soft deletion keeps mail for the grace period.** A reconciled message is soft-deleted, and
  its chunks and vectors survive until the sweep, 30 days by default
  ([`storage.md`](../storage.md) §8.2); its retained bytes, if any, until blob collection. Removed
  from results is not removed from disk. An installation that must forget promptly sets
  `soft_delete_grace_s` accordingly.

**A message restored from Trash comes back.** Its `labelRemoved` record yields it at discovery
(§4.2), and the pipeline must restore the soft-deleted document rather than skip it as
unchanged. [#460](https://github.com/mgd43b/manicule/issues/460) tests that round trip.

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
  is a re-fetch, at the §6 ceiling — a parser change that touches every message of a
  100,000-message mailbox is seventeen hours of quota, and an offline rebuild is not available at
  all.

The connector does not override the installation's storage default; the deployment decides, and
this section is what it decides with.

---

## 10. Privacy

**The index is not permission-aware** ([`confluence.md`](confluence.md) §9), and for mail that
means: **whoever can search the workspace reads every message in it**, attachments and generated
answers included. Roles decide what a person may do, not what they may read.

- **Mail belongs in a workspace of its own**, holding nothing else, so that searching anything
  else never returns mail. A process serves one workspace at a time today, so in practice that
  is an installation of its own.
- **That installation serves with authentication.** The `gmail` connector type is registered as
  *sensitive*, and an installation with an enabled sensitive connector refuses to serve with
  `security.auth.mode = "none"`; `--no-authentication` does not override it
  ([#462](https://github.com/mgd43b/manicule/issues/462)). A configuration slip should not be
  able to publish a mailbox to whoever can route to a port.
- **Models see the text.** Embedding sends every chunk to the configured embedder, and `ask`
  sends retrieved passages to the configured generator. With a remote provider, mail leaves the
  machine in both directions. Use local models for a mail installation, or decide that this is
  acceptable knowing it happens.
- **At rest**: retained bytes (§9), soft-deleted mail for the grace period (§7), and any backup
  of the data directory hold mail. Exclude the installation from backups that are kept longer or
  more widely than the mail itself would be.
- **Diagnostics are aggregate.** No log, telemetry record, status surface, error message or
  fixture carries an address, subject, label name, message id or body.

---

## 11. Not yet verified against a live mailbox

The connector will be built and tested against a synthetic Gmail
([#459](https://github.com/mgd43b/manicule/issues/459)). Where a real mailbox could still
differ, it is here — check these first when one is available:

- Whether `messageAdded` history records carry `labelIds` in practice. §4.2 pays a `minimal` get
  when they do not; if they always do, that cost disappears.
- Whether saving a draft replaces its message id, and whether `in:drafts` in a list query excludes
  every draft revision.
- Whether messages labeled `CHAT` (old Hangouts conversations saved to Gmail) appear in
  `messages.list` and parse as mail. They are in scope unless excluded.
- Whether the `rfc822msgid` search link (§2) opens the right message with several accounts signed
  in, and what it does for a `Message-ID` containing characters that need escaping.
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
