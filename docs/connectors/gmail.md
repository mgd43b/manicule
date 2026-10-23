# Gmail ingestion

Design for the Gmail connector. Ticket [#457](https://github.com/mgd43b/manicule/issues/457),
epic [#464](https://github.com/mgd43b/manicule/issues/464).

> **Status: implemented.**
> [#458](https://github.com/mgd43b/manicule/issues/458) is the configuration (`GmailConfig`,
> `manicule/connectors/config.py`), the refresh-token credential and the mailbox check
> (`manicule/connectors/credentials.py`), the pipeline's pinned-identity check
> (`manicule/ingest/pipeline.py`) and the `gmail-credential` `doctor` check
> (`manicule/app/service.py`). [#459](https://github.com/mgd43b/manicule/issues/459) is the REST
> client, whose calls, pacing and error typing (§5, §6) are in `manicule/connectors/gmail_client.py`,
> tested against the synthetic Gmail in `tests/connectors/fake_gmail.py`.
> [#462](https://github.com/mgd43b/manicule/issues/462) is the refusal to serve a sensitive
> source without authentication (§10, `manicule/app/bind.py` and `manicule/app/sensitive.py`),
> in force before any mail can be read.
> [#460](https://github.com/mgd43b/manicule/issues/460) is the connector itself,
> `manicule/connectors/gmail.py`: scope by label id, the full and incremental walks over Gmail's
> history, the raw fetch, complete reconciliation, and the mailbox check at setup and at the start
> of every run (§2–§7). Reconciliation running in a served installation
> ([#466](https://github.com/mgd43b/manicule/issues/466)) is what makes deletions take effect.
> [#461](https://github.com/mgd43b/manicule/issues/461) is `connector login`, which mints the
> refresh token (§1.3, `manicule/connectors/gmail_login.py`). §12 says which module implements
> which part; when a ticket changes that, it updates this page.

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
browser (`manicule connector login`, §1.3) and then has to survive every
restart of a server that may run where no browser exists. Holding it only in memory would mean
a person signs in again after every restart, deploy and node failure.

So the refresh token is a long-lived secret supplied by the deployment's own secret store, read
from the environment exactly as the Confluence API token is: an explicit value in configuration
wins, the named environment variable fills it otherwise, and printing the configuration never
shows it. A credential in a configuration file is a credential in version control eventually,
so the documented path is the environment variable.

**It is resolved in the factory, before the connector exists** (implemented,
`resolve_gmail_credentials`). A missing client secret or refresh token is a `ConfigError`
naming the setting and the environment variable, never a value. The factory is synchronous, so
it does no network I/O: it validates, resolves the secrets and constructs the credential
object, and returns the connector. The first network call is the connector's `setup()`, which the
container awaits before it hands the connector to anyone (§1.4).

**Whether a refresh succeeds is checked by `doctor`** (the `gmail-credential` check,
[`surfaces.md`](../surfaces.md)), so a revoked grant is visible before a sync rather than at its
first request. The connector makes the same check in its setup and at the start of each run,
where it refuses to begin.

**Access tokens are minted from it and cached until shortly before `expires_in`** (implemented,
`RefreshTokenCredential`: the cache expires 60 seconds early). The REST client
([#459](https://github.com/mgd43b/manicule/issues/459)) asks it for `access_token()`, asks again
with `refresh=True` after a 401 and retries once, and raises the credential's `rejected()` on a
second 401, which is a `TokenRejectedError`; `send(attempt)` does the same refresh-and-resend for
a caller that has no retry of its own. Google answering the
refresh with `invalid_grant` — the grant was revoked, the account's password changed under a
policy that revokes tokens, or the token aged out — raises `GrantRevokedError`, and its renewal
text names `manicule connector login <name>`. The token endpoint is a constant in the module,
not a setting: a setting for it would let one line of configuration send the client secret and
the refresh token to any host.

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

Then, as the account whose mail is indexed, with the source configured (below) and
`GMAIL_CLIENT_SECRET` set in the environment of the shell that runs `login`:

```console
$ manicule connector login mail --token-file ./gmail-refresh-token   # opens a browser; see below
$ manicule doctor                        # the gmail-credential check should pass (implemented)
$ manicule connector sync mail           # implemented (#460)
```

```toml
[connectors.mail]
type = "gmail"
enabled = true

[connectors.mail.options]
account = "someone@example.org"   # required: the primary address of the one mailbox this
                                  # instance indexes (§1.4); stored lowercase
client_id = "1234567890-example.apps.googleusercontent.com"   # required; not a secret
client_secret_env = "GMAIL_CLIENT_SECRET"    # the default; client_secret = "..." wins if set
refresh_token_env = "GMAIL_REFRESH_TOKEN"    # the default; refresh_token = "..." wins if set
exclude_labels = []          # label names; SPAM, TRASH and DRAFT are always excluded and
                             # listing one is refused (§3)
quota_units_per_minute = 5400     # the default; accepted range 60 to 6000 (§6)
max_retry_after_seconds = 120     # the default; longest Retry-After honored
max_retries = 5                   # the default; attempts after the first
request_timeout_seconds = 30      # the default
```

Any other key is refused at startup, which is how the settings that are deliberately absent
(a `query`, an `include_spam_trash`, a `user`, a token endpoint) stay absent.

#### Minting the token: `manicule connector login <name>`

`login` runs Google's installed-app flow for the instance, on the machine you are sitting at,
and needs no running server: the token is handed to you, not to a process.

1. It listens on `127.0.0.1`, on a port the operating system picks for this one sign-in, and
   opens your browser at Google's sign-in page. The address is also printed to standard error,
   for a machine where no browser could be started.
2. You choose the account the instance indexes. manicule never asks for, sees or handles a
   password or a second-factor code; the browser does the sign-in.
3. Google redirects the browser to that port with an authorization code. The request is PKCE
   (`S256`), and the redirect is refused unless it carries the `state` the command chose. The
   only scope requested is `gmail.readonly`, with `access_type=offline` and `prompt=consent`, so
   Google issues a refresh token on every run and not only the first. A token Google reports as
   carrying any other scope is discarded.
4. The command exchanges the code for the refresh token, then checks it exactly as `doctor`
   does: one refresh and one `users.getProfile`. **A token whose mailbox is not the instance's
   `account` is refused and written nowhere** (§1.4). Neither address is printed; an alias of
   the account counts as another mailbox. The grant that was refused stays on that account until
   it is removed under the account's third-party access.
5. Only then is the token written, to where you said:

| You pass | The token goes to |
|---|---|
| `--token-file PATH` | A new file created with mode `0600`, containing the token and a newline. An existing path is refused; `--overwrite-token-file` replaces it by writing a new `0600` file beside it and renaming it into place. |
| nothing | Standard output, **only when it is not a terminal**, as in `manicule connector login mail \| <store-secret>`. On a terminal the command refuses before opening the browser. |

Where the token goes is checked *before* the browser opens. It is never written to the
configuration file, to a terminal, to standard error or to a log, and the result the command
prints names where it went and never what. With the token on standard output the result moves
to standard error, so the pipe receives the token alone; `--json` is refused in that case.
`--timeout SECONDS` bounds the wait for the sign-in (five minutes by default).

Take the token to the secret store your deployment already uses, and expose it to the server as
the variable `refresh_token_env` names. `<store-secret>` below stands for whatever command that
store provides:

```console
$ manicule connector login mail | <store-secret> --name gmail-refresh-token
$ # or, through a file you remove once it is stored:
$ manicule connector login mail --token-file ./gmail-refresh-token
$ <store-secret> --name gmail-refresh-token < ./gmail-refresh-token && rm ./gmail-refresh-token
```

then give the server `GMAIL_REFRESH_TOKEN` from that store, in its environment. Running `login`
again mints a replacement (§1.2): put the new token in the store and restart the server.

### 1.4 One instance, one mailbox, pinned

**A token can only read the mailbox that granted it, so the token decides which mailbox is
indexed — and a token is replaced routinely.** Renewing a revoked grant, or re-running
`connector login` from a browser signed in to a different account, would otherwise switch an
instance to another mailbox with every configured field unchanged. The pipeline keys stored
sync state on the connector's name and scope fingerprint and reuses a watermark whenever the
fingerprint matches, so the new mailbox would resume from the old one's history position and
unfinished inventory, while the old mailbox's mail stayed indexed beside it.

So the mailbox is configuration, not a side effect of the credential:

- **`account` is required**, and it is the address the instance indexes. It is stored
  lowercase, because Google compares addresses without regard to case and two spellings of one
  address must not be two scopes. A malformed value is refused without echoing it.
- **It is checked against `users.getProfile`'s `emailAddress`.** Implemented, as one rule
  (`mailbox_mismatch`, `credentials.py`) applied in three places: `doctor`'s `gmail-credential`
  check (`RefreshTokenCredential.confirm_mailbox`, which does its own refresh and `getProfile`),
  the connector's `setup()`, and the start of every discovery and every reconciliation. Each
  raises `MailboxMismatchError` for a token that belongs to another mailbox, without printing
  either address, since run-level errors are aggregate (§10), and the connector does so before a
  message or a label is read: the only call made is the `getProfile` it needed for its
  `historyId` anyway. `connector login` (§1.3) makes the same check before it writes the token it
  has just minted. The factory cannot do it, because it is synchronous and does no network
  I/O (§1.2), which is why it is `setup()`'s. A run that resumes at acquisition, after a fetch
  failed once enumeration had finished, never passes through discovery, so `fetch` repeats the
  check when the last one is more than a minute old: one `getProfile` (1 unit) at most a minute,
  against 60 for a raw fetch, and the refusal comes before the message is read. `getProfile`
  reports the *primary* address, so configure the primary address: an alias of the account is
  refused as a mismatch rather than followed.
- **`account` is part of the scope fingerprint** as well (`GmailConfig.scope_fingerprint`, which
  also hashes the source scope, the resolved label ids and the representation version), so
  stored state from one mailbox is never resumed for another even if the check were bypassed.
  The fingerprint alone does not protect the first mailbox's mail, though: a different account
  makes the stored watermark invisible rather than wrong, and the run would begin a complete
  walk of someone else's mailbox beside the first person's. That is what the next bullet is for.
- **Editing `account` on an instance that has synced is refused.** This is implemented in the
  pipeline, not in the fingerprint. A connector may expose a `pinned_identity` — for Gmail, a
  digest of the account, never the address — and before a run reads a watermark or enumerates
  anything, the pipeline compares it with the one it recorded for that source and raises
  `PinnedIdentityError` if they differ and the source has *synced*. A reconciliation pass (§7) and
  the confirmation of one are held to the same pin before they read anything, because a replaced
  credential would otherwise diff one account's inventory against another's documents. "Synced"
  means the source still holds documents, or has a run that began and did not settle; an instance that has never
  synced may change its account freely, and deleting its documents releases the pin. The
  fingerprint cannot do this job because it cannot tell an edit to the account (which must not
  keep the source) from an edit to the scope (which does).
- **Indexing a different mailbox is a different instance**, under a new connector name and
  therefore a new source identity. The old instance's documents do not leave by themselves —
  reconciliation (§7) compares only the documents filed under an instance's own name, and it
  never runs for an instance that is disabled or no longer configured — so they are deleted
  explicitly before or after the switch.
  The refusal above says so, rather than treating the edit as a scope change: a scope change
  keeps the source and re-walks it, leaving removal to reconciliation, and what would be left
  behind here is someone else's mail.

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
  show it without them. A message without a `Message-ID` falls back to `#all/<gmail id>`. Discovery
  yields the fallback, because listing returns ids and nothing else. The fetch reads the header, and
  the search link travels as the provenance record's canonical address, which the pipeline stores
  as the document's address; the fetched envelope keeps discovery's address, because retained
  bytes are reusable only while the envelope names the address of the record asking for them.
  §11 lists the link as unverified against a
  live account.
- **Title** is `Subject`, which the parser also makes the first element of the heading path. A
  listing does not carry it, so it reaches the document at fetch, in the provenance record
  (`PROVENANCE_KEY`) beside the link, the message id and the arrival time. A subject that cannot
  be recorded (a control character) leaves the message indexed and cited by its link, with a
  record that says so in words that do not quote it.
- **Metadata** carries the thread id (`gmail_thread_id`) and `internalDate` as an ISO 8601
  timestamp (`internal_date`). Both belong to the message's own document. Neither appears on any
  status surface: status reports aggregate counts only, never an id, address, subject or label.

---

## 3. Scope

**Everything in the mailbox except SPAM, TRASH and DRAFT, minus any labels in
`exclude_labels`.** There is no inclusion list and no free-form query.

**SPAM, TRASH and DRAFT are always excluded, and not configurable.** A mail index that contains
spam is a phishing corpus that answers questions; trash is mail its owner removed. Drafts are
unsent and rewritten continuously while being edited, so an indexed draft is a stale copy within
minutes and a sent message is indexed separately once it is sent.

**`exclude_labels` takes label names**, resolved to label ids once per run through
`users.labels.list` (the listing is the REST client's, [#459](https://github.com/mgd43b/manicule/issues/459);
the rules below are implemented in `manicule/connectors/config.py`):

- An unknown name is a `ConfigError`, not an empty exclusion — a typo that silently excluded
  nothing would index precisely the mail the operator meant to keep out. Matching is exact, so a
  nested label is written with its slashes (`Parent/Child`). The error names only the name the
  operator configured and never lists the labels the mailbox has, since label names come from
  the mail and run-level errors carry none (§10). The resolution is `resolve_label_ids`, a pure
  function over whatever `labels.list` returned.
- **Listing SPAM, TRASH or DRAFT is refused at startup.** They are always excluded, so a setting
  that named one would do nothing while reading like a rule in force.
- Duplicate names are dropped, and an empty name is refused.

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
reconciliation against the new inventory removes them. A completed full walk is a full sync, so
the pass follows it (§7), and a narrowing large enough to trip the deletion ceiling waits for
`manicule connector reconcile NAME --confirm` (§7).

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
message is yielded with the new token and fetched. An unfinished run recorded under the
old version is not resumed either: a run is claimed under the fingerprint, so a bump mid-run
starts a fresh complete walk rather than finishing the old one (tested through the pipeline in
`tests/connectors/test_gmail_pipeline.py`).

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

**The history position is also the watermark's value, and what makes it valid is in its
metadata.** The watermark is `Watermark(value=<historyId>)`, readable only after a walk
completed (a walk abandoned part-way advances nothing), and it records the scope fingerprint and
the pinned identity it was written under. The pipeline already reuses a stored position only under
the fingerprint it was claimed with; the connector checks both again itself, because the
fingerprint it hands the pipeline is a synchronous property over the label ids resolved at setup
or at the start of the last run (resolving needs `labels.list`, and the pipeline reads the
fingerprint before discovery begins), so a label deleted and recreated under the same name
between runs would otherwise be read stale. The cost of the stale read is one complete walk, not
a wrong scope.

**A candidate's lookup is remembered for the length of the walk.** The ids a `history.list` walk
has already looked up are held in a set so that a message named by many records costs one
`format=minimal` get (§4.2). It holds ids the log named, not the mailbox's inventory, and it is
gone when the walk is.

### 4.3 When the history is gone

Google keeps history "typically" for at least a week, and in rare cases only for hours. A
`startHistoryId` that is too old is answered with **404**, typed as `CursorExpiredError`, and the
run falls back to a complete walk from §4.1.

**The fallback is reported as what it is.** The pipeline decides whether a sync was a full
enumeration before discovery runs, from the watermark it hands over, so a walk that began as an
incremental one would otherwise be treated as a delta and skip the reconciliation that follows a
complete walk (§7). The connector says, once discovery has ended, that it walked everything
(`walked_full_inventory`), and the pipeline reads that as it reads the other signal. The same
holds when a stored position is set aside because the scope or the representation changed.

**That fallback is cheap, and the constant version token is the reason.** A complete walk lists
ids at 5 quota units per request of up to 500 ids, and fetches only the ids not already stored
(the retained bytes of a message are found in whichever earlier promoted run fetched it, not only
in the newest manifest, which after an incremental sync names only that sync's delta),
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
proves nothing about anything else in the mailbox. The run records it and the next complete walk
no longer lists the message.

**Size** is the pipeline's existing limit on a fetched body, `ingest.max_fetch_bytes` (256 MiB by
default, well above the 25 MB Gmail accepts), not a Gmail setting. A message that exceeds it is
refused, never truncated: a truncated message would parse, cite and answer from the half that
arrived. The refusal is the pipeline's own for any source, and it is an *omission*, not a
document: the message gets no row, the run counts it as an omission of type `fetch_failed`
(an aggregate; no id, subject or label reaches a status surface), and the run reports that a
retry is owed. Under the default `ingest.snapshot_promotion_policy`, a run with an omission is not
complete and advances no watermark, so the next sync is a complete walk again; under
`allow_omissions` the rest of the mailbox moves on and the message is retried on every sync until
the limit is raised. Either way the cost is a 60-unit fetch per attempt (§6). An earlier draft of
this page said the refusal was recorded on the message's own document; the pipeline does not do
that for any source, and changing it is a change to the acquisition journal, not to this
connector.

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
else using the same project), and waits before a request would exceed it. The budget is a
rolling minute, not a bucket that resets on the clock: the units spent in *any* sixty seconds
stay within it, and every attempt is charged, a retry included. A budget below 60 is refused at
construction, because no raw fetch could ever fit in it. A 429 is still
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

**Errors**, as the client types them ([#459](https://github.com/mgd43b/manicule/issues/459)).
A 403 is a rate limit when its `reason` is `rateLimitExceeded` or `userRateLimitExceeded`,
compared without regard to case or underscores and read from both `error.errors[]` and
`error.details[]` (which of the two Google uses is not yet verified, §11). A 404 is typed by the
call that received it, so a 404 from `getProfile` or `labels.list` is an ordinary unusable
answer rather than a deleted message. A second 401 raises whatever the credential
builds for it, which carries its own renewal sentence. Every message names the operation and a status, never an id,
an address or a label name (§10), and the client redacts Gmail's request URLs from `httpx`'s
own INFO log, which would otherwise print each message id (other clients' records are left
alone):

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

**Reconciliation runs in a served manicule, for every enabled connector.** The scheduler calls
`manicule.ingest.reconcile.reconcile` on `ingest.reconcile_interval_s` (default weekly) or on the
instance's own `reconcile_interval_s`, and a completed full sync is followed by a pass whether or
not the interval has elapsed ([`ingest.md`](../ingest.md) §11, delivered by
[#466](https://github.com/mgd43b/manicule/issues/466)). A mail instance sets its own interval,
`reconcile_interval_s = 86400` for daily or `3600` for hourly, and that interval is how long
deleted mail can stay searchable, plus at most one scheduler poll (an hour). An incremental sync
still cannot see a deletion; the pass is what removes it. `manicule connector reconcile NAME` runs
a pass now, `--dry-run` previews one, and `--confirm` or `--discard` answers a proposal the
deletion ceiling refused. `connector list` and `doctor` report a pass that is overdue and a
proposal that is waiting. Everything a pass returns is counts: never a message id, subject or
address.

That meets the prerequisite this section used to state: reconciliation runs on a schedule before
the Gmail connector ships. The schedule lives in `manicule serve`, so an installation that is not
served has no process to run it, and only the command above reconciles it.

A pass and a held deletion proposal are fenced to the scope they were built under
(`reconciliation_scope`, which carries the scope fingerprint), so a proposal made while a label
was excluded cannot be confirmed after the exclusion is removed, which would delete messages that
are in scope again.

**Two behaviors matter more for mail than for anything else:**

- **The deletion ceiling.** A reconciliation that would delete more than
  `reconcile_max_delete_fraction` (default 10%) of the connector's live documents is refused and
  recorded as a proposal for confirmation (§11.1). Emptying a large folder into Trash, or adding
  a busy label to `exclude_labels`, will trip it. That is the guard working, and the proposal is
  confirmed with `manicule connector reconcile NAME --confirm` rather than the ceiling raised.
  `--discard` keeps the messages and restarts the interval, for the case where the mailbox was
  what looked wrong.
- **Removed from results is removed from disk late, and nothing here erases a message on
  demand.** A reconciled message is soft-deleted. When the grace period ends
  (`soft_delete_grace_s`, 30 days by default, [`storage.md`](../storage.md) §8.2), the sweep
  removes its chunks and vectors — the derived text — and releases the document's reference to
  the retained bytes, keeping the document row, which still carries the subject as its title, so
  that a citation can explain itself. The raw bytes are not yet gone: the acquisition journal that
  fetched them names them too, for 30 days after the last run that listed the message settled,
  and the next durable sync cleans that up (`storage.md` §7). The scheduled blob collection
  deletes the file once both have let go. The two clocks run side by side, so in a plain
  installation **a message the source deleted has its raw bytes removed about one grace period
  after the deletion** — 30 days with the defaults, plus the wait for the next sync and the next
  sweep, and longer if no sync runs to clean the journal. It is longer still when something else
  holds the run that listed the message: unfinished records in it, a rebuild generation or
  snapshot that names it, or a source-inventory replacement still in progress (`storage.md` §7).
  A message deleted while the source still lists it is renewed in the journal by every run that
  lists it, and its file stays for as long as that continues. Lowering `soft_delete_grace_s`
  shortens the wait; the journal retention is not configurable. Until the file is deleted it is
  still in the data directory and in any backup of it.

  Erasing one message at once is manual: `manicule document delete --hard` removes the row, its
  versions and its chunks, and the raw bytes go once no acquisition record still references them
  and the next blob collection runs. For an installation that must be able to forget, the honest
  configuration is retention off **from the first sync** (§9), which leaves no raw copy to erase —
  at the re-fetch cost §9 states. Turning it off later does not remove copies already kept, though
  a deleted message's copy goes on the schedule above. Erasure at the moment the source deletes a
  message is not designed here.

**A message restored from Trash comes back**, unless it still carries another excluded label.
Its `labelRemoved` record makes it a candidate and its current labels decide (§4.2). Once it is
yielded, the pipeline restores the soft-deleted document rather than skipping it as unchanged: a
soft-deleted row is invisible to change detection, so the message is fetched again and written
over the old row, clearing the deletion. `tests/connectors/test_gmail_pipeline.py` drives that
round trip (trashed, reconciled away, restored, indexed again), and the same walk with another
excluded label still on the message yields nothing. A restore costs one raw fetch, because the
message was deleted from the index and not merely hidden.

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
already-retained message's raw copy in place, removed by a hard delete of its document followed
by blob collection, or — for a message the source deletes — by the purge and the journal cleanup
(§7). Switching it on later retains only messages fetched from
then on. Choose before the first sync, and choose off then if the installation must be able to forget.

---

## 10. Privacy

**The index is not permission-aware** ([`confluence.md`](confluence.md) §9), and for mail that
means: **whoever can search the workspace reads every message in it**, attachments and generated
answers included. Roles decide what a person may do, not what they may read.

- **Mail belongs in a workspace of its own**, holding nothing else, so that searching anything
  else never returns mail. A process serves one workspace at a time today, so in practice that
  is an installation of its own.
- **That installation serves with authentication, for as long as it holds any mail.** The
  `gmail` connector type is registered as *sensitive* (`sensitive=True` beside its
  `config_model`; `ComponentRegistry.sensitive_names`). An installation refuses to serve with
  `security.auth.mode = "none"` when either holds:
  - an enabled connector instance is of a sensitive type; or
  - **the data directory holds any document, in any state, from a source a sensitive connector
    has synced.** The first sync of a sensitive source records that durably, in
    `connectors.sensitive_synced_at` on the source's row, beside its stored sync state — before
    the sync enumerates anything, so a first sync that fails halfway still leaves it. Nothing
    clears it. The check asks whether any `documents` row, in any workspace of the data
    directory, still carries a marked source's name, with no filter on status or `deleted_at`:
    soft-deleted rows count, and so do purged ones, which keep their title — for mail, the
    subject (§7).

  The refusal applies in the two places that make a listener, `resolve_bind` (the MCP and HTTP
  socket servers) and `build_app` (an application started by something other than
  `manicule serve`), on **every address, loopback included** — another local account or
  process can reach a loopback port — and **`--no-authentication` does not override it**:
  there is no flag, setting or environment variable that does, and the error says so when the
  flag was passed. stdio opens no socket and is not refused. The message names each source by
  instance name and says what to do.

  Disabling the connector or removing it from configuration does not lift the refusal, because
  neither removes a message. Only deleting the source's documents does:
  `manicule document delete-source <name> --yes` hard-deletes every document of that source in
  the workspace, trashed and purged rows included, and returns the count. It is a write, so it
  runs in the serving process, which means serving *with* authentication first; it is not
  offered over MCP or HTTP. `manicule document delete <id> --hard` removes one live document
  at a time and cannot reach the trashed or purged ones, which is why the bulk command exists.
  It removes the `documents` rows (and what cascades from them: chunks, versions, tags,
  relations); the retained source bytes and acquisition journal rows are released by the journal
  cleanup (§7), and the vectors of the removed chunks by the vector sweep, as for any hard
  delete.
  Checking configuration alone was the first proposal and was rejected: it stops protecting the
  mail at the moment somebody disables the connector, and the mail is still there.

  **The mark follows the documents.** `ingest_raw`, which every document write passes through,
  marks a sensitive instance's source too, so a forced sync or an archive import cannot leave
  mail unmarked. `manicule export` writes `"sensitive": true` on the entries of a marked source
  (and only then, so an archive with no mail is unchanged), and `manicule import` marks that
  source before writing its documents, so mail carried to another installation arrives with the
  refusal attached rather than leaving it behind with the configuration.

  **A database that cannot be read refuses; it does not pass.** Only "no database yet" and a
  database from before the mark existed are treated as holding nothing. A lock that outlasts the
  read timeout, or a corrupt file, refuses when authentication is off and is ignored when it is
  on. `manicule doctor`'s `transport` check reports the same facts as a failure, without
  serving, with the instance names in its `sensitive_sources` fact (and nothing from inside a
  message).
- **Models see the text.** Embedding sends every chunk to the configured embedder, and `ask`
  sends retrieved passages to the configured generator. With a remote provider, mail leaves the
  machine in both directions. Use local models for a mail installation, or decide that this is
  acceptable knowing it happens.
- **At rest**: retained bytes (§9) — for as long as the message is indexed, and for a deleted
  message until the purge and the journal cleanup have both let go (§7) — soft-deleted
  mail's derived text for the grace period, purged rows' subjects, and any backup of the data
  directory hold mail. Exclude the installation from backups that are kept longer or
  more widely than the mail itself would be.
- **Diagnostics are aggregate.** No log, telemetry record, status surface, run-level error or
  fixture carries an address, subject, label name, message id or body. A record about one
  message — its status, a refusal (§5) — belongs to that message's document and is as private
  as the message is; it is never copied into anything wider.

---

## 11. Not yet verified against a live mailbox

The connector is built and tested against a synthetic Gmail
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

## 12. Where this lives

| Concern | Module | Ticket | Status |
|---|---|---|---|
| `GmailConfig`, credential resolution, label-name resolution, scope fingerprint, pinned identity | `manicule/connectors/config.py` | [#458](https://github.com/mgd43b/manicule/issues/458) | Implemented |
| Token refresh and caching, 401 retry, mailbox check, credential errors | `manicule/connectors/credentials.py`, `manicule/connectors/errors.py` | [#458](https://github.com/mgd43b/manicule/issues/458) | Implemented |
| Holding a synced source to its pinned identity | `manicule/ingest/pipeline.py` | [#458](https://github.com/mgd43b/manicule/issues/458) | Implemented |
| The `gmail-credential` `doctor` check | `manicule/app/service.py` | [#458](https://github.com/mgd43b/manicule/issues/458) | Implemented |
| Registration and the factory (resolves the credential, returns the connector), marked `sensitive=True` | `manicule/connectors/plugin.py` | [#458](https://github.com/mgd43b/manicule/issues/458), [#460](https://github.com/mgd43b/manicule/issues/460), [#462](https://github.com/mgd43b/manicule/issues/462) | Implemented |
| REST calls, pacing, error typing | `manicule/connectors/gmail_client.py` | [#459](https://github.com/mgd43b/manicule/issues/459) | Implemented |
| `discover`, `fetch`, `reconcile`, watermarks, the mailbox check at setup and run start | `manicule/connectors/gmail.py` | [#460](https://github.com/mgd43b/manicule/issues/460) | Implemented |
| `connector login` for Gmail | `manicule/connectors/gmail_login.py`, `manicule/app/service.py` (`connector_login_gmail`) and the CLI | [#461](https://github.com/mgd43b/manicule/issues/461) | Implemented |
| Refusing to serve a sensitive source unauthenticated; the durable mark; `document delete-source` | `manicule/app/bind.py`, `manicule/app/sensitive.py`, `manicule/ingest/pipeline.py` | [#462](https://github.com/mgd43b/manicule/issues/462) | Implemented |
| Reconciliation on a schedule and after a full walk | `manicule/ingest/reconcile.py`, `manicule/connectors/gmail.py` (the inventory) | [#466](https://github.com/mgd43b/manicule/issues/466), [#460](https://github.com/mgd43b/manicule/issues/460) | Implemented |
| The synthetic Gmail | `tests/connectors/fake_gmail.py` | [#459](https://github.com/mgd43b/manicule/issues/459) | Implemented |

No Google client library is added. The six REST calls (`getProfile`, `labels.list`,
`messages.list`, `messages.get` in `raw` and `minimal`, and `history.list`) go through the HTTP
stack the Confluence client already uses, imported by the factory rather than at plugin
discovery: `gmail_client.py` imports it inside `GmailClient.setup()`, so importing the module
costs no HTTP stack either.

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
| Deletions | complete-inventory reconciliation, on a schedule and after every full walk | deleted mail searchable forever |
| Exposure | refused unless served with authentication | a mailbox open to whoever can reach the port |
