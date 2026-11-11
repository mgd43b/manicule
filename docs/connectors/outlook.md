# Outlook ingestion

Design for the Outlook connector: one Microsoft 365 mailbox and its calendar, read-only, through
Microsoft Graph.

> **Status: implemented, and never run against a live tenant.** Everything here is built and
> tested against a synthetic Graph and Entra (`tests/connectors/fake_graph.py`) that reproduces the
> traps the documentation describes. §11 lists which of Microsoft's behaviors were read in its
> documentation and which could not be exercised without a tenant. Read it before pointing this at
> a real mailbox, and run `manicule doctor` first.

A mailbox is the most sensitive source manicule has been pointed at, and the obvious
implementation gets four things wrong quietly: it re-downloads the whole mailbox to find what
changed, it never notices that a message was deleted, it indexes the Deleted Items and Junk folders
the owner already threw away, and it serves the result to whoever can reach the port. This
connector keeps one delta link per folder, removes what left the mailbox, never reads the folders
the owner discarded, and refuses to run in an installation that serves without authentication.

The connector ends at `RawDocument`. A message is fetched as its whole MIME bytes and handed to the
existing `message/rfc822` parser ([`parsing.md`](../parsing.md) §10); an event is rendered to
`text/calendar` and handed to the existing calendar parser. There is no Outlook-specific parsing.

It is the second mail connector after [Gmail](gmail.md), and it keeps Gmail's decisions wherever
the source allows: a constant version token, deletion only by complete reconciliation, a pinned
mailbox, errors that name no mail. §0 says where it differs and why.

---

## 0. What is different from Gmail

| | Gmail | Outlook |
|---|---|---|
| Change signal | one `historyId` for the mailbox | **one delta link per folder** (Graph's message delta is per folder) |
| Scope | label ids | **folder ids**, with subfolders |
| Sign-in | a client secret and a refresh token | **a public client**: no secret anywhere, one refresh token |
| Refresh token | stable | **rotated by Microsoft on every refresh**, and the old one is meant to be discarded |
| Pacing | a metered budget of quota units | **four concurrent requests** and a `Retry-After` |
| Ids | a mailbox-wide id | an id that **changes when an item moves** unless immutable ids are asked for |
| Calendar | not indexed | **indexed**, by listing a date window afresh each sync |

---

## 1. Auth

**A delegated, read-only sign-in as a public client.** An Entra app registration of the *Mobile and
desktop application* kind, the person signing in once with a browser, a refresh token, and these
scopes and no others: `offline_access` (without it Microsoft issues no refresh token), `User.Read`
(the mailbox check), `Mail.Read` when `mail` is on and `Calendars.Read` when `calendar` is on.
None can send, modify or delete anything, and a token Microsoft reports as carrying any scope
beyond those (and the OpenID ones it adds itself) is discarded at login (§1.3).

### 1.1 The app registration

This connector cannot create one for you, and it will not borrow another application's client id.
Someone with the right in the directory registers an application, and its two ids are the only
things that go in configuration.

| Setting in the Entra admin center | Value |
|---|---|
| Supported account types | *Accounts in this organizational directory only* (single tenant) |
| Authentication, Platform | *Mobile and desktop applications*, redirect URI `http://localhost` |
| Authentication, Allow public client flows | **Yes** |
| API permissions (delegated) | `Mail.Read`, `Calendars.Read`, `User.Read`, `offline_access` |
| Admin consent | whatever the directory's policy needs for those |

Entra ignores the port of a `localhost` redirect, so one registered address serves the ephemeral
port the login opens. It cannot register `http://127.0.0.1` through the portal (only through the
application manifest), which is why the login names `localhost` by default and
`login_redirect_host = "127.0.0.1"` is the setting for a manifest that registers the other.

**An employer's directory may refuse any of this.** User consent to `Mail.Read` may be disabled,
a conditional-access policy may block a public client or a sign-in from an unmanaged machine, and
an administrator may have to create the registration. Those are the directory's decisions and
`connector login` reports them by the code Microsoft returned and nothing else.

### 1.2 The token, and where it goes

| What | Where it comes from | How a request uses it |
|---|---|---|
| Tenant id, client id | configuration (neither is a secret) | the token endpoint's path, the form body |
| Refresh token | `refresh_token`, else `refresh_token_file`, else `$refresh_token_env` | exchanged for an access token |
| Access token | minted by manicule, held in memory | `Authorization: Bearer <token>` |

**Microsoft rotates refresh tokens.** Each refresh may return a replacement, and Microsoft's own
guidance is to adopt it and discard the old one. The credential adopts it in memory at once. With
`refresh_token_file` set, the replacement is also written back to that file (a new file readable
only by its owner, renamed into place, so a failure part way never leaves half a token where the
working one was). The old token is not revoked when it is used, so a replacement that cannot be
written costs the token's sliding lifetime and nothing else: the failure is logged by its type,
never with the token or the path, and the run goes on.

So the recommended arrangement is a file, `connector login --token-file PATH`, and the same path as
`refresh_token_file`. A token supplied only through the environment works, and is never rotated on
disk: it keeps working until Microsoft expires it, which is a lifetime this connector cannot
observe (§11).

**`doctor` never writes a token.** Its probe holds a rotated token in memory and drops it, because
a diagnostic that rewrote the stored secret would race the server for it.

### 1.3 Configuration and `connector login`

```toml
[connectors.work]
type = "outlook"

[connectors.work.options]
account = "pat@example.org"                          # the sign-in name or the primary address
tenant_id = "00000000-0000-0000-0000-000000000000"   # the directory's GUID
client_id = "11111111-1111-1111-1111-111111111111"   # the app registration's id
refresh_token_file = "/run/secrets/outlook-token"
exclude_folders = ["Archive/2019"]
calendar_days_back = 90
calendar_days_forward = 180
```

| Option | Meaning |
|---|---|
| `account` | required; compared with both `userPrincipalName` and `mail` (§1.4) |
| `tenant_id` | required; a **GUID**. `common`, `organizations` and `consumers` are refused by name |
| `client_id` | required; a GUID |
| `refresh_token`, `refresh_token_env`, `refresh_token_file` | where the token comes from (§1.2) |
| `mail`, `calendar` | what to index; at least one |
| `exclude_folders` | folder **paths** with slashes, `Archive/2019`, excluded with everything below them |
| `calendar_days_back`, `calendar_days_forward` | the window around now (§7.1) |
| `max_concurrent_requests` | at most 4, which is Outlook's own cap |
| `max_retries`, `max_retry_after_seconds`, `request_timeout_seconds` | retry policy (§8) |
| `login_redirect_host` | `localhost` (default) or `127.0.0.1` (§1.1) |

**What is deliberately not a setting**, so that its absence reads as a decision, and unknown keys
are a startup error: a client secret (there is none), an authority or token URL, a Graph URL, a
free-form `$filter` or `$search`, and any way to bring Junk, Deleted Items, Drafts, Outbox or Sync
Issues back into scope.

**The token endpoint is a constant built from the tenant GUID.** `https://login.microsoftonline.com/<guid>/oauth2/v2.0/token`.
The GUID is validated as a GUID because it becomes part of an address, so no setting can send a
refresh token to another host. Public cloud only: the national clouds are different hosts and are
not supported rather than guessed at.

`manicule connector login NAME` mints the token. It opens a browser at Microsoft's authorization
endpoint, listens on a port it opened on `127.0.0.1` for exactly one redirect, and exchanges the
code with PKCE (`S256`). It shares its listener and its token sinks with the Gmail login
(`gmail_login.py`), so the parts that must not differ do not: `state` is checked in constant time
before the code is looked at, a `Host` other than the one the socket opened is refused, the token
reaches only a file created `0600` or a standard output that is not a terminal, and the
destination is checked before the browser opens. Then, in this order and no other:

1. the token's scopes are checked against the instance's (§1.1); anything wider, or a missing
   `Mail.Read` the person declined, is discarded;
2. the token is used once to read `/me`, and a token for another mailbox is discarded and written
   nowhere (§1.4);
3. only then is it written.

No server has to be running. `--timeout` bounds the wait.

### 1.4 One instance, one mailbox, pinned

**A token can only read the mailbox that granted it, so the token decides which mailbox is
indexed, and a token is replaced routinely.** Renewing a revoked grant from a browser signed in to
another account would otherwise switch an instance to another mailbox with every configured field
unchanged, resuming from the old one's delta links while the old one's mail stayed indexed.

So the mailbox is configuration, and it is checked against `GET /me`:

- **`account` matches either identity.** In Microsoft 365 the sign-in name
  (`userPrincipalName`) and the primary address (`mail`) often differ, so the account matches if it
  equals either one, ignoring case. An alias is not accepted. This is a choice: the alternative,
  pinning the directory object id, is stable across renames, but it is not something an operator
  can read off their own mailbox, and the tenant is already pinned by the token endpoint.
- **The pinned identity is a digest of the tenant and the account**
  (`OutlookConfig.pinned_identity`), stored beside the instance's run metadata, which status
  surfaces read. The pipeline refuses an instance that has synced and is pointed at another.
- **It is checked in three places by one rule** (`identity_mismatch`): `doctor`'s
  `outlook-credential` check, the connector's `setup()`, and the start of every discovery and
  reconciliation. A fetch that resumes without discovery re-checks if the last confirmation is more
  than a minute old. Each raises `MailboxMismatchError` without printing either address.

---

## 2. What a document is

| | |
|---|---|
| Source id, message | `msg:` and Graph's **immutable id** |
| Source id, event | `evt:` and Graph's immutable id |
| Title | the message's `Subject`, or the event's subject |
| Link | Graph's `webLink` (opens in Outlook on the web), else a constant deep-link form |
| Media type | `message/rfc822`, or `text/calendar` |
| Metadata | `mail_message_id` (the normalized `Message-ID`), `mail_thread_id` (Graph's `conversationId`), `internal_date`, `version_token` |

**Every request asks for immutable ids** (`Prefer: IdType="ImmutableId"`), delta requests included.
Without it a message's id changes when it moves between folders, and a message the owner files away
would be a different document. With it the id survives a move within the mailbox and changes only
if the item moves to an archive mailbox or is exported and re-imported (`outlook-immutable-id` in
Microsoft's documentation). The two prefixes keep a message and an event from ever being the same
document, and neither can contain `!/`, which the pipeline reserves for attachments.

**`mail_message_id` makes this connector part of the cross-source deduplication.** The mbox
connector's `skip_known_message_ids` asks every other mail source, Outlook instances included, which
`Message-ID` values it holds ([`mbox.md`](mbox.md) §6), so an archived copy of a message the mailbox
already holds is not indexed again.

**`mail_thread_id` makes Outlook conversations collapse.** The retrieval `collapse` stage reads its
configured `thread_key` (`gmail_thread_id`) and then `mail_thread_id`, so a reply chain that quotes
itself is collapsed for this source as it is for Gmail ([`retrieval.md`](../retrieval.md) §5.5).

---

## 3. Scope

**Folders, and nothing else.** Graph's message delta is a per-folder operation that does not
descend into subfolders, so scope has to be a set of folders the walk can enumerate and every path
can test.

1. The mailbox's folder tree is read (`/me/mailFolders` and each folder's `/childFolders`, hidden
   folders included) at the start of every run.
2. **Always excluded**, whatever the configuration says: Junk Email, Deleted Items, Drafts, Outbox
   and Sync Issues, resolved through Graph's well-known names (`junkemail`, `deleteditems`,
   `drafts`, `outbox`, `syncissues`) so a localized mailbox is handled, and every folder below any
   of them.
3. **`exclude_folders`** paths are matched exactly per segment, ignoring case, and resolved to ids.
   **A path the mailbox does not have is a configuration error**, and the message names the
   operator's own text and never lists the mailbox's folders: a typo that silently excluded
   nothing would index precisely the mail the operator meant to keep out.
4. Everything else is in scope.

**`exclude_folders` is a scope field.** Its resolved *ids* are in the scope fingerprint, so a
renamed folder is the same scope, and changing it discards every stored delta link. **A walk does
not remove anything**: widening the scope is handled by it, and narrowing is not. Messages that are
now excluded stay indexed until a reconciliation against the new inventory removes them; a
narrowing large enough to trip the deletion ceiling waits for `manicule connector reconcile NAME
--confirm` (§6).

A folder name that contains a `/` is ambiguous in a path. Exchange allows it; this connector does
not disambiguate it.

---

## 4. Discovery and change detection

### 4.1 The version token

**A message's token is a constant, `mime/1`.** A message's bytes do not change after delivery, and
ids are immutable, so a stored message is never fetched again by id: a flag change, a move or a
category all name a message again, and at the same token the pipeline's version check skips it.

**The `1` is a representation version, and changing it alone would change nothing**, for the reason
Gmail's is ([`gmail.md`](gmail.md) §4.1): a message with no new delta is never yielded, so it never
reaches the version comparison. The representation versions (`mime/1` for messages, `ics/1` for
events) are therefore **also part of the scope fingerprint**, so bumping either discards every
stored delta link and the next sync is a complete walk.

**An event's token is its `changeKey` and the event representation** (`ck-7|ics/1`), because an
event is edited after it is created.

### 4.2 The watermark

A JSON map from folder id to the delta link that folder's last completed round ended on, with the
scope fingerprint and the pinned identity recorded beside it. **The watermark exists only after a
walk has completed**: an abandoned enumeration offers none, so a position past documents nobody
received can never be stored (`assert_connector_contract` checks this).

### 4.3 The walk

For each in-scope folder:

- **With a stored link**: follow it, verbatim, to its end. Each page is one discovery batch; a
  `@removed` entry is counted and ignored, because **a deletion belongs to reconciliation** (§6).
  The link it ends on replaces the stored one.
- **Without one** (a new folder, or one whose state is gone): start a delta round with no token,
  which returns every message currently in the folder, and keep the link it ends on. The first round
  and the incremental ones are the same call, so there is no separate listing to keep in step.

**Links are checked before they are followed.** A `@odata.nextLink` or `@odata.deltaLink` is a full
URL, and a stored one is replayed with a bearer token days later. Both must be `https`, on
`graph.microsoft.com`, under `/v1.0/me/`; anything else raises `UntrustedLinkError` and the access
token is never sent to it, so a poisoned watermark cannot exfiltrate it. A link seen twice in one
round is a loop and is refused.

**A folder that was deleted between the tree read and its walk** is skipped with a warning, and
reconciliation is what removes its mail.

**Expired state.** Microsoft documents that an Outlook delta token's lifetime is not fixed (it
depends on the size of an internal cache) and that an expired one is answered with a 40x error
such as `syncStateNotFound`, and a reset with `410 Gone`. A `410`, a `404`, or a `400` whose code is
`syncStateNotFound`, `syncStateInvalid` or `resyncRequired` on any request of a delta round raises
`CursorExpiredError`, and **that folder alone** is walked from scratch. Nothing already yielded is
wrong, and the walk yields everything again at the same constant token, so only messages not
already stored are fetched. The other folders still resume.

**`walked_full_inventory`** is true only when *every* folder started from scratch, because only a
walk that listed the whole scope may be followed by a reconciliation that treats it as one
([`ingest.md`](../ingest.md) §11).

### 4.4 Batches

One batch per delta page (at most 100 messages, `Prefer: odata.maxpagesize=100`), so durable ingest
commits a page before asking for the next.

---

## 5. Fetch

**A message is `GET /me/messages/{id}/$value`**: the whole MIME message, attachments inline, as
`message/rfc822`. The mail parser anchors citations to lines of the canonical message text and
sends attachments through the parser chain. The headers are read from the front of the bytes only
and never past a megabyte, so a 25 MB message with a large attachment costs a header's worth of
parsing.

Discovery carries the three facts that are not in the MIME (`webLink`, `conversationId`,
`receivedDateTime`) in the document reference, so a fetch is one request. A fetch without them (a
resumed run) asks Graph once. The reference's address, not the fetched link, is the envelope's, so a
retained fetch stays reusable while the envelope names the same address as the record asking for it.

A message removed between discovery and fetch is `NotFoundError`, which the pipeline treats as a
typed acquisition retry and not a failed run.

---

## 6. Reconciliation

A deleted message simply stops appearing in a delta, so without a reconciliation pass the index
serves deleted mail forever. `reconcile()` lists the ids of every message in every in-scope folder
(`/messages?$select=id`, no bodies) and, when the calendar is on, every event in the window, and the
pipeline diffs that against what it holds ([`ingest.md`](../ingest.md) §11).

- **It is built from the same folders the walk covers**, so the three paths (full walk, incremental
  walk, reconciliation) agree about scope.
- **It starts with the mailbox check**, so a token for another mailbox is refused before an id is
  listed.
- **It raises on any failure rather than ending early**: only exhaustion of the iterator is a
  complete inventory, so a failure part way deletes nothing, and the 10% deletion ceiling and
  `connector reconcile --confirm` apply as they do for every connector.
- **What it removes**: a message deleted, a message moved to Deleted Items or any other excluded
  folder, a message in a folder that was deleted, and (§7.1) an event that left the window.

A completed full walk is a full sync, so a reconciliation follows it. A calendar-only instance
lists its window on every sync, so it is always a full walk.

---

## 7. The calendar

### 7.1 The window, and why it is listed and not tracked

`calendarView` is the only way to read recurring events as their occurrences, and Graph's delta for
it (`/me/calendarView/delta`) **fixes its date range when a round begins, cannot select properties,
and reports removals for events outside the range** (Microsoft's `event: delta`). A rolling window
would mean re-baselining that delta constantly. So the calendar keeps **no position at all**: every
sync lists the events between `now - calendar_days_back` (default 90) and `now +
calendar_days_forward` (default 180), as ids and change keys, one small request per hundred events,
and an event whose `changeKey` is unchanged is not fetched again.

**Consequences, stated rather than discovered:**

- **The calendar is a rolling window.** An event that moves out of it is removed by the next
  reconciliation, which is the intended behavior and not a loss. A window that rolls a day at a time
  removes a day's events at a time, far under the deletion ceiling; one that is long idle may trip
  the ceiling and wait for confirmation.
- **A recurring event is its occurrences**, as `calendarView` returns them, each with its own id and
  change key. A daily meeting is a document a day inside the window. The series master is not
  indexed.
- The window is not in the scope fingerprint, because the calendar has no position to invalidate.

### 7.2 The representation: iCalendar, not JSON

Graph v1.0 has no MIME or ICS form of an event, only JSON. The choice was between a new JSON parser
and rendering iCalendar in the connector. **It renders `text/calendar`** (`outlook_ics.py`), because
the repository already has a calendar parser, with one canonical text and one citation shape for the
events that arrive as invitations in mail and the events that arrive from a calendar.

- The client asks for times in UTC (`Prefer: outlook.timezone="UTC"`) and the body as text
  (`outlook.body-content-type="text"`), and the renderer writes `Z` times. The parser prints them
  as `UTC` and does not convert a zone it cannot verify.
- An all-day event is a `DATE` value and keeps no time.
- A property Graph did not send is not written, and a time that is not one of Graph's shapes is left
  out rather than reinterpreted.
- `RRULE` is not written: an occurrence is a single instance.
- Text is escaped and folded as RFC 5545 asks, never inside a multibyte character, so any other
  reader of the retained bytes sees a valid calendar.

---

## 8. Throttling and concurrency

- **Four concurrent requests per app and mailbox**, and 10,000 requests in ten minutes (Microsoft's
  Outlook service limits). A fifth concurrent request is throttled, not queued, so a semaphore holds
  the client to `max_concurrent_requests`, whose ceiling is four.
- **429** is answered with `Retry-After` in both its forms (seconds, and an HTTP date); a pause over
  `max_retry_after_seconds` stops the run with `RateLimitedError` rather than appearing to run while
  waiting. A request that is still limited after `max_retries` attempts stops the run.
- **5xx** is retried with deterministic exponential backoff (no jitter: one index syncing one
  mailbox is not a fleet), then typed as `RemoteError`.
- **401** is refreshed once and the request resent; a second 401 is the credential's rejection
  (`TokenRejectedError`) and not a retry loop.
- Graph does not meter in units the way Gmail does, so there is no pacer: the first sync of a large
  mailbox is bound by the four-request cap and by `$value` fetches of whole messages.

---

## 9. Sensitivity

**The index is not permission-aware.** Everything is fetched as the signed-in account, so the index
holds everything that account can read and anyone who can search the installation can retrieve it.
The connector therefore registers as **sensitive**, exactly as Gmail and mbox do, and an
installation that holds Outlook mail, or any document from such a source, is **refused when it would
serve without authentication** ([`deployment.md`](../deployment.md); `app/sensitive.py`). The mark
is on the data directory, not only on the configuration, so deleting the source does not reopen the
port.

---

## 10. What is never logged

Run-level errors and log records are read by whoever reads a log or a status surface, a wider
audience than whoever may read the mail. So **nothing carries a message id, an event id, a folder
name, an address, a subject, a delta link or a token**: errors are built from the operation's name
(`messages.delta`), a status code and, for a refusal, Graph's short `error.code` token when it
matches `[A-Za-z_]{1,64}`. Microsoft's `error_description` and `message` are free text that carries
the account and a trace id, and are never repeated. The HTTP library's own exception carries the
URL, so it is not chained, and a log filter redacts Graph URLs from its INFO records. A record about
one message belongs to that message's document.

`doctor`'s `outlook-credential` check carries source names and a closed set of outcome words
(`ok`, `invalid_options`, `missing_secret`, `dependency_missing`, `different_mailbox`,
`grant_revoked`, `credential_rejected`, `unreachable`) and nothing else. **Its name is a contract**:
monitors select on it.

---

## 11. What was and was not checked against Microsoft

**Read in Microsoft's documentation** (and built to match; the synthetic Graph reproduces it):

- Message delta is per folder and does not descend into subfolders; it supports `$select`, `$top`
  and `$expand` and the `Prefer: odata.maxpagesize` header; `@removed` marks a deleted message; a
  next link carries `$skiptoken` and a delta link `$deltatoken`, both encoding the original query.
- Immutable ids (`Prefer: IdType="ImmutableId"`) cover messages and events, work with delta (links
  are compatible with both formats), and survive a move within the mailbox.
- A message's MIME is `GET .../messages/{id}/$value`.
- An expired Outlook delta token is a 40x error such as `syncStateNotFound`; a reset is `410 Gone`.
  Outlook token lifetime is not fixed.
- Outlook limits: four concurrent requests and 10,000 requests per ten minutes per app and mailbox.
- `calendarView/delta` fixes its window, cannot `$select`, and reports out-of-window removals.
- A refresh token is not revoked when used; the client is to adopt the new one and discard the old;
  `offline_access` is what makes Microsoft issue one.
- A loopback `localhost` redirect ignores the port when matching; `http://127.0.0.1` can only be
  registered through the manifest.

**Not exercised against a live tenant**, so treat each as unverified until a real mailbox has run
it:

- The exact status and `error.code` a real expired delta link returns, for each of the three shapes
  this connector accepts (§4.3), and whether a mid-round `$skiptoken` can expire.
- That `includeHiddenFolders` is honored on `childFolders` and preserved in a folder listing's next
  link, and that the well-known names resolve in every locale.
- That `$select` of `webLink`, `conversationId` and `receivedDateTime` on a delta behaves as on a
  plain listing, and that `changeKey` changes on every edit that matters.
- That `calendarView` accepts `$top=100` and the `Prefer` headers of §7.2, and returns occurrences
  with stable immutable ids.
- The shape of the token response's `scope` (the documentation shows it percent-encoded in one
  example); the check handles prefixed, bare and encoded forms.
- How long a refresh token that is never rotated on disk survives, and what a conditional-access
  policy does to a refresh from a different address than the sign-in.
- Whether Microsoft's `Retry-After` is sent on every 429 from the Outlook service.
- **The whole sign-in against a directory whose consent and conditional-access policies are not
  yours.**

---

## 12. Where this lives

| Concern | Module |
|---|---|
| Configuration, the tenant and GUID refusals, the scope fingerprint | `manicule/connectors/outlook_config.py` |
| The credential, rotation, the mailbox check | `manicule/connectors/outlook_auth.py` |
| The Graph client: links, retry, concurrency, typed failures | `manicule/connectors/outlook_client.py` |
| Discovery, fetch, reconciliation, the watermark | `manicule/connectors/outlook.py` |
| An event as iCalendar | `manicule/connectors/outlook_ics.py` |
| `connector login`, the scope check | `manicule/connectors/outlook_login.py` |
| The loopback listener and token sinks it shares | `manicule/connectors/gmail_login.py` |
| Header reading and provenance shared with Gmail | `manicule/connectors/mail_headers.py` |
| Registration through the `manicule.plugins` entry point | `manicule/connectors/plugin.py` |
| The `outlook-credential` check and the login operation | `manicule/app/service.py` |
| The synthetic Graph and Entra | `tests/connectors/fake_graph.py` |

---

## 13. Not built

- **App-only (client credentials) access.** It has no refresh token and nothing to rotate, which
  suits a headless server, but it needs a directory administrator and an Exchange access policy to
  keep it to one mailbox. This connector is the delegated flow because that is what a person who
  cannot create an app registration can be given.
- **Shared and group mailboxes, other users' mailboxes**, and the archive mailbox. The connector
  reads `/me`.
- **Personal Microsoft accounts.** The tenant is a directory GUID; `consumers` is refused.
- **National clouds.**
- **Attachments as separate documents beyond what the MIME parser already does**, contacts, tasks
  and Teams chat.
- **Change notifications (webhooks)**, which would replace polling with a push and need a public
  endpoint the server does not have.
