# Storage & data model

Design for the metadata store, the vector store, the lexical index and the retained
source bytes. Ticket [#2](https://github.com/mgd43b/manicule/issues/2).

The store choice is settled in `PLAN.md` §2 and `docs/contracts.md` §4 — **SQLite plus
LanceDB, and therefore FTS5 for BM25**. That reasoning is not repeated here. This document
starts one level below it: given those three stores, what the schema is, which store owns
what, and what happens when they disagree.

> **Prior art.** OpenDocuments is referenced in clearly-marked callouts like this one where
> the comparison carries design information. Everything outside these callouts stands on
> its own.

---

## 1. The whole design in four sentences

**SQLite is authoritative. LanceDB and FTS5 are derived indexes. The blob store is
immutable. Nothing is ever deleted from a derived store synchronously.**

Every decision below follows from those four. They matter because two stores with no shared
transaction *will* diverge on a crash, and there are only two possible recovery stories:
reconcile two peers, or rebuild the derived side. Reconciling peers requires a tie-breaker
that does not exist. Rebuilding does not. So one store is declared the truth and the others
are declared disposable, and the design spends its effort making "rebuild the derived side"
cheap rather than making divergence impossible.

### The blast-radius ladder

Every repair path lands on one of four rungs. The storage layer's whole job is to keep
failures on the lowest rung that is correct.

| Rung | Rebuild | Cost | Needs |
|---|---|---|---|
| 1 | FTS5 index | seconds | `chunks` table |
| 2 | Vectors | minutes–hours | `chunks.embed_text` |
| 3 | Chunks | minutes | retained original bytes |
| 4 | Documents | hours, rate-limited, **may fail** | the upstream source |

Rung 4 is the only one that can fail for reasons outside the machine: the page was deleted,
the token expired, the API is down, the content changed. It is also the only rung whose
result is not reproducible. Retained original bytes (§7) exist to make rung 4 unreachable
for anything except genuinely new content — which is what `original_ref` in
`docs/contracts.md` §2 is for.

---

## 2. Layout on disk

```
<data_dir>/
  manicule.lock            exclusive; one instance per directory
  manicule.db              SQLite — the authority
  manicule.db-wal
  manicule.db-shm
  vectors/                 LanceDB — derived
    chunks__<fp8>.lance/   vector table, name carries the fingerprint hash
    _manicule_meta.lance/  one row: the fingerprints this directory was built with
  blobs/                   immutable, content-addressed
    sha256/ab/cd/abcd…     retained original bytes
```

`manicule.lock` is held for the process lifetime. The recovery sweep, the tombstone sweep and
the blob GC all assume a single writer, and WAL permits several — so the assumption is enforced
rather than hoped for ([`ingest.md`](ingest.md) §6.5).

`<data_dir>` resolution and precedence belong to config
([#1](https://github.com/mgd43b/manicule/issues/1)). The only requirement storage places on
it: **all three live under one root**, because a backup is a snapshot of that root and a
restore is a replacement of it. Splitting the vector store onto a different volume is a
supported deployment, but it makes the backup procedure in §9 the operator's problem rather
than manicule's, and `doctor` says so.

---

## 3. SQLite: conventions before tables

These apply to every table and are not repeated per-column.

### 3.1 Connection setup

```python
@event.listens_for(engine.sync_engine, "connect")
def _configure(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA busy_timeout = 5000")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("PRAGMA wal_autocheckpoint = 1000")
    cur.close()
```

**`foreign_keys` is per-connection and defaults to OFF.** This is the single most common way
a SQLite schema full of `REFERENCES` clauses turns out to enforce nothing: the pragma is set
once at startup, the pool opens a second connection under load, and every write on that
connection skips referential integrity silently. Setting it in a `connect` event listener is
the only placement that covers every connection the pool will ever open. A test asserts
`PRAGMA foreign_keys` returns `1` on a freshly-checked-out connection.

`busy_timeout` matters because `aiosqlite` runs each connection on its own thread, so
"async SQLAlchemy" does not serialise writers for you. Without it, concurrent ingest and a
web request produce `SQLITE_BUSY` immediately rather than after a wait. WAL permits many
readers with one writer; manicule keeps a single write path and lets readers run concurrently.

**Minimum SQLite 3.35**, checked at startup and reported by `doctor`, along with a probe that
actually creates a temporary FTS5 table. Python's `sqlite3` links against whatever the
platform provides, and a build without FTS5 fails at the first query rather than at install.

### 3.2 Identifiers

- **Entity IDs are `uuid4` in canonical dashed form, stored as `TEXT`.** Opaque, safe in a
  URL, no coordination.
- **`chunks.id` is content-derived**, from `manicule.core.ids.chunk_id` — a length-prefixed
  blake2b digest over `document_id`, `position` and `text`.

  A chunk's identity is its content-in-context, so **editing one paragraph of a document
  leaves every other chunk's id unchanged** — their vectors survive the re-parse and only the
  edited chunk is re-embedded. It also makes a stale citation *dangle* rather than silently
  re-point: if the text a citation named no longer exists, its id no longer exists either.
  That is the rule `docs/contracts.md` §1 states for anchors — a location is correct, or it
  is absent.

  **`position` is part of the digest, and the trade that makes is worth stating.** An earlier
  draft of this document derived the id from content alone and disambiguated byte-identical
  chunks with a `:n` suffix. The shipped scheme is simpler and has no suffix, because
  position already separates duplicates — but it moves the churn rather than removing it:

  | | id includes `position` (shipped) | content-only with a `:n` suffix (rejected) |
  |---|---|---|
  | Edit one paragraph in place | only that chunk changes id | only that chunk changes id |
  | **Insert** a paragraph | every later chunk changes id | unaffected |
  | Delete one of several byte-identical chunks | unaffected | the survivors renumber |

  So an insertion re-embeds the tail of the document. That is the cost, it is real, and it is
  paid on a re-parse rather than on a query. It was accepted because the failure runs in the
  conservative direction in both schemes — a needless re-embed and a dangling citation, never
  a citation silently re-pointed at different text — and because a positional digest has no
  enumeration step, so there is no case where an id depends on how many *other* chunks
  happen to share its content.

  > **Prior art.** OpenDocuments builds chunk IDs as `${documentId}_chunk_${i}` and then
  > parses them back out with `/^(.+)_chunk_(\d+)$/` to find neighbours, and deletes FTS rows
  > with `LIKE 'docid_chunk_%'`. Position is load-bearing inside the identifier, so
  > re-chunking a document silently re-points every stored citation at different text, and
  > any document ID containing `_chunk_` breaks the parse.

- **`chunks` additionally has `seq INTEGER PRIMARY KEY`** — a rowid alias, required as the
  `content_rowid` for external-content FTS5 (§6.1). `chunks.id` carries a `UNIQUE` constraint.

### 3.3 Timestamps

**One writer, one format.** All timestamps are set in Python as timezone-aware UTC through a
`TypeDecorator`; **no column has a `server_default` of `datetime('now')`.**

```python
class UtcDateTime(TypeDecorator[datetime]):
    impl, cache_ok = DateTime, True

    def process_bind_param(self, value, dialect):
        if value is None: return None
        if value.tzinfo is None:
            raise ValueError("naive datetime rejected; pass an aware UTC value")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return None if value is None else value.replace(tzinfo=UTC)
```

SQLite has no date type; a timestamp is text, and text sorts lexicographically. Two writers
means two formats in one column, and `ORDER BY created_at DESC` then returns an order that
is neither chronological nor stable.

> **Prior art.** OpenDocuments declares `created_at TEXT DEFAULT (datetime('now'))`, which
> yields `2026-08-10 12:00:00`, and *also* inserts `new Date().toISOString()`, which yields
> `2026-08-10T12:00:00.123Z`. Both land in the same column. Space is `0x20` and `T` is
> `0x54`, so every SQL-defaulted row sorts before every application-written row regardless
> of when either happened.

### 3.4 Typed columns

| Kind | Type | Note |
|---|---|---|
| JSON payloads | `JSON` | Validated by a Pydantic model wherever the shape is known |
| Booleans | `Boolean` | Not `INTEGER DEFAULT 0` |
| Closed value sets | `Enum(..., native_enum=False)` | Renders `VARCHAR` + `CHECK` |
| Timestamps | `UtcDateTime` | §3.3 |

Closed sets get a real `CHECK` constraint, not just a Python enum. A document whose `status`
is misspelled is invisible to retrieval — the join in §6.2 filters on `status = 'indexed'` —
and it fails silently, forever, for one document. A `CHECK` turns that into an error at the
write that caused it.

The cost is honest: SQLite cannot alter a constraint, so adding a status value means an
Alembic batch migration that rebuilds the table. `documents` is a per-document table, not a
per-chunk one, so the rebuild is affordable and the value set changes rarely.

**`STRICT` tables, rejected.** They would catch the same class of error more broadly, but
`STRICT` admits only `INT`/`INTEGER`/`REAL`/`TEXT`/`BLOB`/`ANY` as column types, which
collides with every `JSON`, `BOOLEAN` and `DATETIME` type name SQLAlchemy renders. Adopting
it means declaring every column as `TEXT` or `INTEGER` and moving all conversion into
`TypeDecorator`s. Revisit if the ORM grows a dialect flag for it.

### 3.5 Join tables are `WITHOUT ROWID`

`document_tags`, `collection_documents`, `chunk_relations`, `workspace_members` are pure
composite-key association tables with no rowid consumer. `WITHOUT ROWID` stores them in the
primary-key B-tree directly instead of maintaining a separate index over a rowid nothing
reads.

`chunks` is emphatically **not** `WITHOUT ROWID` — it needs its rowid for FTS5.

---

## 4. The tables

Sixteen carry over from the shape `PLAN.md` §2 names. Four are added, each with a job that
one of the sixteen cannot do. `alembic_version` and the FTS5 shadow tables also exist and
are managed, not modelled.

### 4.1 The four additions

| Table | Why it must exist |
|---|---|
| `chunks` | Chunks carry an `Anchor`, and anchors are the type `docs/contracts.md` §1 calls the most important in the system and locks once ingest runs. Storing them only inside a columnar vector store's JSON blob puts the system's most valuable data in its most disposable store. A real table also gives `chunk_relations` something to point a foreign key at, and gives rung 2 of the ladder somewhere to read `embed_text` from. |
| `blobs` | Content-addressed retained source bytes with media type, size and compression, plus a target for `documents.original_ref` to reference. §7. |
| `index_state` | One row recording the fingerprints and the derived-index names this data directory was built with. §6.3. |
| `vector_tombstones` | Chunk IDs deleted from SQLite whose vectors have not yet been swept from LanceDB. §8.2. |

> **Prior art.** OpenDocuments has no `chunks` table. Chunk text lives in LanceDB *and*
> again in `chunks_fts` — two copies, neither authoritative — and everything else about a
> chunk lives in a `metadata_json` string column inside LanceDB. The consequences are
> visible in its own code: a BM25 hit has to round-trip through the vector store
> (`vectorDb.getByIds`) just to hydrate metadata the lexical leg never needed;
> `chunk_relations` has no foreign keys because there is no table to point at, so orphan
> cleanup is a `LIKE` over string-formatted IDs; and `document_versions.snapshot_chunk_ids`
> is a JSON array of IDs that nothing can validate.

**The alternative, recorded.** Keep sixteen tables and put chunk metadata in LanceDB, as the
prior art does. It is fewer moving parts and one less place for ingest to write. Rejected
because it puts anchors — locked, irreplaceable, and the product — in the store designed to
be thrown away and rebuilt, and because it makes the lexical leg depend on the vector store
for data the vector store has no reason to hold.

### 4.2 `documents`

```python
class Document(Base):
    __tablename__ = "documents"

    id:            Mapped[str]  = mapped_column(Text, primary_key=True)
    workspace_id:  Mapped[str]  = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    connector_id:  Mapped[str]  = mapped_column(ForeignKey("connectors.id", ondelete="RESTRICT"))

    source:        Mapped[str]  = mapped_column(Text)          # "confluence", "filesystem"
    source_id:     Mapped[str]  = mapped_column(Text)          # connector-stable identity
    uri:           Mapped[str]  = mapped_column(Text)          # citable location; mutable
    title:         Mapped[str]  = mapped_column(Text)
    media_type:    Mapped[str | None]
    size_bytes:    Mapped[int | None]

    content_hash:  Mapped[str | None]                          # sha256 of fetched bytes
    version_token: Mapped[str | None]                          # opaque, connector-defined
    original_ref:  Mapped[str | None] = mapped_column(ForeignKey("blobs.hash", ondelete="RESTRICT"))
    original_omitted_reason: Mapped[str | None]

    container_id:  Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    container_depth: Mapped[int] = mapped_column(default=0)

    status:        Mapped[DocumentStatus]
    error_message: Mapped[str | None]
    parser:        Mapped[str | None]
    parse_duration_ms: Mapped[int | None]

    chunk_fp:      Mapped[str | None]                          # short hash, §6.4
    embed_fp:      Mapped[str | None]

    metadata_:     Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at:    Mapped[datetime]
    updated_at:    Mapped[datetime]
    indexed_at:    Mapped[datetime | None]
    deleted_at:    Mapped[datetime | None]
```

**Identity is `(workspace_id, source, source_id)`, and the workspace is part of the id
itself.** `manicule.core.ids.document_id` takes all three, so two workspaces indexing the same
upstream source derive different ids rather than colliding on one row. This has to be settled
before anything is indexed: `chunk_id` derives from `document_id`, so changing the scheme later
re-derives every chunk id, which invalidates every vector and forces a full re-embed — the
same "settle it before you index" class as vector dimensionality and chunk size.

The cost is real and small. The same source synced into two workspaces produces two documents,
two chunk sets and two sets of vectors, which is what isolation *means*. It does not duplicate
the corpus: retained bytes are content-addressed (§7), so both workspaces reference one blob.
The partial unique index below is then a second line of defence rather than the mechanism.

**`source_id` is the other half of it.** Document identity must be whatever the
connector can *promise* is stable, and a URI is not that. A URI is display data — the string
a citation points at, chosen for a human to read — and nothing obliges a source to keep it
fixed. Identity has to be the handle the source itself uses.

Every connector already has one, and it is visible in how each one is addressed rather than
in how it is displayed. `docs/connectors/confluence.md` fetches by page ID
(`GET /wiki/api/v2/pages/{id}`, §4) and tracks change by `version.number` per page ID (§2);
its citation template (§8) is `…/pages/{pageId}/{slug}`, where the ID and the trailing
human-readable component are separate fields — the API is keyed on one of them. For the local
filesystem connector (`PLAN.md` §6) the case needs no citation at all: a file that is moved or
renamed is the same file, and watch mode will report exactly that.

Keying identity on the URI instead means any source that re-mints a URI for an unchanged
document creates a second row, while the first keeps serving stale content forever, invisible
to reconciliation because the connector never reports it again. Keying on `source_id` costs
nothing and cannot fail that way. `uri` remains, because a citation needs somewhere to point.

> **Not established here.** An earlier draft claimed the `{slug}` component of a Confluence
> URL is title-derived and changes on rename. Nothing in `docs/connectors/confluence.md`
> supports that, and it is not verified against Atlassian's documentation, so it has been
> removed rather than softened. The design deliberately does not depend on it: if slugs turn
> out to be perfectly stable, keying on `source_id` is still correct for every other source,
> and still free.

```python
Index("uq_documents_identity", "workspace_id", "connector_id", "source_id",
      unique=True, sqlite_where=text("deleted_at IS NULL"))
```

A **partial** unique index, because a soft-deleted document must not block re-ingesting the
same source.

> **Prior art.** The equivalent index is `(workspace_id, source_path)` and is not unique.
> Two live rows for the same source are representable, and the lookup takes the first row
> returned — so a duplicate ingest silently splits a document in two and half the citations
> point at the stale half.

**`connector_id` is `NOT NULL`.** Everything arrives through a connector, including the
local filesystem and web upload, which are connectors in `PLAN.md` §6. That removes a
nullable foreign key whose null state means "this document can never be reconciled again",
and it gives connector sync state exactly one home. Deleting a connector is a soft delete;
hard deletion is `RESTRICT` and requires its documents be dealt with first.

**`container_id` is a self-referential cascade**, for archive members. An archive member is
its own `Document` with a `zip:<container>!/<inner/path>` URI, and deleting the container
must delete its members. That is a foreign key, so it is a column — it cannot be a key
inside `metadata`, and `chunk_relations` is the wrong table because this relates documents,
not chunks. `container_depth` carries a `CHECK` against a configured maximum so a nested
archive cannot recurse without bound.

**Re-deriving a container's members is a reconcile, not a delete-then-insert.**
`docs/parsing.md` re-derives every member when a container's bytes change, on the grounds
that matching old members to new ones is guesswork. That is right at the parse stage and
does *not* have to reach storage as a wholesale replacement — because the inner path is a
stable key. A member's `source_id` is its `zip:…!/inner/path` string, so storage reconciles
the derived set against the stored one exactly as `Connector.reconcile` does for a source:
members present in both are **upserted by `source_id`**, members no longer present are
soft-deleted. No content matching and no heuristics.

This matters because delete-then-insert would mint new `documents.id` values for every
member, discarding their `document_versions` history and dangling every citation into the
archive — including members whose bytes did not change. Under reconcile, an unchanged
member keeps its ID, keeps its history, and its unchanged `content_hash` lets ingest skip
re-parsing and re-embedding it entirely.

**`status`** is a `CHECK`-constrained enum owned by
[#1](https://github.com/mgd43b/manicule/issues/1). Storage requires only that it includes
the terminal no-content states — `no_extractable_text`, `failed`,
`unsupported_media_type`, `container` — because **a document with zero chunks is a normal,
valid, round-trippable row**, not an error. This is a second reason there is no
`chunk_count` column: `0` cannot distinguish "not indexed yet" from "nothing to index", and
`status` can.

**No `chunk_count`.** It is derivable with an index, and as a stored counter it is a
divergence source that is load-bearing exactly when it is wrong — the natural use is
"delete chunks above the previous count", which leaks stale rows the moment the counter
drifts. Stale-chunk deletion is `DELETE FROM chunks WHERE document_id = ? AND id NOT IN (…)`,
which needs no counter and is correct even after a crash.

Indexes: `(workspace_id, status)`, `(workspace_id, uri)`, `(content_hash)`,
`(workspace_id, deleted_at) WHERE deleted_at IS NOT NULL` (the trash view; live-row queries
fold `deleted_at IS NULL` into the partial indexes above), `(connector_id)`,
`(container_id)`, `(chunk_fp)`, `(embed_fp)`.

### 4.3 `chunks`

```python
class Chunk(Base):
    __tablename__ = "chunks"

    seq:          Mapped[int] = mapped_column(Integer, primary_key=True)   # rowid alias
    id:           Mapped[str] = mapped_column(Text, unique=True)
    document_id:  Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))

    text:         Mapped[str]                     # cited and shown
    embed_text:   Mapped[str]                     # exactly what the embedder saw
    heading_text: Mapped[str]                     # " > "-joined breadcrumb, indexed by FTS
    heading_path: Mapped[list[str]] = mapped_column(JSON)

    kind:         Mapped[BlockKind]               # prose|heading|table|code|list|panel|media
    lang:         Mapped[str | None]
    position:     Mapped[int]
    token_count:  Mapped[int]

    anchor:       Mapped[dict] = mapped_column(JSON)   # tagged union, contracts.md §1
    metadata_:    Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at:   Mapped[datetime]
```

**`embed_text` is stored, not recomputed.** It is derivable from `heading_text` and `text`
today, and that is exactly the problem: the derivation rule is code, code changes, and a
changed rule silently produces different vectors from the same chunks. The fingerprint guard
in §6.3 is only meaningful if the embedder's actual input is recoverable. Storing it also
puts rung 2 of the ladder entirely inside SQLite.

**`heading_text` alongside `heading_path` is a deliberate small redundancy.** FTS5 indexes a
string, and joining a JSON array of arbitrary length is not expressible in a SQLite generated
column. `heading_path` stays because `HeadingAnchor.path` is a list and a heading may itself
contain the separator. Both are written from one place — a SQLAlchemy validator on
`heading_path` — so they cannot drift.

**`anchor` is JSON holding the tagged union.** It is the one JSON column whose shape is
locked (`docs/contracts.md` §1), so it is validated by a Pydantic discriminated union on
both read and write, and a round-trip test on every parser is already an obligation.

Indexes: `(document_id, position)` unique, `(kind)`.

### 4.4 `chunk_relations`

Now that chunks are a table, both columns become real foreign keys with `ON DELETE CASCADE`,
and the `LIKE`-based orphan cleanup disappears entirely.

```python
source_chunk_id / target_chunk_id  → chunks.id, CASCADE
relation_type                       TEXT
PrimaryKeyConstraint(source, target, relation_type)
CheckConstraint("source_chunk_id <> target_chunk_id")
Index("ix_chunk_relations_target", "target_chunk_id")
```

The index on `target_chunk_id` is not redundant with the primary key. Lookups are
`WHERE source = ? OR target = ?`, and a composite key leading with `source` cannot serve the
second half of that predicate.

### 4.5 `blobs`

```python
hash: Mapped[str] = mapped_column(Text, primary_key=True)   # sha256 hex of original bytes
algo / media_type / size_bytes / stored_bytes / compression / created_at
```

`size_bytes` is the original; `stored_bytes` is what is on disk after compression. Both,
because "how much source am I holding" and "how much disk is this costing" are different
questions and `doctor` reports both.

### 4.6 `index_state`

A singleton. One row, enforced.

```python
id:                Mapped[int] = mapped_column(primary_key=True)  # CHECK (id = 1)
vector_table:      Mapped[str]                # e.g. "chunks__7f3a91c2"
embed_fingerprint: Mapped[str] = mapped_column(Text)   # canonical bytes, verbatim
chunk_fingerprint: Mapped[str] = mapped_column(Text)
fts_tokenizer:     Mapped[str]                # "porter unicode61 remove_diacritics 2"
created_at / updated_at
```

**`vector_table` is a pointer, not a constant.** This is what makes the re-embed path in
§6.5 crash-safe: a new table is built alongside the old one and the pointer is moved in a
single SQLite transaction.

**The fingerprints are `Text` holding canonical bytes, not `JSON` holding a dict.** §6.3
compares them for byte equality, and a `JSON` column cannot support that: SQLAlchemy's default
serialiser does not sort keys, so the stored bytes depend on the insertion order of the dict
that was passed in. Verified — the same fingerprint built in two field orders produces two
different stored values, which would either false-mismatch or silently rely on
re-canonicalising at every read. Storing the canonical bytes makes the comparison literal.
SQLite's `json_extract` still works on that text, so `doctor`'s field-by-field diff loses
nothing.

**The canonical form, pinned.** One function produces fingerprint bytes and nothing else may:

```python
def canonical(fp: Mapping[str, object]) -> bytes:
    return json.dumps(fp, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

fp8 = hashlib.sha256(canonical(fp)).hexdigest()[:8]   # the §6.5 table-name suffix
```

`sort_keys` makes it order-independent; `separators` pins whitespace rather than describing it;
`ensure_ascii` keeps the output pure ASCII so encoding and Unicode normalisation never enter a
comparison; `allow_nan=False` rejects the two values that are not JSON.

**Fingerprint values are `str | int | bool | None`, or nested containers of those — never
floats**, enforced at construction. A float's text form depends on how it was computed:
`0.1 + 0.2` serialises as `0.30000000000000004` and `0.3` as `0.3`, so the same intended
setting arriving by two routes yields two different table names and a spurious refusal.
Anything fractional is carried as a string.

### 4.7 The other twelve, and what changed

| Table | Kept as-is | Changed, and why |
|---|---|---|
| `workspaces` | `id`, `name UNIQUE`, `mode`, `settings JSON` | `mode` gets a `CHECK` (`personal`/`team`) |
| `workspace_members` | composite PK, `role` | **`api_key` column removed.** It held a raw, unhashed key — precisely what `api_keys.key_hash` exists to avoid. `api_keys` is the only key store. `role` gets a `CHECK` (`admin`/`member`/`viewer`) |
| `connectors` | `type`, `config JSON`, `sync_interval_seconds`, `last_synced_at`, `status`, `error_message`, `deleted_at` | `config` validated by the connector's Pydantic schema, not stored blind. `UNIQUE (workspace_id, name) WHERE deleted_at IS NULL`. `watermark JSON` added — `Connector.discover` takes a watermark (`contracts.md` §3) and it has to persist somewhere; `last_synced_at` is a timestamp, and a watermark is not always a timestamp. `metadata JSON` added, matching `documents` — last-run counters live here ([`ingest.md`](ingest.md) §13.1), overwritten per run rather than accumulated, which is the right retention policy for a diagnostic |
| `tags` | `UNIQUE(workspace_id, name)`, `color` | `workspace_id NOT NULL` |
| `document_tags` | composite PK, both cascades | `WITHOUT ROWID` |
| `collections` | `auto_rules JSON` | `UNIQUE (workspace_id, name)`; `workspace_id NOT NULL` |
| `collection_documents` | composite PK, both cascades | `WITHOUT ROWID` |
| `document_versions` | `version`, `content_hash`, `changes`, `snapshot_chunk_ids JSON` | `UNIQUE (document_id, version)` — a non-unique index permits two rows claiming version 3. `original_ref` added so a prior version's bytes survive independently of the current one |
| `conversations` | `share_token UNIQUE`, `shared`, soft delete | `shared` becomes `Boolean`; `share_token` from `secrets.token_urlsafe` |
| `messages` | `role`, `content`, `sources JSON`, `confidence_score`, `response_time_ms` | `role` gets a `CHECK`. **`sources` embeds `Anchor`s**, so it inherits the ⚠️ lock from `contracts.md` §1 — a stored conversation's citations must keep resolving. Index `(conversation_id, created_at)` |
| `query_logs` | everything | `workspace_id` keeps `ON DELETE CASCADE`. Query text is user content scoped to a workspace; retaining it past workspace deletion is a data-retention problem, not a feature. [#15](https://github.com/mgd43b/manicule/issues/15) exports what it needs rather than treating live rows as an archive |
| `audit_logs` | everything, **including no foreign key** | Deliberate and now documented: `workspace_id` and `user_id` are plain `TEXT`. An audit log that cascades away when the thing it audits is deleted is not an audit log. Index `(workspace_id, created_at)` added alongside `(event_type, created_at)` |
| `api_keys` | `key_hash UNIQUE`, `key_prefix`, `scopes`, `rate_limit`, `expires_at`, `last_used_at`, `revoked_at`, `allowed_ips` | `idx_api_keys_hash` **dropped** — `UNIQUE` already creates that index, so it was a second copy of the same B-tree maintained on every write |
| `plugins` | `name` PK, `type`, `version`, `config JSON`, `status` | **`permissions` column removed**, per `contracts.md` §5. And this table becomes the actual plugin registry rather than a JSON file beside the database, so plugin state is inside the same transactional and backup boundary as everything else |

> **Prior art.** The `plugins` table is created by `001_initial.sql` and never read or
> written; the real registry is `installed-plugins.json` on disk. It is in the backup file
> list, so a restore can leave the registry and the database describing different worlds.

---

## 5. Alembic

```
alembic.ini
src/manicule/storage/migrations/
    env.py
    script.py.mako
    versions/20260810_0001_initial.py
```

> **Prior art.** OpenDocuments hand-rolls a runner that lists `*.sql`, sorts by filename,
> and executes anything not in a `schema_migrations` table. There is no downgrade path, no
> autogenerate, no branch handling, and every schema change since the initial file is an
> `ALTER TABLE … ADD COLUMN` — which is the only `ALTER` SQLite has historically supported,
> so the tooling shape and the schema evolution constrained each other. The initial schema
> is also replayed on every fresh install, meaning three of the eight files exist only to
> add columns that could simply be in the first one.

manicule starts from a clean slate, so `0001_initial` contains the final shape — the
`revoked_at`, `allowed_ips` and `version_token` columns are simply columns, not migrations.

Four things `env.py` must do:

**1. Async engine.**

```python
async def run_migrations_online() -> None:
    engine = async_engine_from_config(config.get_section(config.config_ini_section),
                                      poolclass=pool.NullPool)
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
```

**2. `render_as_batch=True`.** SQLite cannot drop a column or alter a constraint. Batch mode
does create-copy-swap under the hood. Without it, the first migration that needs to change a
`CHECK` fails and the only remedy is hand-written DDL.

**3. A `MetaData` naming convention.** Batch mode has to *name* the constraint it is dropping,
and SQLite auto-generates anonymous names. This is the most common way an Alembic-on-SQLite
project discovers, months in, that it cannot migrate:

```python
NAMING = {
  "ix": "ix_%(table_name)s_%(column_0_N_name)s",
  "uq": "uq_%(table_name)s_%(column_0_N_name)s",
  "ck": "ck_%(table_name)s_%(constraint_name)s",
  "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
  "pk": "pk_%(table_name)s",
}
```

**4. An `include_name` filter excluding the FTS5 tables.** Autogenerate does not model virtual
tables or triggers, so it sees `chunks_fts` and its shadow tables (`chunks_fts_data`,
`_idx`, `_content`, `_docsize`, `_config`) as tables that exist in the database and not in the
models — and helpfully emits `op.drop_table` for all of them. Exclude anything matching
`chunks_fts%`, and write the FTS5 and trigger DDL by hand with `op.execute`.

**Downgrades are required and tested.** For every revision: on a scratch database,
`upgrade head` → `downgrade -1` → `upgrade head`, asserting the schema is identical at both
`head`s. A downgrade that has never run is not a downgrade path.

**`alembic check` runs in CI** to catch a model edited without a migration. The repo has no
CI yet — [#1](https://github.com/mgd43b/manicule/issues/1) is building it — so this is filed
rather than wired up.

---

## 6. The derived indexes

### 6.1 FTS5 — the lexical leg

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    heading_text,
    content='chunks',
    content_rowid='seq',
    tokenize='porter unicode61 remove_diacritics 2'
);
```

**External content.** FTS5 stores only the inverted index and reads the original text from
`chunks` when it needs it. One copy of the corpus text, in the authoritative store, and a
whole class of "the two copies disagree" bugs that cannot occur.

**Kept in sync by triggers, not by the application.**

```sql
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading_text)
    VALUES (new.seq, new.text, new.heading_text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
    VALUES ('delete', old.seq, old.text, old.heading_text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE OF text, heading_text ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
    VALUES ('delete', old.seq, old.text, old.heading_text);
  INSERT INTO chunks_fts(rowid, text, heading_text)
    VALUES (new.seq, new.text, new.heading_text);
END;
```

A trigger runs inside the same transaction as the row change and cannot be bypassed by a
migration, a `doctor` repair, or a hand-written fix. Application-level synchronisation covers
only the write paths someone remembered.

**One dependency, verified rather than assumed.** Hard-deleting a document reaches `chunks`
through `ON DELETE CASCADE`, and for the triggers to clean up, a *cascaded* delete must fire
the child table's `AFTER DELETE` trigger. It does — checked directly on SQLite 3.51, at two
cascade levels (`documents` → `documents` via `container_id` → `chunks`), with and without
`PRAGMA recursive_triggers`, which turns out not to govern this case. It is not worth relying
on from memory: the `COUNT(chunks)` versus FTS row-count check and the FTS `integrity-check`
in §10 exist partly to catch an environment where it does not hold, and a unit test asserts it
directly.

> **Prior art.** `chunks_fts` there is a standalone table holding a second copy of every
> chunk's text, populated from the ingest path with a hand-written compensating-delete in a
> `catch` block. Every other write path — and there are several — skips it.

**Two columns, weighted.** BM25 scoring uses `bm25(chunks_fts, 1.0, 0.4)`. Note the sign:
FTS5's `bm25()` returns a **negative** number, and a better match is *more* negative, so the
ordering is `ORDER BY bm25(chunks_fts, 1.0, 0.4)` **ascending**. Anything that treats it as a
similarity — or takes its absolute value — inverts the ranking or flattens it, and the result
still looks like a plausible ranked list. The value fed to RRF is a rank, not a score, which
sidesteps the question entirely; where a score is genuinely needed, negate it. Note also that
`bm25()`'s first argument must be the FTS table's real name — it is parsed as a column
reference, so a query that aliases `chunks_fts` fails with `no such column`. The breadcrumb
has to be searchable for the same reason it is prefixed to `embed_text`: "Configuration" is
unfindable without knowing what it configures. But it repeats on every chunk of a page, so
indexing it at full weight both floods term frequencies and depresses the IDF of the very
terms that identify the page. Separate columns let it contribute without dominating; a
single concatenated column cannot.

**Query it as one joined statement, never as MATCH-then-hydrate.** This is the canonical
lexical query and the shape is load-bearing:

```sql
SELECT c.id, c.text, d.uri, d.title
FROM chunks_fts
JOIN chunks    c ON c.seq = chunks_fts.rowid
JOIN documents d ON d.id  = c.document_id
WHERE chunks_fts MATCH :q
  AND d.workspace_id IN (:workspaces)
  AND d.deleted_at IS NULL
  AND d.status = 'indexed'
ORDER BY bm25(chunks_fts, 1.0, 0.4)
LIMIT :k
```

The `LIMIT` has to be applied *after* the joins and filters, not before. Running `MATCH … LIMIT k`
first and filtering the results afterwards silently returns fewer than `k` live rows —
because deletion is deferred (§8.2), the chunks of soft-deleted documents are still in the
index and still compete for those `k` slots, as are chunks belonging to other workspaces,
which `chunks_fts` cannot distinguish because the workspace lives on `documents`.

Measured on a fixture with three matching live chunks in the target workspace, five in a
soft-deleted document and five in another workspace: **`MATCH`-then-hydrate with `k = 5`
returned zero live in-workspace results; the joined statement returned all three.** Not a
marginal loss of recall — a total one, silently, with a well-formed empty result set. SQLite
pushes the `MATCH` down and applies the `LIMIT` last, so the joined form costs nothing.

**Tokenizer: `porter unicode61 remove_diacritics 2`.**

> **Prior art, and the evidence for this choice.** OpenDocuments ships `tokenize='unicode61'`
> — surface forms only. An unmerged experiment on a feature branch (`009_fts5_porter_stemming.sql`,
> **not part of its shipped schema**) records the measured consequence: `"authenticate"`
> scored zero hits against a chunk containing `"authentication"`, and likewise
> `rotate`/`rotates`, `debug`/`debugging`, `fail`/`failed`. Because the lexical leg feeds one
> half of an RRF merge, those were candidates *removed from retrieval*, not merely reordered.
> The same branch carries `010_add_chunk_sparse.sql`, a learned-sparse inverted index — also
> unmerged, and out of scope here: it is a retrieval feature and `PLAN.md` §8 gates those on
> a measured improvement.

`remove_diacritics 2` rather than `1`; option `1` fails to strip diacritics that are encoded
as separate codepoints and is retained only for backward compatibility. The full tokenizer
string above was checked against a live FTS5 build: it parses, and `authenticate` matches a
chunk containing `authentication`.

**No `tokenchars`.** A technical corpus is full of `snake_case`, `X-Forwarded-For` and `3.12`,
and adding `_`/`-`/`.` as token characters keeps those whole — at the cost of `foo_bar` no
longer matching a query for `bar`. Splitting favours recall, and code identifiers are served
by the dense leg and by tree-sitter symbols (`PLAN.md` §5) rather than by BM25. Recorded so
the trade is visible if lexical code search turns out to matter.

**The tokenizer is fixed at table creation.** Changing it is a rebuild, so it is recorded in
`index_state.fts_tokenizer`, not hardcoded — the same discipline as the embedding fingerprint,
for the same reason. This also makes the English-only limitation of Porter a configuration
decision rather than a silent assumption: on a non-English corpus, `unicode61` alone is the
better choice and the operator can make it.

Rebuilding is rung 1 of the ladder: `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`.
Nothing is re-fetched, re-parsed or re-embedded.

### 6.2 LanceDB behind `VectorStore`

The Lance table holds the minimum needed to find a chunk and to filter before finding it:

| Column | Type | Purpose |
|---|---|---|
| `id` | `string` | `chunks.id`; the join key |
| `vector` | `fixed_size_list<float32, D>` | `D` from the fingerprint, never a literal |
| `document_id` | `string` | pushdown filter, and delete-by-document |
| `kind` | `string` | pushdown filter |
| `lang` | `string` | pushdown filter |
| `position` | `int64` | pushdown filter |
| `chunk_json` | `string` | the chunk itself — see below |

**`chunk_json` is a departure from the original design, forced by the protocol.** An earlier
draft of this section said the Lance row holds no text: a search would return `(id, distance)`
and SQLite would hydrate. That is not implementable against the merged contract.
`VectorStore.search` returns `Candidate`, which carries a whole `Chunk`, and
`assert_vector_store_is_dimension_agnostic` in `manicule.testing` exercises a store with no
relational database behind it and requires the returned chunks to carry their text. A vector
store that cannot answer on its own does not satisfy the protocol.

So the chunk travels with the vector, as one canonical JSON column rather than a spread of
typed columns — `Chunk.model_dump_json` and its inverse round-trip whatever the type has
*now*, whereas a hand-written column mapping is a second place to remember when the type
gains a field, and the field nobody remembers is the one that goes missing in silence.

**The hydrating join still exists and is still the enforcement point** (§8) — it is simply
downstream of the vector store rather than inside it. Retrieval joins candidates to
`documents`, so a vector whose chunk row is gone, or whose document is soft-deleted or not yet
`indexed`, is invisible whatever Lance returns:

```sql
SELECT c.*, d.uri, d.title
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.id IN (…) AND d.deleted_at IS NULL AND d.status = 'indexed'
```

Nothing needs to be deleted from LanceDB for a result to stop being served. The cost of
`chunk_json` is a second copy of the corpus text, and the honest accounting is that the
protocol bought self-sufficiency with duplication.

**No `workspace_id` column.** Workspace lives on `documents`, and the join above applies it.
Promoting it into Lance would make it a value that can disagree with SQLite.

**Vectors are L2-normalised and the metric is cosine**, so `score = 1 - _distance` is an
actual cosine similarity in `[-1, 1]`. `PLAN.md` §8 has confidence scoring, and a confidence
derived from an arbitrary monotone transform of an L2 distance is a number with no meaning
outside its own ranking.

**No ANN index below a threshold.** Exhaustive search over a few tens of thousands of vectors
is fast and *exact*; an IVF_PQ index built too early costs recall for latency nobody is
waiting on. Build one when the row count crosses `ann_index_threshold` (default 100 000),
with `num_partitions ≈ sqrt(n)`. Note the interaction with §6.6: once an ANN index exists,
a pushed-down predicate is applied against partitions rather than the full set, so filter
selectivity starts affecting recall as well as speed.

### 6.3 Fingerprints, and the refusal

`docs/contracts.md` §4: *the vector table is created at first ingest, and ingest must refuse
to start if the fingerprint does not match what the index was built with.*

**Dimensional equality is not vector-space equality.** This is the whole reason the check
keys on model identity:

- `bge-large-en-v1.5` and `e5-large-v2` are both 1024-dimensional. Swapping one for the
  other passes any dimension check, and every stored vector becomes noise relative to every
  new query. Retrieval degrades toward random with no error raised anywhere.
- The same weights with CLS pooling versus mean pooling diverge to **0.69 cosine at a
  450-token chunk** (`PLAN.md` §7) — identical dimension, incompatible space, and pooling is a
  parameter manicule itself chooses. The divergence is a length-dependent curve rather than a
  constant, and it widens with chunk length, so at the 512-token budget
  ([`parsing.md`](parsing.md) §1) the two poolings are at their furthest apart.
- **And that divergence is architecture-dependent, which is the sharper point.** The same
  comparison stays at 0.87–0.96 on BERT while reaching 0.69 on ModernBERT. A fingerprint that
  omitted `pooling` would pass every test on a BERT checkpoint and corrupt the space on a
  ModernBERT one. A field that only matters on some architectures is exactly the field someone
  leaves out.
- E5 and BGE expect an instruction prefix on queries (`"query: "`, `"Represent this
  sentence…"`). Same weights, same pooling, different prefix convention, different space in
  practice.

> **Prior art.** `ensureCollection(name, dimensions)` compares `existingDimensions !== dimensions`
> and nothing else. All three cases above pass it.

**What is compared.** `EmbedFingerprint.identity()` — the declared subset of fields that
decide comparability — in the canonical form pinned in §4.6, compared for byte equality. Not
field-by-field, so a field added to that subset later cannot be silently ignored by a
comparison that predates it. As shipped in `manicule.core.embedding`, identity is `model_id`,
`revision`, `dimension`, `pooling`, `normalized` and `tokenizer_id`.

**Three fields are recorded but deliberately excluded, and two of them rest on one
measurement.** `max_sequence_length` is out because including it would force a full re-embed
whenever the limit *rises*, which changes nothing about the stored vectors; what matters is
whether any text was truncated, and `require_within_context` checks that against the actual
batch — in particular on the re-embed path, which reads stored `embed_text` without re-chunking
and so never runs the chunker's own budget refusal.

`backend` is out because the same model under MLX and under ONNX produces interchangeable
vectors, which keeps a corpus portable between machines. That was an assumption when this was
written and is now a measurement: cosine 1.000000 with a largest component difference of
1.8 × 10⁻⁵ on `bge-m3`, asserted per run in `tests/test_embedding_backends.py`
([`embeddings.md`](embeddings.md) §3.3). **If parity ever stops holding, moving `backend` into
the identity set is the correction** — it makes a runtime change a loud error with a re-embed
path, which is right if the vectors really do differ.

`weights_ref` is out on the same measurement and needs the same care. A backend rarely runs the
canonical repository's own files — `BAAI/bge-m3` publishes no safetensors, so MLX executes a
community conversion — and recording which one ran is what makes an index diagnosable. The one
re-encoding that *does* move the vectors is quantisation, and that is refused at load rather
than absorbed here: 4-bit `bge-m3` sits at cosine 0.92–0.97 to the same model in fp16
([`embeddings.md`](embeddings.md) §1.0).

**`architecture` is not currently an identity field, and there is an argument that it should
be.** `mlx-embeddings` binds `last_hidden_state` to the *pooled* vector on some architectures
and not others, so architecture determines which tensor the extraction path reads — upstream
of pooling rather than beside it. Two checkpoints agreeing on model id, revision, dtype,
pooling and dimension could still land in different spaces. It is left to
[#3](https://github.com/mgd43b/manicule/issues/3), which owns the type and the measurement;
recorded here because storage is what would fail to notice.

**`architecture` is in that list for a concrete reason**, and it is a second instance of why
the comparison is by bytes. `mlx-embeddings` binds `last_hidden_state` to the *pooled* vector
only on some architectures and not others, so architecture determines which tensor the
extraction path even reads — upstream of pooling rather than beside it. Two checkpoints
agreeing on model id, revision, dtype, pooling and dimension can still land in different
spaces. Byte equality catches the field nobody anticipated; a hand-written comparison catches
only the fields its author thought of.

A `ChunkFingerprint` — chunker, version, `max_tokens`, `overlap_tokens`, `tokenizer_id` and a
per-language tree-sitter grammar map ([`parsing.md`](parsing.md) §1.7) — sits beside it in the same row, on the same terms. Changing it
means re-chunk *and* re-embed, so it is a strictly larger invalidation than the embedding
fingerprint and gets the same refusal.

**Where it is persisted — three places, all compared.**

| Location | Says |
|---|---|
| Config | what the operator has asked for **now** |
| `index_state` (SQLite) | what ingest last committed to |
| `_manicule_meta` (a one-row Lance table in `vectors/`) | what these vector files were actually built with |

Three, because two cannot detect the interesting failure: swapping the `vectors/` directory
for another instance's, or restoring half a backup. Writing the fingerprint into the vector
directory is also what makes that directory self-describing, which is the difference between
an open storage format and a restorable one.

A one-row Lance table rather than Arrow schema metadata, because it is readable through the
same API as everything else and does not depend on schema-metadata round-tripping.

**The refusal is total, not ingest-only.** If the fingerprints disagree, the store refuses to
open for **retrieval as well as ingest**. Querying an old index with a new model is the
silent-degradation case in its purest form: it returns plausible, ranked, entirely
meaningless results. Only the repair paths — `doctor`, `reindex`, `backup` — may open a
mismatched store, and they open it knowing.

**What the operator sees.** Not "dimension mismatch". A field-by-field diff of the two
fingerprints, and two exits:

```
Embedding fingerprint mismatch — refusing to open the index.

  field           index was built with        config now asks for
  model           BAAI/bge-large-en-v1.5      intfloat/e5-large-v2
  revision        d4fa1d2                     8f2e0b1
  pooling         cls                         mean
  dim             1024                        1024                  (unchanged)

The stored vectors were produced by a different model and cannot be compared
with vectors from this one. Equal dimensionality does not mean a shared space.

Either:
  restore the previous configuration —
      embedding.model    = BAAI/bge-large-en-v1.5
      embedding.revision = d4fa1d2
      embedding.pooling  = cls
  or re-embed the existing corpus —
      manicule reindex --re-embed        (412 908 chunks, ~22 min, no re-fetch)
```

The second line of that estimate is the point: `--re-embed` reads `chunks.embed_text` and
touches neither the network nor a parser. Rung 2, not rung 4.

### 6.4 Per-document lineage

`index_state` records what the store as a whole was built with. `documents.chunk_fp` and
`documents.embed_fp` record, per document, the short hash of the fingerprints *that document*
was last built with.

This is what makes invalidation set-valued rather than total. A tree-sitter grammar upgrade
changes code parse trees and therefore code chunk boundaries — and nothing else. With
per-document lineage that is a query:

```sql
SELECT id FROM documents
WHERE chunk_fp <> :current AND media_type IN (:code_types)
```

Without it, the only expressible repair is "everything". The global refusal in §6.3 still
stands — one vector table cannot hold two spaces — but after adopting a new fingerprint, the
repair is targeted instead of total.

### 6.5 Creating and replacing the vector table

**First ingest** creates `chunks__<fp8>` where `<fp8>` is the first eight hex characters of
the canonical fingerprint hash, writes `_manicule_meta`, and sets `index_state.vector_table`
in the same SQLite transaction that records the fingerprint.

**`reindex --re-embed`** never mutates the live table:

1. Create `chunks__<newfp8>` alongside the existing one.
2. Embed from `chunks.embed_text` into it, in batches, resumable.
3. In one SQLite transaction: update `index_state.vector_table`,
   `index_state.embed_fingerprint`, and every `documents.embed_fp`.
4. Drop the old table.

Crash at any point before step 3 leaves the old table live and pointed at; the partial new
table is garbage collected by name. Crash after step 3 leaves an orphan table, swept the same
way. This is the reason `vector_table` is a pointer in §4.6 rather than a constant in the
code — an in-place rebuild has a window in which the index is neither the old thing nor the
new one.

> **Prior art.** The equivalent operation renames the `vectors/` directory aside, resets every
> document to `pending`, clears the FTS table, and instructs the operator to re-sync every
> connector and re-run indexing — a full re-crawl, rung 4, because there is no stored chunk
> text to re-embed from and no retained bytes to re-parse from.

### 6.6 `Filter` — proposed, not settled

`docs/contracts.md` §6 leaves the `Filter` shape open, narrowed to "a LanceDB predicate plus
a metadata pre-filter, with the exact split wanting contact with real data volumes."
**This section proposes a shape; it does not close the question.**

```python
class Filter(BaseModel):
    workspace_ids:  frozenset[str]                      # required
    document_ids:   frozenset[str] | None = None
    connector_ids:  frozenset[str] | None = None
    collection_ids: frozenset[str] | None = None
    tag_ids:        frozenset[str] | None = None
    kinds:          frozenset[BlockKind] | None = None
    langs:          frozenset[str] | None = None
    updated_after:  datetime | None = None
    updated_before: datetime | None = None
```

**`workspace_ids` is required and is not optional.** Workspace isolation is a security
boundary enforced on every query (`PLAN.md` §14), and a boundary you can forget to pass is
not a boundary. It is a set rather than a single value because `PLAN.md` §16 has admin
cross-workspace search, which is otherwise the exact pressure that turns a required field
back into an optional one. `filter=None` in the protocol therefore means "no *additional*
constraint" — the workspace scope arrives with the store handle.

**The split.** Fields backed by a promoted Lance column (`workspace_ids`, `document_ids`,
`connector_ids`, `kinds`, `langs`) push down as a Lance predicate. Fields requiring a join
(`collection_ids`, `tag_ids`, `updated_*`) resolve in SQLite first into a set of document IDs
and push down as an `IN` list.

That works while the resolved set is small. Above some size the `IN` list stops being a
useful predicate and the better plan inverts: search unfiltered with an inflated `k` and
post-filter through the SQLite join. The threshold — `prefilter_id_limit` — is genuinely
unknown, depends on whether an ANN index exists (§6.2), and is exactly what "contact with
real data volumes" means.

**One correctness trap regardless of which regime wins.** Over-fetching `k' > k` and
post-filtering does *not* guarantee `k` survivors. The implementation must either expand and
retry, or report the result set as truncated. It must never quietly return fewer than `k` and
let the caller assume the corpus had no more.

---

## 7. Retained original bytes

`docs/contracts.md` §2: *`original_ref` points at the retained source bytes, so re-parsing
never means re-fetching.* That is rung 3 of the ladder, and it is the only thing standing
between a parser bug fix and a full re-crawl.

**Content-addressed, immutable, sharded.**

```
<data_dir>/blobs/sha256/ab/cd/abcd1234…      (+ .zst when compressed)
```

`original_ref` stores the bare hash and is a foreign key to `blobs.hash`; the `Document`
domain object presents it as `blob:sha256:<hex>` so nothing downstream depends on a
filesystem path and the data directory can move.

**Why content-addressed.** The same PDF attached to forty Confluence pages is stored once.
Immutability means the directory is safe to copy at any moment, incrementally, with any tool.
And verification is free: a blob whose bytes do not hash to its own name is corrupt, and
`doctor` can say so without a reference copy.

**Write ordering: content first, reference second.** Write to a temporary file in the target
directory, fsync, rename into place, insert the `blobs` row, and only then write the
`documents.original_ref` that points at it. A reference therefore always resolves. Deletion
runs the ordering in reverse.

**Compression** with zstd for text-ish media types, recorded per blob. `size_bytes` and
`stored_bytes` are both kept because they answer different questions.

**Not everything is retained.** Above `max_original_bytes` (default 256 MiB) the bytes are
dropped, `original_ref` is `NULL`, and `original_omitted_reason` says why. A 4 GB video
attachment should not silently double the data directory. This is the same rule as `Unlocated`
in `docs/contracts.md` §1: absent with a stated reason, visible in diagnostics, never a
silent partial success.

**Garbage collection is mark-and-sweep, never refcounts.** A refcount is a number that has to
survive every crash in every path that touches it, and when it is wrong it is wrong silently
in both directions. The sweep is a query:

```sql
SELECT hash FROM blobs WHERE hash NOT IN (
    SELECT original_ref FROM documents         WHERE original_ref IS NOT NULL
    UNION
    SELECT original_ref FROM document_versions WHERE original_ref IS NOT NULL
)
```

For each candidate: delete the row in a transaction, **then** unlink the file. Crashing
between the two leaks a file, which a directory scan against `blobs` reclaims on the next
pass. The reverse order would leave a row pointing at nothing, which is a broken reference
rather than wasted disk. Every ordering decision in this document resolves the same way:
prefer the failure that costs space over the failure that costs correctness.

**Retention.** The live document's bytes are kept for as long as the document exists.
Prior versions' bytes are kept for `retain_version_bytes` (default 30 days), because a chatty
wiki otherwise grows the blob store without bound. This is a real policy decision that no
existing document made; it is a default, and it is configurable.

### 7.1 What the data directory now contains

Stated here rather than tracked elsewhere, on the same principle as the permission-awareness
warning in `docs/connectors/confluence.md` §9: this is a consequence of a design decision, and
it should ship with the design rather than be discovered by whoever finds the directory.

**Before retention, `<data_dir>` held derived artefacts** — extracted text, vectors, metadata.
Recovering the source documents from it would have been lossy and partial.

**With retention it holds the corpus itself.** Every PDF, every attachment, every page body,
byte-identical to what the connector fetched. And because the index is not permission-aware —
content is fetched as the sync user, per `docs/connectors/confluence.md` §9 — the directory
accumulates everything that user can see, now in original form. Two consequences follow, and
both are storage's to handle:

**Permissions are part of the layout.** `<data_dir>` and everything under it is created
`0700`/`0600`, owned by the running user. `doctor` fails — not warns — if the directory is
group- or world-readable, and names the offending path. A default that depends on the
operator's `umask` is not a default.

**`manicule backup` refuses a world-readable target** unless `--allow-insecure-target` is
passed, and creates its own output `0700`. §9 makes backup a routine, exercised operation,
which makes an unprotected backup the most likely way a copy of the corpus ends up somewhere
it should not be. A procedure that is safe only when performed carefully is not safe.

**What is genuinely not storage's to decide** stays with the security surface
([#13](https://github.com/mgd43b/manicule/issues/13),
[#19](https://github.com/mgd43b/manicule/issues/19)): encryption at rest and its key
management, whether retention is opt-out per connector, and the deployment-guide wording. The
disclosure itself is discharged here.

---

## 8. Two stores, one crash

### 8.1 The invariants

| | |
|---|---|
| **I1** | Every servable chunk has a row in `chunks`. Retrieval hydrates through the inner join in §6.2, so this is enforced by construction, not by convention. |
| **I2** | LanceDB may be a **superset** of `chunks`. It is never treated as a subset. Extra vectors are inert — invisible to the join — and are swept. |
| **I3** | `documents.status = 'indexed'` is written **last**, in the transaction that is the commit point of ingest. Until then the document is not servable whatever exists elsewhere. |
| **I4** | FTS5 cannot diverge from `chunks` at all, because the triggers in §6.1 run inside the same transaction as the row change. Its only failure mode is corruption, which `integrity-check` detects and `rebuild` fixes for free. |

### 8.2 Ingest write order, and every crash window

1. Write `chunks` rows (SQLite transaction). Document is still `parsing`/`embedding`.
2. Upsert vectors into the Lance table.
3. Set `documents.status = 'indexed'`, `indexed_at`, `embed_fp`, `chunk_fp` (SQLite transaction).

- **Crash between 1 and 2** — chunks exist, vectors do not, document is not `indexed`. Nothing
  is served. Repair re-embeds the chunks that have no vector. Rung 2.
- **Crash between 2 and 3** — vectors exist for a document that is not `indexed`. Nothing is
  served (I3). Repair re-runs step 2 idempotently (upsert by `chunks.id`) and step 3.
- **Crash during 3** — SQLite transaction, so it either happened or it did not.

**Deletion is deferred, always.** Soft-deleting a document sets `deleted_at` and touches
nothing else: chunks stay, vectors stay, FTS rows stay, and all of them become invisible at
the join. Restore is clearing the timestamp — no re-embed, no re-parse, no re-fetch. Hard
deletion cascades to `chunks`, whose `AFTER DELETE` trigger writes the IDs into
`vector_tombstones`, and a sweep removes them from LanceDB later.

**Deferred deletion is not free, and the cost lands on retrieval.** "Invisible at the join"
is true of the *result*, but the rows are still in the derived indexes and still compete for
top-`k` slots before the join runs. A document soft-deleted an hour ago can crowd live
content out of a result set entirely. The two legs pay for this differently:

- **The lexical leg does not pay.** FTS5 and `documents` are in the same database, so the
  filter and the `LIMIT` go in one statement and `k` means `k` live rows (§6.1).
- **The vector leg does pay**, and cannot be fixed the same way: `deleted_at` and `status`
  live in SQLite, not in the Lance table, so the ANN search selects its top-`k` before
  anything can filter on them. It must over-fetch and expand-retry — the trap already named
  in §6.6, now with a guaranteed cause rather than a hypothetical one.

Promoting `deleted_at` into the Lance table would fix that and is rejected: it makes every
soft delete a write to the vector store, which is exactly the coupling deferred deletion
exists to remove, and it re-introduces a value that can disagree with SQLite.

**So the sweep also collects soft-deleted documents, after a grace period.** Within
`soft_delete_grace` (default 30 days) restore is free. After it, the sweep purges the
chunks — and restore then costs a re-parse from retained bytes, rung 3, still not a re-fetch.
This is a real trade rather than a free lunch: unbounded free restore would mean unbounded
dilution of every vector search. `doctor` reports the soft-deleted fraction of the vector
table so the over-fetch factor can be set against a measured number.

```sql
CREATE TRIGGER chunks_ad_tomb AFTER DELETE ON chunks BEGIN
  INSERT OR IGNORE INTO vector_tombstones(chunk_id, deleted_at)
    VALUES (old.id, datetime('now'));
END;
```

This is the one place a SQL-side timestamp is acceptable, because nothing orders or compares
`vector_tombstones` against another table's timestamps — the sweep reads the whole table. It
is called out so the §3.3 rule reads as absolute where it matters.

> **Prior art.** Soft delete there sets `deleted_at` in SQLite and *hard*-deletes the vectors,
> with a comment noting LanceDB has no soft delete. Restoring a document therefore means
> re-embedding it. Making the join the filter means the vector store never needs to know that
> a document was deleted.

**Why a tombstone table rather than an anti-join sweep.** Sweeping by comparing all Lance IDs
against `chunks` races with concurrent ingest: an ID written after the scan began looks like
an orphan, and the sweep deletes a live vector. A tombstone list only ever names things that
*were* deleted, so it cannot. It is also cheap — the sweep reads a small table instead of the
whole index.

### 8.3 Recovery

`manicule reindex --repair` is the general recovery path and it never leaves rung 2:

| Symptom | Repair |
|---|---|
| FTS corrupt or out of date | `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` |
| Chunks with no vector | Embed those chunk IDs |
| Vectors with no chunk | Sweep (tombstones, then a bounded reconciliation pass) |
| Document `indexed` with zero chunks and a non-terminal status | Re-parse from `original_ref` |
| `original_ref` dangling | Re-fetch that document only — the one rung-4 case |

---

## 9. Backup and restore

[#14](https://github.com/mgd43b/manicule/issues/14) notes that an open storage format is not
a restore procedure. This is the storage half of that.

### 9.1 The consistency problem, named

Two stores, no shared transaction, and copying the vector directory takes long enough that
SQLite moves underneath it. Naively there is no instant at which both are captured.

**Ordering alone gets most of the way.** Because SQLite is authoritative and LanceDB may be a
superset (I2), the safe skew is **LanceDB ahead of SQLite**, never behind: extra vectors are
inert, missing vectors mean a document marked `indexed` that silently returns nothing. So
snapshot **SQLite first, LanceDB second** — which is the opposite of the intuition that you
capture the big slow thing first.

**Ordering alone is not sufficient**, because a delete landing between the two snapshots
removes vectors that the already-captured SQLite still references, reproducing exactly the
failure the ordering was meant to avoid. That is why §8.2 makes deletion deferred: between
sweeps the vector store only grows.

**So the procedure is:**

1. Take the backup lock. It blocks the vector sweep and the blob GC — the only two things
   that remove data — and nothing else. Ingest and queries continue.
2. Snapshot SQLite with the online backup API (`sqlite3.Connection.backup()`) or
   `VACUUM INTO`. Both produce one consistent file from a live WAL database.
3. Record `table.version` for the Lance table. Lance is versioned; this pins the instant.
4. Copy the Lance table directory and the blob directory. Both are append-only under the lock.
5. Write the manifest.
6. Release the lock.

On restore, `checkout` the recorded Lance version, discarding anything written after step 2.
The pair is then consistent as of one instant, with writes never having stopped.

> **Prior art, and the trap to avoid.** The existing procedure copies `opendocuments.db`,
> `.db-wal` and `.db-shm` with `copyFileSync`. A file-by-file copy of a live WAL database is
> not a snapshot — the three files are captured at three different instants and a checkpoint
> between them produces an unopenable result. It is guarded by refusing to run when a PID
> file indicates a live server, which does not cover a second CLI process holding the
> database, and it makes backup an offline operation. `sqlite3.Connection.backup()` is in the
> standard library and is correct against a live database, so there is no reason to copy the
> files.
>
> Two parts of that implementation are worth carrying over as-is, and are: a manifest with a
> per-file sha256 inventory, and a validator that rejects symlinks, absolute paths, path
> traversal, and any file present on disk but absent from the inventory. That validation is
> careful work.

### 9.2 The manifest

Everything the existing manifest carries — format, version, timestamp, per-file size and
sha256 — plus what it must:

```json
{
  "format": "manicule-backup", "version": 1,
  "created_at": "2026-08-10T14:22:31.104Z",
  "manicule_version": "0.3.1",
  "alembic_revision": "9c1f0a7d2b44",
  "embed_fingerprint": { … },
  "chunk_fingerprint": { … },
  "fts_tokenizer": "porter unicode61 remove_diacritics 2",
  "vector_table": "chunks__7f3a91c2",
  "lance_version": 5182,
  "counts": { "documents": 14203, "chunks": 412908, "blobs": 9871 },
  "files": [ { "path": "manicule.db", "size": 184287232, "sha256": "…" } ]
}
```

Restore refuses when:

- `alembic_revision` is unknown to this build — the backup is from a newer version, and
  running old code against a newer schema corrupts it quietly.
- `embed_fingerprint` disagrees with the configured embedder, unless `--adopt-fingerprint` is
  passed, which also rewrites the configuration. Without this, restore is a back door that
  reintroduces the §6.3 mismatch through a path that never asked.
- Any inventoried file fails its hash, or any file on disk is missing from the inventory.
- Any `documents.original_ref` names a blob absent from the snapshot.

`counts` is not decoration. After restore, they are re-counted and compared; a restore that
succeeds while producing a different number of chunks has not succeeded.

### 9.3 Exercised, not documented

The acceptance test for [#2](https://github.com/mgd43b/manicule/issues/2) and the storage half
of [#14](https://github.com/mgd43b/manicule/issues/14):

1. Ingest a fixture corpus covering at least one multi-chunk document, one archive, one
   binary attachment, and one document in a terminal no-content status.
2. Run a fixed query; record the top-`k` chunk IDs **and scores**.
3. Back up. Destroy the data directory. Restore.
4. Re-run the query; assert **identical** IDs and identical scores. Not "similar" — the
   vectors are bytes and the ranking is deterministic.
5. Assert `alembic current` matches the manifest revision.
6. Assert every `original_ref` resolves and every blob's bytes hash to its own name.
7. Assert `PRAGMA integrity_check`, `PRAGMA foreign_key_check`, and the FTS5
   `integrity-check` all pass.

Plus two variants, because they are the cases the procedure exists for:

- **Hot backup with a concurrent writer** — ingest running throughout step 3; the restored
  instance must satisfy I1 and I2 and must not serve a document whose vectors are missing.
- **Restore into a differently-configured instance** — asserts a refusal with the §6.3
  message, not a successful restore into a silently broken index.

---

## 10. `doctor` — the storage checks

Each of these corresponds to a failure mode named above, and each has a repair that names its
rung on the ladder.

| Check | Detects |
|---|---|
| `sqlite_version()` ≥ 3.35, FTS5 probe | A platform SQLite too old or built without FTS5 |
| `PRAGMA foreign_keys` on a fresh connection | The §3.1 trap |
| `PRAGMA integrity_check`, `foreign_key_check` | Database corruption, orphaned references |
| `INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)` | FTS index corruption **and** disagreement with `chunks` — see below |
| Soft-deleted fraction of the vector table | Dilution of vector top-`k` (§8.2); calibrates the over-fetch factor |
| Chunks with no vector | Interrupted ingest — rung 2 |
| Vectors with no chunk | Unswept tombstones or an interrupted delete |
| Fingerprint agreement across config / `index_state` / `_manicule_meta` | A swapped or half-restored vector directory |
| `alembic current` vs `head` | An un-migrated database |
| Dangling `original_ref` | The one rung-4 case |
| Blobs on disk absent from `blobs` | A leaked GC sweep — reports reclaimable bytes |
| `<data_dir>` mode is `0700` and files `0600` | A corpus readable by other local users (§7.1) — **fails**, does not warn |
| `size_bytes` vs `stored_bytes` totals, by media type | What retention is actually costing |

**The `1` argument on the FTS integrity check is the whole check.** Two obvious ways to
verify the lexical index against `chunks` do not work, and both were in an earlier draft of
this table:

- **`COUNT(*)` on each table can never disagree.** `chunks_fts` is an external-content table,
  so `SELECT count(*) FROM chunks_fts` reads through to `chunks`. The two counts are the same
  number by construction, whatever the index contains.
- **A bare `integrity-check` passes on a completely empty index.** Without an argument it
  verifies only that the index is internally consistent — an empty index is perfectly
  consistent with itself.

Checked directly: with the triggers dropped and two rows inserted, both counts report `2`,
`MATCH` returns nothing, and a bare `integrity-check` passes. Only
`INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)` compares the index
against the content table, and it raises `database disk image is malformed` on that fixture.
A silently empty lexical index halves hybrid retrieval while every health check reports green,
so this is the difference between a diagnostic and a decoration.

---

## Appendix A: what this design decided that nothing else had

Flagged because these were calls made in the absence of a stated position, not derived from
one.

| Decision | Rationale in |
|---|---|
| Four tables added beyond the sixteen: `chunks`, `blobs`, `index_state`, `vector_tombstones` | §4.1 |
| `chunks.id` is content-derived; `position` is part of the digest, and the trade is stated | §3.2 |
| Identity is `(workspace_id, source, source_id)`; the workspace is part of the derived id, settled before any corpus exists | §4.2 |
| `documents.connector_id` is `NOT NULL`; filesystem and upload are connectors | §4.2 |
| `documents.container_id` self-referential cascade for archive members | §4.2 |
| `chunk_count` dropped | §4.2 |
| `workspace_members.api_key` dropped | §4.7 |
| `plugins` becomes the real registry; `permissions` column dropped | §4.7 |
| `connectors.watermark` added | §4.7 |
| `audit_logs` deliberately has no foreign keys; `query_logs` deliberately cascades | §4.7 |
| Column renames to match `docs/contracts.md` §2: `source_type`→`source`, `source_path`→`uri`, `file_type`→`media_type`, `source_version`→`version_token`, `parser_used`→`parser` | §4.2 |
| FTS5 is external-content over `chunks` and trigger-maintained | §6.1 |
| The vector row carries the chunk as `chunk_json`, because the protocol requires a self-sufficient store | §6.2 |
| The lexical query is one joined statement so `LIMIT` applies after filtering | §6.1 |
| The sweep collects soft-deleted documents after a grace period, trading free restore against vector top-`k` dilution | §8.2 |
| Two FTS columns with BM25 weights `1.0 / 0.4` | §6.1 |
| `porter unicode61 remove_diacritics 2`, no `tokenchars` | §6.1 |
| The refusal covers retrieval, not just ingest | §6.3 |
| Fingerprints persisted in three places and all three compared | §6.3 |
| Per-document `chunk_fp` / `embed_fp` lineage for partial invalidation | §6.4 |
| `index_state.vector_table` is a pointer; re-embed builds alongside and swaps | §6.5 |
| `Filter.workspace_ids` required and set-valued | §6.6 — **the rest of `Filter` stays open** |
| Blob retention: current version forever, prior versions 30 days; 256 MiB cap | §7 |
| `<data_dir>` is `0700`/`0600`; `doctor` fails on a looser mode; backup refuses a world-readable target | §7.1 |
| Deletion from derived stores is always deferred to a sweep | §8.2 |
| Backup order is SQLite first, LanceDB second, under a sweep-blocking lock | §9.1 |
| `STRICT` tables rejected for v1 | §3.4 |

## Appendix B: deliberately not here

- **Learned-sparse retrieval.** An inverted index for BGE-M3-style sparse vectors is a
  retrieval feature, and `PLAN.md` §8 gates those on a measured improvement from
  [#15](https://github.com/mgd43b/manicule/issues/15). It would be a table, and it can be
  added by a migration when it earns one.
- **Encryption at rest, and its key management.** §7.1 states what the data directory now
  contains and sets its permissions; encrypting it is a different problem with a key-handling
  design behind it, and belongs to [#13](https://github.com/mgd43b/manicule/issues/13) /
  [#19](https://github.com/mgd43b/manicule/issues/19).
- **Any store other than SQLite.** Settled in `PLAN.md` §2 and not reopened here.
- **The full `Filter` shape.** Still open per `docs/contracts.md` §6. §6.6 proposes; it does
  not decide.
